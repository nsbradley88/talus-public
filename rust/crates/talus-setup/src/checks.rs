//! Health checks: one verdict per check, a plain-language detail, and the fix (if any) as data.
//!
//! `talus doctor` prints them and applies fixes; `talus setup` shows the hardware ones as its first step; talus-tui
//! can show the same list as a health panel. Nothing here changes the system: `run` only reads (and, for the server,
//! sends one local GET /health).

use crate::boot;
use crate::config;
use crate::download;
use crate::manifest::{self, Prep};
use crate::paths::Paths;
use crate::plan::{self, Fit};
use crate::probe::SysInfo;
use crate::state::SetupState;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verdict {
    Pass,
    Info,
    Skip,
    Warn,
    Fail,
}

/// A fix, as data. `safe()` fixes are applied by `talus doctor --fix`; the others need the user (sudo, boot).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fix {
    /// Download a model file again (missing, partial or corrupt).
    Redownload {
        model: String,
        file: String,
    },
    /// Build Flash-Next's pack / MTP draft layer.
    Prepare {
        model: String,
    },
    /// Write the model's run config again from setup.json.
    RegenerateConfig {
        model: String,
    },
    /// Write the systemd user unit again.
    InstallService,
    StartService,
    RestartService,
    /// chmod 600.
    ChmodPrivate(PathBuf),
    /// Kernel parameters: the confirm-and-diff flow, never automatic.
    BootParams,
    /// A command that needs sudo; shown and asked for.
    Sudo {
        argv: Vec<String>,
        after: String,
    },
    /// Rebuild Talus (`talus update --rebuild`).
    Rebuild,
}

impl Fix {
    pub fn safe(&self) -> bool {
        matches!(
            self,
            Fix::Redownload { .. }
                | Fix::Prepare { .. }
                | Fix::RegenerateConfig { .. }
                | Fix::InstallService
                | Fix::StartService
                | Fix::RestartService
                | Fix::ChmodPrivate(_)
        )
    }

    pub fn describe(&self) -> String {
        match self {
            Fix::Redownload { model, file } => format!("download {file} for {model}"),
            Fix::Prepare { model } => format!("prepare {model} (pack and draft layer)"),
            Fix::RegenerateConfig { model } => format!("write the run config of {model} again"),
            Fix::InstallService => "write the systemd user service again".into(),
            Fix::StartService => "start the talus service".into(),
            Fix::RestartService => "restart the talus service".into(),
            Fix::ChmodPrivate(p) => format!("make {} readable by you only", p.display()),
            Fix::BootParams => "set the kernel parameters".into(),
            Fix::Sudo { argv, .. } => format!("sudo {}", boot::shell_join(argv)),
            Fix::Rebuild => "rebuild Talus (talus update --rebuild)".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    pub id: String,
    pub title: String,
    pub verdict: Verdict,
    /// One line: what was found.
    pub detail: String,
    /// One line: what to do, in plain words (also when there is nothing to apply).
    pub advice: Option<String>,
    pub fixes: Vec<Fix>,
}

impl Check {
    fn new(id: &str, title: &str, verdict: Verdict, detail: impl Into<String>) -> Check {
        Check {
            id: id.to_string(),
            title: title.to_string(),
            verdict,
            detail: detail.into(),
            advice: None,
            fixes: vec![],
        }
    }
    fn advice(mut self, a: impl Into<String>) -> Check {
        self.advice = Some(a.into());
        self
    }
    fn fix(mut self, f: Fix) -> Check {
        self.fixes.push(f);
        self
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Opts {
    /// Hash every model file that has no verification stamp (slow: minutes per 100 GB).
    pub deep: bool,
    /// Ask GitHub whether a newer Talus exists.
    pub network: bool,
}

pub struct Ctx<'a> {
    pub sys: &'a SysInfo,
    pub paths: &'a Paths,
    pub state: Option<&'a SetupState>,
    /// The models whose needs count (installed ones, or the ones being chosen in setup).
    pub models: Vec<&'static manifest::ModelInstall>,
    pub opts: Opts,
}

impl<'a> Ctx<'a> {
    pub fn new(
        sys: &'a SysInfo,
        paths: &'a Paths,
        state: Option<&'a SetupState>,
        opts: Opts,
    ) -> Ctx<'a> {
        let models = state
            .map(|s| s.models.keys().filter_map(|k| manifest::get(k)).collect())
            .unwrap_or_default();
        Ctx {
            sys,
            paths,
            state,
            models,
            opts,
        }
    }
}

// ------------------------------------------------------------------------------------------------- hardware

pub fn check_os(sys: &SysInfo) -> Check {
    let name = if sys.os_pretty.is_empty() {
        "unknown Linux".to_string()
    } else {
        sys.os_pretty.clone()
    };
    if sys.arch != "x86_64" {
        return Check::new(
            "os",
            "Operating system",
            Verdict::Fail,
            format!("{name} on {}", sys.arch),
        )
        .advice("Talus needs x86-64 Linux (Strix Halo).");
    }
    let kmaj: u32 = sys
        .kernel
        .split('.')
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let known = crate::support::distro(sys);
    let detail = format!(
        "{name}, kernel {}",
        if sys.kernel.is_empty() {
            "?"
        } else {
            &sys.kernel
        }
    );
    if kmaj < 6 {
        return Check::new("os", "Operating system", Verdict::Fail, detail)
            .advice("This kernel is too old for the Radeon 8060S; use a current distribution (we test Ubuntu 26.04, kernel 7.0).");
    }
    match known {
        crate::support::Support::Tested => {
            if kmaj >= 7 {
                Check::new("os", "Operating system", Verdict::Pass, detail)
            } else {
                Check::new("os", "Operating system", Verdict::Warn, detail)
                    .advice("Kernels before 7.0 are untested with Talus; the release was measured on 7.0.")
            }
        }
        crate::support::Support::Handled => Check::new("os", "Operating system", Verdict::Pass, detail)
            .advice("The installer handles this distribution; Talus is measured on Ubuntu 26.04."),
        crate::support::Support::Unknown => Check::new("os", "Operating system", Verdict::Warn, detail)
            .advice("Not a distribution the installer knows: install a C/C++ compiler, git, curl and Python 3 with venv yourself."),
    }
}

pub fn check_gpu(sys: &SysInfo) -> Check {
    let cpu = if sys.cpu_model.is_empty() {
        "unknown CPU".to_string()
    } else {
        sys.cpu_model.clone()
    };
    if sys.has_gfx1151() {
        return Check::new(
            "gpu",
            "GPU",
            Verdict::Pass,
            format!("gfx1151 (Radeon 8060S) in {cpu}"),
        );
    }
    if sys.gpus.is_empty() {
        return Check::new(
            "gpu",
            "GPU",
            Verdict::Fail,
            format!("no AMD GPU compute device found ({cpu})"),
        )
        .advice(
            "Talus needs a Strix Halo GPU (gfx1151) with the amdgpu driver loaded (/dev/kfd).",
        );
    }
    Check::new("gpu", "GPU", Verdict::Fail, format!("found {} ({cpu})", sys.gpus.join(", ")))
        .advice("Only gfx1151 (Ryzen AI Max / Radeon 8060S) is supported; other GPUs have no tuning tables yet.")
}

pub fn check_groups(sys: &SysInfo) -> Check {
    let mut need: Vec<String> = Vec::new();
    for g in [sys.kfd_group.as_deref(), sys.render_group.as_deref()]
        .into_iter()
        .flatten()
    {
        if g != "root" && !sys.in_group(g) && !need.iter().any(|x| x == g) {
            need.push(g.to_string());
        }
    }
    if sys.groups.is_empty() {
        return Check::new(
            "groups",
            "GPU access",
            Verdict::Skip,
            "could not read your groups",
        );
    }
    if need.is_empty() {
        let mut via: Vec<&str> = [sys.kfd_group.as_deref(), sys.render_group.as_deref()]
            .into_iter()
            .flatten()
            .collect();
        via.dedup();
        let d = if via.is_empty() {
            "no GPU device nodes".to_string()
        } else {
            format!("you are in {}", via.join(", "))
        };
        return Check::new(
            "groups",
            "GPU access",
            if via.is_empty() {
                Verdict::Skip
            } else {
                Verdict::Pass
            },
            d,
        );
    }
    let user = std::env::var("USER").unwrap_or_else(|_| "$USER".into());
    Check::new(
        "groups",
        "GPU access",
        Verdict::Fail,
        format!(
            "you are not in the {} group(s) that own /dev/kfd and /dev/dri",
            need.join(", ")
        ),
    )
    .advice("Add yourself to the group(s), then log out and back in.")
    .fix(Fix::Sudo {
        argv: vec!["usermod".into(), "-aG".into(), need.join(","), user],
        after: "Log out and back in (or reboot) for the new group to apply.".into(),
    })
}

pub fn check_ram(sys: &SysInfo) -> Check {
    let t = sys.mem_total_gib();
    let d = format!("{t:.0} GiB total, {:.0} GiB free now", sys.mem_avail_gib());
    if t >= 120.0 {
        Check::new("ram", "Memory", Verdict::Pass, d)
    } else if t >= 60.0 {
        Check::new("ram", "Memory", Verdict::Warn, d)
            .advice("Talus is built for 128 GB machines; with less, only Qwen3.8-27B can fit.")
    } else {
        Check::new("ram", "Memory", Verdict::Fail, d)
            .advice("Every Talus model needs a 128 GB Strix Halo.")
    }
}

/// The kernel-parameter checks. `need_pool`: a model that needs the raised pool is installed or chosen.
pub fn check_boot(sys: &SysInfo, need_pool: bool) -> Vec<Check> {
    let st = boot::status(sys);
    let get = |k: &str| {
        st.iter()
            .find(|s| s.param.key == k)
            .cloned()
            .expect("documented")
    };
    let pool = get("ttm.pages_limit");
    let kho = get("kho");
    let lock = get("amdgpu.lockup_timeout");
    let mut out = Vec::new();
    let pool_c = if pool.ok {
        Check::new(
            "gpu_pool",
            "GPU memory pool",
            Verdict::Pass,
            format!("{} (120 GiB needed by GLM and MiMo)", pool.current),
        )
    } else if need_pool {
        Check::new("gpu_pool", "GPU memory pool", Verdict::Fail, format!("{}; GLM-5.3-Flash and MiMo-V2.6-Flash need 120 GiB", pool.current))
            .advice("Raise it with the ttm kernel parameters (talus doctor --fix shows the exact change, asks, and never reboots for you).")
            .fix(Fix::BootParams)
    } else {
        Check::new(
            "gpu_pool",
            "GPU memory pool",
            Verdict::Info,
            format!(
                "{}: fine for your models (GLM and MiMo would need 120 GiB)",
                pool.current
            ),
        )
    };
    out.push(pool_c);
    let kho_c = if kho.ok {
        Check::new("kho", "Kexec-handover reserve", Verdict::Pass, kho.current)
    } else if need_pool {
        Check::new(
            "kho",
            "Kexec-handover reserve",
            Verdict::Warn,
            format!("{}: about 9 GiB held back", kho.current),
        )
        .advice("Add kho=off to the kernel parameters (part of the same change as the GPU pool).")
        .fix(Fix::BootParams)
    } else {
        Check::new(
            "kho",
            "Kexec-handover reserve",
            Verdict::Info,
            format!("{}: about 9 GiB held back; kho=off frees it", kho.current),
        )
    };
    out.push(kho_c);
    out.push(if lock.ok {
        Check::new("lockup", "GPU timeout", Verdict::Pass, lock.current)
    } else {
        Check::new("lockup", "GPU timeout", Verdict::Warn, lock.current)
            .advice("Very long prompts can hit the driver's timeout; amdgpu.lockup_timeout=10000,60000,10000,10000 gives them 60 s.")
            .fix(Fix::BootParams)
    });
    out
}

pub fn hardware(sys: &SysInfo, need_pool: bool) -> Vec<Check> {
    let mut v = vec![
        check_os(sys),
        check_gpu(sys),
        check_groups(sys),
        check_ram(sys),
    ];
    v.extend(check_boot(sys, need_pool));
    v
}

// ------------------------------------------------------------------------------------------------- install

/// The ROCm version of an SDK root: TheRock wheels keep it in the `rocm_sdk_devel-<v>.dist-info` next to
/// `_rocm_sdk_devel`; tarballs and system installs in `.info/version`.
pub fn rocm_version(root: &Path) -> Option<String> {
    if let Some(parent) = root.parent() {
        if let Ok(rd) = std::fs::read_dir(parent) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                if let Some(v) = n
                    .strip_prefix("rocm_sdk_devel-")
                    .and_then(|r| r.strip_suffix(".dist-info"))
                {
                    return Some(v.to_string());
                }
            }
        }
    }
    std::fs::read_to_string(root.join(".info/version"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub fn check_rocm(st: Option<&SetupState>, paths: &Paths) -> Check {
    // the prebuilt programs find ROCm through install.sh's link: a dangling link breaks them
    let link = paths.rocm_link();
    if link.symlink_metadata().is_ok() && !link.join("lib/libamdhip64.so").exists() {
        return Check::new(
            "rocm",
            "ROCm SDK",
            Verdict::Fail,
            format!("{} points at a missing SDK", paths.tilde(&link)),
        )
        .advice("Run the installer again (it re-creates the link), or: ln -sfn <your ROCm root> ~/.local/share/talus/rocm-sdk")
        .fix(Fix::Rebuild);
    }
    let recorded = st.and_then(|s| s.rocm_sdk.clone());
    let Some(root) = recorded.or_else(|| std::fs::canonicalize(&link).ok()) else {
        return Check::new("rocm", "ROCm SDK", Verdict::Fail, "no ROCm SDK recorded")
            .advice("Run the installer (install.sh) again; it installs the ROCm SDK in your home directory.");
    };
    let hip = root.join("lib/libamdhip64.so");
    if !hip.exists() {
        return Check::new(
            "rocm",
            "ROCm SDK",
            Verdict::Fail,
            format!("{} has no lib/libamdhip64.so", root.display()),
        )
        .advice("The SDK is missing or incomplete: run the installer again.");
    }
    match rocm_version(&root) {
        Some(v) if v == manifest::rocm_version() => Check::new("rocm", "ROCm SDK", Verdict::Pass, format!("{v} (the release's)")),
        Some(v) => Check::new("rocm", "ROCm SDK", Verdict::Warn, format!("{v}; the release was built and measured with {}", manifest::rocm_version()))
            .advice("Other ROCm versions should work but are untested; report problems with `talus doctor --report`."),
        None => Check::new("rocm", "ROCm SDK", Verdict::Warn, format!("version unknown at {}", root.display())),
    }
}

pub fn check_binaries(paths: &Paths) -> Check {
    let bin = paths.bin();
    let need = [
        "libtalus_kernels.so",
        "talus-cli",
        "talus-server",
        "talus-tui",
    ];
    let missing: Vec<&str> = need
        .iter()
        .copied()
        .filter(|n| !bin.join(n).exists())
        .collect();
    if missing.is_empty() {
        Check::new(
            "binaries",
            "Talus programs",
            Verdict::Pass,
            format!("kernel library and programs in {}", paths.tilde(&bin)),
        )
    } else {
        Check::new(
            "binaries",
            "Talus programs",
            Verdict::Fail,
            format!("missing in {}: {}", paths.tilde(&bin), missing.join(", ")),
        )
        .advice("Build them again.")
        .fix(Fix::Rebuild)
    }
}

/// Files of one installed model.
pub fn check_model_files(m: &manifest::ModelInstall, st: &SetupState, deep: bool) -> Check {
    let name = m.spec().name;
    let id = format!("files:{}", m.id);
    let choices = st.models.get(m.id).copied().unwrap_or_default();
    let dir = st.model_dir(m.id);
    let mut bad: Vec<String> = Vec::new();
    let mut unverified = 0;
    let mut fixes = Vec::new();
    let mut total = 0u64;
    for f in m.selected_files(&choices) {
        total += f.size;
        let p = dir.join(f.local_name());
        let size = std::fs::metadata(&p).map(|m| m.len()).ok();
        let problem = match size {
            None => Some(if dir.join(plan::part_name(f)).exists() {
                "partly downloaded"
            } else {
                "missing"
            }),
            Some(s) if s != f.size => Some("wrong size"),
            Some(_) if download::stamped_ok(&dir, f) => None,
            // small files are cheap to hash, so they are always checked when their stamp is stale
            Some(_) if deep || f.size < 4 << 30 => match download::sha256_file(&p, None) {
                Ok(h) if h == f.sha256 => {
                    let _ = download::write_stamp(&dir, f.local_name(), &h);
                    None
                }
                Ok(_) => Some("checksum mismatch"),
                Err(_) => Some("unreadable"),
            },
            Some(_) => {
                unverified += 1;
                None
            }
        };
        if let Some(why) = problem {
            bad.push(format!("{} {why}", f.local_name()));
            fixes.push(Fix::Redownload {
                model: m.id.to_string(),
                file: f.local_name().to_string(),
            });
        }
    }
    let title = format!("{name} files");
    if !bad.is_empty() {
        let mut c = Check::new(&id, &title, Verdict::Fail, bad.join("; "))
            .advice("Download them (a stopped download resumes; a corrupt file is replaced).");
        c.fixes = fixes;
        return c;
    }
    if unverified > 0 {
        return Check::new(
            &id,
            &title,
            Verdict::Warn,
            format!(
                "{} present, {unverified} not checksum-verified yet",
                plan::gb(total)
            ),
        )
        .advice("`talus doctor --deep` hashes them (a few minutes per 100 GB).");
    }
    Check::new(
        &id,
        &title,
        Verdict::Pass,
        format!("{} present and verified", plan::gb(total)),
    )
}

pub fn check_prep(m: &manifest::ModelInstall, st: &SetupState) -> Option<Check> {
    if m.prep != Prep::FlashNextPack {
        return None;
    }
    let dir = st.model_dir(m.id);
    let c = st.models.get(m.id).copied().unwrap_or_default();
    let pack = config::flash_next_pack(&dir);
    let mut missing = Vec::new();
    if !pack.join("native_experts.txt").exists() || !pack.join("tokenizer/vocab.json").exists() {
        missing.push("pack");
    }
    if c.drafters && !config::flash_next_mtp(&dir).join("rt/experts.bin").exists() {
        missing.push("MTP draft layer");
    }
    let title = format!("{} preparation", m.spec().name);
    Some(if missing.is_empty() {
        Check::new(
            &format!("prep:{}", m.id),
            &title,
            Verdict::Pass,
            "pack and draft layer ready",
        )
    } else {
        Check::new(
            &format!("prep:{}", m.id),
            &title,
            Verdict::Fail,
            format!("not built: {}", missing.join(", ")),
        )
        .fix(Fix::Prepare {
            model: m.id.to_string(),
        })
    })
}

/// The run config: present, the one this setup would write, and everything it names on disk. Missing model files
/// are the files check's business: here they only mean "waiting for the downloads".
pub fn check_config(m: &manifest::ModelInstall, paths: &Paths, st: &SetupState) -> Check {
    let p = paths.run_config(m.id);
    let title = format!("{} run config", m.spec().name);
    let id = format!("config:{}", m.id);
    let regen = Fix::RegenerateConfig {
        model: m.id.to_string(),
    };
    let text = match std::fs::read_to_string(&p) {
        Err(_) => {
            return Check::new(
                &id,
                &title,
                Verdict::Fail,
                format!("{} is missing", paths.tilde(&p)),
            )
            .fix(regen)
        }
        Ok(t) => t,
    };
    let bad = config::validate(&text, m.id);
    let expected = config::expected_text(paths, st, m.id);
    if expected.as_deref() != Some(text.as_str()) {
        let why = if bad.iter().any(|b| b.starts_with("not valid JSON")) {
            "not valid JSON".to_string()
        } else {
            "differs from your setup (edited by hand, or written by an older version)".to_string()
        };
        return Check::new(
            &id,
            &title,
            Verdict::Fail,
            format!("{}: {why}", paths.tilde(&p)),
        )
        .advice("Write it again from your setup; the old file is kept as a backup next to it.")
        .fix(regen);
    }
    let model_dir = st.model_dir(m.id);
    let (waiting, other): (Vec<&String>, Vec<&String>) = bad
        .iter()
        .partition(|b| b.contains(&*model_dir.to_string_lossy()));
    if !other.is_empty() {
        let names: Vec<&str> = other.iter().map(|s| s.as_str()).collect();
        return Check::new(&id, &title, Verdict::Fail, names.join("; "))
            .advice("Something it needs outside the model folder is missing: run `talus update --rebuild` or the installer again.")
            .fix(Fix::Rebuild);
    }
    if !waiting.is_empty() {
        return Check::new(
            &id,
            &title,
            Verdict::Warn,
            "ready, waiting for the model files (see the files check)",
        );
    }
    Check::new(&id, &title, Verdict::Pass, paths.tilde(&p))
}

/// The memory a model has now, unless the server is already serving (then the numbers include it).
pub fn check_memory(ctx: &Ctx, serving: bool) -> Vec<Check> {
    let pool_ok = boot::status(ctx.sys)
        .iter()
        .find(|s| s.param.key == "ttm.pages_limit")
        .map(|s| s.ok)
        .unwrap_or(false);
    let free = format!("{:.0} GiB free now", ctx.sys.mem_avail_gib());
    if serving {
        return vec![Check::new(
            "memory",
            "Free memory",
            Verdict::Info,
            format!("{free} while serving"),
        )];
    }
    let mut out = Vec::new();
    for m in &ctx.models {
        let fit = plan::live_fit(m, ctx.sys, pool_ok);
        let title = format!("Memory for {}", m.spec().name);
        let id = format!("memory:{}", m.id);
        out.push(match fit {
            Fit::Fits => Check::new(&id, &title, Verdict::Pass, format!("fits ({free})")),
            Fit::Tight => Check::new(&id, &title, Verdict::Warn, format!("tight ({free})")).advice(
                "It runs close to the free-memory floor: close large programs before starting it.",
            ),
            Fit::NotNow(why) => Check::new(&id, &title, Verdict::Warn, why),
        });
    }
    if ctx
        .models
        .iter()
        .any(|m| m.desktop_fit == manifest::DesktopFit::HeadlessOnly)
        && ctx.sys.desktop_active
    {
        out.push(
            Check::new("headless", "Desktop session", Verdict::Warn, "a desktop is running; GLM-5.3-Flash needs the machine without one")
                .advice("Boot to the console: sudo systemctl set-default multi-user.target, then reboot (graphical.target undoes it)."),
        );
    }
    out
}

/// One local HTTP GET. Returns the status code and body, or the connection error.
pub fn http_get_local(port: u16, path: &str, timeout: Duration) -> std::io::Result<(u16, String)> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut s = TcpStream::connect_timeout(&addr, timeout)?;
    s.set_read_timeout(Some(timeout))?;
    s.set_write_timeout(Some(timeout))?;
    // one write: a server that reads the request once must see all of it
    s.write_all(
        format!("GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n").as_bytes(),
    )?;
    let mut buf = Vec::new();
    s.take(1 << 20).read_to_end(&mut buf)?;
    let text = String::from_utf8_lossy(&buf).into_owned();
    let code = text
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    Ok((code, body))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServerState {
    Up(String),
    Down,
    /// Something else listens on the port.
    Conflict(String),
}

pub fn server_state(port: u16) -> ServerState {
    match http_get_local(port, "/health", Duration::from_secs(3)) {
        Err(_) => ServerState::Down,
        Ok((200, body)) if body.contains("status") || body.contains("ok") => {
            ServerState::Up(body.trim().chars().take(120).collect())
        }
        Ok((code, body)) => ServerState::Conflict(format!(
            "HTTP {code} {}",
            body.trim().chars().take(60).collect::<String>()
        )),
    }
}

pub fn check_server(st: &SetupState, state: &ServerState) -> Check {
    let at = format!("{}:{}", st.host, st.port);
    match state {
        ServerState::Up(_) => Check::new(
            "server",
            "Server",
            Verdict::Pass,
            format!("answering on {at}"),
        ),
        ServerState::Down => {
            let c = Check::new(
                "server",
                "Server",
                Verdict::Info,
                format!("not running on {at}"),
            );
            if st.service {
                c.advice("Start it: systemctl --user start talus")
                    .fix(Fix::StartService)
            } else {
                c.advice("Start it with `talus serve` (or install the service with `talus setup`).")
            }
        }
        ServerState::Conflict(what) => Check::new(
            "server",
            "Server",
            Verdict::Fail,
            format!("port {} is used by another program ({what})", st.port),
        )
        .advice("Stop that program, or pick another port with `talus setup --port N`."),
    }
}

pub fn check_api_key(paths: &Paths, st: &SetupState) -> Check {
    use std::os::unix::fs::PermissionsExt;
    let p = paths.api_key();
    let lan = st.host != "127.0.0.1" && st.host != "localhost";
    match std::fs::metadata(&p) {
        Err(_) if lan => Check::new("api_key", "API key", Verdict::Warn, format!("the server listens on {} without an API key file", st.host))
            .advice("Anyone on your network can use it: set a key in talus-tui (Server > API key) or rerun `talus setup --advanced`."),
        Err(_) => Check::new("api_key", "API key", Verdict::Info, "none (the server answers this computer only)"),
        Ok(md) if md.permissions().mode() & 0o077 != 0 => {
            Check::new("api_key", "API key", Verdict::Warn, format!("{} is readable by other users", paths.tilde(&p)))
                .fix(Fix::ChmodPrivate(p))
        }
        Ok(_) => Check::new("api_key", "API key", Verdict::Pass, format!("{} (readable by you only)", paths.tilde(&p))),
    }
}

pub fn check_service(paths: &Paths, st: &SetupState, sys: &SysInfo) -> Check {
    if !st.service {
        return Check::new(
            "service",
            "Service",
            Verdict::Info,
            "not installed (start the server with `talus serve`)",
        );
    }
    if !paths.unit_file().exists() {
        return Check::new(
            "service",
            "Service",
            Verdict::Fail,
            format!("{} is missing", paths.tilde(&paths.unit_file())),
        )
        .fix(Fix::InstallService);
    }
    if !sys.user_systemd {
        return Check::new(
            "service",
            "Service",
            Verdict::Skip,
            "systemctl --user is not reachable from here",
        );
    }
    let q = |what: &str| -> String {
        std::process::Command::new("systemctl")
            .args(["--user", what, crate::service::UNIT])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    };
    let (enabled, active) = (q("is-enabled"), q("is-active"));
    let sub = std::process::Command::new("systemctl")
        .args([
            "--user",
            "show",
            "-p",
            "SubState",
            "--value",
            crate::service::UNIT,
        ])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    if sub == "auto-restart" || (active == "activating" && sub != "start") {
        return Check::new("service", "Service", Verdict::Fail, "the server keeps failing to start and the service restarts it")
            .advice("See why: journalctl --user -u talus -n 50 (and the other checks above); stop it with systemctl --user stop talus.");
    }
    match (enabled.as_str(), active.as_str()) {
        ("enabled", "active") => {
            Check::new("service", "Service", Verdict::Pass, "enabled and running")
        }
        (_, "failed") => Check::new("service", "Service", Verdict::Fail, "the service failed")
            .advice("See why: journalctl --user -u talus -n 50")
            .fix(Fix::RestartService),
        (e, a) => Check::new("service", "Service", Verdict::Warn, format!("{e}, {a}"))
            .fix(Fix::StartService),
    }
}

pub fn check_disk(st: &SetupState) -> Check {
    match crate::probe::free_bytes(&st.models_dir) {
        None => Check::new(
            "disk",
            "Disk space",
            Verdict::Skip,
            "cannot read the models folder's filesystem",
        ),
        Some(b) if b < 2 << 30 => Check::new(
            "disk",
            "Disk space",
            Verdict::Fail,
            format!("{} free for models", plan::gb(b)),
        )
        .advice("Free some space: prompt caches and logs need room too."),
        Some(b) if b < 20 << 30 => Check::new(
            "disk",
            "Disk space",
            Verdict::Warn,
            format!("{} free for models", plan::gb(b)),
        ),
        Some(b) => Check::new(
            "disk",
            "Disk space",
            Verdict::Pass,
            format!("{} free for models", plan::gb(b)),
        ),
    }
}

/// Is there a newer Talus? Compares the checkout's commit with its upstream branch (network).
pub fn check_update(st: &SetupState) -> Check {
    let git = |args: &[&str]| -> Option<String> {
        let o = std::process::Command::new("git")
            .arg("-C")
            .arg(&st.src)
            .args(args)
            .output()
            .ok()?;
        o.status
            .success()
            .then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let Some(head) = git(&["rev-parse", "HEAD"]) else {
        return Check::new(
            "update",
            "Talus version",
            Verdict::Skip,
            "the source is not a git checkout",
        );
    };
    let Some(remote) = git(&["ls-remote", "origin", "HEAD"])
        .and_then(|s| s.split_whitespace().next().map(str::to_string))
    else {
        return Check::new(
            "update",
            "Talus version",
            Verdict::Skip,
            format!("{} (could not reach GitHub)", &head[..12.min(head.len())]),
        );
    };
    if remote == head {
        Check::new(
            "update",
            "Talus version",
            Verdict::Pass,
            format!("{} (up to date)", &head[..12]),
        )
    } else {
        Check::new(
            "update",
            "Talus version",
            Verdict::Info,
            format!(
                "{} installed, {} available",
                &head[..12],
                &remote[..12.min(remote.len())]
            ),
        )
        .advice("Update with `talus update`.")
    }
}

/// Every check, in order.
pub fn run(ctx: &Ctx) -> Vec<Check> {
    let need_pool = ctx.models.iter().any(|m| m.needs_big_gtt);
    let mut out = hardware(ctx.sys, need_pool);
    out.push(check_rocm(ctx.state, ctx.paths));
    out.push(check_binaries(ctx.paths));
    let Some(st) = ctx.state else {
        out.push(
            Check::new(
                "setup",
                "Installed models",
                Verdict::Warn,
                "no setup found (~/.config/talus/setup.json)",
            )
            .advice("Run `talus setup` to choose and download models."),
        );
        return out;
    };
    if st.models.is_empty() {
        out.push(
            Check::new("setup", "Installed models", Verdict::Warn, "none")
                .advice("Add one with `talus setup`."),
        );
    }
    for m in &ctx.models {
        out.push(check_model_files(m, st, ctx.opts.deep));
        if let Some(c) = check_prep(m, st) {
            out.push(c);
        }
        out.push(check_config(m, ctx.paths, st));
    }
    let server = server_state(st.port);
    out.extend(check_memory(ctx, matches!(server, ServerState::Up(_))));
    out.push(check_server(st, &server));
    out.push(check_api_key(ctx.paths, st));
    out.push(check_service(ctx.paths, st, ctx.sys));
    out.push(check_disk(st));
    if ctx.opts.network {
        out.push(check_update(st));
    }
    out
}

/// The worst verdict (for the exit code).
pub fn worst(checks: &[Check]) -> Verdict {
    checks
        .iter()
        .map(|c| c.verdict)
        .max()
        .unwrap_or(Verdict::Pass)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Choices;

    fn halo_box() -> SysInfo {
        let root = crate::testutil::fake_halo_root("checks", true);
        let mut s = crate::probe::probe(&root);
        std::fs::remove_dir_all(root).ok();
        s.groups = vec!["user1".into(), "video".into(), "render".into()];
        s.kfd_group = Some("render".into());
        s.render_group = Some("render".into());
        s.arch = "x86_64".into();
        s
    }

    #[test]
    fn hardware_on_a_tuned_halo_passes() {
        let s = halo_box();
        let v = hardware(&s, true);
        for c in &v {
            assert_eq!(c.verdict, Verdict::Pass, "{c:?}");
        }
    }

    #[test]
    fn hardware_problems_have_fixes() {
        let mut s = halo_box();
        s.groups = vec!["user1".into()];
        let g = check_groups(&s);
        assert_eq!(g.verdict, Verdict::Fail);
        assert!(
            matches!(&g.fixes[0], Fix::Sudo { argv, .. } if argv[0] == "usermod" && argv[2] == "render")
        );
        assert!(!g.fixes[0].safe());
        s.cmdline = "ro quiet".into();
        s.ttm_pages_limit = None;
        s.ttm_page_pool_size = None;
        let b = check_boot(&s, true);
        assert_eq!(b[0].verdict, Verdict::Fail);
        assert_eq!(b[0].fixes, vec![Fix::BootParams]);
        assert!(!Fix::BootParams.safe());
        // without GLM or MiMo the default pool is fine
        let b = check_boot(&s, false);
        assert_eq!(b[0].verdict, Verdict::Info);
        s.gpus = vec!["gfx1100".into()];
        assert_eq!(check_gpu(&s).verdict, Verdict::Fail);
        s.gpus.clear();
        assert_eq!(check_gpu(&s).verdict, Verdict::Fail);
        s.arch = "aarch64".into();
        assert_eq!(check_os(&s).verdict, Verdict::Fail);
    }

    #[test]
    fn install_checks_find_missing_and_corrupt_files() {
        let home = crate::testutil::tempdir("doctor");
        let paths = Paths::under(&home);
        let mut st = SetupState::new(&paths, home.join("src"), None);
        let m = manifest::get("laguna-s-2.1").unwrap();
        st.models.insert(
            m.id.into(),
            Choices {
                drafters: false,
                ..Default::default()
            },
        );
        let c = check_model_files(m, &st, false);
        assert_eq!(c.verdict, Verdict::Fail);
        assert_eq!(c.fixes.len(), 3, "the three shards");
        assert!(c.fixes.iter().all(Fix::safe));
        let cfg = check_config(m, &paths, &st);
        assert_eq!(cfg.verdict, Verdict::Fail);
        assert_eq!(
            cfg.fixes,
            vec![Fix::RegenerateConfig { model: m.id.into() }]
        );
        assert_eq!(check_rocm(Some(&st), &paths).verdict, Verdict::Fail);
        // a dangling rocm-sdk link (prebuilt install whose SDK moved) is reported
        std::fs::create_dir_all(&paths.data).unwrap();
        std::os::unix::fs::symlink(home.join("gone"), paths.rocm_link()).unwrap();
        assert!(check_rocm(None, &paths)
            .detail
            .contains("points at a missing SDK"));
        assert_eq!(check_binaries(&paths).fixes, vec![Fix::Rebuild]);
        assert_eq!(check_server(&st, &ServerState::Down).verdict, Verdict::Info);
        assert_eq!(
            check_server(&st, &ServerState::Conflict("x".into())).verdict,
            Verdict::Fail
        );
        std::fs::remove_dir_all(home).ok();
    }

    #[test]
    fn run_config_states() {
        let home = crate::testutil::tempdir("cfgstate");
        let paths = Paths::under(&home);
        let mut st = SetupState::new(&paths, home.join("src"), Some(home.join("rocm")));
        let m = manifest::get("qwen3.8-27b").unwrap();
        st.models.insert(m.id.into(), Choices::default());
        let x = crate::exec::Exec::new(false);
        crate::wizard::write_configs(&x, &paths, &st, "s").unwrap();
        // the engine is outside the model folder: without it the check asks for a rebuild
        let c = check_config(m, &paths, &st);
        assert_eq!(c.fixes, vec![Fix::Rebuild], "{c:?}");
        std::fs::create_dir_all(paths.bin()).unwrap();
        std::fs::write(paths.bin().join("talus-cli"), b"").unwrap();
        let c = check_config(m, &paths, &st);
        assert_eq!(
            c.verdict,
            Verdict::Warn,
            "waiting for the model files: {c:?}"
        );
        assert!(c.fixes.is_empty());
        // hand-edited: regenerate
        let p = paths.run_config(m.id);
        let t = std::fs::read_to_string(&p)
            .unwrap()
            .replace("65536", "32768");
        std::fs::write(&p, t).unwrap();
        let c = check_config(m, &paths, &st);
        assert_eq!(c.fixes, vec![Fix::RegenerateConfig { model: m.id.into() }]);
        // the setup moved its models folder: the old config is out of date
        crate::wizard::write_configs(&x, &paths, &st, "s2").unwrap();
        st.models_dir = home.join("elsewhere");
        assert_eq!(check_config(m, &paths, &st).verdict, Verdict::Fail);
        std::fs::remove_file(&p).unwrap();
        assert!(check_config(m, &paths, &st).detail.contains("is missing"));
        std::fs::remove_dir_all(home).ok();
    }

    #[test]
    fn rocm_version_from_wheels_and_tarballs() {
        let d = crate::testutil::tempdir("rocmv");
        let root = d.join("site-packages/_rocm_sdk_devel");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(d.join("site-packages/rocm_sdk_devel-10.2.0a20261014.dist-info"))
            .unwrap();
        assert_eq!(rocm_version(&root).as_deref(), Some("10.2.0a20261014"));
        let t = d.join("tar/rocm");
        std::fs::create_dir_all(t.join(".info")).unwrap();
        std::fs::write(t.join(".info/version"), "7.14.1\n").unwrap();
        assert_eq!(rocm_version(&t).as_deref(), Some("7.14.1"));
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn api_key_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let home = crate::testutil::tempdir("key");
        let paths = Paths::under(&home);
        let mut st = SetupState::new(&paths, home.join("src"), None);
        assert_eq!(check_api_key(&paths, &st).verdict, Verdict::Info);
        st.host = "0.0.0.0".into();
        assert_eq!(check_api_key(&paths, &st).verdict, Verdict::Warn);
        std::fs::create_dir_all(&paths.config).unwrap();
        std::fs::write(paths.api_key(), "k").unwrap();
        std::fs::set_permissions(paths.api_key(), std::fs::Permissions::from_mode(0o644)).unwrap();
        let c = check_api_key(&paths, &st);
        assert_eq!(c.fixes, vec![Fix::ChmodPrivate(paths.api_key())]);
        std::fs::set_permissions(paths.api_key(), std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(check_api_key(&paths, &st).verdict, Verdict::Pass);
        std::fs::remove_dir_all(home).ok();
    }

    #[test]
    fn local_http_probe() {
        use std::net::TcpListener;
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let t = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut req = Vec::new();
            let mut b = [0u8; 512];
            while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = s.read(&mut b).unwrap();
                if n == 0 {
                    break;
                }
                req.extend_from_slice(&b[..n]);
            }
            s.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"ok\"}",
            )
            .unwrap();
        });
        assert!(matches!(server_state(port), ServerState::Up(_)));
        t.join().unwrap();
        // nothing listening now
        assert_eq!(server_state(port), ServerState::Down);
    }
}
