//! `talus setup`: the guided install. Few questions; Enter takes each default.
//!
//! 1. This computer: a verdict per hardware check.
//! 2. Models: the five, with download size, context, fit next to a desktop, release speed and license.
//! 3. Options: drafters on; Qwen3.8-27B's 4-bit prompt weights opt-in (`--advanced`); disk space check.
//! 4. Downloads: resumable, checksum-verified; Flash-Next's pack.
//! 5. System settings: the kernel parameters GLM and MiMo need, shown as a diff, applied only on an explicit "yes".
//! 6. Run configs, the optional user service, start the server, open the TUI.
//!
//! Re-running offers add / remove / repair. `--dry-run` prints every action without doing it. `--yes` takes every
//! default and never touches boot settings unless `--allow-system-changes` is also given.

use crate::boot;
use crate::checks::{self, Verdict};
use crate::config;
use crate::download;
use crate::exec::Exec;
use crate::manifest::{self, Choices, ModelInstall, Prep, Role};
use crate::paths::Paths;
use crate::plan::{self, gb, Fit};
use crate::probe::{self, SysInfo};
use crate::service;
use crate::state::SetupState;
use crate::ui::{self, Prompter};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default)]
pub struct Args {
    pub advanced: bool,
    pub yes: bool,
    pub dry_run: bool,
    pub allow_system_changes: bool,
    /// Install exactly these (ids, numbers or short names).
    pub models: Option<String>,
    pub add: Option<String>,
    pub remove: Option<String>,
    pub models_dir: Option<PathBuf>,
    pub src: Option<PathBuf>,
    pub rocm_sdk: Option<PathBuf>,
    pub port: Option<u16>,
    /// Listen on the network (with an API key).
    pub bind_lan: bool,
    pub no_start: bool,
    pub no_service: bool,
    /// Continue on hardware that cannot run Talus (downloads and configs only).
    pub force: bool,
    /// Skip downloads (configs only; for tests and pre-seeded model folders).
    pub no_download: bool,
    /// Also download opt-in drafters (Laguna-S-2.1's DFlash drafter; license unclear).
    pub opt_in_drafters: bool,
}

const STEPS: usize = 6;

pub fn now_stamp() -> (String, String) {
    // date for humans, stamp for file names; from the system clock without a date crate
    let out = std::process::Command::new("date")
        .args(["+%Y-%m-%d %Y%m%dT%H%M%S"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let mut it = out.split_whitespace();
    (
        it.next().unwrap_or("today").to_string(),
        it.next().unwrap_or("backup").to_string(),
    )
}

/// The ROCm root of a TheRock venv (`rocm-sdk path --root`).
pub fn rocm_root_from_venv(venv: &Path) -> Option<PathBuf> {
    let o = std::process::Command::new(venv.join("bin/rocm-sdk"))
        .args(["path", "--root"])
        .output()
        .ok()?;
    let p = PathBuf::from(String::from_utf8_lossy(&o.stdout).trim());
    (o.status.success() && p.join("lib").is_dir()).then_some(p)
}

fn print_checks(cs: &[checks::Check]) {
    for c in cs {
        println!("  {} {:<26} {}", ui::tag(c.verdict), c.title, c.detail);
        if matches!(c.verdict, Verdict::Warn | Verdict::Fail) {
            if let Some(a) = &c.advice {
                println!("  {:<6} {:<26} {}", "", "", ui::dim(a));
            }
        }
    }
}

fn speed(m: &ModelInstall) -> String {
    format!("{:.0} / {:.0} tok/s", m.chat_tps, m.prompt_tps)
}

fn short_license(l: &str) -> &str {
    l.split(" (").next().unwrap_or(l)
}

/// The model table.
pub fn model_table(st: Option<&SetupState>, sys: &SysInfo, pool_ok: bool) -> Vec<String> {
    let mut lines = vec![format!(
        "  {:<3}{:<20}{:>9}  {:<8}{:<15}{:<17}{}",
        "#", "Model", "Download", "Context", "With desktop", "Answer / prompt", "License"
    )];
    for (i, m) in manifest::MODELS.iter().enumerate() {
        let spec = m.spec();
        let installed = st.map(|s| s.models.contains_key(m.id)).unwrap_or(false);
        let size = gb(m.download_bytes(&Choices::default()));
        let fit = m.desktop_fit.label();
        let row = format!(
            "  {:<3}{:<20}{:>9}  {:<8}{:<15}{:<17}{}{}",
            i + 1,
            spec.name,
            size,
            m.context,
            fit,
            speed(m),
            short_license(spec.license),
            if installed { "  (installed)" } else { "" }
        );
        lines.push(row);
        if let Fit::NotNow(why) = plan::live_fit(m, sys, pool_ok) {
            lines.push(format!(
                "     {}",
                ui::dim(&format!("on this machine now: {why}"))
            ));
        }
    }
    lines
}

fn ask_models(
    p: &mut Prompter,
    st: Option<&SetupState>,
    sys: &SysInfo,
    pool_ok: bool,
) -> Vec<&'static ModelInstall> {
    for l in model_table(st, sys, pool_ok) {
        println!("{l}");
    }
    ui::note(&format!(
        "Speeds: tokens per second in chat (answer) and reading a 2K prompt, release {} on a 70 W HP ZBook Ultra G1a.",
        manifest::RELEASE
    ));
    ui::note("One model serves at a time; install several to switch between them in talus-tui.");
    let have = st.map(|s| !s.models.is_empty()).unwrap_or(false);
    let default = if have {
        "none".to_string()
    } else {
        "1".to_string()
    };
    loop {
        let ans = p.ask("Models to install (numbers like 1,5 or all):", &default);
        if ans == "none" {
            return vec![];
        }
        match manifest::parse_list(&ans) {
            Ok(v) if !v.is_empty() => return v,
            Ok(_) => return vec![],
            Err(e) => {
                ui::warn(&e);
                if !p.interactive() {
                    return vec![];
                }
            }
        }
    }
}

fn ask_choices(
    p: &mut Prompter,
    m: &ModelInstall,
    advanced: bool,
    prev: Option<Choices>,
    opt_in_drafters: bool,
) -> Choices {
    let mut c = prev.unwrap_or_default();
    let spec = m.spec();
    if !advanced {
        if opt_in_drafters {
            c.opt_drafter = true;
        }
        return c;
    }
    println!("  {}", ui::bold(spec.name));
    if m.has_role(Role::Drafter) || m.has_role(Role::DraftVocab) {
        c.drafters = p.confirm(
            "Drafters (speculative decoding: same output, much faster answers)?",
            c.drafters,
        );
    }
    if c.drafters && m.has_role(Role::DrafterOptIn) {
        ui::warn(manifest::OPT_DRAFTER_NOTE);
        c.opt_drafter = p.confirm("Download it anyway?", c.opt_drafter || opt_in_drafters);
    }
    if m.has_role(Role::W4i) {
        if let Some(o) = spec.option("w4_prompt") {
            ui::say(&format!("{}: {}", o.label, o.description));
            ui::note(o.tradeoff);
        }
        c.w4i = p.confirm(
            "Download the 4-bit prompt weights (2.1 GB; turn them on later in talus-tui)?",
            c.w4i,
        );
    }
    if m.has_role(Role::W4iFast) {
        if let Some(o) = spec.option("w4_fast") {
            ui::warn(&format!("{}: {}", o.label, o.tradeoff));
        }
        c.w4i_fast = p.confirm(
            "Download the Fast-tier prompt weights anyway (8.6 GB)?",
            c.w4i_fast,
        );
    }
    c
}

/// Remove models: their files, their run configs, and their entries in setup.json.
fn remove_models(
    x: &Exec,
    p: &mut Prompter,
    paths: &Paths,
    st: &mut SetupState,
    list: &[&ModelInstall],
) -> Result<(), String> {
    for m in list {
        if !st.models.contains_key(m.id) {
            ui::warn(&format!("{} is not installed", m.spec().name));
            continue;
        }
        let dir = st.model_dir(m.id);
        let size: u64 = walk_size(&dir);
        if !p.confirm(
            &format!(
                "Delete {} ({} in {})?",
                m.spec().name,
                gb(size),
                paths.tilde(&dir)
            ),
            true,
        ) {
            continue;
        }
        x.remove_dir(&dir).map_err(|e| e.to_string())?;
        x.remove_file(&paths.run_config(m.id))
            .map_err(|e| e.to_string())?;
        // the TUI's last applied choice may name it: drop it, the server then starts with the default model
        let active = paths.control_dir().join("active.json");
        if std::fs::read_to_string(&active)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| {
                v.get("model_name")
                    .and_then(|n| n.as_str())
                    .map(|n| n == m.id)
            })
            .unwrap_or(false)
        {
            x.remove_file(&active).map_err(|e| e.to_string())?;
        }
        st.models.remove(m.id);
        if st.default_model.as_deref() == Some(m.id) {
            st.default_model = st.models.keys().next().cloned();
        }
        ui::say(&format!("removed {}", m.spec().name));
    }
    Ok(())
}

fn walk_size(p: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(p) else {
        return std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    };
    rd.flatten()
        .map(|e| {
            let ft = e.file_type().ok();
            if ft.map(|t| t.is_dir()).unwrap_or(false) {
                walk_size(&e.path())
            } else {
                e.metadata().map(|m| m.len()).unwrap_or(0)
            }
        })
        .sum()
}

fn random_key() -> String {
    use std::io::Read;
    let mut b = [0u8; 24];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        let _ = f.read_exact(&mut b);
    }
    format!("tk-{}", download::hex(&b))
}

/// Write every installed model's run config. Existing files are backed up when they change.
pub fn write_configs(x: &Exec, paths: &Paths, st: &SetupState, stamp: &str) -> Result<(), String> {
    x.mkdir(&paths.control_dir()).map_err(|e| e.to_string())?;
    for id in st.models.keys() {
        let (Some(m), Some(text)) = (manifest::get(id), config::expected_text(paths, st, id))
        else {
            continue;
        };
        let p = paths.run_config(id);
        let old = std::fs::read_to_string(&p).ok();
        if old.as_deref() == Some(text.as_str()) {
            continue;
        }
        if old.is_some() {
            x.backup(&p, stamp).map_err(|e| e.to_string())?;
        }
        // 0600: a network-reachable config carries the API key
        x.write(
            &p,
            &text,
            0o600,
            &format!("run config of {}", m.spec().name),
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn install_service(x: &Exec, paths: &Paths) -> Result<(), String> {
    x.write(
        &paths.unit_file(),
        &service::unit_text(paths),
        0o644,
        "systemd user service",
    )
    .map_err(|e| e.to_string())?;
    x.run(
        &["systemctl".into(), "--user".into(), "daemon-reload".into()],
        "load the service",
    )?;
    x.run(
        &[
            "systemctl".into(),
            "--user".into(),
            "enable".into(),
            service::UNIT.into(),
        ],
        "start it at login",
    )
}

/// Boot settings: show what is missing and why, the exact change, then apply only on a typed "yes".
pub fn system_settings(
    x: &Exec,
    p: &mut Prompter,
    sys: &SysInfo,
    models: &[&ModelInstall],
    allow: bool,
    stamp: (&str, &str),
) -> Result<bool, String> {
    let need_pool = models.iter().any(|m| m.needs_big_gtt);
    let st = boot::status(sys);
    let missing: Vec<boot::Param> = st
        .iter()
        .filter(|s| !s.ok)
        .filter(|s| need_pool || s.param.key == "amdgpu.lockup_timeout")
        .map(|s| s.param.clone())
        .collect();
    for s in &st {
        let v = if s.ok {
            Verdict::Pass
        } else if need_pool || s.param.key == "amdgpu.lockup_timeout" {
            Verdict::Warn
        } else {
            Verdict::Info
        };
        println!("  {} {:<22} now: {}", ui::tag(v), s.param.key, s.current);
    }
    if missing.is_empty() {
        ui::say("Nothing to change for the models you chose.");
        return Ok(false);
    }
    if need_pool {
        ui::say("GLM-5.3-Flash and MiMo-V2.6-Flash do not fit until the GPU may use 120 GiB of the shared memory.");
    } else {
        ui::say(
            "Your models fit as is; the GPU timeout is still worth raising for very long prompts.",
        );
    }
    for m in &missing {
        ui::say(&format!("{}: {}", ui::bold(&m.arg()), m.why));
    }
    let root = Path::new("/");
    let have = |c: &str| {
        std::process::Command::new("sh")
            .args([
                "-c",
                &format!("command -v {c} >/dev/null 2>&1 || test -x /usr/sbin/{c}"),
            ])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };
    let loader = boot::detect_loader(sys, root, &have);
    let existing =
        std::fs::read_to_string(Path::new(boot::DROPIN_DIR).join(boot::DROPIN_NAME)).ok();
    let plan = boot::plan(&loader, &missing, existing.as_deref(), stamp.0, stamp.1);
    if let Some(d) = &plan.diff {
        ui::say("The change (a new file; nothing else is edited):");
        ui::block(d);
    }
    if plan.steps.is_empty() {
        ui::warn("The installer cannot apply this on your boot loader. Do it yourself:");
        ui::block(&plan.manual);
        return Ok(false);
    }
    ui::say("What would run (backups first; nothing reboots):");
    for s in &plan.steps {
        ui::say(&format!("  {}", s.describe()));
    }
    let want = if p.interactive() {
        p.confirm("Make this change now? (No shows the lines to copy)", false)
            && p.confirm_typed("This changes how your computer boots.", "yes", false)
    } else {
        p.confirm_typed("Apply the kernel parameters?", "yes", allow)
    };
    if !want {
        ui::say("Skipped. To do it yourself, run:");
        ui::block(&plan.manual);
        return Ok(false);
    }
    for s in &plan.steps {
        match s {
            boot::Step::Sudo(a) => x.sudo(a, "kernel parameters")?,
            boot::Step::WriteRoot { path, content } => {
                x.sudo_write(path, content, "kernel parameters drop-in")?
            }
        }
    }
    ui::say(&ui::good("Done. Reboot when convenient for the new settings to take effect (the installer never reboots)."));
    Ok(true)
}

fn start_server(x: &Exec, paths: &Paths, st: &SetupState, use_service: bool) -> Result<(), String> {
    if use_service {
        x.run(
            &[
                "systemctl".into(),
                "--user".into(),
                "restart".into(),
                service::UNIT.into(),
            ],
            "start the server",
        )?;
    } else {
        let log = paths.state.join("server.log");
        x.mkdir(&paths.state).map_err(|e| e.to_string())?;
        x.run(
            &[
                "sh".into(),
                "-c".into(),
                format!(
                    "nohup '{}' serve >>'{}' 2>&1 </dev/null &",
                    paths.bin().join("talus-setup").display(),
                    log.display()
                ),
            ],
            "start the server in the background",
        )?;
    }
    if x.dry_run {
        return Ok(());
    }
    ui::say("Loading the model (a few minutes the first time) ...");
    for i in 0..180 {
        if let checks::ServerState::Up(_) = checks::server_state(st.port) {
            ui::say(&ui::good(&format!(
                "The server answers on http://{}:{}/v1",
                st.host, st.port
            )));
            return Ok(());
        }
        if i % 6 == 5 {
            ui::note(&format!("still loading ({} s)", (i + 1) * 5));
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
    Err(format!(
        "the server did not answer within 15 minutes; see {} and `talus doctor`",
        paths.tilde(&paths.state)
    ))
}

pub fn run(a: Args) -> i32 {
    match run_inner(a) {
        Ok(()) => 0,
        Err(e) => {
            ui::error(&e);
            1
        }
    }
}

fn run_inner(a: Args) -> Result<(), String> {
    let paths = Paths::from_env();
    let x = Exec::new(a.dry_run);
    let mut p = Prompter::new(a.yes);
    let stamp = now_stamp();
    ui::header(if a.dry_run {
        "setup (dry run: nothing is changed)"
    } else {
        "setup"
    });
    let prev = SetupState::load(&paths)?;
    let mut st = prev.clone().unwrap_or_else(|| {
        let src = a.src.clone().unwrap_or_else(|| paths.src());
        SetupState::new(&paths, src, None)
    });
    if let Some(s) = &a.src {
        st.src = s.clone();
    }
    if let Some(r) = &a.rocm_sdk {
        st.rocm_sdk = Some(r.clone());
    } else if st.rocm_sdk.is_none() {
        st.rocm_sdk = std::fs::canonicalize(paths.rocm_link())
            .ok()
            .filter(|p| p.join("lib").is_dir())
            .or_else(|| rocm_root_from_venv(&paths.rocm_venv()))
            .or_else(|| {
                std::env::var_os("ROCM_SDK")
                    .map(PathBuf::from)
                    .filter(|p| p.join("lib").is_dir())
            });
    }
    if let Some(d) = &a.models_dir {
        st.models_dir = d.clone();
    }
    if let Some(port) = a.port {
        st.port = port;
    }

    // 1. this computer
    ui::step(1, STEPS, "This computer");
    let sys = probe::probe(Path::new("/"));
    let hw = checks::hardware(&sys, false);
    print_checks(&hw[..4]);
    let blocking = hw[..4]
        .iter()
        .any(|c| c.verdict == Verdict::Fail && c.id != "groups");
    if blocking && !a.force {
        return Err("Talus cannot run on this computer (see above). `--force` continues anyway (downloads and configs only).".into());
    }
    let pool_ok = boot::status(&sys)
        .iter()
        .find(|s| s.param.key == "ttm.pages_limit")
        .map(|s| s.ok)
        .unwrap_or(false);

    // 2. models
    ui::step(2, STEPS, "Models");
    if let Some(r) = &a.remove {
        let list = manifest::parse_list(r)?;
        remove_models(&x, &mut p, &paths, &mut st, &list)?;
        if !a.dry_run {
            st.save(&paths).map_err(|e| e.to_string())?;
        }
        write_configs(&x, &paths, &st, &stamp.1)?;
        ui::say("Done.");
        return Ok(());
    }
    let mut chosen: Vec<&ModelInstall> = if let Some(l) = a.models.as_ref().or(a.add.as_ref()) {
        let v = manifest::parse_list(l)?;
        for l in model_table(prev.as_ref(), &sys, pool_ok) {
            println!("{l}");
        }
        ui::say(&format!(
            "Chosen: {}",
            v.iter()
                .map(|m| m.spec().name)
                .collect::<Vec<_>>()
                .join(", ")
        ));
        v
    } else if prev.as_ref().map(|s| !s.models.is_empty()).unwrap_or(false) && p.interactive() {
        let installed: Vec<&str> = st
            .models
            .keys()
            .filter_map(|k| manifest::get(k))
            .map(|m| m.spec().name)
            .collect();
        ui::say(&format!("Installed: {}", installed.join(", ")));
        ui::say("1 add models   2 remove models   3 check and repair   4 change settings only");
        match p.ask("What would you like to do?", "1").as_str() {
            "2" => {
                let l = p.ask("Models to remove (numbers):", "none");
                if l != "none" {
                    let list = manifest::parse_list(&l)?;
                    remove_models(&x, &mut p, &paths, &mut st, &list)?;
                    if !a.dry_run {
                        st.save(&paths).map_err(|e| e.to_string())?;
                    }
                }
                return Ok(());
            }
            "3" => {
                return match crate::doctor::run(crate::doctor::Args {
                    fix: true,
                    dry_run: a.dry_run,
                    allow_system_changes: a.allow_system_changes,
                    ..Default::default()
                }) {
                    0 => Ok(()),
                    _ => Err("some problems remain (see above)".into()),
                };
            }
            "4" => vec![],
            _ => ask_models(&mut p, Some(&st), &sys, pool_ok),
        }
    } else {
        ask_models(&mut p, prev.as_ref(), &sys, pool_ok)
    };
    chosen.retain(|m| {
        if let Fit::NotNow(why) = plan::live_fit(m, &sys, true) {
            ui::warn(&format!(
                "{}: {why}. Installing it anyway; it starts once that is fixed.",
                m.spec().name
            ));
        }
        true
    });
    for m in &chosen {
        if m.id == "qwen3.8-flash-next" {
            ui::note(&format!("{} license: {}", m.spec().name, m.spec().license));
        }
    }

    // 3. options and disk
    ui::step(3, STEPS, "Options");
    if a.advanced {
        let d = p.ask(
            "Folder for model files:",
            &st.models_dir.display().to_string(),
        );
        st.models_dir = PathBuf::from(d);
    }
    let mut picks: Vec<(&ModelInstall, Choices)> = Vec::new();
    for m in &chosen {
        let c = ask_choices(
            &mut p,
            m,
            a.advanced,
            st.models.get(m.id).copied(),
            a.opt_in_drafters,
        );
        picks.push((m, c));
    }
    if !a.advanced && !chosen.is_empty() {
        ui::say("Drafters on (speculative decoding: same output, faster answers).");
        if chosen.iter().any(|m| m.has_role(Role::W4i)) {
            ui::note("Qwen3.8-27B's 4-bit prompt weights are an opt-in: `talus setup --advanced` offers them.");
        }
        if chosen.iter().any(|m| m.has_role(Role::DrafterOptIn)) && !a.opt_in_drafters {
            ui::note("Laguna-S-2.1 drafts with prompt lookup. Its DFlash drafter is an opt-in (license unclear): \
                      `talus setup --advanced` explains, `--opt-in-drafters` takes it.");
        }
    }
    let need = plan::disk_need(&picks, &st.models_dir);
    let free = probe::free_bytes(&st.models_dir).unwrap_or(0);
    if !chosen.is_empty() {
        ui::say(&format!(
            "Disk: {} to download{}, {} free in {}",
            gb(need.remaining()),
            if need.present > 0 {
                format!(" ({} already here)", gb(need.present))
            } else {
                String::new()
            },
            gb(free),
            paths.tilde(&st.models_dir)
        ));
        if !need.fits(free) && a.dry_run {
            ui::warn(&format!(
                "not enough disk space ({} needed with a {} margin, {} free); a real run stops here. Continuing the dry run.",
                gb(need.remaining()),
                gb(need.margin()),
                gb(free)
            ));
        } else if !need.fits(free) {
            return Err(format!(
                "not enough disk space: {} needed (with a {} margin), {} free. Free space or choose another folder with --models-dir DIR.",
                gb(need.remaining()),
                gb(need.margin()),
                gb(free)
            ));
        }
    }
    if a.bind_lan
        || (a.advanced
            && p.confirm(
                "Make the server reachable from other computers on your network (with an API key)?",
                st.host != "127.0.0.1",
            ))
    {
        st.host = "0.0.0.0".into();
        if !paths.api_key().exists() {
            let k = random_key();
            x.write(&paths.api_key(), &format!("{k}\n"), 0o600, "API key")
                .map_err(|e| e.to_string())?;
            ui::say(&format!(
                "A new API key is in {} (readable by you only). Clients: export TALUS_API_KEY=$(cat {}) and send \
                 it as Authorization: Bearer; talus-tui reads TALUS_API_KEY too.",
                paths.tilde(&paths.api_key()),
                paths.tilde(&paths.api_key())
            ));
        }
    }

    // 4. downloads
    ui::step(4, STEPS, "Downloads");
    if chosen.is_empty() {
        ui::say("No new models.");
    } else if a.no_download {
        ui::say(
            "Skipped (--no-download): `talus doctor --fix` or `talus setup` downloads them later.",
        );
    }
    if download::token().is_some() {
        ui::note("Using HF_TOKEN for Hugging Face rate limits (not shown, not stored).");
    }
    let mut cache = Default::default();
    for (m, c) in &picks {
        let dir = st.model_dir(m.id);
        ui::say(&ui::bold(m.spec().name));
        for f in m.selected_files(c) {
            if a.dry_run || a.no_download {
                let have = plan::bytes_present(&dir, f);
                let what = if have == f.size {
                    "check".to_string()
                } else if have > 0 {
                    format!("resume ({} of {} here)", gb(have), gb(f.size))
                } else {
                    format!("download {}", gb(f.size))
                };
                if a.dry_run {
                    ui::dry(&format!(
                        "{what}: {} -> {}  (sha256 {}...)",
                        download::resolve_url(&download::endpoint(), f.repo, f.path),
                        paths.tilde(&download::dest_of(&dir, f)),
                        &f.sha256[..16]
                    ));
                }
                continue;
            }
            download::fetch(
                f,
                &dir,
                &download::FetchOpts {
                    quiet: false,
                    check_published: true,
                },
                &mut cache,
            )?;
        }
        if m.prep == Prep::FlashNextPack && !a.no_download {
            crate::prepare::prepare_flash_next(&x, &paths, &st, m, c)?;
        }
        st.models.insert(m.id.to_string(), *c);
        if st.default_model.is_none() {
            st.default_model = Some(m.id.to_string());
        }
    }

    // 5. system settings
    ui::step(5, STEPS, "System settings");
    let all: Vec<&ModelInstall> = st.models.keys().filter_map(|k| manifest::get(k)).collect();
    let changed_boot = system_settings(
        &x,
        &mut p,
        &sys,
        &all,
        a.allow_system_changes,
        (&stamp.0, &stamp.1),
    )?;
    if all
        .iter()
        .any(|m| m.desktop_fit == manifest::DesktopFit::HeadlessOnly)
        && sys.desktop_active
    {
        ui::warn(
            "GLM-5.3-Flash needs the machine without a desktop session. To boot to the console:",
        );
        ui::block("sudo systemctl set-default multi-user.target   # undo: sudo systemctl set-default graphical.target");
    }
    if let Some(c) = hw
        .iter()
        .find(|c| c.id == "groups" && c.verdict == Verdict::Fail)
    {
        ui::warn(&c.detail);
        if let Some(checks::Fix::Sudo { argv, after }) = c.fixes.first() {
            ui::block(&format!("sudo {}\n# {after}", boot::shell_join(argv)));
        }
    }

    // 6. config, service, start
    ui::step(6, STEPS, "Configure and start");
    if st.models.is_empty() {
        ui::say("No models installed; run `talus setup` again to add one.");
        return Ok(());
    }
    if st.rocm_sdk.is_none() {
        ui::warn("No ROCm SDK found: the run configs assume /opt/rocm. Run install.sh, or pass --rocm-sdk DIR.");
    }
    write_configs(&x, &paths, &st, &stamp.1)?;
    if !a.dry_run {
        st.save(&paths).map_err(|e| e.to_string())?;
    } else {
        ui::dry(&format!(
            "write {} (what was installed)",
            paths.tilde(&paths.setup_json())
        ));
    }
    x.symlink(
        &paths.bin().join("talus-setup"),
        &paths.user_bin.join("talus"),
    )
    .map_err(|e| e.to_string())?;
    x.symlink(
        &paths.bin().join("talus-tui"),
        &paths.user_bin.join("talus-tui"),
    )
    .map_err(|e| e.to_string())?;
    let mut use_service = st.service && paths.unit_file().exists();
    // a service that cannot start would only restart in a loop: offer it once the first model can run here
    let first_ready = st
        .default_model
        .as_deref()
        .and_then(manifest::get)
        .map(|m| {
            let c = st.models.get(m.id).copied().unwrap_or_default();
            let dir = st.model_dir(m.id);
            m.selected_files(&c).iter().all(|f| {
                plan::bytes_present(&dir, f) == f.size && dir.join(f.local_name()).exists()
            })
        })
        .unwrap_or(false);
    let service_blocker = if !sys.has_gfx1151() {
        Some("there is no gfx1151 GPU here")
    } else if !first_ready && !a.dry_run {
        Some("the first model's files are not all downloaded")
    } else {
        None
    };
    if !a.no_service && !use_service && service_blocker.is_some() {
        ui::note(&format!(
            "No service yet: {}. `talus setup` offers it again.",
            service_blocker.unwrap_or_default()
        ));
    } else if !a.no_service && !use_service {
        if sys.user_systemd {
            if p.confirm(
                "Install a user service so the server starts when you log in?",
                true,
            ) {
                install_service(&x, &paths)?;
                st.service = true;
                use_service = true;
                if !a.dry_run {
                    st.save(&paths).map_err(|e| e.to_string())?;
                }
                ui::note("To start it at boot without logging in: loginctl enable-linger");
            }
        } else {
            ui::note("systemd --user is not available here; start the server with `talus serve`.");
        }
    }
    let first = st.default_model.clone().and_then(|m| manifest::get(&m));
    let fits_now = first
        .map(|m| !matches!(plan::live_fit(m, &sys, pool_ok), Fit::NotNow(_)))
        .unwrap_or(false);
    let can_start =
        !a.no_start && sys.has_gfx1151() && paths.bin().join("talus-server").exists() && fits_now;
    if !can_start && !a.no_start {
        if changed_boot || first.map(|m| m.needs_big_gtt && !pool_ok).unwrap_or(false) {
            ui::say("Reboot first (the kernel parameters apply at boot), then start the server with `talus serve`.");
        } else if !fits_now {
            ui::say("The first model does not fit as the machine is now (see above); start it later with `talus serve`.");
        } else {
            ui::note("Not starting the server here (no gfx1151 GPU or no build).");
        }
    } else if can_start
        && p.confirm(
            &format!(
                "Start the server now with {}?",
                first.map(|m| m.spec().name).unwrap_or("?")
            ),
            true,
        )
    {
        start_server(&x, &paths, &st, use_service)?;
        if !a.dry_run && p.interactive() && p.confirm("Open talus-tui (the dashboard)?", true) {
            let _ = std::process::Command::new(paths.bin().join("talus-tui"))
                .args(["--server", &format!("http://127.0.0.1:{}", st.port)])
                .status();
        }
    }
    println!();
    ui::say(&ui::bold("Next"));
    ui::say(&format!(
        "API        http://{}:{}/v1  (OpenAI and Anthropic compatible)",
        if st.host == "0.0.0.0" {
            "<this-machine>"
        } else {
            "127.0.0.1"
        },
        st.port
    ));
    ui::say("Dashboard  talus-tui");
    ui::say("Health     talus doctor        (talus doctor --report for a GitHub issue)");
    ui::say("Update     talus update");
    ui::say("Models     talus setup         (add, remove, repair)");
    if a.dry_run {
        println!();
        ui::say("Dry run: every step above was printed, nothing was changed.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_lists_five_models_with_fit_and_license() {
        let sys = SysInfo {
            mem_total_kib: 125 * 1048576,
            mem_avail_kib: 116 * 1048576,
            ..Default::default()
        };
        let t = model_table(None, &sys, false).join("\n");
        for n in [
            "Qwen3.8-27B",
            "Qwen3.8-Flash-Next",
            "GLM-5.3-Flash",
            "MiMo-V2.6-Flash",
            "Laguna-S-2.1",
        ] {
            assert!(t.contains(n), "{n}");
        }
        assert!(t.contains("headless only") && t.contains("tight") && t.contains("Apache-2.0"));
        assert!(t.contains("needs the GPU memory pool raised"));
        assert!(
            t.contains("Qwen Community License 1.0") && !t.contains("commercial services need")
        );
    }

    #[test]
    fn configs_and_removal_in_a_fake_home() {
        let home = crate::testutil::tempdir("wiz");
        let paths = Paths::under(&home);
        let mut st = SetupState::new(&paths, home.join("src"), Some(home.join("rocm")));
        st.models.insert("qwen3.8-27b".into(), Choices::default());
        st.models.insert("laguna-s-2.1".into(), Choices::default());
        st.default_model = Some("qwen3.8-27b".into());
        let x = Exec::new(false);
        write_configs(&x, &paths, &st, "s1").unwrap();
        let q = std::fs::read_to_string(paths.run_config("qwen3.8-27b")).unwrap();
        assert!(q.contains("\"model_name\": \"qwen3.8-27b\""));
        assert!(!q.contains("api_key"));
        // unchanged: no backup; changed: backup kept
        write_configs(&x, &paths, &st, "s2").unwrap();
        assert!(!paths.control_dir().join("qwen3.8-27b.json.bak-s2").exists());
        st.models.insert(
            "qwen3.8-27b".into(),
            Choices {
                drafters: false,
                ..Default::default()
            },
        );
        write_configs(&x, &paths, &st, "s3").unwrap();
        assert!(paths.control_dir().join("qwen3.8-27b.json.bak-s3").exists());
        // LAN: the key goes into the configs
        st.host = "0.0.0.0".into();
        std::fs::write(paths.api_key(), "tk-abc\n").unwrap();
        write_configs(&x, &paths, &st, "s4").unwrap();
        assert!(std::fs::read_to_string(paths.run_config("laguna-s-2.1"))
            .unwrap()
            .contains("\"api_key\": \"tk-abc\""));
        // remove Laguna (and an active.json that names it)
        std::fs::create_dir_all(st.model_dir("laguna-s-2.1")).unwrap();
        std::fs::write(st.model_dir("laguna-s-2.1").join("x.gguf"), b"12345").unwrap();
        std::fs::write(
            paths.control_dir().join("active.json"),
            r#"{"model_name":"laguna-s-2.1"}"#,
        )
        .unwrap();
        let mut p = Prompter::new(true);
        remove_models(
            &x,
            &mut p,
            &paths,
            &mut st,
            &[manifest::get("laguna-s-2.1").unwrap()],
        )
        .unwrap();
        assert!(!st.model_dir("laguna-s-2.1").exists());
        assert!(!paths.run_config("laguna-s-2.1").exists());
        assert!(!paths.control_dir().join("active.json").exists());
        assert_eq!(st.models.len(), 1);
        std::fs::remove_dir_all(home).ok();
    }

    #[test]
    fn keys_look_random() {
        let a = random_key();
        assert!(a.starts_with("tk-") && a.len() == 51);
        assert_ne!(a, random_key());
        assert!(a.len() >= talus_presets::MIN_API_KEY_LEN);
    }
}
