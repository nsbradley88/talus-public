//! `talus doctor`: run the checks, print a verdict per line, and (with `--fix`) apply the fixes step by step.
//!
//! Safe fixes (download a bad file again, rewrite a run config, restart the service, tighten a key file's mode) are
//! applied after a yes (or straight away with `--yes`). Anything that needs sudo is shown and asked for; boot
//! settings go through the same show-the-diff, type-yes flow as setup, and never happen with `--yes` alone.

use crate::checks::{self, Check, Fix, Verdict};
use crate::exec::Exec;
use crate::manifest;
use crate::paths::Paths;
use crate::probe;
use crate::report;
use crate::state::SetupState;
use crate::ui::{self, Prompter};
use crate::wizard;
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct Args {
    pub fix: bool,
    pub yes: bool,
    pub report: bool,
    pub deep: bool,
    pub offline: bool,
    pub dry_run: bool,
    pub allow_system_changes: bool,
    /// Only the hardware checks (what `talus setup` shows first).
    pub hw_only: bool,
}

pub fn print(cs: &[Check]) {
    for c in cs {
        println!("  {} {:<30} {}", ui::tag(c.verdict), c.title, c.detail);
        if matches!(c.verdict, Verdict::Warn | Verdict::Fail) {
            if let Some(a) = &c.advice {
                println!("  {:<6} {:<30} {}", "", "", ui::dim(a));
            }
            for f in &c.fixes {
                println!(
                    "  {:<6} {:<30} {} {}",
                    "",
                    "",
                    ui::accent("fix:"),
                    ui::dim(&format!(
                        "{}{}",
                        f.describe(),
                        if f.safe() { "" } else { " (asks first)" }
                    ))
                );
            }
        }
    }
}

fn talus_version(st: Option<&SetupState>) -> String {
    let Some(st) = st else {
        return "unknown (no setup.json)".into();
    };
    std::process::Command::new("git")
        .arg("-C")
        .arg(&st.src)
        .args(["log", "-1", "--format=%h %cs"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| format!("unknown ({} is not a git checkout)", st.src.display()))
}

fn apply(
    fix: &Fix,
    x: &Exec,
    p: &mut Prompter,
    paths: &Paths,
    st: &mut SetupState,
    sys: &probe::SysInfo,
    a: &Args,
) -> Result<(), String> {
    let (date, stamp) = wizard::now_stamp();
    match fix {
        Fix::Redownload { model, file } => {
            let m = manifest::get(model).ok_or("unknown model")?;
            let f = m
                .files
                .iter()
                .find(|f| f.local_name() == file)
                .ok_or("unknown file")?;
            let dir = st.model_dir(model);
            if x.dry_run {
                ui::dry(&format!(
                    "download {} again into {} (resumes a partial file)",
                    file,
                    paths.tilde(&dir)
                ));
                return Ok(());
            }
            let mut cache = Default::default();
            crate::download::fetch(
                f,
                &dir,
                &crate::download::FetchOpts {
                    quiet: false,
                    check_published: true,
                },
                &mut cache,
            )
            .map(|_| ())
        }
        Fix::Prepare { model } => {
            let m = manifest::get(model).ok_or("unknown model")?;
            let c = st.models.get(model).copied().unwrap_or_default();
            crate::prepare::prepare_flash_next(x, paths, st, m, &c)
        }
        Fix::RegenerateConfig { .. } => wizard::write_configs(x, paths, st, &stamp),
        Fix::InstallService => wizard::install_service(x, paths),
        Fix::StartService => x.run(
            &[
                "systemctl".into(),
                "--user".into(),
                "start".into(),
                crate::service::UNIT.into(),
            ],
            "start the server",
        ),
        Fix::RestartService => x.run(
            &[
                "systemctl".into(),
                "--user".into(),
                "restart".into(),
                crate::service::UNIT.into(),
            ],
            "restart the server",
        ),
        Fix::ChmodPrivate(path) => x.run(
            &["chmod".into(), "600".into(), path.display().to_string()],
            "key readable by you only",
        ),
        Fix::BootParams => {
            let models: Vec<&manifest::ModelInstall> =
                st.models.keys().filter_map(|k| manifest::get(k)).collect();
            wizard::system_settings(x, p, sys, &models, a.allow_system_changes, (&date, &stamp))
                .map(|_| ())
        }
        Fix::Sudo { argv, after } => {
            ui::say(&format!(
                "This needs administrator rights: sudo {}",
                crate::boot::shell_join(argv)
            ));
            let go = if p.interactive() {
                p.confirm("Run it now?", false)
            } else {
                p.confirm_typed("Run it?", "yes", a.allow_system_changes)
            };
            if go {
                x.sudo(argv, "system change")?;
                ui::say(after);
            } else {
                ui::say("Skipped. To do it yourself:");
                ui::block(&format!("sudo {}", crate::boot::shell_join(argv)));
            }
            Ok(())
        }
        Fix::Rebuild => {
            if p.confirm(
                "Rebuild Talus now (it compiles the kernels: tens of minutes)?",
                !p.interactive(),
            ) {
                crate::update::rebuild(x, paths, st)
            } else {
                ui::say("Skipped. Later: talus update --rebuild");
                Ok(())
            }
        }
    }
}

pub fn run(a: Args) -> i32 {
    let paths = Paths::from_env();
    let sys = probe::probe(Path::new("/"));
    let st = match SetupState::load(&paths) {
        Ok(s) => s,
        Err(e) => {
            ui::error(&format!("cannot read the setup: {e}"));
            None
        }
    };
    if !a.report {
        ui::header(if a.hw_only {
            "hardware report"
        } else {
            "doctor"
        });
    }
    let ctx = checks::Ctx::new(
        &sys,
        &paths,
        st.as_ref(),
        checks::Opts {
            deep: a.deep,
            network: !a.offline,
        },
    );
    let cs = if a.hw_only {
        let need = ctx.models.iter().any(|m| m.needs_big_gtt);
        checks::hardware(&sys, need)
    } else {
        checks::run(&ctx)
    };
    if a.report {
        let rocm = st
            .as_ref()
            .and_then(|s| s.rocm_sdk.as_ref())
            .and_then(|r| checks::rocm_version(r))
            .unwrap_or_else(|| "unknown".into());
        let text = report::render(&cs, &sys, &talus_version(st.as_ref()), &rocm);
        let red = report::Redactor::from_env(&paths);
        println!(
            "<details><summary>talus doctor report</summary>\n\n{}\n</details>",
            red.apply(&text)
        );
        eprintln!("(home directory, user and host names, IP addresses and keys are removed; check before posting)");
        return if checks::worst(&cs) == Verdict::Fail {
            1
        } else {
            0
        };
    }
    print(&cs);
    let fixes: Vec<Fix> = {
        let mut v: Vec<Fix> = Vec::new();
        for c in cs
            .iter()
            .filter(|c| matches!(c.verdict, Verdict::Warn | Verdict::Fail))
        {
            for f in &c.fixes {
                // one config rewrite and one boot change cover every check that asks for them
                let dup = v.iter().any(|g| {
                    g == f
                        || matches!(
                            (g, f),
                            (Fix::RegenerateConfig { .. }, Fix::RegenerateConfig { .. })
                        )
                });
                if !dup {
                    v.push(f.clone());
                }
            }
        }
        v
    };
    // the safe fixes first, then what needs sudo, then boot settings
    let mut fixes = fixes;
    fixes.sort_by_key(|f| match f {
        _ if f.safe() => 0,
        Fix::Rebuild => 1,
        Fix::Sudo { .. } => 2,
        _ => 3,
    });
    let fails = cs.iter().filter(|c| c.verdict == Verdict::Fail).count();
    let warns = cs.iter().filter(|c| c.verdict == Verdict::Warn).count();
    println!();
    if fails + warns == 0 {
        ui::say(&ui::good("Everything looks right."));
        return 0;
    }
    ui::say(&format!("{fails} problem(s), {warns} warning(s)."));
    if fixes.is_empty() {
        return i32::from(fails > 0);
    }
    if !a.fix {
        ui::say("`talus doctor --fix` applies the fixes listed above (it asks before anything that needs sudo).");
        return i32::from(fails > 0);
    }
    let Some(mut st) = st else {
        ui::say("Run `talus setup` first.");
        return 1;
    };
    let x = Exec::new(a.dry_run);
    let mut p = Prompter::new(a.yes);
    println!();
    ui::say(&ui::bold("Fixing"));
    for f in &fixes {
        let go = if f.safe() {
            p.confirm(&format!("{}?", capitalize(&f.describe())), true)
        } else {
            true
        };
        if !go {
            continue;
        }
        if let Err(e) = apply(f, &x, &mut p, &paths, &mut st, &sys, &a) {
            ui::error(&e);
        }
    }
    if a.dry_run {
        return 0;
    }
    println!();
    ui::say(&ui::bold("After the fixes"));
    let ctx = checks::Ctx::new(
        &sys,
        &paths,
        Some(&st),
        checks::Opts {
            deep: false,
            network: false,
        },
    );
    let after = checks::run(&ctx);
    print(&after);
    i32::from(checks::worst(&after) == Verdict::Fail)
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}
