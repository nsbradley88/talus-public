//! `talus` (installed as a link to this binary): setup, doctor, update, serve.

use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use talus_setup::{doctor, paths::Paths, service, state::SetupState, ui, update, wizard};

const HELP: &str = "\
talus: install, check and update Talus

USAGE
  talus setup   [--advanced] [--yes] [--dry-run] [--allow-system-changes]
                [--models LIST | --add LIST | --remove LIST] [--models-dir DIR]
                [--port N] [--lan] [--no-start] [--no-service] [--force]
                [--opt-in-drafters]  (Laguna-S-2.1's DFlash drafter: license unclear, off by default)
  talus doctor  [--fix] [--yes] [--report] [--deep] [--offline] [--dry-run] [--allow-system-changes]
  talus hw      the hardware report only (read-only)
  talus update  [--check] [--rebuild] [--yes] [--dry-run]
  talus serve   run the server on the installed models (what the service runs)
  talus tui     open the dashboard (talus-tui)
  talus fetch   MODEL FILE... [--dir DIR]   download single model files (resumable, verified)

LIST is model numbers, ids or short names, comma-separated: 1,5  or  27b,laguna  or  all.
  1 qwen3.8-27b   2 qwen3.8-flash-next   3 glm-5.3-flash   4 mimo-v2.6-flash   5 laguna-s-2.1

--yes takes every default without asking. Boot settings (kernel parameters) change only after you type
\"yes\", or with --yes --allow-system-changes. Nothing ever reboots by itself.
--dry-run prints every action instead of doing it.
HF_TOKEN (optional) raises Hugging Face rate limits; it is never printed or stored.
";

fn need_value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_setup(args: Vec<String>) -> Result<wizard::Args, String> {
    let mut a = wizard::Args::default();
    let mut it = args.into_iter();
    while let Some(f) = it.next() {
        match f.as_str() {
            "--advanced" => a.advanced = true,
            "--yes" | "-y" => a.yes = true,
            "--dry-run" | "-n" => a.dry_run = true,
            "--allow-system-changes" => a.allow_system_changes = true,
            "--models" => a.models = Some(need_value(&mut it, &f)?),
            "--add" => a.add = Some(need_value(&mut it, &f)?),
            "--remove" => a.remove = Some(need_value(&mut it, &f)?),
            "--models-dir" => a.models_dir = Some(PathBuf::from(need_value(&mut it, &f)?)),
            "--src" => a.src = Some(PathBuf::from(need_value(&mut it, &f)?)),
            "--rocm-sdk" => a.rocm_sdk = Some(PathBuf::from(need_value(&mut it, &f)?)),
            "--port" => {
                a.port = Some(
                    need_value(&mut it, &f)?
                        .parse()
                        .map_err(|_| "--port takes a number")?,
                )
            }
            "--lan" => a.bind_lan = true,
            "--no-start" => a.no_start = true,
            "--no-service" => a.no_service = true,
            "--no-download" => a.no_download = true,
            "--opt-in-drafters" => a.opt_in_drafters = true,
            "--force" => a.force = true,
            other => return Err(format!("unknown option for setup: {other}")),
        }
    }
    Ok(a)
}

fn parse_doctor(args: Vec<String>) -> Result<doctor::Args, String> {
    let mut a = doctor::Args::default();
    for f in args {
        match f.as_str() {
            "--fix" => a.fix = true,
            "--yes" | "-y" => a.yes = true,
            "--report" => a.report = true,
            "--deep" => a.deep = true,
            "--offline" => a.offline = true,
            "--dry-run" | "-n" => a.dry_run = true,
            "--allow-system-changes" => a.allow_system_changes = true,
            "--hw" => a.hw_only = true,
            other => return Err(format!("unknown option for doctor: {other}")),
        }
    }
    Ok(a)
}

fn parse_update(args: Vec<String>) -> Result<update::Args, String> {
    let mut a = update::Args::default();
    for f in args {
        match f.as_str() {
            "--yes" | "-y" => a.yes = true,
            "--dry-run" | "-n" => a.dry_run = true,
            "--check" => a.check = true,
            "--rebuild" => a.rebuild = true,
            other => return Err(format!("unknown option for update: {other}")),
        }
    }
    Ok(a)
}

/// `talus fetch MODEL FILE... [--dir DIR]`: single files through the verified downloader.
fn fetch(args: Vec<String>) -> Result<i32, String> {
    use talus_setup::{download, manifest};
    let mut dir: Option<PathBuf> = None;
    let mut rest = Vec::new();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        if a == "--dir" {
            dir = Some(PathBuf::from(need_value(&mut it, "--dir")?));
        } else {
            rest.push(a);
        }
    }
    let model = rest
        .first()
        .ok_or("usage: talus fetch MODEL FILE... [--dir DIR]")?;
    let m = manifest::resolve(model).ok_or_else(|| format!("unknown model: {model}"))?;
    let dir = match dir {
        Some(d) => d,
        None => {
            let paths = Paths::from_env();
            SetupState::load(&paths)?
                .map(|s| s.model_dir(m.id))
                .unwrap_or_else(|| paths.default_models().join(m.id))
        }
    };
    let mut cache = Default::default();
    for name in &rest[1..] {
        let f = m
            .files
            .iter()
            .find(|f| f.local_name() == name.as_str())
            .ok_or_else(|| format!("{} has no file {name}", m.id))?;
        let opts = download::FetchOpts {
            quiet: false,
            check_published: true,
        };
        let out = download::fetch(f, &dir, &opts, &mut cache)?;
        ui::say(&format!("{name}: {out:?}, sha256 {}", f.sha256));
    }
    Ok(0)
}

fn serve() -> i32 {
    let paths = Paths::from_env();
    let st = match SetupState::load(&paths) {
        Ok(Some(s)) => s,
        Ok(None) => {
            ui::error("nothing installed yet: run `talus setup`");
            return 1;
        }
        Err(e) => {
            ui::error(&e);
            return 1;
        }
    };
    let args = match service::server_args(&paths, &st) {
        Ok(a) => a,
        Err(e) => {
            ui::error(&e);
            return 1;
        }
    };
    let _ = std::fs::create_dir_all(&paths.state);
    let err = std::process::Command::new(service::server_exe(&paths))
        .args(&args)
        .exec();
    ui::error(&format!(
        "cannot start {}: {err}",
        service::server_exe(&paths).display()
    ));
    1
}

fn main() {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = if argv.is_empty() {
        String::from("help")
    } else {
        argv.remove(0)
    };
    let code = match cmd.as_str() {
        "setup" | "install" => parse_setup(argv).map(wizard::run),
        "doctor" | "check" => parse_doctor(argv).map(doctor::run),
        "hw" | "hardware" => parse_doctor(argv).map(|mut a| {
            a.hw_only = true;
            doctor::run(a)
        }),
        "update" | "upgrade" => parse_update(argv).map(update::run),
        "serve" => Ok(serve()),
        // run by `talus update` with the new binary
        "migrate" => Ok(update::run_migrate(
            argv.iter().any(|a| a == "--yes" || a == "-y"),
        )),
        "fetch" => fetch(argv),
        "tui" => {
            let paths = Paths::from_env();
            let err = std::process::Command::new(paths.bin().join("talus-tui"))
                .args(argv)
                .exec();
            Err(format!("cannot start talus-tui: {err}"))
        }
        "help" | "--help" | "-h" => {
            print!("{HELP}");
            Ok(0)
        }
        "version" | "--version" | "-V" => {
            println!("talus-setup {}", env!("CARGO_PKG_VERSION"));
            Ok(0)
        }
        other => Err(format!("unknown command: {other} (try `talus help`)")),
    };
    match code {
        Ok(c) => std::process::exit(c),
        Err(e) => {
            ui::error(&e);
            std::process::exit(2);
        }
    }
}
