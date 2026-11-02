//! `talus update`: fetch the new release, rebuild, migrate the run configs, restart the service, run the doctor.
//!
//! The build itself is install.sh's (`install.sh --build-only`), so there is one build recipe. When the operator
//! publishes prebuilt binaries, install.sh takes them instead of compiling (`TALUS_PREBUILT_URL`).

use crate::exec::Exec;
use crate::paths::Paths;
use crate::state::{SetupState, STATE_VERSION};
use crate::ui::{self, Prompter};
use std::process::Command;

#[derive(Clone, Debug, Default)]
pub struct Args {
    pub yes: bool,
    pub dry_run: bool,
    /// Only say whether an update exists.
    pub check: bool,
    /// Rebuild even without new commits.
    pub rebuild: bool,
}

fn git(src: &std::path::Path, args: &[&str]) -> Result<String, String> {
    let o = Command::new("git")
        .arg("-C")
        .arg(src)
        .args(args)
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if o.status.success() {
        Ok(String::from_utf8_lossy(&o.stdout).trim().to_string())
    } else {
        Err(String::from_utf8_lossy(&o.stderr).trim().to_string())
    }
}

/// The newest `vX.Y.Z` tag in `git ls-remote --tags --refs` output (release candidates and other tags left out).
pub fn newest_release_tag(ls_remote: &str) -> Option<String> {
    ls_remote
        .lines()
        .filter_map(|l| l.split("refs/tags/").nth(1))
        .filter_map(|t| {
            let v: Vec<u32> = t
                .strip_prefix('v')?
                .split('.')
                .map(|x| x.parse().ok())
                .collect::<Option<_>>()?;
            (v.len() == 3).then(|| ((v[0], v[1], v[2]), t.to_string()))
        })
        .max()
        .map(|(_, t)| t)
}

/// A source installed from a release tag (install.sh clones the tag of the binaries, talus#50 A3): move it to the
/// newest release tag. Returns Ok(false) when it is already there.
fn update_tag_checkout(
    x: &Exec,
    p: &mut Prompter,
    src: &std::path::Path,
    check: bool,
) -> Result<bool, String> {
    let current = git(src, &["describe", "--tags", "--exact-match", "HEAD"]).unwrap_or_default();
    let tags = git(src, &["ls-remote", "--tags", "--refs", "origin", "v*"])?;
    let Some(latest) = newest_release_tag(&tags) else {
        ui::say("No release tag found upstream; Talus stays as it is.");
        return Ok(false);
    };
    if latest == current {
        ui::say(&format!("Talus is up to date ({current})."));
        return Ok(false);
    }
    ui::say(&format!(
        "New release: {} -> {latest}",
        if current.is_empty() {
            "this checkout"
        } else {
            &current
        }
    ));
    if check {
        ui::say("Update with `talus update`.");
        return Ok(false);
    }
    if !p.confirm(
        "Update now (fetch the release, rebuild, migrate configs)?",
        true,
    ) {
        return Ok(false);
    }
    let dirty = git(src, &["status", "--porcelain", "--untracked-files=no"]).unwrap_or_default();
    if !dirty.is_empty() {
        return Err(format!(
            "{} has local changes; nothing was updated. Commit or stash them (git -C {} stash), then run again.",
            src.display(),
            src.display()
        ));
    }
    let s = src.display().to_string();
    x.run(
        &[
            "git".into(),
            "-C".into(),
            s.clone(),
            "fetch".into(),
            "--quiet".into(),
            "--depth".into(),
            "1".into(),
            "origin".into(),
            "tag".into(),
            latest.clone(),
        ],
        "get the new release",
    )?;
    x.run(
        &[
            "git".into(),
            "-C".into(),
            s,
            "checkout".into(),
            "--quiet".into(),
            latest.clone(),
        ],
        "switch the source to it",
    )?;
    Ok(true)
}

/// Build and install the binaries from `st.src` with install.sh.
pub fn rebuild(x: &Exec, paths: &Paths, st: &SetupState) -> Result<(), String> {
    let script = st.src.join("install.sh");
    let mut a = vec!["env".to_string()];
    if let Some(r) = &st.rocm_sdk {
        a.push(format!("ROCM_SDK={}", r.display()));
    }
    a.push(format!("TALUS_HOME={}", paths.data.display()));
    a.extend([
        "sh".to_string(),
        script.display().to_string(),
        "--build-only".to_string(),
        "--src".to_string(),
        st.src.display().to_string(),
    ]);
    x.run(
        &a,
        "build the kernel library and the Rust programs, install them",
    )
}

/// Bring setup.json and the run configs up to this version (configs are regenerated; changed ones are backed up).
pub fn migrate(x: &Exec, paths: &Paths, st: &mut SetupState) -> Result<(), String> {
    if st.version < STATE_VERSION {
        st.version = STATE_VERSION;
    }
    let (_, stamp) = crate::wizard::now_stamp();
    crate::wizard::write_configs(x, paths, st, &stamp)?;
    if st.service && paths.unit_file().exists() {
        x.write(
            &paths.unit_file(),
            &crate::service::unit_text(paths),
            0o644,
            "systemd user service",
        )
        .map_err(|e| e.to_string())?;
    }
    if !x.dry_run {
        st.save(paths).map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn run(a: Args) -> i32 {
    let paths = Paths::from_env();
    ui::header(if a.dry_run {
        "update (dry run)"
    } else {
        "update"
    });
    let mut st = match SetupState::load(&paths) {
        Ok(Some(s)) => s,
        Ok(None) => {
            ui::error(
                "no setup found: install with the one-liner first (see README, Easy install)",
            );
            return 1;
        }
        Err(e) => {
            ui::error(&e);
            return 1;
        }
    };
    let x = Exec::new(a.dry_run);
    let mut p = Prompter::new(a.yes);
    let src = st.src.clone();
    let head = match git(&src, &["rev-parse", "HEAD"]) {
        Ok(h) => h,
        Err(_) => {
            ui::error(&format!(
                "{} is not a git checkout: run the install one-liner again to get a fresh copy",
                src.display()
            ));
            return 1;
        }
    };
    ui::say("Checking for a new release ...");
    // a release-tag checkout (detached HEAD, install.sh's default): follow the release tags, not a branch
    if git(&src, &["symbolic-ref", "-q", "HEAD"]).is_err() {
        match update_tag_checkout(&x, &mut p, &src, a.check) {
            Ok(true) => {}
            Ok(false) if !a.rebuild || a.check => return 0,
            Ok(false) => {}
            Err(e) => {
                ui::error(&e);
                return 1;
            }
        }
        ui::say(&ui::bold("Building"));
        if let Err(e) = rebuild(&x, &paths, &st) {
            ui::error(&e);
            return 1;
        }
        return finish_update(&x, &mut p, &paths, &mut st, &a);
    }
    if let Err(e) = git(&src, &["fetch", "--quiet", "origin"]) {
        ui::error(&format!("git fetch failed: {e}"));
        return 1;
    }
    let upstream = git(&src, &["rev-parse", "@{u}"]).unwrap_or_else(|_| head.clone());
    let behind: Vec<String> = git(&src, &["log", "--oneline", "HEAD..@{u}"])
        .unwrap_or_default()
        .lines()
        .map(str::to_string)
        .collect();
    if behind.is_empty() {
        ui::say(&format!(
            "Talus is up to date ({}).",
            &head[..12.min(head.len())]
        ));
        if a.check || !a.rebuild {
            return 0;
        }
    } else {
        ui::say(&format!(
            "{} new commit(s): {} -> {}",
            behind.len(),
            &head[..12.min(head.len())],
            &upstream[..12.min(upstream.len())]
        ));
        for l in behind.iter().take(10) {
            ui::note(l);
        }
        if a.check {
            ui::say("Update with `talus update`.");
            return 0;
        }
        if !p.confirm("Update now (pull, rebuild, migrate configs)?", true) {
            return 0;
        }
        let dirty =
            git(&src, &["status", "--porcelain", "--untracked-files=no"]).unwrap_or_default();
        if !dirty.is_empty() {
            ui::error(&format!(
                "{} has local changes; nothing was updated. Commit or stash them (git -C {} stash), then run again.",
                src.display(),
                src.display()
            ));
            return 1;
        }
        if let Err(e) = x.run(
            &[
                "git".into(),
                "-C".into(),
                src.display().to_string(),
                "pull".into(),
                "--ff-only".into(),
                "--quiet".into(),
            ],
            "get the new release",
        ) {
            ui::error(&e);
            return 1;
        }
    }
    ui::say(&ui::bold("Building"));
    if let Err(e) = rebuild(&x, &paths, &st) {
        ui::error(&e);
        return 1;
    }
    finish_update(&x, &mut p, &paths, &mut st, &a)
}

/// After the build: the NEW version does the rest (migrating configs and checking the result are its business).
fn finish_update(x: &Exec, p: &mut Prompter, paths: &Paths, st: &mut SetupState, a: &Args) -> i32 {
    let new = paths.bin().join("talus-setup");
    if a.dry_run || !new.exists() {
        return post_update(x, p, paths, st, a.dry_run);
    }
    let mut cmd = Command::new(&new);
    cmd.arg("migrate");
    if a.yes {
        cmd.arg("--yes");
    }
    match cmd.status() {
        Ok(s) if s.success() => 0,
        Ok(_) => 1,
        Err(e) => {
            ui::error(&format!(
                "cannot run the new version ({e}); run `talus doctor --fix`"
            ));
            1
        }
    }
}

/// `talus migrate` (run by `talus update` with the freshly built binary): configs and unit to this version, a
/// restart of a running server (asked), then the doctor. The doctor's verdict is shown, not the exit code: the update
/// itself succeeded.
pub fn post_update(
    x: &Exec,
    p: &mut Prompter,
    paths: &Paths,
    st: &mut SetupState,
    dry_run: bool,
) -> i32 {
    ui::say(&ui::bold("Migrating configs"));
    if let Err(e) = migrate(x, paths, st) {
        ui::error(&e);
        return 1;
    }
    if st.service {
        let active = Command::new("systemctl")
            .args(["--user", "is-active", crate::service::UNIT])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "active")
            .unwrap_or(false);
        let _ = x.run(
            &["systemctl".into(), "--user".into(), "daemon-reload".into()],
            "load the updated service",
        );
        if active
            && p.confirm(
                "Restart the server on the new version (serving pauses while it reloads)?",
                true,
            )
        {
            let _ = x.run(
                &[
                    "systemctl".into(),
                    "--user".into(),
                    "restart".into(),
                    crate::service::UNIT.into(),
                ],
                "restart the server",
            );
        }
    }
    if dry_run {
        return 0;
    }
    ui::say(&ui::good("Updated."));
    println!();
    crate::doctor::run(crate::doctor::Args {
        offline: true,
        ..Default::default()
    });
    0
}

/// Entry point of `talus migrate`.
pub fn run_migrate(yes: bool) -> i32 {
    let paths = Paths::from_env();
    let mut st = match SetupState::load(&paths) {
        Ok(Some(s)) => s,
        Ok(None) => return 0,
        Err(e) => {
            ui::error(&e);
            return 1;
        }
    };
    let x = Exec::new(false);
    let mut p = Prompter::new(yes);
    post_update(&x, &mut p, &paths, &mut st, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newest_release_tag_skips_candidates() {
        let out = "a1\trefs/tags/v0.1.0\nb2\trefs/tags/v0.1.10\nc3\trefs/tags/v0.1.2\nd4\trefs/tags/v0.2.0-rc1\ne5\trefs/tags/talus-x\n";
        assert_eq!(newest_release_tag(out).as_deref(), Some("v0.1.10"));
        assert_eq!(newest_release_tag("x\trefs/tags/v1.0-rc\n"), None);
        assert_eq!(newest_release_tag(""), None);
    }

    #[test]
    fn rebuild_goes_through_install_sh() {
        let home = crate::testutil::tempdir("upd");
        let paths = Paths::under(&home);
        let st = SetupState::new(&paths, home.join("src"), Some(home.join("rocm")));
        let x = Exec::new(true);
        rebuild(&x, &paths, &st).unwrap();
        let a = x.actions().join("\n");
        assert!(a.contains("install.sh --build-only --src"), "{a}");
        assert!(a.contains("ROCM_SDK="), "{a}");
        std::fs::remove_dir_all(home).ok();
    }
}
