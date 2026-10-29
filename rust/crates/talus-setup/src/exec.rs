//! Every change the installer makes goes through `Exec`, so `--dry-run` can print each action instead of doing it.

use crate::boot::shell_join;
use crate::ui;
use std::path::Path;
use std::process::Command;

#[derive(Clone, Debug, Default)]
pub struct Exec {
    pub dry_run: bool,
    /// Actions that ran or would run, in order (for tests and the dry-run summary).
    pub log: std::cell::RefCell<Vec<String>>,
}

impl Exec {
    pub fn new(dry_run: bool) -> Exec {
        Exec {
            dry_run,
            log: Default::default(),
        }
    }

    fn note(&self, what: String) {
        if self.dry_run {
            ui::dry(&what);
        }
        self.log.borrow_mut().push(what);
    }

    pub fn mkdir(&self, p: &Path) -> std::io::Result<()> {
        if p.is_dir() {
            return Ok(());
        }
        self.note(format!("create directory {}", p.display()));
        if self.dry_run {
            return Ok(());
        }
        std::fs::create_dir_all(p)
    }

    /// Write a user file (atomically). `why` is shown in the dry run.
    pub fn write(&self, p: &Path, content: &str, mode: u32, why: &str) -> std::io::Result<()> {
        let same = std::fs::read_to_string(p)
            .map(|s| s == content)
            .unwrap_or(false);
        if same {
            return Ok(());
        }
        let verb = if p.exists() { "update" } else { "write" };
        self.note(format!("{verb} {} ({why})", p.display()));
        if self.dry_run {
            return Ok(());
        }
        crate::state::write_atomic(p, content.as_bytes(), mode)
    }

    /// Keep a copy of `p` as `p.bak-<stamp>` before changing it.
    pub fn backup(&self, p: &Path, stamp: &str) -> std::io::Result<()> {
        if !p.exists() {
            return Ok(());
        }
        let b = p.with_file_name(format!(
            "{}.bak-{stamp}",
            p.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        ));
        self.note(format!("back up {} to {}", p.display(), b.display()));
        if self.dry_run {
            return Ok(());
        }
        std::fs::copy(p, &b).map(|_| ())
    }

    pub fn remove_file(&self, p: &Path) -> std::io::Result<()> {
        if !p.exists() && p.symlink_metadata().is_err() {
            return Ok(());
        }
        self.note(format!("remove {}", p.display()));
        if self.dry_run {
            return Ok(());
        }
        std::fs::remove_file(p)
    }

    pub fn remove_dir(&self, p: &Path) -> std::io::Result<()> {
        if !p.exists() {
            return Ok(());
        }
        self.note(format!(
            "remove directory {} and everything in it",
            p.display()
        ));
        if self.dry_run {
            return Ok(());
        }
        std::fs::remove_dir_all(p)
    }

    pub fn symlink(&self, target: &Path, link: &Path) -> std::io::Result<()> {
        if std::fs::read_link(link).ok().as_deref() == Some(target) {
            return Ok(());
        }
        self.note(format!("link {} -> {}", link.display(), target.display()));
        if self.dry_run {
            return Ok(());
        }
        if let Some(d) = link.parent() {
            std::fs::create_dir_all(d)?;
        }
        if link.symlink_metadata().is_ok() {
            std::fs::remove_file(link)?;
        }
        std::os::unix::fs::symlink(target, link)
    }

    /// Run a command as the user. Output goes to the terminal.
    pub fn run(&self, argv: &[String], why: &str) -> Result<(), String> {
        self.note(format!("run: {}   ({why})", shell_join(argv)));
        if self.dry_run {
            return Ok(());
        }
        let st = Command::new(&argv[0])
            .args(&argv[1..])
            .status()
            .map_err(|e| format!("{}: {e}", argv[0]))?;
        if st.success() {
            Ok(())
        } else {
            Err(format!("`{}` failed ({st})", shell_join(argv)))
        }
    }

    /// Run a command with sudo. The caller has asked the user first.
    pub fn sudo(&self, argv: &[String], why: &str) -> Result<(), String> {
        let mut a = vec!["sudo".to_string()];
        a.extend(argv.iter().cloned());
        self.run(&a, why)
    }

    /// Write a root-owned file: a user-owned temporary file, then `sudo install`.
    pub fn sudo_write(&self, p: &Path, content: &str, why: &str) -> Result<(), String> {
        self.note(format!("write {} as root ({why})", p.display()));
        if self.dry_run {
            return Ok(());
        }
        let tmp = std::env::temp_dir().join(format!("talus-{}.tmp", std::process::id()));
        std::fs::write(&tmp, content).map_err(|e| e.to_string())?;
        let r = Command::new("sudo")
            .args(["install", "-m", "0644", "-o", "root", "-g", "root"])
            .arg(&tmp)
            .arg(p)
            .status();
        let _ = std::fs::remove_file(&tmp);
        match r {
            Ok(s) if s.success() => Ok(()),
            Ok(s) => Err(format!("sudo install {}: {s}", p.display())),
            Err(e) => Err(e.to_string()),
        }
    }

    pub fn actions(&self) -> Vec<String> {
        self.log.borrow().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_run_touches_nothing_and_lists_everything() {
        let dir = crate::testutil::tempdir("exec");
        let x = Exec::new(true);
        let f = dir.join("a/b.json");
        x.mkdir(&dir.join("a")).unwrap();
        x.write(&f, "{}", 0o644, "run config").unwrap();
        x.symlink(Path::new("/bin/true"), &dir.join("l")).unwrap();
        x.run(&["false".to_string()], "test").unwrap();
        assert!(!dir.join("a").exists());
        let a = x.actions();
        assert_eq!(a.len(), 4);
        assert!(a[1].starts_with("write ") && a[1].ends_with("(run config)"));
        assert!(a[3].starts_with("run: false"));
        // for real, and a second identical write is not an action
        let y = Exec::new(false);
        y.write(&f, "{}", 0o600, "c").unwrap();
        y.write(&f, "{}", 0o600, "c").unwrap();
        assert_eq!(y.actions().len(), 1);
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&f).unwrap().permissions().mode() & 0o777,
            0o600
        );
        y.backup(&f, "s").unwrap();
        assert!(dir.join("a/b.json.bak-s").exists());
        assert!(y.run(&["false".to_string()], "t").is_err());
        std::fs::remove_dir_all(dir).ok();
    }
}
