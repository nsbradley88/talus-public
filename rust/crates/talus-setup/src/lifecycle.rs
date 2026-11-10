//! `talus start`, `talus stop`, `talus status` (v0.1.7, public #11).
//!
//! Two ways a server runs after setup: the systemd --user service (`talus.service`, when setup installed it), or a
//! background `talus serve` started by setup (no service). The background start writes a pidfile at
//! `<state>/server.pid` (`talus serve` writes its own pid there, then execs talus-server, so the pid is the server's).
//! Before signalling a pid from that file we check that it still runs our talus-server (`/proc/<pid>/exe`): a stale
//! file, or a pid the kernel has since given to another program, is never signalled. Nothing is killed by name.

use crate::paths::Paths;
use crate::state::SetupState;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// How long `stop` waits for a background server to exit after SIGTERM (as scripts/talus-serve.sh).
pub const STOP_WAIT_SECS: u64 = 90;

pub fn pidfile(paths: &Paths) -> PathBuf {
    paths.state.join("server.pid")
}

/// The world as `start`/`stop`/`status` see it; tests replace it.
pub trait Host {
    /// `systemctl --user is-active talus.service` (trimmed stdout: active, inactive, failed, activating, ...).
    fn service_active(&self) -> String;
    /// `systemctl --user start|stop talus.service`.
    fn service(&self, verb: &str) -> Result<(), String>;
    /// The executable of a live pid (`/proc/<pid>/exe`), None if no such process.
    fn exe_of(&self, pid: u32) -> Option<PathBuf>;
    /// SIGTERM.
    fn terminate(&self, pid: u32) -> Result<(), String>;
    /// Start `talus serve` in the background (nohup, log appended to `log`).
    fn spawn_background(&self, setup_exe: &Path, log: &Path) -> Result<(), String>;
    /// GET /health on 127.0.0.1:port: Some(body) if it answers 200.
    fn health(&self, port: u16) -> Option<String>;
    fn sleep(&self, d: Duration);
}

/// Who runs the server now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Running {
    Service,
    Background(u32),
    No,
}

/// Read the pidfile and keep the pid only if it is our talus-server.
/// Returns (pid, why-not) so status can say what it found.
pub fn background_pid(h: &dyn Host, paths: &Paths) -> (Option<u32>, Option<String>) {
    let f = pidfile(paths);
    let Ok(s) = std::fs::read_to_string(&f) else {
        return (None, None);
    };
    let Ok(pid) = s.trim().parse::<u32>() else {
        return (None, Some(format!("{} is not a pid", f.display())));
    };
    match h.exe_of(pid) {
        None => (
            None,
            Some(format!("stale pidfile: pid {pid} is not running")),
        ),
        Some(exe) if is_our_server(&exe, paths) => (Some(pid), None),
        Some(exe) => (
            None,
            Some(format!(
                "pid {pid} is now {} (not our talus-server): left alone",
                exe.display()
            )),
        ),
    }
}

/// `/proc/<pid>/exe` of our server: bin/talus-server under the install (a replaced binary reads "... (deleted)").
fn is_our_server(exe: &Path, paths: &Paths) -> bool {
    let want = crate::service::server_exe(paths);
    let s = exe.to_string_lossy();
    let s = s.strip_suffix(" (deleted)").unwrap_or(&s);
    Path::new(s) == want
        || std::fs::canonicalize(&want)
            .map(|w| Path::new(s) == w)
            .unwrap_or(false)
}

pub fn running(h: &dyn Host, paths: &Paths, st: &SetupState) -> Running {
    if st.service
        && matches!(
            h.service_active().as_str(),
            "active" | "activating" | "reloading"
        )
    {
        return Running::Service;
    }
    match background_pid(h, paths).0 {
        Some(p) => Running::Background(p),
        None => Running::No,
    }
}

pub fn stop(h: &dyn Host, paths: &Paths, st: &SetupState) -> Result<String, String> {
    match running(h, paths, st) {
        Running::Service => {
            h.service("stop")?;
            Ok("stopped the talus service".into())
        }
        Running::Background(pid) => {
            h.terminate(pid)?;
            for _ in 0..STOP_WAIT_SECS {
                if h.exe_of(pid).is_none() {
                    let _ = std::fs::remove_file(pidfile(paths));
                    return Ok(format!("stopped the background server (pid {pid})"));
                }
                h.sleep(Duration::from_secs(1));
            }
            Err(format!(
                "the server (pid {pid}) is still running after {STOP_WAIT_SECS} s; it may be finishing a request"
            ))
        }
        Running::No => {
            // a stale file only misleads the next status
            if let (None, Some(_)) = background_pid(h, paths) {
                let _ = std::fs::remove_file(pidfile(paths));
            }
            Ok("not running".into())
        }
    }
}

/// Start the server and wait until /health answers (as setup does: up to 15 minutes, 5 s steps).
pub fn start(h: &dyn Host, paths: &Paths, st: &SetupState, wait: bool) -> Result<String, String> {
    match running(h, paths, st) {
        Running::Service => {
            return Err("already running (talus service); `talus stop` first".into())
        }
        Running::Background(p) => {
            return Err(format!(
                "already running (background, pid {p}); `talus stop` first"
            ))
        }
        Running::No => {}
    }
    if h.health(st.port).is_some() {
        return Err(format!(
            "something already answers on port {}: not started (see `talus status`)",
            st.port
        ));
    }
    let how = if st.service {
        h.service("start")?;
        "service"
    } else {
        std::fs::create_dir_all(&paths.state).map_err(|e| e.to_string())?;
        h.spawn_background(
            &paths.bin().join("talus-setup"),
            &paths.state.join("server.log"),
        )?;
        "background"
    };
    if !wait {
        return Ok(format!("started ({how})"));
    }
    for _ in 0..180 {
        if h.health(st.port).is_some() {
            return Ok(format!(
                "started ({how}); the server answers on http://{}:{}/v1",
                st.host, st.port
            ));
        }
        h.sleep(Duration::from_secs(5));
    }
    Err(format!(
        "started ({how}) but the server did not answer within 15 minutes; see {} and `talus doctor`",
        paths.tilde(&paths.state)
    ))
}

pub fn status(h: &dyn Host, paths: &Paths, st: &SetupState) -> String {
    let mut out = vec![];
    let health = h.health(st.port);
    match running(h, paths, st) {
        Running::Service => out.push("running: yes (talus service)".to_string()),
        Running::Background(p) => out.push(format!("running: yes (background, pid {p})")),
        Running::No if health.is_some() => out.push(
            "running: something answers on the port, but not a server this install started".into(),
        ),
        Running::No => out.push("running: no".into()),
    }
    if let (None, Some(why)) = background_pid(h, paths) {
        out.push(format!("note: {why}"));
    }
    out.push(format!("port: {} ({})", st.port, st.host));
    match health {
        Some(body) => {
            let v: Option<serde_json::Value> = serde_json::from_str(&body).ok();
            let g = |k: &str| {
                v.as_ref()
                    .and_then(|v| v.get(k))
                    .and_then(|x| x.as_str())
                    .unwrap_or("?")
                    .to_string()
            };
            out.push(format!("model: {}", g("model")));
            out.push(format!("health: {}", g("status")));
        }
        None => out.push("health: no answer".into()),
    }
    out.join("\n")
}

/// The real host.
pub struct Sys;

impl Host for Sys {
    fn service_active(&self) -> String {
        std::process::Command::new("systemctl")
            .args(["--user", "is-active", crate::service::UNIT])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    }
    fn service(&self, verb: &str) -> Result<(), String> {
        let s = std::process::Command::new("systemctl")
            .args(["--user", verb, crate::service::UNIT])
            .status()
            .map_err(|e| format!("systemctl: {e}"))?;
        if s.success() {
            Ok(())
        } else {
            Err(format!(
                "systemctl --user {verb} {} failed",
                crate::service::UNIT
            ))
        }
    }
    fn exe_of(&self, pid: u32) -> Option<PathBuf> {
        std::fs::read_link(format!("/proc/{pid}/exe")).ok()
    }
    fn terminate(&self, pid: u32) -> Result<(), String> {
        // SAFETY: plain kill(2) on a pid verified above to be our talus-server.
        let r = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
        if r == 0 {
            Ok(())
        } else {
            Err(format!(
                "cannot signal pid {pid}: {}",
                std::io::Error::last_os_error()
            ))
        }
    }
    fn spawn_background(&self, setup_exe: &Path, log: &Path) -> Result<(), String> {
        let s = std::process::Command::new("sh")
            .args([
                "-c",
                &format!(
                    "nohup '{}' serve >>'{}' 2>&1 </dev/null &",
                    setup_exe.display(),
                    log.display()
                ),
            ])
            .status()
            .map_err(|e| e.to_string())?;
        if s.success() {
            Ok(())
        } else {
            Err("could not start the server in the background".into())
        }
    }
    fn health(&self, port: u16) -> Option<String> {
        match crate::checks::http_get_local(port, "/health", Duration::from_secs(3)) {
            Ok((200, b)) => Some(b),
            _ => None,
        }
    }
    fn sleep(&self, d: Duration) {
        std::thread::sleep(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    struct Fake {
        active: String,
        procs: RefCell<HashMap<u32, PathBuf>>,
        /// pids that exit after SIGTERM (others ignore it)
        obey: bool,
        health: RefCell<Option<String>>,
        calls: RefCell<Vec<String>>,
    }
    impl Fake {
        fn new(active: &str) -> Fake {
            Fake {
                active: active.into(),
                procs: RefCell::new(HashMap::new()),
                obey: true,
                health: RefCell::new(None),
                calls: RefCell::new(vec![]),
            }
        }
    }
    impl Host for Fake {
        fn service_active(&self) -> String {
            self.active.clone()
        }
        fn service(&self, verb: &str) -> Result<(), String> {
            self.calls.borrow_mut().push(format!("systemctl {verb}"));
            if verb == "start" {
                *self.health.borrow_mut() = Some(r#"{"status":"ok","model":"m"}"#.into());
            }
            Ok(())
        }
        fn exe_of(&self, pid: u32) -> Option<PathBuf> {
            self.procs.borrow().get(&pid).cloned()
        }
        fn terminate(&self, pid: u32) -> Result<(), String> {
            self.calls.borrow_mut().push(format!("kill {pid}"));
            if self.obey {
                self.procs.borrow_mut().remove(&pid);
            }
            Ok(())
        }
        fn spawn_background(&self, _: &Path, _: &Path) -> Result<(), String> {
            self.calls.borrow_mut().push("spawn".into());
            *self.health.borrow_mut() = Some(r#"{"status":"ok","model":"qwen3.8-27b"}"#.into());
            Ok(())
        }
        fn health(&self, _: u16) -> Option<String> {
            self.health.borrow().clone()
        }
        fn sleep(&self, _: Duration) {}
    }

    fn setup(tag: &str, service: bool) -> (Paths, SetupState) {
        let d = crate::testutil::tempdir(tag);
        let p = Paths::under(&d);
        std::fs::create_dir_all(&p.state).unwrap();
        let mut st = SetupState::new(&p, d.join("src"), None);
        st.service = service;
        (p, st)
    }

    #[test]
    fn stale_pidfile_is_not_signalled_and_is_removed() {
        let (p, st) = setup("lc-stale", false);
        std::fs::write(pidfile(&p), "4242\n").unwrap();
        let h = Fake::new("inactive");
        assert_eq!(running(&h, &p, &st), Running::No);
        assert!(status(&h, &p, &st).contains("stale pidfile"));
        assert_eq!(stop(&h, &p, &st).unwrap(), "not running");
        assert!(h.calls.borrow().is_empty());
        assert!(!pidfile(&p).exists());
    }

    #[test]
    fn reused_pid_of_another_program_is_left_alone() {
        let (p, st) = setup("lc-reuse", false);
        std::fs::write(pidfile(&p), "777").unwrap();
        let h = Fake::new("inactive");
        h.procs
            .borrow_mut()
            .insert(777, PathBuf::from("/usr/bin/firefox"));
        assert_eq!(running(&h, &p, &st), Running::No);
        assert!(status(&h, &p, &st).contains("not our talus-server"));
        assert_eq!(stop(&h, &p, &st).unwrap(), "not running");
        assert!(h.calls.borrow().is_empty(), "never signal another program");
        // the file no longer names our server: dropped so the next status is clean; the other program runs on
        assert!(!pidfile(&p).exists());
        assert!(h.exe_of(777).is_some());
    }

    #[test]
    fn background_server_is_stopped_by_pid() {
        let (p, st) = setup("lc-bg", false);
        std::fs::write(pidfile(&p), "555").unwrap();
        let h = Fake::new("inactive");
        h.procs
            .borrow_mut()
            .insert(555, crate::service::server_exe(&p));
        assert_eq!(running(&h, &p, &st), Running::Background(555));
        assert!(stop(&h, &p, &st).unwrap().contains("pid 555"));
        assert_eq!(*h.calls.borrow(), vec!["kill 555"]);
        assert!(!pidfile(&p).exists());
    }

    #[test]
    fn replaced_binary_still_counts_as_ours() {
        let (p, st) = setup("lc-deleted", false);
        std::fs::write(pidfile(&p), "556").unwrap();
        let h = Fake::new("inactive");
        let exe = format!("{} (deleted)", crate::service::server_exe(&p).display());
        h.procs.borrow_mut().insert(556, PathBuf::from(exe));
        assert_eq!(running(&h, &p, &st), Running::Background(556));
    }

    #[test]
    fn stop_reports_a_server_that_ignores_sigterm() {
        let (p, st) = setup("lc-stuck", false);
        std::fs::write(pidfile(&p), "558").unwrap();
        let mut h = Fake::new("inactive");
        h.obey = false;
        h.procs
            .borrow_mut()
            .insert(558, crate::service::server_exe(&p));
        let e = stop(&h, &p, &st).unwrap_err();
        assert!(e.contains("still running after 90 s"));
        assert!(pidfile(&p).exists());
    }

    #[test]
    fn service_path_uses_systemctl_and_never_signals() {
        let (p, st) = setup("lc-svc", true);
        // even with a pidfile of a live server, an active service wins
        std::fs::write(pidfile(&p), "559").unwrap();
        let h = Fake::new("active");
        h.procs
            .borrow_mut()
            .insert(559, crate::service::server_exe(&p));
        assert_eq!(running(&h, &p, &st), Running::Service);
        assert_eq!(stop(&h, &p, &st).unwrap(), "stopped the talus service");
        assert_eq!(*h.calls.borrow(), vec!["systemctl stop"]);
    }

    #[test]
    fn start_uses_the_service_when_installed_else_background() {
        let (p, st) = setup("lc-start-svc", true);
        let h = Fake::new("inactive");
        assert!(start(&h, &p, &st, true).unwrap().contains("(service)"));
        assert_eq!(*h.calls.borrow(), vec!["systemctl start"]);

        let (p, st) = setup("lc-start-bg", false);
        let h = Fake::new("inactive");
        assert!(start(&h, &p, &st, true).unwrap().contains("(background)"));
        assert_eq!(*h.calls.borrow(), vec!["spawn"]);
        assert!(status(&h, &p, &st).contains("model: qwen3.8-27b"));
    }

    #[test]
    fn start_refuses_when_already_running_or_port_taken() {
        let (p, st) = setup("lc-start-busy", false);
        std::fs::write(pidfile(&p), "560").unwrap();
        let h = Fake::new("inactive");
        h.procs
            .borrow_mut()
            .insert(560, crate::service::server_exe(&p));
        assert!(start(&h, &p, &st, true)
            .unwrap_err()
            .contains("already running"));

        let (p, st) = setup("lc-start-port", false);
        let h = Fake::new("inactive");
        *h.health.borrow_mut() = Some("{}".into());
        assert!(start(&h, &p, &st, true)
            .unwrap_err()
            .contains("already answers"));
        assert!(h.calls.borrow().is_empty());

        let (p, st) = setup("lc-start-svc-up", true);
        let h = Fake::new("active");
        assert!(start(&h, &p, &st, true)
            .unwrap_err()
            .contains("talus service"));
    }
}
