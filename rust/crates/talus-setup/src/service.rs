//! `talus serve` (what the service runs) and the systemd --user unit.

use crate::paths::Paths;
use crate::state::SetupState;
use std::path::PathBuf;

pub const UNIT: &str = "talus.service";

/// talus-server's arguments: the control directory (so the TUI can switch models), plus the first model's config
/// until the TUI has applied a choice (then `active.json` in the control directory is what starts).
pub fn server_args(paths: &Paths, st: &SetupState) -> Result<Vec<String>, String> {
    let dir = paths.control_dir();
    let mut a = vec!["--control-dir".to_string(), dir.display().to_string()];
    if dir.join("active.json").exists() {
        return Ok(a);
    }
    let first = st
        .default_model
        .clone()
        .filter(|m| st.models.contains_key(m))
        .or_else(|| st.models.keys().next().cloned())
        .ok_or("no model is installed: run `talus setup` first")?;
    let cfg = paths.run_config(&first);
    if !cfg.exists() {
        return Err(format!(
            "{} is missing: run `talus doctor --fix`",
            cfg.display()
        ));
    }
    a.extend(["--config".to_string(), cfg.display().to_string()]);
    Ok(a)
}

pub fn server_exe(paths: &Paths) -> PathBuf {
    paths.bin().join("talus-server")
}

pub fn unit_text(paths: &Paths) -> String {
    let exe = paths.bin().join("talus-setup");
    format!(
        "[Unit]\n\
         Description=Talus LLM server (OpenAI-compatible API)\n\
         Documentation=https://github.com/nsbradley88/talus-public\n\
         After=network-online.target\n\
         # give up after 5 failed starts in 10 minutes instead of retrying forever (talus doctor says why)\n\
         StartLimitIntervalSec=600\n\
         StartLimitBurst=5\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={} serve\n\
         Restart=on-failure\n\
         RestartSec=15\n\
         # loading a model takes a while; stopping drains running requests\n\
         TimeoutStartSec=900\n\
         TimeoutStopSec=150\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n",
        exe.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Choices;

    #[test]
    fn serve_picks_active_then_default() {
        let home = crate::testutil::tempdir("serve");
        let paths = Paths::under(&home);
        let mut st = SetupState::new(&paths, home.join("src"), None);
        assert!(server_args(&paths, &st).is_err());
        st.models.insert("laguna-s-2.1".into(), Choices::default());
        st.models.insert("qwen3.8-27b".into(), Choices::default());
        st.default_model = Some("qwen3.8-27b".into());
        assert!(server_args(&paths, &st).unwrap_err().contains("doctor"));
        std::fs::create_dir_all(paths.control_dir()).unwrap();
        std::fs::write(paths.run_config("qwen3.8-27b"), "{}").unwrap();
        let a = server_args(&paths, &st).unwrap();
        assert_eq!(a[2], "--config");
        assert!(a[3].ends_with("models/qwen3.8-27b.json"));
        std::fs::write(paths.control_dir().join("active.json"), "{}").unwrap();
        assert_eq!(server_args(&paths, &st).unwrap().len(), 2);
        let u = unit_text(&paths);
        assert!(u.contains("ExecStart=") && u.contains("talus-setup serve"));
        assert!(u.contains("WantedBy=default.target"));
        std::fs::remove_dir_all(home).ok();
    }
}
