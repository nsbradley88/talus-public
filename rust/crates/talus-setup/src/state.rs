//! `~/.config/talus/setup.json`: what was installed and where. Written by `talus setup`, read by doctor and update.

use crate::manifest::Choices;
use crate::paths::Paths;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const STATE_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SetupState {
    pub version: u32,
    /// The Talus source checkout the binaries were built from.
    pub src: PathBuf,
    /// The ROCm SDK root (has `bin/`, `lib/`, `lib/llvm/`).
    pub rocm_sdk: Option<PathBuf>,
    /// Model files go to `<models_dir>/<model-id>/`.
    pub models_dir: PathBuf,
    /// Installed models and their choices.
    #[serde(default)]
    pub models: BTreeMap<String, Choices>,
    /// The model the server starts with when the TUI has not switched it yet.
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// A systemd --user service was installed.
    #[serde(default)]
    pub service: bool,
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    8000
}

impl SetupState {
    pub fn new(paths: &Paths, src: PathBuf, rocm_sdk: Option<PathBuf>) -> SetupState {
        SetupState {
            version: STATE_VERSION,
            src,
            rocm_sdk,
            models_dir: paths.default_models(),
            models: BTreeMap::new(),
            default_model: None,
            host: default_host(),
            port: default_port(),
            service: false,
        }
    }

    pub fn model_dir(&self, id: &str) -> PathBuf {
        self.models_dir.join(id)
    }

    pub fn load(paths: &Paths) -> Result<Option<SetupState>, String> {
        Self::load_from(&paths.setup_json())
    }

    pub fn load_from(p: &Path) -> Result<Option<SetupState>, String> {
        match std::fs::read_to_string(p) {
            Ok(s) => serde_json::from_str(&s)
                .map(Some)
                .map_err(|e| format!("{}: {e}", p.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("{}: {e}", p.display())),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("state serializes") + "\n"
    }

    pub fn save(&self, paths: &Paths) -> std::io::Result<()> {
        write_atomic(&paths.setup_json(), self.to_json().as_bytes(), 0o644)
    }
}

/// Write through a temporary file and rename, so a crash never leaves half a file.
pub fn write_atomic(p: &Path, bytes: &[u8], mode: u32) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = p.with_extension(format!(
        "{}tmp{}",
        p.extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default(),
        std::process::id()
    ));
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(mode)
            .open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(mode))?;
    std::fs::rename(&tmp, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_defaults() {
        let dir = crate::testutil::tempdir("state");
        let paths = Paths::under(&dir);
        assert_eq!(SetupState::load(&paths).unwrap(), None);
        let mut s = SetupState::new(&paths, dir.join("src"), None);
        s.models.insert("qwen3.8-27b".into(), Choices::default());
        s.save(&paths).unwrap();
        let back = SetupState::load(&paths).unwrap().unwrap();
        assert_eq!(back, s);
        // an older file without the optional fields still loads
        std::fs::write(
            paths.setup_json(),
            r#"{"version":1,"src":"/s","rocm_sdk":null,"models_dir":"/m"}"#,
        )
        .unwrap();
        let old = SetupState::load(&paths).unwrap().unwrap();
        assert_eq!(old.port, 8000);
        assert_eq!(old.host, "127.0.0.1");
        assert!(old.models.is_empty());
        std::fs::remove_dir_all(dir).ok();
    }
}
