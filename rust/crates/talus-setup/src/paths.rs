//! Where an install lives. Everything is in the user's home; nothing system-wide.
//!
//! ```text
//! ~/.local/share/talus/          TALUS_HOME (default $XDG_DATA_HOME/talus)
//!     src/                       the Talus source (git clone), unless install.sh ran from a checkout
//!     rocm/                      the ROCm SDK (TheRock wheels in a Python venv), unless ROCM_SDK was given
//!     bin/                       talus-cli, talus-server, talus-tui, talus-setup, libtalus_kernels.so
//!     models/<model-id>/         downloaded model files (or --models-dir)
//!     pyenv/                     Python tools for the Flash-Next pack (numpy, regex, llama.cpp's gguf-py)
//! ~/.config/talus/               $XDG_CONFIG_HOME/talus
//!     setup.json                 what was installed and where (this file drives doctor and update)
//!     models/<model-id>.json     one run config per model: talus-server's control directory (TUI model switching)
//!     api-key                    only when the server is reachable from the network
//! ~/.local/state/talus/          logs
//! ~/.local/bin/talus             -> bin/talus-setup
//! ~/.config/systemd/user/talus.service   (optional)
//! ```

use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Paths {
    pub home: PathBuf,
    pub data: PathBuf,
    pub config: PathBuf,
    pub state: PathBuf,
    /// Where the `talus` and `talus-tui` links go (`~/.local/bin`).
    pub user_bin: PathBuf,
    /// `~/.config/systemd/user`.
    pub systemd_user: PathBuf,
}

fn env_dir(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
}

impl Paths {
    pub fn from_env() -> Paths {
        let home = env_dir("HOME").unwrap_or_else(|| PathBuf::from("/"));
        let xdg_data = env_dir("XDG_DATA_HOME").unwrap_or_else(|| home.join(".local/share"));
        let xdg_config = env_dir("XDG_CONFIG_HOME").unwrap_or_else(|| home.join(".config"));
        let xdg_state = env_dir("XDG_STATE_HOME").unwrap_or_else(|| home.join(".local/state"));
        Paths {
            data: env_dir("TALUS_HOME").unwrap_or_else(|| xdg_data.join("talus")),
            config: xdg_config.join("talus"),
            state: xdg_state.join("talus"),
            user_bin: home.join(".local/bin"),
            systemd_user: xdg_config.join("systemd/user"),
            home,
        }
    }

    /// All paths under one root (tests, and a throwaway prefix).
    pub fn under(home: &Path) -> Paths {
        Paths {
            home: home.to_path_buf(),
            data: home.join(".local/share/talus"),
            config: home.join(".config/talus"),
            state: home.join(".local/state/talus"),
            user_bin: home.join(".local/bin"),
            systemd_user: home.join(".config/systemd/user"),
        }
    }

    pub fn src(&self) -> PathBuf {
        self.data.join("src")
    }
    pub fn bin(&self) -> PathBuf {
        self.data.join("bin")
    }
    pub fn rocm_venv(&self) -> PathBuf {
        self.data.join("rocm")
    }
    /// install.sh's link to the ROCm SDK root; the prebuilt programs find ROCm through it (RUNPATH).
    pub fn rocm_link(&self) -> PathBuf {
        self.data.join("rocm-sdk")
    }
    pub fn pyenv(&self) -> PathBuf {
        self.data.join("pyenv")
    }
    pub fn default_models(&self) -> PathBuf {
        self.data.join("models")
    }
    pub fn setup_json(&self) -> PathBuf {
        self.config.join("setup.json")
    }
    /// talus-server's control directory: one run config per model.
    pub fn control_dir(&self) -> PathBuf {
        self.config.join("models")
    }
    pub fn run_config(&self, model_id: &str) -> PathBuf {
        self.control_dir().join(format!("{model_id}.json"))
    }
    pub fn api_key(&self) -> PathBuf {
        self.config.join("api-key")
    }
    pub fn unit_file(&self) -> PathBuf {
        self.systemd_user.join(crate::service::UNIT)
    }
    pub fn engine_log(&self, model_id: &str) -> PathBuf {
        self.state.join(format!("{model_id}.engine.log"))
    }

    /// `p` with the home directory written as `~` (for messages and reports).
    pub fn tilde(&self, p: &Path) -> String {
        tilde_with(&self.home, p)
    }
}

pub fn tilde_with(home: &Path, p: &Path) -> String {
    if home != Path::new("/") {
        if let Ok(rest) = p.strip_prefix(home) {
            return if rest.as_os_str().is_empty() {
                "~".to_string()
            } else {
                format!("~/{}", rest.display())
            };
        }
    }
    p.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_under_home() {
        let p = Paths::under(Path::new("/h/u"));
        assert_eq!(
            p.run_config("glm-5.3-flash"),
            Path::new("/h/u/.config/talus/models/glm-5.3-flash.json")
        );
        assert_eq!(p.tilde(&p.bin()), "~/.local/share/talus/bin");
        assert_eq!(p.tilde(Path::new("/data/models")), "/data/models");
        assert_eq!(tilde_with(Path::new("/"), Path::new("/x")), "/x");
    }
}
