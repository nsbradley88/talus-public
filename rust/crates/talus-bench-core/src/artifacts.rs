//! The artifact directory (spec §9). Files are written whole through a temp file + rename, so a reader (the TUI,
//! a second CLI) never sees half a file. Artifacts are local only; nothing is uploaded.

use serde_json::Value;
use std::path::{Path, PathBuf};

pub struct Artifacts {
    root: PathBuf,
}

impl Artifacts {
    pub fn create(root: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(root).map_err(|e| format!("{}: {e}", root.display()))?;
        Ok(Self {
            root: root.to_path_buf(),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn write_text(&self, rel: &str, text: &str) -> Result<(), String> {
        let p = self.root.join(rel);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        let tmp = p.with_extension("tmp-write");
        std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &p).map_err(|e| format!("{}: {e}", p.display()))
    }

    pub fn write_json(&self, rel: &str, v: &Value) -> Result<(), String> {
        self.write_text(
            rel,
            &(serde_json::to_string_pretty(v).unwrap_or_default() + "\n"),
        )
    }
}
