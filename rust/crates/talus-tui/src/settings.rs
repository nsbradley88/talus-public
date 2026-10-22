//! The TUI's own settings file (`~/.config/talus/tui.json`) and API key file (`~/.config/talus/api-key`).
//!
//! The settings file keeps the server address, the polling interval and the user's choices per model, so picking a
//! model again brings back what they chose last time. The API key never goes into the settings file.

use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use talus_presets::Selection;

pub const DEFAULT_SERVER: &str = "http://127.0.0.1:8000";
pub const DEFAULT_INTERVAL_MS: u64 = 2000;

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub server: String,
    /// Status polling interval (1-10 s).
    pub interval_ms: u64,
    /// Model id -> the user's choices (only non-defaults).
    pub models: BTreeMap<String, Selection>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            server: DEFAULT_SERVER.to_string(),
            interval_ms: DEFAULT_INTERVAL_MS,
            models: BTreeMap::new(),
        }
    }
}

pub fn config_dir() -> Option<PathBuf> {
    if let Some(x) = std::env::var_os("XDG_CONFIG_HOME").filter(|x| !x.is_empty()) {
        return Some(PathBuf::from(x).join("talus"));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config").join("talus"))
}

pub fn clamp_interval(ms: u64) -> u64 {
    ms.clamp(1000, 10_000)
}

impl Settings {
    pub fn from_json(v: &Value) -> Settings {
        let mut s = Settings::default();
        if let Some(srv) = v.get("server").and_then(Value::as_str) {
            s.server = srv.to_string();
        }
        if let Some(ms) = v.get("interval_ms").and_then(Value::as_u64) {
            s.interval_ms = clamp_interval(ms);
        }
        if let Some(models) = v.get("models").and_then(Value::as_object) {
            for (id, choices) in models {
                let sel: Selection = choices
                    .as_object()
                    .map(|o| {
                        o.iter()
                            .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string())))
                            .collect()
                    })
                    .unwrap_or_default();
                // drop choices the catalog no longer knows (an older settings file)
                if let Some(spec) = talus_presets::model(id) {
                    let sel: Selection = sel
                        .into_iter()
                        .filter(|(k, v)| spec.option(k).and_then(|o| o.choice(v)).is_some())
                        .collect();
                    s.models.insert(id.clone(), spec.normalize(&sel));
                }
            }
        }
        s
    }

    pub fn to_json(&self) -> Value {
        let models: Map<String, Value> = self
            .models
            .iter()
            .filter(|(_, sel)| !sel.is_empty())
            .map(|(id, sel)| {
                let o: Map<String, Value> = sel
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                    .collect();
                (id.clone(), Value::Object(o))
            })
            .collect();
        json!({"server": self.server, "interval_ms": self.interval_ms, "models": models})
    }

    pub fn load(path: &Path) -> Settings {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .map(|v| Settings::from_json(&v))
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let text = serde_json::to_string_pretty(&self.to_json()).unwrap_or_default();
        std::fs::write(path, text + "\n")
    }
}

pub fn read_key_file(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Write the API key readable by this user only.
pub fn write_key_file(path: &Path, key: &str) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(path)?;
    f.write_all(key.as_bytes())?;
    f.write_all(b"\n")
}

/// A random API key: 32 hex characters from the OS random source.
pub fn generate_key() -> std::io::Result<String> {
    use std::io::Read;
    let mut buf = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut buf)?;
    Ok(format!(
        "talus-{}",
        buf.iter().map(|b| format!("{b:02x}")).collect::<String>()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_cleanup() {
        let v = json!({
            "server": "http://box:8000",
            "interval_ms": 50,
            "models": {
                "qwen3.8-27b": {"context": "32768", "spec": "on", "gone": "x"},
                "unknown-model": {"context": "1"},
                "mimo-v2.6-flash": {"iq1m": "maybe"}
            }
        });
        let s = Settings::from_json(&v);
        assert_eq!(s.server, "http://box:8000");
        assert_eq!(s.interval_ms, 1000);
        assert_eq!(s.models.len(), 2);
        let q = &s.models["qwen3.8-27b"];
        assert_eq!(
            q.len(),
            1,
            "defaults and unknown options are dropped: {q:?}"
        );
        assert!(s.models["mimo-v2.6-flash"].is_empty());
        let back = Settings::from_json(&s.to_json());
        assert_eq!(back.models.get("qwen3.8-27b"), s.models.get("qwen3.8-27b"));
    }

    #[test]
    fn files() {
        let d = std::env::temp_dir().join(format!("talus-tui-settings-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let p = d.join("tui.json");
        assert_eq!(Settings::load(&p), Settings::default());
        let mut s = Settings::default();
        s.models.insert(
            "glm-5.3-flash".into(),
            [("context".to_string(), "32768".to_string())].into(),
        );
        s.save(&p).unwrap();
        assert_eq!(Settings::load(&p), s);
        let k = d.join("api-key");
        let key = generate_key().unwrap();
        assert!(key.len() >= talus_presets::MIN_API_KEY_LEN);
        write_key_file(&k, &key).unwrap();
        assert_eq!(read_key_file(&k).as_deref(), Some(key.as_str()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&k).unwrap().permissions().mode() & 0o077,
                0
            );
        }
        let _ = std::fs::remove_dir_all(&d);
    }
}
