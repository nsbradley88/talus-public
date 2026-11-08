//! Build/weights fingerprint for `GET /v1/talus/status` (`build` object; bench server extension E3).
//!
//! What the bench records so a report can say what changed between two runs: the engine executable (path,
//! sha256, commit), the kernels library, the weight files named by the run config (path, size, mtime, sha256),
//! the sha256 of the served run config and the numerics-relevant environment variables.
//!
//! Cost discipline (default behaviour must not change):
//! - At start only `stat` calls and small metadata reads (`COMMIT`, `MANIFEST.json`, the hash cache).
//! - Files up to [`SMALL_FILE`] bytes (executable, kernels library, draft vocabularies) are hashed in a
//!   background thread started by the first status request.
//! - Larger files (the model weights, up to 100+ GB) are hashed only when a client asks for it
//!   (`GET /v1/talus/status?hash=weights`, local peer or API key), at the lowest CPU priority and without
//!   filling the page cache. Results are cached in `<control-dir>/fingerprints.json` keyed by
//!   (path, size, mtime), so a restart does not hash again.

use crate::config::ServerConfig;
use crate::json::JsonValue;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Files up to this size are hashed without being asked (first status request).
pub const SMALL_FILE: u64 = 256 << 20;
/// Hash cache file in the control directory.
pub const CACHE_FILE: &str = "fingerprints.json";
/// Prefixes of the environment variables that select numerics / kernels.
pub const ENV_PREFIXES: &[&str] = &["TALUS_", "STRATA_", "GLM_", "MIMO_"];
/// Run-config keys left out of `run_config_sha256` (secrets; the TUI's selection record with its timestamp).
const CONFIG_KEYS_SKIPPED: &[&str] = &["api_key", crate::control::SELECTION_KEY];

#[derive(Clone, Debug, PartialEq)]
pub struct FileEntry {
    /// The run-config flag that named the file (`--model`, `--mtp-gguf`, ...), if any.
    pub arg: Option<String>,
    pub path: PathBuf,
    pub size: u64,
    /// Modification time, nanoseconds since the epoch (cache key); reported in seconds.
    pub mtime_ns: i128,
    pub sha256: Option<String>,
    pub hashing: bool,
    pub error: Option<String>,
}

impl FileEntry {
    pub fn stat(arg: Option<&str>, path: &Path) -> Option<Self> {
        let md = std::fs::metadata(path).ok()?;
        if !md.is_file() {
            return None;
        }
        let mtime_ns = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as i128)
            .unwrap_or(0);
        Some(Self {
            arg: arg.map(str::to_string),
            path: path.to_path_buf(),
            size: md.len(),
            mtime_ns,
            sha256: None,
            hashing: false,
            error: None,
        })
    }

    fn key(&self) -> (String, u64, i128) {
        (
            self.path.to_string_lossy().into_owned(),
            self.size,
            self.mtime_ns,
        )
    }

    fn to_json(&self) -> JsonValue {
        let mut o = JsonValue::obj(vec![
            (
                "path",
                JsonValue::from(self.path.to_string_lossy().as_ref()),
            ),
            ("size", JsonValue::from(self.size)),
            (
                "mtime",
                JsonValue::from((self.mtime_ns / 1_000_000_000) as i64),
            ),
            (
                "sha256",
                self.sha256
                    .as_deref()
                    .map(JsonValue::from)
                    .unwrap_or(JsonValue::Null),
            ),
            ("hashing", JsonValue::from(self.hashing)),
        ]);
        if let Some(a) = &self.arg {
            o.insert("arg", a.as_str());
        }
        if let Some(e) = &self.error {
            o.insert("error", e.as_str());
        }
        o
    }
}

/// What the executable's directory says about the build: (commit, dirty, source, tag, gfx targets).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BuildMeta {
    pub commit: Option<String>,
    pub dirty: Option<bool>,
    pub commit_source: Option<String>,
    pub tag: Option<String>,
    pub gfx_targets: Vec<String>,
}

/// `COMMIT` next to the executable (first token = sha; "dirty" anywhere marks a dirty tree), else the release
/// tarball's `MANIFEST.json` one directory up (`commit`, `tag`, `targets.gpu`).
pub fn build_meta(exe: &Path) -> BuildMeta {
    let mut m = BuildMeta::default();
    let Some(dir) = exe.parent() else {
        return m;
    };
    if let Ok(t) = std::fs::read_to_string(dir.join("COMMIT")) {
        if let Some(sha) = t.split_whitespace().next() {
            m.commit = Some(sha.to_string());
            m.dirty = Some(t.contains("dirty"));
            m.commit_source = Some("COMMIT".to_string());
        }
    }
    for cand in [dir.join("MANIFEST.json"), dir.join("../MANIFEST.json")] {
        let Some(man) = std::fs::read_to_string(&cand)
            .ok()
            .and_then(|t| JsonValue::parse(&t).ok())
        else {
            continue;
        };
        if m.commit.is_none() {
            if let Some(c) = man.get_str("commit") {
                m.commit = Some(c.to_string());
                m.commit_source = Some("MANIFEST.json".to_string());
            }
        }
        m.tag = man.get_str("tag").map(str::to_string);
        m.gfx_targets = man
            .get("targets")
            .and_then(|t| t.get("gpu"))
            .and_then(JsonValue::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        break;
    }
    m
}

fn resolve(p: &str, cwd: Option<&Path>) -> PathBuf {
    let pb = PathBuf::from(p);
    match cwd {
        Some(c) if pb.is_relative() => c.join(pb),
        _ => pb,
    }
}

/// Every run-config argument that names an existing regular file, with the flag before it (deduplicated by path).
pub fn weight_files(cfg: &ServerConfig) -> Vec<FileEntry> {
    let cwd = cfg.cwd.as_deref();
    let mut out: Vec<FileEntry> = Vec::new();
    let mut push = |arg: Option<&str>, value: &str| {
        if value.is_empty() || value.starts_with('-') {
            return;
        }
        let p = resolve(value, cwd);
        if out.iter().any(|e| e.path == p) {
            return;
        }
        if let Some(e) = FileEntry::stat(arg, &p) {
            out.push(e);
        }
    };
    if !cfg.model_path.is_empty() {
        push(Some("--model"), &cfg.model_path);
    }
    let args = &cfg.engine_args;
    for (i, a) in args.iter().enumerate() {
        if a.starts_with('-') {
            continue;
        }
        let flag = i
            .checked_sub(1)
            .map(|j| args[j].as_str())
            .filter(|f| f.starts_with('-'));
        push(flag, a);
    }
    out
}

/// `libtalus_kernels.so`: `TALUS_KERNELS_LIB` in the run config's environment, else the first `lib_dirs` entry
/// that has it, else next to the engine executable.
pub fn kernels_lib(cfg: &ServerConfig) -> Option<FileEntry> {
    const NAME: &str = "libtalus_kernels.so";
    let cwd = cfg.cwd.as_deref();
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Some((_, v)) = cfg
        .engine_env
        .iter()
        .find(|(k, _)| k == "TALUS_KERNELS_LIB")
    {
        cands.push(resolve(v, cwd));
    }
    for d in &cfg.lib_dirs {
        cands.push(resolve(d, cwd).join(NAME));
    }
    if let Some(dir) = cfg.exe_path.as_deref().and_then(Path::parent) {
        cands.push(dir.join(NAME));
    }
    cands.iter().find_map(|p| FileEntry::stat(None, p))
}

fn secret_name(name: &str) -> bool {
    let u = name.to_ascii_uppercase();
    ["KEY", "TOKEN", "SECRET", "PASSW", "AUTH", "CRED"]
        .iter()
        .any(|s| u.contains(s))
}

/// The numerics environment the engine sees: the server's own `TALUS_*`/`STRATA_*`/`GLM_*`/`MIMO_*` variables
/// overridden by the run config's `env`, sorted by name. Nothing else, and never a name that looks like a secret.
pub fn numerics_env(
    process: impl IntoIterator<Item = (String, String)>,
    config: &[(String, String)],
) -> Vec<(String, String)> {
    let keep = |k: &str| ENV_PREFIXES.iter().any(|p| k.starts_with(p)) && !secret_name(k);
    let mut m: std::collections::BTreeMap<String, String> =
        process.into_iter().filter(|(k, _)| keep(k)).collect();
    for (k, v) in config {
        if keep(k) {
            m.insert(k.clone(), v.clone());
        }
    }
    m.into_iter().collect()
}

/// Canonical JSON: object keys sorted (recursively), no whitespace.
pub fn canonical_json(v: &JsonValue) -> String {
    match v {
        JsonValue::Object(entries) => {
            let mut e: Vec<&(String, JsonValue)> = entries.iter().collect();
            e.sort_by(|a, b| a.0.cmp(&b.0));
            let mut s = String::from("{");
            for (i, (k, val)) in e.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                crate::json::write_escaped_str(&mut s, k);
                s.push(':');
                s.push_str(&canonical_json(val));
            }
            s.push('}');
            s
        }
        JsonValue::Array(a) => {
            let parts: Vec<String> = a.iter().map(canonical_json).collect();
            format!("[{}]", parts.join(","))
        }
        other => other.to_json_string(),
    }
}

pub fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// sha256 of the run config being served (file config: its JSON without `api_key` and the TUI's selection record;
/// command line only: exe, args, env and model). Returns (sha256, source).
pub fn run_config_sha256(cfg: &ServerConfig) -> (String, &'static str) {
    if let Some(entries) = cfg.raw_config.as_object().filter(|o| !o.is_empty()) {
        let kept: Vec<(String, JsonValue)> = entries
            .iter()
            .filter(|(k, _)| !CONFIG_KEYS_SKIPPED.contains(&k.as_str()))
            .cloned()
            .collect();
        return (
            sha256_hex(canonical_json(&JsonValue::Object(kept)).as_bytes()),
            "config",
        );
    }
    let synth = JsonValue::obj(vec![
        (
            "exe",
            cfg.exe_path
                .as_ref()
                .map(|p| JsonValue::from(p.to_string_lossy().as_ref()))
                .unwrap_or(JsonValue::Null),
        ),
        (
            "args",
            JsonValue::Array(
                cfg.engine_args
                    .iter()
                    .map(|a| JsonValue::from(a.as_str()))
                    .collect(),
            ),
        ),
        (
            "env",
            JsonValue::Object(
                cfg.engine_env
                    .iter()
                    .filter(|(k, _)| !secret_name(k))
                    .map(|(k, v)| (k.clone(), JsonValue::from(v.as_str())))
                    .collect(),
            ),
        ),
        ("model", JsonValue::from(cfg.model_path.as_str())),
    ]);
    (
        sha256_hex(canonical_json(&synth).as_bytes()),
        "command_line",
    )
}

/// sha256 of a file, read sequentially; on Linux the pages read are dropped from the page cache as it goes
/// (pages another process has mapped, like the engine's weights, stay).
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 4 << 20];
    let mut done: i64 = 0;
    let mut dropped: i64 = 0;
    fadvise(&f, 0, 0, FADV_SEQUENTIAL);
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        done += n as i64;
        if done - dropped >= 64 << 20 {
            fadvise(&f, dropped, done - dropped, FADV_DONTNEED);
            dropped = done;
        }
    }
    fadvise(&f, dropped, 0, FADV_DONTNEED);
    Ok(hex(&h.finalize()))
}

const FADV_SEQUENTIAL: i32 = 2;
const FADV_DONTNEED: i32 = 4;

#[cfg(target_os = "linux")]
fn fadvise(f: &std::fs::File, off: i64, len: i64, advice: i32) {
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    }
    unsafe {
        let _ = posix_fadvise(f.as_raw_fd(), off, len, advice);
    }
}

#[cfg(not(target_os = "linux"))]
fn fadvise(_f: &std::fs::File, _off: i64, _len: i64, _advice: i32) {}

/// Lowest CPU priority for the calling thread (Linux: nice applies per thread).
#[cfg(target_os = "linux")]
fn lower_thread_priority() {
    extern "C" {
        fn setpriority(which: i32, who: u32, prio: i32) -> i32;
    }
    unsafe {
        let _ = setpriority(0, 0, 19);
    }
}

#[cfg(not(target_os = "linux"))]
fn lower_thread_priority() {}

struct Inner {
    exe: Option<FileEntry>,
    kernels: Option<FileEntry>,
    weights: Vec<FileEntry>,
}

impl Inner {
    fn entries_mut(&mut self) -> impl Iterator<Item = &mut FileEntry> {
        self.exe
            .iter_mut()
            .chain(self.kernels.iter_mut())
            .chain(self.weights.iter_mut())
    }
}

pub struct Fingerprint {
    inner: Mutex<Inner>,
    meta: BuildMeta,
    mock: bool,
    run_config: (String, &'static str),
    run_config_path: Option<PathBuf>,
    env: Vec<(String, String)>,
    cache_path: Option<PathBuf>,
    /// One hashing thread at a time.
    busy: AtomicBool,
    small_started: AtomicBool,
}

impl Fingerprint {
    /// Cheap: `stat` calls, small metadata files and the hash cache. Nothing is hashed here.
    pub fn new(cfg: &ServerConfig) -> Self {
        let engine_exe = cfg.exe_path.as_deref().filter(|_| !cfg.mock_mode);
        let exe = engine_exe.and_then(|p| {
            let p = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
            FileEntry::stat(None, &p)
        });
        let meta = exe
            .as_ref()
            .map(|e| build_meta(&e.path))
            .unwrap_or_default();
        let fp = Self {
            inner: Mutex::new(Inner {
                exe,
                kernels: if engine_exe.is_some() {
                    kernels_lib(cfg)
                } else {
                    None
                },
                weights: weight_files(cfg),
            }),
            meta,
            mock: engine_exe.is_none(),
            run_config: run_config_sha256(cfg),
            run_config_path: cfg.config_path.clone(),
            env: numerics_env(std::env::vars(), &cfg.engine_env),
            cache_path: cfg.control_dir.as_ref().map(|d| d.join(CACHE_FILE)),
            busy: AtomicBool::new(false),
            small_started: AtomicBool::new(false),
        };
        fp.apply_cache();
        fp
    }

    fn read_cache(&self) -> Vec<(String, u64, i128, String)> {
        let Some(t) = self
            .cache_path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
        else {
            return Vec::new();
        };
        let Ok(v) = JsonValue::parse(&t) else {
            return Vec::new();
        };
        v.get("files")
            .and_then(JsonValue::as_array)
            .unwrap_or(&[])
            .iter()
            .filter_map(|e| {
                Some((
                    e.get_str("path")?.to_string(),
                    e.get_u64("size")?,
                    e.get_str("mtime_ns")?.parse::<i128>().ok()?,
                    e.get_str("sha256")?.to_string(),
                ))
            })
            .collect()
    }

    fn apply_cache(&self) {
        let cache = self.read_cache();
        let mut g = self.inner.lock().unwrap();
        for e in g.entries_mut() {
            let k = e.key();
            if let Some(c) = cache.iter().find(|c| (&c.0, c.1, c.2) == (&k.0, k.1, k.2)) {
                e.sha256 = Some(c.3.clone());
            }
        }
    }

    /// Merge `done` into the cache file (entries of other files are kept).
    fn write_cache(&self, done: &[FileEntry]) {
        let Some(path) = &self.cache_path else {
            return;
        };
        let mut all = self.read_cache();
        for e in done {
            let Some(sha) = &e.sha256 else { continue };
            let k = e.key();
            all.retain(|c| c.0 != k.0);
            all.push((k.0, k.1, k.2, sha.clone()));
        }
        let files: Vec<JsonValue> = all
            .into_iter()
            .map(|(p, s, m, h)| {
                JsonValue::obj(vec![
                    ("path", JsonValue::from(p)),
                    ("size", JsonValue::from(s)),
                    ("mtime_ns", JsonValue::from(m.to_string())),
                    ("sha256", JsonValue::from(h)),
                ])
            })
            .collect();
        let body = JsonValue::obj(vec![
            ("version", JsonValue::from(1u64)),
            ("files", JsonValue::Array(files)),
        ]);
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, body.to_json_pretty()).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }

    /// Start hashing what is not hashed yet in a background thread: small files only, or every file with
    /// `include_large`. Returns false when a hashing thread is already running.
    pub fn start_hashing(self: &Arc<Self>, include_large: bool) -> bool {
        if !include_large && self.small_started.swap(true, Ordering::SeqCst) {
            return false;
        }
        if self.busy.swap(true, Ordering::SeqCst) {
            if !include_large {
                // retry the small files on a later request
                self.small_started.store(false, Ordering::SeqCst);
            }
            return false;
        }
        let todo: Vec<PathBuf> = {
            let mut g = self.inner.lock().unwrap();
            g.entries_mut()
                .filter(|e| e.sha256.is_none() && (include_large || e.size <= SMALL_FILE))
                .map(|e| {
                    e.hashing = true;
                    e.error = None;
                    e.path.clone()
                })
                .collect()
        };
        if todo.is_empty() {
            self.busy.store(false, Ordering::SeqCst);
            return true;
        }
        let me = Arc::clone(self);
        std::thread::Builder::new()
            .name("talus-fingerprint".to_string())
            .spawn(move || {
                lower_thread_priority();
                let mut done = Vec::new();
                for p in todo {
                    let r = sha256_file(&p);
                    let mut g = me.inner.lock().unwrap();
                    for e in g.entries_mut().filter(|e| e.path == p) {
                        e.hashing = false;
                        match &r {
                            Ok(h) => e.sha256 = Some(h.clone()),
                            Err(err) => e.error = Some(err.to_string()),
                        }
                        done.push(e.clone());
                    }
                }
                me.write_cache(&done);
                me.busy.store(false, Ordering::SeqCst);
            })
            .map(|_| true)
            .unwrap_or_else(|_| {
                self.busy.store(false, Ordering::SeqCst);
                false
            })
    }

    /// True when every file has a sha256 (or failed) and no thread runs.
    pub fn complete(&self) -> bool {
        let mut g = self.inner.lock().unwrap();
        !self.busy.load(Ordering::SeqCst)
            && g.entries_mut()
                .all(|e| e.sha256.is_some() || e.error.is_some())
    }

    /// The `build` object. `engine_info` reads keys from the engine's `INFO` lines.
    pub fn to_json(&self, engine_info: &dyn Fn(&str) -> Option<String>) -> JsonValue {
        let g = self.inner.lock().unwrap();
        let (commit, dirty, source) = match engine_info("commit") {
            Some(c) => (
                Some(c),
                engine_info("dirty").map(|d| d == "1" || d == "true"),
                Some("engine INFO".to_string()),
            ),
            None => (
                self.meta.commit.clone(),
                self.meta.dirty,
                self.meta.commit_source.clone(),
            ),
        };
        let opt = |v: Option<String>| v.map(JsonValue::from).unwrap_or(JsonValue::Null);
        let engine_exe = match &g.exe {
            Some(e) => {
                let mut o = e.to_json();
                o.insert("commit", opt(commit));
                o.insert(
                    "dirty",
                    dirty.map(JsonValue::from).unwrap_or(JsonValue::Null),
                );
                o.insert("commit_source", opt(source));
                o.insert("tag", opt(self.meta.tag.clone()));
                o
            }
            None => JsonValue::Null,
        };
        let (gfx, gfx_source) = match engine_info("gfx_target").or_else(|| engine_info("gfx")) {
            Some(g) => (Some(g), Some("engine INFO")),
            None if !self.meta.gfx_targets.is_empty() => (
                Some(self.meta.gfx_targets.join(",")),
                Some("MANIFEST.json targets.gpu"),
            ),
            None => (None, None),
        };
        JsonValue::obj(vec![
            ("mock", JsonValue::from(self.mock)),
            ("engine_exe", engine_exe),
            (
                "kernels_lib",
                g.kernels
                    .as_ref()
                    .map(FileEntry::to_json)
                    .unwrap_or(JsonValue::Null),
            ),
            ("gfx_target", opt(gfx)),
            (
                "gfx_target_source",
                gfx_source.map(JsonValue::from).unwrap_or(JsonValue::Null),
            ),
            (
                "weights",
                JsonValue::Array(g.weights.iter().map(FileEntry::to_json).collect()),
            ),
            (
                "run_config_sha256",
                JsonValue::from(self.run_config.0.as_str()),
            ),
            ("run_config_source", JsonValue::from(self.run_config.1)),
            (
                "run_config_path",
                self.run_config_path
                    .as_ref()
                    .map(|p| JsonValue::from(p.to_string_lossy().as_ref()))
                    .unwrap_or(JsonValue::Null),
            ),
            (
                "numerics_env",
                JsonValue::Object(
                    self.env
                        .iter()
                        .map(|(k, v)| (k.clone(), JsonValue::from(v.as_str())))
                        .collect(),
                ),
            ),
            (
                "hash_large",
                JsonValue::from(
                    "GET /v1/talus/status?hash=weights hashes files over 256 MiB (cached by path, size, mtime)",
                ),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("talus-fp-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let d = tmpdir("vec");
        let f = d.join("abc");
        std::fs::write(&f, b"abc").unwrap();
        assert_eq!(
            sha256_file(&f).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // larger than one read buffer and one page-cache drop window
        let big: Vec<u8> = (0..(70u32 << 20) / 4)
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let g = d.join("big");
        std::fs::write(&g, &big).unwrap();
        assert_eq!(sha256_file(&g).unwrap(), sha256_hex(&big));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn canonical_sorts_keys() {
        let a = JsonValue::parse(r#"{"b":1,"a":{"y":[1,{"d":2,"c":3}],"x":"s"}}"#).unwrap();
        assert_eq!(
            canonical_json(&a),
            r#"{"a":{"x":"s","y":[1,{"c":3,"d":2}]},"b":1}"#
        );
    }

    #[test]
    fn env_filter_prefixes_and_secrets() {
        let process = vec![
            ("PATH".to_string(), "/bin".to_string()),
            ("HOME".to_string(), "/h".to_string()),
            ("TALUS_API_KEY".to_string(), "sekrit".to_string()),
            ("STRATA_API_KEY".to_string(), "sekrit".to_string()),
            ("HF_TOKEN".to_string(), "sekrit".to_string()),
            ("GLM_PCH".to_string(), "2048".to_string()),
            ("TALUS_X".to_string(), "proc".to_string()),
        ];
        let config = vec![
            ("TALUS_X".to_string(), "cfg".to_string()),
            ("MIMO_Y".to_string(), "1".to_string()),
            ("LD_LIBRARY_PATH".to_string(), "/l".to_string()),
            ("TALUS_HF_TOKEN".to_string(), "sekrit".to_string()),
        ];
        let e = numerics_env(process, &config);
        assert_eq!(
            e,
            vec![
                ("GLM_PCH".to_string(), "2048".to_string()),
                ("MIMO_Y".to_string(), "1".to_string()),
                ("TALUS_X".to_string(), "cfg".to_string()),
            ]
        );
        assert!(!format!("{e:?}").contains("sekrit"));
    }

    fn cfg_with(d: &Path) -> ServerConfig {
        let exe = d.join("bin/talus-cli");
        std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
        std::fs::write(&exe, b"#!/bin/true\n").unwrap();
        std::fs::write(d.join("bin/libtalus_kernels.so"), b"kernels").unwrap();
        std::fs::write(d.join("bin/COMMIT"), b"0123abcd dirty\n").unwrap();
        std::fs::write(d.join("model.gguf"), b"weights-1").unwrap();
        std::fs::write(d.join("mtp.gguf"), b"mtp").unwrap();
        let raw = format!(
            r#"{{"exe": "{}", "args": ["--model", "{}", "--mtp-gguf", "mtp.gguf", "--ctx", "4096", "--spec-draft-vocab", "/nonexistent/x.bin"], "cwd": "{}", "env": {{"TALUS_A": "1", "TALUS_API_KEY": "sekrit-key-123"}}, "api_key": "sekrit-key-123", "control_dir": "{}"}}"#,
            exe.display(),
            d.join("model.gguf").display(),
            d.display(),
            d.join("ctl").display()
        );
        std::fs::create_dir_all(d.join("ctl")).unwrap();
        ServerConfig::from_json(&JsonValue::parse(&raw).unwrap()).unwrap()
    }

    #[test]
    fn fingerprint_hashes_and_cache() {
        let d = tmpdir("full");
        let cfg = cfg_with(&d);
        let fp = Arc::new(Fingerprint::new(&cfg));
        let none = |_: &str| -> Option<String> { None };
        let j = fp.to_json(&none);
        let w = j.get("weights").unwrap().as_array().unwrap();
        let paths: Vec<&str> = w.iter().map(|e| e.get_str("path").unwrap()).collect();
        assert_eq!(paths.len(), 2, "{paths:?}");
        assert!(paths[0].ends_with("model.gguf") && paths[1].ends_with("mtp.gguf"));
        assert_eq!(w[1].get_str("arg"), Some("--mtp-gguf"));
        assert!(w[0].get("sha256").unwrap().is_null());
        let exe = j.get("engine_exe").unwrap();
        assert_eq!(exe.get_str("commit"), Some("0123abcd"));
        assert_eq!(exe.get_bool("dirty"), Some(true));
        assert!(j.get("kernels_lib").unwrap().is_object());
        let text = j.to_json_string();
        assert!(!text.contains("sekrit"), "{text}");
        assert_eq!(j.get("numerics_env").unwrap().get_str("TALUS_A"), Some("1"));

        assert!(fp.start_hashing(false));
        // the hashing thread is slow on a loaded machine (5 s failed CI on busy build VMs): poll up to 30 s
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while !fp.complete() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(fp.complete());
        let j = fp.to_json(&none);
        assert_eq!(
            j.get("weights").unwrap().as_array().unwrap()[0].get_str("sha256"),
            Some(sha256_hex(b"weights-1").as_str())
        );
        assert_eq!(
            j.get("engine_exe").unwrap().get_str("sha256"),
            Some(sha256_hex(b"#!/bin/true\n").as_str())
        );
        // the engine's own INFO commit wins over the COMMIT file
        let info = |k: &str| (k == "commit").then(|| "feedbeef".to_string());
        assert_eq!(
            fp.to_json(&info)
                .get("engine_exe")
                .unwrap()
                .get_str("commit"),
            Some("feedbeef")
        );

        // a restart reuses the cache: hashes are there before any hashing
        let cache = d.join("ctl").join(CACHE_FILE);
        assert!(cache.is_file());
        let fp2 = Fingerprint::new(&cfg);
        assert!(fp2.complete());
        // a changed file (size differs) is not taken from the cache
        std::fs::write(d.join("model.gguf"), b"weights-22").unwrap();
        let fp3 = Fingerprint::new(&cfg);
        assert!(!fp3.complete());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn run_config_hash_ignores_key_and_selection() {
        let d = tmpdir("rc");
        let cfg = cfg_with(&d);
        let (h1, src) = run_config_sha256(&cfg);
        assert_eq!(src, "config");
        let mut cfg2 = cfg.clone();
        cfg2.raw_config.insert("api_key", "another-key-456789");
        cfg2.raw_config.insert(
            crate::control::SELECTION_KEY,
            JsonValue::obj(vec![("applied_at", JsonValue::from(5u64))]),
        );
        assert_eq!(run_config_sha256(&cfg2).0, h1);
        cfg2.raw_config.insert("port", 9999u64);
        assert_ne!(run_config_sha256(&cfg2).0, h1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn manifest_commit() {
        let d = tmpdir("man");
        std::fs::create_dir_all(d.join("bin")).unwrap();
        std::fs::write(
            d.join("MANIFEST.json"),
            r#"{"commit":"3ad31886","tag":"v0.1.2-rc1","targets":{"gpu":["gfx1151"]}}"#,
        )
        .unwrap();
        let m = build_meta(&d.join("bin/talus-cli"));
        assert_eq!(m.commit.as_deref(), Some("3ad31886"));
        assert_eq!(m.dirty, None);
        assert_eq!(m.tag.as_deref(), Some("v0.1.2-rc1"));
        assert_eq!(m.gfx_targets, vec!["gfx1151".to_string()]);
        let _ = std::fs::remove_dir_all(&d);
    }
}
