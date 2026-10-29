//! Downloads from Hugging Face: resumable, checked against SHA-256, with a progress line.
//!
//! The transfer is `curl` (every system that ran the install one-liner has it): `-C -` resumes a partial file, and
//! `--retry` rides out dropped connections. A file is downloaded to `<name>.part`, hashed, and only then renamed;
//! a file whose hash does not match is kept as `<name>.bad` and reported. A verified file gets a stamp (size, mtime,
//! hash) in `<model dir>/.talus-verified.json`, so later checks need not hash 100 GB again.
//!
//! No token is needed (every repository is public). `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) is used when set, for
//! rate limits: it goes to curl on its standard input (`-H @-`), never on a command line, and is never printed or
//! stored. `HF_ENDPOINT` points at a mirror.

use crate::manifest::FileSpec;
use crate::plan::{gb, part_name};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const STAMP_FILE: &str = ".talus-verified.json";

pub fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

/// The token for rate limits, if the user set one.
pub fn token() -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        .iter()
        .find_map(|k| std::env::var(k).ok())
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

pub fn resolve_url(endpoint: &str, repo: &str, path: &str) -> String {
    format!("{endpoint}/{repo}/resolve/main/{path}")
}

/// The curl arguments for a transfer (the token, if any, is fed on stdin and is not in here).
pub fn curl_args(
    url: &str,
    out: &Path,
    resume: bool,
    with_token: bool,
    retry_all: bool,
) -> Vec<String> {
    let mut a: Vec<String> = [
        "-fL",
        "-sS",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    if retry_all {
        a.push("--retry-all-errors".into());
    }
    if resume {
        a.extend(["-C".to_string(), "-".to_string()]);
    }
    if with_token {
        a.extend(["-H".to_string(), "@-".to_string()]);
    }
    a.extend(["-o".to_string(), out.display().to_string(), url.to_string()]);
    a
}

fn curl_supports_retry_all() -> bool {
    // --retry-all-errors arrived in curl 7.71
    let Ok(out) = Command::new("curl").arg("--version").output() else {
        return false;
    };
    let v = String::from_utf8_lossy(&out.stdout);
    let ver = v.split_whitespace().nth(1).unwrap_or("0.0");
    let mut it = ver.split('.').map(|x| x.parse::<u32>().unwrap_or(0));
    let (maj, min) = (it.next().unwrap_or(0), it.next().unwrap_or(0));
    (maj, min) >= (7, 71)
}

/// Run curl with the token (if any) on its stdin. Returns curl's exit status and its stderr.
fn run_curl(args: &[String], tok: Option<&str>) -> std::io::Result<(i32, String)> {
    let mut child = Command::new("curl")
        .args(args)
        .stdin(if tok.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let (Some(t), Some(mut stdin)) = (tok, child.stdin.take()) {
        let _ = stdin.write_all(format!("Authorization: Bearer {t}\n").as_bytes());
    }
    let out = child.wait_with_output()?;
    Ok((
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stderr).trim().to_string(),
    ))
}

/// GET a small text resource.
pub fn http_get_text(url: &str) -> Result<String, String> {
    let tok = token();
    let mut args: Vec<String> = ["-fsSL", "--max-time", "60", "--retry", "3"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    if tok.is_some() {
        args.extend(["-H".to_string(), "@-".to_string()]);
    }
    args.push(url.to_string());
    let mut child = Command::new("curl")
        .args(&args)
        .stdin(if tok.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("curl: {e}"))?;
    if let (Some(t), Some(mut stdin)) = (tok.as_deref(), child.stdin.take()) {
        let _ = stdin.write_all(format!("Authorization: Bearer {t}\n").as_bytes());
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `sha256  name` lines of a SHA256SUMS file.
pub fn parse_sha256sums(s: &str) -> BTreeMap<String, String> {
    s.lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let h = it.next()?;
            let name = it.next()?.trim_start_matches('*');
            (h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()))
                .then(|| (name.to_string(), h.to_ascii_lowercase()))
        })
        .collect()
}

/// The LFS sha256 of `path` in a Hugging Face tree listing (JSON).
pub fn lfs_oid_in_tree(tree_json: &str, path: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(tree_json).ok()?;
    v.as_array()?.iter().find_map(|e| {
        (e.get("path")?.as_str()? == path)
            .then(|| e.get("lfs")?.get("oid")?.as_str().map(str::to_string))
            .flatten()
    })
}

/// The SHA-256 the repository publishes for `f` right now: our SHA256SUMS for j2ipper repositories, the LFS object
/// id elsewhere. `None` when it cannot be fetched (offline): the manifest's value is used alone then.
pub fn published_sha256(f: &FileSpec, cache: &mut BTreeMap<String, String>) -> Option<String> {
    let ep = endpoint();
    let (key, url) = if f.has_sha256sums() {
        (
            format!("sums:{}", f.repo),
            resolve_url(&ep, f.repo, "SHA256SUMS"),
        )
    } else {
        let dir = f.path.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
        let url = if dir.is_empty() {
            format!("{ep}/api/models/{}/tree/main", f.repo)
        } else {
            format!("{ep}/api/models/{}/tree/main/{dir}", f.repo)
        };
        (format!("tree:{}:{dir}", f.repo), url)
    };
    if !cache.contains_key(&key) {
        let text = http_get_text(&url).ok()?;
        cache.insert(key.clone(), text);
    }
    let text = &cache[&key];
    if f.has_sha256sums() {
        parse_sha256sums(text).get(f.path).cloned()
    } else {
        lfs_oid_in_tree(text, f.path)
    }
}

/// Progress on stderr: a redrawn line on a terminal, a line per 10 % otherwise.
pub struct Progress {
    label: String,
    total: u64,
    start: Instant,
    start_bytes: u64,
    tty: bool,
    last_decile: u64,
    quiet: bool,
}

impl Progress {
    pub fn new(label: &str, total: u64, start_bytes: u64, quiet: bool) -> Progress {
        Progress {
            label: label.to_string(),
            total,
            start: Instant::now(),
            start_bytes,
            tty: std::io::stderr().is_terminal(),
            last_decile: start_bytes * 10 / total.max(1),
            quiet,
        }
    }

    pub fn line(&self, done: u64, width: usize) -> String {
        let frac = if self.total == 0 {
            1.0
        } else {
            done as f64 / self.total as f64
        };
        let filled = ((frac * width as f64) as usize).min(width);
        let secs = self.start.elapsed().as_secs_f64().max(0.001);
        let rate = (done.saturating_sub(self.start_bytes)) as f64 / secs;
        let eta = if rate > 1.0 {
            fmt_secs((self.total.saturating_sub(done)) as f64 / rate)
        } else {
            "--".to_string()
        };
        format!(
            "  {:<44} [{}{}] {:>3.0}%  {} / {}  {:>6.1} MB/s  ETA {}",
            truncate(&self.label, 44),
            "#".repeat(filled),
            "-".repeat(width - filled),
            frac * 100.0,
            gb(done),
            gb(self.total),
            rate / 1e6,
            eta
        )
    }

    pub fn update(&mut self, done: u64) {
        if self.quiet {
            return;
        }
        if self.tty {
            eprint!("\r{}\x1b[K", self.line(done, 24));
            let _ = std::io::stderr().flush();
        } else {
            let d = done * 10 / self.total.max(1);
            if d > self.last_decile {
                self.last_decile = d;
                eprintln!("{}", self.line(done, 10));
            }
        }
    }

    pub fn finish(&mut self, done: u64) {
        if self.quiet {
            return;
        }
        if self.tty {
            eprintln!("\r{}\x1b[K", self.line(done, 24));
        } else if done * 10 / self.total.max(1) > self.last_decile {
            eprintln!("{}", self.line(done, 10));
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let t: String = s.chars().take(n.saturating_sub(1)).collect();
        format!("{t}~")
    }
}

pub fn fmt_secs(s: f64) -> String {
    let s = s.max(0.0) as u64;
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

/// SHA-256 of a file, with an optional progress line.
pub fn sha256_file(p: &Path, progress: Option<&mut Progress>) -> std::io::Result<String> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 4 << 20];
    let mut done = 0u64;
    let mut last = Instant::now();
    let mut prog = progress;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        done += n as u64;
        if let Some(pr) = prog.as_deref_mut() {
            if last.elapsed() > Duration::from_millis(500) {
                pr.update(done);
                last = Instant::now();
            }
        }
    }
    if let Some(pr) = prog {
        pr.finish(done);
    }
    Ok(hex(&h.finalize()))
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ------------------------------------------------------------------------------------------------- stamps

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Stamp {
    pub size: u64,
    pub mtime: i64,
    pub sha256: String,
}

/// Modification time in nanoseconds (seconds alone miss a rewrite within the same second).
fn mtime(p: &Path) -> Option<i64> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(p).ok().map(|m| {
        m.mtime()
            .saturating_mul(1_000_000_000)
            .saturating_add(m.mtime_nsec())
    })
}

pub fn read_stamps(dir: &Path) -> BTreeMap<String, Stamp> {
    std::fs::read_to_string(dir.join(STAMP_FILE))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn write_stamp(dir: &Path, name: &str, sha256: &str) -> std::io::Result<()> {
    let p = dir.join(name);
    let mut all = read_stamps(dir);
    all.insert(
        name.to_string(),
        Stamp {
            size: std::fs::metadata(&p)?.len(),
            mtime: mtime(&p).unwrap_or(0),
            sha256: sha256.to_string(),
        },
    );
    let text = serde_json::to_string_pretty(&all).expect("json") + "\n";
    crate::state::write_atomic(&dir.join(STAMP_FILE), text.as_bytes(), 0o644)
}

/// The file is there, has the right size, and was hashed to the expected value since it last changed.
pub fn stamped_ok(dir: &Path, f: &FileSpec) -> bool {
    let p = dir.join(f.local_name());
    let Some(st) = read_stamps(dir).remove(f.local_name()) else {
        return false;
    };
    st.sha256 == f.sha256
        && std::fs::metadata(&p).map(|m| m.len()).ok() == Some(f.size)
        && st.size == f.size
        && mtime(&p) == Some(st.mtime)
}

// ------------------------------------------------------------------------------------------------- fetch

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Already there and verified earlier.
    Present,
    /// There with the right size; hashed now and good.
    Verified,
    Downloaded,
}

pub struct FetchOpts {
    pub quiet: bool,
    /// Check the repository's published hash against the manifest before downloading.
    pub check_published: bool,
}

/// Get `f` into `dir`, verified. Resumes a `.part` left by an earlier run.
pub fn fetch(
    f: &FileSpec,
    dir: &Path,
    opts: &FetchOpts,
    sums_cache: &mut BTreeMap<String, String>,
) -> Result<Outcome, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let dest = dir.join(f.local_name());
    let part = dir.join(part_name(f));
    if stamped_ok(dir, f) {
        return Ok(Outcome::Present);
    }
    if std::fs::metadata(&dest).map(|m| m.len()).ok() == Some(f.size) {
        let mut pr = Progress::new(
            &format!("checking {}", f.local_name()),
            f.size,
            0,
            opts.quiet,
        );
        let h = sha256_file(&dest, Some(&mut pr)).map_err(|e| e.to_string())?;
        if h == f.sha256 {
            write_stamp(dir, f.local_name(), &h).map_err(|e| e.to_string())?;
            return Ok(Outcome::Verified);
        }
        // corrupt: download it again (the bad copy is deleted once the new one verifies)
        let bad = dir.join(format!("{}.bad", f.local_name()));
        std::fs::rename(&dest, &bad).map_err(|e| e.to_string())?;
        eprintln!(
            "  {} did not match its checksum; downloading it again",
            f.local_name()
        );
    }
    if opts.check_published {
        if let Some(pubh) = published_sha256(f, sums_cache) {
            if pubh != f.sha256 {
                return Err(format!(
                    "{}/{} changed upstream since this Talus release (published sha256 {}, expected {}). \
                     Not downloading it: run `talus update`, or report this.",
                    f.repo,
                    f.path,
                    &pubh[..12],
                    &f.sha256[..12]
                ));
            }
        }
    }
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if have > f.size {
        std::fs::remove_file(&part).map_err(|e| e.to_string())?;
    }
    let have = have.min(f.size);
    if have < f.size {
        let url = resolve_url(&endpoint(), f.repo, f.path);
        let tok = token();
        let args = curl_args(&url, &part, true, tok.is_some(), curl_supports_retry_all());
        let mut pr = Progress::new(f.local_name(), f.size, have, opts.quiet);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let watcher = {
            let stop = stop.clone();
            let part = part.clone();
            let total = f.size;
            let quiet = opts.quiet;
            let label = f.local_name().to_string();
            std::thread::spawn(move || {
                let mut pr = Progress::new(&label, total, have, quiet);
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let n = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
                    pr.update(n.min(total));
                    std::thread::sleep(Duration::from_millis(500));
                }
            })
        };
        let res = run_curl(&args, tok.as_deref());
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = watcher.join();
        let got = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        pr.finish(got.min(f.size));
        match res {
            Ok((0, _)) => {}
            Ok((code, err)) => {
                return Err(format!(
                    "download of {} stopped (curl exit {code}: {err}). Run the same command again to resume.",
                    f.local_name()
                ))
            }
            Err(e) => return Err(format!("could not run curl: {e}")),
        }
    }
    let got = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if got != f.size {
        return Err(format!(
            "{}: got {} of {} bytes. Run the same command again to resume.",
            f.local_name(),
            got,
            f.size
        ));
    }
    let mut pr = Progress::new(
        &format!("verifying {}", f.local_name()),
        f.size,
        0,
        opts.quiet,
    );
    let h = sha256_file(&part, Some(&mut pr)).map_err(|e| e.to_string())?;
    if h != f.sha256 {
        // do not keep tens of GB of a bad download around; the next run starts over
        let _ = std::fs::remove_file(&part);
        return Err(format!(
            "{}: checksum mismatch after download (got {}, expected {}); the file was deleted. \
             Run again to download it once more; if it fails again, report it.",
            f.local_name(),
            &h[..12],
            &f.sha256[..12]
        ));
    }
    std::fs::rename(&part, &dest).map_err(|e| e.to_string())?;
    write_stamp(dir, f.local_name(), &h).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(dir.join(format!("{}.bad", f.local_name())));
    Ok(Outcome::Downloaded)
}

/// Where a file of `f` would go.
pub fn dest_of(dir: &Path, f: &FileSpec) -> PathBuf {
    dir.join(f.local_name())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        let dir = crate::testutil::tempdir("sha");
        let p = dir.join("x");
        std::fs::write(&p, b"").unwrap();
        assert_eq!(
            sha256_file(&p, None).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p, None).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // more than one read buffer
        std::fs::write(&p, vec![b'a'; 5_000_000]).unwrap();
        let mut h = Sha256::new();
        h.update(vec![b'a'; 5_000_000]);
        assert_eq!(sha256_file(&p, None).unwrap(), hex(&h.finalize()));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn sums_and_tree_parsing() {
        let s = "55d63bbd101508ba4a4ec6ded0736c151dd538b52345420bd33260e45699c4df  glm-dv-65536.bin\n\
                 junk line\n\
                 1ba03853c291fd3fdaa7cacdf0ed8b7b15dda1fa4482f48d8516c3d8975bb553 *recipe/make_vocab.py\n";
        let m = parse_sha256sums(s);
        assert_eq!(m.len(), 2);
        assert!(m["glm-dv-65536.bin"].starts_with("55d63bbd"));
        assert!(m.contains_key("recipe/make_vocab.py"));
        let tree = r#"[{"type":"file","path":"MTP/README.md","size":5},
                       {"type":"file","path":"MTP/mtp-Qwen3.8-27B-Q4_0.gguf","size":1,"lfs":{"oid":"50d9ce","size":1}}]"#;
        assert_eq!(
            lfs_oid_in_tree(tree, "MTP/mtp-Qwen3.8-27B-Q4_0.gguf").as_deref(),
            Some("50d9ce")
        );
        assert_eq!(lfs_oid_in_tree(tree, "MTP/README.md"), None);
        assert_eq!(lfs_oid_in_tree("not json", "x"), None);
    }

    #[test]
    fn the_token_never_reaches_the_command_line() {
        let a = curl_args(
            "https://huggingface.co/r/resolve/main/f",
            Path::new("/tmp/f.part"),
            true,
            true,
            true,
        );
        assert!(a.windows(2).any(|w| w[0] == "-H" && w[1] == "@-"));
        assert!(a.windows(2).any(|w| w[0] == "-C" && w[1] == "-"));
        assert!(!a.iter().any(|x| x.contains("Bearer")));
        let b = curl_args("u", Path::new("/o"), false, false, false);
        assert!(!b.contains(&"-H".to_string()) && !b.contains(&"-C".to_string()));
        assert_eq!(
            resolve_url(
                "https://huggingface.co",
                "j2ipper/draft-vocab",
                "glm-dv-65536.bin"
            ),
            "https://huggingface.co/j2ipper/draft-vocab/resolve/main/glm-dv-65536.bin"
        );
    }

    #[test]
    fn stamps_track_size_and_mtime() {
        let dir = crate::testutil::tempdir("stamp");
        let body = b"hello talus";
        let mut h = Sha256::new();
        h.update(body);
        let sha = hex(&h.finalize());
        let f = FileSpec {
            repo: "r/x",
            path: "a/b.bin",
            size: body.len() as u64,
            sha256: Box::leak(sha.clone().into_boxed_str()),
            role: crate::manifest::Role::Weights,
        };
        assert!(!stamped_ok(&dir, &f));
        std::fs::write(dir.join("b.bin"), body).unwrap();
        write_stamp(&dir, "b.bin", &sha).unwrap();
        assert!(stamped_ok(&dir, &f));
        // touching the file invalidates the stamp
        let t = std::time::SystemTime::now() + Duration::from_secs(5);
        std::fs::File::options()
            .write(true)
            .open(dir.join("b.bin"))
            .unwrap()
            .set_modified(t)
            .unwrap();
        assert!(!stamped_ok(&dir, &f));
        std::fs::remove_dir_all(dir).ok();
    }

    /// Every manifest hash equals what its repository publishes now (SHA256SUMS or LFS oid). Network; run with
    /// `cargo test -p talus-setup -- --ignored published`.
    #[test]
    #[ignore]
    fn published_hashes_match_the_manifest() {
        let mut cache = BTreeMap::new();
        for m in crate::manifest::MODELS {
            for f in m.files {
                let p = published_sha256(f, &mut cache)
                    .unwrap_or_else(|| panic!("no published hash for {}/{}", f.repo, f.path));
                assert_eq!(p, f.sha256, "{}/{}", f.repo, f.path);
            }
        }
    }

    /// End to end through curl with a file:// URL: resume from a partial file, verify, stamp, and catch corruption.
    #[test]
    fn fetch_resumes_and_verifies_via_file_urls() {
        if Command::new("curl").arg("--version").output().is_err() {
            return;
        }
        let dir = crate::testutil::tempdir("fetch");
        let src = dir.join("hub/org/repo/resolve/main");
        std::fs::create_dir_all(&src).unwrap();
        let body: Vec<u8> = (0..3_000_000u32).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(src.join("w.gguf"), &body).unwrap();
        let mut h = Sha256::new();
        h.update(&body);
        let sha = hex(&h.finalize());
        let f = FileSpec {
            repo: "org/repo",
            path: "w.gguf",
            size: body.len() as u64,
            sha256: Box::leak(sha.into_boxed_str()),
            role: crate::manifest::Role::Weights,
        };
        let out = dir.join("models");
        std::fs::create_dir_all(&out).unwrap();
        // a third of it already there
        std::fs::write(out.join("w.gguf.part"), &body[..1_000_000]).unwrap();
        std::env::set_var(
            "HF_ENDPOINT",
            format!("file://{}", dir.join("hub").display()),
        );
        let opts = FetchOpts {
            quiet: true,
            check_published: false,
        };
        let mut cache = BTreeMap::new();
        assert_eq!(
            fetch(&f, &out, &opts, &mut cache).unwrap(),
            Outcome::Downloaded
        );
        assert_eq!(std::fs::read(out.join("w.gguf")).unwrap(), body);
        assert!(!out.join("w.gguf.part").exists());
        assert_eq!(
            fetch(&f, &out, &opts, &mut cache).unwrap(),
            Outcome::Present
        );
        // corrupt the file: the next fetch moves it aside and downloads it again
        let mut bad = body.clone();
        bad[10] ^= 1;
        std::fs::write(out.join("w.gguf"), &bad).unwrap();
        assert_eq!(
            fetch(&f, &out, &opts, &mut cache).unwrap(),
            Outcome::Downloaded
        );
        assert!(
            !out.join("w.gguf.bad").exists(),
            "the bad copy goes once the new one verifies"
        );
        assert_eq!(std::fs::read(out.join("w.gguf")).unwrap(), body);
        // a source that does not match the manifest is refused and deleted
        let g = FileSpec {
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
            ..f
        };
        std::fs::remove_file(out.join("w.gguf")).unwrap();
        std::fs::remove_file(out.join(STAMP_FILE)).unwrap();
        let e = fetch(&g, &out, &opts, &mut cache).unwrap_err();
        assert!(e.contains("checksum mismatch"), "{e}");
        assert!(!out.join("w.gguf.part").exists() && !out.join("w.gguf").exists());
        assert!(!out.join("w.gguf.part").exists() && !out.join("w.gguf").exists());
        std::env::remove_var("HF_ENDPOINT");
        std::fs::remove_dir_all(dir).ok();
    }
}
