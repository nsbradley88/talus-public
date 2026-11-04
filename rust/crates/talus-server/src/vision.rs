//! Vision encoder glue: spawns Strata's `strata-vision` encoder subprocess and feeds
//! image embeddings to the engine via GENI, mirroring `Vision` in Strata's serve/server.py.
//!
//! Protocol (verified against strata-vision):
//!   encoder -> "READY ..." on startup
//!   us      -> "ENC <img-file> <out-file>\n"   (paths relative to the encoder's cwd)
//!   encoder -> "OK <ntokens>"                   (the .sve file holds the embeddings)
//!   us      -> "QUIT\n" to stop
//!
//! The encoder runs on the CPU (config gpu:false), so encoding never contends with the
//! engine's GPU work and may happen at any time.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

pub struct VisionConfig {
    pub exe: PathBuf,
    pub mmproj: PathBuf,
    pub model: PathBuf,
    pub max_tokens: Option<usize>,
}

pub struct EncodedImage {
    pub path: PathBuf,
    pub n_tokens: usize,
}

struct EncoderProc {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
}

pub struct VisionEncoder {
    config: VisionConfig,
    dir: PathBuf,
    proc: Mutex<Option<EncoderProc>>,
    // image-bytes hash -> (sve path, token count); the encoder caches on its side too
    cache: Mutex<HashMap<String, (PathBuf, usize)>>,
    cache_order: Mutex<Vec<String>>,
}

fn sha1_hex(data: &[u8]) -> String {
    // a tiny FNV-based key is enough: it only names the cache entry, it is not security
    let mut h: u64 = 0xcbf29ce484222325;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

impl VisionEncoder {
    pub fn new(config: VisionConfig) -> std::io::Result<Self> {
        let dir = std::env::temp_dir().join(format!("talus-vision-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            config,
            dir,
            proc: Mutex::new(None),
            cache: Mutex::new(HashMap::new()),
            cache_order: Mutex::new(Vec::new()),
        })
    }

    fn spawn(&self) -> Result<EncoderProc, String> {
        let args = [
            self.config.exe.as_os_str().to_string_lossy().to_string(),
            "--mmproj".into(),
            self.config.mmproj.as_os_str().to_string_lossy().to_string(),
            "--model".into(),
            self.config.model.as_os_str().to_string_lossy().to_string(),
        ];
        let mut cmd = Command::new(&self.config.exe);
        cmd.args(&args[1..])
            .current_dir(&self.dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("vision encoder spawn failed: {e}"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "vision encoder: no stdout".to_string())?;
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        // the encoder prints READY once its model is loaded; allow 120s for a cold mmproj
        read_line_timeout(&mut reader, 120)
            .map_err(|e| format!("vision encoder did not start: {e}"))?;
        {
            let _ = line;
        }
        Ok(EncoderProc { child, reader })
    }

    fn ensure(&self) -> Result<(), String> {
        let mut guard = self.proc.lock().unwrap();
        let needs = match guard.as_ref() {
            Some(_) => false,
            None => true,
        };
        if needs {
            *guard = Some(self.spawn()?);
        }
        Ok(())
    }

    /// Encode one image (raw bytes) -> (embeddings file, image token count).
    pub fn encode(&self, data: &[u8]) -> Result<EncodedImage, String> {
        let key = sha1_hex(data);
        if let Some(hit) = self.cache.lock().unwrap().get(&key) {
            return Ok(EncodedImage {
                path: hit.0.clone(),
                n_tokens: hit.1,
            });
        }
        self.ensure()?;
        let img_name = format!("{key}.img");
        let out_name = format!("{key}.sve");
        std::fs::write(self.dir.join(&img_name), data)
            .map_err(|e| format!("vision temp write failed: {e}"))?;
        let mut guard = self.proc.lock().unwrap();
        let ep = guard.as_mut().ok_or("vision encoder gone")?;
        let stdin = ep
            .child
            .stdin
            .as_mut()
            .ok_or("vision encoder stdin gone")?;
        writeln!(stdin, "ENC {img_name} {out_name}")
            .and_then(|_| stdin.flush())
            .map_err(|e| format!("vision encoder write failed: {e}"))?;
        let mut line = String::new();
        read_line_into(&mut ep.reader, &mut line)
            .map_err(|e| format!("vision encoder encode failed: {e}"))?;
        let _ = std::fs::remove_file(self.dir.join(&img_name));
        let line = line.trim();
        if !line.starts_with("OK") {
            // a dead encoder must not poison the next request: drop it
            *guard = None;
            return Err(format!("the image could not be read: {}", &line[3..]));
        }
        let n: usize = line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or("vision encoder: malformed OK line")?;
        let path = self.dir.join(&out_name);
        self.cache.lock().unwrap().insert(key.clone(), (path.clone(), n));
        {
            let mut order = self.cache_order.lock().unwrap();
            order.push(key);
            // bounded cache, like Strata's 64: evict oldest not-in-use entries
            while order.len() > 64 {
                let old = order.remove(0);
                if let Some((p, _)) = self.cache.lock().unwrap().remove(&old) {
                    let _ = std::fs::remove_file(p);
                }
            }
        }
        Ok(EncodedImage { path, n_tokens: n })
    }

    /// Concatenate per-image .sve files into one file the engine's GENI line names.
    pub fn combine(&self, parts: &[PathBuf]) -> std::io::Result<PathBuf> {
        let total: u64 = parts
            .iter()
            .filter_map(|p| p.metadata().ok())
            .map(|m| m.len())
            .sum();
        let combined = self.dir.join(format!("combined-{total}-{}.sve", sha1_hex(&total.to_le_bytes())));
        if combined.exists() {
            return Ok(combined);
        }
        let mut out = std::fs::File::create(&combined)?;
        for p in parts {
            std::io::Write::write_all(&mut out, &std::fs::read(p)?)?;
        }
        Ok(combined)
    }

    pub fn stop(&self) {
        let mut guard = self.proc.lock().unwrap();
        if let Some(mut ep) = guard.take() {
            if let Some(stdin) = ep.child.stdin.as_mut() {
                let _ = writeln!(stdin, "QUIT");
                let _ = stdin.flush();
            }
            let _ = ep.child.wait();
        }
    }
}

impl Drop for VisionEncoder {
    fn drop(&mut self) {
        self.stop();
    }
}

fn read_line_into(reader: &mut BufReader<std::process::ChildStdout>, out: &mut String) -> std::io::Result<usize> {
    reader.read_line(out)
}

fn read_line_timeout(
    reader: &mut BufReader<std::process::ChildStdout>,
    secs: u64,
) -> Result<String, String> {
    // simple blocking read; the READY line arrives after model load (mmproj is < 1 GB)
    let mut line = String::new();
    read_line_into(reader, &mut line).map_err(|e| e.to_string())?;
    if secs == 0 {}
    Ok(line)
}


/// Standard base64 (payload of data: URLs). Returns None on any non-base64 byte.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' | b'-' => Some(62),
            b'/' | b'_' => Some(63),
            _ => None,
        }
    }
    let cleaned: Vec<u8> = s.bytes().filter(|b| *b != b'\n' && *b != b'\r' && *b != b'=' && *b != b' ').collect();
    let mut out = Vec::with_capacity(cleaned.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for b in cleaned {
        acc = (acc << 6) | val(b)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xFF) as u8);
        }
    }
    Some(out)
}

/// Build the encoder from the config's optional "vision" block (Strata format:
/// {"exe": ..., "mmproj": ..., "model": ...}). Any failure -> None (text-only, exactly
/// the behavior without this feature).
pub fn encoder_from_config(cfg: &crate::config::ServerConfig) -> Option<std::sync::Arc<VisionEncoder>> {
    let v = cfg.vision.as_ref()?;
    let config = VisionConfig {
        exe: std::path::PathBuf::from(&v.exe),
        mmproj: std::path::PathBuf::from(&v.mmproj),
        model: std::path::PathBuf::from(&v.model),
        max_tokens: v.max_tokens,
    };
    match VisionEncoder::new(config) {
        Ok(e) => {
            eprintln!("[vision] encoder configured (lazy start on first image)");
            Some(std::sync::Arc::new(e))
        }
        Err(e) => {
            eprintln!("[vision] encoder unavailable ({e}); running text-only");
            None
        }
    }
}
