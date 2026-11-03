//! Layer 1, the only way out of the agent container. The agent container has no network (`--network none`).
//! The host side of the bridge listens on a unix socket that is bind-mounted into the container; inside, a small
//! stdlib-only Python forwarder (`FORWARDER_PY`, run as root before the agent starts, outside the audited
//! process tree) exposes it on 127.0.0.1:PORT for opencode's OpenAI-compatible client.
//!
//! The host side allows exactly `POST /v1/chat/completions` and `GET /v1/models`, one request per connection
//! (`Connection: close` is forced, so a kept-alive connection cannot smuggle a second request past the check),
//! drops any client `Authorization` header and adds the endpoint's key itself (the key never enters the
//! container), and logs every request as a JSON line (`allowed`, method, path, status, bytes, ms). Anything else
//! is answered 403 and becomes a `bridge_denied` audit flag.

use crate::util::{rfc3339, unix_now};
use serde_json::json;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const ALLOWED: [(&str, &str); 2] = [("POST", "/v1/chat/completions"), ("GET", "/v1/models")];

/// `python3 forwarder.py PORT SOCKET` (standard library only; Python 3.6+).
pub const FORWARDER_PY: &str = r#"
import socket, sys, threading
port, path = int(sys.argv[1]), sys.argv[2]
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port)); srv.listen(16)
def pipe(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d: break
            b.sendall(d)
    except Exception: pass
    for s in (a, b):
        try: s.shutdown(socket.SHUT_RDWR)
        except Exception: pass
def handle(c):
    u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try: u.connect(path)
    except Exception: c.close(); return
    threading.Thread(target=pipe, args=(c, u), daemon=True).start()
    pipe(u, c)
print("ready", flush=True)
while True:
    c, _ = srv.accept()
    threading.Thread(target=handle, args=(c,), daemon=True).start()
"#;

pub struct Bridge {
    pub socket: PathBuf,
    pub log: PathBuf,
    stop: Arc<AtomicBool>,
}

impl Drop for Bridge {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = UnixStream::connect(&self.socket); // wake accept()
        let _ = std::fs::remove_file(&self.socket);
    }
}

fn upstream_addr(base_url: &str) -> Result<(String, u16), String> {
    let rest = base_url
        .trim()
        .trim_end_matches('/')
        .trim_start_matches("http://");
    let rest = rest.strip_suffix("/v1").unwrap_or(rest);
    match rest.rsplit_once(':') {
        Some((h, p)) => Ok((
            h.to_string(),
            p.parse().map_err(|_| format!("bad port in {base_url}"))?,
        )),
        None => Ok((rest.to_string(), 80)),
    }
}

fn handle(
    mut client: UnixStream,
    upstream: &(String, u16),
    key: &Option<String>,
    log: &Mutex<std::fs::File>,
) {
    let t0 = Instant::now();
    let _ = client.set_read_timeout(Some(Duration::from_secs(60)));
    let mut reader = BufReader::new(client.try_clone().expect("clone"));
    let mut first = String::new();
    if reader.read_line(&mut first).unwrap_or(0) == 0 {
        return;
    }
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("").to_string();
    let path = target.split('?').next().unwrap_or("").to_string();
    let mut headers = Vec::new();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        let low = h.to_ascii_lowercase();
        if let Some(v) = low.strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
        if low.starts_with("authorization:")
            || low.starts_with("connection:")
            || low.starts_with("host:")
            || low.starts_with("x-api-key:")
        {
            continue;
        }
        headers.push(h);
    }
    let allowed = ALLOWED.iter().any(|(m, p)| *m == method && *p == path);
    let entry = |status: u16, bytes: usize, usage: Option<serde_json::Value>| {
        let mut line = json!({"t": rfc3339(unix_now()), "allowed": allowed, "method": method, "path": target,
                          "status": status, "bytes": bytes, "ms": t0.elapsed().as_millis() as u64});
        if let Some(u) = usage {
            line["usage"] = u;
        }
        if let Ok(mut f) = log.lock() {
            let _ = writeln!(f, "{line}");
        }
    };
    if !allowed {
        let body = r#"{"error":{"message":"talus-bench bridge: only the model API is reachable"}}"#;
        let _ = write!(client, "HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        entry(403, 0, None);
        return;
    }
    let mut body = vec![0u8; len.min(64 << 20)];
    if reader.read_exact(&mut body).is_err() {
        entry(400, 0, None);
        return;
    }
    let up = (upstream.0.as_str(), upstream.1)
        .to_socket_addrs()
        .ok()
        .and_then(|mut a| a.next())
        .and_then(|a| TcpStream::connect(a).ok());
    let Some(mut up) = up else {
        let _ = write!(
            client,
            "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        entry(502, 0, None);
        return;
    };
    let mut req = format!(
        "{method} {target} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n",
        upstream.0, upstream.1
    );
    if let Some(k) = key {
        req.push_str(&format!("Authorization: Bearer {k}\r\n"));
    }
    for h in headers {
        req.push_str(&h);
    }
    req.push_str("\r\n");
    let _ = up
        .write_all(req.as_bytes())
        .and_then(|_| up.write_all(&body));
    // stream the reply back as it comes (SSE included)
    let mut buf = [0u8; 16384];
    let mut total = 0usize;
    // the reply's tail: talus-server puts `usage` (prompt, completion, cached) in the last SSE chunk or the body
    let mut tail: Vec<u8> = Vec::new();
    let mut status = 0u16;
    loop {
        match up.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if status == 0 {
                    status = String::from_utf8_lossy(&buf[..n.min(16)])
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                }
                total += n;
                tail.extend_from_slice(&buf[..n]);
                if tail.len() > TAIL {
                    tail.drain(..tail.len() - TAIL);
                }
                if client.write_all(&buf[..n]).is_err() {
                    break;
                }
            }
        }
    }
    let _ = client.shutdown(std::net::Shutdown::Both);
    entry(status, total, last_usage(&String::from_utf8_lossy(&tail)));
}

const TAIL: usize = 64 << 10;

/// The last non-null `"usage": {...}` object in a reply (JSON body or SSE stream), with talus-server's
/// `timings` beside it when present.
pub fn last_usage(text: &str) -> Option<serde_json::Value> {
    let obj_at = |from: usize| -> Option<serde_json::Value> {
        let rest = &text[from..];
        let start = rest.find('{')?;
        if rest[..start].trim().trim_start_matches(':').trim() != "" {
            return None;
        }
        let (mut depth, mut in_str, mut esc) = (0i32, false, false);
        for (i, c) in rest[start..].char_indices() {
            match c {
                _ if esc => esc = false,
                '\\' if in_str => esc = true,
                '"' => in_str = !in_str,
                '{' if !in_str => depth += 1,
                '}' if !in_str => {
                    depth -= 1;
                    if depth == 0 {
                        return serde_json::from_str(&rest[start..start + i + 1]).ok();
                    }
                }
                _ => {}
            }
        }
        None
    };
    let mut usage = None;
    let mut pos = 0;
    while let Some(i) = text[pos..].find("\"usage\"") {
        let at = pos + i + 7;
        if let Some(v) = obj_at(at) {
            usage = Some(v);
        }
        pos = at;
    }
    let mut u = usage?;
    let mut pos = 0;
    while let Some(i) = text[pos..].find("\"timings\"") {
        let at = pos + i + 9;
        if let Some(v) = obj_at(at) {
            u["timings"] = v;
        }
        pos = at;
    }
    Some(u)
}

/// Token usage of an agent run from the bridge log: model turns and summed prompt / completion / cached tokens.
pub fn usage_summary(log: &str) -> serde_json::Value {
    let (mut turns, mut with_usage, mut p, mut c, mut k, mut maxp) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    for line in log.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v["allowed"] != true
            || v["method"] != "POST"
            || !v["path"]
                .as_str()
                .unwrap_or("")
                .contains("/chat/completions")
        {
            continue;
        }
        turns += 1;
        let u = &v["usage"];
        if u.is_object() {
            with_usage += 1;
            let pt = u["prompt_tokens"].as_u64().unwrap_or(0);
            p += pt;
            maxp = maxp.max(pt);
            c += u["completion_tokens"].as_u64().unwrap_or(0);
            k += u["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0);
        }
    }
    json!({"turns": turns, "turns_with_usage": with_usage, "prompt_tokens": p, "completion_tokens": c,
           "cached_tokens": k, "total_tokens": p + c, "max_prompt_tokens": maxp})
}

/// Start the host side: `socket` (created), forwarding to `base_url` (`http://host:port[/v1]`), logging to `log`.
pub fn start(
    socket: &Path,
    base_url: &str,
    api_key: Option<String>,
    log: &Path,
) -> Result<Bridge, String> {
    let upstream = upstream_addr(base_url)?;
    let _ = std::fs::remove_file(socket);
    let listener = UnixListener::bind(socket).map_err(|e| format!("{}: {e}", socket.display()))?;
    // the container's agent user must be able to connect; the socket lives in a 0700 directory of ours
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o666));
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .map_err(|e| format!("{}: {e}", log.display()))?;
    let logm = Arc::new(Mutex::new(file));
    let stop = Arc::new(AtomicBool::new(false));
    let st = Arc::clone(&stop);
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            if st.load(Ordering::Relaxed) {
                break;
            }
            if let Ok(c) = conn {
                let (u, k, l) = (upstream.clone(), api_key.clone(), Arc::clone(&logm));
                std::thread::spawn(move || handle(c, &u, &k, &l));
            }
        }
    });
    Ok(Bridge {
        socket: socket.to_path_buf(),
        log: log.to_path_buf(),
        stop,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn usage_from_json_and_sse_replies() {
        let body = r#"HTTP/1.1 200 OK

{"id":"x","choices":[{"message":{"content":"a \"usage\": trap"}}],"usage":{"prompt_tokens":1200,"completion_tokens":80,"total_tokens":1280,"prompt_tokens_details":{"cached_tokens":1024}},"timings":{"prompt_ms":50.5,"predicted_ms":900.0}}"#;
        let u = super::last_usage(body).unwrap();
        assert_eq!(u["prompt_tokens"], 1200);
        assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 1024);
        assert_eq!(u["timings"]["predicted_ms"], 900.0);
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":null}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\ndata: [DONE]\n\n";
        assert_eq!(super::last_usage(sse).unwrap()["completion_tokens"], 2);
        assert!(super::last_usage("{\"usage\": null}").is_none());
        let log = format!(
            "{}\n{}\n{}\n",
            json!({"allowed": true, "method": "POST", "path": "/v1/chat/completions", "usage": u}),
            json!({"allowed": true, "method": "POST", "path": "/v1/chat/completions", "usage": {"prompt_tokens": 2000, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 1200}}}),
            json!({"allowed": false, "method": "GET", "path": "/v1/talus/status"})
        );
        let s = super::usage_summary(&log);
        assert_eq!(s["turns"], 2);
        assert_eq!(s["prompt_tokens"], 3200);
        assert_eq!(s["completion_tokens"], 100);
        assert_eq!(s["cached_tokens"], 2224);
        assert_eq!(s["max_prompt_tokens"], 2000);
    }

    use super::*;
    use crate::mock::{spawn_openai, Degrade};

    #[test]
    fn only_the_model_api_passes_through_the_forwarder() {
        if std::process::Command::new("python3")
            .arg("-V")
            .output()
            .is_err()
        {
            return;
        }
        let (addr, count) = spawn_openai(0, Degrade::NONE).unwrap();
        let dir = std::env::temp_dir().join(format!("tb-bridge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("model.sock");
        let log = dir.join("bridge.jsonl");
        let _b = start(
            &sock,
            &format!("http://{addr}"),
            Some("host-side-key".into()),
            &log,
        )
        .unwrap();
        // the in-container forwarder, here on the host
        let fwd = dir.join("fwd.py");
        std::fs::write(&fwd, FORWARDER_PY).unwrap();
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let mut child = std::process::Command::new("python3")
            .arg("-I")
            .arg(&fwd)
            .arg(port.to_string())
            .arg(&sock)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut ready = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut ready)
            .unwrap();
        assert_eq!(ready.trim(), "ready");
        let c = crate::http::Client::new(
            &format!("http://127.0.0.1:{port}"),
            Some("agent-guess".into()),
            10,
        )
        .unwrap();
        let ok = c
            .post(
                "/v1/chat/completions",
                &json!({"messages": [{"role": "user", "content": "hi"}]}),
            )
            .unwrap();
        assert_eq!(ok.status, 200);
        assert_eq!(c.get("/v1/models").unwrap().status, 200);
        for (m, p) in [
            ("GET", "/v1/talus/status"),
            ("POST", "/v1/talus/apply"),
            ("GET", "/health"),
        ] {
            let r = if m == "GET" {
                c.get(p)
            } else {
                c.post(p, &json!({}))
            }
            .unwrap();
            assert_eq!(r.status, 403, "{m} {p}");
        }
        assert_eq!(
            *count.lock().unwrap(),
            1,
            "denied requests never reach the endpoint"
        );
        child.kill().ok();
        child.wait().ok();
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(&log)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 5);
        assert_eq!(lines.iter().filter(|l| l["allowed"] == true).count(), 2);
        let mut a = super::super::audit::Audit::default();
        super::super::audit::audit_bridge(&std::fs::read_to_string(&log).unwrap(), &mut a);
        assert_eq!(a.flags.len(), 3);
        std::fs::remove_dir_all(&dir).ok();
    }
}
