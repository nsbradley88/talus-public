//! A small blocking HTTP/1.1 client for talus-server (one request per connection, plain HTTP, JSON bodies).
//! Same approach as talus-tui's client, with long, configurable timeouts for generation and a typed error so a
//! timeout, a refused connection and a server error stay distinct case statuses.

use serde_json::Value;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq)]
pub enum HttpError {
    /// Could not resolve or connect.
    Connect(String),
    /// No complete reply within the timeout.
    Timeout(String),
    /// Connected, but the reply was not valid HTTP/JSON.
    Protocol(String),
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpError::Connect(m) => write!(f, "connect: {m}"),
            HttpError::Timeout(m) => write!(f, "timeout: {m}"),
            HttpError::Protocol(m) => write!(f, "protocol: {m}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Reply {
    pub status: u16,
    pub body: Value,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug)]
pub struct Client {
    host: String,
    port: u16,
    api_key: Option<String>,
    timeout: Duration,
}

impl Client {
    /// `url` like `http://127.0.0.1:8000` or `http://host:8000/v1` (the `/v1` suffix is dropped).
    pub fn new(url: &str, api_key: Option<String>, timeout_s: u64) -> Result<Self, String> {
        let rest = url.trim().trim_end_matches('/');
        if rest.starts_with("https://") {
            return Err("https is not supported; talus-server speaks plain HTTP".into());
        }
        let rest = rest.strip_prefix("http://").unwrap_or(rest);
        let rest = rest.strip_suffix("/v1").unwrap_or(rest);
        if rest.contains('/') {
            return Err(format!("{url:?}: only http://host:port[/v1] is supported"));
        }
        let (host, port) = match rest.rsplit_once(':') {
            Some((h, p)) if !h.is_empty() => (
                h.trim_start_matches('[').trim_end_matches(']').to_string(),
                p.parse::<u16>()
                    .map_err(|_| format!("bad port in {url:?}"))?,
            ),
            _ => (rest.to_string(), 8000),
        };
        if host.is_empty() {
            return Err(format!("no host in {url:?}"));
        }
        Ok(Self {
            host,
            port,
            api_key: api_key.filter(|k| !k.is_empty()),
            timeout: Duration::from_secs(timeout_s.max(1)),
        })
    }

    pub fn url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    pub fn with_timeout(&self, secs: u64) -> Self {
        let mut c = self.clone();
        c.timeout = Duration::from_secs(secs.max(1));
        c
    }

    pub fn get(&self, path: &str) -> Result<Reply, HttpError> {
        self.request("GET", path, None)
    }

    pub fn post(&self, path: &str, body: &Value) -> Result<Reply, HttpError> {
        self.request("POST", path, Some(body.to_string()))
    }

    /// talus-server's teacher-forced scoring endpoint (server extension E2): `POST /v1/talus/score` with
    /// `prompt_ids`/`prompt`, `continuation_ids`/`continuation`, `top_k` (0..=64) and optionally `return`.
    /// The reply is returned as is; [`unsupported_reason`] tells an engine or server without scoring apart.
    pub fn score(&self, body: &Value) -> Result<Reply, HttpError> {
        self.post("/v1/talus/score", body)
    }

    fn request(&self, method: &str, path: &str, body: Option<String>) -> Result<Reply, HttpError> {
        let t0 = Instant::now();
        let addr = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(|e| HttpError::Connect(format!("{}: {e}", self.url())))?
            .next()
            .ok_or_else(|| HttpError::Connect(format!("{}: no address", self.url())))?;
        let connect_to = self.timeout.min(Duration::from_secs(10));
        let mut s = TcpStream::connect_timeout(&addr, connect_to)
            .map_err(|e| HttpError::Connect(format!("{}: {e}", self.url())))?;
        s.set_read_timeout(Some(self.timeout)).ok();
        s.set_write_timeout(Some(self.timeout)).ok();
        let body = body.unwrap_or_default();
        let mut req = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}:{}\r\nAccept: application/json\r\nUser-Agent: talus-bench/{}\r\nConnection: close\r\n",
            self.host,
            self.port,
            env!("CARGO_PKG_VERSION")
        );
        if let Some(k) = &self.api_key {
            req.push_str(&format!("Authorization: Bearer {k}\r\n"));
        }
        if method == "POST" {
            req.push_str("Content-Type: application/json\r\n");
            req.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
        req.push_str("\r\n");
        req.push_str(&body);
        s.write_all(req.as_bytes())
            .map_err(|e| HttpError::Connect(format!("{}: {e}", self.url())))?;
        let mut raw = Vec::new();
        let deadline = t0 + self.timeout;
        let mut buf = [0u8; 16384];
        loop {
            if Instant::now() > deadline {
                return Err(HttpError::Timeout(format!(
                    "{} {path}: no complete reply in {} s",
                    self.url(),
                    self.timeout.as_secs()
                )));
            }
            match s.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => raw.extend_from_slice(&buf[..n]),
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    return Err(HttpError::Timeout(format!(
                        "{} {path}: no reply in {} s",
                        self.url(),
                        self.timeout.as_secs()
                    )))
                }
                Err(e) => return Err(HttpError::Protocol(format!("{}: {e}", self.url()))),
            }
        }
        let (status, body) = parse_response(&raw).map_err(HttpError::Protocol)?;
        Ok(Reply {
            status,
            body,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

/// The error for a reply without a header end: the peer closed without answering, or the headers are cut short.
fn no_header_end(n: usize) -> String {
    if n == 0 {
        "connection closed before any response".to_string()
    } else {
        format!("malformed HTTP response: no end of headers in {n} bytes")
    }
}

/// Status code and JSON body of a raw HTTP/1.1 response (Content-Length or read-to-close; chunked is decoded).
pub fn parse_response(raw: &[u8]) -> Result<(u16, Value), String> {
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| no_header_end(raw.len()))?;
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let payload = &raw[split + 4..];
    let code: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .ok_or_else(|| "malformed HTTP status line".to_string())?;
    let chunked = head.lines().any(|l| {
        let l = l.to_ascii_lowercase();
        l.starts_with("transfer-encoding:") && l.contains("chunked")
    });
    let body = if chunked {
        dechunk(payload)
    } else {
        payload.to_vec()
    };
    let text = String::from_utf8_lossy(&body);
    let value = if text.trim().is_empty() {
        Value::Null
    } else {
        match serde_json::from_str(&text) {
            Ok(v) => v,
            // non-JSON error pages are kept as text so the case record shows them
            Err(_) if code >= 400 => Value::String(text.chars().take(2000).collect()),
            Err(e) => return Err(format!("bad JSON from server ({code}): {e}")),
        }
    };
    Ok((code, value))
}

fn dechunk(s: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(p) = rest.windows(2).position(|w| w == b"\r\n") {
        let size_line = String::from_utf8_lossy(&rest[..p]);
        let size = usize::from_str_radix(size_line.trim(), 16).unwrap_or(0);
        let after = &rest[p + 2..];
        if size == 0 || after.len() < size {
            break;
        }
        out.extend_from_slice(&after[..size]);
        rest = &after[size..];
        if rest.starts_with(b"\r\n") {
            rest = &rest[2..];
        }
    }
    out
}

/// The `error.message` of an error reply, else the whole body.
pub fn error_message(v: &Value) -> String {
    v.get("error")
        .and_then(|e| e.get("message").or(Some(e)))
        .and_then(|m| m.as_str().map(str::to_string))
        .unwrap_or_else(|| v.to_string())
}

/// Why a logprob or scoring request cannot work on this endpoint, when the reply says so: 501 (talus-server:
/// "scoring not supported by this engine"), a 400 whose message says the engine does not support logprobs, or
/// 404/405 (a server without the route: an older talus-server or another OpenAI-compatible server). None for
/// every other reply (success, a bad request, an engine failure), which are ordinary case errors.
pub fn unsupported_reason(r: &Reply) -> Option<String> {
    let msg = error_message(&r.body);
    let unsupported_400 = r.status == 400 && msg.contains("not supported by this engine");
    match r.status {
        501 => Some(format!("HTTP 501: {msg}")),
        404 | 405 => Some(format!(
            "HTTP {}: the server has no such route (needs talus-server with the scoring extension)",
            r.status
        )),
        _ if unsupported_400 => Some(format!("HTTP 400: {msg}")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls() {
        assert_eq!(
            Client::new("http://127.0.0.1:8000/v1/", None, 5)
                .unwrap()
                .url(),
            "http://127.0.0.1:8000"
        );
        assert_eq!(
            Client::new("gpu-box", None, 5).unwrap().url(),
            "http://gpu-box:8000"
        );
        assert!(Client::new("https://a:1", None, 5).is_err());
        assert!(Client::new("http://a:1/v2/x", None, 5).is_err());
        assert!(Client::new("http://a:zz", None, 5).is_err());
    }

    #[test]
    fn closed_connection_is_not_called_malformed() {
        assert_eq!(
            parse_response(b"").unwrap_err(),
            "connection closed before any response"
        );
        // partial headers keep their own message, with the byte count
        assert_eq!(
            parse_response(b"HTTP/1.1 200 OK\r\n").unwrap_err(),
            "malformed HTTP response: no end of headers in 17 bytes"
        );
    }

    /// A server stopped mid-request: request read, connection closed, nothing sent. Still `Protocol`, so the
    /// case stays a retryable `protocol_error` and nothing is scored.
    #[test]
    fn server_closing_without_a_reply_stays_a_retryable_protocol_error() {
        use crate::case::CaseStatus;
        let reply = |bytes: &'static [u8]| {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("http://{}", l.local_addr().unwrap());
            let t = std::thread::spawn(move || {
                let (mut s, _) = l.accept().unwrap();
                let mut got = Vec::new();
                let mut b = [0u8; 1024];
                while !got.windows(4).any(|w| w == b"\r\n\r\n") {
                    let n = s.read(&mut b).unwrap();
                    assert!(n > 0);
                    got.extend_from_slice(&b[..n]);
                }
                s.write_all(bytes).unwrap();
            });
            let r = Client::new(&url, None, 5).unwrap().get("/x");
            t.join().unwrap();
            r
        };
        let e = reply(b"").unwrap_err();
        assert_eq!(
            e,
            HttpError::Protocol("connection closed before any response".into())
        );
        assert_eq!(
            e.to_string(),
            "protocol: connection closed before any response"
        );
        assert_eq!(
            reply(b"HTTP/1.1 2").unwrap_err(),
            HttpError::Protocol("malformed HTTP response: no end of headers in 10 bytes".into())
        );
        assert!(!CaseStatus::ProtocolError.scored());
    }

    #[test]
    fn responses() {
        let (c, v) =
            parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\n{\"a\":1}").unwrap();
        assert_eq!((c, v["a"].as_i64()), (200, Some(1)));
        let (c, v) = parse_response(
            b"HTTP/1.1 503 X\r\nTransfer-Encoding: chunked\r\n\r\n5\r\n{\"err\r\n14\r\nor\":{\"message\":\"x\"}}\r\n0\r\n\r\n",
        )
        .unwrap();
        assert_eq!(c, 503);
        assert_eq!(error_message(&v), "x");
        let (c, v) = parse_response(b"HTTP/1.1 502 Bad\r\n\r\n<html>oops</html>").unwrap();
        assert_eq!(c, 502);
        assert_eq!(v, Value::String("<html>oops</html>".into()));
        assert!(parse_response(b"HTTP/1.1 200 OK\r\n\r\nnot json").is_err());
    }

    #[test]
    fn unsupported_replies() {
        let r = |status: u16, body: Value| Reply {
            status,
            body,
            elapsed_ms: 0.0,
        };
        let e = |m: &str| serde_json::json!({"error": {"message": m}});
        assert!(
            unsupported_reason(&r(501, e("scoring not supported by this engine")))
                .unwrap()
                .contains("scoring not supported")
        );
        assert!(unsupported_reason(&r(400, e("logprobs not supported by this engine"))).is_some());
        assert!(unsupported_reason(&r(404, e("not found"))).is_some());
        assert!(
            unsupported_reason(&r(400, e("top_k: must be an integer between 0 and 64"))).is_none()
        );
        assert!(unsupported_reason(&r(500, e("engine died"))).is_none());
        assert!(unsupported_reason(&r(200, serde_json::json!({"positions": []}))).is_none());
    }

    #[test]
    fn connect_errors_are_typed() {
        // port 1 on loopback: refused
        let c = Client::new("http://127.0.0.1:1", None, 2).unwrap();
        assert!(matches!(c.get("/health"), Err(HttpError::Connect(_))));
    }
}
