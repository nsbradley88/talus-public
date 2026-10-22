//! A small blocking HTTP/1.1 client for talus-server's JSON endpoints (one request per connection).

use serde_json::Value;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Client {
    host: String,
    port: u16,
    api_key: Option<String>,
    timeout: Duration,
}

impl Client {
    /// `url` like `http://127.0.0.1:8000` (the scheme and a trailing slash are optional; only plain HTTP).
    pub fn new(url: &str, api_key: Option<String>) -> Result<Self, String> {
        let rest = url.trim().trim_end_matches('/');
        if rest.starts_with("https://") {
            return Err("https is not supported; talus-server speaks plain HTTP".to_string());
        }
        let rest = rest.strip_prefix("http://").unwrap_or(rest);
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
            timeout: Duration::from_secs(3),
        })
    }

    pub fn url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    pub fn has_key(&self) -> bool {
        self.api_key.is_some()
    }

    pub fn set_api_key(&mut self, key: Option<String>) {
        self.api_key = key.filter(|k| !k.is_empty());
    }

    pub fn get(&self, path: &str) -> Result<(u16, Value), String> {
        self.request("GET", path, None)
    }

    pub fn post(&self, path: &str, body: &Value) -> Result<(u16, Value), String> {
        self.request("POST", path, Some(body.to_string()))
    }

    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<String>,
    ) -> Result<(u16, Value), String> {
        let addr = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(|e| format!("{}: {e}", self.url()))?
            .next()
            .ok_or_else(|| format!("{}: no address", self.url()))?;
        let mut s = TcpStream::connect_timeout(&addr, self.timeout)
            .map_err(|e| format!("cannot reach {}: {e}", self.url()))?;
        s.set_read_timeout(Some(self.timeout)).ok();
        s.set_write_timeout(Some(self.timeout)).ok();
        let body = body.unwrap_or_default();
        let mut req = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}:{}\r\nAccept: application/json\r\nConnection: close\r\n",
            self.host, self.port
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
            .map_err(|e| format!("{}: {e}", self.url()))?;
        let mut raw = Vec::new();
        s.read_to_end(&mut raw)
            .map_err(|e| format!("{}: {e}", self.url()))?;
        parse_response(&raw)
    }
}

/// Status code and JSON body of a raw HTTP/1.1 response (Content-Length or read-to-close; chunked is decoded).
pub fn parse_response(raw: &[u8]) -> Result<(u16, Value), String> {
    let text = String::from_utf8_lossy(raw);
    let (head, payload) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_string())?;
    let code: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .ok_or_else(|| "malformed HTTP status line".to_string())?;
    let chunked = head.lines().any(|l| {
        l.to_ascii_lowercase().starts_with("transfer-encoding:")
            && l.to_ascii_lowercase().contains("chunked")
    });
    let body = if chunked {
        dechunk(payload)
    } else {
        payload.to_string()
    };
    let value = if body.trim().is_empty() {
        Value::Null
    } else {
        serde_json::from_str(&body).map_err(|e| format!("bad JSON from server ({code}): {e}"))?
    };
    Ok((code, value))
}

fn dechunk(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some((size_line, after)) = rest.split_once("\r\n") {
        let size = usize::from_str_radix(size_line.trim(), 16).unwrap_or(0);
        if size == 0 || after.len() < size {
            break;
        }
        out.push_str(&after[..size]);
        rest = after[size..].trim_start_matches("\r\n");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_parsing() {
        let c = Client::new("http://127.0.0.1:8000/", None).unwrap();
        assert_eq!(c.url(), "http://127.0.0.1:8000");
        let c = Client::new("box:9000", Some(String::new())).unwrap();
        assert_eq!(c.url(), "http://box:9000");
        assert!(!c.has_key());
        assert_eq!(
            Client::new("localhost", None).unwrap().url(),
            "http://localhost:8000"
        );
        assert!(Client::new("https://x:1", None).is_err());
        assert!(Client::new("http://x:notaport", None).is_err());
    }

    #[test]
    fn responses() {
        let (c, v) =
            parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\n{\"a\":1}").unwrap();
        assert_eq!(c, 200);
        assert_eq!(v["a"], 1);
        let (c, v) = parse_response(
            b"HTTP/1.1 400 Bad\r\nTransfer-Encoding: chunked\r\n\r\n5\r\n{\"err\r\n14\r\nor\":{\"message\":\"x\"}}\r\n0\r\n\r\n",
        )
        .unwrap();
        assert_eq!(c, 400);
        assert_eq!(error_message(&v), "x");
        assert!(parse_response(b"garbage").is_err());
    }
}
