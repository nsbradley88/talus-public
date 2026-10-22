//! A fake talus-server for developing and testing the TUI without a GPU: it serves the same `/v1/talus/*` API with
//! synthetic statistics (made-up numbers, roughly the right size per model) and simulates the restart on apply.
//! Run it with `talus-tui fake-daemon [--port N]`.

use serde_json::{json, Map, Value};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use talus_presets::{Bind, Level, Selection, MIN_API_KEY_LEN};

pub struct FakeState {
    pub model: &'static str,
    pub selection: Selection,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub control: bool,
    /// Model ids without a run config (apply answers 404).
    pub missing: Vec<&'static str>,
    started: Instant,
    restart_until: Option<Instant>,
    pending: Option<(&'static str, Selection, Bind)>,
    history: VecDeque<Value>,
    totals: [u64; 5],
    busy_until: Option<Instant>,
    rng: u64,
    pub restart_secs: f64,
}

impl FakeState {
    pub fn new(model: &'static str) -> Self {
        Self {
            model,
            selection: Selection::new(),
            host: "127.0.0.1".into(),
            port: 0,
            api_key: None,
            control: true,
            missing: Vec::new(),
            started: Instant::now(),
            restart_until: None,
            pending: None,
            history: VecDeque::new(),
            totals: [0; 5],
            busy_until: None,
            rng: 0x9e37_79b9_7f4a_7c15,
            restart_secs: 3.0,
        }
    }

    fn rand(&mut self) -> f64 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        (self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn restarting(&mut self) -> bool {
        match self.restart_until {
            Some(t) if Instant::now() < t => true,
            Some(_) => {
                self.restart_until = None;
                if let Some((m, sel, bind)) = self.pending.take() {
                    self.model = m;
                    self.selection = sel;
                    self.host = bind.host().into();
                }
                self.started = Instant::now();
                self.history.clear();
                self.totals = [0; 5];
                false
            }
            None => false,
        }
    }

    /// Typical speeds per model (illustrative only): answer tok/s, prompt tok/s, draft acceptance.
    fn base(&self) -> (f64, f64, f64) {
        match self.model {
            "qwen3.8-27b" => (45.0, 400.0, 0.70),
            "qwen3.8-flash-next" => (65.0, 1100.0, 0.72),
            "glm-5.3-flash" => (31.0, 360.0, 0.80),
            "mimo-v2.6-flash" => (33.0, 950.0, 0.58),
            _ => (40.0, 800.0, 0.55),
        }
    }

    /// Add one synthetic request.
    pub fn tick(&mut self) {
        if self.restarting() {
            return;
        }
        if self.rand() < 0.35 {
            return;
        }
        let spec = talus_presets::model(self.model).expect("fake model in catalog");
        let (mut dec, mut pp, acc) = self.base();
        if spec.value(&self.selection, "spec") == Some("off") {
            dec *= 0.45;
        }
        match spec.level_of(&self.selection) {
            Some(Level::Reference) => {
                dec *= 0.93;
                pp *= 0.88;
            }
            Some(Level::Fast) => {
                pp *= 1.08;
                dec *= 1.05;
            }
            _ => {}
        }
        let ctx: u64 = spec
            .value(&self.selection, "context")
            .and_then(|v| v.parse().ok())
            .unwrap_or(65536);
        let prompt = (300.0 + self.rand() * (ctx as f64 * 0.4).min(24000.0)) as u64;
        let reused = if self.rand() < 0.6 {
            (prompt as f64 * (0.5 + 0.45 * self.rand())) as u64
        } else {
            0
        };
        let out = (40.0 + self.rand() * 700.0) as u64;
        let noise = 0.9 + 0.2 * self.rand();
        let prompt_ms = (prompt - reused).max(1) as f64 * 1000.0 / (pp * noise);
        let decode_ms = out as f64 * 1000.0 / (dec * noise);
        let drafts = if spec.value(&self.selection, "spec") == Some("off") {
            0
        } else {
            (out as f64 * 0.9) as u64
        };
        let accepted = (drafts as f64 * (acc * (0.9 + 0.2 * self.rand())).min(0.98)) as u64;
        let finish = if self.rand() < 0.85 { "stop" } else { "length" };
        self.history.push_back(json!({
            "prompt_tokens": prompt, "reused": reused, "output_tokens": out,
            "prompt_ms": (prompt_ms * 10.0).round() / 10.0, "decode_ms": (decode_ms * 10.0).round() / 10.0,
            "finish": finish,
            "ttft_ms": ((prompt_ms + 18.0) * 10.0).round() / 10.0,
            "drafts": drafts, "accepted": accepted
        }));
        if self.history.len() > 100 {
            self.history.pop_front();
        }
        self.totals[0] += 1;
        self.totals[1] += prompt;
        self.totals[2] += out;
        self.totals[3] += drafts;
        self.totals[4] += accepted;
        self.busy_until =
            Some(Instant::now() + Duration::from_millis((self.rand() * 1200.0) as u64));
    }

    pub fn status(&mut self) -> Value {
        let restarting = self.restarting();
        let busy = self.busy_until.map(|t| Instant::now() < t).unwrap_or(false);
        let spec = talus_presets::model(self.model).expect("fake model in catalog");
        let ctx: u64 = spec
            .value(&self.selection, "context")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let last = self.history.back().cloned().unwrap_or(json!({}));
        let in_use = last["prompt_tokens"].as_u64().unwrap_or(0)
            + last["output_tokens"].as_u64().unwrap_or(0);
        let total = 125.0;
        let weights = match self.model {
            "qwen3.8-27b" => 43.0,
            "qwen3.8-flash-next" => 74.0,
            "glm-5.3-flash" => 115.0,
            "mimo-v2.6-flash" => 107.0,
            _ => 83.0,
        };
        let avail = ((total - weights) - self.rand() * 2.0) * 10.0;
        let recent: Vec<Value> = self.history.iter().rev().take(20).cloned().collect();
        let level = spec
            .level_of(&self.selection)
            .map(|l| l.id())
            .unwrap_or("custom");
        let choices: Map<String, Value> = self
            .selection
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        json!({
            "service": "talus", "model": self.model, "preset": self.model,
            "state": if restarting { "restarting" } else if busy { "generating" } else { "idle" },
            "loaded": !restarting, "running": u64::from(busy && !restarting), "queued": 0,
            "uptime_s": self.started.elapsed().as_secs(),
            "context": {"max": ctx, "in_use": in_use},
            "memory": {"available_gib": avail.round() / 10.0, "total_gib": total},
            "totals": {"requests": self.totals[0], "prompt_tokens": self.totals[1], "reused": 0,
                       "output_tokens": self.totals[2], "prompt_ms": 0.0, "decode_ms": 0.0,
                       "drafts_offered": self.totals[3], "drafts_accepted": self.totals[4]},
            "recent": recent,
            "server": {"host": self.host, "port": self.port, "api_key": self.api_key.is_some(), "api_key_from_env": false},
            "control": {"enabled": self.control, "selection": {"model": self.model, "level": level, "choices": choices,
                        "bind": Bind::from_host(&self.host).id()}}
        })
    }

    pub fn presets(&self) -> Value {
        let models: Vec<Value> = talus_presets::MODELS
            .iter()
            .map(|m| json!({"id": m.id, "name": m.name, "available": self.control && !self.missing.contains(&m.id)}))
            .collect();
        json!({"enabled": self.control, "models": models, "active": self.model})
    }

    fn apply(&mut self, body: &Value) -> (u16, Value) {
        let err = |code: u16, m: &str| {
            (
                code,
                json!({"error": {"type": "invalid_request_error", "message": m}}),
            )
        };
        if !self.control {
            return err(
                403,
                "control is off: start talus-server with --control-dir DIR (see docs/TUI.md)",
            );
        }
        if self.restarting() {
            return err(409, "a restart is already under way");
        }
        let Some(spec) = body["model"].as_str().and_then(talus_presets::model) else {
            return err(400, "unknown model");
        };
        if self.missing.contains(&spec.id) {
            return err(404, &format!("no run config for {}", spec.id));
        }
        let sel: Selection = body["choices"]
            .as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let edits = match spec.edits(&sel) {
            Ok(e) => e,
            Err(e) => return err(400, &e),
        };
        let bind = match body.get("bind").and_then(Value::as_str) {
            None => Bind::from_host(&self.host),
            Some(b) => match Bind::parse(b) {
                Some(b) => b,
                None => return err(400, "bind must be \"local\" or \"lan\""),
            },
        };
        let key = body.get("api_key").and_then(Value::as_str);
        if let Some(k) = key {
            if k.len() < MIN_API_KEY_LEN {
                return err(400, "an API key needs at least 16 characters");
            }
        }
        if bind == Bind::Lan && key.is_none() && self.api_key.is_none() {
            return err(400, "local network access needs an API key");
        }
        let mut changes: Vec<String> = Vec::new();
        if spec.id != self.model {
            changes.push(format!("model: {} -> {}", self.model, spec.name));
        }
        changes.extend(edits.iter().map(talus_presets::describe_edit));
        if body["dry_run"].as_bool() == Some(true) {
            return (
                200,
                json!({"status": "dry_run", "model": spec.id, "changes": changes}),
            );
        }
        if let Some(k) = key {
            self.api_key = Some(k.to_string());
        }
        let secs = if spec.id != self.model {
            self.restart_secs * 2.0
        } else {
            self.restart_secs
        };
        self.restart_until = Some(Instant::now() + Duration::from_secs_f64(secs));
        self.pending = Some((spec.id, spec.normalize(&sel), bind));
        (
            202,
            json!({"status": "restarting", "model": spec.id, "changes": changes}),
        )
    }
}

fn handle(mut stream: TcpStream, state: Arc<Mutex<FakeState>>) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut len = 0usize;
    let mut auth = None;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
        if lower.starts_with("authorization:") {
            auth = h
                .split_once(':')
                .map(|(_, v)| v.trim().trim_start_matches("Bearer ").to_string());
        }
    }
    let mut body = vec![0u8; len.min(1 << 20)];
    reader.read_exact(&mut body)?;
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);

    let (code, reply) = {
        let mut st = state.lock().unwrap();
        let authorized = st.api_key.is_none() || auth.as_deref() == st.api_key.as_deref();
        match (method.as_str(), path.as_str()) {
            ("GET", "/health") => (200, json!({"status": "ok", "model": st.model})),
            _ if !authorized => (
                401,
                json!({"error": {"message": "missing or wrong API key"}}),
            ),
            ("GET", "/v1/talus/status") => (200, st.status()),
            ("GET", "/v1/talus/presets") => (200, st.presets()),
            ("POST", "/v1/talus/apply") => st.apply(&body),
            _ => (404, json!({"error": {"message": "not found"}})),
        }
    };
    let text = reply.to_string();
    let head = format!(
        "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        text.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(text.as_bytes())?;
    stream.flush()
}

/// Start the fake daemon on 127.0.0.1:`port` (0 = any) in background threads.
pub fn spawn(port: u16, state: FakeState) -> std::io::Result<(SocketAddr, Arc<Mutex<FakeState>>)> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let addr = listener.local_addr()?;
    let state = Arc::new(Mutex::new(state));
    state.lock().unwrap().port = addr.port();
    let ticker = Arc::clone(&state);
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_millis(1500));
        ticker.lock().unwrap().tick();
    });
    let st = Arc::clone(&state);
    std::thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            let st = Arc::clone(&st);
            std::thread::spawn(move || {
                let _ = handle(conn, st);
            });
        }
    });
    Ok((addr, state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::Client;

    #[test]
    fn fake_daemon_round_trip() {
        let mut st = FakeState::new("qwen3.8-27b");
        st.restart_secs = 0.2;
        st.missing = vec!["glm-5.3-flash"];
        for _ in 0..20 {
            st.tick();
        }
        let (addr, state) = spawn(0, st).unwrap();
        let c = Client::new(&format!("http://{addr}"), None).unwrap();
        let (code, v) = c.get("/v1/talus/status").unwrap();
        assert_eq!(code, 200);
        let snap = crate::stats::parse_status(&v);
        assert_eq!(snap.preset.as_deref(), Some("qwen3.8-27b"));
        assert!(!snap.recent.is_empty());
        assert!(snap.recent[0].decode_tps().unwrap() > 10.0);

        let (code, _) = c
            .post("/v1/talus/apply", &json!({"model": "glm-5.3-flash"}))
            .unwrap();
        assert_eq!(code, 404);
        let (code, _) = c
            .post(
                "/v1/talus/apply",
                &json!({"model": "laguna-s-2.1", "choices": {"context": "7"}}),
            )
            .unwrap();
        assert_eq!(code, 400);
        let (code, v) = c
            .post("/v1/talus/apply", &json!({"model": "laguna-s-2.1", "choices": {"verify_gemm": "on"}, "dry_run": true}))
            .unwrap();
        assert_eq!(code, 200);
        assert!(v["changes"]
            .to_string()
            .contains("TALUS_LAGUNA_VERIFY_GEMM=9"));
        let (code, _) = c
            .post(
                "/v1/talus/apply",
                &json!({"model": "laguna-s-2.1", "choices": {"verify_gemm": "on"}, "bind": "lan", "api_key": "0123456789abcdefXYZ"}),
            )
            .unwrap();
        assert_eq!(code, 202);
        let (_, v) = c.get("/v1/talus/status").unwrap();
        assert_eq!(v["error"]["message"], "missing or wrong API key");
        let c = Client::new(
            &format!("http://{addr}"),
            Some("0123456789abcdefXYZ".into()),
        )
        .unwrap();
        let (_, v) = c.get("/v1/talus/status").unwrap();
        assert_eq!(v["state"], "restarting");
        std::thread::sleep(Duration::from_millis(500));
        let (_, v) = c.get("/v1/talus/status").unwrap();
        assert_eq!(v["preset"], "laguna-s-2.1");
        assert_eq!(v["control"]["selection"]["level"], "fast");
        assert_eq!(v["server"]["host"], "0.0.0.0");
        drop(state);
    }
}
