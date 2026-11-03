//! talus-bench server extensions E3 (status `build` fingerprint), E4 (usage/timing detail, RSS) and E5 (energy)
//! against the mock engine: the new keys are there, and every pre-existing key keeps its place and value.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use talus_server::config::ServerConfig;
use talus_server::engine::ScriptedMockEngine;
use talus_server::json::JsonValue;
use talus_server::server::{ServerHandle, ServiceState, TalusServer};
use talus_server::tokenizer::ByteTokenizer;

fn http(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let b = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
        b.len()
    );
    s.write_all(req.as_bytes()).unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    let (head, payload) = resp.split_once("\r\n\r\n").unwrap();
    let code = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (code, payload.to_string())
}

fn json(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, JsonValue) {
    let (c, p) = http(port, method, path, body);
    (c, JsonValue::parse(&p).unwrap_or(JsonValue::Null))
}

fn keys(v: &JsonValue) -> Vec<String> {
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, _)| k.clone())
        .collect()
}

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("talus-benchext-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn start(cfg: ServerConfig, scripts: Vec<String>) -> ServerHandle {
    let engine = ScriptedMockEngine::with_scripts(scripts);
    let state = Arc::new(ServiceState::new(
        cfg,
        Arc::new(engine),
        ByteTokenizer::new(),
    ));
    TalusServer::bind(state).unwrap().start_background()
}

fn skip() -> bool {
    // the server would demand a key from this test client
    std::env::var("STRATA_API_KEY").is_ok() || std::env::var("TALUS_API_KEY").is_ok()
}

#[test]
fn status_build_engine_energy() {
    if skip() {
        return;
    }
    let d = scratch("status");
    let ctl = d.join("ctl");
    std::fs::create_dir_all(&ctl).unwrap();
    std::fs::write(d.join("model.gguf"), b"abc").unwrap();
    let cfg = ServerConfig {
        model_name: "qwen3.8-27b".to_string(),
        port: 0,
        model_path: d.join("model.gguf").to_string_lossy().into_owned(),
        engine_args: vec!["--ctx".into(), "4096".into()],
        engine_env: vec![
            ("TALUS_X".into(), "1".into()),
            ("TALUS_API_KEY".into(), "do-not-print-me".into()),
        ],
        control_dir: Some(ctl.clone()),
        ..Default::default()
    };
    let srv = start(cfg, vec!["Hello.".to_string()]);
    let port = srv.port();
    let (code, st) = json(port, "GET", "/v1/talus/status", None);
    assert_eq!(code, 200);
    // the existing keys first, in their old order; the new ones after them
    assert_eq!(
        keys(&st),
        [
            "service", "model", "preset", "state", "loaded", "running", "queued", "uptime_s",
            "context", "memory", "totals", "recent", "server", "control", "engine", "build",
            "energy"
        ]
    );
    assert_eq!(
        keys(st.get("server").unwrap()),
        ["host", "port", "api_key", "api_key_from_env", "rss_bytes"]
    );
    assert!(st.get("server").unwrap().get_i64("rss_bytes").unwrap() > 0);
    // mock engine: no child process
    assert!(st.get("engine").unwrap().get("pid").unwrap().is_null());
    let b = st.get("build").unwrap();
    assert_eq!(b.get_bool("mock"), Some(true));
    assert!(b.get("engine_exe").unwrap().is_null());
    assert_eq!(b.get_str("run_config_source"), Some("command_line"));
    assert_eq!(b.get_str("run_config_sha256").unwrap().len(), 64);
    let env = b.get("numerics_env").unwrap();
    assert_eq!(env.get_str("TALUS_X"), Some("1"));
    assert!(env.get("TALUS_API_KEY").is_none());
    let text = st.to_json_string();
    assert!(!text.contains("do-not-print-me"));
    // `energy` is null without an amdgpu hwmon power file, else an object with the counters
    let e = st.get("energy").unwrap();
    assert!(e.is_null() || e.get_f64("joules_total").is_some());

    // the weights (3 bytes = a "small" file) get hashed after the first status request
    let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    let mut got = None;
    for _ in 0..200 {
        let (_, st) = json(port, "GET", "/v1/talus/status?hash=weights", None);
        let w = st
            .get("build")
            .unwrap()
            .get("weights")
            .unwrap()
            .as_array()
            .unwrap()[0]
            .clone();
        if let Some(h) = w.get_str("sha256") {
            got = Some(h.to_string());
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(got.as_deref(), Some(abc));
    // the hash is published before the hashing thread writes the cache file: wait for the file
    let mut cache = String::new();
    for _ in 0..200 {
        cache = std::fs::read_to_string(ctl.join("fingerprints.json")).unwrap_or_default();
        if cache.contains(abc) {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(cache.contains(abc), "fingerprints.json: {cache}");
    drop(srv);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn chat_usage_and_timing_split() {
    if skip() {
        return;
    }
    let cfg = ServerConfig {
        model_name: "qwen3.8-27b".to_string(),
        port: 0,
        ..Default::default()
    };
    let srv = start(
        cfg,
        vec!["Analyzing request.</think>\n\nHello from Talus!".to_string()],
    );
    let port = srv.port();
    let body = r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello!"}]}"#;
    let (code, r) = json(port, "POST", "/v1/chat/completions", Some(body));
    assert_eq!(code, 200);
    assert_eq!(
        keys(&r),
        ["id", "object", "created", "model", "choices", "usage", "timings"]
    );
    let msg = r.get("choices").unwrap().as_array().unwrap()[0]
        .get("message")
        .unwrap()
        .clone();
    assert_eq!(msg.get_str("content"), Some("Hello from Talus!"));
    let u = r.get("usage").unwrap();
    assert_eq!(
        keys(u),
        [
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details"
        ]
    );
    let ct = u.get_usize("completion_tokens").unwrap();
    let rt = u
        .get("completion_tokens_details")
        .unwrap()
        .get_usize("reasoning_tokens")
        .unwrap();
    // byte tokenizer: one token per byte of "Analyzing request."
    assert_eq!(rt, "Analyzing request.".len());
    let t = r.get("timings").unwrap();
    assert_eq!(
        keys(t)[..9],
        [
            "cache_n",
            "prompt_n",
            "prompt_ms",
            "prompt_per_token_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_token_ms",
            "predicted_per_second"
        ]
    );
    let tail: Vec<String> = keys(t)[keys(t).len() - 4..].to_vec();
    assert_eq!(
        tail,
        ["reasoning_n", "answer_n", "reasoning_ms", "answer_ms"]
    );
    assert_eq!(t.get_usize("reasoning_n"), Some(rt));
    assert_eq!(
        t.get_usize("reasoning_n").unwrap() + t.get_usize("answer_n").unwrap(),
        ct
    );
    let sum = t.get_f64("reasoning_ms").unwrap() + t.get_f64("answer_ms").unwrap();
    assert!((sum - t.get_f64("predicted_ms").unwrap()).abs() <= 0.11);

    // streaming: the final chunk carries the same detail
    let body =
        r#"{"model":"qwen3.8-27b","stream":true,"messages":[{"role":"user","content":"Hi"}]}"#;
    let (code, sse) = http(port, "POST", "/v1/chat/completions", Some(body));
    assert_eq!(code, 200);
    let last = sse
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .filter_map(|d| JsonValue::parse(d).ok())
        .find(|c| c.get("usage").is_some())
        .unwrap();
    assert_eq!(
        last.get("usage")
            .unwrap()
            .get("completion_tokens_details")
            .unwrap()
            .get_usize("reasoning_tokens"),
        Some(rt)
    );
    assert!(last.get("timings").unwrap().get("answer_ms").is_some());
    drop(srv);
}
