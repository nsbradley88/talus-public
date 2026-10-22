//! `/v1/talus/*` (talus-tui status and control) against a mock engine. The restart is replaced by a hook, so no
//! process is re-executed.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use talus_server::config::ServerConfig;
use talus_server::engine::ScriptedMockEngine;
use talus_server::json::JsonValue;
use talus_server::server::{ServerHandle, ServiceState, TalusServer};
use talus_server::tokenizer::ByteTokenizer;

fn http(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, JsonValue) {
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
    (code, JsonValue::parse(payload).unwrap_or(JsonValue::Null))
}

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("talus-control-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_preset(dir: &Path, id: &str, args: &[&str]) {
    let args: Vec<String> = args.iter().map(|a| format!("{a:?}")).collect();
    std::fs::write(
        dir.join(format!("{id}.json")),
        format!(
            "{{\"mock\": true, \"model_name\": \"{id}\", \"host\": \"127.0.0.1\", \"args\": [{}], \"env\": {{\"KEEP\": \"1\"}}}}",
            args.join(", ")
        ),
    )
    .unwrap();
}

fn start(control_dir: Option<PathBuf>) -> (ServerHandle, mpsc::Receiver<PathBuf>) {
    let tok = ByteTokenizer::new();
    let engine = ScriptedMockEngine::with_scripts(vec!["Hello there, this is a mock.".to_string()]);
    engine.set_fixed_drafts(Some((12, 9)));
    let cfg = ServerConfig {
        model_name: "qwen3.8-27b".to_string(),
        port: 0,
        control_dir,
        ..Default::default()
    };
    let state = Arc::new(ServiceState::new(cfg, Arc::new(engine), tok));
    let (tx, rx) = mpsc::channel();
    let tx = std::sync::Mutex::new(tx);
    state.control.set_restart_hook(move |p| {
        let _ = tx.lock().unwrap().send(p.to_path_buf());
    });
    let srv = TalusServer::bind(state).unwrap().start_background();
    (srv, rx)
}

#[test]
fn status_reports_recent_requests() {
    if std::env::var("STRATA_API_KEY").is_ok() {
        return; // the server would demand a key from this test client
    }
    let (srv, _rx) = start(None);
    let port = srv.port();
    let (code, st) = http(port, "GET", "/v1/talus/status", None);
    assert_eq!(code, 200);
    assert_eq!(st.get_str("service"), Some("talus"));
    assert_eq!(st.get_str("preset"), Some("qwen3.8-27b"));
    assert_eq!(st.get_str("state"), Some("idle"));
    assert_eq!(st.get("control").unwrap().get_bool("enabled"), Some(false));
    assert!(st.get("context").unwrap().get_u64("max").unwrap() > 0);

    let (code, _) = http(
        port,
        "POST",
        "/completion",
        Some(r#"{"prompt": "Hello world, say something", "n_predict": 8}"#),
    );
    assert_eq!(code, 200);
    let (_, st) = http(port, "GET", "/v1/talus/status", None);
    let recent = st.get("recent").unwrap().as_array().unwrap();
    assert_eq!(recent.len(), 1);
    let r = &recent[0];
    assert!(r.get_u64("output_tokens").unwrap() > 0);
    assert!(r.get("ttft_ms").is_some());
    assert_eq!(r.get_u64("drafts"), Some(12));
    assert_eq!(r.get_u64("accepted"), Some(9));
    assert_eq!(
        st.get("totals").unwrap().get_u64("drafts_accepted"),
        Some(9)
    );
    assert!(st.get("context").unwrap().get_u64("in_use").unwrap() > 0);

    // control is off without --control-dir
    let (code, body) = http(
        port,
        "POST",
        "/v1/talus/apply",
        Some(r#"{"model": "qwen3.8-27b"}"#),
    );
    assert_eq!(code, 403, "{body:?}");
}

#[test]
fn apply_builds_config_and_restarts() {
    if std::env::var("STRATA_API_KEY").is_ok() {
        return;
    }
    let dir = scratch("apply");
    write_preset(
        &dir,
        "mimo-v2.6-flash",
        &[
            "--model",
            "m.gguf",
            "--max-context",
            "196608",
            "--spec",
            "4",
            "--serve",
        ],
    );
    write_preset(
        &dir,
        "qwen3.8-27b",
        &["--model", "q.gguf", "--max-context", "65536"],
    );
    let (srv, rx) = start(Some(dir.clone()));
    let port = srv.port();

    let (code, p) = http(port, "GET", "/v1/talus/presets", None);
    assert_eq!(code, 200);
    assert_eq!(p.get_bool("enabled"), Some(true));
    let models = p.get("models").unwrap().as_array().unwrap();
    let avail = |id: &str| {
        models
            .iter()
            .find(|m| m.get_str("id") == Some(id))
            .and_then(|m| m.get_bool("available"))
    };
    assert_eq!(avail("mimo-v2.6-flash"), Some(true));
    assert_eq!(avail("glm-5.3-flash"), Some(false));

    // validation
    let (code, _) = http(
        port,
        "POST",
        "/v1/talus/apply",
        Some(r#"{"model": "nope"}"#),
    );
    assert_eq!(code, 400);
    let (code, _) = http(
        port,
        "POST",
        "/v1/talus/apply",
        Some(r#"{"model": "mimo-v2.6-flash", "choices": {"context": "12345"}}"#),
    );
    assert_eq!(code, 400);
    let (code, _) = http(
        port,
        "POST",
        "/v1/talus/apply",
        Some(r#"{"model": "mimo-v2.6-flash", "choices": {"args": "--evil"}}"#),
    );
    assert_eq!(code, 400);
    let (code, _) = http(
        port,
        "POST",
        "/v1/talus/apply",
        Some(r#"{"model": "glm-5.3-flash"}"#),
    );
    assert_eq!(code, 404);
    let (code, _) = http(
        port,
        "POST",
        "/v1/talus/apply",
        Some(r#"{"model": "mimo-v2.6-flash", "bind": "lan"}"#),
    );
    assert_eq!(code, 400, "LAN without an API key");
    let (code, _) = http(
        port,
        "POST",
        "/v1/talus/apply",
        Some(r#"{"model": "mimo-v2.6-flash", "bind": "lan", "api_key": "short"}"#),
    );
    assert_eq!(code, 400);

    // dry run names the changes and writes nothing
    let body = r#"{"model": "mimo-v2.6-flash", "choices": {"context": "65536", "iq1m": "on", "spec": "off"}, "dry_run": true}"#;
    let (code, r) = http(port, "POST", "/v1/talus/apply", Some(body));
    assert_eq!(code, 200, "{r:?}");
    let changes: Vec<&str> = r
        .get("changes")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c.as_str())
        .collect();
    assert!(
        changes.iter().any(|c| c.starts_with("model:")),
        "{changes:?}"
    );
    assert!(changes.contains(&"--max-context 65536"), "{changes:?}");
    assert!(changes.contains(&"MIMO_IQ1M_TPMOE=1"), "{changes:?}");
    assert!(changes.contains(&"--no-mtp"), "{changes:?}");
    assert!(!dir.join("active.json").exists());

    // the real thing
    let body = r#"{"model": "mimo-v2.6-flash", "choices": {"context": "65536", "iq1m": "on"}, "bind": "lan", "api_key": "0123456789abcdef-test"}"#;
    let (code, r) = http(port, "POST", "/v1/talus/apply", Some(body));
    assert_eq!(code, 202, "{r:?}");
    assert_eq!(r.get_str("status"), Some("restarting"));
    let path = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("restart hook");
    assert_eq!(path, dir.join("active.json"));

    let cfg = JsonValue::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let args: Vec<&str> = cfg
        .get("args")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|a| a.as_str())
        .collect();
    assert_eq!(
        args,
        [
            "--model",
            "m.gguf",
            "--max-context",
            "65536",
            "--spec",
            "4",
            "--serve"
        ]
    );
    let env = cfg.get("env").unwrap();
    assert_eq!(env.get_str("KEEP"), Some("1"));
    assert_eq!(env.get_str("MIMO_IQ1M_TPMOE"), Some("1"));
    assert_eq!(cfg.get_str("host"), Some("0.0.0.0"));
    assert_eq!(cfg.get_str("api_key"), Some("0123456789abcdef-test"));
    assert_eq!(cfg.get_bool("mock"), Some(true));
    let sel = cfg.get("_talus_tui").unwrap();
    assert_eq!(sel.get_str("model"), Some("mimo-v2.6-flash"));
    assert_eq!(sel.get_str("level"), Some("fast"));
    assert_eq!(
        sel.get("choices").unwrap().get_str("context"),
        Some("65536")
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(
            mode & 0o077,
            0,
            "active.json must be private (it holds the key)"
        );
    }
    // the reply never echoes the key
    assert!(!r.to_json_string().contains("0123456789abcdef"));

    let (_, st) = http(port, "GET", "/v1/talus/status", None);
    assert_eq!(st.get_str("state"), Some("restarting"));
    let (code, _) = http(
        port,
        "POST",
        "/v1/talus/apply",
        Some(r#"{"model": "qwen3.8-27b"}"#),
    );
    assert_eq!(code, 409);

    // the generated file is a valid run config for the server itself
    let parsed = ServerConfig::from_json_file(&path).unwrap();
    assert_eq!(parsed.max_context, 65536);
    assert_eq!(parsed.host, "0.0.0.0");
    assert_eq!(parsed.control_dir.as_deref(), Some(dir.as_path()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn control_dir_loads_active_config() {
    let dir = scratch("cli");
    std::fs::write(
        dir.join("active.json"),
        r#"{"mock": true, "model_name": "laguna-s-2.1", "args": ["--max-context", "32768"]}"#,
    )
    .unwrap();
    let d = dir.to_string_lossy().to_string();
    let cfg = ServerConfig::parse_cli_args(["--control-dir", d.as_str(), "--mock"]).unwrap();
    assert_eq!(cfg.model_name, "laguna-s-2.1");
    assert_eq!(cfg.max_context, 32768);
    assert_eq!(cfg.control_dir.as_deref(), Some(dir.as_path()));
    // an explicit --config wins over active.json
    let other = dir.join("other.json");
    std::fs::write(&other, r#"{"mock": true, "model_name": "glm-5.3-flash"}"#).unwrap();
    let o = other.to_string_lossy().to_string();
    let cfg = ServerConfig::parse_cli_args(["--control-dir", d.as_str(), "--config", o.as_str()])
        .unwrap();
    assert_eq!(cfg.model_name, "glm-5.3-flash");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn failed_apply_falls_back_to_last_good() {
    let dir = scratch("recover");
    let good = r#"{"mock": true, "model_name": "qwen3.8-27b", "args": ["--max-context", "65536"]}"#;
    std::fs::write(dir.join("last-good.json"), good).unwrap();
    std::fs::write(
        dir.join("active.json"),
        r#"{"exe": "/nonexistent/talus-cli", "model_name": "mimo-v2.6-flash", "_talus_tui": {"model": "mimo-v2.6-flash"}}"#,
    )
    .unwrap();
    let mut cfg = ServerConfig::from_json_file(&dir.join("active.json")).unwrap();
    cfg.control_dir = Some(dir.clone());
    let prev = talus_server::control::recover_after_failed_start(&cfg, "failed to start engine")
        .expect("falls back");
    assert_eq!(prev.model_name, "qwen3.8-27b");
    assert_eq!(
        std::fs::read_to_string(dir.join("active.json")).unwrap(),
        good
    );
    assert!(dir.join("active.failed.json").exists());
    let err =
        JsonValue::parse(&std::fs::read_to_string(dir.join("last-error.json")).unwrap()).unwrap();
    assert_eq!(err.get_str("error"), Some("failed to start engine"));
    // a config that is not the control directory's active.json is not touched
    let mut other = prev.clone();
    other.config_path = Some(dir.join("last-good.json"));
    assert!(talus_server::control::recover_after_failed_start(&other, "x").is_none());
    let _ = std::fs::remove_dir_all(&dir);
}
