//! Black-box HTTP parity test suite for `talus-server`.
//!
//! Verifies all 12 core endpoint groups in Rust against `talus-server`,
//! exercises `StdioSubprocessEngine` over the `READY`/`GEN`/`PP`/`RESUME`/`T`/`DONE`
//! line protocol, and invokes `tests/run_dual_server_parity.py` to verify
//! black-box HTTP parity against both `serve/server.py` and `talus-server`.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use talus_server::config::ServerConfig;
use talus_server::engine::{
    Engine, GenerateRequest, SamplingParams, ScriptedMockEngine, StdioSubprocessEngine,
};
use talus_server::json::JsonValue;
use talus_server::server::{ServiceState, TalusServer};
use talus_server::tokenizer::ByteTokenizer;

fn raw_http_request(
    port: u16,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&str>,
) -> (u16, String, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect to test server");
    let body_bytes = body.unwrap_or("").as_bytes();
    let has_host = headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("host"));
    let has_ct = headers
        .iter()
        .any(|(k, _)| k.eq_ignore_ascii_case("content-type"));

    let mut req = format!("{method} {path} HTTP/1.1\r\n");
    if !has_host {
        req.push_str(&format!("Host: 127.0.0.1:{port}\r\n"));
    }
    if body.is_some() && !has_ct {
        req.push_str("Content-Type: application/json\r\n");
    }
    req.push_str(&format!("Content-Length: {}\r\n", body_bytes.len()));
    req.push_str("Connection: close\r\n");
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");

    stream.write_all(req.as_bytes()).unwrap();
    if !body_bytes.is_empty() {
        stream.write_all(body_bytes).unwrap();
    }
    stream.flush().unwrap();

    let mut resp = Vec::new();
    stream.read_to_end(&mut resp).unwrap();
    let text = String::from_utf8_lossy(&resp).to_string();
    let (head, payload) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let status_line = head.lines().next().unwrap_or("");
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    (status, head.to_string(), payload.to_string())
}

fn parse_sse(raw: &str) -> Vec<(Option<String>, String)> {
    let mut events = Vec::new();
    for block in raw.split("\n\n") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        let mut ev_name = None;
        let mut data_lines = Vec::new();
        for line in block.lines() {
            if let Some(rest) = line.strip_prefix("event:") {
                ev_name = Some(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.trim().to_string());
            }
        }
        if !data_lines.is_empty() {
            events.push((ev_name, data_lines.join("\n")));
        }
    }
    events
}

#[test]
fn test_blackbox_all_12_endpoint_checks_in_process() {
    let tok = ByteTokenizer::new();
    let scripts = vec![
        "Completion alpha.".to_string(),
        "Completion beta.".to_string(),
        "Token array completion.".to_string(),
        "OpenAI text completion sync.".to_string(),
        "OpenAI text completion stream.".to_string(),
        "Analyzing request.</think>\n\nHello from Talus!".to_string(),
        "Step 1.</think>\n\nStreamed reply!".to_string(),
        "Call weather tool.</think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>".to_string(),
        "</think>\n<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>".to_string(),
        "Reasoning trace.</think>\n\nResponse API answer.".to_string(),
        "</think>Streamed response output.".to_string(),
        "Anthropic thought.</think>\n\nAnthropic message text.".to_string(),
        "Thinking for tool.</think><tool_call><function=get_weather><parameter=city>Berlin</parameter></function></tool_call>".to_string(),
    ];
    let engine = Arc::new(ScriptedMockEngine::with_scripts(scripts));

    let scratch_slots = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("talus-slots-{}", std::process::id()));
    let cfg = ServerConfig {
        model_name: "qwen3.8-27b".to_string(),
        model_aliases: vec!["qwen27".to_string()],
        slot_save_path: scratch_slots.to_string_lossy().to_string(),
        ..Default::default()
    };

    let state = Arc::new(ServiceState::new(cfg, engine, tok));
    let mut srv = TalusServer::bind(state)
        .expect("start server")
        .start_background();
    let port = srv.port();

    // 1. /health + DNS rebinding protection
    let (st, _, body) = raw_http_request(port, "GET", "/health", &[], None);
    assert_eq!(st, 200);
    let h = JsonValue::parse(&body).unwrap();
    assert_eq!(h.get_str("status"), Some("ok"));
    let (st_rebind, _, _) = raw_http_request(
        port,
        "GET",
        "/health",
        &[("Host", "evil.attacker.example.com")],
        None,
    );
    assert_eq!(st_rebind, 403);

    // 2. /v1/models
    let (st, _, body) = raw_http_request(port, "GET", "/v1/models", &[], None);
    assert_eq!(st, 200);
    let models = JsonValue::parse(&body).unwrap();
    let arr = models.get("data").and_then(|v| v.as_array()).unwrap();
    assert_eq!(arr.len(), 2);

    // 3. /props
    let (st, _, body) = raw_http_request(port, "GET", "/props", &[], None);
    assert_eq!(st, 200);
    let props = JsonValue::parse(&body).unwrap();
    assert!(props.get("default_generation_settings").is_some());
    assert!(props.get_str("chat_template").is_some());

    // 4. /slots
    let (st, _, body) = raw_http_request(port, "GET", "/slots", &[], None);
    assert_eq!(st, 200);
    let slots = JsonValue::parse(&body).unwrap();
    assert_eq!(slots.as_array().unwrap().len(), 1);
    let (st_bad_slot, _, _) = raw_http_request(
        port,
        "POST",
        "/slots/0?action=save",
        &[],
        Some(r#"{"filename":"../escape.bin"}"#),
    );
    assert_eq!(st_bad_slot, 400);

    // 5. /tokenize + /detokenize
    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/tokenize",
        &[],
        Some(r#"{"content":"Hello Talus!"}"#),
    );
    assert_eq!(st, 200);
    let tok_res = JsonValue::parse(&body).unwrap();
    let tokens_json = tok_res.get("tokens").unwrap().to_json_string();
    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/detokenize",
        &[],
        Some(&format!(r#"{{"tokens":{tokens_json}}}"#)),
    );
    assert_eq!(st, 200);
    let detok_res = JsonValue::parse(&body).unwrap();
    assert_eq!(detok_res.get_str("content"), Some("Hello Talus!"));

    // 6. /completion (ignore_eos, cache_prompt, timings, token array)
    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/completion",
        &[],
        Some(
            r#"{"prompt":"Shared prompt prefix","n_predict":17,"ignore_eos":true,"cache_prompt":true}"#,
        ),
    );
    assert_eq!(st, 200);
    let c1 = JsonValue::parse(&body).unwrap();
    assert_eq!(c1.get_str("content"), Some("Completion alpha."));
    assert!(c1.get("timings").is_some());

    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/completion",
        &[],
        Some(r#"{"prompt":"Shared prompt prefix extended","n_predict":32,"cache_prompt":true}"#),
    );
    assert_eq!(st, 200);
    let c2 = JsonValue::parse(&body).unwrap();
    assert!(
        c2.get("timings")
            .and_then(|t| t.get_i64("cache_n"))
            .unwrap_or(0)
            > 0
    );

    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/completion",
        &[],
        Some(r#"{"prompt":[72,105],"n_predict":32}"#),
    );
    assert_eq!(st, 200);
    let c3 = JsonValue::parse(&body).unwrap();
    assert_eq!(c3.get_str("content"), Some("Token array completion."));

    // 7. /v1/completions (sync + stream)
    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/v1/completions",
        &[],
        Some(r#"{"model":"qwen3.8-27b","prompt":"Hello","max_tokens":32}"#),
    );
    assert_eq!(st, 200);
    let vc1 = JsonValue::parse(&body).unwrap();
    assert_eq!(
        vc1.get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get_str("text")),
        Some("OpenAI text completion sync.")
    );

    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/v1/completions",
        &[],
        Some(r#"{"model":"qwen3.8-27b","prompt":"Hello","max_tokens":32,"stream":true}"#),
    );
    assert_eq!(st, 200);
    let vc_sse = parse_sse(&body);
    assert_eq!(vc_sse.last().map(|(_, d)| d.as_str()), Some("[DONE]"));

    // 8. /v1/chat/completions (sync + stream + thinking + Qwen/GLM tool_calls + cached_tokens)
    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/v1/chat/completions",
        &[],
        Some(r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello!"}]}"#),
    );
    assert_eq!(st, 200);
    let cc1 = JsonValue::parse(&body).unwrap();
    let msg1 = cc1
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("message"))
        .unwrap();
    assert_eq!(msg1.get_str("content"), Some("Hello from Talus!"));
    assert_eq!(
        msg1.get_str("reasoning_content"),
        Some("Analyzing request.")
    );

    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/v1/chat/completions",
        &[],
        Some(
            r#"{"model":"qwen27","stream":true,"messages":[{"role":"user","content":"Hello!"},{"role":"assistant","content":"Hi"},{"role":"user","content":"More"}]}"#,
        ),
    );
    assert_eq!(st, 200);
    let cc_sse = parse_sse(&body);
    assert_eq!(cc_sse.last().map(|(_, d)| d.as_str()), Some("[DONE]"));

    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/v1/chat/completions",
        &[],
        Some(
            r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Weather?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}"#,
        ),
    );
    assert_eq!(st, 200);
    let tc1 = JsonValue::parse(&body).unwrap();
    assert_eq!(
        tc1.get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get_str("finish_reason")),
        Some("tool_calls")
    );

    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/v1/chat/completions",
        &[],
        Some(
            r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Weather?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}"#,
        ),
    );
    assert_eq!(st, 200);
    let tc2 = JsonValue::parse(&body).unwrap();
    assert_eq!(
        tc2.get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get_str("finish_reason")),
        Some("tool_calls")
    );

    // 9. /v1/responses (sync + stream + GET/DELETE 404)
    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/v1/responses",
        &[],
        Some(
            r#"{"model":"qwen3.8-27b","input":"Explain gravity","include":["reasoning.encrypted_content"]}"#,
        ),
    );
    assert_eq!(st, 200);
    let r1 = JsonValue::parse(&body).unwrap();
    assert_eq!(r1.get_str("object"), Some("response"));
    assert_eq!(r1.get_str("status"), Some("completed"));

    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/v1/responses",
        &[],
        Some(r#"{"model":"qwen3.8-27b","input":"Follow up","stream":true}"#),
    );
    assert_eq!(st, 200);
    let r_sse = parse_sse(&body);
    assert!(r_sse
        .iter()
        .any(|(ev, _)| ev.as_deref() == Some("response.completed")));

    let (st_get_r, _, _) = raw_http_request(port, "GET", "/v1/responses/resp_123", &[], None);
    assert_eq!(st_get_r, 404);
    let (st_del_r, _, _) = raw_http_request(port, "DELETE", "/v1/responses/resp_123", &[], None);
    assert_eq!(st_del_r, 404);

    // 10. /v1/messages + /v1/messages/count_tokens
    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/v1/messages/count_tokens",
        &[],
        Some(r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Count me"}]}"#),
    );
    assert_eq!(st, 200);
    let ct = JsonValue::parse(&body).unwrap();
    assert!(ct.get_i64("input_tokens").unwrap_or(0) > 0);

    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/v1/messages",
        &[],
        Some(
            r#"{"model":"qwen3.8-27b","max_tokens":128,"thinking":{"type":"enabled","budget_tokens":32},"messages":[{"role":"user","content":"Hello"}]}"#,
        ),
    );
    assert_eq!(st, 200);
    let am1 = JsonValue::parse(&body).unwrap();
    assert_eq!(am1.get_str("type"), Some("message"));

    let (st, _, body) = raw_http_request(
        port,
        "POST",
        "/v1/messages",
        &[],
        Some(
            r#"{"model":"qwen3.8-27b","max_tokens":256,"stream":true,"thinking":{"type":"enabled","budget_tokens":32},"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],"messages":[{"role":"user","content":"Berlin weather"}]}"#,
        ),
    );
    assert_eq!(st, 200);
    let am_sse = parse_sse(&body);
    assert!(am_sse
        .iter()
        .any(|(ev, _)| ev.as_deref() == Some("message_stop")));

    // 11. /metrics
    let (st, _, body) = raw_http_request(port, "GET", "/metrics?format=prometheus", &[], None);
    assert_eq!(st, 200);
    assert!(body.contains("vllm:prompt_tokens_total"));
    assert!(body.contains("strata:live_state"));

    srv.stop();
    let _ = std::fs::remove_dir_all(&scratch_slots);

    // 12. API key auth enforcement
    let tok2 = ByteTokenizer::new();
    let engine2 = Arc::new(ScriptedMockEngine::new(tok2.clone()));
    let cfg2 = ServerConfig {
        api_key: Some("secret-token".to_string()),
        ..Default::default()
    };
    let state2 = Arc::new(ServiceState::new(cfg2, engine2, tok2));
    let mut srv2 = TalusServer::bind(state2)
        .expect("start auth server")
        .start_background();
    let port2 = srv2.port();

    let (st_health, _, _) = raw_http_request(port2, "GET", "/health", &[], None);
    assert_eq!(st_health, 200);
    let (st_unauth, _, _) = raw_http_request(port2, "GET", "/v1/models", &[], None);
    assert_eq!(st_unauth, 401);
    let (st_bearer, _, _) = raw_http_request(
        port2,
        "GET",
        "/v1/models",
        &[("Authorization", "Bearer secret-token")],
        None,
    );
    assert_eq!(st_bearer, 200);
    let (st_xkey, _, _) = raw_http_request(
        port2,
        "GET",
        "/v1/models",
        &[("x-api-key", "secret-token")],
        None,
    );
    assert_eq!(st_xkey, 200);
    srv2.stop();
}

#[test]
fn test_stdio_subprocess_engine_protocol() {
    let py_mock = r#"
import sys
sys.stdout.write("READY 8192\n")
sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if line.startswith("GEN "):
        sys.stdout.write("RESUME 3\n")
        sys.stdout.write("PP 2 5\n")
        for b in [72, 105, 33]:
            sys.stdout.write(f"T {b}\n")
        sys.stdout.write("DONE 3 5 5.0 9.0 stop 1 2 3\n")
        sys.stdout.flush()
    elif line == "QUIT":
        break
"#;
    let engine = StdioSubprocessEngine::spawn(
        "python3",
        &[
            "-u".to_string(),
            "-c".to_string(),
            py_mock.to_string(),
            "--serve".to_string(),
        ],
        &[],
        4096,
    )
    .expect("spawn python mock stdio engine");
    assert_eq!(engine.max_context(), 8192);

    let req = GenerateRequest {
        prompt_ids: vec![1, 2, 3, 4, 5],
        max_new_tokens: 16,
        sampling: SamplingParams::default(),
        stop_tokens: vec![],
        stop_strings: vec![],
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let mut streamed = Vec::new();
    let mut pp_events = Vec::new();
    let timings = engine
        .generate(
            &req,
            cancel,
            &mut |tok| {
                streamed.push(tok);
                true
            },
            &mut |read, total| {
                pp_events.push((read, total));
            },
        )
        .expect("generate via stdio engine");

    assert_eq!(streamed, vec![72, 105, 33]);
    assert_eq!(pp_events, vec![(5, 5)]);
    assert_eq!(timings.cache_n, 3);
    assert_eq!(timings.predicted_n, 3);
    assert_eq!(timings.draft_n, Some(2));
    assert_eq!(timings.draft_accepted, Some(1));
}

#[test]
fn test_dual_server_python_and_rust_parity_suite() {
    let bin_path = PathBuf::from(env!("CARGO_BIN_EXE_talus-server"));
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script_path = manifest_dir.join("tests").join("run_dual_server_parity.py");

    let output = Command::new("python3")
        .arg(&script_path)
        .env("TALUS_SERVER_BIN", &bin_path)
        .output()
        .expect("failed to execute tests/run_dual_server_parity.py");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "run_dual_server_parity.py failed (status {:?}):\nSTDOUT:\n{}\nSTDERR:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
    assert!(stdout.contains("ALL 12/12 DUAL-SERVER PARITY CHECKS PASSED"));
}
