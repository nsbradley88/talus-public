//! E1 (opt-in top-k logprobs) and E2 (`POST /v1/talus/score`) against the in-process mock engine and against the
//! stdio protocol (`talus-server --mock-stdio-worker` with `TALUS_MOCK_LOGPROBS=1`, i.e. `INFO logprobs=1 score=1`,
//! `L` and `S` lines). bench-design/server-extensions.md.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use talus_server::config::ServerConfig;
use talus_server::engine::{Engine, ScriptedMockEngine, StdioSubprocessEngine};
use talus_server::json::JsonValue;
use talus_server::server::{ServerHandle, ServiceState, TalusServer};
use talus_server::tokenizer::ByteTokenizer;

fn http_raw(port: u16, path: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    let (head, payload) = resp.split_once("\r\n\r\n").unwrap();
    let code = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (code, payload.to_string())
}

fn post(port: u16, path: &str, body: &str) -> (u16, JsonValue) {
    let (code, payload) = http_raw(port, path, body);
    (code, JsonValue::parse(&payload).unwrap_or(JsonValue::Null))
}

/// SSE `data:` payloads (chunked transfer framing tolerated: only lines starting with `data: {`).
fn sse(port: u16, path: &str, body: &str) -> Vec<JsonValue> {
    let (code, payload) = http_raw(port, path, body);
    assert_eq!(code, 200, "{payload}");
    payload
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|l| l.starts_with('{'))
        .map(|l| JsonValue::parse(l).unwrap())
        .collect()
}

fn skip() -> bool {
    std::env::var("STRATA_API_KEY").is_ok() || std::env::var("TALUS_API_KEY").is_ok()
}

fn serve(engine: Arc<dyn Engine>) -> ServerHandle {
    let cfg = ServerConfig {
        model_name: "qwen3.8-27b".to_string(),
        port: 0,
        ..Default::default()
    };
    let state = Arc::new(ServiceState::new(cfg, engine, ByteTokenizer::new()));
    TalusServer::bind(state).unwrap().start_background()
}

fn mock(lp: bool, score: bool) -> ServerHandle {
    let e = ScriptedMockEngine::with_scripts(vec!["Hello there, this is a mock.".to_string()]);
    e.set_logprobs_support(lp);
    e.set_score_support(score);
    serve(Arc::new(e))
}

fn stdio(lp: bool) -> ServerHandle {
    let env = if lp {
        vec![("TALUS_MOCK_LOGPROBS".to_string(), "1".to_string())]
    } else {
        Vec::new()
    };
    let e = StdioSubprocessEngine::spawn(
        env!("CARGO_BIN_EXE_talus-server"),
        &["--mock-stdio-worker".to_string()],
        &env,
        32768,
    )
    .unwrap();
    assert_eq!(e.supports_logprobs(), lp);
    assert_eq!(e.supports_score(), lp);
    serve(Arc::new(e))
}

fn content_entries(v: &JsonValue) -> Vec<JsonValue> {
    v.get("content")
        .and_then(|c| c.as_array())
        .map(|a| a.to_vec())
        .unwrap_or_default()
}

fn check_entry(e: &JsonValue, k: usize) {
    assert!(e.get_str("token").is_some(), "{}", e.to_json_string());
    let lp = e.get_f64("logprob").unwrap();
    assert!(lp <= 0.0);
    assert!(e.get("bytes").and_then(|b| b.as_array()).is_some());
    let top = e.get("top_logprobs").and_then(|t| t.as_array()).unwrap();
    assert_eq!(top.len(), k);
    if k > 0 {
        // the mock's chosen token is its top-1 (as a greedy engine's must be)
        assert_eq!(top[0].get_f64("logprob").unwrap(), lp);
        assert_eq!(top[0].get_i64("id"), e.get_i64("id"));
        for w in top.windows(2) {
            assert!(w[0].get_f64("logprob").unwrap() >= w[1].get_f64("logprob").unwrap());
        }
    }
    let ch = e.get_str("channel").unwrap();
    assert!(ch == "reasoning" || ch == "content");
}

const CHAT: &str = r#""messages":[{"role":"user","content":"hi"}],"temperature":0,"max_tokens":64"#;

fn run_e1_suite(srv: &ServerHandle) {
    let port = srv.port();
    // off: no key change (chat has no logprobs key; completions keep null)
    let (code, v) = post(port, "/v1/chat/completions", &format!("{{{CHAT}}}"));
    assert_eq!(code, 200);
    let choice = &v.get("choices").unwrap().as_array().unwrap()[0];
    assert!(choice.get("logprobs").is_none());
    let off_msg = choice.get("message").unwrap().to_json_string();
    let (code, v) = post(
        port,
        "/v1/completions",
        r#"{"prompt":"abc","max_tokens":8,"temperature":0}"#,
    );
    assert_eq!(code, 200);
    let choice = &v.get("choices").unwrap().as_array().unwrap()[0];
    assert!(choice.get("logprobs").unwrap().is_null());
    let (_, v) = post(port, "/completion", r#"{"prompt":"abc","n_predict":8}"#);
    assert!(v.get("logprobs").is_none());

    // chat on: one entry per generated token, same message as off
    let (code, v) = post(
        port,
        "/v1/chat/completions",
        &format!(r#"{{{CHAT},"logprobs":true,"top_logprobs":3}}"#),
    );
    assert_eq!(code, 200, "{}", v.to_json_string());
    let choice = &v.get("choices").unwrap().as_array().unwrap()[0];
    assert_eq!(choice.get("message").unwrap().to_json_string(), off_msg);
    let entries = content_entries(choice.get("logprobs").unwrap());
    let ct = v
        .get("usage")
        .unwrap()
        .get_usize("completion_tokens")
        .unwrap();
    assert_eq!(entries.len(), ct);
    for e in &entries {
        check_entry(e, 3);
    }
    // logprobs: true alone = chosen token only
    let (_, v) = post(
        port,
        "/v1/chat/completions",
        &format!(r#"{{{CHAT},"logprobs":true}}"#),
    );
    let choice = &v.get("choices").unwrap().as_array().unwrap()[0];
    for e in content_entries(choice.get("logprobs").unwrap()) {
        check_entry(&e, 0);
    }

    // bounds
    for bad in [
        r#""logprobs":true,"top_logprobs":21"#,
        r#""logprobs":true,"top_logprobs":-1"#,
        r#""top_logprobs":2"#,
        r#""logprobs":"yes""#,
    ] {
        let (code, v) = post(port, "/v1/chat/completions", &format!("{{{CHAT},{bad}}}"));
        assert_eq!(code, 400, "{bad}: {}", v.to_json_string());
    }
    let (code, _) = post(
        port,
        "/v1/completions",
        r#"{"prompt":"abc","max_tokens":4,"logprobs":21}"#,
    );
    assert_eq!(code, 400);

    // chat stream: the chunks' entries cover every generated token, in order
    let chunks = sse(
        port,
        "/v1/chat/completions",
        &format!(r#"{{{CHAT},"stream":true,"logprobs":true,"top_logprobs":2}}"#),
    );
    let mut n = 0usize;
    let mut ct = 0usize;
    for c in &chunks {
        if let Some(ch) = c
            .get("choices")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
        {
            if let Some(lp) = ch.get("logprobs") {
                for e in content_entries(lp) {
                    check_entry(&e, 2);
                    n += 1;
                }
            }
        }
        if let Some(u) = c.get("usage") {
            ct = u.get_usize("completion_tokens").unwrap();
        }
    }
    assert!(ct > 0);
    assert_eq!(n, ct);
    // stream off: no logprobs key in the chunks
    let chunks = sse(
        port,
        "/v1/chat/completions",
        &format!(r#"{{{CHAT},"stream":true}}"#),
    );
    for c in &chunks {
        if let Some(ch) = c
            .get("choices")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
        {
            assert!(ch.get("logprobs").is_none());
        }
    }

    // completions on (non-stream and stream)
    let (code, v) = post(
        port,
        "/v1/completions",
        r#"{"prompt":"abc","max_tokens":8,"temperature":0,"logprobs":4}"#,
    );
    assert_eq!(code, 200);
    let choice = &v.get("choices").unwrap().as_array().unwrap()[0];
    let entries = content_entries(choice.get("logprobs").unwrap());
    assert_eq!(
        entries.len(),
        v.get("usage")
            .unwrap()
            .get_usize("completion_tokens")
            .unwrap()
    );
    for e in &entries {
        check_entry(e, 4);
    }
    let chunks = sse(
        port,
        "/v1/completions",
        r#"{"prompt":"abc","max_tokens":8,"stream":true,"logprobs":1}"#,
    );
    let n: usize = chunks
        .iter()
        .filter_map(|c| {
            c.get("choices")
                .and_then(|a| a.as_array())
                .and_then(|a| a.first())
                .cloned()
        })
        .map(|ch| content_entries(ch.get("logprobs").unwrap()).len())
        .sum();
    assert_eq!(n, 8);
    let (code, v) = post(
        port,
        "/completion",
        r#"{"prompt":"abc","n_predict":8,"logprobs":2}"#,
    );
    assert_eq!(code, 200);
    let entries = content_entries(v.get("logprobs").unwrap());
    assert_eq!(entries.len(), v.get_usize("tokens_predicted").unwrap());
}

fn run_e2_suite(srv: &ServerHandle) {
    let port = srv.port();
    let (code, v) = post(
        port,
        "/v1/talus/score",
        r#"{"prompt_ids":[1,2,3],"continuation_ids":[4,5,6,7],"top_k":3}"#,
    );
    assert_eq!(code, 200, "{}", v.to_json_string());
    let pos = v.get("positions").unwrap().as_array().unwrap();
    assert_eq!(pos.len(), 4);
    assert_eq!(v.get_usize("continuation_tokens"), Some(4));
    assert_eq!(v.get_usize("prompt_tokens"), Some(3));
    for (i, p) in pos.iter().enumerate() {
        assert_eq!(p.get_i64("id"), Some(4 + i as i64));
        assert!(p.get_f64("logprob").is_some());
        assert!(p.get_i64("rank").is_some());
        assert!(p.get_f64("entropy").is_some());
        assert_eq!(p.get("top_k").unwrap().as_array().unwrap().len(), 3);
    }
    // text inputs are tokenized and echoed back; `return` narrows the fields
    let (code, v) = post(
        port,
        "/v1/talus/score",
        r#"{"prompt":"ab","continuation":"cd","return":["logprob"]}"#,
    );
    assert_eq!(code, 200);
    let cont: Vec<i64> = v
        .get("continuation_ids")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect();
    assert_eq!(cont, vec![99, 100]);
    let p0 = &v.get("positions").unwrap().as_array().unwrap()[0];
    assert!(p0.get("rank").is_none() && p0.get("top_k").is_none() && p0.get("logprob").is_some());
    // limits
    for bad in [
        r#"{"prompt_ids":[1],"continuation_ids":[2],"top_k":65}"#,
        r#"{"prompt_ids":[],"continuation_ids":[2]}"#,
        r#"{"prompt_ids":[1]}"#,
        r#"{"prompt_ids":[1],"continuation_ids":[2],"return":["nope"]}"#,
    ] {
        let (code, _) = post(port, "/v1/talus/score", bad);
        assert_eq!(code, 400, "{bad}");
    }
    let long: Vec<String> = (0..4097).map(|_| "5".to_string()).collect();
    let (code, v) = post(
        port,
        "/v1/talus/score",
        &format!(
            r#"{{"prompt_ids":[1],"continuation_ids":[{}]}}"#,
            long.join(",")
        ),
    );
    assert_eq!(code, 400);
    assert!(v.to_json_string().contains("scoring limit"));
}

#[test]
fn mock_engine_e1_e2() {
    if skip() {
        return;
    }
    let srv = mock(true, true);
    run_e1_suite(&srv);
    run_e2_suite(&srv);
}

#[test]
fn stdio_protocol_e1_e2() {
    if skip() {
        return;
    }
    let srv = stdio(true);
    run_e1_suite(&srv);
    run_e2_suite(&srv);
}

#[test]
fn unsupported_engines_answer_honestly() {
    if skip() {
        return;
    }
    for srv in [mock(false, false), stdio(false)] {
        let port = srv.port();
        let (code, v) = post(
            port,
            "/v1/chat/completions",
            &format!(r#"{{{CHAT},"logprobs":true,"top_logprobs":2}}"#),
        );
        assert_eq!(code, 400);
        assert!(v
            .to_json_string()
            .contains("logprobs not supported by this engine"));
        let (code, _) = post(
            port,
            "/v1/completions",
            r#"{"prompt":"abc","max_tokens":4,"logprobs":0}"#,
        );
        assert_eq!(code, 400);
        let (code, v) = post(
            port,
            "/v1/talus/score",
            r#"{"prompt_ids":[1,2],"continuation_ids":[3]}"#,
        );
        assert_eq!(code, 501);
        assert!(v
            .to_json_string()
            .contains("scoring not supported by this engine"));
        // a request without the fields still works
        let (code, _) = post(port, "/v1/chat/completions", &format!("{{{CHAT}}}"));
        assert_eq!(code, 200);
    }
}

#[test]
fn gen_line_is_unchanged_when_off() {
    use talus_server::engine::SamplingParams;
    let s = SamplingParams::default();
    assert!(!s.to_gen_keys().contains("logprobs"));
    let s = SamplingParams {
        logprobs: Some(5),
        ..SamplingParams::default()
    };
    assert!(s.to_gen_keys().ends_with(" logprobs=5"));
}
