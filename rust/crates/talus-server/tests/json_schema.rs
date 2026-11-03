//! E6: `response_format` json_schema / json_object on /v1/chat/completions against the mock engine.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use talus_server::config::ServerConfig;
use talus_server::engine::{SamplingParams, ScriptedMockEngine};
use talus_server::json::JsonValue;
use talus_server::server::{response_format_schema, ServerHandle, ServiceState, TalusServer};
use talus_server::tokenizer::ByteTokenizer;

fn post(port: u16, path: &str, body: &str) -> (u16, JsonValue) {
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
    (code, JsonValue::parse(payload).unwrap_or(JsonValue::Null))
}

fn start(support: bool) -> (ServerHandle, Arc<ScriptedMockEngine>) {
    let engine = Arc::new(ScriptedMockEngine::with_scripts(vec![
        r#"{"city": "Lyon"}"#.to_string(),
    ]));
    engine.set_json_schema_support(support);
    let cfg = ServerConfig {
        model_name: "qwen3.8-27b".to_string(),
        port: 0,
        ..Default::default()
    };
    let state = Arc::new(ServiceState::new(cfg, engine.clone(), ByteTokenizer::new()));
    (TalusServer::bind(state).unwrap().start_background(), engine)
}

const SCHEMA_REQ: &str = r#"{"messages":[{"role":"user","content":"city?"}],"max_tokens":32,
  "response_format":{"type":"json_schema","json_schema":{"name":"c","strict":true,
  "schema":{"type":"object","title":"T","properties":{"city":{"type":"string","description":"d"}},"required":["city"],"additionalProperties":false}}}}"#;

#[test]
fn unsupported_engine_gets_400_and_json_object_stays_ignored() {
    if std::env::var("STRATA_API_KEY").is_ok() {
        return;
    }
    let (srv, eng) = start(false);
    let (code, body) = post(srv.port(), "/v1/chat/completions", SCHEMA_REQ);
    assert_eq!(code, 400);
    assert!(body
        .to_json_string()
        .contains("json_schema not supported by this engine"));
    let (code, _) = post(
        srv.port(),
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"response_format":{"type":"json_object"}}"#,
    );
    assert_eq!(code, 200);
    assert_eq!(eng.last_json_schema(), None);
}

#[test]
fn supported_engine_gets_the_compact_schema_without_thinking() {
    if std::env::var("STRATA_API_KEY").is_ok() {
        return;
    }
    let (srv, eng) = start(true);
    let tok = ByteTokenizer::new();
    let (code, body) = post(srv.port(), "/v1/chat/completions", SCHEMA_REQ);
    assert_eq!(code, 200, "{}", body.to_json_string());
    assert_eq!(
        eng.last_json_schema().as_deref(),
        Some(
            r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}"#
        )
    );
    let prompt = tok.decode(&eng.last_prompt());
    assert!(!prompt.contains("Reasoning effort"), "{prompt}");

    // no response_format: unchanged request (thinking on by default, no schema)
    let (code, _) = post(
        srv.port(),
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"city?"}],"max_tokens":8}"#,
    );
    assert_eq!(code, 200);
    assert_eq!(eng.last_json_schema(), None);
    assert!(tok.decode(&eng.last_prompt()).contains("Reasoning effort"));

    // json_object -> any object
    let (code, _) = post(
        srv.port(),
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"response_format":{"type":"json_object"}}"#,
    );
    assert_eq!(code, 200);
    assert_eq!(
        eng.last_json_schema().as_deref(),
        Some(r#"{"type":"object"}"#)
    );

    // unsupported keyword, bad type, tools -> 400
    for bad in [
        r#"{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"string","pattern":"a"}}}}"#,
        r#"{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"yaml"}}"#,
        r#"{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema"}}"#,
        r#"{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"response_format":{"type":"json_object"}}"#,
    ] {
        let (code, body) = post(srv.port(), "/v1/chat/completions", bad);
        assert_eq!(code, 400, "{bad} -> {}", body.to_json_string());
    }
}

#[test]
fn response_format_parsing_and_gen_key() {
    let p = |s: &str| response_format_schema(&JsonValue::parse(s).unwrap());
    assert_eq!(p(r#"{}"#), Ok(None));
    assert_eq!(p(r#"{"response_format":null}"#), Ok(None));
    assert_eq!(p(r#"{"response_format":{"type":"text"}}"#), Ok(None));
    assert_eq!(
        p(r#"{"response_format":{"type":"json_schema","schema":{"enum":["a b"]}}}"#),
        Ok(Some(r#"{"enum":["a b"]}"#.to_string()))
    );
    let sp = SamplingParams {
        json_schema: Some(r#"{"enum":["a b","50%"]}"#.to_string()),
        ..Default::default()
    };
    assert!(sp
        .to_gen_keys()
        .ends_with(r#" json_schema={"enum":["a%20b","50%25"]}"#));
    assert!(!SamplingParams::default()
        .to_gen_keys()
        .contains("json_schema"));
}
