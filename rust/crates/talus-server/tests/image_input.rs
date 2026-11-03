//! Non-text content parts (image_url, input_image, Anthropic image blocks, audio, files) get an honest HTTP 400
//! `unsupported_content` instead of a reply that silently ignored the input (talus-server had no vision encoder
//! and swapped images for placeholder text). Text-only requests are unaffected.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use talus_server::config::ServerConfig;
use talus_server::engine::ScriptedMockEngine;
use talus_server::json::JsonValue;
use talus_server::server::{ServerHandle, ServiceState, TalusServer};
use talus_server::tokenizer::ByteTokenizer;

fn http(port: u16, path: &str, body: &str) -> (u16, String) {
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

fn start(scripts: Vec<String>) -> ServerHandle {
    let cfg = ServerConfig {
        model_name: "qwen3.8-27b".to_string(),
        port: 0,
        ..Default::default()
    };
    let state = Arc::new(ServiceState::new(
        cfg,
        Arc::new(ScriptedMockEngine::with_scripts(scripts)),
        ByteTokenizer::new(),
    ));
    TalusServer::bind(state).unwrap().start_background()
}

fn skip() -> bool {
    std::env::var("STRATA_API_KEY").is_ok() || std::env::var("TALUS_API_KEY").is_ok()
}

#[test]
fn image_parts_get_400_unsupported_content() {
    if skip() {
        return;
    }
    let srv = start(vec!["</think>\n\nshould not be sent".to_string(); 4]);
    let port = srv.port();

    let chat = r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":[{"type":"text","text":"What is in this picture?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]}]}"#;
    let (code, body) = http(port, "/v1/chat/completions", chat);
    assert_eq!(code, 400, "{body}");
    let v = JsonValue::parse(&body).unwrap();
    let e = v.get("error").unwrap();
    assert_eq!(
        e.get_str("message"),
        Some("image input is not supported by this model")
    );
    assert_eq!(e.get_str("type"), Some("invalid_request_error"));
    assert_eq!(e.get_str("code"), Some("unsupported_content"));

    let resp = r#"{"model":"qwen3.8-27b","input":[{"role":"user","content":[{"type":"input_text","text":"describe"},{"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo="}]}]}"#;
    let (code, body) = http(port, "/v1/responses", resp);
    assert_eq!(code, 400, "{body}");
    let e = JsonValue::parse(&body).unwrap();
    let e = e.get("error").unwrap();
    assert_eq!(e.get_str("code"), Some("unsupported_content"));
    assert_eq!(
        e.get_str("message"),
        Some("image input is not supported by this model")
    );

    let anth = r#"{"model":"qwen3.8-27b","max_tokens":16,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}},{"type":"text","text":"describe"}]}]}"#;
    let (code, body) = http(port, "/v1/messages", anth);
    assert_eq!(code, 400, "{body}");
    let v = JsonValue::parse(&body).unwrap();
    assert_eq!(v.get_str("type"), Some("error"));
    assert_eq!(
        v.get("error").unwrap().get_str("message"),
        Some("image input is not supported by this model")
    );

    // a text-only request on the same server is still answered
    let text = r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":[{"type":"text","text":"Hello!"}]}]}"#;
    let (code, body) = http(port, "/v1/chat/completions", text);
    assert_eq!(code, 200, "{body}");
    let v = JsonValue::parse(&body).unwrap();
    let msg = v.get("choices").unwrap().as_array().unwrap()[0]
        .get("message")
        .unwrap()
        .clone();
    assert_eq!(msg.get_str("content"), Some("should not be sent"));
}
