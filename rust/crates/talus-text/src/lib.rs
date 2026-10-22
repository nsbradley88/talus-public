//! BPE tokenizer, chat templates, and reasoning/tool-call parsers.
//!
//! Implements WP-1.4a (`talus-text`), porting `tools/strata_tokenizer.py` and
//! `serve/frontend.py` + `serve/chat_template.jinja`.

pub mod parser;
pub mod template;
pub mod tokenizer;

pub use parser::{
    call_end, forced_call, function_end, json_tool_call, param_end, parse_any_tool_call,
    parse_tool_call, to_xml_call, tool_choice_of, Event, OutputParser, ParseEvent, ParserError,
    StreamDelta, StreamParser, ToolCall, ToolCallDelta, ToolCallFunctionDelta, CALL_END,
    CALL_START, FUNC_END, FUNC_START, PARAM_END, PARAM_START, RCALL_MAX, THINK_END, THINK_START,
};
pub use template::{
    anthropic_to_messages, budget_effort, default_literal_tags, effort_kwargs, has_image,
    images_of, literal_tags, mark_think_literals, openai_to_messages, pin_billing_stamp,
    strip_generation_tags, text_of, to_python_json, tool_arguments, unmark_think_literals,
    ChatTemplate, TemplateCapabilities, TemplateError, TemplateRequestError, CONTROL_MARK0,
};
pub use tokenizer::{
    bytes_to_unicode, pre_patterns, read_gguf_tokenizer_metadata, Detokenizer,
    GgufTokenizerMetadata, Tokenizer, TokenizerError, GLM4_PATTERN, GPT2_PATTERN, LAGUNA_PATTERNS,
    QWEN2_PATTERN, QWEN35_PATTERN,
};

pub const CRATE_NAME: &str = "talus-text";

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    #[test]
    fn crate_identity() {
        assert_eq!(CRATE_NAME, "talus-text");
    }

    // -------------------------------------------------------------------------
    // 1. ChatTemplate golden parity & {% generation %} stripping tests
    // -------------------------------------------------------------------------

    #[test]
    fn golden_chat_template_parity() {
        let root = repo_root();
        let jinja_path = root.join("serve/chat_template.jinja");
        let golden_path = root.join("serve/chat_golden.json");

        let tmpl = ChatTemplate::from_file(&jinja_path).expect("load chat_template.jinja");
        let golden_text = std::fs::read_to_string(&golden_path).expect("read chat_golden.json");
        let golden: Value = serde_json::from_str(&golden_text).expect("parse chat_golden.json");
        let cases = golden.as_array().expect("golden top-level array");
        assert_eq!(cases.len(), 10);

        for case in cases {
            let name = case["name"].as_str().unwrap();
            let messages = case["messages"].as_array().unwrap();
            let tools = case.get("tools").and_then(|t| t.as_array()).cloned();
            let add_gen = case
                .get("add_generation_prompt")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let enable_thinking = case.get("enable_thinking").and_then(|v| v.as_bool());
            let expected = case["rendered"].as_str().unwrap();

            let rendered = tmpl
                .render_with_thinking(messages, tools.as_deref(), add_gen, enable_thinking)
                .unwrap_or_else(|e| panic!("case {name} failed to render: {e}"));
            assert_eq!(rendered, expected, "golden mismatch on case '{name}'");
        }
    }

    #[test]
    fn strip_generation_tags_and_laguna_template_render() {
        let src = r#"{{- bos_token -}}
{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- '<|im_start|>assistant\n' -}}
        {%- generation -%}
        {{- message['content'] + '<|im_end|>\n' -}}
        {%- endgeneration -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n' -}}
{%- endif -%}"#;

        let stripped = strip_generation_tags(src);
        assert!(!stripped.contains("endgeneration"));
        assert!(!stripped.contains("% generation"));

        let tmpl = ChatTemplate::from_str(src).expect("compile template with generation tags");
        let msgs = vec![
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "Hi there!"}),
            json!({"role": "user", "content": "How are you?"}),
        ];
        let mut extra = serde_json::Map::new();
        extra.insert("bos_token".to_string(), json!(""));
        let out = tmpl
            .render(&msgs, None, true, Some(&extra))
            .expect("render laguna template");
        assert_eq!(
            out,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn glm_arg_key_template_render() {
        let glm_src = r#"[gMASK]<sop>
{%- for m in messages -%}
{%- if m.role == 'user' -%}<|user|>
{{ m.content }}
{%- elif m.role == 'assistant' -%}<|assistant|>
{%- if m.reasoning_content is defined and m.reasoning_content -%}<think>{{ m.reasoning_content }}</think>{%- endif -%}
{{ m.content }}
{%- if m.tool_calls is defined and m.tool_calls -%}
{%- for tc in m.tool_calls -%}
<tool_call>{{ tc.function.name }}
{%- for k, v in tc.function.arguments.items() -%}
<arg_key>{{ k }}</arg_key><arg_value>{{ v if v is string else (v | tojson) }}</arg_value>
{%- endfor -%}</tool_call>
{%- endfor -%}
{%- endif -%}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}<|assistant|>
{%- if enable_thinking is defined and not enable_thinking -%}<think></think>{%- else -%}<think>{%- endif -%}
{%- endif -%}"#;

        let tmpl = ChatTemplate::from_str(glm_src).expect("compile glm template");
        let msgs = vec![
            json!({"role": "user", "content": "Read foo.rs"}),
            json!({
                "role": "assistant",
                "content": "",
                "reasoning_content": "Let's read foo.rs",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "foo.rs", "limit": 50}
                    }
                }]
            }),
        ];
        let out = tmpl
            .render_with_thinking(&msgs, None, true, Some(true))
            .expect("render glm");
        assert!(out.contains(
            "<tool_call>read_file<arg_key>path</arg_key><arg_value>foo.rs</arg_value><arg_key>limit</arg_key><arg_value>50</arg_value></tool_call>"
        ));
        assert!(out.ends_with("<|assistant|><think>"));
    }

    #[test]
    fn template_capabilities_and_literal_marking() {
        let root = repo_root();
        let tmpl =
            ChatTemplate::from_file(root.join("serve/chat_template.jinja")).expect("load template");
        let caps = tmpl.capabilities();
        assert!(caps.supports_system_role);
        assert!(caps.supports_tools);
        assert!(caps.supports_tool_calls);
        assert!(caps.supports_parallel_tool_calls);
        assert!(caps.supports_preserve_reasoning);

        // Literal tag marking in user/tool messages vs assistant messages starting with <think>...</think>
        let tags = literal_tags(["<|im_start|>", "<|im_end|>"]);
        let msgs = vec![
            json!({"role": "user", "content": "Literal <think> and <|im_start|> in user"}),
            json!({"role": "assistant", "content": "<think>ok</think>Assistant reply"}),
        ];
        let (marked, _, changed) = mark_think_literals(&msgs, None, Some(&tags));
        assert!(changed);
        let rendered = tmpl
            .render_with_thinking(&marked, None, true, Some(false))
            .expect("render marked");
        let (clean, spans) = unmark_think_literals(&rendered, Some(&tags));
        assert!(!clean.contains('\u{F0E01}'));
        assert_eq!(spans.len(), 2);
        assert_eq!(&clean[spans[0].0..spans[0].1], "<think>");
        assert_eq!(&clean[spans[1].0..spans[1].1], "<|im_start|>");
    }

    #[test]
    fn openai_and_anthropic_request_normalization() {
        let oai_req = json!({
            "messages": [
                {"role": "developer", "content": "Be concise."},
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Boston\"}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": {"temp": 72}}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }],
            "reasoning_effort": "low"
        });
        let (msgs, tools, kwargs) = openai_to_messages(&oai_req).expect("openai_to_messages");
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "Be concise.");
        assert_eq!(
            msgs[2]["tool_calls"][0]["function"]["arguments"]["city"],
            "Boston"
        );
        assert!(tools.is_some());
        assert_eq!(kwargs.get("reasoning_effort"), Some(&json!("low")));

        let anth_req = json!({
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.170.bf4; cch=abc123;"},
                {"type": "text", "text": "System prompt"}
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Run tool"}]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Thinking..."},
                        {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"cmd": "ls"}}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file1\nfile2"},
                        {"type": "text", "text": "Done?"}
                    ]
                }
            ],
            "thinking": {"type": "enabled", "budget_tokens": 1024}
        });
        let (amsgs, _, akwargs) =
            anthropic_to_messages(&anth_req, true).expect("anthropic_to_messages");
        assert_eq!(
            amsgs[0]["content"],
            "x-anthropic-billing-header: cc_version=2.1.170.fff; cch=ffffff;System prompt"
        );
        assert_eq!(amsgs[1]["role"], "user");
        assert_eq!(amsgs[2]["role"], "assistant");
        assert_eq!(amsgs[2]["reasoning_content"], "Thinking...");
        assert_eq!(amsgs[3]["role"], "tool");
        assert_eq!(amsgs[4]["role"], "user");
        assert_eq!(akwargs.get("reasoning_effort"), Some(&json!("low")));
    }

    // -------------------------------------------------------------------------
    // 2. OutputParser & StreamParser tests (Qwen XML, Qwen JSON, GLM <arg_key>,
    //    <think> splitting, fence tracking, and rcall_specimens.json)
    // -------------------------------------------------------------------------

    #[test]
    fn rcall_specimens_fixture_parity() {
        let root = repo_root();
        let spec_path = root.join("serve/fixtures/rcall_specimens.json");
        let spec_text = std::fs::read_to_string(&spec_path).expect("read rcall_specimens.json");
        let spec_json: Value =
            serde_json::from_str(&spec_text).expect("parse rcall_specimens.json");

        let specimens = spec_json["specimens"].as_array().unwrap();
        assert_eq!(specimens.len(), 37);

        let mut positive_count = 0usize;
        for sp in specimens {
            let name = sp["name"].as_str().unwrap();
            let finish = sp["finish"].as_str();
            let tool_names = sp["tools"].as_array().unwrap();
            let tools: Vec<Value> = tool_names
                .iter()
                .map(|n| json!({"name": n.as_str().unwrap(), "parameters": {"properties": {}}}))
                .collect();
            let input = sp["input"].as_str().unwrap();
            let want_calls: Vec<String> = sp["calls"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            if !want_calls.is_empty() {
                positive_count += 1;
            }

            // Test across chunk widths (1, 2, 3, 7, whole) and stream_tools=(true, false), matching test_reasoning_rescue.py!
            let mut seen: HashSet<(String, Vec<String>)> = HashSet::new();
            for width in [1usize, 2, 3, 7, 100_000] {
                for stream_tools in [true, false] {
                    let mut p = OutputParser::new(true, Some(&tools), stream_tools, false, false);
                    let mut events = Vec::new();
                    let chars: Vec<char> = input.chars().collect();
                    for chunk in chars.chunks(width) {
                        let s: String = chunk.iter().collect();
                        events.extend(p.feed(&s));
                    }
                    events.extend(p.finish(finish));

                    let got_calls: Vec<String> = events
                        .iter()
                        .filter(|e| e.kind == "tool_call")
                        .map(|e| e.call.as_ref().unwrap().name.clone())
                        .collect();
                    let reasoning: String = events
                        .iter()
                        .filter(|e| e.kind == "reasoning")
                        .map(|e| e.text.as_str())
                        .collect();

                    assert_eq!(
                        got_calls, want_calls,
                        "specimen {name} (width={width}, stream_tools={stream_tools}): calls mismatch"
                    );
                    seen.insert((reasoning, got_calls));
                }
            }
            assert_eq!(
                seen.len(),
                1,
                "specimen {name}: streamed and whole results differed: {seen:?}"
            );
        }
        assert_eq!(positive_count, 16);
    }

    #[test]
    fn qwen_xml_qwen_json_and_glm_arg_key_tool_calls() {
        let schema = json!({
            "name": "edit_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "force": {"type": "boolean"},
                    "tags": {"type": "array"},
                    "meta": {"type": "object"},
                    "content": {"type": "string"}
                }
            }
        });
        let mut schemas = std::collections::HashMap::new();
        schemas.insert("edit_file".to_string(), schema.clone());

        // 1. Qwen XML with leading space in function name, type coercion, and literal </parameter> inside content
        let xml_body = "<function= edit_file>\n\
            <parameter=path>src/main.rs</parameter>\n\
            <parameter=line>42</parameter>\n\
            <parameter=ratio>0.75</parameter>\n\
            <parameter=force>true</parameter>\n\
            <parameter=tags>[\"a\", \"b\"]</parameter>\n\
            <parameter=meta>{\"k\": 1}</parameter>\n\
            <parameter=content>Use </parameter> and </tool_call> carefully\n</parameter>\n\
            </function>";
        let tc = parse_tool_call(xml_body, Some(&schema)).expect("parse Qwen XML");
        assert_eq!(tc.name, "edit_file");
        assert_eq!(tc.arguments["path"], json!("src/main.rs"));
        assert_eq!(tc.arguments["line"], json!(42));
        assert_eq!(tc.arguments["ratio"], json!(0.75));
        assert_eq!(tc.arguments["force"], json!(true));
        assert_eq!(tc.arguments["tags"], json!(["a", "b"]));
        assert_eq!(tc.arguments["meta"], json!({"k": 1}));
        assert_eq!(
            tc.arguments["content"],
            json!("Use </parameter> and </tool_call> carefully")
        );

        // 2. Qwen JSON inside <tool_call>
        let json_body =
            r#"{"name": "edit_file", "arguments": {"path": "lib.rs", "line": 10, "force": false}}"#;
        let tc_json = json_tool_call(json_body, Some(&schemas)).expect("parse Qwen JSON");
        assert_eq!(tc_json.name, "edit_file");
        assert_eq!(tc_json.arguments["path"], json!("lib.rs"));
        assert_eq!(tc_json.arguments["line"], json!(10));
        assert_eq!(tc_json.arguments["force"], json!(false));

        // 3. GLM <arg_key>...</arg_key><arg_value>...</arg_value> inside <tool_call>
        let glm_body = "edit_file\
            <arg_key>path</arg_key><arg_value>glm.rs</arg_value>\
            <arg_key>line</arg_key><arg_value>99</arg_value>\
            <arg_key>ratio</arg_key><arg_value>1.5</arg_value>\
            <arg_key>force</arg_key><arg_value>false</arg_value>\
            <arg_key>tags</arg_key><arg_value>[\"x\",\"y\"]</arg_value>\
            <arg_key>meta</arg_key><arg_value>{\"nested\":true}</arg_value>";
        let tc_glm = parse_any_tool_call(glm_body, Some(&schema)).expect("parse GLM arg_key");
        assert_eq!(tc_glm.name, "edit_file");
        assert_eq!(tc_glm.arguments["path"], json!("glm.rs"));
        assert_eq!(tc_glm.arguments["line"], json!(99));
        assert_eq!(tc_glm.arguments["ratio"], json!(1.5));
        assert_eq!(tc_glm.arguments["force"], json!(false));
        assert_eq!(tc_glm.arguments["tags"], json!(["x", "y"]));
        assert_eq!(tc_glm.arguments["meta"], json!({"nested": true}));
    }

    #[test]
    fn stream_parser_incremental_tool_call_deltas() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"}
                    }
                }
            }
        })];

        // Test streaming across Qwen XML, Qwen JSON, and GLM <arg_key> formats in 3-char chunks!
        let payloads = [
            (
                "qwen_xml",
                false,
                false,
                "Thinking step</think>\nHello!\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la & pwd\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>",
            ),
            (
                "qwen_json",
                true,
                false,
                "Thinking step</think>\nHello!\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls -la & pwd\", \"timeout\": 30}}\n</tool_call>",
            ),
            (
                "glm_arg_key",
                false,
                true,
                "Thinking step</think>\nHello!\n<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la & pwd</arg_value><arg_key>timeout</arg_key><arg_value>30</arg_value></tool_call>",
            ),
        ];

        for (label, recover, glm, text) in payloads {
            let mut sp = StreamParser::new(true, Some(&tools), true, recover, glm);
            let mut reasoning = String::new();
            let mut content = String::new();
            let mut tool_name = String::new();
            let mut tool_id = String::new();
            let mut tool_args_streamed = String::new();

            let chars: Vec<char> = text.chars().collect();
            for chunk in chars.chunks(3) {
                let s: String = chunk.iter().collect();
                for delta in sp.feed(&s) {
                    if let Some(r) = delta.reasoning_content {
                        reasoning.push_str(&r);
                    }
                    if let Some(c) = delta.content {
                        content.push_str(&c);
                    }
                    for tc_delta in delta.tool_calls {
                        assert_eq!(tc_delta.index, 0);
                        if let Some(id) = tc_delta.id {
                            tool_id = id;
                        }
                        if let Some(n) = tc_delta.function.name {
                            tool_name.push_str(&n);
                        }
                        tool_args_streamed.push_str(&tc_delta.function.arguments);
                    }
                }
            }
            for delta in sp.finish(Some("stop")) {
                for tc_delta in delta.tool_calls {
                    if let Some(id) = tc_delta.id {
                        tool_id = id;
                    }
                    if let Some(n) = tc_delta.function.name {
                        tool_name.push_str(&n);
                    }
                    tool_args_streamed.push_str(&tc_delta.function.arguments);
                }
            }

            let final_calls = sp.completed_tool_calls();
            assert_eq!(reasoning, "Thinking step", "{label}: reasoning");
            assert_eq!(content, "Hello!", "{label}: content");
            assert_eq!(tool_name, "bash", "{label}: tool_name");
            assert!(tool_id.starts_with("call_"), "{label}: tool_id");
            assert_eq!(final_calls.len(), 1, "{label}: final_calls len");
            assert!(sp.has_complete_tool_calls(), "{label}: complete tool calls");

            let parsed_streamed: Value =
                serde_json::from_str(&tool_args_streamed).unwrap_or_else(|e| {
                    panic!("{label}: invalid streamed JSON {tool_args_streamed:?}: {e}")
                });
            assert_eq!(
                parsed_streamed,
                json!({"command": "ls -la & pwd", "timeout": 30}),
                "{label}: streamed args mismatch"
            );
            assert_eq!(
                Value::Object(final_calls[0].arguments.clone()),
                parsed_streamed,
                "{label}: completed call args mismatch"
            );
        }
    }

    #[test]
    fn parser_track_backticks_and_fences() {
        // Port of serve/test_parser_track.py: <tool_call> inside inline code or fenced blocks must not trigger tool calls
        let tools = vec![json!({"name": "write"})];
        let text = format!(
            "{}`<tool_call>`\n\n```xml\n{}<tool_call><function=quoted></function></tool_call>\n```\n<tool_call><function=write><parameter=text>ok</parameter></function></tool_call>\nDone",
            "prefix ".repeat(64),
            "x".repeat(256)
        );
        for recover in [false, true] {
            for stream in [false, true] {
                let mut p = OutputParser::new(false, Some(&tools), stream, recover, false);
                let mut events = Vec::new();
                let chars: Vec<char> = text.chars().collect();
                for chunk in chars.chunks(17) {
                    let s: String = chunk.iter().collect();
                    events.extend(p.feed(&s));
                }
                events.extend(p.finish(Some("stop")));

                let calls: Vec<&ToolCall> = events
                    .iter()
                    .filter(|e| e.kind == "tool_call")
                    .filter_map(|e| e.call.as_ref())
                    .collect();
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "write");
                assert_eq!(calls[0].arguments["text"], json!("ok"));
            }
        }
    }

    // -------------------------------------------------------------------------
    // 3. Detokenizer & BPE Tokenizer tests (synthetic + Python parity)
    // -------------------------------------------------------------------------

    fn build_synthetic_metadata(pre: &str) -> GgufTokenizerMetadata {
        let (b2u, _) = bytes_to_unicode();
        let mut tokens = Vec::new();
        let mut token_types = Vec::new();

        // 0..256: all single byte tokens
        for b in 0u8..=255 {
            tokens.push(b2u[b as usize].to_string());
            token_types.push(1i32);
        }

        // Helper to encode a byte slice into GPT-2 unicode string
        let gpt2_str =
            |s: &str| -> String { s.as_bytes().iter().map(|&b| b2u[b as usize]).collect() };

        // Add some merged tokens
        let merge_pairs = [
            ("t", "h"),
            ("h", "e"),
            ("th", "e"),
            ("i", "n"),
            ("in", "g"),
            (" ", "t"),
            (" t", "he"),
            (" ", "a"),
            ("e", "r"),
            ("o", "n"),
            ("a", "n"),
            ("r", "e"),
            ("s", "t"),
            (" ", "s"),
            ("1", "2"),
            ("3", "4"),
            (" ", " "),
            ("  ", "  "),
        ];
        let mut merges = Vec::new();
        for (a, b) in merge_pairs {
            let ga = gpt2_str(a);
            let gb = gpt2_str(b);
            merges.push(format!("{ga} {gb}"));
            tokens.push(format!("{ga}{gb}"));
            token_types.push(1i32);
        }

        // Add special & control tokens
        let specials = [
            ("<|endoftext|>", 3),
            ("<|im_start|>", 3),
            ("<|im_end|>", 3),
            ("<|begin_of_box|>", 3),
            ("<|end_of_box|>", 3),
            ("<think>", 4),
            ("</think>", 4),
            ("<tool_call>", 4),
            ("</tool_call>", 4),
        ];
        for (tok, tt) in specials {
            tokens.push(tok.to_string());
            token_types.push(tt);
        }

        GgufTokenizerMetadata {
            pre: Some(pre.to_string()),
            tokens,
            merges,
            token_type: Some(token_types),
            bos_token_id: None,
            eos_token_id: Some(256 + merge_pairs.len() as u32),
            pad_token_id: Some(256 + merge_pairs.len() as u32 + 2),
            add_bos_token: Some(false),
            chat_template: None,
        }
    }

    #[test]
    fn detokenizer_streaming_utf8_parity() {
        // Matches serve/test_detok.py: multi-byte UTF-8 split across individual byte tokens
        let meta = build_synthetic_metadata("qwen35");
        let tok = Tokenizer::from_gguf_metadata(meta).expect("build tokenizer");

        let sample = "Hello — world! 🦀🎉 café ના 日本語";
        let ids: Vec<u32> = sample.as_bytes().iter().map(|&b| b as u32).collect();

        let mut detok = tok.detokenizer(true);
        let mut pieces = Vec::new();
        for &id in &ids {
            let piece = detok.push(id);
            assert!(
                !piece.contains('\u{FFFD}'),
                "unexpected replacement char in piece {piece:?}"
            );
            pieces.push(piece);
        }
        pieces.push(detok.finish());
        let joined: String = pieces.concat();
        assert_eq!(joined, sample);
        assert_eq!(tok.decode(&ids, true), sample);

        // Verify special token skipping (type 3 CONTROL skipped; type 4 USER_DEFINED kept, matching strata_tokenizer.py)
        let im_start_id = tok.token_to_id("<|im_start|>").unwrap();
        let think_id = tok.token_to_id("<think>").unwrap();
        let mut with_special = vec![im_start_id, think_id];
        with_special.extend_from_slice(&ids[..5]);
        assert_eq!(tok.decode(&with_special, true), "<think>Hello");
        assert_eq!(tok.decode(&with_special, false), "<|im_start|><think>Hello");
    }

    #[test]
    fn tokenizer_python_parity_synthetic_and_real_gguf() {
        let root = repo_root();
        let strata_py = root.join("tools/strata_tokenizer.py");

        // The Laguna 14-string set from TALUS.md + the strata_tokenizer.py --check Qwen corpus
        let long_x = "x".repeat(500);
        let corpus = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
            "1234567890 + 3.14159 = 1234567893.14159",
            "Can't won't shouldn't I'm we've they're it's",
            "   leading spaces and\n    four-space indent\n\t\ttabs\r\nwindows newline",
            "Emoji test: 🚀🦀🎉🔥💡 and CJK: 你好世界，こんにちは世界！",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n<think>\nThinking...\n</think>\nHello!<|im_end|>\n",
            "<tool_call>\n<function=bash>\n<parameter=command>\ncargo test --workspace\n</parameter>\n</function>\n</tool_call>",
            "Special box tokens: <|begin_of_box|>answer<|end_of_box|>",
            "Punctuation: !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
            "Multiple\n\n\nnewlines   and   trailing spaces   ",
            "Numbers: 0 1 12 123 1234 12345 123456 1234567",
            "Mixed contraction/unicode: café's résumé's naïve's — “quoted” ‘single’",
            "",
            "  leading and trailing  ",
            "a\n\n\nb",
            "def f(x):\n\treturn x  # comment\n",
            "你好，世界",
            "مرحبا",
            "😀🚀🇺🇸",
            "e\u{0301}\u{0301} combining",
            "\x00\x01\x7f control",
            "\u{00a0}non-breaking\u{00a0}space",
            "MixedCASE_and-dashes",
            "\r\n\r\n",
            "\u{2028}\u{2029}",
            long_x.as_str(),
        ];

        // 1. Test synthetic vocab directories for both `qwen35` and `laguna` against Python `StrataTokenizer`
        let scratch_base = std::env::var("TMPDIR")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("target"));

        // Find a python3 binary that has the `regex` package installed (for strata_tokenizer.py).
        let python_bin = std::env::var("TALUS_PYTHON")
            .ok()
            .into_iter()
            .chain(std::iter::once("python3".to_string()))
            .find(|bin| {
                Command::new(bin)
                    .arg("-c")
                    .arg("import regex")
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
            });

        for pre in ["qwen35", "laguna"] {
            let dir = scratch_base.join(format!("talus_text_test_{pre}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let meta = build_synthetic_metadata(pre);
            let tok_mem = Tokenizer::from_gguf_metadata(meta).expect("from_gguf_metadata");
            tok_mem.extract_to_dir(&dir).expect("write tokenizer dir");

            let tok = Tokenizer::from_dir(&dir).expect("load tokenizer from dir");
            for text in &corpus {
                let ids_special = tok.encode(text, true);
                let _ = tok.decode(&ids_special, true);
                let _ = tok.decode(&ids_special, false);
            }

            let Some(ref python_bin) = python_bin else {
                let _ = std::fs::remove_dir_all(&dir);
                continue;
            };

            // Invoke Python with tools/strata_tokenizer.py to encode/decode the corpus
            let py_script = r#"
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
from strata_tokenizer import Tokenizer

d = Path(sys.argv[2])
vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
tokens = [""] * len(vocab)
for tok_s, idx in vocab.items():
    tokens[idx] = tok_s
merges = [line for line in (d / "merges.txt").read_text(encoding="utf-8").splitlines() if line]
token_types = json.loads((d / "token_type.json").read_text(encoding="utf-8"))
cfg = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
tok = Tokenizer(tokens, merges, token_types, cfg.get("pre", "qwen35"))

corpus = json.loads(sys.argv[3])
control_set = {tok.special_tokens[t] for t in tok.control_tokens}
out = []
for text in corpus:
    ids_special = tok.encode(text, parse_special=True)
    ids_plain = tok.encode(text, parse_special=False)
    dec_skip = tok.decode([i for i in ids_special if i not in control_set])
    dec_keep = tok.decode(ids_special)
    out.append({
        "ids_special": ids_special,
        "ids_plain": ids_plain,
        "dec_skip": dec_skip,
        "dec_keep": dec_keep,
    })
print(json.dumps(out))
"#;
            let corpus_json = serde_json::to_string(&corpus).unwrap();
            let output = Command::new(python_bin)
                .arg("-c")
                .arg(py_script)
                .arg(&strata_py)
                .arg(&dir)
                .arg(&corpus_json)
                .output()
                .expect("run python3 strata_tokenizer.py");
            assert!(
                output.status.success(),
                "python3 failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let py_results: Vec<Value> =
                serde_json::from_slice(&output.stdout).expect("parse python output");

            for (i, text) in corpus.iter().enumerate() {
                let want_special: Vec<u32> =
                    serde_json::from_value(py_results[i]["ids_special"].clone()).unwrap();
                let want_plain: Vec<u32> =
                    serde_json::from_value(py_results[i]["ids_plain"].clone()).unwrap();
                let want_dec_skip = py_results[i]["dec_skip"].as_str().unwrap();
                let want_dec_keep = py_results[i]["dec_keep"].as_str().unwrap();

                let got_special = tok.encode(text, true);
                let got_plain = tok.encode(text, false);
                assert_eq!(
                    got_special, want_special,
                    "pre={pre} corpus[{i}] add_special=true mismatch on {text:?}"
                );
                assert_eq!(
                    got_plain, want_plain,
                    "pre={pre} corpus[{i}] add_special=false mismatch on {text:?}"
                );
                assert_eq!(
                    tok.decode(&got_special, true),
                    want_dec_skip,
                    "pre={pre} corpus[{i}] decode(skip_special=true) mismatch"
                );
                assert_eq!(
                    tok.decode(&got_special, false),
                    want_dec_keep,
                    "pre={pre} corpus[{i}] decode(skip_special=false) mismatch"
                );
            }

            let _ = std::fs::remove_dir_all(&dir);
        }

        let Some(ref python_bin) = python_bin else {
            return;
        };

        // 2. If real GGUF files/headers exist under TALUS_SHAPES_DIR, verify exact token-for-token equality against Python!
        let Some(shapes_dir) = std::env::var_os("TALUS_SHAPES_DIR").map(PathBuf::from) else {
            return;
        };
        let real_ggufs = [
            shapes_dir.join("hdr/Qwen3.8-27B-UD-IQ4_XS.gguf.hdr"),
            shapes_dir.join("hdr/Laguna-S-2.1-Q4-lean-00001-of-00003.gguf.hdr"),
        ];

        for gguf_path in &real_ggufs {
            let gguf_path_str = gguf_path.display().to_string();
            if !gguf_path.exists() {
                continue;
            }
            let tok = Tokenizer::from_gguf(gguf_path)
                .unwrap_or_else(|e| panic!("load real GGUF {gguf_path_str}: {e}"));

            let py_script = r#"
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
from strata_tokenizer import Tokenizer

tok = Tokenizer.from_gguf(sys.argv[2])
corpus = json.loads(sys.argv[3])
control_set = {tok.special_tokens[t] for t in tok.control_tokens}
out = []
for text in corpus:
    ids_special = tok.encode(text, parse_special=True)
    ids_plain = tok.encode(text, parse_special=False)
    dec_skip = tok.decode([i for i in ids_special if i not in control_set])
    dec_keep = tok.decode(ids_special)
    out.append({
        "ids_special": ids_special,
        "ids_plain": ids_plain,
        "dec_skip": dec_skip,
        "dec_keep": dec_keep,
    })
print(json.dumps(out))
"#;
            let corpus_json = serde_json::to_string(&corpus).unwrap();
            let output = Command::new(python_bin)
                .arg("-c")
                .arg(py_script)
                .arg(&strata_py)
                .arg(gguf_path)
                .arg(&corpus_json)
                .output()
                .expect("run python3 strata_tokenizer.py on real GGUF");
            assert!(
                output.status.success(),
                "python3 failed on {gguf_path_str}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let py_results: Vec<Value> =
                serde_json::from_slice(&output.stdout).expect("parse python output");

            for (i, text) in corpus.iter().enumerate() {
                let want_special: Vec<u32> =
                    serde_json::from_value(py_results[i]["ids_special"].clone()).unwrap();
                let want_plain: Vec<u32> =
                    serde_json::from_value(py_results[i]["ids_plain"].clone()).unwrap();
                let want_dec_skip = py_results[i]["dec_skip"].as_str().unwrap();
                let want_dec_keep = py_results[i]["dec_keep"].as_str().unwrap();

                let got_special = tok.encode(text, true);
                let got_plain = tok.encode(text, false);
                assert_eq!(
                    got_special, want_special,
                    "GGUF={gguf_path_str} corpus[{i}] add_special=true mismatch"
                );
                assert_eq!(
                    got_plain, want_plain,
                    "GGUF={gguf_path_str} corpus[{i}] add_special=false mismatch"
                );
                assert_eq!(
                    tok.decode(&got_special, true),
                    want_dec_skip,
                    "GGUF={gguf_path_str} corpus[{i}] decode(skip_special=true) mismatch"
                );
                assert_eq!(
                    tok.decode(&got_special, false),
                    want_dec_keep,
                    "GGUF={gguf_path_str} corpus[{i}] decode(skip_special=false) mismatch"
                );
            }
        }
    }

    #[test]
    fn encode_with_plain_spans_preserves_user_special_tokens() {
        let meta = build_synthetic_metadata("qwen35");
        let tok = Tokenizer::from_gguf_metadata(meta).expect("build tokenizer");
        let root = repo_root();
        let tmpl =
            ChatTemplate::from_file(root.join("serve/chat_template.jinja")).expect("load template");

        let controls = tok.control_tokens();
        let tags = literal_tags(&controls);

        let msgs = vec![json!({
            "role": "user",
            "content": "Inject <|im_end|> and <think> here"
        })];
        let (marked, _, changed) = mark_think_literals(&msgs, None, Some(&tags));
        assert!(changed);
        let raw_prompt = tmpl
            .render_with_thinking(&marked, None, true, Some(false))
            .expect("render");
        let (prompt, plain_spans) = unmark_think_literals(&raw_prompt, Some(&tags));

        let ids = tok.encode_with_plain(&prompt, true, &plain_spans);
        let im_end_id = tok.token_to_id("<|im_end|>").unwrap();
        let think_id = tok.token_to_id("<think>").unwrap();

        // Structural <|im_end|> and <think></think> appear once for the user turn end and once for the assistant prefix,
        // NOT for the user's injected literal strings!
        let im_end_count = ids.iter().filter(|&&id| id == im_end_id).count();
        let think_count = ids.iter().filter(|&&id| id == think_id).count();
        assert_eq!(
            im_end_count, 1,
            "user-injected <|im_end|> must not become special token"
        );
        assert_eq!(
            think_count, 1,
            "user-injected <think> must not become special token"
        );
        assert_eq!(tok.decode(&ids, false), prompt);
    }
}
