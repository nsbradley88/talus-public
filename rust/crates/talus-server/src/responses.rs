//! Stateless OpenAI Responses API (`/v1/responses`) matching `serve/responses.py`.

use crate::frontend::{
    clean_tools, text_of_content, tool_choice_of, Event, EventKind, TemplateKwargs,
};
use crate::json::{
    base64_std_encode, base64_url_decode, next_hex_id, unix_timestamp_secs, JsonValue,
};
use std::collections::HashMap;

pub const ENCRYPTED_PREFIX: &str = "strata.r1:";
pub const HOSTED_TOOLS: &[&str] = &[
    "web_search",
    "web_search_preview",
    "file_search",
    "computer_use_preview",
    "computer_use",
    "code_interpreter",
    "image_generation",
    "local_shell",
    "mcp",
    "tool_search",
];

#[derive(Clone, Debug)]
pub struct ResponsesError {
    pub message: String,
    pub param: Option<String>,
    pub code: Option<String>,
    pub status: u16,
    pub error_type: String,
}

impl ResponsesError {
    pub fn invalid(message: impl Into<String>, param: Option<&str>, code: Option<&str>) -> Self {
        Self {
            message: message.into(),
            param: param.map(|s| s.to_string()),
            code: code.map(|s| s.to_string()),
            status: 400,
            error_type: "invalid_request_error".to_string(),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            param: Some("response_id".to_string()),
            code: Some("not_found".to_string()),
            status: 404,
            error_type: "invalid_request_error".to_string(),
        }
    }

    pub fn to_body(&self) -> JsonValue {
        responses_error_body(
            &self.message,
            &self.error_type,
            self.param.as_deref(),
            self.code.as_deref(),
        )
    }
}

pub fn responses_error_body(
    message: &str,
    error_type: &str,
    param: Option<&str>,
    code: Option<&str>,
) -> JsonValue {
    JsonValue::obj(vec![(
        "error",
        JsonValue::obj(vec![
            ("message", JsonValue::from(message)),
            ("type", JsonValue::from(error_type)),
            ("param", JsonValue::from(param)),
            ("code", JsonValue::from(code)),
        ]),
    )])
}

pub fn encode_reasoning(text: &str) -> String {
    let payload = JsonValue::obj(vec![("text", JsonValue::from(text))]).to_json_string();
    format!(
        "{ENCRYPTED_PREFIX}{}",
        base64_std_encode(payload.as_bytes())
    )
}

pub fn decode_reasoning(item: &JsonValue, param: &str) -> Result<String, ResponsesError> {
    if let Some(arr) = item.get("content").and_then(|v| v.as_array()) {
        let mut texts = Vec::new();
        for p in arr {
            if matches!(p.get_str("type"), Some("reasoning_text" | "text")) {
                if let Some(t) = p.get_str("text") {
                    texts.push(t);
                }
            }
        }
        if !texts.is_empty() {
            return Ok(texts.join(""));
        }
    }
    let Some(enc) = item.get("encrypted_content") else {
        return summary_text(item.get("summary"));
    };
    if enc.is_null() {
        return summary_text(item.get("summary"));
    }
    let Some(s) = enc.as_str() else {
        return Err(ResponsesError::invalid(
            "reasoning.encrypted_content must be a string",
            Some(&format!("{param}.encrypted_content")),
            Some("invalid_encrypted_content"),
        ));
    };
    let Some(b64) = s.strip_prefix(ENCRYPTED_PREFIX) else {
        return Ok(String::new());
    };
    let bytes = base64_url_decode(b64).map_err(|_| {
        ResponsesError::invalid(
            "this reasoning item's encrypted_content is damaged",
            Some(&format!("{param}.encrypted_content")),
            Some("invalid_encrypted_content"),
        )
    })?;
    let utf8 = String::from_utf8(bytes).map_err(|_| {
        ResponsesError::invalid(
            "this reasoning item's encrypted_content is damaged",
            Some(&format!("{param}.encrypted_content")),
            Some("invalid_encrypted_content"),
        )
    })?;
    if let Ok(parsed) = JsonValue::parse(&utf8) {
        if let Some(t) = parsed.get_str("text") {
            return Ok(t.to_string());
        }
    }
    Err(ResponsesError::invalid(
        "this reasoning item's encrypted_content is damaged",
        Some(&format!("{param}.encrypted_content")),
        Some("invalid_encrypted_content"),
    ))
}

pub fn check_request(req: &JsonValue) -> Result<(), ResponsesError> {
    if !req.is_object() {
        return Err(ResponsesError::invalid(
            "the request body must be a JSON object",
            None,
            None,
        ));
    }
    if let Some(prev) = req.get("previous_response_id") {
        if !prev.is_null() && prev.as_str() != Some("") && prev.as_bool() != Some(false) {
            return Err(ResponsesError::invalid(
                "this server keeps no responses (stateless): send the whole conversation in input instead of previous_response_id",
                Some("previous_response_id"),
                Some("unsupported_parameter"),
            ));
        }
    }
    if let Some(conv) = req.get("conversation") {
        if !conv.is_null() && conv.as_str() != Some("") && conv.as_bool() != Some(false) {
            return Err(ResponsesError::invalid(
                "this server keeps no conversations (stateless): send the whole conversation in input",
                Some("conversation"),
                Some("unsupported_parameter"),
            ));
        }
    }
    if req.get_bool("background") == Some(true) {
        return Err(ResponsesError::invalid(
            "background responses are not supported: use stream or wait for the answer",
            Some("background"),
            Some("unsupported_parameter"),
        ));
    }
    if let Some(p) = req.get("prompt") {
        if !p.is_null() {
            return Err(ResponsesError::invalid(
                "stored prompt templates are not supported: send instructions and input",
                Some("prompt"),
                Some("unsupported_parameter"),
            ));
        }
    }
    if let Some(mo) = req.get("max_output_tokens") {
        if !mo.is_null() {
            match mo.as_i64() {
                Some(n) if n >= 1 => {}
                _ => {
                    return Err(ResponsesError::invalid(
                        "max_output_tokens must be a positive whole number",
                        Some("max_output_tokens"),
                        None,
                    ));
                }
            }
        }
    }
    if let Some(r) = req.get("reasoning") {
        if !r.is_null() && !r.is_object() {
            return Err(ResponsesError::invalid(
                "reasoning must be an object",
                Some("reasoning"),
                None,
            ));
        }
    }
    Ok(())
}

fn summary_text(summary: Option<&JsonValue>) -> Result<String, ResponsesError> {
    let Some(v) = summary else {
        return Ok(String::new());
    };
    if v.is_null() {
        return Ok(String::new());
    }
    if let Some(s) = v.as_str() {
        return Ok(s.to_string());
    }
    let Some(arr) = v.as_array() else {
        return Err(ResponsesError::invalid(
            "reasoning.summary must be a list",
            Some("input"),
            None,
        ));
    };
    let mut parts = Vec::new();
    for item in arr {
        if let Some(s) = item.as_str() {
            parts.push(s.to_string());
        } else if item.is_object() {
            if let Some(t) = item.get_str("text") {
                parts.push(t.to_string());
            }
        }
    }
    Ok(parts.join("\n"))
}

pub fn input_messages(req: &JsonValue) -> Result<Vec<JsonValue>, ResponsesError> {
    let mut out: Vec<JsonValue> = Vec::new();
    if let Some(inst) = req.get("instructions") {
        if !inst.is_null() {
            let Some(s) = inst.as_str() else {
                return Err(ResponsesError::invalid(
                    "instructions must be a string",
                    Some("instructions"),
                    None,
                ));
            };
            if !s.trim().is_empty() {
                out.push(JsonValue::obj(vec![
                    ("role", JsonValue::from("system")),
                    ("content", JsonValue::from(s)),
                ]));
            }
        }
    }

    let raw = req.get("input");
    let items: Vec<JsonValue> = match raw {
        None | Some(JsonValue::Null) => {
            return Err(ResponsesError::invalid(
                "input is required: a string or an array of input items",
                Some("input"),
                Some("missing_required_parameter"),
            ));
        }
        Some(JsonValue::Str(s)) => {
            vec![JsonValue::obj(vec![
                ("type", JsonValue::from("message")),
                ("role", JsonValue::from("user")),
                ("content", JsonValue::from(s.as_str())),
            ])]
        }
        Some(JsonValue::Array(arr)) => arr.clone(),
        _ => {
            return Err(ResponsesError::invalid(
                "input must be a string or an array of input items",
                Some("input"),
                None,
            ));
        }
    };

    let mut sys_parts: Vec<String> = out
        .drain(..)
        .filter_map(|m| m.get_str("content").map(|s| s.to_string()))
        .collect();
    let mut rest: Vec<JsonValue> = Vec::new();
    let mut pending_reasoning: Vec<String> = Vec::new();
    let mut pending_calls: Vec<JsonValue> = Vec::new();

    let flush_pending = |rest: &mut Vec<JsonValue>,
                         pending_reasoning: &mut Vec<String>,
                         pending_calls: &mut Vec<JsonValue>| {
        if pending_reasoning.is_empty() && pending_calls.is_empty() {
            return;
        }
        let mut msg = JsonValue::obj(vec![
            ("role", JsonValue::from("assistant")),
            ("content", JsonValue::from("")),
        ]);
        if !pending_reasoning.is_empty() {
            msg.insert(
                "reasoning_content",
                JsonValue::from(pending_reasoning.join("\n")),
            );
            pending_reasoning.clear();
        }
        if !pending_calls.is_empty() {
            msg.insert(
                "tool_calls",
                JsonValue::Array(std::mem::take(pending_calls)),
            );
        }
        rest.push(msg);
    };

    for (idx, item) in items.iter().enumerate() {
        let param = format!("input[{idx}]");
        if !item.is_object() {
            return Err(ResponsesError::invalid(
                "expected an input item object",
                Some(&param),
                None,
            ));
        }
        let kind = item
            .get_str("type")
            .or_else(|| {
                if item.get("role").is_some() {
                    Some("message")
                } else {
                    None
                }
            })
            .unwrap_or("");
        match kind {
            "reasoning" => {
                let text = decode_reasoning(item, &param)?;
                if !text.is_empty() {
                    if !pending_calls.is_empty() {
                        flush_pending(&mut rest, &mut pending_reasoning, &mut pending_calls);
                    }
                    pending_reasoning.push(text);
                }
            }
            "function_call" => {
                let name = item
                    .get_str("name")
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        ResponsesError::invalid(
                            "function_call item is missing `name`",
                            Some("input"),
                            None,
                        )
                    })?;
                let args = match item.get("arguments") {
                    None | Some(JsonValue::Null) => JsonValue::empty_obj(),
                    Some(JsonValue::Str(s)) => {
                        if s.trim().is_empty() {
                            JsonValue::empty_obj()
                        } else {
                            JsonValue::parse(s).map_err(|_| {
                                ResponsesError::invalid(
                                    format!(
                                        "function_call {name:?}: `arguments` is not valid JSON"
                                    ),
                                    Some("input"),
                                    None,
                                )
                            })?
                        }
                    }
                    Some(v) if v.is_object() => v.clone(),
                    _ => {
                        return Err(ResponsesError::invalid(
                            format!("function_call {name:?}: `arguments` must be a JSON string or object"),
                            Some("input"),
                            None,
                        ));
                    }
                };
                pending_calls.push(JsonValue::obj(vec![
                    ("name", JsonValue::from(name)),
                    ("arguments", args),
                ]));
            }
            "custom_tool_call" => {
                let name = item
                    .get_str("name")
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        ResponsesError::invalid(
                            "custom_tool_call item is missing `name`",
                            Some("input"),
                            None,
                        )
                    })?;
                let inp = item.get_str("input").unwrap_or("");
                pending_calls.push(JsonValue::obj(vec![
                    ("name", JsonValue::from(name)),
                    (
                        "arguments",
                        JsonValue::obj(vec![("input", JsonValue::from(inp))]),
                    ),
                ]));
            }
            "function_call_output" | "custom_tool_call_output" => {
                flush_pending(&mut rest, &mut pending_reasoning, &mut pending_calls);
                let out_val = item.get("output");
                let text = match out_val {
                    None | Some(JsonValue::Null) => String::new(),
                    Some(JsonValue::Str(s)) => s.clone(),
                    Some(JsonValue::Array(_)) => text_of_content(out_val)
                        .map_err(|e| ResponsesError::invalid(e, Some("input"), None))?,
                    Some(other) => other.to_json_string(),
                };
                rest.push(JsonValue::obj(vec![
                    ("role", JsonValue::from("tool")),
                    ("content", JsonValue::from(text)),
                ]));
            }
            "item_reference" => {
                return Err(ResponsesError::invalid(
                    "item_reference is not supported: this server does not store items; send the item itself in `input`",
                    Some("input"),
                    None,
                ));
            }
            "compaction" | "compaction_summary" => {}
            "message" => {
                let role = item.get_str("role").ok_or_else(|| {
                    ResponsesError::invalid(
                        "message item in `input` is missing `role`",
                        Some("input"),
                        None,
                    )
                })?;
                if role != "assistant" || !pending_calls.is_empty() {
                    flush_pending(&mut rest, &mut pending_reasoning, &mut pending_calls);
                }
                let text = text_of_content(item.get("content"))
                    .map_err(|e| ResponsesError::invalid(e, Some("input"), None))?;
                if role == "system" || role == "developer" {
                    if !text.trim().is_empty() {
                        if rest.is_empty() {
                            sys_parts.push(text);
                        } else {
                            rest.push(JsonValue::obj(vec![
                                ("role", JsonValue::from("user")),
                                ("content", JsonValue::from(text)),
                            ]));
                        }
                    }
                } else if role == "assistant" {
                    let mut msg = JsonValue::obj(vec![
                        ("role", JsonValue::from("assistant")),
                        ("content", JsonValue::from(text)),
                    ]);
                    if !pending_reasoning.is_empty() {
                        msg.insert(
                            "reasoning_content",
                            JsonValue::from(pending_reasoning.join("\n")),
                        );
                        pending_reasoning.clear();
                    }
                    rest.push(msg);
                } else if role == "user" {
                    rest.push(JsonValue::obj(vec![
                        ("role", JsonValue::from("user")),
                        ("content", JsonValue::from(text)),
                    ]));
                } else {
                    return Err(ResponsesError::invalid(
                        format!("unsupported message role {role:?} in `input`"),
                        Some("input"),
                        None,
                    ));
                }
            }
            other => {
                return Err(ResponsesError::invalid(
                    format!("unsupported input item type {other:?}"),
                    Some("input"),
                    None,
                ));
            }
        }
    }

    flush_pending(&mut rest, &mut pending_reasoning, &mut pending_calls);

    let mut final_msgs = Vec::new();
    if !sys_parts.is_empty() {
        final_msgs.push(JsonValue::obj(vec![
            ("role", JsonValue::from("system")),
            ("content", JsonValue::from(sys_parts.join("\n\n"))),
        ]));
    }
    final_msgs.extend(rest);
    if final_msgs.is_empty() {
        return Err(ResponsesError::invalid(
            "input produced no messages",
            Some("input"),
            None,
        ));
    }
    Ok(final_msgs)
}

pub type RequestToolsResult = (Option<Vec<JsonValue>>, HashMap<String, JsonValue>);

pub fn request_tools(req: &JsonValue) -> Result<RequestToolsResult, ResponsesError> {
    let (choice_mode, chosen_name) = tool_choice_of(req.get("tool_choice"))
        .map_err(|e| ResponsesError::invalid(e, Some("tool_choice"), None))?;
    if choice_mode == "none" {
        return Ok((None, HashMap::new()));
    }
    let raw = match req.get("tools") {
        None | Some(JsonValue::Null) => return Ok((None, HashMap::new())),
        Some(v) => v,
    };
    let Some(arr) = raw.as_array() else {
        return Err(ResponsesError::invalid(
            "tools must be a list",
            Some("tools"),
            None,
        ));
    };

    let mut fn_tools = Vec::new();
    let mut by_name = HashMap::new();
    for t in arr {
        if !t.is_object() {
            return Err(ResponsesError::invalid(
                "each tool must be an object",
                Some("tools"),
                None,
            ));
        }
        let kind = t.get_str("type").unwrap_or("function");
        if kind == "custom" {
            let name = t.get_str("name").filter(|s| !s.is_empty()).ok_or_else(|| {
                ResponsesError::invalid(
                    "custom tool is missing a non-empty `name`",
                    Some("tools"),
                    None,
                )
            })?;
            let desc = t.get_str("description").unwrap_or("");
            let converted = JsonValue::obj(vec![
                ("name", JsonValue::from(name)),
                ("description", JsonValue::from(desc)),
                (
                    "parameters",
                    JsonValue::obj(vec![
                        ("type", JsonValue::from("object")),
                        (
                            "properties",
                            JsonValue::obj(vec![(
                                "input",
                                JsonValue::obj(vec![("type", JsonValue::from("string"))]),
                            )]),
                        ),
                        ("required", JsonValue::Array(vec![JsonValue::from("input")])),
                    ]),
                ),
            ]);
            by_name.insert(name.to_string(), t.clone());
            fn_tools.push(converted);
        } else if kind == "function" {
            let inner = t.get("function").unwrap_or(t);
            let name = inner
                .get_str("name")
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    ResponsesError::invalid(
                        "function tool is missing a non-empty `name`",
                        Some("tools"),
                        None,
                    )
                })?;
            by_name.insert(name.to_string(), t.clone());
            fn_tools.push(t.clone());
        }
    }

    let cleaned = clean_tools(Some(&JsonValue::Array(fn_tools)))
        .map_err(|e| ResponsesError::invalid(e, Some("tools"), None))?;
    let Some(mut tools) = cleaned else {
        if choice_mode == "required" || choice_mode == "function" {
            return Err(ResponsesError::invalid(
                "tool_choice requires a tool, but no function tools were given",
                Some("tool_choice"),
                None,
            ));
        }
        return Ok((None, by_name));
    };

    if choice_mode == "function" {
        let want = chosen_name.unwrap();
        tools.retain(|t| t.get_str("name") == Some(want.as_str()));
        if tools.is_empty() {
            return Err(ResponsesError::invalid(
                format!("tool_choice names unknown tool {want:?}"),
                Some("tool_choice"),
                None,
            ));
        }
    }
    Ok((Some(tools), by_name))
}

pub fn template_kwargs(
    req: &JsonValue,
    default_effort: Option<&str>,
) -> Result<TemplateKwargs, ResponsesError> {
    let mut kw = TemplateKwargs {
        enable_thinking: None,
        reasoning_effort: default_effort.map(|s| s.to_string()),
        preserve_thinking: None,
        add_generation_prompt: true,
    };
    if let Some(r) = req.get("reasoning") {
        if !r.is_null() {
            if !r.is_object() {
                return Err(ResponsesError::invalid(
                    "reasoning must be an object",
                    Some("reasoning"),
                    None,
                ));
            }
            if let Some(eff) = r.get_str("effort") {
                match eff {
                    "none" | "minimal" | "off" | "disabled" => {
                        kw.enable_thinking = Some(false);
                        kw.reasoning_effort = None;
                    }
                    "high" | "xhigh" => kw.reasoning_effort = Some("xhigh".to_string()),
                    "medium" | "low" => kw.reasoning_effort = Some(eff.to_string()),
                    other => {
                        return Err(ResponsesError::invalid(
                            format!("reasoning.effort: unsupported value {other:?} (expected low, medium, high, xhigh, or none)"),
                            Some("reasoning.effort"),
                            None,
                        ));
                    }
                }
            }
        }
    }
    Ok(kw)
}

/// Stateful assembler that turns `OutputParser` events into OpenAI Responses SSE events and final `response` JSON.
pub struct Assembler {
    pub id: String,
    pub created_at: i64,
    pub model: String,
    req: JsonValue,
    prompt_tokens: usize,
    tools_by_name: HashMap<String, JsonValue>,
    want_encrypted: bool,
    seq: usize,
    pub output: Vec<JsonValue>,
    open_kind: Option<String>,
    open_item: Option<JsonValue>,
    text_parts: Vec<String>,
}

impl Assembler {
    pub fn new(
        req: &JsonValue,
        model: String,
        prompt_tokens: usize,
        tools_by_name: HashMap<String, JsonValue>,
        want_encrypted: bool,
    ) -> Self {
        Self {
            id: format!("resp_{}", next_hex_id(24)),
            created_at: unix_timestamp_secs(),
            model,
            req: req.clone(),
            prompt_tokens,
            tools_by_name,
            want_encrypted,
            seq: 0,
            output: Vec::new(),
            open_kind: None,
            open_item: None,
            text_parts: Vec::new(),
        }
    }

    fn emit(&mut self, event_type: &str, mut fields: Vec<(&str, JsonValue)>) -> JsonValue {
        let s = self.seq;
        self.seq += 1;
        let mut entries = Vec::with_capacity(fields.len() + 2);
        entries.push(("type", JsonValue::from(event_type)));
        entries.append(&mut fields);
        entries.push(("sequence_number", JsonValue::from(s)));
        JsonValue::obj(entries)
    }

    pub fn response_object(
        &self,
        status: &str,
        usage: Option<JsonValue>,
        incomplete_details: Option<JsonValue>,
        error: Option<JsonValue>,
    ) -> JsonValue {
        let r = &self.req;
        let reasoning_cfg = r
            .get("reasoning")
            .filter(|v| v.is_object())
            .cloned()
            .unwrap_or_else(|| {
                JsonValue::obj(vec![
                    ("effort", JsonValue::Null),
                    ("summary", JsonValue::Null),
                ])
            });
        let text_cfg = r
            .get("text")
            .filter(|v| v.is_object())
            .cloned()
            .unwrap_or_else(|| {
                JsonValue::obj(vec![(
                    "format",
                    JsonValue::obj(vec![("type", JsonValue::from("text"))]),
                )])
            });
        let metadata = r
            .get("metadata")
            .filter(|v| v.is_object())
            .cloned()
            .unwrap_or_else(JsonValue::empty_obj);
        let tools = r
            .get("tools")
            .filter(|v| v.is_array())
            .cloned()
            .unwrap_or_else(JsonValue::empty_arr);

        JsonValue::obj(vec![
            ("id", JsonValue::from(self.id.as_str())),
            ("object", JsonValue::from("response")),
            ("created_at", JsonValue::from(self.created_at)),
            ("status", JsonValue::from(status)),
            ("background", JsonValue::from(false)),
            ("error", error.unwrap_or(JsonValue::Null)),
            (
                "incomplete_details",
                incomplete_details.unwrap_or(JsonValue::Null),
            ),
            (
                "instructions",
                r.get("instructions").cloned().unwrap_or(JsonValue::Null),
            ),
            (
                "max_output_tokens",
                r.get("max_output_tokens")
                    .cloned()
                    .unwrap_or(JsonValue::Null),
            ),
            ("model", JsonValue::from(self.model.as_str())),
            ("output", JsonValue::Array(self.output.clone())),
            (
                "parallel_tool_calls",
                JsonValue::from(r.get_bool("parallel_tool_calls").unwrap_or(true)),
            ),
            ("previous_response_id", JsonValue::Null),
            ("reasoning", reasoning_cfg),
            ("store", JsonValue::from(false)),
            (
                "temperature",
                JsonValue::from(r.get_f64("temperature").unwrap_or(1.0)),
            ),
            ("text", text_cfg),
            (
                "tool_choice",
                r.get("tool_choice")
                    .cloned()
                    .unwrap_or_else(|| JsonValue::from("auto")),
            ),
            ("tools", tools),
            ("top_p", JsonValue::from(r.get_f64("top_p").unwrap_or(1.0))),
            (
                "truncation",
                JsonValue::from(r.get_str("truncation").unwrap_or("disabled")),
            ),
            ("usage", usage.unwrap_or(JsonValue::Null)),
            ("user", r.get("user").cloned().unwrap_or(JsonValue::Null)),
            ("metadata", metadata),
        ])
    }

    pub fn start(&mut self) -> Vec<JsonValue> {
        let r1 = self.response_object("in_progress", None, None, None);
        let r2 = self.response_object("in_progress", None, None, None);
        vec![
            self.emit("response.created", vec![("response", r1)]),
            self.emit("response.in_progress", vec![("response", r2)]),
        ]
    }

    fn close_open(&mut self, item_status: &str) -> Vec<JsonValue> {
        let Some(kind) = self.open_kind.take() else {
            return Vec::new();
        };
        let Some(mut item) = self.open_item.take() else {
            return Vec::new();
        };
        let idx = self.output.len();
        let text = std::mem::take(&mut self.text_parts).join("");
        let item_id = item.get_str("id").unwrap_or("").to_string();
        let mut evs = Vec::new();

        item.insert("status", JsonValue::from(item_status));
        if kind == "reasoning" {
            let part = JsonValue::obj(vec![
                ("type", JsonValue::from("reasoning_text")),
                ("text", JsonValue::from(text.as_str())),
            ]);
            item.insert("content", JsonValue::Array(vec![part]));
            if self.want_encrypted {
                item.insert(
                    "encrypted_content",
                    JsonValue::from(encode_reasoning(&text)),
                );
            }
            evs.push(self.emit(
                "response.reasoning_text.done",
                vec![
                    ("item_id", JsonValue::from(item_id.as_str())),
                    ("output_index", JsonValue::from(idx)),
                    ("content_index", JsonValue::from(0)),
                    ("text", JsonValue::from(text)),
                ],
            ));
        } else if kind == "message" {
            let part = JsonValue::obj(vec![
                ("type", JsonValue::from("output_text")),
                ("text", JsonValue::from(text.as_str())),
                ("annotations", JsonValue::empty_arr()),
                ("logprobs", JsonValue::empty_arr()),
            ]);
            item.insert("content", JsonValue::Array(vec![part.clone()]));
            evs.push(self.emit(
                "response.output_text.done",
                vec![
                    ("item_id", JsonValue::from(item_id.as_str())),
                    ("output_index", JsonValue::from(idx)),
                    ("content_index", JsonValue::from(0)),
                    ("text", JsonValue::from(text)),
                    ("logprobs", JsonValue::empty_arr()),
                ],
            ));
            evs.push(self.emit(
                "response.content_part.done",
                vec![
                    ("item_id", JsonValue::from(item_id.as_str())),
                    ("output_index", JsonValue::from(idx)),
                    ("content_index", JsonValue::from(0)),
                    ("part", part),
                ],
            ));
        }

        self.output.push(item.clone());
        evs.push(self.emit(
            "response.output_item.done",
            vec![("output_index", JsonValue::from(idx)), ("item", item)],
        ));
        evs
    }

    pub fn feed(&mut self, ev: &Event) -> Vec<JsonValue> {
        let mut out = Vec::new();
        match ev.kind {
            EventKind::Reasoning if !ev.text.is_empty() => {
                if self.open_kind.as_deref() != Some("reasoning") {
                    out.extend(self.close_open("completed"));
                    let idx = self.output.len();
                    let item_id = format!("rs_{}", next_hex_id(24));
                    let item = JsonValue::obj(vec![
                        ("id", JsonValue::from(item_id.as_str())),
                        ("type", JsonValue::from("reasoning")),
                        ("status", JsonValue::from("in_progress")),
                        ("summary", JsonValue::empty_arr()),
                        (
                            "content",
                            JsonValue::Array(vec![JsonValue::obj(vec![
                                ("type", JsonValue::from("reasoning_text")),
                                ("text", JsonValue::from("")),
                            ])]),
                        ),
                    ]);
                    self.open_kind = Some("reasoning".to_string());
                    self.open_item = Some(item.clone());
                    self.text_parts.clear();
                    out.push(self.emit(
                        "response.output_item.added",
                        vec![("output_index", JsonValue::from(idx)), ("item", item)],
                    ));
                }
                let idx = self.output.len();
                let item_id = self
                    .open_item
                    .as_ref()
                    .and_then(|i| i.get_str("id"))
                    .unwrap_or("")
                    .to_string();
                self.text_parts.push(ev.text.clone());
                out.push(self.emit(
                    "response.reasoning_text.delta",
                    vec![
                        ("item_id", JsonValue::from(item_id)),
                        ("output_index", JsonValue::from(idx)),
                        ("content_index", JsonValue::from(0)),
                        ("delta", JsonValue::from(ev.text.as_str())),
                    ],
                ));
            }
            EventKind::Content if !ev.text.is_empty() => {
                if self.open_kind.as_deref() != Some("message") {
                    out.extend(self.close_open("completed"));
                    let idx = self.output.len();
                    let item_id = format!("msg_{}", next_hex_id(24));
                    let item = JsonValue::obj(vec![
                        ("id", JsonValue::from(item_id.as_str())),
                        ("type", JsonValue::from("message")),
                        ("status", JsonValue::from("in_progress")),
                        ("role", JsonValue::from("assistant")),
                        ("content", JsonValue::empty_arr()),
                    ]);
                    self.open_kind = Some("message".to_string());
                    self.open_item = Some(item.clone());
                    self.text_parts.clear();
                    out.push(self.emit(
                        "response.output_item.added",
                        vec![("output_index", JsonValue::from(idx)), ("item", item)],
                    ));
                    out.push(self.emit(
                        "response.content_part.added",
                        vec![
                            ("item_id", JsonValue::from(item_id.as_str())),
                            ("output_index", JsonValue::from(idx)),
                            ("content_index", JsonValue::from(0)),
                            (
                                "part",
                                JsonValue::obj(vec![
                                    ("type", JsonValue::from("output_text")),
                                    ("text", JsonValue::from("")),
                                    ("annotations", JsonValue::empty_arr()),
                                    ("logprobs", JsonValue::empty_arr()),
                                ]),
                            ),
                        ],
                    ));
                }
                let idx = self.output.len();
                let item_id = self
                    .open_item
                    .as_ref()
                    .and_then(|i| i.get_str("id"))
                    .unwrap_or("")
                    .to_string();
                self.text_parts.push(ev.text.clone());
                out.push(self.emit(
                    "response.output_text.delta",
                    vec![
                        ("item_id", JsonValue::from(item_id)),
                        ("output_index", JsonValue::from(idx)),
                        ("content_index", JsonValue::from(0)),
                        ("delta", JsonValue::from(ev.text.as_str())),
                        ("logprobs", JsonValue::empty_arr()),
                    ],
                ));
            }
            EventKind::ToolCall => {
                let Some(ref call) = ev.call else {
                    return out;
                };
                out.extend(self.close_open("completed"));
                let idx = self.output.len();
                let tool_spec = self.tools_by_name.get(&call.name);
                let is_custom = tool_spec.and_then(|t| t.get_str("type")) == Some("custom");
                if is_custom {
                    let item_id = format!("ctc_{}", next_hex_id(24));
                    let inp = call
                        .arguments
                        .get_str("input")
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| call.arguments.to_json_string());
                    let added_item = JsonValue::obj(vec![
                        ("id", JsonValue::from(item_id.as_str())),
                        ("type", JsonValue::from("custom_tool_call")),
                        ("status", JsonValue::from("in_progress")),
                        ("call_id", JsonValue::from(call.id.as_str())),
                        ("name", JsonValue::from(call.name.as_str())),
                        ("input", JsonValue::from("")),
                    ]);
                    out.push(self.emit(
                        "response.output_item.added",
                        vec![("output_index", JsonValue::from(idx)), ("item", added_item)],
                    ));
                    if !inp.is_empty() {
                        out.push(self.emit(
                            "response.custom_tool_call_input.delta",
                            vec![
                                ("item_id", JsonValue::from(item_id.as_str())),
                                ("output_index", JsonValue::from(idx)),
                                ("delta", JsonValue::from(inp.as_str())),
                            ],
                        ));
                    }
                    out.push(self.emit(
                        "response.custom_tool_call_input.done",
                        vec![
                            ("item_id", JsonValue::from(item_id.as_str())),
                            ("output_index", JsonValue::from(idx)),
                            ("input", JsonValue::from(inp.as_str())),
                        ],
                    ));
                    let done_item = JsonValue::obj(vec![
                        ("id", JsonValue::from(item_id)),
                        ("type", JsonValue::from("custom_tool_call")),
                        ("status", JsonValue::from("completed")),
                        ("call_id", JsonValue::from(call.id.as_str())),
                        ("name", JsonValue::from(call.name.as_str())),
                        ("input", JsonValue::from(inp)),
                    ]);
                    self.output.push(done_item.clone());
                    out.push(self.emit(
                        "response.output_item.done",
                        vec![("output_index", JsonValue::from(idx)), ("item", done_item)],
                    ));
                } else {
                    let item_id = format!("fc_{}", next_hex_id(24));
                    let args_str = call.arguments.to_json_string();
                    let added_item = JsonValue::obj(vec![
                        ("id", JsonValue::from(item_id.as_str())),
                        ("type", JsonValue::from("function_call")),
                        ("status", JsonValue::from("in_progress")),
                        ("call_id", JsonValue::from(call.id.as_str())),
                        ("name", JsonValue::from(call.name.as_str())),
                        ("arguments", JsonValue::from("")),
                    ]);
                    out.push(self.emit(
                        "response.output_item.added",
                        vec![("output_index", JsonValue::from(idx)), ("item", added_item)],
                    ));
                    if !args_str.is_empty() {
                        out.push(self.emit(
                            "response.function_call_arguments.delta",
                            vec![
                                ("item_id", JsonValue::from(item_id.as_str())),
                                ("output_index", JsonValue::from(idx)),
                                ("delta", JsonValue::from(args_str.as_str())),
                            ],
                        ));
                    }
                    out.push(self.emit(
                        "response.function_call_arguments.done",
                        vec![
                            ("item_id", JsonValue::from(item_id.as_str())),
                            ("output_index", JsonValue::from(idx)),
                            ("name", JsonValue::from(call.name.as_str())),
                            ("arguments", JsonValue::from(args_str.as_str())),
                        ],
                    ));
                    let done_item = JsonValue::obj(vec![
                        ("id", JsonValue::from(item_id)),
                        ("type", JsonValue::from("function_call")),
                        ("status", JsonValue::from("completed")),
                        ("call_id", JsonValue::from(call.id.as_str())),
                        ("name", JsonValue::from(call.name.as_str())),
                        ("arguments", JsonValue::from(args_str)),
                    ]);
                    self.output.push(done_item.clone());
                    out.push(self.emit(
                        "response.output_item.done",
                        vec![("output_index", JsonValue::from(idx)), ("item", done_item)],
                    ));
                }
            }
            _ => {}
        }
        out
    }

    pub fn finish(
        &mut self,
        finish_reason: &str,
        completion_tokens: usize,
        reused: usize,
        reasoning_tokens: usize,
    ) -> Vec<JsonValue> {
        let item_status = if finish_reason == "length" {
            "incomplete"
        } else {
            "completed"
        };
        let mut out = self.close_open(item_status);
        let usage = JsonValue::obj(vec![
            ("input_tokens", JsonValue::from(self.prompt_tokens)),
            (
                "input_tokens_details",
                JsonValue::obj(vec![(
                    "cached_tokens",
                    JsonValue::from(reused.min(self.prompt_tokens)),
                )]),
            ),
            ("output_tokens", JsonValue::from(completion_tokens)),
            (
                "output_tokens_details",
                JsonValue::obj(vec![(
                    "reasoning_tokens",
                    JsonValue::from(reasoning_tokens),
                )]),
            ),
            (
                "total_tokens",
                JsonValue::from(self.prompt_tokens + completion_tokens),
            ),
        ]);
        if finish_reason == "length" {
            let inc = JsonValue::obj(vec![("reason", JsonValue::from("max_output_tokens"))]);
            let resp = self.response_object("incomplete", Some(usage), Some(inc), None);
            out.push(self.emit("response.incomplete", vec![("response", resp)]));
        } else {
            let mut resp = self.response_object("completed", Some(usage), None, None);
            resp.insert("completed_at", JsonValue::from(unix_timestamp_secs()));
            out.push(self.emit("response.completed", vec![("response", resp)]));
        }
        out
    }
}

pub fn collect_responses(events: &[JsonValue]) -> Option<JsonValue> {
    let mut last = None;
    for ev in events {
        let t = ev.get_str("type").unwrap_or("");
        if matches!(
            t,
            "response.completed" | "response.incomplete" | "response.failed"
        ) {
            last = ev.get("response").cloned();
        }
    }
    last
}
