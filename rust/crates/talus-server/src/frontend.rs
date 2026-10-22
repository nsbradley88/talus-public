//! Chat template rendering, OpenAI & Anthropic request normalization, and streaming `OutputParser`
//! (`<think>` reasoning, Qwen XML `<function=...><parameter=...>`, JSON tool calls, and GLM `<arg_key>...<arg_value>`).

use crate::json::{next_hex_id, JsonValue};

pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
pub const TOOL_OPEN: &str = "<tool_call>";
pub const TOOL_CLOSE: &str = "</tool_call>";
pub const FUNC_OPEN: &str = "<function=";
pub const FUNC_CLOSE: &str = "</function>";
pub const ARG_KEY_OPEN: &str = "<arg_key>";
pub const ARG_KEY_CLOSE: &str = "</arg_key>";
pub const ARG_VAL_OPEN: &str = "<arg_value>";
pub const ARG_VAL_CLOSE: &str = "</arg_value>";

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: JsonValue,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EventKind {
    Reasoning,
    Content,
    ToolStart,
    ToolArgs,
    ToolCall,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub kind: EventKind,
    pub text: String,
    pub call: Option<ToolCall>,
}

impl Event {
    pub fn reasoning(text: impl Into<String>) -> Self {
        Self {
            kind: EventKind::Reasoning,
            text: text.into(),
            call: None,
        }
    }

    pub fn content(text: impl Into<String>) -> Self {
        Self {
            kind: EventKind::Content,
            text: text.into(),
            call: None,
        }
    }

    pub fn tool_start(call: ToolCall) -> Self {
        Self {
            kind: EventKind::ToolStart,
            text: String::new(),
            call: Some(call),
        }
    }

    pub fn tool_args(text: impl Into<String>, call: ToolCall) -> Self {
        Self {
            kind: EventKind::ToolArgs,
            text: text.into(),
            call: Some(call),
        }
    }

    pub fn tool_call(call: ToolCall) -> Self {
        Self {
            kind: EventKind::ToolCall,
            text: String::new(),
            call: Some(call),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TemplateKwargs {
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub preserve_thinking: Option<bool>,
    pub add_generation_prompt: bool,
}

impl TemplateKwargs {
    pub fn thinking_enabled(&self) -> bool {
        self.enable_thinking != Some(false)
    }
}

/// Extract text from a content field (string, array of content parts, or null).
pub fn text_of_content(content: Option<&JsonValue>) -> Result<String, String> {
    match content {
        None | Some(JsonValue::Null) => Ok(String::new()),
        Some(JsonValue::Str(s)) => Ok(s.clone()),
        Some(JsonValue::Array(parts)) => {
            let mut out = Vec::new();
            for p in parts {
                match p {
                    JsonValue::Str(s) => out.push(s.clone()),
                    JsonValue::Object(_) => {
                        let t = p.get_str("type").unwrap_or("text");
                        if t == "text"
                            || t == "input_text"
                            || t == "output_text"
                            || p.contains_key("text")
                        {
                            let txt = p.get_str("text").ok_or_else(|| {
                                "content part is missing string 'text'".to_string()
                            })?;
                            out.push(txt.to_string());
                        } else if t == "image"
                            || t == "image_url"
                            || p.contains_key("image")
                            || p.contains_key("image_url")
                        {
                            out.push("<|vision_start|><|image_pad|><|vision_end|>".to_string());
                        }
                    }
                    _ => return Err("content part must be an object or string".to_string()),
                }
            }
            Ok(out.join(""))
        }
        _ => Err("message content must be a string, array, or null".to_string()),
    }
}

/// Normalize a tools list (OpenAI or Anthropic style) into the template's `[{"name", "description", "parameters"}]`.
pub fn clean_tools(raw_tools: Option<&JsonValue>) -> Result<Option<Vec<JsonValue>>, String> {
    let tools_val = match raw_tools {
        None | Some(JsonValue::Null) => return Ok(None),
        Some(JsonValue::Str(s)) => {
            if s.trim().is_empty() {
                return Ok(None);
            }
            let parsed = JsonValue::parse(s)
                .map_err(|_| "tools: must be a list of tool definitions".to_string())?;
            return clean_tools(Some(&parsed));
        }
        Some(v) => v,
    };
    let arr = tools_val
        .as_array()
        .ok_or_else(|| "tools: must be a list of tool definitions".to_string())?;
    if arr.is_empty() {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(arr.len());
    for t in arr {
        let obj = if let Some(inner) = t.get("function") {
            if inner.is_object() {
                inner
            } else {
                t
            }
        } else {
            t
        };
        if !obj.is_object() {
            return Err("tool definition must be an object".to_string());
        }
        let name = obj
            .get_str("name")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "tool definition is missing a non-empty 'name'".to_string())?;
        let desc = obj.get_str("description").unwrap_or("");
        let params_raw = obj.get("parameters").or_else(|| obj.get("input_schema"));
        let params = match params_raw {
            None | Some(JsonValue::Null) => JsonValue::empty_obj(),
            Some(JsonValue::Str(s)) => {
                if s.trim().is_empty() {
                    JsonValue::empty_obj()
                } else {
                    let p = JsonValue::parse(s).map_err(|_| {
                        format!("tool {name:?}: parameters must be a JSON Schema object")
                    })?;
                    if !p.is_object() {
                        return Err(format!(
                            "tool {name:?}: parameters must be a JSON Schema object"
                        ));
                    }
                    p
                }
            }
            Some(v) if v.is_object() => v.clone(),
            _ => {
                return Err(format!(
                    "tool {name:?}: parameters must be a JSON Schema object"
                ))
            }
        };
        out.push(JsonValue::obj(vec![
            ("name", JsonValue::from(name)),
            ("description", JsonValue::from(desc)),
            ("parameters", params),
        ]));
    }
    Ok(Some(out))
}

/// Normalize `tool_choice` into `("auto" | "none" | "required" | "function", Option<name>)`.
pub fn tool_choice_of(
    choice: Option<&JsonValue>,
) -> Result<(&'static str, Option<String>), String> {
    let c = match choice {
        None | Some(JsonValue::Null) => return Ok(("auto", None)),
        Some(v) => v,
    };
    if let Some(s) = c.as_str() {
        return match s {
            "auto" => Ok(("auto", None)),
            "none" => Ok(("none", None)),
            "required" | "any" => Ok(("required", None)),
            _ => Err(format!(
                "tool_choice: expected 'auto', 'none', or 'required', got {s:?}"
            )),
        };
    }
    if c.is_object() {
        let kind = c.get_str("type").unwrap_or("");
        match kind {
            "auto" => return Ok(("auto", None)),
            "none" => return Ok(("none", None)),
            "any" | "required" => return Ok(("required", None)),
            "function" | "tool" | "custom" => {
                let name = c
                    .get("function")
                    .and_then(|f| f.get_str("name"))
                    .or_else(|| c.get_str("name"))
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| "tool_choice is missing a tool name".to_string())?;
                return Ok(("function", Some(name.to_string())));
            }
            _ => {}
        }
    }
    Err("invalid tool_choice".to_string())
}

/// Compute forced tool call opening if `tool_choice` requires a specific tool or any tool.
pub fn forced_call(
    choice: Option<&JsonValue>,
    tools: Option<&[JsonValue]>,
) -> Result<Option<String>, String> {
    let (mode, name) = tool_choice_of(choice)?;
    match mode {
        "auto" | "none" => Ok(None),
        "required" => {
            let list = tools.filter(|t| !t.is_empty()).ok_or_else(|| {
                "tool_choice requires a tool call, but the request has no tools".to_string()
            })?;
            if list.len() == 1 {
                let n = list[0].get_str("name").unwrap_or("");
                Ok(Some(format!("{TOOL_OPEN}\n{FUNC_OPEN}{n}>\n")))
            } else {
                Ok(Some(format!("{TOOL_OPEN}\n{FUNC_OPEN}")))
            }
        }
        "function" => {
            let target = name.unwrap();
            let list = tools.unwrap_or(&[]);
            if !list
                .iter()
                .any(|t| t.get_str("name") == Some(target.as_str()))
            {
                return Err(format!("tool_choice names unknown tool {target:?}"));
            }
            Ok(Some(format!("{TOOL_OPEN}\n{FUNC_OPEN}{target}>\n")))
        }
        _ => Ok(None),
    }
}

pub type NormalizedMessages = (Vec<JsonValue>, Option<Vec<JsonValue>>, TemplateKwargs);

/// Convert OpenAI `/v1/chat/completions` request into normalized messages, tools, and template kwargs.
pub fn openai_to_messages(req: &JsonValue) -> Result<NormalizedMessages, String> {
    openai_to_messages_with_effort(req, None)
}

pub fn openai_to_messages_with_effort(
    req: &JsonValue,
    default_effort: Option<&str>,
) -> Result<NormalizedMessages, String> {
    if !req.is_object() {
        return Err("request body must be a JSON object".to_string());
    }
    let raw_msgs = req
        .get("messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "messages: expected a non-empty list of messages".to_string())?;
    if raw_msgs.is_empty() {
        return Err("messages: expected a non-empty list of messages".to_string());
    }

    let mut sys_parts: Vec<String> = Vec::new();
    let mut rest_msgs: Vec<JsonValue> = Vec::new();

    for m in raw_msgs {
        if !m.is_object() {
            return Err("each message must be a JSON object".to_string());
        }
        let role = m
            .get_str("role")
            .ok_or_else(|| "message is missing 'role'".to_string())?;
        if role == "system" || role == "developer" {
            let txt = text_of_content(m.get("content"))?;
            if !txt.trim().is_empty() {
                if rest_msgs.is_empty() {
                    sys_parts.push(txt);
                } else {
                    rest_msgs.push(JsonValue::obj(vec![
                        ("role", JsonValue::from("user")),
                        ("content", JsonValue::from(txt)),
                    ]));
                }
            }
            continue;
        }
        let content_txt = text_of_content(m.get("content"))?;
        let mut msg = JsonValue::obj(vec![
            ("role", JsonValue::from(role)),
            ("content", JsonValue::from(content_txt.as_str())),
        ]);
        if role == "assistant" {
            if let Some(rc) = m
                .get_str("reasoning_content")
                .or_else(|| m.get_str("reasoning"))
            {
                msg.insert("reasoning_content", JsonValue::from(rc));
            }
            if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()) {
                let mut norm_calls = Vec::new();
                for tc in tcs {
                    let fn_obj = tc.get("function").unwrap_or(tc);
                    let name = fn_obj
                        .get_str("name")
                        .ok_or_else(|| "tool_call is missing function.name".to_string())?;
                    let args_val = match fn_obj.get("arguments") {
                        None | Some(JsonValue::Null) => JsonValue::empty_obj(),
                        Some(JsonValue::Str(s)) => {
                            if s.trim().is_empty() {
                                JsonValue::empty_obj()
                            } else {
                                match JsonValue::parse(s) {
                                    Ok(v) if v.is_object() => v,
                                    _ => {
                                        JsonValue::obj(vec![("_raw", JsonValue::from(s.as_str()))])
                                    }
                                }
                            }
                        }
                        Some(v) if v.is_object() => v.clone(),
                        _ => {
                            return Err(
                                "tool_call arguments must be an object or JSON string".to_string()
                            )
                        }
                    };
                    let fn_entry = JsonValue::obj(vec![
                        ("name", JsonValue::from(name)),
                        ("arguments", args_val.clone()),
                    ]);
                    norm_calls.push(JsonValue::obj(vec![
                        ("name", JsonValue::from(name)),
                        ("arguments", args_val),
                        ("function", fn_entry),
                    ]));
                }
                if !norm_calls.is_empty() {
                    msg.insert("tool_calls", JsonValue::Array(norm_calls));
                }
            } else if let Some(fc) = m.get("function_call") {
                if let Some(name) = fc.get_str("name") {
                    let args_val = fc
                        .get_str("arguments")
                        .and_then(|s| JsonValue::parse(s).ok())
                        .filter(|v| v.is_object())
                        .unwrap_or_else(JsonValue::empty_obj);
                    let fn_entry = JsonValue::obj(vec![
                        ("name", JsonValue::from(name)),
                        ("arguments", args_val.clone()),
                    ]);
                    msg.insert(
                        "tool_calls",
                        JsonValue::Array(vec![JsonValue::obj(vec![
                            ("name", JsonValue::from(name)),
                            ("arguments", args_val),
                            ("function", fn_entry),
                        ])]),
                    );
                }
            }
            if content_txt.contains(THINK_CLOSE) && !msg.contains_key("reasoning_content") {
                if let Some((before, after)) = content_txt.split_once(THINK_CLOSE) {
                    let r = before.trim_start_matches(THINK_OPEN).trim();
                    msg.insert("reasoning_content", JsonValue::from(r));
                    msg.insert("content", JsonValue::from(after.trim_start()));
                }
            }
        }
        rest_msgs.push(msg);
    }

    let mut out = Vec::new();
    if !sys_parts.is_empty() {
        out.push(JsonValue::obj(vec![
            ("role", JsonValue::from("system")),
            ("content", JsonValue::from(sys_parts.join("\n\n"))),
        ]));
    }
    out.extend(rest_msgs);
    if out.is_empty() {
        return Err("messages: expected at least one message".to_string());
    }

    let tools = clean_tools(req.get("tools"))?;

    let mut kw = TemplateKwargs {
        enable_thinking: None,
        reasoning_effort: default_effort.map(|s| s.to_string()),
        preserve_thinking: None,
        add_generation_prompt: true,
    };

    if let Some(ctk) = req.get("chat_template_kwargs") {
        if let Some(et) = ctk.get_bool("enable_thinking") {
            kw.enable_thinking = Some(et);
        }
        if let Some(re) = ctk.get_str("reasoning_effort") {
            kw.reasoning_effort = Some(re.to_string());
        }
        if let Some(pt) = ctk.get_bool("preserve_thinking") {
            kw.preserve_thinking = Some(pt);
        }
    }
    if let Some(et) = req.get_bool("enable_thinking") {
        kw.enable_thinking = Some(et);
    }
    if let Some(thinking_obj) = req.get("thinking") {
        if let Some(t) = thinking_obj.get_str("type") {
            if t == "disabled" {
                kw.enable_thinking = Some(false);
            } else if t == "enabled" || t == "adaptive" {
                kw.enable_thinking = Some(true);
            }
        }
    }
    let effort = req
        .get_str("reasoning_effort")
        .or_else(|| req.get("reasoning").and_then(|r| r.get_str("effort")))
        .or_else(|| {
            req.get("chat_template_kwargs")
                .and_then(|c| c.get_str("reasoning_effort"))
        });
    if let Some(eff) = effort {
        match eff {
            "none" | "minimal" | "off" | "disabled" => {
                kw.enable_thinking = Some(false);
                kw.reasoning_effort = None;
            }
            "high" => {
                kw.reasoning_effort = Some("xhigh".to_string());
            }
            "xhigh" | "medium" | "low" => {
                kw.reasoning_effort = Some(eff.to_string());
            }
            other => {
                return Err(format!(
                    "Unexpected reasoning effort {other}. Supported types are xhigh (default), high, medium, low, and none."
                ));
            }
        }
    }
    if kw.enable_thinking == Some(false) {
        kw.reasoning_effort = None;
    }

    Ok((out, tools, kw))
}

/// Convert Anthropic `/v1/messages` request into normalized messages, tools, and template kwargs.
pub fn anthropic_to_messages(
    req: &JsonValue,
    think_unasked: bool,
) -> Result<NormalizedMessages, String> {
    anthropic_to_messages_with_effort(req, think_unasked, None)
}

pub fn anthropic_to_messages_with_effort(
    req: &JsonValue,
    think_unasked: bool,
    default_effort: Option<&str>,
) -> Result<NormalizedMessages, String> {
    if !req.is_object() {
        return Err("request body must be a JSON object".to_string());
    }
    let raw_msgs = req
        .get("messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "messages: expected a non-empty list of messages".to_string())?;
    if raw_msgs.is_empty() {
        return Err("messages: expected a non-empty list of messages".to_string());
    }

    let mut out = Vec::new();
    if let Some(sys) = req.get("system") {
        let sys_text = text_of_content(Some(sys))?;
        if !sys_text.trim().is_empty() {
            out.push(JsonValue::obj(vec![
                ("role", JsonValue::from("system")),
                ("content", JsonValue::from(sys_text)),
            ]));
        }
    }

    for m in raw_msgs {
        if !m.is_object() {
            return Err("each message must be a JSON object".to_string());
        }
        let role = m
            .get_str("role")
            .ok_or_else(|| "message is missing 'role'".to_string())?;
        let blocks = match m.get("content") {
            None | Some(JsonValue::Null) => Vec::new(),
            Some(JsonValue::Str(s)) => {
                vec![JsonValue::obj(vec![
                    ("type", JsonValue::from("text")),
                    ("text", JsonValue::from(s.as_str())),
                ])]
            }
            Some(JsonValue::Array(arr)) => arr.clone(),
            _ => return Err("message content must be a string or list of blocks".to_string()),
        };

        let mut text_parts: Vec<String> = Vec::new();
        let mut thinking_parts: Vec<String> = Vec::new();
        let mut calls: Vec<JsonValue> = Vec::new();
        let mut tool_results: Vec<String> = Vec::new();

        for b in &blocks {
            if let Some(s) = b.as_str() {
                text_parts.push(s.to_string());
                continue;
            }
            if !b.is_object() {
                return Err("content block must be an object".to_string());
            }
            let btype = b.get_str("type").unwrap_or("text");
            match btype {
                "text" => {
                    let t = b
                        .get_str("text")
                        .ok_or_else(|| "text block is missing 'text'".to_string())?;
                    text_parts.push(t.to_string());
                }
                "thinking" => {
                    if let Some(th) = b.get_str("thinking") {
                        thinking_parts.push(th.to_string());
                    }
                }
                "tool_use" => {
                    let name = b
                        .get_str("name")
                        .ok_or_else(|| "tool_use block is missing 'name'".to_string())?;
                    let input = b.get("input").cloned().unwrap_or_else(JsonValue::empty_obj);
                    let fn_entry = JsonValue::obj(vec![
                        ("name", JsonValue::from(name)),
                        ("arguments", input.clone()),
                    ]);
                    calls.push(JsonValue::obj(vec![
                        ("name", JsonValue::from(name)),
                        ("arguments", input),
                        ("function", fn_entry),
                    ]));
                }
                "tool_result" => {
                    let c = b.get("content");
                    let txt = match c {
                        None | Some(JsonValue::Null) => String::new(),
                        Some(JsonValue::Str(s)) => s.clone(),
                        Some(JsonValue::Array(_)) => text_of_content(c)?,
                        Some(other) => other.to_json_string(),
                    };
                    tool_results.push(txt);
                }
                _ => {}
            }
        }

        for tr in tool_results {
            out.push(JsonValue::obj(vec![
                ("role", JsonValue::from("tool")),
                ("content", JsonValue::from(tr)),
            ]));
        }

        if !text_parts.is_empty()
            || !thinking_parts.is_empty()
            || !calls.is_empty()
            || blocks.is_empty()
        {
            let mut msg = JsonValue::obj(vec![
                ("role", JsonValue::from(role)),
                ("content", JsonValue::from(text_parts.join(""))),
            ]);
            if !thinking_parts.is_empty() {
                msg.insert(
                    "reasoning_content",
                    JsonValue::from(thinking_parts.join("\n")),
                );
            }
            if !calls.is_empty() {
                msg.insert("tool_calls", JsonValue::Array(calls));
            }
            out.push(msg);
        }
    }

    let tools = clean_tools(req.get("tools"))?;
    let mut kw = TemplateKwargs {
        enable_thinking: None,
        reasoning_effort: default_effort.map(|s| s.to_string()),
        preserve_thinking: None,
        add_generation_prompt: true,
    };

    let ttype = req
        .get("thinking")
        .and_then(|t| t.get_str("type"))
        .unwrap_or(if think_unasked { "enabled" } else { "disabled" });
    if ttype == "disabled" {
        kw.enable_thinking = Some(false);
        kw.reasoning_effort = None;
    } else if let Some(eff) = req
        .get("output_config")
        .and_then(|o| o.get_str("effort"))
        .or_else(|| req.get_str("reasoning_effort"))
    {
        match eff {
            "none" | "minimal" | "off" | "disabled" => {
                kw.enable_thinking = Some(false);
                kw.reasoning_effort = None;
            }
            "max" | "xhigh" | "high" => {
                kw.reasoning_effort = Some("xhigh".to_string());
            }
            "medium" | "low" => {
                kw.reasoning_effort = Some(eff.to_string());
            }
            other => {
                return Err(format!("Unexpected reasoning effort {other}"));
            }
        }
    }

    Ok((out, tools, kw))
}

const MARK_OPEN: &str = "\x00STRATA_LIT_OPEN\x00";
const MARK_CLOSE: &str = "\x00STRATA_LIT_CLOSE\x00";

const TEMPLATE_SPECIALS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "<|image_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
];

/// Render chat messages and tools into a Qwen/Strata chat template prompt string
/// with sentinel markers around user/tool literal special tokens.
pub fn render_chat_prompt(
    messages: &[JsonValue],
    tools: Option<&[JsonValue]>,
    kwargs: &TemplateKwargs,
) -> String {
    if messages.is_empty() {
        return String::new();
    }

    let mark_str = |s: &str| -> String {
        let mut res = s.to_string();
        for &tok in TEMPLATE_SPECIALS {
            if res.contains(tok) {
                let replacement = format!("{MARK_OPEN}{tok}{MARK_CLOSE}");
                res = res.replace(tok, &replacement);
            }
        }
        res
    };

    let mut sys_count = 0usize;
    let mut merged_system = String::new();
    for (idx, m) in messages.iter().enumerate() {
        let role = m.get_str("role").unwrap_or("");
        if idx == sys_count && (role == "system" || role == "developer") {
            let content = m.get_str("content").unwrap_or("").trim();
            if !content.is_empty() {
                if !merged_system.is_empty() {
                    merged_system.push('\n');
                }
                merged_system.push_str(&mark_str(content));
            }
            sys_count += 1;
        }
    }

    let mut reasoning_instructions = "";
    if kwargs.thinking_enabled() {
        let eff = kwargs.reasoning_effort.as_deref().unwrap_or("xhigh");
        let resolved = if eff == "high" { "xhigh" } else { eff };
        match resolved {
            "xhigh" => {
                reasoning_instructions = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
            }
            "low" => {
                reasoning_instructions = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
            }
            _ => {
                reasoning_instructions = "";
            }
        }
    }

    let mut out = String::with_capacity(512);
    if let Some(tool_list) = tools.filter(|t| !t.is_empty()) {
        out.push_str("<|im_start|>system\n");
        if !reasoning_instructions.is_empty() {
            out.push_str(reasoning_instructions);
            out.push_str("\n\n");
        }
        out.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for t in tool_list {
            out.push('\n');
            out.push_str(&mark_str(&t.to_json_string()));
        }
        out.push_str("\n</tools>");
        out.push_str("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
        if !merged_system.is_empty() {
            out.push_str("\n\n");
            out.push_str(&merged_system);
        }
        out.push_str("<|im_end|>\n");
    } else if !merged_system.is_empty() {
        out.push_str("<|im_start|>system\n");
        if !reasoning_instructions.is_empty() {
            out.push_str(reasoning_instructions);
            out.push_str("\n\n");
        }
        out.push_str(&merged_system);
        out.push_str("<|im_end|>\n");
    } else if !reasoning_instructions.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(reasoning_instructions);
        out.push_str("<|im_end|>\n");
    }

    let mut last_query_index = messages.len().saturating_sub(1);
    for (idx, m) in messages.iter().enumerate().rev() {
        if m.get_str("role") == Some("user") {
            let c = m.get_str("content").unwrap_or("").trim();
            if !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>")) {
                last_query_index = idx;
                break;
            }
        }
    }

    for (idx, m) in messages.iter().enumerate().skip(sys_count) {
        let role = m.get_str("role").unwrap_or("");
        let raw_content = m.get_str("content").unwrap_or("").trim();
        let content = mark_str(raw_content);
        match role {
            "user" | "system" | "developer" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(&content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                let rc = mark_str(m.get_str("reasoning_content").unwrap_or("").trim());
                let preserve = kwargs.preserve_thinking.unwrap_or(true) || idx > last_query_index;
                if preserve {
                    out.push_str("<|im_start|>assistant\n<think>\n");
                    out.push_str(&rc);
                    out.push_str("\n</think>\n\n");
                    out.push_str(&content);
                } else {
                    out.push_str("<|im_start|>assistant\n");
                    out.push_str(&content);
                }
                if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()) {
                    for (tc_idx, tc) in tcs.iter().enumerate() {
                        let fn_obj = tc.get("function").unwrap_or(tc);
                        let name = fn_obj.get_str("name").unwrap_or("");
                        if tc_idx == 0 {
                            if !content.is_empty() {
                                out.push_str("\n\n<tool_call>\n<function=");
                            } else {
                                out.push_str("<tool_call>\n<function=");
                            }
                        } else {
                            out.push_str("\n<tool_call>\n<function=");
                        }
                        out.push_str(name);
                        out.push_str(">\n");
                        if let Some(args) = fn_obj.get("arguments").and_then(|a| a.as_object()) {
                            for (k, v) in args {
                                out.push_str("<parameter=");
                                out.push_str(k);
                                out.push_str(">\n");
                                let val_str = match v {
                                    JsonValue::Str(s) => mark_str(s),
                                    other => mark_str(&other.to_json_string()),
                                };
                                out.push_str(&val_str);
                                out.push_str("\n</parameter>\n");
                            }
                        }
                        out.push_str("</function>\n</tool_call>");
                    }
                }
                out.push_str("<|im_end|>\n");
            }
            "tool" => {
                let prev_role = if idx > sys_count {
                    messages.get(idx - 1).and_then(|p| p.get_str("role"))
                } else {
                    None
                };
                if prev_role != Some("tool") {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(&content);
                out.push_str("\n</tool_response>");
                let next_role = messages.get(idx + 1).and_then(|n| n.get_str("role"));
                if next_role != Some("tool") {
                    out.push_str("<|im_end|>\n");
                }
            }
            _ => {}
        }
    }

    if kwargs.add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if kwargs.enable_thinking == Some(false) {
            out.push_str("<think>\n\n</think>\n\n");
        } else {
            out.push_str("<think>\n");
        }
    }

    out
}

/// Convert a `JsonValue` to `serde_json::Value` for `talus_text::ChatTemplate` rendering.
pub fn to_serde_value(val: &JsonValue) -> serde_json::Value {
    match val {
        JsonValue::Null => serde_json::Value::Null,
        JsonValue::Bool(b) => serde_json::Value::Bool(*b),
        JsonValue::Int(i) => serde_json::Value::Number((*i).into()),
        JsonValue::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        JsonValue::Str(s) => serde_json::Value::String(s.clone()),
        JsonValue::Array(arr) => serde_json::Value::Array(arr.iter().map(to_serde_value).collect()),
        JsonValue::Object(entries) => {
            let mut map = serde_json::Map::with_capacity(entries.len());
            for (k, v) in entries {
                map.insert(k.clone(), to_serde_value(v));
            }
            serde_json::Value::Object(map)
        }
    }
}

/// Split a rendered prompt into `(segment, is_plain_user_literal)` pairs so `ByteTokenizer::encode_with_plain`
/// encodes user-supplied `<think>` / `<|im_end|>` literals as plain UTF-8 bytes rather than special token IDs.
pub fn unmark_think_literals(rendered: &str) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let mut rest = rendered;
    while let Some(open_pos) = rest.find(MARK_OPEN) {
        if open_pos > 0 {
            out.push((rest[..open_pos].to_string(), false));
        }
        let after_open = &rest[open_pos + MARK_OPEN.len()..];
        if let Some(close_pos) = after_open.find(MARK_CLOSE) {
            let lit = &after_open[..close_pos];
            out.push((lit.to_string(), true));
            rest = &after_open[close_pos + MARK_CLOSE.len()..];
        } else {
            out.push((after_open.to_string(), false));
            rest = "";
            break;
        }
    }
    if !rest.is_empty() {
        out.push((rest.to_string(), false));
    }
    out
}

fn schema_types(prop: Option<&JsonValue>) -> Vec<String> {
    let p = match prop {
        Some(v) if v.is_object() => v,
        _ => return Vec::new(),
    };
    let mut out = Vec::new();
    match p.get("type") {
        Some(JsonValue::Str(s)) => out.push(s.clone()),
        Some(JsonValue::Array(arr)) => {
            for item in arr {
                if let Some(s) = item.as_str() {
                    out.push(s.to_string());
                }
            }
        }
        _ => {}
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(arr) = p.get(key).and_then(|v| v.as_array()) {
            for sub in arr {
                out.extend(schema_types(Some(sub)));
            }
        }
    }
    out
}

pub fn coerce_param_value(raw: &str, prop_schema: Option<&JsonValue>) -> JsonValue {
    let want = schema_types(prop_schema);
    if want.iter().any(|t| t == "string")
        && !want.iter().any(|t| {
            matches!(
                t.as_str(),
                "integer" | "number" | "boolean" | "array" | "object"
            )
        })
    {
        return JsonValue::Str(raw.to_string());
    }
    let trimmed = raw.trim();
    if let Ok(parsed) = JsonValue::parse(trimmed) {
        let kind = match &parsed {
            JsonValue::Null => "null",
            JsonValue::Bool(_) => "boolean",
            JsonValue::Int(_) => "integer",
            JsonValue::Float(_) => "number",
            JsonValue::Str(_) => "string",
            JsonValue::Array(_) => "array",
            JsonValue::Object(_) => "object",
        };
        if want.is_empty()
            || want.iter().any(|t| t == kind)
            || (kind == "integer" && want.iter().any(|t| t == "number"))
        {
            return parsed;
        }
    }
    JsonValue::Str(raw.to_string())
}

/// Parse the inner body of a `<tool_call>...</tool_call>` block.
/// Supports:
/// 1. Qwen XML `<function=NAME><parameter=K>V</parameter></function>`
/// 2. JSON `{"name": "NAME", "arguments": {...}}`
/// 3. GLM `<arg_key>` `NAME<arg_key>K</arg_key><arg_value>V</arg_value>`
pub fn parse_tool_call(body: &str, schemas: Option<&[JsonValue]>) -> Option<ToolCall> {
    let trimmed = body.trim();
    let get_schema_props = |fn_name: &str| -> Option<&JsonValue> {
        schemas?
            .iter()
            .find(|t| t.get_str("name") == Some(fn_name))
            .and_then(|t| t.get("parameters"))
            .and_then(|p| p.get("properties"))
    };

    // 1. Qwen XML: <function=NAME>...</function>
    if let Some(fn_start) = trimmed.find(FUNC_OPEN) {
        let after_fn = &trimmed[fn_start + FUNC_OPEN.len()..];
        if let Some(gt_pos) = after_fn.find('>') {
            let name = after_fn[..gt_pos].trim().to_string();
            let rest = &after_fn[gt_pos + 1..];
            let end_pos = rest.rfind(FUNC_CLOSE)?;
            let inner = &rest[..end_pos];
            let props = get_schema_props(&name);
            let mut args = JsonValue::empty_obj();
            let mut cursor = inner;
            while let Some(p_start) = cursor.find("<parameter=") {
                let after_p = &cursor[p_start + "<parameter=".len()..];
                let Some(p_gt) = after_p.find('>') else {
                    break;
                };
                let key = after_p[..p_gt].trim().to_string();
                let val_rest = &after_p[p_gt + 1..];
                let Some(p_end) = val_rest.find("</parameter>") else {
                    break;
                };
                let mut raw_val = &val_rest[..p_end];
                if let Some(s) = raw_val.strip_prefix("\r\n") {
                    raw_val = s;
                } else if let Some(s) = raw_val.strip_prefix('\n') {
                    raw_val = s;
                }
                if let Some(s) = raw_val.strip_suffix("\r\n") {
                    raw_val = s;
                } else if let Some(s) = raw_val.strip_suffix('\n') {
                    raw_val = s;
                }
                let prop_schema = props.and_then(|p| p.get(&key));
                args.insert(key, coerce_param_value(raw_val, prop_schema));
                cursor = &val_rest[p_end + "</parameter>".len()..];
            }
            return Some(ToolCall {
                id: format!("call_{}", next_hex_id(24)),
                name,
                arguments: args,
            });
        }
    }

    // 2. GLM <arg_key>: NAME<arg_key>K</arg_key><arg_value>V</arg_value>
    if !trimmed.starts_with('{') && !trimmed.starts_with('<') {
        let name_end = trimmed.find('<').unwrap_or(trimmed.len());
        let name = trimmed[..name_end].trim().to_string();
        if !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':'))
        {
            let rest = &trimmed[name_end..];
            if rest.trim().is_empty() || rest.contains(ARG_KEY_OPEN) {
                let props = get_schema_props(&name);
                let mut args = JsonValue::empty_obj();
                let mut cursor = rest;
                while let Some(k_start) = cursor.find(ARG_KEY_OPEN) {
                    let after_k = &cursor[k_start + ARG_KEY_OPEN.len()..];
                    let Some(k_end) = after_k.find(ARG_KEY_CLOSE) else {
                        break;
                    };
                    let key = after_k[..k_end].trim().to_string();
                    let after_k_close = &after_k[k_end + ARG_KEY_CLOSE.len()..];
                    let Some(v_start) = after_k_close.find(ARG_VAL_OPEN) else {
                        break;
                    };
                    let after_v = &after_k_close[v_start + ARG_VAL_OPEN.len()..];
                    let Some(v_end) = after_v.find(ARG_VAL_CLOSE) else {
                        break;
                    };
                    let mut raw_val = &after_v[..v_end];
                    if let Some(s) = raw_val.strip_prefix("\r\n") {
                        raw_val = s;
                    } else if let Some(s) = raw_val.strip_prefix('\n') {
                        raw_val = s;
                    }
                    if let Some(s) = raw_val.strip_suffix("\r\n") {
                        raw_val = s;
                    } else if let Some(s) = raw_val.strip_suffix('\n') {
                        raw_val = s;
                    }
                    let prop_schema = props.and_then(|p| p.get(&key));
                    args.insert(key, coerce_param_value(raw_val, prop_schema));
                    cursor = &after_v[v_end + ARG_VAL_CLOSE.len()..];
                }
                return Some(ToolCall {
                    id: format!("call_{}", next_hex_id(24)),
                    name,
                    arguments: args,
                });
            }
        }
    }

    // 3. JSON tool call: {"name": "NAME", "arguments": {...}}
    if let Ok(obj) = JsonValue::parse(trimmed) {
        if let Some(name) = obj.get_str("name") {
            let args = match obj.get("arguments").or_else(|| obj.get("parameters")) {
                None | Some(JsonValue::Null) => JsonValue::empty_obj(),
                Some(JsonValue::Str(s)) => {
                    JsonValue::parse(s).unwrap_or_else(|_| JsonValue::empty_obj())
                }
                Some(v) if v.is_object() => v.clone(),
                _ => JsonValue::empty_obj(),
            };
            return Some(ToolCall {
                id: format!("call_{}", next_hex_id(24)),
                name: name.to_string(),
                arguments: args,
            });
        }
    }

    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParserState {
    Reasoning,
    Content,
    Call,
    ReasoningCall,
}

/// Incremental streaming parser for `<think>...</think>` and `<tool_call>...</tool_call>`.
pub struct OutputParser {
    pub state: ParserState,
    buf: String,
    lead: bool,
    tools: Option<Vec<JsonValue>>,
    has_tools: bool,
    recover: bool,
    pub rescued: usize,
    pub refused: usize,
    pub recovered_from_reasoning: bool,
}

impl OutputParser {
    pub fn new(
        thinking: bool,
        tools: Option<Vec<JsonValue>>,
        _stream_tool_args: bool,
        recover: bool,
        _strict_tool_names: bool,
    ) -> Self {
        let has_tools = tools.as_ref().map(|t| !t.is_empty()).unwrap_or(true);
        Self {
            state: if thinking {
                ParserState::Reasoning
            } else {
                ParserState::Content
            },
            buf: String::new(),
            lead: false,
            tools,
            has_tools,
            recover,
            rescued: 0,
            refused: 0,
            recovered_from_reasoning: false,
        }
    }

    pub fn feed(&mut self, piece: &str) -> Vec<Event> {
        self.buf.push_str(piece);
        let mut out = Vec::new();
        loop {
            match self.state {
                ParserState::Reasoning => {
                    if let Some(pos) = self.buf.find(THINK_CLOSE) {
                        let before = self.buf[..pos].to_string();
                        self.buf.drain(..pos + THINK_CLOSE.len());
                        let clean = before.strip_prefix(THINK_OPEN).unwrap_or(&before);
                        if !clean.is_empty() {
                            out.push(Event::reasoning(clean));
                        }
                        self.lead = true;
                        self.state = ParserState::Content;
                        continue;
                    }
                    // Strip leading <think> if model emitted it redundantly
                    if let Some(stripped) = self.buf.strip_prefix(THINK_OPEN) {
                        self.buf = stripped.to_string();
                    }
                    let hold = partial_suffix_len(&self.buf, &[THINK_CLOSE, THINK_OPEN]);
                    let emit_len = self.buf.len().saturating_sub(hold);
                    if emit_len > 0 && self.buf.is_char_boundary(emit_len) {
                        let text = self.buf[..emit_len].to_string();
                        self.buf.drain(..emit_len);
                        out.push(Event::reasoning(text));
                    }
                    break;
                }
                ParserState::Content => {
                    if self.lead {
                        let trimmed = self.buf.trim_start_matches(['\r', '\n']).to_string();
                        self.buf = trimmed;
                        if self.buf.is_empty() {
                            break;
                        }
                        self.lead = false;
                    }
                    // Check for <tool_call> or (when recover is true) bare <function=
                    let tc_pos = if self.has_tools {
                        self.buf.find(TOOL_OPEN)
                    } else {
                        None
                    };
                    let bare_pos = if self.has_tools && self.recover {
                        self.buf.find(FUNC_OPEN)
                    } else {
                        None
                    };
                    let open_match = match (tc_pos, bare_pos) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (Some(a), None) => Some(a),
                        (None, Some(b)) => Some(b),
                        (None, None) => None,
                    };
                    if let Some(pos) = open_match {
                        if pos > 0 {
                            let before = self.buf[..pos].to_string();
                            out.push(Event::content(before));
                        }
                        let is_tc = self.buf[pos..].starts_with(TOOL_OPEN);
                        let skip = if is_tc { TOOL_OPEN.len() } else { 0 };
                        self.buf.drain(..pos + skip);
                        self.state = ParserState::Call;
                        continue;
                    }
                    let mut guards = vec![TOOL_OPEN];
                    if self.recover {
                        guards.push(FUNC_OPEN);
                    }
                    let hold = if self.has_tools {
                        partial_suffix_len(&self.buf, &guards)
                    } else {
                        0
                    };
                    let emit_len = self.buf.len().saturating_sub(hold);
                    if emit_len > 0 && self.buf.is_char_boundary(emit_len) {
                        let text = self.buf[..emit_len].to_string();
                        self.buf.drain(..emit_len);
                        out.push(Event::content(text));
                    }
                    break;
                }
                ParserState::Call | ParserState::ReasoningCall => {
                    if let Some(end_pos) = self.buf.find(TOOL_CLOSE) {
                        let raw_call = self.buf[..end_pos].to_string();
                        self.buf.drain(..end_pos + TOOL_CLOSE.len());
                        if let Some(call) = parse_tool_call(&raw_call, self.tools.as_deref()) {
                            let start_call = ToolCall {
                                id: call.id.clone(),
                                name: call.name.clone(),
                                arguments: JsonValue::empty_obj(),
                            };
                            let args_json = call.arguments.to_json_string();
                            out.push(Event::tool_start(start_call.clone()));
                            out.push(Event::tool_args(args_json, start_call));
                            out.push(Event::tool_call(call));
                        } else {
                            out.push(Event::content(format!("{TOOL_OPEN}{raw_call}{TOOL_CLOSE}")));
                        }
                        let trimmed = self.buf.trim_start_matches(['\r', '\n']).to_string();
                        self.buf = trimmed;
                        self.state = ParserState::Content;
                        continue;
                    }
                    if self.recover {
                        if let Some(end_pos) = self.buf.find(FUNC_CLOSE) {
                            let after = &self.buf[end_pos + FUNC_CLOSE.len()..];
                            if !after.trim_start().starts_with(TOOL_CLOSE) && after.contains('\n') {
                                let raw_call = self.buf[..end_pos + FUNC_CLOSE.len()].to_string();
                                self.buf.drain(..end_pos + FUNC_CLOSE.len());
                                if let Some(call) =
                                    parse_tool_call(&raw_call, self.tools.as_deref())
                                {
                                    let start_call = ToolCall {
                                        id: call.id.clone(),
                                        name: call.name.clone(),
                                        arguments: JsonValue::empty_obj(),
                                    };
                                    let args_json = call.arguments.to_json_string();
                                    out.push(Event::tool_start(start_call.clone()));
                                    out.push(Event::tool_args(args_json, start_call));
                                    out.push(Event::tool_call(call));
                                }
                                self.state = ParserState::Content;
                                continue;
                            }
                        }
                    }
                    break;
                }
            }
        }
        out
    }

    pub fn finish(&mut self) -> Vec<Event> {
        self.finish_with_reason("stop")
    }

    pub fn finish_with_reason(&mut self, finish_reason: &str) -> Vec<Event> {
        let mut out = Vec::new();
        if self.buf.is_empty() {
            return out;
        }
        let leftover = std::mem::take(&mut self.buf);
        match self.state {
            ParserState::Reasoning => {
                // Rescue tool call inside unclosed <think> if finish_reason == "stop"
                if finish_reason == "stop" && self.has_tools && self.recover {
                    if let Some(tc_pos) = leftover.find(TOOL_OPEN) {
                        if let Some(end_pos) = leftover[tc_pos..].find(TOOL_CLOSE) {
                            let before = &leftover[..tc_pos];
                            let raw_call = &leftover[tc_pos + TOOL_OPEN.len()..tc_pos + end_pos];
                            if let Some(call) = parse_tool_call(raw_call, self.tools.as_deref()) {
                                if !before.is_empty() {
                                    out.push(Event::reasoning(before));
                                }
                                let start_call = ToolCall {
                                    id: call.id.clone(),
                                    name: call.name.clone(),
                                    arguments: JsonValue::empty_obj(),
                                };
                                let args_json = call.arguments.to_json_string();
                                out.push(Event::tool_start(start_call.clone()));
                                out.push(Event::tool_args(args_json, start_call));
                                out.push(Event::tool_call(call));
                                self.rescued += 1;
                                self.recovered_from_reasoning = true;
                                return out;
                            }
                        }
                    }
                }
                out.push(Event::reasoning(leftover));
            }
            ParserState::Content => {
                out.push(Event::content(leftover));
            }
            ParserState::Call | ParserState::ReasoningCall => {
                if let Some(call) = parse_tool_call(&leftover, self.tools.as_deref()) {
                    let start_call = ToolCall {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        arguments: JsonValue::empty_obj(),
                    };
                    let args_json = call.arguments.to_json_string();
                    out.push(Event::tool_start(start_call.clone()));
                    out.push(Event::tool_args(args_json, start_call));
                    out.push(Event::tool_call(call));
                } else {
                    out.push(Event::content(format!("{TOOL_OPEN}{leftover}")));
                }
            }
        }
        out
    }
}

fn partial_suffix_len(s: &str, tags: &[&str]) -> usize {
    let mut best = 0usize;
    for &tag in tags {
        let max_k = tag.len().saturating_sub(1).min(s.len());
        for k in (1..=max_k).rev() {
            let start = s.len() - k;
            if s.is_char_boundary(start) && tag.starts_with(&s[start..]) {
                if k > best {
                    best = k;
                }
                break;
            }
        }
    }
    best
}
