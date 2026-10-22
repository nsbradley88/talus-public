//! Incremental reasoning (`<think>...</think>`) and tool-call (`<tool_call>...</tool_call>`)
//! output and stream parsers.
//!
//! Direct port of `serve/frontend.py`'s `OutputParser`, `to_xml_call`, `parse_tool_call`,
//! `json_tool_call`, `param_end`, `call_end`, `function_end`, `tool_choice_of`, and `forced_call`,
//! plus OpenAI-compatible incremental [`StreamParser`] emitting [`ToolCallDelta`] chunks.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const CALL_START: &str = "<tool_call>";
pub const CALL_END: &str = "</tool_call>";
pub const PARAM_START: &str = "<parameter=";
pub const PARAM_END: &str = "</parameter>";
pub const FUNC_START: &str = "<function=";
pub const FUNC_END: &str = "</function>";

/// Maximum bytes of a `<tool_call>` inside reasoning held while waiting for its closing tag (#804).
pub const RCALL_MAX: usize = 32768;

static CALL_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_call_id() -> String {
    let n = CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("call_{n:024x}")
}

/// Errors produced during tool-call parsing.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParserError {
    #[error("malformed tool call: {0}")]
    MalformedToolCall(String),
}

/// A parsed tool call with function `name`, JSON `arguments` object, and call `id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
    pub id: String,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: Map<String, Value>) -> Self {
        Self {
            name: name.into(),
            arguments,
            id: next_call_id(),
        }
    }

    pub fn with_id(
        name: impl Into<String>,
        arguments: Map<String, Value>,
        id: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            arguments,
            id: id.into(),
        }
    }
}

/// Event emitted by [`OutputParser`].
///
/// `kind` is one of `"reasoning"`, `"content"`, `"tool_start"`, `"tool_args"`, or `"tool_call"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParseEvent {
    pub kind: String,
    pub text: String,
    pub call: Option<ToolCall>,
}

impl ParseEvent {
    pub fn reasoning(text: impl Into<String>) -> Self {
        Self {
            kind: "reasoning".to_string(),
            text: text.into(),
            call: None,
        }
    }

    pub fn content(text: impl Into<String>) -> Self {
        Self {
            kind: "content".to_string(),
            text: text.into(),
            call: None,
        }
    }

    pub fn tool_start(call: ToolCall) -> Self {
        Self {
            kind: "tool_start".to_string(),
            text: String::new(),
            call: Some(call),
        }
    }

    pub fn tool_args(text: impl Into<String>, call: Option<ToolCall>) -> Self {
        Self {
            kind: "tool_args".to_string(),
            text: text.into(),
            call,
        }
    }

    pub fn tool_call(call: ToolCall) -> Self {
        Self {
            kind: "tool_call".to_string(),
            text: String::new(),
            call: Some(call),
        }
    }
}

/// Alias matching `serve/frontend.py`'s `Event`.
pub type Event = ParseEvent;

/// OpenAI-style function delta inside a [`ToolCallDelta`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub arguments: String,
}

/// OpenAI-style incremental `tool_calls` delta entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    pub function: ToolCallFunctionDelta,
}

/// OpenAI-style incremental chunk produced by [`StreamParser`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tool_calls: Vec<ToolCallDelta>,
}

/// Find where a parameter value in `text` ends: the first `</parameter>` followed (after whitespace)
/// by the next `<parameter=` or `</function>`.
///
/// Returns `-1` if none yet, or `-2` if a candidate `</parameter>` was found whose follower has not
/// yet arrived (accepted when `final_mode` is `true`).
pub fn param_end(text: &str, final_mode: bool) -> isize {
    let mut search_from = 0;
    while let Some(rel) = text[search_from..].find(PARAM_END) {
        let at = search_from + rel;
        let after = text[at + PARAM_END.len()..].trim_start();
        if after.starts_with(PARAM_START) || after.starts_with(FUNC_END) {
            return at as isize;
        }
        if after.is_empty() || PARAM_START.starts_with(after) || FUNC_END.starts_with(after) {
            return if final_mode { at as isize } else { -2 };
        }
        search_from = at + 1;
    }
    -1
}

/// Find where a tool call's body in `text` ends (the byte index of `</tool_call>` after `</function>`),
/// walking parameters with [`param_end`] so parameter values may contain literal `</parameter>` or `</tool_call>`.
pub fn call_end(text: &str) -> isize {
    let s = text.trim_start();
    let mut pos = text.len() - s.len();
    if !s.starts_with(FUNC_START) {
        return if !FUNC_START.starts_with(s) {
            text.find(CALL_END).map(|i| i as isize).unwrap_or(-1)
        } else {
            -1
        };
    }
    let Some(gt_rel) = text[pos..].find('>') else {
        return -1;
    };
    pos += gt_rel + 1;
    loop {
        let rest = &text[pos..];
        let s = rest.trim_start();
        pos += rest.len() - s.len();
        if s.starts_with(PARAM_START) {
            let Some(gt_rel) = text[pos..].find('>') else {
                return -1;
            };
            let gt = pos + gt_rel;
            let end = param_end(&text[gt + 1..], false);
            if end < 0 {
                return -1;
            }
            pos = gt + 1 + (end as usize) + PARAM_END.len();
        } else if s.starts_with(FUNC_END) {
            let rest = &text[pos + FUNC_END.len()..];
            let s = rest.trim_start();
            if s.starts_with(CALL_END) {
                return (pos + FUNC_END.len() + (rest.len() - s.len())) as isize;
            }
            return if CALL_END.starts_with(s) {
                -1
            } else {
                text[pos..]
                    .find(CALL_END)
                    .map(|rel| (pos + rel) as isize)
                    .unwrap_or(-1)
            };
        } else if s.is_empty() || PARAM_START.starts_with(s) || FUNC_END.starts_with(s) {
            return -1;
        } else {
            return text[pos..]
                .find(CALL_END)
                .map(|rel| (pos + rel) as isize)
                .unwrap_or(-1);
        }
    }
}

/// Find where a `<function=NAME>...</function>` at the start of `text` ends (just past `</function>`).
pub fn function_end(text: &str) -> isize {
    let s = text.trim_start();
    let mut pos = text.len() - s.len();
    if !s.starts_with(FUNC_START) {
        return -1;
    }
    let Some(gt_rel) = text[pos..].find('>') else {
        return -1;
    };
    pos += gt_rel + 1;
    loop {
        let rest = &text[pos..];
        let s = rest.trim_start();
        pos += rest.len() - s.len();
        if s.starts_with(PARAM_START) {
            let Some(gt_rel) = text[pos..].find('>') else {
                return -1;
            };
            let gt = pos + gt_rel;
            let end = param_end(&text[gt + 1..], false);
            if end < 0 {
                return -1;
            }
            pos = gt + 1 + (end as usize) + PARAM_END.len();
        } else if s.starts_with(FUNC_END) {
            return (pos + FUNC_END.len()) as isize;
        } else {
            return -1;
        }
    }
}

/// Normalize a client's `tool_choice` in OpenAI or Anthropic shape into `("auto" | "none" | "required" | "named", Option<name>)`.
pub fn tool_choice_of(tool_choice: Option<&Value>) -> (&'static str, Option<String>) {
    let Some(tc) = tool_choice else {
        return ("auto", None);
    };
    if let Some(s) = tc.as_str() {
        return match s {
            "auto" => ("auto", None),
            "none" => ("none", None),
            "required" => ("required", None),
            _ => ("unknown", None),
        };
    }
    if let Some(obj) = tc.as_object() {
        let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match kind {
            "auto" => return ("auto", None),
            "none" => return ("none", None),
            "any" | "required" => return ("required", None),
            "function" | "tool" => {
                let name = obj
                    .get("function")
                    .and_then(|f| f.as_object())
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .or_else(|| obj.get("name").and_then(|n| n.as_str()));
                if let Some(n) = name {
                    if !n.is_empty() {
                        return ("named", Some(n.to_string()));
                    }
                }
            }
            _ => {}
        }
    }
    ("unknown", None)
}

/// Compute the forced tool-call prefix for `tool_choice` and declared `tools`, or `None` if the model decides.
pub fn forced_call(tool_choice: Option<&Value>, tools: Option<&[Value]>) -> Option<String> {
    let (mut kind, mut name) = tool_choice_of(tool_choice);
    let names: Vec<String> = tools
        .unwrap_or(&[])
        .iter()
        .filter_map(|t| {
            t.as_object()?
                .get("name")?
                .as_str()
                .map(ToString::to_string)
        })
        .collect();
    if matches!(kind, "auto" | "none") {
        return None;
    }
    if kind == "required" && names.len() == 1 {
        kind = "named";
        name = Some(names[0].clone());
    }
    if kind == "required" && !names.is_empty() {
        return Some(format!("{CALL_START}\n<function="));
    }
    if kind == "named" {
        if let Some(n) = name {
            if names.iter().any(|x| x == &n) {
                return Some(format!("{CALL_START}\n<function={n}>\n"));
            }
        }
    }
    None
}

fn strip_json_fence(body: &str) -> &str {
    let body = body.trim();
    let rx = Regex::new(r"(?i)^```[ \t]*(?:json)?[ \t]*\r?\n\s*").unwrap();
    if let Ok(Some(m)) = rx.find(body) {
        let rest = body[m.end()..].trim_end();
        return rest.strip_suffix("```").unwrap_or(rest);
    }
    body
}

/// Parse a Qwen JSON-style tool call `{"name": N, "arguments": {...}}` (optionally inside a ````json` fence).
///
/// When `schemas` is `Some(map)`, only tools present in `map` are accepted (matching `serve/frontend.py`).
/// When `schemas` is `None`, any tool name is accepted.
pub fn json_tool_call(body: &str, schemas: Option<&HashMap<String, Value>>) -> Option<ToolCall> {
    let inner = strip_outer_tool_call_tags(body);
    let cleaned = strip_json_fence(inner);
    let obj: Value = serde_json::from_str(cleaned).ok()?;
    let map = obj.as_object()?;
    let name = map.get("name")?.as_str()?;
    if name.is_empty() {
        return None;
    }
    if let Some(s) = schemas {
        if !s.contains_key(name) {
            return None;
        }
    }
    let raw_args = map
        .get("arguments")
        .or_else(|| map.get("parameters"))
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let args_map = match raw_args {
        Value::Null => Map::new(),
        Value::Object(m) => m,
        Value::String(s) => {
            let parsed: Value = serde_json::from_str(&s).ok()?;
            parsed.as_object()?.clone()
        }
        _ => return None,
    };
    Some(ToolCall::new(name, args_map))
}

fn is_glm_name_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '.' | '-' | ':')
}

/// Convert a GLM-style `<arg_key>` tool-call body (`NAME<arg_key>K</arg_key><arg_value>V</arg_value>...`)
/// into the canonical `<function=NAME><parameter=K>V</parameter>...</function>` XML form.
///
/// Bodies already in `<function=...>` form or not matching the GLM shape are returned unchanged.
pub fn to_xml_call(body: &str) -> String {
    if body.trim_start().starts_with(FUNC_START) {
        return body.to_string();
    }
    let trimmed = body.trim_start();
    let name_len: usize = trimmed
        .chars()
        .take_while(|&c| is_glm_name_char(c))
        .map(char::len_utf8)
        .sum();
    if name_len == 0 {
        return body.to_string();
    }
    let name = &trimmed[..name_len];
    let after_name = trimmed[name_len..].trim_start();
    if !after_name.is_empty() && !after_name.starts_with("<arg_key>") {
        return body.to_string();
    }

    let mut parts = vec![format!("{FUNC_START}{name}>\n")];
    let mut rest = after_name;
    while let Some(k_start) = rest.find("<arg_key>") {
        let after_k_open = &rest[k_start + "<arg_key>".len()..];
        let Some(k_end) = after_k_open.find("</arg_key>") else {
            break;
        };
        let key = after_k_open[..k_end].trim();
        let after_k_close = after_k_open[k_end + "</arg_key>".len()..].trim_start();
        if !after_k_close.starts_with("<arg_value>") {
            rest = after_k_close;
            continue;
        }
        let after_v_open = &after_k_close["<arg_value>".len()..];
        let Some(v_end) = after_v_open.find("</arg_value>") else {
            break;
        };
        let val = &after_v_open[..v_end];
        parts.push(format!("<parameter={key}>\n{val}\n{PARAM_END}\n"));
        rest = &after_v_open[v_end + "</arg_value>".len()..];
    }
    parts.push(FUNC_END.to_string());
    parts.concat()
}

fn strip_outer_tool_call_tags(body: &str) -> &str {
    let trimmed = body.trim();
    if let Some(inner) = trimmed
        .strip_prefix(CALL_START)
        .and_then(|s| s.strip_suffix(CALL_END))
    {
        inner
    } else {
        body
    }
}

/// Parse a `<function=NAME>\n<parameter=P>\nVALUE\n</parameter>...</function>` or GLM
/// `NAME<arg_key>P</arg_key><arg_value>VALUE</arg_value>...` tool-call body into a [`ToolCall`].
///
/// Parameter values are JSON-decoded when the tool's `schema` does not declare the parameter as `"string"`
/// (or when no schema is provided and the value is valid JSON).
pub fn parse_tool_call(body: &str, schema: Option<&Value>) -> Result<ToolCall, ParserError> {
    let inner = strip_outer_tool_call_tags(body);
    let converted = to_xml_call(inner);
    let body_trimmed = converted.trim();
    if !body_trimmed.starts_with(FUNC_START) {
        let preview: String = body_trimmed.chars().take(80).collect();
        return Err(ParserError::MalformedToolCall(preview));
    }
    let Some(gt) = body_trimmed.find('>') else {
        let preview: String = body_trimmed.chars().take(80).collect();
        return Err(ParserError::MalformedToolCall(preview));
    };
    let name = body_trimmed[FUNC_START.len()..gt].trim().to_string();
    let mut rest = &body_trimmed[gt + 1..];

    let props = schema.and_then(|s| {
        let obj = s.as_object()?;
        let params = obj
            .get("parameters")
            .or_else(|| obj.get("input_schema"))
            .and_then(|p| p.as_object())
            .unwrap_or(obj);
        params.get("properties")?.as_object()
    });

    let mut args = Map::new();
    while let Some(p_idx) = rest.find(PARAM_START) {
        rest = &rest[p_idx + PARAM_START.len()..];
        let Some(p_gt) = rest.find('>') else {
            break;
        };
        let pname = rest[..p_gt].to_string();
        rest = &rest[p_gt + 1..];
        let end = param_end(rest, true);
        let mut value = if end >= 0 {
            let v = &rest[..end as usize];
            rest = &rest[end as usize + PARAM_END.len()..];
            v
        } else {
            let v = rest;
            rest = "";
            v
        };
        if let Some(stripped) = value.strip_prefix('\n') {
            value = stripped;
        }
        if let Some(stripped) = value.strip_suffix('\n') {
            value = stripped;
        }
        let declared_type = props
            .and_then(|p| p.get(&pname))
            .and_then(|v| v.as_object())
            .and_then(|v| v.get("type"))
            .and_then(|t| t.as_str());

        if declared_type == Some("string") {
            args.insert(pname, Value::String(value.to_string()));
        } else if let Ok(parsed_val) = serde_json::from_str::<Value>(value) {
            args.insert(pname, parsed_val);
        } else {
            args.insert(pname, Value::String(value.to_string()));
        }
    }

    Ok(ToolCall::new(name, args))
}

/// Parse any supported tool call format (Qwen XML `<function=...>`, GLM `<arg_key>`, or Qwen JSON `{"name": ...}`).
pub fn parse_any_tool_call(body: &str, schema: Option<&Value>) -> Result<ToolCall, ParserError> {
    if let Ok(tc) = parse_tool_call(body, schema) {
        return Ok(tc);
    }
    if let Some(tc) = json_tool_call(body, None) {
        return Ok(tc);
    }
    let preview: String = body.trim().chars().take(80).collect();
    Err(ParserError::MalformedToolCall(preview))
}

/// Incremental parser of model output text (`<think>...</think>` reasoning and `<tool_call>` tool calls).
///
/// Matches `serve/frontend.py`'s `OutputParser`.
#[derive(Debug, Clone)]
pub struct OutputParser {
    pub state: String,
    pub glm: bool,
    pub buf: String,
    pub lead: bool,
    pub schemas: HashMap<String, Value>,
    has_tools_filter: bool,
    pub stream_tools: bool,
    pub pending: Vec<(String, Option<ToolCall>)>,
    pub rescued: usize,
    pub refused: usize,
    pub fence: String,
    pub line: String,
    pub ticks: usize,
    pub recover: bool,
    pub bare: bool,
    pub batch: bool,
    at_start: bool,
    lead_think: bool,
    sp: usize,
    ss: String,
    scall: Option<ToolCall>,
    sfirst: bool,
    sval_started: bool,
    sdeclared: HashMap<String, String>,
}

impl Default for OutputParser {
    fn default() -> Self {
        Self::new(true, None, false, false, false)
    }
}

impl OutputParser {
    /// Create a new [`OutputParser`].
    ///
    /// - `thinking`: whether the prompt already opened `<think>` (`true` starts in `"reasoning"` state,
    ///   `false` starts in `"content"` state; in both cases a leading `<think>` emitted in output enters `"reasoning"`).
    /// - `tools`: optional slice of tool definitions (`{"name": ..., "parameters": ...}`).
    /// - `stream_tools`: when `true`, emits `"tool_start"` and incremental `"tool_args"` events while a call streams.
    /// - `recover`: when `true`, enables tool-call recovery (`<parameter=NAME>` opener, JSON inside `<tool_call>`,
    ///   bare `<function=NAME>` at line start outside code, and multiple calls in one `<tool_call>`).
    /// - `glm`: when `true`, enables GLM `<arg_key>` tool-call conversion in `<tool_call>`.
    pub fn new(
        thinking: bool,
        tools: Option<&[Value]>,
        stream_tools: bool,
        recover: bool,
        glm: bool,
    ) -> Self {
        let mut schemas = HashMap::new();
        let has_tools_filter = tools.is_some();
        if let Some(tool_list) = tools {
            for t in tool_list {
                if let Some(obj) = t.as_object() {
                    let fn_obj = obj
                        .get("function")
                        .and_then(|f| f.as_object())
                        .unwrap_or(obj);
                    if let Some(name) = fn_obj.get("name").and_then(|n| n.as_str()) {
                        schemas.insert(name.to_string(), Value::Object(fn_obj.clone()));
                    }
                }
            }
        }
        let mut parser = Self {
            state: if thinking {
                "reasoning".to_string()
            } else {
                "content".to_string()
            },
            glm,
            buf: String::new(),
            lead: false,
            schemas,
            has_tools_filter,
            stream_tools,
            pending: Vec::new(),
            rescued: 0,
            refused: 0,
            fence: String::new(),
            line: String::new(),
            ticks: 0,
            recover,
            bare: false,
            batch: false,
            at_start: true,
            lead_think: false,
            sp: 0,
            ss: "name".to_string(),
            scall: None,
            sfirst: true,
            sval_started: false,
            sdeclared: HashMap::new(),
        };
        parser.reset_scan();
        parser
    }

    fn reset_scan(&mut self) {
        self.sp = 0;
        self.ss = "name".to_string();
        self.scall = None;
        self.sfirst = true;
        self.sval_started = false;
        self.sdeclared.clear();
    }

    /// Track markdown code fences (` ``` ` or `~~~`), current line, and inline backticks.
    pub fn track(&mut self, text: &str) -> String {
        if !text.contains('\n') {
            self.line.push_str(text);
            return text.to_string();
        }
        let parts: Vec<&str> = text.split('\n').collect();
        for (k, part) in parts.iter().enumerate() {
            if k < parts.len() - 1 {
                self.line.push_str(part);
                let s = self.line.trim_start();
                if !self.fence.is_empty() {
                    let f3 = self.fence.repeat(3);
                    if s.starts_with(&f3) {
                        self.fence.clear();
                    }
                } else if (s.starts_with("```") && !s[3..].contains("```"))
                    || (s.starts_with("~~~") && !s[3..].contains("~~~"))
                {
                    self.fence = s[..1].to_string();
                } else if s.is_empty() {
                    self.ticks = 0;
                } else {
                    self.ticks += self.line.matches('`').count();
                }
                self.line.clear();
            } else {
                self.line.push_str(part);
            }
        }
        text.to_string()
    }

    /// Whether the reasoning text so far puts a `<tool_call>` at the start of a line,
    /// outside a code fence and outside inline code.
    pub fn opener_ok(&self) -> bool {
        self.fence.is_empty() && self.line.trim().is_empty() && self.ticks.is_multiple_of(2)
    }

    /// Whether the text so far leaves the next character inside a code fence or inline code.
    pub fn in_code(&self) -> bool {
        !self.fence.is_empty() || !(self.ticks + self.line.matches('`').count()).is_multiple_of(2)
    }

    fn ok_at(&mut self, p: usize) -> bool {
        let snap = (self.fence.clone(), self.line.clone(), self.ticks);
        let prefix = self.buf[..p].to_string();
        self.track(&prefix);
        let ok = self.opener_ok();
        self.fence = snap.0;
        self.line = snap.1;
        self.ticks = snap.2;
        ok
    }

    fn follower(&self, after: &str) -> &'static str {
        if after.starts_with(FUNC_START) {
            return "call";
        }
        if after.starts_with(PARAM_START) {
            let Some(gt) = after.find('>') else {
                return if !after.contains('\n') {
                    "wait"
                } else {
                    "text"
                };
            };
            let name = &after[PARAM_START.len()..gt];
            return if self.schemas.contains_key(name) {
                "param"
            } else {
                "text"
            };
        }
        if after.starts_with('{') {
            return if !self.schemas.is_empty() || !self.has_tools_filter {
                "json"
            } else {
                "text"
            };
        }
        if after.starts_with("```") || (!after.is_empty() && "```".starts_with(after)) {
            let rx = Regex::new(r"(?i)^```[ \t]*(?:json)?[ \t]*\r?\n\s*").unwrap();
            if let Ok(Some(m)) = rx.find(after) {
                let rest = &after[m.end()..];
                if rest.starts_with('{') {
                    return if !self.schemas.is_empty() || !self.has_tools_filter {
                        "json"
                    } else {
                        "text"
                    };
                }
                return if rest.is_empty() { "wait" } else { "text" };
            }
            return if !after.contains('\n') {
                "wait"
            } else {
                "text"
            };
        }
        if after.is_empty() || FUNC_START.starts_with(after) || PARAM_START.starts_with(after) {
            return "wait";
        }
        "text"
    }

    fn bare_opener(&mut self) -> (isize, bool) {
        let mut search_from = 0;
        while let Some(rel) = self.buf[search_from..].find(FUNC_START) {
            let p = search_from + rel;
            if self.ok_at(p) {
                let gt = self.buf[p..].find('>').map(|r| p + r);
                let name_end = gt.unwrap_or(self.buf.len());
                let name = &self.buf[p + FUNC_START.len()..name_end];
                if gt.is_none() && !name.chars().any(char::is_whitespace) {
                    return (p as isize, false);
                }
                if gt.is_some() && self.schemas.contains_key(name) {
                    return (p as isize, true);
                }
            }
            search_from = p + 1;
        }
        (-1, false)
    }

    fn quoted_think_end(&self, i: usize) -> Option<bool> {
        let before = if i > 0 {
            self.buf[..i].chars().next_back()
        } else {
            self.line.chars().next_back()
        };
        let Some(q) = before.filter(|c| matches!(c, '"' | '\'' | '`')) else {
            return Some(false);
        };
        let full_before = format!("{}{}", self.line, &self.buf[..i]);
        let cur_line = full_before.rsplit('\n').next().unwrap_or(&full_before);
        if q == '`' {
            if (self.ticks + cur_line.matches('`').count()).is_multiple_of(2) {
                return Some(false);
            }
        } else {
            let mut chars_rev = cur_line.chars().rev();
            let _ = chars_rev.next();
            let prev_alnum = chars_rev.next().is_some_and(|c| c.is_alphanumeric());
            if cur_line.matches(q).count().is_multiple_of(2) || prev_alnum {
                return Some(false);
            }
        }
        let after = &self.buf[i + THINK_END.len()..];
        let first_after = after.chars().next()?;
        Some(!matches!(first_after, '\r' | '\n'))
    }

    fn release(&mut self, deliver: bool) -> Vec<ParseEvent> {
        let mut out = Vec::new();
        let pending = std::mem::take(&mut self.pending);
        for (raw, call) in pending {
            if let Some(c) = call {
                if deliver {
                    out.push(ParseEvent::tool_call(c));
                    self.rescued += 1;
                } else {
                    self.refused += 1;
                    if !raw.is_empty() {
                        let tracked = self.track(&raw);
                        out.push(ParseEvent::reasoning(tracked));
                    }
                }
            } else if !raw.is_empty() {
                let tracked = self.track(&raw);
                out.push(ParseEvent::reasoning(tracked));
            }
        }
        out
    }

    fn hold(text: &str, tags: &[&str]) -> usize {
        let mut best = 0;
        for tag in tags {
            for n in 1..tag.len() {
                if text.ends_with(&tag[..n]) {
                    best = best.max(n);
                }
            }
        }
        best
    }

    fn scan(&mut self) -> Vec<ParseEvent> {
        let mut out = Vec::new();
        loop {
            let rest = &self.buf[self.sp..];
            if self.ss == "name" {
                let Some(a) = rest.find(FUNC_START) else {
                    return out;
                };
                let Some(b_rel) = rest[a + FUNC_START.len()..].find('>') else {
                    return out;
                };
                let b = a + FUNC_START.len() + b_rel;
                let name = rest[a + FUNC_START.len()..b].trim().to_string();
                let call = ToolCall::new(name.clone(), Map::new());
                let mut sdeclared = HashMap::new();
                if let Some(schema) = self.schemas.get(&name) {
                    if let Some(props) = schema
                        .get("parameters")
                        .and_then(|p| p.get("properties"))
                        .and_then(|p| p.as_object())
                    {
                        for (k, v) in props {
                            if let Some(ty) = v.get("type").and_then(|t| t.as_str()) {
                                sdeclared.insert(k.clone(), ty.to_string());
                            }
                        }
                    }
                }
                self.sdeclared = sdeclared;
                self.scall = Some(call.clone());
                out.push(ParseEvent::tool_start(call.clone()));
                out.push(ParseEvent::tool_args("{", Some(call)));
                self.sp += b + 1;
                self.ss = "between".to_string();
            } else if self.ss == "between" {
                let stripped = rest.trim_start();
                self.sp += rest.len() - stripped.len();
                if stripped.starts_with(PARAM_START) {
                    let Some(b) = stripped.find('>') else {
                        return out;
                    };
                    let pname = stripped[PARAM_START.len()..b].to_string();
                    let key_json = serde_json::to_string(&pname).unwrap_or_default();
                    let prefix = if self.sfirst { "" } else { "," };
                    self.sfirst = false;
                    out.push(ParseEvent::tool_args(
                        format!("{prefix}{key_json}:"),
                        self.scall.clone(),
                    ));
                    self.sp += b + 1;
                    if self.sdeclared.get(&pname).map(String::as_str) == Some("string") {
                        out.push(ParseEvent::tool_args("\"", self.scall.clone()));
                        self.ss = "str".to_string();
                        self.sval_started = false;
                    } else {
                        self.ss = "raw".to_string();
                    }
                } else if stripped.starts_with(FUNC_END) {
                    out.push(ParseEvent::tool_args("}", self.scall.clone()));
                    self.sp += FUNC_END.len();
                    self.ss = "done".to_string();
                } else {
                    return out;
                }
            } else if self.ss == "str" {
                let mut rest = &self.buf[self.sp..];
                if !self.sval_started {
                    if rest.is_empty() {
                        return out;
                    }
                    if rest.starts_with('\n') {
                        self.sp += 1;
                        rest = &self.buf[self.sp..];
                    }
                    self.sval_started = true;
                }
                let end = param_end(rest, false);
                if end >= 0 {
                    let end_u = end as usize;
                    let mut value = &rest[..end_u];
                    if let Some(stripped) = value.strip_suffix('\n') {
                        value = stripped;
                    }
                    let encoded = serde_json::to_string(value).unwrap_or_else(|_| "\"\"".into());
                    let inner = &encoded[1..encoded.len() - 1];
                    out.push(ParseEvent::tool_args(
                        format!("{inner}\""),
                        self.scall.clone(),
                    ));
                    self.sp += end_u + PARAM_END.len();
                    self.ss = "between".to_string();
                    continue;
                }
                let mut safe = if end == -2 {
                    rest.find(PARAM_END).unwrap_or(0)
                } else {
                    rest.len().saturating_sub(Self::hold(rest, &[PARAM_END]))
                };
                if safe > 0 && rest.as_bytes()[safe - 1] == b'\n' {
                    safe -= 1;
                }
                if safe > 0 {
                    let chunk = &rest[..safe];
                    let encoded = serde_json::to_string(chunk).unwrap_or_else(|_| "\"\"".into());
                    let inner = &encoded[1..encoded.len() - 1];
                    if !inner.is_empty() {
                        out.push(ParseEvent::tool_args(inner, self.scall.clone()));
                    }
                    self.sp += safe;
                }
                return out;
            } else if self.ss == "raw" {
                let end = param_end(rest, false);
                if end < 0 {
                    return out;
                }
                let end_u = end as usize;
                let mut value = &rest[..end_u];
                if let Some(s) = value.strip_prefix('\n') {
                    value = s;
                }
                if let Some(s) = value.strip_suffix('\n') {
                    value = s;
                }
                let serialized = if let Ok(v) = serde_json::from_str::<Value>(value) {
                    crate::template::to_python_json(&v, None)
                } else {
                    serde_json::to_string(value).unwrap_or_default()
                };
                out.push(ParseEvent::tool_args(serialized, self.scall.clone()));
                self.sp += end_u + PARAM_END.len();
                self.ss = "between".to_string();
            } else {
                return out;
            }
        }
    }

    fn close_scan(&mut self) -> Vec<ParseEvent> {
        let mut out = Vec::new();
        if self.scall.is_none() || self.ss == "done" {
            return out;
        }
        let mut tail = String::new();
        if self.ss == "str" {
            tail.push('"');
        } else if self.ss == "raw" {
            let trimmed = self.buf[self.sp..].trim_matches('\n');
            tail.push_str(&serde_json::to_string(trimmed).unwrap_or_default());
        }
        tail.push('}');
        out.push(ParseEvent::tool_args(tail, self.scall.clone()));
        self.ss = "done".to_string();
        out
    }

    fn finish_call(&mut self, body: &str) -> Result<Vec<ParseEvent>, ParserError> {
        let xml = to_xml_call(body);
        let xml_trimmed = xml.trim();
        let name = xml_trimmed
            .strip_prefix(FUNC_START)
            .and_then(|s| s.split_once('>'))
            .map(|(n, _)| n.trim().to_string())
            .unwrap_or_default();
        let bare = self.bare;
        self.bare = false;
        match parse_tool_call(body, self.schemas.get(&name)) {
            Ok(mut call) => {
                if let Some(sc) = &self.scall {
                    call.id = sc.id.clone();
                }
                self.reset_scan();
                self.state = "content".to_string();
                self.lead = true;
                Ok(vec![ParseEvent::tool_call(call)])
            }
            Err(err) => {
                if !self.recover {
                    return Err(err);
                }
                self.reset_scan();
                self.state = "content".to_string();
                let raw = if bare {
                    body.to_string()
                } else {
                    format!("{CALL_START}{body}{CALL_END}")
                };
                let tracked = self.track(&raw);
                Ok(vec![ParseEvent::content(tracked)])
            }
        }
    }

    /// Feed a text delta to the parser and return any completed [`ParseEvent`] items.
    pub fn feed(&mut self, delta: &str) -> Vec<ParseEvent> {
        self.feed_checked(delta).unwrap_or_default()
    }

    /// Feed a text delta to the parser, returning a [`ParserError`] if a malformed `<function=...>`
    /// call is encountered when `recover == false`.
    pub fn feed_checked(&mut self, delta: &str) -> Result<Vec<ParseEvent>, ParserError> {
        self.buf.push_str(delta);
        let mut out = Vec::new();

        loop {
            if self.at_start {
                if self.buf.is_empty() {
                    return Ok(out);
                }
                if self.buf.starts_with(THINK_START) {
                    self.buf.drain(..THINK_START.len());
                    self.state = "reasoning".to_string();
                    self.lead_think = true;
                    self.at_start = false;
                } else if THINK_START.starts_with(&self.buf) {
                    return Ok(out);
                } else {
                    self.at_start = false;
                }
            }
            if self.lead_think {
                let stripped = self.buf.trim_start_matches('\n');
                if stripped.is_empty() {
                    self.buf.clear();
                    return Ok(out);
                }
                let trim_len = self.buf.len() - stripped.len();
                if trim_len > 0 {
                    self.buf.drain(..trim_len);
                }
                self.lead_think = false;
            }

            if self.state == "rcall" {
                let body = &self.buf[CALL_START.len()..];
                let end = call_end(body);
                let think = body.find(THINK_END).map(|i| i as isize).unwrap_or(-1);
                if end >= 0 && (think < 0 || think >= end) {
                    let end_u = end as usize;
                    let body_slice = &body[..end_u];
                    let xml = to_xml_call(body_slice);
                    let name = xml
                        .trim()
                        .strip_prefix(FUNC_START)
                        .and_then(|s| s.split_once('>'))
                        .map(|(n, _)| n.trim().to_string())
                        .unwrap_or_default();
                    let call = if self.schemas.contains_key(&name) {
                        parse_tool_call(body_slice, self.schemas.get(&name)).ok()
                    } else {
                        None
                    };
                    let raw_end = CALL_START.len() + end_u + CALL_END.len();
                    let raw = self.buf[..raw_end].to_string();
                    let tail = body[end_u + CALL_END.len()..].to_string();
                    if call.is_some() || !self.pending.is_empty() {
                        self.pending.push((raw, call));
                    } else {
                        let tracked = self.track(&raw);
                        out.push(ParseEvent::reasoning(tracked));
                    }
                    self.buf = tail;
                    self.state = "reasoning".to_string();
                } else if think >= 0 {
                    let think_u = think as usize;
                    let before_think = self.buf[..CALL_START.len() + think_u].to_string();
                    let tail = body[think_u..].to_string();
                    out.extend(self.release(false));
                    let tracked = self.track(&before_think);
                    out.push(ParseEvent::reasoning(tracked));
                    self.buf = tail;
                    self.state = "reasoning".to_string();
                } else if body.len() > RCALL_MAX {
                    out.extend(self.release(false));
                    let all = std::mem::take(&mut self.buf);
                    let tracked = self.track(&all);
                    out.push(ParseEvent::reasoning(tracked));
                    self.state = "reasoning".to_string();
                } else {
                    return Ok(out);
                }
            } else if self.state == "reasoning" {
                if !self.pending.is_empty() {
                    let stripped_len = self.buf.trim_start().len();
                    if stripped_len < self.buf.len() {
                        let ws_len = self.buf.len() - stripped_len;
                        let ws = self.buf[..ws_len].to_string();
                        if let Some(last) = self.pending.last_mut() {
                            last.0.push_str(&ws);
                        }
                        self.buf.drain(..ws_len);
                    }
                    if self.buf.is_empty() {
                        return Ok(out);
                    }
                    if self.buf.starts_with(CALL_START) {
                        self.state = "rcall".to_string();
                    } else if self.buf.starts_with(THINK_END) {
                        out.extend(self.release(true));
                    } else if CALL_START.starts_with(&self.buf) || THINK_END.starts_with(&self.buf)
                    {
                        return Ok(out);
                    } else {
                        out.extend(self.release(false));
                    }
                    continue;
                }

                let i = self.buf.find(THINK_END);
                let tool = if !self.schemas.is_empty() {
                    self.buf.find(CALL_START)
                } else {
                    None
                };

                if let Some(t) = tool {
                    if i.is_none_or(|think_pos| t < think_pos) {
                        if t > 0 {
                            let prefix = self.buf[..t].to_string();
                            let tracked = self.track(&prefix);
                            out.push(ParseEvent::reasoning(tracked));
                        }
                        if self.opener_ok() {
                            self.buf.drain(..t);
                            self.state = "rcall".to_string();
                        } else {
                            let tracked = self.track(CALL_START);
                            out.push(ParseEvent::reasoning(tracked));
                            self.buf.drain(..t + CALL_START.len());
                        }
                        continue;
                    }
                }

                let Some(i) = i else {
                    let keep = if !self.schemas.is_empty() {
                        Self::hold(&self.buf, &[THINK_END, CALL_START])
                    } else {
                        Self::hold(&self.buf, &[THINK_END])
                    };
                    if self.buf.len() > keep {
                        let emit_len = self.buf.len() - keep;
                        let chunk = self.buf[..emit_len].to_string();
                        let tracked = self.track(&chunk);
                        out.push(ParseEvent::reasoning(tracked));
                        self.buf.drain(..emit_len);
                    }
                    return Ok(out);
                };

                let quoted = self.quoted_think_end(i);
                let Some(is_quoted) = quoted else {
                    if i > 0 {
                        let prefix = self.buf[..i].to_string();
                        let tracked = self.track(&prefix);
                        out.push(ParseEvent::reasoning(tracked));
                        self.buf.drain(..i);
                    }
                    return Ok(out);
                };
                if is_quoted {
                    let end_pos = i + THINK_END.len();
                    let chunk = self.buf[..end_pos].to_string();
                    let tracked = self.track(&chunk);
                    out.push(ParseEvent::reasoning(tracked));
                    self.buf.drain(..end_pos);
                    continue;
                }
                if i > 0 {
                    let prefix = self.buf[..i].to_string();
                    let tracked = self.track(&prefix);
                    out.push(ParseEvent::reasoning(tracked));
                }
                self.buf.drain(..i + THINK_END.len());
                self.state = "content".to_string();
                self.lead = true;
                self.fence.clear();
                self.line.clear();
                self.ticks = 0;
            } else if self.state == "content" {
                if self.lead {
                    let stripped = self.buf.trim_start_matches('\n');
                    if stripped.is_empty() {
                        self.buf.clear();
                        return Ok(out);
                    }
                    let trim_len = self.buf.len() - stripped.len();
                    if trim_len > 0 {
                        self.buf.drain(..trim_len);
                    }
                    self.lead = false;
                }

                let i = self.buf.find(CALL_START);
                let (b, decided) = if self.recover && !self.schemas.is_empty() {
                    self.bare_opener()
                } else {
                    (-1, false)
                };

                if b >= 0 && i.is_none_or(|call_pos| (b as usize) < call_pos) {
                    let b_u = b as usize;
                    let mut j = b_u;
                    while j > 0 && self.buf.as_bytes()[j - 1] == b'\n' {
                        j -= 1;
                    }
                    if j > 0 && !self.buf[..j].trim().is_empty() {
                        let prefix = self.buf[..j].to_string();
                        let tracked = self.track(&prefix);
                        out.push(ParseEvent::content(tracked));
                    }
                    if !decided {
                        self.buf.drain(..j);
                        return Ok(out);
                    }
                    self.buf.drain(..b_u);
                    self.state = "call".to_string();
                    self.bare = true;
                    continue;
                }

                let Some(i) = i else {
                    let mut j = self
                        .buf
                        .len()
                        .saturating_sub(Self::hold(&self.buf, &[CALL_START]));
                    if self.recover && !self.schemas.is_empty() {
                        let k = self.buf.rfind('\n').map(|pos| pos + 1).unwrap_or(0);
                        let tail = &self.buf[k..];
                        if !tail.is_empty() && FUNC_START.starts_with(tail) && self.ok_at(k) {
                            j = j.min(k);
                        }
                    }
                    while j > 0 && self.buf.as_bytes()[j - 1] == b'\n' {
                        j -= 1;
                    }
                    if j > 0 {
                        let prefix = self.buf[..j].to_string();
                        let tracked = self.track(&prefix);
                        out.push(ParseEvent::content(tracked));
                        self.buf.drain(..j);
                    }
                    return Ok(out);
                };

                let snap = (self.fence.clone(), self.line.clone(), self.ticks);
                let before_call = self.buf[..i].to_string();
                self.track(&before_call);
                let in_code = self.in_code();
                self.fence = snap.0;
                self.line = snap.1;
                self.ticks = snap.2;
                if in_code {
                    let end_pos = i + CALL_START.len();
                    let chunk = self.buf[..end_pos].to_string();
                    let tracked = self.track(&chunk);
                    out.push(ParseEvent::content(tracked));
                    self.buf.drain(..end_pos);
                    continue;
                }

                let after_str = self.buf[i + CALL_START.len()..].trim_start().to_string();
                let mut after = after_str.as_str();
                let kind = if self.recover {
                    Some(self.follower(after))
                } else {
                    None
                };

                if matches!(kind, Some("param" | "json")) {
                    if i > 0 && !self.buf[..i].trim().is_empty() {
                        let prefix = self.buf[..i].trim_end_matches('\n').to_string();
                        let tracked = self.track(&prefix);
                        out.push(ParseEvent::content(tracked));
                    }
                    let mut rest = self.buf[i + CALL_START.len()..].to_string();
                    if kind == Some("param") {
                        let at = rest.len() - after.len();
                        let replaced = format!(
                            "{}{}{}",
                            &rest[..at],
                            FUNC_START,
                            &after[PARAM_START.len()..]
                        );
                        rest = replaced;
                    }
                    self.buf = rest;
                    self.state = if kind == Some("param") {
                        "call".to_string()
                    } else {
                        "json".to_string()
                    };
                    continue;
                }
                if kind == Some("wait") {
                    after = "";
                }

                let looks_like_glm = self.glm || after.contains("<arg_key>");
                if looks_like_glm
                    && !after.is_empty()
                    && !after.starts_with(FUNC_START)
                    && !FUNC_START.starts_with(after)
                {
                    let e_opt = self.buf[i + CALL_START.len()..]
                        .find(CALL_END)
                        .map(|rel| i + CALL_START.len() + rel);
                    let Some(e) = e_opt else {
                        if i > 0 && !self.buf[..i].trim().is_empty() {
                            let prefix = self.buf[..i].trim_end_matches('\n').to_string();
                            let tracked = self.track(&prefix);
                            out.push(ParseEvent::content(tracked));
                            self.buf.drain(..i);
                        }
                        return Ok(out);
                    };
                    let conv = to_xml_call(&self.buf[i + CALL_START.len()..e]);
                    if conv.trim_start().starts_with(FUNC_START) {
                        if i > 0 && !self.buf[..i].trim().is_empty() {
                            let prefix = self.buf[..i].trim_end_matches('\n').to_string();
                            let tracked = self.track(&prefix);
                            out.push(ParseEvent::content(tracked));
                        }
                        let tail = self.buf[e + CALL_END.len()..].to_string();
                        self.buf = format!("{conv}{CALL_END}{tail}");
                        self.state = "call".to_string();
                        continue;
                    }
                }

                if !after.is_empty()
                    && !after.starts_with(FUNC_START)
                    && !FUNC_START.starts_with(after)
                {
                    let end_pos = i + CALL_START.len();
                    let chunk = self.buf[..end_pos].to_string();
                    let tracked = self.track(&chunk);
                    out.push(ParseEvent::content(tracked));
                    self.buf.drain(..end_pos);
                    continue;
                }
                if !after.starts_with(FUNC_START) {
                    let mut j = i;
                    while j > 0 && self.buf.as_bytes()[j - 1] == b'\n' {
                        j -= 1;
                    }
                    if j > 0 {
                        let prefix = self.buf[..j].to_string();
                        let tracked = self.track(&prefix);
                        out.push(ParseEvent::content(tracked));
                        self.buf.drain(..j);
                    }
                    return Ok(out);
                }
                if i > 0 && !self.buf[..i].trim().is_empty() {
                    let prefix = self.buf[..i].trim_end_matches('\n').to_string();
                    let tracked = self.track(&prefix);
                    out.push(ParseEvent::content(tracked));
                }
                self.buf.drain(..i + CALL_START.len());
                self.state = "call".to_string();
            } else if self.state == "json" {
                let Some(i) = self.buf.find(CALL_END) else {
                    return Ok(out);
                };
                let schemas_ref = if self.has_tools_filter {
                    Some(&self.schemas)
                } else {
                    None
                };
                let call = json_tool_call(self.buf[..i].trim(), schemas_ref);
                let is_call = call.is_some();
                if let Some(c) = call {
                    out.push(ParseEvent::tool_call(c));
                } else {
                    let raw = format!("{CALL_START}{}", &self.buf[..i + CALL_END.len()]);
                    let tracked = self.track(&raw);
                    out.push(ParseEvent::content(tracked));
                }
                self.buf.drain(..i + CALL_END.len());
                self.state = "content".to_string();
                self.lead = is_call;
            } else {
                // state == "call"
                let mut drop: Option<usize> = None;
                let mut i: isize;
                if self.bare {
                    i = function_end(&self.buf);
                    if i >= 0 {
                        let i_u = i as usize;
                        let after = self.buf[i_u..].trim_start();
                        if after.starts_with(CALL_END) {
                            drop = Some((self.buf.len() - i_u) - after.len() + CALL_END.len());
                        } else if !after.is_empty() && !CALL_END.starts_with(after) {
                            drop = Some(0);
                        }
                    }
                } else {
                    i = call_end(&self.buf);
                    if i >= 0 {
                        drop = Some(CALL_END.len());
                    }
                    if self.recover {
                        let f = function_end(&self.buf);
                        if f >= 0 {
                            let f_u = f as usize;
                            let nxt_trimmed = self.buf[f_u..].trim_start().to_string();
                            let mut nxt = nxt_trimmed.as_str();
                            if nxt.starts_with(PARAM_START) && self.follower(nxt) == "param" {
                                let at = self.buf.len() - nxt.len();
                                self.buf = format!(
                                    "{}{}{}",
                                    &self.buf[..at],
                                    FUNC_START,
                                    &nxt[PARAM_START.len()..]
                                );
                                nxt = &self.buf[at..];
                            }
                            if nxt.starts_with(FUNC_START) {
                                i = f;
                                drop = Some(0);
                                self.batch = true;
                            }
                        }
                    }
                }

                if self.stream_tools {
                    if i >= 0 {
                        let i_u = i as usize;
                        let whole = self.buf.clone();
                        self.buf.truncate(i_u);
                        out.extend(self.scan());
                        out.extend(self.close_scan());
                        self.buf = whole;
                    } else {
                        out.extend(self.scan());
                    }
                }

                if i < 0 || drop.is_none() {
                    return Ok(out);
                }
                let i_u = i as usize;
                let drop_u = drop.unwrap();
                let body = self.buf[..i_u].to_string();
                self.buf.drain(..i_u + drop_u);
                let batch = self.batch;
                self.batch = false;
                let finished = self.finish_call(&body)?;
                let last_is_call = finished
                    .last()
                    .map(|e| e.kind == "tool_call")
                    .unwrap_or(false);
                out.extend(finished);
                if batch && last_is_call {
                    self.state = "call".to_string();
                    self.lead = false;
                }
            }
        }
    }

    /// Flush any remaining buffered text at the end of generation.
    ///
    /// `reason` indicates how generation ended (`None` or `Some("stop")` rescues pending calls
    /// from reasoning; `"length"` / `"cancel"` / `"error"` refuses them back into reasoning text).
    pub fn finish(&mut self, reason: Option<&str>) -> Vec<ParseEvent> {
        self.finish_checked(reason).unwrap_or_default()
    }

    /// Checked version of [`OutputParser::finish`].
    pub fn finish_checked(&mut self, reason: Option<&str>) -> Result<Vec<ParseEvent>, ParserError> {
        self.at_start = false;
        self.lead_think = false;
        let mut out = Vec::new();
        if !self.pending.is_empty() {
            let deliver = matches!(reason, None | Some("stop"))
                && self.state == "reasoning"
                && self.buf.is_empty();
            out.extend(self.release(deliver));
        }
        if self.state == "call" && self.bare && function_end(&self.buf) >= 0 {
            let i_u = function_end(&self.buf) as usize;
            let body = self.buf[..i_u].to_string();
            let rest = self.buf[i_u..].trim().to_string();
            self.buf.clear();
            out.extend(self.finish_call(&body)?);
            if !rest.is_empty() && !CALL_END.starts_with(&rest) {
                let tracked = self.track(&rest);
                out.push(ParseEvent::content(tracked));
            }
            return Ok(out);
        }
        if self.state == "call" && self.stream_tools && self.scall.is_some() {
            out.extend(self.scan());
            if self.ss == "done" {
                if let Some(call) = self.scall.take() {
                    out.push(ParseEvent::tool_call(call));
                }
            }
            self.buf.clear();
            self.reset_scan();
            return Ok(out);
        }
        if self.state == "reasoning" && self.buf == THINK_END {
            self.buf.clear();
            self.state = "content".to_string();
        }
        if !self.buf.is_empty() {
            let kind = match self.state.as_str() {
                "reasoning" | "rcall" => "reasoning",
                _ => "content",
            };
            let text = if matches!(self.state.as_str(), "call" | "json") && !self.bare {
                format!("{CALL_START}{}", self.buf)
            } else {
                self.buf.clone()
            };
            self.buf.clear();
            out.push(ParseEvent {
                kind: kind.to_string(),
                text,
                call: None,
            });
        }
        Ok(out)
    }
}

/// Incremental streaming parser that wraps [`OutputParser`] and translates events into
/// OpenAI-compatible [`StreamDelta`] chunks (`reasoning_content`, `content`, and `tool_calls`
/// deltas with `index`, `id`, `function.name`, and `function.arguments`).
#[derive(Debug, Clone)]
pub struct StreamParser {
    pub inner: OutputParser,
    calls_count: usize,
    streamed_indices: HashMap<String, usize>,
    finished_ids: HashSet<String>,
    completed_calls: Vec<ToolCall>,
}

impl StreamParser {
    pub fn new(
        thinking: bool,
        tools: Option<&[Value]>,
        stream_tools: bool,
        recover: bool,
        glm: bool,
    ) -> Self {
        Self {
            inner: OutputParser::new(thinking, tools, stream_tools, recover, glm),
            calls_count: 0,
            streamed_indices: HashMap::new(),
            finished_ids: HashSet::new(),
            completed_calls: Vec::new(),
        }
    }

    /// Feed a text delta and return OpenAI-style [`StreamDelta`] chunks.
    pub fn feed(&mut self, delta: &str) -> Vec<StreamDelta> {
        let events = self.inner.feed(delta);
        self.events_to_deltas(events)
    }

    /// Finish the stream and return any final [`StreamDelta`] chunks.
    pub fn finish(&mut self, reason: Option<&str>) -> Vec<StreamDelta> {
        let events = self.inner.finish(reason);
        self.events_to_deltas(events)
    }

    /// Return all completed [`ToolCall`] items collected so far.
    pub fn completed_tool_calls(&self) -> &[ToolCall] {
        &self.completed_calls
    }

    /// Whether at least one tool call was emitted and all announced tool calls completed cleanly.
    pub fn has_complete_tool_calls(&self) -> bool {
        self.calls_count > 0
            && self
                .streamed_indices
                .keys()
                .all(|id| self.finished_ids.contains(id))
    }

    fn events_to_deltas(&mut self, events: Vec<ParseEvent>) -> Vec<StreamDelta> {
        let mut out = Vec::new();
        for ev in events {
            match ev.kind.as_str() {
                "reasoning" if !ev.text.is_empty() => {
                    out.push(StreamDelta {
                        reasoning_content: Some(ev.text),
                        content: None,
                        tool_calls: Vec::new(),
                    });
                }
                "content" if !ev.text.is_empty() => {
                    out.push(StreamDelta {
                        reasoning_content: None,
                        content: Some(ev.text),
                        tool_calls: Vec::new(),
                    });
                }
                "tool_start" => {
                    if let Some(call) = ev.call {
                        let idx = self.calls_count;
                        self.calls_count += 1;
                        self.streamed_indices.insert(call.id.clone(), idx);
                        out.push(StreamDelta {
                            reasoning_content: None,
                            content: None,
                            tool_calls: vec![ToolCallDelta {
                                index: idx,
                                id: Some(call.id),
                                kind: Some("function".to_string()),
                                function: ToolCallFunctionDelta {
                                    name: Some(call.name),
                                    arguments: String::new(),
                                },
                            }],
                        });
                    }
                }
                "tool_args" => {
                    if let Some(call) = ev.call {
                        if let Some(&idx) = self.streamed_indices.get(&call.id) {
                            out.push(StreamDelta {
                                reasoning_content: None,
                                content: None,
                                tool_calls: vec![ToolCallDelta {
                                    index: idx,
                                    id: None,
                                    kind: None,
                                    function: ToolCallFunctionDelta {
                                        name: None,
                                        arguments: ev.text,
                                    },
                                }],
                            });
                        }
                    }
                }
                "tool_call" => {
                    if let Some(call) = ev.call {
                        self.completed_calls.push(call.clone());
                        if self.streamed_indices.contains_key(&call.id) {
                            self.finished_ids.insert(call.id);
                        } else {
                            let idx = self.calls_count;
                            self.calls_count += 1;
                            self.streamed_indices.insert(call.id.clone(), idx);
                            self.finished_ids.insert(call.id.clone());
                            let args_json = crate::template::to_python_json(
                                &Value::Object(call.arguments),
                                None,
                            );
                            out.push(StreamDelta {
                                reasoning_content: None,
                                content: None,
                                tool_calls: vec![ToolCallDelta {
                                    index: idx,
                                    id: Some(call.id),
                                    kind: Some("function".to_string()),
                                    function: ToolCallFunctionDelta {
                                        name: Some(call.name),
                                        arguments: args_json,
                                    },
                                }],
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }
}
