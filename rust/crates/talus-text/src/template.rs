//! Jinja chat template engine, Hugging Face `{% generation %}` stripping,
//! literal special-token protection, and OpenAI / Anthropic request normalization.
//!
//! Direct port of `serve/frontend.py`'s `ChatTemplate`, `mark_think_literals`,
//! `unmark_think_literals`, `literal_tags`, `effort_kwargs`, `budget_effort`,
//! `pin_billing_stamp`, `openai_to_messages`, and `anthropic_to_messages`.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::{LazyLock, Mutex};

use aho_corasick::{AhoCorasickBuilder, MatchKind};
use fancy_regex::Regex;
use minijinja::value::{Kwargs, Value as MjValue, ValueKind};
use minijinja::{AutoEscape, Environment, Error as MjError, ErrorKind};
use serde::{Deserialize, Serialize};
use serde_json::ser::{Formatter, PrettyFormatter, Serializer};
use serde_json::{json, Map, Value};
use thiserror::Error;

use crate::parser::parse_tool_call;

/// Errors produced when loading or rendering a [`ChatTemplate`] or normalizing a request.
#[derive(Debug, Error)]
pub enum TemplateError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("template error: {0}")]
    Jinja(String),
    #[error("{0}")]
    Request(String),
}

/// Alias matching `serve/frontend.py`'s `TemplateRequestError`.
pub type TemplateRequestError = TemplateError;

// ---------------------------------------------------------------------------
// Python-compatible `json.dumps(..., ensure_ascii=False)` formatter
// ---------------------------------------------------------------------------

struct PythonCompactFormatter;

impl Formatter for PythonCompactFormatter {
    #[inline]
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    #[inline]
    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    #[inline]
    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }
}

/// Format a [`serde_json::Value`] identically to Python's `json.dumps(val, ensure_ascii=False, indent=indent)`.
pub fn to_python_json(val: &Value, indent: Option<usize>) -> String {
    let mut buf = Vec::new();
    match indent {
        None => {
            let mut ser = Serializer::with_formatter(&mut buf, PythonCompactFormatter);
            let _ = val.serialize(&mut ser);
        }
        Some(n) => {
            let indent_bytes = vec![b' '; n];
            let fmt = PrettyFormatter::with_indent(&indent_bytes);
            let mut ser = Serializer::with_formatter(&mut buf, fmt);
            let _ = val.serialize(&mut ser);
        }
    }
    String::from_utf8(buf).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Hugging Face `{% generation %}` / `{% endgeneration %}` stripping
// ---------------------------------------------------------------------------

static GENERATION_TAG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{%-?\s*(?:end)?generation\s*-?%\}").unwrap());

/// Strip Hugging Face `{% generation %}` and `{% endgeneration %}` tags from a Jinja template source.
pub fn strip_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut pos = 0;
    let mut iter = GENERATION_TAG_RE.find_iter(source);
    while let Some(Ok(m)) = iter.next() {
        out.push_str(&source[pos..m.start()]);
        pos = m.end();
    }
    out.push_str(&source[pos..]);
    out
}

// ---------------------------------------------------------------------------
// MiniJinja environment helpers (tojson filter, raise_exception, py methods)
// ---------------------------------------------------------------------------

fn filter_tojson(value: MjValue, kwargs: Kwargs) -> Result<MjValue, MjError> {
    let indent: Option<usize> = kwargs.get("indent").ok().flatten();
    let sort_keys: bool = kwargs
        .get::<Option<bool>>("sort_keys")
        .ok()
        .flatten()
        .unwrap_or(false);
    let mut json_val: Value = serde_json::to_value(&value).map_err(|e| {
        MjError::new(
            ErrorKind::InvalidOperation,
            format!("tojson serialization failed: {e}"),
        )
    })?;
    if sort_keys {
        json_val = sort_json_keys(json_val);
    }
    Ok(MjValue::from_safe_string(to_python_json(&json_val, indent)))
}

fn sort_json_keys(val: Value) -> Value {
    match val {
        Value::Object(map) => {
            let mut entries: Vec<(String, Value)> = map
                .into_iter()
                .map(|(k, v)| (k, sort_json_keys(v)))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let mut sorted = Map::new();
            for (k, v) in entries {
                sorted.insert(k, v);
            }
            Value::Object(sorted)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(sort_json_keys).collect()),
        other => other,
    }
}

fn match_prefix_or_tuple(s: &str, arg: &MjValue, end: bool) -> bool {
    if let Some(pat) = arg.as_str() {
        return if end {
            s.ends_with(pat)
        } else {
            s.starts_with(pat)
        };
    }
    if let Ok(iter) = arg.try_iter() {
        for item in iter {
            if let Some(pat) = item.as_str() {
                if (end && s.ends_with(pat)) || (!end && s.starts_with(pat)) {
                    return true;
                }
            }
        }
    }
    false
}

fn py_unknown_method(
    _state: &minijinja::State<'_, '_>,
    value: &MjValue,
    method: &str,
    args: &[MjValue],
) -> Result<MjValue, MjError> {
    if let Some(s) = value.as_str() {
        match method {
            "strip" => {
                let chars = args.first().and_then(|v| v.as_str());
                let res = match chars {
                    Some(c) => s.trim_matches(|ch| c.contains(ch)),
                    None => s.trim(),
                };
                return Ok(MjValue::from(res));
            }
            "lstrip" => {
                let chars = args.first().and_then(|v| v.as_str());
                let res = match chars {
                    Some(c) => s.trim_start_matches(|ch| c.contains(ch)),
                    None => s.trim_start(),
                };
                return Ok(MjValue::from(res));
            }
            "rstrip" => {
                let chars = args.first().and_then(|v| v.as_str());
                let res = match chars {
                    Some(c) => s.trim_end_matches(|ch| c.contains(ch)),
                    None => s.trim_end(),
                };
                return Ok(MjValue::from(res));
            }
            "startswith" => {
                let Some(arg) = args.first() else {
                    return Err(MjError::new(
                        ErrorKind::MissingArgument,
                        "startswith requires an argument",
                    ));
                };
                return Ok(MjValue::from(match_prefix_or_tuple(s, arg, false)));
            }
            "endswith" => {
                let Some(arg) = args.first() else {
                    return Err(MjError::new(
                        ErrorKind::MissingArgument,
                        "endswith requires an argument",
                    ));
                };
                return Ok(MjValue::from(match_prefix_or_tuple(s, arg, true)));
            }
            "split" => {
                let sep = args.first().and_then(|v| v.as_str());
                let maxsplit: Option<isize> = args
                    .get(1)
                    .and_then(|v| i64::try_from(v.clone()).ok().map(|n| n as isize));
                let parts: Vec<MjValue> = match (sep, maxsplit) {
                    (None, None) => s.split_whitespace().map(MjValue::from).collect(),
                    (None, Some(m)) if m < 0 => s.split_whitespace().map(MjValue::from).collect(),
                    (None, Some(m)) => {
                        let mut out = Vec::new();
                        let mut rem = s.trim_start();
                        for _ in 0..(m as usize) {
                            let Some(idx) = rem.find(char::is_whitespace) else {
                                break;
                            };
                            out.push(MjValue::from(&rem[..idx]));
                            rem = rem[idx..].trim_start();
                        }
                        if !rem.is_empty() {
                            out.push(MjValue::from(rem));
                        }
                        out
                    }
                    (Some(sep_str), Some(m)) if m >= 0 => s
                        .splitn((m as usize) + 1, sep_str)
                        .map(MjValue::from)
                        .collect(),
                    (Some(sep_str), _) => s.split(sep_str).map(MjValue::from).collect(),
                };
                return Ok(MjValue::from(parts));
            }
            "rsplit" => {
                let sep = args.first().and_then(|v| v.as_str());
                let maxsplit: Option<isize> = args
                    .get(1)
                    .and_then(|v| i64::try_from(v.clone()).ok().map(|n| n as isize));
                let parts: Vec<MjValue> = match (sep, maxsplit) {
                    (Some(sep_str), Some(m)) if m >= 0 => {
                        let mut v: Vec<MjValue> = s
                            .rsplitn((m as usize) + 1, sep_str)
                            .map(MjValue::from)
                            .collect();
                        v.reverse();
                        v
                    }
                    (Some(sep_str), _) => s.split(sep_str).map(MjValue::from).collect(),
                    (None, _) => s.split_whitespace().map(MjValue::from).collect(),
                };
                return Ok(MjValue::from(parts));
            }
            "replace" => {
                let old = args.first().and_then(|v| v.as_str()).unwrap_or("");
                let new = args.get(1).and_then(|v| v.as_str()).unwrap_or("");
                let count = args
                    .get(2)
                    .and_then(|v| usize::try_from(v.clone()).ok())
                    .unwrap_or(usize::MAX);
                return Ok(MjValue::from(s.replacen(old, new, count)));
            }
            "lower" => return Ok(MjValue::from(s.to_lowercase())),
            "upper" => return Ok(MjValue::from(s.to_uppercase())),
            "find" => {
                let sub = args.first().and_then(|v| v.as_str()).unwrap_or("");
                let pos = s.find(sub).map(|i| i as i64).unwrap_or(-1);
                return Ok(MjValue::from(pos));
            }
            _ => {}
        }
    }

    if value.kind() == ValueKind::Map {
        match method {
            "items" => {
                let mut pairs = Vec::new();
                if let Ok(iter) = value.try_iter() {
                    for k in iter {
                        let v = value.get_item(&k).unwrap_or(MjValue::UNDEFINED);
                        pairs.push(MjValue::from(vec![k, v]));
                    }
                }
                return Ok(MjValue::from(pairs));
            }
            "keys" => {
                let mut keys = Vec::new();
                if let Ok(iter) = value.try_iter() {
                    keys.extend(iter);
                }
                return Ok(MjValue::from(keys));
            }
            "values" => {
                let mut vals = Vec::new();
                if let Ok(iter) = value.try_iter() {
                    for k in iter {
                        vals.push(value.get_item(&k).unwrap_or(MjValue::UNDEFINED));
                    }
                }
                return Ok(MjValue::from(vals));
            }
            "get" => {
                let Some(key) = args.first() else {
                    return Ok(MjValue::from(()));
                };
                let default = args.get(1).cloned().unwrap_or_else(|| MjValue::from(()));
                match value.get_item(key) {
                    Ok(v) if !v.is_undefined() => return Ok(v),
                    _ => return Ok(default),
                }
            }
            _ => {}
        }
    }

    Err(MjError::from(ErrorKind::UnknownMethod))
}

/// Capabilities probed from a [`ChatTemplate`] at load time (matching `serve/frontend.py`'s `_detect_caps`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplateCapabilities {
    pub supports_tools: bool,
    pub supports_tool_calls: bool,
    pub supports_system_role: bool,
    pub supports_parallel_tool_calls: bool,
    pub supports_preserve_reasoning: bool,
}

/// Model chat template engine matching `serve/frontend.py`'s `ChatTemplate`.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    pub source: String,
    pub stripped_source: String,
    pub glm_tools: bool,
    pub caps: HashMap<String, bool>,
}

impl ChatTemplate {
    /// Load a [`ChatTemplate`] from a `.jinja` file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TemplateError> {
        let source = std::fs::read_to_string(path)?;
        Self::from_str(&source)
    }

    /// Construct a [`ChatTemplate`] from a Jinja source string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(source: &str) -> Result<Self, TemplateError> {
        let glm_tools = source.contains("<arg_key>");
        let stripped_source = strip_generation_tags(source);

        // Validate compilation eagerly.
        let mut env = Self::build_env();
        env.add_template("chat", &stripped_source)
            .map_err(|e| TemplateError::Jinja(e.to_string()))?;

        let mut tpl = Self {
            source: source.to_string(),
            stripped_source,
            glm_tools,
            caps: HashMap::new(),
        };
        tpl.caps = tpl.detect_caps();
        Ok(tpl)
    }

    fn build_env() -> Environment<'static> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.add_filter("tojson", filter_tojson);
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<MjValue, MjError> {
                Err(MjError::new(ErrorKind::InvalidOperation, msg))
            },
        );
        env.set_unknown_method_callback(py_unknown_method);
        env
    }

    /// Render the chat template with `messages`, `tools`, `add_generation_prompt`, and optional `kwargs`.
    pub fn render(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        add_generation_prompt: bool,
        kwargs: Option<&Map<String, Value>>,
    ) -> Result<String, TemplateError> {
        let keep_empty = std::env::var("STRATA_KEEP_EMPTY_TURNS").as_deref() == Ok("1");
        let n = messages.len();
        let filtered: Vec<&Value> = messages
            .iter()
            .enumerate()
            .filter(|&(i, m)| i == n.saturating_sub(1) || keep_empty || !is_empty_assistant_turn(m))
            .map(|(_, m)| m)
            .collect();

        let mut ctx = Map::new();
        if let Some(extra) = kwargs {
            for (k, v) in extra {
                ctx.insert(k.clone(), v.clone());
            }
        }
        ctx.insert(
            "messages".to_string(),
            serde_json::to_value(&filtered).unwrap_or(Value::Array(Vec::new())),
        );
        if let Some(t) = tools {
            ctx.insert("tools".to_string(), Value::Array(t.to_vec()));
        }
        ctx.insert(
            "add_generation_prompt".to_string(),
            Value::Bool(add_generation_prompt),
        );

        let mut env = Self::build_env();
        env.add_template("chat", &self.stripped_source)
            .map_err(|e| TemplateError::Jinja(e.to_string()))?;
        let tmpl = env
            .get_template("chat")
            .map_err(|e| TemplateError::Jinja(e.to_string()))?;

        tmpl.render(MjValue::from_serialize(&ctx)).map_err(|e| {
            let detail = e
                .detail()
                .map(ToString::to_string)
                .unwrap_or_else(|| e.to_string());
            TemplateError::Request(detail)
        })
    }

    /// Convenience method to render with `enable_thinking` explicitly set.
    pub fn render_with_thinking(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String, TemplateError> {
        let mut kw = Map::new();
        if let Some(et) = enable_thinking {
            kw.insert("enable_thinking".to_string(), Value::Bool(et));
        }
        self.render(messages, tools, add_generation_prompt, Some(&kw))
    }

    /// Return typed [`TemplateCapabilities`].
    pub fn capabilities(&self) -> TemplateCapabilities {
        TemplateCapabilities {
            supports_tools: self.caps.get("supports_tools").copied().unwrap_or(false),
            supports_tool_calls: self
                .caps
                .get("supports_tool_calls")
                .copied()
                .unwrap_or(false),
            supports_system_role: self
                .caps
                .get("supports_system_role")
                .copied()
                .unwrap_or(false),
            supports_parallel_tool_calls: self
                .caps
                .get("supports_parallel_tool_calls")
                .copied()
                .unwrap_or(false),
            supports_preserve_reasoning: self
                .caps
                .get("supports_preserve_reasoning")
                .copied()
                .unwrap_or(false),
        }
    }

    fn detect_caps(&self) -> HashMap<String, bool> {
        let try_render = |msgs: &[Value], tools: Option<&[Value]>| -> String {
            self.render(msgs, tools, true, None).unwrap_or_default()
        };

        let user = json!({"role": "user", "content": "strata_caps_user"});
        let tools: Vec<Value> = (0..2)
            .map(|i| {
                json!({
                    "name": format!("strata_caps_call_{i}"),
                    "description": "strata_caps_description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arg": {"type": "string"}
                        }
                    }
                })
            })
            .collect();

        let tool_prompt = try_render(std::slice::from_ref(&user), Some(&tools));

        let calls_supported = |count: usize| -> bool {
            let calls: Vec<Value> = (0..count)
                .map(|i| {
                    json!({
                        "name": format!("strata_caps_call_{i}"),
                        "arguments": {"arg": format!("strata_caps_arg_{i}")}
                    })
                })
                .collect();
            let tc_wrapped: Vec<Value> = calls.iter().map(|c| json!({"function": c})).collect();
            let mut msgs = vec![
                user.clone(),
                json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tc_wrapped
                }),
            ];
            let replies: Vec<String> = (0..count)
                .map(|i| format!("strata_caps_result_{i}"))
                .collect();
            for reply in &replies {
                msgs.push(json!({"role": "tool", "content": reply}));
            }
            msgs.push(user.clone());
            let prompt = try_render(&msgs, Some(&tools));

            let rx = Regex::new(
                r"(?s)<tool_call>\s*(<function=strata_caps_call_\d+>.*?</function>)\s*</tool_call>",
            )
            .unwrap();
            let mut parsed = Vec::new();
            let mut iter = rx.captures_iter(&prompt);
            while let Some(Ok(cap)) = iter.next() {
                if let Some(m) = cap.get(1) {
                    let Ok(tc) = parse_tool_call(m.as_str(), None) else {
                        return false;
                    };
                    parsed.push(json!({
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }));
                }
            }
            parsed == calls && replies.iter().all(|r| prompt.contains(r))
        };

        let history = vec![
            user.clone(),
            json!({
                "role": "assistant",
                "content": "strata_caps_answer",
                "reasoning_content": "strata_caps_reasoning"
            }),
            user.clone(),
        ];
        let sys_msgs = vec![
            json!({"role": "system", "content": "strata_caps_system"}),
            user.clone(),
        ];

        let mut caps = HashMap::new();
        caps.insert(
            "supports_tools".to_string(),
            [
                "strata_caps_call_0",
                "strata_caps_description",
                "<tool_call>",
                "<function=",
            ]
            .iter()
            .all(|s| tool_prompt.contains(s)),
        );
        caps.insert("supports_tool_calls".to_string(), calls_supported(1));
        caps.insert(
            "supports_system_role".to_string(),
            try_render(&sys_msgs, None).contains("strata_caps_system"),
        );
        caps.insert(
            "supports_parallel_tool_calls".to_string(),
            calls_supported(2),
        );
        caps.insert(
            "supports_preserve_reasoning".to_string(),
            try_render(&history, None).contains("strata_caps_reasoning"),
        );
        caps
    }
}

fn is_empty_assistant_turn(m: &Value) -> bool {
    let Some(obj) = m.as_object() else {
        return false;
    };
    if obj.get("role").and_then(|v| v.as_str()) != Some("assistant") {
        return false;
    }
    if let Some(tc) = obj.get("tool_calls") {
        if !tc.is_null() && tc.as_array().is_none_or(|a| !a.is_empty()) {
            return false;
        }
    }
    let content = obj.get("content");
    text_of(content).trim().is_empty() && !has_image(content)
}

// ---------------------------------------------------------------------------
// Content & effort helpers
// ---------------------------------------------------------------------------

const IMAGE_PARTS: &[&str] = &["image_url", "input_image", "image"];

fn is_text_part_type(ty: Option<&str>) -> bool {
    matches!(ty, Some("text" | "input_text") | None)
}

pub fn text_of(content: Option<&Value>) -> String {
    let Some(c) = content else {
        return String::new();
    };
    if let Some(s) = c.as_str() {
        return s.to_string();
    }
    if let Some(arr) = c.as_array() {
        let mut out = String::new();
        for part in arr {
            if let Some(obj) = part.as_object() {
                let ty = obj.get("type").and_then(|v| v.as_str());
                if is_text_part_type(ty) {
                    if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                        out.push_str(t);
                    }
                }
            }
        }
        return out;
    }
    String::new()
}

pub fn has_image(content: Option<&Value>) -> bool {
    let Some(Value::Array(arr)) = content else {
        return false;
    };
    arr.iter().any(|p| {
        p.as_object()
            .and_then(|o| o.get("type"))
            .and_then(|t| t.as_str())
            .is_some_and(|ty| IMAGE_PARTS.contains(&ty))
    })
}

fn image_source(part: &Map<String, Value>) -> String {
    if part.get("type").and_then(|v| v.as_str()) == Some("image") {
        if let Some(src) = part.get("source").and_then(|v| v.as_object()) {
            if src.get("type").and_then(|v| v.as_str()) == Some("base64") {
                let media_type = src
                    .get("media_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("image/png");
                let data = src.get("data").and_then(|v| v.as_str()).unwrap_or("");
                return format!("data:{media_type};base64,{data}");
            }
            return src
                .get("url")
                .or_else(|| src.get("path"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
        }
        return String::new();
    }
    if let Some(url_val) = part.get("image_url") {
        if let Some(s) = url_val.as_str() {
            return s.to_string();
        }
        if let Some(obj) = url_val.as_object() {
            if let Some(s) = obj.get("url").and_then(|v| v.as_str()) {
                return s.to_string();
            }
        }
    }
    String::new()
}

fn parts_of(content: Option<&Value>) -> Value {
    if !has_image(content) {
        return Value::String(text_of(content));
    }
    let Some(Value::Array(arr)) = content else {
        return Value::String(String::new());
    };
    let mut items = Vec::new();
    for part in arr {
        let Some(obj) = part.as_object() else {
            continue;
        };
        let ty = obj.get("type").and_then(|v| v.as_str());
        if let Some(t) = ty {
            if IMAGE_PARTS.contains(&t) {
                items.push(json!({"type": "image", "source": image_source(obj)}));
                continue;
            }
        }
        if is_text_part_type(ty) && obj.contains_key("text") {
            let txt = obj.get("text").and_then(|v| v.as_str()).unwrap_or("");
            items.push(json!({"type": "text", "text": txt}));
        }
    }
    Value::Array(items)
}

fn check_parts(content: Option<&Value>) -> Option<Value> {
    let c = content?.clone();
    let Value::Array(arr) = c else {
        return Some(c);
    };
    let mut out = arr;
    for part in &mut out {
        if let Some(obj) = part.as_object_mut() {
            if obj.get("type").and_then(|v| v.as_str()) == Some("refusal") {
                let refusal = obj
                    .get("refusal")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                obj.clear();
                obj.insert("type".to_string(), Value::String("text".to_string()));
                obj.insert("text".to_string(), Value::String(refusal));
            }
        }
    }
    Some(Value::Array(out))
}

/// Extract image sources from normalized messages in prompt order.
pub fn images_of(messages: &[Value]) -> Vec<String> {
    let mut out = Vec::new();
    for m in messages {
        if let Some(arr) = m.get("content").and_then(|c| c.as_array()) {
            for item in arr {
                if item.get("type").and_then(|t| t.as_str()) == Some("image") {
                    if let Some(src) = item.get("source").and_then(|s| s.as_str()) {
                        out.push(src.to_string());
                    }
                }
            }
        }
    }
    out
}

/// Map a client's reasoning effort value onto template kwargs (`enable_thinking` or `reasoning_effort`).
pub fn effort_kwargs(value: Option<&Value>) -> Result<Map<String, Value>, TemplateError> {
    let mut out = Map::new();
    let Some(v) = value else {
        return Ok(out);
    };
    if v.is_null() {
        return Ok(out);
    }
    if v.as_bool() == Some(false) {
        out.insert("enable_thinking".to_string(), Value::Bool(false));
        return Ok(out);
    }
    let raw_str = match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if raw_str.is_empty() {
        return Ok(out);
    }
    let key = raw_str.trim().to_lowercase();
    match key.as_str() {
        "none" | "off" | "minimal" | "disabled" | "false" => {
            out.insert("enable_thinking".to_string(), Value::Bool(false));
        }
        "low" => {
            out.insert(
                "reasoning_effort".to_string(),
                Value::String("low".to_string()),
            );
        }
        "medium" => {
            out.insert(
                "reasoning_effort".to_string(),
                Value::String("medium".to_string()),
            );
        }
        "high" | "xhigh" | "max" | "maximum" => {
            out.insert(
                "reasoning_effort".to_string(),
                Value::String("xhigh".to_string()),
            );
        }
        _ => {
            return Err(TemplateError::Request(format!(
                "unknown reasoning effort {raw_str:?}: use none, low, medium or high"
            )));
        }
    }
    Ok(out)
}

/// Map Anthropic's `budget_tokens` onto template `reasoning_effort` kwargs.
pub fn budget_effort(tokens: &Value) -> Map<String, Value> {
    let mut out = Map::new();
    let n = tokens
        .as_i64()
        .or_else(|| tokens.as_str().and_then(|s| s.parse::<i64>().ok()));
    if let Some(n) = n {
        let level = if n < 2048 {
            "low"
        } else if n < 8192 {
            "medium"
        } else {
            "xhigh"
        };
        out.insert(
            "reasoning_effort".to_string(),
            Value::String(level.to_string()),
        );
    }
    out
}

// ---------------------------------------------------------------------------
// Literal tag marking & unmarking (#537, #554, #2026)
// ---------------------------------------------------------------------------

pub const CONTROL_MARK0: u32 = 0xF0E10;

const TOOL_LITERAL_KEYS: &[&str] = &[
    "</tool_call>",
    "<tool_call>",
    "<tool_response>",
    "</tool_response>",
];

/// Return the default literal tag-to-private-use-character map (`LITERAL_TAGS` in `serve/frontend.py`),
/// ordered longest-first.
pub fn default_literal_tags() -> Vec<(String, String)> {
    let mut pairs = vec![
        ("<think>".to_string(), "\u{F0E01}".to_string()),
        ("</think>".to_string(), "\u{F0E02}".to_string()),
        ("<|vision_start|>".to_string(), "\u{F0E03}".to_string()),
        ("<|image_pad|>".to_string(), "\u{F0E04}".to_string()),
        ("<|vision_end|>".to_string(), "\u{F0E05}".to_string()),
        ("<|video_pad|>".to_string(), "\u{F0E06}".to_string()),
        ("</tool_call>".to_string(), "\u{F0E07}".to_string()),
        ("<tool_call>".to_string(), "\u{F0E08}".to_string()),
        ("<tool_response>".to_string(), "\u{F0E09}".to_string()),
        ("</tool_response>".to_string(), "\u{F0E0A}".to_string()),
    ];
    pairs.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
    pairs
}

/// Build the literal tag map including private-use markers for all tokenizer `controls` (`<|im_start|>`, etc.),
/// sorted longest-first.
pub fn literal_tags<I, S>(controls: I) -> Vec<(String, String)>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut pairs = default_literal_tags();
    let mut k = 0u32;
    for c in controls {
        let s = c.as_ref();
        if !pairs.iter().any(|(existing, _)| existing == s) {
            if let Some(ch) = char::from_u32(CONTROL_MARK0 + k) {
                pairs.push((s.to_string(), ch.to_string()));
                k += 1;
            }
        }
    }
    pairs.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
    pairs
}

fn mark_str(text: &str, tags: &[(String, String)]) -> String {
    let mut out = text.to_string();
    for (tag, mark) in tags {
        if out.contains(tag) {
            out = out.replace(tag, mark);
        }
    }
    out
}

fn mark_deep(v: &Value, tags: &[(String, String)]) -> Value {
    match v {
        Value::String(s) => Value::String(mark_str(s, tags)),
        Value::Object(map) => {
            let mut out = Map::new();
            for (k, x) in map {
                out.insert(k.clone(), mark_deep(x, tags));
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(|x| mark_deep(x, tags)).collect()),
        other => other.clone(),
    }
}

fn has_tag(v: &Value, tags: &[(String, String)]) -> bool {
    match v {
        Value::String(s) => tags.iter().any(|(tag, _)| s.contains(tag)),
        Value::Object(map) => map.values().any(|x| has_tag(x, tags)),
        Value::Array(arr) => arr.iter().any(|x| has_tag(x, tags)),
        _ => false,
    }
}

const TOOL_RESULT_OPEN: &str = "<tool_response>";
const TOOL_RESULT_CLOSE: &str = "</tool_response>";

fn keep_tool_result_wrap(content: Option<&Value>, tags: &[(String, String)]) -> Option<Value> {
    let c = content?;
    let o = TOOL_RESULT_OPEN;
    let cl = TOOL_RESULT_CLOSE;

    let wrap = |text: &str, first: bool, last: bool| -> String {
        let i = if first {
            text.find(o).unwrap() + o.len()
        } else {
            0
        };
        let j = if last {
            text.rfind(cl).unwrap()
        } else {
            text.len()
        };
        let head = if first {
            format!("{}{o}", mark_str(&text[..i - o.len()], tags))
        } else {
            String::new()
        };
        let tail = if last {
            format!("{cl}{}", mark_str(&text[j + cl.len()..], tags))
        } else {
            String::new()
        };
        format!("{head}{}{tail}", mark_str(&text[i..j], tags))
    };

    if let Some(s) = c.as_str() {
        let body = s.trim();
        if body.len() >= o.len() + cl.len() && body.starts_with(o) && body.ends_with(cl) {
            return Some(Value::String(wrap(s, true, true)));
        }
        return None;
    }

    let arr = c.as_array()?;
    if arr.is_empty() {
        return None;
    }
    let mut texts = Vec::with_capacity(arr.len());
    for x in arr {
        let obj = x.as_object()?;
        let t = obj.get("text")?.as_str()?;
        texts.push(t);
    }
    let joined = texts.concat();
    let body = joined.trim();
    if !(body.len() >= o.len() + cl.len() && body.starts_with(o) && body.ends_with(cl)) {
        return None;
    }
    if !(texts[0].trim_start().starts_with(o) && texts.last().unwrap().trim_end().ends_with(cl)) {
        return None;
    }
    if arr.len() == 1 {
        let mut item = arr[0].as_object()?.clone();
        item.insert(
            "text".to_string(),
            Value::String(wrap(texts[0], true, true)),
        );
        return Some(Value::Array(vec![Value::Object(item)]));
    }
    let mut out = Vec::with_capacity(arr.len());
    let mut first_item = arr[0].as_object()?.clone();
    first_item.insert(
        "text".to_string(),
        Value::String(wrap(texts[0], true, false)),
    );
    out.push(Value::Object(first_item));
    for x in &arr[1..arr.len() - 1] {
        out.push(mark_deep(x, tags));
    }
    let mut last_item = arr.last()?.as_object()?.clone();
    last_item.insert(
        "text".to_string(),
        Value::String(wrap(texts.last().unwrap(), false, true)),
    );
    out.push(Value::Object(last_item));
    Some(Value::Array(out))
}

/// Swap literal `<think>`, `</think>`, vision tags, tool tags, and optional control tokens inside
/// `messages` and `tools` with private-use markers so they are not tokenized as special tokens.
///
/// Returns `(marked_messages, marked_tools, changed)`.
pub fn mark_think_literals(
    messages: &[Value],
    tools: Option<&[Value]>,
    tags: Option<&[(String, String)]>,
) -> (Vec<Value>, Option<Vec<Value>>, bool) {
    let default_tags;
    let tags = match tags {
        Some(t) => t,
        None => {
            default_tags = default_literal_tags();
            &default_tags
        }
    };

    let msgs_have = messages.iter().any(|m| has_tag(m, tags));
    let tools_have = tools.is_some_and(|t| t.iter().any(|x| has_tag(x, tags)));
    if !msgs_have && !tools_have {
        return (messages.to_vec(), tools.map(|t| t.to_vec()), false);
    }

    let own: Vec<(String, String)> = tags
        .iter()
        .filter(|(t, _)| !TOOL_LITERAL_KEYS.contains(&t.as_str()))
        .cloned()
        .collect();

    let mut out_msgs = Vec::with_capacity(messages.len());
    for m in messages {
        let Some(obj) = m.as_object() else {
            out_msgs.push(m.clone());
            continue;
        };
        let role = obj.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let orig_content = obj.get("content");
        let mine = if role == "assistant" {
            own.as_slice()
        } else {
            tags
        };

        let mut new_obj = Map::new();
        for (k, v) in obj {
            if k == "role" {
                new_obj.insert(k.clone(), v.clone());
            } else {
                new_obj.insert(k.clone(), mark_deep(v, mine));
            }
        }
        if role == "user" {
            if let Some(kept) = keep_tool_result_wrap(orig_content, tags) {
                new_obj.insert("content".to_string(), kept);
            }
        }
        if role == "assistant" {
            if let Some(s) = orig_content.and_then(|c| c.as_str()) {
                if s.trim_start().starts_with("<think>") && s.contains("</think>") {
                    let i = s.find("<think>").unwrap() + "<think>".len();
                    let j = s.find("</think>").unwrap();
                    let rebuilt = format!(
                        "{}{}</think>{}",
                        &s[..i],
                        mark_str(&s[i..j], &own),
                        mark_str(&s[j + "</think>".len()..], &own)
                    );
                    new_obj.insert("content".to_string(), Value::String(rebuilt));
                }
            }
        }
        out_msgs.push(Value::Object(new_obj));
    }

    let out_tools = tools.map(|t| t.iter().map(|x| mark_deep(x, tags)).collect());
    (out_msgs, out_tools, true)
}

/// Restore private-use markers in the rendered `prompt` back to their literal tag text, returning
/// `(unmarked_prompt, plain_byte_spans)` for [`crate::tokenizer::Tokenizer::encode_with_plain`].
pub fn unmark_think_literals(
    prompt: &str,
    tags: Option<&[(String, String)]>,
) -> (String, Vec<(usize, usize)>) {
    let default_tags;
    let tags = match tags {
        Some(t) => t,
        None => {
            default_tags = default_literal_tags();
            &default_tags
        }
    };
    let marks: Vec<&str> = tags.iter().map(|(_, m)| m.as_str()).collect();
    let tag_strs: Vec<&str> = tags.iter().map(|(t, _)| t.as_str()).collect();
    let Ok(ac) = AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostFirst)
        .build(&marks)
    else {
        return (prompt.to_string(), Vec::new());
    };

    let mut out = String::with_capacity(prompt.len());
    let mut spans = Vec::new();
    let mut pos = 0;
    for m in ac.find_iter(prompt) {
        out.push_str(&prompt[pos..m.start()]);
        let tag = tag_strs[m.pattern().as_usize()];
        let start_byte = out.len();
        out.push_str(tag);
        spans.push((start_byte, start_byte + tag.len()));
        pos = m.end();
    }
    out.push_str(&prompt[pos..]);
    (out, spans)
}

// ---------------------------------------------------------------------------
// OpenAI & Anthropic request normalization
// ---------------------------------------------------------------------------

const BILLING_HEADER: &str = "x-anthropic-billing-header:";

/// Pin volatile `cch=` and 4th-component `cc_version=` stamps in an `x-anthropic-billing-header:`
/// system prefix to `'f'`s so prefix caches stay valid across requests.
pub fn pin_billing_stamp(system: &str) -> String {
    if !system.starts_with(BILLING_HEADER) {
        return system.to_string();
    }
    let mut bytes = system.as_bytes().to_vec();
    if let Some(rel) = system[BILLING_HEADER.len()..].find("cch=") {
        let cch = BILLING_HEADER.len() + rel;
        if cch <= 160 {
            let v = cch + 4;
            if let Some(end_rel) = system[v..].find(';') {
                let end = v + end_rel;
                if end > v && end - v <= 16 {
                    bytes[v..end].fill(b'f');
                }
            }
        }
    }
    if let Some(rel) = system[BILLING_HEADER.len()..].find("cc_version=") {
        let cv = BILLING_HEADER.len() + rel;
        if cv <= 160 {
            let v = cv + "cc_version=".len();
            if let Some(end_rel) = system[v..].find(';') {
                let end = v + end_rel;
                if end > v && end - v <= 64 {
                    let parts: Vec<&str> = system[v..end].split('.').collect();
                    if parts.len() > 3 {
                        let prefix_len = parts[..3].join(".").len() + 1;
                        let tail = v + prefix_len;
                        if tail <= end {
                            bytes[tail..end].fill(b'f');
                        }
                    }
                }
            }
        }
    }
    String::from_utf8(bytes).unwrap_or_else(|_| system.to_string())
}

/// Normalize a tool call's arguments from client history into a JSON object map.
pub fn tool_arguments(raw: Option<&Value>) -> Map<String, Value> {
    let Some(r) = raw else {
        return Map::new();
    };
    match r {
        Value::Null => Map::new(),
        Value::Object(m) => m.clone(),
        Value::String(s) => {
            if s.trim().is_empty() {
                return Map::new();
            }
            if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(s) {
                m
            } else {
                let mut m = Map::new();
                m.insert("arguments".to_string(), Value::String(s.clone()));
                m
            }
        }
        other => {
            let mut m = Map::new();
            m.insert(
                "arguments".to_string(),
                Value::String(to_python_json(other, None)),
            );
            m
        }
    }
}

fn object_list(
    value: Option<&Value>,
    name: &str,
) -> Result<Vec<Map<String, Value>>, TemplateError> {
    let Some(v) = value else {
        return Ok(Vec::new());
    };
    if v.is_null() {
        return Ok(Vec::new());
    }
    let parsed_owned: Value;
    let target = if let Some(s) = v.as_str() {
        parsed_owned = serde_json::from_str(s).map_err(|_| {
            TemplateError::Request(format!(
                "{name} must be a list of objects (a string was sent that is not JSON)"
            ))
        })?;
        &parsed_owned
    } else {
        v
    };
    let Some(arr) = target.as_array() else {
        return Err(TemplateError::Request(format!(
            "{name} must be a list of objects"
        )));
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let Some(obj) = item.as_object() else {
            return Err(TemplateError::Request(format!(
                "{name} must be a list of objects"
            )));
        };
        out.push(obj.clone());
    }
    Ok(out)
}

type ToolOrderCache = Vec<(Vec<String>, Vec<String>)>;

static TOOL_ORDERS: LazyLock<Mutex<ToolOrderCache>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// Normalized request tuple `(messages, tools, template_kwargs)`.
pub type NormalizedRequest = (Vec<Value>, Option<Vec<Value>>, Map<String, Value>);

fn stable_tool_order<F>(tools: Vec<Map<String, Value>>, name_of: F) -> Vec<Map<String, Value>>
where
    F: Fn(&Map<String, Value>) -> String,
{
    if tools.len() < 2 {
        return tools;
    }
    let names: Vec<String> = tools.iter().map(&name_of).collect();
    let mut sorted_key = names.clone();
    sorted_key.sort();
    sorted_key.dedup();
    if sorted_key.len() != names.len() {
        return tools;
    }
    let order = {
        let mut cache = TOOL_ORDERS.lock().unwrap();
        if let Some(pos) = cache.iter().position(|(k, _)| k == &sorted_key) {
            let entry = cache.remove(pos);
            let ord = entry.1.clone();
            cache.push(entry);
            ord
        } else {
            cache.push((sorted_key, names));
            if cache.len() > 64 {
                cache.remove(0);
            }
            return tools;
        }
    };
    let pos_map: HashMap<&str, usize> = order
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let mut reordered = tools;
    reordered.sort_by_key(|t| pos_map.get(name_of(t).as_str()).copied().unwrap_or(0));
    reordered
}

fn tool_list(
    value: Option<&Value>,
    wrapper: Option<&str>,
) -> Result<Vec<Map<String, Value>>, TemplateError> {
    let Some(v) = value else {
        return Ok(Vec::new());
    };
    if v.is_null() || v.as_array().is_some_and(|a| a.is_empty()) {
        return Ok(Vec::new());
    }
    let shape = if wrapper.is_some() {
        r#"{"type": "function", "function": {"name": ..., "parameters": {...}}}"#
    } else {
        r#"{"name": ..., "input_schema": {...}}"#
    };
    let tools = object_list(Some(v), "tools").map_err(|_| {
        TemplateError::Request(format!(
            "tools must be a list of tool objects, each {shape}"
        ))
    })?;
    let key = if wrapper.is_some() {
        "parameters"
    } else {
        "input_schema"
    };
    for (i, t) in tools.iter().enumerate() {
        let fn_obj = if let Some(w) = wrapper {
            if t.get("type").and_then(|x| x.as_str()) == Some(w) {
                t.get(w).and_then(|x| x.as_object())
            } else {
                Some(t)
            }
        } else {
            Some(t)
        };
        let Some(fn_map) = fn_obj else {
            return Err(TemplateError::Request(format!(
                "tools[{i}] has no name: each tool must be {shape}"
            )));
        };
        let name = fn_map.get("name").and_then(|n| n.as_str()).unwrap_or("");
        if name.is_empty() {
            return Err(TemplateError::Request(format!(
                "tools[{i}] has no name: each tool must be {shape}"
            )));
        }
        if let Some(schema) = fn_map.get(key) {
            if !schema.is_null() && !schema.is_object() {
                return Err(TemplateError::Request(format!(
                    "tools[{i}] ({name}): \"{key}\" must be an object (the JSON schema of its parameters)"
                )));
            }
        }
    }

    Ok(stable_tool_order(tools, |t| {
        let fn_map = if let Some(w) = wrapper {
            if t.get("type").and_then(|x| x.as_str()) == Some(w) {
                t.get(w).and_then(|x| x.as_object()).unwrap_or(t)
            } else {
                t
            }
        } else {
            t
        };
        fn_map
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string()
    }))
}

fn late_system_to_user(messages: Vec<Value>) -> Vec<Value> {
    messages
        .into_iter()
        .enumerate()
        .map(|(i, mut m)| {
            if i > 0 {
                if let Some(obj) = m.as_object_mut() {
                    if obj.get("role").and_then(|r| r.as_str()) == Some("system") {
                        obj.insert("role".to_string(), Value::String("user".to_string()));
                    }
                }
            }
            m
        })
        .collect()
}

/// Convert an OpenAI Chat Completions request JSON object into `(messages, tools, template_kwargs)`.
pub fn openai_to_messages(req: &Value) -> Result<NormalizedRequest, TemplateError> {
    let mut messages = Vec::new();
    let raw_msgs = object_list(req.get("messages"), "messages")?;
    for m in raw_msgs {
        let mut role = m
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("user")
            .to_string();
        if role == "developer" {
            role = "system".to_string();
        }
        let images = matches!(role.as_str(), "user" | "tool" | "assistant");
        let checked = check_parts(m.get("content"));
        let content_val = if images {
            parts_of(checked.as_ref())
        } else {
            Value::String(text_of(checked.as_ref()))
        };

        let mut out = Map::new();
        out.insert("role".to_string(), Value::String(role));
        out.insert("content".to_string(), content_val);
        if let Some(rc) = m.get("reasoning_content") {
            if !rc.is_null() && rc.as_str() != Some("") {
                out.insert("reasoning_content".to_string(), rc.clone());
            }
        }
        if let Some(tc_val) = m.get("tool_calls") {
            if !tc_val.is_null() && tc_val.as_array().is_none_or(|a| !a.is_empty()) {
                let mut calls = Vec::new();
                for c in object_list(Some(tc_val), "tool_calls")? {
                    let fn_val = c.get("function").cloned().unwrap_or(Value::Object(c));
                    let Some(fn_obj) = fn_val.as_object() else {
                        return Err(TemplateError::Request(
                            "tool_calls must be a list of objects (each with a \"function\" object)"
                                .to_string(),
                        ));
                    };
                    let args = tool_arguments(fn_obj.get("arguments"));
                    calls.push(json!({
                        "function": {
                            "name": fn_obj.get("name"),
                            "arguments": args
                        }
                    }));
                }
                out.insert("tool_calls".to_string(), Value::Array(calls));
            }
        }
        messages.push(Value::Object(out));
    }

    let raw_tools = tool_list(req.get("tools"), Some("function"))?;
    let tools = if raw_tools.is_empty() {
        None
    } else {
        Some(
            raw_tools
                .into_iter()
                .map(|t| {
                    if t.get("type").and_then(|x| x.as_str()) == Some("function") {
                        t.get("function").cloned().unwrap_or(Value::Object(t))
                    } else {
                        Value::Object(t)
                    }
                })
                .collect(),
        )
    };

    let mut kwargs = Map::new();
    let reasoning_obj = req.get("reasoning").and_then(|r| r.as_object());
    let given = req.get("reasoning_effort");
    let effort_val = if given.is_some_and(|g| !g.is_null() && g.as_str() != Some("")) {
        given
    } else {
        reasoning_obj.and_then(|r| r.get("effort"))
    };
    for (k, v) in effort_kwargs(effort_val)? {
        kwargs.insert(k, v);
    }

    if let Some(ctk) = req.get("chat_template_kwargs").and_then(|c| c.as_object()) {
        for (k, v) in ctk {
            if k == "enable_thinking" && v.as_bool() == Some(false) {
                kwargs.clear();
                kwargs.insert("enable_thinking".to_string(), Value::Bool(false));
            } else if k == "reasoning_effort" && !kwargs.contains_key("enable_thinking") {
                for (ek, ev) in effort_kwargs(Some(v))? {
                    kwargs.insert(ek, ev);
                }
            }
        }
    }

    Ok((late_system_to_user(messages), tools, kwargs))
}

/// Convert an Anthropic Messages request JSON object into `(messages, tools, template_kwargs)`.
pub fn anthropic_to_messages(
    req: &Value,
    think_unasked: bool,
) -> Result<NormalizedRequest, TemplateError> {
    let mut messages = Vec::new();
    if let Some(sys) = req.get("system") {
        if !sys.is_null() && sys.as_str() != Some("") {
            let checked = check_parts(Some(sys));
            let text = pin_billing_stamp(&text_of(checked.as_ref()));
            if !text.is_empty() {
                messages.push(json!({"role": "system", "content": text}));
            }
        }
    }

    for m in object_list(req.get("messages"), "messages")? {
        let role = m
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("user")
            .to_string();
        let checked = check_parts(m.get("content"));
        if let Some(s) = checked.as_ref().and_then(|c| c.as_str()) {
            messages.push(json!({"role": role, "content": s}));
            continue;
        }
        let has_tool_result = checked
            .as_ref()
            .and_then(|c| c.as_array())
            .is_some_and(|arr| {
                arr.iter().any(|b| {
                    b.as_object()
                        .and_then(|o| o.get("type"))
                        .and_then(|t| t.as_str())
                        == Some("tool_result")
                })
            });
        if role == "user" && has_image(checked.as_ref()) && !has_tool_result {
            messages.push(json!({"role": "user", "content": parts_of(checked.as_ref())}));
            continue;
        }

        let mut text = Vec::new();
        let mut reasoning = Vec::new();
        let mut calls = Vec::new();
        let mut parts = Vec::new();

        if let Some(arr) = checked.as_ref().and_then(|c| c.as_array()) {
            for block in arr {
                let Some(b_obj) = block.as_object() else {
                    continue;
                };
                let kind = b_obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match kind {
                    "text" => {
                        let t = b_obj.get("text").and_then(|v| v.as_str()).unwrap_or("");
                        text.push(t.to_string());
                        parts.push(block.clone());
                    }
                    k if IMAGE_PARTS.contains(&k) => {
                        parts.push(block.clone());
                    }
                    "thinking" => {
                        let th = b_obj.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                        reasoning.push(th.to_string());
                    }
                    "tool_use" => {
                        let args = tool_arguments(b_obj.get("input"));
                        calls.push(json!({
                            "function": {
                                "name": b_obj.get("name"),
                                "arguments": args
                            }
                        }));
                    }
                    "tool_result" => {
                        let tr_checked = check_parts(b_obj.get("content"));
                        messages.push(json!({
                            "role": "tool",
                            "content": parts_of(tr_checked.as_ref())
                        }));
                    }
                    _ => {}
                }
            }
        }

        if !text.is_empty() || !calls.is_empty() || !reasoning.is_empty() || !parts.is_empty() {
            let parts_val = Value::Array(parts);
            let content_val = if has_image(Some(&parts_val)) {
                parts_of(Some(&parts_val))
            } else {
                Value::String(text.concat())
            };
            let mut out = Map::new();
            out.insert("role".to_string(), Value::String(role));
            out.insert("content".to_string(), content_val);
            if !reasoning.is_empty() {
                out.insert(
                    "reasoning_content".to_string(),
                    Value::String(reasoning.concat()),
                );
            }
            if !calls.is_empty() {
                out.insert("tool_calls".to_string(), Value::Array(calls));
            }
            messages.push(Value::Object(out));
        }
    }

    let raw_tools = tool_list(req.get("tools"), None)?;
    let tools = if raw_tools.is_empty() {
        None
    } else {
        Some(
            raw_tools
                .into_iter()
                .map(|t| {
                    json!({
                        "name": t.get("name"),
                        "description": t.get("description").and_then(|d| d.as_str()).unwrap_or(""),
                        "parameters": t.get("input_schema").cloned().unwrap_or_else(|| json!({})),
                    })
                })
                .collect(),
        )
    };

    let mut kwargs = Map::new();
    let thinking = req.get("thinking");
    let effort = req
        .get("output_config")
        .and_then(|o| o.as_object())
        .and_then(|o| o.get("effort"));

    if thinking
        .and_then(|t| t.as_object())
        .and_then(|t| t.get("type"))
        .and_then(|t| t.as_str())
        == Some("disabled")
    {
        kwargs.insert("enable_thinking".to_string(), Value::Bool(false));
    } else if effort.is_some_and(|e| !e.is_null() && e.as_str() != Some("")) {
        for (k, v) in effort_kwargs(effort)? {
            kwargs.insert(k, v);
        }
    } else if let Some(budget) = thinking
        .and_then(|t| t.as_object())
        .and_then(|t| t.get("budget_tokens"))
    {
        for (k, v) in budget_effort(budget) {
            kwargs.insert(k, v);
        }
    } else if (thinking.is_none() || thinking.is_some_and(|t| t.is_null()))
        && req
            .get("reasoning_budget_tokens")
            .is_none_or(|b| b.is_null())
        && !think_unasked
    {
        kwargs.insert("enable_thinking".to_string(), Value::Bool(false));
    }

    Ok((late_system_to_user(messages), tools, kwargs))
}
