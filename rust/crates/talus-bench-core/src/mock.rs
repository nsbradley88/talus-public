//! A mock OpenAI-compatible server for tests and demos (no GPU, no model). It recognises the bench's own cases
//! and answers them from the ground truth ("oracle"), then degrades a deterministic fraction of answers: wrong
//! answers (`wrong_rate`) and HTTP 500s (`error_rate`). A case is degraded or not by a hash of its prompt and the
//! seed, so runs are reproducible. Its outputs are synthetic and never appear in a real report.

use crate::case::Case;
use crate::config::Tier;
use crate::tasks::{self, TaskContext};
use crate::util::{sha256_hex, Rng};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Degrade {
    pub wrong_rate: f64,
    pub error_rate: f64,
    pub seed: u64,
}

impl Degrade {
    pub const NONE: Degrade = Degrade {
        wrong_rate: 0.0,
        error_rate: 0.0,
        seed: 0,
    };
}

/// How the mock answers teacher-forced scoring (`POST /v1/talus/score`, server extension E2) and logprob requests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScoreMode {
    /// Scores with a synthetic, deterministic distribution.
    Supported,
    /// An engine without scoring: 501 "scoring not supported by this engine", and 400 "logprobs not supported by
    /// this engine" for chat requests that ask for logprobs (talus-server's answers).
    NotSupported,
    /// A server without the route (older talus-server, other servers): 404.
    Absent,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreMock {
    pub mode: ScoreMode,
    /// 0 = the reference distribution. > 0 lowers the forced token's logit by `drift` and perturbs the others by
    /// up to +-`drift` (a degraded candidate: every forced token's logprob drops, KL > 0).
    pub drift: f64,
    pub seed: u64,
}

impl Default for ScoreMock {
    fn default() -> Self {
        ScoreMock {
            mode: ScoreMode::Supported,
            drift: 0.0,
            seed: 0,
        }
    }
}

/// Size of the mock's per-position vocabulary (every other token has probability 0).
const MOCK_SCORE_VOCAB: u64 = 32;

fn ids_or_text(b: &Value, ids: &str, text: &str) -> Result<Vec<u64>, String> {
    if let Some(a) = b[ids].as_array() {
        return a
            .iter()
            .map(|x| {
                x.as_u64()
                    .ok_or_else(|| format!("{ids}: token ids must be non-negative integers"))
            })
            .collect();
    }
    if let Some(t) = b[text].as_str() {
        return Ok(t.chars().map(|c| c as u64).collect());
    }
    Err(format!("{ids} or {text} is required"))
}

/// The mock's distribution before `tok`, given the previous token: (id, logprob) sorted by logprob, descending
/// (ties by id). Deterministic in (prev, tok) for the reference; `drift` degrades it.
fn mock_distribution(prev: u64, tok: u64, pos: usize, s: &ScoreMock) -> Vec<(u64, f64)> {
    let mut ids = vec![tok];
    let mut k = 0u64;
    while (ids.len() as u64) < MOCK_SCORE_VOCAB {
        let id = (prev.wrapping_mul(2_654_435_761).wrapping_add(k * 7919 + 13)) % 50_000;
        k += 1;
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let logits: Vec<f64> = ids
        .iter()
        .map(|&id| {
            let base = unit_hash(&format!("{prev}:{id}"), 0, 3) * 4.0 - 3.0;
            let forced = if id == tok { 2.0 - s.drift } else { 0.0 };
            // the other tokens move by at most +-drift, so the forced token's logprob can only drop
            let noise = if s.drift > 0.0 && id != tok {
                s.drift * (unit_hash(&format!("{prev}:{id}:{pos}"), s.seed, 4) * 2.0 - 1.0)
            } else {
                0.0
            };
            base + forced + noise
        })
        .collect();
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = m + logits.iter().map(|l| (l - m).exp()).sum::<f64>().ln();
    let mut d: Vec<(u64, f64)> = ids
        .into_iter()
        .zip(logits.iter().map(|l| l - lse))
        .collect();
    d.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    d
}

/// The reply to `POST /v1/talus/score` in talus-server's shape (positions, ids echoed, timings).
pub fn score_reply(b: &Value, s: &ScoreMock) -> (u16, Value) {
    let err = |code: u16, m: &str| {
        (
            code,
            json!({"error": {"message": m, "type": if code == 501 { "not_supported" } else { "invalid_request_error" }}}),
        )
    };
    match s.mode {
        ScoreMode::NotSupported => return err(501, "scoring not supported by this engine"),
        ScoreMode::Absent => return (404, json!({"error": {"message": "not found"}})),
        ScoreMode::Supported => {}
    }
    let prompt = match ids_or_text(b, "prompt_ids", "prompt") {
        Ok(v) => v,
        Err(e) => return err(400, &e),
    };
    let cont = match ids_or_text(b, "continuation_ids", "continuation") {
        Ok(v) => v,
        Err(e) => return err(400, &e),
    };
    if prompt.is_empty() || cont.is_empty() {
        return err(
            400,
            "the prompt and the continuation must both be non-empty",
        );
    }
    let top_k = match &b["top_k"] {
        Value::Null => 0,
        v => match v.as_u64().filter(|k| *k <= 64) {
            Some(k) => k as usize,
            None => return err(400, "top_k: must be an integer between 0 and 64"),
        },
    };
    if cont.len() > 4096 {
        return err(
            400,
            "the continuation exceeds the scoring limit (4096 tokens)",
        );
    }
    let positions: Vec<Value> = cont
        .iter()
        .enumerate()
        .map(|(j, &tok)| {
            let prev = if j == 0 {
                *prompt.last().unwrap_or(&0)
            } else {
                cont[j - 1]
            };
            let d = mock_distribution(prev, tok, j, s);
            let lp = d
                .iter()
                .find(|x| x.0 == tok)
                .map(|x| x.1)
                .unwrap_or(-9999.0);
            let rank = d.iter().position(|x| x.0 == tok).unwrap_or(0);
            let entropy: f64 = -d.iter().map(|x| x.1.exp() * x.1).sum::<f64>();
            let top: Vec<Value> = d
                .iter()
                .take(top_k)
                .map(|&(id, l)| json!({"id": id, "token": format!("<{id}>"), "logprob": l as f32}))
                .collect();
            json!({"token": format!("<{tok}>"), "id": tok, "logprob": lp as f32, "rank": rank,
                   "entropy": entropy as f32, "top_k": top})
        })
        .collect();
    (
        200,
        json!({"model": "mock-model", "positions": positions, "prompt_tokens": prompt.len(),
               "continuation_tokens": cont.len(), "prompt_ids": prompt, "continuation_ids": cont,
               "timings": {"prompt_ms": 0.1, "engine_ms": 0.2, "wall_ms": 0.3}}),
    )
}

/// Last user message -> the case it belongs to (every embedded case of every native task, smoke tier).
fn index() -> &'static HashMap<String, Case> {
    static IDX: OnceLock<HashMap<String, Case>> = OnceLock::new();
    IDX.get_or_init(|| {
        let mut m = HashMap::new();
        for (id, a, _) in tasks::KNOWN {
            if *a != tasks::TaskAvailability::Native {
                continue;
            }
            let t = tasks::native(id).expect("native");
            if t.needs_server() {
                continue;
            }
            for c in t
                .cases(&TaskContext::for_tests(Tier::Smoke))
                .unwrap_or_default()
            {
                if let Some(u) = last_user(&Value::Array(c.messages.clone())) {
                    m.insert(u, c);
                }
            }
        }
        m
    })
}

fn last_user(messages: &Value) -> Option<String> {
    messages
        .as_array()?
        .iter()
        .rev()
        .find(|m| m["role"] == "user")
        .and_then(|m| m["content"].as_str())
        .map(str::to_string)
}

/// A schema-valid instance: required properties, minimum sizes, first enum/const/branch.
pub fn instance(schema: &Value) -> Value {
    let s = match schema {
        Value::Object(o) => o,
        _ => return Value::Null,
    };
    if let Some(c) = s.get("const") {
        return c.clone();
    }
    if let Some(Value::Array(e)) = s.get("enum") {
        return e.first().cloned().unwrap_or(Value::Null);
    }
    for k in ["oneOf", "anyOf"] {
        if let Some(Value::Array(b)) = s.get(k) {
            return b.first().map(instance).unwrap_or(Value::Null);
        }
    }
    let t = match s.get("type") {
        Some(Value::String(t)) => t.as_str(),
        Some(Value::Array(ts)) => ts.first().and_then(Value::as_str).unwrap_or("null"),
        _ => {
            if s.contains_key("properties") {
                "object"
            } else {
                "null"
            }
        }
    };
    match t {
        "object" => {
            let mut o = serde_json::Map::new();
            let props = s.get("properties").and_then(Value::as_object);
            for r in s
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                let sub = props
                    .and_then(|p| p.get(r))
                    .cloned()
                    .unwrap_or(Value::Bool(true));
                o.insert(r.to_string(), instance(&sub));
            }
            Value::Object(o)
        }
        "array" => {
            let n = s.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
            let item = s.get("items").cloned().unwrap_or(Value::Bool(true));
            (0..n)
                .map(|i| match instance(&item) {
                    Value::String(x) => Value::String(format!("{x}{i}")),
                    Value::Number(_) if item["type"] == "integer" || item["type"] == "number" => {
                        json!(i)
                    }
                    v => v,
                })
                .collect()
        }
        "string" => {
            let n = s
                .get("minLength")
                .and_then(Value::as_u64)
                .unwrap_or(1)
                .max(1) as usize;
            Value::String("x".repeat(n))
        }
        "integer" | "number" => {
            let lo = s
                .get("minimum")
                .and_then(Value::as_f64)
                .or(s
                    .get("exclusiveMinimum")
                    .and_then(Value::as_f64)
                    .map(|x| x + 1.0))
                .unwrap_or(0.0);
            json!(lo.ceil() as i64)
        }
        "boolean" => Value::Bool(false),
        _ => Value::Null,
    }
}

fn set_pointer(doc: &mut Value, ptr: &str, v: Value) {
    let mut cur = doc;
    let parts: Vec<&str> = ptr.trim_start_matches('/').split('/').collect();
    for (i, p) in parts.iter().enumerate() {
        let last = i + 1 == parts.len();
        if let Ok(idx) = p.parse::<usize>() {
            if let Value::Array(a) = cur {
                while a.len() <= idx {
                    a.push(Value::Null);
                }
                if last {
                    a[idx] = v;
                    return;
                }
                cur = &mut a[idx];
                continue;
            }
        }
        if !cur.is_object() {
            *cur = json!({});
        }
        let o = cur.as_object_mut().expect("object");
        if last {
            o.insert(p.to_string(), v);
            return;
        }
        cur = o.entry(p.to_string()).or_insert(json!({}));
    }
}

/// The correct answer text for a case.
/// A mock reply: answer text and tool calls (null = none).
pub struct MockAnswer {
    pub content: String,
    pub tool_calls: Value,
}

impl MockAnswer {
    fn text(s: String) -> Self {
        MockAnswer {
            content: s,
            tool_calls: Value::Null,
        }
    }
}

/// The first concrete value of a possible-answer option (dicts hold option lists per key).
fn bfcl_value(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(
            m.iter()
                .filter_map(|(k, opts)| {
                    let first = opts.as_array()?.iter().find(|x| **x != json!(""))?;
                    Some((k.clone(), bfcl_value(first)))
                })
                .collect(),
        ),
        Value::Array(a) if a.iter().all(Value::is_object) && !a.is_empty() => {
            Value::Array(a.iter().map(bfcl_value).collect())
        }
        x => x.clone(),
    }
}

fn bfcl_calls(c: &Case) -> Value {
    let mut calls = Vec::new();
    for (i, gt) in c.expected["ground_truth"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let Some((name, params)) = gt.as_object().and_then(|m| m.iter().next()) else {
            continue;
        };
        let mut args = serde_json::Map::new();
        for (k, opts) in params.as_object().into_iter().flatten() {
            if let Some(v) = opts
                .as_array()
                .and_then(|o| o.iter().find(|x| **x != json!("")))
            {
                args.insert(k.clone(), bfcl_value(v));
            }
        }
        calls.push(json!({"id": format!("call_{i}"), "type": "function",
                          "function": {"name": name.replace('.', "_"), "arguments": Value::Object(args).to_string()}}));
    }
    Value::Array(calls)
}

/// The correct answer for a case (IFEval: a fixed text, as the mock cannot follow arbitrary instructions).
pub fn oracle(c: &Case) -> MockAnswer {
    match c.task.as_str() {
        "gsm8k" => MockAnswer::text(format!(
            "Working through it step by step.\n#### {}",
            c.expected["answer"].as_str().unwrap_or("0")
        )),
        "json_schema" | "json_schema_strict" => {
            let mut doc = instance(&c.meta["schema"]);
            for (p, v) in c.expected["must"].as_object().into_iter().flatten() {
                set_pointer(&mut doc, p, v.clone());
            }
            MockAnswer::text(doc.to_string())
        }
        "mmlu_pro" => MockAnswer::text(format!(
            "Let's reason it through.\nThe answer is ({}).",
            c.expected["answer"].as_str().unwrap_or("A")
        )),
        "bfcl" if c.expected["category"] == "irrelevance" => {
            MockAnswer::text("None of the available functions apply.".into())
        }
        "bfcl" => MockAnswer {
            content: String::new(),
            tool_calls: bfcl_calls(c),
        },
        "greedy_identity" => MockAnswer::text(format!(
            "deterministic answer {}",
            &sha256_hex(
                c.messages[0]["content"]
                    .as_str()
                    .unwrap_or_default()
                    .as_bytes()
            )[..16]
        )),
        _ => MockAnswer::text("This is a mock response.".into()),
    }
}

/// A plausible wrong answer for a case.
pub fn wrong(c: &Case) -> MockAnswer {
    match c.task.as_str() {
        "gsm8k" => {
            let n: f64 = c.expected["answer"]
                .as_str()
                .and_then(|a| a.parse().ok())
                .unwrap_or(0.0);
            MockAnswer::text(format!("I think it is this.\n#### {}", n + 1.0))
        }
        "json_schema" | "json_schema_strict" => {
            MockAnswer::text(format!("```json\n{}\n```", oracle(c).content))
        }
        "mmlu_pro" => {
            let g = c.expected["answer"]
                .as_str()
                .unwrap_or("A")
                .chars()
                .next()
                .unwrap_or('A');
            let w = if g == 'A' { 'B' } else { 'A' };
            MockAnswer::text(format!("The answer is ({w})."))
        }
        "bfcl" if c.expected["category"] == "irrelevance" => MockAnswer {
            content: String::new(),
            tool_calls: json!([{"id": "call_0", "type": "function", "function": {"name": "irrelevant", "arguments": "{}"}}]),
        },
        "bfcl" => MockAnswer::text("I would call a function here.".into()),
        "greedy_identity" => MockAnswer::text(format!("{} (drifted)", oracle(c).content)),
        _ => MockAnswer::text("no idea".into()),
    }
}

/// Needle prompts: answer from the inserted comments (drop the last one when degraded).
fn needle_answer(user: &str, degrade: bool) -> Option<String> {
    let re = regex::Regex::new(
        r"# NOTE: the deployment codename for the (\w+) module is ([a-z]+-\d+)\.",
    )
    .ok()?;
    let mut found: Vec<String> = re
        .captures_iter(user)
        .map(|c| format!("{}: {}", &c[1], &c[2]))
        .collect();
    if found.is_empty() {
        return None;
    }
    if degrade {
        found.pop();
    }
    Some(found.join("\n"))
}

fn unit_hash(s: &str, seed: u64, salt: u64) -> f64 {
    let h = sha256_hex(format!("{seed}:{salt}:{s}").as_bytes());
    let mut r = Rng::new(u64::from_str_radix(&h[..16], 16).unwrap_or(0));
    r.unit()
}

/// The reply for one chat request (status, body).
pub fn chat_reply(req: &Value, d: Degrade) -> (u16, Value) {
    let user = last_user(&req["messages"]).unwrap_or_default();
    if d.error_rate > 0.0 && unit_hash(&user, d.seed, 1) < d.error_rate {
        return (500, json!({"error": {"message": "mock engine failure"}}));
    }
    let degrade = d.wrong_rate > 0.0 && unit_hash(&user, d.seed, 2) < d.wrong_rate;
    let ans = match index().get(&user) {
        Some(c) if degrade => wrong(c),
        Some(c) => oracle(c),
        None => match needle_answer(&user, degrade) {
            Some(a) => MockAnswer::text(a),
            None => scripted_agent(req)
                .unwrap_or_else(|| MockAnswer::text("I cannot help with that.".into())),
        },
    };
    let pt = req["messages"].to_string().len() / 4;
    let ct = ans.content.len() / 4 + 1;
    let mut msg = json!({"role": "assistant", "content": if ans.content.is_empty() { Value::Null } else { json!(ans.content) }});
    let finish = if ans.tool_calls.is_null() {
        "stop"
    } else {
        "tool_calls"
    };
    if !ans.tool_calls.is_null() {
        msg["tool_calls"] = ans.tool_calls;
    }
    (
        200,
        json!({
            "id": "chatcmpl-mock", "object": "chat.completion", "model": req["model"],
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        }),
    )
}

/// Marks a handler result that `write_reply` sends as server-sent events (one `data:` line per element).
pub const SSE_KEY: &str = "__sse";

/// A non-streaming chat completion as the `stream: true` chunk sequence (role+content, tool calls, finish, usage).
pub fn as_stream(v: &Value) -> Value {
    let ch = &v["choices"][0];
    let msg = &ch["message"];
    let base = |delta: Value, finish: Value| {
        json!({"id": v["id"], "object": "chat.completion.chunk", "model": v["model"],
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    };
    let mut ev = vec![base(
        json!({"role": "assistant", "content": msg["content"].as_str().unwrap_or("")}),
        Value::Null,
    )];
    if let Some(calls) = msg["tool_calls"].as_array() {
        for (i, c) in calls.iter().enumerate() {
            let mut c = c.clone();
            c["index"] = json!(i);
            ev.push(base(json!({"tool_calls": [c]}), Value::Null));
        }
    }
    ev.push(base(json!({}), ch["finish_reason"].clone()));
    let mut u = base(json!({}), Value::Null);
    u["choices"] = json!([]);
    u["usage"] = v["usage"].clone();
    ev.push(u);
    json!({ SSE_KEY: ev })
}

/// A scripted coding agent turn for requests that offer a `bash` tool (the SWE-bench trial's agent path):
/// first one benign tool call that also leaves a patch, then, once a tool result is in the history, a final answer.
fn scripted_agent(req: &Value) -> Option<MockAnswer> {
    let offers_bash = req["tools"]
        .as_array()?
        .iter()
        .any(|t| t["function"]["name"] == "bash");
    if !offers_bash {
        return None;
    }
    let msgs: Vec<&Value> = req["messages"].as_array().into_iter().flatten().collect();
    let has_result = msgs.iter().any(|m| m["role"] == "tool");
    // the mini-SWE-agent loop (its prompt asks for an explicit submit): edit, write patch.txt, submit
    if msgs.iter().any(|m| {
        m["content"]
            .to_string()
            .contains("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
    }) {
        let step = msgs.iter().filter(|m| m["role"] == "tool").count();
        let cmd = match step {
            0 => "f=$(git ls-files | grep -v '^\\.' | head -1); echo '# tb-mock fix' >> \"$f\"; git status --short",
            1 => "git diff > patch.txt; cat patch.txt | head -5",
            _ => "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt",
        };
        return Some(MockAnswer {
            content: format!("THOUGHT: step {step}"),
            tool_calls: json!([{"id": format!("call_mini_{step}"), "type": "function", "function": {"name": "bash",
                "arguments": json!({"command": cmd}).to_string()}}]),
        });
    }
    Some(if has_result {
        MockAnswer::text("Done: inspected the repository and left a marker file.".into())
    } else {
        MockAnswer {
            content: String::new(),
            tool_calls: json!([{"id": "call_tb_trial", "type": "function", "function": {"name": "bash",
                "arguments": json!({"command": "git status --short | head -5; git log --oneline -3; ls / >/dev/null; find / -maxdepth 3 -name '*.cfg' 2>/dev/null | head -3; python -c 'import sys; print(sys.version, sys.path)'; pip list 2>/dev/null | head -3; timeout 120 python -m pytest --collect-only -q 2>/dev/null | tail -2; echo trial > tb_trial_marker.txt",
                                    "description": "inspect the repo"}).to_string()}}]),
        }
    })
}

/// A character-level stand-in for talus-server's `/tokenize` and `/detokenize` (one token per char).
pub fn tokenizer_reply(path: &str, b: &Value) -> (u16, Value) {
    if path == "/tokenize" {
        let t: Vec<u32> = b["content"]
            .as_str()
            .unwrap_or_default()
            .chars()
            .map(|c| c as u32)
            .collect();
        (200, json!({"tokens": t}))
    } else {
        let s: String = b["tokens"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|t| t.as_u64().and_then(|x| char::from_u32(x as u32)))
            .collect();
        (200, json!({"content": s}))
    }
}

pub type Handler = Arc<dyn Fn(&str, &str, &Value) -> (u16, Value) + Send + Sync>;

/// Read one HTTP request: (method, path, auth bearer, JSON body).
pub fn read_request(
    stream: &TcpStream,
) -> std::io::Result<(String, String, Option<String>, Value)> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut len = 0usize;
    let mut auth = None;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
        if lower.starts_with("authorization:") {
            auth = h
                .split_once(':')
                .map(|(_, v)| v.trim().trim_start_matches("Bearer ").to_string());
        }
    }
    let mut body = vec![0u8; len.min(16 << 20)];
    reader.read_exact(&mut body)?;
    Ok((
        method,
        path,
        auth,
        serde_json::from_slice(&body).unwrap_or(Value::Null),
    ))
}

pub fn write_reply(mut stream: &TcpStream, code: u16, v: &Value) -> std::io::Result<()> {
    if let Some(events) = v.get(SSE_KEY).and_then(Value::as_array) {
        let mut text = String::new();
        for e in events {
            text.push_str(&format!("data: {e}\n\n"));
        }
        text.push_str("data: [DONE]\n\n");
        let head = format!(
            "HTTP/1.1 {code} X\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            text.len()
        );
        stream.write_all(head.as_bytes())?;
        stream.write_all(text.as_bytes())?;
        return stream.flush();
    }
    let text = v.to_string();
    let head = format!(
        "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        text.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(text.as_bytes())?;
    stream.flush()
}

/// Serve `handler(method, path, body)` on 127.0.0.1:`port` (0 = any) in background threads.
pub fn spawn(port: u16, handler: Handler) -> std::io::Result<SocketAddr> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let addr = listener.local_addr()?;
    std::thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            let h = Arc::clone(&handler);
            std::thread::spawn(move || {
                if let Ok((m, p, _, b)) = read_request(&conn) {
                    let (code, v) = h(&m, &p, &b);
                    let _ = write_reply(&conn, code, &v);
                }
            });
        }
    });
    Ok(addr)
}

/// A plain mock endpoint (no Talus control API) with fixed degradation. Returns the address and a request counter.
/// It answers teacher-forced scoring with the reference distribution ([`spawn_openai_with`] changes that).
pub fn spawn_openai(port: u16, d: Degrade) -> std::io::Result<(SocketAddr, Arc<Mutex<u64>>)> {
    spawn_openai_with(port, d, ScoreMock::default())
}

/// [`spawn_openai`] with a chosen scoring behaviour (drift, or an engine/server without scoring).
pub fn spawn_openai_with(
    port: u16,
    d: Degrade,
    sm: ScoreMock,
) -> std::io::Result<(SocketAddr, Arc<Mutex<u64>>)> {
    let count = Arc::new(Mutex::new(0u64));
    let c2 = Arc::clone(&count);
    let addr = spawn(
        port,
        Arc::new(move |m: &str, p: &str, b: &Value| match (m, p) {
            ("GET", "/health") => (200, json!({"status": "ok"})),
            ("GET", "/v1/models") => (
                200,
                json!({"object": "list", "data": [{"id": "mock-model", "object": "model"}]}),
            ),
            ("POST", "/v1/chat/completions") => {
                *c2.lock().unwrap() += 1;
                let asks_logprobs = b["logprobs"] == true || b["top_logprobs"].is_u64();
                if asks_logprobs && sm.mode != ScoreMode::Supported {
                    return (
                        400,
                        json!({"error": {"message": "logprobs not supported by this engine", "type": "invalid_request_error"}}),
                    );
                }
                let (code, v) = chat_reply(b, d);
                if code == 200 && b["stream"] == true {
                    (code, as_stream(&v))
                } else {
                    (code, v)
                }
            }
            ("POST", "/tokenize") | ("POST", "/detokenize") => tokenizer_reply(p, b),
            ("POST", "/v1/talus/score") => score_reply(b, &sm),
            _ => (404, json!({"error": {"message": "not found"}})),
        }),
    )?;
    Ok((addr, count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::RawOutput;

    #[test]
    fn oracle_passes_every_embedded_case() {
        for (id, a, _) in tasks::KNOWN
            .iter()
            .filter(|k| k.1 == tasks::TaskAvailability::Native)
        {
            let _ = a;
            let t = tasks::native(id).unwrap();
            if t.needs_server() {
                continue;
            }
            let cs = t
                .cases(&TaskContext {
                    limit: None,
                    ..TaskContext::for_tests(Tier::Smoke)
                })
                .unwrap();
            for c in &cs {
                let o = oracle(c);
                let out = RawOutput {
                    content: o.content,
                    tool_calls: o.tool_calls,
                    finish_reason: "stop".into(),
                    ..Default::default()
                };
                let s = t.score(c, &out);
                if c.task == "ifeval" {
                    continue; // the mock cannot follow arbitrary instructions
                }
                assert!(
                    s.pass,
                    "{}: oracle {:?} -> {:?}",
                    c.case_id, out.content, s.details
                );
                let w = wrong(c);
                let bad = RawOutput {
                    content: w.content,
                    tool_calls: w.tool_calls,
                    finish_reason: "stop".into(),
                    ..Default::default()
                };
                if c.task != "greedy_identity" {
                    assert!(!t.score(c, &bad).pass, "{}: wrong answer passed", c.case_id);
                }
            }
        }
    }

    #[test]
    fn streaming_and_scripted_agent() {
        let req = json!({"model": "m", "stream": true, "tools": [{"type": "function", "function": {"name": "bash"}}],
                         "messages": [{"role": "user", "content": "fix the bug"}]});
        let (code, v) = chat_reply(&req, Degrade::NONE);
        assert_eq!(code, 200);
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        let s = as_stream(&v);
        let ev = s[SSE_KEY].as_array().unwrap();
        assert_eq!(
            ev[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "bash"
        );
        assert_eq!(ev[2]["choices"][0]["finish_reason"], "tool_calls");
        assert!(ev.last().unwrap()["usage"].is_object());
        let mut req2 = req.clone();
        req2["messages"]
            .as_array_mut()
            .unwrap()
            .push(json!({"role": "tool", "tool_call_id": "call_tb_trial", "content": "ok"}));
        assert_eq!(
            chat_reply(&req2, Degrade::NONE).1["choices"][0]["finish_reason"],
            "stop"
        );
    }

    #[test]
    fn score_mock_shapes_and_modes() {
        let req = json!({"prompt_ids": [1, 2], "continuation_ids": [3, 4, 5], "top_k": 20});
        let (code, v) = score_reply(&req, &ScoreMock::default());
        assert_eq!(code, 200);
        let pos = v["positions"].as_array().unwrap();
        assert_eq!(pos.len(), 3);
        assert_eq!(pos[1]["id"], 4);
        assert_eq!(pos[1]["top_k"].as_array().unwrap().len(), 20);
        // the top-k list is descending and covers less than the whole distribution (a tail exists)
        let lps: Vec<f64> = pos[0]["top_k"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["logprob"].as_f64().unwrap())
            .collect();
        assert!(lps.windows(2).all(|w| w[0] >= w[1]));
        let mass: f64 = lps.iter().map(|l| l.exp()).sum();
        assert!(mass < 1.0 && mass > 0.5, "{mass}");
        assert_eq!(v["continuation_ids"], json!([3, 4, 5]));
        // deterministic, and drift changes the distribution and lowers the forced token's logprob
        assert_eq!(score_reply(&req, &ScoreMock::default()).1, v);
        let drifted = ScoreMock {
            drift: 0.5,
            seed: 3,
            ..Default::default()
        };
        let (_, w) = score_reply(&req, &drifted);
        let sum = |x: &Value| -> f64 {
            x["positions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["logprob"].as_f64().unwrap())
                .sum()
        };
        assert!(sum(&w) < sum(&v));
        let no = ScoreMock {
            mode: ScoreMode::NotSupported,
            ..Default::default()
        };
        assert_eq!(score_reply(&req, &no).0, 501);
        let absent = ScoreMock {
            mode: ScoreMode::Absent,
            ..Default::default()
        };
        assert_eq!(score_reply(&req, &absent).0, 404);
        assert_eq!(
            score_reply(&json!({"prompt_ids": [1]}), &ScoreMock::default()).0,
            400
        );
        assert_eq!(
            score_reply(
                &json!({"prompt_ids": [1], "continuation_ids": [2], "top_k": 65}),
                &ScoreMock::default()
            )
            .0,
            400
        );
    }

    #[test]
    fn degradation_is_deterministic_and_close_to_the_rate() {
        let msgs: Vec<String> = index().keys().cloned().collect();
        let d = Degrade {
            wrong_rate: 0.3,
            error_rate: 0.0,
            seed: 9,
        };
        let wrong = msgs
            .iter()
            .filter(|u| unit_hash(u, d.seed, 2) < d.wrong_rate)
            .count();
        let again = msgs
            .iter()
            .filter(|u| unit_hash(u, d.seed, 2) < d.wrong_rate)
            .count();
        assert_eq!(wrong, again);
        let frac = wrong as f64 / msgs.len() as f64;
        assert!((0.15..0.45).contains(&frac), "{frac}");
    }
}
