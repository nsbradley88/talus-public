//! talus-bench server extensions E1 (opt-in top-k logprobs) and E2 (teacher-forced `POST /v1/talus/score`);
//! bench-design/server-extensions.md. Kept out of server.rs so the request handlers only gain a few lines.
//!
//! Off by default: a request without the logprobs fields sends the engine the same `GEN` line as before and gets
//! the same JSON back (no `logprobs` key change).

use crate::engine::{EngineError, ScoreRequest, TokenLogprob};
use crate::json::JsonValue;
use crate::server::ServiceState;
use crate::tokenizer::ByteTokenizer;
use std::time::Instant;

/// Default (and default maximum) of `top_logprobs` / `logprobs: k`; the run config's `logprobs_max` overrides it.
pub const DEFAULT_TOP_LOGPROBS_MAX: usize = 20;
/// E2 `top_k` maximum (the engines cap it at 64 as well).
pub const SCORE_TOP_K_MAX: usize = 64;
/// Default continuation limit for E2; the run config's `score_max_continuation` overrides it.
pub const DEFAULT_SCORE_MAX_CONTINUATION: usize = 4096;

pub const NOT_SUPPORTED_MSG: &str = "logprobs not supported by this engine";
pub const SCORE_NOT_SUPPORTED_MSG: &str = "scoring not supported by this engine";

fn cap(state: &ServiceState) -> usize {
    state
        .config
        .raw_config
        .get_usize("logprobs_max")
        .unwrap_or(DEFAULT_TOP_LOGPROBS_MAX)
}

fn max_continuation(state: &ServiceState) -> usize {
    state
        .config
        .raw_config
        .get_usize("score_max_continuation")
        .unwrap_or(DEFAULT_SCORE_MAX_CONTINUATION)
}

/// A request's E1 ask: `Ok(None)` = off. `chat`: OpenAI chat (`logprobs: true`, `top_logprobs: k`); else
/// completions (`logprobs: k`; `true` means k = 0). Errors are 400 messages (bounds, types, unsupported engine).
pub fn parse_request(
    req: &JsonValue,
    chat: bool,
    state: &ServiceState,
) -> Result<Option<usize>, String> {
    let max = cap(state);
    let k = if chat {
        let on = match req.get("logprobs") {
            None | Some(JsonValue::Null) => false,
            Some(v) => v
                .as_bool()
                .ok_or_else(|| "logprobs: must be a boolean".to_string())?,
        };
        let top = match req.get("top_logprobs") {
            None | Some(JsonValue::Null) => None,
            Some(v) => Some(
                v.as_i64()
                    .ok_or_else(|| "top_logprobs: must be an integer".to_string())?,
            ),
        };
        if !on {
            if top.is_some_and(|t| t > 0) {
                return Err("top_logprobs: requires logprobs: true".to_string());
            }
            return Ok(None);
        }
        top.unwrap_or(0)
    } else {
        match req.get("logprobs") {
            None | Some(JsonValue::Null) => return Ok(None),
            Some(JsonValue::Bool(false)) => return Ok(None),
            Some(JsonValue::Bool(true)) => 0,
            Some(v) => v
                .as_i64()
                .ok_or_else(|| "logprobs: must be an integer".to_string())?,
        }
    };
    if k < 0 || k as usize > max {
        return Err(format!(
            "{}: must be between 0 and {max}",
            if chat { "top_logprobs" } else { "logprobs" }
        ));
    }
    if !state.engine.supports_logprobs() {
        return Err(NOT_SUPPORTED_MSG.to_string());
    }
    Ok(Some(k as usize))
}

/// A logprob as a JSON number: the f32's shortest decimal form; minus infinity as -9999.0 (OpenAI's convention).
pub fn lp_json(lp: f32) -> JsonValue {
    if !lp.is_finite() {
        return JsonValue::from(-9999.0f64);
    }
    JsonValue::from(format!("{lp}").parse::<f64>().unwrap_or(lp as f64))
}

/// The token's bytes (byte-level BPE pieces may be partial UTF-8) and its lossy text.
pub fn token_bytes(tok: &ByteTokenizer, id: u32) -> Vec<u8> {
    if let Some(real) = tok.real_tokenizer() {
        return real.token_bytes(id).to_vec();
    }
    tok.decode(&[id]).into_bytes()
}

fn token_fields(tok: &ByteTokenizer, id: u32) -> (JsonValue, JsonValue) {
    let bytes = token_bytes(tok, id);
    let text = String::from_utf8_lossy(&bytes).into_owned();
    (
        JsonValue::from(text),
        JsonValue::Array(
            bytes
                .into_iter()
                .map(|b| JsonValue::from(b as i64))
                .collect(),
        ),
    )
}

/// One `logprobs.content[]` entry (OpenAI shape plus `id` and `channel`).
pub fn entry_json(tok: &ByteTokenizer, id: u32, lp: &TokenLogprob, channel: &str) -> JsonValue {
    let (text, bytes) = token_fields(tok, id);
    let top: Vec<JsonValue> = lp
        .top
        .iter()
        .map(|&(tid, tlp)| {
            let (t, b) = token_fields(tok, tid);
            JsonValue::obj(vec![
                ("token", t),
                ("logprob", lp_json(tlp)),
                ("bytes", b),
                ("id", JsonValue::from(tid as i64)),
            ])
        })
        .collect();
    JsonValue::obj(vec![
        ("token", text),
        ("logprob", lp_json(lp.logprob)),
        ("bytes", bytes),
        ("id", JsonValue::from(id as i64)),
        ("top_logprobs", JsonValue::Array(top)),
        ("channel", JsonValue::from(channel)),
    ])
}

/// `{"content": [entries]}`
pub fn logprobs_obj(entries: Vec<JsonValue>) -> JsonValue {
    JsonValue::obj(vec![("content", JsonValue::Array(entries))])
}

/// Which channel a generated chat token belongs to: tracks reasoning vs answer from the parser's events.
#[derive(Default)]
pub struct ChannelTracker {
    seen_answer: bool,
}

impl ChannelTracker {
    /// `reasoning_ev` / `answer_ev`: the token produced reasoning / non-reasoning events; `thinking`: the turn
    /// starts in reasoning. A token that produced no event (e.g. `</think>`) keeps the current channel.
    pub fn channel(&mut self, reasoning_ev: bool, answer_ev: bool, thinking: bool) -> &'static str {
        if answer_ev && !reasoning_ev {
            self.seen_answer = true;
            return "content";
        }
        if reasoning_ev {
            return "reasoning";
        }
        if self.seen_answer || !thinking {
            "content"
        } else {
            "reasoning"
        }
    }
}

fn ids_of(
    req: &JsonValue,
    ids_key: &str,
    text_key: &str,
    tok: &ByteTokenizer,
) -> Result<Vec<u32>, String> {
    if let Some(v) = req.get(ids_key).filter(|v| !v.is_null()) {
        let arr = v
            .as_array()
            .ok_or_else(|| format!("{ids_key}: must be an array of token ids"))?;
        return arr
            .iter()
            .map(|x| {
                x.as_i64()
                    .filter(|&n| n >= 0 && n <= u32::MAX as i64)
                    .map(|n| n as u32)
                    .ok_or_else(|| format!("{ids_key}: token ids must be non-negative integers"))
            })
            .collect();
    }
    if let Some(s) = req.get(text_key).and_then(|v| v.as_str()) {
        return Ok(tok.encode(s, true));
    }
    Err(format!("{ids_key} or {text_key} is required"))
}

/// The reply of `POST /v1/talus/score`: (HTTP status, body).
pub fn handle_score(
    req: &JsonValue,
    state: &ServiceState,
    peer_is_local: bool,
) -> (u16, JsonValue) {
    let err = |code: u16, msg: &str| {
        (
            code,
            JsonValue::obj(vec![(
                "error",
                JsonValue::obj(vec![
                    ("message", JsonValue::from(msg)),
                    (
                        "type",
                        JsonValue::from(if code == 501 {
                            "not_supported"
                        } else {
                            "invalid_request_error"
                        }),
                    ),
                ]),
            )]),
        )
    };
    // never exposed without auth on a LAN bind (with an API key set, the connection already passed the key check)
    if state.config.api_key.is_none() && !peer_is_local {
        return err(
            403,
            "scoring is served to loopback clients only unless an API key is set",
        );
    }
    if !state.engine.supports_score() {
        return err(501, SCORE_NOT_SUPPORTED_MSG);
    }
    let prompt_ids = match ids_of(req, "prompt_ids", "prompt", &state.tokenizer) {
        Ok(v) => v,
        Err(e) => return err(400, &e),
    };
    let continuation_ids = match ids_of(req, "continuation_ids", "continuation", &state.tokenizer) {
        Ok(v) => v,
        Err(e) => return err(400, &e),
    };
    if prompt_ids.is_empty() || continuation_ids.is_empty() {
        return err(
            400,
            "the prompt and the continuation must both be non-empty",
        );
    }
    let top_k = match req.get("top_k") {
        None | Some(JsonValue::Null) => 0,
        Some(v) => match v.as_i64() {
            Some(k) if (0..=SCORE_TOP_K_MAX as i64).contains(&k) => k as usize,
            _ => {
                return err(
                    400,
                    &format!("top_k: must be an integer between 0 and {SCORE_TOP_K_MAX}"),
                )
            }
        },
    };
    let mut want = ["logprob", "rank", "top_k", "entropy"].map(|_| true);
    if let Some(v) = req.get("return").filter(|v| !v.is_null()) {
        let Some(arr) = v.as_array() else {
            return err(400, "return: must be an array of field names");
        };
        want = [false; 4];
        for x in arr {
            match x.as_str() {
                Some("logprob") => want[0] = true,
                Some("rank") => want[1] = true,
                Some("top_k") => want[2] = true,
                Some("entropy") => want[3] = true,
                _ => {
                    return err(
                        400,
                        "return: allowed values are logprob, rank, top_k, entropy",
                    )
                }
            }
        }
    }
    let max_cont = max_continuation(state);
    if continuation_ids.len() > max_cont {
        return err(
            400,
            &format!(
                "the continuation ({} tokens) exceeds the scoring limit ({max_cont} tokens)",
                continuation_ids.len()
            ),
        );
    }
    let ctx = state.reported_ctx();
    let total = prompt_ids.len() + continuation_ids.len();
    if total > ctx {
        return err(
            400,
            &format!("prompt + continuation ({total} tokens) exceeds the context ({ctx} tokens)"),
        );
    }
    let sreq = ScoreRequest {
        prompt_ids: prompt_ids.clone(),
        continuation_ids: continuation_ids.clone(),
        top_k,
    };
    // one request for the single-request scheduler (running/waiting in status reflect it)
    let t0 = Instant::now();
    let res = {
        let _running = state.begin_score_request(prompt_ids.len());
        state.engine.score(&sreq)
    };
    let res = match res {
        Ok(r) => r,
        Err(EngineError::Unsupported(m)) => return err(501, &m),
        Err(EngineError::InvalidRequest(m)) => return err(400, &m),
        Err(e) => return err(500, &e.to_string()),
    };
    if res.positions.len() != continuation_ids.len() {
        return err(
            500,
            &format!(
                "the engine scored {} positions for {} continuation tokens",
                res.positions.len(),
                continuation_ids.len()
            ),
        );
    }
    let tok = &state.tokenizer;
    let positions: Vec<JsonValue> = res
        .positions
        .iter()
        .map(|p| {
            let (t, _) = token_fields(tok, p.id);
            let mut o = JsonValue::obj(vec![("token", t), ("id", JsonValue::from(p.id as i64))]);
            if want[0] {
                o.insert("logprob", lp_json(p.logprob));
            }
            if want[1] {
                o.insert("rank", JsonValue::from(p.rank as i64));
            }
            if want[3] {
                o.insert("entropy", lp_json(p.entropy));
            }
            if want[2] {
                let top: Vec<JsonValue> = p
                    .top
                    .iter()
                    .map(|&(id, lp)| {
                        let (t, _) = token_fields(tok, id);
                        JsonValue::obj(vec![
                            ("id", JsonValue::from(id as i64)),
                            ("token", t),
                            ("logprob", lp_json(lp)),
                        ])
                    })
                    .collect();
                o.insert("top_k", JsonValue::Array(top));
            }
            o
        })
        .collect();
    let body = JsonValue::obj(vec![
        ("model", JsonValue::from(state.model_for(req))),
        ("positions", JsonValue::Array(positions)),
        ("prompt_tokens", JsonValue::from(prompt_ids.len())),
        (
            "continuation_tokens",
            JsonValue::from(continuation_ids.len()),
        ),
        (
            "prompt_ids",
            JsonValue::Array(
                prompt_ids
                    .iter()
                    .map(|&i| JsonValue::from(i as i64))
                    .collect(),
            ),
        ),
        (
            "continuation_ids",
            JsonValue::Array(
                continuation_ids
                    .iter()
                    .map(|&i| JsonValue::from(i as i64))
                    .collect(),
            ),
        ),
        (
            "timings",
            JsonValue::obj(vec![
                ("prompt_ms", JsonValue::from(res.prompt_ms)),
                ("engine_ms", JsonValue::from(res.total_ms)),
                (
                    "wall_ms",
                    JsonValue::from((t0.elapsed().as_secs_f64() * 1e4).round() / 10.0),
                ),
            ]),
        ),
    ]);
    (200, body)
}
