//! Cases (what is asked) and case results (what came back and how it scored): the per-case schema of spec §9.
//! One `CaseResult` per (model, case) goes to `cases.jsonl`; pairing joins reference and candidate rows by
//! `case_id` and refuses to pair rows whose prompt hash, dataset revision or scorer version differ.

use crate::util::hash_json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// One evaluation item.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Case {
    pub task: String,
    /// Stable id, `<task>/<dataset-native id or index>`.
    pub case_id: String,
    pub dataset: String,
    pub dataset_revision: String,
    /// OpenAI chat messages sent as is.
    pub messages: Vec<Value>,
    /// Ground truth for the scorer.
    pub expected: Value,
    /// Task-specific data the scorer needs (e.g. the JSON schema).
    #[serde(default)]
    pub meta: Value,
    /// A baseline-pass / candidate-fail on this case needs manual review (spec §8).
    #[serde(default)]
    pub critical: bool,
    /// Task-defined output budget (replaces `inference.max_output_tokens` when set).
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// OpenAI `tools` sent with the request (function-calling tasks), else null.
    #[serde(default)]
    pub tools: Value,
    /// Request fields this task fixes regardless of the inference settings (e.g. the needle gate's sampling,
    /// the greedy-identity mixes' seeds), merged over the generated body; null = none.
    #[serde(default)]
    pub request: Value,
}

impl Case {
    /// sha256 over everything the model sees for this case: messages, the per-case token budget, and the tools
    /// and request overrides when present (absent ones are left out, so older hashes stay valid).
    pub fn prompt_hash(&self) -> String {
        let mut v = json!({"messages": self.messages, "max_tokens": self.max_tokens});
        if !self.tools.is_null() {
            v["tools"] = self.tools.clone();
        }
        if !self.request.is_null() {
            v["request"] = self.request.clone();
        }
        hash_json(&v)
    }
}

/// Execution status of one case. Every status stays in the denominator; only `Ok` and `Truncated` were scored
/// on real output, the rest count as failures and are listed in the report.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CaseStatus {
    /// Got a complete answer; scored.
    Ok,
    /// The answer hit max_tokens (finish_reason length); scored on what came back, flagged.
    Truncated,
    /// The output budget went to reasoning and no answer content came back.
    TruncatedByReasoning,
    /// No reply within request_timeout_s.
    Timeout,
    /// Connection refused / reset / unresolvable.
    ConnectError,
    /// HTTP 4xx (the request was rejected).
    HttpError,
    /// HTTP 5xx (engine failure, OOM, unloaded...).
    ServerError,
    /// Reply was not a chat completion.
    ProtocolError,
    /// Present in the other run but not in this one (pairing only).
    Missing,
}

impl CaseStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            CaseStatus::Ok => "ok",
            CaseStatus::Truncated => "truncated",
            CaseStatus::TruncatedByReasoning => "truncated_by_reasoning",
            CaseStatus::Timeout => "timeout",
            CaseStatus::ConnectError => "connect_error",
            CaseStatus::HttpError => "http_error",
            CaseStatus::ServerError => "server_error",
            CaseStatus::ProtocolError => "protocol_error",
            CaseStatus::Missing => "missing",
        }
    }

    /// The model produced output that was scored.
    pub fn scored(self) -> bool {
        matches!(self, CaseStatus::Ok | CaseStatus::Truncated)
    }
}

/// What the model returned (content only is scored; reasoning is kept for audit).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct RawOutput {
    pub content: String,
    #[serde(default)]
    pub reasoning_content: String,
    #[serde(default)]
    pub tool_calls: Value,
    #[serde(default)]
    pub finish_reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Score {
    pub pass: bool,
    /// 1.0 / 0.0 for binary tasks.
    pub value: f64,
    /// What the scorer extracted from the output (e.g. the number, the parsed JSON).
    #[serde(default)]
    pub parsed: Value,
    /// Scorer-specific details (violations, extraction method...).
    #[serde(default)]
    pub details: Value,
}

impl Score {
    pub fn binary(pass: bool, parsed: Value, details: Value) -> Self {
        Self {
            pass,
            value: if pass { 1.0 } else { 0.0 },
            parsed,
            details,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Usage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    /// `usage.completion_tokens_details.reasoning_tokens` (talus-server, OpenAI).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
}

/// Passive timings of the quality request (not a performance measurement; see the perf tier).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Timings {
    pub client_ms: f64,
    pub server_prompt_ms: Option<f64>,
    pub server_decode_ms: Option<f64>,
    pub draft_n: Option<u64>,
    pub draft_accepted: Option<u64>,
    /// Decode split by channel (talus-server `timings.reasoning_n` / `answer_n` / `reasoning_ms` / `answer_ms`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_n: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_n: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_ms: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CaseResult {
    pub experiment_id: String,
    pub model_id: String,
    pub task: String,
    pub case_id: String,
    /// The sampling seed of this trial when the run repeats cases over `inference.seeds`; pairing joins on
    /// (case_id, seed) and statistics average the trials per case.
    #[serde(default)]
    pub seed: Option<u64>,
    pub dataset: String,
    pub dataset_revision: String,
    pub prompt_hash: String,
    pub scorer_version: String,
    /// sha256 of the canonical generation settings; the settings themselves are in the manifest.
    pub settings_hash: String,
    pub status: CaseStatus,
    /// Prompt messages, or null when redacted / not stored.
    #[serde(default)]
    pub prompt: Value,
    #[serde(default)]
    pub expected: Value,
    /// Null when raw outputs are not stored.
    #[serde(default)]
    pub raw_output: Option<RawOutput>,
    pub score: Score,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub timings: Timings,
    /// Transport attempts (1 = no retry).
    pub attempts: u32,
    #[serde(default)]
    pub critical: bool,
    pub started_at: String,
}

impl CaseResult {
    pub fn pass(&self) -> bool {
        self.status.scored() && self.score.pass
    }

    /// Score in [0, 1] for averaging over trials: the scorer's value when scored, else 0.
    pub fn value(&self) -> f64 {
        if self.status.scored() {
            self.score.value.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// A transient transport/server failure that a resumed run should retry rather than keep.
    pub fn retryable(&self) -> bool {
        matches!(
            self.status,
            CaseStatus::Timeout
                | CaseStatus::ConnectError
                | CaseStatus::ServerError
                | CaseStatus::ProtocolError
        )
    }

    /// The pairing key.
    pub fn key(&self) -> (String, Option<u64>) {
        (self.case_id.clone(), self.seed)
    }
}

/// Read a cases.jsonl file.
pub fn read_jsonl(path: &std::path::Path) -> Result<Vec<CaseResult>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str(l).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))
        })
        .collect()
}

pub fn to_jsonl(rows: &[CaseResult]) -> String {
    let mut s = String::new();
    for r in rows {
        s.push_str(&serde_json::to_string(r).unwrap_or_default());
        s.push('\n');
    }
    s
}
