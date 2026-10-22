//! The status snapshot from `GET /v1/talus/status`, and the numbers the status screen derives from it.

use serde_json::Value;
use talus_presets::Selection;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Request {
    pub prompt_tokens: u64,
    pub reused: u64,
    pub output_tokens: u64,
    pub prompt_ms: f64,
    pub decode_ms: f64,
    pub ttft_ms: f64,
    pub drafts: u64,
    pub accepted: u64,
    pub finish: String,
}

impl Request {
    /// Answer (decode) speed in tokens per second.
    pub fn decode_tps(&self) -> Option<f64> {
        (self.decode_ms > 0.0 && self.output_tokens > 1)
            .then(|| self.output_tokens as f64 * 1000.0 / self.decode_ms)
    }

    /// Prompt reading speed over the tokens actually read (reused tokens excluded).
    pub fn prompt_tps(&self) -> Option<f64> {
        let fresh = self.prompt_tokens.saturating_sub(self.reused);
        (self.prompt_ms > 0.0 && fresh > 0).then(|| fresh as f64 * 1000.0 / self.prompt_ms)
    }

    pub fn acceptance(&self) -> Option<f64> {
        (self.drafts > 0).then(|| self.accepted as f64 / self.drafts as f64)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Totals {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    pub drafts_offered: u64,
    pub drafts_accepted: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AppliedSelection {
    pub model: String,
    pub level: String,
    pub choices: Selection,
    pub bind: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    pub model: String,
    /// Catalog id of the served model, when the server knows it.
    pub preset: Option<String>,
    pub state: String,
    pub loaded: bool,
    pub running: u64,
    pub queued: u64,
    pub uptime_s: u64,
    pub ctx_max: u64,
    pub ctx_in_use: u64,
    pub mem_avail_gib: Option<f64>,
    pub mem_total_gib: Option<f64>,
    pub totals: Totals,
    /// Newest first.
    pub recent: Vec<Request>,
    pub host: String,
    pub port: u16,
    pub api_key: bool,
    pub api_key_from_env: bool,
    pub control_enabled: bool,
    pub selection: Option<AppliedSelection>,
    /// The last applied settings that failed to start (the server went back to the previous ones).
    pub last_error: Option<String>,
}

fn u(v: &Value, k: &str) -> u64 {
    v.get(k).and_then(Value::as_u64).unwrap_or(0)
}

fn f(v: &Value, k: &str) -> f64 {
    v.get(k).and_then(Value::as_f64).unwrap_or(0.0)
}

fn s(v: &Value, k: &str) -> String {
    v.get(k)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

pub fn parse_selection(v: &Value) -> Option<AppliedSelection> {
    let model = v.get("model")?.as_str()?.to_string();
    let choices = v
        .get("choices")
        .and_then(Value::as_object)
        .map(|o| {
            o.iter()
                .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string())))
                .collect()
        })
        .unwrap_or_default();
    Some(AppliedSelection {
        model,
        level: s(v, "level"),
        choices,
        bind: s(v, "bind"),
    })
}

pub fn parse_status(v: &Value) -> Snapshot {
    let null = Value::Null;
    let ctx = v.get("context").unwrap_or(&null);
    let mem = v.get("memory").unwrap_or(&null);
    let tot = v.get("totals").unwrap_or(&null);
    let srv = v.get("server").unwrap_or(&null);
    let ctl = v.get("control").unwrap_or(&null);
    let recent = v
        .get("recent")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|r| Request {
                    prompt_tokens: u(r, "prompt_tokens"),
                    reused: u(r, "reused"),
                    output_tokens: u(r, "output_tokens"),
                    prompt_ms: f(r, "prompt_ms"),
                    decode_ms: f(r, "decode_ms"),
                    ttft_ms: f(r, "ttft_ms"),
                    drafts: u(r, "drafts"),
                    accepted: u(r, "accepted"),
                    finish: s(r, "finish"),
                })
                .collect()
        })
        .unwrap_or_default();
    Snapshot {
        model: s(v, "model"),
        preset: v.get("preset").and_then(Value::as_str).map(str::to_string),
        state: s(v, "state"),
        loaded: v.get("loaded").and_then(Value::as_bool).unwrap_or(false),
        running: u(v, "running"),
        queued: u(v, "queued"),
        uptime_s: u(v, "uptime_s"),
        ctx_max: u(ctx, "max"),
        ctx_in_use: u(ctx, "in_use"),
        mem_avail_gib: mem.get("available_gib").and_then(Value::as_f64),
        mem_total_gib: mem.get("total_gib").and_then(Value::as_f64),
        totals: Totals {
            requests: u(tot, "requests"),
            prompt_tokens: u(tot, "prompt_tokens"),
            output_tokens: u(tot, "output_tokens"),
            drafts_offered: u(tot, "drafts_offered"),
            drafts_accepted: u(tot, "drafts_accepted"),
        },
        recent,
        host: s(srv, "host"),
        port: u(srv, "port") as u16,
        api_key: srv.get("api_key").and_then(Value::as_bool).unwrap_or(false),
        api_key_from_env: srv
            .get("api_key_from_env")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        control_enabled: ctl.get("enabled").and_then(Value::as_bool).unwrap_or(false),
        selection: ctl.get("selection").and_then(parse_selection),
        last_error: ctl.get("last_error").and_then(|e| {
            let msg = e.get("error")?.as_str()?;
            let model = e
                .get("selection")
                .and_then(|s| s.get("model"))
                .and_then(Value::as_str)
                .and_then(talus_presets::model)
                .map(|m| m.name)
                .unwrap_or("the new settings");
            Some(format!("{model} did not start: {msg}"))
        }),
    }
}

/// Averages over the newest `n` requests.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Summary {
    pub n: usize,
    /// Token-weighted answer speed (total output tokens / total decode time).
    pub decode_tps: Option<f64>,
    pub prompt_tps: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub acceptance: Option<f64>,
}

pub fn summarize(recent: &[Request], n: usize) -> Summary {
    let take: Vec<&Request> = recent.iter().take(n).collect();
    let (mut out, mut dms, mut fresh, mut pms, mut dr, mut acc) =
        (0u64, 0.0, 0u64, 0.0, 0u64, 0u64);
    let mut ttft = Vec::new();
    for r in &take {
        if r.decode_ms > 0.0 && r.output_tokens > 1 {
            out += r.output_tokens;
            dms += r.decode_ms;
        }
        let fr = r.prompt_tokens.saturating_sub(r.reused);
        if r.prompt_ms > 0.0 && fr > 0 {
            fresh += fr;
            pms += r.prompt_ms;
        }
        dr += r.drafts;
        acc += r.accepted;
        if r.ttft_ms > 0.0 {
            ttft.push(r.ttft_ms);
        }
    }
    Summary {
        n: take.len(),
        decode_tps: (dms > 0.0).then(|| out as f64 * 1000.0 / dms),
        prompt_tps: (pms > 0.0).then(|| fresh as f64 * 1000.0 / pms),
        ttft_ms: (!ttft.is_empty()).then(|| ttft.iter().sum::<f64>() / ttft.len() as f64),
        acceptance: (dr > 0).then(|| acc as f64 / dr as f64),
    }
}

pub fn fmt_uptime(s: u64) -> String {
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m {}s", s / 60, s % 60),
        3600..=86399 => format!("{}h {}m", s / 3600, (s % 3600) / 60),
        _ => format!("{}d {}h", s / 86400, (s % 86400) / 3600),
    }
}

/// 65536 -> "64K", 196608 -> "192K", 1234 -> "1,234".
pub fn fmt_tokens(n: u64) -> String {
    if n >= 1024 && n.is_multiple_of(1024) {
        return format!("{}K", n / 1024);
    }
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_and_summarize() {
        let v = json!({
            "model": "qwen3.8-27b", "preset": "qwen3.8-27b", "state": "idle", "loaded": true,
            "running": 0, "queued": 0, "uptime_s": 4000,
            "context": {"max": 65536, "in_use": 2000},
            "memory": {"available_gib": 66.5, "total_gib": 125.0},
            "totals": {"requests": 2, "prompt_tokens": 3000, "output_tokens": 300, "drafts_offered": 100, "drafts_accepted": 70},
            "recent": [
                {"prompt_tokens": 2000, "reused": 1000, "output_tokens": 200, "prompt_ms": 2500.0, "decode_ms": 4000.0, "ttft_ms": 2600.0, "drafts": 60, "accepted": 40, "finish": "stop"},
                {"prompt_tokens": 1000, "reused": 0, "output_tokens": 100, "prompt_ms": 2500.0, "decode_ms": 2000.0, "ttft_ms": 2550.0, "drafts": 40, "accepted": 30, "finish": "length"}
            ],
            "server": {"host": "127.0.0.1", "port": 8000, "api_key": false},
            "control": {"enabled": true, "selection": {"model": "qwen3.8-27b", "level": "balanced", "choices": {"context": "32768"}, "bind": "local"}}
        });
        let s = parse_status(&v);
        assert_eq!(s.ctx_max, 65536);
        assert_eq!(s.recent.len(), 2);
        assert_eq!(s.recent[0].decode_tps(), Some(50.0));
        assert_eq!(s.recent[0].prompt_tps(), Some(400.0));
        assert_eq!(s.recent[1].acceptance(), Some(0.75));
        let sel = s.selection.as_ref().unwrap();
        assert_eq!(
            sel.choices.get("context").map(String::as_str),
            Some("32768")
        );
        let sum = summarize(&s.recent, 10);
        assert_eq!(sum.n, 2);
        assert_eq!(sum.decode_tps, Some(50.0));
        assert_eq!(sum.prompt_tps, Some(400.0));
        assert_eq!(sum.acceptance, Some(0.7));
        assert_eq!(sum.ttft_ms, Some(2575.0));
        // missing fields don't panic
        let empty = parse_status(&json!({}));
        assert_eq!(summarize(&empty.recent, 10).decode_tps, None);
    }

    #[test]
    fn formatting() {
        assert_eq!(fmt_tokens(65536), "64K");
        assert_eq!(fmt_tokens(196608), "192K");
        assert_eq!(fmt_tokens(1234567), "1,234,567");
        assert_eq!(fmt_tokens(12), "12");
        assert_eq!(fmt_uptime(4000), "1h 6m");
        assert_eq!(fmt_uptime(59), "59s");
    }
}
