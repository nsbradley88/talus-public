//! Prometheus exposition format (`GET /metrics` with `Accept: text/plain` or `?format=prometheus`,
//! plus `GET /metrics/prometheus`) matching `serve/prometheus.py` and `serve/server.py`.

use std::fmt::Write as _;

pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

pub const BUCKETS: &[(f64, &str)] = &[
    (0.001, "0.001"),
    (0.005, "0.005"),
    (0.01, "0.01"),
    (0.02, "0.02"),
    (0.04, "0.04"),
    (0.06, "0.06"),
    (0.08, "0.08"),
    (0.1, "0.1"),
    (0.25, "0.25"),
    (0.5, "0.5"),
    (0.75, "0.75"),
    (1.0, "1.0"),
    (2.5, "2.5"),
    (5.0, "5.0"),
    (7.5, "7.5"),
    (10.0, "10.0"),
    (20.0, "20.0"),
    (40.0, "40.0"),
    (80.0, "80.0"),
    (160.0, "160.0"),
    (640.0, "640.0"),
    (2560.0, "2560.0"),
];

#[derive(Clone, Debug)]
struct Hist {
    bins: Vec<u64>,
    count: u64,
    sum: f64,
}

impl Default for Hist {
    fn default() -> Self {
        Self {
            bins: vec![0; BUCKETS.len()],
            count: 0,
            sum: 0.0,
        }
    }
}

impl Hist {
    fn add(&mut self, value: f64, weight: u64) {
        if !value.is_finite() || value < 0.0 || weight == 0 {
            return;
        }
        self.count += weight;
        self.sum += value * (weight as f64);
        for (i, &(edge, _)) in BUCKETS.iter().enumerate() {
            if value <= edge {
                self.bins[i] += weight;
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LatencyHistograms {
    ttft: Hist,
    itl: Hist,
    e2e: Hist,
}

impl LatencyHistograms {
    pub fn observe(
        &mut self,
        ttft_s: Option<f64>,
        output_tokens: usize,
        decode_s: f64,
        e2e_s: f64,
        _finish_reason: &str,
    ) {
        if let Some(ttft) = ttft_s {
            self.ttft.add(ttft, 1);
        }
        if output_tokens > 1 && decode_s > 0.0 {
            let gaps = output_tokens - 1;
            self.itl.add(decode_s / (gaps as f64), gaps as u64);
        }
        self.e2e.add(e2e_s, 1);
    }
}

pub struct PrometheusSnapshot {
    pub model_name: String,
    pub loaded: bool,
    pub max_context: usize,
    pub running: usize,
    pub waiting: usize,
    pub active_prompt_tokens: usize,
    pub active_generated_tokens: usize,
    pub requests_total: u64,
    pub prompt_tokens_total: u64,
    pub reused_tokens_total: u64,
    pub output_tokens_total: u64,
    pub prompt_ms_total: f64,
    pub decode_ms_total: f64,
    pub drafts_offered_total: u64,
    pub drafts_accepted_total: u64,
    pub reasoning_recoveries_total: u64,
    pub tool_calls_from_reasoning_total: u64,
    pub histograms: LatencyHistograms,
}

pub fn wants_prometheus(accept: Option<&str>, query: Option<&str>) -> bool {
    if let Some(q) = query {
        if q.contains("format=prometheus") {
            return true;
        }
    }
    if let Some(acc) = accept {
        let lower = acc.to_ascii_lowercase();
        if lower.contains("text/plain") || lower.contains("openmetrics") {
            return true;
        }
    }
    false
}

fn prom_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

fn write_hist(out: &mut String, name: &str, help: &str, m: &str, h: &Hist) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} histogram");
    for ((_, le), &count) in BUCKETS.iter().zip(h.bins.iter()) {
        let _ = writeln!(
            out,
            "{name}_bucket{{model_name=\"{m}\",le=\"{le}\"}} {count}"
        );
    }
    let _ = writeln!(
        out,
        "{name}_bucket{{model_name=\"{m}\",le=\"+Inf\"}} {}",
        h.count
    );
    let _ = writeln!(out, "{name}_sum{{model_name=\"{m}\"}} {:.6}", h.sum);
    let _ = writeln!(out, "{name}_count{{model_name=\"{m}\"}} {}", h.count);
}

pub fn render_prometheus(snap: &PrometheusSnapshot) -> String {
    let m = prom_label(&snap.model_name);
    let used = snap.active_prompt_tokens + snap.active_generated_tokens;
    let ctx_ratio = if snap.max_context > 0 && snap.running > 0 {
        ((used as f64) / (snap.max_context as f64)).min(1.0)
    } else {
        0.0
    };

    let mut out = String::with_capacity(4096);
    fn metric(out: &mut String, name: &str, kind: &str, help: &str, lines: &[String]) {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} {kind}");
        for l in lines {
            let _ = writeln!(out, "{l}");
        }
    }

    metric(
        &mut out,
        "vllm:num_requests_running",
        "gauge",
        "Requests reading their prompt or generating.",
        &[format!(
            "vllm:num_requests_running{{model_name=\"{m}\"}} {}",
            snap.running
        )],
    );
    metric(
        &mut out,
        "vllm:num_requests_waiting",
        "gauge",
        "Requests waiting for their turn.",
        &[format!(
            "vllm:num_requests_waiting{{model_name=\"{m}\"}} {}",
            snap.waiting
        )],
    );
    metric(
        &mut out,
        "vllm:kv_cache_usage_perc",
        "gauge",
        "The running request's share of the context (1 = full).",
        &[format!(
            "vllm:kv_cache_usage_perc{{model_name=\"{m}\"}} {ctx_ratio:.4}"
        )],
    );
    metric(
        &mut out,
        "vllm:prompt_tokens_total",
        "counter",
        "Prompt tokens of the finished requests.",
        &[format!(
            "vllm:prompt_tokens_total{{model_name=\"{m}\"}} {}",
            snap.prompt_tokens_total
        )],
    );
    metric(
        &mut out,
        "vllm:generation_tokens_total",
        "counter",
        "Generated tokens.",
        &[format!(
            "vllm:generation_tokens_total{{model_name=\"{m}\"}} {}",
            snap.output_tokens_total
        )],
    );
    metric(
        &mut out,
        "vllm:request_success_total",
        "counter",
        "Finished requests.",
        &[format!(
            "vllm:request_success_total{{model_name=\"{m}\"}} {}",
            snap.requests_total
        )],
    );
    metric(
        &mut out,
        "vllm:prefix_cache_queries_total",
        "counter",
        "Prompt tokens looked up in the prompt cache.",
        &[format!(
            "vllm:prefix_cache_queries_total{{model_name=\"{m}\"}} {}",
            snap.prompt_tokens_total
        )],
    );
    metric(
        &mut out,
        "vllm:prefix_cache_hits_total",
        "counter",
        "Prompt tokens the prompt cache already held.",
        &[format!(
            "vllm:prefix_cache_hits_total{{model_name=\"{m}\"}} {}",
            snap.reused_tokens_total
        )],
    );
    metric(
        &mut out,
        "vllm:spec_decode_num_draft_tokens_total",
        "counter",
        "MTP draft tokens offered.",
        &[format!(
            "vllm:spec_decode_num_draft_tokens_total{{model_name=\"{m}\"}} {}",
            snap.drafts_offered_total
        )],
    );
    metric(
        &mut out,
        "vllm:spec_decode_num_accepted_tokens_total",
        "counter",
        "MTP draft tokens accepted.",
        &[format!(
            "vllm:spec_decode_num_accepted_tokens_total{{model_name=\"{m}\"}} {}",
            snap.drafts_accepted_total
        )],
    );
    metric(
        &mut out,
        "vllm:num_preemptions_total",
        "counter",
        "Preempted requests (Strata does not preempt).",
        &[format!(
            "vllm:num_preemptions_total{{model_name=\"{m}\"}} 0"
        )],
    );

    write_hist(
        &mut out,
        "vllm:time_to_first_token_seconds",
        "Time to the first token.",
        &m,
        &snap.histograms.ttft,
    );
    write_hist(
        &mut out,
        "vllm:inter_token_latency_seconds",
        "Time between two tokens.",
        &m,
        &snap.histograms.itl,
    );
    write_hist(
        &mut out,
        "vllm:e2e_request_latency_seconds",
        "A request from start to end.",
        &m,
        &snap.histograms.e2e,
    );

    let live_state = if !snap.loaded {
        "unloaded"
    } else if snap.running > 0 {
        "generating"
    } else {
        "idle"
    };
    metric(
        &mut out,
        "strata:live_state",
        "gauge",
        "1 for what the engine is doing (live.state).",
        &["unloaded", "idle", "reading", "generating"]
            .iter()
            .map(|st| {
                format!(
                    "strata:live_state{{model_name=\"{m}\",state=\"{st}\"}} {}",
                    usize::from(*st == live_state)
                )
            })
            .collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "strata:engine_max_context",
        "gauge",
        "The engine's context (engine.max_context).",
        &[format!(
            "strata:engine_max_context{{model_name=\"{m}\"}} {}",
            snap.max_context
        )],
    );
    metric(
        &mut out,
        "strata:totals_prompt_seconds_total",
        "counter",
        "Time spent reading prompts (totals.prompt_ms).",
        &[format!(
            "strata:totals_prompt_seconds_total{{model_name=\"{m}\"}} {:.3}",
            snap.prompt_ms_total / 1000.0
        )],
    );
    metric(
        &mut out,
        "strata:totals_decode_seconds_total",
        "counter",
        "Time spent generating (totals.decode_ms).",
        &[format!(
            "strata:totals_decode_seconds_total{{model_name=\"{m}\"}} {:.3}",
            snap.decode_ms_total / 1000.0
        )],
    );

    out
}

pub fn render_llamacpp_prometheus(snap: &PrometheusSnapshot) -> String {
    let mut out = String::with_capacity(1024);
    let prompt_n = snap
        .prompt_tokens_total
        .saturating_sub(snap.reused_tokens_total);
    let p_sec = snap.prompt_ms_total / 1000.0;
    let d_sec = snap.decode_ms_total / 1000.0;
    let p_tps = if p_sec > 0.0 {
        (prompt_n as f64) / p_sec
    } else {
        0.0
    };
    let d_tps = if d_sec > 0.0 {
        (snap.output_tokens_total as f64) / d_sec
    } else {
        0.0
    };
    let rows: &[(&str, &str, &str, String)] = &[
        (
            "llamacpp:prompt_tokens_total",
            "counter",
            "Prompt tokens read",
            prompt_n.to_string(),
        ),
        (
            "llamacpp:prompt_tokens_cached_total",
            "counter",
            "Prompt tokens the conversation cache already held",
            snap.reused_tokens_total.to_string(),
        ),
        (
            "llamacpp:prompt_seconds_total",
            "counter",
            "Prompt read time",
            format!("{p_sec:.6}"),
        ),
        (
            "llamacpp:tokens_predicted_total",
            "counter",
            "Tokens generated",
            snap.output_tokens_total.to_string(),
        ),
        (
            "llamacpp:tokens_predicted_seconds_total",
            "counter",
            "Generation time",
            format!("{d_sec:.6}"),
        ),
        (
            "llamacpp:prompt_tokens_seconds",
            "gauge",
            "Average prompt read throughput in tokens/s",
            format!("{p_tps:.3}"),
        ),
        (
            "llamacpp:predicted_tokens_seconds",
            "gauge",
            "Average generation throughput in tokens/s",
            format!("{d_tps:.3}"),
        ),
        (
            "llamacpp:requests_processing",
            "gauge",
            "Requests running",
            snap.running.to_string(),
        ),
        (
            "llamacpp:requests_deferred",
            "gauge",
            "Requests waiting their turn",
            snap.waiting.to_string(),
        ),
        (
            "strata:requests_total",
            "counter",
            "Requests finished",
            snap.requests_total.to_string(),
        ),
        (
            "strata:drafts_offered_total",
            "counter",
            "Speculative draft tokens offered",
            snap.drafts_offered_total.to_string(),
        ),
        (
            "strata:drafts_accepted_total",
            "counter",
            "Speculative draft tokens accepted",
            snap.drafts_accepted_total.to_string(),
        ),
        (
            "strata:model_loaded",
            "gauge",
            "1 while the model is loaded",
            usize::from(snap.loaded).to_string(),
        ),
    ];
    for (name, kind, help, val) in rows {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} {kind}");
        let _ = writeln!(out, "{name} {val}");
    }
    out
}
