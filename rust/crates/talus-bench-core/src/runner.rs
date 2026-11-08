//! Executes an experiment: resolve tasks and cases, fingerprint each endpoint, put Talus on the requested
//! selection (Talus-vs-Talus), run every case sequentially, then pair, gate and report. Progress goes to a
//! callback (the CLI prints it, the TUI's Bench tab renders it); the runner itself never draws.

use crate::artifacts::Artifacts;
use crate::case::{Case, CaseResult, CaseStatus, RawOutput, Score, Timings, Usage};
use crate::compare::{self, PairRow, PairedTask};
use crate::config::{Config, Endpoint, EndpointKind, TalusSelection, Thinking};
use crate::datasets;
use crate::gates::{self, GateResult, GateStatus, Verdict};
use crate::http::{error_message, Client, HttpError};
use crate::report;
use crate::tasks::{self, BatchRow, ExecEnv, ExecOutcome, Task, TaskContext};
use crate::util::{hash_json, rfc3339, sha256_hex, unix_now};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Progress events for the CLI / TUI.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Phase(String),
    ModelStart {
        model: String,
        total: usize,
    },
    Case {
        model: String,
        done: usize,
        total: usize,
        case_id: String,
        status: CaseStatus,
        pass: bool,
    },
    ModelDone {
        model: String,
        correct: usize,
        total: usize,
    },
    Log(String),
}

#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// Override the output directory.
    pub output_dir: Option<PathBuf>,
    /// Poll interval while waiting for an engine restart (default 1 s; tests use less).
    pub poll_ms: Option<u64>,
    /// Key for endpoints without `api_key_env` (the TUI passes the key it already holds; never written out).
    pub api_key: Option<String>,
    /// Reuse the scored rows already in the output directory (same prompt, settings and scorer) and run only
    /// the missing or transiently failed cases.
    pub resume: bool,
    /// Stop cleanly before starting a case or an engine restart after this instant (GPU lock segments); the
    /// run returns an error starting with `INCOMPLETE` and `--resume` continues it.
    pub deadline: Option<Instant>,
}

/// Prefix of the error a run returns when it stopped at its segment deadline.
pub const INCOMPLETE: &str = "INCOMPLETE";

#[derive(Clone, Debug)]
pub struct RunOutcome {
    pub output_dir: PathBuf,
    pub verdicts: Vec<Verdict>,
    pub report_md: String,
    pub report_json: Value,
}

impl RunOutcome {
    /// CLI exit code: 0 all PASS, 1 any FAIL, 3 otherwise (WARN / NOT_RUN). 2 is reserved for config errors.
    pub fn exit_code(&self) -> i32 {
        let all = self.verdicts.iter().map(|v| v.overall);
        let v: Vec<GateStatus> = all.collect();
        if v.contains(&GateStatus::Fail) {
            1
        } else if !v.is_empty() && v.iter().all(|s| *s == GateStatus::Pass) {
            0
        } else {
            3
        }
    }
}

/// The talus-presets selection an endpoint asks for: the level's accuracy options, then explicit choices.
pub fn talus_choices(t: &TalusSelection) -> talus_presets::Selection {
    let Some(spec) = talus_presets::model(&t.model) else {
        return t.choices.clone();
    };
    let level = t
        .level
        .as_deref()
        .and_then(talus_presets::Level::parse)
        .unwrap_or(talus_presets::Level::Balanced);
    let mut sel = spec.apply_level(&talus_presets::Selection::new(), level);
    sel.extend(t.choices.clone());
    spec.normalize(&sel)
}

/// The request-shaping settings every case of a model is generated with; hashed into `settings_hash`.
pub fn settings_value(cfg: &Config) -> Value {
    let i = &cfg.inference;
    let mut v = json!({
        "temperature": i.temperature, "top_p": i.top_p, "max_output_tokens": i.max_output_tokens,
        "seed": i.seed, "thinking": format!("{:?}", i.thinking).to_lowercase(),
    });
    if let Some(seeds) = &i.seeds {
        v["seeds"] = json!(seeds);
    }
    v
}

struct Resolved {
    task: Box<dyn Task>,
    /// Static cases (None for tasks built per endpoint).
    cases: Option<Vec<Case>>,
    limit: Option<usize>,
}

fn api_key(ep: &Endpoint) -> Option<String> {
    ep.api_key_env
        .as_deref()
        .and_then(|v| std::env::var(v).ok())
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
}

fn client_for(
    ep: &Endpoint,
    timeout_s: u64,
    fallback_key: Option<&String>,
) -> Result<Client, String> {
    let key = api_key(ep).or_else(|| fallback_key.cloned());
    Client::new(ep.endpoint.as_deref().unwrap_or(""), key, timeout_s)
}

/// What the server says about itself; fields it does not expose are recorded as unavailable. On talus-server the
/// `build` fingerprint of `/v1/talus/status` (engine/kernels/weights sha256, commit, run-config sha256, numerics
/// environment) is recorded, and hashing of the weight files is requested (`?hash=weights`; it runs in the
/// background and is cached by the server, so [`refresh_build`] after the model's cases picks the hashes up).
pub fn fingerprint(c: &Client) -> Value {
    let get = |p: &str| match c.get(p) {
        Ok(r) if r.status == 200 => Some(r.body),
        _ => None,
    };
    let models = get("/v1/models");
    let props = get("/props");
    let status = get("/v1/talus/status?hash=weights").or_else(|| get("/v1/talus/status"));
    let model_ids: Vec<Value> = models
        .as_ref()
        .and_then(|m| m["data"].as_array().cloned())
        .unwrap_or_default()
        .iter()
        .map(|m| m["id"].clone())
        .collect();
    let mut fp = json!({
        "url": c.url(),
        "models": model_ids,
        "props": props.as_ref().map(|p| json!({
            "build_info": p["build_info"], "n_ctx": p["default_generation_settings"]["n_ctx"],
            "model_path": p["model_path"], "params": p["default_generation_settings"]["params"],
        })).unwrap_or(Value::Null),
        "talus_status": status.as_ref().map(|s| json!({
            "model": s["model"], "preset": s["preset"], "selection": s["control"]["selection"],
            "context_max": s["context"]["max"], "mem_available_gib": s["memory"]["available_gib"],
            "build": s["build"], "engine_rss_bytes": s["engine"]["rss_bytes"],
            "server_rss_bytes": s["server"]["rss_bytes"], "energy": s["energy"],
        })).unwrap_or(Value::Null),
    });
    fp["unavailable"] = unavailable_of(&fp["talus_status"]["build"]);
    fp
}

/// The fingerprint fields a `build` object (null: an older server) does not give.
pub fn unavailable_of(build: &Value) -> Value {
    let mut out = Vec::new();
    if build["engine_exe"]["commit"].is_null() {
        out.push("engine build commit");
    }
    if build["engine_exe"]["sha256"].is_null() {
        out.push("engine sha256");
    }
    let weights = build["weights"].as_array();
    if weights.map(|w| w.is_empty() || w.iter().any(|x| x["sha256"].is_null())) != Some(false) {
        out.push("weights sha256");
    }
    if build["run_config_sha256"].is_null() {
        out.push("run config sha256");
    }
    if build["kernels_lib"]["sha256"].is_null() {
        out.push("kernels library sha256");
    }
    if build["gfx_target"].is_null() {
        out.push("gfx target");
    }
    json!(out)
}

/// Re-read `build` (and RSS / energy) after a model's cases: weight hashes requested at the start are often done.
pub fn refresh_build(c: &Client, server_fp: &mut Value) {
    let Ok(r) = c.get("/v1/talus/status") else {
        return;
    };
    if r.status != 200 || r.body["build"].is_null() || !server_fp["talus_status"].is_object() {
        return;
    }
    let st = &mut server_fp["talus_status"];
    st["build"] = r.body["build"].clone();
    st["engine_rss_bytes_end"] = r.body["engine"]["rss_bytes"].clone();
    st["energy_end"] = r.body["energy"].clone();
    server_fp["unavailable"] = unavailable_of(&server_fp["talus_status"]["build"]);
}

fn talus_status(c: &Client) -> Result<Value, String> {
    match c.get("/v1/talus/status") {
        Ok(r) if r.status == 200 => Ok(r.body),
        Ok(r) => Err(format!(
            "/v1/talus/status answered {}: {}",
            r.status,
            error_message(&r.body)
        )),
        Err(e) => Err(e.to_string()),
    }
}

fn selection_of(status: &Value) -> (Option<String>, talus_presets::Selection) {
    let preset = status["preset"].as_str().map(str::to_string);
    let sel = &status["control"]["selection"];
    let choices = if sel["model"].as_str() == preset.as_deref() {
        sel["choices"]
            .as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                    .collect()
            })
            .unwrap_or_default()
    } else {
        talus_presets::Selection::new()
    };
    (preset, choices)
}

fn on_selection(status: &Value, t: &TalusSelection, want: &talus_presets::Selection) -> bool {
    let (preset, have) = selection_of(status);
    let spec = talus_presets::model(&t.model);
    preset.as_deref() == Some(t.model.as_str())
        && status["state"].as_str() != Some("restarting")
        && status["loaded"].as_bool() != Some(false)
        && spec.map(|s| s.normalize(&have) == *want).unwrap_or(false)
}

/// Make the server serve `t` (apply when asked), or check that it already does.
pub fn ensure_selection(
    c: &Client,
    ep_id: &str,
    t: &TalusSelection,
    poll: Duration,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(Event),
) -> Result<(), String> {
    let want = talus_choices(t);
    let st = talus_status(c)?;
    if on_selection(&st, t, &want) {
        progress(Event::Log(format!(
            "{ep_id}: server already on {} {:?}",
            t.model, want
        )));
        return Ok(());
    }
    if !t.apply {
        let (preset, have) = selection_of(&st);
        return Err(format!(
            "{ep_id}: server serves {preset:?} {have:?}, config wants {} {want:?}; set talus.apply: true or switch it first",
            t.model
        ));
    }
    let choices: serde_json::Map<String, Value> = want
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    progress(Event::Phase(format!(
        "{ep_id}: applying {} {:?} (engine restart)",
        t.model, want
    )));
    let r = c
        .post(
            "/v1/talus/apply",
            &json!({"model": t.model, "choices": choices}),
        )
        .map_err(|e| format!("{ep_id}: apply: {e}"))?;
    if r.status != 202 && r.status != 200 {
        return Err(format!(
            "{ep_id}: apply answered {}: {}",
            r.status,
            error_message(&r.body)
        ));
    }
    let deadline = Instant::now() + Duration::from_secs(t.apply_timeout_s);
    let mut seen_down = false;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        std::thread::sleep(poll);
        match talus_status(c) {
            Ok(s) => {
                if s["state"].as_str() == Some("restarting") {
                    seen_down = true;
                } else if on_selection(&s, t, &want) {
                    progress(Event::Log(format!("{ep_id}: engine back on {}", t.model)));
                    return Ok(());
                } else if seen_down {
                    if let Some(e) = s["last_error"]
                        .as_str()
                        .or(s["control"]["last_error"].as_str())
                    {
                        return Err(format!("{ep_id}: apply failed: {e}"));
                    }
                }
            }
            Err(_) => seen_down = true,
        }
        if Instant::now() > deadline {
            return Err(format!(
                "{ep_id}: engine not back on the requested selection within {} s",
                t.apply_timeout_s
            ));
        }
    }
}

/// Build the chat request for a case.
/// Precedence: the case's own request fields (task protocol, e.g. the needle gate's sampling) over the trial
/// seed over the inference settings. The case's `max_tokens` is the task's budget and replaces the setting.
pub fn request_body(cfg: &Config, ep: &Endpoint, case: &Case, trial_seed: Option<u64>) -> Value {
    let i = &cfg.inference;
    let mut b = json!({
        "model": ep.model.clone().unwrap_or_else(|| "default".into()),
        "messages": case.messages,
        "max_tokens": case.max_tokens.unwrap_or(i.max_output_tokens),
        "temperature": i.temperature,
        "top_p": i.top_p,
        "stream": false,
    });
    if let Some(s) = trial_seed.or(i.seed) {
        b["seed"] = json!(s);
    }
    match i.thinking {
        Thinking::Off => b["chat_template_kwargs"] = json!({"enable_thinking": false}),
        Thinking::On => b["chat_template_kwargs"] = json!({"enable_thinking": true}),
        Thinking::ServerDefault => {}
    }
    if !case.tools.is_null() {
        b["tools"] = case.tools.clone();
    }
    if let Some(o) = case.request.as_object() {
        for (k, v) in o {
            b[k.as_str()] = v.clone();
        }
    }
    b
}

/// Turn an HTTP outcome into (status, raw output, usage, timings, error).
pub fn classify(
    res: Result<crate::http::Reply, HttpError>,
) -> (
    CaseStatus,
    Option<RawOutput>,
    Usage,
    Timings,
    Option<String>,
) {
    let mut timings = Timings::default();
    let r = match res {
        Ok(r) => r,
        Err(HttpError::Timeout(m)) => {
            return (
                CaseStatus::Timeout,
                None,
                Usage::default(),
                timings,
                Some(m),
            )
        }
        Err(HttpError::Connect(m)) => {
            return (
                CaseStatus::ConnectError,
                None,
                Usage::default(),
                timings,
                Some(m),
            )
        }
        Err(HttpError::Protocol(m)) => {
            return (
                CaseStatus::ProtocolError,
                None,
                Usage::default(),
                timings,
                Some(m),
            )
        }
    };
    timings.client_ms = r.elapsed_ms;
    if r.status >= 500 {
        return (
            CaseStatus::ServerError,
            None,
            Usage::default(),
            timings,
            Some(format!("HTTP {}: {}", r.status, error_message(&r.body))),
        );
    }
    if r.status >= 400 {
        return (
            CaseStatus::HttpError,
            None,
            Usage::default(),
            timings,
            Some(format!("HTTP {}: {}", r.status, error_message(&r.body))),
        );
    }
    let b = &r.body;
    let msg = &b["choices"][0]["message"];
    if !msg.is_object() {
        return (
            CaseStatus::ProtocolError,
            None,
            Usage::default(),
            timings,
            Some("reply has no choices[0].message".into()),
        );
    }
    let out = RawOutput {
        content: msg["content"].as_str().unwrap_or_default().to_string(),
        reasoning_content: msg["reasoning_content"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        tool_calls: msg.get("tool_calls").cloned().unwrap_or(Value::Null),
        finish_reason: b["choices"][0]["finish_reason"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
    };
    let u = &b["usage"];
    let usage = Usage {
        prompt_tokens: u["prompt_tokens"].as_u64(),
        completion_tokens: u["completion_tokens"].as_u64(),
        cached_tokens: u["prompt_tokens_details"]["cached_tokens"].as_u64(),
        reasoning_tokens: u["completion_tokens_details"]["reasoning_tokens"].as_u64(),
    };
    let t = &b["timings"];
    timings.server_prompt_ms = t["prompt_ms"].as_f64();
    timings.server_decode_ms = t["predicted_ms"].as_f64();
    timings.draft_n = t["draft_n"].as_u64();
    timings.draft_accepted = t["draft_n_accepted"].as_u64();
    timings.reasoning_n = t["reasoning_n"].as_u64();
    timings.answer_n = t["answer_n"].as_u64();
    timings.reasoning_ms = t["reasoning_ms"].as_f64();
    timings.answer_ms = t["answer_ms"].as_f64();
    let status = if out.finish_reason == "length" {
        if out.content.trim().is_empty() && !out.reasoning_content.is_empty() {
            CaseStatus::TruncatedByReasoning
        } else {
            CaseStatus::Truncated
        }
    } else {
        CaseStatus::Ok
    };
    (status, Some(out), usage, timings, None)
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    cfg: &Config,
    ep: &Endpoint,
    client: &Client,
    task: &dyn Task,
    case: &Case,
    trial_seed: Option<u64>,
    settings_hash: &str,
) -> CaseResult {
    let started = rfc3339(unix_now());
    let body = request_body(cfg, ep, case, trial_seed);
    let mut attempts = 0;
    let (status, raw, usage, timings, error) = loop {
        attempts += 1;
        let r = classify(client.post("/v1/chat/completions", &body));
        let transport = matches!(r.0, CaseStatus::ConnectError | CaseStatus::Timeout);
        if !transport || attempts > cfg.inference.retries {
            break r;
        }
        std::thread::sleep(Duration::from_millis(500));
    };
    let score = match (&raw, status.scored()) {
        (Some(o), true) => task.score(case, o),
        _ => Score::binary(false, Value::Null, json!({"not_scored": status.as_str()})),
    };
    let redact = cfg.report.redact_prompts;
    CaseResult {
        experiment_id: cfg.experiment.id.clone(),
        model_id: ep.id.clone(),
        task: case.task.clone(),
        case_id: case.case_id.clone(),
        seed: trial_seed,
        dataset: case.dataset.clone(),
        dataset_revision: case.dataset_revision.clone(),
        prompt_hash: case.prompt_hash(),
        scorer_version: task.scorer_version().into(),
        settings_hash: settings_hash.into(),
        status,
        prompt: if redact {
            Value::Null
        } else {
            Value::Array(case.messages.clone())
        },
        expected: if redact {
            Value::Null
        } else {
            case.expected.clone()
        },
        raw_output: if cfg.report.store_raw_outputs {
            raw
        } else {
            None
        },
        score,
        error,
        usage,
        timings,
        attempts,
        critical: case.critical,
        started_at: started,
    }
}

/// Frozen reference: `<dir>/frozen.json` + `<dir>/cases.jsonl`, verified by content address.
pub fn load_frozen(dir: &Path, settings_hash: &str) -> Result<(Vec<CaseResult>, Value), String> {
    let meta: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("frozen.json"))
            .map_err(|e| format!("{}/frozen.json: {e}", dir.display()))?,
    )
    .map_err(|e| format!("{}/frozen.json: {e}", dir.display()))?;
    let bytes = std::fs::read(dir.join("cases.jsonl"))
        .map_err(|e| format!("{}/cases.jsonl: {e}", dir.display()))?;
    let got = sha256_hex(&bytes);
    if meta["content_sha256"].as_str() != Some(got.as_str()) {
        return Err(format!(
            "frozen reference {}: cases.jsonl sha256 {got} != recorded {} (modified after freezing)",
            dir.display(),
            meta["content_sha256"]
        ));
    }
    if meta["settings_hash"].as_str() != Some(settings_hash) {
        return Err(format!(
            "frozen reference {} was generated with different inference settings ({}); current settings hash {settings_hash}",
            dir.display(),
            meta["settings"]
        ));
    }
    let rows = crate::case::read_jsonl(&dir.join("cases.jsonl"))?;
    Ok((rows, meta))
}

/// Freeze one model's cases from a finished run into `<root>/<model>-<hash12>/` and return that directory.
pub fn freeze(run_dir: &Path, model: Option<&str>, root: &Path) -> Result<PathBuf, String> {
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(run_dir.join("manifest.json"))
            .map_err(|e| format!("{}/manifest.json: {e}", run_dir.display()))?,
    )
    .map_err(|e| e.to_string())?;
    let ref_id = manifest["models"][0]["id"]
        .as_str()
        .unwrap_or("reference")
        .to_string();
    let model = model.map(str::to_string).unwrap_or(ref_id.clone());
    let src = if model == ref_id {
        run_dir.join("baseline/cases.jsonl")
    } else {
        run_dir.join(format!("candidates/{model}/cases.jsonl"))
    };
    let bytes = std::fs::read(&src).map_err(|e| format!("{}: {e}", src.display()))?;
    let rows = crate::case::read_jsonl(&src)?;
    if rows.is_empty() {
        return Err(format!("{}: no cases", src.display()));
    }
    let bad = rows.iter().filter(|r| !r.status.scored()).count();
    let hash = sha256_hex(&bytes);
    let dir = root.join(format!("{model}-{}", &hash[..12]));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    std::fs::write(dir.join("cases.jsonl"), &bytes).map_err(|e| e.to_string())?;
    let fp = manifest["models"]
        .as_array()
        .and_then(|m| m.iter().find(|x| x["id"] == model.as_str()))
        .cloned()
        .unwrap_or(Value::Null);
    let meta = json!({
        "frozen_at": rfc3339(unix_now()),
        "model_id": model,
        "content_sha256": hash,
        "source_run": run_dir.display().to_string(),
        "experiment_id": manifest["experiment"]["id"],
        "settings": manifest["settings"],
        "settings_hash": manifest["settings_hash"],
        "datasets": manifest["datasets"],
        "model": fp,
        "cases": rows.len(),
        "unscored_cases": bad,
    });
    std::fs::write(
        dir.join("frozen.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    Ok(dir)
}

/// A task error's gate status: NOT_SUPPORTED when the backend or this machine cannot run it, else NOT_RUN.
fn err_status(e: &str) -> GateStatus {
    if e.starts_with(tasks::NOT_SUPPORTED) {
        GateStatus::NotSupported
    } else {
        GateStatus::NotRun
    }
}

fn base_row(cfg: &Config, ep: &Endpoint, settings_hash: &str) -> CaseResult {
    CaseResult {
        experiment_id: cfg.experiment.id.clone(),
        model_id: ep.id.clone(),
        task: String::new(),
        case_id: String::new(),
        seed: None,
        dataset: String::new(),
        dataset_revision: String::new(),
        prompt_hash: String::new(),
        scorer_version: String::new(),
        settings_hash: settings_hash.into(),
        status: CaseStatus::Ok,
        prompt: Value::Null,
        expected: Value::Null,
        raw_output: None,
        score: Score::default(),
        error: None,
        usage: Usage::default(),
        timings: Timings::default(),
        attempts: 1,
        critical: false,
        started_at: rfc3339(unix_now()),
    }
}

/// A job case's row (agent sessions run by the adapter itself).
fn exec_row(
    cfg: &Config,
    ep: &Endpoint,
    task: &dyn Task,
    case: &Case,
    seed: Option<u64>,
    settings_hash: &str,
    o: ExecOutcome,
) -> CaseResult {
    let mut r = base_row(cfg, ep, settings_hash);
    r.task = case.task.clone();
    r.case_id = case.case_id.clone();
    r.seed = seed;
    r.dataset = case.dataset.clone();
    r.dataset_revision = case.dataset_revision.clone();
    r.prompt_hash = case.prompt_hash();
    r.scorer_version = task.scorer_version().into();
    r.status = o.status;
    r.expected = case.expected.clone();
    r.raw_output = if cfg.report.store_raw_outputs {
        o.raw
    } else {
        None
    };
    r.score = o.score;
    r.error = o.error;
    r.timings.client_ms = o.wall_ms;
    // job tasks that drive a model themselves (SWE-bench agents) report their summed token usage
    let u = &r.score.details["agent_usage"];
    if u.is_object() {
        r.usage.prompt_tokens = u["prompt_tokens"].as_u64();
        r.usage.completion_tokens = u["completion_tokens"].as_u64();
        r.usage.cached_tokens = u["cached_tokens"].as_u64();
    }
    r.critical = case.critical;
    r
}

/// A row reported by an external framework.
fn batch_row(
    cfg: &Config,
    ep: &Endpoint,
    task: &str,
    b: BatchRow,
    settings_hash: &str,
) -> CaseResult {
    let mut r = base_row(cfg, ep, settings_hash);
    r.task = task.into();
    r.case_id = b.case_id;
    r.seed = b.epoch.filter(|e| *e > 1);
    r.dataset = b.dataset;
    r.dataset_revision = b.dataset_revision;
    r.prompt_hash = b.prompt_hash;
    r.scorer_version = b.scorer_version;
    r.status = b.status;
    r.expected = if cfg.report.redact_prompts {
        Value::Null
    } else {
        b.expected
    };
    r.raw_output = if cfg.report.store_raw_outputs {
        b.raw
    } else {
        None
    };
    r.score = b.score;
    r
}

/// Rows of an earlier (partial) run of this model that a resumed run may keep, keyed by (case_id, seed).
fn prior_rows(dir: &Path) -> BTreeMap<(String, Option<u64>), CaseResult> {
    let mut out = BTreeMap::new();
    for f in ["cases.jsonl", "cases.partial.jsonl"] {
        if let Ok(rows) = crate::case::read_jsonl(&dir.join(f)) {
            for r in rows {
                out.insert(r.key(), r);
            }
        }
    }
    out
}

/// Appends each finished case to `cases.partial.jsonl`, so an interrupted run loses at most one case.
struct PartialWriter {
    file: std::fs::File,
}

impl PartialWriter {
    fn create(dir: &Path, initial: &[CaseResult]) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let p = dir.join("cases.partial.jsonl");
        std::fs::write(&p, crate::case::to_jsonl(initial))
            .map_err(|e| format!("{}: {e}", p.display()))?;
        let file = std::fs::OpenOptions::new()
            .append(true)
            .open(&p)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        Ok(Self { file })
    }

    fn push(&mut self, r: &CaseResult) -> Result<(), String> {
        use std::io::Write;
        let line = serde_json::to_string(r).unwrap_or_default() + "\n";
        self.file
            .write_all(line.as_bytes())
            .and_then(|_| self.file.flush())
            .map_err(|e| format!("cases.partial.jsonl: {e}"))
    }
}

fn past(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
}

/// Run the whole experiment.
pub fn run(
    cfg: &Config,
    opts: &RunOptions,
    cancel: Arc<AtomicBool>,
    progress: &mut dyn FnMut(Event),
) -> Result<RunOutcome, String> {
    let v = cfg.validate();
    if !v.ok() {
        return Err(format!("invalid config:\n  {}", v.errors.join("\n  ")));
    }
    for w in &v.warnings {
        progress(Event::Log(format!("warning: {w}")));
    }
    let started = unix_now();
    let out_dir = opts.output_dir.clone().unwrap_or_else(|| cfg.output_dir());
    let art = Artifacts::create(&out_dir)?;
    art.write_text("config.resolved.yaml", &cfg.to_yaml())?;
    let cache = cfg
        .benchmarks
        .dataset_cache
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(datasets::default_cache_dir);
    let required = cfg.benchmarks.required();
    let settings = settings_value(cfg);
    let settings_hash = hash_json(&settings);
    let poll = Duration::from_millis(opts.poll_ms.unwrap_or(1000));
    let tier = cfg.benchmarks.tier;
    let trial_seeds: Vec<Option<u64>> = match &cfg.inference.seeds {
        Some(s) => s.iter().map(|x| Some(*x)).collect(),
        None => vec![None],
    };
    let null = Value::Null;
    let params_of = |id: &str| cfg.benchmarks.params.get(id).unwrap_or(&null);

    // 1. tasks and static cases
    progress(Event::Phase("loading tasks and datasets".into()));
    let mut resolved: Vec<Resolved> = Vec::new();
    let mut unavailable: Vec<(String, GateStatus, String)> = Vec::new();
    for id in &cfg.benchmarks.tasks {
        match tasks::native(id) {
            Some(task) => {
                let limit = cfg
                    .benchmarks
                    .limits
                    .get(id)
                    .copied()
                    .or_else(|| task.default_limit(tier));
                if task.needs_server() {
                    resolved.push(Resolved {
                        task,
                        cases: None,
                        limit,
                    });
                    continue;
                }
                let ctx = TaskContext {
                    tier,
                    limit,
                    cache: &cache,
                    params: params_of(id),
                    client: None,
                    run_dir: Some(&out_dir),
                    resume: opts.resume,
                };
                match task.cases(&ctx) {
                    Ok(cases) => resolved.push(Resolved {
                        task,
                        cases: Some(cases),
                        limit,
                    }),
                    Err(e) => unavailable.push((id.clone(), err_status(&e), e)),
                }
            }
            None => {
                let a = tasks::availability(id);
                unavailable.push((id.clone(), a.status(), a.reason(id)));
            }
        }
    }
    for (id, s, why) in &unavailable {
        progress(Event::Log(format!("{id}: {} ({why})", s.as_str())));
    }
    // tasks whose per-endpoint cases could not be built (task -> reason); they become NOT_RUN gates
    let mut server_task_errors: BTreeMap<String, String> = BTreeMap::new();
    let mut server_case_counts: BTreeMap<String, usize> = BTreeMap::new();

    // 2. models
    let mut model_rows: Vec<(String, Vec<CaseResult>)> = Vec::new();
    let mut fingerprints: Vec<Value> = Vec::new();
    let mut reused_total = 0usize;
    for (idx, ep) in cfg.models().enumerate() {
        let role = if idx == 0 { "reference" } else { "candidate" };
        let sub = if idx == 0 {
            "baseline".to_string()
        } else {
            format!("candidates/{}", ep.id)
        };
        let sub_dir = out_dir.join(&sub);
        let mut fp = json!({"id": ep.id, "role": role, "kind": format!("{:?}", ep.kind).to_lowercase(),
                            "provenance": ep.provenance, "talus": ep.talus.as_ref().map(|t| json!({
                                "model": t.model, "level": t.level, "choices": talus_choices(t), "apply": t.apply}))});
        let rows: Vec<CaseResult> = match ep.kind {
            EndpointKind::Frozen => {
                let dir = PathBuf::from(ep.frozen.clone().unwrap_or_default());
                progress(Event::Phase(format!(
                    "{}: loading frozen reference {}",
                    ep.id,
                    dir.display()
                )));
                let (rows, meta) = load_frozen(&dir, &settings_hash)?;
                fp["frozen"] = json!({"dir": dir.display().to_string(), "content_sha256": meta["content_sha256"],
                                      "frozen_at": meta["frozen_at"], "model": meta["model"]});
                let wanted: std::collections::BTreeSet<&str> =
                    resolved.iter().map(|r| r.task.id()).collect();
                let rows: Vec<CaseResult> = rows
                    .into_iter()
                    .filter(|r| wanted.contains(r.task.as_str()))
                    .collect();
                progress(Event::ModelDone {
                    model: ep.id.clone(),
                    correct: rows.iter().filter(|r| r.pass()).count(),
                    total: rows.len(),
                });
                rows
            }
            EndpointKind::Http => {
                let client =
                    client_for(ep, cfg.inference.request_timeout_s, opts.api_key.as_ref())?;
                let ctl = client.with_timeout(15);
                // the case plan: static cases + cases built against this endpoint
                let mut plan: Vec<(&Resolved, Case)> = Vec::new();
                for r in &resolved {
                    match &r.cases {
                        Some(cs) => plan.extend(cs.iter().map(|c| (r, c.clone()))),
                        None => {
                            let id = r.task.id();
                            if server_task_errors.contains_key(id) {
                                continue;
                            }
                            let slow = client.with_timeout(600);
                            let ctx = TaskContext {
                                tier,
                                limit: r.limit,
                                cache: &cache,
                                params: params_of(id),
                                client: Some(&slow),
                                run_dir: Some(&out_dir),
                                resume: opts.resume,
                            };
                            match r.task.cases(&ctx) {
                                Ok(cs) => {
                                    server_case_counts.insert(id.to_string(), cs.len());
                                    plan.extend(cs.into_iter().map(|c| (r, c)));
                                }
                                Err(e) => {
                                    progress(Event::Log(format!(
                                        "{id}: {} ({e})",
                                        err_status(&e).as_str()
                                    )));
                                    server_task_errors.insert(id.to_string(), e);
                                }
                            }
                        }
                    }
                }
                let trials: Vec<(&Resolved, &Case, Option<u64>)> = plan
                    .iter()
                    .flat_map(|(r, c)| trial_seeds.iter().map(move |s| (*r, c, *s)))
                    .collect();
                // resume: keep rows of the same case, prompt, settings and scorer that did not fail transiently
                let prior = if opts.resume {
                    prior_rows(&sub_dir)
                } else {
                    BTreeMap::new()
                };
                let mut kept: BTreeMap<(String, Option<u64>), CaseResult> = BTreeMap::new();
                for (r, c, s) in &trials {
                    let key = (c.case_id.clone(), *s);
                    if let Some(p) = prior.get(&key) {
                        if p.prompt_hash == c.prompt_hash()
                            && p.settings_hash == settings_hash
                            && p.scorer_version == r.task.scorer_version()
                            && !p.retryable()
                        {
                            kept.insert(key, p.clone());
                        }
                    }
                }
                // batch (external framework) tasks: all of a task's rows are kept, or the task runs again
                let mut batch_kept: Vec<CaseResult> = Vec::new();
                let mut batch_todo: Vec<&Resolved> = Vec::new();
                for r in resolved.iter().filter(|r| r.task.is_batch()) {
                    let id = r.task.id();
                    if server_task_errors.contains_key(id) {
                        continue;
                    }
                    let prev: Vec<CaseResult> =
                        prior.values().filter(|p| p.task == id).cloned().collect();
                    if !prev.is_empty()
                        && prev
                            .iter()
                            .all(|p| !p.retryable() && p.settings_hash == settings_hash)
                    {
                        batch_kept.extend(prev);
                    } else {
                        batch_todo.push(r);
                    }
                }
                let todo = trials.len() - kept.len() + batch_todo.len();
                reused_total += kept.len() + batch_kept.len();
                if !kept.is_empty() {
                    progress(Event::Log(format!(
                        "{}: resuming, {} of {} cases kept from the earlier segment",
                        ep.id,
                        kept.len(),
                        trials.len()
                    )));
                }
                let fp_file = sub_dir.join("fingerprint.json");
                if todo > 0 {
                    if past(opts.deadline) {
                        return Err(format!(
                            "{INCOMPLETE}: segment time limit reached before {} (resume with --resume)",
                            ep.id
                        ));
                    }
                    if let Some(t) = &ep.talus {
                        ensure_selection(&ctl, &ep.id, t, poll, &cancel, progress)?;
                    }
                    fp["server"] = fingerprint(&ctl);
                    art.write_json(&format!("{sub}/fingerprint.json"), &fp["server"])?;
                } else {
                    fp["server"] = std::fs::read_to_string(&fp_file)
                        .ok()
                        .and_then(|t| serde_json::from_str(&t).ok())
                        .unwrap_or(Value::Null);
                }
                let mut kept_rows: Vec<CaseResult> = kept.values().cloned().collect();
                kept_rows.extend(batch_kept.iter().cloned());
                let mut writer = PartialWriter::create(&sub_dir, &kept_rows)?;
                let served_model = ep
                    .model
                    .clone()
                    .or_else(|| fp["server"]["models"][0].as_str().map(str::to_string))
                    .unwrap_or_else(|| "default".into());
                let served_ctx = fp["server"]["talus_status"]["context_max"]
                    .as_u64()
                    .filter(|c| *c > 0)
                    .or_else(|| fp["server"]["props"]["n_ctx"].as_u64());
                let key_for_jobs = api_key(ep).or_else(|| opts.api_key.clone());
                let mk_env = |id: &str, limit: Option<usize>| ExecEnv {
                    base_url: client.url(),
                    api_key: key_for_jobs.clone(),
                    model: served_model.clone(),
                    context: served_ctx,
                    work_dir: sub_dir.join("jobs"),
                    label: format!("{}-{}", cfg.experiment.id, ep.id),
                    params: params_of(id).clone(),
                    limit,
                    temperature: cfg.inference.temperature,
                    seed: cfg.inference.seed,
                    max_tokens: cfg.inference.max_output_tokens,
                    cancel: Arc::clone(&cancel),
                };
                progress(Event::ModelStart {
                    model: ep.id.clone(),
                    total: trials.len(),
                });
                let mut rows = Vec::with_capacity(trials.len());
                for (i, (r, case, seed)) in trials.iter().enumerate() {
                    let key = (case.case_id.clone(), *seed);
                    let row = match kept.remove(&key) {
                        Some(row) => row,
                        None => {
                            if cancel.load(Ordering::Relaxed) {
                                return Err("cancelled".into());
                            }
                            if past(opts.deadline) {
                                return Err(format!(
                                    "{INCOMPLETE}: segment time limit reached in {} after {} of {} cases (resume with --resume)",
                                    ep.id, i, trials.len()
                                ));
                            }
                            let row = match r.task.execute(case, &mk_env(r.task.id(), r.limit)) {
                                Some(o) => {
                                    if cancel.load(Ordering::Relaxed) {
                                        return Err("cancelled".into());
                                    }
                                    exec_row(
                                        cfg,
                                        ep,
                                        r.task.as_ref(),
                                        case,
                                        *seed,
                                        &settings_hash,
                                        o,
                                    )
                                }
                                None => run_case(
                                    cfg,
                                    ep,
                                    &client,
                                    r.task.as_ref(),
                                    case,
                                    *seed,
                                    &settings_hash,
                                ),
                            };
                            writer.push(&row)?;
                            row
                        }
                    };
                    progress(Event::Case {
                        model: ep.id.clone(),
                        done: i + 1,
                        total: trials.len(),
                        case_id: case.case_id.clone(),
                        status: row.status,
                        pass: row.pass(),
                    });
                    rows.push(row);
                }
                for r in batch_todo {
                    let id = r.task.id();
                    if cancel.load(Ordering::Relaxed) {
                        return Err("cancelled".into());
                    }
                    if past(opts.deadline) {
                        return Err(format!(
                            "{INCOMPLETE}: segment time limit reached in {} before {id} (resume with --resume)",
                            ep.id
                        ));
                    }
                    progress(Event::Phase(format!(
                        "{}: running {id} (external framework)",
                        ep.id
                    )));
                    match r.task.run_batch(&mk_env(id, r.limit)) {
                        Ok(brs) => {
                            for b in brs {
                                let row = batch_row(cfg, ep, id, b, &settings_hash);
                                writer.push(&row)?;
                                rows.push(row);
                            }
                        }
                        Err(e) => {
                            if cancel.load(Ordering::Relaxed) {
                                return Err("cancelled".into());
                            }
                            progress(Event::Log(format!(
                                "{id}: {} ({e})",
                                err_status(&e).as_str()
                            )));
                            server_task_errors.insert(id.to_string(), e);
                        }
                    }
                }
                rows.extend(batch_kept);
                if todo > 0 && !fp["server"]["talus_status"]["build"].is_null() {
                    refresh_build(&ctl, &mut fp["server"]);
                    art.write_json(&format!("{sub}/fingerprint.json"), &fp["server"])?;
                }
                progress(Event::ModelDone {
                    model: ep.id.clone(),
                    correct: rows.iter().filter(|r| r.pass()).count(),
                    total: rows.len(),
                });
                rows
            }
        };
        art.write_text(&format!("{sub}/cases.jsonl"), &crate::case::to_jsonl(&rows))?;
        let _ = std::fs::remove_file(sub_dir.join("cases.partial.jsonl"));
        art.write_json(
            &format!("{sub}/metrics.json"),
            &compare::model_metrics(&rows, cfg.quality_gates.confidence),
        )?;
        if idx > 0 {
            art.write_json(&format!("{sub}/perf.json"), &passive_perf(&rows))?;
        }
        fp["cases"] = json!(rows.len());
        fingerprints.push(fp);
        model_rows.push((ep.id.clone(), rows));
    }
    for (id, e) in &server_task_errors {
        unavailable.push((id.clone(), err_status(e), e.clone()));
    }
    // a frozen reference may hold more cases than this run planned (other limits): keep the planned ones
    if cfg.reference.kind == EndpointKind::Frozen && model_rows.len() > 1 {
        let planned: std::collections::BTreeSet<(String, Option<u64>)> = model_rows[1..]
            .iter()
            .flat_map(|(_, rs)| rs.iter().map(CaseResult::key))
            .collect();
        model_rows[0].1.retain(|r| planned.contains(&r.key()));
        art.write_text(
            "baseline/cases.jsonl",
            &crate::case::to_jsonl(&model_rows[0].1),
        )?;
    }
    let datasets_json: Vec<Value> = resolved
        .iter()
        .map(|r| {
            let id = r.task.id();
            json!({"task": id, "scorer_version": r.task.scorer_version(),
                   "cases": r.cases.as_ref().map(Vec::len).or(server_case_counts.get(id).copied()),
                   "limit": r.limit, "built_per_endpoint": r.cases.is_none(),
                   "datasets": r.task.datasets(tier), "provenance": r.task.provenance()})
        })
        .collect();
    art.write_json("datasets.json", &json!(datasets_json))?;

    // 3. pairing and gates
    progress(Event::Phase("pairing, statistics and gates".into()));
    let g = &cfg.quality_gates;
    let ref_rows = &model_rows[0].1;
    let mut paired_all: Vec<PairedTask> = Vec::new();
    let mut pair_rows: Vec<(String, PairRow)> = Vec::new();
    let mut scored_pairs: Vec<compare::ScoredPair> = Vec::new();
    let mut verdicts = Vec::new();
    for (ci, (cand_id, rows)) in model_rows.iter().enumerate().skip(1) {
        let mut gate_list: Vec<GateResult> = Vec::new();
        // the served model (per-model kld_topk limits): the candidate's, else the reference's
        let served = [ci, 0]
            .iter()
            .find_map(|&i| fingerprints.get(i)?["server"]["models"][0].as_str())
            .unwrap_or("")
            .to_string();
        for r in &resolved {
            let tid = r.task.id();
            if server_task_errors.contains_key(tid) {
                continue;
            }
            let req = required.iter().any(|x| x == tid);
            let (p, prs) = compare::pair_task(
                tid,
                cand_id,
                ref_rows,
                rows,
                g.bootstrap_resamples,
                g.confidence,
                cfg.experiment.seed,
            );
            let mut gated = false;
            if let Some(t) = g.max_loss_pp.get(tid) {
                gate_list.push(gates::loss_gate(&p, *t, g, req));
                gated = true;
            }
            let mine: Vec<&CaseResult> = rows.iter().filter(|x| x.task == tid).collect();
            if let Some(t) = g.min_rate_pct.get(tid) {
                gate_list.push(gates::rate_gate(
                    tid,
                    mine.iter().filter(|x| x.pass()).count(),
                    mine.len(),
                    *t,
                    g,
                    req,
                ));
                gated = true;
            }
            if g.all_pass.iter().any(|x| x == tid) {
                gate_list.push(gates::all_pass_gate(tid, &mine, req));
                gated = true;
            }
            if g.identity.iter().any(|x| x == tid) {
                gate_list.push(gates::identity_gate(&p, req));
                gated = true;
            }
            // teacher-forced scoring tasks: the reference's scores are compared position by position
            if tid == "perplexity" || tid == "kld_topk" {
                let sp = compare::pair_scored(
                    tid,
                    cand_id,
                    ref_rows,
                    rows,
                    g.bootstrap_resamples,
                    g.confidence,
                    cfg.experiment.seed,
                );
                gate_list.push(if tid == "perplexity" {
                    gates::ppl_ratio_gate(&sp, g.ppl_ratio, g, req)
                } else {
                    gates::kld_topk_gate(&sp, &g.kld_topk, &served, req)
                });
                gated = true;
                scored_pairs.push(sp);
            }
            if !gated {
                gate_list.push(gates::availability_gate(
                    tid,
                    GateStatus::Warn,
                    "no gate configured for this task (reported only)".into(),
                    false,
                ));
            }
            pair_rows.extend(prs.into_iter().map(|x| (cand_id.clone(), x)));
            paired_all.push(p);
        }
        for (id, s, why) in &unavailable {
            gate_list.push(gates::availability_gate(
                id,
                *s,
                why.clone(),
                required.contains(id),
            ));
        }
        verdicts.push(gates::overall(
            cand_id,
            gate_list,
            cfg.benchmarks.tier,
            &cfg.benchmarks.tasks,
        ));
    }

    // 4. artifacts
    art.write_text(
        "comparisons/paired_transitions.csv",
        &report::transitions_csv(&pair_rows),
    )?;
    art.write_text(
        "comparisons/benchmark_summary.csv",
        &report::summary_csv(&paired_all),
    )?;
    art.write_json(
        "comparisons/confidence_intervals.json",
        &json!(paired_all.iter().map(|p| json!({"candidate": p.candidate_id, "task": p.task, "loss_pp": p.loss_pp,
            "ci_loss_pp": p.ci_loss_pp, "mcnemar_p": p.mcnemar_p, "n": p.n_paired, "trials": p.n_trials,
            "method": format!("paired percentile bootstrap over case ids (trials averaged per case), {} resamples, seed {}", g.bootstrap_resamples, cfg.experiment.seed)})).collect::<Vec<_>>()),
    )?;
    if !scored_pairs.is_empty() {
        art.write_json(
            "comparisons/teacher_forced.json",
            &json!({"note": "perplexity / kld_topk paired over identical token ids (POST /v1/talus/score); KL is the top-k screening approximation",
                    "pairs": scored_pairs}),
        )?;
    }
    let regressions =
        report::regressions_jsonl(&paired_all, &model_rows, cfg.report.redact_prompts);
    art.write_text("comparisons/regressions.jsonl", &regressions)?;
    let finished = unix_now();
    let manifest = json!({
        "schema": "talus-bench/manifest/1",
        "experiment": cfg.experiment,
        "tier": cfg.benchmarks.tier.id(),
        "started_at": rfc3339(started), "finished_at": rfc3339(finished), "wall_s": (finished - started).round(),
        "resumed": {"enabled": opts.resume, "rows_kept": reused_total},
        "bench": {"name": "talus-bench", "version": env!("CARGO_PKG_VERSION"),
                  "commit": option_env!("TALUS_BUILD_COMMIT").unwrap_or("unrecorded")},
        "client": {"host": crate::util::hostname(), "os": std::env::consts::OS, "arch": std::env::consts::ARCH},
        "config_sha256": sha256_hex(cfg.to_yaml().as_bytes()),
        "settings": settings, "settings_hash": settings_hash,
        "datasets": datasets_json,
        "models": fingerprints,
        "validation_warnings": v.warnings,
        "task_caveats": resolved.iter().flat_map(|r| r.task.caveats()).collect::<Vec<_>>(),
        "limitations": [
            "full-vocabulary KLD is NOT_SUPPORTED over HTTP (GPU-host mode kld_teacher_forced); perplexity and the kld_topk screening KL need talus-server's POST /v1/talus/score and are NOT_SUPPORTED on servers or engines without it",
            "json_schema measures prompt-following JSON (no response_format); json_schema_strict sends response_format json_schema (grammar-constrained, server extension E6; engines without it answer HTTP 400)",
            "per-case timings are passive (quality requests); performance is measured by the perf tier only",
        ],
    });
    art.write_json("manifest.json", &manifest)?;
    let ctx = report::ReportCtx {
        cfg,
        manifest: &manifest,
        paired: &paired_all,
        verdicts: &verdicts,
        unavailable: &unavailable,
        model_rows: &model_rows,
    };
    let report_json = report::report_json(&ctx);
    let report_md = report::report_md(&ctx);
    art.write_json("report.json", &report_json)?;
    art.write_text("report.md", &report_md)?;
    progress(Event::Phase(format!(
        "done: report at {}",
        out_dir.join("report.md").display()
    )));
    Ok(RunOutcome {
        output_dir: out_dir,
        verdicts,
        report_md,
        report_json,
    })
}

/// Passive request statistics of a quality run (not a benchmark; see the perf tier for measured cells).
pub fn passive_perf(rows: &[CaseResult]) -> Value {
    let mut by: BTreeMap<&str, (u64, f64, usize)> = BTreeMap::new();
    // answer-vs-reasoning decode split (talus-server timings.reasoning_n/answer_n/reasoning_ms/answer_ms):
    // (reasoning tokens, reasoning ms, answer tokens, answer ms, requests)
    let mut split: BTreeMap<&str, (u64, f64, u64, f64, usize)> = BTreeMap::new();
    for r in rows {
        if let (Some(n), Some(ms)) = (r.usage.completion_tokens, r.timings.server_decode_ms) {
            let e = by.entry(r.task.as_str()).or_default();
            e.0 += n;
            e.1 += ms;
            e.2 += 1;
        }
        let t = &r.timings;
        if let (Some(rn), Some(rms), Some(an), Some(ams)) =
            (t.reasoning_n, t.reasoning_ms, t.answer_n, t.answer_ms)
        {
            let e = split.entry(r.task.as_str()).or_default();
            e.0 += rn;
            e.1 += rms;
            e.2 += an;
            e.3 += ams;
            e.4 += 1;
        }
    }
    let rate = |n: u64, ms: f64| {
        if ms > 0.0 {
            Some(n as f64 * 1000.0 / ms)
        } else {
            None
        }
    };
    let mut v = json!({
        "kind": "passive",
        "note": "summed server decode timings of the quality requests; mixed prompts, not comparable to perf-tier cells",
        "tasks": by.iter().map(|(k, (n, ms, c))| (k.to_string(), json!({"requests": c, "completion_tokens": n,
            "decode_ms": ms, "decode_tok_s": rate(*n, *ms)}))).collect::<serde_json::Map<_, _>>(),
    });
    if !split.is_empty() {
        v["channels"] = json!({
            "note": "decode split at the first answer token (server token arrival times); reasoning + answer = decode",
            "tasks": split.iter().map(|(k, (rn, rms, an, ams, c))| (k.to_string(), json!({"requests": c,
                "reasoning_tokens": rn, "reasoning_ms": rms, "reasoning_tok_s": rate(*rn, *rms),
                "answer_tokens": an, "answer_ms": ams, "answer_tok_s": rate(*an, *ams)}))).collect::<serde_json::Map<_, _>>(),
        });
    }
    v
}
