//! The experiment config (YAML, spec §6.1 adapted to Talus) and its validation.
//!
//! Secrets never live in the file: an endpoint names the environment variable that holds its key (`api_key_env`).
//! Unknown fields are errors, so a typo (or a pasted `api_key:`) is caught instead of being ignored.

use crate::tasks::{self, TaskAvailability};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub experiment: Experiment,
    pub reference: Endpoint,
    /// Empty = a reference-only run: absolute rates with Wilson intervals, no paired gates (exit 3).
    #[serde(default)]
    pub candidates: Vec<Endpoint>,
    #[serde(default)]
    pub inference: Inference,
    pub benchmarks: Benchmarks,
    #[serde(default)]
    pub quality_gates: GateConfig,
    #[serde(default)]
    pub report: ReportConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Experiment {
    pub id: String,
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Default: `artifacts/<id>` (relative to the working directory).
    #[serde(default)]
    pub output_dir: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

fn default_seed() -> u64 {
    42
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum EndpointKind {
    /// An OpenAI-compatible server reached over HTTP (talus-server, local or remote).
    #[default]
    Http,
    /// A frozen, content-addressed reference run (`talus-bench freeze`); only valid for `reference`.
    Frozen,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub id: String,
    #[serde(default)]
    pub kind: EndpointKind,
    /// `http://host:port` (an optional trailing `/v1` is accepted). Plain HTTP only.
    #[serde(default)]
    pub endpoint: Option<String>,
    /// The `model` field sent with each request (talus-server serves one model; any value it accepts).
    #[serde(default)]
    pub model: Option<String>,
    /// Name of the environment variable holding the API key (never the key itself).
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Directory of a frozen reference (kind = frozen).
    #[serde(default)]
    pub frozen: Option<String>,
    /// Talus-vs-Talus: put the server on this model/level/choices before the run (through /v1/talus/apply).
    #[serde(default)]
    pub talus: Option<TalusSelection>,
    #[serde(default)]
    pub provenance: Provenance,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TalusSelection {
    /// talus-presets model id, e.g. `qwen3.8-27b`.
    pub model: String,
    /// reference | balanced | fast (talus-presets accuracy levels). Default balanced.
    #[serde(default)]
    pub level: Option<String>,
    /// Extra option choices on top of the level (talus-presets option id -> value).
    #[serde(default)]
    pub choices: BTreeMap<String, String>,
    /// POST /v1/talus/apply before running (needs a server started with --control-dir). If false, the selection
    /// is only checked against /v1/talus/status and recorded.
    #[serde(default)]
    pub apply: bool,
    /// Seconds to wait for the engine to come back after an apply. Default 900 (a model switch loads weights).
    #[serde(default = "default_apply_timeout")]
    pub apply_timeout_s: u64,
}

fn default_apply_timeout() -> u64 {
    900
}

/// What the operator knows about the weights and runtime. The bench records it verbatim; fields the server can
/// report itself (see DESIGN.md "server extensions") are fingerprinted separately in the manifest.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    #[serde(default)]
    pub base_checkpoint: Option<String>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub quantizer: Option<String>,
    #[serde(default)]
    pub group_size: Option<u32>,
    #[serde(default)]
    pub calibration_dataset_revision: Option<String>,
    #[serde(default)]
    pub activation_precision: Option<String>,
    #[serde(default)]
    pub kv_cache_precision: Option<String>,
    #[serde(default)]
    pub weights_sha256: Option<String>,
    #[serde(default)]
    pub engine_build: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Thinking {
    /// Send `chat_template_kwargs.enable_thinking = false` (default: objective tasks score the answer only).
    #[default]
    Off,
    On,
    /// Send nothing; the server's template default applies.
    ServerDefault,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Inference {
    #[serde(default)]
    pub temperature: f64,
    #[serde(default = "one")]
    pub top_p: f64,
    #[serde(default = "default_max_tokens")]
    pub max_output_tokens: u32,
    #[serde(default = "default_inference_seed")]
    pub seed: Option<u64>,
    #[serde(default)]
    pub thinking: Thinking,
    #[serde(default = "default_timeout")]
    pub request_timeout_s: u64,
    /// Talus serves one request at a time; quality runs are sequential. Only 1 is accepted.
    #[serde(default = "one_u32")]
    pub concurrency_for_quality: u32,
    /// Retries for transport errors only (never for a wrong answer). Each retry is recorded on the case.
    #[serde(default)]
    pub retries: u32,
    /// Repeat every case once per seed (stochastic sampling). Results are paired per (case, seed) and averaged
    /// per case for the statistics. Replaces `seed` when set.
    #[serde(default)]
    pub seeds: Option<Vec<u64>>,
}

fn one() -> f64 {
    1.0
}
fn one_u32() -> u32 {
    1
}
fn default_max_tokens() -> u32 {
    1024
}
fn default_inference_seed() -> Option<u64> {
    Some(42)
}
fn default_timeout() -> u64 {
    600
}

impl Default for Inference {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            max_output_tokens: default_max_tokens(),
            seed: default_inference_seed(),
            thinking: Thinking::Off,
            request_timeout_s: default_timeout(),
            concurrency_for_quality: 1,
            retries: 0,
            seeds: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    #[default]
    Smoke,
    Full,
    Qualification,
}

impl Tier {
    pub fn id(self) -> &'static str {
        match self {
            Tier::Smoke => "smoke",
            Tier::Full => "full",
            Tier::Qualification => "qualification",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Benchmarks {
    #[serde(default)]
    pub tier: Tier,
    pub tasks: Vec<String>,
    /// Per-task case limit (first N cases of the pinned dataset, in dataset order).
    #[serde(default)]
    pub limits: BTreeMap<String, usize>,
    /// Tasks whose gate must run for an overall PASS. Default: every task listed.
    #[serde(default)]
    pub required: Option<Vec<String>>,
    /// Where fetched datasets are cached. Default `~/.cache/talus-bench/datasets`.
    #[serde(default)]
    pub dataset_cache: Option<String>,
    /// Task parameters, e.g. `needle: {lengths: [60000], seeds: [1, 2, 3]}`, `mmlu_pro: {split: full}`.
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
}

impl Benchmarks {
    pub fn required(&self) -> Vec<String> {
        self.required.clone().unwrap_or_else(|| self.tasks.clone())
    }
}

/// A two-level threshold, as in spec §8: `target` (initial target) and `investigate`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Threshold {
    pub target: f64,
    pub investigate: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GateConfig {
    /// Fewer paired cases than this gives at most WARN ("sample too small").
    #[serde(default = "default_min_cases")]
    pub min_cases: usize,
    #[serde(default = "default_resamples")]
    pub bootstrap_resamples: u32,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    /// Absolute accuracy loss in percentage points (reference - candidate), per task.
    #[serde(default = "default_loss_gates")]
    pub max_loss_pp: BTreeMap<String, Threshold>,
    /// Minimum absolute candidate rate in percent, per task (e.g. strict JSON validity).
    #[serde(default = "default_rate_gates")]
    pub min_rate_pct: BTreeMap<String, Threshold>,
    /// Tasks whose every candidate case must pass (release gates such as needle 15/15).
    #[serde(default = "default_all_pass")]
    pub all_pass: Vec<String>,
    /// Tasks whose candidate outputs must be byte-identical to the reference's (greedy identity).
    #[serde(default = "default_identity")]
    pub identity: Vec<String>,
    /// `perplexity`: candidate / reference perplexity over the same tokens (teacher-forced). Default target 1.01
    /// (the release ledger's "PPL ratio <= 1.01"), investigate 1.05 (spec §8 "perplexity increase > 5%").
    #[serde(default = "default_ppl_ratio")]
    pub ppl_ratio: Threshold,
    /// `kld_topk`: screening KL from top-k lists. Provisional and WARN-only by default (not a release gate).
    #[serde(default)]
    pub kld_topk: KldTopkGate,
}

/// Thresholds of the `kld_topk` screening gate. Top-k KL differs from the full-vocabulary KL of the release
/// gates (kld_teacher_forced), so these values are provisional until derived from paired full-vocabulary runs.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct KldTopkGate {
    /// Mean per-position KL(reference || candidate) in nats.
    #[serde(default = "default_kld_topk_mean")]
    pub max_mean: f64,
    /// Minimum top-1 agreement in percent.
    #[serde(default = "default_kld_topk_top1")]
    pub min_top1_pct: f64,
    /// false (default): a breach WARNs; true: it FAILs like a release gate.
    #[serde(default)]
    pub enforce: bool,
}

fn default_kld_topk_mean() -> f64 {
    0.05
}
fn default_kld_topk_top1() -> f64 {
    90.0
}

impl Default for KldTopkGate {
    fn default() -> Self {
        Self {
            max_mean: default_kld_topk_mean(),
            min_top1_pct: default_kld_topk_top1(),
            enforce: false,
        }
    }
}

pub fn default_ppl_ratio() -> Threshold {
    Threshold {
        target: 1.01,
        investigate: 1.05,
    }
}

fn default_all_pass() -> Vec<String> {
    vec!["needle".into(), "agentic_go".into()]
}

fn default_identity() -> Vec<String> {
    vec!["greedy_identity".into()]
}

fn default_min_cases() -> usize {
    30
}
fn default_resamples() -> u32 {
    2000
}
fn default_confidence() -> f64 {
    0.95
}

/// Spec §8 provisional table, plus Talus additions (gsm8k), keyed by task id. Provisional engineering thresholds.
pub fn default_loss_gates() -> BTreeMap<String, Threshold> {
    let t = |a, b| Threshold {
        target: a,
        investigate: b,
    };
    [
        ("mmlu_pro", t(1.0, 2.0)),
        ("gpqa_diamond", t(2.0, 4.0)),
        ("livecodebench", t(2.0, 4.0)),
        ("ifeval", t(1.0, 2.0)),
        ("bfcl", t(1.0, 2.0)),
        ("agentic_go", t(2.0, 5.0)),
        ("swebench_verified", t(2.0, 5.0)),
        ("gsm8k", t(2.0, 4.0)),
        ("json_schema", t(1.0, 2.0)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

pub fn default_rate_gates() -> BTreeMap<String, Threshold> {
    [(
        "json_schema".to_string(),
        Threshold {
            target: 99.5,
            investigate: 99.0,
        },
    )]
    .into()
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            min_cases: default_min_cases(),
            bootstrap_resamples: default_resamples(),
            confidence: default_confidence(),
            max_loss_pp: default_loss_gates(),
            min_rate_pct: default_rate_gates(),
            all_pass: default_all_pass(),
            identity: default_identity(),
            ppl_ratio: default_ppl_ratio(),
            kld_topk: KldTopkGate::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReportConfig {
    /// Keep raw model outputs in cases.jsonl (local-only artifacts). Default true.
    #[serde(default = "yes")]
    pub store_raw_outputs: bool,
    /// Replace prompts in cases.jsonl and the report with their hash (for private holdout data).
    #[serde(default)]
    pub redact_prompts: bool,
    /// Representative regressions shown per task in report.md.
    #[serde(default = "default_examples")]
    pub max_examples: usize,
}

fn yes() -> bool {
    true
}
fn default_examples() -> usize {
    5
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            store_raw_outputs: true,
            redact_prompts: false,
            max_examples: default_examples(),
        }
    }
}

/// Validation result: errors stop the run, warnings are printed and recorded in the manifest.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Validation {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl Validation {
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }
}

fn valid_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 80
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
        && !s.starts_with('.')
}

impl Config {
    pub fn from_yaml(text: &str) -> Result<Self, String> {
        yaml_serde::from_str(text).map_err(|e| {
            let m = e.to_string();
            if m.contains("api_key`") || m.contains("unknown field `api_key`") {
                format!("{m}\nhint: keys never go in the config; set `api_key_env: NAME` and export NAME")
            } else {
                m
            }
        })
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_yaml(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn to_yaml(&self) -> String {
        yaml_serde::to_string(self).unwrap_or_default()
    }

    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(
            self.experiment
                .output_dir
                .clone()
                .unwrap_or_else(|| format!("artifacts/{}", self.experiment.id)),
        )
    }

    pub fn models(&self) -> impl Iterator<Item = &Endpoint> {
        std::iter::once(&self.reference).chain(self.candidates.iter())
    }

    pub fn validate(&self) -> Validation {
        let mut v = Validation::default();
        let e = &mut v.errors;
        if !valid_id(&self.experiment.id) {
            e.push(format!(
                "experiment.id {:?}: use letters, digits, '-', '_', '.' (max 80)",
                self.experiment.id
            ));
        }
        if self.candidates.is_empty() {
            if self.reference.kind == EndpointKind::Frozen {
                e.push("candidates: a frozen reference needs at least one candidate".into());
            } else {
                v.warnings.push(
                    "no candidates: reference-only run (absolute rates with Wilson intervals; no paired gates)".into(),
                );
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for (role, ep) in std::iter::once(("reference", &self.reference))
            .chain(self.candidates.iter().map(|c| ("candidate", c)))
        {
            if !valid_id(&ep.id) {
                e.push(format!(
                    "{role} id {:?}: use letters, digits, '-', '_', '.'",
                    ep.id
                ));
            }
            if !seen.insert(ep.id.clone()) {
                e.push(format!("{role} id {:?} is used twice", ep.id));
            }
            match ep.kind {
                EndpointKind::Http => match ep.endpoint.as_deref() {
                    None | Some("") => e.push(format!("{role} {}: endpoint is required", ep.id)),
                    Some(u) if u.starts_with("https://") => e.push(format!(
                        "{role} {}: https is not supported (talus-server speaks plain HTTP; use an SSH tunnel)",
                        ep.id
                    )),
                    Some(u) => {
                        if crate::http::Client::new(u, None, 1).is_err() {
                            e.push(format!("{role} {}: bad endpoint {u:?}", ep.id));
                        }
                    }
                },
                EndpointKind::Frozen => {
                    if role != "reference" {
                        e.push(format!(
                            "candidate {}: kind frozen is only valid for the reference",
                            ep.id
                        ));
                    }
                    if ep.frozen.as_deref().unwrap_or("").is_empty() {
                        e.push(format!("{role} {}: frozen: <dir> is required", ep.id));
                    }
                    if ep.talus.as_ref().is_some_and(|t| t.apply) {
                        e.push(format!("{role} {}: a frozen reference cannot apply settings", ep.id));
                    }
                }
            }
            if let Some(var) = &ep.api_key_env {
                if var.is_empty() || !var.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
                    e.push(format!(
                        "{role} {}: api_key_env must be an environment variable name",
                        ep.id
                    ));
                }
            }
            if let Some(t) = &ep.talus {
                match talus_presets::model(&t.model) {
                    None => e.push(format!(
                        "{role} {}: talus.model {:?} is not in the talus-presets catalog",
                        ep.id, t.model
                    )),
                    Some(spec) => {
                        if let Some(l) = &t.level {
                            if talus_presets::Level::parse(l).is_none() {
                                e.push(format!(
                                    "{role} {}: talus.level {l:?} (use reference, balanced or fast)",
                                    ep.id
                                ));
                            }
                        }
                        if let Err(msg) = spec.edits(&crate::runner::talus_choices(t)) {
                            e.push(format!("{role} {}: talus choices: {msg}", ep.id));
                        }
                    }
                }
            }
        }
        let inf = &self.inference;
        if !(0.0..=2.0).contains(&inf.temperature) {
            e.push("inference.temperature must be in 0..2".into());
        }
        if !(inf.top_p > 0.0 && inf.top_p <= 1.0) {
            e.push("inference.top_p must be in (0, 1]".into());
        }
        if inf.max_output_tokens == 0 {
            e.push("inference.max_output_tokens must be > 0".into());
        }
        if inf.concurrency_for_quality != 1 {
            e.push(
                "inference.concurrency_for_quality: talus-server serves one request at a time; only 1 is supported"
                    .into(),
            );
        }
        if inf.request_timeout_s == 0 {
            e.push("inference.request_timeout_s must be > 0".into());
        }
        if let Some(seeds) = &inf.seeds {
            let uniq: std::collections::BTreeSet<_> = seeds.iter().collect();
            if seeds.is_empty() || uniq.len() != seeds.len() {
                e.push("inference.seeds must be a non-empty list of distinct seeds".into());
            }
            if inf.temperature == 0.0 {
                v.warnings.push("inference.seeds with temperature 0: the trials will mostly repeat the same greedy answer".into());
            }
        }
        if inf.temperature > 0.0 {
            v.warnings.push(format!(
                "inference.temperature {} > 0: exact-answer tasks become stochastic; repeat with several seeds before trusting a delta",
                inf.temperature
            ));
        }
        let b = &self.benchmarks;
        if b.tasks.is_empty() {
            v.errors.push("benchmarks.tasks is empty".into());
        }
        let mut tseen = std::collections::BTreeSet::new();
        for t in &b.tasks {
            if !tseen.insert(t) {
                v.errors
                    .push(format!("benchmarks.tasks: {t:?} listed twice"));
            }
            match tasks::availability(t) {
                TaskAvailability::Unknown => v.errors.push(format!(
                    "benchmarks.tasks: unknown task {t:?} (known: {})",
                    tasks::KNOWN
                        .iter()
                        .map(|k| k.0)
                        .collect::<Vec<_>>()
                        .join(", ")
                )),
                TaskAvailability::Native => {}
                TaskAvailability::Adapter(tool) => v.warnings.push(format!(
                    "task {t}: runs {tool} as a subprocess; reported as NOT_SUPPORTED if it is missing on this machine"
                )),
                other => v.warnings.push(format!(
                    "task {t}: {} in this build; it will be reported as {}",
                    other.reason(t),
                    other.status().as_str()
                )),
            }
        }
        for k in b.limits.keys() {
            if !b.tasks.contains(k) {
                v.warnings
                    .push(format!("benchmarks.limits.{k}: not in tasks"));
            }
        }
        for (k, n) in &b.limits {
            if *n == 0 {
                v.errors.push(format!("benchmarks.limits.{k} must be > 0"));
            }
        }
        if let Some(req) = &b.required {
            for r in req {
                if !b.tasks.contains(r) {
                    v.errors.push(format!(
                        "benchmarks.required: {r:?} is not in benchmarks.tasks (a required test that is never run can never pass)"
                    ));
                }
            }
        }
        let g = &self.quality_gates;
        if !(0.5..1.0).contains(&g.confidence) {
            v.errors
                .push("quality_gates.confidence must be in [0.5, 1)".into());
        }
        if g.bootstrap_resamples < 200 {
            v.errors
                .push("quality_gates.bootstrap_resamples must be >= 200".into());
        }
        for (k, t) in g.max_loss_pp.iter() {
            if t.investigate < t.target {
                v.errors.push(format!(
                    "quality_gates.max_loss_pp.{k}: investigate must be >= target"
                ));
            }
        }
        if !(g.ppl_ratio.target >= 1.0 && g.ppl_ratio.investigate >= g.ppl_ratio.target) {
            v.errors.push(
                "quality_gates.ppl_ratio: need 1 <= target <= investigate (a ratio of candidate to reference perplexity)".into(),
            );
        }
        if !(g.kld_topk.max_mean >= 0.0 && (0.0..=100.0).contains(&g.kld_topk.min_top1_pct)) {
            v.errors.push(
                "quality_gates.kld_topk: need max_mean >= 0 and min_top1_pct in 0..100".into(),
            );
        }
        for (k, t) in g.min_rate_pct.iter() {
            if t.investigate > t.target || t.target > 100.0 {
                v.errors.push(format!(
                    "quality_gates.min_rate_pct.{k}: need investigate <= target <= 100"
                ));
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub const SAMPLE: &str = r#"
experiment:
  id: smoke-27b-fast-vs-balanced
  seed: 7
reference:
  id: q27-balanced
  endpoint: http://127.0.0.1:8000
  api_key_env: TALUS_API_KEY
  talus: {model: qwen3.8-27b, level: balanced}
candidates:
  - id: q27-reference-numerics
    endpoint: http://127.0.0.1:8000/v1
    talus: {model: qwen3.8-27b, level: reference, apply: true}
    provenance: {quantization: UD-IQ4_XS, kv_cache_precision: f16}
benchmarks:
  tier: smoke
  tasks: [json_schema, gsm8k]
  limits: {gsm8k: 20}
"#;

    #[test]
    fn parses_and_validates_sample() {
        let c = Config::from_yaml(SAMPLE).unwrap();
        assert_eq!(c.experiment.seed, 7);
        assert_eq!(c.inference.temperature, 0.0);
        assert_eq!(c.inference.thinking, Thinking::Off);
        assert_eq!(c.quality_gates.max_loss_pp["gsm8k"].target, 2.0);
        assert_eq!(c.quality_gates.min_rate_pct["json_schema"].target, 99.5);
        assert_eq!(c.benchmarks.required(), vec!["json_schema", "gsm8k"]);
        assert_eq!(
            c.output_dir(),
            PathBuf::from("artifacts/smoke-27b-fast-vs-balanced")
        );
        let v = c.validate();
        assert!(v.ok(), "{v:?}");
        // round trip through YAML keeps everything
        let again = Config::from_yaml(&c.to_yaml()).unwrap();
        assert_eq!(again, c);
    }

    #[test]
    fn rejects_inline_keys_and_typos() {
        let bad = SAMPLE.replace("api_key_env: TALUS_API_KEY", "api_key: sk-123");
        let err = Config::from_yaml(&bad).unwrap_err();
        assert!(err.contains("api_key_env"), "{err}");
        let typo = SAMPLE.replace("seed: 7", "sed: 7");
        assert!(Config::from_yaml(&typo).is_err());
    }

    #[test]
    fn validation_errors() {
        let mut c = Config::from_yaml(SAMPLE).unwrap();
        c.candidates[0].id = "q27-balanced".into();
        c.inference.concurrency_for_quality = 4;
        c.benchmarks.tasks.push("mystery_bench".into());
        c.benchmarks.required = Some(vec!["needle".into()]);
        c.candidates[0].talus.as_mut().unwrap().level = Some("turbo".into());
        c.reference.endpoint = Some("https://x:1".into());
        let v = c.validate();
        let all = v.errors.join("\n");
        for needle in [
            "used twice",
            "one request at a time",
            "unknown task \"mystery_bench\"",
            "required test that is never run",
            "talus.level \"turbo\"",
            "https is not supported",
        ] {
            assert!(all.contains(needle), "missing {needle:?} in:\n{all}");
        }
    }

    #[test]
    fn planned_tasks_warn_not_error() {
        let mut c = Config::from_yaml(SAMPLE).unwrap();
        c.benchmarks.tasks.extend([
            "aime".into(),
            "kld".into(),
            "perplexity".into(),
            "kld_topk".into(),
        ]);
        let v = c.validate();
        assert!(v.ok(), "{v:?}");
        let w = v.warnings.join("\n");
        assert!(w.contains("aime") && w.contains("NOT_RUN"), "{w}");
        // the full-vocabulary KL stays NOT_SUPPORTED over HTTP and points at the HTTP screening task
        assert!(
            w.contains("task kld:") && w.contains("NOT_SUPPORTED") && w.contains("kld_topk"),
            "{w}"
        );
        // perplexity and kld_topk are native now (NOT_SUPPORTED only at run time, on a server without scoring)
        assert!(
            !w.contains("task perplexity") && !w.contains("task kld_topk"),
            "{w}"
        );
    }

    #[test]
    fn logprob_gate_defaults_and_validation() {
        let c = Config::from_yaml(SAMPLE).unwrap();
        assert_eq!(c.quality_gates.ppl_ratio.target, 1.01);
        assert!(!c.quality_gates.kld_topk.enforce);
        assert_eq!(c.quality_gates.kld_topk.max_mean, 0.05);
        let y = SAMPLE.to_string()
            + "quality_gates: {ppl_ratio: {target: 0.9, investigate: 1.0}, kld_topk: {max_mean: 0.01, enforce: true}}\n";
        let c = Config::from_yaml(&y).unwrap();
        assert!(c.quality_gates.kld_topk.enforce);
        assert_eq!(c.quality_gates.kld_topk.min_top1_pct, 90.0);
        assert!(c.validate().errors.join("\n").contains("ppl_ratio"));
        let typo =
            SAMPLE.to_string() + "quality_gates: {kld_topk: {max_mean: 0.01, enforced: true}}\n";
        assert!(Config::from_yaml(&typo).is_err());
    }

    #[test]
    fn frozen_reference_rules() {
        let mut c = Config::from_yaml(SAMPLE).unwrap();
        c.reference.kind = EndpointKind::Frozen;
        c.reference.frozen = None;
        c.candidates[0].kind = EndpointKind::Frozen;
        let all = c.validate().errors.join("\n");
        assert!(all.contains("frozen: <dir> is required"), "{all}");
        assert!(all.contains("only valid for the reference"), "{all}");
    }
}
