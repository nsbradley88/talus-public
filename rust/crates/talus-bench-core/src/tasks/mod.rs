//! Tasks: the adapter interface and the registry of every task id talus-bench knows.
//!
//! A *native* task builds its cases from a pinned dataset and scores outputs in Rust. Every other known id maps to
//! an explicit status instead of a substitute metric: planned native tasks are NOT_RUN, external Python
//! frameworks are NOT_SUPPORTED unless their adapter is implemented and the tool is installed. Numerical
//! fidelity over HTTP uses talus-server's teacher-forced scoring endpoint (`perplexity`, `kld_topk`; NOT_SUPPORTED
//! on a server or engine without it); the full-vocabulary KL (`kld`) needs logits HTTP does not carry and stays
//! NOT_SUPPORTED there (GPU-host mode `kld_teacher_forced`).

pub mod agentic;
pub mod bfcl;
pub mod external;
pub mod greedy;
pub mod gsm8k;
pub mod ifeval;
pub mod json_schema;
pub mod json_schema_strict;
pub mod livecodebench;
pub mod mmlu_pro;
pub mod needle;
pub mod textscore;

use crate::case::{Case, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::DatasetPin;
use crate::gates::GateStatus;
use crate::http::Client;
use serde_json::Value;
use std::path::Path;

/// Inputs a task needs to build its cases.
pub struct TaskContext<'a> {
    pub tier: Tier,
    pub limit: Option<usize>,
    pub cache: &'a Path,
    /// `benchmarks.params.<task>` from the config (null when absent).
    pub params: &'a Value,
    /// The endpoint under test, for tasks whose cases depend on the server (needle: tokenizer).
    pub client: Option<&'a Client>,
    /// The run's output directory (tasks that persist a per-run draw keep it there).
    pub run_dir: Option<&'a Path>,
    /// The run resumes an earlier one (reuse its draw instead of drawing again).
    pub resume: bool,
}

impl TaskContext<'_> {
    pub fn for_tests(tier: Tier) -> TaskContext<'static> {
        TaskContext {
            tier,
            limit: None,
            cache: Path::new("/nonexistent"),
            params: &Value::Null,
            client: None,
            run_dir: None,
            resume: false,
        }
    }
}

/// The adapter interface. Native tasks implement it directly; external adapters (lm-eval, Inspect,
/// LiveCodeBench) will implement it by running a subprocess and translating its per-sample log into `Case` /
/// `Score` rows (DESIGN.md "Adapters").
pub trait Task: Send + Sync {
    fn id(&self) -> &str;
    /// Bumped whenever scoring changes; pairing refuses to compare rows with different versions.
    fn scorer_version(&self) -> &'static str;
    /// The dataset pins used for this tier (empty when the cases are built from a runtime source; the case's
    /// `dataset_revision` then carries that source's hash).
    fn datasets(&self, tier: Tier) -> Vec<&'static DatasetPin>;
    /// Build the cases. Tasks that `needs_server` get `ctx.client` and are built per endpoint.
    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String>;
    fn score(&self, case: &Case, out: &RawOutput) -> Score;
    /// Cases depend on the endpoint (e.g. its tokenizer).
    fn needs_server(&self) -> bool {
        false
    }
    /// Case limit applied when the config sets none, per tier (None = every case).
    fn default_limit(&self, _tier: Tier) -> Option<usize> {
        None
    }
    /// Job adapters run each case themselves (e.g. an agent session) instead of one chat request.
    fn execute(&self, _case: &Case, _env: &ExecEnv) -> Option<ExecOutcome> {
        None
    }
    /// Batch adapters (external frameworks) run every case in one subprocess per endpoint and return the rows;
    /// `cases` is not used for them.
    fn is_batch(&self) -> bool {
        false
    }
    fn run_batch(&self, _env: &ExecEnv) -> Result<Vec<BatchRow>, String> {
        Err("not a batch task".into())
    }
    /// Run-specific provenance for the manifest (e.g. a random draw's seed and instance ids).
    fn provenance(&self) -> Value {
        Value::Null
    }
    /// Caveats for the report's "What cannot be concluded".
    fn caveats(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Prefix of a task error that means "this backend/host cannot run it" (NOT_SUPPORTED, not NOT_RUN).
pub const NOT_SUPPORTED: &str = "NOT_SUPPORTED:";

/// What a job or batch adapter needs to know about the endpoint under test.
#[derive(Clone, Debug)]
pub struct ExecEnv {
    /// `http://host:port`
    pub base_url: String,
    pub api_key: Option<String>,
    /// The model id the server reports in /v1/models (or the config's `model`).
    pub model: String,
    /// The served context length, when the server reports it.
    pub context: Option<u64>,
    /// A directory for this endpoint's job files (inside the run's output directory).
    pub work_dir: std::path::PathBuf,
    /// A label unique to the run and endpoint.
    pub label: String,
    pub params: Value,
    pub limit: Option<usize>,
    pub temperature: f64,
    pub seed: Option<u64>,
    pub max_tokens: u32,
    pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// One executed job case.
#[derive(Clone, Debug)]
pub struct ExecOutcome {
    pub status: crate::case::CaseStatus,
    pub raw: Option<RawOutput>,
    pub score: Score,
    pub error: Option<String>,
    pub wall_ms: f64,
}

/// One row produced by a batch adapter.
#[derive(Clone, Debug)]
pub struct BatchRow {
    pub case_id: String,
    pub prompt_hash: String,
    pub dataset: String,
    pub dataset_revision: String,
    pub scorer_version: String,
    pub expected: Value,
    pub status: crate::case::CaseStatus,
    pub raw: Option<RawOutput>,
    pub score: Score,
    pub epoch: Option<u64>,
}

/// Run a command to completion with a timeout and the cancel flag; returns (exit code, stdout, stderr tail).
pub fn run_command(
    cmd: &mut std::process::Command,
    timeout: std::time::Duration,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<(i32, String, String), String> {
    use std::io::Read;
    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start {:?}: {e}", cmd.get_program()))?;
    let mut out = child.stdout.take().expect("piped");
    let mut err = child.stderr.take().expect("piped");
    let to = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = out.read_to_string(&mut s);
        s
    });
    let te = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = err.read_to_string(&mut s);
        s
    });
    let t0 = std::time::Instant::now();
    let code = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st.code().unwrap_or(-1),
            Ok(None) => {}
            Err(e) => return Err(e.to_string()),
        }
        if cancel.load(std::sync::atomic::Ordering::Relaxed) || t0.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(if t0.elapsed() > timeout {
                format!("timed out after {} s", timeout.as_secs())
            } else {
                "cancelled".into()
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    let stdout = to.join().unwrap_or_default();
    let stderr = te.join().unwrap_or_default();
    let tail: String = stderr
        .chars()
        .rev()
        .take(2000)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    Ok((code, stdout, tail))
}

/// `name` on PATH (or an absolute/relative path that exists).
pub fn which(name: &str) -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(name);
    if name.contains('/') {
        return p.is_file().then(|| p.to_path_buf());
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join(name))
            .find(|c| c.is_file())
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskAvailability {
    Native,
    /// Native implementation planned for the named release.
    Planned(&'static str),
    /// A wrapper around an external tool (subprocess), planned for the named release.
    External(&'static str),
    /// Needs full-vocabulary logits, which talus-server does not return over HTTP (it returns top-k lists).
    NeedsLogprobs,
    /// Needs engine access on the GPU host (talus-cli); not reachable from an HTTP run.
    GpuHostOnly,
    /// A performance measurement: run by the perf tier, never as a quality task.
    Perf,
    /// An adapter around an external tool; it runs when the tool is installed on the machine running the bench,
    /// else the task reports NOT_SUPPORTED with what is missing.
    Adapter(&'static str),
    Unknown,
}

impl TaskAvailability {
    pub fn status(self) -> GateStatus {
        match self {
            TaskAvailability::Native | TaskAvailability::Adapter(_) => GateStatus::Pass,
            TaskAvailability::Planned(_) | TaskAvailability::Perf => GateStatus::NotRun,
            TaskAvailability::External(_)
            | TaskAvailability::NeedsLogprobs
            | TaskAvailability::GpuHostOnly
            | TaskAvailability::Unknown => GateStatus::NotSupported,
        }
    }

    pub fn reason(self, id: &str) -> String {
        match self {
            TaskAvailability::Native => "native".into(),
            TaskAvailability::Adapter(tool) => format!("runs {tool}; NOT_SUPPORTED where it is not installed"),
            TaskAvailability::Planned(v) => format!("not implemented yet (native task planned for {v})"),
            TaskAvailability::External(v) => format!(
                "external adapter for {id} is not implemented in this build (planned {v}; runs as a subprocess when the tool is installed)"
            ),
            TaskAvailability::NeedsLogprobs => "needs full-vocabulary logits, which talus-server does not return over HTTP (only top-k lists); use kld_topk for an HTTP screening KL, or the GPU-host mode (kld_teacher_forced) for the full-vocabulary release gate".into(),
            TaskAvailability::GpuHostOnly => "needs engine access on the GPU host (talus-cli); run the GPU-host mode".into(),
            TaskAvailability::Perf => "performance cell: measured by the perf tier, separately from quality".into(),
            TaskAvailability::Unknown => format!("unknown task {id}"),
        }
    }
}

/// Every task id the bench knows, with its availability in this build and a one-line description.
pub const KNOWN: &[(&str, TaskAvailability, &str)] = &[
    (
        "swebench_verified",
        TaskAvailability::Adapter("docker + the SWE-bench Verified images"),
        "SWE-bench Verified: fresh stratified draw per run, agent in a no-network container, hidden tests in a fresh offline container, anti-cheating audit",
    ),
    (
        "swebench_live",
        TaskAvailability::Planned("later"),
        "contamination-resistant companion set (SWE-rebench / SWE-bench-Live tasks created after the model's cutoff), reported separately",
    ),
    (
        "json_schema",
        TaskAvailability::Native,
        "structured output: parse, schema compliance, semantic checks, extra fields",
    ),
    (
        "json_schema_strict",
        TaskAvailability::Native,
        "json_schema cases with response_format json_schema (grammar-constrained, server extension E6; informational, not a gate)",
    ),
    (
        "gsm8k",
        TaskAvailability::Native,
        "grade-school math, exact numeric answer",
    ),
    (
        "ifeval",
        TaskAvailability::Native,
        "IFEval strict prompt-level accuracy on the 20 of 25 instruction types with reimplementable rules (Apache-2.0)",
    ),
    (
        "bfcl",
        TaskAvailability::Native,
        "BFCL v3 simple/multiple/parallel/irrelevance via native tool calls, AST checker reimplemented (Apache-2.0)",
    ),
    (
        "mmlu_pro",
        TaskAvailability::Native,
        "MMLU-Pro official 5-shot CoT prompt and answer extraction (dataset MIT)",
    ),
    (
        "gpqa_diamond",
        TaskAvailability::Planned("v1.1"),
        "GPQA Diamond (CC-BY-4.0, gated; fetched only, never committed)",
    ),
    (
        "aime",
        TaskAvailability::Planned("v1.1"),
        "AIME pinned year, final-answer normalization, repeated trials",
    ),
    (
        "needle",
        TaskAvailability::Native,
        "needle gate as in METHODOLOGY.md: 5 codenames at 10-90% depth of an N-token corpus prompt, per seed",
    ),
    (
        "greedy_identity",
        TaskAvailability::Native,
        "greedy byte-identity of the bench27 code and edit mixes against the reference",
    ),
    (
        "multi_turn",
        TaskAvailability::Planned("v1.1"),
        "multi-turn constraint retention and stale tool output recovery",
    ),
    (
        "agentic_go",
        TaskAvailability::Adapter("python3, go, opencode, bwrap + bench/agentic"),
        "agentic Go suite: opencode on bubbles v0.21.1, hidden go tests, one case per task",
    ),
    (
        "agent_k8s",
        TaskAvailability::Planned("later"),
        "mocked Kubernetes diagnosis (sandboxed kind/kwok or recorded fixtures only)",
    ),
    (
        "agent_router",
        TaskAvailability::Planned("later"),
        "router/BGP config generation checked by an offline parser",
    ),
    (
        "lm_eval",
        TaskAvailability::Adapter("lm-eval (lm-evaluation-harness)"),
        "EleutherAI lm-evaluation-harness task via local-chat-completions, e.g. lm_eval:gsm8k",
    ),
    (
        "inspect",
        TaskAvailability::Adapter("inspect (Inspect AI) + openai"),
        "Inspect AI task via the openai-api provider, e.g. inspect:inspect_evals/gsm8k",
    ),
    (
        "livecodebench",
        TaskAvailability::Adapter("docker + the pinned Python sandbox image (talus-bench lcb-prep)"),
        "LiveCodeBench code generation, release v6 pinned, pass@1 (shipped sampling or greedy), code run in an offline, resource-limited container",
    ),
    (
        "kld",
        TaskAvailability::NeedsLogprobs,
        "full-vocabulary token KL vs the reference (not over HTTP: see kld_topk, kld_teacher_forced)",
    ),
    (
        "kld_topk",
        TaskAvailability::Native,
        "screening KL(reference || candidate) from top-k lists of teacher-forced scoring (POST /v1/talus/score), tail mass recorded; WARN-only by default",
    ),
    (
        "perplexity",
        TaskAvailability::Native,
        "perplexity of pinned public-domain prose, teacher-forced via POST /v1/talus/score; gate on the PPL ratio vs the reference",
    ),
    (
        "kld_teacher_forced",
        TaskAvailability::GpuHostOnly,
        "teacher-forced full-vocab KL via talus-cli (release gate)",
    ),
    (
        "perf",
        TaskAvailability::Perf,
        "TTFT, prefill at depth, decode 2K/32K (perf tier)",
    ),
];

pub fn availability(id: &str) -> TaskAvailability {
    let base = id.split(':').next().unwrap_or(id);
    KNOWN
        .iter()
        .find(|k| k.0 == base)
        .map(|k| k.1)
        .unwrap_or(TaskAvailability::Unknown)
}

/// The native implementation of `id`, if this build has one.
pub fn native(id: &str) -> Option<Box<dyn Task>> {
    match id {
        "json_schema" => Some(Box::new(json_schema::JsonSchemaTask)),
        "json_schema_strict" => Some(Box::new(json_schema_strict::JsonSchemaStrictTask)),
        "gsm8k" => Some(Box::new(gsm8k::Gsm8kTask)),
        "ifeval" => Some(Box::new(ifeval::IfevalTask)),
        "bfcl" => Some(Box::new(bfcl::BfclTask)),
        "mmlu_pro" => Some(Box::new(mmlu_pro::MmluProTask)),
        "needle" => Some(Box::new(needle::NeedleTask)),
        "greedy_identity" => Some(Box::new(greedy::GreedyTask)),
        "perplexity" => Some(Box::new(textscore::PerplexityTask)),
        "kld_topk" => Some(Box::new(textscore::KldTopkTask)),
        "agentic_go" => Some(Box::new(agentic::AgenticTask)),
        "livecodebench" => Some(Box::new(livecodebench::LcbTask::default())),
        "swebench_verified" => Some(Box::new(crate::swe::SwebenchTask::default())),
        _ if id.starts_with("lm_eval:") => Some(Box::new(external::ExternalTask::lm_eval(id))),
        _ if id.starts_with("inspect:") => Some(Box::new(external::ExternalTask::inspect(id))),
        _ => None,
    }
}

/// Apply a per-task limit (first N in dataset order; never invents cases).
pub fn take_limit(mut cases: Vec<Case>, limit: Option<usize>) -> Vec<Case> {
    if let Some(n) = limit {
        cases.truncate(n);
    }
    cases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_consistent() {
        for (id, a, _) in KNOWN {
            let runs = matches!(a, TaskAvailability::Native | TaskAvailability::Adapter(_));
            if !id.contains(':') && *id != "lm_eval" && *id != "inspect" {
                assert_eq!(native(id).is_some(), runs, "{id}");
            }
            assert_eq!(availability(id), *a);
        }
        assert!(matches!(
            availability("lm_eval:mmlu"),
            TaskAvailability::Adapter(_)
        ));
        assert!(
            native("lm_eval:gsm8k").is_some() && native("inspect:inspect_evals/gsm8k").is_some()
        );
        assert!(matches!(
            availability("livecodebench"),
            TaskAvailability::Adapter(_)
        ));
        assert_eq!(availability("nope"), TaskAvailability::Unknown);
        assert_eq!(availability("kld").status(), GateStatus::NotSupported);
        assert!(availability("kld").reason("kld").contains("kld_topk"));
        assert_eq!(availability("perplexity"), TaskAvailability::Native);
        assert_eq!(availability("kld_topk"), TaskAvailability::Native);
        assert!(native("perplexity").unwrap().needs_server());
        assert_eq!(availability("aime").status(), GateStatus::NotRun);
    }
}
