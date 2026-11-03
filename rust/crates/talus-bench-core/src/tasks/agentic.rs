//! `agentic_go`: the agentic Go suite of docs/METHODOLOGY.md as a job adapter. Each task of
//! `bench/agentic/tasks.py` is one case: the adapter copies the suite into the run's job directory, writes an
//! opencode provider config that points at the endpoint under test, runs `run_agentic.py LABEL talus/MODEL TASK`
//! (opencode edits a fresh bubbles v0.21.1 checkout, then the hidden `go test` grades it, offline, under bwrap)
//! and reads the JSON record the script prints. The case passes when the record says `pass`.
//!
//! Requirements on the machine running talus-bench: python3, go (with bubbles v0.21.1 in the module cache),
//! opencode, bwrap (unless `mask: false`), and the suite directory. Missing pieces make the task
//! NOT_SUPPORTED with the list.
//!
//! Params (`benchmarks.params.agentic_go`): `dir` (the suite; default `$TALUS_AGENTIC_DIR`, `./bench/agentic`, or
//! the source tree this binary was built from), `tasks` (subset), `timeout_s` (per task, default 3600),
//! `autocompact` (default: on when the served context is <= 64K, as the methodology does), `mask` (default
//! true), `check_tools` (default true; tests turn it off), `python` (default python3).

use super::{run_command, which, ExecEnv, ExecOutcome, Task, TaskContext, NOT_SUPPORTED};
use crate::case::{Case, CaseStatus, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::DatasetPin;
use crate::util::sha256_hex;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub struct AgenticTask;

fn suite_dir(params: &Value) -> PathBuf {
    if let Some(d) = params["dir"].as_str() {
        return PathBuf::from(d);
    }
    if let Ok(d) = std::env::var("TALUS_AGENTIC_DIR") {
        if !d.is_empty() {
            return PathBuf::from(d);
        }
    }
    let local = PathBuf::from("bench/agentic");
    if local.join("run_agentic.py").is_file() {
        return local;
    }
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../bench/agentic"
    ))
}

/// Task names in tasks.py order (`    "name": dict(` entries of TASKS).
pub fn task_names(tasks_py: &str) -> Vec<String> {
    let re = regex::Regex::new(r#"(?m)^    "([A-Za-z0-9_-]+)": dict\("#).expect("re");
    re.captures_iter(tasks_py)
        .map(|c| c[1].to_string())
        .collect()
}

/// What is missing to run the suite here.
fn missing(dir: &Path, params: &Value) -> Vec<String> {
    let mut m = Vec::new();
    for f in ["run_agentic.py", "tasks.py", "opencode-test.json"] {
        if !dir.join(f).is_file() {
            m.push(format!("{}", dir.join(f).display()));
        }
    }
    if params["check_tools"].as_bool() != Some(false) {
        let python = params["python"].as_str().unwrap_or("python3");
        let mut tools = vec![python, "go", "opencode"];
        if params["mask"].as_bool() != Some(false) {
            tools.push("bwrap");
        }
        for t in tools {
            if which(t).is_none() {
                m.push(t.to_string());
            }
        }
    }
    m
}

/// The opencode provider config for the endpoint under test (the suite's permissions are kept).
pub fn opencode_config(original: &Value, env: &ExecEnv) -> Value {
    let mut cfg = original.clone();
    let ctx = env.context.unwrap_or(65_536);
    cfg["provider"] = json!({
        "talus": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "talus-bench endpoint under test",
            "options": {"baseURL": format!("{}/v1", env.base_url), "apiKey": "{env:STRATA_API_KEY}"},
            "models": {env.model.clone(): {"name": env.model, "limit": {"context": ctx, "output": 16384}}}
        }
    });
    cfg
}

impl Task for AgenticTask {
    fn id(&self) -> &str {
        "agentic_go"
    }

    fn scorer_version(&self) -> &'static str {
        "agentic_go/1"
    }

    fn datasets(&self, _tier: Tier) -> Vec<&'static DatasetPin> {
        Vec::new()
    }

    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String> {
        let dir = suite_dir(ctx.params);
        let miss = missing(&dir, ctx.params);
        if !miss.is_empty() {
            return Err(format!(
                "{NOT_SUPPORTED} agentic_go needs {} on this machine",
                miss.join(", ")
            ));
        }
        let tasks_py = std::fs::read_to_string(dir.join("tasks.py")).map_err(|e| e.to_string())?;
        let runner =
            std::fs::read_to_string(dir.join("run_agentic.py")).map_err(|e| e.to_string())?;
        let rev = sha256_hex(format!("{runner}\u{0}{tasks_py}").as_bytes());
        let mut names = task_names(&tasks_py);
        if let Some(want) = ctx.params["tasks"].as_array() {
            let want: Vec<String> = want
                .iter()
                .filter_map(|t| t.as_str().map(str::to_string))
                .collect();
            for w in &want {
                if !names.contains(w) {
                    return Err(format!(
                        "agentic_go: no task {w:?} in tasks.py ({})",
                        names.join(", ")
                    ));
                }
            }
            names.retain(|n| want.contains(n));
        }
        if names.is_empty() {
            return Err("agentic_go: tasks.py defines no tasks".into());
        }
        Ok(super::take_limit(
            names
                .into_iter()
                .map(|n| Case {
                    task: self.id().into(),
                    case_id: format!("agentic_go/{n}"),
                    dataset: "bubbles-v0.21.1".into(),
                    dataset_revision: format!("agentic@{}", &rev[..12]),
                    messages: Vec::new(),
                    expected: json!({"task": n}),
                    meta: json!({"dir": dir.display().to_string()}),
                    critical: false,
                    max_tokens: None,
                    tools: Value::Null,
                    request: json!({"agentic_task": n}),
                })
                .collect(),
            ctx.limit,
        ))
    }

    fn score(&self, _case: &Case, _out: &RawOutput) -> Score {
        Score::default() // execute() scores
    }

    fn execute(&self, case: &Case, env: &ExecEnv) -> Option<ExecOutcome> {
        let t0 = Instant::now();
        let fail = |status: CaseStatus, e: String| ExecOutcome {
            status,
            raw: None,
            score: Score::binary(false, Value::Null, json!({"stage": "harness"})),
            error: Some(e),
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
        };
        let task = case.expected["task"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let src = PathBuf::from(case.meta["dir"].as_str().unwrap_or_default());
        // a private copy of the suite: results/ and the provider config live there, never in the source tree
        let job = env.work_dir.join("agentic_go");
        let suite = job.join("agentic");
        if let Err(e) = std::fs::create_dir_all(&suite) {
            return Some(fail(
                CaseStatus::ProtocolError,
                format!("{}: {e}", suite.display()),
            ));
        }
        for f in ["run_agentic.py", "tasks.py", "opencode-test.json"] {
            if let Err(e) = std::fs::copy(src.join(f), suite.join(f)) {
                return Some(fail(CaseStatus::ProtocolError, format!("copy {f}: {e}")));
            }
        }
        let original: Value = std::fs::read_to_string(src.join("opencode-test.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(json!({}));
        let cfg = opencode_config(&original, env);
        if let Err(e) = std::fs::write(
            suite.join("opencode-test.json"),
            serde_json::to_string_pretty(&cfg).unwrap_or_default(),
        ) {
            return Some(fail(
                CaseStatus::ProtocolError,
                format!("opencode config: {e}"),
            ));
        }
        let timeout_s = env.params["timeout_s"].as_u64().unwrap_or(3600);
        let autocompact = env.params["autocompact"]
            .as_bool()
            .unwrap_or(env.context.is_some_and(|c| c <= 65_536));
        let python = env.params["python"].as_str().unwrap_or("python3");
        let mut cmd = std::process::Command::new(python);
        cmd.arg(suite.join("run_agentic.py"))
            .arg(&env.label)
            .arg(format!("talus/{}", env.model))
            .arg(&task)
            .current_dir(&job)
            .env("AGENTIC_TIMEOUT", timeout_s.to_string())
            .env(
                "STRATA_API_KEY",
                env.api_key.clone().unwrap_or_else(|| "none".into()),
            );
        if autocompact {
            cmd.env("AGENTIC_AUTOCOMPACT", "1");
        }
        if env.params["mask"].as_bool() == Some(false) {
            cmd.env("AGENTIC_MASK", "0");
        }
        // the script's own timeout plus grading; a hung run is killed after that
        let budget = Duration::from_secs(timeout_s + 900);
        let (code, stdout, stderr) = match run_command(&mut cmd, budget, &env.cancel) {
            Ok(x) => x,
            Err(e) if e.starts_with("timed out") => return Some(fail(CaseStatus::Timeout, e)),
            Err(e) => return Some(fail(CaseStatus::ProtocolError, e)),
        };
        let rec = stdout
            .lines()
            .rev()
            .filter_map(|l| serde_json::from_str::<Value>(l.trim()).ok())
            .find(|v| v["task"].as_str() == Some(task.as_str()));
        let Some(rec) = rec else {
            return Some(fail(
                CaseStatus::ProtocolError,
                format!(
                    "run_agentic.py exit {code}, no record for {task}: {}",
                    stderr.trim()
                ),
            ));
        };
        let pass = rec["pass"].as_bool().unwrap_or(false);
        let clean = rec["clean"].as_bool().unwrap_or(true);
        let stage = if pass {
            "pass"
        } else if rec["agent_rc"] == "timeout" {
            "agent_timeout"
        } else if rec["tests_touched"]
            .as_array()
            .is_some_and(|a| !a.is_empty())
        {
            "touched_tests"
        } else {
            "hidden_tests_failed"
        };
        let mut details = rec.clone();
        details["stage"] = json!(stage);
        details["audit_clean"] = json!(clean);
        Some(ExecOutcome {
            status: CaseStatus::Ok,
            raw: Some(RawOutput {
                content: format!(
                    "{}\n{}",
                    rec["diff"].as_str().unwrap_or_default(),
                    rec["fail_tail"].as_str().unwrap_or_default()
                ),
                finish_reason: "job".into(),
                ..Default::default()
            }),
            score: Score::binary(pass, Value::Null, details),
            error: None,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_names_from_the_repo_suite() {
        let t = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../bench/agentic/tasks.py"
        ))
        .unwrap();
        assert_eq!(
            task_names(&t),
            vec![
                "progress-downward",
                "paginator-broken",
                "viewport-impl",
                "viewport-ensurevisible",
                "paginator-wrap"
            ]
        );
    }

    #[test]
    fn missing_tools_mean_not_supported() {
        let params = json!({"dir": "/nonexistent/agentic", "check_tools": true});
        let e = AgenticTask
            .cases(&TaskContext {
                params: &params,
                ..TaskContext::for_tests(Tier::Full)
            })
            .unwrap_err();
        assert!(e.starts_with(NOT_SUPPORTED), "{e}");
        assert!(e.contains("run_agentic.py"));
    }

    #[test]
    fn provider_config_points_at_the_endpoint() {
        let env = ExecEnv {
            base_url: "http://127.0.0.1:8100".into(),
            api_key: None,
            model: "qwen3.8-27b".into(),
            context: Some(65536),
            work_dir: PathBuf::from("/tmp"),
            label: "l".into(),
            params: Value::Null,
            limit: None,
            temperature: 0.0,
            seed: None,
            max_tokens: 1,
            cancel: Default::default(),
        };
        let c = opencode_config(
            &json!({"permission": {"edit": "allow"}, "provider": {"glm": {}}}),
            &env,
        );
        assert_eq!(
            c["provider"]["talus"]["options"]["baseURL"],
            "http://127.0.0.1:8100/v1"
        );
        assert_eq!(
            c["provider"]["talus"]["models"]["qwen3.8-27b"]["limit"]["context"],
            65536
        );
        assert!(c["provider"].get("glm").is_none());
        assert_eq!(c["permission"]["edit"], "allow");
    }
}
