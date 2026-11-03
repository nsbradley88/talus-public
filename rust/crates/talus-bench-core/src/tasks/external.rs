//! External framework adapters, run as subprocesses when the tool is installed (never presented as native):
//!
//! - `lm_eval:<task>`: EleutherAI lm-evaluation-harness (MIT), verified with lm-eval 0.4.13:
//!   `lm-eval run --model local-chat-completions --model_args model=M,base_url=URL/v1/chat/completions,
//!   num_concurrent=1,max_retries=3,tokenized_requests=False --tasks T --apply_chat_template --log_samples
//!   --output_path DIR [--limit N] [--seed S] [--gen_kwargs temperature=T]`, key in `OPENAI_API_KEY`. Rows come
//!   from `samples_<task>_*.jsonl` (one line per document and filter): `doc_id`, lm-eval's own `prompt_hash`, the
//!   metric value of the chosen filter (params `filter`, default the task's first; `metric`, default the first);
//!   the dataset revision is lm-eval's `task_hashes` entry. Only generate-until tasks work: log-likelihood tasks
//!   need prompt logprobs (`echo`) on the completions API, which talus-server does not return (its opt-in
//!   logprobs cover generated tokens only); they fail inside lm-eval and the task reports NOT_RUN. Teacher-forced
//!   scoring of fixed text is the native `perplexity` / `kld_topk` tasks (POST /v1/talus/score).
//! - `inspect:<task>`: Inspect AI (MIT), verified with inspect_ai 0.3.285 + inspect_evals 0.26.0: `inspect eval
//!   TASK --model openai-api/talus/M --log-format json --log-dir DIR --display plain --max-connections 1
//!   [--limit N] [--seed S] [--temperature T]` with `TALUS_BASE_URL` / `TALUS_API_KEY`; needs the `openai` Python
//!   package. Rows come from the JSON log's `samples`: `id`, `epoch`, the first scorer's value (C = 1, P = 0.5,
//!   I / N = 0, or a number); the revision is the task version and package versions.
//!
//! Both pass the run's temperature and seed so reference and candidate get identical commands; thinking mode
//! follows the server's template default (neither framework sends chat_template_kwargs). Params
//! (`benchmarks.params."lm_eval:gsm8k"` etc.): `bin` (default `lm-eval` / `inspect` on PATH), `extra_args`
//! (list appended to the command), `timeout_s` (default 14400), `filter`, `metric`.

use super::{run_command, which, BatchRow, ExecEnv, Task, TaskContext, NOT_SUPPORTED};
use crate::case::{Case, CaseStatus, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::DatasetPin;
use crate::util::hash_json;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Framework {
    LmEval,
    Inspect,
}

pub struct ExternalTask {
    pub framework: Framework,
    /// The framework's task name (after the `lm_eval:` / `inspect:` prefix).
    pub task: String,
    /// The configured id, e.g. `lm_eval:gsm8k`.
    pub full_id: String,
}

impl ExternalTask {
    pub fn lm_eval(id: &str) -> Self {
        Self {
            framework: Framework::LmEval,
            task: id.trim_start_matches("lm_eval:").to_string(),
            full_id: id.to_string(),
        }
    }

    pub fn inspect(id: &str) -> Self {
        Self {
            framework: Framework::Inspect,
            task: id.trim_start_matches("inspect:").to_string(),
            full_id: id.to_string(),
        }
    }

    fn prefix(&self) -> &'static str {
        match self.framework {
            Framework::LmEval => "lm_eval",
            Framework::Inspect => "inspect",
        }
    }

    fn default_bin(&self) -> &'static str {
        match self.framework {
            Framework::LmEval => "lm-eval",
            Framework::Inspect => "inspect",
        }
    }

    /// The command for this endpoint (exposed for tests and the manifest).
    pub fn command(&self, env: &ExecEnv, out: &Path) -> Result<std::process::Command, String> {
        let bin_name = env.params["bin"].as_str().unwrap_or(self.default_bin());
        let bin = which(bin_name).ok_or_else(|| {
            format!(
                "{NOT_SUPPORTED} {} is not installed here ({bin_name} not found)",
                self.prefix()
            )
        })?;
        let mut c = std::process::Command::new(bin);
        let key = env.api_key.clone().unwrap_or_else(|| "none".into());
        match self.framework {
            Framework::LmEval => {
                c.arg("run")
                    .args(["--model", "local-chat-completions", "--model_args"])
                    .arg(format!(
                        "model={},base_url={}/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False",
                        env.model, env.base_url
                    ))
                    .args(["--tasks", &self.task, "--apply_chat_template", "--log_samples", "--output_path"])
                    .arg(out)
                    .args(["--gen_kwargs", &format!("temperature={}", env.temperature)])
                    .env("OPENAI_API_KEY", key);
                if let Some(n) = env.limit {
                    c.args(["--limit", &n.to_string()]);
                }
                if let Some(s) = env.seed {
                    c.args(["--seed", &s.to_string()]);
                }
            }
            Framework::Inspect => {
                c.args(["eval", &self.task, "--model"])
                    .arg(format!("openai-api/talus/{}", env.model))
                    .args(["--log-format", "json", "--log-dir"])
                    .arg(out)
                    .args(["--display", "plain", "--max-connections", "1"])
                    .args(["--temperature", &env.temperature.to_string()])
                    .env("TALUS_BASE_URL", format!("{}/v1", env.base_url))
                    .env("TALUS_API_KEY", key);
                if let Some(n) = env.limit {
                    c.args(["--limit", &n.to_string()]);
                }
                if let Some(s) = env.seed {
                    c.args(["--seed", &s.to_string()]);
                }
            }
        }
        for a in env.params["extra_args"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            c.arg(a);
        }
        Ok(c)
    }
}

fn read_dir_files(dir: &Path, pred: &dyn Fn(&str) -> bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if pred(&e.file_name().to_string_lossy()) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// lm-eval output directory -> rows.
pub fn parse_lm_eval(out: &Path, params: &Value) -> Result<Vec<BatchRow>, String> {
    let results = read_dir_files(out, &|n| n.starts_with("results_") && n.ends_with(".json"));
    let res: Value = results
        .last()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str(&t).ok())
        .ok_or("lm-eval wrote no results_*.json")?;
    let version = res["lm_eval_version"].as_str().unwrap_or("?").to_string();
    let mut rows = Vec::new();
    for f in read_dir_files(out, &|n| n.starts_with("samples_") && n.ends_with(".jsonl")) {
        let name = f
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        // samples_<task>_<timestamp>.jsonl
        let task = name
            .trim_start_matches("samples_")
            .rsplit_once('_')
            .map(|x| x.0)
            .unwrap_or("")
            .to_string();
        let text = std::fs::read_to_string(&f).map_err(|e| e.to_string())?;
        let lines: Vec<Value> = text
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        let filter = params["filter"]
            .as_str()
            .map(str::to_string)
            .or_else(|| {
                lines
                    .first()
                    .and_then(|l| l["filter"].as_str())
                    .map(str::to_string)
            })
            .unwrap_or_default();
        let thash = res["task_hashes"][&task]
            .as_str()
            .unwrap_or("")
            .chars()
            .take(12)
            .collect::<String>();
        for l in lines
            .iter()
            .filter(|l| l["filter"].as_str().unwrap_or("") == filter)
        {
            let metric = params["metric"]
                .as_str()
                .map(str::to_string)
                .or_else(|| l["metrics"][0].as_str().map(str::to_string))
                .unwrap_or_else(|| "acc".into());
            let value = l[&metric].as_f64().unwrap_or(0.0);
            let resp = l["resps"][0][0].as_str().unwrap_or_default().to_string();
            rows.push(BatchRow {
                case_id: format!("lm_eval:{task}/{}", l["doc_id"]),
                prompt_hash: l["prompt_hash"].as_str().unwrap_or_default().to_string(),
                dataset: format!("lm_eval:{task}"),
                dataset_revision: format!("lm-eval {version} task {thash}"),
                scorer_version: format!("lm_eval@{version}/{task}/{filter}/{metric}"),
                expected: json!({"target": l["target"]}),
                status: CaseStatus::Ok,
                raw: Some(RawOutput { content: resp, finish_reason: "external".into(), ..Default::default() }),
                score: Score {
                    pass: value >= 1.0 - 1e-9,
                    value,
                    parsed: l["filtered_resps"].clone(),
                    details: json!({"stage": if value >= 1.0 - 1e-9 { "pass" } else { "wrong_answer" },
                                    "filter": filter, "metric": metric, "doc_hash": l["doc_hash"]}),
                },
                epoch: None,
            });
        }
    }
    if rows.is_empty() {
        return Err(
            "lm-eval wrote no samples (log-likelihood task? talus-server returns no prompt logprobs)"
                .into(),
        );
    }
    Ok(rows)
}

fn inspect_value(v: &Value) -> f64 {
    match v {
        Value::String(s) => match s.as_str() {
            "C" => 1.0,
            "P" => 0.5,
            _ => 0.0,
        },
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::Bool(b) => *b as u8 as f64,
        _ => 0.0,
    }
}

/// Inspect JSON log directory -> rows.
pub fn parse_inspect(out: &Path) -> Result<Vec<BatchRow>, String> {
    let logs = read_dir_files(out, &|n| n.ends_with(".json"));
    let log: Value = logs
        .last()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str(&t).ok())
        .ok_or("inspect wrote no JSON log")?;
    if log["status"].as_str() != Some("success") {
        return Err(format!(
            "inspect eval status {}: {}",
            log["status"], log["error"]["message"]
        ));
    }
    let e = &log["eval"];
    let task = e["task"].as_str().unwrap_or("?");
    let rev = format!(
        "{} v{} {}",
        e["dataset"]["name"].as_str().unwrap_or("?"),
        e["task_version"],
        e["packages"]
            .as_object()
            .map(|p| p
                .iter()
                .map(|(k, v)| format!("{k} {}", v.as_str().unwrap_or("?")))
                .collect::<Vec<_>>()
                .join(", "))
            .unwrap_or_default()
    );
    let mut rows = Vec::new();
    for s in log["samples"].as_array().into_iter().flatten() {
        let (scorer, sc) = s["scores"]
            .as_object()
            .and_then(|m| m.iter().next())
            .map(|(k, v)| (k.clone(), v.clone()))
            .unwrap_or_default();
        let value = inspect_value(&sc["value"]);
        let epoch = s["epoch"].as_u64();
        rows.push(BatchRow {
            case_id: format!("inspect:{task}/{}", s["id"].as_str().map(str::to_string).unwrap_or_else(|| s["id"].to_string())),
            prompt_hash: hash_json(&s["input"]),
            dataset: format!("inspect:{task}"),
            dataset_revision: rev.clone(),
            scorer_version: format!("inspect/{task}/{scorer}"),
            expected: json!({"target": s["target"]}),
            status: if s["error"].is_null() { CaseStatus::Ok } else { CaseStatus::ServerError },
            raw: Some(RawOutput {
                content: s["output"]["completion"].as_str().unwrap_or_default().to_string(),
                finish_reason: "external".into(),
                ..Default::default()
            }),
            score: Score {
                pass: value >= 1.0 - 1e-9,
                value,
                parsed: sc["answer"].clone(),
                details: json!({"stage": if value >= 1.0 - 1e-9 { "pass" } else { "wrong_answer" }, "scorer": scorer, "value": sc["value"]}),
            },
            epoch,
        });
    }
    if rows.is_empty() {
        return Err("inspect log has no samples".into());
    }
    Ok(rows)
}

impl Task for ExternalTask {
    fn id(&self) -> &str {
        &self.full_id
    }

    fn scorer_version(&self) -> &'static str {
        "external"
    }

    fn datasets(&self, _tier: Tier) -> Vec<&'static DatasetPin> {
        Vec::new()
    }

    fn cases(&self, _ctx: &TaskContext) -> Result<Vec<Case>, String> {
        Ok(Vec::new())
    }

    fn score(&self, _case: &Case, _out: &RawOutput) -> Score {
        Score::default()
    }

    fn is_batch(&self) -> bool {
        true
    }

    fn run_batch(&self, env: &ExecEnv) -> Result<Vec<BatchRow>, String> {
        let out = env.work_dir.join(format!(
            "{}-{}",
            self.prefix(),
            self.task.replace(['/', ':'], "_")
        ));
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
        let mut cmd = self.command(env, &out)?;
        let timeout = Duration::from_secs(env.params["timeout_s"].as_u64().unwrap_or(14_400));
        let (code, _stdout, stderr) = run_command(&mut cmd, timeout, &env.cancel)?;
        if code != 0 {
            return Err(format!(
                "{} exited {code}: {}",
                self.prefix(),
                stderr.trim()
            ));
        }
        match self.framework {
            Framework::LmEval => parse_lm_eval(&out, &env.params),
            Framework::Inspect => parse_inspect(&out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lm_eval_samples() {
        let d = std::env::temp_dir().join(format!("tb-lmeval-{}", std::process::id()));
        let m = d.join("model");
        std::fs::create_dir_all(&m).unwrap();
        std::fs::write(
            m.join("results_2026.json"),
            r#"{"lm_eval_version": "0.4.13", "task_hashes": {"gsm8k": "b59a0cf647dd6ed8"}}"#,
        )
        .unwrap();
        let line = |id: u32, filter: &str, em: f64| {
            json!({"doc_id": id, "target": "#### 18", "resps": [["#### 18"]], "filtered_resps": ["18"], "filter": filter,
                   "metrics": ["exact_match"], "doc_hash": "d", "prompt_hash": format!("p{id}"), "exact_match": em})
            .to_string()
        };
        let body = [
            line(0, "strict-match", 1.0),
            line(0, "flexible-extract", 1.0),
            line(1, "strict-match", 0.0),
            line(1, "flexible-extract", 1.0),
        ]
        .join("\n");
        std::fs::write(m.join("samples_gsm8k_2026-10-26T14-27-01.jsonl"), body).unwrap();
        let rows = parse_lm_eval(&d, &Value::Null).unwrap();
        assert_eq!(rows.len(), 2, "first filter (strict-match) only");
        assert_eq!(rows[0].case_id, "lm_eval:gsm8k/0");
        assert!(rows[0].score.pass && !rows[1].score.pass);
        assert_eq!(
            rows[1].scorer_version,
            "lm_eval@0.4.13/gsm8k/strict-match/exact_match"
        );
        assert_eq!(rows[0].dataset_revision, "lm-eval 0.4.13 task b59a0cf647dd");
        let flex = parse_lm_eval(&d, &json!({"filter": "flexible-extract"})).unwrap();
        assert!(flex.iter().all(|r| r.score.pass));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn parses_inspect_log() {
        let d = std::env::temp_dir().join(format!("tb-inspect-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let log = json!({"status": "success", "eval": {"task": "inspect_evals/gsm8k", "task_version": 2,
        "dataset": {"name": "openai/gsm8k"}, "packages": {"inspect_ai": "0.3.285"}},
        "samples": [
            {"id": "gsm8k_1", "epoch": 1, "input": "q1", "target": "18", "output": {"completion": "18"}, "scores": {"match": {"value": "C", "answer": "18"}}, "error": null},
            {"id": "gsm8k_2", "epoch": 1, "input": "q2", "target": "3", "output": {"completion": "4"}, "scores": {"match": {"value": "I", "answer": "4"}}, "error": null}
        ]});
        std::fs::write(d.join("x.json"), log.to_string()).unwrap();
        let rows = parse_inspect(&d).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].case_id, "inspect:inspect_evals/gsm8k/gsm8k_1");
        assert!(rows[0].score.pass && !rows[1].score.pass);
        assert_eq!(rows[0].scorer_version, "inspect/inspect_evals/gsm8k/match");
        assert!(rows[0].dataset_revision.contains("inspect_ai 0.3.285"));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn missing_tool_is_not_supported() {
        let t = ExternalTask::lm_eval("lm_eval:gsm8k");
        let env = ExecEnv {
            base_url: "http://x:1".into(),
            api_key: None,
            model: "m".into(),
            context: None,
            work_dir: std::env::temp_dir(),
            label: "l".into(),
            params: json!({"bin": "/nonexistent/lm-eval"}),
            limit: Some(5),
            temperature: 0.0,
            seed: Some(42),
            max_tokens: 10,
            cancel: Default::default(),
        };
        let e = t.command(&env, Path::new("/tmp/x")).err().unwrap();
        assert!(e.starts_with(NOT_SUPPORTED), "{e}");
    }
}
