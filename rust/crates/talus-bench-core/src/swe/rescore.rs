//! `talus-bench swe-rescore RUN_DIR`: re-judge a finished (or interrupted) run's SWE-bench instances from their
//! retained artifacts (snapshot head, verification, environment scan, gzipped strace, opencode events, bridge log,
//! patch) with the current auditor. The agent's behaviour does not depend on the auditor, so fixing an audit rule
//! after a run and re-judging every instance with one auditor is unbiased; re-running only the affected instances
//! would not be. A patch that becomes valid is evaluated offline (CPU only); an unchanged valid patch keeps its
//! evaluation. The previous rows are kept as `cases.jsonl.pre-rescore-<unix time>`.

use super::{dataset, judge, runtime::AgentRun, SwebenchTask};
use crate::case::CaseStatus;
use crate::config::Config;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

#[derive(Debug, Default)]
pub struct Rescored {
    pub file: PathBuf,
    /// (instance, old stage, new stage, new flag kinds)
    pub changes: Vec<(String, String, String, Vec<String>)>,
    pub kept: usize,
    pub skipped: Vec<(String, String)>,
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_default()
}

/// The agent run of one instance, rebuilt from its job directory (`trace.log.gz` is unpacked to `tmp`).
fn agent_run(job: &Path, details: &Value, tmp: &Path) -> Result<AgentRun, String> {
    let a = job.join("audit");
    if !a.is_dir() {
        return Err(format!("{}: no audit directory", a.display()));
    }
    let json_of = |f: &str| serde_json::from_str(read(&a.join(f)).trim()).unwrap_or(Value::Null);
    let gz = a.join("trace.log.gz");
    let strace_file = if gz.exists() {
        let out = tmp.join("trace.log");
        let file = std::fs::File::create(&out).map_err(|e| e.to_string())?;
        let st = std::process::Command::new("gzip")
            .arg("-dc")
            .arg(&gz)
            .stdout(file)
            .status()
            .map_err(|e| e.to_string())?;
        if !st.success() {
            return Err(format!("{}: gzip -dc failed", gz.display()));
        }
        Some(out)
    } else if a.join("trace.log").exists() {
        let out = tmp.join("trace.log");
        std::fs::copy(a.join("trace.log"), &out).map_err(|e| e.to_string())?;
        Some(out)
    } else {
        // no trace: the agent never started (verification failed) or the audit is missing
        Some(tmp.join("trace.log"))
    };
    Ok(AgentRun {
        head: read(&a.join("head")).trim().to_string(),
        patch: read(&a.join("patch.diff")),
        events: read(&a.join("events.jsonl")),
        strace: String::new(),
        strace_file,
        verify: json_of("verify.json"),
        sitepkg: json_of("sitepkg.json"),
        exit: details["agent_exit"].as_str().unwrap_or("").to_string(),
        wall_s: details["agent_wall_s"].as_f64().unwrap_or(0.0),
    })
}

/// Re-judge the `swebench_verified` rows of `RUN_DIR/<sub>` (sub: `baseline` or `candidates/<id>`).
pub fn rescore(run_dir: &Path, sub: &str, dry_run: bool) -> Result<Rescored, String> {
    rescore_with(run_dir, sub, dry_run, false)
}

/// [`rescore`]; `re_eval` evaluates every patch again (e.g. under a new evaluation network) instead of keeping
/// the evaluation an unchanged patch already has.
pub fn rescore_with(
    run_dir: &Path,
    sub: &str,
    dry_run: bool,
    re_eval: bool,
) -> Result<Rescored, String> {
    let cfg = Config::load(&run_dir.join("config.resolved.yaml"))?;
    let params = cfg
        .benchmarks
        .params
        .get("swebench_verified")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let draw: Value = serde_json::from_str(&read(&run_dir.join("swebench_draw.json")))
        .map_err(|e| format!("{}/swebench_draw.json: {e}", run_dir.display()))?;
    let cache = cfg
        .benchmarks
        .dataset_cache
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(crate::datasets::default_cache_dir);
    let snap = dataset::read_cache(&cache, draw["content_sha256"].as_str())?;
    let dir = run_dir.join(sub);
    let file = ["cases.jsonl", "cases.partial.jsonl"]
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.exists())
        .ok_or_else(|| format!("{}: no cases.jsonl", dir.display()))?;
    let mut rows = crate::case::read_jsonl(&file)?;
    let task = SwebenchTask::default();
    // the image index lives next to the default dataset cache, as in `execute`
    let image_cache = crate::datasets::default_cache_dir();
    let rt = task.runtime(
        &params,
        params["image_cache_dir"]
            .as_str()
            .map(Path::new)
            .unwrap_or(&image_cache),
    )?;
    let port = params["bridge_port"].as_u64().unwrap_or(18555) as u16;
    let mut out = Rescored {
        file: file.clone(),
        ..Default::default()
    };
    for row in rows.iter_mut().filter(|r| r.task == "swebench_verified") {
        let d = row.score.details.clone();
        let id = d["instance_id"].as_str().unwrap_or("").to_string();
        let old = d["stage"].as_str().unwrap_or("").to_string();
        // rows without an agent run (environment / harness failures) have nothing to re-judge
        if ![
            "invalid",
            "empty_patch",
            "resolved",
            "unresolved",
            "patch_not_applied",
            "not_evaluated",
        ]
        .contains(&old.as_str())
        {
            out.skipped.push((id, format!("stage {old}")));
            continue;
        }
        let Some(inst) = snap.instances.iter().find(|i| i.instance_id == id) else {
            out.skipped.push((id, "not in the dataset snapshot".into()));
            continue;
        };
        let job = dir.join("jobs/swe-jobs").join(&id);
        let tmp = super::short_private_dir()?;
        let run = match agent_run(&job, &d, &tmp) {
            Ok(r) => r,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&tmp);
                out.skipped.push((id, e));
                continue;
            }
        };
        let mut run = run;
        // a scan saved by an older scanner (no `matches_base`) describes the same pre-agent image state a fresh
        // offline container of the image reproduces: scan again with the current scanner
        let mut rescanned = false;
        if run.sitepkg.is_object() && run.sitepkg.get("matches_base").is_none() && !dry_run {
            let _ = rt.prepare(inst);
            if let Some(v) = rt.rescan_environment(inst) {
                run.sitepkg = v;
                rescanned = true;
            }
        }
        let bridge_log = read(&job.join("bridge.jsonl"));
        // an evaluation may be needed (a patch that was invalid before): make sure the image is there
        let needs_image = re_eval || !d["eval"].is_object();
        if needs_image && !dry_run {
            if let Err(e) = rt.prepare(inst) {
                let _ = std::fs::remove_dir_all(&tmp);
                out.skipped.push((id, format!("image prep: {e}")));
                continue;
            }
        }
        let o = if dry_run {
            // audit only: report what the new auditor says without evaluating
            let mut o = judge(
                inst,
                &run,
                &bridge_log,
                &NoEval(rt.as_ref()),
                &params,
                port,
                d["prep"].clone(),
                std::time::Instant::now(),
                (!re_eval).then_some(&d),
            );
            if o.error.as_deref().is_some_and(|e| e.contains("dry run")) {
                o.score.details["stage"] = json!("valid (dry run: not evaluated)");
            }
            o
        } else {
            judge(
                inst,
                &run,
                &bridge_log,
                rt.as_ref(),
                &params,
                port,
                d["prep"].clone(),
                std::time::Instant::now(),
                (!re_eval).then_some(&d),
            )
        };
        let _ = std::fs::remove_dir_all(&tmp);
        let new = o.score.details["stage"].as_str().unwrap_or("?").to_string();
        let kinds: Vec<String> = o.score.details["flags"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|f| f["kind"].as_str().map(str::to_string))
            .collect();
        if new != old || d["flags"] != o.score.details["flags"] {
            out.changes
                .push((id.clone(), old.clone(), new.clone(), kinds));
        } else {
            out.kept += 1;
        }
        let mut details = o.score.details.clone();
        details["rescored"] = json!({"previous_stage": old, "previous_flags": d["flags"], "environment_rescanned": rescanned,
            "at": crate::util::unix_now(), "by": option_env!("TALUS_BUILD_COMMIT").unwrap_or("unrecorded")});
        // keep the original agent timing and request accounting
        for k in ["agent_wall_s", "agent_exit", "prep"] {
            details[k] = d[k].clone();
        }
        row.score = o.score;
        row.score.details = details;
        row.status = if o.status == CaseStatus::Ok {
            CaseStatus::Ok
        } else {
            o.status
        };
        row.error = o.error;
        if let Some(raw) = o.raw {
            row.raw_output = Some(raw);
        }
    }
    if dry_run {
        return Ok(out);
    }
    // a run still in progress appends to cases.partial.jsonl: never rewrite it, write beside it
    if file.ends_with("cases.partial.jsonl") {
        let side = dir.join("cases.rescored.partial.jsonl");
        std::fs::write(&side, crate::case::to_jsonl(&rows))
            .map_err(|e| format!("{}: {e}", side.display()))?;
        out.file = side;
        return Ok(out);
    }
    let backup = file.with_extension(format!(
        "jsonl.pre-rescore-{}",
        crate::util::unix_now() as u64
    ));
    std::fs::copy(&file, &backup).map_err(|e| format!("{}: {e}", backup.display()))?;
    std::fs::write(&file, crate::case::to_jsonl(&rows))
        .map_err(|e| format!("{}: {e}", file.display()))?;
    std::fs::write(
        dir.join("metrics.json"),
        serde_json::to_string_pretty(&crate::compare::model_metrics(
            &rows,
            cfg.quality_gates.confidence,
        ))
        .unwrap_or_default(),
    )
    .map_err(|e| e.to_string())?;
    // a finished reference-only run: re-render the report from the re-judged rows
    if cfg.candidates.is_empty() && file.ends_with("cases.jsonl") && sub == "baseline" {
        let mut manifest: Value =
            serde_json::from_str(&read(&run_dir.join("manifest.json"))).unwrap_or(Value::Null);
        if manifest.is_object() {
            manifest["rescored"] = json!({"at": crate::util::rfc3339(crate::util::unix_now()),
                "by": option_env!("TALUS_BUILD_COMMIT").unwrap_or("unrecorded"), "changed": out.changes.len(),
                "note": "swebench_verified rows re-judged from retained artifacts with the current auditor (swe-rescore)"});
            let model_rows = vec![(cfg.reference.id.clone(), rows)];
            let ctx = crate::report::ReportCtx {
                cfg: &cfg,
                manifest: &manifest,
                paired: &[],
                verdicts: &[],
                unavailable: &[],
                model_rows: &model_rows,
            };
            let w =
                |f: &str, t: String| std::fs::write(run_dir.join(f), t).map_err(|e| e.to_string());
            w(
                "manifest.json",
                serde_json::to_string_pretty(&manifest).unwrap_or_default(),
            )?;
            w(
                "report.json",
                serde_json::to_string_pretty(&crate::report::report_json(&ctx)).unwrap_or_default(),
            )?;
            w("report.md", crate::report::report_md(&ctx))?;
        }
    }
    Ok(out)
}

/// A runtime wrapper for dry runs: everything but the evaluation.
struct NoEval<'a>(&'a dyn super::runtime::Runtime);

impl super::runtime::Runtime for NoEval<'_> {
    fn prepare(&self, inst: &dataset::Instance) -> Result<super::runtime::PrepInfo, String> {
        self.0.prepare(inst)
    }
    fn run_agent(
        &self,
        inst: &dataset::Instance,
        spec: &super::runtime::AgentSpec,
    ) -> Result<AgentRun, String> {
        self.0.run_agent(inst, spec)
    }
    fn run_eval(
        &self,
        _inst: &dataset::Instance,
        _patch: &str,
    ) -> Result<super::runtime::EvalRun, String> {
        Err("dry run: not evaluated".into())
    }
    fn policy(
        &self,
        inst: &dataset::Instance,
        module: &str,
        dist: &str,
        port: u16,
    ) -> super::audit::Policy {
        self.0.policy(inst, module, dist, port)
    }
}
