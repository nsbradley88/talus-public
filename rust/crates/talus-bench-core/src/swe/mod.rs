//! `swebench_verified`: SWE-bench Verified as a native job task with layered anti-cheating. DESIGN.md §21 has
//! the full design; in short, per run:
//! - a LIVE fetch of the 500 instances (cache only as offline fallback, said in the report) and a FRESH stratified
//!   draw of `n` (default 50) with a CSPRNG seed recorded in the manifest (`params.seed` reproduces a run); the
//!   reference and every candidate get the same draw; a resumed run reuses its draw;
//! - per instance: image prep (LRU cache), an agent container with no network except the model bridge, a
//!   single-commit snapshot verified before the agent starts, an environment scan for future copies of the
//!   library, the agent (opencode) traced with strace, then checks (audit flags, test-file edits), a FRESH offline
//!   evaluation container with the official apply chain, eval script and grading; gold-patch similarity and the
//!   knowledge-cutoff split are measured and reported, never used to pass or fail.
//!
//! Any audit, history, environment or test-edit flag makes the instance INVALID (counted as not resolved).

pub mod audit;
pub mod bridge;
pub mod dataset;
pub mod egress;
pub mod harness;
pub mod patch;
pub mod rescore;
pub mod runtime;
pub mod sitepkg;
pub mod snapshot;

use crate::case::{Case, CaseStatus, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::DatasetPin;
use crate::tasks::{ExecEnv, ExecOutcome, Task, TaskContext};
use dataset::{Draw, Instance};
use runtime::{AgentRun, AgentSpec, Runtime};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

pub const DRAW_FILE: &str = "swebench_draw.json";

#[derive(Default)]
struct State {
    instances: BTreeMap<String, Instance>,
    draw: Option<Draw>,
}

#[derive(Default)]
pub struct SwebenchTask {
    state: Mutex<State>,
}

static TEST_RUNTIME: OnceLock<Mutex<Option<Arc<dyn Runtime>>>> = OnceLock::new();

/// Tests only: the runtime used when `params.runtime` is "test".
pub fn set_test_runtime(r: Option<Arc<dyn Runtime>>) {
    *TEST_RUNTIME
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap() = r;
}

fn test_runtime() -> Option<Arc<dyn Runtime>> {
    TEST_RUNTIME
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap()
        .clone()
}

/// The prompt the agent gets: the issue text only (no hints, no tests, no gold).
pub fn agent_prompt(inst: &Instance) -> String {
    format!(
        "You are working in the repository at /testbed (the {} project). Resolve the following issue by editing the source code. \
Do not modify or add tests. Work offline: there is no network access. When you are done, leave the changes in the working tree.\n\n<issue>\n{}\n</issue>",
        inst.repo, inst.problem_statement
    )
}

fn param_u64(p: &Value, k: &str, d: u64) -> u64 {
    p[k].as_u64().unwrap_or(d)
}

impl SwebenchTask {
    pub(crate) fn runtime(&self, params: &Value, cache: &Path) -> Result<Arc<dyn Runtime>, String> {
        match params["runtime"].as_str().unwrap_or("docker") {
            "docker" => Ok(Arc::new(runtime::DockerRuntime::new(
                cache,
                params["image_cache_gb"].as_f64().unwrap_or(150.0),
                param_u64(params, "eval_timeout_s", 1800),
                params["validate_gold"].as_bool().unwrap_or(true),
            ))),
            "test" => test_runtime().ok_or_else(|| "no test runtime set".to_string()),
            other => Err(format!("unknown swebench runtime {other}")),
        }
    }
}

/// A new 0700 directory with a short path (`/tmp/tbsw-<pid>-<n>`).
pub fn short_private_dir() -> Result<PathBuf, String> {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let d = PathBuf::from(format!(
        "/tmp/tbsw-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o700))
        .map_err(|e| e.to_string())?;
    Ok(d)
}

/// Remove the hunks of `files` from a unified diff.
pub fn strip_files(diff: &str, files: &[String]) -> String {
    let mut out = String::new();
    let mut keep = true;
    for l in diff.split_inclusive('\n') {
        if let Some(rest) = l.strip_prefix("diff --git ") {
            keep = !files.iter().any(|f| {
                rest.contains(&format!("a/{f} ")) || rest.trim_end().ends_with(&format!("b/{f}"))
            });
        }
        if keep {
            out.push_str(l);
        }
    }
    out
}

impl Task for SwebenchTask {
    fn id(&self) -> &str {
        "swebench_verified"
    }

    fn scorer_version(&self) -> &'static str {
        "swebench_verified-harness5.0.2/1"
    }

    fn datasets(&self, _tier: Tier) -> Vec<&'static DatasetPin> {
        Vec::new()
    }

    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String> {
        let n = param_u64(ctx.params, "n", 50) as usize;
        let draw_path = ctx.run_dir.map(|d| d.join(DRAW_FILE));
        let previous: Option<Draw> = if ctx.resume {
            draw_path
                .as_ref()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .and_then(|t| serde_json::from_str(&t).ok())
        } else {
            None
        };
        let (snap, draw) = match previous {
            Some(d) => {
                let snap = dataset::read_cache(ctx.cache, Some(&d.content_sha256))
                    .map_err(|e| format!("resuming swebench draw {}: {e}", d.seed))?;
                let mut d = d;
                d.seed_source = "resumed".into();
                (snap, d)
            }
            None => {
                let snap = if ctx.params["offline"].as_bool() == Some(true) {
                    let mut s = dataset::read_cache(ctx.cache, None)?;
                    s.source = "cache (offline run requested)".into();
                    s
                } else {
                    dataset::load(ctx.cache, &dataset::live_fetch)?
                };
                // a fixed subset (`instances_file`: JSON with `instance_ids`, or `instance_ids` inline) replaces
                // the random draw, e.g. the same 20 tasks for every arm of an ablation
                let fixed: Option<(Vec<String>, String, u64)> = match (
                    ctx.params["instances_file"].as_str(),
                    ctx.params["instance_ids"].as_array(),
                ) {
                    (Some(f), _) => {
                        let v: Value = serde_json::from_str(
                            &std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?,
                        )
                        .map_err(|e| format!("{f}: {e}"))?;
                        let ids = v["instance_ids"]
                            .as_array()
                            .ok_or_else(|| format!("{f}: no instance_ids"))?
                            .iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect();
                        let name = Path::new(f)
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default();
                        Some((
                            ids,
                            format!("fixed:{name}"),
                            v["seed"].as_u64().unwrap_or(0),
                        ))
                    }
                    (None, Some(a)) => Some((
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect(),
                        "fixed:params".into(),
                        0,
                    )),
                    _ => None,
                };
                let (seed, src, ids, per_repo) = match fixed {
                    Some((ids, src, seed)) => {
                        let mut per_repo: BTreeMap<String, usize> = BTreeMap::new();
                        for id in &ids {
                            let inst = snap
                                .instances
                                .iter()
                                .find(|i| &i.instance_id == id)
                                .ok_or_else(|| {
                                    format!("fixed subset: {id} is not in SWE-bench Verified")
                                })?;
                            *per_repo.entry(inst.repo.clone()).or_default() += 1;
                        }
                        (seed, src, ids, per_repo)
                    }
                    None => {
                        let (seed, src) = match ctx.params["seed"].as_u64() {
                            Some(s) => (s, "param".to_string()),
                            None => (dataset::csprng_seed()?, "csprng".to_string()),
                        };
                        let (ids, per_repo) = dataset::draw(&snap.instances, n, seed);
                        (seed, src, ids, per_repo)
                    }
                };
                let d = Draw {
                    seed,
                    seed_source: src,
                    n: ids.len(),
                    dataset: dataset::HF_DATASET.into(),
                    revision: snap.revision.clone(),
                    content_sha256: snap.content_sha256.clone(),
                    source: snap.source.clone(),
                    drawn_at: crate::util::rfc3339(crate::util::unix_now()),
                    per_repo,
                    instance_ids: ids,
                };
                if let Some(p) = &draw_path {
                    if let Some(dir) = p.parent() {
                        let _ = std::fs::create_dir_all(dir);
                    }
                    std::fs::write(p, serde_json::to_string_pretty(&d).unwrap_or_default())
                        .map_err(|e| format!("{}: {e}", p.display()))?;
                }
                (snap, d)
            }
        };
        let by_id: BTreeMap<String, Instance> = snap
            .instances
            .into_iter()
            .map(|i| (i.instance_id.clone(), i))
            .collect();
        let rev = format!(
            "{}@{}",
            draw.revision
                .as_deref()
                .map(|r| &r[..12.min(r.len())])
                .unwrap_or("?"),
            &draw.content_sha256[..12]
        );
        let mut cases = Vec::new();
        for id in &draw.instance_ids {
            let inst = by_id
                .get(id)
                .ok_or_else(|| format!("drawn instance {id} is not in the dataset snapshot"))?;
            cases.push(Case {
                task: self.id().into(),
                case_id: format!("swebench_verified/{id}"),
                dataset: dataset::HF_DATASET.into(),
                dataset_revision: rev.clone(),
                messages: vec![json!({"role": "user", "content": agent_prompt(inst)})],
                // no gold patch, tests or test lists here: cases are written to the run's artifacts
                expected: json!({"instance_id": id, "repo": inst.repo, "created_at": inst.created_at, "version": inst.version}),
                meta: Value::Null,
                critical: false,
                max_tokens: None,
                tools: Value::Null,
                request: json!({"swebench": id}),
            });
        }
        let mut st = self.state.lock().unwrap();
        st.instances = by_id;
        st.draw = Some(draw);
        Ok(cases)
    }

    fn score(&self, _case: &Case, _out: &RawOutput) -> Score {
        Score::default()
    }

    fn execute(&self, case: &Case, env: &ExecEnv) -> Option<ExecOutcome> {
        let t0 = std::time::Instant::now();
        let id = case.expected["instance_id"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let inst = self.state.lock().unwrap().instances.get(&id).cloned();
        let fail = |status: CaseStatus, stage: &str, e: String| ExecOutcome {
            status,
            raw: None,
            score: Score::binary(
                false,
                Value::Null,
                json!({"stage": stage, "instance_id": id}),
            ),
            error: Some(e),
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
        };
        let Some(inst) = inst else {
            return Some(fail(
                CaseStatus::ProtocolError,
                "harness",
                format!("{id}: not in the drawn snapshot"),
            ));
        };
        // the image index is shared by every run (LRU across runs), next to the dataset cache
        let cache = crate::datasets::default_cache_dir();
        let rt = match self.runtime(
            &env.params,
            env.params["image_cache_dir"]
                .as_str()
                .map(Path::new)
                .unwrap_or(&cache),
        ) {
            Ok(r) => r,
            Err(e) => return Some(fail(CaseStatus::ProtocolError, "harness", e)),
        };
        // 1. image (outside any agent run; network allowed here only)
        let prep = match rt.prepare(&inst) {
            Ok(p) => p,
            Err(e) => {
                return Some(fail(
                    CaseStatus::ServerError,
                    "environment",
                    format!("image prep: {e}"),
                ))
            }
        };
        if prep.gold_ok == Some(false) {
            return Some(fail(
                CaseStatus::ServerError,
                "environment",
                "the gold patch does not resolve offline in this image".into(),
            ));
        }
        // 2. the model bridge (the agent's only way out) and the agent
        // (a neutral name: paths naming SWE-bench are themselves audit flags)
        let work = env.work_dir.join("swe-jobs").join(&id);
        let _ = std::fs::remove_dir_all(&work);
        // unix socket paths are limited to 108 bytes: the bridge directory lives in a short private /tmp dir
        let sockdir = match short_private_dir() {
            Ok(d) => d,
            Err(e) => return Some(fail(CaseStatus::ProtocolError, "harness", e)),
        };
        if let Err(e) = std::fs::create_dir_all(&work) {
            return Some(fail(CaseStatus::ProtocolError, "harness", e.to_string()));
        }
        let port = param_u64(&env.params, "bridge_port", 18555) as u16;
        let bridge = match bridge::start(
            &sockdir.join("model.sock"),
            &env.base_url,
            env.api_key.clone(),
            &work.join("bridge.jsonl"),
        ) {
            Ok(b) => b,
            Err(e) => return Some(fail(CaseStatus::ProtocolError, "harness", e)),
        };
        let opencode = env.params["opencode"]
            .as_str()
            .map(PathBuf::from)
            .or_else(|| crate::tasks::which("opencode"))
            .unwrap_or_else(|| PathBuf::from("/usr/local/bin/opencode"));
        let spec = AgentSpec {
            model: env.model.clone(),
            bridge_socket_dir: sockdir.clone(),
            bridge_port: port,
            work_dir: work.clone(),
            opencode,
            timeout: Duration::from_secs(param_u64(&env.params, "agent_timeout_s", 2700)),
            prompt: agent_prompt(&inst),
            context_tokens: param_u64(&env.params, "context_tokens", 65536),
            output_tokens: param_u64(&env.params, "output_tokens", 16384),
            mini_bundle: (env.params["agent"].as_str() == Some("mini")).then(|| {
                env.params["mini_bundle"]
                    .as_str()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("/opt/talus-bench/mini-swe-agent-2.4.6"))
            }),
            problem: inst.problem_statement.clone(),
        };
        let run = match rt.run_agent(&inst, &spec) {
            Ok(r) => r,
            Err(e) => {
                return Some(fail(
                    CaseStatus::ProtocolError,
                    "harness",
                    format!("agent container: {e}"),
                ))
            }
        };
        let bridge_log = std::fs::read_to_string(&bridge.log).unwrap_or_default();
        drop(bridge);
        let _ = std::fs::remove_dir_all(&sockdir);
        Some(judge(
            &inst,
            &run,
            &bridge_log,
            rt.as_ref(),
            &env.params,
            port,
            json!(prep),
            t0,
            None,
        ))
    }

    fn provenance(&self) -> Value {
        match &self.state.lock().unwrap().draw {
            Some(d) => serde_json::to_value(d).unwrap_or(Value::Null),
            None => Value::Null,
        }
    }

    fn caveats(&self) -> Vec<String> {
        let st = self.state.lock().unwrap();
        let mut c = vec![
            "swebench_verified: memorization cannot be blocked, only measured. All 500 Verified instances were created between 2013 and 2023, before current models' training cutoffs; the gold-patch similarity (near-verbatim >= 0.9 per instance) and the after-cutoff split are reported in the case details. Contamination-resistant companion sets (SWE-rebench / SWE-bench-Live tasks created after the model's cutoff) are planned and would be reported separately.".to_string(),
            "swebench_verified: each run draws a fresh random sample; results of different runs are not comparable instance for instance. Use `talus-bench pool` to combine runs into one estimate with a confidence interval; one run of 50 has a wide interval.".to_string(),
        ];
        if let Some(d) = &st.draw {
            if d.source != "live" {
                c.push(format!("swebench_verified: the instance list came from the local {} (the live fetch was not used).", d.source));
            }
            c.push(format!("swebench_verified: draw seed {} ({}), {} instances, dataset {} revision {} content sha256 {}.", d.seed, d.seed_source, d.n, d.dataset, d.revision.as_deref().unwrap_or("?"), &d.content_sha256[..16]));
        }
        c
    }
}

/// Steps 3-4 of an instance once the agent has run: history/environment checks, the audit trail, test edits,
/// then (valid, non-empty patch) the fresh offline evaluation. Shared by `execute` and `swe-rescore` (which
/// rebuilds `run` from a finished run's retained artifacts).
#[allow(clippy::too_many_arguments)]
pub fn judge(
    inst: &Instance,
    run: &AgentRun,
    bridge_log: &str,
    rt: &dyn Runtime,
    params: &Value,
    port: u16,
    prep: Value,
    t0: std::time::Instant,
    prior: Option<&Value>,
) -> ExecOutcome {
    let fail = |status: CaseStatus, stage: &str, e: String| ExecOutcome {
        status,
        raw: None,
        score: Score::binary(
            false,
            Value::Null,
            json!({"stage": stage, "instance_id": inst.instance_id}),
        ),
        error: Some(e),
        wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
    };
    // 3. checks: history, environment, audit trail, test edits
    let (dist, module) = sitepkg::target_of(&inst.repo);
    let mut policy = rt.policy(inst, module, dist, port);
    policy.head = run.head.clone();
    let mut a = audit::Audit::default();
    for p in snapshot::problems(&run.verify) {
        a.flags.push(audit::Flag {
            kind: "history".into(),
            detail: p,
            source: "verify".into(),
        });
    }
    if run.sitepkg.is_null() {
        a.flags.push(audit::Flag {
            kind: "future_copy".into(),
            detail: "environment scan did not run".into(),
            source: "sitepkg".into(),
        });
    }
    for n in sitepkg::notes(&run.sitepkg) {
        a.notes.push(audit::Flag {
            kind: "base_copy_installed".into(),
            detail: n,
            source: "sitepkg".into(),
        });
    }
    for f in sitepkg::flags(&run.sitepkg)
        .into_iter()
        .filter(|_| !run.sitepkg.is_null())
    {
        a.flags.push(audit::Flag {
            kind: "future_copy".into(),
            detail: f,
            source: "sitepkg".into(),
        });
    }
    match &run.strace_file {
        // the agent never started (verification stopped it): nothing to trace
        Some(f) if !f.exists() && run.exit.starts_with("verification failed") => {}
        // the docker runtime's trace can be GBs: stream it, then keep it gzipped as evidence
        Some(f) => {
            use std::io::BufRead;
            match std::fs::File::open(f) {
                Ok(_) => {
                    // two streaming passes (process tree, then audit)
                    let lines = || {
                        std::fs::File::open(f)
                            .map(|file| {
                                std::io::BufReader::with_capacity(1 << 20, file)
                                    .lines()
                                    .map_while(Result::ok)
                            })
                            .into_iter()
                            .flatten()
                    };
                    audit::audit_strace_source(lines, &policy, &policy.repo.clone(), &mut a);
                    let _ = std::process::Command::new("gzip").arg("-f").arg(f).status();
                }
                Err(e) => a.flags.push(audit::Flag {
                    kind: "audit_missing".into(),
                    detail: format!("strace log unreadable: {e}"),
                    source: "strace".into(),
                }),
            }
        }
        None => audit::audit_strace(&run.strace, &policy, &policy.repo.clone(), &mut a),
    }
    audit::audit_events(&run.events, &policy, &mut a);
    audit::audit_bridge(bridge_log, &mut a);
    let mut patch_text = run.patch.clone();
    let edits = patch::test_edits(&patch_text, &inst.test_patch);
    if !edits.is_empty() {
        if params["test_edits"].as_str() == Some("strip") {
            patch_text = strip_files(&patch_text, &edits);
        } else {
            a.flags.push(audit::Flag {
                kind: "test_edit".into(),
                detail: edits.join(", "),
                source: "patch".into(),
            });
        }
    }
    for s in egress::patch_network_code(&patch_text) {
        a.notes.push(audit::Flag {
            kind: "patch_network_code".into(),
            detail: s,
            source: "patch".into(),
        });
    }
    let sim = patch::similarity(&patch_text, &inst.patch);
    let cutoff = params["model_cutoff"].as_str().map(str::to_string);
    let after_cutoff = cutoff
        .as_ref()
        .map(|c| inst.created_at.as_str() > c.as_str());
    let mut details = json!({
        "instance_id": inst.instance_id, "repo": inst.repo, "created_at": inst.created_at, "model_cutoff": cutoff,
        "after_cutoff": after_cutoff, "similarity": sim, "near_verbatim": sim.ratio >= patch::NEAR_VERBATIM,
        "flags": a.flags, "notes": a.notes, "harness_network": a.harness_network, "harness_commands": a.harness_commands, "package_commands": a.package_commands, "audit_counts":
            {"execs": a.execs, "file_accesses": a.file_accesses, "tool_calls": a.tool_calls},
        "prep": prep, "agent_exit": run.exit, "agent_wall_s": run.wall_s, "agent_usage": bridge::usage_summary(bridge_log), "patch_lines": patch_text.lines().count(),
    });
    let raw = Some(RawOutput {
        content: patch_text.chars().take(20_000).collect(),
        finish_reason: "job".into(),
        ..Default::default()
    });
    // pass/fail and flags are reported separately: a patch is evaluated even when a leak vector was flagged
    // (`eval`, `resolved_ignoring_flags`); only an unflagged instance can pass
    let mut invalid = a.invalid();
    let patch_sha = crate::util::sha256_hex(patch_text.as_bytes());
    details["patch_sha256"] = json!(patch_sha);
    let finish = |mut details: Value, eval_stage: &str, resolved: bool, invalid: bool| {
        details["eval_stage"] = json!(eval_stage);
        details["resolved_ignoring_flags"] = json!(resolved);
        details["stage"] = json!(if invalid { "invalid" } else { eval_stage });
        ExecOutcome {
            status: CaseStatus::Ok,
            raw: raw.clone(),
            score: Score::binary(resolved && !invalid, Value::Null, details),
            error: None,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
    };
    if patch_text.trim().is_empty() {
        return finish(details, "empty_patch", false, invalid);
    }
    // a re-judged run (swe-rescore) keeps the evaluation it already has for the same patch
    if let Some(p) = prior.filter(|p| {
        p["eval"].is_object()
            && (p["patch_sha256"] == json!(patch_sha)
                || (p["patch_sha256"].is_null() && p["patch_lines"] == details["patch_lines"]))
    }) {
        let e = p["eval"].clone();
        let st = eval_stage(e["resolved"] == true, e["patch_applied"] == true);
        details["eval"] = e.clone();
        details["eval_reused"] = json!(true);
        // the reused evaluation's network use still counts: its flags and notes come from the eval, not the
        // agent audit, so carry them over (otherwise a patch that fetched upstream at test time is cleared)
        if p["eval_network"].is_object() {
            details["eval_network"] = p["eval_network"].clone();
            for (key, kind) in [("flags", "eval_network"), ("notes", "eval_network_blocked")] {
                let mut list = details[key].as_array().cloned().unwrap_or_default();
                for f in p[key].as_array().into_iter().flatten() {
                    if f["kind"] == kind && !list.contains(f) {
                        list.push(f.clone());
                        invalid |= key == "flags";
                    }
                }
                details[key] = json!(list);
            }
        }
        return finish(details, st, e["resolved"] == true, invalid);
    }
    // 4. a fresh, offline evaluation container with the hidden tests
    let log = match rt.run_eval(inst, &patch_text) {
        Ok(er) => {
            // the tests' network use, against the gold run's: anything new is an attempt to reach elsewhere
            // (e.g. the upstream fix fetched at import)
            if !er.network.is_null() {
                let novel: Vec<String> = er.network["novel"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect();
                if !novel.is_empty() {
                    let mut flags = details["flags"].as_array().cloned().unwrap_or_default();
                    flags.push(json!({"kind": "eval_network", "detail": format!("the tests contacted what the gold run did not: {}", novel.join(", ")), "source": "eval"}));
                    details["flags"] = json!(flags);
                    invalid = true;
                }
                let blocked: Vec<String> = er.network["novel_blocked"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect();
                if !blocked.is_empty() {
                    let mut notes = details["notes"].as_array().cloned().unwrap_or_default();
                    notes.push(json!({"kind": "eval_network_blocked", "detail": format!("new attempts at blackholed private addresses: {}", blocked.join(", ")), "source": "eval"}));
                    details["notes"] = json!(notes);
                }
                details["eval_network"] = er.network;
            }
            er.log
        }
        Err(e) if invalid => {
            details["eval_error"] = json!(e);
            return finish(details, "not_evaluated", false, invalid);
        }
        Err(e) => {
            return fail(
                CaseStatus::ServerError,
                "environment",
                format!("eval container: {e}"),
            )
        }
    };
    let g = match harness::grade(
        &inst.log_parser,
        &inst.eval_type,
        &inst.fail_to_pass,
        &inst.pass_to_pass,
        &log,
    ) {
        Ok(g) => g,
        Err(e) => return fail(CaseStatus::ProtocolError, "harness", e),
    };
    details["eval"] = json!({"patch_applied": g.patch_applied, "resolved": g.resolved,
            "f2p": g.report.as_ref().map(|r| (r.f2p_success.len(), r.f2p_success.len() + r.f2p_failure.len())),
            "p2p": g.report.as_ref().map(|r| (r.p2p_success.len(), r.p2p_success.len() + r.p2p_failure.len()))});
    finish(
        details,
        eval_stage(g.resolved, g.patch_applied),
        g.resolved,
        invalid,
    )
}

fn eval_stage(resolved: bool, applied: bool) -> &'static str {
    if resolved {
        "resolved"
    } else if applied {
        "unresolved"
    } else {
        "patch_not_applied"
    }
}

#[cfg(test)]
mod tests;
