//! `livecodebench`: LiveCodeBench code generation (pinned release increment, sandboxed execution).
//!
//! Dataset: `livecodebench/code_generation_lite` at a pinned Hub revision, release increment v6 (`test6.jsonl`,
//! 175 problems from AtCoder and LeetCode, contest dates 2025-01-04 .. 2025-04-06; the newest release published).
//! Fetched once into the dataset cache (`talus-bench datasets fetch livecodebench-v6`) and verified by sha256 on
//! every load; never embedded or committed (the dataset card declares only `license: cc`). Optional date window
//! (`date_from` / `date_to`), difficulty and platform filters; `model_cutoff` splits the pass rate before/after.
//!
//! Prompt: the problem statement, then either the LeetCode starter code (functional problems) or "read stdin,
//! write stdout" (stdin problems), asking for one fenced Python block. Sampling (`sampling`): `shipped` (default:
//! temperature/top_p sent as null, so the server applies the model's shipped sampling; one sample per case and
//! seed = pass@1, `inference.seeds` gives several samples per case), `greedy` (temperature 0), or `config` (the
//! run's inference settings). `max_tokens` default 16384 (reasoning models).
//!
//! Scoring: the last fenced code block of the answer content runs against the public + hidden tests in a fresh
//! sandbox container per case (`lcb_runner.py`): `docker run --network none --read-only` (tmpfs /tmp only),
//! memory/swap, CPU, pids and per-test time limits, every capability dropped except SETUID/SETGID/KILL (the
//! runner demotes each test's process to uid 65534, and kills it on timeout), `no-new-privileges`, image pinned by digest and never pulled at run
//! time (`talus-bench lcb-prep` pulls it once). The tests reach the container on stdin and live only in the
//! root-owned runner's memory, so the solution cannot read the expected outputs. A Python audit hook in the
//! solution's process records network, process-spawn, file-write and ctypes attempts: any such flag makes the case
//! INVALID (not passed, `stage: sandbox_violation`); `passed_ignoring_flags` keeps the plain test outcome.
//! Grading follows LiveCodeBench's published rules (reimplemented, see `lcb_runner.py`).

use super::{Task, TaskContext, NOT_SUPPORTED};
use crate::case::{Case, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::{self, DatasetPin};
use crate::util::sha256_hex;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The sandbox runner (container entrypoint, `python3 -I -B -c`).
pub const RUNNER_PY: &str = include_str!("lcb_runner.py");

/// `python:3.12-slim-bookworm`, multi-arch index digest (Docker Hub, 2026-11-02). Python stdlib only.
pub const DEFAULT_IMAGE: &str =
    "python@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337";
pub const DEFAULT_DATASET: &str = "livecodebench-v6";

const SYSTEM: &str = "You are an expert Python programmer. Write a correct and efficient Python 3 program for the problem below; it will be run against hidden tests.";
const FORMAT_FUNCTIONAL: &str = "Complete the following starter code and return the whole solution in a single fenced Python code block:";
const FORMAT_STDIN: &str = "The program must read the input from standard input and write the answer to standard output (do not hard-code the sample tests). Return the whole program in a single fenced Python code block:";

#[derive(Default)]
pub struct LcbTask {
    chosen: Mutex<Option<&'static DatasetPin>>,
    prov: Mutex<Value>,
}

/// Sandbox settings, carried in each case's `meta` so scoring is a function of (case, output).
#[derive(Clone, Debug)]
pub struct Sandbox {
    pub executor: String,
    pub docker: String,
    pub image: String,
    pub memory: String,
    pub cpus: String,
    pub pids: u64,
    pub tmpfs_mb: u64,
    pub test_timeout_s: f64,
    pub eval_timeout_s: u64,
}

impl Sandbox {
    pub fn from_params(p: &Value) -> Self {
        let s = |k: &str, d: &str| p[k].as_str().unwrap_or(d).to_string();
        Sandbox {
            executor: s("executor", "docker"),
            docker: s("docker", "docker"),
            image: s("image", DEFAULT_IMAGE),
            memory: s("memory", "2g"),
            cpus: s("cpus", "1"),
            pids: p["pids"].as_u64().unwrap_or(128),
            tmpfs_mb: p["tmpfs_mb"].as_u64().unwrap_or(256),
            test_timeout_s: p["test_timeout_s"].as_f64().unwrap_or(6.0),
            eval_timeout_s: p["eval_timeout_s"].as_u64().unwrap_or(900),
        }
    }

    pub fn to_value(&self) -> Value {
        json!({"executor": self.executor, "docker": self.docker, "image": self.image, "memory": self.memory,
               "cpus": self.cpus, "pids": self.pids, "tmpfs_mb": self.tmpfs_mb,
               "test_timeout_s": self.test_timeout_s, "eval_timeout_s": self.eval_timeout_s})
    }

    pub fn from_value(v: &Value) -> Self {
        Self::from_params(v)
    }

    /// The `docker run` arguments (everything after `docker`).
    pub fn docker_args(&self, name: &str) -> Vec<String> {
        let mut a: Vec<String> = [
            "run",
            "--rm",
            "-i",
            "--pull",
            "never",
            "--name",
            name,
            "--label",
            "talus-bench=livecodebench",
            "--network",
            "none",
            "--read-only",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        a.push("--tmpfs".into());
        a.push(format!(
            "/tmp:rw,nosuid,nodev,noexec,size={}m,mode=1777",
            self.tmpfs_mb
        ));
        for (k, v) in [
            ("--memory", self.memory.clone()),
            ("--memory-swap", self.memory.clone()),
            ("--cpus", self.cpus.clone()),
            ("--pids-limit", self.pids.to_string()),
            ("--cap-drop", "ALL".into()),
            ("--cap-add", "SETUID".into()),
            ("--cap-add", "SETGID".into()),
            // the runner (root, no other capability) must be able to kill the demoted test processes
            ("--cap-add", "KILL".into()),
            ("--security-opt", "no-new-privileges".into()),
            ("--ulimit", "core=0".into()),
            ("--user", "0:0".into()),
            ("-e", "PYTHONDONTWRITEBYTECODE=1".into()),
        ] {
            a.push(k.into());
            a.push(v);
        }
        a.push("--init".into());
        a.push(self.image.clone());
        for x in ["python3", "-I", "-B", "-c", RUNNER_PY] {
            a.push(x.into());
        }
        a
    }
}

fn param_str<'a>(p: &'a Value, k: &str) -> Option<&'a str> {
    p[k].as_str().filter(|s| !s.is_empty())
}

fn str_list(p: &Value, k: &str) -> Option<Vec<String>> {
    p[k].as_array().map(|a| {
        a.iter()
            .filter_map(|x| x.as_str().map(|s| s.to_lowercase()))
            .collect()
    })
}

/// One parsed LiveCodeBench problem (the test fields kept verbatim).
#[derive(Clone, Debug)]
pub struct Problem {
    pub question_id: String,
    pub title: String,
    pub content: String,
    pub platform: String,
    pub contest_id: String,
    pub contest_date: String,
    pub difficulty: String,
    pub starter_code: String,
    pub fn_name: Option<String>,
    pub public: String,
    pub private: String,
}

pub fn parse_problems(text: &str, id: &str) -> Result<Vec<Problem>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let r: Value = serde_json::from_str(line).map_err(|e| format!("{id} row {i}: {e}"))?;
        let s = |k: &str| r[k].as_str().unwrap_or_default().to_string();
        let meta: Value =
            serde_json::from_str(r["metadata"].as_str().unwrap_or("{}")).unwrap_or(Value::Null);
        let q = s("question_id");
        if q.is_empty() {
            return Err(format!("{id} row {i}: no question_id"));
        }
        out.push(Problem {
            question_id: q,
            title: s("question_title"),
            content: s("question_content"),
            platform: s("platform"),
            contest_id: s("contest_id"),
            contest_date: s("contest_date"),
            difficulty: s("difficulty"),
            starter_code: s("starter_code"),
            fn_name: meta["func_name"]
                .as_str()
                .filter(|f| !f.is_empty())
                .map(str::to_string),
            public: s("public_test_cases"),
            private: s("private_test_cases"),
        });
    }
    Ok(out)
}

/// Deterministic case order: by (contest date, question id) within each difficulty, then interleaved
/// easy/medium/hard, so the first N of any limit are stratified by difficulty.
pub fn order(mut ps: Vec<Problem>) -> Vec<Problem> {
    ps.sort_by(|a, b| {
        (a.contest_date.as_str(), a.question_id.as_str())
            .cmp(&(b.contest_date.as_str(), b.question_id.as_str()))
    });
    let mut groups: BTreeMap<u8, std::collections::VecDeque<Problem>> = BTreeMap::new();
    for p in ps {
        let k = match p.difficulty.as_str() {
            "easy" => 0,
            "medium" => 1,
            "hard" => 2,
            _ => 3,
        };
        groups.entry(k).or_default().push_back(p);
    }
    let mut out = Vec::new();
    loop {
        let mut any = false;
        for g in groups.values_mut() {
            if let Some(p) = g.pop_front() {
                out.push(p);
                any = true;
            }
        }
        if !any {
            break;
        }
    }
    out
}

pub fn prompt(p: &Problem) -> Vec<Value> {
    let fmt = if p.fn_name.is_some() || !p.starter_code.trim().is_empty() {
        format!(
            "{FORMAT_FUNCTIONAL}\n```python\n{}\n```",
            p.starter_code.trim_end()
        )
    } else {
        format!("{FORMAT_STDIN}\n```python\n# your code here\n```")
    };
    vec![
        json!({"role": "system", "content": SYSTEM}),
        json!({"role": "user", "content": format!("### Problem\n{}\n\n### Format\n{fmt}\n\n### Answer", p.content.trim_end())}),
    ]
}

/// The code to run: the last fenced block tagged python/py/python3 or untagged; else the last fenced block.
pub fn extract_code(s: &str) -> Option<String> {
    let mut blocks: Vec<(String, String)> = Vec::new();
    let mut cur: Option<(String, Vec<&str>)> = None;
    for line in s.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("```") {
            match cur.take() {
                None => cur = Some((rest.trim().to_lowercase(), Vec::new())),
                Some((lang, body)) => blocks.push((lang, body.join("\n"))),
            }
        } else if let Some((_, body)) = cur.as_mut() {
            body.push(line);
        }
    }
    let py = |l: &str| matches!(l, "" | "python" | "py" | "python3");
    blocks
        .iter()
        .rev()
        .find(|(l, b)| py(l) && !b.trim().is_empty())
        .or_else(|| blocks.iter().rev().find(|(_, b)| !b.trim().is_empty()))
        .map(|(_, b)| b.clone())
}

fn tests_dir(cache: &Path, d: &DatasetPin) -> PathBuf {
    let base = cache.join("lcb-tests").join(&d.sha256[..12]);
    if std::fs::create_dir_all(&base).is_ok() {
        return base;
    }
    let t = std::env::temp_dir().join(format!("talus-bench-lcb-tests-{}", &d.sha256[..12]));
    let _ = std::fs::create_dir_all(&t);
    t
}

/// Write (once) the problem's tests, as given by the dataset, to `dir/<question_id>.json`; returns (path, sha256).
fn write_tests(dir: &Path, p: &Problem) -> Result<(PathBuf, String), String> {
    let body = json!({"public": p.public, "private": p.private}).to_string();
    let h = sha256_hex(body.as_bytes());
    let safe: String = p
        .question_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let path = dir.join(format!("{safe}.json"));
    let fresh = std::fs::metadata(&path)
        .map(|m| m.len() != body.len() as u64)
        .unwrap_or(true);
    if fresh {
        let tmp = path.with_extension(format!("part{}", std::process::id()));
        std::fs::write(&tmp, &body).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok((path, h))
}

/// The sandbox can run here: docker answers and the pinned image is present (nothing is pulled).
pub fn check_sandbox(sb: &Sandbox) -> Result<(), String> {
    if sb.executor == "local" {
        return if cfg!(test) {
            Ok(())
        } else {
            Err(format!("{NOT_SUPPORTED} livecodebench executor \"local\" runs model code unsandboxed and exists for the unit tests only"))
        };
    }
    if sb.executor != "docker" {
        return Err(format!(
            "livecodebench: unknown executor {:?} (docker)",
            sb.executor
        ));
    }
    let o = Command::new(&sb.docker)
        .args(["image", "inspect", "--format", "{{.Id}}", &sb.image])
        .output()
        .map_err(|e| {
            format!(
                "{NOT_SUPPORTED} livecodebench needs docker for its sandbox ({}: {e})",
                sb.docker
            )
        })?;
    if !o.status.success() {
        return Err(format!(
            "{NOT_SUPPORTED} livecodebench sandbox image {} is not present; run `talus-bench lcb-prep` once (pulls it; no pull at run time)",
            sb.image
        ));
    }
    Ok(())
}

/// Pull the pinned sandbox image (the only network step) and check it runs offline.
pub fn prep(sb: &Sandbox) -> Result<String, String> {
    let o = Command::new(&sb.docker)
        .args(["pull", &sb.image])
        .output()
        .map_err(|e| format!("{}: {e}", sb.docker))?;
    if !o.status.success() {
        return Err(format!(
            "docker pull {} failed: {}",
            sb.image,
            String::from_utf8_lossy(&o.stderr).trim()
        ));
    }
    let job = json!({"code": "print(int(input()) * 2)", "mode": "stdin", "fn_name": null,
                     "public": [{"input": "21\n", "output": "42\n"}], "private": [], "timeout_s": 6, "demote": true});
    let r = execute(sb, &job.to_string())?;
    if r["passed"] != 1 {
        return Err(format!("sandbox self-check failed: {r}"));
    }
    Ok(format!(
        "{} present; offline self-check passed ({} s)",
        sb.image, r["wall_s"]
    ))
}

static SEQ: AtomicU64 = AtomicU64::new(0);

/// Run the runner on `job` (JSON) in the sandbox; returns its result object.
pub fn execute(sb: &Sandbox, job: &str) -> Result<Value, String> {
    let name = format!(
        "tb-lcb-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let mut cmd = match sb.executor.as_str() {
        "docker" => {
            let mut c = Command::new(&sb.docker);
            c.args(sb.docker_args(&name));
            c
        }
        "local" if cfg!(test) => {
            let mut c = Command::new("python3");
            c.args(["-I", "-B", "-c", RUNNER_PY]);
            c
        }
        e => return Err(format!("livecodebench: executor {e:?} is not available")),
    };
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start the sandbox ({:?}): {e}", cmd.get_program()))?;
    let mut sin = child.stdin.take().expect("piped");
    let mut sout = child.stdout.take().expect("piped");
    let mut serr = child.stderr.take().expect("piped");
    let body = job.as_bytes().to_vec();
    let tw = std::thread::spawn(move || {
        let _ = sin.write_all(&body);
    });
    let to = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = sout.read_to_string(&mut s);
        s
    });
    let te = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = serr.read_to_string(&mut s);
        s
    });
    let t0 = Instant::now();
    let limit = Duration::from_secs(sb.eval_timeout_s.max(1));
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break Some(st),
            Ok(None) => {}
            Err(e) => return Err(e.to_string()),
        }
        if t0.elapsed() > limit {
            if sb.executor == "docker" {
                let _ = Command::new(&sb.docker).args(["rm", "-f", &name]).output();
            }
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let _ = tw.join();
    let out = to.join().unwrap_or_default();
    let err = te.join().unwrap_or_default();
    let Some(st) = status else {
        return Ok(json!({"eval_timeout": true, "wall_s": t0.elapsed().as_secs_f64()}));
    };
    let last = out.lines().rev().find(|l| l.trim_start().starts_with('{'));
    match last.and_then(|l| serde_json::from_str::<Value>(l).ok()) {
        Some(v) => Ok(v),
        None => {
            let tail: String = err.lines().rev().take(4).collect::<Vec<_>>().join(" | ");
            Err(format!("sandbox exited {st} without a result: {tail}"))
        }
    }
}

fn date_of(s: &str) -> &str {
    s.get(..10).unwrap_or(s)
}

impl Task for LcbTask {
    fn id(&self) -> &str {
        "livecodebench"
    }

    fn scorer_version(&self) -> &'static str {
        "livecodebench-sandbox/1"
    }

    fn datasets(&self, _tier: Tier) -> Vec<&'static DatasetPin> {
        let d = *self.chosen.lock().unwrap();
        vec![d.unwrap_or_else(|| datasets::pin(DEFAULT_DATASET).expect("pinned"))]
    }

    fn default_limit(&self, tier: Tier) -> Option<usize> {
        match tier {
            Tier::Smoke => Some(20),
            Tier::Full | Tier::Qualification => None,
        }
    }

    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String> {
        let p = ctx.params;
        let did = param_str(p, "dataset").unwrap_or(DEFAULT_DATASET);
        let d = datasets::pin(did)
            .filter(|d| d.id.starts_with("livecodebench") || d.id.starts_with("lcb-"))
            .ok_or_else(|| format!("livecodebench: unknown dataset {did:?}"))?;
        *self.chosen.lock().unwrap() = Some(d);
        let sb = Sandbox::from_params(p);
        check_sandbox(&sb)?;
        let text = datasets::load(d, ctx.cache)?;
        let all = parse_problems(&text, d.id)?;
        let n_all = all.len();
        let from = param_str(p, "date_from");
        let to = param_str(p, "date_to");
        let diffs = str_list(p, "difficulties");
        let plats = str_list(p, "platforms");
        let sel: Vec<Problem> = all
            .into_iter()
            .filter(|x| from.is_none_or(|f| date_of(&x.contest_date) >= f))
            .filter(|x| to.is_none_or(|t| date_of(&x.contest_date) <= t))
            .filter(|x| diffs.as_ref().is_none_or(|v| v.contains(&x.difficulty)))
            .filter(|x| plats.as_ref().is_none_or(|v| v.contains(&x.platform)))
            .collect();
        let n_window = sel.len();
        let ordered = take_limit_problems(order(sel), ctx.limit);
        let sampling = param_str(p, "sampling").unwrap_or("shipped");
        let request = match sampling {
            "shipped" => json!({"temperature": null, "top_p": null}),
            "greedy" => json!({"temperature": 0.0}),
            "config" => Value::Null,
            s => {
                return Err(format!(
                    "livecodebench: sampling {s:?} (shipped | greedy | config)"
                ))
            }
        };
        let max_tokens = p["max_tokens"].as_u64().unwrap_or(16384) as u32;
        let cutoff = param_str(p, "model_cutoff");
        let dir = tests_dir(ctx.cache, d);
        let mut out = Vec::new();
        let mut per_diff: BTreeMap<String, usize> = BTreeMap::new();
        for x in &ordered {
            let (path, h) = write_tests(&dir, x)?;
            *per_diff.entry(x.difficulty.clone()).or_default() += 1;
            let functional = x.fn_name.is_some();
            out.push(Case {
                task: self.id().into(),
                case_id: format!("livecodebench/{}", x.question_id),
                dataset: d.id.into(),
                dataset_revision: format!("{}@{}", d.revision, &d.sha256[..12]),
                messages: prompt(x),
                expected: json!({"question_id": x.question_id, "title": x.title, "platform": x.platform,
                                 "contest_id": x.contest_id, "contest_date": x.contest_date,
                                 "difficulty": x.difficulty,
                                 "testtype": if functional { "functional" } else { "stdin" },
                                 "fn_name": x.fn_name, "tests_sha256": h,
                                 "after_cutoff": cutoff.map(|c| date_of(&x.contest_date) > c)}),
                meta: json!({"tests_file": path, "sandbox": sb.to_value()}),
                critical: false,
                max_tokens: Some(max_tokens),
                tools: Value::Null,
                request: request.clone(),
            });
        }
        *self.prov.lock().unwrap() = json!({
            "dataset": d.id, "revision": d.revision, "sha256": d.sha256, "license": d.license,
            "problems_in_release": n_all, "in_window": n_window, "cases": out.len(),
            "window": {"date_from": from, "date_to": to, "difficulties": diffs, "platforms": plats},
            "per_difficulty": per_diff, "sampling": sampling, "max_tokens": max_tokens,
            "model_cutoff": cutoff, "sandbox": sb.to_value(),
            "order": "by contest date then id within difficulty, interleaved easy/medium/hard",
        });
        Ok(out)
    }

    fn score(&self, case: &Case, out: &RawOutput) -> Score {
        let e = &case.expected;
        let mut details = json!({
            "question_id": e["question_id"], "difficulty": e["difficulty"], "platform": e["platform"],
            "testtype": e["testtype"], "contest_date": e["contest_date"], "after_cutoff": e["after_cutoff"],
        });
        let fail = |mut d: Value, stage: &str| {
            d["stage"] = json!(stage);
            d["passed_ignoring_flags"] = json!(false);
            Score::binary(false, Value::Null, d)
        };
        let Some(code) = extract_code(&out.content) else {
            return fail(details, "no_code");
        };
        let tests = match case.meta["tests_file"]
            .as_str()
            .ok_or("no tests_file".to_string())
            .and_then(|f| std::fs::read_to_string(f).map_err(|err| format!("{f}: {err}")))
        {
            Ok(t) if sha256_hex(t.as_bytes()) == e["tests_sha256"].as_str().unwrap_or("") => t,
            Ok(_) => {
                details["error"] = json!("tests file does not match its sha256");
                return fail(details, "sandbox_error");
            }
            Err(err) => {
                details["error"] = json!(err);
                return fail(details, "sandbox_error");
            }
        };
        let t: Value = serde_json::from_str(&tests).unwrap_or(Value::Null);
        let sb = Sandbox::from_value(&case.meta["sandbox"]);
        let job = json!({
            "code": code, "mode": e["testtype"], "fn_name": e["fn_name"],
            "public": t["public"], "private": t["private"],
            "timeout_s": sb.test_timeout_s, "demote": true,
        });
        let r = match execute(&sb, &job.to_string()) {
            Ok(r) => r,
            Err(err) => {
                details["error"] = json!(err);
                return fail(details, "sandbox_error");
            }
        };
        details["code_sha256"] = json!(sha256_hex(code.as_bytes()));
        details["eval_s"] = r["wall_s"].clone();
        if r["eval_timeout"] == true {
            details["error"] = json!(format!("evaluation exceeded {} s", sb.eval_timeout_s));
            return fail(details, "timeout");
        }
        if !r["decode_error"].is_null() {
            details["error"] = r["decode_error"].clone();
            return fail(details, "sandbox_error");
        }
        for k in [
            "tests_total",
            "tests_run",
            "passed",
            "first_fail",
            "max_test_ms",
        ] {
            details[k] = r[k].clone();
        }
        let flags: Vec<Value> = r["flags"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|f| json!({"kind": f["kind"], "event": f["event"], "detail": f["detail"], "source": "audit_hook"}))
            .collect();
        let total = r["tests_total"].as_u64().unwrap_or(0);
        let all = total > 0 && r["passed"].as_u64() == Some(total) && r["first_fail"].is_null();
        details["passed_ignoring_flags"] = json!(all);
        let stage = if !flags.is_empty() {
            "sandbox_violation"
        } else if all {
            "pass"
        } else if total == 0 {
            "sandbox_error"
        } else {
            r["first_fail"]["kind"].as_str().unwrap_or("wrong_answer")
        };
        details["flags"] = json!(flags);
        details["stage"] = json!(stage);
        Score::binary(stage == "pass", json!(code), details)
    }

    fn provenance(&self) -> Value {
        self.prov.lock().unwrap().clone()
    }

    fn caveats(&self) -> Vec<String> {
        vec![
            "livecodebench: the newest published release (v6, contests 2025-01..2025-04) predates the knowledge cutoff of the models served here, so absolute pass@1 may include memorized problems; the paired reference-vs-candidate difference is what the gate uses (set params.model_cutoff to split before/after)".into(),
            "livecodebench: grading is a reimplementation of LiveCodeBench's rules in a Python 3.12 stdlib-only sandbox (no numpy/sortedcontainers); absolute numbers are not leaderboard numbers".into(),
            "livecodebench: per-test time limits are wall-clock on a shared host; a loaded host can turn a slow solution into a timeout (the reference and candidate are scored on the same host)".into(),
        ]
    }
}

fn take_limit_problems(mut v: Vec<Problem>, limit: Option<usize>) -> Vec<Problem> {
    if let Some(n) = limit {
        v.truncate(n);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn python_ok() -> bool {
        Command::new("python3")
            .arg("-c")
            .arg("import sys; assert sys.version_info >= (3, 9)")
            .status()
            .is_ok_and(|s| s.success())
    }

    fn fixture_cases(extra: Value) -> Vec<Case> {
        let mut p = json!({"dataset": "lcb-fixture-v1", "executor": "local", "test_timeout_s": 2});
        for (k, v) in extra.as_object().into_iter().flatten() {
            p[k] = v.clone();
        }
        LcbTask::default()
            .cases(&TaskContext {
                params: &p,
                ..TaskContext::for_tests(Tier::Full)
            })
            .unwrap()
    }

    fn answer(code: &str) -> RawOutput {
        RawOutput {
            content: format!("Here is my solution.\n```python\n{code}\n```\nDone."),
            finish_reason: "stop".into(),
            ..Default::default()
        }
    }

    pub(crate) const ADD_OK: &str = "a, b = map(int, input().split())\nprint(a + b)";
    pub(crate) const MAXSUB_OK: &str = "class Solution:\n    def maxSubArray(self, nums: List[int]) -> int:\n        best = cur = nums[0]\n        for x in nums[1:]:\n            cur = max(x, cur + x)\n            best = max(best, cur)\n        return best";
    pub(crate) const HALF_OK: &str = "n = int(input())\nprint(n / 2)";
    /// Tries the network, a file write and a process spawn, then answers correctly.
    pub(crate) const ESCAPE: &str = "import socket, os\ntry:\n    socket.create_connection(('127.0.0.1', 9), timeout=0.5)\nexcept OSError:\n    pass\ntry:\n    open('/tmp/tb-lcb-escape-probe', 'w').write('x')\n    os.remove('/tmp/tb-lcb-escape-probe')\nexcept OSError:\n    pass\ntry:\n    os.system('true')\nexcept Exception:\n    pass\na, b = map(int, input().split())\nprint(a + b)";

    #[test]
    fn code_extraction() {
        assert_eq!(
            extract_code("x\n```python\nprint(1)\n```\ny\n```\nprint(2)\n```").as_deref(),
            Some("print(2)")
        );
        assert_eq!(
            extract_code("```python\nprint(1)\n```\n```text\nsample\n```").as_deref(),
            Some("print(1)"),
            "a later non-Python block does not win"
        );
        assert_eq!(extract_code("no code here"), None);
        assert_eq!(extract_code("```python\nunterminated"), None);
        assert_eq!(
            extract_code("  ```py\n  x = 1\n  ```").as_deref(),
            Some("  x = 1")
        );
    }

    #[test]
    fn fixture_cases_are_built_and_ordered() {
        let cs = fixture_cases(json!({}));
        assert_eq!(cs.len(), 3);
        let diffs: Vec<&str> = cs
            .iter()
            .map(|c| c.expected["difficulty"].as_str().unwrap())
            .collect();
        assert_eq!(diffs, vec!["easy", "medium", "hard"]);
        assert_eq!(cs[0].case_id, "livecodebench/tbfix_add");
        assert_eq!(cs[1].expected["testtype"], "functional");
        assert_eq!(cs[1].expected["fn_name"], "maxSubArray");
        assert!(cs[1].messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("def maxSubArray"));
        assert!(cs[0].messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("standard input"));
        // shipped sampling by default: null temperature/top_p, the server applies the model's sampling
        assert_eq!(cs[0].request, json!({"temperature": null, "top_p": null}));
        assert_eq!(cs[0].max_tokens, Some(16384));
        // the hidden tests never enter the case row
        let row = serde_json::to_string(&cs[0].expected).unwrap();
        assert!(!row.contains("2000000000"), "{row}");
        let g = fixture_cases(json!({"sampling": "greedy", "max_tokens": 4096}));
        assert_eq!(g[0].request, json!({"temperature": 0.0}));
        assert_ne!(g[0].prompt_hash(), cs[0].prompt_hash());
        // window and filters
        let w = fixture_cases(json!({"date_from": "2025-02-01", "date_to": "2025-02-28"}));
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].expected["question_id"], "tbfix_maxsub");
        let f = fixture_cases(json!({"platforms": ["atcoder"], "model_cutoff": "2025-02-15"}));
        assert_eq!(f.len(), 2);
        assert_eq!(f[0].expected["after_cutoff"], false);
        assert_eq!(f[1].expected["after_cutoff"], true);
    }

    #[test]
    fn docker_args_are_locked_down() {
        let a = Sandbox::from_params(&Value::Null)
            .docker_args("n")
            .join(" ");
        for want in [
            "--network none",
            "--read-only",
            "--pull never",
            "--cap-drop ALL",
            "--security-opt no-new-privileges",
            "--memory 2g --memory-swap 2g",
            "--pids-limit 128",
            "--cpus 1",
            DEFAULT_IMAGE,
            "noexec",
        ] {
            assert!(a.contains(want), "{want} missing");
        }
        assert!(!a.contains(" -v ") && !a.contains("--volume") && !a.contains("--privileged"));
    }

    #[test]
    fn sandbox_scoring_correct_wrong_timeout_crash_escape() {
        if !python_ok() {
            eprintln!("python3 >= 3.9 not found: skipping the local-executor scoring test");
            return;
        }
        let cs = fixture_cases(json!({"test_timeout_s": 1}));
        let t = LcbTask::default();
        let (add, maxsub, half) = (&cs[0], &cs[1], &cs[2]);
        let s = t.score(add, &answer(ADD_OK));
        assert!(s.pass, "{}", s.details);
        assert_eq!(s.details["stage"], "pass");
        assert_eq!(s.details["tests_total"], 4, "public + decoded hidden tests");
        let s = t.score(maxsub, &answer(MAXSUB_OK));
        assert!(s.pass, "{}", s.details);
        assert_eq!(s.details["tests_total"], 4);
        // decimal comparison: 3.5 matches the expected 3.50
        let s = t.score(half, &answer(HALF_OK));
        assert!(s.pass, "{}", s.details);
        // wrong
        let s = t.score(
            add,
            &answer("a, b = map(int, input().split())\nprint(a - b)"),
        );
        assert!(!s.pass);
        assert_eq!(s.details["stage"], "wrong_answer");
        let s = t.score(
            maxsub,
            &answer("class Solution:\n    def maxSubArray(self, nums):\n        return sum(nums)"),
        );
        assert_eq!(s.details["stage"], "wrong_answer");
        // passes the public test, fails a hidden one
        let s = t.score(
            add,
            &answer("a, b = map(int, input().split())\nprint(3 if a == 1 else 0)"),
        );
        assert_eq!(s.details["stage"], "wrong_answer");
        assert_eq!(s.details["first_fail"]["public"], false);
        assert_eq!(s.details["passed"], 1);
        // timeout
        let s = t.score(add, &answer("while True:\n    pass"));
        assert!(!s.pass);
        assert_eq!(s.details["stage"], "timeout", "{}", s.details);
        // crash
        let s = t.score(add, &answer("import sys\nsys.exit(3)"));
        assert_eq!(s.details["stage"], "runtime_error");
        let s = t.score(
            maxsub,
            &answer("class Solution:\n    def maxSubArray(self, nums):\n        return 1 // 0"),
        );
        assert_eq!(s.details["stage"], "runtime_error");
        assert!(s.details["first_fail"]["detail"]
            .as_str()
            .unwrap()
            .contains("ZeroDivisionError"));
        let s = t.score(add, &answer("def broken(:\n    pass"));
        assert_eq!(
            s.details["stage"], "runtime_error",
            "a syntax error is a crash"
        );
        // escape attempt: flagged and INVALID even though the answers are right
        let s = t.score(add, &answer(ESCAPE));
        assert!(!s.pass);
        assert_eq!(s.details["stage"], "sandbox_violation", "{}", s.details);
        assert_eq!(s.details["passed_ignoring_flags"], true);
        let kinds: Vec<&str> = s.details["flags"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["kind"].as_str().unwrap())
            .collect();
        for k in ["network", "file_write", "spawn"] {
            assert!(kinds.contains(&k), "{k} not flagged: {kinds:?}");
        }
        // no code
        let s = t.score(
            add,
            &RawOutput {
                content: "I would add them.".into(),
                ..Default::default()
            },
        );
        assert_eq!(s.details["stage"], "no_code");
        // a tampered tests file is refused
        let mut bad = add.clone();
        bad.expected["tests_sha256"] = json!("0".repeat(64));
        assert_eq!(
            t.score(&bad, &answer(ADD_OK)).details["stage"],
            "sandbox_error"
        );
    }

    #[test]
    fn real_release_needs_the_cache_and_the_image() {
        // the default executor is docker: without the image (or docker) the task is NOT_SUPPORTED, never a
        // download
        let p = json!({"docker": "/nonexistent/docker"});
        let e = LcbTask::default()
            .cases(&TaskContext {
                params: &p,
                ..TaskContext::for_tests(Tier::Smoke)
            })
            .unwrap_err();
        assert!(e.starts_with(NOT_SUPPORTED), "{e}");
        let p = json!({"executor": "local"});
        let e = LcbTask::default()
            .cases(&TaskContext {
                params: &p,
                ..TaskContext::for_tests(Tier::Smoke)
            })
            .unwrap_err();
        assert!(e.contains("datasets fetch livecodebench-v6"), "{e}");
    }

    #[test]
    fn metrics_group_unscored_rows_by_their_case() {
        use crate::case::CaseStatus;
        let mut a =
            crate::compare::tests_support::row("m", "livecodebench", "a", true, CaseStatus::Ok);
        a.score.details = json!({"difficulty": "easy", "platform": "atcoder", "testtype": "stdin", "stage": "pass"});
        let mut b = crate::compare::tests_support::row(
            "m",
            "livecodebench",
            "b",
            false,
            CaseStatus::TruncatedByReasoning,
        );
        b.expected =
            json!({"difficulty": "hard", "platform": "leetcode", "testtype": "functional"});
        let m = crate::compare::model_metrics(&[a, b], 0.95);
        let x = &m["livecodebench"]["task_metrics"];
        assert_eq!(x["by_difficulty"]["hard"]["n"], 1, "{x}");
        assert_eq!(x["by_difficulty"]["easy"]["passed"], 1);
        assert_eq!(x["stages"]["not_scored:truncated_by_reasoning"], 1);
        assert!((x["mean_pass_at_1"].as_f64().unwrap() - 0.5).abs() < 1e-9);
    }

    /// A scripted mock model: answers each fixture problem with the given code (by problem title).
    fn mock_model(
        wrong_maxsub: bool,
        seen: std::sync::Arc<Mutex<Vec<Value>>>,
    ) -> std::net::SocketAddr {
        crate::mock::spawn(
            0,
            std::sync::Arc::new(move |m: &str, p: &str, b: &Value| match (m, p) {
                ("GET", "/health") => (200, json!({"status": "ok"})),
                ("GET", "/v1/models") => (200, json!({"object": "list", "data": [{"id": "mock-model"}]})),
                ("POST", "/v1/chat/completions") => {
                    seen.lock().unwrap().push(b.clone());
                    let u = b["messages"].to_string();
                    let code = if u.contains("A+B") {
                        ADD_OK
                    } else if u.contains("contiguous subarray") {
                        if wrong_maxsub {
                            "class Solution:\n    def maxSubArray(self, nums):\n        return max(nums)"
                        } else {
                            MAXSUB_OK
                        }
                    } else {
                        HALF_OK
                    };
                    let content = format!("```python\n{code}\n```");
                    (200, json!({"id": "x", "object": "chat.completion", "model": "mock-model",
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}))
                }
                _ => (404, json!({"error": {"message": "not found"}})),
            }),
        )
        .unwrap()
    }

    #[test]
    fn runner_pairs_reference_and_candidate_with_breakdown() {
        if !python_ok() {
            eprintln!("python3 >= 3.9 not found: skipping");
            return;
        }
        let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
        let a = mock_model(false, seen.clone());
        let b = mock_model(true, seen.clone());
        let root = std::env::temp_dir().join(format!("tb-lcb-run-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let cfg = crate::config::Config::from_yaml(&format!(
            r#"
experiment: {{id: lcb-run, seed: 3}}
reference: {{id: ref, endpoint: "http://{a}"}}
candidates: [{{id: cand, endpoint: "http://{b}"}}]
benchmarks:
  tier: smoke
  tasks: [livecodebench]
  dataset_cache: "{}"
  params: {{livecodebench: {{dataset: lcb-fixture-v1, executor: local, model_cutoff: "2025-01-31"}}}}
quality_gates: {{min_cases: 2}}
"#,
            root.join("cache").display()
        ))
        .unwrap();
        let out = root.join("run");
        let o = crate::runner::run(
            &cfg,
            &crate::runner::RunOptions {
                output_dir: Some(out.clone()),
                ..Default::default()
            },
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            &mut |_| {},
        )
        .unwrap();
        let base = crate::case::read_jsonl(&out.join("baseline/cases.jsonl")).unwrap();
        let cand = crate::case::read_jsonl(&out.join("candidates/cand/cases.jsonl")).unwrap();
        assert_eq!(base.len(), 3);
        assert!(
            base.iter().all(|r| r.pass()),
            "{:?}",
            base.iter().map(|r| &r.score.details).collect::<Vec<_>>()
        );
        assert_eq!(cand.iter().filter(|r| r.pass()).count(), 2);
        // shipped sampling reached the server as null temperature/top_p
        for req in seen.lock().unwrap().iter() {
            assert!(
                req["temperature"].is_null() && req.get("temperature").is_some(),
                "{req}"
            );
            assert_eq!(req["max_tokens"], 16384);
        }
        // no hidden test (or its expected output) in any artifact row
        let rows = std::fs::read_to_string(out.join("baseline/cases.jsonl")).unwrap();
        assert!(!rows.contains("2000000000"));
        let m: Value = serde_json::from_str(
            &std::fs::read_to_string(out.join("candidates/cand/metrics.json")).unwrap(),
        )
        .unwrap();
        let x = &m["livecodebench"]["task_metrics"];
        assert_eq!(x["by_difficulty"]["medium"]["passed"], 0);
        assert_eq!(x["by_difficulty"]["easy"]["passed"], 1);
        assert!(x["by_difficulty"]["easy"]["wilson"]["lo"].is_number());
        assert_eq!(x["by_cutoff"]["after"]["n"], 2);
        assert_eq!(x["stages"]["wrong_answer"], 1);
        let md = std::fs::read_to_string(out.join("report.md")).unwrap();
        assert!(md.contains("livecodebench pass@1"), "{md}");
        assert!(
            md.contains("livecodebench accuracy loss") || md.contains("livecodebench"),
            "{md}"
        );
        let ds = std::fs::read_to_string(out.join("datasets.json")).unwrap();
        assert!(
            ds.contains("lcb-fixture-v1") && ds.contains("\"sampling\": \"shipped\""),
            "{ds}"
        );
        assert!(!o.verdicts.is_empty());
        std::fs::remove_dir_all(&root).ok();
    }

    /// Opt-in (TALUS_BENCH_DOCKER_TESTS=1, needs docker and the pinned image from `talus-bench lcb-prep`; pulls
    /// nothing): the real container blocks the network, the root filesystem, the tests in the runner's memory,
    /// fork bombs and memory hogs.
    #[test]
    fn docker_sandbox_enforces() {
        if std::env::var("TALUS_BENCH_DOCKER_TESTS").as_deref() != Ok("1") {
            return;
        }
        let mut cs = fixture_cases(json!({}));
        for c in cs.iter_mut() {
            c.meta["sandbox"]["executor"] = json!("docker");
            c.meta["sandbox"]["test_timeout_s"] = json!(3);
            c.meta["sandbox"]["memory"] = json!("512m");
        }
        let t = LcbTask::default();
        let add = &cs[0];
        let s = t.score(add, &answer(ADD_OK));
        assert!(s.pass, "{}", s.details);
        let s = t.score(&cs[1], &answer(MAXSUB_OK));
        assert!(s.pass, "{}", s.details);
        // network: every attempt fails inside the container, and is flagged
        let net = "import socket\nok = 0\nfor h in [('1.1.1.1', 443), ('8.8.8.8', 53), ('140.82.112.3', 443), ('192.168.0.1', 80)]:\n    try:\n        socket.create_connection(h, timeout=2); ok += 1\n    except OSError:\n        pass\na, b = map(int, input().split())\nprint(a + b + ok)";
        let s = t.score(add, &answer(net));
        assert_eq!(s.details["stage"], "sandbox_violation", "{}", s.details);
        assert_eq!(
            s.details["passed_ignoring_flags"], true,
            "no connection succeeded: {}",
            s.details
        );
        // the root filesystem is read-only; a write is flagged and fails
        let w = "ok = 0\nfor p in ['/etc/tb-probe', '/usr/tb-probe', '/tb-probe']:\n    try:\n        open(p, 'w').write('x'); ok += 1\n    except OSError:\n        pass\na, b = map(int, input().split())\nprint(a + b + ok)";
        let s = t.score(add, &answer(w));
        assert_eq!(
            s.details["passed_ignoring_flags"], true,
            "no write succeeded: {}",
            s.details
        );
        assert_eq!(s.details["stage"], "sandbox_violation");
        // the expected outputs are not reachable: other processes' memory, fds, environment and files (the
        // probe builds its needle at run time, else it would find it in its own source file)
        let leak = "import os, glob\nT = ('2' + '0' * 9).encode()\nfound = 0\nfor p in glob.glob('/proc/[0-9]*'):\n    if p == '/proc/%d' % os.getpid():\n        continue\n    for f in ['environ', 'cmdline', 'mem', 'fd/0', 'fd/1']:\n        try:\n            with open(p + '/' + f, 'rb') as h:\n                d = h.read(1 << 20)\n            if T in d:\n                found += 1\n        except OSError:\n            pass\nfor f in glob.glob('/tmp/**', recursive=True):\n    try:\n        if T in open(f, 'rb').read():\n            found += 1\n    except OSError:\n        pass\na, b = map(int, input().split())\nprint(a + b + found)";
        let s = t.score(add, &answer(leak));
        assert!(
            s.pass,
            "the solution found the expected outputs: {}",
            s.details
        );
        // a fork bomb is contained (pids limit) and the case ends
        let bomb = "import os\nwhile True:\n    try:\n        os.fork()\n    except OSError:\n        pass";
        let t0 = Instant::now();
        let s = t.score(add, &answer(bomb));
        assert!(!s.pass);
        assert!(t0.elapsed() < Duration::from_secs(30), "{:?}", t0.elapsed());
        // a memory hog is killed
        let hog = "x = bytearray(3 << 30)\nprint(len(x))";
        let s = t.score(add, &answer(hog));
        assert!(!s.pass);
        assert!(
            matches!(
                s.details["stage"].as_str(),
                Some("killed") | Some("runtime_error")
            ),
            "{}",
            s.details
        );
        // no container is left behind
        let o = Command::new("docker")
            .args([
                "ps",
                "-a",
                "-q",
                "--filter",
                "label=talus-bench=livecodebench",
            ])
            .output()
            .unwrap();
        assert!(String::from_utf8_lossy(&o.stdout).trim().is_empty());
    }
}
