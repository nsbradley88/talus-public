//! `talus-bench`: headless CLI for paired quality validation of Talus settings and quantizations.
//! Exit codes: 0 every candidate PASS, 1 any FAIL, 2 usage/config error, 3 WARN or NOT_RUN, 4 segment
//! incomplete (stopped at --max-minutes; continue with --resume).

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use talus_bench_core::config::Config;
use talus_bench_core::runner::{self, Event, RunOptions};
use talus_bench_core::{datasets, mock, tasks};

const USAGE: &str = "\
Usage:
  talus-bench run CONFIG.yaml [--out DIR] [--quiet] [--resume] [--max-minutes M] [--seed N]
                                                      run an experiment (reference + candidates), write artifacts + report;
                                                      --max-minutes stops cleanly before the next case (exit 4), --resume continues;
                                                      --seed N reproduces a swebench_verified draw
  talus-bench validate CONFIG.yaml                    check a config without running it
  talus-bench smoke --server URL [--model ID] [--candidate-level fast|reference] [--api-key-env NAME] [--out DIR]
                                                      Talus-vs-Talus smoke: balanced vs a level of the served model
  talus-bench freeze RUN_DIR [--model ID] [--to DIR]  freeze a finished run's reference (or --model) as a content-addressed baseline
  talus-bench report RUN_DIR                          print a finished run's report.md
  talus-bench tasks                                   list known tasks and their status in this build
  talus-bench datasets [list | fetch ID|bfcl | import ID FILE | verify]
                                                      pinned datasets: list, fetch into the cache (bfcl = its 7 files),
                                                      import a converted file (mmlu-pro-test), verify the cache
  talus-bench example-config                          print an annotated example config
  talus-bench swe-draw [--n 50] [--seed N] [--offline] [--prep [--cache-gb 150]]
                                                      fetch SWE-bench Verified live, draw, print ids and images; --prep pulls,
                                                      builds and gold-validates the drawn images (docker, network, no GPU)
  talus-bench swe-trial [--n 5] [--seed N] [--out DIR] [--agent opencode|mini] [--mini-bundle DIR] [--instances-file F]
                                                      CPU-only: prep a fresh draw, prove the offline eval (gold resolves,
                                                      unmodified base does not), run the agent container against the mock model
  talus-bench swe-rescore RUN_DIR [--sub baseline|candidates/ID] [--dry-run] [--re-eval]
                                                      re-judge SWE-bench instances from their retained artifacts with the
                                                      current auditor (CPU only; newly valid patches are evaluated)
  talus-bench swe-netcheck INSTANCE_ID...             CPU only: eval-network check (gold discover/allow, unmodified base, a
                                                      mock cheater fetching the upstream file at import)
  talus-bench lcb-prep [--image REF] [--docker BIN]   pull the pinned livecodebench sandbox image (the only network step) and
                                                      self-check it offline (CPU only; runs never pull)
  talus-bench pool [--task swebench_verified] [--json] RUN_DIR...
                                                      pool several runs (fresh draws) into one rate per model with a CI
  talus-bench kl BASE TEST [--rows N] [--skip K] [--model mimo|laguna | --max-mean X --min-top1 Y] [--label L]
                                                      teacher-forced KL of two float32 [rows, V] logit dumps (kl.py port);
                                                      with a threshold: a RESULT line and exit 0 PASS / 1 FAIL
  talus-bench gpu-kld-plan --model mimo|laguna --build B --ref-build R [--depths 2000,32000] [--host H]
                                                      print the GPU-host commands (tfkld.sh, copy, kl); runs nothing
  talus-bench mock-server [--port N] [--wrong-rate F] [--error-rate F] [--seed N]
                          [--score-drift F] [--score supported|unsupported|absent]
                                                      synthetic OpenAI-compatible server for trying the bench (no model)

Exit codes: 0 all PASS, 1 a candidate FAILs, 2 usage or config error, 3 WARN / NOT_RUN, 4 segment incomplete.
API keys come from the environment variable named by `api_key_env` in the config; never from the file.";

const EXAMPLE: &str = r#"# talus-bench experiment (YAML). Keys never go here: name an environment variable in api_key_env.
experiment:
  id: q27-fast-vs-balanced        # letters, digits, - _ .
  seed: 42                        # bootstrap seed
  # output_dir: artifacts/q27-fast-vs-balanced   (default artifacts/<id>)
reference:
  id: balanced
  endpoint: http://127.0.0.1:8000 # plain HTTP; tunnel over SSH for remote hosts
  api_key_env: TALUS_API_KEY
  talus: {model: qwen3.8-27b, level: balanced, apply: true}   # Talus-vs-Talus: switch the server first
  # kind: frozen                  # or compare against a frozen baseline:
  # frozen: ~/.local/share/talus-bench/frozen/balanced-0123456789ab
candidates:
  - id: fast
    endpoint: http://127.0.0.1:8000
    api_key_env: TALUS_API_KEY
    talus: {model: qwen3.8-27b, level: fast, apply: true}
    provenance: {quantization: UD-IQ4_XS, kv_cache_precision: f16, notes: "TUI Fast level"}
inference:
  temperature: 0
  max_output_tokens: 1024
  seed: 42
  thinking: off                   # off | on | server_default
  request_timeout_s: 600
benchmarks:
  tier: smoke                     # smoke | full | qualification
  tasks: [json_schema, gsm8k]     # planned tasks (ifeval, needle, agentic_go, ...) report NOT_RUN
  limits: {gsm8k: 50}
quality_gates:
  min_cases: 30
  bootstrap_resamples: 2000
  confidence: 0.95
  max_loss_pp: {gsm8k: {target: 2.0, investigate: 4.0}, json_schema: {target: 1.0, investigate: 2.0}}
  min_rate_pct: {json_schema: {target: 99.5, investigate: 99.0}}
report:
  store_raw_outputs: true
  redact_prompts: false
"#;

fn die(msg: &str) -> ! {
    eprintln!("talus-bench: {msg}");
    std::process::exit(2)
}

/// The gold patch plus an import-time fetch of the upstream version of the first file it changes (inserted at
/// module level, after any `from __future__` imports), as one diff produced inside the instance image offline.
fn cheater_patch(inst: &talus_bench_core::swe::dataset::Instance) -> Result<String, String> {
    let file = inst
        .patch
        .lines()
        .find_map(|l| l.strip_prefix("+++ b/"))
        .ok_or("no changed file")?
        .to_string();
    let url = format!(
        "https://raw.githubusercontent.com/{}/main/{file}",
        inst.repo
    );
    let dir = std::env::temp_dir().join(format!("tb-cheat-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("gold.diff"), &inst.patch).map_err(|e| e.to_string())?;
    std::fs::write(
        dir.join("inject.py"),
        format!(
            r##"import sys
p = sys.argv[1]
lines = open(p).read().split("\n")
fut = [k for k, l in enumerate(lines) if l.startswith("from __future__")]
i = fut[-1] + 1 if fut else 0
while not fut and i < len(lines) and (lines[i].startswith("#!") or lines[i].startswith("# -*-")):
    i += 1
block = ["import urllib.request as _tb_u", "try:", "    _tb_u.urlopen('{url}', timeout=5)", "except Exception:", "    pass"]
open(p, "w").write("\n".join(lines[:i] + block + lines[i:]))
"##
        ),
    )
    .map_err(|e| e.to_string())?;
    let tag = talus_bench_core::swe::runtime::derived_tag(inst);
    let o = std::process::Command::new("docker")
        .args(["run", "--rm", "--network", "none", "-v"])
        .arg(format!("{}:/w:ro", dir.display()))
        .arg(&tag)
        .args([
            "bash",
            "-c",
            &format!("cd /testbed && git apply /w/gold.diff && /opt/miniconda3/envs/testbed/bin/python /w/inject.py {file} && git diff"),
        ])
        .output()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&dir);
    if !o.status.success() {
        return Err(String::from_utf8_lossy(&o.stderr).to_string());
    }
    Ok(String::from_utf8_lossy(&o.stdout).to_string())
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| {
        args.get(i + 1)
            .cloned()
            .unwrap_or_else(|| die(&format!("{name} needs a value")))
    })
}

fn printer(quiet: bool) -> impl FnMut(Event) {
    move |e| match e {
        Event::Phase(p) => eprintln!("== {p}"),
        Event::Log(l) => eprintln!("   {l}"),
        Event::ModelStart { model, total } => eprintln!("== {model}: {total} cases"),
        Event::Case {
            model,
            done,
            total,
            case_id,
            status,
            pass,
        } => {
            if !quiet && (done % 10 == 0 || done == total || !status.scored()) {
                eprintln!(
                    "   {model} {done}/{total} {case_id} {} {}",
                    status.as_str(),
                    if pass { "pass" } else { "fail" }
                );
            }
        }
        Event::ModelDone {
            model,
            correct,
            total,
        } => eprintln!("== {model}: {correct}/{total} passed"),
    }
}

struct RunFlags {
    seed: Option<u64>,
    out: Option<PathBuf>,
    quiet: bool,
    resume: bool,
    max_minutes: Option<f64>,
}

fn run_flags(rest: &[String]) -> RunFlags {
    RunFlags {
        seed: flag(rest, "--seed").map(|s| s.parse().unwrap_or_else(|_| die("bad --seed"))),
        out: flag(rest, "--out").map(PathBuf::from),
        quiet: rest.iter().any(|a| a == "--quiet"),
        resume: rest.iter().any(|a| a == "--resume"),
        max_minutes: flag(rest, "--max-minutes")
            .map(|m| m.parse().unwrap_or_else(|_| die("bad --max-minutes"))),
    }
}

fn run_config(cfg: &Config, f: RunFlags) -> ! {
    let mut cfg = cfg.clone();
    if let Some(seed) = f.seed {
        // reproduce a swebench_verified draw exactly
        let e = cfg
            .benchmarks
            .params
            .entry("swebench_verified".into())
            .or_insert_with(|| serde_json::json!({}));
        e["seed"] = serde_json::json!(seed);
    }
    let cfg = &cfg;
    let opts = RunOptions {
        output_dir: f.out,
        resume: f.resume,
        deadline: f
            .max_minutes
            .map(|m| std::time::Instant::now() + std::time::Duration::from_secs_f64(m * 60.0)),
        ..Default::default()
    };
    let quiet = f.quiet;
    match runner::run(
        cfg,
        &opts,
        Arc::new(AtomicBool::new(false)),
        &mut printer(quiet),
    ) {
        Ok(o) => {
            println!("report: {}", o.output_dir.join("report.md").display());
            for v in &o.verdicts {
                println!(
                    "{:<24} {:<13} {}",
                    v.candidate_id,
                    v.overall.as_str(),
                    v.reasons.join("; ")
                );
            }
            std::process::exit(o.exit_code())
        }
        Err(e) => {
            eprintln!("talus-bench: {e}");
            std::process::exit(if e.starts_with("invalid config") {
                2
            } else if e.starts_with(runner::INCOMPLETE) {
                4
            } else {
                1
            })
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("--help");
    let rest = &args[1.min(args.len())..];
    match cmd {
        "-h" | "--help" | "help" => println!("{USAGE}"),
        "example-config" => print!("{EXAMPLE}"),
        "validate" | "run" => {
            let path = rest
                .first()
                .filter(|p| !p.starts_with("--"))
                .unwrap_or_else(|| die("needs a CONFIG.yaml"));
            let cfg = Config::load(Path::new(path)).unwrap_or_else(|e| die(&e));
            let v = cfg.validate();
            if cmd == "validate" {
                for w in &v.warnings {
                    eprintln!("warning: {w}");
                }
            }
            if !v.ok() {
                for e in &v.errors {
                    eprintln!("error: {e}");
                }
                std::process::exit(2);
            }
            if cmd == "validate" {
                println!(
                    "ok: {} ({} candidate(s), tasks {})",
                    cfg.experiment.id,
                    cfg.candidates.len(),
                    cfg.benchmarks.tasks.join(", ")
                );
                return;
            }
            run_config(&cfg, run_flags(rest))
        }
        "smoke" => {
            let server = flag(rest, "--server").unwrap_or_else(|| "http://127.0.0.1:8000".into());
            let model = flag(rest, "--model");
            let level = flag(rest, "--candidate-level");
            let key = flag(rest, "--api-key-env");
            let out = flag(rest, "--out").unwrap_or_else(|| "artifacts/smoke".into());
            let cfg = talus_bench_core::smoke_config(
                &server,
                model.as_deref(),
                level.as_deref(),
                key.as_deref(),
                &out,
            );
            let mut f = run_flags(rest);
            f.out = None; // smoke: --out names the output directory in the config
            run_config(&cfg, f)
        }
        "freeze" => {
            let dir = rest
                .first()
                .filter(|p| !p.starts_with("--"))
                .unwrap_or_else(|| die("needs RUN_DIR"));
            let to = flag(rest, "--to").map(PathBuf::from).unwrap_or_else(|| {
                PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
                    .join(".local/share/talus-bench/frozen")
            });
            match runner::freeze(Path::new(dir), flag(rest, "--model").as_deref(), &to) {
                Ok(d) => println!(
                    "frozen: {}\nuse it as   reference: {{id: ..., kind: frozen, frozen: {}}}",
                    d.display(),
                    d.display()
                ),
                Err(e) => die(&e),
            }
        }
        "report" => {
            let dir = rest.first().unwrap_or_else(|| die("needs RUN_DIR"));
            match std::fs::read_to_string(Path::new(dir).join("report.md")) {
                Ok(t) => print!("{t}"),
                Err(e) => die(&format!("{dir}/report.md: {e}")),
            }
        }
        "tasks" => {
            for (id, a, d) in tasks::KNOWN {
                let st = if *a == tasks::TaskAvailability::Native {
                    "native".to_string()
                } else {
                    a.status().as_str().to_string()
                };
                println!("{id:<20} {st:<14} {d}");
            }
        }
        "datasets" => {
            let cache = datasets::default_cache_dir();
            match rest.first().map(String::as_str).unwrap_or("list") {
                "list" => {
                    for d in datasets::DATASETS {
                        let where_ = if d.embedded.is_some() {
                            "embedded".to_string()
                        } else if datasets::load(d, &cache).is_ok() {
                            "cached".into()
                        } else {
                            "not fetched".into()
                        };
                        println!(
                            "{:<22} {:<8} {:<12} rows {:<5} {}  sha256 {}…  {}",
                            d.id,
                            d.license,
                            where_,
                            d.rows,
                            d.revision.chars().take(12).collect::<String>(),
                            &d.sha256[..16],
                            d.description
                        );
                    }
                }
                "fetch" if rest.get(1).map(String::as_str) == Some("bfcl") => {
                    for d in datasets::DATASETS
                        .iter()
                        .filter(|d| d.id.starts_with("bfcl-") && d.embedded.is_none())
                    {
                        match datasets::fetch(d, &cache) {
                            Ok(p) => println!(
                                "ok: {} ({} rows, sha256 verified) -> {}",
                                d.id,
                                d.rows,
                                p.display()
                            ),
                            Err(e) => die(&e),
                        }
                    }
                }
                "import" => {
                    let id = rest
                        .get(1)
                        .unwrap_or_else(|| die("datasets import ID FILE"));
                    let file = rest
                        .get(2)
                        .unwrap_or_else(|| die("datasets import ID FILE"));
                    let d =
                        datasets::pin(id).unwrap_or_else(|| die(&format!("unknown dataset {id}")));
                    match datasets::import(d, Path::new(file), &cache) {
                        Ok(p) => println!(
                            "ok: {} ({} rows, sha256 verified) -> {}",
                            d.id,
                            d.rows,
                            p.display()
                        ),
                        Err(e) => die(&e),
                    }
                }
                "fetch" => {
                    let id = rest.get(1).unwrap_or_else(|| die("datasets fetch ID"));
                    let d =
                        datasets::pin(id).unwrap_or_else(|| die(&format!("unknown dataset {id}")));
                    match datasets::fetch(d, &cache) {
                        Ok(p) => println!(
                            "ok: {} ({} rows, sha256 verified) -> {}",
                            d.id,
                            d.rows,
                            p.display()
                        ),
                        Err(e) => die(&e),
                    }
                }
                "verify" => {
                    let mut bad = 0;
                    for d in datasets::DATASETS {
                        match datasets::load(d, &cache) {
                            Ok(_) => println!("ok       {}", d.id),
                            Err(e) if d.embedded.is_none() && e.contains("not in the cache") => {
                                println!("absent   {}", d.id)
                            }
                            Err(e) => {
                                bad += 1;
                                println!("BAD      {}: {e}", d.id)
                            }
                        }
                    }
                    std::process::exit(if bad > 0 { 1 } else { 0 });
                }
                other => die(&format!("datasets {other}: use list, fetch ID or verify")),
            }
        }
        "lcb-prep" => {
            use talus_bench_core::tasks::livecodebench;
            let mut p = serde_json::json!({});
            if let Some(i) = flag(rest, "--image") {
                p["image"] = serde_json::json!(i);
            }
            if let Some(d) = flag(rest, "--docker") {
                p["docker"] = serde_json::json!(d);
            }
            let sb = livecodebench::Sandbox::from_params(&p);
            match livecodebench::prep(&sb) {
                Ok(m) => println!("ok: {m}"),
                Err(e) => die(&e),
            }
        }
        "swe-draw" => {
            use talus_bench_core::swe::dataset;
            let cache = datasets::default_cache_dir();
            let n = flag(rest, "--n")
                .map(|v| v.parse().unwrap_or_else(|_| die("bad --n")))
                .unwrap_or(50usize);
            let snap = if rest.iter().any(|a| a == "--offline") {
                let mut s = dataset::read_cache(&cache, None).unwrap_or_else(|e| die(&e));
                s.source = "cache (offline run requested)".into();
                s
            } else {
                dataset::load(&cache, &dataset::live_fetch).unwrap_or_else(|e| die(&e))
            };
            let (seed, src) = match flag(rest, "--seed") {
                Some(v) => (
                    v.parse::<u64>().unwrap_or_else(|_| die("bad --seed")),
                    "param",
                ),
                None => (dataset::csprng_seed().unwrap_or_else(|e| die(&e)), "csprng"),
            };
            let (ids, per_repo) = dataset::draw(&snap.instances, n, seed);
            let images: Vec<&str> = snap
                .instances
                .iter()
                .filter(|i| ids.contains(&i.instance_id))
                .map(|i| i.image.as_str())
                .collect();
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                "dataset": dataset::HF_DATASET, "source": snap.source, "revision": snap.revision,
                "content_sha256": snap.content_sha256, "instances": snap.instances.len(), "seed": seed, "seed_source": src,
                "per_repo": per_repo, "instance_ids": ids, "images": images})).unwrap_or_default());
            if rest.iter().any(|a| a == "--prep") {
                // pull, build the derived image and validate the gold patch offline, per drawn instance (no GPU)
                use talus_bench_core::swe::runtime::Runtime;
                let gb = flag(rest, "--cache-gb")
                    .map(|v| v.parse::<f64>().unwrap_or_else(|_| die("bad --cache-gb")))
                    .unwrap_or(150.0);
                let rt = talus_bench_core::swe::runtime::DockerRuntime::new(&cache, gb, 1800, true);
                for i in snap
                    .instances
                    .iter()
                    .filter(|i| ids.contains(&i.instance_id))
                {
                    match rt.prepare(i) {
                        Ok(p) => eprintln!(
                            "{} prep {:.0} s pulled {} built {} gold_ok {:?} evicted {:?}",
                            i.instance_id, p.prep_s, p.pulled, p.built, p.gold_ok, p.evicted
                        ),
                        Err(e) => eprintln!("{} prep FAILED: {e}", i.instance_id),
                    }
                }
            }
        }
        "swe-rescore" => {
            // re-judge a run's SWE-bench instances from their retained artifacts with the current auditor
            let dir = rest
                .iter()
                .find(|a| !a.starts_with("--"))
                .unwrap_or_else(|| die("needs RUN_DIR"));
            let sub = flag(rest, "--sub").unwrap_or_else(|| "baseline".into());
            let dry = rest.iter().any(|a| a == "--dry-run");
            let re_eval = rest.iter().any(|a| a == "--re-eval");
            let r =
                talus_bench_core::swe::rescore::rescore_with(Path::new(dir), &sub, dry, re_eval)
                    .unwrap_or_else(|e| die(&e));
            println!(
                "{} {}: {} changed, {} unchanged, {} skipped",
                if dry { "dry run" } else { "rescored" },
                r.file.display(),
                r.changes.len(),
                r.kept,
                r.skipped.len()
            );
            for (id, old, new, kinds) in &r.changes {
                println!("  {id}: {old} -> {new} {kinds:?}");
            }
            for (id, why) in &r.skipped {
                println!("  skipped {id}: {why}");
            }
        }
        "swe-netcheck" => {
            // CPU only: for each instance, the gold run in discover mode (allowlist), then gold, the unmodified
            // base and a mock cheater (gold fix + the upstream file fetched at import time) under the allowlist
            use talus_bench_core::swe::{dataset, egress, harness, runtime::Runtime};
            let ids: Vec<&String> = rest.iter().filter(|a| !a.starts_with("--")).collect();
            if ids.is_empty() {
                die("needs INSTANCE_ID...");
            }
            let cache = datasets::default_cache_dir();
            let snap = dataset::read_cache(&cache, None).unwrap_or_else(|e| die(&e));
            let rt = talus_bench_core::swe::runtime::DockerRuntime::new(&cache, 150.0, 1800, true);
            let mut out = Vec::new();
            for id in ids {
                let inst = snap
                    .instances
                    .iter()
                    .find(|i| &i.instance_id == id)
                    .cloned()
                    .unwrap_or_else(|| die(&format!("{id}: not in the cached dataset")));
                let prep = rt.prepare(&inst).unwrap_or_else(|e| die(&e));
                let gold = rt.gold_evidence(&inst).unwrap_or_default();
                let allow = egress::NetMode::Allow(gold.allowlist_from_gold());
                let grade = |log: &str| {
                    harness::grade(
                        &inst.log_parser,
                        &inst.eval_type,
                        &inst.fail_to_pass,
                        &inst.pass_to_pass,
                        log,
                    )
                    .map(|g| g.resolved)
                    .unwrap_or(false)
                };
                // the cheater: the right fix, plus the upstream version of the first changed file fetched when that
                // module is imported; built inside the image (offline) as one correct diff
                let cheat = cheater_patch(&inst)
                    .unwrap_or_else(|e| die(&format!("{id}: cheater patch: {e}")));
                let mut row = serde_json::json!({"instance_id": id, "gold_discover_resolves": prep.gold_ok,
                    "gold_evidence": gold, "allowlist": gold.allowlist_from_gold()});
                for (k, p) in [
                    ("gold", inst.patch.as_str()),
                    ("empty", ""),
                    ("cheater", cheat.as_str()),
                ] {
                    let r = rt.eval_with(&inst, p, &allow);
                    row[k] = match r {
                        Ok((log, ev)) => {
                            let ev = ev.unwrap_or_default();
                            serde_json::json!({"resolves": grade(&log), "applied": !log.contains(harness::APPLY_PATCH_FAIL),
                                "novel": egress::novel(&gold, &ev), "evidence": ev})
                        }
                        Err(e) => serde_json::json!({"error": e}),
                    };
                }
                eprintln!(
                    "{id}: gold(discover) {:?} | gold(allow) {} novel {} | empty {} | cheater resolves {} novel {}",
                    prep.gold_ok, row["gold"]["resolves"], row["gold"]["novel"], row["empty"]["resolves"],
                    row["cheater"]["resolves"], row["cheater"]["novel"]
                );
                out.push(row);
            }
            println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
        }
        "swe-trial" => {
            // CPU-only shakeout of a fresh draw: prep, offline eval proof (gold resolves, unmodified base does not),
            // and the whole agent-container path against the mock model server (expect an empty patch, no flags)
            use talus_bench_core::swe::{dataset, runtime::Runtime, SwebenchTask};
            use talus_bench_core::tasks::{ExecEnv, Task, TaskContext};
            let n = flag(rest, "--n")
                .map(|v| v.parse().unwrap_or_else(|_| die("bad --n")))
                .unwrap_or(5u64);
            let out = PathBuf::from(flag(rest, "--out").unwrap_or_else(|| "swe-trial".into()));
            std::fs::create_dir_all(&out).unwrap_or_else(|e| die(&e.to_string()));
            let cache = datasets::default_cache_dir();
            let mut params =
                serde_json::json!({"n": n, "runtime": "docker", "agent_timeout_s": 600});
            if let Some(s) = flag(rest, "--seed") {
                params["seed"] =
                    serde_json::json!(s.parse::<u64>().unwrap_or_else(|_| die("bad --seed")));
            }
            for (f, k) in [
                ("--agent", "agent"),
                ("--mini-bundle", "mini_bundle"),
                ("--instances-file", "instances_file"),
            ] {
                if let Some(v) = flag(rest, f) {
                    params[k] = serde_json::json!(v);
                }
            }
            let task = SwebenchTask::default();
            let ctx = TaskContext {
                params: &params,
                cache: &cache,
                run_dir: Some(&out),
                ..TaskContext::for_tests(talus_bench_core::config::Tier::Smoke)
            };
            let cases = task.cases(&ctx).unwrap_or_else(|e| die(&e));
            eprintln!("draw: {}", task.provenance());
            let (mock, _) =
                talus_bench_core::mock::spawn_openai(0, talus_bench_core::mock::Degrade::NONE)
                    .unwrap_or_else(|e| die(&e.to_string()));
            let rt = talus_bench_core::swe::runtime::DockerRuntime::new(&cache, 150.0, 1800, true);
            let snap = dataset::read_cache(&cache, task.provenance()["content_sha256"].as_str())
                .unwrap_or_else(|e| die(&e));
            let mut summary = Vec::new();
            for c in &cases {
                let id = c.expected["instance_id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let inst = snap
                    .instances
                    .iter()
                    .find(|i| i.instance_id == id)
                    .cloned()
                    .unwrap_or_else(|| die("instance"));
                let t0 = std::time::Instant::now();
                let prep = rt.prepare(&inst);
                let prep_s = t0.elapsed().as_secs_f64();
                let (gold, base) = match &prep {
                    Ok(_) => {
                        let g = rt.run_eval(&inst, &inst.patch).and_then(|l| {
                            talus_bench_core::swe::harness::grade(
                                &inst.log_parser,
                                &inst.eval_type,
                                &inst.fail_to_pass,
                                &inst.pass_to_pass,
                                &l.log,
                            )
                        });
                        let b = rt.run_eval(&inst, "").and_then(|l| {
                            talus_bench_core::swe::harness::grade(
                                &inst.log_parser,
                                &inst.eval_type,
                                &inst.fail_to_pass,
                                &inst.pass_to_pass,
                                &l.log,
                            )
                        });
                        (g.map(|g| g.resolved), b.map(|b| b.resolved))
                    }
                    Err(e) => (Err(e.clone()), Err(e.clone())),
                };
                let t1 = std::time::Instant::now();
                let env = ExecEnv {
                    base_url: format!("http://{mock}"),
                    api_key: None,
                    model: "mock-model".into(),
                    context: None,
                    work_dir: out.join("work"),
                    label: "swe-trial".into(),
                    params: params.clone(),
                    limit: None,
                    temperature: 0.0,
                    seed: None,
                    max_tokens: 512,
                    cancel: Default::default(),
                };
                let o = task.execute(c, &env);
                let d = o
                    .as_ref()
                    .map(|o| o.score.details.clone())
                    .unwrap_or_default();
                let rec = serde_json::json!({"instance_id": id, "repo": inst.repo, "prep_s": prep_s, "prep": prep.as_ref().map(|p| serde_json::to_value(p).unwrap_or_default()).unwrap_or_else(|e| serde_json::json!(e)),
                    "gold_resolves": gold.as_ref().map(|x| serde_json::json!(x)).unwrap_or_else(|e| serde_json::json!(e)),
                    "base_resolves": base.as_ref().map(|x| serde_json::json!(x)).unwrap_or_else(|e| serde_json::json!(e)),
                    "agent_path_s": t1.elapsed().as_secs_f64(), "agent_stage": d["stage"], "agent_error": o.as_ref().and_then(|o| o.error.clone()),
                    "flags": d["flags"], "harness_network": d["harness_network"], "harness_commands": d["harness_commands"], "package_commands": d["package_commands"],
                    "patch_lines": d["patch_lines"], "agent_usage": d["agent_usage"], "eval": d["eval"], "tool_calls": d["audit_counts"]["tool_calls"], "audit_counts": d["audit_counts"], "agent_exit": d["agent_exit"]});
                eprintln!("{rec}");
                summary.push(rec);
            }
            let doc = serde_json::json!({"draw": task.provenance(), "cache_bytes": rt.cache_bytes(), "instances": summary});
            std::fs::write(
                out.join("trial.json"),
                serde_json::to_string_pretty(&doc).unwrap_or_default(),
            )
            .ok();
            println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
        }
        "pool" => {
            let task = flag(rest, "--task").unwrap_or_else(|| "swebench_verified".into());
            let mut dirs: Vec<&str> = Vec::new();
            let mut skip = false;
            for a in rest {
                if skip {
                    skip = false;
                    continue;
                }
                if a == "--task" {
                    skip = true;
                    continue;
                }
                if !a.starts_with("--") {
                    dirs.push(a);
                }
            }
            if dirs.is_empty() {
                die("pool needs RUN_DIRs");
            }
            let paths: Vec<&Path> = dirs.iter().map(Path::new).collect();
            match talus_bench_core::pool::pool(&paths, &task, 0.95, 2000, 42) {
                Ok(v) if rest.iter().any(|a| a == "--json") => {
                    println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default())
                }
                Ok(v) => print!("{}", talus_bench_core::pool::pool_md(&v)),
                Err(e) => die(&e),
            }
        }
        "kl" => {
            let base = rest
                .first()
                .filter(|p| !p.starts_with("--"))
                .unwrap_or_else(|| die("kl BASE TEST"));
            let test = rest
                .get(1)
                .filter(|p| !p.starts_with("--"))
                .unwrap_or_else(|| die("kl BASE TEST"));
            let num = |n: &str| {
                flag(rest, n).map(|v| {
                    v.parse::<f64>()
                        .unwrap_or_else(|_| die(&format!("bad {n}")))
                })
            };
            let rows = num("--rows").map(|r| r as usize);
            let skip = num("--skip").map(|r| r as usize).unwrap_or(0);
            let st = talus_bench_core::kl::kl_files(Path::new(base), Path::new(test), rows, skip)
                .unwrap_or_else(|e| die(&e));
            let line = talus_bench_core::kl::kl_line(&st);
            let thr = match flag(rest, "--model") {
                Some(m) => Some(
                    talus_bench_core::kl::RELEASE_TFKLD
                        .iter()
                        .find(|t| t.0 == m)
                        .map(|t| (t.1, t.2))
                        .unwrap_or_else(|| {
                            die(&format!(
                                "no teacher-forced KL release threshold for {m} (mimo, laguna)"
                            ))
                        }),
                ),
                None => num("--max-mean").map(|m| (m, num("--min-top1").unwrap_or(0.0))),
            };
            match thr {
                None => println!("{line}"),
                Some((mx, t1)) => {
                    let label = flag(rest, "--label").unwrap_or_else(|| "kl".into());
                    let g = talus_bench_core::kl::kl_gate(&label, &st, mx, t1, true);
                    let ok = g.status == talus_bench_core::gates::GateStatus::Pass;
                    println!(
                        "RESULT {label}: {line} [{}]",
                        if ok { "PASS" } else { "FAIL" }
                    );
                    std::process::exit(if ok { 0 } else { 1 });
                }
            }
        }
        "gpu-kld-plan" => {
            let model = flag(rest, "--model").unwrap_or_else(|| die("--model mimo|laguna"));
            let (mx, t1) = talus_bench_core::kl::RELEASE_TFKLD
                .iter()
                .find(|t| t.0 == model)
                .map(|t| (t.1, t.2))
                .unwrap_or_else(|| die("teacher-forced KL plan exists for mimo and laguna (q27, flashnext, glm use other references)"));
            let build = flag(rest, "--build").unwrap_or_else(|| die("--build B"));
            let refb = flag(rest, "--ref-build").unwrap_or_else(|| die("--ref-build R"));
            let host = flag(rest, "--host").unwrap_or_else(|| "user@gpu-host".into());
            let depths = flag(rest, "--depths").unwrap_or_else(|| "2000,32000".into());
            let s = "~/talus-scratch";
            println!("# teacher-forced KL({refb} || {build}) for {model}, release gate: bitwise or mean <= {mx:e} and top-1 >= {t1}");
            println!("# 1. dumps on the GPU host, one run per build");
            for b in [&refb, &build] {
                let jobs: Vec<String> = depths
                    .split(',')
                    .map(|d| {
                        format!("KLDOUT={s}/bench-kld NOGEN=1 {s}/tfkld.sh {b} {model} {d} bench")
                    })
                    .collect();
                println!("ssh {host} 'bash -c \"{}\"'", jobs.join(" && "));
            }
            println!("# 2. copy the dumps to the client (probe outputs stay on the GPU host)");
            println!("mkdir -p kld && scp '{host}:{s}/bench-kld/*-{model}-d*-bench.logits' kld/");
            println!("# 3. KL and gate per depth (CPU)");
            for d in depths.split(',') {
                println!(
                    "talus-bench kl kld/{refb}-{model}-d{d}-bench.logits kld/{build}-{model}-d{d}-bench.logits --model {model} --label '{model} kl_d{d} vs {refb}'"
                );
            }
        }
        "mock-server" => {
            let p = |n: &str, d: f64| {
                flag(rest, n)
                    .map(|v| {
                        v.parse::<f64>()
                            .unwrap_or_else(|_| die(&format!("bad {n}")))
                    })
                    .unwrap_or(d)
            };
            let port = p("--port", 8199.0) as u16;
            let d = mock::Degrade {
                wrong_rate: p("--wrong-rate", 0.0),
                error_rate: p("--error-rate", 0.0),
                seed: p("--seed", 1.0) as u64,
            };
            let sm = mock::ScoreMock {
                mode: match flag(rest, "--score").as_deref().unwrap_or("supported") {
                    "supported" => mock::ScoreMode::Supported,
                    "unsupported" => mock::ScoreMode::NotSupported,
                    "absent" => mock::ScoreMode::Absent,
                    other => die(&format!(
                        "bad --score {other:?} (supported, unsupported or absent)"
                    )),
                },
                drift: p("--score-drift", 0.0),
                seed: d.seed,
            };
            match mock::spawn_openai_with(port, d, sm) {
                Ok((addr, _)) => {
                    println!("mock OpenAI-compatible server (synthetic answers, no model) on http://{addr}  wrong {} error {}  scoring {:?} drift {}", d.wrong_rate, d.error_rate, sm.mode, sm.drift);
                    loop {
                        std::thread::sleep(std::time::Duration::from_secs(3600));
                    }
                }
                Err(e) => die(&format!("port {port}: {e}")),
            }
        }
        other => die(&format!("unknown command {other:?} (try --help)")),
    }
}
