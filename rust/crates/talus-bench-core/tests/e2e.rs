//! End-to-end runs against mock servers: identical candidate passes, a deliberately degraded candidate fails its
//! gates, server errors stay in the denominator, a frozen reference replaces the live one, and a required task
//! that cannot run blocks an overall PASS.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use talus_bench_core::config::{Config, EndpointKind};
use talus_bench_core::gates::GateStatus;
use talus_bench_core::mock::{spawn_openai, spawn_openai_with, Degrade, ScoreMock, ScoreMode};
use talus_bench_core::runner::{self, Event, RunOptions};

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("talus-bench-e2e-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn config(reference: &str, candidate: &str, extra_tasks: &str) -> Config {
    Config::from_yaml(&format!(
        r#"
experiment: {{id: e2e, seed: 11}}
reference: {{id: ref, endpoint: "http://{reference}"}}
candidates:
  - id: cand
    endpoint: "http://{candidate}/v1"
    provenance: {{quantization: mock-q4}}
inference: {{request_timeout_s: 10}}
benchmarks:
  tier: smoke
  tasks: [json_schema, gsm8k{extra_tasks}]
quality_gates: {{min_cases: 20, bootstrap_resamples: 1000}}
"#
    ))
    .unwrap()
}

fn run(cfg: &Config, out: &Path) -> (runner::RunOutcome, Vec<Event>) {
    let mut events = Vec::new();
    let o = runner::run(
        cfg,
        &RunOptions {
            output_dir: Some(out.to_path_buf()),
            poll_ms: Some(20),
            api_key: None,
            ..Default::default()
        },
        Arc::new(AtomicBool::new(false)),
        &mut |e| events.push(e),
    )
    .unwrap();
    (o, events)
}

fn gate<'a>(o: &'a runner::RunOutcome, name: &str) -> &'a talus_bench_core::gates::GateResult {
    o.verdicts[0]
        .gates
        .iter()
        .find(|g| g.gate == name)
        .unwrap_or_else(|| panic!("no gate {name}"))
}

#[test]
fn identical_candidate_passes_and_artifacts_are_complete() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, count) = spawn_openai(0, Degrade::NONE).unwrap();
    let out = tmp("same");
    let (o, events) = run(&config(&r.to_string(), &c.to_string(), ""), &out);
    assert_eq!(
        o.verdicts[0].overall,
        GateStatus::Pass,
        "{:#?}",
        o.verdicts[0]
    );
    assert_eq!(o.exit_code(), 0);
    assert_eq!(
        *count.lock().unwrap(),
        70,
        "20 json_schema + 50 gsm8k cases"
    );
    assert_eq!(gate(&o, "gsm8k accuracy loss").value, Some(0.0));
    for f in [
        "manifest.json",
        "config.resolved.yaml",
        "datasets.json",
        "baseline/cases.jsonl",
        "baseline/metrics.json",
        "candidates/cand/cases.jsonl",
        "candidates/cand/metrics.json",
        "candidates/cand/perf.json",
        "comparisons/paired_transitions.csv",
        "comparisons/regressions.jsonl",
        "comparisons/benchmark_summary.csv",
        "comparisons/confidence_intervals.json",
        "report.md",
        "report.json",
    ] {
        assert!(out.join(f).is_file(), "missing {f}");
    }
    let rows =
        talus_bench_core::case::read_jsonl(&out.join("candidates/cand/cases.jsonl")).unwrap();
    assert_eq!(rows.len(), 70);
    let r0 = &rows[0];
    assert_eq!(r0.experiment_id, "e2e");
    assert_eq!(r0.prompt_hash.len(), 64);
    assert!(r0.raw_output.is_some());
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest["models"][1]["provenance"]["quantization"],
        "mock-q4"
    );
    assert_eq!(manifest["models"][0]["server"]["models"][0], "mock-model");
    assert!(events.iter().any(|e| matches!(
        e,
        Event::Case {
            done: 70,
            total: 70,
            ..
        }
    )));
    let md = std::fs::read_to_string(out.join("report.md")).unwrap();
    for s in [
        "## Verdict",
        "## 1. What changed",
        "## 3. What regressed",
        "## 5. What cannot be concluded",
        "## 6. What should deploy",
        "not measured",
    ] {
        assert!(md.contains(s), "report.md lacks {s:?}");
    }
    std::fs::remove_dir_all(&out).ok();
}

#[test]
fn degraded_candidate_fails_its_gates() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, _) = spawn_openai(
        0,
        Degrade {
            wrong_rate: 0.4,
            error_rate: 0.0,
            seed: 5,
        },
    )
    .unwrap();
    let out = tmp("degraded");
    let (o, _) = run(&config(&r.to_string(), &c.to_string(), ""), &out);
    let v = &o.verdicts[0];
    assert_eq!(v.overall, GateStatus::Fail, "{v:#?}");
    assert_eq!(o.exit_code(), 1);
    let g = gate(&o, "gsm8k accuracy loss");
    assert_eq!(g.status, GateStatus::Fail);
    assert!(g.interval.unwrap().lo > 2.0);
    assert_eq!(gate(&o, "json_schema rate").status, GateStatus::Fail);
    // the transition table matches the CSV and the regression list
    let paired = &o.report_json["paired"];
    let gsm = paired
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["task"] == "gsm8k")
        .unwrap();
    let regs = gsm["transitions"]["regressions"].as_u64().unwrap();
    assert_eq!(gsm["transitions"]["improvements"], 0);
    assert_eq!(regs + gsm["transitions"]["both_pass"].as_u64().unwrap(), 50);
    let csv = std::fs::read_to_string(out.join("comparisons/paired_transitions.csv")).unwrap();
    assert_eq!(
        csv.lines()
            .filter(|l| l.contains(",gsm8k,") && l.ends_with(",regression"))
            .count() as u64,
        regs
    );
    let jl = std::fs::read_to_string(out.join("comparisons/regressions.jsonl")).unwrap();
    assert_eq!(
        jl.lines()
            .filter(|l| l.contains("\"task\":\"gsm8k\""))
            .count() as u64,
        regs
    );
    let md = std::fs::read_to_string(out.join("report.md")).unwrap();
    assert!(md.contains("| cand | **FAIL** |"), "{md}");
    assert!(md.contains("Representative regressions"));
    std::fs::remove_dir_all(&out).ok();
}

#[test]
fn server_errors_stay_in_the_denominator() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, _) = spawn_openai(
        0,
        Degrade {
            wrong_rate: 0.0,
            error_rate: 0.25,
            seed: 3,
        },
    )
    .unwrap();
    let out = tmp("errors");
    let (o, _) = run(&config(&r.to_string(), &c.to_string(), ""), &out);
    let m: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("candidates/cand/metrics.json")).unwrap(),
    )
    .unwrap();
    let errs = m["gsm8k"]["statuses"]["server_error"].as_u64().unwrap();
    assert!(errs > 0);
    assert_eq!(m["gsm8k"]["n"], 50);
    assert_eq!(m["gsm8k"]["correct"].as_u64().unwrap(), 50 - errs);
    assert_eq!(o.verdicts[0].overall, GateStatus::Fail);
    std::fs::remove_dir_all(&out).ok();
}

#[test]
fn frozen_reference_replaces_the_live_one() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let out = tmp("freeze-src");
    let cfg = config(&r.to_string(), &c.to_string(), "");
    run(&cfg, &out);
    let root = tmp("frozen");
    let dir = runner::freeze(&out, None, &root).unwrap();
    assert!(dir
        .file_name()
        .unwrap()
        .to_string_lossy()
        .starts_with("ref-"));

    // later: frozen reference vs a degraded candidate; the live reference server is not contacted
    let (bad, _) = spawn_openai(
        0,
        Degrade {
            wrong_rate: 0.4,
            error_rate: 0.0,
            seed: 5,
        },
    )
    .unwrap();
    let mut cfg2 = config("127.0.0.1:1", &bad.to_string(), "");
    cfg2.reference.kind = EndpointKind::Frozen;
    cfg2.reference.endpoint = None;
    cfg2.reference.frozen = Some(dir.display().to_string());
    let out2 = tmp("frozen-run");
    let (o, _) = run(&cfg2, &out2);
    assert_eq!(o.verdicts[0].overall, GateStatus::Fail);
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out2.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["models"][0]["kind"], "frozen");
    assert_eq!(
        manifest["models"][0]["frozen"]["content_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );

    // different settings: refused
    let mut cfg3 = cfg2.clone();
    cfg3.inference.max_output_tokens = 99;
    let e = runner::run(
        &cfg3,
        &RunOptions {
            output_dir: Some(tmp("x")),
            poll_ms: None,
            api_key: None,
            ..Default::default()
        },
        Arc::new(AtomicBool::new(false)),
        &mut |_| {},
    )
    .unwrap_err();
    assert!(e.contains("different inference settings"), "{e}");

    // tampered: refused
    let p = dir.join("cases.jsonl");
    let mut t = std::fs::read_to_string(&p).unwrap();
    t = t.replacen("\"pass\":true", "\"pass\":false", 1);
    std::fs::write(&p, t).unwrap();
    let e = runner::run(
        &cfg2,
        &RunOptions {
            output_dir: Some(tmp("y")),
            poll_ms: None,
            api_key: None,
            ..Default::default()
        },
        Arc::new(AtomicBool::new(false)),
        &mut |_| {},
    )
    .unwrap_err();
    assert!(e.contains("modified after freezing"), "{e}");
    for d in [out, root, out2, tmp("x"), tmp("y")] {
        std::fs::remove_dir_all(d).ok();
    }
}

#[test]
fn required_task_not_run_blocks_pass() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let out = tmp("notrun");
    let (o, _) = run(&config(&r.to_string(), &c.to_string(), ", aime, kld"), &out);
    let v = &o.verdicts[0];
    assert_eq!(v.overall, GateStatus::NotRun, "{v:#?}");
    assert_eq!(o.exit_code(), 3);
    assert_eq!(gate(&o, "aime").status, GateStatus::NotRun);
    assert_eq!(gate(&o, "kld").status, GateStatus::NotSupported);
    assert!(gate(&o, "kld").reasons[0].contains("full-vocabulary logits"));
    let md = std::fs::read_to_string(out.join("report.md")).unwrap();
    assert!(md.contains("**kld**: NOT_SUPPORTED"));
    std::fs::remove_dir_all(&out).ok();
}

#[test]
fn unreachable_candidate_is_recorded_not_dropped() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let out = tmp("down");
    let mut cfg = config(&r.to_string(), "127.0.0.1:1", "");
    cfg.benchmarks.limits.insert("gsm8k".into(), 5);
    let (o, _) = run(&cfg, &out);
    let rows =
        talus_bench_core::case::read_jsonl(&out.join("candidates/cand/cases.jsonl")).unwrap();
    assert_eq!(rows.len(), 25);
    assert!(rows.iter().all(
        |r| r.status == talus_bench_core::case::CaseStatus::ConnectError && r.error.is_some()
    ));
    assert_eq!(o.verdicts[0].overall, GateStatus::Fail);
    std::fs::remove_dir_all(&out).ok();
}

// ---- Phase 3a: every native task, gates, resume, seeds ----

fn corpus_file(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("talus-bench-e2e-{}-{name}.txt", std::process::id()));
    let text: String = (0..800)
        .map(|i| format!("def f{i}(x):\n    return x + {i}\n"))
        .collect();
    std::fs::write(&p, text).unwrap();
    p
}

fn all_tasks_config(reference: &str, candidate: &str, corpus: &Path, extra: &str) -> Config {
    Config::from_yaml(&format!(
        r#"
experiment: {{id: e2e-all, seed: 3}}
reference: {{id: ref, endpoint: "http://{reference}"}}
candidates: [{{id: cand, endpoint: "http://{candidate}"}}]
inference: {{request_timeout_s: 10{extra}}}
benchmarks:
  tier: smoke
  tasks: [json_schema, gsm8k, ifeval, bfcl, mmlu_pro, greedy_identity, needle]
  limits: {{gsm8k: 20}}
  params: {{needle: {{lengths: [3000], seeds: [1, 2], corpus: "{}"}}}}
quality_gates: {{min_cases: 20, bootstrap_resamples: 500}}
"#,
        corpus.display()
    ))
    .unwrap()
}

#[test]
fn every_native_task_end_to_end() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, count) = spawn_openai(0, Degrade::NONE).unwrap();
    let corpus = corpus_file("all");
    let out = tmp("all-same");
    let (o, _) = run(
        &all_tasks_config(&r.to_string(), &c.to_string(), &corpus, ""),
        &out,
    );
    let v = &o.verdicts[0];
    // json 20 + gsm8k 20 + ifeval 30 + bfcl 40 + mmlu 28 + greedy 9 + needle 2
    assert_eq!(*count.lock().unwrap(), 149);
    assert_eq!(v.overall, GateStatus::Pass, "{v:#?}");
    assert_eq!(gate(&o, "needle all cases pass").status, GateStatus::Pass);
    assert_eq!(
        gate(&o, "greedy_identity byte identity").status,
        GateStatus::Pass
    );
    assert_eq!(gate(&o, "bfcl accuracy loss").value, Some(0.0));
    let rows =
        talus_bench_core::case::read_jsonl(&out.join("candidates/cand/cases.jsonl")).unwrap();
    let needle: Vec<_> = rows.iter().filter(|r| r.task == "needle").collect();
    assert_eq!(needle.len(), 2);
    assert!(needle
        .iter()
        .all(|r| r.pass() && r.dataset_revision.starts_with("corpus@")));
    let md = std::fs::read_to_string(out.join("report.md")).unwrap();
    assert!(md.contains("Output identity: 9/9"), "identity line");
    let ds = std::fs::read_to_string(out.join("datasets.json")).unwrap();
    assert!(ds.contains("\"built_per_endpoint\": true"));
    for d in [out, corpus] {
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_file(&d);
    }
}

#[test]
fn degraded_candidate_fails_release_gates() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, _) = spawn_openai(
        0,
        Degrade {
            wrong_rate: 0.5,
            error_rate: 0.0,
            seed: 11,
        },
    )
    .unwrap();
    let corpus = corpus_file("deg");
    let out = tmp("all-degraded");
    let (o, _) = run(
        &all_tasks_config(&r.to_string(), &c.to_string(), &corpus, ""),
        &out,
    );
    assert_eq!(o.verdicts[0].overall, GateStatus::Fail);
    assert_eq!(
        gate(&o, "greedy_identity byte identity").status,
        GateStatus::Fail
    );
    assert_eq!(gate(&o, "bfcl accuracy loss").status, GateStatus::Fail);
    assert_eq!(gate(&o, "mmlu_pro accuracy loss").status, GateStatus::Fail);
    let needle = gate(&o, "needle all cases pass");
    // the needle cases are degraded or not by hash; whichever way, the gate agrees with the rows
    let rows =
        talus_bench_core::case::read_jsonl(&out.join("candidates/cand/cases.jsonl")).unwrap();
    let needle_ok = rows.iter().filter(|r| r.task == "needle").all(|r| r.pass());
    assert_eq!(needle.status == GateStatus::Pass, needle_ok);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_file(&corpus);
}

#[test]
fn cancelled_run_resumes_without_repeating_cases() {
    let (r, rc) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, cc) = spawn_openai(0, Degrade::NONE).unwrap();
    let out = tmp("resume");
    let cfg = config(&r.to_string(), &c.to_string(), "");
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancel);
    let err = runner::run(
        &cfg,
        &RunOptions {
            output_dir: Some(out.clone()),
            ..Default::default()
        },
        cancel,
        &mut |e| {
            if let Event::Case {
                model, done: 30, ..
            } = &e
            {
                if model == "cand" {
                    flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        },
    )
    .unwrap_err();
    assert_eq!(err, "cancelled");
    assert_eq!(*rc.lock().unwrap(), 70);
    assert_eq!(*cc.lock().unwrap(), 30);
    let partial =
        talus_bench_core::case::read_jsonl(&out.join("candidates/cand/cases.partial.jsonl"))
            .unwrap();
    assert_eq!(partial.len(), 30);
    // resume: the reference is complete, the candidate runs only its 40 missing cases
    let o = runner::run(
        &cfg,
        &RunOptions {
            output_dir: Some(out.clone()),
            resume: true,
            ..Default::default()
        },
        Arc::new(AtomicBool::new(false)),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(*rc.lock().unwrap(), 70, "reference fully reused");
    assert_eq!(*cc.lock().unwrap(), 70);
    assert_eq!(o.verdicts[0].overall, GateStatus::Pass);
    assert!(!out.join("candidates/cand/cases.partial.jsonl").exists());
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m["resumed"]["rows_kept"], 100);
    // a transient failure is retried on resume, a scored row is kept
    let p = out.join("candidates/cand/cases.jsonl");
    let mut rows = talus_bench_core::case::read_jsonl(&p).unwrap();
    rows[0].status = talus_bench_core::case::CaseStatus::ServerError;
    std::fs::write(&p, talus_bench_core::case::to_jsonl(&rows)).unwrap();
    runner::run(
        &cfg,
        &RunOptions {
            output_dir: Some(out.clone()),
            resume: true,
            ..Default::default()
        },
        Arc::new(AtomicBool::new(false)),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(
        *cc.lock().unwrap(),
        71,
        "only the server_error row was re-run"
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn segment_deadline_stops_cleanly() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let out = tmp("deadline");
    let cfg = config(&r.to_string(), &c.to_string(), "");
    let err = runner::run(
        &cfg,
        &RunOptions {
            output_dir: Some(out.clone()),
            deadline: Some(std::time::Instant::now()),
            ..Default::default()
        },
        Arc::new(AtomicBool::new(false)),
        &mut |_| {},
    )
    .unwrap_err();
    assert!(err.starts_with(runner::INCOMPLETE), "{err}");
    let o = runner::run(
        &cfg,
        &RunOptions {
            output_dir: Some(out.clone()),
            resume: true,
            ..Default::default()
        },
        Arc::new(AtomicBool::new(false)),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(o.verdicts[0].overall, GateStatus::Pass);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn repeated_seeds_pair_per_trial() {
    let (r, rc) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let out = tmp("seeds");
    let mut cfg = config(&r.to_string(), &c.to_string(), "");
    cfg.inference.seeds = Some(vec![1, 2, 3]);
    cfg.inference.temperature = 0.7;
    cfg.benchmarks.limits.insert("gsm8k".into(), 10);
    let (o, _) = run(&cfg, &out);
    assert_eq!(*rc.lock().unwrap(), 90, "(20 + 10) cases x 3 seeds");
    let gsm = o.report_json["paired"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["task"] == "gsm8k")
        .unwrap()
        .clone();
    assert_eq!(gsm["n_paired"], 10);
    assert_eq!(gsm["n_trials"], 30);
    let rows = talus_bench_core::case::read_jsonl(&out.join("baseline/cases.jsonl")).unwrap();
    assert!(rows.iter().all(|r| r.seed.is_some()));
    let _ = std::fs::remove_dir_all(&out);
}

// ---- Phase 3b: job and batch adapters with fake tools ----

fn write_exec(p: &Path, body: &str) {
    std::fs::write(p, body).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// A stand-in for bench/agentic: two tasks; run_agentic.py fails a task when the provider baseURL in the copied
/// opencode config is the one named by TB_FAKE_AGENTIC_FAIL_URL.
fn fake_agentic_suite(name: &str) -> PathBuf {
    let d = tmp(&format!("agentic-suite-{name}"));
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("tasks.py"), "TASKS = {\n    \"alpha\": dict(\n        pkg=\"a\"),\n    \"beta\": dict(\n        pkg=\"b\"),\n}\n").unwrap();
    std::fs::write(
        d.join("opencode-test.json"),
        r#"{"permission": {"edit": "allow"}, "provider": {"glm": {}}}"#,
    )
    .unwrap();
    write_exec(
        &d.join("run_agentic.py"),
        r#"#!/usr/bin/env python3
import json, os, sys
label, model, task = sys.argv[1], sys.argv[2], sys.argv[3]
here = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(here, "opencode-test.json")))
url = cfg["provider"]["talus"]["options"]["baseURL"]
fail = os.environ.get("TB_FAKE_AGENTIC_FAIL_URL", "") == url and task == "beta"
print(json.dumps({"label": label, "model": model, "task": task, "pass": not fail, "go_test_rc": 1 if fail else 0,
                  "tests_touched": [], "agent_rc": 0, "wall_s": 1, "tool_events": 3, "diff": "1 file changed",
                  "clean": True, "fail_tail": "FAIL" if fail else ""}))
"#,
    );
    d
}

#[test]
fn agentic_job_adapter_end_to_end() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let suite = fake_agentic_suite("e2e");
    std::env::set_var("TB_FAKE_AGENTIC_FAIL_URL", format!("http://{c}/v1"));
    let mut cfg = config(&r.to_string(), &c.to_string(), "");
    cfg.benchmarks.tasks = vec!["agentic_go".into()];
    cfg.benchmarks.params.insert(
        "agentic_go".into(),
        serde_json::json!({"dir": suite.display().to_string(), "check_tools": false, "mask": false}),
    );
    let out = tmp("agentic-run");
    let (o, _) = run(&cfg, &out);
    let rows =
        talus_bench_core::case::read_jsonl(&out.join("candidates/cand/cases.jsonl")).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].case_id, "agentic_go/alpha");
    assert!(rows[0].pass() && !rows[1].pass());
    assert_eq!(rows[1].score.details["stage"], "hidden_tests_failed");
    let g = gate(&o, "agentic_go all cases pass");
    assert_eq!(g.status, GateStatus::Fail);
    assert!(g.reasons[0].contains("agentic_go/beta"));
    // the provider config written for the candidate points at the candidate
    let cfgjson = std::fs::read_to_string(
        out.join("candidates/cand/jobs/agentic_go/agentic/opencode-test.json"),
    )
    .unwrap();
    assert!(cfgjson.contains(&format!("http://{c}/v1")));
    assert!(!cfgjson.contains("\"glm\""));
    // missing tools on a real check: NOT_SUPPORTED, not a crash
    let mut cfg2 = cfg.clone();
    cfg2.benchmarks.params.insert(
        "agentic_go".into(),
        serde_json::json!({"dir": "/nonexistent"}),
    );
    let (o2, _) = run(&cfg2, &tmp("agentic-missing"));
    assert_eq!(gate(&o2, "agentic_go").status, GateStatus::NotSupported);
    assert_eq!(o2.verdicts[0].overall, GateStatus::NotRun);
    for d in [out, suite, tmp("agentic-missing")] {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[test]
fn lm_eval_batch_adapter_end_to_end_and_resume() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let bin_dir = tmp("fake-lmeval");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let calls = bin_dir.join("calls");
    // writes the lm-eval 0.4.13 layout: <out>/<model>/results_*.json + samples_<task>_*.jsonl;
    // doc 2 fails when the base_url is the candidate's
    write_exec(
        &bin_dir.join("lm-eval"),
        &format!(
            r#"#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
open("{calls}", "a").write(" ".join(a) + "\n")
out = a[a.index("--output_path") + 1]; task = a[a.index("--tasks") + 1]
margs = dict(kv.split("=", 1) for kv in a[a.index("--model_args") + 1].split(","))
bad = "{cand}" in margs["base_url"]
d = os.path.join(out, "mock-model"); os.makedirs(d, exist_ok=True)
json.dump({{"lm_eval_version": "0.4.13", "task_hashes": {{task: "abcdef0123456789"}}}}, open(os.path.join(d, "results_x.json"), "w"))
with open(os.path.join(d, "samples_%s_2026.jsonl" % task), "w") as f:
    for i in range(4):
        for flt in ("strict-match", "flexible-extract"):
            em = 0.0 if (bad and i == 2) else 1.0
            f.write(json.dumps({{"doc_id": i, "target": "1", "resps": [["the answer is 1"]], "filtered_resps": ["1"], "filter": flt,
                "metrics": ["exact_match"], "doc_hash": "d%d" % i, "prompt_hash": "p%d" % i, "exact_match": em}}) + "\n")
"#,
            calls = calls.display(),
            cand = c
        ),
    );
    let mut cfg = config(&r.to_string(), &c.to_string(), "");
    cfg.benchmarks.tasks = vec!["lm_eval:gsm8k".into()];
    cfg.benchmarks.params.insert(
        "lm_eval:gsm8k".into(),
        serde_json::json!({"bin": bin_dir.join("lm-eval").display().to_string()}),
    );
    cfg.quality_gates.max_loss_pp.insert(
        "lm_eval:gsm8k".into(),
        talus_bench_core::config::Threshold {
            target: 1.0,
            investigate: 2.0,
        },
    );
    cfg.quality_gates.min_cases = 4;
    let out = tmp("lmeval-run");
    let (o, _) = run(&cfg, &out);
    let rows =
        talus_bench_core::case::read_jsonl(&out.join("candidates/cand/cases.jsonl")).unwrap();
    assert_eq!(rows.len(), 4, "one filter");
    assert_eq!(rows[0].task, "lm_eval:gsm8k");
    assert_eq!(rows[2].case_id, "lm_eval:gsm8k/2");
    assert!(!rows[2].pass());
    let g = gate(&o, "lm_eval:gsm8k accuracy loss");
    assert_eq!(g.value, Some(25.0));
    let invocations = std::fs::read_to_string(&calls).unwrap();
    assert_eq!(invocations.lines().count(), 2);
    assert!(
        invocations.contains("--model local-chat-completions")
            && invocations.contains("--log_samples")
    );
    // resume: complete batch rows are kept, the tool is not run again
    runner::run(
        &cfg,
        &RunOptions {
            output_dir: Some(out.clone()),
            resume: true,
            ..Default::default()
        },
        Arc::new(AtomicBool::new(false)),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(std::fs::read_to_string(&calls).unwrap().lines().count(), 2);
    // not installed: NOT_SUPPORTED
    let mut cfg2 = cfg.clone();
    cfg2.benchmarks.params.insert(
        "lm_eval:gsm8k".into(),
        serde_json::json!({"bin": "/nonexistent/lm-eval"}),
    );
    let (o2, _) = run(&cfg2, &tmp("lmeval-missing"));
    assert_eq!(gate(&o2, "lm_eval:gsm8k").status, GateStatus::NotSupported);
    for d in [out, bin_dir, tmp("lmeval-missing")] {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[test]
fn reference_only_run_reports_absolute_rates() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let mut cfg = config(&r.to_string(), &r.to_string(), "");
    cfg.candidates.clear();
    let v = cfg.validate();
    assert!(v.errors.is_empty(), "{:?}", v.errors);
    assert!(v.warnings.iter().any(|w| w.contains("reference-only")));
    let out = tmp("refonly");
    let (o, _) = run(&cfg, &out);
    assert!(o.verdicts.is_empty());
    assert_eq!(o.exit_code(), 3);
    assert!(o.report_md.contains("Absolute results"), "{}", o.report_md);
    assert!(o.report_md.contains("| ref | gsm8k |"));
    assert!(out.join("baseline/metrics.json").exists());
    // a frozen reference still needs a candidate
    cfg.reference.kind = EndpointKind::Frozen;
    assert!(!cfg.validate().errors.is_empty());
    let _ = std::fs::remove_dir_all(&out);
}

// ---- teacher-forced scoring (talus-server E2): perplexity and kld_topk ----

fn scoring_config(reference: &str, candidate: &str) -> Config {
    let mut cfg = config(reference, candidate, ", perplexity, kld_topk");
    // small windows keep the mock fast; 24 perplexity windows >= min_cases 20
    for (task, windows) in [("perplexity", 24), ("kld_topk", 6)] {
        cfg.benchmarks.params.insert(
            task.into(),
            serde_json::json!({"windows": windows, "prompt_tokens": 64, "continuation_tokens": 64}),
        );
    }
    cfg
}

#[test]
fn teacher_forced_tasks_end_to_end() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let out = tmp("tf-same");
    let (o, _) = run(&scoring_config(&r.to_string(), &c.to_string()), &out);
    assert_eq!(
        o.verdicts[0].overall,
        GateStatus::Pass,
        "{:#?}",
        o.verdicts[0]
    );
    let g = gate(&o, "perplexity PPL ratio");
    assert_eq!(g.status, GateStatus::Pass, "{g:?}");
    assert_eq!(g.value, Some(1.0));
    assert_eq!(g.n, 24);
    let k = gate(&o, "kld_topk screening KL");
    assert_eq!(k.status, GateStatus::Pass, "{k:?}");
    assert_eq!(k.value, Some(0.0));
    assert_eq!(k.n, 6 * 64);
    assert!(k.reasons[0].contains("top-1 100.00%"), "{:?}", k.reasons);
    // the rows carry the scores; the reference keeps its top-k lists for the pairing (or a later freeze)
    let rows = talus_bench_core::case::read_jsonl(&out.join("baseline/cases.jsonl")).unwrap();
    let ppl: Vec<_> = rows.iter().filter(|r| r.task == "perplexity").collect();
    assert_eq!(ppl.len(), 24);
    assert_eq!(ppl[0].case_id, "perplexity/w000");
    assert_eq!(ppl[0].score.parsed["tokens"], 64);
    assert!(ppl[0].score.parsed["ppl"].as_f64().unwrap() > 1.0);
    let kr = rows.iter().find(|r| r.task == "kld_topk").unwrap();
    assert_eq!(kr.score.parsed["top"][0].as_array().unwrap().len(), 40);
    let cov = kr.score.details["covered_mean"].as_f64().unwrap();
    assert!(cov > 0.5 && cov < 1.0, "{cov}");
    let tf: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("comparisons/teacher_forced.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(tf["pairs"].as_array().unwrap().len(), 2);
    assert_eq!(tf["pairs"][0]["per_case"].as_array().unwrap().len(), 24);
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("baseline/metrics.json")).unwrap())
            .unwrap();
    assert_eq!(m["perplexity"]["task_metrics"]["tokens"], 24 * 64);
    let md = std::fs::read_to_string(out.join("report.md")).unwrap();
    assert!(
        md.contains("| perplexity PPL ratio | yes | **PASS** | 1.0000x |"),
        "{md}"
    );
    assert!(md.contains("(WARN only)"));
    std::fs::remove_dir_all(&out).ok();
}

#[test]
fn drifted_candidate_fails_ppl_and_warns_kld_topk() {
    let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let (c, _) = spawn_openai_with(
        0,
        Degrade::NONE,
        ScoreMock {
            drift: 1.0,
            seed: 7,
            ..Default::default()
        },
    )
    .unwrap();
    let out = tmp("tf-drift");
    let (o, _) = run(&scoring_config(&r.to_string(), &c.to_string()), &out);
    let g = gate(&o, "perplexity PPL ratio");
    assert_eq!(g.status, GateStatus::Fail, "{g:?}");
    assert!(g.value.unwrap() > 1.05);
    assert!(g.interval.unwrap().lo > 1.01);
    let k = gate(&o, "kld_topk screening KL");
    assert_eq!(k.status, GateStatus::Warn, "{k:?}");
    assert!(k.value.unwrap() > 0.05);
    assert_eq!(o.verdicts[0].overall, GateStatus::Fail);
    // enforced, the screening gate fails too
    let mut cfg = scoring_config(&r.to_string(), &c.to_string());
    cfg.quality_gates.kld_topk.enforce = true;
    let out2 = tmp("tf-drift-enforce");
    let (o2, _) = run(&cfg, &out2);
    assert_eq!(gate(&o2, "kld_topk screening KL").status, GateStatus::Fail);
    std::fs::remove_dir_all(&out).ok();
    std::fs::remove_dir_all(&out2).ok();
}

#[test]
fn scoring_unsupported_is_not_supported() {
    for mode in [ScoreMode::NotSupported, ScoreMode::Absent] {
        let (r, _) = spawn_openai(0, Degrade::NONE).unwrap();
        let (c, _) = spawn_openai_with(
            0,
            Degrade::NONE,
            ScoreMock {
                mode,
                ..Default::default()
            },
        )
        .unwrap();
        let out = tmp("tf-unsupported");
        let (o, _) = run(&scoring_config(&r.to_string(), &c.to_string()), &out);
        for t in ["perplexity", "kld_topk"] {
            let g = gate(&o, t);
            assert_eq!(g.status, GateStatus::NotSupported, "{mode:?} {g:?}");
            assert!(g.reasons[0].contains("/v1/talus/score"), "{:?}", g.reasons);
        }
        if mode == ScoreMode::NotSupported {
            assert!(gate(&o, "perplexity").reasons[0].contains("HTTP 501"));
        }
        // a required test that cannot run blocks an overall PASS
        assert_eq!(o.verdicts[0].overall, GateStatus::NotRun);
        std::fs::remove_dir_all(&out).ok();
    }
}
