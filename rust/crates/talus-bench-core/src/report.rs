//! Reports (spec §10): report.md for people, report.json for machines, plus the comparison CSV/JSONL files.
//! Every number comes from this run's case rows; nothing is estimated or filled in. Where something was not
//! measured the report says so.

use crate::case::CaseResult;
use crate::compare::{PairRow, PairedTask};
use crate::config::Config;
use crate::gates::{GateStatus, Verdict};
use crate::util::{rfc3339, unix_now};
use serde_json::{json, Value};
use std::fmt::Write as _;

pub struct ReportCtx<'a> {
    pub cfg: &'a Config,
    pub manifest: &'a Value,
    pub paired: &'a [PairedTask],
    pub verdicts: &'a [Verdict],
    pub unavailable: &'a [(String, GateStatus, String)],
    pub model_rows: &'a [(String, Vec<CaseResult>)],
}

fn pct(x: Option<f64>) -> String {
    x.map(|v| format!("{:.1}%", v * 100.0))
        .unwrap_or_else(|| "n/a".into())
}

fn num(x: Option<f64>, d: usize) -> String {
    x.map(|v| format!("{v:.d$}"))
        .unwrap_or_else(|| "n/a".into())
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn snippet(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    let t = t.replace('\n', " ⏎ ").replace('|', "\\|");
    if s.chars().count() > n {
        format!("{t}…")
    } else {
        t
    }
}

pub fn transitions_csv(rows: &[(String, PairRow)]) -> String {
    let mut s = String::from("candidate,task,case_id,reference_pass,candidate_pass,reference_status,candidate_status,transition\n");
    for (c, r) in rows {
        let t = match (r.reference_pass, r.candidate_pass) {
            (true, true) => "retained",
            (true, false) => "regression",
            (false, true) => "improvement",
            (false, false) => "shared_failure",
        };
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{},{t}",
            csv_field(c),
            csv_field(&r.task),
            csv_field(&r.case_id),
            r.reference_pass,
            r.candidate_pass,
            r.reference_status.as_str(),
            r.candidate_status.as_str()
        );
    }
    s
}

pub fn summary_csv(p: &[PairedTask]) -> String {
    let mut s = String::from(
        "candidate,task,n_paired,reference_correct,candidate_correct,reference_accuracy,candidate_accuracy,loss_pp,retention_pct,ci_lo_pp,ci_hi_pp,regressions,improvements,both_pass,both_fail,mcnemar_p,not_comparable,n_trials,identical,identity_compared\n",
    );
    for x in p {
        let f = |v: Option<f64>| v.map(|v| format!("{v:.6}")).unwrap_or_default();
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{},{},{},{}",
            csv_field(&x.candidate_id),
            x.task,
            x.n_paired,
            x.reference_correct,
            x.candidate_correct,
            f(x.reference_accuracy),
            f(x.candidate_accuracy),
            f(x.loss_pp),
            f(x.retention_pct),
            f(x.ci_loss_pp.map(|c| c.lo)),
            f(x.ci_loss_pp.map(|c| c.hi)),
            x.transitions.regressions,
            x.transitions.improvements,
            x.transitions.both_pass,
            x.transitions.both_fail,
            x.mcnemar_p,
            x.not_comparable.len(),
            x.n_trials,
            x.identical,
            x.identity_compared
        );
    }
    s
}

/// The row of `model` for a pair label (`case_id` or `case_id#s<seed>`).
fn find<'a>(
    rows: &'a [(String, Vec<CaseResult>)],
    model: &str,
    label: &str,
) -> Option<&'a CaseResult> {
    let (id, seed) = match label.rsplit_once("#s") {
        Some((id, s)) if s.parse::<u64>().is_ok() => (id, s.parse::<u64>().ok()),
        _ => (label, None),
    };
    rows.iter()
        .find(|(m, _)| m == model)?
        .1
        .iter()
        .find(|r| r.case_id == id && r.seed == seed)
}

pub fn regressions_jsonl(
    p: &[PairedTask],
    rows: &[(String, Vec<CaseResult>)],
    redact: bool,
) -> String {
    let ref_id = rows.first().map(|r| r.0.as_str()).unwrap_or("");
    let mut s = String::new();
    for x in p {
        for id in &x.regression_ids {
            let r = find(rows, ref_id, id);
            let c = find(rows, &x.candidate_id, id);
            let out = |c: Option<&CaseResult>| {
                c.and_then(|c| c.raw_output.as_ref())
                    .map(|o| o.content.chars().take(4000).collect::<String>())
            };
            let v = json!({
                "candidate": x.candidate_id, "task": x.task, "case_id": id,
                "critical": x.critical_regression_ids.contains(id),
                "expected": if redact { Value::Null } else { r.map(|r| r.expected.clone()).unwrap_or(Value::Null) },
                "reference_status": r.map(|r| r.status.as_str()).unwrap_or("missing"),
                "candidate_status": c.map(|c| c.status.as_str()).unwrap_or("missing"),
                "reference_output": out(r), "candidate_output": out(c),
                "candidate_score": c.map(|c| c.score.details.clone()),
                "candidate_error": c.and_then(|c| c.error.clone()),
            });
            s.push_str(&v.to_string());
            s.push('\n');
        }
    }
    s
}

fn recommendation(v: &[Verdict]) -> Vec<String> {
    let mut out = Vec::new();
    for x in v {
        out.push(match x.overall {
            GateStatus::Pass => format!(
                "{}: PASS on every configured gate. Eligible; choose among eligible candidates on the quality/performance/memory frontier (perf tier and memory not measured in this run).",
                x.candidate_id
            ),
            GateStatus::Warn => format!(
                "{}: WARN. No verified regression beyond a target, but the evidence is not strong enough for PASS ({}). Run a larger tier or review the warned gates before deploying.",
                x.candidate_id,
                x.reasons.join("; ")
            ),
            GateStatus::Fail => format!("{}: FAIL. Do not deploy: {}.", x.candidate_id, x.reasons.join("; ")),
            GateStatus::NotRun | GateStatus::NotSupported => format!(
                "{}: no overall PASS possible: {}.",
                x.candidate_id,
                x.reasons.join("; ")
            ),
        });
    }
    out
}

pub fn report_json(c: &ReportCtx) -> Value {
    json!({
        "schema": "talus-bench/report/1",
        "generated_at": rfc3339(unix_now()),
        "experiment": c.cfg.experiment,
        "tier": c.cfg.benchmarks.tier.id(),
        "verdicts": c.verdicts,
        "paired": c.paired,
        "unavailable": c.unavailable.iter().map(|(t, s, r)| json!({"task": t, "status": s, "reason": r})).collect::<Vec<_>>(),
        "models": c.manifest["models"],
        "settings": c.manifest["settings"],
        "limitations": c.manifest["limitations"],
        "performance": {"measured": false, "note": "quality run; performance is measured separately by the perf tier"},
        "recommendation": recommendation(c.verdicts),
    })
}

pub fn report_md(c: &ReportCtx) -> String {
    let mut s = String::new();
    let m = c.manifest;
    let _ = writeln!(s, "# talus-bench report: {}\n", c.cfg.experiment.id);
    if let Some(d) = &c.cfg.experiment.description {
        let _ = writeln!(s, "{d}\n");
    }
    let _ = writeln!(
        s,
        "Tier **{}** · started {} · finished {} · talus-bench {} · settings `{}` (hash {})\n",
        c.cfg.benchmarks.tier.id(),
        m["started_at"].as_str().unwrap_or("?"),
        m["finished_at"].as_str().unwrap_or("?"),
        m["bench"]["version"].as_str().unwrap_or("?"),
        m["settings"],
        &m["settings_hash"].as_str().unwrap_or("?")
            [..12.min(m["settings_hash"].as_str().unwrap_or("?").len())]
    );

    if c.verdicts.is_empty() {
        absolute_md(&mut s, c);
    }
    let _ = writeln!(
        s,
        "## Verdict\n\n| Candidate | Overall | Why |\n|---|---|---|"
    );
    for v in c.verdicts {
        let _ = writeln!(
            s,
            "| {} | **{}** | {} |",
            v.candidate_id,
            v.overall.as_str(),
            if v.reasons.is_empty() {
                "all gates pass".into()
            } else {
                v.reasons.join("; ")
            }
        );
    }

    let _ = writeln!(s, "\n## 1. What changed\n\n| Model | Role | Source | Talus selection | Server reports | Provenance |\n|---|---|---|---|---|---|");
    for x in m["models"].as_array().into_iter().flatten() {
        let source = x["frozen"]["dir"]
            .as_str()
            .map(|d| {
                format!(
                    "frozen `{}` ({})",
                    d,
                    &x["frozen"]["content_sha256"].as_str().unwrap_or("")
                        [..12.min(x["frozen"]["content_sha256"].as_str().unwrap_or("").len())]
                )
            })
            .or_else(|| x["server"]["url"].as_str().map(str::to_string))
            .unwrap_or_default();
        let talus = if x["talus"].is_null() {
            "-".into()
        } else {
            format!(
                "{} {} {}",
                x["talus"]["model"].as_str().unwrap_or(""),
                x["talus"]["level"].as_str().unwrap_or("balanced"),
                x["talus"]["choices"]
            )
        };
        let st = &x["server"]["talus_status"];
        let server = if st.is_null() {
            "no /v1/talus/status".to_string()
        } else {
            let level = match st["selection"]["level"].as_str() {
                Some(l) => format!("level {l}"),
                None => "served config, no apply".to_string(),
            };
            format!(
                "{} ({level}, ctx {})",
                st["preset"].as_str().unwrap_or("?"),
                st["context_max"]
            )
        };
        let prov: Vec<String> = x["provenance"]
            .as_object()
            .into_iter()
            .flatten()
            .filter(|(_, v)| !v.is_null())
            .map(|(k, v)| {
                format!(
                    "{k}={}",
                    v.as_str().map(str::to_string).unwrap_or(v.to_string())
                )
            })
            .collect();
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} |",
            x["id"].as_str().unwrap_or(""),
            x["role"].as_str().unwrap_or(""),
            source,
            talus,
            server,
            if prov.is_empty() {
                "not recorded".into()
            } else {
                prov.join(", ")
            }
        );
    }
    fingerprint_table(&mut s, m);

    let _ = writeln!(s, "\n## 2. Gates\n");
    for v in c.verdicts {
        let _ = writeln!(s, "### {} — {}\n\n| Gate | Required | Status | Value | Threshold (target / investigate) | Interval | n | Notes |\n|---|---|---|---|---|---|---|---|", v.candidate_id, v.overall.as_str());
        for g in &v.gates {
            let unit = crate::gates::unit(&g.kind);
            let th = g
                .threshold
                .map(|t| match g.kind.as_str() {
                    "min_rate_pct" => format!(">= {} / < {} investigate", t.target, t.investigate),
                    "all_pass" => "every case passes".to_string(),
                    "identity" => "every answer byte-identical".to_string(),
                    "kld_topk" => {
                        // the resolved limits (config, per-model or generic); top-1 and the source are in the notes
                        let k = &c.cfg.quality_gates.kld_topk;
                        format!(
                            "mean <= {}{}",
                            t.target,
                            if k.enforce { "" } else { " (WARN only)" }
                        )
                    }
                    _ => format!("<= {} / > {} investigate", t.target, t.investigate),
                })
                .unwrap_or_else(|| "-".into());
            // ratios and KL need more digits than accuracies
            let fine = matches!(g.kind.as_str(), "ppl_ratio" | "kld_topk");
            let iv = g
                .interval
                .map(|i| {
                    if fine {
                        format!("[{:.4}, {:.4}] ({:.0}%)", i.lo, i.hi, i.confidence * 100.0)
                    } else {
                        format!("[{:.2}, {:.2}] ({:.0}%)", i.lo, i.hi, i.confidence * 100.0)
                    }
                })
                .unwrap_or_else(|| "-".into());
            let _ = writeln!(
                s,
                "| {} | {} | **{}** | {} | {} | {} | {} | {} |",
                g.gate,
                if g.required { "yes" } else { "no" },
                g.status.as_str(),
                g.value
                    .map(|v| match g.kind.as_str() {
                        "ppl_ratio" => format!("{v:.4}{unit}"),
                        "kld_topk" => format!("{v:.3e}{unit}"),
                        _ => format!("{v:.2}{unit}"),
                    })
                    .unwrap_or_else(|| "-".into()),
                th,
                iv,
                g.n,
                g.reasons.join("; ")
            );
        }
        let _ = writeln!(s);
    }

    let _ = writeln!(s, "## 3. What regressed\n");
    for p in c.paired {
        let _ = writeln!(s, "### {} on {}\n", p.candidate_id, p.task);
        let _ = writeln!(s, "| Reference acc. | Candidate acc. | Loss (pp) | Retention | Loss CI (pp) | McNemar p | Paired n |\n|---|---|---|---|---|---|---|");
        let _ = writeln!(
            s,
            "| {} ({}/{}) | {} ({}/{}) | {} | {} | {} | {:.4} | {} |",
            pct(p.reference_accuracy),
            p.reference_correct,
            p.n_trials,
            pct(p.candidate_accuracy),
            p.candidate_correct,
            p.n_trials,
            num(p.loss_pp, 2),
            p.retention_pct
                .map(|r| format!("{r:.1}%"))
                .unwrap_or_else(|| "undefined".into()),
            p.ci_loss_pp
                .map(|i| format!("[{:.2}, {:.2}]", i.lo, i.hi))
                .unwrap_or_else(|| "insufficient sample".into()),
            p.mcnemar_p,
            if p.n_trials != p.n_paired {
                format!("{} cases x trials = {}", p.n_paired, p.n_trials)
            } else {
                p.n_paired.to_string()
            }
        );
        if let (Some(rw), Some(cw)) = (
            crate::stats::wilson(
                p.reference_correct,
                p.n_trials,
                c.cfg.quality_gates.confidence,
            ),
            crate::stats::wilson(
                p.candidate_correct,
                p.n_trials,
                c.cfg.quality_gates.confidence,
            ),
        ) {
            let _ = writeln!(
                s,
                "\nRates with {:.0}% Wilson intervals: reference {}/{} [{:.1}, {:.1}]%, candidate {}/{} [{:.1}, {:.1}]%",
                rw.confidence * 100.0, p.reference_correct, p.n_trials, rw.lo * 100.0, rw.hi * 100.0,
                p.candidate_correct, p.n_trials, cw.lo * 100.0, cw.hi * 100.0
            );
        }
        if p.identity_compared > 0 {
            let gated = c.cfg.quality_gates.identity.contains(&p.task);
            let note = if p.identical == p.identity_compared {
                ""
            } else if gated {
                " (identity gate: see the gate table)"
            } else {
                " (numerics or text drift; informational, no identity gate on this task)"
            };
            let _ = writeln!(
                s,
                "\nOutput identity: {}/{} answers byte-identical to the reference{note}",
                p.identical, p.identity_compared
            );
        }
        let t = p.transitions;
        let _ = writeln!(s, "\n| Reference | Candidate | Cases | Meaning |\n|---|---|---|---|\n| pass | pass | {} | retained success |\n| pass | fail | {} | regression |\n| fail | pass | {} | improvement |\n| fail | fail | {} | shared failure |\n", t.both_pass, t.regressions, t.improvements, t.both_fail);
        if !p.missing_in_candidate.is_empty()
            || !p.missing_in_reference.is_empty()
            || !p.not_comparable.is_empty()
        {
            let _ = writeln!(
                s,
                "Missing in candidate: {} · missing in reference: {} · not comparable: {}\n",
                p.missing_in_candidate.len(),
                p.missing_in_reference.len(),
                p.not_comparable.len()
            );
        }
        if p.task == "livecodebench" {
            let ref_id = c.model_rows.first().map(|r| r.0.as_str()).unwrap_or("");
            for (m, rows) in c
                .model_rows
                .iter()
                .filter(|(m, _)| m == ref_id || *m == p.candidate_id)
            {
                let lcb: Vec<CaseResult> = rows
                    .iter()
                    .filter(|r| r.task == "livecodebench")
                    .cloned()
                    .collect();
                if !lcb.is_empty() {
                    let mm = crate::compare::model_metrics(&lcb, c.cfg.quality_gates.confidence);
                    let _ = writeln!(
                        s,
                        "{}\n",
                        livecodebench_md(m, &mm["livecodebench"]["task_metrics"])
                    );
                }
            }
        }
        if let Some((_, rows)) = c.model_rows.iter().find(|(m, _)| *m == p.candidate_id) {
            let mut st: std::collections::BTreeMap<&str, usize> = Default::default();
            for r in rows.iter().filter(|r| r.task == p.task) {
                let k = if r.status.scored() {
                    r.score.details["stage"]
                        .as_str()
                        .unwrap_or(r.status.as_str())
                } else {
                    r.status.as_str()
                };
                if !r.pass() {
                    *st.entry(k).or_default() += 1;
                }
            }
            if !st.is_empty() {
                let _ = writeln!(
                    s,
                    "Candidate failure categories: {}\n",
                    st.iter()
                        .map(|(k, v)| format!("{k} {v}"))
                        .collect::<Vec<_>>()
                        .join(" · ")
                );
            }
        }
        let ref_id = c.model_rows.first().map(|r| r.0.as_str()).unwrap_or("");
        let shown: Vec<&String> = p
            .critical_regression_ids
            .iter()
            .chain(
                p.regression_ids
                    .iter()
                    .filter(|i| !p.critical_regression_ids.contains(i)),
            )
            .take(c.cfg.report.max_examples)
            .collect();
        if !shown.is_empty() {
            let _ = writeln!(s, "Representative regressions ({} of {}; full list in comparisons/regressions.jsonl):\n\n| Case | Critical | Expected | Reference answer | Candidate answer | Candidate status |\n|---|---|---|---|---|---|", shown.len(), p.regression_ids.len());
            for id in shown {
                let r = find(c.model_rows, ref_id, id);
                let k = find(c.model_rows, &p.candidate_id, id);
                let ans = |x: Option<&CaseResult>| {
                    x.and_then(|x| x.raw_output.as_ref())
                        .map(|o| snippet(o.content.trim(), 80))
                        .unwrap_or_else(|| "-".into())
                };
                let exp = if c.cfg.report.redact_prompts {
                    "(redacted)".into()
                } else {
                    r.map(|r| snippet(&r.expected.to_string(), 60))
                        .unwrap_or_default()
                };
                let _ = writeln!(
                    s,
                    "| `{}` | {} | {} | {} | {} | {} |",
                    id,
                    if p.critical_regression_ids.contains(id) {
                        "yes"
                    } else {
                        ""
                    },
                    exp,
                    ans(r),
                    ans(k),
                    k.map(|k| k.status.as_str()).unwrap_or("missing")
                );
            }
            let _ = writeln!(s);
        }
    }

    let _ = writeln!(s, "## 4. What improved\n");
    for p in c.paired {
        let _ = writeln!(
            s,
            "- {} on {}: {} improvement(s) (reference fail, candidate pass).",
            p.candidate_id, p.task, p.transitions.improvements
        );
    }
    let _ = writeln!(s, "\nPerformance, memory and energy were **not measured** in this run. Quality runs record only passive request timings (candidates' perf.json), which are not benchmark cells. Run the perf tier (same-session control → candidate → control, noise bands from bands.json) for speed claims.\n");

    let _ = writeln!(s, "## 5. What cannot be concluded\n");
    for (t, st, r) in c.unavailable {
        let _ = writeln!(s, "- **{t}**: {} — {r}", st.as_str());
    }
    for p in c
        .paired
        .iter()
        .filter(|p| p.n_paired < c.cfg.quality_gates.min_cases)
    {
        let _ = writeln!(
            s,
            "- {} on {}: only {} paired cases (min_cases {}); intervals are wide.",
            p.candidate_id, p.task, p.n_paired, c.cfg.quality_gates.min_cases
        );
    }
    for l in m["task_caveats"].as_array().into_iter().flatten() {
        let _ = writeln!(s, "- {}", l.as_str().unwrap_or(""));
    }
    for l in m["limitations"].as_array().into_iter().flatten() {
        let _ = writeln!(s, "- {}", l.as_str().unwrap_or(""));
    }
    let _ = writeln!(s, "- Contamination: GSM8K is public and likely in training data; it measures relative retention between settings, not absolute ability. The json_schema cases are written for Talus and public in its repository. No private holdout was used in this run.");

    let _ = writeln!(s, "\n## 6. What should deploy\n");
    for r in recommendation(c.verdicts) {
        let _ = writeln!(s, "- {r}");
    }
    let _ = writeln!(s, "\n---\nArtifacts: manifest.json, config.resolved.yaml, datasets.json, baseline/, candidates/, comparisons/. Every figure above comes from this run's case rows.");
    s
}

/// Reference-only runs: per-task absolute rates (Wilson), plus the SWE-bench validity counters.
fn absolute_md(s: &mut String, c: &ReportCtx) {
    let conf = c.cfg.quality_gates.confidence;
    let _ = writeln!(
        s,
        "## Absolute results (reference-only run: no candidate, no paired gates)\n"
    );
    let _ = writeln!(
        s,
        "| Model | Task | Passed | n | Rate | {:.0}% Wilson CI |\n|---|---|---|---|---|---|",
        conf * 100.0
    );
    for (model, rows) in c.model_rows {
        let mut by: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
        for r in rows {
            let e = by.entry(r.task.as_str()).or_default();
            e.1 += 1;
            if r.pass() {
                e.0 += 1;
            }
        }
        for (task, (k, n)) in by {
            let ci = crate::stats::wilson(k, n, conf)
                .map(|i| format!("{:.1}–{:.1}%", i.lo * 100.0, i.hi * 100.0))
                .unwrap_or_else(|| "n/a".into());
            let _ = writeln!(
                s,
                "| {model} | {task} | {k} | {n} | {:.1}% | {ci} |",
                if n > 0 {
                    k as f64 * 100.0 / n as f64
                } else {
                    0.0
                }
            );
        }
        let lcb: Vec<crate::case::CaseResult> = rows
            .iter()
            .filter(|r| r.task == "livecodebench")
            .cloned()
            .collect();
        if !lcb.is_empty() {
            let m = crate::compare::model_metrics(&lcb, conf);
            let _ = writeln!(
                s,
                "\n{}",
                livecodebench_md(model, &m["livecodebench"]["task_metrics"])
            );
        }
        let swe: Vec<&crate::case::CaseResult> = rows
            .iter()
            .filter(|r| r.task == "swebench_verified")
            .collect();
        if !swe.is_empty() {
            let m = crate::compare::model_metrics(
                &swe.iter().map(|r| (*r).clone()).collect::<Vec<_>>(),
                conf,
            );
            let x = &m["swebench_verified"]["task_metrics"];
            let _ = writeln!(
                s,
                "\n{model} swebench_verified: invalid (anti-cheat leak vector) {} · resolved ignoring flags {} (of which flagged {}) · environment errors {} · near-verbatim patches {} · flags by kind {} · informational notes by kind {} · stages {}\n",
                x["invalid_instances"], x["resolved_ignoring_flags"], x["flagged_but_resolved"], x["environment_errors"], x["near_verbatim"], x["flags_by_kind"], x["notes_by_kind"], x["stages"]
            );
        }
    }
    let _ = writeln!(s);
}

fn short(v: &Value) -> String {
    match v.as_str() {
        Some(h) => h[..12.min(h.len())].to_string(),
        None => "-".to_string(),
    }
}

fn file_cell(f: &Value) -> String {
    if f.is_null() {
        return "-".into();
    }
    match f["sha256"].as_str() {
        Some(h) => format!("`{}`", &h[..12.min(h.len())]),
        None if f["hashing"].as_bool() == Some(true) => "hashing".into(),
        None => "not hashed".into(),
    }
}

/// One row per fingerprint field of talus-server's `/v1/talus/status` `build` (bench extension E3), one column per
/// model; a field whose value differs from the reference's is marked **changed**.
fn fingerprint_table(s: &mut String, m: &Value) {
    let models: Vec<&Value> = m["models"].as_array().into_iter().flatten().collect();
    let builds: Vec<&Value> = models
        .iter()
        .map(|x| &x["server"]["talus_status"]["build"])
        .collect();
    let missing: Vec<&str> = models
        .iter()
        .zip(&builds)
        .filter(|(_, b)| !b.is_object())
        .map(|(x, _)| x["id"].as_str().unwrap_or(""))
        .collect();
    if builds.iter().all(|b| !b.is_object()) {
        let _ = writeln!(s, "\nNot fingerprinted (the server does not report a `build` in /v1/talus/status; talus-server with bench extension E3 does): engine build commit, weights sha256, run-config sha256, kernels library. See DESIGN.md \"server extensions\".");
        return;
    }
    let weights = |b: &Value| -> String {
        let w: Vec<String> = b["weights"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|f| {
                let name = f["path"]
                    .as_str()
                    .and_then(|p| p.rsplit('/').next())
                    .unwrap_or("?");
                format!("{name} {}", file_cell(f))
            })
            .collect();
        if w.is_empty() {
            "-".into()
        } else {
            w.join("<br>")
        }
    };
    let env = |b: &Value| -> String {
        let e: Vec<String> = b["numerics_env"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, v)| format!("{k}={}", v.as_str().unwrap_or("")))
            .collect();
        if e.is_empty() {
            "-".into()
        } else {
            e.join(" ")
        }
    };
    type Field = (&'static str, fn(&Value) -> String);
    let fields: Vec<Field> = vec![
        ("Engine commit", |b| {
            let c = short(&b["engine_exe"]["commit"]);
            match b["engine_exe"]["dirty"].as_bool() {
                Some(true) => format!("{c} (dirty)"),
                _ => c,
            }
        }),
        ("Release tag", |b| {
            b["engine_exe"]["tag"].as_str().unwrap_or("-").to_string()
        }),
        ("Engine sha256", |b| file_cell(&b["engine_exe"])),
        ("Kernels library sha256", |b| file_cell(&b["kernels_lib"])),
        ("GPU target", |b| {
            b["gfx_target"].as_str().unwrap_or("-").to_string()
        }),
        ("Run config sha256", |b| short(&b["run_config_sha256"])),
    ];
    let _ = write!(
        s,
        "\nBuild fingerprint (talus-server /v1/talus/status `build`):\n\n| Field |"
    );
    for x in &models {
        let _ = write!(s, " {} |", x["id"].as_str().unwrap_or(""));
    }
    let _ = write!(s, "\n|---|");
    for _ in &models {
        let _ = write!(s, "---|");
    }
    let _ = writeln!(s);
    let row = |s: &mut String, name: &str, cells: Vec<String>| {
        let _ = write!(s, "| {name} |");
        for (i, (c, b)) in cells.iter().zip(&builds).enumerate() {
            let cell = if !b.is_object() {
                "not reported".to_string()
            } else if i > 0
                && builds[0].is_object()
                && *c != cells[0]
                && ![c, &cells[0]]
                    .iter()
                    .any(|x| x.contains("hashing") || x.contains("not hashed"))
            {
                format!("{c} **changed**")
            } else {
                c.clone()
            };
            let _ = write!(s, " {cell} |");
        }
        let _ = writeln!(s);
    };
    for (name, f) in &fields {
        let cells = builds.iter().map(|b| f(b)).collect();
        row(s, name, cells);
    }
    row(
        s,
        "Weights sha256",
        builds.iter().map(|b| weights(b)).collect(),
    );
    row(
        s,
        "Numerics environment",
        builds.iter().map(|b| env(b)).collect(),
    );
    if !missing.is_empty() {
        let _ = writeln!(
            s,
            "\nNot fingerprinted for {} (no `build` in /v1/talus/status).",
            missing.join(", ")
        );
    }
}

/// One model's livecodebench breakdown (pass@1 per difficulty with Wilson intervals, stages, sandbox flags).
pub fn livecodebench_md(model: &str, x: &Value) -> String {
    let ci = |v: &Value| {
        v["wilson"]["lo"]
            .as_f64()
            .zip(v["wilson"]["hi"].as_f64())
            .map(|(lo, hi)| format!("{:.1}–{:.1}%", lo * 100.0, hi * 100.0))
            .unwrap_or_else(|| "n/a".into())
    };
    let mut parts = Vec::new();
    for key in ["by_difficulty", "by_platform", "by_testtype"] {
        for (k, v) in x[key].as_object().into_iter().flatten() {
            parts.push(format!("{k} {}/{} ({})", v["passed"], v["n"], ci(v)));
        }
    }
    format!(
        "{model} livecodebench pass@1: {} · mean pass@1 over {} problems {:.1}% · sandbox violations {} (flags {}) · sandbox errors {} · stages {}",
        parts.join(" · "),
        x["problems"],
        x["mean_pass_at_1"].as_f64().unwrap_or(0.0) * 100.0,
        x["sandbox_violations"],
        x["flags_by_kind"],
        x["sandbox_errors"],
        x["stages"]
    )
}

#[cfg(test)]
mod fingerprint_tests {
    use super::*;
    use crate::case::Timings;

    fn build(commit: &str, w: Option<&str>, env: &str) -> Value {
        json!({
            "engine_exe": {"path": "/x/bin/talus-cli", "sha256": "aa".repeat(32), "commit": commit, "dirty": false, "tag": "v0.1.2"},
            "kernels_lib": {"path": "/x/bin/libtalus_kernels.so", "sha256": "bb".repeat(32)},
            "gfx_target": "gfx1151",
            "weights": [{"path": "/m/model.gguf", "sha256": w, "hashing": w.is_none()}],
            "run_config_sha256": "cc".repeat(32),
            "numerics_env": {"GLM_PCH": env},
        })
    }

    fn manifest(builds: Vec<Value>) -> Value {
        let models: Vec<Value> = builds
            .into_iter()
            .enumerate()
            .map(|(i, b)| json!({"id": format!("m{i}"), "server": {"talus_status": {"build": b}}}))
            .collect();
        json!({ "models": models })
    }

    #[test]
    fn table_marks_changes() {
        let m = manifest(vec![
            build("3ad318869db8", Some(&"dd".repeat(32)), "4096"),
            build("96221159abcd", Some(&"dd".repeat(32)), "2048"),
            build("3ad318869db8", None, "4096"),
            Value::Null,
        ]);
        let mut s = String::new();
        fingerprint_table(&mut s, &m);
        assert!(s.contains("| Engine commit | 3ad318869db8 | 96221159abcd **changed** | 3ad318869db8 | not reported |"), "{s}");
        assert!(s.contains("| Weights sha256 | model.gguf `dddddddddddd` | model.gguf `dddddddddddd` | model.gguf hashing | not reported |"), "{s}");
        assert!(s.contains("GLM_PCH=2048 **changed**"), "{s}");
        assert!(s.contains("Not fingerprinted for m3"), "{s}");
        assert!(!s.contains("hashing **changed**"), "{s}");

        let mut s = String::new();
        fingerprint_table(&mut s, &manifest(vec![Value::Null, Value::Null]));
        assert!(
            s.contains("Not fingerprinted (the server does not report"),
            "{s}"
        );
    }

    #[test]
    fn unavailable_fields() {
        assert_eq!(
            crate::runner::unavailable_of(&Value::Null),
            json!([
                "engine build commit",
                "engine sha256",
                "weights sha256",
                "run config sha256",
                "kernels library sha256",
                "gfx target"
            ])
        );
        let b = build("3ad3", Some("ee"), "1");
        assert_eq!(crate::runner::unavailable_of(&b), json!([]));
        assert_eq!(
            crate::runner::unavailable_of(&build("3ad3", None, "1")),
            json!(["weights sha256"])
        );
    }

    #[test]
    fn passive_perf_channel_split() {
        use crate::case::CaseStatus;
        let mut a = crate::compare::tests_support::row("m", "gsm8k", "1", true, CaseStatus::Ok);
        a.usage.completion_tokens = Some(100);
        a.timings.server_decode_ms = Some(2000.0);
        let mut b = a.clone();
        a.timings.reasoning_n = Some(80);
        a.timings.answer_n = Some(20);
        a.timings.reasoning_ms = Some(1600.0);
        a.timings.answer_ms = Some(400.0);
        b.case_id = "gsm8k/2".into();
        let p = crate::runner::passive_perf(&[a.clone(), b.clone()]);
        let c = &p["channels"]["tasks"]["gsm8k"];
        assert_eq!(c["requests"], json!(1));
        assert_eq!(c["reasoning_tok_s"], json!(50.0));
        assert_eq!(c["answer_tok_s"], json!(50.0));
        assert_eq!(p["tasks"]["gsm8k"]["requests"], json!(2));
        // no split reported by the server: no channels key (older output unchanged)
        assert!(crate::runner::passive_perf(&[b])["channels"].is_null());
        // old case rows (no split fields) still parse, and rows without the fields serialize as before
        let text = serde_json::to_string(&a.timings).unwrap();
        assert!(text.contains("reasoning_ms"));
        let plain = serde_json::to_string(&Timings::default()).unwrap();
        assert!(!plain.contains("reasoning"), "{plain}");
        let back: Timings = serde_json::from_str(
            r#"{"client_ms":1.0,"server_prompt_ms":null,"server_decode_ms":null,"draft_n":null,"draft_accepted":null}"#,
        )
        .unwrap();
        let want = Timings {
            client_ms: 1.0,
            ..Default::default()
        };
        assert_eq!(back, want);
    }
}
