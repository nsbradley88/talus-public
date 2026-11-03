//! Per-model metrics and reference/candidate pairing by case id (spec §6.3, §7).
//!
//! Pairing rules:
//! - Rows join on `case_id`. A case that exists on one side only is paired with a `missing` row on the other
//!   side, which counts as a failure there (failures never leave the denominator).
//! - Rows whose prompt hash, dataset revision or scorer version differ are *not comparable*: they are excluded
//!   from the transition table and listed, and the gate for that task becomes NOT_RUN (re-run the reference).

use crate::case::{CaseResult, CaseStatus};
use crate::stats::{self, Interval};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Transitions {
    /// reference pass, candidate pass: retained success
    pub both_pass: usize,
    /// reference pass, candidate fail: regression
    pub regressions: usize,
    /// reference fail, candidate pass: improvement
    pub improvements: usize,
    /// both fail: shared failure
    pub both_fail: usize,
}

impl Transitions {
    pub fn total(&self) -> usize {
        self.both_pass + self.regressions + self.improvements + self.both_fail
    }

    pub fn add(&mut self, r: bool, c: bool) {
        match (r, c) {
            (true, true) => self.both_pass += 1,
            (true, false) => self.regressions += 1,
            (false, true) => self.improvements += 1,
            (false, false) => self.both_fail += 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Mismatch {
    pub case_id: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PairedTask {
    pub task: String,
    pub candidate_id: String,
    /// Paired cases (the unit of the bootstrap).
    pub n_paired: usize,
    /// Paired trials (= cases unless the run repeats over seeds); transitions and McNemar count trials.
    pub n_trials: usize,
    /// Trials whose answer text and reasoning are byte-identical between reference and candidate, out of
    /// `identity_compared` (pairs where both raw outputs were stored).
    pub identical: usize,
    pub identity_compared: usize,
    pub differing_ids: Vec<String>,
    pub missing_in_candidate: Vec<String>,
    pub missing_in_reference: Vec<String>,
    pub not_comparable: Vec<Mismatch>,
    pub transitions: Transitions,
    pub reference_correct: usize,
    pub candidate_correct: usize,
    pub reference_accuracy: Option<f64>,
    pub candidate_accuracy: Option<f64>,
    pub loss_pp: Option<f64>,
    pub retention_pct: Option<f64>,
    pub ci_loss_pp: Option<Interval>,
    pub mcnemar_p: f64,
    pub regression_ids: Vec<String>,
    pub improvement_ids: Vec<String>,
    pub critical_regression_ids: Vec<String>,
}

/// One row per pair, for paired_transitions.csv.
#[derive(Clone, Debug, PartialEq)]
pub struct PairRow {
    pub task: String,
    pub case_id: String,
    pub reference_pass: bool,
    pub candidate_pass: bool,
    pub reference_status: CaseStatus,
    pub candidate_status: CaseStatus,
}

/// Tool calls for identity comparison: the server assigns a fresh call id per request, so only the type,
/// function name and arguments count.
fn calls_without_ids(v: &Value) -> Value {
    match v {
        Value::Array(a) => Value::Array(
            a.iter()
                .map(|c| {
                    let mut c = c.clone();
                    if let Some(o) = c.as_object_mut() {
                        o.remove("id");
                    }
                    c
                })
                .collect(),
        ),
        x => x.clone(),
    }
}

fn label(k: &(String, Option<u64>)) -> String {
    match k.1 {
        Some(s) => format!("{}#s{s}", k.0),
        None => k.0.clone(),
    }
}

/// Pair reference and candidate rows of one task, by (case_id, seed).
pub fn pair_task(
    task: &str,
    candidate_id: &str,
    reference: &[CaseResult],
    candidate: &[CaseResult],
    resamples: u32,
    confidence: f64,
    seed: u64,
) -> (PairedTask, Vec<PairRow>) {
    type Key = (String, Option<u64>);
    let r: BTreeMap<Key, &CaseResult> = reference
        .iter()
        .filter(|x| x.task == task)
        .map(|x| (x.key(), x))
        .collect();
    let c: BTreeMap<Key, &CaseResult> = candidate
        .iter()
        .filter(|x| x.task == task)
        .map(|x| (x.key(), x))
        .collect();
    let mut keys: Vec<&Key> = r.keys().chain(c.keys()).collect();
    keys.sort_unstable();
    keys.dedup();
    let mut t = Transitions::default();
    let mut rows = Vec::new();
    let (mut miss_c, mut miss_r, mut mism) = (Vec::new(), Vec::new(), Vec::new());
    let (mut regs, mut imps, mut crit) = (Vec::new(), Vec::new(), Vec::new());
    let (mut identical, mut compared, mut differing) = (0usize, 0usize, Vec::new());
    // per case: (sum ref, sum cand, trials)
    let mut per_case: BTreeMap<String, (f64, f64, usize)> = BTreeMap::new();
    for key in keys {
        let id = label(key);
        let (rp, rs, cp, cs, critical) = match (r.get(key), c.get(key)) {
            (Some(a), Some(b)) => {
                let mut why = Vec::new();
                if a.prompt_hash != b.prompt_hash {
                    why.push("prompt_hash");
                }
                if a.dataset_revision != b.dataset_revision {
                    why.push("dataset_revision");
                }
                if a.scorer_version != b.scorer_version {
                    why.push("scorer_version");
                }
                if !why.is_empty() {
                    mism.push(Mismatch {
                        case_id: id,
                        reason: format!("{} differ", why.join(", ")),
                    });
                    continue;
                }
                if let (Some(x), Some(y)) = (&a.raw_output, &b.raw_output) {
                    compared += 1;
                    if x.content == y.content
                        && x.reasoning_content == y.reasoning_content
                        && calls_without_ids(&x.tool_calls) == calls_without_ids(&y.tool_calls)
                    {
                        identical += 1;
                    } else {
                        differing.push(id.clone());
                    }
                }
                (
                    a.pass(),
                    a.status,
                    b.pass(),
                    b.status,
                    a.critical || b.critical,
                )
            }
            (Some(a), None) => {
                miss_c.push(id.clone());
                (a.pass(), a.status, false, CaseStatus::Missing, a.critical)
            }
            (None, Some(b)) => {
                miss_r.push(id.clone());
                (false, CaseStatus::Missing, b.pass(), b.status, b.critical)
            }
            (None, None) => unreachable!(),
        };
        t.add(rp, cp);
        let e = per_case.entry(key.0.clone()).or_default();
        e.0 += rp as u8 as f64;
        e.1 += cp as u8 as f64;
        e.2 += 1;
        if rp && !cp {
            regs.push(id.clone());
            if critical {
                crit.push(id.clone());
            }
        }
        if !rp && cp {
            imps.push(id.clone());
        }
        rows.push(PairRow {
            task: task.into(),
            case_id: id,
            reference_pass: rp,
            candidate_pass: cp,
            reference_status: rs,
            candidate_status: cs,
        });
    }
    let n_cases = per_case.len();
    let n_trials = t.total();
    let means: Vec<(f64, f64)> = per_case
        .values()
        .map(|(a, b, n)| (a / *n as f64, b / *n as f64))
        .collect();
    let ra = (n_cases > 0).then(|| means.iter().map(|m| m.0).sum::<f64>() / n_cases as f64);
    let ca = (n_cases > 0).then(|| means.iter().map(|m| m.1).sum::<f64>() / n_cases as f64);
    let diffs: Vec<f64> = means.iter().map(|(a, b)| a - b).collect();
    let p = PairedTask {
        task: task.into(),
        candidate_id: candidate_id.into(),
        n_paired: n_cases,
        n_trials,
        identical,
        identity_compared: compared,
        differing_ids: differing,
        missing_in_candidate: miss_c,
        missing_in_reference: miss_r,
        not_comparable: mism,
        transitions: t,
        reference_correct: t.both_pass + t.regressions,
        candidate_correct: t.both_pass + t.improvements,
        reference_accuracy: ra,
        candidate_accuracy: ca,
        loss_pp: ra.zip(ca).map(|(a, b)| stats::loss_pp(a, b)),
        retention_pct: ra.zip(ca).and_then(|(a, b)| stats::retention_pct(a, b)),
        ci_loss_pp: stats::paired_bootstrap_mean(&diffs, resamples, confidence, seed),
        mcnemar_p: stats::mcnemar_exact_p(t.regressions, t.improvements),
        regression_ids: regs,
        improvement_ids: imps,
        critical_regression_ids: crit,
    };
    (p, rows)
}

/// Paired teacher-forced metrics of `perplexity` and `kld_topk` (their rows carry the scores in `score.parsed`):
/// perplexity of both sides over the same tokens and their ratio, and for `kld_topk` the screening KL statistics.
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct ScoredPair {
    pub task: String,
    pub candidate_id: String,
    /// Cases scored on both sides with identical prompts and ids.
    pub n_paired: usize,
    /// Scored tokens in those cases (per side).
    pub tokens: usize,
    pub not_comparable: Vec<Mismatch>,
    /// Cases without scores on one side (error, missing): they keep the gate from passing.
    pub unscored: Vec<String>,
    pub reference_ppl: Option<f64>,
    pub candidate_ppl: Option<f64>,
    /// candidate / reference perplexity = exp(mean NLL difference per token).
    pub ppl_ratio: Option<f64>,
    /// Paired bootstrap interval of the ratio over cases.
    pub ppl_ratio_ci: Option<Interval>,
    pub kl: Option<crate::kl::TopkKlStats>,
    /// Positions or cases whose top-k data could not be compared.
    pub kl_errors: Vec<String>,
    /// Per case: id, tokens, reference / candidate PPL, ratio, and (kld_topk) mean KL and top-1 agreement.
    pub per_case: Vec<Value>,
}

/// Pair the scored rows of a teacher-forced task by (case_id, seed).
pub fn pair_scored(
    task: &str,
    candidate_id: &str,
    reference: &[CaseResult],
    candidate: &[CaseResult],
    resamples: u32,
    confidence: f64,
    seed: u64,
) -> ScoredPair {
    type Key = (String, Option<u64>);
    let side = |rows: &[CaseResult]| -> BTreeMap<Key, CaseResult> {
        rows.iter()
            .filter(|x| x.task == task)
            .map(|x| (x.key(), x.clone()))
            .collect()
    };
    let r = side(reference);
    let c = side(candidate);
    let mut keys: Vec<&Key> = r.keys().chain(c.keys()).collect();
    keys.sort_unstable();
    keys.dedup();
    let mut out = ScoredPair {
        task: task.into(),
        candidate_id: candidate_id.into(),
        ..Default::default()
    };
    let scored = |x: &CaseResult| {
        x.status == CaseStatus::Ok
            && x.score.pass
            && x.score.parsed["nll_sum"].is_number()
            && x.score.parsed["tokens"].as_u64().unwrap_or(0) > 0
    };
    let (mut nll_r, mut nll_c) = (0.0f64, 0.0f64);
    let mut diffs = Vec::new();
    let mut kls: Vec<crate::kl::TopkKl> = Vec::new();
    for key in keys {
        let id = label(key);
        let (a, b) = match (r.get(key), c.get(key)) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                out.unscored.push(id);
                continue;
            }
        };
        let mut why = Vec::new();
        if a.prompt_hash != b.prompt_hash {
            why.push("prompt_hash (token ids)");
        }
        if a.dataset_revision != b.dataset_revision {
            why.push("dataset_revision");
        }
        if a.scorer_version != b.scorer_version {
            why.push("scorer_version");
        }
        if !why.is_empty() {
            out.not_comparable.push(Mismatch {
                case_id: id,
                reason: format!("{} differ", why.join(", ")),
            });
            continue;
        }
        if !scored(a) || !scored(b) {
            out.unscored.push(id);
            continue;
        }
        let (pa, pb) = (&a.score.parsed, &b.score.parsed);
        let n = pa["tokens"].as_u64().unwrap_or(0) as usize;
        if pb["tokens"].as_u64() != Some(n as u64) {
            out.not_comparable.push(Mismatch {
                case_id: id,
                reason: "scored token counts differ".into(),
            });
            continue;
        }
        let (sa, sb) = (
            pa["nll_sum"].as_f64().unwrap_or(0.0),
            pb["nll_sum"].as_f64().unwrap_or(0.0),
        );
        nll_r += sa;
        nll_c += sb;
        out.tokens += n;
        out.n_paired += 1;
        let d = (sb - sa) / n as f64;
        diffs.push(d);
        let mut row = json!({"case_id": id, "tokens": n, "reference_ppl": (sa / n as f64).exp(),
                             "candidate_ppl": (sb / n as f64).exp(), "ppl_ratio": d.exp()});
        if task == "kld_topk" {
            match (
                crate::tasks::textscore::topk_rows(pa),
                crate::tasks::textscore::topk_rows(pb),
            ) {
                (Some(ra), Some(rb)) if ra.len() == rb.len() => {
                    let mut case_kl = Vec::with_capacity(ra.len());
                    for (j, (x, y)) in ra.iter().zip(&rb).enumerate() {
                        match crate::kl::topk_kl(x, y) {
                            Ok(k) => case_kl.push(k),
                            Err(e) => out.kl_errors.push(format!("{id} position {j}: {e}")),
                        }
                    }
                    if let Some(s) = crate::kl::topk_kl_stats(&case_kl) {
                        row["kl_mean"] = json!(s.mean);
                        row["kl_max"] = json!(s.max);
                        row["top1"] = json!(s.top1);
                        row["reference_covered_mean"] = json!(s.ref_covered_mean);
                    }
                    kls.extend(case_kl);
                }
                _ => out
                    .kl_errors
                    .push(format!("{id}: top-k data missing or of different lengths")),
            }
        }
        out.per_case.push(row);
    }
    if out.tokens > 0 {
        let t = out.tokens as f64;
        out.reference_ppl = Some((nll_r / t).exp());
        out.candidate_ppl = Some((nll_c / t).exp());
        out.ppl_ratio = Some(((nll_c - nll_r) / t).exp());
    }
    // paired_bootstrap_mean reports 100 * mean(d); d is the per-token NLL difference of a case
    out.ppl_ratio_ci =
        stats::paired_bootstrap_mean(&diffs, resamples, confidence, seed).map(|i| Interval {
            lo: (i.lo / 100.0).exp(),
            hi: (i.hi / 100.0).exp(),
            confidence: i.confidence,
        });
    out.kl = crate::kl::topk_kl_stats(&kls);
    out
}

/// perplexity / kld_topk: token-weighted perplexity of one model's scored rows (and kld_topk's own top-k mass).
fn scored_metrics(rs: &[&CaseResult]) -> Value {
    let ok: Vec<&&CaseResult> = rs
        .iter()
        .filter(|r| r.status == CaseStatus::Ok && r.score.pass)
        .collect();
    let tokens: u64 = ok
        .iter()
        .filter_map(|r| r.score.parsed["tokens"].as_u64())
        .sum();
    let nll: f64 = ok
        .iter()
        .filter_map(|r| r.score.parsed["nll_sum"].as_f64())
        .sum();
    let cover: Vec<f64> = ok
        .iter()
        .filter_map(|r| r.score.details["covered_mean"].as_f64())
        .collect();
    json!({
        "scored_cases": ok.len(), "tokens": tokens,
        "mean_nll": (tokens > 0).then(|| nll / tokens as f64),
        "ppl": (tokens > 0).then(|| (nll / tokens as f64).exp()),
        "neg_inf_tokens": ok.iter().filter_map(|r| r.score.parsed["neg_inf_tokens"].as_u64()).sum::<u64>(),
        "topk_covered_mean": (!cover.is_empty()).then(|| cover.iter().sum::<f64>() / cover.len() as f64),
    })
}

/// swebench_verified: resolved / invalid / environment counts, flags by kind, memorization and cutoff splits.
fn swebench_metrics(rs: &[&CaseResult], confidence: f64) -> Value {
    let stage = |r: &CaseResult| r.score.details["stage"].as_str().unwrap_or("").to_string();
    let n = rs.len();
    let resolved = rs.iter().filter(|r| r.pass()).count();
    let mut stages: BTreeMap<String, usize> = BTreeMap::new();
    let mut flags: BTreeMap<String, usize> = BTreeMap::new();
    let mut notes: BTreeMap<String, usize> = BTreeMap::new();
    for r in rs {
        let mut nk: Vec<String> = r.score.details["notes"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|f| f["kind"].as_str().map(str::to_string))
            .collect();
        nk.sort();
        nk.dedup();
        for k in nk {
            *notes.entry(k).or_default() += 1;
        }
        *stages.entry(stage(r)).or_default() += 1;
        let mut kinds: Vec<String> = r.score.details["flags"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|f| f["kind"].as_str().map(str::to_string))
            .collect();
        kinds.sort();
        kinds.dedup();
        for k in kinds {
            *flags.entry(k).or_default() += 1;
        }
    }
    let split = |after: bool| {
        let sel: Vec<&&CaseResult> = rs
            .iter()
            .filter(|r| r.score.details["after_cutoff"] == after)
            .collect();
        json!({"n": sel.len(), "resolved": sel.iter().filter(|r| r.pass()).count()})
    };
    json!({
        "resolved": resolved, "n": n,
        "resolve_rate": stats::accuracy(resolved, n),
        "wilson": stats::wilson(resolved, n, confidence),
        "stages": stages,
        "invalid_instances": stages.get("invalid").copied().unwrap_or(0),
        "environment_errors": stages.get("environment").copied().unwrap_or(0),
        "flags_by_kind": flags,
        "notes_by_kind": notes,
        // the evaluation regardless of flags (flagged patches are evaluated too), reported separately
        "resolved_ignoring_flags": rs.iter().filter(|r| r.score.details["resolved_ignoring_flags"] == true).count(),
        "flagged_but_resolved": rs.iter().filter(|r| r.score.details["stage"] == "invalid" && r.score.details["resolved_ignoring_flags"] == true).count(),
        "near_verbatim": rs.iter().filter(|r| r.score.details["near_verbatim"] == true).count(),
        "before_cutoff": split(false), "after_cutoff": split(true),
    })
}

/// livecodebench: pass@1 per difficulty / platform / test type / cutoff side (Wilson), mean pass@1 over cases
/// (several seeds = several samples per case), failure stages and sandbox flags.
fn livecodebench_metrics(rs: &[&CaseResult], confidence: f64) -> Value {
    let group = |key: &dyn Fn(&CaseResult) -> String| {
        let mut m: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for r in rs {
            let e = m.entry(key(r)).or_default();
            e.1 += 1;
            if r.pass() {
                e.0 += 1;
            }
        }
        let mut o = serde_json::Map::new();
        for (k, (p, n)) in m {
            o.insert(
                k,
                json!({"passed": p, "n": n, "pass_at_1": stats::accuracy(p, n), "wilson": stats::wilson(p, n, confidence)}),
            );
        }
        Value::Object(o)
    };
    // unscored rows (timeouts, truncated by reasoning, ...) carry no score details: fall back to the case's expected
    let d = |r: &CaseResult, k: &str| {
        r.score.details[k]
            .as_str()
            .or_else(|| r.expected[k].as_str())
            .unwrap_or("unknown")
            .to_string()
    };
    let mut stages: BTreeMap<String, usize> = BTreeMap::new();
    let mut flags: BTreeMap<String, usize> = BTreeMap::new();
    let mut per_case: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for r in rs {
        let st = if r.status.scored() {
            d(r, "stage")
        } else {
            format!("not_scored:{}", r.status.as_str())
        };
        *stages.entry(st).or_default() += 1;
        let mut kinds: Vec<String> = r.score.details["flags"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|f| f["kind"].as_str().map(str::to_string))
            .collect();
        kinds.sort();
        kinds.dedup();
        for k in kinds {
            *flags.entry(k).or_default() += 1;
        }
        let e = per_case.entry(r.case_id.as_str()).or_default();
        e.1 += 1;
        if r.pass() {
            e.0 += 1;
        }
    }
    let mean_pass_at_1 = if per_case.is_empty() {
        0.0
    } else {
        per_case
            .values()
            .map(|(p, n)| *p as f64 / *n as f64)
            .sum::<f64>()
            / per_case.len() as f64
    };
    json!({
        "problems": per_case.len(),
        "samples": rs.len(),
        "mean_pass_at_1": mean_pass_at_1,
        "by_difficulty": group(&|r| d(r, "difficulty")),
        "by_platform": group(&|r| d(r, "platform")),
        "by_testtype": group(&|r| d(r, "testtype")),
        "by_cutoff": group(&|r| match r.score.details["after_cutoff"]
            .as_bool()
            .or_else(|| r.expected["after_cutoff"].as_bool())
        {
            Some(true) => "after".into(),
            Some(false) => "before".into(),
            None => "no_cutoff_given".into(),
        }),
        "stages": stages,
        "flags_by_kind": flags,
        "sandbox_violations": rs.iter().filter(|r| r.score.details["stage"] == "sandbox_violation").count(),
        "sandbox_errors": rs.iter().filter(|r| r.score.details["stage"] == "sandbox_error").count(),
        "passed_ignoring_flags": rs.iter().filter(|r| r.score.details["passed_ignoring_flags"] == true).count(),
    })
}

/// Per-model, per-task metrics (metrics.json): counts, accuracy with a Wilson interval, status breakdown and
/// task-specific failure stages.
pub fn model_metrics(rows: &[CaseResult], confidence: f64) -> Value {
    let mut by_task: BTreeMap<&str, Vec<&CaseResult>> = BTreeMap::new();
    for r in rows {
        by_task.entry(r.task.as_str()).or_default().push(r);
    }
    let mut out = serde_json::Map::new();
    for (task, rs) in by_task {
        let n = rs.len();
        let correct = rs.iter().filter(|r| r.pass()).count();
        let mut statuses: BTreeMap<&str, usize> = BTreeMap::new();
        for r in &rs {
            *statuses.entry(r.status.as_str()).or_default() += 1;
        }
        let count = |f: &dyn Fn(&CaseResult) -> bool| rs.iter().filter(|r| f(r)).count();
        let extra = match task {
            "json_schema" | "json_schema_strict" => {
                let d = |r: &CaseResult, k: &str| r.status.scored() && r.score.details[k] == true;
                let parsed = count(&|r| d(r, "parse_ok"));
                json!({
                    "parse_ok": parsed,
                    "fenced_only": count(&|r| d(r, "fenced")),
                    "schema_ok": count(&|r| d(r, "schema_ok")),
                    "semantic_ok": count(&|r| d(r, "semantic_ok")),
                    "with_extra_fields": count(&|r| r.status.scored() && r.score.details["extra_fields"].as_u64().unwrap_or(0) > 0),
                    "extra_field_rate": if parsed > 0 { count(&|r| r.status.scored() && r.score.details["extra_fields"].as_u64().unwrap_or(0) > 0) as f64 / parsed as f64 } else { 0.0 },
                })
            }
            "gsm8k" => json!({
                "strict_correct": count(&|r| r.status.scored() && r.score.details["strict_pass"] == true),
                "no_answer_extracted": count(&|r| r.status.scored() && r.score.details["extraction"] == "none"),
            }),
            "swebench_verified" => swebench_metrics(&rs, confidence),
            "perplexity" | "kld_topk" => scored_metrics(&rs),
            "livecodebench" => livecodebench_metrics(&rs, confidence),
            _ => Value::Null,
        };
        let completion: u64 = rs.iter().filter_map(|r| r.usage.completion_tokens).sum();
        out.insert(
            task.to_string(),
            json!({
                "n": n, "correct": correct,
                "accuracy": stats::accuracy(correct, n),
                "wilson": stats::wilson(correct, n, confidence),
                "statuses": statuses,
                "completion_tokens": completion,
                "task_metrics": extra,
            }),
        );
    }
    Value::Object(out)
}

#[cfg(test)]
pub(crate) mod tests_support {
    pub use super::tests::row;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::case::{Score, Timings, Usage};

    pub fn row(model: &str, task: &str, id: &str, pass: bool, status: CaseStatus) -> CaseResult {
        CaseResult {
            experiment_id: "e".into(),
            model_id: model.into(),
            task: task.into(),
            case_id: format!("{task}/{id}"),
            seed: None,
            dataset: "d".into(),
            dataset_revision: "r1".into(),
            prompt_hash: format!("h-{id}"),
            scorer_version: "s/1".into(),
            settings_hash: "x".into(),
            status,
            prompt: Value::Null,
            expected: Value::Null,
            raw_output: None,
            score: Score::binary(pass, Value::Null, Value::Null),
            error: None,
            usage: Usage::default(),
            timings: Timings::default(),
            attempts: 1,
            critical: false,
            started_at: String::new(),
        }
    }

    #[test]
    fn transition_counts() {
        use CaseStatus::*;
        let outcomes = [
            (true, true),
            (true, false),
            (false, true),
            (false, false),
            (true, true),
            (true, false),
        ];
        let r: Vec<_> = outcomes
            .iter()
            .enumerate()
            .map(|(i, o)| row("ref", "t", &i.to_string(), o.0, Ok))
            .collect();
        let c: Vec<_> = outcomes
            .iter()
            .enumerate()
            .map(|(i, o)| row("cand", "t", &i.to_string(), o.1, Ok))
            .collect();
        let (p, rows) = pair_task("t", "cand", &r, &c, 500, 0.95, 1);
        assert_eq!(
            p.transitions,
            Transitions {
                both_pass: 2,
                regressions: 2,
                improvements: 1,
                both_fail: 1
            }
        );
        assert_eq!(p.n_paired, 6);
        assert_eq!(rows.len(), 6);
        assert_eq!((p.reference_correct, p.candidate_correct), (4, 3));
        assert!((p.loss_pp.unwrap() - 100.0 / 6.0).abs() < 1e-9);
        assert!((p.retention_pct.unwrap() - 75.0).abs() < 1e-9);
        assert_eq!(p.regression_ids, vec!["t/1", "t/5"]);
        assert_eq!(p.improvement_ids, vec!["t/2"]);
    }

    #[test]
    fn errors_and_missing_stay_in_the_denominator() {
        use CaseStatus::*;
        // a scored-but-failed status, a timeout that the scorer marked "pass" (must not count), and a missing case
        let r = vec![
            row("r", "t", "a", true, Ok),
            row("r", "t", "b", true, Ok),
            row("r", "t", "c", true, Ok),
        ];
        let c = vec![
            row("c", "t", "a", true, Ok),
            row("c", "t", "b", true, Timeout),
            row("c", "t", "z", true, Ok),
        ];
        let (p, rows) = pair_task("t", "c", &r, &c, 500, 0.95, 1);
        assert_eq!(
            p.n_paired, 4,
            "a, b, c (missing in candidate), z (missing in reference)"
        );
        assert_eq!(p.missing_in_candidate, vec!["t/c"]);
        assert_eq!(p.missing_in_reference, vec!["t/z"]);
        assert_eq!(p.transitions.regressions, 2, "b timed out, c missing");
        assert_eq!(p.transitions.improvements, 1, "z: reference missing");
        assert!(rows
            .iter()
            .any(|x| x.case_id == "t/c" && x.candidate_status == Missing));
    }

    #[test]
    fn mismatched_rows_are_not_paired() {
        use CaseStatus::*;
        let r = vec![row("r", "t", "a", true, Ok), row("r", "t", "b", true, Ok)];
        let mut c = vec![row("c", "t", "a", false, Ok), row("c", "t", "b", true, Ok)];
        c[0].prompt_hash = "other".into();
        c[1].scorer_version = "s/2".into();
        let (p, _) = pair_task("t", "c", &r, &c, 500, 0.95, 1);
        assert_eq!(p.n_paired, 0);
        assert_eq!(p.not_comparable.len(), 2);
        assert!(p.not_comparable[0].reason.contains("prompt_hash"));
        assert!(p.not_comparable[1].reason.contains("scorer_version"));
        assert_eq!(p.loss_pp, None);
        assert_eq!(p.ci_loss_pp, None);
    }

    #[test]
    fn critical_regressions_are_flagged() {
        use CaseStatus::*;
        let mut r = vec![row("r", "t", "a", true, Ok)];
        r[0].critical = true;
        let c = vec![row("c", "t", "a", false, Ok)];
        let (p, _) = pair_task("t", "c", &r, &c, 500, 0.95, 1);
        assert_eq!(p.critical_regression_ids, vec!["t/a"]);
    }

    #[test]
    fn repeated_trials_average_per_case() {
        use CaseStatus::*;
        // case a: ref 2/2, cand 1/2; case b: ref 0/2, cand 0/2 -> loss = 100 * ((1 - 0.5) + 0) / 2 = 25 pp
        let mut r = Vec::new();
        let mut c = Vec::new();
        for (id, s, rp, cp) in [
            ("a", 1, true, true),
            ("a", 2, true, false),
            ("b", 1, false, false),
            ("b", 2, false, false),
        ] {
            let mut x = row("r", "t", id, rp, Ok);
            x.seed = Some(s);
            r.push(x);
            let mut y = row("c", "t", id, cp, Ok);
            y.seed = Some(s);
            c.push(y);
        }
        let (p, rows) = pair_task("t", "c", &r, &c, 500, 0.95, 1);
        assert_eq!((p.n_paired, p.n_trials), (2, 4));
        assert_eq!(p.transitions.regressions, 1);
        assert!((p.loss_pp.unwrap() - 25.0).abs() < 1e-9);
        assert_eq!(p.regression_ids, vec!["t/a#s2"]);
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn identity_counts() {
        use crate::case::RawOutput;
        use CaseStatus::*;
        let mut r = vec![row("r", "t", "a", true, Ok), row("r", "t", "b", true, Ok)];
        let mut c = vec![row("c", "t", "a", true, Ok), row("c", "t", "b", true, Ok)];
        for (i, x) in r.iter_mut().enumerate() {
            x.raw_output = Some(RawOutput {
                content: format!("same{i}"),
                ..Default::default()
            });
        }
        c[0].raw_output = Some(RawOutput {
            content: "same0".into(),
            ..Default::default()
        });
        c[1].raw_output = Some(RawOutput {
            content: "other".into(),
            ..Default::default()
        });
        let (p, _) = pair_task("t", "c", &r, &c, 500, 0.95, 1);
        assert_eq!((p.identical, p.identity_compared), (1, 2));
        assert_eq!(p.differing_ids, vec!["t/b"]);
        // tool calls differing only in the server-assigned id count as identical
        let call = |id: &str| serde_json::json!([{"id": id, "type": "function", "function": {"name": "f", "arguments": "{}"}}]);
        r[1].raw_output = Some(RawOutput {
            tool_calls: call("call_a"),
            ..Default::default()
        });
        c[1].raw_output = Some(RawOutput {
            tool_calls: call("call_b"),
            ..Default::default()
        });
        let (p, _) = pair_task("t", "c", &r, &c, 500, 0.95, 1);
        assert_eq!(p.identical, 2);
    }

    #[test]
    fn metrics_count_statuses() {
        use CaseStatus::*;
        let rows = vec![
            row("m", "gsm8k", "a", true, Ok),
            row("m", "gsm8k", "b", false, Truncated),
            row("m", "gsm8k", "c", true, ServerError),
        ];
        let m = model_metrics(&rows, 0.95);
        assert_eq!(m["gsm8k"]["n"], 3);
        assert_eq!(m["gsm8k"]["correct"], 1);
        assert_eq!(m["gsm8k"]["statuses"]["server_error"], 1);
    }
}
