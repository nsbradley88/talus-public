//! Release gates (spec §8) with statistical uncertainty.
//!
//! Loss gate (accuracy loss in pp, target T, investigate I, paired bootstrap CI [lo, hi]):
//! - FAIL when lo > T: the whole interval is beyond the target, so the degradation is verified;
//! - PASS when hi <= T, at least `min_cases` pairs, and no critical regression;
//! - WARN otherwise (interval overlaps T, small sample, point loss > I "investigate", critical regressions).
//!
//! Rate gate (candidate rate in %, e.g. strict JSON validity; Wilson interval [lo, hi]):
//! - FAIL when hi < T (verified below target); PASS when rate >= T with at least `min_cases` cases; else WARN.
//!
//! Perplexity ratio gate (candidate / reference PPL over the same teacher-forced tokens, target T, investigate I,
//! paired bootstrap interval [lo, hi] over cases): FAIL when lo > T; PASS when hi <= T with at least `min_cases`
//! cases and every case scored on both sides; else WARN (point ratio > I adds "investigate").
//!
//! Screening KL gate (`kld_topk`): NOT_RUN without comparable positions; a mean above `max_mean` or a top-1
//! agreement below `min_top1_pct` WARNs (FAILs only with `enforce: true`); else PASS.
//!
//! Overall per candidate: FAIL if a required gate fails; else NOT_RUN if a required gate is NOT_RUN or
//! NOT_SUPPORTED (no overall PASS while required tests are missing); else WARN if anything warns (a non-required
//! FAIL also warns); else PASS.

use crate::compare::{PairedTask, ScoredPair};
use crate::config::{GateConfig, KldTopkGate, Threshold, Tier};
use crate::stats::{self, Interval};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GateStatus {
    #[serde(rename = "PASS")]
    Pass,
    #[serde(rename = "WARN")]
    Warn,
    #[serde(rename = "FAIL")]
    Fail,
    #[serde(rename = "NOT_RUN")]
    NotRun,
    #[serde(rename = "NOT_SUPPORTED")]
    NotSupported,
}

impl GateStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            GateStatus::Pass => "PASS",
            GateStatus::Warn => "WARN",
            GateStatus::Fail => "FAIL",
            GateStatus::NotRun => "NOT_RUN",
            GateStatus::NotSupported => "NOT_SUPPORTED",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GateResult {
    /// e.g. "gsm8k accuracy loss", "json_schema validity".
    pub gate: String,
    pub task: String,
    /// loss_pp | min_rate_pct | availability
    pub kind: String,
    pub required: bool,
    pub status: GateStatus,
    pub value: Option<f64>,
    pub threshold: Option<Threshold>,
    pub interval: Option<Interval>,
    pub n: usize,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Verdict {
    pub candidate_id: String,
    pub overall: GateStatus,
    pub gates: Vec<GateResult>,
    pub reasons: Vec<String>,
}

/// Display unit of a gate's value.
pub fn unit(kind: &str) -> &'static str {
    match kind {
        "loss_pp" => " pp",
        "min_rate_pct" | "all_pass" | "identity" => "%",
        "ppl_ratio" => "x",
        "kld_topk" => " nats",
        _ => "",
    }
}

pub fn loss_gate(p: &PairedTask, t: Threshold, cfg: &GateConfig, required: bool) -> GateResult {
    let mut g = GateResult {
        gate: format!("{} accuracy loss", p.task),
        task: p.task.clone(),
        kind: "loss_pp".into(),
        required,
        status: GateStatus::Warn,
        value: p.loss_pp,
        threshold: Some(t),
        interval: p.ci_loss_pp,
        n: p.n_paired,
        reasons: Vec::new(),
    };
    if !p.not_comparable.is_empty() {
        g.status = GateStatus::NotRun;
        g.reasons.push(format!(
            "{} case(s) not comparable with the reference ({}); re-run or re-freeze the reference",
            p.not_comparable.len(),
            p.not_comparable[0].reason
        ));
        return g;
    }
    let Some(loss) = p.loss_pp else {
        g.status = GateStatus::NotRun;
        g.reasons.push("no paired cases".into());
        return g;
    };
    if let Some(ci) = p.ci_loss_pp {
        if ci.lo > t.target {
            g.status = GateStatus::Fail;
            g.reasons.push(format!(
                "loss {loss:.2} pp, {:.0}% CI [{:.2}, {:.2}] lies above the {:.2} pp target",
                ci.confidence * 100.0,
                ci.lo,
                ci.hi,
                t.target
            ));
            return g;
        }
    }
    let mut ok = true;
    match p.ci_loss_pp {
        Some(ci) if ci.hi > t.target => {
            ok = false;
            g.reasons.push(format!(
                "{:.0}% CI upper bound {:.2} pp exceeds the {:.2} pp target",
                ci.confidence * 100.0,
                ci.hi,
                t.target
            ));
        }
        None => {
            ok = false;
            g.reasons.push("too few pairs for an interval".into());
        }
        _ => {}
    }
    if loss > t.investigate {
        ok = false;
        g.reasons.push(format!(
            "point loss {loss:.2} pp > {:.2} pp: investigate",
            t.investigate
        ));
    }
    if p.n_paired < cfg.min_cases {
        ok = false;
        g.reasons.push(format!(
            "sample too small ({} < min_cases {})",
            p.n_paired, cfg.min_cases
        ));
    }
    if !p.critical_regression_ids.is_empty() {
        ok = false;
        g.reasons.push(format!(
            "critical regression(s) need manual review: {}",
            p.critical_regression_ids.join(", ")
        ));
    }
    if ok {
        g.status = GateStatus::Pass;
    }
    g
}

pub fn rate_gate(
    task: &str,
    correct: usize,
    n: usize,
    t: Threshold,
    cfg: &GateConfig,
    required: bool,
) -> GateResult {
    let mut g = GateResult {
        gate: format!("{task} rate"),
        task: task.into(),
        kind: "min_rate_pct".into(),
        required,
        status: GateStatus::Warn,
        value: stats::accuracy(correct, n).map(|a| a * 100.0),
        threshold: Some(t),
        interval: stats::wilson(correct, n, cfg.confidence).map(|w| Interval {
            lo: w.lo * 100.0,
            hi: w.hi * 100.0,
            confidence: w.confidence,
        }),
        n,
        reasons: Vec::new(),
    };
    let (Some(rate), Some(w)) = (g.value, g.interval) else {
        g.status = GateStatus::NotRun;
        g.reasons.push("no cases".into());
        return g;
    };
    if w.hi < t.target {
        g.status = GateStatus::Fail;
        g.reasons.push(format!(
            "rate {rate:.2}% ({correct}/{n}), {:.0}% Wilson interval [{:.2}, {:.2}] lies below the {:.2}% target",
            w.confidence * 100.0,
            w.lo,
            w.hi,
            t.target
        ));
        return g;
    }
    let mut ok = true;
    if rate < t.target {
        ok = false;
        g.reasons.push(format!(
            "rate {rate:.2}% below the {:.2}% target (not verified)",
            t.target
        ));
    }
    if rate < t.investigate {
        g.reasons
            .push(format!("rate below {:.2}%: investigate", t.investigate));
    }
    if n < cfg.min_cases {
        ok = false;
        g.reasons.push(format!(
            "sample too small ({n} < min_cases {})",
            cfg.min_cases
        ));
    }
    if ok {
        g.status = GateStatus::Pass;
        if w.lo < t.target {
            g.reasons.push(format!(
                "PASS on the point estimate (small sample): Wilson lower bound {:.1}% is below the {:.2}% target",
                w.lo, t.target
            ));
        }
    }
    g
}

/// Every candidate case must pass (release gates such as needle 15/15 or agentic 5/5).
pub fn all_pass_gate(task: &str, rows: &[&crate::case::CaseResult], required: bool) -> GateResult {
    let n = rows.len();
    let failed: Vec<String> = rows
        .iter()
        .filter(|r| !r.pass())
        .map(|r| match r.seed {
            Some(s) => format!("{}#s{s}", r.case_id),
            None => r.case_id.clone(),
        })
        .collect();
    let mut g = GateResult {
        gate: format!("{task} all cases pass"),
        task: task.into(),
        kind: "all_pass".into(),
        required,
        status: GateStatus::Pass,
        value: stats::accuracy(n - failed.len(), n).map(|a| a * 100.0),
        threshold: Some(Threshold {
            target: 100.0,
            investigate: 100.0,
        }),
        interval: None,
        n,
        reasons: Vec::new(),
    };
    if n == 0 {
        g.status = GateStatus::NotRun;
        g.reasons.push("no cases".into());
    } else if !failed.is_empty() {
        g.status = GateStatus::Fail;
        g.reasons.push(format!(
            "{} of {n} failed: {}",
            failed.len(),
            failed.join(", ")
        ));
    }
    g
}

/// The candidate's outputs must be byte-identical to the reference's (greedy identity for exact changes).
pub fn identity_gate(p: &PairedTask, required: bool) -> GateResult {
    let mut g = GateResult {
        gate: format!("{} byte identity", p.task),
        task: p.task.clone(),
        kind: "identity".into(),
        required,
        status: GateStatus::Pass,
        value: (p.identity_compared > 0)
            .then(|| 100.0 * p.identical as f64 / p.identity_compared as f64),
        threshold: Some(Threshold {
            target: 100.0,
            investigate: 100.0,
        }),
        interval: None,
        n: p.identity_compared,
        reasons: Vec::new(),
    };
    let expected = p.n_trials;
    if p.identity_compared == 0 || !p.not_comparable.is_empty() {
        g.status = GateStatus::NotRun;
        g.reasons.push(if p.identity_compared == 0 {
            "no output pairs to compare (raw outputs not stored?)".into()
        } else {
            "cases not comparable with the reference".into()
        });
    } else if p.identical < expected {
        g.status = GateStatus::Fail;
        let missing = expected - p.identity_compared;
        g.reasons.push(format!(
            "{} of {expected} differ{}: {}",
            expected - p.identical,
            if missing > 0 {
                format!(" ({missing} without an answer to compare)")
            } else {
                String::new()
            },
            p.differing_ids
                .iter()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    g
}

fn scored_not_comparable(p: &ScoredPair) -> Option<String> {
    (!p.not_comparable.is_empty()).then(|| {
        format!(
            "{} case(s) not comparable with the reference ({}: {}); the endpoints must tokenize the text identically, else re-run or re-freeze the reference",
            p.not_comparable.len(),
            p.not_comparable[0].case_id,
            p.not_comparable[0].reason
        )
    })
}

/// `perplexity`: candidate / reference perplexity ratio (see the module docs).
pub fn ppl_ratio_gate(
    p: &ScoredPair,
    t: Threshold,
    cfg: &GateConfig,
    required: bool,
) -> GateResult {
    let mut g = GateResult {
        gate: format!("{} PPL ratio", p.task),
        task: p.task.clone(),
        kind: "ppl_ratio".into(),
        required,
        status: GateStatus::Warn,
        value: p.ppl_ratio,
        threshold: Some(t),
        interval: p.ppl_ratio_ci,
        n: p.n_paired,
        reasons: Vec::new(),
    };
    if let Some(why) = scored_not_comparable(p) {
        g.status = GateStatus::NotRun;
        g.reasons.push(why);
        return g;
    }
    let (Some(ratio), Some(rp), Some(cp)) = (p.ppl_ratio, p.reference_ppl, p.candidate_ppl) else {
        g.status = GateStatus::NotRun;
        g.reasons.push(format!(
            "no case scored on both sides ({} unscored)",
            p.unscored.len()
        ));
        return g;
    };
    g.reasons.push(format!(
        "PPL reference {rp:.4} -> candidate {cp:.4} over {} tokens in {} cases",
        p.tokens, p.n_paired
    ));
    if let Some(ci) = p.ppl_ratio_ci {
        if ci.lo > t.target {
            g.status = GateStatus::Fail;
            g.reasons.push(format!(
                "ratio {ratio:.4}, {:.0}% CI [{:.4}, {:.4}] lies above the {} target",
                ci.confidence * 100.0,
                ci.lo,
                ci.hi,
                t.target
            ));
            return g;
        }
    }
    let mut ok = true;
    match p.ppl_ratio_ci {
        Some(ci) if ci.hi > t.target => {
            ok = false;
            g.reasons.push(format!(
                "{:.0}% CI upper bound {:.4} exceeds the {} target",
                ci.confidence * 100.0,
                ci.hi,
                t.target
            ));
        }
        None => {
            ok = false;
            g.reasons.push("too few cases for an interval".into());
        }
        _ => {}
    }
    if ratio > t.investigate {
        ok = false;
        g.reasons.push(format!(
            "point ratio {ratio:.4} > {}: investigate",
            t.investigate
        ));
    }
    if p.n_paired < cfg.min_cases {
        ok = false;
        g.reasons.push(format!(
            "sample too small ({} < min_cases {})",
            p.n_paired, cfg.min_cases
        ));
    }
    if !p.unscored.is_empty() {
        ok = false;
        g.reasons.push(format!(
            "{} case(s) without scores on one side: {}",
            p.unscored.len(),
            p.unscored
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if ok {
        g.status = GateStatus::Pass;
    }
    g
}

/// `kld_topk`: screening KL from top-k lists (see the module docs). Value: mean KL in nats.
pub fn kld_topk_gate(p: &ScoredPair, t: &KldTopkGate, required: bool) -> GateResult {
    let mut g = GateResult {
        gate: format!("{} screening KL", p.task),
        task: p.task.clone(),
        kind: "kld_topk".into(),
        required,
        status: GateStatus::Warn,
        value: p.kl.as_ref().map(|k| k.mean),
        threshold: Some(Threshold {
            target: t.max_mean,
            investigate: t.max_mean,
        }),
        interval: None,
        n: p.kl.as_ref().map(|k| k.positions).unwrap_or(0),
        reasons: Vec::new(),
    };
    if let Some(why) = scored_not_comparable(p) {
        g.status = GateStatus::NotRun;
        g.reasons.push(why);
        return g;
    }
    let Some(k) = &p.kl else {
        g.status = GateStatus::NotRun;
        g.reasons.push(format!(
            "no comparable positions ({} case(s) unscored{})",
            p.unscored.len(),
            p.kl_errors
                .first()
                .map(|e| format!("; {e}"))
                .unwrap_or_default()
        ));
        return g;
    };
    g.reasons.push(format!(
        "KL mean {:.3e} p50 {:.3e} p99 {:.3e} max {:.3e} nats, top-1 {:.2}%, |dlogprob| of the forced token mean {:.4} max {:.4}, over {} positions in {} cases; reference top-k covers {:.2}% of the mass on average (min {:.2}%; the rest is the recorded tail), candidate {:.2}%",
        k.mean,
        k.p50,
        k.p99,
        k.max,
        k.top1 * 100.0,
        k.abs_dlp_mean,
        k.abs_dlp_max,
        k.positions,
        p.n_paired,
        k.ref_covered_mean * 100.0,
        k.ref_covered_min * 100.0,
        k.cand_covered_mean * 100.0
    ));
    let mut breach = Vec::new();
    if k.mean > t.max_mean {
        breach.push(format!("mean {:.3e} > {}", k.mean, t.max_mean));
    }
    if k.top1 * 100.0 < t.min_top1_pct {
        breach.push(format!(
            "top-1 {:.2}% < {}%",
            k.top1 * 100.0,
            t.min_top1_pct
        ));
    }
    let incomplete = !p.unscored.is_empty() || !p.kl_errors.is_empty();
    g.status = if !breach.is_empty() {
        if t.enforce {
            GateStatus::Fail
        } else {
            GateStatus::Warn
        }
    } else if incomplete {
        GateStatus::Warn
    } else {
        GateStatus::Pass
    };
    if !breach.is_empty() {
        g.reasons.push(breach.join(", "));
    }
    if incomplete {
        g.reasons.push(format!(
            "{} case(s) unscored, {} comparison error(s)",
            p.unscored.len(),
            p.kl_errors.len()
        ));
    }
    if !t.enforce {
        g.reasons.push("screening metric with provisional thresholds: WARN-only (quality_gates.kld_topk.enforce: false); the full-vocabulary release gate is kld_teacher_forced on the GPU host".into());
    }
    g
}

pub fn availability_gate(
    task: &str,
    status: GateStatus,
    reason: String,
    required: bool,
) -> GateResult {
    GateResult {
        gate: task.to_string(),
        task: task.into(),
        kind: "availability".into(),
        required,
        status,
        value: None,
        threshold: None,
        interval: None,
        n: 0,
        reasons: vec![reason],
    }
}

/// Combine gate results into the candidate's overall status.
pub fn overall(
    candidate_id: &str,
    gates: Vec<GateResult>,
    tier: Tier,
    tasks_run: &[String],
) -> Verdict {
    let mut reasons = Vec::new();
    let req_fail: Vec<&GateResult> = gates
        .iter()
        .filter(|g| g.required && g.status == GateStatus::Fail)
        .collect();
    let req_missing: Vec<&GateResult> = gates
        .iter()
        .filter(|g| g.required && matches!(g.status, GateStatus::NotRun | GateStatus::NotSupported))
        .collect();
    let warn = gates
        .iter()
        .any(|g| g.status == GateStatus::Warn || (!g.required && g.status == GateStatus::Fail));
    let mut status = if !req_fail.is_empty() {
        reasons.push(format!("required gate(s) failed: {}", names(&req_fail)));
        GateStatus::Fail
    } else if !req_missing.is_empty() {
        reasons.push(format!(
            "required test(s) not run or not supported: {}",
            names(&req_missing)
        ));
        GateStatus::NotRun
    } else if warn {
        reasons.push("some gates warn; see the gate table".into());
        GateStatus::Warn
    } else {
        GateStatus::Pass
    };
    if tier == Tier::Qualification
        && !tasks_run.iter().any(|t| t == "agentic_go")
        && status != GateStatus::Fail
    {
        reasons.push(
            "qualification tier without the custom agent task (agentic_go): no production PASS"
                .into(),
        );
        status = GateStatus::NotRun;
    }
    if gates.is_empty() {
        reasons.push("no gates evaluated".into());
        status = GateStatus::NotRun;
    }
    Verdict {
        candidate_id: candidate_id.into(),
        overall: status,
        gates,
        reasons,
    }
}

fn names(g: &[&GateResult]) -> String {
    g.iter()
        .map(|x| x.gate.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::{Mismatch, Transitions};

    fn paired(n: usize, loss: f64, lo: f64, hi: f64) -> PairedTask {
        PairedTask {
            task: "gsm8k".into(),
            candidate_id: "c".into(),
            n_paired: n,
            n_trials: n,
            identical: 0,
            identity_compared: 0,
            differing_ids: vec![],
            missing_in_candidate: vec![],
            missing_in_reference: vec![],
            not_comparable: vec![],
            transitions: Transitions::default(),
            reference_correct: 0,
            candidate_correct: 0,
            reference_accuracy: Some(0.8),
            candidate_accuracy: Some(0.8 - loss / 100.0),
            loss_pp: Some(loss),
            retention_pct: None,
            ci_loss_pp: Some(Interval {
                lo,
                hi,
                confidence: 0.95,
            }),
            mcnemar_p: 1.0,
            regression_ids: vec![],
            improvement_ids: vec![],
            critical_regression_ids: vec![],
        }
    }

    const T: Threshold = Threshold {
        target: 2.0,
        investigate: 4.0,
    };

    #[test]
    fn loss_gate_cases() {
        let cfg = GateConfig::default();
        assert_eq!(
            loss_gate(&paired(200, 0.0, 0.0, 0.0), T, &cfg, true).status,
            GateStatus::Pass
        );
        assert_eq!(
            loss_gate(&paired(200, -1.0, -3.0, 1.5), T, &cfg, true).status,
            GateStatus::Pass
        );
        let g = loss_gate(&paired(200, 1.0, -1.0, 3.0), T, &cfg, true);
        assert_eq!(g.status, GateStatus::Warn);
        assert!(g.reasons[0].contains("upper bound"));
        let g = loss_gate(&paired(200, 8.0, 4.5, 11.0), T, &cfg, true);
        assert_eq!(g.status, GateStatus::Fail);
        // exactly at the target is not "beyond" it
        assert_ne!(
            loss_gate(&paired(200, 3.0, 2.0, 4.0), T, &cfg, true).status,
            GateStatus::Fail
        );
        let g = loss_gate(&paired(10, 0.0, 0.0, 0.0), T, &cfg, true);
        assert_eq!(g.status, GateStatus::Warn);
        assert!(g.reasons.iter().any(|r| r.contains("sample too small")));
        let g = loss_gate(&paired(200, 5.0, 1.0, 9.0), T, &cfg, true);
        assert!(g.reasons.iter().any(|r| r.contains("investigate")));
        let mut p = paired(200, 0.0, 0.0, 0.0);
        p.critical_regression_ids = vec!["x".into()];
        assert_eq!(loss_gate(&p, T, &cfg, true).status, GateStatus::Warn);
        let mut p = paired(200, 0.0, 0.0, 0.0);
        p.not_comparable = vec![Mismatch {
            case_id: "a".into(),
            reason: "prompt_hash differ".into(),
        }];
        assert_eq!(loss_gate(&p, T, &cfg, true).status, GateStatus::NotRun);
        let mut p = paired(1, 0.0, 0.0, 0.0);
        p.ci_loss_pp = None;
        assert_eq!(loss_gate(&p, T, &cfg, true).status, GateStatus::Warn);
    }

    #[test]
    fn rate_gate_cases() {
        let cfg = GateConfig::default();
        let t = Threshold {
            target: 99.5,
            investigate: 99.0,
        };
        assert_eq!(
            rate_gate("j", 1000, 1000, t, &cfg, true).status,
            GateStatus::Pass
        );
        let g = rate_gate("j", 20, 20, t, &cfg, true);
        assert_eq!(g.status, GateStatus::Warn, "{:?}", g.reasons);
        assert_eq!(
            rate_gate("j", 15, 20, t, &cfg, true).status,
            GateStatus::Fail
        );
        assert_eq!(
            rate_gate("j", 0, 0, t, &cfg, true).status,
            GateStatus::NotRun
        );
        // 996/1000 = 99.6%: pass on the point estimate with an adequate sample
        assert_eq!(
            rate_gate("j", 996, 1000, t, &cfg, true).status,
            GateStatus::Pass
        );
    }

    #[test]
    fn all_pass_and_identity_gates() {
        use crate::case::CaseStatus;
        let mk = |id: &str, pass: bool| {
            let mut r = crate::compare::tests_support::row("c", "needle", id, pass, CaseStatus::Ok);
            r.seed = None;
            r
        };
        let rows = [mk("a", true), mk("b", true), mk("c", true)];
        let refs: Vec<&crate::case::CaseResult> = rows.iter().collect();
        assert_eq!(
            all_pass_gate("needle", &refs, true).status,
            GateStatus::Pass
        );
        let rows2 = [mk("a", true), mk("b", false)];
        let refs2: Vec<&crate::case::CaseResult> = rows2.iter().collect();
        let g = all_pass_gate("needle", &refs2, true);
        assert_eq!(g.status, GateStatus::Fail);
        assert!(g.reasons[0].contains("needle/b"));
        assert_eq!(
            all_pass_gate("needle", &[], true).status,
            GateStatus::NotRun
        );
        let mut p = paired(9, 0.0, 0.0, 0.0);
        p.identity_compared = 9;
        p.identical = 9;
        assert_eq!(identity_gate(&p, true).status, GateStatus::Pass);
        p.identical = 8;
        p.differing_ids = vec!["greedy_identity/edit-2".into()];
        let g = identity_gate(&p, true);
        assert_eq!(g.status, GateStatus::Fail);
        assert!(g.reasons[0].contains("edit-2"));
        p.identity_compared = 0;
        assert_eq!(identity_gate(&p, true).status, GateStatus::NotRun);
        // rate gate: PASS on a small sample says so
        let cfg = GateConfig {
            min_cases: 20,
            ..Default::default()
        };
        let t = Threshold {
            target: 99.5,
            investigate: 99.0,
        };
        let g = rate_gate("j", 20, 20, t, &cfg, true);
        assert_eq!(g.status, GateStatus::Pass);
        assert!(g.reasons[0].contains("small sample"), "{:?}", g.reasons);
    }

    fn scored(n: usize, ratio: f64, lo: f64, hi: f64) -> ScoredPair {
        ScoredPair {
            task: "perplexity".into(),
            candidate_id: "c".into(),
            n_paired: n,
            tokens: n * 256,
            reference_ppl: Some(5.0),
            candidate_ppl: Some(5.0 * ratio),
            ppl_ratio: Some(ratio),
            ppl_ratio_ci: Some(Interval {
                lo,
                hi,
                confidence: 0.95,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn ppl_ratio_gate_cases() {
        let cfg = GateConfig::default();
        let t = crate::config::default_ppl_ratio();
        assert_eq!(
            ppl_ratio_gate(&scored(32, 1.0, 1.0, 1.0), t, &cfg, true).status,
            GateStatus::Pass
        );
        let g = ppl_ratio_gate(&scored(32, 1.008, 1.004, 1.012), t, &cfg, true);
        assert_eq!(g.status, GateStatus::Warn);
        assert!(g.reasons.iter().any(|r| r.contains("upper bound")));
        let g = ppl_ratio_gate(&scored(32, 1.07, 1.06, 1.08), t, &cfg, true);
        assert_eq!(g.status, GateStatus::Fail);
        assert_eq!(
            ppl_ratio_gate(&scored(10, 1.0, 1.0, 1.0), t, &cfg, true).status,
            GateStatus::Warn
        );
        let mut p = scored(32, 1.0, 1.0, 1.0);
        p.unscored = vec!["perplexity/w003".into()];
        assert_eq!(ppl_ratio_gate(&p, t, &cfg, true).status, GateStatus::Warn);
        let mut p = scored(32, 1.0, 1.0, 1.0);
        p.not_comparable = vec![crate::compare::Mismatch {
            case_id: "perplexity/w000".into(),
            reason: "prompt_hash (token ids) differ".into(),
        }];
        let g = ppl_ratio_gate(&p, t, &cfg, true);
        assert_eq!(g.status, GateStatus::NotRun);
        assert!(g.reasons[0].contains("tokenize"));
        let empty = ScoredPair::default();
        assert_eq!(
            ppl_ratio_gate(&empty, t, &cfg, true).status,
            GateStatus::NotRun
        );
    }

    #[test]
    fn kld_topk_gate_is_warn_only_by_default() {
        let stats = |mean: f64, top1: f64| crate::kl::TopkKlStats {
            positions: 4096,
            mean,
            p50: mean,
            p99: mean * 3.0,
            max: mean * 10.0,
            top1,
            ref_covered_mean: 0.97,
            ref_covered_min: 0.6,
            cand_covered_mean: 0.96,
            abs_dlp_mean: 0.01,
            abs_dlp_max: 0.2,
        };
        let mut p = ScoredPair {
            task: "kld_topk".into(),
            n_paired: 16,
            kl: Some(stats(1e-3, 0.99)),
            ..Default::default()
        };
        let t = KldTopkGate::default();
        let g = kld_topk_gate(&p, &t, true);
        assert_eq!(g.status, GateStatus::Pass);
        assert!(g.reasons[0].contains("covers 97.00%"), "{:?}", g.reasons);
        assert!(g.reasons.iter().any(|r| r.contains("WARN-only")));
        p.kl = Some(stats(0.2, 0.99));
        assert_eq!(kld_topk_gate(&p, &t, true).status, GateStatus::Warn);
        p.kl = Some(stats(1e-3, 0.5));
        assert_eq!(kld_topk_gate(&p, &t, true).status, GateStatus::Warn);
        let enforce = KldTopkGate {
            enforce: true,
            ..Default::default()
        };
        assert_eq!(kld_topk_gate(&p, &enforce, true).status, GateStatus::Fail);
        p.kl = None;
        assert_eq!(kld_topk_gate(&p, &t, true).status, GateStatus::NotRun);
    }

    fn g(task: &str, s: GateStatus, required: bool) -> GateResult {
        availability_gate(task, s, String::new(), required)
    }

    #[test]
    fn overall_rules() {
        use GateStatus::*;
        let run = vec!["a".to_string()];
        assert_eq!(
            overall("c", vec![g("a", Pass, true)], Tier::Smoke, &run).overall,
            Pass
        );
        assert_eq!(
            overall(
                "c",
                vec![g("a", Pass, true), g("b", Warn, true)],
                Tier::Smoke,
                &run
            )
            .overall,
            Warn
        );
        assert_eq!(
            overall(
                "c",
                vec![g("a", Pass, true), g("b", NotRun, true)],
                Tier::Smoke,
                &run
            )
            .overall,
            NotRun
        );
        assert_eq!(
            overall(
                "c",
                vec![g("a", Pass, true), g("b", NotSupported, true)],
                Tier::Smoke,
                &run
            )
            .overall,
            NotRun
        );
        assert_eq!(
            overall(
                "c",
                vec![g("a", Fail, true), g("b", NotRun, true)],
                Tier::Smoke,
                &run
            )
            .overall,
            Fail
        );
        assert_eq!(
            overall(
                "c",
                vec![g("a", Pass, true), g("b", Fail, false)],
                Tier::Smoke,
                &run
            )
            .overall,
            Warn
        );
        assert_eq!(
            overall(
                "c",
                vec![g("a", Pass, true), g("b", NotRun, false)],
                Tier::Smoke,
                &run
            )
            .overall,
            Pass
        );
        assert_eq!(overall("c", vec![], Tier::Smoke, &run).overall, NotRun);
        let v = overall("c", vec![g("a", Pass, true)], Tier::Qualification, &run);
        assert_eq!(v.overall, NotRun);
        assert!(v.reasons[0].contains("agentic_go"));
    }
}
