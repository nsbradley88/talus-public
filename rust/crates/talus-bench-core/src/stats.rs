//! Statistics for paired binary outcomes (spec §7): accuracy, percentage-point loss, relative retention, the
//! paired bootstrap over case ids, McNemar's exact test, and the Wilson interval for a single rate.
//! Percentage points and relative retention are kept apart; retention is undefined (None) at a zero baseline.

use crate::util::Rng;
use serde::{Deserialize, Serialize};

/// correct / total; None when total = 0.
pub fn accuracy(correct: usize, total: usize) -> Option<f64> {
    (total > 0).then(|| correct as f64 / total as f64)
}

/// 100 * (reference - candidate), in percentage points. Positive = the candidate lost accuracy.
pub fn loss_pp(reference_acc: f64, candidate_acc: f64) -> f64 {
    100.0 * (reference_acc - candidate_acc)
}

/// 100 * candidate / reference. None (undefined) when the reference accuracy is 0. May exceed 100.
pub fn retention_pct(reference_acc: f64, candidate_acc: f64) -> Option<f64> {
    (reference_acc > 0.0).then(|| 100.0 * candidate_acc / reference_acc)
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
    pub confidence: f64,
}

/// Percentile bootstrap CI of the paired loss in pp: resample case indices with replacement, recompute
/// 100 * mean(ref_i - cand_i). `pairs` are (reference pass, candidate pass) per case. Deterministic for a seed.
/// Fewer than 2 pairs gives None (no interval; the gate reports "sample too small").
pub fn paired_bootstrap_loss(
    pairs: &[(bool, bool)],
    resamples: u32,
    confidence: f64,
    seed: u64,
) -> Option<Interval> {
    let d: Vec<f64> = pairs
        .iter()
        .map(|&(r, c)| r as i32 as f64 - c as i32 as f64)
        .collect();
    paired_bootstrap_mean(&d, resamples, confidence, seed)
}

/// Percentile bootstrap CI of 100 * mean(d) over per-case paired differences d_i = ref_i - cand_i (each in
/// [-1, 1]; with repeated trials, the difference of the per-case means). Resamples cases, so trials of one case
/// stay together (cluster bootstrap). Fewer than 2 cases gives None.
pub fn paired_bootstrap_mean(
    d: &[f64],
    resamples: u32,
    confidence: f64,
    seed: u64,
) -> Option<Interval> {
    let n = d.len();
    if n < 2 {
        return None;
    }
    let mut rng = Rng::new(seed);
    let mut stats: Vec<f64> = (0..resamples.max(1))
        .map(|_| {
            let s: f64 = (0..n).map(|_| d[rng.below(n)]).sum();
            // round to suppress float noise so identical outcomes give exactly [0, 0]
            (100.0 * s / n as f64 * 1e9).round() / 1e9
        })
        .collect();
    stats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let alpha = (1.0 - confidence) / 2.0;
    Some(Interval {
        lo: quantile_sorted(&stats, alpha),
        hi: quantile_sorted(&stats, 1.0 - alpha),
        confidence,
    })
}

/// Linear-interpolated quantile of sorted data (type 7).
pub fn quantile_sorted(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let pos = q.clamp(0.0, 1.0) * (xs.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    xs[lo] + (xs[hi] - xs[lo]) * (pos - lo as f64)
}

/// McNemar's exact (binomial) two-sided p-value for b regressions and c improvements.
/// 1.0 when there are no discordant pairs.
pub fn mcnemar_exact_p(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = b.min(c);
    // P(X <= k), X ~ Bin(n, 1/2), summed in log space
    let ln_half_n = n as f64 * 0.5f64.ln();
    let mut ln_c = 0.0; // ln C(n, 0)
    let mut tail = 0.0;
    for i in 0..=k {
        if i > 0 {
            ln_c += ((n - i + 1) as f64).ln() - (i as f64).ln();
        }
        tail += (ln_c + ln_half_n).exp();
    }
    (2.0 * tail).min(1.0)
}

/// Wilson score interval for a rate (as a fraction), e.g. JSON validity. None when n = 0.
pub fn wilson(successes: usize, n: usize, confidence: f64) -> Option<Interval> {
    if n == 0 {
        return None;
    }
    let z = normal_quantile(0.5 + confidence / 2.0);
    let nf = n as f64;
    let p = successes as f64 / nf;
    let z2 = z * z;
    let denom = 1.0 + z2 / nf;
    let centre = (p + z2 / (2.0 * nf)) / denom;
    let half = z * (p * (1.0 - p) / nf + z2 / (4.0 * nf * nf)).sqrt() / denom;
    Some(Interval {
        lo: (centre - half).max(0.0),
        hi: (centre + half).min(1.0),
        confidence,
    })
}

/// Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9).
pub fn normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383_577_518_672_69e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    ];
    let pl = 0.02425;
    if p < pl {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - pl {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -normal_quantile(1.0 - p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy_loss_retention() {
        assert_eq!(accuracy(3, 4), Some(0.75));
        assert_eq!(accuracy(0, 0), None);
        assert!((loss_pp(0.80, 0.75) - 5.0).abs() < 1e-9);
        assert!(
            (loss_pp(0.70, 0.75) + 5.0).abs() < 1e-9,
            "improvement is a negative loss"
        );
        assert!((retention_pct(0.80, 0.60).unwrap() - 75.0).abs() < 1e-9);
        assert!(
            retention_pct(0.50, 0.60).unwrap() > 100.0,
            "retention may exceed 100%"
        );
        assert_eq!(
            retention_pct(0.0, 0.3),
            None,
            "zero baseline: undefined, not infinite"
        );
    }

    #[test]
    fn mcnemar_known_values() {
        assert_eq!(mcnemar_exact_p(0, 0), 1.0);
        assert!((mcnemar_exact_p(5, 5) - 1.0).abs() < 1e-12);
        // b=10, c=0: 2 * 0.5^10
        assert!((mcnemar_exact_p(10, 0) - 2.0 / 1024.0).abs() < 1e-12);
        // b=8, c=2: 2 * (1 + 10 + 45) / 1024 = 0.109375
        assert!((mcnemar_exact_p(8, 2) - 0.109375).abs() < 1e-12);
        assert_eq!(mcnemar_exact_p(2, 8), mcnemar_exact_p(8, 2));
        // large n stays finite
        let p = mcnemar_exact_p(600, 500);
        assert!(p > 0.0 && p < 0.01, "{p}");
    }

    #[test]
    fn bootstrap_behaviour() {
        // identical outcomes: degenerate interval at 0
        let same: Vec<(bool, bool)> = (0..50).map(|i| (i % 3 == 0, i % 3 == 0)).collect();
        let ci = paired_bootstrap_loss(&same, 1000, 0.95, 1).unwrap();
        assert_eq!((ci.lo, ci.hi), (0.0, 0.0));
        // deterministic for a seed
        let mixed: Vec<(bool, bool)> = (0..60).map(|i| (i % 2 == 0, i % 5 != 0)).collect();
        let a = paired_bootstrap_loss(&mixed, 2000, 0.95, 7).unwrap();
        let b = paired_bootstrap_loss(&mixed, 2000, 0.95, 7).unwrap();
        assert_eq!(a, b);
        // contains the point estimate, and narrows as n grows
        let point =
            100.0 * mixed.iter().map(|&(r, c)| r as i32 - c as i32).sum::<i32>() as f64 / 60.0;
        assert!(a.lo <= point && point <= a.hi, "{a:?} vs {point}");
        let big: Vec<(bool, bool)> = (0..600).map(|i| (i % 2 == 0, i % 5 != 0)).collect();
        let w = paired_bootstrap_loss(&big, 2000, 0.95, 7).unwrap();
        assert!(w.hi - w.lo < (a.hi - a.lo) * 0.6, "{w:?} vs {a:?}");
        // a clear regression has an interval above 0
        let reg: Vec<(bool, bool)> = (0..100).map(|i| (true, i % 4 != 0)).collect();
        let r = paired_bootstrap_loss(&reg, 2000, 0.95, 3).unwrap();
        assert!(r.lo > 10.0 && r.hi < 40.0, "{r:?}");
        assert!(paired_bootstrap_loss(&[(true, false)], 100, 0.95, 1).is_none());
    }

    #[test]
    fn wilson_and_normal() {
        assert!((normal_quantile(0.975) - 1.959964).abs() < 1e-5);
        assert!((normal_quantile(0.5)).abs() < 1e-9);
        assert!((normal_quantile(0.001) + 3.090232).abs() < 1e-5);
        let w = wilson(20, 20, 0.95).unwrap();
        assert!(
            (w.hi - 1.0).abs() < 1e-12 && (w.lo - 0.8389).abs() < 1e-3,
            "{w:?}"
        );
        let w = wilson(50, 100, 0.95).unwrap();
        assert!(
            (w.lo - 0.4038).abs() < 1e-3 && (w.hi - 0.5962).abs() < 1e-3,
            "{w:?}"
        );
        assert!(wilson(0, 0, 0.95).is_none());
    }
}
