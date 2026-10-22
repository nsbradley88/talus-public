//! CPU statistical check of the 27B sampled speculative round (`spec_run_with_callback`): the target's
//! `make_dist` (penalties -> top-k (64 when unset) -> temperature -> min-p / top-p), the row decision
//! `spec_decide_row` (the loop's own code), `draw_dist` for the bonus row. The MTP drafts are drawn autoregressively
//! from q = a CPU model of the device draft sampler (`k_draft_support`: support x/tau >= thr from min-p, top-k, top-p;
//! q = exp(x/tau - lse)), and the n-gram rows use the point-mass rule (`qs = false`). Chi-square on the joint law of
//! the first three generated tokens.

use super::graphs::{draw_dist, make_dist, spec_decide_row, RowDecision, Sampling, SplitMix64};
use talus_spec::ssamp::{chi2_critical, chi2_tv};

const V: usize = 12;
const M: usize = 3;

fn hash(a: u64) -> u64 {
    let mut z = a.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(h: u64) -> f32 {
    ((h >> 40) as f32) / ((1u64 << 24) as f32)
}

fn target(h: &[i32]) -> Vec<f32> {
    let a = *h.last().unwrap_or(&0) as u64;
    let b = if h.len() >= 2 {
        h[h.len() - 2] as u64
    } else {
        7
    };
    (0..V)
        .map(|v| {
            5.0 * unit(hash(
                a * 1_000_003 + b * 1009 + v as u64 + 17 * h.len() as u64,
            )) - 2.0
        })
        .collect()
}

fn mtp_logits(h: &[i32]) -> Vec<f32> {
    let t = target(h);
    let a = *h.last().unwrap_or(&0) as u64;
    (0..V)
        .map(|v| 0.8 * t[v] + 2.0 * unit(hash(a * 29 + v as u64 * 3 + 11)) - 1.0)
        .collect()
}

/// CPU model of the device draft distribution: `(ids, q)` over the support `x / tau >= thr`.
fn device_q(x: &[f32], s: &Sampling) -> Vec<(i32, f64)> {
    let tau = s.temp;
    let v: Vec<f32> = x.iter().map(|&l| l / tau).collect();
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let tm = if s.min_p > 0.0 && s.min_p < 1.0 {
        m + s.min_p.ln()
    } else {
        f32::NEG_INFINITY
    };
    let mut sorted = v.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let k = if s.top_k > 0 { s.top_k as usize } else { 64 }.min(sorted.len());
    let base = sorted[k - 1];
    let mut th = tm.max(base);
    if s.top_p > 0.0 && s.top_p < 1.0 {
        let z: f64 = sorted
            .iter()
            .filter(|&&u| u >= base)
            .map(|&u| ((u - m) as f64).exp())
            .sum();
        let mut c = 0.0;
        for &u in &sorted {
            c += ((u - m) as f64).exp();
            if c >= s.top_p as f64 * z {
                th = th.max(u);
                break;
            }
        }
    }
    th = th.min(m);
    let lse = m as f64
        + v.iter()
            .filter(|&&u| u >= th)
            .map(|&u| ((u - m) as f64).exp())
            .sum::<f64>()
            .ln();
    (0..x.len())
        .filter(|&i| v[i] >= th)
        .map(|i| (i as i32, (v[i] as f64 - lse).exp()))
        .collect()
}

fn draw_q(q: &[(i32, f64)], rng: &mut SplitMix64) -> i32 {
    let tot: f64 = q.iter().map(|x| x.1).sum();
    let r = rng.uniform_01() * tot;
    let mut c = 0.0;
    for &(id, p) in q {
        c += p;
        if r < c {
            return id;
        }
    }
    q.last().unwrap().0
}

fn generate(prompt: &[i32], s: &Sampling, k: usize, mtp: bool, rng: &mut SplitMix64) -> Vec<i32> {
    generate_q(prompt, s, k, mtp, 1.0, rng)
}

/// `qscale`: the drafter's proposal temperature multiplier (TALUS_Q27_QTEMP); the target's `s` is unchanged.
fn generate_q(
    prompt: &[i32],
    s: &Sampling,
    k: usize,
    mtp: bool,
    qscale: f32,
    rng: &mut SplitMix64,
) -> Vec<i32> {
    let sq = Sampling {
        temp: s.temp * qscale,
        ..s.clone()
    };
    let mut h = prompt.to_vec();
    let mut out = vec![];
    let first = draw_dist(&make_dist(&target(&h), s, &h), rng);
    out.push(first);
    h.push(first);
    while out.len() < M {
        let mut drafts = Vec::with_capacity(k);
        let mut qs: Vec<Vec<(i32, f64)>> = Vec::with_capacity(k);
        let mut hd = h.clone();
        for _ in 0..k {
            let lg = mtp_logits(&hd);
            let d = if mtp {
                let q = device_q(&lg, &sq);
                let d = draw_q(&q, rng);
                qs.push(q);
                d
            } else {
                // an n-gram-like deterministic draft
                (0..V).max_by(|&a, &b| lg[a].total_cmp(&lg[b])).unwrap() as i32
            };
            drafts.push(d);
            hd.push(d);
        }
        for i in 0..=k {
            let mut p = make_dist(&target(&h), s, &h);
            if i == k {
                let x = draw_dist(&p, rng);
                out.push(x);
                h.push(x);
                break;
            }
            let d = drafts[i];
            let qof = |t: i32| -> f64 {
                if !mtp {
                    return 0.0;
                }
                qs[i].iter().find(|e| e.0 == t).map_or(0.0, |e| e.1)
            };
            let hh = h.clone();
            match spec_decide_row(&mut p, d, mtp, qof, rng, || make_dist(&target(&hh), s, &hh)) {
                RowDecision::Accept => {
                    out.push(d);
                    h.push(d);
                }
                RowDecision::End(x) => {
                    out.push(x);
                    h.push(x);
                    break;
                }
            }
        }
    }
    out.truncate(M);
    out
}

fn exact_law(prompt: &[i32], s: &Sampling) -> Vec<f64> {
    fn rec(h: &mut Vec<i32>, depth: usize, pr: f64, cell: usize, s: &Sampling, law: &mut [f64]) {
        if depth == M {
            law[cell] += pr;
            return;
        }
        for (t, p) in make_dist(&target(h), s, h) {
            h.push(t);
            rec(h, depth + 1, pr * p as f64, cell * V + t as usize, s, law);
            h.pop();
        }
    }
    let mut law = vec![0.0; V.pow(M as u32)];
    let mut h = prompt.to_vec();
    rec(&mut h, 0, 1.0, 0, s, &mut law);
    law
}

#[test]
fn q27_spec_rows_are_exact() {
    let prompt = [3, 1, 4, 1, 5];
    let base = Sampling::default();
    let cfgs = [
        Sampling {
            temp: 0.8,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            ..base.clone()
        },
        Sampling {
            temp: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            ..base.clone()
        },
        Sampling {
            temp: 0.7,
            top_p: 0.9,
            top_k: 6,
            min_p: 0.02,
            rep_pen: 1.2,
            freq_pen: 0.1,
            pres_pen: 0.1,
            rep_last_n: 4,
            ..base.clone()
        },
    ];
    for s in cfgs {
        let law = exact_law(&prompt, &s);
        for (label, mtp, k) in [("mtp q", true, 3usize), ("point mass", false, 3)] {
            let mut rng = SplitMix64::new(99);
            let mut counts = vec![0u64; law.len()];
            for _ in 0..300_000 {
                let o = generate(&prompt, &s, k, mtp, &mut rng);
                counts[o.iter().fold(0usize, |c, &t| c * V + t as usize)] += 1;
            }
            let (chi2, df, tv) = chi2_tv(&counts, &law);
            let crit = chi2_critical(df.max(1), 1e-6);
            eprintln!(
                "27b [t{} k{} p{} minp{} pen{}] {label}: chi2 {chi2:.1} df {df} crit(1e-6) {crit:.1} tv {tv:.4}",
                s.temp, s.top_k, s.top_p, s.min_p, s.rep_pen
            );
            assert!(chi2 < crit, "{label}: chi2 {chi2:.1} >= {crit:.1}");
            assert!(tv < 0.02);
        }
    }
}

#[test]
fn q27_sharper_q_stays_exact() {
    // TALUS_Q27_QTEMP < 1: the drafts come from a sharper q; the target law must not move
    let prompt = [3, 1, 4, 1, 5];
    let s = Sampling {
        temp: 1.0,
        top_p: 0.95,
        top_k: 40,
        min_p: 0.0,
        ..Sampling::default()
    };
    let law = exact_law(&prompt, &s);
    for qscale in [0.5f32, 0.75] {
        let mut rng = SplitMix64::new(7);
        let mut counts = vec![0u64; law.len()];
        for _ in 0..300_000 {
            let o = generate_q(&prompt, &s, 3, true, qscale, &mut rng);
            counts[o.iter().fold(0usize, |c, &t| c * V + t as usize)] += 1;
        }
        let (chi2, df, tv) = chi2_tv(&counts, &law);
        let crit = chi2_critical(df.max(1), 1e-6);
        eprintln!("27b qtemp {qscale}: chi2 {chi2:.1} df {df} crit(1e-6) {crit:.1} tv {tv:.4}");
        assert!(chi2 < crit && tv < 0.02);
    }
}
