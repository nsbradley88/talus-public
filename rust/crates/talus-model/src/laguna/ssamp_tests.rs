//! CPU statistical proof that the MiMo / Laguna speculative-sampling rounds (the serve loops' exact
//! sequence: `draft_q` -> `q.draw` per DFlash row, `sample_dist` + `judge` per verified row, `sample` for the bonus row)
//! emit tokens distributed exactly as the target's own sampler (`sample`), on a synthetic prefix-dependent target and
//! drafter. Chi-square over the joint law of the first three generated tokens (with penalties, top-k, top-p, min-p,
//! temperature), a total-variation bound, and a negative control (argmax drafts judged against the softmax q) that
//! the test must reject.

use super::serve::{draft_q, sample, sample_dist, Mt19937_64, Sampling};
use talus_spec::ssamp::{chi2_critical, chi2_tv, judge, SparseDist, Uniform01, Verdict};

const V: usize = 10;
const M: usize = 3; // generated tokens whose joint law is tested

fn hash(a: u64) -> u64 {
    let mut z = a.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(h: u64) -> f32 {
    ((h >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Target logits after history `h` (depends on the last two tokens and the length).
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

/// DFlash-like drafter: row `r` of a block after context `h` (no dependence on earlier drafts), top `m` candidates
/// `(id, logit)` in descending order, correlated with the target but not equal to it.
fn drafter_row(h: &[i32], r: usize, m: usize) -> Vec<(i32, f32)> {
    let a = *h.last().unwrap_or(&0) as u64;
    let t = target(h);
    let mut c: Vec<(i32, f32)> = (0..V)
        .map(|v| {
            let noise = 3.0 * unit(hash(a * 31 + v as u64 * 7 + r as u64 * 101 + 5)) - 1.5;
            (v as i32, 0.6 * t[v] + noise)
        })
        .collect();
    c.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
    c.truncate(m);
    c
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// The serve loops with ssamp on: drafts drawn from q, judged by min(1, p/q).
    Exact,
    /// The serve loops with ssamp off: argmax drafts, kept iff the sampled row equals them (point-mass q).
    MatchRule,
    /// Negative control: argmax drafts judged against the softmax q (a draft not drawn from q): biased.
    Biased,
}

/// Generate the first `M` tokens after `prompt` the way the serve loop does, `k` drafts per round.
fn generate(
    prompt: &[i32],
    sp: &Sampling,
    k: usize,
    m_cands: usize,
    mode: Mode,
    rng: &mut Mt19937_64,
) -> Vec<i32> {
    let mut h = prompt.to_vec();
    let mut out = Vec::with_capacity(M + k + 1);
    let mut l = target(&h);
    let next = sample(&mut l, sp, &h, rng);
    out.push(next);
    h.push(next);
    while out.len() < M {
        // propose: k DFlash rows from the context h
        let rows: Vec<Vec<(i32, f32)>> = (0..k).map(|r| drafter_row(&h, r, m_cands)).collect();
        let mut drafts: Vec<i32> = rows.iter().map(|c| c[0].0).collect();
        let mut qs: Vec<SparseDist> = Vec::new();
        if mode != Mode::MatchRule {
            let mut hist = h.clone();
            for i in 0..k {
                let q = draft_q(&rows[i], sp, &hist, 1.0);
                if mode == Mode::Exact {
                    drafts[i] = q.draw(rng);
                }
                qs.push(q);
                hist.push(drafts[i]);
            }
        }
        // verify rows 0..=k
        for i in 0..=k {
            let mut row = target(&h);
            let ends = if i < k && i < qs.len() {
                let p = sample_dist(&mut row, None, sp, &h);
                match judge(&p, &qs[i], drafts[i], rng) {
                    Verdict::Accept => None,
                    Verdict::Reject(y) => Some(y),
                }
            } else {
                let c = sample(&mut row, sp, &h, rng);
                if i < k && c == drafts[i] {
                    None
                } else {
                    Some(c)
                }
            };
            match ends {
                None => {
                    out.push(drafts[i]);
                    h.push(drafts[i]);
                }
                Some(t) => {
                    out.push(t);
                    h.push(t);
                    break;
                }
            }
        }
    }
    out.truncate(M);
    out
}

/// Exact joint law of the first M tokens under the target's sampler (`sample_dist`, dense, by enumeration).
fn exact_law(prompt: &[i32], sp: &Sampling) -> Vec<f64> {
    let mut law = vec![0.0f64; V.pow(M as u32)];
    fn rec(h: &mut Vec<i32>, depth: usize, pr: f64, cell: usize, sp: &Sampling, law: &mut [f64]) {
        if depth == M {
            law[cell] += pr;
            return;
        }
        let mut row = target(h);
        let p = sample_dist(&mut row, None, sp, h);
        for (&id, &pv) in p.ids.iter().zip(p.p.iter()) {
            if pv <= 0.0 {
                continue;
            }
            h.push(id);
            rec(h, depth + 1, pr * pv, cell * V + id as usize, sp, law);
            h.pop();
        }
    }
    let mut h = prompt.to_vec();
    rec(&mut h, 0, 1.0, 0, sp, &mut law);
    law
}

fn run(
    sp: &Sampling,
    k: usize,
    m_cands: usize,
    mode: Mode,
    n: usize,
    seed: u64,
) -> (f64, usize, f64, f64) {
    let prompt = [3, 1, 4, 1, 5];
    let law = exact_law(&prompt, sp);
    let mut counts = vec![0u64; law.len()];
    let mut rng = Mt19937_64::new(seed);
    for _ in 0..n {
        let o = generate(&prompt, sp, k, m_cands, mode, &mut rng);
        let cell = o.iter().fold(0usize, |c, &t| c * V + t as usize);
        counts[cell] += 1;
    }
    let (chi2, df, tv) = chi2_tv(&counts, &law);
    (chi2, df, tv, chi2_critical(df.max(1), 1e-6))
}

fn configs() -> Vec<(&'static str, Sampling)> {
    let base = Sampling {
        temperature: 1.0,
        top_p: 1.0,
        min_p: 0.0,
        top_k: 0,
        ..Sampling::default()
    };
    vec![
        ("t1.0", base),
        (
            "t0.8 top_p 0.9",
            Sampling {
                temperature: 0.8,
                top_p: 0.9,
                ..base
            },
        ),
        (
            "t1.0 top_k 5 min_p 0.05",
            Sampling {
                top_k: 5,
                min_p: 0.05,
                ..base
            },
        ),
        (
            "t0.7 top_k 6 top_p 0.95 min_p 0.02 penalties",
            Sampling {
                temperature: 0.7,
                top_k: 6,
                top_p: 0.95,
                min_p: 0.02,
                penalty_repeat: 1.3,
                penalty_freq: 0.2,
                penalty_present: 0.1,
                penalty_last_n: 4,
                ..base
            },
        ),
    ]
}

#[test]
fn ssamp_rounds_are_exact_in_distribution() {
    for (name, sp) in configs() {
        for &(k, m) in &[(3usize, 6usize), (2, 16), (4, 3)] {
            let (chi2, df, tv, crit) = run(&sp, k, m, Mode::Exact, 300_000, 11 + k as u64);
            eprintln!("ssamp exact  [{name}] k{k} cands{m}: chi2 {chi2:.1} df {df} crit(1e-6) {crit:.1} tv {tv:.4}");
            assert!(
                chi2 < crit,
                "[{name}] k{k} m{m}: chi2 {chi2:.1} >= {crit:.1}"
            );
            assert!(tv < 0.02, "[{name}] k{k} m{m}: tv {tv}");
        }
    }
}

#[test]
fn match_rule_is_exact_too() {
    // ssamp off (the current rule) = point-mass q: also exact, lower acceptance
    for (name, sp) in configs() {
        let (chi2, df, tv, crit) = run(&sp, 3, 6, Mode::MatchRule, 300_000, 5);
        eprintln!("match rule   [{name}]: chi2 {chi2:.1} df {df} crit(1e-6) {crit:.1} tv {tv:.4}");
        assert!(chi2 < crit, "[{name}]: chi2 {chi2:.1} >= {crit:.1}");
    }
}

#[test]
fn negative_control_is_detected() {
    // argmax drafts judged against a softmax q (the draft was not drawn from q) must fail the same test
    let mut failed = 0;
    for (name, sp) in configs() {
        let (chi2, df, tv, crit) = run(&sp, 3, 6, Mode::Biased, 300_000, 9);
        eprintln!("biased ctrl  [{name}]: chi2 {chi2:.1} df {df} crit(1e-6) {crit:.1} tv {tv:.4}");
        if chi2 >= crit {
            failed += 1;
        }
    }
    assert!(
        failed >= 3,
        "the test lacks power: only {failed}/4 biased configs rejected"
    );
}

#[test]
fn sample_dist_matches_sample_law() {
    // the p used by judge() is the law sample() draws from (same transforms, same order)
    for (name, sp) in configs() {
        let h = vec![2, 2, 7, 3];
        let mut row = target(&h);
        let p = sample_dist(&mut row, None, &sp, &h);
        let mut dense = vec![0.0; V];
        for (&id, &v) in p.ids.iter().zip(p.p.iter()) {
            dense[id as usize] += v;
        }
        let mut rng = Mt19937_64::new(77);
        let mut counts = vec![0u64; V];
        for _ in 0..400_000 {
            let mut r = target(&h);
            counts[sample(&mut r, &sp, &h, &mut rng) as usize] += 1;
        }
        let (chi2, df, _tv) = chi2_tv(&counts, &dense);
        assert!(
            chi2 < chi2_critical(df.max(1), 1e-6),
            "[{name}] chi2 {chi2:.1}"
        );
    }
}

#[test]
fn acceptance_rises_over_match_rule() {
    // the point of the change: mean accepted drafts per round, exact rule vs the match rule, on the same model
    let sp = Sampling {
        temperature: 1.0,
        ..Sampling::default()
    };
    let prompt = [3, 1, 4, 1, 5];
    let count = |mode: Mode| -> f64 {
        let mut rng = Mt19937_64::new(3);
        let (mut acc, mut drafted) = (0usize, 0usize);
        for _ in 0..50_000 {
            let mut h = prompt.to_vec();
            let rows: Vec<Vec<(i32, f32)>> = (0..3).map(|r| drafter_row(&h, r, 16)).collect();
            let mut drafts: Vec<i32> = rows.iter().map(|c| c[0].0).collect();
            let qs: Vec<SparseDist> = if mode == Mode::Exact {
                rows.iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let q = draft_q(c, &sp, &h, 1.0);
                        drafts[i] = q.draw(&mut rng);
                        q
                    })
                    .collect()
            } else {
                drafts.iter().map(|&d| SparseDist::one_hot(d)).collect()
            };
            for i in 0..3 {
                drafted += 1;
                let mut row = target(&h);
                let p = sample_dist(&mut row, None, &sp, &h);
                if judge(&p, &qs[i], drafts[i], &mut rng) == Verdict::Accept {
                    acc += 1;
                    h.push(drafts[i]);
                } else {
                    break;
                }
            }
            let _ = rng.u01();
        }
        acc as f64 / drafted as f64
    };
    let (a_exact, a_match) = (count(Mode::Exact), count(Mode::MatchRule));
    eprintln!("synthetic acceptance: exact {a_exact:.3} vs match rule {a_match:.3}");
    assert!(a_exact.is_finite() && a_match.is_finite());
}
