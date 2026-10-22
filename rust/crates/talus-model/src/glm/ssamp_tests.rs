//! CPU statistical check of GLM's MTP speculative sampling (`sample::dist` for p and q with the request's
//! chain top-k -> top-p -> min-p -> temperature, q through a draft vocabulary map, `sample::accept`), in the order of
//! `spec::generate`: d1 ~ q1(h), d2 ~ q2(h + d1) (the MTP chain conditions on its own draft), rows judged in turn,
//! the bonus row sampled from p. Chi-square on the joint law of the first three generated tokens.

use super::{accept, dist, sample, Rng, SamplerParams};
use talus_spec::ssamp::{chi2_critical, chi2_tv};

const V: usize = 10;
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

fn target(h: &[u32]) -> Vec<f32> {
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

/// Draft vocabulary: 7 of the 10 tokens, permuted (index -> token id). Tokens 2, 5, 9 are never drafted.
const DV: [u32; 7] = [8, 0, 6, 3, 1, 7, 4];

/// The MTP head's logits over the draft vocabulary after `h` (conditioned on the whole prefix, drafts included).
fn draft_logits(h: &[u32]) -> Vec<f32> {
    let t = target(h);
    let a = *h.last().unwrap_or(&0) as u64;
    DV.iter()
        .enumerate()
        .map(|(i, &tok)| 0.7 * t[tok as usize] + 2.0 * unit(hash(a * 13 + i as u64 * 5 + 3)) - 1.0)
        .collect()
}

fn generate(prompt: &[u32], sp: &SamplerParams, q_sp: &SamplerParams, rng: &mut Rng) -> Vec<u32> {
    let mut h = prompt.to_vec();
    let mut out = vec![];
    let d0 = dist(&target(&h), sp, None);
    let first = sample(&d0, rng);
    out.push(first);
    h.push(first);
    while out.len() < M {
        let q1 = dist(&draft_logits(&h), q_sp, Some(&DV));
        let d1 = sample(&q1, rng);
        let mut h1 = h.clone();
        h1.push(d1);
        let q2 = dist(&draft_logits(&h1), q_sp, Some(&DV));
        let d2 = sample(&q2, rng);
        let mut a = 0;
        let mut next = None;
        for (dr, qr) in [(d1, &q1), (d2, &q2)] {
            let pr = dist(&target(&h), sp, None);
            match accept(&pr, qr, dr, false, rng) {
                Ok(x) => {
                    a += 1;
                    out.push(x);
                    h.push(x);
                }
                Err(y) => {
                    next = Some(y);
                    break;
                }
            }
        }
        let y = match next {
            Some(y) => y,
            None => {
                debug_assert_eq!(a, 2);
                sample(&dist(&target(&h), sp, None), rng)
            }
        };
        out.push(y);
        h.push(y);
    }
    out.truncate(M);
    out
}

fn exact_law(prompt: &[u32], sp: &SamplerParams) -> Vec<f64> {
    fn rec(
        h: &mut Vec<u32>,
        depth: usize,
        pr: f64,
        cell: usize,
        sp: &SamplerParams,
        law: &mut [f64],
    ) {
        if depth == M {
            law[cell] += pr;
            return;
        }
        for (t, p) in dist(&target(h), sp, None) {
            h.push(t);
            rec(h, depth + 1, pr * p as f64, cell * V + t as usize, sp, law);
            h.pop();
        }
    }
    let mut law = vec![0.0; V.pow(M as u32)];
    let mut h = prompt.to_vec();
    rec(&mut h, 0, 1.0, 0, sp, &mut law);
    law
}

#[test]
fn glm_mtp_spec_sampling_is_exact() {
    let prompt = [3u32, 1, 4, 1, 5];
    let cfgs = [
        SamplerParams {
            temp: 1.0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            seed: 0,
        },
        SamplerParams {
            temp: 0.6,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
        },
        SamplerParams {
            temp: 1.3,
            top_k: 5,
            top_p: 0.9,
            min_p: 0.1,
            seed: 0,
        },
    ];
    for sp in cfgs {
        let law = exact_law(&prompt, &sp);
        for (label, q_sp) in [
            ("q sampled", sp),
            (
                "q argmax (GLM_SSAMP_OFF)",
                SamplerParams { temp: 0.0, ..sp },
            ),
        ] {
            let mut rng = Rng::new(42);
            let mut counts = vec![0u64; law.len()];
            for _ in 0..300_000 {
                let o = generate(&prompt, &sp, &q_sp, &mut rng);
                counts[o.iter().fold(0usize, |c, &t| c * V + t as usize)] += 1;
            }
            let (chi2, df, tv) = chi2_tv(&counts, &law);
            let crit = chi2_critical(df.max(1), 1e-6);
            eprintln!(
                "glm [t{} k{} p{} minp{}] {label}: chi2 {chi2:.1} df {df} crit(1e-6) {crit:.1} tv {tv:.4}",
                sp.temp, sp.top_k, sp.top_p, sp.min_p
            );
            assert!(chi2 < crit, "{label}: chi2 {chi2:.1} >= {crit:.1}");
            assert!(tv < 0.02);
        }
    }
}
