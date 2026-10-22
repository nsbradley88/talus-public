//! Exact speculative sampling for sampled (temperature > 0) requests.
//!
//! Standard speculative rejection sampling (Leviathan et al. 2023, Chen et al. 2023): a draft token `x` drawn from
//! the drafter's distribution `q` is accepted with probability `min(1, p(x) / q(x))`, where `p` is the target's
//! sampling distribution at that row (after the server's own transforms: penalties, top-k, temperature, min-p,
//! top-p, renormalised). On a rejection the replacement is drawn from `norm(max(0, p - q))`. Every emitted token is
//! then distributed exactly as `p`, whatever `q` is, provided that
//!   1. `x` really was drawn from `q` (independently of the verifier's uniforms), and
//!   2. `q` is the same distribution at draft time and at verify time.
//!
//! `q` does not have to match `p`'s transforms for exactness; using the request's transforms on the drafter's logits
//! only makes `q` closer to `p` (acceptance = sum min(p, q)).
//!
//! No `q` (n-gram / prompt-lookup drafts) means a POINT MASS at `x`, never `q = 0`: accept with `p(x)`, residual
//! `p` without `x` - distributionally the same as the old "sample the row and keep the draft iff it equals the
//! sample" rule, which is what the serve loops keep doing for those rows.
//!
//! Kill switch: [`ssamp_enabled`] reads a per-model environment variable (default ON since the operator's 2026-10-24
//! decision; `=0` restores the exact-match rule): the per-seed sampled bytes depend on it, the output distribution does
//! not. Greedy requests never reach this module.

/// Uniform `[0, 1)` source (implemented for both Mersenne Twister ports in the tree).
pub trait Uniform01 {
    /// A uniform draw in `[0, 1)`.
    fn u01(&mut self) -> f64;
}

impl Uniform01 for crate::sampling::Mt19937_64 {
    #[inline]
    fn u01(&mut self) -> f64 {
        self.uniform_01()
    }
}

const MAX_LT_ONE: f64 = f64::from_bits(0x3FEF_FFFF_FFFF_FFFF);

/// Clamp a raw `[0, 1]` draw into `[0, 1)`.
#[inline]
pub fn clamp_u01(u: f64) -> f64 {
    if u >= 1.0 {
        MAX_LT_ONE
    } else if u > 0.0 {
        u
    } else {
        0.0
    }
}

/// A sparse probability distribution over token ids: `(ids[i], p[i])`, ids unique, `p >= 0`, summing to ~1.
/// Entries are kept in the order the transform produced them (usually descending probability).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SparseDist {
    /// Token ids (target vocabulary).
    pub ids: Vec<i32>,
    /// Probabilities, same length as `ids`.
    pub p: Vec<f64>,
}

impl SparseDist {
    /// Empty distribution.
    pub fn new() -> Self {
        Self::default()
    }

    /// Point mass at `x`.
    pub fn one_hot(x: i32) -> Self {
        Self {
            ids: vec![x],
            p: vec![1.0],
        }
    }

    /// Build from `(id, weight)` pairs: drops non-positive / non-finite weights, merges duplicate ids (a draft-vocab
    /// map that is not injective), and normalises. Returns an empty distribution when no mass is left.
    pub fn from_weights(pairs: impl IntoIterator<Item = (i32, f64)>) -> Self {
        let mut d = Self::new();
        for (id, w) in pairs {
            if !(w > 0.0) || !w.is_finite() || id < 0 {
                continue;
            }
            if let Some(k) = d.ids.iter().position(|&t| t == id) {
                d.p[k] += w;
            } else {
                d.ids.push(id);
                d.p.push(w);
            }
        }
        d.normalize();
        d
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True when there is no entry.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Total mass.
    #[inline]
    pub fn total(&self) -> f64 {
        self.p.iter().sum()
    }

    /// Renormalise to sum 1 (no-op on an empty or zero-mass list).
    pub fn normalize(&mut self) {
        let t = self.total();
        if t > 0.0 && t.is_finite() {
            let inv = 1.0 / t;
            for v in &mut self.p {
                *v *= inv;
            }
        }
    }

    /// Probability of `id` (0 when absent).
    #[inline]
    pub fn prob(&self, id: i32) -> f64 {
        self.ids
            .iter()
            .position(|&t| t == id)
            .map_or(0.0, |k| self.p[k])
    }

    /// Draw one token with a single uniform from `rng` (inverse CDF in list order).
    pub fn draw<R: Uniform01 + ?Sized>(&self, rng: &mut R) -> i32 {
        if self.ids.is_empty() {
            return 0;
        }
        if self.ids.len() == 1 {
            // keep the stream position independent of the list length: always consume one uniform
            let _ = rng.u01();
            return self.ids[0];
        }
        let tot = self.total();
        let r = clamp_u01(rng.u01()) * tot;
        let mut acc = 0.0f64;
        let mut last = self.ids[0];
        for (&id, &pv) in self.ids.iter().zip(self.p.iter()) {
            if pv <= 0.0 {
                continue;
            }
            last = id;
            acc += pv;
            if r < acc {
                return id;
            }
        }
        last
    }
}

/// Outcome of judging one draft token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The draft is kept.
    Accept,
    /// The draft is rejected; the row emits this replacement (drawn from the residual) and the round ends.
    Reject(i32),
}

/// The residual `norm(max(0, p - q))` over `p`'s support (empty when `p <= q` everywhere).
pub fn residual(p: &SparseDist, q: &SparseDist) -> SparseDist {
    let mut r = SparseDist {
        ids: p.ids.clone(),
        p: p.p.clone(),
    };
    for (&qid, &qv) in q.ids.iter().zip(q.p.iter()) {
        if qv <= 0.0 {
            continue;
        }
        if let Some(k) = r.ids.iter().position(|&t| t == qid) {
            r.p[k] = (r.p[k] - qv).max(0.0);
        }
    }
    let t = r.total();
    if !(t > 0.0) {
        return SparseDist::new();
    }
    r.normalize();
    r
}

/// Judge draft `x` (drawn from `q`) against the target distribution `p`.
///
/// Consumes one uniform for the accept test and, on a rejection, one more for the residual draw. If `x` has no mass
/// under `q` (a caller that did not draw `x` from `q`), `q` is taken as the point mass at `x`, which is the exact rule
/// for a deterministic draft.
pub fn judge<R: Uniform01 + ?Sized>(
    p: &SparseDist,
    q: &SparseDist,
    x: i32,
    rng: &mut R,
) -> Verdict {
    let px = p.prob(x);
    let mut qx = q.prob(x);
    let one_hot;
    let q = if qx > 0.0 {
        q
    } else {
        one_hot = SparseDist::one_hot(x);
        qx = 1.0;
        &one_hot
    };
    let u = clamp_u01(rng.u01());
    if u * qx < px {
        return Verdict::Accept;
    }
    let r = residual(p, q);
    if r.is_empty() {
        // p <= q everywhere means p == q, where a rejection has probability 0; keep the row valid anyway
        return Verdict::Reject(p.draw(rng));
    }
    Verdict::Reject(r.draw(rng))
}

/// Draw a token from `q` for drafting (the same `q` must later be passed to [`judge`]).
#[inline]
pub fn draft_from<R: Uniform01 + ?Sized>(q: &SparseDist, rng: &mut R) -> i32 {
    q.draw(rng)
}

/// Read a boolean kill switch: `1`/`true`/`on` enable; anything else (or unset) keeps the default `false`.
pub fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        })
        .unwrap_or(false)
}

/// Whether exact speculative sampling is on for a model (`env_name`, e.g. `TALUS_MIMO_SSAMP`): default ON; `0`, `false`
/// or `off` restore the exact-match rule (kill switch).
pub fn ssamp_enabled(env_name: &str) -> bool {
    match std::env::var(env_name) {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    }
}

/// The drafter temperature multiplier `TALUS_SSAMP_QTEMP` (default 1.0): q is built at `scale * temperature`.
/// Any q keeps the output exact; a sharper q (scale < 1) can raise acceptance when the drafter is less confident
/// than the target.
pub fn q_temp_scale() -> f32 {
    std::env::var("TALUS_SSAMP_QTEMP")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| *v > 0.0 && v.is_finite())
        .unwrap_or(1.0)
}

/// Per-model drafter temperature multiplier: `model_env` (e.g. `TALUS_Q27_QTEMP`) if set to a positive number, else
/// [`q_temp_scale`] (`TALUS_SSAMP_QTEMP`, default 1.0). The proposal q is any distribution the drafts are really
/// drawn from, so the output stays exact for every value; it only moves acceptance (sum min(p, q)).
pub fn q_temp_scale_for(model_env: &str) -> f32 {
    std::env::var(model_env)
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| *v > 0.0 && v.is_finite())
        .unwrap_or_else(q_temp_scale)
}

/// [`q_temp_scale_for`] with a per-model `default` when neither `model_env` nor `TALUS_SSAMP_QTEMP` is set.
pub fn q_temp_scale_or(model_env: &str, default: f32) -> f32 {
    let parse = |n: &str| {
        std::env::var(n)
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|v| *v > 0.0 && v.is_finite())
    };
    parse(model_env)
        .or_else(|| parse("TALUS_SSAMP_QTEMP"))
        .unwrap_or(default)
}

/// Chi-square critical value for `df` degrees of freedom at upper-tail probability `alpha` (Wilson-Hilferty).
pub fn chi2_critical(df: usize, alpha: f64) -> f64 {
    // upper normal quantile by bisection on the complementary error function approximation
    let z = normal_upper_quantile(alpha);
    let k = df as f64;
    let a = 2.0 / (9.0 * k);
    k * (1.0 - a + z * a.sqrt()).powi(3)
}

fn normal_upper_quantile(alpha: f64) -> f64 {
    // Acklam-free bisection on 0.5 * erfc(z / sqrt 2) (erfc via Abramowitz-Stegun 7.1.26, |err| < 1.5e-7)
    fn erfc(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
        let y = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        let r = y * (-x * x).exp();
        if x >= 0.0 {
            r
        } else {
            2.0 - r
        }
    }
    let (mut lo, mut hi) = (-10.0f64, 10.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if 0.5 * erfc(mid / std::f64::consts::SQRT_2) > alpha {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Pearson chi-square statistic and total-variation distance of observed counts against expected probabilities.
/// Cells with expected count < 5 are pooled into one. Returns `(chi2, df, tv)`.
pub fn chi2_tv(counts: &[u64], expected_p: &[f64]) -> (f64, usize, f64) {
    let n: u64 = counts.iter().sum();
    let nf = n as f64;
    let mut chi2 = 0.0;
    let mut cells = 0usize;
    let (mut pool_o, mut pool_e) = (0.0f64, 0.0f64);
    let mut tv = 0.0;
    for (&c, &pe) in counts.iter().zip(expected_p.iter()) {
        let e = pe * nf;
        tv += ((c as f64) / nf - pe).abs();
        if e < 5.0 {
            pool_o += c as f64;
            pool_e += e;
            continue;
        }
        chi2 += (c as f64 - e).powi(2) / e;
        cells += 1;
    }
    if pool_e > 0.0 {
        chi2 += (pool_o - pool_e).powi(2) / pool_e.max(1e-300);
        cells += 1;
    }
    (chi2, cells.saturating_sub(1), 0.5 * tv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::Mt19937_64;

    fn dist(v: &[(i32, f64)]) -> SparseDist {
        SparseDist::from_weights(v.iter().copied())
    }

    /// One row: the emitted token's law must be p for any q (draft drawn from q).
    fn one_row_law(p: &SparseDist, q: &SparseDist, nv: usize, n: usize, seed: u64) -> Vec<u64> {
        let mut rng = Mt19937_64::new(seed);
        let mut counts = vec![0u64; nv];
        for _ in 0..n {
            let x = draft_from(q, &mut rng);
            let y = match judge(p, q, x, &mut rng) {
                Verdict::Accept => x,
                Verdict::Reject(y) => y,
            };
            counts[y as usize] += 1;
        }
        counts
    }

    fn dense(p: &SparseDist, nv: usize) -> Vec<f64> {
        let mut d = vec![0.0; nv];
        for (&id, &v) in p.ids.iter().zip(p.p.iter()) {
            d[id as usize] += v;
        }
        d
    }

    #[test]
    fn single_row_is_exact_for_many_q() {
        let nv = 8;
        let p = dist(&[
            (0, 0.30),
            (1, 0.25),
            (2, 0.20),
            (3, 0.10),
            (4, 0.10),
            (5, 0.05),
        ]);
        let qs = [
            dist(&[
                (0, 0.30),
                (1, 0.25),
                (2, 0.20),
                (3, 0.10),
                (4, 0.10),
                (5, 0.05),
            ]), // q == p
            dist(&[(0, 0.9), (1, 0.1)]),           // sharper, truncated
            dist(&[(6, 0.5), (7, 0.3), (0, 0.2)]), // mostly off-support
            dist(&[(5, 1.0)]),                     // point mass
            dist(&[
                (0, 0.125),
                (1, 0.125),
                (2, 0.125),
                (3, 0.125),
                (4, 0.125),
                (5, 0.125),
                (6, 0.125),
                (7, 0.125),
            ]),
        ];
        let crit = chi2_critical(7, 1e-6);
        for (k, q) in qs.iter().enumerate() {
            let counts = one_row_law(&p, q, nv, 400_000, 1000 + k as u64);
            let (chi2, df, tv) = chi2_tv(&counts, &dense(&p, nv));
            assert!(
                chi2 < chi2_critical(df.max(1), 1e-6),
                "q#{k}: chi2 {chi2:.1} df {df} (crit {crit:.1})"
            );
            assert!(tv < 0.004, "q#{k}: tv {tv}");
        }
    }

    #[test]
    fn acceptance_is_sum_min_p_q() {
        let p = dist(&[(0, 0.5), (1, 0.3), (2, 0.2)]);
        let q = dist(&[(0, 0.2), (1, 0.5), (2, 0.3)]);
        let want: f64 = 0.2 + 0.3 + 0.2;
        let mut rng = Mt19937_64::new(7);
        let n = 300_000;
        let mut acc = 0usize;
        for _ in 0..n {
            let x = draft_from(&q, &mut rng);
            if judge(&p, &q, x, &mut rng) == Verdict::Accept {
                acc += 1;
            }
        }
        let got = acc as f64 / n as f64;
        assert!(
            (got - want).abs() < 0.004,
            "acceptance {got} vs sum min {want}"
        );
    }

    #[test]
    fn point_mass_reduces_to_match_rule_acceptance() {
        // q = one-hot at x: acceptance p(x), and the residual never emits x
        let p = dist(&[(0, 0.6), (1, 0.4)]);
        let q = SparseDist::one_hot(1);
        let mut rng = Mt19937_64::new(9);
        let (mut acc, n) = (0usize, 200_000usize);
        for _ in 0..n {
            match judge(&p, &q, 1, &mut rng) {
                Verdict::Accept => acc += 1,
                Verdict::Reject(y) => assert_eq!(y, 0),
            }
        }
        assert!(((acc as f64 / n as f64) - 0.4).abs() < 0.005);
        // a draft outside q's support is judged as a point mass too (never as q = 0)
        let empty = SparseDist::new();
        let mut acc2 = 0usize;
        for _ in 0..n {
            if judge(&p, &empty, 0, &mut rng) == Verdict::Accept {
                acc2 += 1;
            }
        }
        assert!(((acc2 as f64 / n as f64) - 0.6).abs() < 0.005);
    }

    #[test]
    fn from_weights_merges_duplicates_and_normalizes() {
        let d = SparseDist::from_weights([(3, 1.0), (4, 1.0), (3, 2.0), (5, 0.0), (6, f64::NAN)]);
        assert_eq!(d.ids, vec![3, 4]);
        assert!((d.prob(3) - 0.75).abs() < 1e-12 && (d.prob(4) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn chi2_critical_sane() {
        // df 10 at alpha 0.05 is 18.31; df 35 at 1e-4 is 74.93 (regularized gamma)
        assert!((chi2_critical(10, 0.05) - 18.31).abs() < 0.2);
        let c = chi2_critical(35, 1e-4);
        assert!((c - 74.93).abs() < 0.5, "{c}");
    }
}
