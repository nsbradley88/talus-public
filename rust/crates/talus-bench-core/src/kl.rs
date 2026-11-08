//! Teacher-forced KL between two full-vocabulary logit dumps: the CPU half of the GPU-host KL mode.
//!
//! A port of the release tooling's `kl.py`: two float32 little-endian `[rows, V]` dumps
//! written by `talus-cli --dump-full-logits` (via the release tooling's `tfkld.sh`); per row, log-softmax in f64 and
//! KL(base || test) = sum p_base (log p_base - log p_test); reported as mean / p50 / p99 / max (numpy's linear
//! percentile) and top-1 agreement (first argmax, as numpy). `rows` is given, or tried as 129 then 128 as
//! rel-kld.sh does; `skip` drops leading rows (e.g. the prefill row). Byte-identical dumps are reported as
//! bitwise-identical.
//!
//! The release gates that use it (rel-kld.sh, METHODOLOGY.md) are in `RELEASE_TFKLD`: PASS when the dumps are
//! bitwise-identical, or mean <= max_mean and top-1 >= min_top1.

use crate::config::Threshold;
use crate::gates::{GateResult, GateStatus};
use serde::Serialize;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KlStats {
    pub rows: usize,
    pub vocab: usize,
    pub mean: f64,
    pub p50: f64,
    pub p99: f64,
    pub max: f64,
    pub top1: f64,
    pub bitwise_identical: bool,
}

pub fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() % 4 != 0 {
        return Err(format!(
            "{}: size {} is not a multiple of 4",
            path.display(),
            b.len()
        ));
    }
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v as f64));
    let s: f64 = x.iter().map(|&v| (v as f64 - m).exp()).sum();
    let ls = s.ln();
    x.iter().map(|&v| v as f64 - m - ls).collect()
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in x.iter().enumerate() {
        if *v > x[best] {
            best = i;
        }
    }
    best
}

/// KL statistics of two dumps already in memory.
pub fn kl_stats(a: &[f32], b: &[f32], rows: usize, skip: usize) -> Result<KlStats, String> {
    if a.len() != b.len() || rows == 0 || !a.len().is_multiple_of(rows) {
        return Err(format!(
            "sizes {} / {} do not split into {rows} rows",
            a.len(),
            b.len()
        ));
    }
    if skip >= rows {
        return Err(format!("skip {skip} >= rows {rows}"));
    }
    let v = a.len() / rows;
    let mut kl = Vec::with_capacity(rows - skip);
    let mut agree = 0usize;
    for r in skip..rows {
        let ra = &a[r * v..(r + 1) * v];
        let rb = &b[r * v..(r + 1) * v];
        let la = log_softmax(ra);
        let lb = log_softmax(rb);
        kl.push(
            la.iter()
                .zip(&lb)
                .map(|(x, y)| x.exp() * (x - y))
                .sum::<f64>(),
        );
        agree += (argmax(ra) == argmax(rb)) as usize;
    }
    let n = kl.len();
    let mean = kl.iter().sum::<f64>() / n as f64;
    let mut sorted = kl.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    Ok(KlStats {
        rows: n,
        vocab: v,
        mean,
        p50: crate::stats::quantile_sorted(&sorted, 0.5),
        p99: crate::stats::quantile_sorted(&sorted, 0.99),
        max: *sorted.last().unwrap(),
        top1: agree as f64 / n as f64,
        bitwise_identical: a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
    })
}

/// KL of two dump files; `rows` None tries 129 then 128 (rel-kld.sh).
pub fn kl_files(
    base: &Path,
    test: &Path,
    rows: Option<usize>,
    skip: usize,
) -> Result<KlStats, String> {
    let a = read_f32(base)?;
    let b = read_f32(test)?;
    match rows {
        Some(r) => kl_stats(&a, &b, r, skip),
        None => kl_stats(&a, &b, 129, skip).or_else(|_| kl_stats(&a, &b, 128, skip)),
    }
}

/// Python's `{:.3e}` (two-digit exponent with sign).
pub fn py_e3(x: f64) -> String {
    let s = format!("{x:.3e}");
    match s.split_once('e') {
        Some((m, e)) => {
            let (sign, digits) = match e.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', e),
            };
            format!("{m}e{sign}{digits:0>2}")
        }
        None => s,
    }
}

/// The line kl.py prints (results_page.py parses it), plus the bitwise note rel-kld.sh appends.
pub fn kl_line(s: &KlStats) -> String {
    let mut l = format!(
        "rows {} V {} KL mean {} p50 {} p99 {} max {} top1 {:.4}",
        s.rows,
        s.vocab,
        py_e3(s.mean),
        py_e3(s.p50),
        py_e3(s.p99),
        py_e3(s.max),
        s.top1
    );
    if s.bitwise_identical {
        l.push_str(" bitwise-identical");
    }
    l
}

/// Teacher-forced KL release thresholds per model (rel-kld.sh; METHODOLOGY.md "KLD references per model"):
/// (model, max mean KL, min top-1). Both at depths 2K and 32K against the previous gated build.
pub const RELEASE_TFKLD: [(&str, f64, f64); 2] = [("mimo", 1e-3, 0.99), ("laguna", 3e-3, 0.98)];

pub fn kl_gate(
    label: &str,
    s: &KlStats,
    max_mean: f64,
    min_top1: f64,
    required: bool,
) -> GateResult {
    let ok = s.bitwise_identical || (s.mean <= max_mean && s.top1 >= min_top1);
    GateResult {
        gate: format!("{label} teacher-forced KL"),
        task: "kld_teacher_forced".into(),
        kind: "kl".into(),
        required,
        status: if ok {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        value: Some(s.mean),
        threshold: Some(Threshold {
            target: max_mean,
            investigate: max_mean,
        }),
        interval: None,
        n: s.rows,
        reasons: vec![format!(
            "{} (needs bitwise, or mean <= {max_mean:e} and top-1 >= {min_top1})",
            kl_line(s)
        )],
    }
}

// ---- Screening KL from top-k lists (HTTP `kld_topk`, server extension E2) ----
//
// A scoring endpoint reports, per position, the forced token's logprob and the k most likely tokens; the rest of
// the vocabulary is unseen. The KL is computed over a common partition of the vocabulary: every token id in
// either top-k list or forced, plus one "rest" bucket. A side's probability of an id outside its own top-k list
// (and not its forced token) is unknown: such ids share that side's tail mass (1 - known mass) equally with the
// rest bucket, each capped at the side's smallest top-k probability (it was not in the top k, so it cannot be
// larger). Buckets are floored at 1e-12 and both vectors renormalized. Identical inputs give exactly 0; the value
// is a coarse-grained approximation of the full-vocabulary KL, so its thresholds are derived separately
// (DESIGN.md 12.2) and it never replaces the GPU-host full-vocabulary gate.

/// One scored position: the forced (continuation) token, its logprob, and the top-k list `(id, logprob)`.
/// A logprob of -9999 (the server's encoding of minus infinity) is probability 0.
#[derive(Clone, Debug, PartialEq)]
pub struct TopkRow {
    pub id: u32,
    pub lp: f64,
    pub top: Vec<(u32, f64)>,
}

/// The screening comparison of one position, KL(reference || candidate).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TopkKl {
    pub kl: f64,
    /// The most likely token (highest known probability, ties to the smaller id) is the same on both sides.
    pub top1_agree: bool,
    /// Probability mass of the reference's top-k list (1 - this is the reference tail the top-k list misses).
    pub ref_covered: f64,
    pub cand_covered: f64,
    /// |candidate logprob - reference logprob| of the forced token.
    pub abs_dlp: f64,
}

const TOPK_FLOOR: f64 = 1e-12;

fn prob(lp: f64) -> f64 {
    if lp <= -9999.0 || !lp.is_finite() {
        0.0
    } else {
        lp.exp()
    }
}

/// The known probabilities of one side: its top-k list plus its forced token.
fn known(r: &TopkRow) -> std::collections::BTreeMap<u32, f64> {
    let mut m: std::collections::BTreeMap<u32, f64> =
        r.top.iter().map(|&(id, lp)| (id, prob(lp))).collect();
    m.entry(r.id).or_insert(prob(r.lp));
    m
}

fn argmax_known(m: &std::collections::BTreeMap<u32, f64>) -> Option<u32> {
    // BTreeMap iterates ids ascending; a strict > keeps the smaller id on ties
    let mut best: Option<(u32, f64)> = None;
    for (&id, &p) in m {
        if best.is_none_or(|(_, bp)| p > bp) {
            best = Some((id, p));
        }
    }
    best.map(|b| b.0)
}

/// One side's probability vector over `union` + the rest bucket.
fn bucketed(r: &TopkRow, union: &[u32]) -> Vec<f64> {
    let k = known(r);
    let known_mass: f64 = k.values().sum();
    let tail = (1.0 - known_mass).max(0.0);
    let cap = r
        .top
        .iter()
        .map(|&(_, lp)| prob(lp))
        .fold(f64::INFINITY, f64::min);
    let unknown = union.iter().filter(|id| !k.contains_key(id)).count();
    let share = (tail / (unknown + 1) as f64).min(cap);
    let mut v: Vec<f64> = union
        .iter()
        .map(|id| k.get(id).copied().unwrap_or(share))
        .collect();
    v.push((tail - share * unknown as f64).max(0.0));
    let floored: Vec<f64> = v.iter().map(|p| p.max(TOPK_FLOOR)).collect();
    let s: f64 = floored.iter().sum();
    floored.into_iter().map(|p| p / s).collect()
}

/// Screening KL(reference || candidate) of one teacher-forced position from the two top-k lists.
pub fn topk_kl(r: &TopkRow, c: &TopkRow) -> Result<TopkKl, String> {
    if r.id != c.id {
        return Err(format!(
            "forced tokens differ ({} vs {}): the two sides did not score the same ids",
            r.id, c.id
        ));
    }
    let kr = known(r);
    let kc = known(c);
    let union: Vec<u32> = kr
        .keys()
        .chain(kc.keys())
        .copied()
        .collect::<std::collections::BTreeSet<u32>>()
        .into_iter()
        .collect();
    let p = bucketed(r, &union);
    let q = bucketed(c, &union);
    let kl = p
        .iter()
        .zip(&q)
        .map(|(a, b)| if *a > 0.0 { a * (a / b).ln() } else { 0.0 })
        .sum::<f64>()
        .max(0.0);
    let cover = |x: &TopkRow| x.top.iter().map(|&(_, lp)| prob(lp)).sum::<f64>().min(1.0);
    let lp_or_floor = |lp: f64| if lp <= -9999.0 { -9999.0 } else { lp };
    Ok(TopkKl {
        kl,
        top1_agree: argmax_known(&kr) == argmax_known(&kc),
        ref_covered: cover(r),
        cand_covered: cover(c),
        abs_dlp: (lp_or_floor(c.lp) - lp_or_floor(r.lp)).abs(),
    })
}

/// Aggregate of the per-position screening comparisons.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TopkKlStats {
    pub positions: usize,
    pub mean: f64,
    pub p50: f64,
    pub p99: f64,
    pub max: f64,
    /// Fraction of positions whose most likely token agrees (0..1).
    pub top1: f64,
    /// Mean and minimum probability mass the reference's top-k lists cover (the recorded tail is 1 - covered).
    pub ref_covered_mean: f64,
    pub ref_covered_min: f64,
    pub cand_covered_mean: f64,
    pub abs_dlp_mean: f64,
    pub abs_dlp_max: f64,
}

pub fn topk_kl_stats(xs: &[TopkKl]) -> Option<TopkKlStats> {
    if xs.is_empty() {
        return None;
    }
    let n = xs.len() as f64;
    let mut kl: Vec<f64> = xs.iter().map(|x| x.kl).collect();
    kl.sort_by(|a, b| a.total_cmp(b));
    let mean = |f: &dyn Fn(&TopkKl) -> f64| xs.iter().map(f).sum::<f64>() / n;
    Some(TopkKlStats {
        positions: xs.len(),
        mean: kl.iter().sum::<f64>() / n,
        p50: crate::stats::quantile_sorted(&kl, 0.5),
        p99: crate::stats::quantile_sorted(&kl, 0.99),
        max: *kl.last().unwrap_or(&0.0),
        top1: xs.iter().filter(|x| x.top1_agree).count() as f64 / n,
        ref_covered_mean: mean(&|x| x.ref_covered),
        ref_covered_min: xs.iter().map(|x| x.ref_covered).fold(1.0, f64::min),
        cand_covered_mean: mean(&|x| x.cand_covered),
        abs_dlp_mean: mean(&|x| x.abs_dlp),
        abs_dlp_max: xs.iter().map(|x| x.abs_dlp).fold(0.0, f64::max),
    })
}

/// What a scoring endpoint (E2) reports for one teacher-forced position whose full-vocabulary logits are `x`: the
/// forced token's logprob and the `k` most likely tokens, from a log-softmax in f64 at temperature 1 (ties to the
/// smaller id).
pub fn topk_row_from_logits(x: &[f32], k: usize, forced: u32) -> TopkRow {
    let ls = log_softmax(x);
    let mut idx: Vec<u32> = (0..x.len() as u32).collect();
    let by = |a: &u32, b: &u32| ls[*b as usize].total_cmp(&ls[*a as usize]).then(a.cmp(b));
    let k = k.min(idx.len());
    if k > 0 && k < idx.len() {
        idx.select_nth_unstable_by(k - 1, by);
    }
    idx.truncate(k);
    idx.sort_by(by);
    TopkRow {
        id: forced,
        lp: ls.get(forced as usize).copied().unwrap_or(-9999.0),
        top: idx.iter().map(|&i| (i, ls[i as usize])).collect(),
    }
}

/// The full-vocabulary KL of two dumps next to the `kld_topk` screening KL of the same rows: each row reduced to
/// what the scoring endpoint would report (`topk_row_from_logits`) and compared with `topk_kl`. This is how the
/// screening thresholds are derived from the full-vocabulary release KL, per model, on real logits. `forced`:
/// the teacher-forced token of each compared row (rows `skip..rows`); None takes the base row's argmax, which is in
/// both lists' union anyway (then the forced token adds nothing).
pub fn topk_vs_full(
    a: &[f32],
    b: &[f32],
    rows: usize,
    skip: usize,
    k: usize,
    forced: Option<&[u32]>,
) -> Result<(KlStats, TopkKlStats), String> {
    let full = kl_stats(a, b, rows, skip)?;
    let v = a.len() / rows;
    if let Some(f) = forced {
        if f.len() != rows - skip {
            return Err(format!(
                "{} forced tokens for {} compared rows",
                f.len(),
                rows - skip
            ));
        }
    }
    let mut xs = Vec::with_capacity(rows - skip);
    for r in skip..rows {
        let ra = &a[r * v..(r + 1) * v];
        let rb = &b[r * v..(r + 1) * v];
        let id = forced.map(|f| f[r - skip]).unwrap_or(argmax(ra) as u32);
        xs.push(topk_kl(
            &topk_row_from_logits(ra, k, id),
            &topk_row_from_logits(rb, k, id),
        )?);
    }
    let tk = topk_kl_stats(&xs).ok_or("no rows")?;
    Ok((full, tk))
}

#[cfg(test)]
// the fixtures are numpy float32 values printed as Python floats (exact in f32)
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    const A: [f32; 54] = [
        0.0036904599983245134,
        0.8962365984916687,
        -0.8224135637283325,
        -2.6717755794525146,
        -1.364012360572815,
        -2.9749395847320557,
        0.1804308146238327,
        4.020645618438721,
        -1.4766196012496948,
        -1.8614246845245361,
        1.4695261716842651,
        1.0706610679626465,
        0.3162427544593811,
        -2.7914042472839355,
        -0.0877554640173912,
        2.085909605026245,
        -4.032643795013428,
        -1.372847318649292,
        -5.703668117523193,
        -3.8686132431030273,
        -5.525205135345459,
        -0.7052733898162842,
        -3.802339553833008,
        0.8137930631637573,
        0.47025325894355774,
        -0.560792863368988,
        -7.550279140472412,
        -1.6160787343978882,
        -0.14550283551216125,
        0.33992695808410645,
        -4.590407371520996,
        -1.4332598447799683,
        -2.9355571269989014,
        -2.426511764526367,
        3.1826958656311035,
        -2.4226040840148926,
        -0.09756511449813843,
        2.653169631958008,
        -1.7508013248443604,
        -0.335105836391449,
        0.33139243721961975,
        0.19134531915187836,
        -3.6751675605773926,
        0.228420689702034,
        4.076470375061035,
        -4.641434192657471,
        2.578148126602173,
        0.35806208848953247,
        -1.9244111776351929,
        6.00124979019165,
        2.2867791652679443,
        -3.5978667736053467,
        0.22354868054389954,
        1.7300688028335571,
    ];
    const B: [f32; 54] = [
        -0.05294417962431908,
        1.1011097431182861,
        -0.8423687815666199,
        -2.4716012477874756,
        -0.9324555397033691,
        -3.177638292312622,
        0.24137240648269653,
        3.881653308868408,
        -1.438439130783081,
        -2.217583179473877,
        1.2957357168197632,
        1.011802315711975,
        0.5858719348907471,
        -2.4478375911712646,
        -0.4848138391971588,
        1.8475168943405151,
        -3.8385727405548096,
        -1.9705733060836792,
        -5.703668117523193,
        -3.8686132431030273,
        -5.525205135345459,
        -0.7052733898162842,
        -3.802339553833008,
        0.8137930631637573,
        0.47025325894355774,
        -0.560792863368988,
        -7.550279140472412,
        -1.7071828842163086,
        -0.03972610831260681,
        0.30369582772254944,
        -4.649592876434326,
        -1.7674800157546997,
        -2.9390134811401367,
        -2.5595860481262207,
        3.532534122467041,
        -2.226677417755127,
        -0.10480819642543793,
        2.8536839485168457,
        -1.852762222290039,
        -0.01946789026260376,
        0.32977256178855896,
        0.36636003851890564,
        -4.062435626983643,
        0.33242470026016235,
        3.570009231567383,
        -5.25203275680542,
        2.486804962158203,
        0.08808380365371704,
        -1.8751953840255737,
        6.674676895141602,
        2.037262201309204,
        -3.7850499153137207,
        0.2851698696613312,
        1.8779728412628174,
    ];

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * b.abs().max(1e-300)
    }

    #[test]
    fn matches_kl_py() {
        // numpy reference (kl.py logic) on the same arrays
        let s = kl_stats(&A, &B, 6, 0).unwrap();
        assert_eq!((s.rows, s.vocab), (6, 9));
        assert!(close(s.mean, 0.014865548576266617), "{}", s.mean);
        assert!(close(s.p50, 0.010022909982590207), "{}", s.p50);
        assert!(close(s.p99, 0.047188327381092486), "{}", s.p99);
        assert!(close(s.max, 0.04879977197581095));
        assert_eq!(s.top1, 1.0);
        assert!(!s.bitwise_identical);
        assert_eq!(
            kl_line(&s),
            "rows 6 V 9 KL mean 1.487e-02 p50 1.002e-02 p99 4.719e-02 max 4.880e-02 top1 1.0000"
        );
        let s1 = kl_stats(&A, &B, 6, 1).unwrap();
        assert!(close(s1.mean, 0.017083294404486568) && close(s1.p99, 0.04751061630003617));
        assert_eq!(s1.rows, 5);
    }

    #[test]
    fn identical_disagreeing_and_gates() {
        let s = kl_stats(&A, &A, 6, 0).unwrap();
        assert!(s.bitwise_identical && s.mean == 0.0 && s.top1 == 1.0);
        assert!(kl_line(&s).ends_with("bitwise-identical"));
        assert_eq!(
            kl_gate("mimo d2000", &s, 1e-3, 0.99, true).status,
            GateStatus::Pass
        );
        // flip the top token of one row
        let a = [0.0f32, 5.0, 0.0, 0.0];
        let b = [5.0f32, 0.0, 0.0, 0.0];
        let d = kl_stats(&a, &b, 1, 0).unwrap();
        assert_eq!(d.top1, 0.0);
        assert!(d.mean > 1.0);
        let g = kl_gate("laguna d32000", &d, 3e-3, 0.98, true);
        assert_eq!(g.status, GateStatus::Fail);
        assert!(g.reasons[0].contains("top1 0.0000"));
        assert!(kl_stats(&a, &b[..3], 1, 0).is_err());
    }

    #[test]
    fn files_and_row_fallback() {
        let d = std::env::temp_dir().join(format!("tb-kl-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        // 128 rows x 3: 129 does not divide 384, so the fallback picks 128
        let a: Vec<f32> = (0..384).map(|i| (i % 7) as f32 * 0.5).collect();
        let w = |p: &Path, v: &[f32]| {
            std::fs::write(
                p,
                v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
            )
            .unwrap()
        };
        w(&d.join("a"), &a);
        w(&d.join("b"), &a);
        let s = kl_files(&d.join("a"), &d.join("b"), None, 0).unwrap();
        assert_eq!((s.rows, s.vocab), (128, 3));
        assert!(s.bitwise_identical);
        assert_eq!(py_e3(0.0), "0.000e+00");
        assert_eq!(py_e3(12345.0), "1.234e+04");
        std::fs::remove_dir_all(&d).ok();
    }

    fn row(id: u32, probs: &[(u32, f64)]) -> TopkRow {
        let lp = probs
            .iter()
            .find(|x| x.0 == id)
            .map(|x| x.1.ln())
            .unwrap_or(-9999.0);
        TopkRow {
            id,
            lp,
            top: probs.iter().map(|&(i, p)| (i, p.ln())).collect(),
        }
    }

    /// Plain KL over a common support (the reference for the cases where top-k sees the whole distribution).
    fn exact_kl(p: &[f64], q: &[f64]) -> f64 {
        p.iter().zip(q).map(|(a, b)| a * (a / b).ln()).sum()
    }

    #[test]
    fn topk_rows_from_full_logits() {
        // 6 tokens; k = 3 keeps the three largest, sorted, ties to the smaller id
        let x = [1.0f32, 3.0, 2.0, 3.0, -1.0, 0.0];
        let r = topk_row_from_logits(&x, 3, 4);
        let ids: Vec<u32> = r.top.iter().map(|t| t.0).collect();
        assert_eq!(ids, vec![1, 3, 2]);
        let ls = log_softmax(&x);
        assert_eq!(r.lp, ls[4]);
        assert_eq!(r.top[0].1, ls[1]);
        // k >= V: the whole distribution, so the screening KL is the full KL
        let y = [0.5f32, 2.5, 1.0, 2.0, -0.5, 0.3];
        let (full, tk) = topk_vs_full(&x, &y, 1, 0, 6, None).unwrap();
        assert!(
            (full.mean - tk.mean).abs() < 1e-9,
            "{} vs {}",
            full.mean,
            tk.mean
        );
        // k < V: close to the full KL when the tail is small, never NaN; identical dumps give 0 for both
        let (full, tk) = topk_vs_full(&x, &y, 1, 0, 3, Some(&[2])).unwrap();
        assert!(tk.mean.is_finite() && tk.mean >= 0.0 && full.mean > 0.0);
        let (full, tk) = topk_vs_full(&x, &x, 1, 0, 3, None).unwrap();
        assert_eq!((full.mean, tk.mean), (0.0, 0.0));
        assert!(topk_vs_full(&x, &y, 1, 0, 3, Some(&[1, 2])).is_err());
    }

    #[test]
    fn topk_identical_is_zero() {
        let r = row(3, &[(3, 0.5), (7, 0.3), (1, 0.15)]);
        let x = topk_kl(&r, &r.clone()).unwrap();
        assert_eq!(x.kl, 0.0);
        assert!(x.top1_agree);
        assert!((x.ref_covered - 0.95).abs() < 1e-12);
        assert_eq!(x.abs_dlp, 0.0);
    }

    #[test]
    fn topk_full_support_matches_exact_kl() {
        // the top-k lists cover the whole distribution: the bucketed KL is the exact KL (rest bucket ~ 0)
        let p = [0.6, 0.3, 0.1];
        let q = [0.5, 0.3, 0.2];
        let r = row(0, &[(0, p[0]), (1, p[1]), (2, p[2])]);
        let c = row(0, &[(0, q[0]), (1, q[1]), (2, q[2])]);
        let x = topk_kl(&r, &c).unwrap();
        assert!((x.kl - exact_kl(&p, &q)).abs() < 1e-9, "{}", x.kl);
        assert!((x.abs_dlp - (0.6f64.ln() - 0.5f64.ln())).abs() < 1e-12);
        assert!(x.top1_agree);
    }

    #[test]
    fn topk_tail_mass_is_a_bucket() {
        // both cover 0.8; the tails (0.2 vs 0.4) form the rest bucket, ids are the same
        let r = row(1, &[(1, 0.5), (2, 0.3)]);
        let c = row(1, &[(1, 0.4), (2, 0.2)]);
        let x = topk_kl(&r, &c).unwrap();
        let want = exact_kl(&[0.5, 0.3, 0.2], &[0.4, 0.2, 0.4]);
        assert!((x.kl - want).abs() < 1e-9, "{} vs {want}", x.kl);
        assert!((x.ref_covered - 0.8).abs() < 1e-12 && (x.cand_covered - 0.6).abs() < 1e-12);
    }

    #[test]
    fn topk_disjoint_lists() {
        // the candidate's top tokens are absent from the reference's list and vice versa (forced token 9 is in
        // neither list: its logprob is known on both sides). Each unknown id gets an equal share of the side's
        // tail, capped by that side's smallest top-k probability.
        let r = TopkRow {
            id: 9,
            lp: 0.01f64.ln(),
            top: vec![(1, 0.7f64.ln()), (2, 0.2f64.ln())],
        };
        let c = TopkRow {
            id: 9,
            lp: 0.02f64.ln(),
            top: vec![(5, 0.6f64.ln()), (6, 0.3f64.ln())],
        };
        let x = topk_kl(&r, &c).unwrap();
        assert!(!x.top1_agree);
        // union [1, 2, 5, 6, 9] + rest; reference: tail 0.09 over 2 unknown ids + rest -> 0.03 each (cap 0.2)
        let p = [0.7, 0.2, 0.03, 0.03, 0.01, 0.03];
        // candidate: tail 0.08 over 2 unknown ids + rest -> 0.08/3 each (cap 0.3)
        let t = 0.08 / 3.0;
        let q = [t, t, 0.6, 0.3, 0.02, t];
        assert!((x.kl - exact_kl(&p, &q)).abs() < 1e-9, "{}", x.kl);
        assert!(x.kl > 1.0);
        // the cap: a single tiny top-k entry bounds what an unknown id may receive
        let r2 = TopkRow {
            id: 1,
            lp: 0.5f64.ln(),
            top: vec![(1, 0.5f64.ln()), (2, 0.001f64.ln())],
        };
        let c2 = TopkRow {
            id: 1,
            lp: 0.5f64.ln(),
            top: vec![(1, 0.5f64.ln()), (3, 0.3f64.ln())],
        };
        let y = topk_kl(&r2, &c2).unwrap();
        // reference: id 3 unknown, tail 0.499, share min(0.2495, 0.001) = 0.001, rest 0.498
        // candidate: id 2 unknown, tail 0.2, share min(0.1, 0.3) = 0.1, rest 0.1
        let want = exact_kl(&[0.5, 0.001, 0.001, 0.498], &[0.5, 0.1, 0.3, 0.1]);
        assert!((y.kl - want).abs() < 1e-9, "{} vs {want}", y.kl);
        // forced tokens must agree
        let mut c3 = c2.clone();
        c3.id = 4;
        assert!(topk_kl(&r2, &c3).is_err());
    }

    #[test]
    fn topk_minus_infinity_and_stats() {
        let r = TopkRow {
            id: 1,
            lp: -9999.0,
            top: vec![(2, 0.0)],
        };
        let c = TopkRow {
            id: 1,
            lp: -9999.0,
            top: vec![(2, 0.0)],
        };
        let x = topk_kl(&r, &c).unwrap();
        assert_eq!(x.kl, 0.0);
        assert!(x.kl.is_finite());
        let s = topk_kl_stats(&[
            x,
            TopkKl {
                kl: 0.5,
                top1_agree: false,
                ref_covered: 0.5,
                cand_covered: 0.4,
                abs_dlp: 2.0,
            },
        ])
        .unwrap();
        assert_eq!(s.positions, 2);
        assert!((s.mean - 0.25).abs() < 1e-12 && s.max == 0.5);
        assert_eq!(s.top1, 0.5);
        assert_eq!(s.ref_covered_min, 0.5);
        assert_eq!(s.abs_dlp_max, 2.0);
        assert!(topk_kl_stats(&[]).is_none());
    }
}
