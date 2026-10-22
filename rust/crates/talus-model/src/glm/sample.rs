//! Host sampling for GLM: llama.cpp-style chain (top-k -> top-p -> min-p -> temperature) and speculative
//! rejection sampling (exact in distribution) for MTP drafts.

#[derive(Debug, Clone, Copy)]
pub struct SamplerParams {
    pub temp: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
}

impl Default for SamplerParams {
    fn default() -> Self {
        // llama-server defaults (the halo reference serves GLM with them; clients send temperature / top_p)
        Self {
            temp: 1.0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            seed: 0,
        }
    }
}

impl SamplerParams {
    pub fn greedy(&self) -> bool {
        self.temp <= 0.0
    }
}

/// SplitMix64 RNG.
#[derive(Debug, Clone)]
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// uniform in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// `GLM_SSAMP_OFF=1`: draft the MTP head's argmax (point-mass q) instead of sampling q (A/B reference only).
pub fn ssamp_off() -> bool {
    std::env::var("GLM_SSAMP_OFF")
        .map(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn argmax(l: &[f32]) -> u32 {
    let mut b = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in l.iter().enumerate() {
        if v > bv {
            bv = v;
            b = i;
        }
    }
    b as u32
}

/// Candidate distribution after the sampler chain: (token, probability), probabilities sum to 1, sorted by
/// descending probability. `ids` maps logit index -> token id (draft vocabulary) when given.
pub fn dist(logits: &[f32], p: &SamplerParams, ids: Option<&[u32]>) -> Vec<(u32, f32)> {
    let tok = |i: usize| ids.map(|m| m[i]).unwrap_or(i as u32);
    if p.greedy() {
        let a = argmax(logits) as usize;
        return vec![(tok(a), 1.0)];
    }
    let k = if p.top_k <= 0 {
        logits.len()
    } else {
        (p.top_k as usize).min(logits.len())
    };
    let mut idx: Vec<u32>;
    if k <= 256 {
        // single pass: keep the k best in a small sorted buffer (no 154880-entry index vector)
        idx = Vec::with_capacity(k + 1);
        let mut thr = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if idx.len() < k || v > thr {
                let pos = idx.partition_point(|&j| logits[j as usize] >= v);
                idx.insert(pos, i as u32);
                if idx.len() > k {
                    idx.pop();
                }
                if idx.len() == k {
                    thr = logits[idx[k - 1] as usize];
                }
            }
        }
    } else {
        idx = (0..logits.len() as u32).collect();
        if k < idx.len() {
            idx.select_nth_unstable_by(k - 1, |&a, &b| {
                logits[b as usize].total_cmp(&logits[a as usize])
            });
            idx.truncate(k);
        }
        idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    }
    // softmax at temperature 1 for top-p / min-p (llama.cpp applies temperature last)
    let mx = logits[idx[0] as usize];
    let mut pr: Vec<f64> = idx
        .iter()
        .map(|&i| ((logits[i as usize] - mx) as f64).exp())
        .collect();
    let s: f64 = pr.iter().sum();
    for v in pr.iter_mut() {
        *v /= s;
    }
    let mut n = pr.len();
    if p.top_p < 1.0 {
        let mut c = 0.0;
        for (i, v) in pr.iter().enumerate() {
            c += v;
            if c >= p.top_p as f64 {
                n = i + 1;
                break;
            }
        }
    }
    if p.min_p > 0.0 {
        let thr = pr[0] * p.min_p as f64;
        let m = pr[..n].iter().take_while(|&&v| v >= thr).count().max(1);
        n = n.min(m);
    }
    // temperature
    let t = p.temp.max(1e-6) as f64;
    let mut q: Vec<f64> = idx[..n]
        .iter()
        .map(|&i| (((logits[i as usize] - mx) as f64) / t).exp())
        .collect();
    let s: f64 = q.iter().sum();
    for v in q.iter_mut() {
        *v /= s;
    }
    idx[..n]
        .iter()
        .zip(q)
        .map(|(&i, v)| (tok(i as usize), v as f32))
        .collect()
}

pub fn sample(d: &[(u32, f32)], rng: &mut Rng) -> u32 {
    if d.len() == 1 {
        return d[0].0;
    }
    let r = rng.uniform() as f32;
    let mut c = 0.0f32;
    for &(t, p) in d {
        c += p;
        if r < c {
            return t;
        }
    }
    d[d.len() - 1].0
}

pub fn prob_of(d: &[(u32, f32)], t: u32) -> f32 {
    d.iter().find(|x| x.0 == t).map(|x| x.1).unwrap_or(0.0)
}

/// Speculative acceptance of draft token `x` drafted from `q` against target `p`: Ok(x) if accepted, Err(y) with a
/// replacement sampled from norm(max(0, p - q)) otherwise. Greedy: accept iff x == argmax p.
pub fn accept(
    p: &[(u32, f32)],
    q: &[(u32, f32)],
    x: u32,
    greedy: bool,
    rng: &mut Rng,
) -> Result<u32, u32> {
    if greedy {
        return if p[0].0 == x { Ok(x) } else { Err(p[0].0) };
    }
    let px = prob_of(p, x);
    let qx = prob_of(q, x).max(1e-20);
    if (rng.uniform() as f32) < (px / qx).min(1.0) {
        return Ok(x);
    }
    let mut res: Vec<(u32, f32)> = p
        .iter()
        .map(|&(t, pt)| (t, (pt - prob_of(q, t)).max(0.0)))
        .collect();
    let s: f32 = res.iter().map(|x| x.1).sum();
    if s <= 0.0 {
        return Err(sample(p, rng));
    }
    for v in res.iter_mut() {
        v.1 /= s;
    }
    Err(sample(&res, rng))
}

#[cfg(test)]
#[path = "ssamp_tests.rs"]
mod ssamp_tests;
