//! Token logprobs for the stdio engine protocol's opt-in `GEN ... logprobs=K` (`L` lines) and the teacher-forced
//! `SCORE` command (`S` lines); bench-design/server-extensions.md E1/E2.
//!
//! Semantics: the target model's distribution at temperature 1 before any sampling filter or penalty, i.e.
//! `log_softmax(logits)` over the full vocabulary of one logits row, accumulated in f64. This first version runs on
//! the host over a D2H copy of the needed rows (an engine calls it only when a request asked for logprobs, so the
//! default path does no extra work).

/// One row's logprobs: the chosen (or teacher-forced) token, its logprob, its 1-based rank, the row's entropy (nats)
/// and the top `k` tokens in descending order (ties: lower id first).
#[derive(Clone, Debug, PartialEq)]
pub struct RowLogprobs {
    pub id: i32,
    pub logprob: f32,
    pub rank: u32,
    pub entropy: f32,
    pub top: Vec<(i32, f32)>,
}

/// Largest `k` an engine accepts for `logprobs=K` / `SCORE <k>`.
pub const MAX_TOP_K: usize = 64;

/// log-softmax facts of one logits row for token `chosen`, with the top `k` (clamped to [`MAX_TOP_K`]).
/// Non-finite logits count as minus infinity.
pub fn row_logprobs(logits: &[f32], chosen: i32, k: usize) -> RowLogprobs {
    let k = k.min(MAX_TOP_K).min(logits.len());
    let fin = |x: f32| if x.is_finite() { x } else { f32::NEG_INFINITY };
    let mut m = f32::NEG_INFINITY;
    for &x in logits {
        let x = fin(x);
        if x > m {
            m = x;
        }
    }
    if m == f32::NEG_INFINITY {
        return RowLogprobs {
            id: chosen,
            logprob: f32::NEG_INFINITY,
            rank: 1,
            entropy: 0.0,
            top: Vec::new(),
        };
    }
    let md = m as f64;
    let mut sum = 0.0f64;
    let mut sum_xe = 0.0f64;
    for &x in logits {
        let x = fin(x);
        if x == f32::NEG_INFINITY {
            continue;
        }
        let d = x as f64 - md;
        let e = d.exp();
        sum += e;
        sum_xe += e * d;
    }
    let lse = sum.ln();
    // H = -sum p (d - lse) = lse - E[d]
    let entropy = (lse - sum_xe / sum).max(0.0) as f32;
    let lp_of = |x: f32| -> f32 {
        let x = fin(x);
        if x == f32::NEG_INFINITY {
            f32::NEG_INFINITY
        } else {
            (x as f64 - md - lse) as f32
        }
    };
    let cx = if chosen >= 0 && (chosen as usize) < logits.len() {
        fin(logits[chosen as usize])
    } else {
        f32::NEG_INFINITY
    };
    // top-k: a small sorted buffer (k <= 64); rank = 1 + #logits strictly above the chosen token's
    let mut top: Vec<(i32, f32)> = Vec::with_capacity(k + 1);
    let mut above = 0u32;
    for (i, &x) in logits.iter().enumerate() {
        let x = fin(x);
        if x > cx {
            above += 1;
        }
        if k == 0 {
            continue;
        }
        if top.len() == k && x <= top[k - 1].1 {
            continue;
        }
        let pos = top.partition_point(|&(_, v)| v >= x);
        top.insert(pos, (i as i32, x));
        if top.len() > k {
            top.pop();
        }
    }
    RowLogprobs {
        id: chosen,
        logprob: lp_of(cx),
        rank: above + 1,
        entropy,
        top: top.into_iter().map(|(i, x)| (i, lp_of(x))).collect(),
    }
}

fn top_list(top: &[(i32, f32)]) -> String {
    if top.is_empty() {
        return "-".to_string();
    }
    let mut s = String::with_capacity(top.len() * 16);
    for (n, (id, lp)) in top.iter().enumerate() {
        if n > 0 {
            s.push(',');
        }
        s.push_str(&format!("{id}:{lp}"));
    }
    s
}

/// `L <lp> <k> <id>:<lp>,...` (`-` for an empty list), sent right before the token's `T <id>` line.
pub fn format_l_line(r: &RowLogprobs) -> String {
    format!("L {} {} {}", r.logprob, r.top.len(), top_list(&r.top))
}

/// `S <id> <lp> <rank> <entropy> <k> <id>:<lp>,...` (`-` for an empty list), one per scored position.
pub fn format_s_line(r: &RowLogprobs) -> String {
    format!(
        "S {} {} {} {} {} {}",
        r.id,
        r.logprob,
        r.rank,
        r.entropy,
        r.top.len(),
        top_list(&r.top)
    )
}

/// Parse a `SCORE <topk> <csv prompt ids>|<csv continuation ids>` line into (k, prompt, continuation).
pub fn parse_score_line(line: &str) -> Result<(usize, Vec<i32>, Vec<i32>), String> {
    let rest = line
        .trim()
        .strip_prefix("SCORE ")
        .ok_or_else(|| "ERR expected: SCORE <topk> <prompt ids>|<continuation ids>".to_string())?;
    let mut it = rest.trim_start().splitn(2, ' ');
    let k: usize = it
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "ERR SCORE: bad topk".to_string())?;
    let ids = it.next().unwrap_or("").trim();
    let (p, c) = ids
        .split_once('|')
        .ok_or_else(|| "ERR SCORE: expected <prompt ids>|<continuation ids>".to_string())?;
    let parse = |s: &str| -> Result<Vec<i32>, String> {
        s.split(',')
            .filter(|x| !x.trim().is_empty())
            .map(|x| {
                x.trim()
                    .parse::<i32>()
                    .map_err(|_| format!("ERR SCORE: bad id {x}"))
            })
            .collect()
    };
    let prompt = parse(p)?;
    let cont = parse(c)?;
    if prompt.is_empty() || cont.is_empty() {
        return Err("ERR SCORE: the prompt and the continuation must be non-empty".to_string());
    }
    if k > MAX_TOP_K {
        return Err(format!("ERR SCORE: topk must be <= {MAX_TOP_K}"));
    }
    Ok((k, prompt, cont))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact(logits: &[f32]) -> Vec<f64> {
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let s: f64 = logits.iter().map(|&x| (x as f64 - m).exp()).sum();
        logits.iter().map(|&x| x as f64 - m - s.ln()).collect()
    }

    #[test]
    fn matches_exact_log_softmax_and_topk() {
        let logits: Vec<f32> = (0..1000)
            .map(|i| ((i * 7919) % 1013) as f32 * 0.013 - 3.0)
            .collect();
        let ex = exact(&logits);
        let r = row_logprobs(&logits, 5, 20);
        assert!((r.logprob as f64 - ex[5]).abs() < 1e-5);
        assert_eq!(r.top.len(), 20);
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
        for (n, (id, lp)) in r.top.iter().enumerate() {
            assert_eq!(*id as usize, idx[n]);
            assert!((*lp as f64 - ex[idx[n]]).abs() < 1e-5);
        }
        let rank = 1 + logits.iter().filter(|&&x| x > logits[5]).count() as u32;
        assert_eq!(r.rank, rank);
        let h: f64 = -ex.iter().map(|&l| l.exp() * l).sum::<f64>();
        assert!((r.entropy as f64 - h).abs() < 1e-4);
    }

    #[test]
    fn greedy_top1_is_chosen() {
        let logits = vec![0.5f32, 2.0, -1.0, 2.0 - 1e-6, f32::NAN];
        let r = row_logprobs(&logits, 1, 3);
        assert_eq!(r.top[0].0, 1);
        assert_eq!(r.top[0].1, r.logprob);
        assert_eq!(r.rank, 1);
        assert_eq!(r.top.len(), 3);
    }

    #[test]
    fn lines_round_trip() {
        let r = RowLogprobs {
            id: 7,
            logprob: -0.25,
            rank: 2,
            entropy: 1.5,
            top: vec![(3, -0.1), (7, -0.25)],
        };
        assert_eq!(format_l_line(&r), "L -0.25 2 3:-0.1,7:-0.25");
        assert_eq!(format_s_line(&r), "S 7 -0.25 2 1.5 2 3:-0.1,7:-0.25");
        let r0 = RowLogprobs { top: vec![], ..r };
        assert_eq!(format_l_line(&r0), "L -0.25 0 -");
        assert_eq!(
            parse_score_line("SCORE 5 1,2,3|4,5").unwrap(),
            (5, vec![1, 2, 3], vec![4, 5])
        );
        assert!(parse_score_line("SCORE 5 1,2,3").is_err());
        assert!(parse_score_line("SCORE 99 1|2").is_err());
    }
}
