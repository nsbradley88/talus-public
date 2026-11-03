//! Pick a speculative window's draft depth from a measured per-width round-cost table
//! instead of fixed confidence thresholds.
//!
//! A round that verifies `k` drafts (`k + 1` rows) costs `C(k)` ms (the table, measured per verify width at two context
//! depths and interpolated linearly in between) and emits `E(k) = 1 + P_1 + ... + P_k` tokens on average, where `P_j`
//! is the probability that drafts `1..=j` are all accepted. `P_j` is estimated from the drafter's own per-draft
//! confidences (`c_1 * ... * c_j`, each calibrated as `c^gamma`). The rule keeps the `k` that maximises `E(k) / C(k)`
//! (tokens per ms), never fewer than `dmin` drafts.
//!
//! Exactness: greedy outputs are the target's argmax whatever the window, so any window length is exact as long as the
//! engine's windows of that width are bitwise verify(1) (27B: every width; FN: 1..=8 rows since bug #22's fix).
//!
//! Switches (per model prefix `P` = `Q27` / `FN`; read once by the engine):
//!   TALUS_<P>_COSTDEPTH=1|0             on / off (default: the model's; 27B MTP and FN on since w1, 27B DFlash2 off)
//!   TALUS_<P>_COSTDEPTH_MS="c1,c2,..."   round cost in ms for 1, 2, ... verify rows at the near depth
//!   TALUS_<P>_COSTDEPTH_MS_FAR="..."     the same at the far depth (default: = near)
//!   TALUS_<P>_COSTDEPTH_FAR=N            the far depth in tokens (default 32768; near depth is 0..2K)
//!   TALUS_<P>_COSTDEPTH_GAMMA=g          calibration exponent (default: the model's; 27B 1.6, FN 1.5)
//!   TALUS_<P>_COSTDEPTH_DMIN=n           drafts always kept (default 1)
//!   TALUS_<P>_COSTDEPTH_STEP_MS=x        cost of one more chained draft step when the rule wants to extend (27B MTP)

/// Maximum verify rows the table holds.
pub const CD_MAX_ROWS: usize = 16;

/// Cost-table draft depth rule (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct CostDepth {
    /// Round cost (ms) by verify rows (index 1..=CD_MAX_ROWS; 0 unused) at depth <= `near`.
    pub near_ms: [f64; CD_MAX_ROWS + 1],
    /// Round cost (ms) by verify rows at depth >= `far`.
    pub far_ms: [f64; CD_MAX_ROWS + 1],
    /// Near / far depths (tokens) of the two tables.
    pub near: f64,
    pub far: f64,
    /// Calibration exponent on each draft's confidence.
    pub gamma: f64,
    /// Drafts always kept.
    pub dmin: usize,
    /// Cost of one more chained draft step (ms), used by [`CostDepth::want_more`].
    pub step_ms: f64,
}

fn parse_table(s: &str) -> Option<[f64; CD_MAX_ROWS + 1]> {
    let v: Vec<f64> = s
        .split(',')
        .map(|x| x.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .ok()?;
    if v.is_empty() || v.iter().any(|x| !(x.is_finite() && *x > 0.0)) {
        return None;
    }
    let mut t = [0.0; CD_MAX_ROWS + 1];
    for r in 1..=CD_MAX_ROWS {
        t[r] = if r <= v.len() {
            v[r - 1]
        } else {
            // extrapolate with the last measured step
            let n = v.len();
            let step = if n >= 2 { v[n - 1] - v[n - 2] } else { 0.0 };
            v[n - 1] + step.max(0.0) * (r - n) as f64
        };
    }
    Some(t)
}

impl CostDepth {
    /// A rule from explicit tables (`near` / `far`: costs for 1, 2, ... rows).
    pub fn new(near: &[f64], far: &[f64], far_depth: f64, gamma: f64, dmin: usize) -> Option<Self> {
        let j = |v: &[f64]| {
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let n = parse_table(&j(near))?;
        let f = if far.is_empty() {
            n
        } else {
            parse_table(&j(far))?
        };
        Some(Self {
            near_ms: n,
            far_ms: f,
            near: 2048.0,
            far: far_depth.max(4096.0),
            gamma: if gamma.is_finite() && gamma > 0.0 {
                gamma
            } else {
                1.0
            },
            dmin,
            step_ms: 0.0,
        })
    }

    /// The rule from `TALUS_<prefix>_COSTDEPTH*`, falling back to `default_near` / `default_far` tables when the
    /// variables are not set. `None` = off.
    pub fn from_env(
        prefix: &str,
        default_near: &[f64],
        default_far: &[f64],
        default_on: bool,
    ) -> Option<Self> {
        Self::from_env_with(prefix, default_near, default_far, default_on, 1.0, 1, 0.0)
    }

    /// [`CostDepth::from_env`] with the model's default calibration exponent, minimum drafts and chain step cost.
    pub fn from_env_with(
        prefix: &str,
        default_near: &[f64],
        default_far: &[f64],
        default_on: bool,
        default_gamma: f64,
        default_dmin: usize,
        default_step_ms: f64,
    ) -> Option<Self> {
        let k = |s: &str| format!("TALUS_{prefix}_COSTDEPTH{s}");
        let on = match std::env::var(k("")) {
            Ok(v) => v.trim() != "0" && !v.trim().is_empty(),
            Err(_) => default_on,
        };
        if !on {
            return None;
        }
        let fnum = |s: &str| {
            std::env::var(k(s))
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
        };
        let near = std::env::var(k("_MS"))
            .ok()
            .and_then(|v| parse_table(&v))
            .or_else(|| {
                CostDepth::new(default_near, default_far, 32768.0, 1.0, 1).map(|c| c.near_ms)
            })?;
        let far = std::env::var(k("_MS_FAR"))
            .ok()
            .and_then(|v| parse_table(&v))
            .or_else(|| {
                if std::env::var(k("_MS")).is_ok() {
                    Some(near)
                } else {
                    CostDepth::new(default_near, default_far, 32768.0, 1.0, 1).map(|c| c.far_ms)
                }
            })?;
        let gamma = fnum("_GAMMA")
            .filter(|g| g.is_finite() && *g > 0.0)
            .unwrap_or(default_gamma);
        Some(Self {
            near_ms: near,
            far_ms: far,
            near: 2048.0,
            far: fnum("_FAR").unwrap_or(32768.0).max(4096.0),
            gamma,
            dmin: fnum("_DMIN")
                .map(|v| v.max(0.0) as usize)
                .unwrap_or(default_dmin),
            step_ms: fnum("_STEP_MS").unwrap_or(default_step_ms).max(0.0),
        })
    }

    /// Round cost (ms) of a window with `rows` verify rows at context `depth`.
    pub fn cost(&self, rows: usize, depth: f64) -> f64 {
        let r = rows.clamp(1, CD_MAX_ROWS);
        let w = ((depth - self.near) / (self.far - self.near)).clamp(0.0, 1.0);
        self.near_ms[r] * (1.0 - w) + self.far_ms[r] * w
    }

    fn cal(&self, c: f32) -> f64 {
        let c = if c.is_finite() {
            (c as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if self.gamma == 1.0 {
            c
        } else {
            c.powf(self.gamma)
        }
    }

    /// Of the `n` drafts with confidences `conf[0..n]`, how many to verify at context `depth`: the `k` in
    /// `dmin.min(n)..=n` maximising expected tokens per ms (ties: the shorter window).
    pub fn keep(&self, conf: &[f32], n: usize, depth: f64) -> usize {
        let n = n.min(conf.len()).min(CD_MAX_ROWS - 1);
        let lo = self.dmin.min(n);
        let mut p = 1.0f64;
        let mut e = 1.0f64;
        let mut best_k = lo;
        let mut best = f64::NEG_INFINITY;
        for k in 0..=n {
            if k > 0 {
                p *= self.cal(conf[k - 1]);
                e += p;
            }
            if k < lo {
                continue;
            }
            let r = e / self.cost(k + 1, depth);
            if r > best * (1.0 + 1e-9) {
                best = r;
                best_k = k;
            }
        }
        best_k
    }

    /// Whether to chain one more draft after `conf[0..n]` (all of which the rule keeps): the next draft is assumed
    /// as confident as the last one, and it must pay for its verify row plus `step_ms`.
    pub fn want_more(&self, conf: &[f32], n: usize, depth: f64) -> bool {
        let n = n.min(conf.len());
        if n == 0 || n + 2 > CD_MAX_ROWS {
            return false;
        }
        let mut p = 1.0f64;
        let mut e = 1.0f64;
        for c in &conf[..n] {
            p *= self.cal(*c);
            e += p;
        }
        let pn = p * self.cal(conf[n - 1]);
        let now = e / self.cost(n + 1, depth);
        let more = (e + pn) / (self.cost(n + 2, depth) + self.step_ms);
        more > now
    }

    /// One-line description for the engine log.
    pub fn describe(&self) -> String {
        let t = |v: &[f64; CD_MAX_ROWS + 1]| {
            v[1..=8]
                .iter()
                .map(|x| format!("{x:.1}"))
                .collect::<Vec<_>>()
                .join("/")
        };
        format!(
            "cost-table draft depth (rows 1..8 ms {} near, {} at {:.0}; gamma {:.2}, dmin {}, step {:.1} ms)",
            t(&self.near_ms),
            t(&self.far_ms),
            self.far,
            self.gamma,
            self.dmin,
            self.step_ms
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule() -> CostDepth {
        CostDepth::new(&[70.0, 75.0, 80.0, 85.0, 90.0, 95.0], &[], 32768.0, 1.0, 1).unwrap()
    }

    #[test]
    fn keeps_confident_and_cuts_unlikely() {
        let r = rule();
        assert_eq!(r.keep(&[0.99, 0.99, 0.99, 0.99, 0.99], 5, 0.0), 5);
        // 1 + 0.05: (1.05)/75 < 1/70 -> no draft beyond dmin? dmin 1 keeps one
        assert_eq!(r.keep(&[0.05, 0.9, 0.9], 3, 0.0), 1);
        let mut r0 = rule();
        r0.dmin = 0;
        assert_eq!(r0.keep(&[0.05, 0.9, 0.9], 3, 0.0), 0);
        // a weak middle draft ends the useful prefix
        assert_eq!(r.keep(&[0.95, 0.95, 0.1, 0.99], 4, 0.0), 2);
    }

    #[test]
    fn table_extrapolates_and_interpolates() {
        let r = CostDepth::new(&[10.0, 12.0], &[20.0, 30.0], 32768.0, 1.0, 1).unwrap();
        assert_eq!(r.cost(4, 0.0), 16.0);
        assert_eq!(r.cost(1, 32768.0), 20.0);
        assert!((r.cost(1, (2048.0 + 32768.0) / 2.0) - 15.0).abs() < 1e-9);
        assert_eq!(r.cost(3, 1e9), 40.0);
    }

    #[test]
    fn want_more_needs_confidence() {
        let r = rule();
        assert!(r.want_more(&[0.99, 0.99, 0.99], 3, 0.0));
        assert!(!r.want_more(&[0.5, 0.5, 0.5], 3, 0.0));
        assert!(!r.want_more(&[], 0, 0.0));
    }

    #[test]
    fn nan_conf_is_zero() {
        let r = rule();
        assert_eq!(r.keep(&[0.99, f32::NAN, 0.99], 3, 0.0), 1);
    }
}
