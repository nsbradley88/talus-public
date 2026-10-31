//! confident-prefix verify for block (parallel) drafters: the MiMo and Laguna DFlash drafters draft a whole
//! block in one forward, so a round's drafts all arrive at once with their rows' distributions. Verify only the longest
//! prefix whose running product of the drafter's per-row top-1 probabilities stays >= a threshold; the cut rows are
//! not verified (they cost verify rows, mostly expert loads on the MoE targets, and are rarely accepted).
//! Idea from hipfire (warpfront/hipfire, Apache-2.0: DFlash2 verifies the prefix whose running product of selector
//! confidences stays >= 0.5); ideas only, this is our own implementation. The 27B DFlash2 path already has the
//! selector-weight form (`DraftLen::keep_dfl`, STRATA_DFL_STOP), the 27B MTP and FN MTP chains have their own
//! confidence stops (STRATA_ADRAFT=2, spec_min_p); this module covers the drafters that had none.
//!
//! Exactness. Greedy: every emitted token is the target's argmax at its position whatever the drafts are, and the cut
//! only shortens a verify window to a width the engine already verifies (MiMo: 2..=5 rows, the copy windows' widths;
//! Laguna: every width is bitwise verify(1)), so greedy output is byte-identical. Sampled: a row's top-1 probability is
//! the drafter's own (T = 1, before the request's transforms) and DFlash rows do not condition on earlier drafts, so
//! the cut does not depend on any drawn draft and rejection sampling stays exact in distribution; the per-seed bytes
//! change (fewer draws), so sampled requests have their own switch, off by default.
//!
//! Switches (read once per process; `MODEL` = `MIMO` / `LAGUNA`, the model's variable wins over the generic one):
//!   TALUS_<MODEL>_DF_CONF / TALUS_DF_CONF        greedy threshold on the running product (0 = off; unset = the
//!                                                model's default: MIMO 0.6, LAGUNA 0.45 - greedy output unchanged,
//!                                                faster code; other models off)
//!   TALUS_<MODEL>_DF_CONF_S / TALUS_DF_CONF_S    sampled threshold (unset / 0 = off)
//!   TALUS_DF_CONF_DMIN                           drafts always verified (default 1)
//!   TALUS_DF_CONF_TRACE=1                        compute the tops and log one `dfconf` line per DFlash round to stderr
//!                                                (with the thresholds off: calibration data, output and timing of
//!                                                the verify unchanged)

/// Confident-prefix rule for one model (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfPrefix {
    /// Running-product threshold [greedy, sampled]; <= 0 = off.
    pub thr: [f32; 2],
    /// Drafts always kept (the rule is checked from draft `dmin` on).
    pub dmin: usize,
    /// Log one line per round (`dfconf ...`).
    pub trace: bool,
}

impl Default for ConfPrefix {
    fn default() -> Self {
        Self {
            thr: [0.0, 0.0],
            dmin: 1,
            trace: false,
        }
    }
}

fn env_f32(keys: &[String]) -> Option<f32> {
    keys.iter()
        .find_map(|k| std::env::var(k).ok())
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
}

/// Default greedy threshold per model when no variable is set (the trace replay maximum,
/// tier1 interleaved c1/t/c2 greedy 9/9 and faster on code beyond the control spread; chat/edit within noise).
pub fn default_greedy_threshold(model: &str) -> f32 {
    match model.to_ascii_uppercase().as_str() {
        "MIMO" => 0.6,
        "LAGUNA" => 0.45,
        _ => 0.0,
    }
}

impl ConfPrefix {
    /// The rule from the environment for `model` (`"MIMO"`, `"LAGUNA"`).
    pub fn from_env(model: &str) -> Self {
        let m = model.to_ascii_uppercase();
        let g = env_f32(&[format!("TALUS_{m}_DF_CONF"), "TALUS_DF_CONF".to_string()]);
        let s = env_f32(&[
            format!("TALUS_{m}_DF_CONF_S"),
            "TALUS_DF_CONF_S".to_string(),
        ]);
        let dmin = std::env::var("TALUS_DF_CONF_DMIN")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(1);
        let trace = std::env::var("TALUS_DF_CONF_TRACE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        Self {
            thr: [
                g.unwrap_or_else(|| default_greedy_threshold(&m))
                    .clamp(0.0, 1.0),
                s.unwrap_or(0.0).clamp(0.0, 1.0),
            ],
            dmin,
            trace,
        }
    }

    /// True when the engine must compute the per-row top-1 probabilities (a threshold or the trace is on).
    pub fn active(&self) -> bool {
        self.thr[0] > 0.0 || self.thr[1] > 0.0 || self.trace
    }

    /// True when the rule may cut a round of this kind.
    pub fn cuts(&self, sampled: bool) -> bool {
        self.thr[usize::from(sampled)] > 0.0
    }

    /// The number of the `k` drafts to verify: the longest prefix (at least `min(dmin, k)`) whose running product of
    /// `tops` stays >= the threshold. `k` itself when the rule is off for this kind of request or `tops` is short
    /// (no probability for a row = no evidence to cut it).
    pub fn keep(&self, tops: &[f32], k: usize, sampled: bool) -> usize {
        let thr = self.thr[usize::from(sampled)];
        if thr <= 0.0 || tops.len() < k {
            return k;
        }
        let mut p = 1.0f32;
        for (i, &t) in tops.iter().enumerate().take(k) {
            p *= t;
            // NaN-safe: a non-finite probability cuts (never extends) the prefix
            if i >= self.dmin && !(p >= thr) {
                return i;
            }
        }
        k
    }

    /// One trace line per DFlash round (only with `trace`): drafted `k`, verified `kept`, accepted `acc` (of `kept`),
    /// verify wall time, the drafter's top-1 probabilities of all `k` rows.
    pub fn trace_round(
        &self,
        model: &str,
        sampled: bool,
        tops: &[f32],
        k: usize,
        kept: usize,
        acc: usize,
        verify_ms: f64,
    ) {
        if !self.trace {
            return;
        }
        let t: Vec<String> = tops.iter().take(k).map(|v| format!("{v:.4}")).collect();
        eprintln!(
            "dfconf model={model} s={} thr={:.3} k={k} keep={kept} acc={acc} ms={verify_ms:.3} tops={}",
            u8::from(sampled),
            self.thr[usize::from(sampled)],
            t.join(",")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(g: f32, s: f32, dmin: usize) -> ConfPrefix {
        ConfPrefix {
            thr: [g, s],
            dmin,
            trace: false,
        }
    }

    #[test]
    fn model_defaults() {
        assert_eq!(default_greedy_threshold("mimo"), 0.6);
        assert_eq!(default_greedy_threshold("LAGUNA"), 0.45);
        assert_eq!(default_greedy_threshold("glm"), 0.0);
    }

    #[test]
    fn off_keeps_everything() {
        let r = ConfPrefix::default();
        assert!(!r.active());
        assert_eq!(r.keep(&[0.01, 0.01, 0.01], 3, false), 3);
        assert_eq!(r.keep(&[0.01, 0.01, 0.01], 3, true), 3);
    }

    #[test]
    fn running_product_cut() {
        let r = rule(0.5, 0.0, 1);
        // 0.9, 0.81, 0.405 -> cut before draft 2
        assert_eq!(r.keep(&[0.9, 0.9, 0.5, 0.99], 4, false), 2);
        // sampled is off: untouched
        assert_eq!(r.keep(&[0.9, 0.9, 0.5, 0.99], 4, true), 4);
        // all confident
        assert_eq!(r.keep(&[0.99, 0.99, 0.99, 0.99], 4, false), 4);
        // k smaller than the tops
        assert_eq!(r.keep(&[0.99, 0.2, 0.99], 1, false), 1);
    }

    #[test]
    fn dmin_rows_always_kept() {
        let r = rule(0.5, 0.0, 1);
        // the first draft is kept even at 0.1
        assert_eq!(r.keep(&[0.1, 0.99], 2, false), 1);
        let r2 = rule(0.5, 0.0, 2);
        assert_eq!(r2.keep(&[0.1, 0.99, 0.99], 3, false), 2);
        let r0 = rule(0.5, 0.0, 0);
        assert_eq!(r0.keep(&[0.1, 0.99], 2, false), 0);
        // dmin beyond k: k
        let r9 = rule(0.5, 0.0, 9);
        assert_eq!(r9.keep(&[0.1, 0.1], 2, false), 2);
    }

    #[test]
    fn short_or_bad_tops() {
        let r = rule(0.5, 0.5, 1);
        // missing probabilities: no cut
        assert_eq!(r.keep(&[0.1], 3, false), 3);
        // NaN cuts
        assert_eq!(r.keep(&[0.9, f32::NAN, 0.9], 3, true), 1);
        assert_eq!(r.keep(&[], 0, false), 0);
    }

    #[test]
    fn keep_is_monotone_in_threshold() {
        let tops = [0.97, 0.8, 0.95, 0.6, 0.99, 0.7, 0.9];
        let mut last = usize::MAX;
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let k = rule(t, 0.0, 1).keep(&tops, tops.len(), false);
            assert!(k <= last, "threshold {t}: {k} > {last}");
            assert!(k >= 1);
            last = k;
        }
    }
}
