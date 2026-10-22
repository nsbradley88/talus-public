//! Speculative decoding controller (`SuffixDrafter`, `DraftPolicy`) and `parse_logit_bias`
//! for Flash-Next (`qwen4exp`), ported 1:1 from:
//! - `src/spec/suffix_drafter.cpp`
//! - `src/spec/draft_policy.cpp`
//! - `src/spec/draft_source.cpp`
//! - `include/strata/core/logit_bias.hpp`

pub use talus_spec::{
    parse_logit_bias, CrossoverTable, DfCtx, DraftPick, DraftPolicy, DrafterPick, ModelSpecDecl,
    SpecEngineService, SuffixDrafter,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_drafter_matches_cpp_behavior() {
        let mut d = SuffixDrafter::new(4, 64, 1024);
        // Sequence: [10, 20, 30, 40, 50, 60, 70, 99, 10, 20, 30, 40]
        d.append_slice(&[10, 20, 30, 40, 50, 60, 70, 99, 10, 20, 30, 40]);
        let mut out = [0i32; 8];
        let k = d.propose(3, &mut out);
        assert_eq!(k, 3);
        assert_eq!(d.last_match(), 4);
        assert_eq!(&out[..3], &[50, 60, 70]);

        // propose_after without mutating history
        let mut out2 = [0i32; 8];
        let k2 = d.propose_after(&[50], 2, &mut out2);
        assert_eq!(k2, 2);
        assert_eq!(d.last_match(), 5);
        assert_eq!(&out2[..2], &[60, 70]);
        assert_eq!(d.history().len(), 12);
    }

    #[test]
    fn draft_policy_matches_cpp_behavior() {
        let mut pol = DraftPolicy::new(4, 0.05);
        assert_eq!(DraftPolicy::bucket(4), 0);
        assert_eq!(DraftPolicy::bucket(8), 1);
        assert_eq!(DraftPolicy::bucket(16), 2);
        assert_eq!(DraftPolicy::bucket(32), 3);

        // Initial probe on long match (>24 -> q = 0.96 >= 0.85) should probe full lookup window
        let pk = pol.choose(2, 3, 28);
        assert!(pk.lookup);
        assert_eq!(pk.t, 4);

        pol.observe(true, 4, 3, 28, 12.0);
        pol.observe(false, 2, 1, 0, 8.0);
        assert!(pol.cost_ms(4) > 0.0);
        assert!(pol.lookup_rate(28) > 0.9);
    }

    #[test]
    fn parse_logit_bias_matches_cpp_contract() {
        assert!(parse_logit_bias("", 100).unwrap().is_empty());
        assert!(parse_logit_bias("5:0,10:0.0", 100).unwrap().is_empty());

        let b = parse_logit_bias("3:-1.5,7:200.0,9:-1e12", 16).unwrap();
        assert_eq!(b.len(), 16);
        assert!((b[3] - (-1.5)).abs() < 1e-6);
        assert!((b[7] - 100.0).abs() < 1e-6);
        assert!((b[9] - (-1e9)).abs() < 1.0);

        assert!(parse_logit_bias("16:1.0", 16).is_err());
        assert!(parse_logit_bias("-1:1.0", 16).is_err());
        assert!(parse_logit_bias("abc", 16).is_err());
    }
}
