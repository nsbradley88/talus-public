//! Speculative decoding drafters (`NgramMod`, `DFlash`), draft vocabulary (`DraftVocab`),
//! adaptive draft policy (`DraftLen`), Wave 2 speculation traits, and exact CPU/rejection
//! sampling (`Mt19937_64`, `SampCfg`, `sample`, `make_dist`, `draw_dist`, `decide_sampled`,
//! `sample_selftest`).

#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::collapsible_if,
    clippy::manual_div_ceil,
    clippy::missing_safety_doc,
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::too_many_arguments
)]

pub const CRATE_NAME: &str = "talus-spec";

pub mod adraft;
pub mod copy;
pub mod dflash;
pub mod draft_vocab;
pub mod ngram;
pub mod pause;
pub mod prompt_lookup;
pub mod sampling;
pub mod service;
pub mod ssamp;
pub mod traits;

pub use adraft::{
    AdraftMode, DFlashStopMode, DFlashStoppingPolicy, DraftLen, DEFAULT_EMA_ALPHA,
    INITIAL_EMA_RATE, K_MAX, MAX_HIST_DRAFTS, THRESH_DOWNSHIFT, THRESH_DOWNSHIFT_FAST,
    THRESH_DOWNSHIFT_MILD, THRESH_UPSHIFT,
};
pub use copy::{
    copy_cfg_from_env, copy_enabled, copy_env_f64, CopyCfg, CopyMatch, CopyProposer, CopyStats,
    COPY_KEY,
};
pub use dflash::{
    DFlash, DFlashDraftFn, DFlashDrafterSpec, DFlashDynConv, DFlashFamily, DFlashHeadMode,
    DFlashLayer, DFlashModelProfile, DFlashSamp, DFlashTargetInfo, DFlashTargetView, DFlashWeight,
    DFLASH_TAPS, DFLASH_TOP_K,
};
pub use draft_vocab::{DraftVocab, DVOC_MAGIC, DVOC_VERSION};
pub use ngram::{NgramConfig, NgramMod, NgramQ36, Q36Ngram};
pub use pause::{
    round_speedup_ratio, AdaptivePauseMetric, AdaptivePausePolicy, NgramEditProfile,
    NgramProfileKind, SpecWorkloadProfile, DEFAULT_COST_PER_DRAFT, DEFAULT_MAX_BACKOFF,
    DEFAULT_MIN_BACKOFF, DEFAULT_PAUSE_EMA_DECAY, DEFAULT_PAUSE_RATIO,
};
pub use prompt_lookup::{
    choose_draft_len, fnv1a_ngram_key, fnv1a_ngram_key_u32, update_acc_est_ema,
    PromptLookupDrafter, DEFAULT_DRAFT_COST, DEFAULT_MAX_K, DEFAULT_MIN_GAIN, DEFAULT_MIN_MATCH,
    FNV1A_OFFSET_BASIS, FNV1A_PRIME, INITIAL_ACC_EST,
};
pub use sampling::{
    decide_greedy, decide_sampled, draw_dist, draw_dist_sparse, generate_canonical_f64,
    greedy_argmax, make_dist, make_dist_sparse, sample, sample_from_logits, sample_selftest,
    select_top_k_cands, uniform_01, Cand, DraftDist, Mt19937_64, SampCfg, Sampling, SamplingParams,
};
pub use service::{
    parse_logit_bias, CrossoverTable, DfCtx, DraftPick, DraftPolicy, DrafterPick, ModelSpecDecl,
    SpecEngineService, SuffixDrafter,
};
pub use traits::{
    Accept, DeviceDrafter, HostDrafter, RoundExecutor, RoundIo, RoundOutcome, RoundPlan,
    RoundStats, SpecController, SpecRoundOutcome,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch_dir() -> PathBuf {
        if let Ok(t) = std::env::var("TMPDIR") {
            if !t.is_empty() && !t.starts_with("/tmp") {
                let p = PathBuf::from(t);
                let _ = std::fs::create_dir_all(&p);
                return p;
            }
        }
        let p = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/test-scratch"
        ));
        let _ = std::fs::create_dir_all(&p);
        p
    }

    #[test]
    fn crate_identity() {
        assert_eq!(CRATE_NAME, "talus-spec");
    }

    #[test]
    fn test_mt19937_64_standard_vector() {
        // C++11 standard requirement for std::mt19937_64:
        // After default-seeding with 5489, the 10,000th consecutive invocation produces
        // 9981545732273789042ULL.
        let mut rng = Mt19937_64::default();
        let mut val = 0u64;
        for _ in 0..10_000 {
            val = rng.next_u64();
        }
        assert_eq!(val, 9_981_545_732_273_789_042u64);
    }

    #[test]
    fn test_mt19937_64_uniform_01() {
        let mut rng = Mt19937_64::new(123456789);
        for _ in 0..1_000 {
            let u = rng.uniform_01();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn test_sample_selftest_passes() {
        assert!(
            sample_selftest(),
            "sample_selftest() must pass all 4 self-tests"
        );
    }

    #[test]
    fn test_make_dist_and_decide_sampled_edge_cases() {
        // Degenerate single-token after top_p / min_p filtering returns token ID >= 0.
        let logits = [10.0f32, -20.0, -20.0, -20.0];
        let cfg = SampCfg {
            temp: 1.0,
            top_k: 4,
            top_p: 0.5,
            min_p: 0.2,
            ..SampCfg::default()
        };
        let mut dist = [0.0f32; 4];
        let rc = make_dist(&logits, 4, &cfg, &[], &mut dist);
        assert_eq!(rc, 0);
        assert!((dist[0] - 1.0).abs() < 1e-6);
        assert_eq!(dist[1], 0.0);

        // Speculative acceptance when all draft tokens match greedy target winner.
        let target_logits = [
            0.0f32, 5.0, 1.0, 0.0, // step 0 -> winner 1
            0.0, 1.0, 6.0, 0.0, // step 1 -> winner 2
            0.0, 0.0, 1.0, 7.0, // bonus -> winner 3
        ];
        let drafts = [1i32, 2];
        let greedy_cfg = SampCfg::default();
        let mut recent = vec![0i32];
        let mut rng = Mt19937_64::new(7);
        let mut accepted = 0usize;
        let bonus = decide_sampled(
            &target_logits,
            4,
            &drafts,
            2,
            &greedy_cfg,
            &mut recent,
            &mut rng,
            &mut accepted,
            None,
        );
        assert_eq!(accepted, 2);
        assert_eq!(bonus, 3);
        assert_eq!(recent, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_ngram_add_draft_and_reset() {
        let mut ng = NgramMod::new();
        ng.init();

        // Fewer than N_NGRAM (3) tokens in history -> returns 0 drafts.
        let mut out = [0i32; 4];
        assert_eq!(ng.draft(&[10, 20], &mut out, 4, 100), 0);

        // Feed sequence [10, 20, 30, 40, 50, 60] via 4-grams (N_NGRAM + 1 = 4)
        let seq = [10i32, 20, 30, 40, 50, 60];
        ng.add_n(&seq, seq.len());

        // History ending with [10, 20, 30] should chain-draft [40, 50, 60]
        let n = ng.draft(&[5, 10, 20, 30], &mut out, 4, 100);
        assert_eq!(n, 3);
        assert_eq!(&out[..3], &[40, 50, 60]);

        // Out-of-range token in table (>= n_vocab) stops drafting.
        ng.add_n(&[40, 50, 60, 150], 4);
        let n2 = ng.draft(&[10, 20, 30], &mut out, 4, 100);
        assert_eq!(n2, 3);

        // HostDrafter trait integration
        let mut vec_out = Vec::new();
        HostDrafter::propose(&mut ng, &[10u32, 20, 30], 2, &mut vec_out);
        assert_eq!(vec_out, vec![40u32, 50]);

        // Reset clears table
        ng.reset();
        assert_eq!(ng.draft(&[10, 20, 30], &mut out, 4, 100), 0);
    }

    #[test]
    fn test_draft_vocab_binary_parse_and_inverse_map() {
        let ids = vec![10i32, 42, 7, 99];
        let expected_vocab = 128i32;
        let dv = DraftVocab::from_ids(&ids, expected_vocab).unwrap();

        assert_eq!(dv.n_draft, 4);
        assert_eq!(dv.n_vocab, 128);
        assert_eq!(dv.to_full(0), 10);
        assert_eq!(dv.to_full(1), 42);
        assert_eq!(dv.to_full(2), 7);
        assert_eq!(dv.to_full(3), 99);
        assert_eq!(dv.to_full(4), -1);

        assert_eq!(dv.to_draft(10), 0);
        assert_eq!(dv.to_draft(42), 1);
        assert_eq!(dv.to_draft(7), 2);
        assert_eq!(dv.to_draft(99), 3);
        assert_eq!(dv.to_draft(0), -1);
        assert_eq!(dv.to_draft(128), -1);

        // Round-trip through DVOC binary format on disk in scratch directory
        let bytes = dv.to_bytes();
        assert_eq!(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            DVOC_MAGIC
        );
        let path = scratch_dir().join("test_dvoc_unit.bin");
        std::fs::write(&path, &bytes).unwrap();
        let loaded = DraftVocab::load(&path, expected_vocab).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.ids, ids);
        assert_eq!(loaded.dv_inv, dv.dv_inv);

        // Mismatched expected_vocab must error
        assert!(DraftVocab::from_bytes(&bytes, 256).is_err());
    }

    #[test]
    fn test_adraft_step_up_down_and_reset() {
        let mut dl = DraftLen::new();
        dl.init_with_alpha(4, true, 0.5);
        dl.n_max = 8;
        assert_eq!(dl.choose(), 4);

        // High acceptance (4/4 = 1.0) -> ema = 0.5 * 0.6 + 0.5 * 1.0 = 0.80 >= 0.72 -> upshift to 5
        dl.observe(4, 4);
        assert_eq!(dl.n_cur, 5);
        assert_eq!(dl.upshifts, 1);
        assert_eq!(dl.hist[4], 1);

        // Very low acceptance (0/5 = 0.0) twice -> drops below 0.30 -> downshift by 2
        dl.observe(5, 0); // ema = 0.40 < 0.50 -> n_cur = 4
        assert_eq!(dl.n_cur, 4);
        dl.observe(4, 0); // ema = 0.20 < 0.30 -> n_cur = 2
        assert_eq!(dl.n_cur, 2);
        assert_eq!(dl.downshifts, 2);

        // Reset request restores n_cur and ema_rate
        dl.n_max = 4;
        dl.reset_request();
        assert_eq!(dl.n_cur, 4);
        assert!((dl.ema_rate - 0.60).abs() < 1e-6);

        // SpecController trait integration
        dl.n_max = 8;
        let plan = dl.plan(&RoundStats {
            proposed: 4,
            accepted: 4,
            round_latency_us: 1000,
            ema_accept_rate: 0.9,
            host_draft_available: false,
            consecutive_zero_accepts: 0,
        });
        assert_eq!(plan.draft_depth, 5);
    }

    #[test]
    fn test_adraft_confidence_and_dflash_stop_policy() {
        let mut dl = DraftLen::new();
        dl.init_arch("gfx1151:sramecc+", 4);
        assert_eq!(dl.mode(), 2);
        assert!(dl.want_conf());
        assert!(dl.stop_mode());
        assert_eq!(dl.dmax(), 7);
        assert_eq!(dl.dmax_for(false), 7);
        assert!(dl.dfl_stop());
        assert!(dl.dfl_sample());

        // High confidence keep -> keeps full chain_len (3)
        assert_eq!(dl.keep(&[0.95f32, 0.90, 0.85], 3, false), 3);
        // Low confidence at step 1 -> truncates to dmin (2)
        assert_eq!(dl.keep(&[0.90f32, 0.10, 0.10], 3, false), 2);
        // DFlash cumulative confidence stop
        assert_eq!(dl.keep_dfl(&[0.8f32, 0.8, 0.8, 0.1], 4, false), 3);
        assert!(!dl.describe().is_empty());
    }

    #[test]
    fn test_ngram_q36_rolling_hash() {
        let mut ng = NgramMod::new();
        let ctx: Vec<i32> = (1..=32).collect();
        ng.update(&ctx);
        let suffix: Vec<i32> = (1..=28).collect();
        let proposed = ng.propose_q36(&suffix, 4, 2);
        assert_eq!(proposed, vec![29, 30, 31, 32]);
    }

    #[test]
    fn test_prompt_lookup_drafter_laguna_parity_and_min_match() {
        // Verify exact FNV-1a key computation matching laguna_serve.cpp
        let trigram = [101i32, 202, 303];
        let mut expected_h = 1_469_598_103_934_665_603u64;
        for &t in &trigram {
            expected_h = (expected_h ^ (t as u32 as u64)).wrapping_mul(1_099_511_628_211u64);
        }
        expected_h ^= 3u64;
        assert_eq!(fnv1a_ngram_key(&trigram), expected_h);

        let mut drafter = PromptLookupDrafter::new();
        assert_eq!(drafter.min_match, 8);

        // Prompt containing a 10-token span [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        // followed by continuation [100, 101, 102, 103, 104]
        let mut prompt: Vec<i32> = vec![1, 2, 3];
        prompt.extend(10..=19); // 10 tokens
        prompt.extend([100, 101, 102, 103, 104]);
        prompt.extend([50, 51, 52]);
        // Append the 8-token suffix [12, 13, 14, 15, 16, 17, 18, 19] (>= min_match = 8)
        prompt.extend(12..=19);
        drafter.reset(&prompt);

        let mut out = [0i32; 7];
        let nd = drafter.draft(5, &mut out, 8);
        assert_eq!(nd, 5);
        assert_eq!(&out[..5], &[100, 101, 102, 103, 104]);

        // Now test min_match rejection: only 5 tokens match before the 3-gram [17, 18, 19],
        // preceded by a mismatch token 999 -> match_len = 5 < min_match (8), must return 0!
        let mut short_suffix_seq: Vec<i32> = vec![1, 2, 3];
        short_suffix_seq.extend(10..=19);
        short_suffix_seq.extend([100, 101, 102]);
        short_suffix_seq.push(999); // breaks suffix match at length 5
        short_suffix_seq.extend(15..=19); // only 5 tokens [15, 16, 17, 18, 19] match
        drafter.reset(&short_suffix_seq);
        assert_eq!(drafter.draft(4, &mut out, 8), 0);
        // With min_match = 5, the same sequence succeeds:
        assert_eq!(drafter.draft(3, &mut out, 5), 3);
        assert_eq!(&out[..3], &[100, 101, 102]);
    }

    #[test]
    fn test_prompt_lookup_choose_draft_len_and_ema_acc_est() {
        let mut drafter = PromptLookupDrafter::new();
        assert!((drafter.acc_est - 0.90).abs() < 1e-12);

        // At acc_est = 0.90, draft_cost = 0.40, min_gain = 1.03:
        // kk=1: ex = 1.90, r = 1.90 / 1.40 = 1.3571
        // kk=2: ex = 2.71, r = 2.71 / 1.80 = 1.5056
        // kk=3: ex = 3.439, r = 3.439 / 2.20 = 1.5632 (peak!)
        // kk=4: ex = 4.0951, r = 4.0951 / 2.60 = 1.5750 (peak at kk=4!)
        // kk=5: ex = 4.68559, r = 4.68559 / 3.00 = 1.5619
        let k_initial = choose_draft_len(7, 100, 0.90, 0.40, 1.03);
        assert_eq!(k_initial, 4);
        assert_eq!(drafter.choose_k(7, 2), 2); // room-clamped

        // Near-perfect acceptance (acc_est = 0.9995 >= 0.999) selects max_k = 7
        assert_eq!(choose_draft_len(7, 100, 0.9995, 0.40, 1.03), 7);

        // Very low acceptance (acc_est = 0.20) yields ex=1.20/1.40 = 0.857 < 1.03 -> 0 drafts
        assert_eq!(choose_draft_len(7, 100, 0.20, 0.40, 1.03), 0);

        // EMA update when 4 drafted, 1 accepted:
        // step 0 (acc): 0.9 * 0.9 + 0.1 = 0.91
        // step 1 (rej): 0.9 * 0.91 = 0.819
        // step 2 (rej): 0.9 * 0.819 = 0.7371
        // step 3 (rej): 0.9 * 0.7371 = 0.66339
        let updated = drafter.update_acc_est(4, 1);
        assert!((updated - 0.66339).abs() < 1e-9);

        // When no draft is proposed (nd == 0), drifts toward 0.90:
        // acc_est += 0.05 * (0.9 - acc_est)
        let drifted = drafter.update_acc_est(0, 0);
        let expected_drift = 0.66339 + 0.05 * (0.90 - 0.66339);
        assert!((drifted - expected_drift).abs() < 1e-9);

        // HostDrafter trait propose + observe round-trip
        let mut seq: Vec<u32> = (100..112).collect(); // 12 tokens
        seq.extend([200u32, 201, 202, 203]);
        seq.extend(104..112); // 8-token suffix matching 104..112
        let mut proposed = Vec::new();
        HostDrafter::propose(&mut drafter, &seq, 4, &mut proposed);
        assert!(!proposed.is_empty());
        assert_eq!(proposed[0], 200);
        HostDrafter::observe(&mut drafter, &[200, 201, 999]);
        assert!(drafter.hist.ends_with(&[200, 201, 999]));
    }

    #[test]
    fn test_adaptive_pause_policy_1_2_and_ngram_edit_profile() {
        let mut policy = AdaptivePausePolicy::new(1.2);
        assert!(!policy.is_paused());
        assert_eq!(policy.backoff, 8);

        // Round with 0 accepted out of 4 proposed drops EMA below 1.2 -> triggers pause!
        let paused = policy.observe_round(4, 0);
        assert!(paused);
        assert!(policy.is_paused());
        assert_eq!(policy.cooldown_remaining, 8);
        assert_eq!(policy.backoff, 16);
        assert_eq!(policy.pause_count, 1);

        // Coordinate with NgramEditProfile::edit(): while device drafter is paused,
        // if host draft is available, RoundPlan falls back to host drafter!
        let edit_prof = NgramEditProfile::edit();
        let plan_with_host = edit_prof.plan_with_pause(
            &mut policy,
            &RoundStats {
                proposed: 0,
                accepted: 0,
                round_latency_us: 500,
                ema_accept_rate: 0.2,
                host_draft_available: true,
                consecutive_zero_accepts: 1,
            },
        );
        assert!(plan_with_host.pause_device_drafter);
        assert!(plan_with_host.prefer_host_draft);
        assert_eq!(plan_with_host.draft_depth, edit_prof.k_max as u32);
        assert_eq!(policy.cooldown_remaining, 7);

        // If a host copy draft achieves strong acceptance (4 accepted of 4 proposed) during cooldown,
        // AdaptivePausePolicy recovers immediately!
        let still_paused = policy.observe_round(4, 4);
        assert!(!still_paused);
        assert!(!policy.is_paused());
        assert_eq!(policy.cooldown_remaining, 0);
        assert_eq!(policy.resume_count, 1);

        // Trigger another pause and let the full cooldown expire via single-token decode steps
        assert!(policy.observe_round(4, 0));
        let cd = policy.cooldown_remaining;
        for rem in (1..=cd).rev() {
            assert_eq!(policy.cooldown_remaining, rem);
            policy.observe_round(0, 0);
        }
        assert!(!policy.is_paused());
        assert_eq!(policy.resume_count, 2);
    }

    #[test]
    fn test_dflash_model_profiles_qwen_laguna_mimo() {
        // 1. Qwen3.8-27B DFlash2
        let qwen = DFlashModelProfile::qwen38_27b_dflash2();
        assert_eq!(qwen.family, DFlashFamily::Qwen27BDFlash2);
        assert_eq!(qwen.n_taps, 5);
        assert_eq!(qwen.block_size, 16);
        assert_eq!(qwen.max_draft, 15);
        assert_eq!(qwen.draft_vocab_size, Some(65_536));
        assert_eq!(qwen.default_draft_vocab_file, Some("qwen-dv-65536.bin"));
        assert!(qwen.is_dflash2);

        // 2. Laguna-S-2.1 DFlash (Phase C2)
        let laguna = DFlashModelProfile::laguna_s_2_1_dflash();
        assert_eq!(laguna.family, DFlashFamily::LagunaDFlash);
        assert_eq!(laguna.swa_window, 1024);
        assert_eq!(laguna.target_vocab, 100_352);
        assert_eq!(laguna.draft_vocab_size, Some(65_536));
        assert!((laguna.adaptive_pause_ratio - 1.2).abs() < 1e-6);
        assert_eq!(laguna.edit_profile.kind, SpecWorkloadProfile::Edit);
        assert!(laguna.has_attn_gate_softplus);

        // Laguna SWA window enforcement (1024 tokens)
        let mut laguna_spec = DFlashDrafterSpec::laguna_s_2_1();
        let long_ctx: Vec<u32> = (0..1500).map(|i| i % 50_000).collect();
        laguna_spec.push_context(&long_ctx);
        assert_eq!(laguna_spec.swa_len(), 1024);
        assert_eq!(laguna_spec.total_committed_tokens, 1500);
        assert_eq!(laguna_spec.swa_tokens[0], 1500 - 1024);

        // 3. MiMo-V2-Flash DFlash (Phase H)
        let mimo = DFlashModelProfile::mimo_v2_flash_dflash();
        assert_eq!(mimo.family, DFlashFamily::MimoDFlash);
        assert_eq!(mimo.target_vocab, 152_064);
        assert_eq!(
            mimo.head_mode,
            DFlashHeadMode::DedicatedQ8_0Head {
                head_vocab: 152_064
            }
        );
        assert_eq!(mimo.draft_vocab_size, Some(65_536));
        assert_eq!(mimo.default_draft_vocab_file, Some("mimo-dv-65536.bin"));
        assert_eq!(mimo.max_draft, 4);
        assert_eq!(mimo.block_size, 5);
        assert!(mimo.has_attn_sinks);

        let mimo_spec = DFlashDrafterSpec::mimo_v2_flash();
        assert_eq!(DeviceDrafter::vocab(&mimo_spec).unwrap().len(), 65_536);
    }

    #[test]
    fn test_round_executor_end_to_end_greedy_and_rejection_invariance() {
        // 1. End-to-end Laguna DFlash + AdaptivePausePolicy(1.2) + NgramEditProfile::edit() greedy rounds
        let n_vocab = 256usize;
        let mut laguna_profile = DFlashModelProfile::laguna_s_2_1_dflash();
        laguna_profile.target_vocab = n_vocab;
        laguna_profile.head_vocab = n_vocab;
        laguna_profile.draft_vocab_size = Some(128);

        // Device drafter proposes [10, 11, 12, 13]
        let mut drafter = DFlashDrafterSpec::new(laguna_profile)
            .with_draft_callback(|_prof, _swa, k, _dv| (0..k).map(|i| (10 + i) as u32).collect());

        let mut exec = RoundExecutor::new(n_vocab, 4);
        // Seed prompt with a verbatim repeatable block [20..28] -> [40, 41, 42, 43]
        let mut prompt: Vec<u32> = (20..28).collect();
        prompt.extend([40, 41, 42, 43]);
        prompt.extend([1, 2, 3]);
        exec.seed_prompt(&prompt);
        drafter.push_context(&prompt);

        // Round 1: Target greedily wants [10, 11, 12, 13, 14] -> all 4 device drafts accepted + bonus 14!
        let r1 = exec
            .run_dflash_round(&mut drafter, Accept::Greedy, |_hist, drafts| {
                let rows = drafts.len() + 1;
                let mut logits = vec![-10.0f32; rows * n_vocab];
                for r in 0..rows {
                    logits[r * n_vocab + (10 + r)] = 20.0;
                }
                logits
            })
            .unwrap();
        assert_eq!(r1.proposed, 4);
        assert_eq!(r1.accepted, 4);
        assert_eq!(r1.bonus_token, 14);
        assert_eq!(r1.committed_tokens, vec![10, 11, 12, 13, 14]);
        assert!(!r1.used_host_drafter);

        // Round 2 & 3: Target rejects all device drafts (wants token 90) -> triggers AdaptivePausePolicy (1.2)!
        for step in 0..2 {
            let r_bad = exec
                .run_dflash_round(&mut drafter, Accept::Greedy, |_hist, drafts| {
                    let rows = drafts.len() + 1;
                    let mut logits = vec![-10.0f32; rows * n_vocab];
                    for r in 0..rows {
                        logits[r * n_vocab + (20 + step)] = 20.0;
                    }
                    logits
                })
                .unwrap();
            assert_eq!(r_bad.accepted, 0);
        }
        assert!(drafter.pause_policy.is_paused());

        // Feed the remaining suffix of [20..28] so PromptLookupDrafter has an 8-token suffix match
        let suffix_rest: Vec<u32> = (22..28).collect();
        drafter.push_context(&suffix_rest);
        exec.history.extend_from_slice(&suffix_rest);

        // Round 4: Device drafter is paused, so RoundExecutor falls back to HostDrafter (PromptLookupDrafter),
        // which proposes [40, 41, 42, 43] and is fully accepted!
        let r_host = exec
            .run_dflash_round(&mut drafter, Accept::Greedy, |_hist, drafts| {
                let rows = drafts.len() + 1;
                let mut logits = vec![-10.0f32; rows * n_vocab];
                for r in 0..rows {
                    logits[r * n_vocab + (40 + r)] = 20.0;
                }
                logits
            })
            .unwrap();
        assert!(r_host.used_host_drafter);
        assert_eq!(r_host.accepted, 4);
        assert_eq!(&r_host.committed_tokens[..4], &[40, 41, 42, 43]);

        // 2. Rejection sampling distribution invariance test through RoundExecutor::verify_logits:
        // Verify that speculative rejection sampling over wrong/arbitrary draft tokens reproduces
        // the exact target categorical distribution!
        let v_small = 4usize;
        let row_logits = [1.4f32, 0.9, 0.2, -0.7];
        let samp_cfg = SampCfg {
            temp: 1.0,
            top_k: 4,
            top_p: 1.0,
            min_p: 0.0,
            ..SampCfg::default()
        };
        let mut target_dist = [0.0f32; 4];
        assert_eq!(
            make_dist(&row_logits, 4, &samp_cfg, &[], &mut target_dist),
            -2
        );

        let mut two_row_logits = Vec::with_capacity(8);
        two_row_logits.extend_from_slice(&row_logits);
        two_row_logits.extend_from_slice(&row_logits);

        let mut s_exec = RoundExecutor::new(v_small, 1)
            .with_sampling(samp_cfg)
            .with_seed(202505);
        let mut counts = [0usize; 4];
        let trials = 12_000usize;
        for t in 0..trials {
            let draft_tok = ((t * 3 + 1) % v_small) as i32;
            let out = s_exec
                .verify_logits(
                    &[draft_tok],
                    &two_row_logits,
                    Accept::Rejection { seed: 202505 },
                    None,
                )
                .unwrap();
            let first_emitted = out.committed_tokens[0] as usize;
            counts[first_emitted] += 1;
        }
        for i in 0..v_small {
            let emp = (counts[i] as f32) / (trials as f32);
            assert!(
                (emp - target_dist[i]).abs() < 0.025,
                "token {i}: empirical {emp:.4} vs target {:.4}",
                target_dist[i]
            );
        }
    }

    #[test]
    fn test_select_top_k_cands_bit_exact_vs_full_sort() {
        let mut rng = Mt19937_64::new(987654);
        let n = 4096usize;
        let mut logits = Vec::with_capacity(n);
        for i in 0..n {
            // Mix random logits with intentional exact ties to test ascending-id tie breaking
            let v = if i % 17 == 0 {
                3.5f32
            } else {
                ((rng.uniform_01() * 20.0) - 10.0) as f32
            };
            logits.push(v);
        }
        let full = select_top_k_cands(&logits, n, n);
        for &k in &[1usize, 8, 20, 40, 256, 512] {
            let top_k = select_top_k_cands(&logits, n, k);
            assert_eq!(top_k.len(), k);
            assert_eq!(&top_k[..], &full[..k]);
        }
    }

    #[test]
    fn test_spec_engine_service_wp5_models_and_host_propose() {
        for decl in [
            ModelSpecDecl::qwen35_27b(),
            ModelSpecDecl::qwen35_35b_a3b(),
            ModelSpecDecl::laguna_s_2_1(),
            ModelSpecDecl::mimo_v2_flash(),
            ModelSpecDecl::glm_4_7_flash(),
        ] {
            let mut svc = SpecEngineService::new(decl.clone());
            let mut prompt: Vec<i32> = (10..30).collect();
            prompt.extend([500, 501, 502, 503, 504]);
            prompt.extend(20..30); // 10-token suffix matching 20..30
            svc.reset_request(&prompt);

            let mut out = [0i32; 8];
            let (nd, match_len) = svc.propose_host(4, 10, 5, &mut out);
            assert_eq!(
                nd, 4,
                "model {} should propose 4 host tokens",
                decl.model_name
            );
            assert!(match_len >= 5);
            assert_eq!(&out[..4], &[500, 501, 502, 503]);

            svc.observe_round(true, 4, 4, match_len, 10.0);
            svc.commit_tokens(&[500, 501, 502, 503, 504]);
        }
    }

    #[test]
    fn test_mimo_adaptive_pause_does_not_lock_out_code_after_prose_intro() {
        let mut svc = SpecEngineService::from_decl(ModelSpecDecl::mimo(4, true));
        svc.reset(&[1, 2, 3, 4]);

        // Brief prose intro with mixed 0/1 acceptance must not lock out DFlash.
        for &acc in &[1usize, 0, 1, 0, 1] {
            assert_eq!(svc.select_draft_len(10, true), 4);
            svc.record_verify(4, acc);
        }
        assert_eq!(svc.select_draft_len(30, true), 4);

        // Even after a 6-round zero-acceptance streak triggers a pause, cooldown expiry
        // gives probe headroom so a single zero-accept probe round does not cascade to 256-step lockout.
        for _ in 0..6 {
            if svc.select_draft_len(40, true) > 0 {
                svc.record_verify(4, 0);
            } else {
                svc.record_verify(0, 0);
            }
        }
        while svc.select_draft_len(50, true) == 0 {
            svc.record_verify(0, 0);
        }
        // First probe round after cooldown with 0 accepted should still allow a second probe round!
        svc.record_verify(4, 0);
        assert_eq!(svc.select_draft_len(60, true), 4);
    }
}
