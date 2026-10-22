//! Laguna-S-2.1 (`laguna`) architecture adapter (`src/laguna/*`, Phase C1).
//!
//! Provides a bitwise-faithful Rust host implementation of the Laguna-S-2.1
//! hybrid 48-layer (12 YaRN full-attention + 36 SWA, layer 0 dense FFN + layers 1..47
//! 256-expert top-10 MoE with shared expert) inference engine, weight loader,
//! `TALUSKV2` disk session manager, prompt-lookup speculative drafter, and stdio server.

pub mod engine;
pub mod ffi;
pub mod serve;
pub mod session;
#[cfg(test)]
mod ssamp_tests;
pub mod weights;

pub use engine::dflash::LagunaDFlash;
pub use engine::{DumpFn, Engine, Options, K_MAX_VERIFY, TAP_LAYERS};
pub use ffi::{
    load_talus_kernels, tk_build_rope_table, HipRuntime, RopeScalingParams, TkContext,
    TK_ROPE_SCALING_LINEAR, TK_ROPE_SCALING_NONE, TK_ROPE_SCALING_YARN,
};
pub use serve::{
    kDraftCost, run_cli_ubatch, sample, serve_main, Drafter, LagunaCliConfig, Mt19937_64, Sampling,
    DEFAULT_EOS_IDS, K_DRAFT_COST,
};
pub use session::{compute_ctx16_and_ring, session_payload_bytes, SessionHeader, SESSION_MAGIC};
pub use weights::{
    compute_model_id, dequant_embd_row, is_full, kEmbd, kEps, kExpScale, kExperts, kFfDense,
    kFfExp, kHeadDim, kHeadKv, kLayers, kSwa, kTopK, kVocab, ks_index, n_head, to_tk_quant_type,
    vs_index, vt_index, DevTensor, LagunaCatalog, LagunaModel, LayerWeights, K_EMBD, K_EPS,
    K_EXPERTS, K_EXP_SCALE, K_FF_DENSE, K_FF_EXP, K_HEAD_DIM, K_HEAD_KV, K_LAYERS, K_SWA, K_TOP_K,
    K_VOCAB, STAGE_BYTES,
};

#[cfg(test)]
mod tests {
    use super::*;
    use talus_core_api::QuantType;
    use talus_gguf::GgufFile;
    use talus_kernels::f32_to_f16;

    #[test]
    fn shape_constants_and_layer_predicates_match_cpp() {
        assert_eq!(K_EMBD, 3072);
        assert_eq!(K_VOCAB, 100352);
        assert_eq!(K_LAYERS, 48);
        assert_eq!(K_HEAD_KV, 8);
        assert_eq!(K_HEAD_DIM, 128);
        assert_eq!(K_SWA, 512);
        assert_eq!(K_EXPERTS, 256);
        assert_eq!(K_TOP_K, 10);
        assert_eq!(K_FF_EXP, 1024);
        assert_eq!(K_FF_DENSE, 12288);
        assert_eq!(K_EPS, 1e-6_f32);
        assert_eq!(K_EXP_SCALE, 2.5_f64);

        let full_layers: Vec<usize> = (0..K_LAYERS).filter(|&il| is_full(il)).collect();
        assert_eq!(
            full_layers,
            vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
        );
        for il in 0..K_LAYERS {
            if il % 4 == 0 {
                assert!(is_full(il));
                assert_eq!(n_head(il), 48);
            } else {
                assert!(!is_full(il));
                assert_eq!(n_head(il), 72);
            }
        }
    }

    #[test]
    fn ctx16_and_ring_calculations_match_cpp() {
        // Default ctx=4096, ubatch=512:
        // ctx16 = ((4096 + 15) & !15) + 64 = 4160
        // ring = min(4160, ((512 + 512 + 64) + 15) & !15) = 1088
        let (ctx16, ring) = compute_ctx16_and_ring(4096, 512);
        assert_eq!(ctx16, 4160);
        assert_eq!(ring, 1088);

        // Small ctx=500, ubatch=512:
        // ctx16 = ((500 + 15) & !15) + 64 = 512 + 64 = 576
        // ring = min(576, 1088) = 576
        let (ctx16_small, ring_small) = compute_ctx16_and_ring(500, 512);
        assert_eq!(ctx16_small, 576);
        assert_eq!(ring_small, 576);
    }

    #[test]
    fn vt_vs_ks_index_math_matches_cpp() {
        let stride = 1088usize;
        // Within a 16-position tile, consecutive positions at fixed (kvh, d) must have stride 1.
        for kvh in 0..K_HEAD_KV {
            for d in [0usize, 1, 63, 127] {
                let base = vt_index(kvh, 32, d, stride);
                for off in 0..16 {
                    assert_eq!(vt_index(kvh, 32 + off, d, stride), base + off);
                }
                // Next dim d+1 in the same 16-pos tile is +16 elements away.
                if d + 1 < K_HEAD_DIM {
                    assert_eq!(vt_index(kvh, 32, d + 1, stride), base + 16);
                }
            }
            for blk in 0..4 {
                let sbase = vs_index(kvh, 48, blk, stride);
                for off in 0..16 {
                    assert_eq!(vs_index(kvh, 48 + off, blk, stride), sbase + off);
                }
            }
        }
        assert_eq!(ks_index(0, 0, 0, 8), 0);
        assert_eq!(ks_index(1, 0, 0, 8), 32);
        assert_eq!(ks_index(2, 3, 2, 8), (2 * 8 + 3) * 4 + 2);
    }

    #[test]
    fn session_header_abi_is_40_bytes_and_matches_cpp_layout() {
        assert_eq!(std::mem::size_of::<SessionHeader>(), 40);
        assert_eq!(std::mem::align_of::<SessionHeader>(), 8);
        assert_eq!(std::mem::offset_of!(SessionHeader, magic), 0);
        assert_eq!(std::mem::offset_of!(SessionHeader, n_layers), 8);
        assert_eq!(std::mem::offset_of!(SessionHeader, kv_q8), 12);
        assert_eq!(std::mem::offset_of!(SessionHeader, cell), 16);
        assert_eq!(std::mem::offset_of!(SessionHeader, n_tokens), 20);
        assert_eq!(std::mem::offset_of!(SessionHeader, model_id), 24);
        assert_eq!(std::mem::offset_of!(SessionHeader, ring), 32);
        assert_eq!(std::mem::offset_of!(SessionHeader, ctx16), 36);

        let hdr = SessionHeader::new(48, true, 1234, 0xDEAD_BEEF_CAFE_BABEu64, 1088, 4160);
        let raw = hdr.to_bytes();
        let decoded = SessionHeader::from_bytes(&raw);
        assert_eq!(hdr, decoded);
        assert!(decoded.validate(48, true, 0xDEAD_BEEF_CAFE_BABEu64, 1088, 4160, 4096));
        assert!(!decoded.validate(47, true, 0xDEAD_BEEF_CAFE_BABEu64, 1088, 4160, 4096));
        assert!(!decoded.validate(48, false, 0xDEAD_BEEF_CAFE_BABEu64, 1088, 4160, 4096));
        assert!(!decoded.validate(48, true, 0xDEAD_BEEF_CAFE_BABFu64, 1088, 4160, 4096));
        assert!(!decoded.validate(48, true, 0xDEAD_BEEF_CAFE_BABEu64, 2048, 4160, 4096));
        assert!(!decoded.validate(48, true, 0xDEAD_BEEF_CAFE_BABEu64, 1088, 8192, 4096));
        assert!(!decoded.validate(48, true, 0xDEAD_BEEF_CAFE_BABEu64, 1088, 4160, 1000));
    }

    #[test]
    fn rope_table_yarn_and_swa_match_cpp_math() {
        let max_ctx = 64usize;
        let mut cos_full = vec![0.0_f32; max_ctx * 32];
        let mut sin_full = vec![0.0_f32; max_ctx * 32];
        let full_p = RopeScalingParams {
            scaling_type: TK_ROPE_SCALING_YARN,
            freq_base: 500_000.0,
            factor: 32.0,
            freq_scale_in: 0.0,
            orig_ctx: 8192,
            ext_factor: 1.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        tk_build_rope_table(&mut cos_full, &mut sin_full, max_ctx, 64, &full_p);

        // At pos=0, angle is 0 for all dims, so sin=0 and cos = mscale = 1 + 0.1 * ln(32)
        let expected_mscale = (1.0_f64 + 0.1_f64 * 32.0_f64.ln()) as f32;
        for i in 0..32 {
            assert_eq!(cos_full[i].to_bits(), expected_mscale.to_bits());
            assert_eq!(sin_full[i].to_bits(), 0.0_f32.to_bits());
        }

        let mut cos_swa = vec![0.0_f32; max_ctx * 64];
        let mut sin_swa = vec![0.0_f32; max_ctx * 64];
        let swa_p = RopeScalingParams {
            scaling_type: TK_ROPE_SCALING_NONE,
            freq_base: 10_000.0,
            ..Default::default()
        };
        tk_build_rope_table(&mut cos_swa, &mut sin_swa, max_ctx, 128, &swa_p);
        for i in 0..64 {
            assert_eq!(cos_swa[i].to_bits(), 1.0_f32.to_bits());
            assert_eq!(sin_swa[i].to_bits(), 0.0_f32.to_bits());
        }
        // Position 1, dim pair 0 has inv = 1.0, angle = 1.0 rad
        assert_eq!(cos_swa[64].to_bits(), (1.0_f64.cos() as f32).to_bits());
        assert_eq!(sin_swa[64].to_bits(), (1.0_f64.sin() as f32).to_bits());
    }

    #[test]
    fn dequant_embd_row_q8_0_f16_f32_work() {
        let n_blocks = K_EMBD / 32;
        let mut raw_q8 = vec![0u8; n_blocks * 34];
        let d_bits = f32_to_f16(0.5_f32).to_le_bytes();
        for b in 0..n_blocks {
            let blk = &mut raw_q8[b * 34..(b + 1) * 34];
            blk[0] = d_bits[0];
            blk[1] = d_bits[1];
            for j in 0..32 {
                blk[2 + j] = (j as i8 - 16) as u8;
            }
        }
        let mut out = vec![0.0_f32; K_EMBD];
        dequant_embd_row(QuantType::Q8_0, &raw_q8, 0, &mut out).unwrap();
        assert_eq!(out[0], -8.0_f32);
        assert_eq!(out[16], 0.0_f32);
        assert_eq!(out[18], 1.0_f32);
    }

    #[test]
    fn drafter_key_push_draft_and_ev_selection_match_cpp() {
        // Test FNV-1a key matching C++ `Drafter::key`
        let k3 = Drafter::key(&[10, 20, 30]);
        let mut expected = 1_469_598_103_934_665_603_u64;
        for &tok in &[10i32, 20, 30] {
            expected = (expected ^ (tok as u32 as u64)).wrapping_mul(1_099_511_628_211_u64);
        }
        expected ^= 3u64;
        assert_eq!(k3, expected);

        // Prompt with verbatim copy of 8 tokens followed by continuation [90, 91, 92, 93]
        let seq = vec![
            1, 2, 3, 4, 5, 6, 7, 8, // first occurrence (indices 0..8)
            90, 91, 92, 93, // continuation at index 8..12
            1, 2, 3, 4, 5, 6, 7, 8, // second occurrence (indices 12..20)
        ];
        let mut drafter = Drafter::new();
        drafter.reset(&seq);

        let mut out = [0i32; 7];
        // With min_match = 8, suffix [1..=8] matches earlier occurrence at index 8:
        // k = 4 drafts [90, 91, 92, 93]; k = 7 drafts [90, 91, 92, 93, 1, 2, 3]
        let nd4 = drafter.draft(4, &mut out, 8);
        assert_eq!(nd4, 4);
        assert_eq!(&out[..4], &[90, 91, 92, 93]);
        let nd7 = drafter.draft(7, &mut out, 8);
        assert_eq!(nd7, 7);
        assert_eq!(&out, &[90, 91, 92, 93, 1, 2, 3]);

        // With min_match = 9, suffix only matches for 8 tokens -> returns 0
        assert_eq!(drafter.draft(7, &mut out, 9), 0);

        // Expected-value best_k selection:
        // At acc_est = 0.9, kDraftCost = 0.40:
        // kk=1: ex=1.9, r=1.9/1.4=1.357
        // kk=4: ex=(1-0.9^5)/0.1=4.0951, r=4.0951/2.6=1.575038
        // kk=5: ex=(1-0.9^6)/0.1=4.68559, r=4.68559/3.0=1.56186
        assert_eq!(Drafter::select_best_k(7, 0.9), 4);
        // At low acceptance acc_est = 0.2, no kk exceeds 1.03 -> best_k = 0
        assert_eq!(Drafter::select_best_k(7, 0.2), 0);
        // At near-certain acceptance acc_est = 0.9995, best_k = min(spec, 7)
        assert_eq!(Drafter::select_best_k(7, 0.9995), 7);
    }

    #[test]
    fn sample_penalties_match_cpp() {
        let mut rng = Mt19937_64::new(42);
        let mut logits = vec![1.0_f32, 5.0_f32, 4.0_f32, 2.0_f32];
        let samp = Sampling {
            temperature: 0.0,
            penalty_repeat: 2.0,
            penalty_freq: 0.5,
            penalty_present: 0.5,
            penalty_last_n: 64,
            ..Default::default()
        };
        // Token 1 has logit 5.0; with recent=[1, 1], penalty_repeat=2.0 -> 2.5, freq=2*0.5=1.0, pres=0.5 -> 1.0.
        // Token 2 has logit 4.0 (unpenalized) -> wins greedy!
        let picked = sample(&mut logits, &samp, &[1, 1], &mut rng);
        assert_eq!(picked, 2);
    }

    #[test]
    fn laguna_gguf_header_shards_validate_814_tensors() {
        let Some(shapes_dir) = std::env::var_os("TALUS_SHAPES_DIR").map(std::path::PathBuf::from)
        else {
            return;
        };
        let hdr_dir = shapes_dir.join("hdr");
        let p1 = hdr_dir.join("Laguna-S-2.1-Q4-lean-00001-of-00003.gguf.hdr");
        let p2 = hdr_dir.join("Laguna-S-2.1-Q4-lean-00002-of-00003.gguf.hdr");
        let p3 = hdr_dir.join("Laguna-S-2.1-Q4-lean-00003-of-00003.gguf.hdr");
        if !p1.exists() || !p2.exists() || !p3.exists() {
            return;
        }
        let v1 = GgufFile::open(&p1).expect("open shard 1 hdr");
        let v2 = GgufFile::open(&p2).expect("open shard 2 hdr");
        let v3 = GgufFile::open(&p3).expect("open shard 3 hdr");

        let cat = LagunaCatalog::validate_views(&[v1, v2, v3], "Laguna-S-2.1", K_LAYERS)
            .expect("valid Laguna catalog");
        assert_eq!(cat.n_layers, 48);
        assert_eq!(cat.n_tensors, 814);
        assert!(cat.total_tensor_bytes > 60_000_000_000);
        assert_ne!(cat.model_id, 0);
    }
}
