//! Typed safe kernel wrappers, `KernelSet` selection, `TuningTables`,
//! `FfiKernelBackend`, and `CpuMockBackend`.

pub mod backend;
pub mod cpu_mock;
pub mod ffi;
pub mod kernel_set;
pub mod quant;
pub mod tuning;

pub use backend::{
    AttnDecodeArgs, AttnPrefillArgs, GdnSeqArgs, GdnStepArgs, KernelBackend, MmvqIdsArgs,
    PrefillGemmArgs, PrefillMmqArgs,
};
pub use cpu_mock::{CpuMockBackend, ROUTER_MIN_DENOM};
pub use ffi::{
    gpu_arch_to_tk, is_tk_error, quant_type_to_tk, tk_to_gpu_arch, tk_to_quant_type,
    translate_tk_status, FfiConfig, FfiKernelBackend, FfiPackedWeights,
};
pub use kernel_set::{KernelId, KernelOp, KernelSet, SUPPORTED_ARCHS};
pub use quant::{
    bf16_to_f32, build_rope_table, dequantize_row, dequantize_row_q4_0, dequantize_row_q4_k,
    dequantize_row_q8_0, dequantize_row_q8_1, f16_to_f32, f32_to_bf16, f32_to_f16, q8_1_bytes,
    quantize_row_q4_0, quantize_row_q8_0, quantize_row_q8_1, row_bytes,
};
pub use tuning::{
    arch_matches_table, strip_arch_features, TuningDtype, TuningRow, TuningTable, TuningTables,
    TUNING_HEADER_MAGIC,
};

/// Re-exported Section 9 (`q36` / `adraft` / `dflash`) C ABI types and entry points from
/// [`talus_kernels_sys`] for Wave 2 (`WP-2a` & `WP-2b`).
pub mod q36 {
    pub use talus_kernels_sys::{
        talus_pf_fa_prefill, talus_pf_i8gemm, talus_pf_i8x_expand, talus_pf_swiglu_q8_mmq,
        talus_pf_w4_gemm, talus_pf_w4_pack, tk_q36_add, tk_q36_add_rms, tk_q36_add_rms_q8,
        tk_q36_argmax, tk_q36_argmax_samp, tk_q36_attn_blas_prefill, tk_q36_attn_merge,
        tk_q36_attn_part, tk_q36_attn_prep, tk_q36_attn_softmax16, tk_q36_attn_wmma,
        tk_q36_chain_prep, tk_q36_conv_ab, tk_q36_conv_hist, tk_q36_conv_par, tk_q36_conv_seq,
        tk_q36_ctl_end, tk_q36_dflash_add_inplace, tk_q36_dflash_attn_chunks,
        tk_q36_dflash_attn_scratch_floats, tk_q36_dflash_block_attn, tk_q36_dflash_block_tokens,
        tk_q36_dflash_dyn_conv, tk_q36_dflash_kv_store, tk_q36_dflash_qk_norm_rope,
        tk_q36_dflash_rms_rows, tk_q36_dflash_select_chain, tk_q36_dflash_select_chain_q,
        tk_q36_dflash_set_i32, tk_q36_dflash_sum5_rms, tk_q36_dflash_swiglu, tk_q36_dflash_top16,
        tk_q36_draft_conf, tk_q36_draft_samp, tk_q36_draft_support, tk_q36_embed_rows,
        tk_q36_expert_grouped, tk_q36_expert_layout, tk_q36_expert_scratch_bytes,
        tk_q36_expert_supported, tk_q36_fattn_mask_bytes, tk_q36_fattn_supported,
        tk_q36_flush_read, tk_q36_gate_mul, tk_q36_gate_rows, tk_q36_gdn_ab, tk_q36_gdn_ab2,
        tk_q36_gdn_gb, tk_q36_gdn_gb2, tk_q36_gdn_norm, tk_q36_gdn_pend, tk_q36_gdn_seq,
        tk_q36_gdn_seq_cols, tk_q36_gdn_seq_warp, tk_q36_gemv_f32, tk_q36_group, tk_q36_group_hy,
        tk_q36_lse, tk_q36_merge_cpu, tk_q36_mmq_built, tk_q36_mmq_iota, tk_q36_mmq_matrix_bytes,
        tk_q36_mmq_q8_bytes, tk_q36_mmq_q8_layout, tk_q36_mmq_quantize, tk_q36_mmq_set_bounds,
        tk_q36_mmq_supported_fits, tk_q36_mmq_swiglu, tk_q36_mmvq_supported, tk_q36_moe_out,
        tk_q36_prefill_init_scratch, tk_q36_q8_0_pack_host, tk_q36_q8_0_pack_should,
        tk_q36_q8_0_packed_register, tk_q36_q8_0_packed_unregister, tk_q36_rms, tk_q36_rms_ld,
        tk_q36_round_prep, tk_q36_route_count, tk_q36_route_scan, tk_q36_route_scatter,
        tk_q36_router_topk, tk_q36_row_copy, tk_q36_sgemm_nt, tk_q36_sum_splits, tk_q36_swiglu,
        tk_q36_to_f16,
    };
}

pub(crate) fn invalid_arg(reason: impl Into<String>) -> talus_core_api::CoreError {
    talus_core_api::CoreError::InvalidArgument {
        reason: reason.into(),
    }
}

pub(crate) fn io_err(message: impl Into<String>) -> talus_core_api::CoreError {
    talus_core_api::CoreError::IoError {
        message: message.into(),
    }
}

pub(crate) fn unsupported_quant(arch: &str, op: &str, quant: &str) -> talus_core_api::CoreError {
    talus_core_api::CoreError::CapabilityRefused {
        arch: arch.to_string(),
        reason: format!("{op}: unsupported quantization type {quant}"),
    }
}

pub const CRATE_NAME: &str = "talus-kernels";

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use talus_core_api::{CoreError, GpuArch, QuantType, StreamHandle};

    fn repo_tools_hip_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tools/hip")
    }

    #[test]
    fn crate_identity() {
        assert_eq!(CRATE_NAME, "talus-kernels");
    }

    #[test]
    fn kernel_set_select_gfx1151_vs_gfx1201_and_other() {
        // Unsupported architectures must return CoreError::UnsupportedArch
        let err_1201 =
            KernelSet::select(&GpuArch::Gfx1201, KernelOp::Mmvq, QuantType::Q4_K, 1).unwrap_err();
        match err_1201 {
            CoreError::UnsupportedArch { arch, supported } => {
                assert_eq!(arch, "gfx1201");
                assert_eq!(supported, &["gfx1151"]);
            }
            other => panic!("expected UnsupportedArch for gfx1201, got {other:?}"),
        }

        let err_other = KernelSet::select(
            &GpuArch::Other("gfx1100".to_string()),
            KernelOp::AttnDecode,
            QuantType::F16,
            1024,
        )
        .unwrap_err();
        assert!(matches!(err_other, CoreError::UnsupportedArch { .. }));
        assert!(KernelSet::for_arch(GpuArch::Gfx1201).is_err());

        let kset = KernelSet::for_arch(GpuArch::Gfx1151).expect("gfx1151 is supported");
        assert_eq!(kset.arch(), &GpuArch::Gfx1151);

        // ncols = 1 MMVQ rows_per_wg table:
        // Q2_K / IQ2_XS / IQ2_S -> 2; Q3_K -> 4; Q4_K / Q5_K / Q6_K / Q8_0 -> 1
        assert_eq!(
            KernelSet::mmvq_rows_per_wg(&GpuArch::Gfx1151, QuantType::Q2_K, 1).unwrap(),
            2
        );
        assert_eq!(
            KernelSet::mmvq_rows_per_wg(&GpuArch::Gfx1151, QuantType::IQ2_XS, 1).unwrap(),
            2
        );
        assert_eq!(
            KernelSet::mmvq_rows_per_wg(&GpuArch::Gfx1151, QuantType::IQ2_S, 1).unwrap(),
            2
        );
        assert_eq!(
            KernelSet::mmvq_rows_per_wg(&GpuArch::Gfx1151, QuantType::Q3_K, 1).unwrap(),
            4
        );
        assert_eq!(
            KernelSet::mmvq_rows_per_wg(&GpuArch::Gfx1151, QuantType::Q4_K, 1).unwrap(),
            1
        );
        assert_eq!(
            KernelSet::mmvq_rows_per_wg(&GpuArch::Gfx1151, QuantType::Q8_0, 1).unwrap(),
            1
        );

        // 2 <= ncols <= 8 MMVQ rows_per_wg table:
        // Q4_K / Q5_K / Q6_K -> 2; IQ3_S / Q3_K / Q2_K -> 4; Q8_0 / Q4_0 -> 1
        for ncols in [2u32, 5, 8] {
            let k_q4k =
                KernelSet::select(&GpuArch::Gfx1151, KernelOp::Mmvq, QuantType::Q4_K, ncols)
                    .unwrap();
            assert_eq!(k_q4k.rows_per_wg, 2);
            if ncols <= 5 {
                assert_eq!(k_q4k.variant, "mmvq_w32_r2");
            } else {
                assert_eq!(k_q4k.variant, "mmvq_w32_wide_r2");
            }

            let k_q5k =
                KernelSet::select(&GpuArch::Gfx1151, KernelOp::Mmvq, QuantType::Q5_K, ncols)
                    .unwrap();
            assert_eq!(k_q5k.rows_per_wg, 2);

            let k_q6k =
                KernelSet::select(&GpuArch::Gfx1151, KernelOp::Mmvq, QuantType::Q6_K, ncols)
                    .unwrap();
            assert_eq!(k_q6k.rows_per_wg, 2);

            let k_iq3s =
                KernelSet::select(&GpuArch::Gfx1151, KernelOp::Mmvq, QuantType::IQ3_S, ncols)
                    .unwrap();
            assert_eq!(k_iq3s.rows_per_wg, 4);

            let k_q3k =
                KernelSet::select(&GpuArch::Gfx1151, KernelOp::Mmvq, QuantType::Q3_K, ncols)
                    .unwrap();
            assert_eq!(k_q3k.rows_per_wg, 4);

            let k_q8 = KernelSet::select(&GpuArch::Gfx1151, KernelOp::Mmvq, QuantType::Q8_0, ncols)
                .unwrap();
            assert_eq!(k_q8.rows_per_wg, 1);
        }

        // 9 <= ncols <= 16: rows_per_wg is always 1 (wide kernel)
        for ncols in [9u32, 12, 16] {
            let k_wide =
                KernelSet::select(&GpuArch::Gfx1151, KernelOp::Mmvq, QuantType::Q4_K, ncols)
                    .unwrap();
            assert_eq!(k_wide.rows_per_wg, 1);
            assert_eq!(k_wide.variant, "mmvq_w32_wide_r1");
        }

        // Attention split-K cell sizes: 128 below 4K (< 4096), 512 at or above 4K (>= 4096)
        let attn_short = KernelSet::select(
            &GpuArch::Gfx1151,
            KernelOp::AttnDecode,
            QuantType::F16,
            4095,
        )
        .unwrap();
        assert_eq!(attn_short.chunk_cells, 128);
        assert_eq!(attn_short.variant, "q36_attn_splitk_c128");

        let attn_long = KernelSet::select(
            &GpuArch::Gfx1151,
            KernelOp::AttnDecode,
            QuantType::F16,
            4096,
        )
        .unwrap();
        assert_eq!(attn_long.chunk_cells, 512);
        assert_eq!(attn_long.variant, "q36_attn_splitk_c512");

        let laguna_short = KernelSet::select(
            &GpuArch::Gfx1151,
            KernelOp::LagunaDecodeAttn,
            QuantType::F16,
            2048,
        )
        .unwrap();
        assert_eq!(laguna_short.chunk_cells, 128);

        let laguna_long = KernelSet::select(
            &GpuArch::Gfx1151,
            KernelOp::LagunaDecodeAttn,
            QuantType::F16,
            8192,
        )
        .unwrap();
        assert_eq!(laguna_long.chunk_cells, 512);
    }

    #[test]
    fn tuning_tables_load_real_gfx1151_files_and_version_sibling_fallback() {
        let hip_dir = repo_tools_hip_dir();
        let file_100400 = hip_dir.join("gfx1151-hipblaslt-100400.txt");
        let file_100401 = hip_dir.join("gfx1151-hipblaslt-100401.txt");
        let file_100500 = hip_dir.join("gfx1151-hipblaslt-100500.txt");

        let t0 = TuningTables::load_file(&file_100400, Some("gfx1151"), Some(100400))
            .expect("should load real gfx1151-hipblaslt-100400.txt");
        assert_eq!(t0.arch(), "gfx1151");
        assert_eq!(t0.hipblaslt_version(), 100400);
        assert!(!t0.is_empty());

        let t1 =
            TuningTables::load_file(&file_100401, Some("gfx1151:sramecc-:xnack-"), Some(100401))
                .expect("should load real gfx1151-hipblaslt-100401.txt");
        assert_eq!(t1.arch(), "gfx1151");
        assert_eq!(t1.hipblaslt_version(), 100401);
        assert!(!t1.is_empty());

        let t2 = TuningTables::load_file(&file_100500, Some("gfx1151"), Some(100500))
            .expect("should load real gfx1151-hipblaslt-100500.txt");
        assert_eq!(t2.arch(), "gfx1151");
        assert_eq!(t2.hipblaslt_version(), 100500);
        assert!(!t2.is_empty());

        // Exact bucket lookup & nearest-bucket tie-breaking in real 100500 table:
        // bf16 1 2560 1 has t_bucket=4096 (sol 1253), t_bucket=8192 (sol 1252)
        assert_eq!(
            t2.select_solution(TuningDtype::Bf16, 1, 2560, 1, 4096),
            Some(1253)
        );
        assert_eq!(
            t2.select_solution(TuningDtype::Bf16, 1, 2560, 1, 8192),
            Some(1252)
        );
        // t=6144 is equidistant between 4096 and 8192 -> tie breaks to smaller t_bucket (4096 -> sol 1253)
        let row_tie = t2
            .closest(TuningDtype::Bf16, 1, 2560, 1, 6144)
            .expect("row exists");
        assert_eq!(row_tie.t_bucket, 4096);
        assert_eq!(row_tie.solution_id, 1253);

        // Commit 5c7e9867 version-sibling fallback rule:
        // Pointing at 100401 file while runtime hipBLASLt is 100500 must fall back to sibling 100500 file!
        let fallback_to_100500 =
            TuningTables::load_with_fallback(&file_100401, "gfx1151:sramecc-:xnack-", 100500)
                .expect("should fall back to sibling gfx1151-hipblaslt-100500.txt");
        assert_eq!(fallback_to_100500.hipblaslt_version(), 100500);
        assert_eq!(
            fallback_to_100500
                .source_path()
                .unwrap()
                .file_name()
                .unwrap(),
            "gfx1151-hipblaslt-100500.txt"
        );

        // Pointing at 100500 file while runtime hipBLASLt is 100401 must fall back to sibling 100401 file!
        let fallback_to_100401 = TuningTables::load_with_fallback(&file_100500, "gfx1151", 100401)
            .expect("should fall back to sibling gfx1151-hipblaslt-100401.txt");
        assert_eq!(fallback_to_100401.hipblaslt_version(), 100401);
        assert_eq!(
            fallback_to_100401
                .source_path()
                .unwrap()
                .file_name()
                .unwrap(),
            "gfx1151-hipblaslt-100401.txt"
        );

        // Loading all tuning files in tools/hip/
        let all = TuningTables::load_all_in_dir(&hip_dir).expect("should load all tuning tables");
        assert!(all.len() >= 3);

        // Architecture family matching rules
        assert!(arch_matches_table("gfx1201:sramecc-", "gfx1200"));
        assert!(arch_matches_table("gfx1102", "gfx1100"));
        assert!(!arch_matches_table("gfx1151", "gfx1100"));
        assert!(!arch_matches_table("gfx1151", "gfx1201"));
    }

    #[test]
    fn cpu_mock_rmsnorm_rope_silu_axpy_embed_softplus() {
        let backend = CpuMockBackend::new();
        assert_eq!(backend.arch(), GpuArch::Gfx1151);

        // 1. RMSNorm
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 1.0, -1.0, 1.0];
        let w = vec![0.5f32, 1.0, 1.5, 2.0];
        let eps = 1e-5f32;
        backend.rms_norm(&mut x, Some(&w), 2, 4, eps).unwrap();
        // Row 0: mean(1+4+9+16)=7.5, rstd = 1/sqrt(7.5 + 1e-5)
        let rstd0 = 1.0f32 / (7.5f32 + eps).sqrt();
        assert!((x[0] - 1.0 * rstd0 * 0.5).abs() < 1e-6);
        assert!((x[3] - 4.0 * rstd0 * 2.0).abs() < 1e-6);
        // Row 1: mean(1+1+1+1)=1.0, rstd = 1/sqrt(1 + 1e-5)
        let rstd1 = 1.0f32 / (1.0f32 + eps).sqrt();
        assert!((x[4] - (-rstd1 * 0.5)).abs() < 1e-6);

        // 2. NEOX partial RoPE + HeadNormRoPE
        let (cos_tab, sin_tab) = build_rope_table(4, 10000.0, 8).unwrap();
        // head_dim = 6, n_rot = 4 (last 2 dims untouched)
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 99.0, -42.0];
        let mut out = vec![0.0f32; 6];
        backend
            .rope_neox(&src, &mut out, 1, 6, 4, &cos_tab, &sin_tab, &[3])
            .unwrap();
        let c0 = cos_tab[3 * 2];
        let s0 = sin_tab[3 * 2];
        let c1 = cos_tab[3 * 2 + 1];
        let s1 = sin_tab[3 * 2 + 1];
        assert!((out[0] - (1.0 * c0 - 3.0 * s0)).abs() < 1e-6);
        assert!((out[2] - (1.0 * s0 + 3.0 * c0)).abs() < 1e-6);
        assert!((out[1] - (2.0 * c1 - 4.0 * s1)).abs() < 1e-6);
        assert!((out[3] - (2.0 * s1 + 4.0 * c1)).abs() < 1e-6);
        assert_eq!(out[4], 99.0);
        assert_eq!(out[5], -42.0);

        // Verify head_norm_rope equals rms_norm followed by rope_neox
        let mut hnr = src.clone();
        let mut sep_norm = src.clone();
        let mut sep_out = vec![0.0f32; 6];
        let hw = vec![1.0f32, 0.5, 2.0, 1.5, 0.25, 0.75];
        backend
            .head_norm_rope(&mut hnr, &hw, 1, 1e-5, 4, &cos_tab, &sin_tab, &[3], 1)
            .unwrap();
        backend
            .rms_norm(&mut sep_norm, Some(&hw), 1, 6, 1e-5)
            .unwrap();
        backend
            .rope_neox(&sep_norm, &mut sep_out, 1, 6, 4, &cos_tab, &sin_tab, &[3])
            .unwrap();
        for i in 0..6 {
            assert!((hnr[i] - sep_out[i]).abs() < 1e-6);
        }

        // 3. SiLU * up & axpy & softplus_head_gate
        let mut gate = vec![0.0f32, 1.0, -1.0];
        let up = vec![2.0f32, 3.0, 4.0];
        backend.silu_mul(&mut gate, &up, 3).unwrap();
        assert!((gate[0] - 0.0).abs() < 1e-6);
        assert!((gate[1] - 3.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
        assert!((gate[2] - (-4.0 / (1.0 + 1.0f32.exp()))).abs() < 1e-6);

        let mut y = vec![1.0f32, 2.0, 3.0];
        backend.axpy(&mut y, &[10.0, 20.0, 30.0], 0.5, 3).unwrap();
        assert_eq!(y, vec![6.0, 12.0, 18.0]);

        let mut o = vec![2.0f32, 4.0, 1.0, 3.0];
        let g = vec![0.0f32, 25.0f32]; // softplus(0) = ln(2), softplus(25) = 25 (linear regime > 20)
        backend.softplus_head_gate(&mut o, &g, 2, 2).unwrap();
        assert!((o[0] - 2.0 * 2.0f32.ln()).abs() < 1e-6);
        assert!((o[2] - 25.0).abs() < 1e-6);
        assert!((o[3] - 75.0).abs() < 1e-6);

        // 4. Q8_0 embedding table lookup
        let mut row0 = vec![0.0f32; 32];
        let mut row1 = vec![0.0f32; 32];
        for i in 0..32 {
            row0[i] = (i as f32) * 0.25;
            row1[i] = -((i as f32) * 0.5);
        }
        let mut table = vec![0u8; 68];
        quantize_row_q8_0(&row0, &mut table[0..34]).unwrap();
        quantize_row_q8_0(&row1, &mut table[34..68]).unwrap();
        let mut emb_out = vec![0.0f32; 64];
        backend
            .embed_q8_0(&table, &[1, 0], &mut emb_out, 32)
            .unwrap();
        for i in 0..32 {
            assert!((emb_out[i] - row1[i]).abs() < 0.08);
            assert!((emb_out[32 + i] - row0[i]).abs() < 0.05);
        }
    }

    #[test]
    fn cpu_mock_matmul_mmvq_routing_moe_and_argmax() {
        let backend = CpuMockBackend::new();

        // 1. f32_matvec & f32_matmul
        let w = vec![
            1.0f32, 2.0, 3.0, // row 0
            -1.0, 0.5, 2.0, // row 1
        ];
        let x = vec![2.0f32, 1.0, -1.0, 0.0, 4.0, 2.0];
        let mut y_mv = vec![0.0f32; 2];
        backend.f32_matvec(&w, &x[0..3], &mut y_mv, 3, 2).unwrap();
        assert!((y_mv[0] - 1.0).abs() < 1e-6); // 2 + 2 - 3 = 1
        assert!((y_mv[1] - (-3.5)).abs() < 1e-6); // -2 + 0.5 - 2 = -3.5

        let mut y_mm = vec![0.0f32; 4];
        backend.f32_matmul(&w, &x, &mut y_mm, 3, 2, 2).unwrap();
        assert!((y_mm[0] - 1.0).abs() < 1e-6);
        assert!((y_mm[1] - (-3.5)).abs() < 1e-6);
        assert!((y_mm[2] - 14.0).abs() < 1e-6); // 0 + 8 + 6 = 14
        assert!((y_mm[3] - 6.0).abs() < 1e-6); // 0 + 2 + 4 = 6

        // 2. quantize_q8_1 + mmvq + mmvq_ids + prefill_mmq + prefill_gemm
        let n_in = 32;
        let n_out = 2;
        let mut w_f32 = vec![0.0f32; n_out * n_in];
        for i in 0..n_in {
            w_f32[i] = ((i as f32) - 16.0) * 0.1;
            w_f32[n_in + i] = ((16 - i as i32) as f32) * 0.05;
        }
        let mut w_q8_0 = vec![0u8; n_out * 34];
        quantize_row_q8_0(&w_f32[0..n_in], &mut w_q8_0[0..34]).unwrap();
        quantize_row_q8_0(&w_f32[n_in..2 * n_in], &mut w_q8_0[34..68]).unwrap();

        let x_act: Vec<f32> = (0..n_in).map(|i| (i as f32) * 0.05).collect();
        let mut x_q8_1 = vec![0u8; 36];
        backend.quantize_q8_1(&x_act, &mut x_q8_1, n_in, 1).unwrap();

        let mut y_mmvq = vec![0.0f32; n_out];
        backend
            .mmvq(
                QuantType::Q8_0,
                &w_q8_0,
                &x_q8_1,
                &mut y_mmvq,
                n_in,
                n_out,
                1,
            )
            .unwrap();

        let mut y_ref = vec![0.0f32; n_out];
        backend
            .f32_matvec(&w_f32, &x_act, &mut y_ref, n_in, n_out)
            .unwrap();
        for o in 0..n_out {
            assert!((y_mmvq[o] - y_ref[o]).abs() < 0.05);
        }

        // mmvq_ids with 2 experts and paired w2/y2
        let mut experts_w = vec![0u8; 2 * 68];
        experts_w[0..68].copy_from_slice(&w_q8_0);
        experts_w[68..136].copy_from_slice(&w_q8_0);
        let mut y_ids = vec![0.0f32; 2 * n_out];
        let mut y2_ids = vec![0.0f32; 2 * n_out];
        backend
            .mmvq_ids(MmvqIdsArgs {
                qtype: QuantType::Q8_0,
                w: &experts_w,
                expert_bytes: 68,
                ids: &[1, 0],
                x_q8_1: &x_q8_1,
                x_per_id: false,
                y: &mut y_ids,
                n_in,
                n_out,
                w2: Some(&experts_w),
                y2: Some(&mut y2_ids),
            })
            .unwrap();
        assert_eq!(y_ids[0..n_out], y_mmvq[..]);
        assert_eq!(y_ids[n_out..2 * n_out], y_mmvq[..]);
        assert_eq!(y2_ids, y_ids);

        // 3. route_sigmoid_topk (ties go to lower expert index; normalized weights * scale)
        let logits = vec![0.0f32, 2.0, 2.0, -2.0];
        let bias = vec![0.0f32, 0.1, 0.1, 5.0]; // expert 3 gets boosted by bias, experts 1 & 2 tie -> expert 1 wins 2nd slot
        let mut ids = vec![0i32; 2];
        let mut weights = vec![0.0f32; 2];
        backend
            .route_sigmoid_topk(&logits, Some(&bias), &mut ids, &mut weights, 4, 2, 2.5)
            .unwrap();
        assert_eq!(ids, vec![3, 1]);
        let p3 = 1.0f32 / (1.0 + 2.0f32.exp());
        let p1 = 1.0f32 / (1.0 + (-2.0f32).exp());
        let sum_p = p3 + p1;
        assert!((weights[0] - (p3 / sum_p) * 2.5).abs() < 1e-6);
        assert!((weights[1] - (p1 / sum_p) * 2.5).abs() < 1e-6);

        // 4. moe_add_slots
        let mut res_x = vec![1.0f32, 2.0];
        let shared = vec![0.5f32, 0.5];
        let slot_y = vec![10.0f32, 20.0, 100.0, 200.0];
        let slot_w = vec![0.2f32, 0.1];
        backend
            .moe_add_slots(
                &mut res_x,
                Some(&shared),
                &slot_y,
                Some(&[1, 0]),
                &slot_w,
                1,
                2,
                2,
            )
            .unwrap();
        // res_x[0] = 1.0 + 0.5 + 0.2 * 100.0 + 0.1 * 10.0 = 22.5
        // res_x[1] = 2.0 + 0.5 + 0.2 * 200.0 + 0.1 * 20.0 = 44.5
        assert!((res_x[0] - 22.5).abs() < 1e-5);
        assert!((res_x[1] - 44.5).abs() < 1e-5);

        // 5. argmax with lower-index tie-breaking
        let vocab_logits = vec![
            0.1f32, 5.0, 5.0, 2.0, // tie between 1 and 2 -> 1
            -3.0, -1.0, -2.0, -0.5, // max at 3
        ];
        let mut out_tok = vec![0i32; 2];
        backend.argmax(&vocab_logits, 2, 4, &mut out_tok).unwrap();
        assert_eq!(out_tok, vec![1, 3]);
    }

    #[test]
    fn cpu_mock_causal_gqa_swa_attention_and_gdn_recurrence() {
        let backend = CpuMockBackend::new();

        // 1. Causal GQA + Sliding Window + Attention Sinks + Query Gate
        // n_tok = 3, n_head = 2, n_kv = 1, head_dim = 2, head_dim_v = 2, kv_stride = 4
        let k_f32: [f32; 8] = [
            1.0, 0.0, // pos 0
            0.0, 1.0, // pos 1
            1.0, 1.0, // pos 2
            0.0, 0.0, // pos 3
        ];
        let v_f32: [f32; 8] = [
            10.0, 1.0, // pos 0
            20.0, 2.0, // pos 1
            30.0, 3.0, // pos 2
            0.0, 0.0, // pos 3
        ];
        let mut k_bytes = Vec::with_capacity(32);
        let mut v_bytes = Vec::with_capacity(32);
        for &v in &k_f32 {
            k_bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &v_f32 {
            v_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let q = vec![
            1.0f32, 0.0, 0.0, 1.0, // tok 0 (pos 0)
            1.0, 0.0, 0.0, 1.0, // tok 1 (pos 1)
            1.0, 0.0, 0.0, 1.0, // tok 2 (pos 2)
        ];
        let mut out_prefill = vec![0.0f32; 12];
        backend
            .attn_prefill(AttnPrefillArgs {
                q: &q,
                k_pool: &k_bytes,
                v_pool: &v_bytes,
                kv_f16: false,
                out: &mut out_prefill,
                n_tok: 3,
                n_head: 2,
                n_kv: 1,
                head_dim: 2,
                head_dim_v: 2,
                kv_stride: 4,
                p0: 0,
                window: 2, // SWA window = 2: tok 2 only attends to pos 1 and pos 2 (not pos 0!)
                scale: 1.0,
                sinks: None,
                q_gate: None,
            })
            .unwrap();

        // At tok 0 (pos 0): only pos 0 is in causal window -> output for both heads equals V[0] = [10, 1]
        assert!((out_prefill[0] - 10.0).abs() < 1e-6);
        assert!((out_prefill[1] - 1.0).abs() < 1e-6);

        // Verify that single-step attn_decode at pos=2 with window=2 matches row 2 of attn_prefill!
        let mut out_decode = vec![0.0f32; 4];
        backend
            .attn_decode(AttnDecodeArgs {
                q: &q[8..12],
                k_pool: &k_bytes,
                v_pool: &v_bytes,
                kv_f16: false,
                k_scale: None,
                v_scale: None,
                out: &mut out_decode,
                scratch: None,
                n_tok: 1,
                n_head: 2,
                n_kv: 1,
                head_dim: 2,
                head_dim_v: 2,
                kv_stride: 4,
                window: 2,
                scale: 1.0,
                pos: &[2],
                sinks: None,
                q_gate: None,
                n_split: 0,
            })
            .unwrap();
        for i in 0..4 {
            assert!((out_decode[i] - out_prefill[8 + i]).abs() < 1e-6);
        }

        // At pos 2 with window=2, head 0 has q=[1, 0], keys are pos 1 ([0, 1], dot=0) and pos 2 ([1, 1], dot=1)
        let w1 = 0.0f32.exp();
        let w2 = 1.0f32.exp();
        let expected_v0 = (w1 * 20.0 + w2 * 30.0) / (w1 + w2);
        assert!((out_decode[0] - expected_v0).abs() < 1e-5);

        // 2. GDN recurrent step and sequence step
        // S = 2, h_k = 1, h_v = 2
        let mut state_step = vec![0.0f32; 2 * 2 * 2];
        let mut state_seq = vec![0.0f32; 2 * 2 * 2];
        let gdn_q = vec![1.0f32, 0.0, 0.0, 1.0]; // 2 tokens, h_k=1, S=2
        let gdn_k = vec![1.0f32, 0.0, 0.0, 1.0];
        let gdn_v = vec![
            2.0f32, 4.0, 6.0, 8.0, // tok 0: hv0=[2,4], hv1=[6,8]
            1.0, 3.0, 5.0, 7.0, // tok 1: hv0=[1,3], hv1=[5,7]
        ];
        let gdn_gate = vec![0.0f32, 0.0, -0.5, -0.5]; // tok 0: exp(0)=1, tok 1: exp(-0.5)
        let gdn_beta = vec![1.0f32, 0.5, 1.0, 0.5];
        let gdn_z = vec![1.0f32; 2 * 2 * 2];
        let gdn_gamma = vec![1.0f32, 1.0];

        let mut seq_out = vec![0.0f32; 2 * 2 * 2];
        backend
            .gdn_seq(GdnSeqArgs {
                state: &mut state_seq,
                q: &gdn_q,
                k: &gdn_k,
                v: &gdn_v,
                gate: &gdn_gate,
                beta: &gdn_beta,
                z: &gdn_z,
                gamma: &gdn_gamma,
                out: &mut seq_out,
                n_tok: 2,
                state_size: 2,
                h_k: 1,
                h_v: 2,
                norm_eps: 1e-5,
                silu_gate: true,
            })
            .unwrap();

        // Run 2 single-token gdn_step calls and verify state matches state_seq identically
        let mut step_out = vec![0.0f32; 4];
        for t in 0..2 {
            backend
                .gdn_step(GdnStepArgs {
                    state: &mut state_step,
                    q: &gdn_q[t * 2..(t + 1) * 2],
                    k: &gdn_k[t * 2..(t + 1) * 2],
                    v: &gdn_v[t * 4..(t + 1) * 4],
                    gate: &gdn_gate[t * 2..(t + 1) * 2],
                    beta: &gdn_beta[t * 2..(t + 1) * 2],
                    out: &mut step_out,
                    state_size: 2,
                    h_k: 1,
                    h_v: 2,
                })
                .unwrap();
        }
        assert_eq!(state_step, state_seq);
    }

    #[test]
    fn ffi_kernel_backend_wraps_c_library_and_translates_errors() {
        use std::env;
        use std::fs;
        use std::process::Command;

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let include_dir = manifest_dir.join("../../../include");
        let base_tmp = env::var_os("TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| manifest_dir.join("../../target"));
        let scratch_dir = base_tmp.join(format!("talus-kernels-ffi-test-{}", std::process::id()));
        fs::create_dir_all(&scratch_dir).unwrap();

        let c_file = scratch_dir.join("stub_backend.c");
        let so_file = scratch_dir.join("libtalus_kernels_backend_stub.so");

        // Build a minimal C11 shared library implementing all 75 entry points, with custom
        // rms_norm / axpy behavior and error reporting via tk_last_error.
        let header_path = include_dir.join("talus_kernels.h");
        let header_text = fs::read_to_string(&header_path).unwrap();
        let mut c_src = String::from(
            r#"#include "talus_kernels.h"
#include <math.h>
#include <stdlib.h>

struct tk_config { int64_t val; };
struct tk_ctx { int32_t dev; tk_arch arch; const char* last_err; };
struct tk_packed_weights { int32_t tag; };

const char* tk_status_string(tk_status status) {
    switch (status) {
        case TK_OK: return "TK_OK";
        case TK_ERR_INVALID_ARG: return "TK_ERR_INVALID_ARG";
        case TK_ERR_HIP: return "TK_ERR_HIP";
        default: return "TK_ERR_OTHER";
    }
}
const char* tk_last_error(const tk_ctx* ctx) {
    return (ctx && ctx->last_err) ? ctx->last_err : "";
}
tk_status tk_config_create(tk_config** out_cfg) {
    *out_cfg = (tk_config*)calloc(1, sizeof(tk_config));
    return TK_OK;
}
tk_status tk_config_create_for_arch(tk_arch a, tk_config** out_cfg) {
    (void)a;
    return tk_config_create(out_cfg);
}
tk_status tk_config_from_env(tk_config* cfg) { (void)cfg; return TK_OK; }
void tk_config_destroy(tk_config* cfg) { free(cfg); }
tk_status tk_config_set_int(tk_config* cfg, const char* k, int64_t v) { (void)k; cfg->val = v; return TK_OK; }
tk_status tk_config_get_int(const tk_config* cfg, const char* k, int64_t* v) { (void)k; *v = cfg->val; return TK_OK; }
tk_status tk_ctx_create(int32_t d, const tk_config* cfg, tk_ctx** out) {
    (void)cfg;
    tk_ctx* c = (tk_ctx*)calloc(1, sizeof(tk_ctx));
    c->dev = d;
    c->arch = TK_ARCH_GFX1151;
    c->last_err = "";
    *out = c;
    return TK_OK;
}
void tk_ctx_destroy(tk_ctx* ctx) { free(ctx); }
tk_status tk_ctx_device(const tk_ctx* ctx, int32_t* d) { *d = ctx->dev; return TK_OK; }
tk_status tk_ctx_arch(const tk_ctx* ctx, tk_arch* a) { *a = ctx->arch; return TK_OK; }
tk_status tk_ctx_load_tuning(tk_ctx* ctx, const char* p) {
    if (!p || p[0] == '\0') {
        ctx->last_err = "empty tuning path";
        return TK_ERR_INVALID_ARG;
    }
    return TK_OK;
}
tk_status tk_axpy(tk_ctx* ctx, float* y, const float* x, float alpha, int32_t n, tk_stream s) {
    (void)s;
    if (n < 0) { ctx->last_err = "negative n"; return TK_ERR_INVALID_ARG; }
    for (int32_t i = 0; i < n; ++i) y[i] += alpha * x[i];
    return TK_OK;
}
size_t tk_q8_1_bytes(int32_t n_in, int32_t ncols) {
    return (size_t)(n_in / 32) * 36u * (size_t)ncols;
}
void tk_packed_free(tk_ctx* ctx, tk_packed_weights* p) { (void)ctx; free(p); }
"#,
        );

        // Auto-generate no-op stubs for the remaining functions in talus_kernels.h
        let already_defined = [
            "tk_status_string",
            "tk_last_error",
            "tk_config_create",
            "tk_config_create_for_arch",
            "tk_config_from_env",
            "tk_config_destroy",
            "tk_config_set_int",
            "tk_config_get_int",
            "tk_ctx_create",
            "tk_ctx_destroy",
            "tk_ctx_device",
            "tk_ctx_arch",
            "tk_ctx_load_tuning",
            "tk_axpy",
            "tk_q8_1_bytes",
            "tk_packed_free",
        ];

        let mut current = String::new();
        let mut collecting = false;
        for line in header_text.lines() {
            let trimmed = line.trim();
            if !collecting {
                if trimmed.starts_with("tk_status tk_")
                    || trimmed.starts_with("const char* tk_")
                    || trimmed.starts_with("void tk_")
                    || trimmed.starts_with("size_t tk_")
                    || trimmed.starts_with("int32_t tk_")
                    || trimmed.starts_with("int tk_")
                    || trimmed.starts_with("int talus_pf_")
                    || trimmed.starts_with("bool tk_")
                {
                    collecting = true;
                    current.clear();
                    current.push_str(trimmed);
                    if trimmed.contains(");") || trimmed.ends_with("TK_NOEXCEPT;") {
                        collecting = false;
                        emit_stub(&mut c_src, &current, &already_defined);
                    }
                }
            } else {
                current.push(' ');
                current.push_str(trimmed);
                if trimmed.contains(");") || trimmed.ends_with("TK_NOEXCEPT;") {
                    collecting = false;
                    emit_stub(&mut c_src, &current, &already_defined);
                }
            }
        }

        fn emit_stub(out: &mut String, decl: &str, skip: &[&str]) {
            for &name in skip {
                if decl.contains(&format!("{name}(")) {
                    return;
                }
            }
            let sig = decl
                .trim_end_matches(';')
                .trim()
                .trim_end_matches("TK_NOEXCEPT")
                .trim();
            out.push_str(sig);
            if sig.starts_with("void ") {
                out.push_str(" {}\n");
            } else {
                out.push_str(" { return 0; }\n");
            }
        }

        fs::write(&c_file, &c_src).unwrap();

        let status = Command::new("cc")
            .args([
                "-std=c11",
                "-shared",
                "-fPIC",
                "-Wno-unused-parameter",
                "-I",
            ])
            .arg(&include_dir)
            .arg(&c_file)
            .arg("-o")
            .arg(&so_file)
            .status()
            .unwrap();
        assert!(status.success());

        let backend = FfiKernelBackend::open(
            &so_file,
            0,
            None,
            StreamHandle::null(talus_core_api::DeviceId(0)),
        )
        .unwrap();
        assert_eq!(backend.arch(), GpuArch::Gfx1151);
        assert_eq!(backend.device_id(), 0);

        let mut y = vec![1.0f32, 2.0, 3.0];
        backend.axpy(&mut y, &[2.0, 4.0, 6.0], 0.5, 3).unwrap();
        assert_eq!(y, vec![2.0, 4.0, 6.0]);

        // Trigger an error in tk_ctx_load_tuning("") and verify tk_last_error message propagation
        let err = backend.load_tuning("").unwrap_err();
        match err {
            CoreError::InvalidArgument { reason } => {
                assert!(reason.contains("TK_ERR_INVALID_ARG"), "msg was: {reason}");
                assert!(reason.contains("empty tuning path"), "msg was: {reason}");
            }
            other => panic!("expected CoreError::InvalidArgument, got {other:?}"),
        }

        let _ = fs::remove_dir_all(&scratch_dir);
    }
}
