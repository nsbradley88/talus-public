//! Unit tests for `talus-model::qwen35` (geometry, options, `Tens`/`Mat`, host dequant,
//! and small `<= 64 MiB` GPU buffer kernel dispatch sanity).

#![allow(unused_unsafe)]

use std::ffi::c_void;
use std::ptr;

use super::ffi::*;
use super::graphs::{
    sample, sample_selftest, DfCtx, DraftLen, DrafterPick, NgramMod, Sampling, SplitMix64,
};
use super::weights::{
    bf16_to_fp32, dequantize_to_f32, fp16_to_fp32, fp32_to_fp16, reference_dense, reference_moe,
    Mat, ModelGeometry, Q36Config, Q36Options, Tens, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_0_PACKED,
};

#[test]
fn test_model_geometry_and_reference_configs() {
    let dense_geom = reference_dense();
    assert_eq!(dense_geom.arch, "qwen35");
    assert!(!dense_geom.is_moe());
    let dense: Q36Config = (&dense_geom).into();
    assert_eq!(dense.n_Layer, 64);
    assert_eq!(dense.d_model, 5120);
    assert_eq!(dense.d_ff, 17408);
    assert_eq!(dense.n_q, 24);
    assert_eq!(dense.n_kv, 4);
    assert_eq!(dense.d_head, 256);
    assert_eq!(dense.d_rot, 64);
    assert_eq!(dense.n_vocab, 248320);
    assert_eq!(dense.gdn_H, 48);
    assert_eq!(dense.gdn_Hk, 16);
    assert_eq!(dense.gdn_D, 128);
    assert_eq!(dense.gdn_S, 128);
    assert_eq!(dense.gdn_Q, 2048);
    assert_eq!(dense.gdn_V, 6144);
    assert_eq!(dense.gdn_C, 10240);
    assert_eq!(dense.max_seq, 65536);

    // Hybrid 3:1 GDN:GQA cadence: every 4th layer (index % 4 == 3) is full attention.
    assert!(!dense_geom.is_full_attention_layer(0));
    assert!(!dense_geom.is_full_attention_layer(1));
    assert!(!dense_geom.is_full_attention_layer(2));
    assert!(dense_geom.is_full_attention_layer(3));
    assert!(dense_geom.is_full_attention_layer(63));

    let moe_geom = reference_moe();
    assert_eq!(moe_geom.arch, "qwen35moe");
    assert!(moe_geom.is_moe());
    let moe: Q36Config = (&moe_geom).into();
    assert!(moe.is_moe);
    assert_eq!(moe.n_Layer, 40);
    assert_eq!(moe.n_exp, 256);
    assert_eq!(moe.n_used, 8);
    assert_eq!(moe.d_exp, 512);
    assert_eq!(moe.d_shexp, 512);

    let cfg: Q36Config = dense.clone();
    assert_eq!(cfg.num_layers(), 64);
    assert_eq!(cfg.hidden_size(), 5120);
    assert_eq!(cfg.vocab_size(), 248320);
}

#[test]
fn test_q36_options_defaults() {
    let opts = Q36Options::new();
    assert!(opts.use_graph);
    assert!(!opts.no_gdn_flush);
    assert!(!opts.no_hipblas);
    assert!(opts.use_fattn);
    assert!(opts.use_bf16_kv);
    assert!(opts.use_w32);
    assert!(opts.use_fused_qkv);
    assert!(opts.use_fused_gate_up);
    assert!(opts.use_fused_PostAttn);
    assert!(opts.use_fused_fattn_q);
    assert!(opts.use_fused_shexp);
    assert!(opts.use_SplitK_down);
    assert!(opts.use_SplitK_out);
    assert!(opts.use_Direct_down);
    assert!(opts.use_Direct_out);
    assert!(opts.use_Wide_out);
    assert!(opts.use_Wide_down);
    assert!(opts.use_mtp_wide);
    assert!(opts.use_mtp_splitk);
    assert!(opts.use_mtp_direct);
    assert!(opts.use_mtp_fp16);
    assert!(opts.use_mtp_graph);
}

#[test]
fn test_tens_and_mat_row_col_calculations() {
    let dummy_buf = [0u8; 34 * 8];
    let tens = Tens {
        ptr: dummy_buf.as_ptr().cast::<c_void>(),
        qtype: GGML_TYPE_Q8_0,
        ne0: 128,
        ne1: 2,
        ne2: 1,
        bytes: dummy_buf.len(),
        w_scale: 1.0,
    };
    assert!(tens.valid());
    assert_eq!(tens.cols(), 128);
    assert_eq!(tens.rows(), 2);
    assert_eq!(tens.row_bytes(), 4 * 34);

    let mat: Mat = tens.into();
    assert!(mat.valid());
    assert_eq!(mat.cols(), 128);
    assert_eq!(mat.rows(), 2);
    assert_eq!(mat.row_bytes(), 4 * 34);
    assert_eq!(mat.tk_qtype(), talus_kernels_sys::tk_qtype::TK_QTYPE_Q8_0);

    let back: Tens = mat.into();
    assert_eq!(back.ne0, 128);
    assert_eq!(back.ne1, 2);
    assert_eq!(back.qtype, GGML_TYPE_Q8_0);
}

#[test]
fn test_host_fp16_bf16_and_q8_0_dequant() {
    let one_h = fp32_to_fp16(1.0);
    assert_eq!(one_h, 0x3C00);
    assert!((fp16_to_fp32(one_h) - 1.0).abs() < 1e-6);
    assert!((bf16_to_fp32(0x3F80) - 1.0).abs() < 1e-6);

    // One Q8_0 block: d = 0.5 (0x3800), qs = [0, 1, 2, ..., 31]
    let mut blk = [0u8; 34];
    blk[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
    for i in 0..32 {
        blk[2 + i] = i as u8;
    }
    let out = dequantize_to_f32(GGML_TYPE_Q8_0, &blk, 32).expect("dequant Q8_0");
    for i in 0..32 {
        assert!((out[i] - (i as f32) * 0.5).abs() < 1e-6);
    }

    // F32 and F16 roundtrip via dequantize_to_f32
    let src_f32 = [1.25f32, -2.5, 0.0, 4.0];
    let src_bytes =
        unsafe { std::slice::from_raw_parts(src_f32.as_ptr().cast::<u8>(), src_f32.len() * 4) };
    let dst_f32 = dequantize_to_f32(GGML_TYPE_F32, src_bytes, 4).expect("dequant F32");
    assert_eq!(&src_f32[..], &dst_f32[..]);

    let src_f16: [u16; 2] = [fp32_to_fp16(2.0), fp32_to_fp16(-0.5)];
    let src_f16_bytes =
        unsafe { std::slice::from_raw_parts(src_f16.as_ptr().cast::<u8>(), src_f16.len() * 2) };
    let dst_from_f16 = dequantize_to_f32(GGML_TYPE_F16, src_f16_bytes, 2).expect("dequant F16");
    assert!((dst_from_f16[0] - 2.0).abs() < 1e-5);
    assert!((dst_from_f16[1] + 0.5).abs() < 1e-5);
}

#[test]
fn test_ngram_and_sampling_selftest() {
    assert!(sample_selftest());
    let mut rng = SplitMix64::new(12345);
    let logits = [0.1f32, 0.2, 5.0, -1.0];
    let greedy = Sampling {
        temp: 0.0,
        ..Sampling::default()
    };
    assert_eq!(sample(&logits, &greedy, &[], &mut rng), 2);

    let mut ng = NgramMod::new();
    for &t in &[10, 20, 30, 40, 50, 10, 20, 30] {
        ng.feed(t);
    }
    let mut prop = [0i32; 4];
    let n = ng.propose(2, &mut prop);
    assert_eq!(n, 2);
    assert_eq!(prop[0], 40);
    assert_eq!(prop[1], 50);

    // Verify 24-gram open-addressed rolling window (`propose_ctx`) chains up to k_max=15 tokens
    // matching `NgramMod::propose` in `q36_engine.cu:446-477`.
    let mut ng24 = NgramMod::with_n(24);
    let mut ctx: Vec<i32> = (100..150).collect(); // 50 tokens: 100..150
    ctx.extend(100..124); // repeat the 24-gram 100..124 at the end of ctx
    let p24 = ng24.propose_ctx(&ctx, 15, 4);
    let expected: Vec<i32> = (124..139).collect();
    assert_eq!(p24, expected);
}

#[test]
fn test_small_gpu_buffer_kernel_dispatch_sanity() {
    // 1. Test `tk_q36_q8_0_pack_eligible` and `tk_q36_q8_0_pack_inplace` on a tiny 128x2 Q8_0 matrix (< 1 KiB).
    assert!(tk_q36_q8_0_pack_eligible(128, 2));
    assert!(!tk_q36_q8_0_pack_eligible(64, 2));
    assert!(!tk_q36_q8_0_pack_eligible(128, 3));

    let Ok(tk) = KernelsContext::new() else {
        // On CPU-only CI hosts without libtalus_kernels.so / ROCm GPU, skip GPU buffer dispatch.
        return;
    };

    let q8_bytes = (128 / 32) * 2 * 34;
    let mut h_q8 = vec![0u8; q8_bytes];
    for b in 0..8 {
        let off = b * 34;
        h_q8[off..off + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        for i in 0..32 {
            h_q8[off + 2 + i] = ((b + i) & 0x7F) as u8;
        }
    }
    let mut d_q8: *mut c_void = ptr::null_mut();
    let mut d_tmp: *mut c_void = ptr::null_mut();
    unsafe {
        assert_eq!(hipMalloc(&mut d_q8, q8_bytes), hipSuccess);
        assert_eq!(hipMalloc(&mut d_tmp, q8_bytes), hipSuccess);
        assert_eq!(
            hipMemcpy(
                d_q8,
                h_q8.as_ptr().cast::<c_void>(),
                q8_bytes,
                HIP_MEMCPY_HOST_TO_DEVICE
            ),
            hipSuccess
        );
    }
    assert_eq!(
        tk_q36_q8_0_pack_inplace(tk.as_ptr(), d_q8, 128, 2, d_tmp, q8_bytes, ptr::null_mut()),
        0
    );
    let _ = GGML_TYPE_Q8_0_PACKED;

    // 2. Test `tk_q36_rms_norm`, `tk_q36_swiglu`, and `tk_q36_argmax` on tiny 128-element buffers (< 16 KiB).
    let cols = 128usize;
    let h_x: Vec<f32> = (0..cols).map(|i| ((i as f32) + 1.0) * 0.01).collect();
    let h_w: Vec<f32> = vec![1.0f32; cols];
    let mut d_x: *mut c_void = ptr::null_mut();
    let mut d_w: *mut c_void = ptr::null_mut();
    let mut d_y: *mut c_void = ptr::null_mut();
    let mut d_u: *mut c_void = ptr::null_mut();
    let mut d_pick: *mut c_void = ptr::null_mut();
    let mut d_pv: *mut c_void = ptr::null_mut();
    let mut d_pi: *mut c_void = ptr::null_mut();

    unsafe {
        assert_eq!(hipMalloc(&mut d_x, cols * 4), hipSuccess);
        assert_eq!(hipMalloc(&mut d_w, cols * 4), hipSuccess);
        assert_eq!(hipMalloc(&mut d_y, cols * 4), hipSuccess);
        assert_eq!(hipMalloc(&mut d_u, cols * 4), hipSuccess);
        assert_eq!(hipMalloc(&mut d_pick, 4), hipSuccess);
        assert_eq!(hipMalloc(&mut d_pv, 16 * 4), hipSuccess);
        assert_eq!(hipMalloc(&mut d_pi, 16 * 4), hipSuccess);

        assert_eq!(
            hipMemcpy(
                d_x,
                h_x.as_ptr().cast::<c_void>(),
                cols * 4,
                HIP_MEMCPY_HOST_TO_DEVICE
            ),
            hipSuccess
        );
        assert_eq!(
            hipMemcpy(
                d_w,
                h_w.as_ptr().cast::<c_void>(),
                cols * 4,
                HIP_MEMCPY_HOST_TO_DEVICE
            ),
            hipSuccess
        );
    }

    assert_eq!(
        tk_q36_rms(
            tk.as_ptr(),
            1,
            d_x.cast::<f32>(),
            d_w.cast::<f32>(),
            d_y.cast::<f32>(),
            cols as i32,
            1e-6,
            ptr::null_mut(),
        ),
        0
    );

    assert_eq!(
        tk_q36_swiglu(
            tk.as_ptr(),
            d_y.cast::<f32>(),
            d_x.cast::<f32>(),
            d_u.cast::<f32>(),
            cols as i64,
            ptr::null_mut(),
        ),
        0
    );

    assert_eq!(
        tk_q36_argmax(
            tk.as_ptr(),
            1,
            1,
            d_x.cast::<f32>(),
            cols as i32,
            d_pick.cast::<i32>(),
            d_pv.cast::<f32>(),
            d_pi.cast::<i32>(),
            false,
            ptr::null(),
            ptr::null_mut(),
        ),
        0
    );
    assert_eq!(
        tk_q36_argmax(
            tk.as_ptr(),
            1,
            1,
            d_x.cast::<f32>(),
            1,
            d_pick.cast::<i32>(),
            d_pv.cast::<f32>(),
            d_pi.cast::<i32>(),
            true,
            ptr::null(),
            ptr::null_mut(),
        ),
        0
    );

    let mut h_y = vec![0.0f32; cols];
    let mut h_pick = -1i32;
    unsafe {
        assert_eq!(hipDeviceSynchronize(), hipSuccess);
        assert_eq!(
            hipMemcpy(
                h_y.as_mut_ptr().cast::<c_void>(),
                d_y,
                cols * 4,
                HIP_MEMCPY_DEVICE_TO_HOST
            ),
            hipSuccess
        );
        assert_eq!(
            hipMemcpy(
                (&mut h_pick as *mut i32).cast::<c_void>(),
                d_pick,
                4,
                HIP_MEMCPY_DEVICE_TO_HOST
            ),
            hipSuccess
        );
    }

    assert!(h_y.iter().all(|v| v.is_finite() && *v > 0.0));
    assert_eq!(h_pick, (cols - 1) as i32);

    // 3. Test DraftLen small buffer init & cleanup and gfx1151 policy defaults
    let mut dl = DraftLen::default();
    dl.init(&tk).expect("DraftLen::init");
    dl.begin_round(4, 0.6, 0.05, 0.95, 20);
    assert_eq!(dl.choose_next_k(4), 4);
    dl.init_arch("gfx1151", 5);
    assert!(dl.stop_mode());
    assert!(dl.want_conf());
    assert_eq!(dl.dmin, 2);
    assert_eq!(dl.dmax(), 7);
    assert_eq!(dl.chain_len(false), 3);
    assert!(dl.dfl_stop());
    assert!(dl.dfl_sample());
    assert_eq!(dl.keep(&[0.9, 0.8, 0.7], 3, false), 3);
    assert_eq!(dl.keep_dfl(&[0.9, 0.8, 0.7], 3, false), 3);
    dl.free_buffers();

    unsafe {
        let _ = hipFree(d_q8);
        let _ = hipFree(d_tmp);
        let _ = hipFree(d_x);
        let _ = hipFree(d_w);
        let _ = hipFree(d_y);
        let _ = hipFree(d_u);
        let _ = hipFree(d_pick);
        let _ = hipFree(d_pv);
        let _ = hipFree(d_pi);
    }
    let _ = ModelGeometry::default();
}

#[test]
fn test_dfctx_and_drafter_pick_depth_policy() {
    let mut dctx = DfCtx::default();
    dctx.init(2048, 4096, 8);
    assert!(dctx.valid(0));
    dctx.at(0);
    dctx.inject(0, 512, 512);
    assert_eq!(dctx.lo(), 0);
    assert_eq!(dctx.hi(), 512);
    assert!(dctx.valid(512));
    assert!(!dctx.valid(513));
    dctx.rewind(256);
    assert_eq!(dctx.hi(), 256);
    assert!(dctx.valid(256));
    assert!(!dctx.valid(300));

    let mut dp = DrafterPick::default();
    dp.init("gfx1151", true, "auto");
    assert_eq!(dp.mode(), DrafterPick::K_AUTO);
    assert!(dp.uses_mtp());
    assert!(dp.want_dfl(1024, false));
    assert!(!dp.want_dfl(20000, false));
    assert!(!dp.mtp_rows_in_dfl(1024, false));
    assert!(dp.mtp_rows_in_dfl(15000, false));
    dp.observe(true, 7, 5, 1024, false);
    assert_eq!(dp.n_dfl, 1);
    dp.fallback(2048);
    assert_eq!(dp.n_fallback, 1);
}
