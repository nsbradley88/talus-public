//! MiMo-V2.6-Flash (`mimo2`) model adapter (WP-3.3, Phase H).
//!
//! Built on the Laguna family (`crate::laguna`), the WP-1.6 MiMo HIP kernels
//! (`tk_attn_swa_sinks`, `tk_attn_k192_v128`, `tk_mmvq_ids`, `tk_prefill_moe_mmq`),
//! the 11 GiB `MemAvailable` floor from `talus-mem` ([`MimoBudgetExt::for_mimo_11gib`]),
//! and the WP-3.2 DFlash speculation engine (`talus-spec`, 152,064 `Q8_0` head +
//! 64K `mimo-dv-65536.bin` draft vocabulary + `n4`).

pub mod engine;
pub mod ffi;
pub mod serve;
pub mod session;
pub mod weights;

pub use engine::{
    compute_mimo_kv_cache_bytes, is_mimo_gguf, route_top8_experts, AllocationStep,
    GenerationResult, MimoBudgetExt, MimoBudgetGuard, MimoDFlashProfile, MimoDFlashRunner,
    MimoDFlashSpec, MimoDispatchTrace, MimoEngine, MimoEngineConfig, MimoSpecStats,
};
pub use serve::{serve_loop, serve_main};
pub use session::{SessionHeader, SESSION_MAGIC_KV2, SESSION_MAGIC_MM1, SESSION_VERSION};
pub use weights::{
    build_synthetic_mimo_dflash_gguf_header, build_synthetic_mimo_ud_gguf_shards,
    default_model_path, is_canonical_swa_layer, is_moe_layer, models_dir,
    resolve_mimo_draft_vocab_path, DevTensor, MimoAttentionKind, MimoCatalog, MimoDFlashCatalog,
    MimoDFlashLayerWeights, MimoDFlashWeights, MimoDraftVocabInfo, MimoFfnCatalog, MimoFfnKind,
    MimoHparams, MimoLayerCatalog, MimoLayerWeights, MimoModel, MimoQkvCatalog, MimoTensorLoc,
    ARCH_NAME, DEFAULT_ATTN_SCALE, DEFAULT_MIMO_DFLASH_PATH, DEFAULT_MIMO_DRAFT_VOCAB_PATH,
    DEFAULT_MIMO_MOPD_DFLASH_PATH, DEFAULT_MIMO_UD_SHARD1_PATH, DEFAULT_MIMO_UD_SHARD2_PATH,
    DFLASH_5L_EXPECTED_TENSOR_COUNT, DFLASH_5L_TARGET_TAPS, DFLASH_8L_TARGET_TAPS,
    DFLASH_ARCH_NAME, DFLASH_BLOCK_SIZE, DFLASH_DEFAULT_DRAFT_MAX, FULL_ATTN_LAYER_IDS, HEAD_DIM_K,
    HEAD_DIM_V, LOCAL_MIMO_DFLASH_HDR_PATH, LOCAL_MIMO_DRAFT_VOCAB_PATH,
    LOCAL_MIMO_MOPD_HDR_SHARDS, MIMO_DRAFT_VOCAB_SIZE, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_DENSE,
    N_FF_EXPERT_MOPD, N_FF_EXPERT_UD, N_FULL_LAYERS, N_HEAD_KV_FULL, N_HEAD_KV_SWA, N_HEAD_Q,
    N_LAYERS, N_SWA_LAYERS, N_VOCAB_MOPD, N_VOCAB_UD, RMS_NORM_EPS, ROPE_DIM, ROPE_FREQ_BASE_FULL,
    ROPE_FREQ_BASE_SWA, SWA_WINDOW, UD_EXPECTED_TENSOR_COUNT,
};

use talus_core_api::{
    Arch, CoreError, GraphBuilder, GraphKey, HiddenTapKind, HiddenTaps, KvCacheFormat, KvLayout,
    LayerKind, Loader, LogitRows, ModelAdapter, ModelDesc, Result, SeqState, StateLayout,
    StreamHandle, WeightSource, WindowIo,
};

/// Build the canonical [`ModelDesc`] for MiMo-V2.6-Flash (`general.architecture == "mimo2"`).
pub fn build_mimo_model_desc(hparams: &MimoHparams) -> ModelDesc {
    let layer_kinds = (0..hparams.n_layers)
        .map(|il| {
            if is_canonical_swa_layer(il) {
                LayerKind::AttnSwa
            } else {
                LayerKind::Attn
            }
        })
        .collect();
    ModelDesc {
        arch: ARCH_NAME.to_string(),
        n_embd: hparams.n_embd as u32,
        n_layers: hparams.n_layers as u32,
        layer_kinds,
        qsa_interval: 6,
        n_head: hparams.n_head_q as u32,
        n_head_per_layer: vec![hparams.n_head_q as u32; hparams.n_layers],
        n_head_kv: hparams.n_head_kv_swa as u32,
        head_dim: hparams.head_dim_k as u32,
        head_dim_v: hparams.head_dim_v as u32,
        n_rot: hparams.rope_dim as u32,
        sliding_window: hparams.sliding_window as u32,
        rope_freq_base_swa: f64::from(hparams.rope_freq_base_swa),
        n_rot_swa: hparams.rope_dim as u32,
        rms_eps: f64::from(hparams.rms_norm_eps),
        rope_freq_base: f64::from(hparams.rope_freq_base_full),
        n_ff_dense: hparams.n_ff_dense as u32,
        n_dense_lead: 1,
        n_expert: hparams.n_expert as u32,
        n_expert_used: hparams.n_expert_used as u32,
        n_ff: hparams.n_ff_exp as u32,
        expert_weights_norm: hparams.expert_weights_norm,
        expert_weights_scale: 1.0,
        n_vocab: hparams.n_vocab as u32,
        n_ctx_train: 262_144,
        ..Default::default()
    }
}

/// `ModelAdapter` implementation for MiMo-V2.6-Flash (`general.architecture == "mimo2"`).
#[derive(Debug, Clone)]
pub struct MimoAdapter {
    desc: ModelDesc,
    max_seq_len: u32,
    last_window_len: u32,
}

impl Default for MimoAdapter {
    fn default() -> Self {
        Self::new(MimoHparams::default(), 512)
    }
}

impl MimoAdapter {
    /// Construct a new `MimoAdapter` from `hparams` and `max_seq_len`.
    pub fn new(hparams: MimoHparams, max_seq_len: u32) -> Self {
        Self {
            desc: build_mimo_model_desc(&hparams),
            max_seq_len,
            last_window_len: 0,
        }
    }
}

impl Arch for MimoAdapter {
    const NAME: &'static str = ARCH_NAME;

    fn describe(g: &talus_core_api::GgufView) -> Result<ModelDesc> {
        if g.architecture != ARCH_NAME {
            return Err(CoreError::UnsupportedArch {
                arch: g.architecture.clone(),
                supported: &[ARCH_NAME],
            });
        }
        Ok(build_mimo_model_desc(&MimoHparams::default()))
    }

    fn load(d: &ModelDesc, _src: &WeightSource, _ld: &mut dyn Loader) -> Result<Self> {
        if d.arch != ARCH_NAME {
            return Err(CoreError::UnsupportedArch {
                arch: d.arch.clone(),
                supported: &[ARCH_NAME],
            });
        }
        Ok(Self {
            desc: d.clone(),
            max_seq_len: d.n_ctx_train.max(512),
            last_window_len: 0,
        })
    }
}

impl ModelAdapter for MimoAdapter {
    fn desc(&self) -> &ModelDesc {
        &self.desc
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout {
            kv: KvLayout {
                num_layers: self.desc.n_layers,
                n_head_kv: self.desc.n_head_kv,
                head_dim_k: self.desc.head_dim,
                head_dim_v: self.desc.head_dim_v,
                max_seq_len: self.max_seq_len,
                ring_window: Some(self.desc.sliding_window),
                kv_pad_stride: 256,
                vt_tile_stride: Some(16),
                format: KvCacheFormat::F16,
            },
            recurrent: None,
            indexer_bytes_per_pos: 0,
            ple_history_bytes: 0,
        }
    }

    fn prefill(
        &mut self,
        st: &mut SeqState,
        ids: &[u32],
        pos0: u32,
        _want: LogitRows,
        _s: &StreamHandle,
    ) -> Result<()> {
        if ids.is_empty() {
            return Err(CoreError::InvalidArgument {
                reason: "prefill prompt tokens must be non-empty".to_string(),
            });
        }
        st.pos = pos0 + ids.len() as u32;
        Ok(())
    }

    fn window(&mut self, _st: &mut SeqState, w: &WindowIo, _s: &StreamHandle) -> Result<()> {
        if w.window_len == 0 || w.window_len > 16 {
            return Err(CoreError::InvalidArgument {
                reason: format!("window_len {} out of range 1..=16", w.window_len),
            });
        }
        self.last_window_len = w.window_len;
        Ok(())
    }

    fn commit(&mut self, st: &mut SeqState, n_keep: u32, _s: &StreamHandle) -> Result<()> {
        if n_keep > self.last_window_len {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "n_keep {} exceeds last_window_len {}",
                    n_keep, self.last_window_len
                ),
            });
        }
        st.pos += n_keep;
        self.last_window_len = 0;
        Ok(())
    }

    fn hidden_taps(&self) -> HiddenTaps {
        let taps: Vec<HiddenTapKind> = DFLASH_5L_TARGET_TAPS
            .iter()
            .map(|&l| HiddenTapKind::LayerPostFfn(l as u32))
            .collect();
        let n = taps.len();
        HiddenTaps {
            taps,
            n_embd: self.desc.n_embd,
            device_ptrs: vec![0; n],
        }
    }

    fn capture(&mut self, _key: GraphKey, g: &mut GraphBuilder) -> Result<()> {
        for _ in 0..self.desc.n_layers {
            g.record_launch();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::Arc;
    use talus_mem::{Budget, BudgetCategory};

    #[test]
    fn test_mimo_catalog_validates_821_tensor_ud_2shard_and_on_disk_hdr() {
        // 1. Validate the canonical 2-shard MiMo-V2.6-Flash-UD-IQ2_XXS header (821 tensors,
        //    40 SWA layers `i % 6 != 5` + 8 full-attn layers `i % 6 == 5`, Layer 0 dense +
        //    Layers 1..47 256-expert top-8 MoE).
        let strix_shard1 = default_model_path(DEFAULT_MIMO_UD_SHARD1_PATH);
        let (s1, s2) = build_synthetic_mimo_ud_gguf_shards().expect("build UD 2-shard headers");
        let total_synth = s1.tensors().len() + s2.tensors().len();
        assert_eq!(total_synth, UD_EXPECTED_TENSOR_COUNT);
        for t in s1.tensors().iter().chain(s2.tensors().iter()) {
            assert!(
                !t.name.contains("mtp_aux"),
                "must not contain fake mtp_aux tensor: {}",
                t.name
            );
        }
        for mtp_il in 48..51 {
            assert!(
                s2.find_tensor(&format!("blk.{mtp_il}.nextn.eh_proj.weight"))
                    .is_some(),
                "missing blk.{mtp_il}.nextn.eh_proj.weight"
            );
            assert!(
                s2.find_tensor(&format!("blk.{mtp_il}.layer_output_norm.weight"))
                    .is_some(),
                "missing blk.{mtp_il}.layer_output_norm.weight"
            );
        }
        let catalog = if strix_shard1.exists() {
            MimoCatalog::open_shards(&[strix_shard1]).expect("open default UD 2-shard GGUF")
        } else {
            MimoCatalog::from_shards(&[&s1, &s2]).expect("validate UD 2-shard headers")
        };

        assert_eq!(catalog.n_shards, 2);
        assert_eq!(catalog.total_tensors, UD_EXPECTED_TENSOR_COUNT); // 821 tensors
        assert_eq!(catalog.hparams.n_layers, 48);
        assert_eq!(catalog.hparams.n_embd, 4096);
        assert_eq!(catalog.hparams.n_ff_dense, 16384);
        assert_eq!(catalog.hparams.n_ff_exp, 1280);
        assert_eq!(catalog.hparams.n_expert, 256);
        assert_eq!(catalog.hparams.n_expert_used, 8);
        assert_eq!(catalog.hparams.n_vocab, 152_064);
        assert_eq!(catalog.hparams.n_head_q, 64);
        assert_eq!(catalog.hparams.n_head_kv_swa, 8);
        assert_eq!(catalog.hparams.n_head_kv_full, 4);
        assert_eq!(catalog.hparams.head_dim_k, 192);
        assert_eq!(catalog.hparams.head_dim_v, 128);
        assert_eq!(catalog.hparams.rope_dim, 64);
        assert_eq!(catalog.hparams.sliding_window, 128);
        assert!((catalog.hparams.rms_norm_eps - 1e-6).abs() < 1e-10);
        assert!(catalog.hparams.expert_weights_norm);

        // Verify 40 SWA layers (`i % 6 != 5`) and 8 full-attention layers (`i % 6 == 5`).
        assert_eq!(catalog.swa_layer_count(), 40);
        assert_eq!(catalog.full_attn_layer_count(), 8);
        assert_eq!(catalog.dense_layer_count(), 1);
        assert_eq!(catalog.moe_layer_count(), 47);

        for il in 0..48 {
            let l = &catalog.layers[il];
            if il % 6 == 5 {
                assert!(!l.is_swa(), "layer {il} must be Full-attention");
                assert_eq!(l.n_head_kv, 4);
                assert!(l.attn_sinks.is_none(), "full-attn layer {il} has no sinks");
                assert!((l.rope_freq_base() - 64_000_000.0).abs() < 1.0);
            } else {
                assert!(l.is_swa(), "layer {il} must be SWA-128");
                assert_eq!(l.n_head_kv, 8);
                assert!(
                    l.attn_sinks.is_some(),
                    "SWA layer {il} must have attn_sinks"
                );
                assert_eq!(l.sliding_window, 128);
                assert!((l.rope_freq_base() - 10_000.0).abs() < 1e-3);
            }
            assert!(l.post_attention_norm.is_some());
            assert!(l.post_ffn_norm.is_some());
            if il == 0 {
                assert!(!l.is_moe(), "layer 0 must be dense FFN");
            } else {
                assert!(l.is_moe(), "layer {il} must be 256-expert MoE");
            }
        }

        // 2. Also validate the real on-disk 3-shard MOPD-a4 `.gguf.hdr` dump if present under TALUS_SHAPES_DIR.
        let local_hdr1 = std::env::var_os("TALUS_SHAPES_DIR")
            .map(|d| {
                std::path::PathBuf::from(d)
                    .join("hdr")
                    .join(LOCAL_MIMO_MOPD_HDR_SHARDS[0])
            })
            .unwrap_or_else(|| std::path::PathBuf::from(LOCAL_MIMO_MOPD_HDR_SHARDS[0]));
        if local_hdr1.exists() {
            let mopd =
                MimoCatalog::open_shards(&[&local_hdr1]).expect("open on-disk MOPD 3-shard hdr");
            assert_eq!(mopd.n_shards, 3);
            assert_eq!(mopd.total_tensors, 508);
            assert_eq!(mopd.hparams.n_layers, 48);
            assert_eq!(mopd.hparams.n_embd, 4096);
            assert_eq!(mopd.hparams.n_ff_dense, 16384);
            assert_eq!(mopd.hparams.n_ff_exp, 2048);
            assert_eq!(mopd.hparams.n_expert, 256);
            assert_eq!(mopd.hparams.n_expert_used, 8);
            assert_eq!(mopd.dense_layer_count(), 1);
            assert_eq!(mopd.moe_layer_count(), 47);
        }
    }

    #[test]
    fn test_mimo_dflash_catalog_59_tensors_and_64k_draft_vocab() {
        // 1. Validate the 59-tensor 5-layer MiMo-V2-Flash-DFlash-Q8_0.gguf header.
        let strix_dflash = default_model_path(DEFAULT_MIMO_DFLASH_PATH);
        let df_cat = if strix_dflash.exists() {
            MimoDFlashCatalog::open(strix_dflash).expect("open default DFlash GGUF")
        } else {
            let gguf =
                build_synthetic_mimo_dflash_gguf_header().expect("build 59-tensor DFlash GGUF");
            MimoDFlashCatalog::from_gguf(&gguf).expect("validate 59-tensor DFlash GGUF")
        };

        assert_eq!(df_cat.total_tensors, DFLASH_5L_EXPECTED_TENSOR_COUNT); // 59 tensors
        assert_eq!(df_cat.block_count, 5);
        assert_eq!(df_cat.embedding_length, 4096);
        assert_eq!(df_cat.feed_forward_length, 20480);
        assert_eq!(df_cat.n_head, 64);
        assert_eq!(df_cat.n_head_kv, 8);
        assert_eq!(df_cat.k_dim, 192);
        assert_eq!(df_cat.v_dim, 128);
        assert_eq!(df_cat.rope_dim, 64);
        assert!((df_cat.rope_freq_base - 64_000_000.0).abs() < 1.0);
        assert_eq!(df_cat.block_size, 16);
        assert_eq!(df_cat.target_layers, vec![1, 12, 24, 36, 47]);
        assert_eq!(df_cat.n_vocab, 152_064);
        assert_eq!(df_cat.output_weight.qtype, talus_core_api::QuantType::Q8_0);
        assert_eq!(df_cat.output_weight.shape(), &[4096, 152_064]);
        assert_eq!(df_cat.output_norm.shape(), &[4096]);

        // 2. Also validate the real on-disk `dflash-MiMo-V2.6-Flash-MOPD-Q8_0.gguf.hdr` if present.
        let local_dflash_hdr = std::env::var_os("TALUS_SHAPES_DIR")
            .map(|d| {
                std::path::PathBuf::from(d)
                    .join("hdr")
                    .join(LOCAL_MIMO_DFLASH_HDR_PATH)
            })
            .unwrap_or_else(|| std::path::PathBuf::from(LOCAL_MIMO_DFLASH_HDR_PATH));
        if local_dflash_hdr.exists() {
            let mopd_df =
                MimoDFlashCatalog::open(&local_dflash_hdr).expect("open on-disk DFlash hdr");
            assert_eq!(mopd_df.total_tensors, 65);
            assert_eq!(mopd_df.block_count, 5);
            assert_eq!(mopd_df.embedding_length, 4096);
            assert_eq!(mopd_df.feed_forward_length, 16384);
            assert_eq!(mopd_df.target_layers, vec![1, 12, 24, 36, 48]);
            assert_eq!(mopd_df.output_weight.qtype, talus_core_api::QuantType::Q8_0);
        }

        // 3. Validate the real 64K draft vocabulary (`mimo-dv-65536.bin`) on disk and wire into
        //    `MimoDFlashProfile::mimo_v2_flash()` + `MimoDFlashSpec::mimo_v2_flash`.
        let profile = MimoDFlashProfile::mimo_v2_flash();
        assert_eq!(profile.block_count, 5);
        assert_eq!(profile.target_layer_taps, vec![1, 12, 24, 36, 47]);
        assert_eq!(profile.target_vocab_size, 152_064);
        assert_eq!(profile.draft_vocab_size, Some(65_536));
        assert_eq!(profile.default_draft_max, 4);

        if let Some(dv_path) = resolve_mimo_draft_vocab_path() {
            let info = MimoDraftVocabInfo::from_path(&dv_path, N_VOCAB_UD)
                .expect("validate real mimo-dv-65536.bin");
            assert_eq!(info.draft_vocab_size, 65_536);
            assert_eq!(info.min_target_id, 0);
            assert!(info.max_target_id < N_VOCAB_MOPD as i32);
            assert!(info.is_sorted_unique);

            let spec = MimoDFlashSpec::mimo_v2_flash(
                default_model_path(DEFAULT_MIMO_DFLASH_PATH),
                &dv_path,
            )
            .expect("build MimoDFlashSpec::mimo_v2_flash");
            assert_eq!(spec.active_vocab_size(), 65_536);
            assert_eq!(spec.map_draft_to_target(0).unwrap(), 0);
        }
    }

    #[test]
    fn test_budget_for_mimo_11gib_ordering_and_floor_enforcement() {
        use crate::mimo::engine::{GIB, MIMO_FLOOR_BYTES};

        // 1. Verify `Budget::for_mimo_11gib(None)` configures the 11 GiB floor.
        let default_budget = Budget::for_mimo_11gib(None);
        assert_eq!(default_budget.floor_bytes(), MIMO_FLOOR_BYTES);
        assert_eq!(MIMO_FLOOR_BYTES, 11 * GIB);

        // 2. Verify reservation ordering: `Weights` and `KvCache` are reserved strictly BEFORE `hipMalloc`.
        let avail_20gib = 20 * GIB;
        let engine = MimoEngine::new_synthetic(
            MimoHparams::synthetic_tiny(),
            MimoEngineConfig::default(),
            Some(avail_20gib),
        )
        .expect("synthetic engine within 20 GiB budget");

        let guard = engine.budget_guard();
        assert!(
            guard.verify_reservation_precedes_hip_malloc(),
            "Weights and KvCache must be reserved before any hipMalloc: {:?}",
            guard.steps()
        );
        assert_eq!(
            guard.steps()[0],
            AllocationStep::BudgetReserve {
                category: BudgetCategory::Weights,
                bytes: 16 * 1024 * 1024,
            }
        );
        assert!(matches!(
            guard.steps()[1],
            AllocationStep::BudgetReserve {
                category: BudgetCategory::KvCache,
                ..
            }
        ));

        // 3. Verify that attempting to reserve when `MemAvailable` would drop below 11 GiB fails with `OutOfMemory`.
        let avail_11_5_gib = 11 * GIB + 512 * 1024 * 1024;
        let tight_budget = Arc::new(Budget::for_mimo_11gib(Some(avail_11_5_gib)));
        let err = MimoBudgetGuard::reserve_before_hip_malloc(
            tight_budget,
            GIB, // 1 GiB weights -> remaining 10.5 GiB < 11 GiB floor!
            64 * 1024 * 1024,
            0,
            16 * 1024 * 1024,
        )
        .expect_err("must reject reservation that breaches 11 GiB floor");

        match err {
            CoreError::OutOfMemory {
                requested_bytes, ..
            } => {
                assert_eq!(requested_bytes, GIB as u64);
            }
            other => panic!("expected OutOfMemory, got {other:?}"),
        }
    }

    #[test]
    fn test_hybrid_swa128_sinks_k192_v128_moe256_top8_and_dflash_n4() {
        use crate::mimo::engine::GIB;
        // Test the full 48-layer canonical `mimo2` topology with scaled hidden/vocab dimensions
        // so all 40 SWA layers + 8 Full-attention layers + Layer 0 dense + 47 MoE layers +
        // DFlash `n4` taps `[1, 12, 24, 36, 47]` execute deterministically.
        let mut hparams = MimoHparams::default();
        hparams.n_embd = 64;
        hparams.n_ff_dense = 128;
        hparams.n_ff_exp = 64;
        hparams.n_vocab = 256;

        let cfg = MimoEngineConfig {
            max_seq_len: 512,
            draft_max: 4, // DFlash n4
            ..Default::default()
        };

        let mut engine = MimoEngine::new_synthetic(hparams, cfg, Some(32 * GIB))
            .expect("create 48-layer synthetic MimoEngine");

        // Verify non-null `vzero` pointer containing `0` for `n_expert_used == 1` `tk_mmvq_ids`.
        let vzero = engine.vzero_ptr();
        assert!(!vzero.is_null());
        assert_eq!(unsafe { *vzero }, 0);

        // 1. Decode / verify batch (`n_tok = 4 <= 8`) -> dispatches `tk_mmvq_ids` on 47 MoE layers.
        let logits = engine
            .forward_batch(&[10, 20, 30, 40])
            .expect("forward_batch decode/verify");
        assert_eq!(logits.len(), 4);
        assert_eq!(logits[0].len(), 256);

        let trace = engine.trace().clone();
        assert_eq!(trace.swa_sinks_calls, 40);
        assert_eq!(trace.full_k192_v128_calls, 8);
        assert_eq!(trace.dense_ffn_calls, 1);
        assert_eq!(trace.moe_topk_calls, 47);
        assert_eq!(trace.moe_mmvq_ids_calls, 47 * 3); // gate_exps, up_exps, down_exps
        assert_eq!(trace.moe_prefill_mmq_calls, 0);
        assert!(trace.non_null_vzero_mmvq_calls >= 52);
        assert_eq!(trace.captured_dflash_taps, vec![1, 12, 24, 36, 47]);

        // 2. Prefill batch (`T = 12 > 8`) -> dispatches `tk_prefill_moe_mmq` on 47 MoE layers.
        engine.reset();
        let prefill_tokens: Vec<i32> = (1..=12).collect();
        let prefill_logits = engine
            .forward_batch(&prefill_tokens)
            .expect("forward_batch prefill T=12");
        assert_eq!(prefill_logits.len(), 12);
        let ptrace = engine.trace().clone();
        assert_eq!(ptrace.swa_sinks_calls, 40);
        assert_eq!(ptrace.full_k192_v128_calls, 8);
        assert_eq!(ptrace.dense_ffn_calls, 1);
        assert_eq!(ptrace.moe_topk_calls, 47);
        assert_eq!(ptrace.moe_prefill_mmq_calls, 47 * 3);

        // 3. Verify 256-expert top-8 router normalization (`expert_weights_norm = true`).
        let (ids, weights) = route_top8_experts(&[0.5; 64], 256, 8);
        assert_eq!(ids.len(), 8);
        assert_eq!(weights.len(), 8);
        let wsum: f32 = weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-5);
        for &id in &ids {
            assert!((0..256).contains(&id));
        }

        // 4. End-to-end generation with DFlash `n4` speculation.
        engine.reset();
        let gen = engine
            .generate(&[1, 2, 3, 4], 10)
            .expect("generate 10 tokens with DFlash n4");
        assert_eq!(gen.tokens.len(), 10);
        let stats = gen.spec_stats.expect("DFlash n4 stats must be recorded");
        assert!(stats.verify_steps >= 1);
        assert_eq!(stats.draft_proposed, stats.verify_steps * 4);
        assert!(stats.total_emitted() >= stats.verify_steps);
    }

    #[test]
    fn test_session_header_and_serve_protocol() {
        use crate::mimo::engine::GIB;
        let hdr = SessionHeader::new(128, 4096, 128, 0xCAFE_BABE_1234_5678);
        let bytes = hdr.to_bytes();
        assert_eq!(&bytes[0..8], b"TALUSMM1");
        let parsed = SessionHeader::from_bytes(&bytes).expect("parse TALUSMM1");
        assert_eq!(parsed, hdr);

        let kv2 = SessionHeader::new_kv2(64, 8192, 64, 99);
        let kv2_parsed = SessionHeader::from_bytes(&kv2.to_bytes()).expect("parse TALUSKV2");
        assert_eq!(kv2_parsed, kv2);

        let mut engine = MimoEngine::new_synthetic(
            MimoHparams::synthetic_tiny(),
            MimoEngineConfig::default(),
            Some(24 * GIB),
        )
        .expect("create synthetic engine");

        let input = b"RESET\nSEED 123\nMTP 4\nGEN 4 0.0 1.0 1 3 10 20 30\nQUIT\n";
        let mut out = Vec::new();
        serve_loop(&mut engine, Cursor::new(&input[..]), &mut out).expect("serve_loop");
        let text = String::from_utf8(out).expect("utf8 output");
        assert!(text.contains("READY 512 4096"), "output: {text}");
        assert!(text.contains("TOK "), "output: {text}");
        assert!(text.contains("DONE "), "output: {text}");
        assert!(text.contains("BYE"), "output: {text}");
    }

    #[test]
    fn test_mimo_adapter_implements_model_adapter_and_arch() {
        use talus_core_api::DeviceId;

        let mut view = talus_core_api::GgufView::default();
        view.architecture = ARCH_NAME.to_string();
        let desc = MimoAdapter::describe(&view).expect("describe mimo2");
        assert_eq!(desc.arch, "mimo2");
        assert_eq!(desc.n_layers, 48);
        assert_eq!(desc.n_embd, 4096);
        assert_eq!(desc.n_expert, 256);
        assert_eq!(desc.n_expert_used, 8);
        assert_eq!(desc.n_vocab, 152_064);

        let mut adapter = MimoAdapter::new(MimoHparams::default(), 4096);
        assert_eq!(adapter.desc().arch, "mimo2");
        let layout = adapter.state_layout();
        assert_eq!(layout.kv.num_layers, 48);
        assert_eq!(layout.kv.head_dim_k, 192);
        assert_eq!(layout.kv.head_dim_v, 128);
        assert_eq!(layout.kv.ring_window, Some(128));
        assert!(layout.recurrent.is_none());

        let taps = adapter.hidden_taps();
        assert_eq!(
            taps.taps,
            vec![
                HiddenTapKind::LayerPostFfn(1),
                HiddenTapKind::LayerPostFfn(12),
                HiddenTapKind::LayerPostFfn(24),
                HiddenTapKind::LayerPostFfn(36),
                HiddenTapKind::LayerPostFfn(47),
            ]
        );

        let stream = StreamHandle::null(DeviceId(0));
        let mut seq = SeqState::new(1, 4096, DeviceId(0));
        adapter
            .prefill(&mut seq, &[10, 20, 30], 0, LogitRows::Last, &stream)
            .expect("prefill");
        assert_eq!(seq.pos, 3);

        let win = WindowIo {
            tokens: vec![30, 31, 32, 33, 34],
            pos0: 3,
            window_len: 5,
            device_tokens_ptr: 0,
            device_pos_ptr: 0,
            device_logits_ptr: 0,
        };
        adapter.window(&mut seq, &win, &stream).expect("window");
        adapter.commit(&mut seq, 4, &stream).expect("commit");
        assert_eq!(seq.pos, 7);

        let key = GraphKey {
            kind: "mimo2_verify".to_string(),
            window_len: 5,
            draft_depth: 4,
            device: DeviceId(0),
        };
        let mut gb = GraphBuilder::new(key, stream);
        adapter
            .capture(gb.key.clone(), &mut gb)
            .expect("capture graph");
        assert_eq!(gb.recorded_nodes, 48);
    }
}
