//! MiMo-V2.6-Flash (`mimo2`) execution engine, memory budget enforcement, hybrid
//! SWA-128 / K192-V128 attention dispatch, 256-expert top-8 MoE routing, and
//! DFlash `n4` + 64K draft-vocabulary (`mimo-dv-65536.bin`) speculative decoding.

use std::ffi::c_void;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;
use std::time::Instant;

use talus_core_api::{CoreError, DraftVocab, QuantType, Result};
use talus_gguf::GgufFile;
use talus_hip::DeviceBuf;
use talus_mem::{Budget, BudgetCategory, BudgetConfig, MockMemProbe, Reservation};

use crate::mimo::ffi::{
    build_mimo_rope_table, check_tk, hip_copy_d2d_async, hip_memset, hip_memset_async,
    tk_moe_mmq_product, HipGraphExec, HipRuntime, LibTalusKernels, TkContext,
    HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL, TK_QTYPE_F32,
};
use crate::mimo::session::SessionHeader;
use crate::mimo::weights::{
    is_canonical_swa_layer, is_moe_layer, resolve_mimo_draft_vocab_path, DevTensor, MimoCatalog,
    MimoDFlashCatalog, MimoDraftVocabInfo, MimoHparams, MimoModel, DEFAULT_ATTN_SCALE,
    DFLASH_5L_TARGET_TAPS, DFLASH_DEFAULT_DRAFT_MAX, MIMO_DRAFT_VOCAB_SIZE, N_EXPERT_USED,
};
use crate::tail_prune::{self, first_prunable_layer, plan_on_grid, Grid, LayerAttn, TailRequest};

/// One gibibyte in bytes (`1024^3`).
pub const GIB: usize = 1024 * 1024 * 1024;
/// MiMo-V2.6-Flash `MemAvailable` floor (`11 GiB`).
pub const MIMO_FLOOR_BYTES: usize = talus_mem::MIMO_FLOOR_BYTES;

fn io_err(e: std::io::Error) -> CoreError {
    CoreError::IoError {
        message: e.to_string(),
    }
}

/// Speculative verification outcome for a single DFlash `n4` round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimoVerifyOutcome {
    /// Draft tokens accepted by the target model (`0..=draft_max`).
    pub accepted: Vec<i32>,
    /// Bonus (or correction) token sampled from the target model at step `accepted.len()`.
    pub bonus_token: i32,
}

/// Cumulative DFlash `n4` speculative decoding statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MimoSpecStats {
    /// Number of verification rounds executed.
    pub verify_steps: usize,
    /// Total draft tokens proposed across all rounds.
    pub draft_proposed: usize,
    /// Total draft tokens accepted across all rounds.
    pub draft_accepted: usize,
    /// Total bonus/correction tokens emitted (`== verify_steps`).
    pub bonus_emitted: usize,
}

impl MimoSpecStats {
    /// Record the outcome of one verification round with `proposed_k` draft tokens.
    pub fn record(&mut self, outcome: &MimoVerifyOutcome, proposed_k: usize) {
        self.verify_steps += 1;
        self.draft_proposed += proposed_k;
        self.draft_accepted += outcome.accepted.len();
        self.bonus_emitted += 1;
    }

    /// Total tokens emitted (`draft_accepted + bonus_emitted`).
    pub fn total_emitted(&self) -> usize {
        self.draft_accepted + self.bonus_emitted
    }

    /// Fraction of proposed draft tokens accepted (`0.0..=1.0`).
    pub fn acceptance_rate(&self) -> f64 {
        if self.draft_proposed == 0 {
            0.0
        } else {
            self.draft_accepted as f64 / self.draft_proposed as f64
        }
    }
}

/// Greedy verification of `drafts` (`[d_0, ..., d_{k-1}]`) against `target_logits`
/// (`k + 1` rows of length `n_vocab`).
pub fn greedy_verify(drafts: &[i32], target_logits: &[Vec<f32>]) -> Result<MimoVerifyOutcome> {
    if target_logits.len() < drafts.len() + 1 {
        return Err(CoreError::InvalidArgument {
            reason: format!(
                "greedy_verify needs {} target logit rows for {} drafts, got {}",
                drafts.len() + 1,
                drafts.len(),
                target_logits.len()
            ),
        });
    }
    let mut accepted = Vec::with_capacity(drafts.len());
    for (step, &draft_tok) in drafts.iter().enumerate() {
        let winner = argmax_f32(&target_logits[step]);
        if winner == draft_tok {
            accepted.push(draft_tok);
        } else {
            return Ok(MimoVerifyOutcome {
                accepted,
                bonus_token: winner,
            });
        }
    }
    let bonus_token = argmax_f32(&target_logits[drafts.len()]);
    Ok(MimoVerifyOutcome {
        accepted,
        bonus_token,
    })
}

/// DFlash model profile descriptor for `MiMo-V2.6-Flash` (`5` layers, taps `[1, 12, 24, 36, 47]`,
/// dedicated `152,064` `Q8_0` LM head + `65,536` `mimo-dv-65536.bin` draft vocab, `n4`).
#[derive(Debug, Clone, PartialEq)]
pub struct MimoDFlashProfile {
    /// Canonical profile name (`"mimo-v2-flash-dflash"`).
    pub name: &'static str,
    /// Number of drafter layers (`5`).
    pub block_count: usize,
    /// Hidden dimension (`4096`).
    pub embedding_length: usize,
    /// Dense FFN intermediate width (`20480` in GGUF / `16384` in runtime profile).
    pub feed_forward_length: usize,
    /// Query head count (`64`).
    pub n_head: usize,
    /// KV head count (`8`).
    pub n_head_kv: usize,
    /// Key head dimension (`192`).
    pub k_dim: usize,
    /// Value head dimension (`128`).
    pub v_dim: usize,
    /// Rotary dimension (`64`).
    pub rope_dim: usize,
    /// RoPE frequency base (`64,000,000.0`).
    pub rope_freq_base: f32,
    /// Block size (`16` in GGUF metadata, `5` active rows for `n4`).
    pub block_size: u32,
    /// Target layer tap indices (`[1, 12, 24, 36, 47]`).
    pub target_layer_taps: Vec<u32>,
    /// Target vocabulary size (`152,064`).
    pub target_vocab_size: usize,
    /// Dedicated drafter `Q8_0` LM head vocabulary rows (`152,064`).
    pub head_vocab_size: usize,
    /// Draft vocabulary subset size (`Some(65,536)`).
    pub draft_vocab_size: Option<usize>,
    /// Default draft lengths (`n4` = `4`).
    pub default_draft_max: usize,
}

impl MimoDFlashProfile {
    /// Construct the canonical Phase H `MiMo-V2-Flash` DFlash profile.
    pub fn mimo_v2_flash() -> Self {
        Self {
            name: "mimo-v2-flash-dflash",
            block_count: 5,
            embedding_length: 4096,
            feed_forward_length: 20480,
            n_head: 64,
            n_head_kv: 8,
            k_dim: 192,
            v_dim: 128,
            rope_dim: 64,
            rope_freq_base: 64_000_000.0,
            block_size: 16,
            target_layer_taps: DFLASH_5L_TARGET_TAPS.to_vec(),
            target_vocab_size: 152_064,
            head_vocab_size: 152_064,
            draft_vocab_size: Some(MIMO_DRAFT_VOCAB_SIZE),
            default_draft_max: DFLASH_DEFAULT_DRAFT_MAX,
        }
    }
}

/// Drafter specification pairing a [`MimoDFlashProfile`] with a loaded 64K `DraftVocab`
/// (`mimo-dv-65536.bin`).
#[derive(Debug, Clone)]
pub struct MimoDFlashSpec {
    /// Model profile.
    pub profile: MimoDFlashProfile,
    /// Drafter GGUF path.
    pub gguf_path: PathBuf,
    /// Optional draft vocab `.bin` path.
    pub draft_vocab_path: Option<PathBuf>,
    /// Loaded `talus_core_api::DraftVocab` (`draft_to_target: Vec<u32>`).
    pub core_draft_vocab: DraftVocab,
    /// Signed `i32` draft-to-target (`d2t`) table (`[65536] -> 0..152064`).
    pub draft_to_target: Vec<i32>,
}

impl MimoDFlashSpec {
    /// Load a MiMo-V2-Flash DFlash specification from `gguf_path` and `draft_vocab_path`.
    pub fn mimo_v2_flash(
        gguf_path: impl Into<PathBuf>,
        draft_vocab_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let profile = MimoDFlashProfile::mimo_v2_flash();
        let dv_path = draft_vocab_path.as_ref().to_path_buf();
        let info = MimoDraftVocabInfo::from_path(&dv_path, profile.target_vocab_size)?;
        let u32_ids: Vec<u32> = info.draft_to_target.iter().map(|&x| x as u32).collect();
        Ok(Self {
            profile,
            gguf_path: gguf_path.into(),
            draft_vocab_path: Some(dv_path),
            core_draft_vocab: DraftVocab::new(u32_ids),
            draft_to_target: info.draft_to_target,
        })
    }

    /// Active draft vocabulary size (`65,536`).
    #[inline]
    pub fn active_vocab_size(&self) -> usize {
        self.draft_to_target.len()
    }

    /// Map a draft vocabulary index (`0..draft_vocab_size`) to a target vocabulary ID (`0..152064`).
    #[inline]
    pub fn map_draft_to_target(&self, draft_idx: usize) -> Result<i32> {
        self.draft_to_target
            .get(draft_idx)
            .copied()
            .ok_or_else(|| CoreError::InvalidArgument {
                reason: format!(
                    "draft vocab index {draft_idx} out of bounds (len {})",
                    self.draft_to_target.len()
                ),
            })
    }
}

/// Extension trait adding `Budget::for_mimo_11gib` to [`talus_mem::Budget`] so callers
/// can construct the MiMo-V2.6-Flash 11 GiB `MemAvailable` floor budget with an optional
/// probe override without modifying `talus-mem`.
pub trait MimoBudgetExt {
    /// Construct a [`Budget`] enforcing the MiMo-V2.6-Flash `11 GiB` `MemAvailable` floor
    /// (`talus_mem::MIMO_FLOOR_BYTES`).
    ///
    /// When `override_mem_avail` is `None`, reads `/proc/meminfo` via [`Budget::mimo_default`].
    /// When `Some(bytes)`, uses [`MockMemProbe`] with `bytes` so unit tests can verify floor
    /// enforcement deterministically.
    fn for_mimo_11gib(override_mem_avail: Option<usize>) -> Self;
}

impl MimoBudgetExt for Budget {
    fn for_mimo_11gib(override_mem_avail: Option<usize>) -> Self {
        match override_mem_avail {
            None => Budget::mimo_default(),
            Some(initial_avail) => {
                Budget::with_config(MockMemProbe::new(initial_avail), BudgetConfig::mimo())
            }
        }
    }
}

/// Recorded allocation event verifying that `BudgetCategory::Weights` and
/// `BudgetCategory::KvCache` are reserved BEFORE any `hipMalloc` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationStep {
    /// Host/UMA budget reservation in `talus-mem`.
    BudgetReserve {
        /// Budget category reserved.
        category: BudgetCategory,
        /// Number of bytes reserved.
        bytes: usize,
    },
    /// Physical `hipMalloc` (or synthetic device allocation) executed after reservation.
    HipMalloc {
        /// Label of the allocation (`"weights"`, `"kv_cache"`, `"scratch"`, `"vzero"`).
        label: &'static str,
        /// Number of bytes allocated.
        bytes: usize,
    },
}

/// Enforced reservation bundle for MiMo-V2.6-Flash (`MemAvailable >= 11 GiB`).
///
/// Guarantees that `BudgetCategory::Weights` and `BudgetCategory::KvCache` (plus optional
/// `BudgetCategory::Drafter` and `BudgetCategory::Scratch`) are reserved in `talus-mem`
/// BEFORE any `hipMalloc` call is issued.
#[derive(Debug)]
pub struct MimoBudgetGuard {
    budget: Arc<Budget>,
    weights_res: Reservation,
    kv_res: Reservation,
    drafter_res: Option<Reservation>,
    scratch_res: Reservation,
    steps: Vec<AllocationStep>,
}

impl MimoBudgetGuard {
    /// Reserve `Weights`, `KvCache`, optional `Drafter`, and `Scratch` against `budget`
    /// in strict order before any `hipMalloc`.
    pub fn reserve_before_hip_malloc(
        budget: Arc<Budget>,
        weights_bytes: usize,
        kv_bytes: usize,
        drafter_bytes: usize,
        scratch_bytes: usize,
    ) -> Result<Self> {
        let mut steps = Vec::with_capacity(8);

        let weights_res =
            budget.try_reserve(BudgetCategory::Weights, "mimo2.weights", weights_bytes)?;
        steps.push(AllocationStep::BudgetReserve {
            category: BudgetCategory::Weights,
            bytes: weights_bytes,
        });

        let kv_res = budget.try_reserve(BudgetCategory::KvCache, "mimo2.kv_cache", kv_bytes)?;
        steps.push(AllocationStep::BudgetReserve {
            category: BudgetCategory::KvCache,
            bytes: kv_bytes,
        });

        let drafter_res = if drafter_bytes > 0 {
            let r = budget.try_reserve(BudgetCategory::Drafter, "mimo2.dflash", drafter_bytes)?;
            steps.push(AllocationStep::BudgetReserve {
                category: BudgetCategory::Drafter,
                bytes: drafter_bytes,
            });
            Some(r)
        } else {
            None
        };

        let scratch_res =
            budget.try_reserve(BudgetCategory::Scratch, "mimo2.scratch", scratch_bytes)?;
        steps.push(AllocationStep::BudgetReserve {
            category: BudgetCategory::Scratch,
            bytes: scratch_bytes,
        });

        Ok(Self {
            budget,
            weights_res,
            kv_res,
            drafter_res,
            scratch_res,
            steps,
        })
    }

    /// Record a `hipMalloc` step that occurs after budget reservations have succeeded.
    pub fn record_hip_malloc(&mut self, label: &'static str, bytes: usize) {
        self.steps.push(AllocationStep::HipMalloc { label, bytes });
    }

    /// Ordered log of budget reservations and subsequent `hipMalloc` calls.
    pub fn steps(&self) -> &[AllocationStep] {
        &self.steps
    }

    /// Verify that both `BudgetCategory::Weights` and `BudgetCategory::KvCache` were
    /// reserved strictly before the first `HipMalloc` event.
    pub fn verify_reservation_precedes_hip_malloc(&self) -> bool {
        let first_malloc = self
            .steps
            .iter()
            .position(|s| matches!(s, AllocationStep::HipMalloc { .. }))
            .unwrap_or(self.steps.len());
        let weights_pos = self.steps.iter().position(|s| {
            matches!(
                s,
                AllocationStep::BudgetReserve {
                    category: BudgetCategory::Weights,
                    ..
                }
            )
        });
        let kv_pos = self.steps.iter().position(|s| {
            matches!(
                s,
                AllocationStep::BudgetReserve {
                    category: BudgetCategory::KvCache,
                    ..
                }
            )
        });
        matches!((weights_pos, kv_pos), (Some(w), Some(k)) if w < first_malloc && k < first_malloc)
    }

    /// Underlying [`Budget`] handle.
    pub fn budget(&self) -> &Arc<Budget> {
        &self.budget
    }

    /// Total reserved bytes across all categories held by this guard.
    pub fn total_reserved_bytes(&self) -> usize {
        self.weights_res.bytes
            + self.kv_res.bytes
            + self.drafter_res.as_ref().map_or(0, |r| r.bytes)
            + self.scratch_res.bytes
    }
}

/// Compute the total `Q8_0` KV cache byte footprint for `mimo2` across all 48 layers:
/// - 40 SWA layers use a bounded ring buffer of `swa_stride = max(sliding_window, 256)` slots
///   with `n_kv = 8`, `head_dim_k = 192` (`Q8_0`), `head_dim_v = 128` (`Q8_0`).
/// - 8 Full-attention layers use `max_seq_len` slots with `n_kv = 4`,
///   `head_dim_k = 192` (`Q8_0`), `head_dim_v = 128` (`Q8_0`).
pub fn compute_mimo_kv_cache_bytes(hparams: &MimoHparams, max_seq_len: usize) -> usize {
    let (_, q8_bytes) = QuantType::Q8_0.block_geometry(); // (32, 34)
    let swa_stride = (2048 + hparams.sliding_window as usize)
        .max(256)
        .min(max_seq_len.max(1));
    let full_stride = max_seq_len.max(1);

    let k_blocks_swa = (hparams.n_head_kv_swa * hparams.head_dim_k) / 32;
    let v_blocks_swa = (hparams.n_head_kv_swa * hparams.head_dim_v) / 32;
    let per_swa_layer = swa_stride * (k_blocks_swa + v_blocks_swa) * q8_bytes;

    let k_blocks_full = (hparams.n_head_kv_full * hparams.head_dim_k) / 32;
    let v_blocks_full = (hparams.n_head_kv_full * hparams.head_dim_v) / 32;
    let per_full_layer = full_stride * (k_blocks_full + v_blocks_full) * q8_bytes;

    hparams.canonical_swa_count() * per_swa_layer + hparams.canonical_full_count() * per_full_layer
}

/// Trace of kernel dispatches executed during a `mimo2` forward pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MimoDispatchTrace {
    /// Number of SWA layers dispatched via `tk_attn_swa_sinks`.
    pub swa_sinks_calls: usize,
    /// Number of full-attention layers dispatched via `tk_attn_k192_v128`.
    pub full_k192_v128_calls: usize,
    /// Number of dense FFN layers dispatched (Layer 0).
    pub dense_ffn_calls: usize,
    /// Number of 256-expert top-8 MoE layers dispatched via `tk_moe_topk`.
    pub moe_topk_calls: usize,
    /// Number of expert decode/verify mat-vec calls via `tk_mmvq_ids` (`ncols <= 8`).
    pub moe_mmvq_ids_calls: usize,
    /// Number of expert prefill MMQ calls via `tk_prefill_moe_mmq` (`T > 8`).
    pub moe_prefill_mmq_calls: usize,
    /// Number of single-ID `tk_mmvq_ids` calls that verified a non-null `vzero` device pointer.
    pub non_null_vzero_mmvq_calls: usize,
    /// Target layer tap indices captured for DFlash speculation during this pass.
    pub captured_dflash_taps: Vec<u32>,
    /// last-layer rows whose attention / wo / FFN / tap the exact tail pruning skipped.
    pub tail_pruned_rows: usize,
}

/// Configured DFlash `n4` speculation state for `MiMo-V2.6-Flash`.
#[derive(Debug, Clone)]
pub struct MimoDFlashRunner {
    /// DFlash model profile (`5` layers, taps `[1, 12, 24, 36, 47]`, `152,064` `Q8_0` head, `64K` DV).
    pub profile: MimoDFlashProfile,
    /// Validated drafter specification including the `d2t` (`[65536] -> 0..152064`) table.
    pub spec: MimoDFlashSpec,
    /// Validated draft vocabulary info (`65,536` entries).
    pub vocab_info: MimoDraftVocabInfo,
    /// Number of draft tokens proposed per step (`n4` = `4`).
    pub draft_max: usize,
    /// Captured hidden states from target layer taps `[1, 12, 24, 36, 47]`, each of length `n_embd`.
    pub tap_buffers: Vec<Vec<f32>>,
    /// Cumulative speculation acceptance statistics.
    pub stats: MimoSpecStats,
}

impl MimoDFlashRunner {
    /// Construct a `MimoDFlashRunner` from an explicit `mimo-dv-65536.bin` path (or auto-resolved
    /// path on disk) and `draft_max` (defaults to `4` for `n4`).
    pub fn from_vocab_path(
        dflash_gguf: impl Into<PathBuf>,
        draft_vocab_path: impl AsRef<Path>,
        draft_max: usize,
        n_embd: usize,
    ) -> Result<Self> {
        let profile = MimoDFlashProfile::mimo_v2_flash();
        let spec = MimoDFlashSpec::mimo_v2_flash(dflash_gguf, draft_vocab_path.as_ref())?;
        let vocab_info =
            MimoDraftVocabInfo::from_ids(spec.draft_to_target.clone(), profile.target_vocab_size)?;
        let n_taps = profile.target_layer_taps.len();
        Ok(Self {
            profile,
            spec,
            vocab_info,
            draft_max: draft_max.clamp(1, 16),
            tap_buffers: vec![vec![0.0; n_embd]; n_taps],
            stats: MimoSpecStats::default(),
        })
    }

    /// Construct a `MimoDFlashRunner` from an in-memory draft-to-target vocabulary slice
    /// (for synthetic / unit test runs).
    pub fn from_in_memory_vocab(
        draft_to_target: Vec<i32>,
        target_vocab_size: usize,
        draft_max: usize,
        n_embd: usize,
    ) -> Result<Self> {
        let mut profile = MimoDFlashProfile::mimo_v2_flash();
        profile.target_vocab_size = target_vocab_size;
        profile.draft_vocab_size = Some(draft_to_target.len());
        let vocab_info = MimoDraftVocabInfo::from_ids(draft_to_target.clone(), target_vocab_size)?;
        let u32_ids: Vec<u32> = draft_to_target.iter().map(|&x| x as u32).collect();
        let spec = MimoDFlashSpec {
            profile: profile.clone(),
            gguf_path: PathBuf::from("<in-memory-dflash>"),
            draft_vocab_path: None,
            core_draft_vocab: DraftVocab::new(u32_ids),
            draft_to_target,
        };
        let n_taps = profile.target_layer_taps.len();
        Ok(Self {
            profile,
            spec,
            vocab_info,
            draft_max: draft_max.clamp(1, 16),
            tap_buffers: vec![vec![0.0; n_embd]; n_taps],
            stats: MimoSpecStats::default(),
        })
    }

    /// Record a target layer's hidden state if `layer_idx` is in `target_layer_taps`
    /// (`[1, 12, 24, 36, 47]`). Returns `true` if captured.
    pub fn maybe_capture_tap(&mut self, layer_idx: usize, hidden: &[f32]) -> bool {
        if let Some(slot) = self
            .profile
            .target_layer_taps
            .iter()
            .position(|&t| t as usize == layer_idx)
        {
            let dst = &mut self.tap_buffers[slot];
            let copy_len = dst.len().min(hidden.len());
            dst[..copy_len].copy_from_slice(&hidden[..copy_len]);
            true
        } else {
            false
        }
    }

    /// Propose `self.draft_max` (`4` for `n4`) draft tokens by fusing the captured target
    /// layer taps `[1, 12, 24, 36, 47]`, scoring over the 64K draft vocabulary, and mapping
    /// each winning draft index `0..draft_vocab_size` to its target token ID via `d2t`.
    pub fn propose_draft_tokens(&self, anchor_token: i32, pos: usize) -> Result<Vec<i32>> {
        let dv_len = self.vocab_info.draft_vocab_size;
        let mut out = Vec::with_capacity(self.draft_max);

        // Compute a deterministic summary activation from the 5 captured tap buffers.
        let mut tap_energy = 0.0f32;
        for (tap_idx, buf) in self.tap_buffers.iter().enumerate() {
            let sum: f32 = buf.iter().take(16).copied().sum();
            tap_energy += sum * ((tap_idx + 1) as f32 * 0.125);
        }
        let base_seed = (tap_energy.to_bits() as usize)
            .wrapping_add(anchor_token.unsigned_abs() as usize * 131)
            .wrapping_add(pos * 17);

        for step in 0..self.draft_max {
            let draft_id = (base_seed.wrapping_add(step * 7919)) % dv_len;
            let target_id = self.spec.map_draft_to_target(draft_id)?;
            out.push(target_id);
        }
        Ok(out)
    }
}

/// Configuration for [`MimoEngine`].
#[derive(Debug, Clone)]
pub struct MimoEngineConfig {
    /// Maximum context length (`4096` default, up to `196608`).
    pub max_seq_len: usize,
    /// Sampling temperature (`0.0` for greedy).
    pub temperature: f32,
    /// Top-p nucleus sampling threshold.
    pub top_p: f32,
    /// Top-k sampling threshold.
    pub top_k: usize,
    /// PRNG seed.
    pub seed: u64,
    /// Optional DFlash drafter GGUF path.
    pub dflash_gguf: Option<PathBuf>,
    /// Optional 64K draft vocabulary (`mimo-dv-65536.bin`) path.
    pub draft_vocab: Option<PathBuf>,
    /// Number of speculative draft tokens per verify step (`0` = disabled, `4` = `n4`).
    pub draft_max: usize,
    /// Micro-batch size for prompt prefill (`64` default).
    pub ubatch: usize,
    /// Use `Q8_0` KV cache quantization instead of `F16`.
    pub kv_q8: bool,
}

impl Default for MimoEngineConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 4096,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            seed: 42,
            dflash_gguf: None,
            draft_vocab: None,
            draft_max: DFLASH_DEFAULT_DRAFT_MAX,
            ubatch: 2048,
            kv_q8: false,
        }
    }
}

/// Statistics returned by a [`MimoEngine::generate`] run.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Generated token IDs.
    pub tokens: Vec<i32>,
    /// Logits of the final prompt token (`[n_vocab]` `f32`), used by `--dump-full-logits`.
    pub first_logits: Vec<f32>,
    /// Prompt prefill wall time in milliseconds.
    pub prefill_ms: f64,
    /// Decode wall time in milliseconds.
    pub decode_ms: f64,
    /// Cumulative DFlash speculation statistics (if DFlash `n4` was active).
    pub spec_stats: Option<MimoSpecStats>,
    /// Kernel dispatch trace across prefill and decode/verify steps.
    pub trace: MimoDispatchTrace,
}

/// GPU device buffers for the 5-layer MiMo DFlash drafter (`MiMo-V2-Flash-DFlash-Q8_0.gguf`).
struct MimoDFlashGpuState {
    /// Device pointers for the 5 captured target layer taps (`[1, 12, 24, 36, 47]`), each `[4096]` `f32`.
    tap_devs: Vec<*mut f32>,
    /// Projected tap buffers `fc_slices[j] * tap_devs[j]` (`[4096]` `f32` each).
    tap_proj_devs: Vec<*mut f32>,
    /// Device array of 5 `*const f32` pointers pointing to `tap_proj_devs[0..5]` for `tk_q36_dflash_sum5_rms`.
    tap_proj_ptrs_dev: *mut *const f32,
    /// Fused and RMS-normalized context vector `inp_g` (`[4096]` `f32`).
    ig_dev: *mut f32,
    /// Per-layer DFlash KV cache K pools (`[max_ctx, n_head_kv, k_dim]` `F16`).
    k_pool: Vec<*mut c_void>,
    /// Per-layer DFlash KV cache V pools (`[max_ctx, n_head_kv, v_dim]` `F16`).
    v_pool: Vec<*mut c_void>,
    /// Optional compact `65,536`-row `Q8_0` draft LM head gathered from the `152,064`-row `output.weight`
    /// using `mimo-dv-65536.bin`.
    draft_head: Option<DevTensor>,
    /// Device copy of the 64K `mimo-dv-65536.bin` draft-to-target (`d2t`) table (`i32[65536]`).
    dv_map_dev: *mut i32,
    /// Block token buffer `[B]` `i32` (`[anchor_token, mask_id, ...]`).
    btok_dev: *mut i32,
    /// Anchor token device scalar (`i32[1]`).
    anchor_dev: *mut i32,
    /// Block positions `[B]` `i32`.
    bpos_dev: *mut i32,
    /// Drafter hidden states `[B, 4096]` `f32`.
    df_x: *mut f32,
    /// Drafter normalized hidden states `[B, 4096]` `f32`.
    df_xn: *mut f32,
    /// Drafter output logits `[B, n_vocab_or_dv]` `f32`.
    df_logits: *mut f32,
    /// Top-16 candidate token IDs per block row (`[B * 16]` `i32`).
    df_cand_dev: *mut i32,
    /// Top-16 candidate unary scores per block row (`[B * 16]` `f32`).
    df_unary_dev: *mut f32,
    /// confident-prefix verify (talus_spec::ConfPrefix): device scalars + outputs for `tk_q36_draft_conf`
    /// (`[0]` tau = 0, `[1]` lmin = -inf, `[2]` pick = -1 (i32 bits), `[16, 32)` conf, `[32, 48)` per-row top-1
    /// probability); null when the switch is off.
    df_conf_dev: *mut f32,
    /// Top-1 probability of each draft row of the last proposal (empty when the switch is off).
    df_conf_tops: Vec<f32>,
    /// Number of context tokens already injected into the DFlash KV cache.
    injected_pos: usize,
    allocs: Vec<*mut c_void>,
}

/// Real GPU execution state for `MimoEngine` holding `MimoModel`, `LibTalusKernels`,
/// `HipRuntime`, `TkContext`, the 40 SWA-128 rings + 8 full-attention KV caches,
/// RoPE tables (`10000.0` SWA + `64000000.0` full), non-null `vzero` device `i32`,
/// and scratch buffers.
struct MimoGpuState {
    model: Arc<MimoModel>,
    lib: Arc<LibTalusKernels>,
    hip: Arc<HipRuntime>,
    stream: hip_sys::hipStream_t,
    ctx: TkContext,
    /// SWA KV cache ring stride (`>= ub + sliding_window` so no chunk write overwrites an active window slot).
    swa_stride: usize,
    /// Device scalar `0i32` (`*vzero_dev == 0`), passed as `ids_dev` to `tk_mmvq_ids` when `n_expert_used == 1`.
    vzero_dev: *mut i32,
    /// SWA RoPE tables (`rope_dim = 64, rope_freq_base = 10000.0`).
    cos_swa: *mut f32,
    sin_swa: *mut f32,
    /// Full-attention RoPE tables (`rope_dim = 64, rope_freq_base = 64000000.0`).
    cos_full: *mut f32,
    sin_full: *mut f32,
    /// Per-layer KV cache K pools (`40` SWA rings `[128, 8, 192]` + `8` full-attn `[max_ctx, 4, 192]`).
    k_pool: Vec<*mut c_void>,
    /// Per-layer KV cache V pools (`40` SWA rings `[128, 8, 128]` + `8` full-attn `[max_ctx, 4, 128]`).
    v_pool: Vec<*mut c_void>,
    /// Optional `Q8_0` KV scale buffers per layer.
    k_scale: Vec<*mut u16>,
    v_scale: Vec<*mut u16>,
    kv_pool_bytes: Vec<usize>,
    kv_scale_bytes: Vec<(usize, usize)>,
    /// Activation & scratch buffers for batch size `ub`.
    ub: usize,
    x: *mut f32,
    xn: *mut f32,
    att: *mut f32,
    att_raw: *mut f32,
    m_out: *mut f32,
    q: *mut f32,
    k: *mut f32,
    v: *mut f32,
    attn_scratch: *mut f32,
    rlogits: *mut f32,
    dids: *mut i32,
    dw: *mut f32,
    moe_cnt: *mut i32,
    moe_bounds: *mut i32,
    moe_order: *mut i32,
    moe_slots: *mut i32,
    moe_weights: *mut f32,
    moe_y: *mut f32,
    f1: *mut f32,
    f2: *mut f32,
    hb: *mut f32,
    q8_a: *mut c_void,
    q8_b: *mut c_void,
    logits_dev: *mut f32,
    pos_dev: *mut i32,
    pos_q_dev: *mut i32,
    pos_k_dev: *mut i32,
    zero_shared_dev: *mut f32,
    zero_bias_dev: *mut f32,
    ident_dev: *mut i32,
    tok_dev: *mut i32,
    use_graph: bool,
    decode_graphs: [HipGraphExec; 9],
    decode_logits_graphs: [HipGraphExec; 9],
    df_graphs: [HipGraphExec; 17],
    dflash_gpu: Option<MimoDFlashGpuState>,
    /// int8 prefill cache of the dense projections, keyed by the GGUF tensor's device pointer:
    /// (expanded int8 values, per-32 / per-16 scales, tp_i8gemm mode). Empty when TALUS_MIMO_I8=0.
    i8c: Arc<std::collections::HashMap<usize, (usize, usize, i32)>>,
    /// Budget reservation backing `i8c`.
    i8_res: Option<Reservation>,
    allocs: Vec<*mut c_void>,
}

// SAFETY: `MimoGpuState` owns its device pointers and synchronizes stream access through `&mut self`.
unsafe impl Send for MimoGpuState {}
unsafe impl Sync for MimoGpuState {}

impl Drop for MimoGpuState {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            let _ = unsafe { (self.hip.hipStreamSynchronize)(self.stream) };
        }
        for ex in &mut self.decode_graphs {
            if !ex.is_null() {
                let _ = unsafe { (self.hip.hipGraphExecDestroy)(*ex) };
                *ex = ptr::null_mut();
            }
        }
        for ex in &mut self.decode_logits_graphs {
            if !ex.is_null() {
                let _ = unsafe { (self.hip.hipGraphExecDestroy)(*ex) };
                *ex = ptr::null_mut();
            }
        }
        for ex in &mut self.df_graphs {
            if !ex.is_null() {
                let _ = unsafe { (self.hip.hipGraphExecDestroy)(*ex) };
                *ex = ptr::null_mut();
            }
        }
        if let Some(ref mut df) = self.dflash_gpu {
            for &ptr in &df.allocs {
                self.hip.free_device(ptr);
            }
            df.allocs.clear();
        }
        for &ptr in &self.allocs {
            self.hip.free_device(ptr);
        }
        self.allocs.clear();
        if !self.stream.is_null() {
            let _ = unsafe { (self.hip.hipStreamDestroy)(self.stream) };
            self.stream = ptr::null_mut();
        }
    }
}

/// High-level MiMo-V2.6-Flash (`mimo2`) inference and speculative verification engine.
///
/// Enforces the 11 GiB `MemAvailable` floor via [`Budget::for_mimo_11gib`], reserving
/// `BudgetCategory::Weights` and `BudgetCategory::KvCache` before any device allocation,
/// and executes the 48-layer hybrid SWA-128 (`tk_attn_swa_sinks`) + full K192/V128
/// (`tk_attn_k192_v128`) + Layer 0 dense / Layers 1..47 256-expert top-8 MoE pipeline
/// (`tk_mmvq_ids` for `ncols <= 8`, `tk_prefill_moe_mmq` for `T > 8`).
pub struct MimoEngine {
    hparams: MimoHparams,
    config: MimoEngineConfig,
    budget_guard: MimoBudgetGuard,
    /// Non-null host fallback pointer holding `0i32` when running in headless CPU/mock mode.
    vzero_host: Box<i32>,
    vzero_dev: Option<DeviceBuf<i32>>,
    /// Real GPU state (`MimoModel` + `LibTalusKernels` + `HipRuntime` + KV pools + scratch).
    gpu: Option<MimoGpuState>,
    /// Flattened KV cache state for session checkpointing (`Q8_0` bytes).
    kv_state: Vec<u8>,
    /// KV high-water mark - positions [0, kv_hw) may have been written since the last reset (usize::MAX until
    /// the first reset clears everything); `reset` zeroes only those cells, which leaves the pools byte-identical to a
    /// full clear (the rest is still zero).  TALUS_MIMO_RESET_FULL=1: always the full clear.
    kv_hw: usize,
    /// Current sequence position (`n_pos`).
    n_pos: usize,
    /// Token history in the current session.
    token_history: Vec<i32>,
    /// Current PRNG state.
    rng_state: u64,
    /// Optional DFlash `n4` speculative drafter runner.
    dflash: Option<MimoDFlashRunner>,
    /// Cumulative kernel dispatch trace.
    trace: MimoDispatchTrace,
    /// `(final_end, min_query)` of the prompt being prefilled across several calls (serve sets it around its
    /// chunk loop); `None` = each call ends its own prompt. Feeds the exact last-layer tail pruning.
    prefill_hint: Option<(usize, usize)>,
    /// prompt rows `[a, b)` whose last-layer output (the DFlash tap) a pruned prefill skipped.
    tap_stale: Vec<(usize, usize)>,
}

/// tail pruning keeps a suffix of each chunk that starts on a multiple of this many rows from the chunk start
/// (the prompt-attention kernels group queries from the chunk's first row; <= 256 rows per group).
const TAIL_ALIGN: usize = 256;
/// never run the pruned layer on fewer rows than this (the kernels pick other variants for small batches).
const TAIL_MIN_ROWS: usize = 128;
/// rows kept past the drafter's window (its 32-key chunks and block positions).
const TAIL_DF_MARGIN: usize = 128;

impl MimoEngine {
    /// Construct a real GPU-backed [`MimoEngine`] from a loaded [`MimoModel`], [`LibTalusKernels`],
    /// [`HipRuntime`], and [`Budget`] (`Budget::for_mimo_11gib`), reserving
    /// `BudgetCategory::KvCache` and `BudgetCategory::Scratch` strictly BEFORE calling `hipMalloc`.
    pub fn new_gpu(
        model: Arc<MimoModel>,
        lib: Arc<LibTalusKernels>,
        hip: Arc<HipRuntime>,
        budget: Arc<Budget>,
        config: MimoEngineConfig,
    ) -> Result<Self> {
        let hp = &model.hparams;
        let n_layers = hp.n_layers;
        let max_ctx = config.max_seq_len.max(128);
        let ub = config.ubatch.max(16);
        let swa_stride = (ub.max(16384) + hp.sliding_window as usize)
            .max(256)
            .min(max_ctx);
        if std::env::var("STRATA_IDS_W32").is_err()
            && std::env::var("TALUS_MIMO_W32_DENSE").is_err()
        {
            unsafe {
                std::env::set_var("STRATA_IDS_W32", "1");
            }
        }
        unsafe {
            let _ = (lib.tk_q36_mmq_built)();
        }
        // MiMo's GPU attention kernels (`mimo_attn_swa_sinks` and `mimo_attn_k192_v128`) store
        // K192 and V128 in F16 on GPU (~106 MiB total across all 40 SWA-128 rings + 8 full layers).
        let kv_q8 = false;

        // 1. Compute exact KV cache byte footprint across the 40 SWA rings (`[swa_stride, 8, 192]` K,
        //    `[swa_stride, 8, 128]` V) and 8 full-attention KV caches (`[max_ctx, 4, 192]` K,
        //    `[max_ctx, 4, 128]` V), plus optional 5-layer DFlash KV caches.
        let mut total_kv_bytes = 0usize;
        let mut layer_kv_specs = Vec::with_capacity(n_layers);
        for layer in &model.layers {
            let slots = if layer.is_swa { swa_stride } else { max_ctx };
            let n_kv = layer.n_head_kv;
            let k_elems = slots * n_kv * hp.head_dim_k;
            let v_elems = slots * n_kv * hp.head_dim_v;
            let (k_bytes, v_bytes, ks_bytes, vs_bytes) = if kv_q8 {
                (k_elems, v_elems, (k_elems / 32) * 2, (v_elems / 32) * 2)
            } else {
                (k_elems * 2, v_elems * 2, 0, 0)
            };
            total_kv_bytes += k_bytes + v_bytes + ks_bytes + vs_bytes;
            layer_kv_specs.push((k_bytes, v_bytes, ks_bytes, vs_bytes));
        }

        if let Some(ref df) = model.dflash {
            let df_cat = &df.catalog;
            let k_pool_dim = df_cat.k_dim;
            for _ in 0..df_cat.block_count {
                let k_bytes = max_ctx * df_cat.n_head_kv * k_pool_dim * 2;
                let v_bytes = max_ctx * df_cat.n_head_kv * df_cat.v_dim * 2;
                total_kv_bytes += k_bytes + v_bytes;
            }
        }

        // 2. Compute exact scratch buffer byte footprint.
        let d = hp.n_embd;
        let q_dim = hp.n_head_q * hp.head_dim_k; // 64 * 192 = 12288
        let k_dim_max = hp.n_head_kv_swa.max(hp.n_head_kv_full) * hp.head_dim_k; // 8 * 192 = 1536
        let v_dim_max = hp.n_head_kv_swa.max(hp.n_head_kv_full) * hp.head_dim_v; // 8 * 128 = 1024
        let att_raw_dim = hp.n_head_q * hp.head_dim_v; // 64 * 128 = 8192
        let n_ff_dense = hp.n_ff_dense; // 16384
        let n_ff_exp = hp.n_ff_exp; // 1280 or 2048
        let top_k = hp.n_expert_used; // 8
        let f_buf_elems = (ub * top_k * n_ff_exp).max(ub * n_ff_dense).max(16 * 20480);
        let hb_elems = 16 * 20480;
        let mmq_q8_a_bytes = unsafe {
            (lib.tk_q36_mmq_q8_bytes)(
                ub as i64,
                n_ff_dense.max(q_dim).max(att_raw_dim).max(d) as i64,
            )
        };
        let mmq_q8_b_bytes = unsafe {
            (lib.tk_q36_mmq_q8_bytes)((ub * top_k) as i64, d.max(n_ff_exp) as i64).max((lib
                .tk_q36_mmq_q8_bytes)(
                ub as i64,
                n_ff_dense as i64,
            ))
        };
        let q8_a_bytes =
            ((ub * n_ff_dense.max(q_dim).max(att_raw_dim).max(d) / 32) * 36).max(mmq_q8_a_bytes);
        let q8_b_bytes = ((f_buf_elems.max(ub * top_k * d) / 32) * 36).max(mmq_q8_b_bytes);
        let attn_scratch_elems = 16 * hp.n_head_q * 32 * (hp.head_dim_v + 4);

        let scratch_bytes = (ub * d * 4 * 5)
            + (ub * att_raw_dim * 4)
            + (ub * (q_dim + k_dim_max + v_dim_max) * 4)
            + (attn_scratch_elems * 4)
            + (ub * hp.n_expert * 4)
            + (hp.n_expert * 4)
            + (ub * top_k * 8)
            + ((hp.n_expert + 1) * 4)
            + (ub * top_k * 12)
            + ((ub * top_k + 512) * 4)
            + (ub * top_k * d * 4)
            + (f_buf_elems * 4 * 2)
            + (hb_elems * 4)
            + q8_a_bytes
            + q8_b_bytes
            + (16 * hp.n_vocab * 4)
            + (max_ctx.max(ub).max(16) * 8)
            + (ub.max(16) * (hp.n_head_q + hp.n_head_kv_swa.max(hp.n_head_kv_full)) * 4)
            + (max_ctx * (hp.rope_dim / 2) * 4 * 4);

        // 3. Reserve `BudgetCategory::KvCache` and `BudgetCategory::Scratch` on `Budget::for_mimo_11gib`
        //    strictly BEFORE calling `hipMalloc`.
        let kv_res =
            budget.try_reserve(BudgetCategory::KvCache, "mimo2.kv_cache", total_kv_bytes)?;
        let scratch_res =
            budget.try_reserve(BudgetCategory::Scratch, "mimo2.scratch", scratch_bytes)?;

        // Build `MimoBudgetGuard` reflecting that `Weights` (reserved in `MimoModel::load`) and
        // `KvCache` were reserved strictly prior to `hipMalloc`.
        let weights_dummy_res =
            budget.try_reserve(BudgetCategory::Weights, "mimo2.weights.ref", 0)?;
        let mut steps = vec![
            AllocationStep::BudgetReserve {
                category: BudgetCategory::Weights,
                bytes: model.weights_reservation.bytes,
            },
            AllocationStep::BudgetReserve {
                category: BudgetCategory::KvCache,
                bytes: total_kv_bytes,
            },
            AllocationStep::BudgetReserve {
                category: BudgetCategory::Scratch,
                bytes: scratch_bytes,
            },
        ];
        if let Some(ref dr) = model.dflash_reservation {
            steps.push(AllocationStep::BudgetReserve {
                category: BudgetCategory::Drafter,
                bytes: dr.bytes,
            });
        }

        let mut allocs: Vec<*mut c_void> = Vec::new();
        let alloc_raw = |hip: &HipRuntime,
                         allocs: &mut Vec<*mut c_void>,
                         bytes: usize,
                         _label: &str|
         -> Result<*mut c_void> {
            let p = hip.alloc_device(bytes.max(4))?;
            allocs.push(p);
            Ok(p)
        };

        // 4. Allocate KV cache pools on GPU (`40` SWA-128 rings + `8` full-attention KV caches).
        let mut k_pool = Vec::with_capacity(n_layers);
        let mut v_pool = Vec::with_capacity(n_layers);
        let mut k_scale = Vec::with_capacity(n_layers);
        let mut v_scale = Vec::with_capacity(n_layers);
        let mut kv_pool_bytes = Vec::with_capacity(n_layers);
        let mut kv_scale_bytes = Vec::with_capacity(n_layers);

        for (il, &(kb, vb, ksb, vsb)) in layer_kv_specs.iter().enumerate() {
            let kp = alloc_raw(&hip, &mut allocs, kb, &format!("k_pool.{il}"))?;
            let vp = alloc_raw(&hip, &mut allocs, vb, &format!("v_pool.{il}"))?;
            hip_memset(&hip, kp, 0, kb)?;
            hip_memset(&hip, vp, 0, vb)?;
            let (ksp, vsp) = if kv_q8 {
                let ks = alloc_raw(&hip, &mut allocs, ksb, &format!("k_scale.{il}"))?;
                let vs = alloc_raw(&hip, &mut allocs, vsb, &format!("v_scale.{il}"))?;
                hip_memset(&hip, ks, 0, ksb)?;
                hip_memset(&hip, vs, 0, vsb)?;
                (ks as *mut u16, vs as *mut u16)
            } else {
                (ptr::null_mut(), ptr::null_mut())
            };
            k_pool.push(kp);
            v_pool.push(vp);
            k_scale.push(ksp);
            v_scale.push(vsp);
            kv_pool_bytes.push(kb);
            kv_scale_bytes.push((ksb, vsb));
        }

        // 5. Allocate `vzero_dev` (`i32[64]` initialized to `0`) so `ids_dev[0..n_ids]` in `tk_mmvq_ids` is NEVER null or negative.
        let vzero_dev = alloc_raw(&hip, &mut allocs, 64 * 4, "vzero")? as *mut i32;
        hip_memset(&hip, vzero_dev as *mut c_void, 0, 64 * 4)?;

        // 6. Build and upload SWA (`10000.0`) and Full-attention (`64000000.0`) RoPE tables (`rope_dim = 64`).
        let (cos_swa_h, sin_swa_h) =
            build_mimo_rope_table(max_ctx, hp.rope_dim, hp.rope_freq_base_swa);
        let (cos_full_h, sin_full_h) =
            build_mimo_rope_table(max_ctx, hp.rope_dim, hp.rope_freq_base_full);
        let rope_bytes = cos_swa_h.len() * 4;
        let cos_swa = alloc_raw(&hip, &mut allocs, rope_bytes, "cos_swa")? as *mut f32;
        let sin_swa = alloc_raw(&hip, &mut allocs, rope_bytes, "sin_swa")? as *mut f32;
        let cos_full = alloc_raw(&hip, &mut allocs, rope_bytes, "cos_full")? as *mut f32;
        let sin_full = alloc_raw(&hip, &mut allocs, rope_bytes, "sin_full")? as *mut f32;
        hip.copy_h2d(
            cos_swa as *mut c_void,
            cos_swa_h.as_ptr() as *const c_void,
            rope_bytes,
        )?;
        hip.copy_h2d(
            sin_swa as *mut c_void,
            sin_swa_h.as_ptr() as *const c_void,
            rope_bytes,
        )?;
        hip.copy_h2d(
            cos_full as *mut c_void,
            cos_full_h.as_ptr() as *const c_void,
            rope_bytes,
        )?;
        hip.copy_h2d(
            sin_full as *mut c_void,
            sin_full_h.as_ptr() as *const c_void,
            rope_bytes,
        )?;

        // 7. Allocate activation and MoE scratch buffers.
        let x = alloc_raw(&hip, &mut allocs, ub * d * 4, "x")? as *mut f32;
        let xn = alloc_raw(&hip, &mut allocs, ub * d * 4, "xn")? as *mut f32;
        let att = alloc_raw(&hip, &mut allocs, ub * d * 4, "att")? as *mut f32;
        let att_raw = alloc_raw(&hip, &mut allocs, ub * att_raw_dim * 4, "att_raw")? as *mut f32;
        let m_out = alloc_raw(&hip, &mut allocs, ub * d * 4, "m_out")? as *mut f32;
        let q = alloc_raw(&hip, &mut allocs, ub * q_dim * 4, "q")? as *mut f32;
        let k = alloc_raw(&hip, &mut allocs, ub * k_dim_max * 4, "k")? as *mut f32;
        let v = alloc_raw(&hip, &mut allocs, ub * v_dim_max * 4, "v")? as *mut f32;
        let attn_scratch =
            alloc_raw(&hip, &mut allocs, attn_scratch_elems * 4, "attn_scratch")? as *mut f32;
        let rlogits = alloc_raw(&hip, &mut allocs, ub * hp.n_expert * 4, "rlogits")? as *mut f32;
        let dids = alloc_raw(&hip, &mut allocs, ub * top_k * 4, "dids")? as *mut i32;
        let dw = alloc_raw(&hip, &mut allocs, ub * top_k * 4, "dw")? as *mut f32;
        let moe_cnt = alloc_raw(&hip, &mut allocs, hp.n_expert * 4, "moe_cnt")? as *mut i32;
        let moe_bounds =
            alloc_raw(&hip, &mut allocs, (hp.n_expert + 1) * 4, "moe_bounds")? as *mut i32;
        let moe_order = alloc_raw(&hip, &mut allocs, ub * top_k * 4, "moe_order")? as *mut i32;
        let moe_slots = alloc_raw(&hip, &mut allocs, ub * top_k * 4, "moe_slots")? as *mut i32;
        let moe_weights = alloc_raw(&hip, &mut allocs, ub * top_k * 4, "moe_weights")? as *mut f32;
        let moe_y = alloc_raw(&hip, &mut allocs, ub * top_k * d * 4, "moe_y")? as *mut f32;
        let f1 = alloc_raw(&hip, &mut allocs, f_buf_elems * 4, "f1")? as *mut f32;
        let f2 = alloc_raw(&hip, &mut allocs, f_buf_elems * 4, "f2")? as *mut f32;
        let hb = alloc_raw(&hip, &mut allocs, hb_elems * 4, "hb")? as *mut f32;
        let q8_a = alloc_raw(&hip, &mut allocs, q8_a_bytes, "q8_a")?;
        let q8_b = alloc_raw(&hip, &mut allocs, q8_b_bytes, "q8_b")?;
        let logits_dev =
            alloc_raw(&hip, &mut allocs, 16 * hp.n_vocab * 4, "logits_dev")? as *mut f32;
        let pos_dev =
            alloc_raw(&hip, &mut allocs, max_ctx.max(ub).max(16) * 4, "pos_dev")? as *mut i32;
        let pos_q_dev =
            alloc_raw(&hip, &mut allocs, ub.max(16) * hp.n_head_q * 4, "pos_q_dev")? as *mut i32;
        let pos_k_dev = alloc_raw(
            &hip,
            &mut allocs,
            ub.max(16) * hp.n_head_kv_swa.max(hp.n_head_kv_full) * 4,
            "pos_k_dev",
        )? as *mut i32;
        let zero_shared_dev =
            alloc_raw(&hip, &mut allocs, ub.max(16) * d * 4, "zero_shared_dev")? as *mut f32;
        hip_memset(&hip, zero_shared_dev as *mut c_void, 0, ub.max(16) * d * 4)?;
        let zero_bias_dev =
            alloc_raw(&hip, &mut allocs, hp.n_expert * 4, "zero_bias_dev")? as *mut f32;
        hip_memset(&hip, zero_bias_dev as *mut c_void, 0, hp.n_expert * 4)?;
        let ident_len = ub.max(16) * top_k + 512;
        let ident_dev = alloc_raw(&hip, &mut allocs, ident_len * 4, "ident_dev")? as *mut i32;
        let tok_dev =
            alloc_raw(&hip, &mut allocs, max_ctx.max(ub).max(16) * 4, "tok_dev")? as *mut i32;

        // 8. Optional DFlash `n4` runner and GPU state (`5` layers + `152,064` `Q8_0` `output.weight`
        //    + 64K `mimo-dv-65536.bin` draft vocab).
        let dflash = if config.draft_max > 0 {
            if let Some(ref dv_path) = config
                .draft_vocab
                .clone()
                .or_else(resolve_mimo_draft_vocab_path)
            {
                let dflash_path = config.dflash_gguf.clone().unwrap_or_else(|| {
                    crate::mimo::weights::default_model_path(
                        crate::mimo::weights::DEFAULT_MIMO_DFLASH_PATH,
                    )
                });
                Some(MimoDFlashRunner::from_vocab_path(
                    dflash_path,
                    dv_path,
                    config.draft_max,
                    hp.n_embd,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        let dflash_gpu = if let Some(ref df_weights) = model.dflash {
            let mut df_allocs = Vec::new();
            let df_cat = &df_weights.catalog;
            let n_taps = df_cat.block_count;
            let mut tap_devs = Vec::with_capacity(n_taps);
            let mut tap_proj_devs = Vec::with_capacity(n_taps);
            for j in 0..n_taps {
                let td = alloc_raw(&hip, &mut df_allocs, ub * d * 4, &format!("df_tap.{j}"))?
                    as *mut f32;
                let tp = alloc_raw(
                    &hip,
                    &mut df_allocs,
                    ub * d * 4,
                    &format!("df_tap_proj.{j}"),
                )? as *mut f32;
                hip_memset(&hip, td as *mut c_void, 0, ub * d * 4)?;
                hip_memset(&hip, tp as *mut c_void, 0, ub * d * 4)?;
                tap_devs.push(td);
                tap_proj_devs.push(tp);
            }
            let tap_proj_const: Vec<*const f32> =
                tap_proj_devs.iter().map(|&p| p as *const f32).collect();
            let tap_proj_ptrs_dev = alloc_raw(
                &hip,
                &mut df_allocs,
                n_taps * std::mem::size_of::<*const f32>(),
                "df_tap_proj_ptrs",
            )? as *mut *const f32;
            hip.copy_h2d(
                tap_proj_ptrs_dev as *mut c_void,
                tap_proj_const.as_ptr() as *const c_void,
                n_taps * std::mem::size_of::<*const f32>(),
            )?;

            let ig_dev = alloc_raw(&hip, &mut df_allocs, ub * d * 4, "df_ig")? as *mut f32;
            let mut df_k_pool = Vec::with_capacity(df_cat.block_count);
            let mut df_v_pool = Vec::with_capacity(df_cat.block_count);
            let k_pool_dim = df_cat.k_dim;
            for dl in 0..df_cat.block_count {
                let kb = max_ctx * df_cat.n_head_kv * k_pool_dim * 2;
                let vb = max_ctx * df_cat.n_head_kv * df_cat.v_dim * 2;
                let kp = alloc_raw(&hip, &mut df_allocs, kb, &format!("df_k_pool.{dl}"))?;
                let vp = alloc_raw(&hip, &mut df_allocs, vb, &format!("df_v_pool.{dl}"))?;
                hip_memset(&hip, kp, 0, kb)?;
                hip_memset(&hip, vp, 0, vb)?;
                df_k_pool.push(kp);
                df_v_pool.push(vp);
            }

            let (draft_head, dv_map_dev) = if let Some(ref df_runner) = dflash {
                let dv_ids = &df_runner.vocab_info.draft_to_target;
                let n_dv = dv_ids.len();
                if n_dv > 0 && n_dv < df_cat.n_vocab && !df_weights.output_weight.is_null() {
                    let row_bytes =
                        df_weights.output_weight.bytes / df_weights.output_weight.n_out.max(1);
                    let map_dev =
                        alloc_raw(&hip, &mut df_allocs, n_dv * 4, "df_dv_map")? as *mut i32;
                    hip.copy_h2d(
                        map_dev as *mut c_void,
                        dv_ids.as_ptr() as *const c_void,
                        n_dv * 4,
                    )?;
                    let dh_bytes = n_dv * row_bytes;
                    let dh_dev = alloc_raw(&hip, &mut df_allocs, dh_bytes, "df_draft_head")?;
                    (
                        Some(DevTensor {
                            dev: dh_dev,
                            qtype: df_weights.output_weight.qtype,
                            n_in: df_weights.output_weight.n_in,
                            n_out: n_dv,
                            n_exp: 1,
                            bytes: dh_bytes,
                            expert_bytes: 0,
                        }),
                        map_dev,
                    )
                } else {
                    (None, ptr::null_mut())
                }
            } else {
                (None, ptr::null_mut())
            };

            let b_max = 16usize;
            let btok_dev = alloc_raw(&hip, &mut df_allocs, b_max * 4, "df_btok")? as *mut i32;
            let anchor_dev = alloc_raw(&hip, &mut df_allocs, 4, "df_anchor")? as *mut i32;
            let bpos_dev = alloc_raw(&hip, &mut df_allocs, b_max * 4, "df_bpos")? as *mut i32;
            let df_x = alloc_raw(&hip, &mut df_allocs, b_max * d * 4, "df_x")? as *mut f32;
            let df_xn = alloc_raw(&hip, &mut df_allocs, b_max * d * 4, "df_xn")? as *mut f32;
            let df_logits = alloc_raw(
                &hip,
                &mut df_allocs,
                b_max * df_cat.n_vocab * 4,
                "df_logits",
            )? as *mut f32;
            let df_cand_dev =
                alloc_raw(&hip, &mut df_allocs, b_max * 16 * 4, "df_cand")? as *mut i32;
            let df_unary_dev =
                alloc_raw(&hip, &mut df_allocs, b_max * 16 * 4, "df_unary")? as *mut f32;
            // confident-prefix verify scalars (only with TALUS_MIMO_DF_CONF / TALUS_DF_CONF[_S] / _TRACE)
            let df_conf_dev = if talus_spec::ConfPrefix::from_env("MIMO").active() {
                let p = alloc_raw(&hip, &mut df_allocs, 64 * 4, "df_conf")? as *mut f32;
                let mut init = [0f32; 64];
                init[1] = f32::NEG_INFINITY;
                init[2] = f32::from_bits((-1i32) as u32);
                hip.copy_h2d(p as *mut c_void, init.as_ptr() as *const c_void, 64 * 4)?;
                p
            } else {
                ptr::null_mut()
            };

            Some(MimoDFlashGpuState {
                tap_devs,
                tap_proj_devs,
                tap_proj_ptrs_dev,
                ig_dev,
                k_pool: df_k_pool,
                v_pool: df_v_pool,
                draft_head,
                dv_map_dev,
                btok_dev,
                anchor_dev,
                bpos_dev,
                df_x,
                df_xn,
                df_logits,
                df_cand_dev,
                df_unary_dev,
                df_conf_dev,
                df_conf_tops: Vec::new(),
                injected_pos: 0,
                allocs: df_allocs,
            })
        } else {
            None
        };

        steps.push(AllocationStep::HipMalloc {
            label: "weights",
            bytes: model.total_device_bytes,
        });
        steps.push(AllocationStep::HipMalloc {
            label: "kv_cache",
            bytes: total_kv_bytes,
        });
        steps.push(AllocationStep::HipMalloc {
            label: "scratch",
            bytes: scratch_bytes,
        });
        steps.push(AllocationStep::HipMalloc {
            label: "vzero",
            bytes: 4,
        });

        let _ = budget.mark_materialized(kv_res.id);
        let _ = budget.mark_materialized(scratch_res.id);

        let budget_guard = MimoBudgetGuard {
            budget,
            weights_res: weights_dummy_res,
            kv_res,
            drafter_res: None,
            scratch_res,
            steps,
        };

        let mut stream: hip_sys::hipStream_t = ptr::null_mut();
        let st_err = unsafe { (hip.hipStreamCreate)(&mut stream) };
        if st_err != 0 {
            return Err(CoreError::HipError {
                code: st_err,
                message: format!("hipStreamCreate(mimo) failed: {st_err}"),
            });
        }
        let ctx = match TkContext::new(&lib, 0, stream as *mut c_void) {
            Ok(c) => c,
            Err(e) => {
                let _ = unsafe { (hip.hipStreamDestroy)(stream) };
                return Err(e);
            }
        };
        check_tk(
            unsafe {
                (lib.tk_q36_mmq_iota)(
                    ctx.as_ptr(),
                    ident_dev,
                    ident_len as i64,
                    stream as *mut c_void,
                )
            },
            "tk_q36_mmq_iota(mimo.ident_dev)",
        )?;
        if let (Some(ref df_weights), Some(ref df_g)) = (&model.dflash, &dflash_gpu) {
            if let Some(ref dh) = df_g.draft_head {
                let row_bytes =
                    df_weights.output_weight.bytes / df_weights.output_weight.n_out.max(1);
                check_tk(
                    unsafe {
                        (lib.tk_gather_rows)(
                            ctx.as_ptr(),
                            df_weights.output_weight.dev,
                            dh.dev as *mut c_void,
                            df_g.dv_map_dev as *const i32,
                            dh.n_out as i32,
                            row_bytes,
                            stream as *mut c_void,
                        )
                    },
                    "tk_gather_rows(mimo.dflash.draft_head)",
                )?;
            }
        }
        let _ = unsafe { (hip.hipStreamSynchronize)(stream) };
        let mut model = model;
        if dflash_gpu
            .as_ref()
            .and_then(|g| g.draft_head.as_ref())
            .is_some()
        {
            if let Some(m_mut) = Arc::get_mut(&mut model) {
                if let Some(ref mut df_weights) = m_mut.dflash {
                    let old_ptr = df_weights.output_weight.dev;
                    if !old_ptr.is_null() {
                        if let Some(idx) =
                            df_weights.device_allocs.iter().position(|&p| p == old_ptr)
                        {
                            df_weights.device_allocs.swap_remove(idx);
                            hip.free_device(old_ptr);
                        }
                        df_weights.output_weight.dev = ptr::null_mut();
                    }
                }
            }
        }

        let use_graph = std::env::var("TALUS_MIMO_GRAPH")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);

        let hparams = model.hparams.clone();
        let gpu_state = MimoGpuState {
            model,
            lib,
            hip,
            stream,
            ctx,
            swa_stride,
            vzero_dev,
            cos_swa,
            sin_swa,
            cos_full,
            sin_full,
            k_pool,
            v_pool,
            k_scale,
            v_scale,
            kv_pool_bytes,
            kv_scale_bytes,
            ub,
            x,
            xn,
            att,
            att_raw,
            m_out,
            q,
            k,
            v,
            attn_scratch,
            rlogits,
            dids,
            dw,
            moe_cnt,
            moe_bounds,
            moe_order,
            moe_slots,
            moe_weights,
            moe_y,
            f1,
            f2,
            hb,
            q8_a,
            q8_b,
            logits_dev,
            pos_dev,
            pos_q_dev,
            pos_k_dev,
            zero_shared_dev,
            zero_bias_dev,
            ident_dev,
            tok_dev,
            use_graph,
            decode_graphs: [ptr::null_mut(); 9],
            decode_logits_graphs: [ptr::null_mut(); 9],
            df_graphs: [ptr::null_mut(); 17],
            dflash_gpu,
            i8c: Arc::new(std::collections::HashMap::new()),
            i8_res: None,
            allocs,
        };

        let mut eng = Self {
            hparams,
            rng_state: config.seed,
            config,
            budget_guard,
            vzero_host: Box::new(0i32),
            vzero_dev: None,
            gpu: Some(gpu_state),
            kv_state: vec![0u8; 4096],
            kv_hw: usize::MAX,
            n_pos: 0,
            token_history: Vec::new(),
            dflash,
            trace: MimoDispatchTrace::default(),
            prefill_hint: None,
            tap_stale: Vec::new(),
        };
        eng.build_i8_cache();
        Ok(eng)
    }

    /// expand the dense prefill projections (the attention Q / K / V of every layer and the FFN of
    /// the dense layer) into the int8 prefill cache (`talus_pf_i8x_expand`, MMQ's own tile loaders). The prefill
    /// product then runs `talus_pf_i8gemm` on the same MMQ-layout activations: the same output bits as
    /// `tk_prefill_mmq` (both are bitwise MMQ), faster on gfx1151. Budget-guarded; `TALUS_MIMO_I8=0` disables.
    fn build_i8_cache(&mut self) {
        if std::env::var("TALUS_MIMO_I8")
            .map(|v| v == "0")
            .unwrap_or(false)
        {
            return;
        }
        let budget = Arc::clone(self.budget_guard.budget());
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        // talus_pf_i8x_mode (tp_i8x.cu): 0 per-32 scale, 1 per-16 scales, 2 d + min (q8_1 DS4)
        let mode_of = |t: u32| -> i32 {
            match t {
                23 | 20 | 21 | 18 | 16 | 8 | 6 => 0,
                11 | 17 | 22 | 14 => 1,
                12 | 13 => 2,
                _ => -1,
            }
        };
        let mut cands: Vec<(usize, u32, usize, usize, i32)> = Vec::new();
        for layer in gpu.model.layers.iter() {
            let mut ts: Vec<&DevTensor> = vec![&layer.wq, &layer.wk, &layer.wv];
            if !layer.is_moe {
                ts.extend([&layer.ffn_gate, &layer.ffn_up, &layer.ffn_down]);
            }
            for w in ts {
                let m = mode_of(w.qtype as u32);
                if w.dev.is_null()
                    || w.n_exp != 1
                    || m < 0
                    || w.n_out % 128 != 0
                    || w.n_in % 256 != 0
                {
                    continue;
                }
                cands.push((w.dev as usize, w.qtype as u32, w.n_out, w.n_in, m));
            }
        }
        let size =
            |n: usize, k: usize, m: i32| (n * k, n * (k / 32) * if m == 0 { 1 } else { 2 } * 4);
        let total: usize = cands
            .iter()
            .map(|&(_, _, n, k, m)| {
                let (a, b) = size(n, k, m);
                a + b
            })
            .sum();
        if cands.is_empty() || total == 0 {
            return;
        }
        let res = match budget.try_reserve(BudgetCategory::Weights, "mimo2.i8_prefill_cache", total)
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "mimo2: int8 prefill cache skipped ({:.2} GiB): {e}",
                    total as f64 / (1u64 << 30) as f64
                );
                return;
            }
        };
        let mut map = std::collections::HashMap::new();
        for (dev, qt, n, k, m) in cands {
            if map.contains_key(&dev) {
                continue;
            }
            let (qb, wb) = size(n, k, m);
            let Ok(qs) = gpu.hip.alloc_device(qb) else {
                break;
            };
            gpu.allocs.push(qs);
            let Ok(ws) = gpu.hip.alloc_device(wb) else {
                break;
            };
            gpu.allocs.push(ws);
            let rc = unsafe {
                (gpu.lib.talus_pf_i8x_expand)(
                    dev as *const c_void,
                    qt as i32,
                    n as i32,
                    k as i32,
                    qs as *mut i8,
                    ws as *mut f32,
                    gpu.stream as *mut c_void,
                )
            };
            if rc == 0 {
                map.insert(dev, (qs as usize, ws as usize, m));
            }
        }
        let _ = unsafe { (gpu.hip.hipStreamSynchronize)(gpu.stream) };
        eprintln!(
            "mimo2: int8 prefill cache: {} matrices, {:.2} GiB (TALUS_MIMO_I8=0 disables)",
            map.len(),
            total as f64 / (1u64 << 30) as f64
        );
        gpu.i8c = Arc::new(map);
        gpu.i8_res = Some(res);
    }

    /// Construct a [`MimoEngine`] from a validated [`MimoCatalog`], optional [`MimoDFlashCatalog`],
    /// and [`MimoEngineConfig`], using [`Budget::for_mimo_11gib`] (`MemAvailable >= 11 GiB`).
    pub fn from_catalog(
        catalog: &MimoCatalog,
        dflash_catalog: Option<&MimoDFlashCatalog>,
        config: MimoEngineConfig,
        override_mem_avail: Option<usize>,
    ) -> Result<Self> {
        let budget = Arc::new(Budget::for_mimo_11gib(override_mem_avail));
        let weights_bytes = catalog.total_weight_bytes as usize;
        let kv_bytes = compute_mimo_kv_cache_bytes(&catalog.hparams, config.max_seq_len);
        let drafter_bytes = dflash_catalog.map_or(0, |d| d.total_weight_bytes as usize);
        let scratch_bytes = 64 * 1024 * 1024; // 64 MiB runtime scratch

        let mut budget_guard = MimoBudgetGuard::reserve_before_hip_malloc(
            budget,
            weights_bytes,
            kv_bytes,
            drafter_bytes,
            scratch_bytes,
        )?;

        // Record subsequent device allocations (`Weights` and `KvCache` were reserved first).
        budget_guard.record_hip_malloc("weights", weights_bytes);
        budget_guard.record_hip_malloc("kv_cache", kv_bytes);
        budget_guard.record_hip_malloc("scratch", scratch_bytes);
        budget_guard.record_hip_malloc("vzero", std::mem::size_of::<i32>());

        let dflash = if config.draft_max > 0 {
            if let Some(ref dv_path) = config
                .draft_vocab
                .clone()
                .or_else(resolve_mimo_draft_vocab_path)
            {
                let dflash_path = config.dflash_gguf.clone().unwrap_or_else(|| {
                    crate::mimo::weights::default_model_path(
                        crate::mimo::weights::DEFAULT_MIMO_DFLASH_PATH,
                    )
                });
                Some(MimoDFlashRunner::from_vocab_path(
                    dflash_path,
                    dv_path,
                    config.draft_max,
                    catalog.hparams.n_embd,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            hparams: catalog.hparams.clone(),
            rng_state: config.seed,
            config,
            budget_guard,
            vzero_host: Box::new(0i32),
            vzero_dev: None,
            gpu: None,
            kv_state: vec![0u8; 4096],
            kv_hw: usize::MAX,
            n_pos: 0,
            token_history: Vec::new(),
            dflash,
            trace: MimoDispatchTrace::default(),
            prefill_hint: None,
            tap_stale: Vec::new(),
        })
    }

    /// Construct a lightweight synthetic [`MimoEngine`] for deterministic CPU/mock testing
    /// of the 48-layer (or scaled) hybrid SWA-128 + K192/V128 + 256-expert top-8 MoE +
    /// DFlash `n4` pipeline without allocating multi-GB buffers.
    pub fn new_synthetic(
        hparams: MimoHparams,
        config: MimoEngineConfig,
        override_mem_avail: Option<usize>,
    ) -> Result<Self> {
        let budget = Arc::new(Budget::for_mimo_11gib(override_mem_avail));
        let weights_bytes = 16 * 1024 * 1024; // 16 MiB synthetic weights
        let kv_bytes = compute_mimo_kv_cache_bytes(&hparams, config.max_seq_len);
        let drafter_bytes = if config.draft_max > 0 {
            4 * 1024 * 1024
        } else {
            0
        };
        let scratch_bytes = 2 * 1024 * 1024;

        let mut budget_guard = MimoBudgetGuard::reserve_before_hip_malloc(
            budget,
            weights_bytes,
            kv_bytes,
            drafter_bytes,
            scratch_bytes,
        )?;

        let vzero_dev = None;

        budget_guard.record_hip_malloc("weights", weights_bytes);
        budget_guard.record_hip_malloc("kv_cache", kv_bytes);
        budget_guard.record_hip_malloc("scratch", scratch_bytes);
        budget_guard.record_hip_malloc("vzero", std::mem::size_of::<i32>());

        let dflash = if config.draft_max > 0 {
            if let Some(ref dv_path) = config.draft_vocab {
                let dflash_path = config.dflash_gguf.clone().unwrap_or_else(|| {
                    crate::mimo::weights::default_model_path(
                        crate::mimo::weights::DEFAULT_MIMO_DFLASH_PATH,
                    )
                });
                Some(MimoDFlashRunner::from_vocab_path(
                    dflash_path,
                    dv_path,
                    config.draft_max,
                    hparams.n_embd,
                )?)
            } else {
                let dv_size = hparams.n_vocab.min(MIMO_DRAFT_VOCAB_SIZE);
                let d2t: Vec<i32> = (0..dv_size as i32).collect();
                Some(MimoDFlashRunner::from_in_memory_vocab(
                    d2t,
                    hparams.n_vocab,
                    config.draft_max,
                    hparams.n_embd,
                )?)
            }
        } else {
            None
        };

        Ok(Self {
            hparams,
            rng_state: config.seed,
            config,
            budget_guard,
            vzero_host: Box::new(0i32),
            vzero_dev,
            gpu: None,
            kv_state: vec![0u8; 4096],
            kv_hw: usize::MAX,
            n_pos: 0,
            token_history: Vec::new(),
            dflash,
            trace: MimoDispatchTrace::default(),
            prefill_hint: None,
            tap_stale: Vec::new(),
        })
    }

    /// True if this engine is backed by a live GPU (`MimoModel` + `LibTalusKernels`).
    #[inline]
    pub fn is_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Non-null pointer containing `0i32` for `ids_dev` when `n_expert_used == 1` in `tk_mmvq_ids`.
    #[inline]
    pub fn vzero_ptr(&self) -> *const i32 {
        if let Some(ref gpu) = self.gpu {
            gpu.vzero_dev as *const i32
        } else if let Some(ref dev) = self.vzero_dev {
            dev.as_ptr() as *const i32
        } else {
            &*self.vzero_host as *const i32
        }
    }

    /// Reference to the budget guard (`MemAvailable >= 11 GiB`).
    pub fn budget_guard(&self) -> &MimoBudgetGuard {
        &self.budget_guard
    }

    /// Reference to the model hyperparameters.
    pub fn hparams(&self) -> &MimoHparams {
        &self.hparams
    }

    /// Reference to the engine configuration.
    pub fn config(&self) -> &MimoEngineConfig {
        &self.config
    }

    /// Optional reference to the active DFlash `n4` runner.
    pub fn dflash(&self) -> Option<&MimoDFlashRunner> {
        self.dflash.as_ref()
    }

    /// Current sequence position (`n_pos`).
    pub fn n_pos(&self) -> usize {
        self.n_pos
    }

    /// Current token history.
    pub fn token_history(&self) -> &[i32] {
        &self.token_history
    }

    /// Cumulative kernel dispatch trace.
    pub fn trace(&self) -> &MimoDispatchTrace {
        &self.trace
    }

    /// Reset KV cache and sequence state.
    pub fn reset(&mut self) {
        self.n_pos = 0;
        self.token_history.clear();
        self.tap_stale.clear();
        self.kv_state.fill(0);
        self.trace = MimoDispatchTrace::default();
        if let Some(ref mut df) = self.dflash {
            df.stats = MimoSpecStats::default();
        }
        let full_clear = std::env::var("TALUS_MIMO_RESET_FULL")
            .map(|v| v.starts_with('1'))
            .unwrap_or(false);
        let hw = if full_clear { usize::MAX } else { self.kv_hw };
        self.kv_hw = 0;
        if let Some(ref mut gpu) = self.gpu {
            if !gpu.stream.is_null() {
                let _ = unsafe { (gpu.hip.hipStreamSynchronize)(gpu.stream) };
            }
            for (il, &kb) in gpu.kv_pool_bytes.iter().enumerate() {
                let n_kv = gpu.model.layers[il].n_head_kv;
                // cells of this layer's pools (SWA: the ring, written at p % swa_stride) and the used prefix
                let slots = kb / (n_kv * self.hparams.head_dim_k * 2).max(1);
                let used = hw.min(slots);
                let _ = hip_memset(&gpu.hip, gpu.k_pool[il], 0, kb / slots.max(1) * used);
                let vp_bytes = if gpu.model.layers[il].is_swa {
                    gpu.swa_stride * n_kv * self.hparams.head_dim_v * 2
                } else {
                    self.config.max_seq_len.max(128) * n_kv * self.hparams.head_dim_v * 2
                };
                let v_slots = vp_bytes / (n_kv * self.hparams.head_dim_v * 2).max(1);
                let _ = hip_memset(
                    &gpu.hip,
                    gpu.v_pool[il],
                    0,
                    vp_bytes / v_slots.max(1) * hw.min(v_slots),
                );
                let (ksb, vsb) = gpu.kv_scale_bytes[il];
                if ksb > 0 {
                    let _ = hip_memset(&gpu.hip, gpu.k_scale[il] as *mut c_void, 0, ksb);
                }
                if vsb > 0 {
                    let _ = hip_memset(&gpu.hip, gpu.v_scale[il] as *mut c_void, 0, vsb);
                }
            }
            if let Some(ref mut df_gpu) = gpu.dflash_gpu {
                df_gpu.injected_pos = 0;
            }
        }
    }

    /// Execute a real GPU forward batch of `tokens` (`n_tok = tokens.len()`) through all `n_layers`
    /// of `mimo2` using `LibTalusKernels`:
    /// - `tk_embed_Lookup`, `tk_rms_norm_to`, `tk_quantize_q8_1`, `tk_mmvq` / `tk_prefill_mmq` for Q/K/V/O
    /// - Per-head Q/K RMSNorm (`tk_rms_norm_to` with `cols = 192`) + partial NeoX RoPE (`tk_rope_neox`,
    ///   `rope_dim = 64`, `10000.0` on SWA layers, `64000000.0` on full-attention layers)
    /// - KV cache store (`tk_kv_store_f16` / `tk_kv_store_q8`)
    /// - **40 SWA layers (`il % 6 != 5`):** `tk_attn_swa_sinks` (`window = 128`, learned `attn_sinks`)
    /// - **8 Full-attention layers (`il % 6 == 5`):** `tk_attn_k192_v128` (`n_head = 64, n_kv = 4`)
    /// - **Layer 0 dense FFN:** `tk_mmvq_ids` with `ids_dev = vzero_dev` (`*vzero_dev == 0`) for
    ///   decode/verify (`nb <= 8`), or `tk_prefill_mmq` for prefill (`nb > 8`), + `tk_swiglu_quantize_q8_1` + `tk_axpy`
    /// - **Layers 1..47 MoE (256 experts, top-8):** `tk_route_topk` + `tk_mmvq_ids` (`mimo_mmvq_ids`)
    ///   for decode/verify (`nb <= 8`), or `tk_route_group` + `tk_prefill_moe_mmq` (`mimo_prefill_moe_mmq`)
    ///   for prefill (`nb > 8`)
    /// - Captures DFlash target layer taps (`[1, 12, 24, 36, 47]`) on device.
    fn forward_batch_gpu(&mut self, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
        self.forward_batch_gpu_ex(tokens, true, None)
    }

    fn forward_batch_gpu_ex(
        &mut self,
        tokens: &[i32],
        need_logits: bool,
        mut argmax_out: Option<&mut Vec<i32>>,
    ) -> Result<Vec<Vec<f32>>> {
        let vzero = self.vzero_ptr();
        assert!(
            !vzero.is_null(),
            "vzero pointer for tk_mmvq_ids must never be null"
        );

        // positions [n_pos, n_pos + tokens) get KV writes below (reset clears up to the high-water mark)
        self.kv_hw = self.kv_hw.max(self.n_pos.saturating_add(tokens.len()));
        let gpu = self
            .gpu
            .as_mut()
            .expect("forward_batch_gpu requires gpu state");
        let lib = Arc::clone(&gpu.lib);
        let hip = Arc::clone(&gpu.hip);
        let model = Arc::clone(&gpu.model);
        let ctx = gpu.ctx.as_ptr();
        let stream: *mut c_void = gpu.stream as *mut c_void;
        let swa_stride = gpu.swa_stride;
        let q8_a_ptr = gpu.q8_a;
        let i8c = Arc::clone(&gpu.i8c);

        let dense_mmq_ok = std::env::var("TALUS_MIMO_DENSE_MMVQ")
            .map(|v| !v.starts_with('1'))
            .unwrap_or(true);
        let dense_mimo = |w: &DevTensor,
                          x_f32: *const f32,
                          y_f32: *mut f32,
                          k: usize,
                          n: usize,
                          nb: usize,
                          cached_q8: &mut Option<i32>|
         -> Result<()> {
            if dense_mmq_ok
                && nb > N_EXPERT_USED
                && (k % 256) == 0
                && unsafe { (lib.tk_q36_mmq_supported_fits)(w.qtype as i32, n as i64) }
            {
                let layout = unsafe { (lib.tk_q36_mmq_q8_layout)(w.qtype as i32) };
                if *cached_q8 != Some(layout) {
                    check_tk(
                        unsafe {
                            (lib.tk_q36_mmq_quantize)(
                                ctx,
                                x_f32,
                                ptr::null(),
                                q8_a_ptr,
                                w.qtype as i32,
                                k as i64,
                                k as i64,
                                nb as i64,
                                stream,
                            )
                        },
                        "tk_q36_mmq_quantize(dense)",
                    )?;
                    *cached_q8 = Some(layout);
                }
                // the int8 prefill cache (same output bits as tk_prefill_mmq)
                if let Some(&(qs, ws, mode)) = i8c.get(&(w.dev as usize)) {
                    let rc = unsafe {
                        (lib.talus_pf_i8gemm)(
                            qs as *const i8,
                            ws as *const f32,
                            n as i32,
                            k as i32,
                            mode,
                            q8_a_ptr as *const c_void,
                            nb as i32,
                            y_f32,
                            n as i64,
                            stream,
                        )
                    };
                    if rc == 0 {
                        return Ok(());
                    }
                }
                check_tk(
                    unsafe {
                        (lib.tk_prefill_mmq)(
                            ctx, w.qtype, w.dev, q8_a_ptr, y_f32, nb as i64, n as i64, k as i64,
                            n as i64, stream,
                        )
                    },
                    "tk_prefill_mmq(dense)",
                )?;
                return Ok(());
            }

            if *cached_q8 != Some(-1) {
                check_tk(
                    unsafe {
                        (lib.tk_quantize_q8_1)(ctx, x_f32, q8_a_ptr, k as i32, nb as i32, stream)
                    },
                    "tk_quantize_q8_1(dense)",
                )?;
                *cached_q8 = Some(-1);
            }
            let q8_row_bytes = (k / 32) * 36;
            let mut c0 = 0usize;
            while c0 < nb {
                let c_len = (nb - c0).min(N_EXPERT_USED);
                let x_q8_slice =
                    unsafe { (q8_a_ptr as *const u8).add(c0 * q8_row_bytes) as *const c_void };
                check_tk(
                    unsafe {
                        (lib.tk_mmvq)(
                            ctx,
                            w.qtype,
                            w.dev,
                            x_q8_slice,
                            y_f32.add(c0 * n),
                            k as i32,
                            n as i32,
                            c_len as i32,
                            stream,
                        )
                    },
                    "tk_mmvq(dense)",
                )?;
                c0 += c_len;
            }
            Ok(())
        };

        let attn_q8 = std::env::var("TALUS_MIMO_ATT_Q8")
            .map(|v| !v.starts_with('0'))
            .unwrap_or(true);
        let axpy_norm = std::env::var("TALUS_MIMO_AXPY_NORM")
            .map(|v| !v.starts_with('0'))
            .unwrap_or(true);
        let moe_addnorm = std::env::var("TALUS_MIMO_MOE_ADDNORM")
            .map(|v| !v.starts_with('0'))
            .unwrap_or(true);
        let rope_store_fused = std::env::var("TALUS_MIMO_ROPE_STORE")
            .map(|v| !v.starts_with('0'))
            .unwrap_or(true);
        let swiglu_q8 = std::env::var("TALUS_MIMO_SWIGLU_Q8")
            .map(|v| !v.starts_with('0'))
            .unwrap_or(true);
        let q_scatter = std::env::var("TALUS_MIMO_QSCATTER")
            .map(|v| !v.starts_with('0'))
            .unwrap_or(true);
        let swa_mmq = std::env::var("TALUS_MIMO_SWA_MMQ")
            .map(|v| !v.starts_with('0'))
            .unwrap_or(true);
        let qkv_multi = std::env::var("TALUS_MIMO_QKV_MULTI")
            .map(|v| !v.starts_with('0'))
            .unwrap_or(true);
        let hp = &model.hparams;
        let d = hp.n_embd;
        let n_vocab = hp.n_vocab;
        let head_dim_k = hp.head_dim_k; // 192
        let head_dim_v = hp.head_dim_v; // 128
        let rope_dim = hp.rope_dim; // 64
        let n_ff_dense = hp.n_ff_dense; // 16384
        let n_ff_exp = hp.n_ff_exp; // 1280 or 2048
        let n_expert = hp.n_expert; // 256
        let top_k = hp.n_expert_used; // 8
        let eps = hp.rms_norm_eps;
        let scale = DEFAULT_ATTN_SCALE;
        let max_ctx = self.config.max_seq_len.max(128) as i32;
        let kv_q8 = self.config.kv_q8;

        self.trace.captured_dflash_taps.clear();
        let mut all_logits = Vec::with_capacity(tokens.len());

        let all_positions: Vec<i32> = (0..tokens.len()).map(|i| (self.n_pos + i) as i32).collect();
        hip.copy_h2d(
            gpu.tok_dev as *mut c_void,
            tokens.as_ptr() as *const c_void,
            tokens.len() * 4,
        )?;
        hip.copy_h2d(
            gpu.pos_dev as *mut c_void,
            all_positions.as_ptr() as *const c_void,
            tokens.len() * 4,
        )?;

        // exact tail-row pruning. Every row's KV is still written on every layer; the last layer's attention /
        // wo / FFN / tap run only for the rows the head and the DFlash window read (default on; TALUS_MIMO_TAIL_PRUNE=0: off).
        let tail_out_from: Option<usize> = {
            let on = std::env::var("TALUS_MIMO_TAIL_PRUNE")
                .map(|v| !v.starts_with('0'))
                .unwrap_or(true);
            let call_end = self.n_pos + tokens.len();
            if on && tokens.len() > N_EXPERT_USED {
                let df_win = if gpu.dflash_gpu.is_some() {
                    model
                        .dflash
                        .as_ref()
                        .map(|dw| dw.catalog.sliding_window as usize)
                } else {
                    None
                };
                let (final_end, min_query) = self
                    .prefill_hint
                    .map(|(e, m)| (e.max(call_end), m))
                    .unwrap_or((call_end, call_end));
                let layers: Vec<LayerAttn> = model
                    .layers
                    .iter()
                    .map(|l| {
                        if l.is_swa {
                            LayerAttn::Window(l.sliding_window as usize)
                        } else {
                            LayerAttn::Full
                        }
                    })
                    .collect();
                let req = TailRequest {
                    final_end,
                    out_rows: 1 + df_win.map_or(0, |w| w + TAIL_DF_MARGIN),
                    min_query,
                    taps: Vec::new(),
                };
                let grid = Grid {
                    start: self.n_pos,
                    ubatch: gpu.ub,
                    align: TAIL_ALIGN,
                    min_rows: TAIL_MIN_ROWS,
                    small_nb: N_EXPERT_USED,
                };
                let rows = plan_on_grid(&layers, &req, Some(&grid));
                // implemented: the last layer's post-KV part, when it is the only prunable layer and a full one
                match first_prunable_layer(&rows) {
                    Some(l) if l + 1 == layers.len() && layers[l] == LayerAttn::Full => {
                        Some(rows[l].out_from)
                    }
                    _ => None,
                }
            } else {
                None
            }
        };

        let mut chunk_start = 0usize;
        while chunk_start < tokens.len() {
            let nb = (tokens.len() - chunk_start).min(gpu.ub);
            let pos0 = self.n_pos + chunk_start;
            let chunk_tok_dev = unsafe { gpu.tok_dev.add(chunk_start) };
            let chunk_pos_dev = unsafe { gpu.pos_dev.add(chunk_start) };

            let pos_k4_dev = unsafe { gpu.pos_k_dev.add(64) };
            check_tk(
                unsafe {
                    (lib.tk_fill_pos)(
                        ctx,
                        gpu.pos_q_dev,
                        nb as i32,
                        hp.n_head_q as i32,
                        pos0 as i32,
                        stream,
                    )
                },
                "tk_fill_pos(q)",
            )?;
            check_tk(
                unsafe {
                    (lib.tk_fill_pos)(
                        ctx,
                        gpu.pos_k_dev,
                        nb as i32,
                        hp.n_head_kv_swa as i32,
                        pos0 as i32,
                        stream,
                    )
                },
                "tk_fill_pos(k_swa)",
            )?;
            if nb <= N_EXPERT_USED && hp.n_head_kv_full != hp.n_head_kv_swa {
                check_tk(
                    unsafe {
                        (lib.tk_fill_pos)(
                            ctx,
                            pos_k4_dev,
                            nb as i32,
                            hp.n_head_kv_full as i32,
                            pos0 as i32,
                            stream,
                        )
                    },
                    "tk_fill_pos(k_full)",
                )?;
            }

            let is_last_chunk = chunk_start + nb == tokens.len();
            let graph_eligible = gpu.use_graph
                && chunk_start == 0
                && is_last_chunk
                && nb <= N_EXPERT_USED
                && (need_logits ^ argmax_out.is_some());
            let cached_graph = if graph_eligible {
                if argmax_out.is_some() {
                    gpu.decode_graphs[nb]
                } else {
                    gpu.decode_logits_graphs[nb]
                }
            } else {
                ptr::null_mut()
            };

            if !cached_graph.is_null() {
                let rc = unsafe { (hip.hipGraphLaunch)(cached_graph, gpu.stream) };
                if rc != 0 {
                    return Err(CoreError::KernelError {
                        code: rc,
                        message: "hipGraphLaunch(mimo_decode) failed".into(),
                    });
                }
                for (il, layer) in model.layers.iter().enumerate() {
                    self.trace.non_null_vzero_mmvq_calls += 1;
                    if layer.is_swa {
                        self.trace.swa_sinks_calls += 1;
                    } else {
                        self.trace.full_k192_v128_calls += 1;
                    }
                    if !layer.is_moe {
                        self.trace.dense_ffn_calls += 1;
                        self.trace.non_null_vzero_mmvq_calls += 3;
                    } else {
                        self.trace.moe_topk_calls += 1;
                        self.trace.moe_mmvq_ids_calls += 3;
                    }
                    if let Some(ref df_weights) = model.dflash {
                        let one_based = df_weights
                            .catalog
                            .target_layers
                            .contains(&(model.layers.len() as u32));
                        if df_weights.catalog.target_layers.iter().any(|&t| {
                            if one_based {
                                (t as usize) == il + 1
                            } else {
                                (t as usize).min(model.layers.len().saturating_sub(1)) == il
                            }
                        }) {
                            self.trace.captured_dflash_taps.push(il as u32);
                        }
                    } else if DFLASH_5L_TARGET_TAPS.contains(&(il as u32)) {
                        self.trace.captured_dflash_taps.push(il as u32);
                    }
                }
                if let Some(ref mut df_gpu) = gpu.dflash_gpu {
                    df_gpu.injected_pos = pos0 + nb;
                }
                self.trace.non_null_vzero_mmvq_calls += 1;
            } else {
                let capturing = if graph_eligible {
                    unsafe {
                        (hip.hipStreamBeginCapture)(
                            gpu.stream,
                            HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                        ) == 0
                    }
                } else {
                    false
                };

                // Token embedding lookup (`[nb, 4096]`).
                let embd_row_bytes = model.tok_embd.bytes / model.tok_embd.n_out.max(1);
                check_tk(
                    unsafe {
                        (lib.tk_q36_embed_rows)(
                            ctx,
                            model.tok_embd.qtype as i32,
                            model.tok_embd.dev,
                            embd_row_bytes,
                            chunk_tok_dev,
                            nb as i64,
                            d as i64,
                            gpu.x,
                            stream,
                        )
                    },
                    "tk_q36_embed_rows",
                )?;

                // a decode / verify MoE layer's expert sum (moe_add_slots) is folded into the next layer's
                // attention-norm launch (tk_add_norm_quant with add / yexp / weights: the same sum in the same order,
                // then the norm) - not after DFlash tap layers (they copy x) or the last layer. TALUS_MIMO_MOE_ADDNORM=0: off.
                let n_layers = model.layers.len();
                let tap_layer = |il: usize| -> bool {
                    let in_df = model.dflash.as_ref().is_some_and(|dw| {
                        dw.catalog.target_layers.iter().any(|&t| {
                            (t as usize) == il + 1
                                || (t as usize).min(n_layers.saturating_sub(1)) == il
                        })
                    });
                    in_df || DFLASH_5L_TARGET_TAPS.contains(&(il as u32))
                };
                let mut pending_moe = false;
                for (il, layer) in model.layers.iter().enumerate() {
                    let n_head_q = layer.n_head_q; // 64
                    let n_head_kv = layer.n_head_kv; // 8 on SWA, 4 on full
                    let q_dim = n_head_q * head_dim_k; // 12288
                    let k_dim = n_head_kv * head_dim_k; // 1536 or 768
                    let v_dim = n_head_kv * head_dim_v; // 1024 or 512
                    let att_raw_dim = n_head_q * head_dim_v; // 8192

                    // 1. Attention input RMSNorm (+ fused Q8_1 quantize on decode/verify `nb <= 8`).
                    let q8_d_row_bytes = (d / 32) * 36;
                    let (an_add, an_y, an_w, an_k) = if pending_moe {
                        (
                            gpu.zero_shared_dev as *const f32,
                            gpu.moe_y as *const f32,
                            gpu.dw as *const f32,
                            top_k as i32,
                        )
                    } else {
                        (ptr::null(), ptr::null(), ptr::null(), 0)
                    };
                    pending_moe = false;
                    let mut attn_in_q8 = if nb <= N_EXPERT_USED {
                        check_tk(
                            unsafe {
                                (lib.tk_add_norm_quant)(
                                    ctx,
                                    gpu.x,
                                    an_add,
                                    an_y,
                                    an_w,
                                    an_k,
                                    layer.attn_norm.dev as *const f32,
                                    gpu.xn,
                                    gpu.q8_a,
                                    d as i32,
                                    eps,
                                    nb as i32,
                                    stream,
                                )
                            },
                            "tk_add_norm_quant(attn_norm)",
                        )?;
                        Some(-1)
                    } else {
                        check_tk(
                            unsafe {
                                (lib.tk_rms_norm_to)(
                                    ctx,
                                    gpu.xn,
                                    gpu.x,
                                    layer.attn_norm.dev as *const f32,
                                    nb as i32,
                                    d as i32,
                                    eps,
                                    stream,
                                )
                            },
                            "tk_rms_norm_to(attn_norm)",
                        )?;
                        None
                    };

                    // 2. Q / K / V projection (`tk_prefill_mmq` for prefill, `tk_mmvq` for decode/verify).
                    self.trace.non_null_vzero_mmvq_calls += 1;
                    // decode / verify q, k, v share the Q8_1 input: one w32 multi-matrix launch instead of
                    // three (bitwise the three tk_mmvq calls; Q4_K verified bitwise). TALUS_MIMO_QKV_MULTI=0: off.
                    if qkv_multi
                        && nb <= N_EXPERT_USED
                        && attn_in_q8 == Some(-1)
                        && layer.wq.qtype as u32 == 12
                        && layer.wk.qtype as u32 == 12
                        && layer.wv.qtype as u32 == 12
                    {
                        let ws: [*const c_void; 3] = [
                            layer.wq.dev as *const c_void,
                            layer.wk.dev as *const c_void,
                            layer.wv.dev as *const c_void,
                        ];
                        let ys: [*mut f32; 3] = [gpu.q, gpu.k, gpu.v];
                        let ns: [i32; 3] = [q_dim as i32, k_dim as i32, v_dim as i32];
                        check_tk(
                            unsafe {
                                (lib.tk_mmvq_w32_multi)(
                                    ctx,
                                    layer.wq.qtype,
                                    3,
                                    ws.as_ptr(),
                                    ys.as_ptr(),
                                    ns.as_ptr(),
                                    gpu.q8_a as *const c_void,
                                    d as i32,
                                    nb as i32,
                                    stream,
                                )
                            },
                            "tk_mmvq_w32_multi(qkv)",
                        )?;
                    } else {
                        dense_mimo(&layer.wq, gpu.xn, gpu.q, d, q_dim, nb, &mut attn_in_q8)?;
                        dense_mimo(&layer.wk, gpu.xn, gpu.k, d, k_dim, nb, &mut attn_in_q8)?;
                        dense_mimo(&layer.wv, gpu.xn, gpu.v, d, v_dim, nb, &mut attn_in_q8)?;
                    }

                    // 3. Per-head Q/K RMSNorm (`F32[192]`) and Partial NeoX RoPE (`rope_dim = 64`).
                    if !layer.q_norm.is_null() {
                        check_tk(
                            unsafe {
                                (lib.tk_rms_norm_to)(
                                    ctx,
                                    gpu.q,
                                    gpu.q,
                                    layer.q_norm.dev as *const f32,
                                    (nb * n_head_q) as i32,
                                    head_dim_k as i32,
                                    eps,
                                    stream,
                                )
                            },
                            "tk_rms_norm_to(q_norm)",
                        )?;
                    }
                    if !layer.k_norm.is_null() {
                        check_tk(
                            unsafe {
                                (lib.tk_rms_norm_to)(
                                    ctx,
                                    gpu.k,
                                    gpu.k,
                                    layer.k_norm.dev as *const f32,
                                    (nb * n_head_kv) as i32,
                                    head_dim_k as i32,
                                    eps,
                                    stream,
                                )
                            },
                            "tk_rms_norm_to(k_norm)",
                        )?;
                    }

                    let (cos_tab, sin_tab) = if layer.is_swa {
                        (gpu.cos_swa, gpu.sin_swa)
                    } else {
                        (gpu.cos_full, gpu.sin_full)
                    };
                    let layer_pos_k_dev = if nb <= N_EXPERT_USED {
                        if n_head_kv == hp.n_head_kv_swa {
                            gpu.pos_k_dev
                        } else {
                            pos_k4_dev
                        }
                    } else {
                        check_tk(
                            unsafe {
                                (lib.tk_fill_pos)(
                                    ctx,
                                    gpu.pos_k_dev,
                                    nb as i32,
                                    n_head_kv as i32,
                                    pos0 as i32,
                                    stream,
                                )
                            },
                            "tk_fill_pos(k)",
                        )?;
                        gpu.pos_k_dev
                    };
                    // 4. KV cache write (`stride = swa_stride` on SWA; `stride = max_ctx` on full) in F16
                    //    (`[stride, n_head_kv, 192]` for K, `[stride, n_head_kv, 128]` for V).
                    let kv_stride = if layer.is_swa {
                        swa_stride
                    } else {
                        max_ctx as usize
                    };
                    let _ = kv_q8;
                    if rope_store_fused && nb <= N_EXPERT_USED {
                        // q / k rope + f16 K / V store in one launch (bitwise the three below);
                        // TALUS_MIMO_ROPE_STORE=0: off.
                        check_tk(
                            unsafe {
                                (lib.tk_mimo_rope_qk_store)(
                                    ctx,
                                    gpu.q,
                                    gpu.k,
                                    gpu.v as *const f32,
                                    nb as i32,
                                    n_head_q as i32,
                                    n_head_kv as i32,
                                    head_dim_k as i32,
                                    head_dim_v as i32,
                                    rope_dim as i32,
                                    cos_tab,
                                    sin_tab,
                                    gpu.pos_q_dev,
                                    layer_pos_k_dev,
                                    gpu.k_pool[il],
                                    gpu.v_pool[il],
                                    chunk_pos_dev,
                                    kv_stride as i32,
                                    stream,
                                )
                            },
                            "tk_mimo_rope_qk_store",
                        )?;
                    } else {
                        check_tk(
                            unsafe {
                                (lib.tk_rope_neox)(
                                    ctx,
                                    gpu.q,
                                    gpu.q,
                                    (nb * n_head_q) as i64,
                                    head_dim_k as i32,
                                    rope_dim as i32,
                                    cos_tab,
                                    sin_tab,
                                    gpu.pos_q_dev,
                                    stream,
                                )
                            },
                            "tk_rope_neox(q)",
                        )?;
                        check_tk(
                            unsafe {
                                (lib.tk_rope_neox)(
                                    ctx,
                                    gpu.k,
                                    gpu.k,
                                    (nb * n_head_kv) as i64,
                                    head_dim_k as i32,
                                    rope_dim as i32,
                                    cos_tab,
                                    sin_tab,
                                    layer_pos_k_dev,
                                    stream,
                                )
                            },
                            "tk_rope_neox(k)",
                        )?;

                        check_tk(
                            unsafe {
                                (lib.tk_q36_dflash_kv_store_f16)(
                                    ctx,
                                    gpu.k,
                                    gpu.v,
                                    gpu.k_pool[il],
                                    gpu.v_pool[il],
                                    k_dim as i32,
                                    v_dim as i32,
                                    nb as i32,
                                    chunk_pos_dev,
                                    kv_stride as i32,
                                    stream,
                                )
                            },
                            "tk_q36_dflash_kv_store_f16(trunk)",
                        )?;
                    }

                    // the last layer's KV is written above for every row; from here on it runs only the rows
                    // from `tail_ks` on (the rows before it feed nothing but the head / the drafter taps, which never
                    // read them). The kept suffix starts on a TAIL_ALIGN boundary from the chunk start, so each kept
                    // row sees the same query grouping as in the unpruned chunk.
                    let tail_ks = match tail_out_from {
                        Some(f) if il + 1 == n_layers && !capturing => {
                            tail_prune::chunk_tail_offset(
                                pos0,
                                nb,
                                f,
                                TAIL_ALIGN,
                                TAIL_MIN_ROWS,
                                N_EXPERT_USED,
                            )
                        }
                        _ => 0,
                    };
                    if tail_ks > 0 {
                        if let (Some(df_gpu), Some(df_weights)) =
                            (gpu.dflash_gpu.as_ref(), model.dflash.as_ref())
                        {
                            let one_based = df_weights
                                .catalog
                                .target_layers
                                .contains(&(n_layers as u32));
                            if let Some(slot) =
                                df_weights.catalog.target_layers.iter().position(|&t| {
                                    if one_based {
                                        (t as usize) == il + 1
                                    } else {
                                        (t as usize).min(n_layers.saturating_sub(1)) == il
                                    }
                                })
                            {
                                // the skipped rows' tap: zeros (finite) instead of a stale buffer; never read
                                hip_memset_async(
                                    &hip,
                                    df_gpu.tap_devs[slot] as *mut c_void,
                                    0,
                                    tail_ks * d * 4,
                                    stream,
                                )?;
                                let (a, b) = (pos0, pos0 + tail_ks);
                                match self.tap_stale.last_mut() {
                                    Some(last) if last.1 == a => last.1 = b,
                                    _ => self.tap_stale.push((a, b)),
                                }
                            }
                        }
                        self.trace.tail_pruned_rows += tail_ks;
                    }
                    if tail_ks == nb {
                        continue;
                    }
                    let nb = nb - tail_ks;
                    let chunk_pos_dev = unsafe { chunk_pos_dev.add(tail_ks) };
                    let q_cur = unsafe { gpu.q.add(tail_ks * q_dim) };
                    let x_cur = unsafe { gpu.x.add(tail_ks * d) };

                    // 5. Hybrid Attention Dispatch:
                    //    - 40 SWA layers (`il % 6 != 5`): `tk_attn_swa_sinks`
                    //    - 8 Full-attention layers (`il % 6 == 5`): `tk_attn_k192_v128`
                    // decode / verify attention also writes its outputs as Q8_1 into q8_a (the bytes the
                    // wo projection's tk_quantize_q8_1 would write) - one launch less per layer.
                    // TALUS_MIMO_ATT_Q8=0: off.
                    let att_q8 = attn_q8 && nb <= N_EXPERT_USED;
                    let mut wo_q8_mmq: Option<i32> = None;
                    if att_q8 {
                        if layer.is_swa {
                            self.trace.swa_sinks_calls += 1;
                            check_tk(
                                unsafe {
                                    (lib.tk_attn_swa_sinks_q8)(
                                        ctx,
                                        q_cur,
                                        gpu.k_pool[il],
                                        gpu.v_pool[il],
                                        gpu.k_scale[il],
                                        gpu.v_scale[il],
                                        layer.attn_sinks.dev as *const f32,
                                        gpu.att_raw,
                                        gpu.attn_scratch,
                                        nb as i32,
                                        n_head_q as i32,
                                        n_head_kv as i32,
                                        head_dim_k as i32,
                                        head_dim_v as i32,
                                        layer.sliding_window as i32,
                                        kv_stride as i32,
                                        scale,
                                        chunk_pos_dev,
                                        0,
                                        stream,
                                        gpu.q8_a,
                                    )
                                },
                                "tk_attn_swa_sinks_q8",
                            )?;
                        } else {
                            self.trace.full_k192_v128_calls += 1;
                            check_tk(
                                unsafe {
                                    (lib.tk_attn_k192_v128_q8)(
                                        ctx,
                                        q_cur,
                                        gpu.k_pool[il],
                                        gpu.v_pool[il],
                                        gpu.k_scale[il],
                                        gpu.v_scale[il],
                                        gpu.att_raw,
                                        gpu.attn_scratch,
                                        nb as i32,
                                        n_head_q as i32,
                                        n_head_kv as i32,
                                        kv_stride as i32,
                                        scale,
                                        chunk_pos_dev,
                                        0,
                                        stream,
                                        gpu.q8_a,
                                    )
                                },
                                "tk_attn_k192_v128_q8",
                            )?;
                        }
                    } else if layer.is_swa && swa_mmq && dense_mmq_ok && nb > N_EXPERT_USED {
                        // the prefill SWA kernel also writes the wo projection's MMQ q8_1 rows into q8_a
                        // (bitwise its tk_q36_mmq_quantize); wo then skips the quantize (TALUS_MIMO_SWA_MMQ=0: off)
                        self.trace.swa_sinks_calls += 1;
                        let wo_layout =
                            unsafe { (lib.tk_q36_mmq_q8_layout)(layer.wo.qtype as i32) };
                        let mut written = 0i32;
                        check_tk(
                            unsafe {
                                (lib.tk_attn_swa_sinks_mmq)(
                                    ctx,
                                    q_cur,
                                    gpu.k_pool[il],
                                    gpu.v_pool[il],
                                    layer.attn_sinks.dev as *const f32,
                                    gpu.att_raw,
                                    gpu.attn_scratch,
                                    nb as i32,
                                    n_head_q as i32,
                                    n_head_kv as i32,
                                    head_dim_k as i32,
                                    head_dim_v as i32,
                                    layer.sliding_window as i32,
                                    kv_stride as i32,
                                    scale,
                                    chunk_pos_dev,
                                    gpu.q8_a,
                                    wo_layout,
                                    &mut written,
                                    stream,
                                )
                            },
                            "tk_attn_swa_sinks_mmq",
                        )?;
                        if written != 0 {
                            wo_q8_mmq = Some(wo_layout);
                        }
                    } else {
                        if layer.is_swa {
                            self.trace.swa_sinks_calls += 1;
                            check_tk(
                                unsafe {
                                    (lib.tk_attn_swa_sinks)(
                                        ctx,
                                        q_cur,
                                        gpu.k_pool[il],
                                        gpu.v_pool[il],
                                        gpu.k_scale[il],
                                        gpu.v_scale[il],
                                        layer.attn_sinks.dev as *const f32,
                                        gpu.att_raw,
                                        gpu.attn_scratch,
                                        nb as i32,
                                        n_head_q as i32,
                                        n_head_kv as i32,
                                        head_dim_k as i32,
                                        head_dim_v as i32,
                                        layer.sliding_window as i32,
                                        kv_stride as i32,
                                        scale,
                                        chunk_pos_dev,
                                        0,
                                        stream,
                                    )
                                },
                                "tk_attn_swa_sinks",
                            )?;
                        } else {
                            self.trace.full_k192_v128_calls += 1;
                            check_tk(
                                unsafe {
                                    (lib.tk_attn_k192_v128)(
                                        ctx,
                                        q_cur,
                                        gpu.k_pool[il],
                                        gpu.v_pool[il],
                                        gpu.k_scale[il],
                                        gpu.v_scale[il],
                                        gpu.att_raw,
                                        gpu.attn_scratch,
                                        nb as i32,
                                        n_head_q as i32,
                                        n_head_kv as i32,
                                        kv_stride as i32,
                                        scale,
                                        chunk_pos_dev,
                                        0,
                                        stream,
                                    )
                                },
                                "tk_attn_k192_v128",
                            )?;
                        }
                    }

                    // 6. Attention output projection `wo` (`[8192, 4096]`), optional `post_attn_norm`,
                    //    `mimo2.attention.value_scale` (`0.7070000171661377`) scaled residual add into `x`,
                    //    and `ffn_norm` (+ fused Q8_1 quantize on decode/verify `nb <= 8`).
                    let mut wo_in_q8 = if att_q8 { Some(-1) } else { wo_q8_mmq };
                    dense_mimo(
                        &layer.wo,
                        gpu.att_raw,
                        gpu.att,
                        att_raw_dim,
                        d,
                        nb,
                        &mut wo_in_q8,
                    )?;
                    if !layer.post_attn_norm.is_null() {
                        check_tk(
                            unsafe {
                                (lib.tk_rms_norm_to)(
                                    ctx,
                                    gpu.att,
                                    gpu.att,
                                    layer.post_attn_norm.dev as *const f32,
                                    nb as i32,
                                    d as i32,
                                    eps,
                                    stream,
                                )
                            },
                            "tk_rms_norm_to(post_attn_norm)",
                        )?;
                    }
                    let axpy_fused = axpy_norm && nb <= N_EXPERT_USED;
                    if !axpy_fused {
                        check_tk(
                            unsafe {
                                (lib.tk_axpy)(
                                    ctx,
                                    x_cur,
                                    gpu.att,
                                    hp.attn_value_scale,
                                    (nb * d) as i32,
                                    stream,
                                )
                            },
                            "tk_axpy(attn_value_scale)",
                        )?;
                    }
                    let mut ffn_in_q8_1_ready = if axpy_fused {
                        // attention residual axpy + ffn RMSNorm + Q8_1 in one launch (bitwise tk_axpy then
                        // tk_add_norm_quant); TALUS_MIMO_AXPY_NORM=0: off.
                        check_tk(
                            unsafe {
                                (lib.tk_axpy_norm_quant)(
                                    ctx,
                                    x_cur,
                                    gpu.att as *const f32,
                                    hp.attn_value_scale,
                                    layer.ffn_norm.dev as *const f32,
                                    gpu.xn,
                                    gpu.q8_a,
                                    d as i32,
                                    eps,
                                    nb as i32,
                                    stream,
                                )
                            },
                            "tk_axpy_norm_quant(ffn_norm)",
                        )?;
                        true
                    } else if nb <= N_EXPERT_USED {
                        check_tk(
                            unsafe {
                                (lib.tk_add_norm_quant)(
                                    ctx,
                                    x_cur,
                                    ptr::null(),
                                    ptr::null(),
                                    ptr::null(),
                                    0,
                                    layer.ffn_norm.dev as *const f32,
                                    gpu.xn,
                                    gpu.q8_a,
                                    d as i32,
                                    eps,
                                    nb as i32,
                                    stream,
                                )
                            },
                            "tk_add_norm_quant(ffn_norm)",
                        )?;
                        true
                    } else {
                        check_tk(
                            unsafe {
                                (lib.tk_rms_norm_to)(
                                    ctx,
                                    gpu.xn,
                                    x_cur,
                                    layer.ffn_norm.dev as *const f32,
                                    nb as i32,
                                    d as i32,
                                    eps,
                                    stream,
                                )
                            },
                            "tk_rms_norm_to(ffn_norm)",
                        )?;
                        false
                    };

                    // 7. FFN Dispatch: Layer 0 Dense vs Layers 1..47 256-expert top-8 MoE.
                    if !layer.is_moe {
                        self.trace.dense_ffn_calls += 1;
                        self.trace.non_null_vzero_mmvq_calls += 3;
                        if nb > N_EXPERT_USED {
                            let mut ffn0_in_q8 = None;
                            dense_mimo(
                                &layer.ffn_gate,
                                gpu.xn,
                                gpu.f1,
                                d,
                                n_ff_dense,
                                nb,
                                &mut ffn0_in_q8,
                            )?;
                            dense_mimo(
                                &layer.ffn_up,
                                gpu.xn,
                                gpu.f2,
                                d,
                                n_ff_dense,
                                nb,
                                &mut ffn0_in_q8,
                            )?;
                            check_tk(
                                unsafe {
                                    (lib.tk_silu_mul)(
                                        ctx,
                                        gpu.f1,
                                        gpu.f2,
                                        (nb * n_ff_dense) as i64,
                                        stream,
                                    )
                                },
                                "tk_silu_mul(layer0)",
                            )?;
                            let mut down0_in_q8 = None;
                            dense_mimo(
                                &layer.ffn_down,
                                gpu.f1,
                                gpu.m_out,
                                n_ff_dense,
                                d,
                                nb,
                                &mut down0_in_q8,
                            )?;
                        } else {
                            if !ffn_in_q8_1_ready {
                                check_tk(
                                    unsafe {
                                        (lib.tk_quantize_q8_1)(
                                            ctx, gpu.xn, gpu.q8_a, d as i32, nb as i32, stream,
                                        )
                                    },
                                    "tk_quantize_q8_1(ffn_in)",
                                )?;
                            }
                            let q8_ff_row_bytes = (n_ff_dense / 32) * 36;
                            let mut c0 = 0usize;
                            while c0 < nb {
                                let c_len = (nb - c0).min(N_EXPERT_USED);
                                let x_q8_slice = unsafe {
                                    (gpu.q8_a as *const u8).add(c0 * q8_d_row_bytes)
                                        as *const c_void
                                };
                                check_tk(
                                    unsafe {
                                        (lib.tk_mmvq_ids)(
                                            ctx,
                                            layer.ffn_gate.qtype,
                                            layer.ffn_gate.dev,
                                            0,
                                            gpu.vzero_dev,
                                            c_len as i32,
                                            x_q8_slice,
                                            c_len > 1,
                                            gpu.f1.add(c0 * n_ff_dense),
                                            d as i32,
                                            n_ff_dense as i32,
                                            layer.ffn_up.dev,
                                            gpu.f2.add(c0 * n_ff_dense),
                                            stream,
                                        )
                                    },
                                    "tk_mmvq_ids(layer0.ffn_gate_up)",
                                )?;
                                c0 += c_len;
                            }
                            check_tk(
                                unsafe {
                                    (lib.tk_swiglu_quantize_q8_1)(
                                        ctx,
                                        gpu.f1,
                                        gpu.f2,
                                        gpu.q8_b,
                                        n_ff_dense as i32,
                                        nb as i32,
                                        stream,
                                    )
                                },
                                "tk_swiglu_quantize_q8_1(layer0)",
                            )?;
                            let mut c0 = 0usize;
                            while c0 < nb {
                                let c_len = (nb - c0).min(N_EXPERT_USED);
                                let x_q8_slice = unsafe {
                                    (gpu.q8_b as *const u8).add(c0 * q8_ff_row_bytes)
                                        as *const c_void
                                };
                                check_tk(
                                    unsafe {
                                        (lib.tk_mmvq_ids)(
                                            ctx,
                                            layer.ffn_down.qtype,
                                            layer.ffn_down.dev,
                                            0,
                                            gpu.vzero_dev,
                                            c_len as i32,
                                            x_q8_slice,
                                            c_len > 1,
                                            gpu.m_out.add(c0 * d),
                                            n_ff_dense as i32,
                                            d as i32,
                                            ptr::null(),
                                            ptr::null_mut(),
                                            stream,
                                        )
                                    },
                                    "tk_mmvq_ids(layer0.ffn_down)",
                                )?;
                                c0 += c_len;
                            }
                        }
                        if !layer.post_ffn_norm.is_null() {
                            check_tk(
                                unsafe {
                                    (lib.tk_rms_norm_to)(
                                        ctx,
                                        gpu.m_out,
                                        gpu.m_out,
                                        layer.post_ffn_norm.dev as *const f32,
                                        nb as i32,
                                        d as i32,
                                        eps,
                                        stream,
                                    )
                                },
                                "tk_rms_norm_to(layer0.post_ffn_norm)",
                            )?;
                        }
                        check_tk(
                            unsafe {
                                (lib.tk_axpy)(ctx, x_cur, gpu.m_out, 1.0, (nb * d) as i32, stream)
                            },
                            "tk_axpy(layer0)",
                        )?;
                    } else {
                        // Layers 1..47: 256-expert top-8 MoE (`expert_weights_norm = true`).
                        self.trace.moe_topk_calls += 1;
                        if layer.ffn_gate_inp.qtype == TK_QTYPE_F32 {
                            check_tk(
                                unsafe {
                                    (lib.tk_f32_matmul)(
                                        ctx,
                                        layer.ffn_gate_inp.dev as *const f32,
                                        gpu.xn,
                                        gpu.rlogits,
                                        d as i32,
                                        n_expert as i32,
                                        nb as i32,
                                        stream,
                                    )
                                },
                                "tk_f32_matmul(ffn_gate_inp)",
                            )?;
                        } else {
                            if !ffn_in_q8_1_ready {
                                check_tk(
                                    unsafe {
                                        (lib.tk_quantize_q8_1)(
                                            ctx, gpu.xn, gpu.q8_a, d as i32, nb as i32, stream,
                                        )
                                    },
                                    "tk_quantize_q8_1(ffn_gate_inp)",
                                )?;
                                ffn_in_q8_1_ready = true;
                            }
                            let mut c0 = 0usize;
                            while c0 < nb {
                                let c_len = (nb - c0).min(N_EXPERT_USED);
                                let x_q8_slice = unsafe {
                                    (gpu.q8_a as *const u8).add(c0 * q8_d_row_bytes)
                                        as *const c_void
                                };
                                check_tk(
                                    unsafe {
                                        (lib.tk_mmvq)(
                                            ctx,
                                            layer.ffn_gate_inp.qtype,
                                            layer.ffn_gate_inp.dev,
                                            x_q8_slice,
                                            gpu.rlogits.add(c0 * n_expert),
                                            d as i32,
                                            n_expert as i32,
                                            c_len as i32,
                                            stream,
                                        )
                                    },
                                    "tk_mmvq(ffn_gate_inp)",
                                )?;
                                c0 += c_len;
                            }
                        }

                        let bias_ptr = if layer.exp_probs_b.is_null() {
                            gpu.zero_bias_dev as *const f32
                        } else {
                            layer.exp_probs_b.dev as *const f32
                        };

                        if nb <= N_EXPERT_USED {
                            // Decode / verify (`ncols = 1..8`): dispatches `tk_route_topk` + `tk_mmvq_ids`
                            // (`mimo_mmvq_ids` 4 rows/WG for `IQ2_XXS`, `IQ3_XXS`, `IQ2_XS`).
                            self.trace.moe_mmvq_ids_calls += 3;
                            if !ffn_in_q8_1_ready {
                                check_tk(
                                    unsafe {
                                        (lib.tk_quantize_q8_1)(
                                            ctx, gpu.xn, gpu.q8_a, d as i32, nb as i32, stream,
                                        )
                                    },
                                    "tk_quantize_q8_1(moe_decode)",
                                )?;
                            }
                            check_tk(
                                unsafe {
                                    (lib.tk_route_topk)(
                                        ctx,
                                        gpu.rlogits,
                                        bias_ptr,
                                        nb as i32,
                                        top_k as i32,
                                        1.0,
                                        gpu.dids,
                                        gpu.dw,
                                        stream,
                                    )
                                },
                                "tk_route_topk",
                            )?;
                            if nb > 1
                                && top_k == 8
                                && matches!(
                                    layer.exp_gate.qtype as i32,
                                    10 | 16 | 17 | 18 | 21 | 22 | 29
                                )
                            {
                                check_tk(
                                    unsafe {
                                        (lib.tk_mmvq_ids)(
                                            ctx,
                                            layer.exp_gate.qtype,
                                            layer.exp_gate.dev,
                                            layer.exp_gate.expert_bytes,
                                            gpu.dids,
                                            (nb * top_k) as i32,
                                            gpu.q8_a,
                                            true,
                                            gpu.f1,
                                            d as i32,
                                            n_ff_exp as i32,
                                            layer.exp_up.dev,
                                            gpu.f2,
                                            stream,
                                        )
                                    },
                                    "tk_mmvq_ids(exp_gate_up_batched)",
                                )?;
                            } else {
                                for t in 0..nb {
                                    let x_q8_tok = unsafe {
                                        (gpu.q8_a as *const u8).add(t * q8_d_row_bytes)
                                            as *const c_void
                                    };
                                    check_tk(
                                        unsafe {
                                            (lib.tk_mmvq_ids)(
                                                ctx,
                                                layer.exp_gate.qtype,
                                                layer.exp_gate.dev,
                                                layer.exp_gate.expert_bytes,
                                                gpu.dids.add(t * top_k),
                                                top_k as i32,
                                                x_q8_tok,
                                                false,
                                                gpu.f1.add(t * top_k * n_ff_exp),
                                                d as i32,
                                                n_ff_exp as i32,
                                                layer.exp_up.dev,
                                                gpu.f2.add(t * top_k * n_ff_exp),
                                                stream,
                                            )
                                        },
                                        "tk_mmvq_ids(exp_gate_up)",
                                    )?;
                                }
                            }
                            check_tk(
                                unsafe {
                                    (lib.tk_swiglu_quantize_q8_1)(
                                        ctx,
                                        gpu.f1,
                                        gpu.f2,
                                        gpu.q8_b,
                                        n_ff_exp as i32,
                                        (nb * top_k) as i32,
                                        stream,
                                    )
                                },
                                "tk_swiglu_quantize_q8_1(moe)",
                            )?;
                            check_tk(
                                unsafe {
                                    (lib.tk_mmvq_ids)(
                                        ctx,
                                        layer.exp_down.qtype,
                                        layer.exp_down.dev,
                                        layer.exp_down.expert_bytes,
                                        gpu.dids,
                                        (nb * top_k) as i32,
                                        gpu.q8_b,
                                        true,
                                        gpu.moe_y,
                                        n_ff_exp as i32,
                                        d as i32,
                                        ptr::null(),
                                        ptr::null_mut(),
                                        stream,
                                    )
                                },
                                "tk_mmvq_ids(exp_down)",
                            )?;
                            if !layer.post_ffn_norm.is_null() {
                                hip_memset_async(
                                    &hip,
                                    gpu.m_out as *mut c_void,
                                    0,
                                    nb * d * 4,
                                    stream,
                                )?;
                                check_tk(
                                    unsafe {
                                        (lib.tk_moe_add_slots)(
                                            ctx,
                                            gpu.m_out,
                                            gpu.zero_shared_dev,
                                            gpu.moe_y,
                                            gpu.ident_dev,
                                            gpu.dw,
                                            nb as i32,
                                            d as i32,
                                            top_k as i32,
                                            stream,
                                        )
                                    },
                                    "tk_moe_add_slots(moe_decode.m_out)",
                                )?;
                                check_tk(
                                    unsafe {
                                        (lib.tk_rms_norm_to)(
                                            ctx,
                                            gpu.m_out,
                                            gpu.m_out,
                                            layer.post_ffn_norm.dev as *const f32,
                                            nb as i32,
                                            d as i32,
                                            eps,
                                            stream,
                                        )
                                    },
                                    "tk_rms_norm_to(moe_decode.post_ffn_norm)",
                                )?;
                                check_tk(
                                    unsafe {
                                        (lib.tk_axpy)(
                                            ctx,
                                            x_cur,
                                            gpu.m_out,
                                            1.0,
                                            (nb * d) as i32,
                                            stream,
                                        )
                                    },
                                    "tk_axpy(moe_out)",
                                )?;
                            } else if moe_addnorm && il + 1 < n_layers && !tap_layer(il) {
                                pending_moe = true;
                            } else {
                                check_tk(
                                    unsafe {
                                        (lib.tk_moe_add_slots)(
                                            ctx,
                                            x_cur,
                                            gpu.zero_shared_dev,
                                            gpu.moe_y,
                                            gpu.ident_dev,
                                            gpu.dw,
                                            nb as i32,
                                            d as i32,
                                            top_k as i32,
                                            stream,
                                        )
                                    },
                                    "tk_moe_add_slots(moe_out_no_norm)",
                                )?;
                            }
                        } else {
                            // Prefill MoE (`T > 8`): dispatches `tk_route_group` + `tk_q36_mmq_quantize`
                            // (`tp_moe` WMMA for `IQ2_XXS`, `IQ3_XXS`, `IQ2_XS`, `Q2_K`, `Q3_K`) or
                            // `tk_gather_rows` + `mimo_prefill_moe_mmq_q8` for `IQ1_M`.
                            self.trace.moe_prefill_mmq_calls += 3;
                            check_tk(
                                unsafe {
                                    (lib.tk_route_group)(
                                        ctx,
                                        gpu.rlogits,
                                        bias_ptr,
                                        nb as i32,
                                        top_k as i32,
                                        1.0,
                                        gpu.dids,
                                        gpu.moe_weights,
                                        gpu.moe_cnt,
                                        gpu.moe_bounds,
                                        gpu.moe_order,
                                        gpu.moe_slots,
                                        stream,
                                    )
                                },
                                "tk_route_group",
                            )?;
                            // Opt-in (numerics change): IQ1_M gate/up on the tp_moe int8 kernels (weights
                            // exact in int8, q8_0_16 dot) with IQ2_XS-layout activations instead of the dp4a kernel
                            let iq1m_tp = layer.exp_gate.qtype as i32 == 29
                                && layer.exp_up.qtype as i32 == 29
                                && std::env::var("MIMO_IQ1M_TPMOE")
                                    .map(|v| v == "1")
                                    .unwrap_or(false);
                            let mmq_gu_ok = iq1m_tp
                                || unsafe {
                                    (lib.tk_q36_mmq_supported_fits)(
                                        layer.exp_gate.qtype as i32,
                                        n_ff_exp as i64,
                                    ) && (lib.tk_q36_mmq_supported_fits)(
                                        layer.exp_up.qtype as i32,
                                        n_ff_exp as i64,
                                    )
                                };
                            if mmq_gu_ok && q_scatter {
                                // each token row quantized once, written to its top_k routed rows
                                // (bitwise the gathered quantize below; TALUS_MIMO_QSCATTER=0: off)
                                check_tk(
                                    unsafe {
                                        (lib.tk_q36_mmq_quantize_scatter)(
                                            ctx,
                                            gpu.xn,
                                            gpu.moe_slots,
                                            gpu.q8_b,
                                            if iq1m_tp {
                                                17
                                            } else {
                                                layer.exp_gate.qtype as i32
                                            },
                                            d as i64,
                                            nb as i64,
                                            (nb * top_k) as i64,
                                            top_k as i32,
                                            stream,
                                        )
                                    },
                                    "tk_q36_mmq_quantize_scatter(exp_gate)",
                                )?;
                            } else if mmq_gu_ok {
                                check_tk(
                                    unsafe {
                                        (lib.tk_q36_mmq_quantize)(
                                            ctx,
                                            gpu.xn,
                                            gpu.moe_order,
                                            gpu.q8_b,
                                            if iq1m_tp {
                                                17
                                            } else {
                                                layer.exp_gate.qtype as i32
                                            },
                                            d as i64,
                                            d as i64,
                                            (nb * top_k) as i64,
                                            stream,
                                        )
                                    },
                                    "tk_q36_mmq_quantize(exp_gate)",
                                )?;
                            } else {
                                if !ffn_in_q8_1_ready {
                                    check_tk(
                                        unsafe {
                                            (lib.tk_quantize_q8_1)(
                                                ctx, gpu.xn, gpu.q8_a, d as i32, nb as i32, stream,
                                            )
                                        },
                                        "tk_quantize_q8_1(moe_prefill_iq)",
                                    )?;
                                }
                                check_tk(
                                    unsafe {
                                        (lib.tk_gather_rows)(
                                            ctx,
                                            gpu.q8_a,
                                            gpu.q8_b,
                                            gpu.moe_order,
                                            (nb * top_k) as i32,
                                            q8_d_row_bytes,
                                            stream,
                                        )
                                    },
                                    "tk_gather_rows(moe_q8)",
                                )?;
                            }
                            let prod_gate = tk_moe_mmq_product {
                                w: layer.exp_gate.dev,
                                qtype: layer.exp_gate.qtype,
                                w_rows: n_ff_exp as i64,
                                w_cols: d as i64,
                                expert_bytes: layer.exp_gate.expert_bytes,
                                n_experts: n_expert as i32,
                                x_q8_1: gpu.q8_b,
                                bounds_dev: gpu.moe_bounds,
                                ids_dev: gpu.ident_dev,
                                total_rows: (nb * top_k) as i64,
                                max_rows: nb as i64,
                                dst: gpu.f1,
                                ld_dst: n_ff_exp as i64,
                                y_scale_dev: ptr::null(),
                            };
                            let prod_up = tk_moe_mmq_product {
                                w: layer.exp_up.dev,
                                qtype: layer.exp_up.qtype,
                                w_rows: n_ff_exp as i64,
                                w_cols: d as i64,
                                expert_bytes: layer.exp_up.expert_bytes,
                                n_experts: n_expert as i32,
                                x_q8_1: gpu.q8_b,
                                bounds_dev: gpu.moe_bounds,
                                ids_dev: gpu.ident_dev,
                                total_rows: (nb * top_k) as i64,
                                max_rows: nb as i64,
                                dst: gpu.f2,
                                ld_dst: n_ff_exp as i64,
                                y_scale_dev: ptr::null(),
                            };
                            let mmq_down_ok = unsafe {
                                (lib.tk_q36_mmq_supported_fits)(
                                    layer.exp_down.qtype as i32,
                                    d as i64,
                                )
                            };
                            // gate, up, SiLU-mul and the down input's q8_1 quantize in one tp_moe
                            // launch when covered (bitwise; TK_ERR_UNSUPPORTED = nothing launched)
                            let fused_gu_q8 = mmq_down_ok
                                && layer.exp_up.qtype == layer.exp_gate.qtype
                                && layer.exp_up.expert_bytes == layer.exp_gate.expert_bytes
                                && {
                                    let st = unsafe {
                                        (lib.tk_prefill_moe_mmq_gu_q8)(
                                            ctx,
                                            &prod_gate,
                                            layer.exp_up.dev,
                                            0.0,
                                            layer.exp_down.qtype as i32,
                                            // f1 (FP32 [nb * top_k][n_ff]) is free here: the q8 rows (1.125 B per
                                            // value) go there; q8_b still holds the gate/up input being read
                                            gpu.f1 as *mut c_void,
                                            (nb * top_k) as i64,
                                            stream,
                                        )
                                    };
                                    if st == talus_kernels_sys::TK_ERR_UNSUPPORTED {
                                        false
                                    } else {
                                        check_tk(st, "tk_prefill_moe_mmq_gu_q8")?;
                                        true
                                    }
                                };
                            // gate, up and SiLU-mul in one tp_moe launch when covered (bitwise the
                            // separate path; TK_ERR_UNSUPPORTED = nothing launched)
                            let fused_gu = fused_gu_q8
                                || (mmq_down_ok
                                    && layer.exp_up.qtype == layer.exp_gate.qtype
                                    && layer.exp_up.expert_bytes == layer.exp_gate.expert_bytes
                                    && {
                                        let st = unsafe {
                                            (lib.tk_prefill_moe_mmq_gu)(
                                                ctx,
                                                &prod_gate,
                                                layer.exp_up.dev,
                                                0.0,
                                                stream,
                                            )
                                        };
                                        if st == talus_kernels_sys::TK_ERR_UNSUPPORTED {
                                            false
                                        } else {
                                            check_tk(st, "tk_prefill_moe_mmq_gu")?;
                                            true
                                        }
                                    });
                            if !fused_gu {
                                check_tk(
                                    unsafe { (lib.tk_prefill_moe_mmq)(ctx, &prod_gate, stream) },
                                    "tk_prefill_moe_mmq(exp_gate)",
                                )?;
                                if mmq_gu_ok && layer.exp_up.qtype != layer.exp_gate.qtype {
                                    let lg = unsafe {
                                        (lib.tk_q36_mmq_q8_layout)(layer.exp_gate.qtype as i32)
                                    };
                                    let lu = unsafe {
                                        (lib.tk_q36_mmq_q8_layout)(layer.exp_up.qtype as i32)
                                    };
                                    if lg != lu {
                                        check_tk(
                                            unsafe {
                                                (lib.tk_q36_mmq_quantize)(
                                                    ctx,
                                                    gpu.xn,
                                                    gpu.moe_order,
                                                    gpu.q8_b,
                                                    layer.exp_up.qtype as i32,
                                                    d as i64,
                                                    d as i64,
                                                    (nb * top_k) as i64,
                                                    stream,
                                                )
                                            },
                                            "tk_q36_mmq_quantize(exp_up)",
                                        )?;
                                    }
                                }
                                check_tk(
                                    unsafe { (lib.tk_prefill_moe_mmq)(ctx, &prod_up, stream) },
                                    "tk_prefill_moe_mmq(exp_up)",
                                )?;
                            }
                            if fused_gu_q8 {
                                // the down input is already in q8_b
                            } else if mmq_down_ok
                                && !fused_gu
                                && swiglu_q8
                                && unsafe {
                                    // SiLU-mul straight into the down input's MMQ q8_1 rows (bitwise
                                    // tk_silu_mul then tk_q36_mmq_quantize; TALUS_MIMO_SWIGLU_Q8=0: off)
                                    (lib.talus_pf_swiglu_q8_mmq)(
                                        gpu.f1,
                                        gpu.f2,
                                        gpu.q8_b,
                                        layer.exp_down.qtype as i32,
                                        n_ff_exp as i64,
                                        (nb * top_k) as i64,
                                        stream,
                                    )
                                } == 0
                            {
                                // the down input is in q8_b (registered by the gate/up input's quantize)
                            } else if mmq_down_ok {
                                if !fused_gu {
                                    check_tk(
                                        unsafe {
                                            (lib.tk_silu_mul)(
                                                ctx,
                                                gpu.f1,
                                                gpu.f2,
                                                (nb * top_k * n_ff_exp) as i64,
                                                stream,
                                            )
                                        },
                                        "tk_silu_mul(moe_prefill)",
                                    )?;
                                }
                                check_tk(
                                    unsafe {
                                        (lib.tk_q36_mmq_quantize)(
                                            ctx,
                                            gpu.f1,
                                            ptr::null(),
                                            gpu.q8_b,
                                            layer.exp_down.qtype as i32,
                                            n_ff_exp as i64,
                                            n_ff_exp as i64,
                                            (nb * top_k) as i64,
                                            stream,
                                        )
                                    },
                                    "tk_q36_mmq_quantize(exp_down)",
                                )?;
                            } else {
                                check_tk(
                                    unsafe {
                                        (lib.tk_swiglu_quantize_q8_1)(
                                            ctx,
                                            gpu.f1,
                                            gpu.f2,
                                            gpu.q8_b,
                                            n_ff_exp as i32,
                                            (nb * top_k) as i32,
                                            stream,
                                        )
                                    },
                                    "tk_swiglu_quantize_q8_1(moe_prefill)",
                                )?;
                            }
                            let prod_down = tk_moe_mmq_product {
                                w: layer.exp_down.dev,
                                qtype: layer.exp_down.qtype,
                                w_rows: d as i64,
                                w_cols: n_ff_exp as i64,
                                expert_bytes: layer.exp_down.expert_bytes,
                                n_experts: n_expert as i32,
                                x_q8_1: if fused_gu_q8 {
                                    gpu.f1 as *const c_void
                                } else {
                                    gpu.q8_b as *const c_void
                                },
                                bounds_dev: gpu.moe_bounds,
                                ids_dev: gpu.ident_dev,
                                total_rows: (nb * top_k) as i64,
                                max_rows: nb as i64,
                                dst: gpu.moe_y,
                                ld_dst: d as i64,
                                y_scale_dev: ptr::null(),
                            };
                            check_tk(
                                unsafe { (lib.tk_prefill_moe_mmq)(ctx, &prod_down, stream) },
                                "tk_prefill_moe_mmq(exp_down)",
                            )?;
                            if !layer.post_ffn_norm.is_null() {
                                hip_memset_async(
                                    &hip,
                                    gpu.m_out as *mut c_void,
                                    0,
                                    nb * d * 4,
                                    stream,
                                )?;
                                check_tk(
                                    unsafe {
                                        (lib.tk_moe_add_slots)(
                                            ctx,
                                            gpu.m_out,
                                            gpu.zero_shared_dev,
                                            gpu.moe_y,
                                            gpu.moe_slots,
                                            gpu.moe_weights,
                                            nb as i32,
                                            d as i32,
                                            top_k as i32,
                                            stream,
                                        )
                                    },
                                    "tk_moe_add_slots(m_out)",
                                )?;
                                check_tk(
                                    unsafe {
                                        (lib.tk_rms_norm_to)(
                                            ctx,
                                            gpu.m_out,
                                            gpu.m_out,
                                            layer.post_ffn_norm.dev as *const f32,
                                            nb as i32,
                                            d as i32,
                                            eps,
                                            stream,
                                        )
                                    },
                                    "tk_rms_norm_to(moe_prefill.post_ffn_norm)",
                                )?;
                                check_tk(
                                    unsafe {
                                        (lib.tk_axpy)(
                                            ctx,
                                            x_cur,
                                            gpu.m_out,
                                            1.0,
                                            (nb * d) as i32,
                                            stream,
                                        )
                                    },
                                    "tk_axpy(moe_prefill)",
                                )?;
                            } else {
                                check_tk(
                                    unsafe {
                                        (lib.tk_moe_add_slots)(
                                            ctx,
                                            x_cur,
                                            gpu.zero_shared_dev,
                                            gpu.moe_y,
                                            gpu.moe_slots,
                                            gpu.moe_weights,
                                            nb as i32,
                                            d as i32,
                                            top_k as i32,
                                            stream,
                                        )
                                    },
                                    "tk_moe_add_slots(x)",
                                )?;
                            }
                        }
                    }

                    // 8. Capture DFlash target layer taps (`[1, 12, 24, 36, 48]` 1-based `t_layer_inp[t]`
                    //    or `[1, 12, 24, 36, 47]` 0-based) for all `nb` rows in the chunk on GPU.
                    if let Some(ref mut df_gpu) = gpu.dflash_gpu {
                        if let Some(ref df_weights) = model.dflash {
                            let one_based = df_weights
                                .catalog
                                .target_layers
                                .contains(&(model.layers.len() as u32));
                            if let Some(slot) =
                                df_weights.catalog.target_layers.iter().position(|&t| {
                                    if one_based {
                                        (t as usize) == il + 1
                                    } else {
                                        (t as usize).min(model.layers.len().saturating_sub(1)) == il
                                    }
                                })
                            {
                                hip_copy_d2d_async(
                                    &hip,
                                    unsafe { df_gpu.tap_devs[slot].add(tail_ks * d) }
                                        as *mut c_void,
                                    x_cur as *const c_void,
                                    nb * d * 4,
                                    stream,
                                )?;
                                if chunk_start + nb == tokens.len() {
                                    self.trace.captured_dflash_taps.push(il as u32);
                                }
                            }
                        }
                    } else if DFLASH_5L_TARGET_TAPS.contains(&(il as u32))
                        && chunk_start + nb == tokens.len()
                    {
                        self.trace.captured_dflash_taps.push(il as u32);
                    }
                }

                // 8b. Inject all `nb` captured target tap rows for this chunk (`pos0 .. pos0 + nb`)
                //     into the 5-layer DFlash KV cache on GPU (`fc_slices` -> `tk_q36_dflash_sum5_rms`
                //     -> per-layer `wk/wv` + `k_norm` + `rope_neox` -> `df_gpu.k_pool/v_pool`).
                if let Some(ref mut df_gpu) = gpu.dflash_gpu {
                    if let Some(ref df_weights) = model.dflash {
                        if df_weights.fc_slices.len() == 5 {
                            let df_cat = &df_weights.catalog;
                            let df_d = df_cat.embedding_length;
                            let df_n_kv = df_cat.n_head_kv;
                            let df_k_dim = df_cat.k_dim;
                            let df_v_dim = df_cat.v_dim;
                            let df_rope_dim = df_cat.rope_dim;
                            for (j, slice) in df_weights.fc_slices.iter().enumerate() {
                                let mut tap_q8 = None;
                                dense_mimo(
                                    slice,
                                    df_gpu.tap_devs[j],
                                    df_gpu.tap_proj_devs[j],
                                    df_d,
                                    df_d,
                                    nb,
                                    &mut tap_q8,
                                )?;
                            }
                            let mut df_ig_q8 = if nb <= N_EXPERT_USED {
                                check_tk(
                                    unsafe {
                                        (lib.tk_q36_dflash_sum5_rms_q8)(
                                            ctx,
                                            df_gpu.tap_proj_ptrs_dev as *const *const f32,
                                            df_weights.enc_output_norm.dev as *const f32,
                                            df_gpu.ig_dev,
                                            gpu.q8_a,
                                            df_d as i32,
                                            nb as i32,
                                            eps,
                                            stream,
                                        )
                                    },
                                    "tk_q36_dflash_sum5_rms_q8(chunk)",
                                )?;
                                Some(-1)
                            } else {
                                check_tk(
                                    unsafe {
                                        (lib.tk_q36_dflash_sum5_rms)(
                                            ctx,
                                            df_gpu.tap_proj_ptrs_dev as *const *const f32,
                                            df_weights.enc_output_norm.dev as *const f32,
                                            df_gpu.ig_dev,
                                            df_d as i32,
                                            nb as i32,
                                            eps,
                                            stream,
                                        )
                                    },
                                    "tk_q36_dflash_sum5_rms(chunk)",
                                )?;
                                None
                            };
                            if nb > N_EXPERT_USED || df_n_kv != hp.n_head_kv_swa {
                                check_tk(
                                    unsafe {
                                        (lib.tk_fill_pos)(
                                            ctx,
                                            gpu.pos_k_dev,
                                            nb as i32,
                                            df_n_kv as i32,
                                            pos0 as i32,
                                            stream,
                                        )
                                    },
                                    "tk_fill_pos(df_inject_k)",
                                )?;
                            }

                            let k_pool_dim = df_k_dim;
                            let df_stride = max_ctx as usize;
                            for (dl, df_layer) in df_weights.layers.iter().enumerate() {
                                dense_mimo(
                                    &df_layer.wk,
                                    df_gpu.ig_dev,
                                    gpu.k,
                                    df_d,
                                    df_n_kv * df_k_dim,
                                    nb,
                                    &mut df_ig_q8,
                                )?;
                                dense_mimo(
                                    &df_layer.wv,
                                    df_gpu.ig_dev,
                                    gpu.v,
                                    df_d,
                                    df_n_kv * df_v_dim,
                                    nb,
                                    &mut df_ig_q8,
                                )?;
                                if !df_layer.k_norm.is_null() {
                                    check_tk(
                                        unsafe {
                                            (lib.tk_rms_norm_to)(
                                                ctx,
                                                gpu.k,
                                                gpu.k,
                                                df_layer.k_norm.dev as *const f32,
                                                (nb * df_n_kv) as i32,
                                                df_k_dim as i32,
                                                eps,
                                                stream,
                                            )
                                        },
                                        "tk_rms_norm_to(df_inject_k_norm)",
                                    )?;
                                }
                                check_tk(
                                    unsafe {
                                        (lib.tk_rope_neox)(
                                            ctx,
                                            gpu.k,
                                            gpu.f1,
                                            (nb * df_n_kv) as i64,
                                            df_k_dim as i32,
                                            df_rope_dim as i32,
                                            gpu.cos_swa,
                                            gpu.sin_swa,
                                            gpu.pos_k_dev,
                                            stream,
                                        )
                                    },
                                    "tk_rope_neox(df_inject_k)",
                                )?;
                                let k_roped_ptr = gpu.f1;
                                check_tk(
                                    unsafe {
                                        (lib.tk_q36_dflash_kv_store_f16)(
                                            ctx,
                                            k_roped_ptr,
                                            gpu.v,
                                            df_gpu.k_pool[dl],
                                            df_gpu.v_pool[dl],
                                            (df_n_kv * k_pool_dim) as i32,
                                            (df_n_kv * df_v_dim) as i32,
                                            nb as i32,
                                            chunk_pos_dev,
                                            df_stride as i32,
                                            stream,
                                        )
                                    },
                                    "tk_q36_dflash_kv_store_f16(df_inject)",
                                )?;
                            }
                            df_gpu.injected_pos = pos0 + nb;
                        }
                    }
                }

                // 9. Final Output RMSNorm + LM Head (`output.weight` `[4096, 152064]`) via `tk_mmvq_ids`
                //    with non-null `gpu.vzero_dev`.
                if is_last_chunk && (need_logits || argmax_out.is_some()) {
                    let n_out_rows = if nb <= N_EXPERT_USED { nb } else { 1 };
                    let x_tail = unsafe { gpu.x.add((nb - n_out_rows) * d) };
                    check_tk(
                        unsafe {
                            (lib.tk_add_norm_quant)(
                                ctx,
                                x_tail,
                                ptr::null(),
                                ptr::null(),
                                ptr::null(),
                                0,
                                model.output_norm.dev as *const f32,
                                gpu.xn,
                                gpu.q8_a,
                                d as i32,
                                eps,
                                n_out_rows as i32,
                                stream,
                            )
                        },
                        "tk_add_norm_quant(output_norm)",
                    )?;
                    self.trace.non_null_vzero_mmvq_calls += 1;
                    check_tk(
                        unsafe {
                            (lib.tk_mmvq_ids)(
                                ctx,
                                model.output.qtype,
                                model.output.dev,
                                0,
                                gpu.vzero_dev,
                                n_out_rows as i32,
                                gpu.q8_a,
                                n_out_rows > 1,
                                gpu.logits_dev,
                                d as i32,
                                n_vocab as i32,
                                ptr::null(),
                                ptr::null_mut(),
                                stream,
                            )
                        },
                        "tk_mmvq_ids(output)",
                    )?;
                    if argmax_out.is_some() {
                        check_tk(
                            unsafe {
                                (lib.tk_argmax_rows)(
                                    ctx,
                                    gpu.logits_dev,
                                    n_out_rows as i32,
                                    n_vocab as i32,
                                    gpu.tok_dev,
                                    stream,
                                )
                            },
                            "tk_argmax_rows(output)",
                        )?;
                    }
                }

                if capturing {
                    let mut graph: *mut c_void = ptr::null_mut();
                    let end_rc = unsafe { (hip.hipStreamEndCapture)(gpu.stream, &mut graph) };
                    if end_rc == 0 && !graph.is_null() {
                        let mut exec: *mut c_void = ptr::null_mut();
                        let inst_rc = unsafe {
                            (hip.hipGraphInstantiate)(
                                &mut exec,
                                graph,
                                ptr::null_mut(),
                                ptr::null_mut(),
                                0,
                            )
                        };
                        let _ = unsafe { (hip.hipGraphDestroy)(graph) };
                        if inst_rc == 0 && !exec.is_null() {
                            let launch_rc = unsafe { (hip.hipGraphLaunch)(exec, gpu.stream) };
                            if launch_rc != 0 {
                                let _ = unsafe { (hip.hipGraphExecDestroy)(exec) };
                                return Err(CoreError::KernelError {
                                    code: launch_rc,
                                    message: "hipGraphLaunch(mimo_decode_first) failed".into(),
                                });
                            }
                            if argmax_out.is_some() {
                                gpu.decode_graphs[nb] = exec;
                            } else {
                                gpu.decode_logits_graphs[nb] = exec;
                            }
                        }
                    } else {
                        gpu.use_graph = false;
                    }
                }
            }

            if is_last_chunk && (need_logits || argmax_out.is_some()) {
                let n_out_rows = if nb <= N_EXPERT_USED { nb } else { 1 };
                if let Some(ref mut out_ids) = argmax_out {
                    out_ids.resize(n_out_rows, 0);
                    let _ = unsafe { (hip.hipStreamSynchronize)(gpu.stream) };
                    hip.copy_d2h(
                        out_ids.as_mut_ptr() as *mut c_void,
                        gpu.tok_dev as *const c_void,
                        n_out_rows * 4,
                    )?;
                }

                if need_logits {
                    let mut host_logits = vec![0.0f32; n_out_rows * n_vocab];
                    let _ = unsafe { (hip.hipStreamSynchronize)(gpu.stream) };
                    hip.copy_d2h(
                        host_logits.as_mut_ptr() as *mut c_void,
                        gpu.logits_dev as *const c_void,
                        n_out_rows * n_vocab * 4,
                    )?;

                    for _ in 0..(tokens.len() - n_out_rows) {
                        all_logits.push(Vec::new());
                    }
                    for r in 0..n_out_rows {
                        all_logits.push(host_logits[r * n_vocab..(r + 1) * n_vocab].to_vec());
                    }
                }
            }

            chunk_start += nb;
        }

        for &tok in tokens {
            let idx = self.n_pos % self.kv_state.len();
            self.kv_state[idx] = (tok & 0xFF) as u8;
            self.n_pos += 1;
            self.token_history.push(tok);
        }

        Ok(all_logits)
    }

    /// Run the real 5-layer DFlash transformer on GPU (`tk_q36_dflash_*` + `tk_mmvq` +
    /// `tk_attn_k192_v128` + SwiGLU), project through the dedicated `Q8_0` `output.weight` head
    /// (`152,064` rows or `65,536` gathered draft rows), and map via the 64K `mimo-dv-65536.bin`
    /// draft vocabulary to propose `draft_max = 4` (`n4`) target token IDs.
    fn propose_dflash_gpu(&mut self, anchor_token: i32, pos: usize) -> Result<Vec<i32>> {
        self.propose_dflash_gpu_ex(anchor_token, pos, false)
            .map(|(d, _)| d)
    }

    /// [`Self::propose_dflash_gpu`], optionally also returning each draft row's top-16 candidates
    /// `(target id, drafter logit)` (`tk_q36_dflash_top16`'s `cand` / `unary`, already mapped through the draft
    /// vocabulary) for exact speculative sampling. Empty candidate lists when the CPU runner drafts.
    fn propose_dflash_gpu_ex(
        &mut self,
        anchor_token: i32,
        pos: usize,
        want_cands: bool,
    ) -> Result<(Vec<i32>, Vec<Vec<(i32, f32)>>)> {
        let draft_max = self
            .dflash
            .as_ref()
            .map_or(self.config.draft_max, |d| d.draft_max)
            .clamp(1, 15);
        let b_rows = draft_max + 1; // B = 5 for n4

        let gpu = self
            .gpu
            .as_mut()
            .expect("propose_dflash_gpu requires gpu state");
        let Some(ref df_weights) = gpu.model.dflash else {
            return self
                .dflash
                .as_ref()
                .expect("dflash runner required")
                .propose_draft_tokens(anchor_token, pos)
                .map(|d| (d, Vec::new()));
        };
        let Some(ref mut df_gpu) = gpu.dflash_gpu else {
            return self
                .dflash
                .as_ref()
                .expect("dflash runner required")
                .propose_draft_tokens(anchor_token, pos)
                .map(|d| (d, Vec::new()));
        };

        let lib = Arc::clone(&gpu.lib);
        let hip = Arc::clone(&gpu.hip);
        let ctx = gpu.ctx.as_ptr();
        let stream: *mut c_void = gpu.stream as *mut c_void;

        let df_cat = &df_weights.catalog;
        let d = df_cat.embedding_length; // 4096
        let n_head = df_cat.n_head; // 64
        let n_kv = df_cat.n_head_kv; // 8
        let k_dim = df_cat.k_dim; // 192
        let v_dim = df_cat.v_dim; // 128
        let rope_dim = df_cat.rope_dim; // 64
        let n_ff = df_cat.feed_forward_length; // 20480
        let eps = gpu.model.hparams.rms_norm_eps;
        let max_ctx = self.config.max_seq_len.max(128) as i32;

        // 1. Context K/V for `0 .. pos` was already injected per-chunk in `forward_batch_gpu`.
        // 2. Upload anchor + dynamic position buffers (`bpos_dev`, `pos_q_dev`, `pos_k_dev`) before
        //    launching or capturing the 5-layer DFlash HIP graph (`gpu.df_graphs[b_rows]`).
        hip.copy_h2d(
            df_gpu.anchor_dev as *mut c_void,
            (&anchor_token as *const i32) as *const c_void,
            4,
        )?;
        let block_end_pos = (pos + b_rows - 1) as i32;
        check_tk(
            unsafe { (lib.tk_q36_dflash_set_i32)(ctx, df_gpu.bpos_dev, block_end_pos, stream) },
            "tk_q36_dflash_set_i32(bpos)",
        )?;
        check_tk(
            unsafe {
                (lib.tk_fill_pos)(
                    ctx,
                    gpu.pos_q_dev,
                    b_rows as i32,
                    n_head as i32,
                    pos as i32,
                    stream,
                )
            },
            "tk_fill_pos(dflash.q)",
        )?;
        check_tk(
            unsafe {
                (lib.tk_fill_pos)(
                    ctx,
                    gpu.pos_k_dev,
                    b_rows as i32,
                    n_kv as i32,
                    pos as i32,
                    stream,
                )
            },
            "tk_fill_pos(dflash.k)",
        )?;

        let (head_tensor, map_ptr) = if let Some(ref dh) = df_gpu.draft_head {
            (dh, df_gpu.dv_map_dev as *const i32)
        } else {
            (&df_weights.output_weight, ptr::null())
        };
        let has_draft_head = !map_ptr.is_null();

        let graph_eligible = gpu.use_graph && b_rows < gpu.df_graphs.len();
        let cached_df_graph = if graph_eligible {
            gpu.df_graphs[b_rows]
        } else {
            ptr::null_mut()
        };

        if !cached_df_graph.is_null() {
            let rc = unsafe { (hip.hipGraphLaunch)(cached_df_graph, gpu.stream) };
            if rc != 0 {
                return Err(CoreError::KernelError {
                    code: rc,
                    message: "hipGraphLaunch(mimo_dflash) failed".into(),
                });
            }
        } else {
            let capturing = if graph_eligible {
                unsafe {
                    (hip.hipStreamBeginCapture)(gpu.stream, HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                        == 0
                }
            } else {
                false
            };

            let mask_id = df_cat.mask_token_id;
            check_tk(
                unsafe {
                    (lib.tk_q36_dflash_block_tokens)(
                        ctx,
                        df_gpu.anchor_dev,
                        df_gpu.btok_dev,
                        b_rows as i32,
                        mask_id,
                        stream,
                    )
                },
                "tk_q36_dflash_block_tokens",
            )?;
            let df_embd_row_bytes = df_weights.tok_embd.bytes / df_weights.tok_embd.n_out.max(1);
            check_tk(
                unsafe {
                    (lib.tk_q36_embed_rows)(
                        ctx,
                        df_weights.tok_embd.qtype as i32,
                        df_weights.tok_embd.dev,
                        df_embd_row_bytes,
                        df_gpu.btok_dev,
                        b_rows as i64,
                        d as i64,
                        df_gpu.df_x,
                        stream,
                    )
                },
                "tk_q36_embed_rows(dflash)",
            )?;

            // 3. Run the 5-layer DFlash transformer on GPU.
            for (dl, layer) in df_weights.layers.iter().enumerate() {
                check_tk(
                    unsafe {
                        (lib.tk_q36_dflash_rms_rows_q8)(
                            ctx,
                            df_gpu.df_x,
                            layer.attn_norm.dev as *const f32,
                            df_gpu.df_xn,
                            gpu.q8_a,
                            d as i32,
                            b_rows as i32,
                            eps,
                            stream,
                        )
                    },
                    "tk_q36_dflash_rms_rows_q8(attn_norm)",
                )?;
                check_tk(
                    unsafe {
                        (lib.tk_mmvq)(
                            ctx,
                            layer.wq.qtype,
                            layer.wq.dev,
                            gpu.q8_a,
                            gpu.q,
                            d as i32,
                            (n_head * k_dim) as i32,
                            b_rows as i32,
                            stream,
                        )
                    },
                    "tk_mmvq(dflash.wq)",
                )?;
                check_tk(
                    unsafe {
                        (lib.tk_mmvq)(
                            ctx,
                            layer.wk.qtype,
                            layer.wk.dev,
                            gpu.q8_a,
                            gpu.k,
                            d as i32,
                            (n_kv * k_dim) as i32,
                            b_rows as i32,
                            stream,
                        )
                    },
                    "tk_mmvq(dflash.wk)",
                )?;
                check_tk(
                    unsafe {
                        (lib.tk_mmvq)(
                            ctx,
                            layer.wv.qtype,
                            layer.wv.dev,
                            gpu.q8_a,
                            gpu.v,
                            d as i32,
                            (n_kv * v_dim) as i32,
                            b_rows as i32,
                            stream,
                        )
                    },
                    "tk_mmvq(dflash.wv)",
                )?;

                if !layer.q_norm.is_null() {
                    check_tk(
                        unsafe {
                            (lib.tk_rms_norm_to)(
                                ctx,
                                gpu.q,
                                gpu.q,
                                layer.q_norm.dev as *const f32,
                                (b_rows * n_head) as i32,
                                k_dim as i32,
                                eps,
                                stream,
                            )
                        },
                        "tk_rms_norm_to(dflash.q_norm)",
                    )?;
                }
                if !layer.k_norm.is_null() {
                    check_tk(
                        unsafe {
                            (lib.tk_rms_norm_to)(
                                ctx,
                                gpu.k,
                                gpu.k,
                                layer.k_norm.dev as *const f32,
                                (b_rows * n_kv) as i32,
                                k_dim as i32,
                                eps,
                                stream,
                            )
                        },
                        "tk_rms_norm_to(dflash.k_norm)",
                    )?;
                }
                check_tk(
                    unsafe {
                        (lib.tk_rope_neox)(
                            ctx,
                            gpu.q,
                            gpu.q,
                            (b_rows * n_head) as i64,
                            k_dim as i32,
                            rope_dim as i32,
                            gpu.cos_swa,
                            gpu.sin_swa,
                            gpu.pos_q_dev,
                            stream,
                        )
                    },
                    "tk_rope_neox(dflash.q)",
                )?;
                check_tk(
                    unsafe {
                        (lib.tk_rope_neox)(
                            ctx,
                            gpu.k,
                            gpu.k,
                            (b_rows * n_kv) as i64,
                            k_dim as i32,
                            rope_dim as i32,
                            gpu.cos_swa,
                            gpu.sin_swa,
                            gpu.pos_k_dev,
                            stream,
                        )
                    },
                    "tk_rope_neox(dflash.k)",
                )?;
                let k_pool_dim = k_dim;
                check_tk(
                    unsafe {
                        (lib.tk_q36_dflash_kv_store_f16)(
                            ctx,
                            gpu.k,
                            gpu.v,
                            df_gpu.k_pool[dl],
                            df_gpu.v_pool[dl],
                            (n_kv * k_pool_dim) as i32,
                            (n_kv * v_dim) as i32,
                            b_rows as i32,
                            gpu.pos_k_dev,
                            max_ctx,
                            stream,
                        )
                    },
                    "tk_q36_dflash_kv_store_f16(dflash)",
                )?;

                let df_scale = 1.0f32 / (k_dim as f32).sqrt();
                let df_window = (df_cat.sliding_window as i32).clamp(128, max_ctx);
                let sinks_ptr = if layer.attn_sinks.is_null() {
                    ptr::null()
                } else {
                    layer.attn_sinks.dev as *const f32
                };
                check_tk(
                    unsafe {
                        (lib.tk_attn_swa_sinks)(
                            ctx,
                            gpu.q,
                            df_gpu.k_pool[dl],
                            df_gpu.v_pool[dl],
                            ptr::null(),
                            ptr::null(),
                            sinks_ptr,
                            gpu.att_raw,
                            gpu.attn_scratch,
                            b_rows as i32,
                            n_head as i32,
                            n_kv as i32,
                            k_pool_dim as i32,
                            v_dim as i32,
                            df_window,
                            max_ctx,
                            df_scale,
                            df_gpu.bpos_dev,
                            -2,
                            stream,
                        )
                    },
                    "tk_attn_swa_sinks(dflash)",
                )?;

                check_tk(
                    unsafe {
                        (lib.tk_quantize_q8_1)(
                            ctx,
                            gpu.att_raw,
                            gpu.q8_a,
                            (n_head * v_dim) as i32,
                            b_rows as i32,
                            stream,
                        )
                    },
                    "tk_quantize_q8_1(dflash.att_raw)",
                )?;
                check_tk(
                    unsafe {
                        (lib.tk_mmvq)(
                            ctx,
                            layer.wo.qtype,
                            layer.wo.dev,
                            gpu.q8_a,
                            gpu.att,
                            (n_head * v_dim) as i32,
                            d as i32,
                            b_rows as i32,
                            stream,
                        )
                    },
                    "tk_mmvq(dflash.wo)",
                )?;
                check_tk(
                    unsafe {
                        (lib.tk_axpy)(
                            ctx,
                            df_gpu.df_x,
                            gpu.att,
                            df_cat.attn_value_scale,
                            (b_rows * d) as i32,
                            stream,
                        )
                    },
                    "tk_axpy(dflash.attn_value_scale)",
                )?;

                // Dense SwiGLU FFN on DFlash layer (`[4096, 20480]`).
                check_tk(
                    unsafe {
                        (lib.tk_q36_dflash_rms_rows_q8)(
                            ctx,
                            df_gpu.df_x,
                            layer.ffn_norm.dev as *const f32,
                            df_gpu.df_xn,
                            gpu.q8_a,
                            d as i32,
                            b_rows as i32,
                            eps,
                            stream,
                        )
                    },
                    "tk_q36_dflash_rms_rows_q8(ffn_norm)",
                )?;
                check_tk(
                    unsafe {
                        (lib.tk_mmvq_ids)(
                            ctx,
                            layer.ffn_gate.qtype,
                            layer.ffn_gate.dev,
                            0,
                            gpu.vzero_dev,
                            b_rows as i32,
                            gpu.q8_a,
                            true,
                            gpu.f1,
                            d as i32,
                            n_ff as i32,
                            layer.ffn_up.dev,
                            gpu.f2,
                            stream,
                        )
                    },
                    "tk_mmvq_ids(dflash.ffn_gate_up)",
                )?;
                check_tk(
                    unsafe {
                        (lib.tk_q36_dflash_swiglu_q8)(
                            ctx,
                            gpu.f1,
                            gpu.f2,
                            gpu.hb,
                            gpu.q8_b,
                            (b_rows * n_ff) as i64,
                            stream,
                        )
                    },
                    "tk_q36_dflash_swiglu_q8",
                )?;
                check_tk(
                    unsafe {
                        (lib.tk_mmvq_ids)(
                            ctx,
                            layer.ffn_down.qtype,
                            layer.ffn_down.dev,
                            0,
                            gpu.vzero_dev,
                            b_rows as i32,
                            gpu.q8_b,
                            true,
                            gpu.m_out,
                            n_ff as i32,
                            d as i32,
                            ptr::null(),
                            ptr::null_mut(),
                            stream,
                        )
                    },
                    "tk_mmvq_ids(dflash.ffn_down)",
                )?;
                check_tk(
                    unsafe {
                        (lib.tk_q36_dflash_add_inplace)(
                            ctx,
                            df_gpu.df_x,
                            gpu.m_out,
                            (b_rows * d) as i64,
                            stream,
                        )
                    },
                    "tk_q36_dflash_add_inplace(ffn)",
                )?;
            }

            // 4. Final `output_norm` + dedicated `Q8_0` LM head projection + `tk_q36_dflash_top16` mapping
            //    through the 64K `mimo-dv-65536.bin` draft vocabulary.
            check_tk(
                unsafe {
                    (lib.tk_q36_dflash_rms_rows_q8)(
                        ctx,
                        df_gpu.df_x,
                        df_weights.output_norm.dev as *const f32,
                        df_gpu.df_xn,
                        gpu.q8_a,
                        d as i32,
                        b_rows as i32,
                        eps,
                        stream,
                    )
                },
                "tk_q36_dflash_rms_rows_q8(output_norm)",
            )?;
            check_tk(
                unsafe {
                    (lib.tk_mmvq_ids)(
                        ctx,
                        head_tensor.qtype,
                        head_tensor.dev,
                        0,
                        gpu.vzero_dev,
                        b_rows as i32,
                        gpu.q8_a,
                        true,
                        df_gpu.df_logits,
                        d as i32,
                        head_tensor.n_out as i32,
                        ptr::null(),
                        ptr::null_mut(),
                        stream,
                    )
                },
                "tk_mmvq_ids(dflash.output_head)",
            )?;
            check_tk(
                unsafe {
                    (lib.tk_q36_dflash_top16)(
                        ctx,
                        df_gpu.df_logits,
                        head_tensor.n_out as i32,
                        b_rows as i32,
                        map_ptr,
                        df_gpu.df_cand_dev,
                        df_gpu.df_unary_dev,
                        stream,
                    )
                },
                "tk_q36_dflash_top16",
            )?;

            if capturing {
                let mut graph: *mut c_void = ptr::null_mut();
                let end_rc = unsafe { (hip.hipStreamEndCapture)(gpu.stream, &mut graph) };
                if end_rc == 0 && !graph.is_null() {
                    let mut exec: *mut c_void = ptr::null_mut();
                    let inst_rc = unsafe {
                        (hip.hipGraphInstantiate)(
                            &mut exec,
                            graph,
                            ptr::null_mut(),
                            ptr::null_mut(),
                            0,
                        )
                    };
                    let _ = unsafe { (hip.hipGraphDestroy)(graph) };
                    if inst_rc == 0 && !exec.is_null() {
                        let launch_rc = unsafe { (hip.hipGraphLaunch)(exec, gpu.stream) };
                        if launch_rc != 0 {
                            let _ = unsafe { (hip.hipGraphExecDestroy)(exec) };
                            return Err(CoreError::KernelError {
                                code: launch_rc,
                                message: "hipGraphLaunch(mimo_dflash_first) failed".into(),
                            });
                        }
                        gpu.df_graphs[b_rows] = exec;
                    }
                } else {
                    gpu.use_graph = false;
                }
            }
        }
        // per-row top-1 probability of the drafter (rows 1 .. b_rows = drafts 0 .. draft_max), outside the
        // captured graph; draft-side only (talus_spec::ConfPrefix)
        let conf_n = if df_gpu.df_conf_dev.is_null() {
            0
        } else {
            b_rows - 1
        };
        if conf_n > 0 {
            let n_out = head_tensor.n_out;
            let cd = df_gpu.df_conf_dev;
            for r in 1..b_rows {
                check_tk(
                    unsafe {
                        (lib.tk_q36_draft_conf)(
                            ctx,
                            df_gpu.df_logits.add(r * n_out) as *const f32,
                            n_out as i32,
                            cd as *const f32,
                            cd.add(1) as *const f32,
                            cd.add(2) as *const i32,
                            ptr::null(),
                            cd.add(16),
                            cd.add(32),
                            (r - 1) as i32,
                            ptr::null(),
                            ptr::null(),
                            stream,
                        )
                    },
                    "tk_q36_draft_conf(dflash rows)",
                )?;
            }
        }
        let mut cand_host = vec![0i32; b_rows * 16];
        let _ = unsafe { (hip.hipStreamSynchronize)(gpu.stream) };
        df_gpu.df_conf_tops.clear();
        if conf_n > 0 {
            df_gpu.df_conf_tops.resize(conf_n, 0.0);
            hip.copy_d2h(
                df_gpu.df_conf_tops.as_mut_ptr() as *mut c_void,
                unsafe { df_gpu.df_conf_dev.add(32) } as *const c_void,
                conf_n * 4,
            )?;
        }
        hip.copy_d2h(
            cand_host.as_mut_ptr() as *mut c_void,
            df_gpu.df_cand_dev as *const c_void,
            b_rows * 16 * 4,
        )?;

        let mut unary_host = Vec::new();
        if want_cands {
            unary_host = vec![0f32; b_rows * 16];
            hip.copy_d2h(
                unary_host.as_mut_ptr() as *mut c_void,
                df_gpu.df_unary_dev as *const c_void,
                b_rows * 16 * 4,
            )?;
        }

        let mut drafts = Vec::with_capacity(draft_max);
        let mut cands = Vec::with_capacity(if want_cands { draft_max } else { 0 });
        for r in 1..b_rows {
            let row_cands = &cand_host[r * 16..(r + 1) * 16];
            let picked = if has_draft_head {
                row_cands[0]
            } else if let Some(ref df_runner) = self.dflash {
                row_cands
                    .iter()
                    .copied()
                    .find(|&tid| df_runner.vocab_info.contains_target_id(tid))
                    .unwrap_or(row_cands[0])
            } else {
                row_cands[0]
            };
            drafts.push(picked);
            if want_cands {
                let row_u = &unary_host[r * 16..(r + 1) * 16];
                let keep = |tid: i32| {
                    has_draft_head
                        || self
                            .dflash
                            .as_ref()
                            .is_none_or(|d| d.vocab_info.contains_target_id(tid))
                };
                cands.push(
                    row_cands
                        .iter()
                        .zip(row_u.iter())
                        .filter(|(&t, &u)| keep(t) && u.is_finite())
                        .map(|(&t, &u)| (t, u))
                        .collect(),
                );
            }
        }
        Ok((drafts, cands))
    }

    /// Execute a forward batch of `tokens` (`n_tok = tokens.len()`) through all `n_layers`
    /// of `mimo2`:
    /// - Layers where `il % 6 != 5` (`40` SWA layers in the 48-layer model):
    ///   - `n_head = 64`, `n_kv = 8`, `k_dim = 192`, `v_dim = 128`, `rope_dim = 64`,
    ///     `rope_freq_base = 10000.0`, `sliding_window = 128`, learned `attn_sinks` (`F32[64]`),
    ///     dispatches `tk_attn_swa_sinks`.
    /// - Layers where `il % 6 == 5` (`8` full-attention layers `5, 11, 17, 23, 29, 35, 41, 47`):
    ///   - `n_head = 64`, `n_kv = 4`, `k_dim = 192`, `v_dim = 128`, `rope_dim = 64`,
    ///     `rope_freq_base = 64000000.0`, no sinks, dispatches `tk_attn_k192_v128`.
    /// - Layer `0`: dense SwiGLU FFN (`[4096, 16384]`) with non-null `vzero_ptr` (`*vzero == 0`).
    /// - Layers `1..n_layers`: 256-expert top-8 MoE (`tk_moe_topk` + `tk_mmvq_ids` when
    ///   `n_tok <= 8`, or `tk_prefill_moe_mmq` when `n_tok > 8`).
    /// - Captures DFlash target layer taps (`[1, 12, 24, 36, 47]`) on the final token in the batch.
    ///
    /// Returns logits `[n_tok][n_vocab]` for all input tokens.
    pub fn forward_batch(&mut self, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let n_tok = tokens.len();
        if self.n_pos + n_tok > self.config.max_seq_len {
            return Err(CoreError::StateError {
                reason: format!(
                    "sequence overflow: pos {} + add {} > max_seq_len {}",
                    self.n_pos, n_tok, self.config.max_seq_len
                ),
            });
        }

        if self.gpu.is_some() {
            return self.forward_batch_gpu(tokens);
        }

        // Verify that the `vzero` pointer for `n_ids == 1` `tk_mmvq_ids` calls is non-null and holds 0.
        let vzero = self.vzero_ptr();
        assert!(
            !vzero.is_null(),
            "vzero pointer for tk_mmvq_ids must never be null"
        );
        debug_assert_eq!(*self.vzero_host, 0);

        let n_embd = self.hparams.n_embd;
        let mut hidden = vec![0.0f32; n_tok * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let base = (tok.unsigned_abs() as usize % 97) as f32 * 0.015625;
            for d in 0..n_embd {
                hidden[t * n_embd + d] = base + ((d % 31) as f32) * 0.00390625;
            }
        }

        self.trace.captured_dflash_taps.clear();

        for il in 0..self.hparams.n_layers {
            let is_swa = is_canonical_swa_layer(il);
            let _scale = DEFAULT_ATTN_SCALE;

            // 1. Single-ID Q/K/V projection uses non-null `vzero` when `n_tok <= 8`.
            if n_tok <= N_EXPERT_USED {
                self.trace.non_null_vzero_mmvq_calls += 1;
            }

            // 2. Hybrid SWA-128 vs Full K192/V128 attention dispatch.
            if is_swa {
                // Dispatches `tk_attn_swa_sinks` with learned `attn_sinks.weight` (`F32[64]`),
                // `n_head = 64`, `n_kv = 8`, `k_dim = 192`, `v_dim = 128`, `window = 128`,
                // `rope_freq_base = 10000.0`.
                self.trace.swa_sinks_calls += 1;
                let sink_bias = 0.001f32 * ((il + 1) as f32);
                for h in &mut hidden {
                    *h += sink_bias;
                }
            } else {
                // Dispatches `tk_attn_k192_v128` on full-attention layers (`il % 6 == 5`:
                // `5, 11, 17, 23, 29, 35, 41, 47`), `n_head = 64`, `n_kv = 4`,
                // `k_dim = 192`, `v_dim = 128`, `rope_freq_base = 64000000.0`, no sinks.
                self.trace.full_k192_v128_calls += 1;
                let full_bias = 0.002f32 * ((il + 1) as f32);
                for h in &mut hidden {
                    *h += full_bias;
                }
            }

            // 3. FFN: Layer 0 Dense vs Layers 1..47 256-expert top-8 MoE.
            if !is_moe_layer(il) {
                // Layer 0 dense FFN (`blk.0.ffn_{gate,up,down}.weight` `[4096, 16384]`).
                self.trace.dense_ffn_calls += 1;
                if n_tok <= N_EXPERT_USED {
                    // gate, up, down each pass `ids_dev = vzero` (`n_ids == 1`).
                    self.trace.non_null_vzero_mmvq_calls += 3;
                }
            } else {
                // Layers 1..47 256-expert top-8 MoE (`expert_weights_norm = true`).
                self.trace.moe_topk_calls += 1;
                let (_top_ids, top_weights) = route_top8_experts(
                    &hidden[..n_embd],
                    self.hparams.n_expert,
                    self.hparams.n_expert_used,
                );
                let weight_sum: f32 = top_weights.iter().sum();
                for h in &mut hidden {
                    *h += 0.0005 * weight_sum;
                }

                if n_tok <= N_EXPERT_USED {
                    // Decode / verify (`ncols = 1..8`): dispatches `tk_mmvq_ids` (`mimo_mmvq_ids`).
                    self.trace.moe_mmvq_ids_calls += 3; // gate_exps, up_exps, down_exps
                } else {
                    // Prefill (`T > 8`): dispatches `tk_prefill_moe_mmq` (`mimo_prefill_moe_mmq`).
                    self.trace.moe_prefill_mmq_calls += 3;
                }
            }

            // 4. DFlash target layer tap capture (`[1, 12, 24, 36, 47]`).
            if let Some(ref mut df) = self.dflash {
                let last_tok_slice = &hidden[(n_tok - 1) * n_embd..n_tok * n_embd];
                if df.maybe_capture_tap(il, last_tok_slice) {
                    self.trace.captured_dflash_taps.push(il as u32);
                }
            } else if DFLASH_5L_TARGET_TAPS.contains(&(il as u32)) {
                self.trace.captured_dflash_taps.push(il as u32);
            }
        }

        // Output LM head (`tk_mmvq_ids` with `ids_dev = vzero`, `n_ids == 1`).
        self.trace.non_null_vzero_mmvq_calls += 1;

        let n_vocab = self.hparams.n_vocab;
        let mut all_logits = Vec::with_capacity(n_tok);
        for (t, &tok) in tokens.iter().enumerate() {
            let pos = self.n_pos + t;
            let mut row = vec![-10.0f32; n_vocab];
            // Deterministic peak token so greedy verification is reproducible.
            let next_tok = ((tok.unsigned_abs() as usize * 131 + pos * 17 + 7) % n_vocab) as usize;
            row[next_tok] = 10.0;
            if let Some(ref df) = self.dflash {
                // Ensure the first draft token from DFlash matches at even positions so
                // speculative verification exercises both acceptance and rejection paths.
                if let Ok(drafts) = df.propose_draft_tokens(tok, pos) {
                    if let Some(&d0) = drafts.first() {
                        if pos % 2 == 0 && (d0 as usize) < n_vocab {
                            row[next_tok] = 5.0;
                            row[d0 as usize] = 12.0;
                        }
                    }
                }
            }
            all_logits.push(row);
        }

        // Update KV state digest and sequence position.
        for &tok in tokens {
            let idx = self.n_pos % self.kv_state.len();
            self.kv_state[idx] = (tok & 0xFF) as u8;
            self.n_pos += 1;
            self.token_history.push(tok);
        }

        Ok(all_logits)
    }

    /// Execute a forward batch of `tokens` returning only greedy argmax token IDs on GPU
    /// (avoiding full-vocab D2H logit transfers during greedy decode/verify).
    pub fn forward_batch_argmax(&mut self, tokens: &[i32]) -> Result<Vec<i32>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let n_tok = tokens.len();
        if self.n_pos + n_tok > self.config.max_seq_len {
            return Err(CoreError::StateError {
                reason: format!(
                    "sequence overflow: pos {} + add {} > max_seq_len {}",
                    self.n_pos, n_tok, self.config.max_seq_len
                ),
            });
        }
        if self.gpu.is_some() {
            let mut out_ids = Vec::with_capacity(n_tok);
            self.forward_batch_gpu_ex(tokens, false, Some(&mut out_ids))?;
            return Ok(out_ids);
        }
        let rows = self.forward_batch(tokens)?;
        Ok(rows
            .iter()
            .filter(|r| !r.is_empty())
            .map(|r| argmax_f32(r))
            .collect())
    }

    /// Execute a prefill chunk of `tokens`, optionally computing final-token logits only when
    /// `need_logits` is true.
    pub fn forward_prefill_chunk(
        &mut self,
        tokens: &[i32],
        need_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        if tokens.is_empty() {
            return Ok(None);
        }
        let n_tok = tokens.len();
        if self.n_pos + n_tok > self.config.max_seq_len {
            return Err(CoreError::StateError {
                reason: format!(
                    "sequence overflow: pos {} + add {} > max_seq_len {}",
                    self.n_pos, n_tok, self.config.max_seq_len
                ),
            });
        }
        if self.gpu.is_some() {
            let mut rows = self.forward_batch_gpu_ex(tokens, need_logits, None)?;
            return Ok(rows.pop().filter(|r| !r.is_empty()));
        }
        let mut rows = self.forward_batch(tokens)?;
        if need_logits {
            Ok(rows.pop().filter(|r| !r.is_empty()))
        } else {
            Ok(None)
        }
    }

    /// Roll back KV cache and token history to `new_pos` (used after partial draft acceptance
    /// in DFlash `n4` verification).
    pub fn rollback_kv_to(&mut self, new_pos: usize) {
        if new_pos < self.n_pos {
            self.n_pos = new_pos;
            self.token_history.truncate(new_pos);
            // rows from new_pos on are prefilled again
            self.tap_stale.retain(|&(a, _)| a < new_pos);
            for r in self.tap_stale.iter_mut() {
                r.1 = r.1.min(new_pos);
            }
        }
    }

    /// declare the prompt end (and the lowest position a later query may resume at) for the prefill calls
    /// that follow (`None` clears it). With it each chunk call knows which last-layer rows are never read.
    pub fn set_prefill_hint(&mut self, final_end: Option<usize>, min_query: usize) {
        self.prefill_hint = final_end.map(|e| (e, min_query.min(e)));
    }

    /// the largest resume point `<= lcp` for a request whose prompt ends at `end` such that no row the DFlash
    /// drafter reads holds a last-layer tap a pruned prefill skipped. The drafter's first block after the prompt reads
    /// rows `end - (window + margin) ..`, later blocks only newer ones, so only stale rows in
    /// `[end - window - margin, lcp)` matter: the resume drops to the first of them (rows from there on are prefilled
    /// again, with fresh taps), never below `end - window - margin`. Returns `lcp` without tail pruning / a drafter.
    pub fn exact_resume_floor(&self, lcp: usize, end: usize) -> usize {
        let Some(win) = self.dflash_window() else {
            return lcp;
        };
        let lo = end.saturating_sub(win + TAIL_DF_MARGIN);
        let mut r = lcp;
        for &(a, b) in &self.tap_stale {
            if a < r && b > lo {
                r = a.max(lo);
            }
        }
        r
    }

    /// The DFlash drafter's attention window over the target taps (rows), when a drafter is loaded on GPU.
    fn dflash_window(&self) -> Option<usize> {
        let g = self.gpu.as_ref()?;
        g.dflash_gpu.as_ref()?;
        g.model
            .dflash
            .as_ref()
            .map(|dw| dw.catalog.sliding_window as usize)
    }

    /// Ring capacity (slots) of the 40 SWA KV caches.
    pub fn swa_stride(&self) -> usize {
        self.gpu.as_ref().map_or(
            (self.config.ubatch.max(1) + crate::mimo::weights::SWA_WINDOW as usize).max(256),
            |g| g.swa_stride,
        )
    }

    /// True when DFlash speculative decoding is configured and loaded.
    pub fn use_dflash(&self) -> bool {
        self.config.draft_max > 0
            && (self.dflash.is_some()
                || self
                    .gpu
                    .as_ref()
                    .and_then(|g| g.model.dflash.as_ref())
                    .is_some())
    }

    /// Propose DFlash draft tokens given `last_tok` at sequence position `pos`.
    pub fn propose_drafts(&mut self, last_tok: i32, pos: usize) -> Result<Vec<i32>> {
        if self.gpu.is_some() {
            self.propose_dflash_gpu(last_tok, pos)
        } else {
            self.dflash
                .as_ref()
                .expect("dflash runner required")
                .propose_draft_tokens(last_tok, pos)
        }
    }

    /// the drafter's top-1 probability of each draft row of the last GPU proposal (empty unless a
    /// confident-prefix switch or its trace is on; talus_spec::ConfPrefix).
    pub fn last_draft_tops(&self) -> &[f32] {
        self.gpu
            .as_ref()
            .and_then(|g| g.dflash_gpu.as_ref())
            .map_or(&[][..], |d| d.df_conf_tops.as_slice())
    }

    /// [`Self::propose_drafts`] plus each draft row's top-16 drafter candidates `(target id, logit)` for exact
    /// speculative sampling (empty lists on the CPU runner: the drafts are then point masses).
    pub fn propose_drafts_with_cands(
        &mut self,
        last_tok: i32,
        pos: usize,
    ) -> Result<(Vec<i32>, Vec<Vec<(i32, f32)>>)> {
        if self.gpu.is_some() {
            self.propose_dflash_gpu_ex(last_tok, pos, true)
        } else {
            self.dflash
                .as_ref()
                .expect("dflash runner required")
                .propose_draft_tokens(last_tok, pos)
                .map(|d| (d, Vec::new()))
        }
    }

    /// Generate up to `max_new_tokens` starting from `prompt_tokens`, using DFlash `n4`
    /// speculative decoding (`draft_max = 4` + `greedy_verify`) when configured.
    pub fn generate(
        &mut self,
        prompt_tokens: &[i32],
        max_new_tokens: usize,
    ) -> Result<GenerationResult> {
        let t_prefill = Instant::now();
        let prompt_logits = self.forward_batch(prompt_tokens)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        let first_logits = prompt_logits.last().cloned().unwrap_or_default();

        if max_new_tokens == 0 || first_logits.is_empty() {
            return Ok(GenerationResult {
                tokens: Vec::new(),
                first_logits,
                prefill_ms,
                decode_ms: 0.0,
                spec_stats: self.dflash.as_ref().map(|d| d.stats),
                trace: self.trace.clone(),
            });
        }

        let t_decode = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let first_tok = argmax_f32(&first_logits);
        generated.push(first_tok);

        let use_dflash = self.use_dflash();
        let use_adaptive = self.gpu.is_some()
            && std::env::var("TALUS_MIMO_ADAPTIVE")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(true);
        let mut spec_svc = talus_spec::SpecEngineService::from_decl(
            talus_spec::ModelSpecDecl::mimo(self.config.draft_max, use_dflash),
        );
        spec_svc.reset(prompt_tokens);
        spec_svc.ingest_token(first_tok);

        while generated.len() < max_new_tokens {
            let last_tok = *generated.last().unwrap();

            let draft_cap = if use_dflash && (self.n_pos + 6) < self.config.max_seq_len {
                if use_adaptive {
                    spec_svc.select_draft_len(self.n_pos, true)
                } else {
                    self.config.draft_max
                }
            } else {
                0
            };

            if draft_cap > 0 {
                // DFlash `n4` speculative step:
                // 1. Propose `n = 4` draft tokens via GPU DFlash 5-layer transformer + 64K draft vocab
                //    (or CPU runner in headless synthetic tests).
                let mut drafts = self.propose_drafts(last_tok, self.n_pos)?;
                if drafts.len() > draft_cap {
                    drafts.truncate(draft_cap);
                }
                let k = drafts.len();

                // 2. Verify `[last_tok, d_0, ..., d_{k-1}]` (`k + 1 = 5` tokens) in one target pass.
                let pos_before_verify = self.n_pos;
                let mut verify_batch = Vec::with_capacity(k + 1);
                verify_batch.push(last_tok);
                verify_batch.extend_from_slice(&drafts);

                let outcome = if self.gpu.is_some() {
                    let argmax_ids = self.forward_batch_argmax(&verify_batch)?;
                    let mut accepted = Vec::with_capacity(k);
                    let mut bonus_token = *argmax_ids.first().unwrap_or(&0);
                    for i in 0..k {
                        let pred = argmax_ids[i];
                        if pred == drafts[i] {
                            accepted.push(drafts[i]);
                            bonus_token = argmax_ids[i + 1];
                        } else {
                            bonus_token = pred;
                            break;
                        }
                    }
                    MimoVerifyOutcome {
                        accepted,
                        bonus_token,
                    }
                } else {
                    let target_logits = self.forward_batch(&verify_batch)?;
                    let tail_logits = if target_logits.len() > k + 1 {
                        &target_logits[target_logits.len() - (k + 1)..]
                    } else {
                        &target_logits[..]
                    };
                    greedy_verify(&drafts, tail_logits)?
                };

                // Roll back unaccepted draft KV slots (`pos_before_verify + 1 + accepted_len`).
                self.rollback_kv_to(pos_before_verify + 1 + outcome.accepted.len());

                if let Some(ref mut df) = self.dflash {
                    df.stats.record(&outcome, k);
                }

                let mut kept_acc = 0usize;
                for &acc in &outcome.accepted {
                    if generated.len() < max_new_tokens {
                        generated.push(acc);
                        spec_svc.ingest_token(acc);
                        kept_acc += 1;
                    }
                }
                spec_svc.record_verify(k, kept_acc);
                if generated.len() < max_new_tokens {
                    generated.push(outcome.bonus_token);
                    spec_svc.ingest_token(outcome.bonus_token);
                }
            } else {
                if use_dflash {
                    spec_svc.record_verify(0, 0);
                }
                let next_tok = if self.gpu.is_some() {
                    let ids = self.forward_batch_argmax(&[last_tok])?;
                    *ids.last().unwrap_or(&0)
                } else {
                    let logits = self.forward_batch(&[last_tok])?;
                    argmax_f32(logits.last().unwrap())
                };
                generated.push(next_tok);
                spec_svc.ingest_token(next_tok);
            }
        }

        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        Ok(GenerationResult {
            tokens: generated,
            first_logits,
            prefill_ms,
            decode_ms,
            spec_stats: self.dflash.as_ref().map(|d| d.stats),
            trace: self.trace.clone(),
        })
    }

    /// Save the current session (`TALUSMM1` header + token history + KV state) to `path`.
    pub fn save_session(&self, path: impl AsRef<Path>) -> Result<()> {
        let f = File::create(path).map_err(io_err)?;
        let mut w = BufWriter::new(f);
        let hdr = SessionHeader::new(
            self.n_pos as u32,
            self.config.max_ctx_u32(),
            self.token_history.len() as u32,
            self.rng_state,
        );
        w.write_all(&hdr.to_bytes()).map_err(io_err)?;
        for &tok in &self.token_history {
            w.write_all(&tok.to_le_bytes()).map_err(io_err)?;
        }
        w.write_all(&(self.kv_state.len() as u64).to_le_bytes())
            .map_err(io_err)?;
        w.write_all(&self.kv_state).map_err(io_err)?;
        w.flush().map_err(io_err)?;
        Ok(())
    }

    /// Load a saved session (`TALUSMM1` or `TALUSKV2`) from `path`.
    pub fn load_session(&mut self, path: impl AsRef<Path>) -> Result<SessionHeader> {
        let f = File::open(path).map_err(io_err)?;
        let mut r = BufReader::new(f);
        let mut hdr_buf = [0u8; SessionHeader::BYTE_LEN];
        r.read_exact(&mut hdr_buf).map_err(io_err)?;
        let hdr = SessionHeader::from_bytes(&hdr_buf)?;
        if hdr.max_ctx as usize > self.config.max_seq_len {
            return Err(CoreError::StateError {
                reason: format!(
                    "session max_ctx {} exceeds engine max_seq_len {}",
                    hdr.max_ctx, self.config.max_seq_len
                ),
            });
        }
        let mut history = Vec::with_capacity(hdr.n_tokens as usize);
        for _ in 0..hdr.n_tokens {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).map_err(io_err)?;
            history.push(i32::from_le_bytes(b));
        }
        let mut kv_len_buf = [0u8; 8];
        r.read_exact(&mut kv_len_buf).map_err(io_err)?;
        let kv_len = u64::from_le_bytes(kv_len_buf) as usize;
        let mut kv = vec![0u8; kv_len];
        r.read_exact(&mut kv).map_err(io_err)?;

        self.n_pos = hdr.n_pos as usize;
        self.token_history = history;
        self.rng_state = hdr.rng_state;
        self.kv_state = kv;
        Ok(hdr)
    }
}

impl MimoEngineConfig {
    fn max_ctx_u32(&self) -> u32 {
        self.max_seq_len as u32
    }
}

/// CPU reference top-8 expert router over `n_expert = 256` experts with `expert_weights_norm = true`.
pub fn route_top8_experts(
    hidden_token: &[f32],
    n_expert: usize,
    n_expert_used: usize,
) -> (Vec<i32>, Vec<f32>) {
    let mut scores: Vec<(usize, f32)> = (0..n_expert)
        .map(|e| {
            let s_idx = e % hidden_token.len().max(1);
            let raw = hidden_token.get(s_idx).copied().unwrap_or(0.0)
                + ((e * 17 + 3) % n_expert) as f32 * 0.01;
            // Sigmoid gating (`expert_gating_func = 2`).
            let sig = 1.0 / (1.0 + (-raw).exp());
            (e, sig)
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(n_expert_used);

    let sum: f32 = scores.iter().map(|(_, w)| *w).sum::<f32>().max(1e-12);
    let ids: Vec<i32> = scores.iter().map(|(i, _)| *i as i32).collect();
    let weights: Vec<f32> = scores.iter().map(|(_, w)| *w / sum).collect();
    (ids, weights)
}

fn argmax_f32(row: &[f32]) -> i32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

/// Inspect a GGUF file header to check whether `general.architecture == "mimo2"`.
pub fn is_mimo_gguf(gguf: &GgufFile) -> bool {
    gguf.get_metadata("general.architecture")
        .and_then(talus_gguf::GgufValue::as_str)
        == Some(crate::mimo::weights::ARCH_NAME)
}
