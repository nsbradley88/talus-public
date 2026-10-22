//! Speculation traits and round execution contracts.
//!
//! Re-exports [`HostDrafter`], [`DeviceDrafter`], [`SpecController`], [`RoundIo`],
//! [`RoundStats`], [`RoundPlan`], and [`Accept`] from `talus-core-api`, and defines
//! [`RoundExecutor`] and [`RoundOutcome`] (`SpecRoundOutcome`) for driving speculative
//! draft-and-verify rounds end-to-end over CPU mock or device backends.

use crate::dflash::DFlashDrafterSpec;
use crate::sampling::{decide_greedy, decide_sampled, Mt19937_64, SampCfg};
use talus_core_api::DeviceId;
pub use talus_core_api::{
    Accept, CoreError, DeviceDrafter, DraftVocab as CoreDraftVocab, GraphBuilder, GraphKey,
    HiddenTaps, HostDrafter, Result, RoundIo, RoundPlan, RoundStats, SeqState, SpecController,
    StreamHandle,
};

/// Result of executing and verifying a single speculative round (`RoundOutcome` / `SpecRoundOutcome`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundOutcome {
    /// Number of draft tokens proposed in this round (`0..=max_draft`).
    pub proposed: usize,
    /// Number of draft tokens accepted (`0..=proposed`).
    pub accepted: usize,
    /// Bonus or resampled token emitted after the last accepted draft token.
    pub bonus_token: i32,
    /// All committed tokens in order (`accepted` draft tokens followed by `bonus_token`).
    pub committed_tokens: Vec<i32>,
    /// True if this round's draft tokens came from a host drafter (`ngram-mod` / `prompt-lookup`).
    pub used_host_drafter: bool,
    /// True if the device drafter was paused by the [`SpecController`] during this round.
    pub device_paused: bool,
    /// The [`RoundPlan`] executed in this round.
    pub plan: RoundPlan,
}

/// Alias for [`RoundOutcome`] (`SpecRoundOutcome`).
pub type SpecRoundOutcome = RoundOutcome;

/// End-to-end speculative round executor (`RoundExecutor`).
///
/// Coordinates [`DeviceDrafter`], [`HostDrafter`], and [`SpecController`] across speculative
/// verification rounds (supporting both `Accept::Greedy` and `Accept::Rejection`).
#[derive(Debug, Clone)]
pub struct RoundExecutor {
    /// Target vocabulary size.
    pub n_vocab: usize,
    /// Maximum draft tokens per round.
    pub max_draft: usize,
    /// Sampling parameters used during rejection verification (`Accept::Rejection`).
    pub samp_cfg: SampCfg,
    /// Mersenne Twister RNG for rejection sampling.
    pub rng: Mt19937_64,
    /// Committed token history (`u32`).
    pub history: Vec<u32>,
    /// Rolling token history (`i32`) passed to [`decide_sampled`].
    pub recent_i32: Vec<i32>,
    /// Sequence state tracked across rounds.
    pub seq_state: SeqState,
    /// Telemetry from the most recently completed speculative round.
    pub last_stats: RoundStats,
    /// Total speculative rounds executed.
    pub rounds_executed: u64,
}

impl Default for RoundExecutor {
    fn default() -> Self {
        Self::new(100_352, 4)
    }
}

impl RoundExecutor {
    /// Creates a new `RoundExecutor` for `n_vocab` and `max_draft`.
    pub fn new(n_vocab: usize, max_draft: usize) -> Self {
        Self {
            n_vocab: n_vocab.max(1),
            max_draft,
            samp_cfg: SampCfg {
                temp: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                ..SampCfg::default()
            },
            rng: Mt19937_64::default(),
            history: Vec::new(),
            recent_i32: Vec::new(),
            seq_state: SeqState::new(1, 32768, DeviceId(0)),
            last_stats: RoundStats {
                proposed: 0,
                accepted: 0,
                round_latency_us: 0,
                ema_accept_rate: 0.9,
                host_draft_available: false,
                consecutive_zero_accepts: 0,
            },
            rounds_executed: 0,
        }
    }

    /// Configures sampling parameters (`SampCfg`) for `Accept::Rejection` verification.
    pub fn with_sampling(mut self, cfg: SampCfg) -> Self {
        self.samp_cfg = cfg;
        self
    }

    /// Seeds the RNG used by `Accept::Rejection` rounds.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng.seed(seed);
        self
    }

    /// Initializes the executor's prompt token history (`prompt_tokens`).
    pub fn seed_prompt(&mut self, prompt_tokens: &[u32]) {
        self.history.clear();
        self.history.extend_from_slice(prompt_tokens);
        self.recent_i32 = prompt_tokens.iter().map(|&t| t as i32).collect();
        self.seq_state.pos = prompt_tokens.len() as u32;
    }

    /// Verifies `drafts` against precomputed `target_logits` of shape `(drafts.len() + 1) * n_vocab`
    /// under `accept` policy, commits the accepted prefix + bonus token into history, and returns [`RoundOutcome`].
    pub fn verify_logits(
        &mut self,
        drafts: &[i32],
        target_logits: &[f32],
        accept: Accept,
        q_draft: Option<&dyn Fn(usize, i32) -> f64>,
    ) -> Result<RoundOutcome> {
        let nd = drafts.len();
        let required = (nd + 1) * self.n_vocab;
        if target_logits.len() < required {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "RoundExecutor::verify_logits expected at least {required} logits for nd={nd}, n_vocab={}, got {}",
                    self.n_vocab,
                    target_logits.len()
                ),
            });
        }

        let mut accepted = 0usize;
        let bonus_token = match accept {
            Accept::Greedy => {
                let b = decide_greedy(target_logits, self.n_vocab, drafts, nd, &mut accepted);
                for &d in &drafts[..accepted] {
                    self.recent_i32.push(d);
                }
                self.recent_i32.push(b);
                b
            }
            Accept::Rejection { seed } => {
                if self.rounds_executed == 0 && seed != 0 {
                    self.rng.seed(seed);
                }
                decide_sampled(
                    target_logits,
                    self.n_vocab as i32,
                    drafts,
                    nd,
                    &self.samp_cfg,
                    &mut self.recent_i32,
                    &mut self.rng,
                    &mut accepted,
                    q_draft,
                )
            }
        };

        let mut committed_tokens = Vec::with_capacity(accepted + 1);
        committed_tokens.extend_from_slice(&drafts[..accepted]);
        committed_tokens.push(bonus_token);

        for &tok in &committed_tokens {
            self.history.push(tok as u32);
        }
        self.seq_state.pos = self.history.len() as u32;
        self.rounds_executed += 1;

        let rate = if nd > 0 {
            (accepted as f32) / (nd as f32)
        } else {
            1.0
        };
        self.last_stats.proposed = nd as u32;
        self.last_stats.accepted = accepted as u32;
        self.last_stats.ema_accept_rate = 0.8 * self.last_stats.ema_accept_rate + 0.2 * rate;
        if nd > 0 && accepted == 0 {
            self.last_stats.consecutive_zero_accepts += 1;
        } else if accepted > 0 {
            self.last_stats.consecutive_zero_accepts = 0;
        }

        let plan = RoundPlan {
            draft_depth: nd as u32,
            pause_device_drafter: false,
            prefer_host_draft: false,
            window_len: (nd + 1) as u32,
        };

        Ok(RoundOutcome {
            proposed: nd,
            accepted,
            bonus_token,
            committed_tokens,
            used_host_drafter: false,
            device_paused: false,
            plan,
        })
    }

    /// Executes a complete speculative round driving [`DFlashDrafterSpec`] (implementing both
    /// [`DeviceDrafter`] and [`SpecController`]) + [`NgramEditProfile`] + [`AdaptivePausePolicy`]
    /// against a CPU mock target model `target_logits_fn`.
    ///
    /// `target_logits_fn` receives `(committed_history, draft_tokens)` and returns
    /// `(draft_tokens.len() + 1) * n_vocab` target logits for verification.
    pub fn run_dflash_round<F>(
        &mut self,
        drafter: &mut DFlashDrafterSpec,
        accept: Accept,
        mut target_logits_fn: F,
    ) -> Result<RoundOutcome>
    where
        F: FnMut(&[u32], &[u32]) -> Vec<f32>,
    {
        if drafter.swa_tokens.is_empty() && !self.history.is_empty() {
            drafter.push_context(&self.history);
        }

        // Probe host drafter availability prior to planning.
        let host_probe = drafter.propose_host_fallback(self.max_draft);
        self.last_stats.host_draft_available = !host_probe.is_empty();

        // Consult SpecController (`NgramEditProfile` + `AdaptivePausePolicy`).
        let plan = SpecController::plan(drafter, &self.last_stats);
        let max_k = (plan.draft_depth as usize).min(self.max_draft);

        let mut used_host = false;
        let drafted_u32: Vec<u32> = if max_k == 0 {
            Vec::new()
        } else if plan.prefer_host_draft && !host_probe.is_empty() {
            used_host = true;
            host_probe.into_iter().take(max_k).collect()
        } else if !plan.pause_device_drafter {
            let stream = StreamHandle {
                raw: 0,
                device: self.seq_state.device,
            };
            let taps = drafter.profile.expected_hidden_taps();
            DeviceDrafter::seed(drafter, &self.seq_state, &taps, &stream)?;

            let mut chain_buf = vec![0u32; max_k + 1];
            chain_buf[0] = self.history.last().copied().unwrap_or(0);
            let io = RoundIo {
                draft_depth: max_k,
                device_draft_tokens_ptr: chain_buf.as_mut_ptr() as u64,
                device_accepted_ptr: 0,
                device_bonus_token_ptr: 0,
                device_pos_ptr: 0,
                host_draft_tokens: Vec::new(),
            };
            let mut gb = GraphBuilder::new(
                GraphKey {
                    kind: drafter.profile.name.to_string(),
                    window_len: (max_k + 1) as u32,
                    draft_depth: max_k as u32,
                    device: self.seq_state.device,
                },
                stream,
            );
            DeviceDrafter::capture_chain(drafter, max_k, &io, &mut gb)?;
            chain_buf[1..].to_vec()
        } else if !host_probe.is_empty() && drafter.edit_profile.enable_on_device_pause {
            used_host = true;
            host_probe.into_iter().take(max_k).collect()
        } else {
            Vec::new()
        };

        let target_logits = target_logits_fn(&self.history, &drafted_u32);
        let drafted_i32: Vec<i32> = drafted_u32.iter().map(|&t| t as i32).collect();

        // Preserve single-stream RNG across rounds when Accept::Rejection is used.
        let effective_accept = match accept {
            Accept::Greedy => Accept::Greedy,
            Accept::Rejection { seed } => {
                let s = if self.rounds_executed == 0 { seed } else { 0 };
                Accept::Rejection { seed: s }
            }
        };

        let mut outcome =
            self.verify_logits(&drafted_i32, &target_logits, effective_accept, None)?;
        outcome.used_host_drafter = used_host;
        outcome.device_paused = plan.pause_device_drafter;
        outcome.plan = plan;

        let committed_u32: Vec<u32> = outcome.committed_tokens.iter().map(|&t| t as u32).collect();
        drafter.push_context(&committed_u32);

        Ok(outcome)
    }
}
