//! GLM-5.3-Flash stdio engine protocol backend (`talus-cli --model GLM serve --engine-protocol`), driven by
//! `serve/server.py --engine strata` or `talus-server`.
//!
//! Prompt cache (talus-livecold exact policy): the recurrent state (KDA + conv) is snapshotted only on the prompt
//! chunk grid (multiples of `GRID` tokens, written by prefill) and never restored from decode-written state; the
//! position-indexed caches (MLA latent, indexer keys / pooled keys, MTP block) are reused up to the snapshot.

use std::io::Write;
use std::time::Instant;

use talus_model::glm::cli::{self as gcli, GlmCli};
use talus_model::glm::engine::{Buf, GlmEngine};
use talus_model::glm::sample::SamplerParams;
use talus_model::glm::spec::{self, GenParams, GenStats, SeqPos, SnapEvent};
use talus_sched::{
    write_done_line, write_pp_line, write_resume_line, write_token_line, CancelToken, DoneStats,
    EngineProtocolBackend, GenCommand,
};

/// Snapshot grid (tokens): a multiple of the prefill window and the k-pool size.
pub const GRID: usize = 512;

struct Snap {
    pos: usize,
    hash: u64,
    h_prev: Vec<f32>,
    buf: Buf,
    used: u64,
}

pub struct GlmBackend {
    eng: GlmEngine,
    cfg: GlmCli,
    /// tokens whose state is committed in the position-indexed caches (positions 0..hist.len())
    hist: Vec<u32>,
    snaps: Vec<Snap>,
    max_snaps: usize,
    clock: u64,
    log: bool,
}

fn hash_prefix(t: &[u32]) -> u64 {
    // FNV-1a over the token ids
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in t {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

impl GlmBackend {
    pub fn new(eng: GlmEngine, cfg: GlmCli) -> Self {
        let max_snaps = std::env::var("GLM_PROMPT_SNAPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        Self {
            eng,
            cfg,
            hist: vec![],
            snaps: vec![],
            max_snaps,
            clock: 0,
            log: std::env::var("GLM_SERVE_LOG").is_ok(),
        }
    }

    fn run(
        &mut self,
        cmd: &GenCommand,
        stop: &CancelToken,
        out: &mut dyn Write,
    ) -> talus_core_api::Result<()> {
        let t0 = Instant::now();
        let prompt: Vec<u32> = cmd.prompt_ids.iter().map(|&x| x as u32).collect();
        if prompt.is_empty() {
            return Err(talus_core_api::CoreError::InvalidArgument {
                reason: "empty prompt".into(),
            });
        }
        // common prefix with the committed history (at least one prompt token is evaluated for its logits)
        let mut l = 0;
        if cmd.use_cache {
            while l < prompt.len() && l < self.hist.len() && prompt[l] == self.hist[l] {
                l += 1;
            }
        }
        let l = l.min(prompt.len() - 1);
        // the newest snapshot at or below l whose prefix matches
        let mut best: Option<usize> = None;
        for (i, s) in self.snaps.iter().enumerate() {
            if s.pos <= l
                && s.pos > best.map(|b| self.snaps[b].pos).unwrap_or(0)
                && s.hash == hash_prefix(&prompt[..s.pos])
            {
                best = Some(i);
            }
        }
        let mut seq = SeqPos::default();
        match best {
            Some(i) => {
                let buf = self.snaps[i].buf;
                self.eng.snap_restore(&buf)?;
                self.clock += 1;
                self.snaps[i].used = self.clock;
                seq.n_past = self.snaps[i].pos;
                seq.h_prev = self.snaps[i].h_prev.clone();
            }
            None => {
                self.eng.reset()?;
            }
        }
        let reuse = seq.n_past;
        write_resume_line(out, reuse).ok();
        // grid snapshots of this prompt: the last grid point before its end (and the one before it with 2+ slots)
        let end = prompt.len();
        let last_grid = (end.saturating_sub(1) / GRID) * GRID;
        let points: Vec<usize> = [last_grid, last_grid.saturating_sub(GRID)]
            .into_iter()
            .take(self.max_snaps.max(1))
            .filter(|&g| g > reuse && g > 0)
            .collect();
        let mtp = self.cfg.mtp;
        let first = {
            // the snapshot copy must happen at the point itself (the state moves on with the next window)
            let snaps = &mut self.snaps;
            let max_snaps = self.max_snaps;
            let clock = &mut self.clock;
            let prompt_ref = &prompt;
            let mut pending: Option<usize> = None; // slot being filled
            let mut cb2 = |eng: &mut GlmEngine,
                           ev: SnapEvent|
             -> talus_core_api::Result<Option<Buf>> {
                if max_snaps == 0 {
                    return Ok(None);
                }
                match ev {
                    SnapEvent::Want(pos) => {
                        let hash = hash_prefix(&prompt_ref[..pos]);
                        *clock += 1;
                        if let Some(s) = snaps.iter_mut().find(|s| s.pos == pos && s.hash == hash) {
                            s.used = *clock;
                            return Ok(None); // already held
                        }
                        let idx = if snaps.len() < max_snaps {
                            let buf = eng.snap_alloc()?;
                            snaps.push(Snap {
                                pos: 0,
                                hash: 0,
                                h_prev: vec![],
                                buf,
                                used: 0,
                            });
                            snaps.len() - 1
                        } else {
                            (0..snaps.len()).min_by_key(|&i| snaps[i].used).unwrap()
                        };
                        // invalid until Done (an error in between leaves it unusable)
                        snaps[idx].pos = usize::MAX;
                        pending = Some(idx);
                        Ok(Some(snaps[idx].buf))
                    }
                    SnapEvent::Done(pos, h) => {
                        if let Some(idx) = pending.take() {
                            let s = &mut snaps[idx];
                            s.pos = pos;
                            s.hash = hash_prefix(&prompt_ref[..pos]);
                            s.h_prev = h.to_vec();
                            s.used = *clock;
                        }
                        Ok(None)
                    }
                }
            };
            spec::prefill_with(
                &mut self.eng,
                &mut seq,
                &prompt[reuse..],
                mtp,
                &points,
                &mut cb2,
            )?
        };
        let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
        let fresh = (prompt.len() - reuse) as i64;
        write_pp_line(out, prompt.len() as i64, fresh, prompt_ms).ok();

        let samp = SamplerParams {
            temp: cmd.temperature,
            top_k: cmd.top_k,
            top_p: cmd.top_p,
            min_p: cmd.min_p,
            seed: cmd.seed,
        };
        let p = GenParams {
            max_new: cmd.max_new.max(1) as usize,
            samp,
            mtp,
            eos: self.eng.w.hp.eos.clone(),
            ignore_eos: cmd.ignore_eos,
            ctx: prompt.clone(),
        };
        let mut st = GenStats::default();
        let mut cancelled = false;
        let n0 = seq.n_past;
        let gen = spec::generate(&mut self.eng, &mut seq, &first, &p, &mut st, |t| {
            write_token_line(out, t as i32).ok();
            if stop.is_cancelled() {
                cancelled = true;
                return false;
            }
            true
        })?;
        // committed history: the prompt + the generated tokens the trunk has processed
        let processed = seq.n_past - n0;
        self.hist = prompt.clone();
        self.hist
            .extend_from_slice(&gen[..processed.min(gen.len())]);
        let finish: &'static str = if cancelled {
            "cancel"
        } else if st.finish == "stop" {
            "stop"
        } else {
            "length"
        };
        if self.log {
            eprintln!(
                "glm-serve: prompt {} (reuse {reuse}) {:.0} ms | gen {} {:.0} ms ({:.1} tok/s) accept {}/{} | avail {:.2} GiB",
                prompt.len(),
                prompt_ms,
                gen.len(),
                st.decode_ms,
                gen.len() as f64 / st.decode_ms.max(1e-3) * 1e3,
                st.accepted,
                st.drafted,
                talus_model::glm::weights::mem_available_bytes() as f64 / (1u64 << 30) as f64
            );
        }
        write_done_line(
            out,
            &DoneStats {
                produced: gen.len() as i64,
                prompt_tokens: prompt.len(),
                prompt_ms,
                decode_ms: st.decode_ms,
                finish,
                accepted: st.accepted as i64,
                offered: st.drafted as i64,
                reuse,
                fresh,
            },
        )
        .ok();
        Ok(())
    }
}

impl EngineProtocolBackend for GlmBackend {
    fn max_context(&self) -> i32 {
        self.eng.cfg.ctx as i32
    }
    fn n_vocab(&self) -> i32 {
        self.eng.w.hp.n_vocab as i32
    }
    fn handle_gen(
        &mut self,
        cmd: &GenCommand,
        stop: &CancelToken,
        out: &mut dyn Write,
    ) -> std::io::Result<()> {
        if let Err(e) = self.run(cmd, stop, out) {
            writeln!(out, "ERR {e}")?;
            out.flush()?;
            // state is unknown after an error: drop the history so the next request starts cold
            self.hist.clear();
        }
        Ok(())
    }
}

/// GLM entry from talus-cli: serve mode here (needs talus-sched), other modes in talus_model::glm::cli.
pub fn run(argv: &[String]) -> i32 {
    let c = match gcli::parse(argv) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("talus-glm: {e}");
            return 2;
        }
    };
    if !c.serve {
        return gcli::run(argv);
    }
    talus_model::glm::weights::spawn_watchdog((c.minavail_gib * (1u64 << 30) as f64) as u64);
    let mut eng = match gcli::load_engine(&c) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("talus-glm: {e}");
            return 1;
        }
    };
    if let Some(dv) = &c.draft_vocab {
        match gcli::read_draft_vocab(dv).and_then(|ids| eng.set_draft_vocab(&ids)) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("talus-glm: draft vocab: {e}");
                return 1;
            }
        }
    }
    let _ = eng.reset();
    let mut be = GlmBackend::new(eng, c);
    talus_sched::serve_stdio(&mut be)
}
