//! Stdio engine protocol (`--serve` / `--engine-protocol`) for MiMo-V2.6-Flash (`mimo2`).
//!
//! Speaks the exact line-oriented text protocol expected by `server.py` and `talus-cli`:
//! - Startup banner: `READY <n_vocab> <max_ctx>`
//! - `RESET` -> `OK`
//! - `SEED <u64>` -> `OK`
//! - `MTP <draft_max>` -> `OK`
//! - `SAVE <path>` -> `OK` or `ERR <msg>`
//! - `LOAD <path>` -> `OK <n_pos> <n_tokens>` or `ERR <msg>`
//! - `GEN <n_gen> <temp> <top_p> <top_k> <n_prompt> <t_0> ...` ->
//!   `TOK <id>` ... `DONE <prefill_ms> <decode_ms> <n_drafted> <n_accepted>`
//! - `QUIT` -> exit `0`

use std::collections::VecDeque;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use talus_core_api::{CoreError, Result};

use crate::laguna::serve::{draft_q, sample, sample_dist, Mt19937_64, Sampling};
use crate::mimo::engine::{MimoEngine, MimoVerifyOutcome};
use crate::mimo::weights::{N_EXPERT_USED, SWA_WINDOW};
use talus_spec::ssamp::{judge, SparseDist, Verdict};

/// Default EOS token IDs for MiMo-V2.6-Flash (`<|endoftext|>` = 151643, `<|im_end|>` = 151645).
pub const DEFAULT_MIMO_EOS_IDS: [i32; 2] = [151643, 151645];

/// Kill switch for exact speculative sampling of sampled requests: default ON; `0`
/// restores the exact-match rule. The per-seed sampled bytes depend on it, the output distribution does not.
pub const SSAMP_ENV: &str = "TALUS_MIMO_SSAMP";

/// Default sampled-request draft cap for MiMo (`TALUS_MIMO_SPEC_SAMPLED`, default `3`; `0` = off).
pub const DEFAULT_MIMO_SPEC_SAMPLED: usize = 3;

/// Parse `TALUS_MIMO_SPEC_SAMPLED` into an optional draft cap (`None` when `0`/`off`/`false`).
#[inline]
pub fn parse_sampled_draft_cap(env_val: Option<&str>) -> Option<usize> {
    match env_val.map(str::trim) {
        None | Some("") => Some(DEFAULT_MIMO_SPEC_SAMPLED),
        Some(v) if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false") => {
            None
        }
        Some(v) => match v.parse::<usize>() {
            Ok(0) => None,
            Ok(n) => Some(n),
            Err(_) => Some(DEFAULT_MIMO_SPEC_SAMPLED),
        },
    }
}

/// Compute the effective per-request maximum draft length for MiMo.
/// Greedy requests (`temperature <= 0.0`) keep `config_draft_max`; sampled requests
/// (`temperature > 0.0`) are capped at `TALUS_MIMO_SPEC_SAMPLED` (default `3`, `0` = off).
#[inline]
pub fn effective_request_draft_max(
    config_draft_max: usize,
    temperature: f32,
    env_val: Option<&str>,
) -> usize {
    if temperature <= 0.0 {
        config_draft_max
    } else if let Some(cap) = parse_sampled_draft_cap(env_val) {
        config_draft_max.min(cap)
    } else {
        config_draft_max
    }
}

fn io_err(e: io::Error) -> CoreError {
    CoreError::IoError {
        message: e.to_string(),
    }
}

/// Run the line-oriented stdio serving loop over `reader` and `writer`.
pub fn serve_loop<R: BufRead, W: Write>(
    engine: &mut MimoEngine,
    reader: R,
    mut writer: W,
) -> Result<()> {
    writeln!(
        writer,
        "READY {} {}",
        engine.hparams().n_vocab,
        engine.config().max_seq_len
    )
    .map_err(io_err)?;
    writer.flush().map_err(io_err)?;

    for line_res in reader.lines() {
        let line = line_res.map_err(io_err)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        match parts[0] {
            "QUIT" | "EXIT" => {
                writeln!(writer, "BYE").map_err(io_err)?;
                writer.flush().map_err(io_err)?;
                break;
            }
            "RESET" => {
                engine.reset();
                writeln!(writer, "OK").map_err(io_err)?;
                writer.flush().map_err(io_err)?;
            }
            "SEED" => {
                writeln!(writer, "OK").map_err(io_err)?;
                writer.flush().map_err(io_err)?;
            }
            "MTP" => {
                writeln!(writer, "OK").map_err(io_err)?;
                writer.flush().map_err(io_err)?;
            }
            "SAVE" => {
                if parts.len() < 2 {
                    writeln!(writer, "ERR missing path for SAVE").map_err(io_err)?;
                } else {
                    match engine.save_session(Path::new(parts[1])) {
                        Ok(()) => writeln!(writer, "OK").map_err(io_err)?,
                        Err(e) => writeln!(writer, "ERR {e}").map_err(io_err)?,
                    }
                }
                writer.flush().map_err(io_err)?;
            }
            "LOAD" => {
                if parts.len() < 2 {
                    writeln!(writer, "ERR missing path for LOAD").map_err(io_err)?;
                } else {
                    match engine.load_session(Path::new(parts[1])) {
                        Ok(hdr) => {
                            writeln!(writer, "OK {} {}", hdr.n_pos, hdr.n_tokens).map_err(io_err)?
                        }
                        Err(e) => writeln!(writer, "ERR {e}").map_err(io_err)?,
                    }
                }
                writer.flush().map_err(io_err)?;
            }
            "GEN" => {
                if parts.len() < 6 {
                    writeln!(writer, "ERR malformed GEN command").map_err(io_err)?;
                    writer.flush().map_err(io_err)?;
                    continue;
                }
                let n_gen: usize = parts[1].parse().unwrap_or(1);
                let n_prompt: usize = parts[5].parse().unwrap_or(0);
                if parts.len() < 6 + n_prompt {
                    writeln!(writer, "ERR not enough prompt tokens in GEN").map_err(io_err)?;
                    writer.flush().map_err(io_err)?;
                    continue;
                }
                let prompt: Vec<i32> = parts[6..6 + n_prompt]
                    .iter()
                    .map(|s| s.parse::<i32>().unwrap_or(0))
                    .collect();

                match engine.generate(&prompt, n_gen) {
                    Ok(res) => {
                        for tok in &res.tokens {
                            writeln!(writer, "TOK {tok}").map_err(io_err)?;
                        }
                        let (drafted, accepted) = res
                            .spec_stats
                            .map(|s| (s.draft_proposed, s.draft_accepted))
                            .unwrap_or((0, 0));
                        writeln!(
                            writer,
                            "DONE {:.3} {:.3} {} {}",
                            res.prefill_ms, res.decode_ms, drafted, accepted
                        )
                        .map_err(io_err)?;
                    }
                    Err(e) => {
                        writeln!(writer, "ERR {e}").map_err(io_err)?;
                    }
                }
                writer.flush().map_err(io_err)?;
            }
            other => {
                writeln!(writer, "ERR unknown command '{other}'").map_err(io_err)?;
                writer.flush().map_err(io_err)?;
            }
        }
    }

    Ok(())
}

struct InputQueue {
    mu: Mutex<(VecDeque<String>, bool)>,
    cv: Condvar,
    stop: AtomicBool,
}

impl InputQueue {
    fn new() -> Self {
        Self {
            mu: Mutex::new((VecDeque::new(), false)),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
        }
    }

    fn next_line(&self) -> Option<String> {
        let mut guard = self.mu.lock().unwrap();
        while guard.0.is_empty() && !guard.1 {
            guard = self.cv.wait(guard).unwrap();
        }
        guard.0.pop_front()
    }
}

#[inline]
fn emit(line: &str) {
    let mut out = io::stdout().lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

#[inline]
fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Run the Strata-compatible stdio engine protocol (`--serve` / `--engine-protocol`) attached to process stdin/stdout.
pub fn serve_main(engine: &mut MimoEngine) -> Result<()> {
    let v_len = engine.hparams().n_vocab;
    let ctx = engine.config().max_seq_len;
    let ubatch = engine.config().ubatch.max(1);
    let kv_str = if engine.config().kv_q8 { "q8" } else { "f16" };
    emit(&format!(
        "INFO engine=talus model=mimo vocab={v_len} ubatch={ubatch} kv={kv_str}"
    ));
    emit(&format!("READY {ctx} stop"));

    let input = Arc::new(InputQueue::new());
    {
        let input_bg = Arc::clone(&input);
        std::thread::spawn(move || {
            let stdin = io::stdin();
            for line_res in stdin.lock().lines() {
                match line_res {
                    Ok(mut line) => {
                        if line.ends_with('\r') {
                            line.pop();
                        }
                        if line == "STOP" {
                            input_bg.stop.store(true, Ordering::SeqCst);
                            continue;
                        }
                        let mut guard = input_bg.mu.lock().unwrap();
                        guard.0.push_back(line);
                        input_bg.cv.notify_one();
                    }
                    Err(_) => break,
                }
            }
            let mut guard = input_bg.mu.lock().unwrap();
            guard.1 = true;
            input_bg.cv.notify_one();
        });
    }

    let mut rng = Mt19937_64::new(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42),
    );

    while let Some(line) = input.next_line() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let cmd = parts.next().unwrap_or("");
        match cmd {
            "QUIT" | "EXIT" => break,
            "RESET" => {
                engine.reset();
                emit("OK");
                continue;
            }
            "SEED" | "MTP" => {
                emit("OK");
                continue;
            }
            "SAVE" | "RESTORE" | "LOAD" => {
                let rest = trimmed[cmd.len()..].trim();
                if rest.is_empty() {
                    emit("SERR invalid 0 no path");
                    continue;
                }
                let ts = Instant::now();
                let path = Path::new(rest);
                if cmd == "SAVE" {
                    match engine.save_session(path) {
                        Ok(()) => emit(&format!("SAVED {} 0 {:.1}", engine.n_pos(), ms_since(ts))),
                        Err(e) => emit(&format!("SERR io 0 {e}")),
                    }
                } else {
                    match engine.load_session(path) {
                        Ok(hdr) => {
                            emit(&format!("RESTORED {} 0 {:.1}", hdr.n_tokens, ms_since(ts)))
                        }
                        Err(e) => emit(&format!("SERR io 0 {e}")),
                    }
                }
                continue;
            }
            "GEN" => {}
            _ => {
                emit("ERR unsupported command (the MiMo engine takes GEN, STOP, QUIT)");
                continue;
            }
        }

        let Some(max_new_str) = parts.next() else {
            emit("ERR expected: GEN <max_new> <id,id,...>");
            continue;
        };
        let Ok(max_new) = max_new_str.parse::<i64>() else {
            emit("ERR expected: GEN <max_new> <id,id,...>");
            continue;
        };
        if max_new < 0 {
            emit("ERR expected: GEN <max_new> <id,id,...>");
            continue;
        }

        let mut sp = Sampling::default();
        let mut ignore_eos = false;
        let mut no_cache = false;
        let mut ids_text = "";
        for tok in parts {
            if let Some((k, v)) = tok.split_once('=') {
                match k {
                    "temperature" => {
                        if let Ok(x) = v.parse::<f32>() {
                            sp.temperature = x;
                        }
                    }
                    "top_p" => {
                        if let Ok(x) = v.parse::<f32>() {
                            sp.top_p = x;
                        }
                    }
                    "top_k" => {
                        if let Ok(x) = v.parse::<i32>() {
                            sp.top_k = x;
                        }
                    }
                    "min_p" => {
                        if let Ok(x) = v.parse::<f32>() {
                            sp.min_p = x;
                        }
                    }
                    "seed" => {
                        if let Ok(x) = v.parse::<i64>() {
                            sp.seed = x;
                        }
                    }
                    "penalty_repeat" => {
                        if let Ok(x) = v.parse::<f32>() {
                            sp.penalty_repeat = x;
                        }
                    }
                    "penalty_freq" => {
                        if let Ok(x) = v.parse::<f32>() {
                            sp.penalty_freq = x;
                        }
                    }
                    "penalty_present" => {
                        if let Ok(x) = v.parse::<f32>() {
                            sp.penalty_present = x;
                        }
                    }
                    "penalty_last_n" => {
                        if let Ok(x) = v.parse::<i32>() {
                            sp.penalty_last_n = x;
                        }
                    }
                    "ignore_eos" => {
                        ignore_eos = v == "1" || v.eq_ignore_ascii_case("true");
                    }
                    "cache" => {
                        no_cache = v == "0" || v.eq_ignore_ascii_case("false");
                    }
                    _ => {}
                }
            } else {
                ids_text = tok;
                break;
            }
        }

        let mut ids: Vec<i32> = Vec::new();
        let mut bad = false;
        for t in ids_text.split(',') {
            if t.is_empty() {
                continue;
            }
            match t.parse::<i32>() {
                Ok(id) if id >= 0 && (id as usize) < v_len => ids.push(id),
                _ => {
                    bad = true;
                    break;
                }
            }
        }
        if bad || ids.is_empty() {
            emit("ERR bad prompt ids");
            continue;
        }
        if (ids.len() as i64) + max_new + 8 > (ctx as i64) {
            emit(&format!(
                "ERR prompt {} + max_new {} exceeds the context {}",
                ids.len(),
                max_new,
                ctx
            ));
            continue;
        }
        if sp.seed >= 0 {
            rng.seed(sp.seed as u64);
        }
        input.stop.store(false, Ordering::SeqCst);

        let tp = Instant::now();
        let mut lcp = 0usize;
        if no_cache {
            engine.reset();
        } else {
            let hist = engine.token_history();
            while lcp < hist.len() && lcp < ids.len() && hist[lcp] == ids[lcp] {
                lcp += 1;
            }
            if lcp == ids.len() && lcp > 0 {
                lcp -= 1;
            }
            let max_rollback = engine.swa_stride().saturating_sub(SWA_WINDOW as usize);
            if lcp == 0 || engine.n_pos().saturating_sub(lcp) > max_rollback {
                engine.reset();
                lcp = 0;
            } else if lcp < engine.n_pos() {
                engine.rollback_kv_to(lcp);
            }
        }
        if lcp > 0 {
            emit(&format!("RESUME {lcp}"));
        }

        let n_read = ids.len() - lcp;
        let mut done = 0usize;
        let mut last_logits = Vec::new();
        let mut prefill_err: Option<CoreError> = None;
        while done < n_read {
            let n = ubatch.min(n_read - done);
            let is_last = done + n == n_read;
            match engine.forward_prefill_chunk(&ids[lcp + done..lcp + done + n], is_last) {
                Ok(Some(last)) => {
                    last_logits = last;
                }
                Ok(None) => {}
                Err(e) => {
                    prefill_err = Some(e);
                    break;
                }
            }
            done += n;
            let ms = ms_since(tp);
            let tps = if ms > 0.0 {
                (done as f64) / (ms / 1000.0)
            } else {
                0.0
            };
            emit(&format!("PP {done} {n_read} {:.0} {:.1}", ms, tps));
        }
        if let Some(e) = prefill_err {
            engine.reset();
            emit(&format!("ERR {e}"));
            continue;
        }
        let prompt_ms = ms_since(tp);

        if max_new == 0 || last_logits.is_empty() {
            emit(&format!(
                "DONE 0 {} {:.1} 0.0 length 0 0 {} 0 0 0 0 0 {}",
                ids.len(),
                prompt_ms,
                lcp,
                n_read
            ));
            continue;
        }

        let td = Instant::now();
        let recent_start = ids.len().saturating_sub(256);
        let mut recent: Vec<i32> = ids[recent_start..].to_vec();
        let mut gen: i64 = 0;
        let mut drafted: i64 = 0;
        let mut accepted_total: i64 = 0;
        let mut finish = "length";
        let mut next = sample(&mut last_logits, &sp, &recent, &mut rng);
        let fast = sp.greedy_plain();
        let use_dflash = engine.use_dflash();
        let env_sampled_cap = std::env::var("TALUS_MIMO_SPEC_SAMPLED").ok();
        let draft_max = effective_request_draft_max(
            engine.config().draft_max,
            sp.temperature,
            env_sampled_cap.as_deref(),
        );
        let mut spec_svc = talus_spec::SpecEngineService::from_decl(
            talus_spec::ModelSpecDecl::mimo(draft_max, use_dflash),
        );
        spec_svc.reset(&ids);
        // exact speculative sampling for sampled requests (default on; TALUS_MIMO_SSAMP=0 = kill switch). Greedy
        // requests (temperature <= 0) never take it.
        let ssamp_on = sp.temperature > 0.0 && talus_spec::ssamp::ssamp_enabled(SSAMP_ENV);
        let q_scale = talus_spec::ssamp::q_temp_scale();
        let pen_on =
            sp.penalty_repeat != 1.0 || sp.penalty_freq != 0.0 || sp.penalty_present != 0.0;
        let is_eos = |tok: i32| !ignore_eos && DEFAULT_MIMO_EOS_IDS.contains(&tok);
        // greedy requests verify a long-span copy of earlier context (talus_spec::CopyProposer, idea from
        // gufo, MIT) instead of the DFlash drafts when one qualifies; the verify window injects the DFlash context
        // as any window does. TALUS_COPY=0 / TALUS_MIMO_COPY=0: off.
        let mut mcopy = (fast
            && use_dflash
            && talus_spec::copy_enabled()
            && std::env::var("TALUS_MIMO_COPY")
                .map(|v| v != "0")
                .unwrap_or(true))
        .then(|| {
            let mut c = talus_spec::CopyProposer::new(talus_spec::copy_cfg_from_env());
            c.extend(&ids);
            c.begin_request();
            c
        });
        // the batch path returns every row's argmax only for windows of up to N_EXPERT_USED rows; default 4 = the
        // DFlash window (verify windows of 6+ rows round differently: greedy 4/9 at 8 rows, 9/9 at 5 in s4 / s5)
        let copy_cap = talus_spec::copy_env_f64("MIMO_MAX")
            .map_or(4, |v| v as usize)
            .clamp(1, N_EXPERT_USED - 1);
        // a copy replaces the DFlash drafts (~0.7 accepted on code): only buckets that keep >= this many copied tokens
        let copy_qmin = talus_spec::copy_env_f64("QMIN").unwrap_or(0.85);
        let copy_row_cost = talus_spec::copy_env_f64("ROWCOST").unwrap_or(0.10).max(0.0);
        let copy_gain = talus_spec::copy_env_f64("GAIN").unwrap_or(0.03);
        let mut copy_buf: Vec<i32> = Vec::new();

        loop {
            emit(&format!("T {next}"));
            gen += 1;
            recent.push(next);
            spec_svc.ingest_token(next);
            if let Some(c) = mcopy.as_mut() {
                c.push(next);
            }
            if is_eos(next) {
                finish = "stop";
                break;
            }
            if input.stop.load(Ordering::SeqCst) {
                finish = "cancel";
                break;
            }
            if gen >= max_new {
                break;
            }

            let draft_cap = if use_dflash && (engine.n_pos() + 6) < ctx {
                let use_adaptive = std::env::var("TALUS_MIMO_ADAPTIVE")
                    .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                    .unwrap_or(true);
                if use_adaptive {
                    spec_svc
                        .select_draft_len(engine.n_pos(), fast)
                        .min(draft_max)
                } else {
                    draft_max
                }
            } else {
                0
            };

            let mut copied: Option<talus_spec::CopyMatch> = None;
            if let Some(c) = mcopy.as_ref() {
                let room = ctx.saturating_sub(engine.n_pos() + 6);
                if let Some(m) =
                    c.propose(&[], copy_cap.min(room), false, false, is_eos, &mut copy_buf)
                {
                    let k =
                        c.choose_len_min(&m, copy_buf.len(), copy_row_cost, copy_gain, copy_qmin);
                    if k > 0 {
                        copy_buf.truncate(k);
                        copied = Some(m);
                    }
                }
            }
            if draft_cap > 0 || copied.is_some() {
                let mut cands: Vec<Vec<(i32, f32)>> = Vec::new();
                let proposed = if copied.is_some() {
                    Ok(std::mem::take(&mut copy_buf))
                } else if ssamp_on {
                    engine
                        .propose_drafts_with_cands(next, engine.n_pos())
                        .map(|(d, c)| {
                            cands = c;
                            d
                        })
                } else {
                    engine.propose_drafts(next, engine.n_pos())
                };
                let mut drafts = match proposed {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("talus-mimo: dflash propose error: {e}");
                        Vec::new()
                    }
                };
                if copied.is_none() && drafts.len() > draft_cap {
                    drafts.truncate(draft_cap);
                }
                // ssamp: draw each draft from its row's q (the drafter's top-16 under the request's transforms) with
                // the request's RNG; the DFlash rows do not condition on earlier drafts, so q_i is fixed per round.
                // A row without candidates keeps its argmax draft as a point mass (the exact-match rule).
                let mut qs: Vec<SparseDist> = Vec::new();
                if ssamp_on && !fast && cands.len() >= drafts.len() {
                    let mut hist = if pen_on { recent.clone() } else { Vec::new() };
                    for i in 0..drafts.len() {
                        let h: &[i32] = if pen_on { &hist } else { &recent };
                        let q = draft_q(&cands[i], &sp, h, q_scale);
                        if q.is_empty() {
                            qs.push(SparseDist::one_hot(drafts[i]));
                        } else {
                            drafts[i] = q.draw(&mut rng);
                            qs.push(q);
                        }
                        if pen_on {
                            hist.push(drafts[i]);
                        }
                    }
                }
                if !drafts.is_empty() {
                    let k = drafts.len();
                    drafted += k as i64;
                    let pos_before_verify = engine.n_pos();
                    let mut verify_batch = Vec::with_capacity(k + 1);
                    verify_batch.push(next);
                    verify_batch.extend_from_slice(&drafts);
                    if fast {
                        let argmax_ids = match engine.forward_batch_argmax(&verify_batch) {
                            Ok(ids) => ids,
                            Err(e) => {
                                eprintln!("talus-mimo: verify error: {e}");
                                finish = "stop";
                                break;
                            }
                        };
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
                        let outcome = MimoVerifyOutcome {
                            accepted,
                            bonus_token,
                        };
                        engine.rollback_kv_to(pos_before_verify + 1 + outcome.accepted.len());
                        accepted_total += outcome.accepted.len() as i64;

                        let mut ended_in_drafts = false;
                        let mut kept_acc = 0usize;
                        for &acc in &outcome.accepted {
                            emit(&format!("T {acc}"));
                            gen += 1;
                            kept_acc += 1;
                            recent.push(acc);
                            spec_svc.ingest_token(acc);
                            if let Some(c) = mcopy.as_mut() {
                                c.push(acc);
                            }
                            if is_eos(acc) {
                                finish = "stop";
                                ended_in_drafts = true;
                                break;
                            }
                            if input.stop.load(Ordering::SeqCst) {
                                finish = "cancel";
                                ended_in_drafts = true;
                                break;
                            }
                            if gen >= max_new {
                                ended_in_drafts = true;
                                break;
                            }
                        }
                        if let (Some(m), Some(c)) = (copied, mcopy.as_mut()) {
                            c.observe(&m, k, kept_acc, false);
                        } else {
                            spec_svc.record_verify(k, kept_acc);
                        }
                        if ended_in_drafts {
                            engine.rollback_kv_to(pos_before_verify + 1 + kept_acc);
                            break;
                        }
                        next = outcome.bonus_token;
                        continue;
                    } else {
                        let target_logits = match engine.forward_batch(&verify_batch) {
                            Ok(l) => l,
                            Err(e) => {
                                eprintln!("talus-mimo: verify error: {e}");
                                finish = "stop";
                                break;
                            }
                        };
                        let mut tail_logits = if target_logits.len() > k + 1 {
                            target_logits[target_logits.len() - (k + 1)..].to_vec()
                        } else {
                            target_logits
                        };
                        let mut ended_in_drafts = false;
                        let mut kept_acc = 0usize;
                        for i in 0..=k {
                            if i >= tail_logits.len() {
                                break;
                            }
                            // `None` = draft i kept; `Some(t)` = the row emits t and the round ends
                            let ends: Option<i32> = if i < k && i < qs.len() {
                                let p = sample_dist(&mut tail_logits[i], None, &sp, &recent);
                                match judge(&p, &qs[i], drafts[i], &mut rng) {
                                    Verdict::Accept => None,
                                    Verdict::Reject(y) => Some(y),
                                }
                            } else {
                                let choice = sample(&mut tail_logits[i], &sp, &recent, &mut rng);
                                if i < k && choice == drafts[i] {
                                    None
                                } else {
                                    Some(choice)
                                }
                            };
                            if ends.is_none() {
                                let acc = drafts[i];
                                emit(&format!("T {acc}"));
                                gen += 1;
                                kept_acc += 1;
                                accepted_total += 1;
                                recent.push(acc);
                                spec_svc.ingest_token(acc);
                                if is_eos(acc) {
                                    finish = "stop";
                                    ended_in_drafts = true;
                                    break;
                                }
                                if input.stop.load(Ordering::SeqCst) {
                                    finish = "cancel";
                                    ended_in_drafts = true;
                                    break;
                                }
                                if gen >= max_new {
                                    ended_in_drafts = true;
                                    break;
                                }
                                continue;
                            }
                            next = ends.unwrap_or(0);
                            break;
                        }
                        spec_svc.record_verify(k, kept_acc);
                        engine.rollback_kv_to(pos_before_verify + 1 + kept_acc);
                        if ended_in_drafts {
                            break;
                        }
                        continue;
                    }
                }
            }

            if use_dflash && draft_cap == 0 {
                spec_svc.record_verify(0, 0);
            }
            if fast {
                match engine.forward_batch_argmax(&[next]) {
                    Ok(ids) => {
                        if let Some(&id) = ids.last() {
                            next = id;
                        } else {
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("talus-mimo: decode error: {e}");
                        break;
                    }
                }
            } else {
                match engine.forward_batch(&[next]) {
                    Ok(mut rows) => {
                        if let Some(mut row) = rows.pop() {
                            next = sample(&mut row, &sp, &recent, &mut rng);
                        } else {
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("talus-mimo: decode error: {e}");
                        break;
                    }
                }
            }
        }

        let decode_ms = ms_since(td);
        if let Some(c) = mcopy.as_mut() {
            let st = c.take_stats();
            if st.windows > 0 {
                eprintln!(
                    "talus copy: {} windows, {} of {} copied drafts accepted",
                    st.windows, st.accepted, st.proposed
                );
            }
        }
        eprintln!(
            "talus-mimo-serve: {} tokens in {:.1} ms ({:.2} tok/s), drafts {}/{}, prompt {} ({} read, {:.1} ms)",
            gen,
            decode_ms,
            (gen as f64) / (decode_ms / 1000.0).max(1e-9),
            accepted_total,
            drafted,
            ids.len(),
            n_read,
            prompt_ms
        );
        emit(&format!(
            "DONE {} {} {:.1} {:.1} {} {} {} {} 0 0 0 0 0 {}",
            gen,
            ids.len(),
            prompt_ms,
            decode_ms,
            finish,
            accepted_total,
            drafted,
            lcp,
            n_read
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampled_draft_cap_greedy_vs_sampled() {
        // Greedy (temperature <= 0) keeps config_draft_max (4) regardless of TALUS_MIMO_SPEC_SAMPLED.
        assert_eq!(effective_request_draft_max(4, 0.0, None), 4);
        assert_eq!(effective_request_draft_max(4, 0.0, Some("3")), 4);
        assert_eq!(effective_request_draft_max(4, 0.0, Some("2")), 4);
        assert_eq!(effective_request_draft_max(4, 0.0, Some("0")), 4);

        // Sampled (temperature > 0) defaults to 3.
        assert_eq!(effective_request_draft_max(4, 1.0, None), 3);
        assert_eq!(effective_request_draft_max(4, 0.8, Some("")), 3);
        assert_eq!(effective_request_draft_max(4, 1.0, Some("3")), 3);
        assert_eq!(effective_request_draft_max(4, 1.0, Some("2")), 2);
        assert_eq!(effective_request_draft_max(2, 1.0, None), 2);

        // TALUS_MIMO_SPEC_SAMPLED=0 disables the sampled cap (restores config_draft_max).
        assert_eq!(effective_request_draft_max(4, 1.0, Some("0")), 4);
        assert_eq!(effective_request_draft_max(4, 1.0, Some("off")), 4);
        assert_eq!(effective_request_draft_max(4, 1.0, Some("false")), 4);
    }

    #[test]
    fn test_sampled_draft_cap_adaptive_select_draft_len() {
        let greedy_max = effective_request_draft_max(4, 0.0, None);
        let mut greedy_svc = talus_spec::SpecEngineService::from_decl(
            talus_spec::ModelSpecDecl::mimo(greedy_max, true),
        );
        greedy_svc.reset(&[1, 2, 3]);
        assert_eq!(greedy_svc.select_draft_len(3, true).min(greedy_max), 4);

        let sampled_max = effective_request_draft_max(4, 1.0, None);
        let mut sampled_svc = talus_spec::SpecEngineService::from_decl(
            talus_spec::ModelSpecDecl::mimo(sampled_max, true),
        );
        sampled_svc.reset(&[1, 2, 3]);
        assert_eq!(sampled_svc.select_draft_len(3, false).min(sampled_max), 3);

        let off_max = effective_request_draft_max(4, 1.0, Some("0"));
        let mut off_svc = talus_spec::SpecEngineService::from_decl(
            talus_spec::ModelSpecDecl::mimo(off_max, true),
        );
        off_svc.reset(&[1, 2, 3]);
        assert_eq!(off_svc.select_draft_len(3, false).min(off_max), 4);
    }
}
