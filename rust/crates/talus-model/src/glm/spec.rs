//! GLM generation: prompt prefill (trunk + MTP KV catch-up), plain decode, and MTP n2 speculative decode
//! (draft 2 tokens with the NextN block, verify 3 with the trunk, rollback through the KDA snapshot slots).

use std::time::Instant;

use talus_core_api::Result;

use super::engine::{asub, kda_slots, pch, Buf, GlmEngine, PCH, TMAX};
use super::sample::{self, Rng, SamplerParams};

#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_new: usize,
    pub samp: SamplerParams,
    pub mtp: bool,
    pub eos: Vec<u32>,
    pub ignore_eos: bool,
    /// the request's context (prompt tokens), the long-span copy proposer's source; empty = no copies
    pub ctx: Vec<u32>,
}

/// the long-span copy proposer (talus_spec::CopyProposer, idea from gufo, MIT) for greedy GLM requests:
/// opt-in (GLM_COPY=1; it also raises the KDA rollback slots 3 -> 8, +0.72 GiB). greedy output differs (verify windows wider than 3 round differently) and
/// GLM's edit replies are mostly prose, so few tokens are copies.
pub fn copy_on() -> bool {
    talus_spec::copy_enabled() && std::env::var("GLM_COPY").map(|v| v == "1").unwrap_or(false)
}

#[derive(Debug, Clone, Default)]
pub struct GenStats {
    pub prompt_ms: f64,
    pub decode_ms: f64,
    pub n_prompt: usize,
    pub n_gen: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub rounds: usize,
    pub finish: String,
}

/// Sequence state between calls (single sequence).
#[derive(Debug, Clone, Default)]
pub struct SeqPos {
    /// tokens whose trunk state is committed (positions 0..n_past)
    pub n_past: usize,
    /// trunk h (post output-norm) of position n_past - 1
    pub h_prev: Vec<f32>,
}

/// Prefill `toks` at positions seq.n_past.. (trunk, plus the MTP KV catch-up when `mtp`). Returns the logits of
/// the last token.
pub fn prefill(eng: &mut GlmEngine, seq: &mut SeqPos, toks: &[u32], mtp: bool) -> Result<Vec<f32>> {
    prefill_with(eng, seq, toks, mtp, &[], &mut |_, _| Ok(None))
}

/// Prompt-cache snapshot protocol of `prefill_with`.
pub enum SnapEvent<'a> {
    /// a snapshot at this position is coming: return the buffer to fill (`snap_save` layout), or None to skip
    Want(usize),
    /// the snapshot at this position is complete (stream-ordered) with the trunk h of position pos - 1
    Done(usize, &'a [f32]),
}

/// Prefill with recurrent-state snapshots at `points` (absolute positions = token counts, multiples of the
/// attention slice). A point at a window end is saved after the window; a point inside a sliced window (2048-row
/// chunks run attention in 512-row slices) is captured layer by layer as the slice ending there completes, so the
/// chunk stays whole. Points that fit neither end the window there.
pub fn prefill_with(
    eng: &mut GlmEngine,
    seq: &mut SeqPos,
    toks: &[u32],
    mtp: bool,
    points: &[usize],
    on_point: &mut dyn FnMut(&mut GlmEngine, SnapEvent) -> Result<Option<Buf>>,
) -> Result<Vec<f32>> {
    let e = eng.w.hp.n_embd;
    if seq.h_prev.len() != e {
        seq.h_prev = vec![0.0; e];
    }
    let mut i = 0;
    let mut last = vec![];
    while i < toks.len() {
        // prefill chunks aligned to absolute multiples of PCH (the MMQ path above TMAX tokens)
        let mut t = (toks.len() - i).min(pch() - (seq.n_past + i) % pch());
        // a snapshot point inside the window: captured mid-window when the window is sliced and the point sits on
        // a slice boundary, else the window ends there
        let at = seq.n_past + i;
        let mut mid: Option<usize> = None;
        if let Some(&p) = points.iter().filter(|&&p| p > at && p < at + t).min() {
            if t > asub() && (p - at) % asub() == 0 {
                mid = Some(p);
            } else {
                t = p - at;
            }
        }
        if let Some(p) = mid {
            eng.snap_mid = on_point(eng, SnapEvent::Want(p))?.map(|b| (p, b));
        }
        if t < 64 {
            // short tails: decode-size windows beat a small MMQ chunk (keeps TMAX alignment)
            t = t.min(TMAX - (seq.n_past + i) % TMAX);
        }
        let pos0 = seq.n_past + i;
        let cur = eng.st.cur_slot as i32;
        let r = eng.trunk(&toks[i..i + t], pos0, &[cur], 1);
        let mid_done = eng.snap_mid.take().is_some();
        r?;
        if mid_done {
            let p = mid.unwrap();
            let h = eng.get_h_row(p - 1 - pos0, false)?;
            on_point(eng, SnapEvent::Done(p, &h))?;
        }
        if i + t == toks.len() {
            last = eng.logits_row(if t <= TMAX { t - 1 } else { 0 }).to_vec();
        }
        if mtp {
            // the MTP block's buffers hold PCH rows: windows above it in PCH-row slices (hin row j of a
            // slice at s = the trunk's h row s + j - 1; the trunk h stays intact across the slices)
            for s0 in (0..t).step_by(PCH) {
                let tt = (t - s0).min(PCH);
                if s0 == 0 {
                    eng.put_hin_row(0, &seq.h_prev)?;
                } else {
                    eng.set_hin(0, Some((s0 - 1, false)))?;
                }
                for j in 1..tt {
                    eng.set_hin(j, Some((s0 + j - 1, false)))?;
                }
                eng.mtp(&toks[i + s0..i + s0 + tt], pos0 + s0, false)?;
            }
        }
        seq.h_prev = eng.get_h_row(t - 1, false)?;
        i += t;
        let pos = seq.n_past + i;
        if points.contains(&pos) {
            if let Some(buf) = on_point(eng, SnapEvent::Want(pos))? {
                eng.snap_save(&buf)?;
                let h = seq.h_prev.clone();
                on_point(eng, SnapEvent::Done(pos, &h))?;
            }
        }
    }
    seq.n_past += toks.len();
    Ok(last)
}

/// Generate after a prefill. `first` = logits of the last prompt token. `on_tok` returns false to stop.
pub fn generate(
    eng: &mut GlmEngine,
    seq: &mut SeqPos,
    first: &[f32],
    p: &GenParams,
    stats: &mut GenStats,
    mut on_tok: impl FnMut(u32) -> bool,
) -> Result<Vec<u32>> {
    let t0 = Instant::now();
    let mut rng = Rng::new(p.samp.seed);
    let greedy = p.samp.greedy();
    // The MTP drafts of a sampled request are drawn from q (the draft head under the request's chain) and judged
    // by exact speculative sampling (sample::accept). A/B reference: GLM_SSAMP_OFF=1 drafts the head's
    // argmax as a point mass instead (the exact-match rule's acceptance; same output distribution).
    let q_samp = if !greedy && sample::ssamp_off() {
        SamplerParams {
            temp: 0.0,
            ..p.samp
        }
    } else {
        p.samp
    };
    let mut out = vec![];
    let is_eos = |t: u32| !p.ignore_eos && p.eos.contains(&t);
    let d0 = sample::dist(first, &p.samp, None);
    let mut id_last = sample::sample(&d0, &mut rng);
    out.push(id_last);
    let mut stop = is_eos(id_last) || !on_tok(id_last);
    let ctx = eng.cfg.ctx;
    let use_mtp = p.mtp && eng.w.layers.len() > eng.w.hp.n_layer;
    let mut catch: Vec<u32> = vec![]; // accepted drafts of the previous round (MTP KV catch-up)
    let mut first_round = true;
    let mut last_row: Option<usize> = None;
    // greedy rounds chained on the device (draft argmax -> next window embedding on the GPU, one host sync per
    // round instead of three, no logits D2H); same tokens as the host path (GLM_DEV_GREEDY=0)
    let devg = greedy
        && if use_mtp {
            eng.dev_greedy_ok()
        } else {
            eng.dev_greedy_plain_ok()
        };
    // greedy MTP requests may verify a copy of earlier context (up to TMAX tokens) instead of the MTP's
    // two drafts; the MTP then only catches up (its KV) in that round (opt-in: see copy_on).
    let slots = kda_slots();
    let mut copy = (greedy && use_mtp && slots > 3 && !p.ctx.is_empty() && copy_on()).then(|| {
        let mut c = talus_spec::CopyProposer::new(talus_spec::copy_cfg_from_env());
        let ids: Vec<i32> = p.ctx.iter().map(|&t| t as i32).collect();
        c.extend(&ids);
        c.begin_request();
        c
    });
    let copy_row_cost = talus_spec::copy_env_f64("ROWCOST").unwrap_or(0.10).max(0.0);
    let copy_gain = talus_spec::copy_env_f64("GAIN").unwrap_or(0.03);
    let copy_min = talus_spec::copy_env_f64("GLM_MINK")
        .map(|v| v as usize)
        .unwrap_or(3);
    let mut copy_buf: Vec<i32> = Vec::new();
    let mut indexed = 0usize; // tokens of `out` in the copy index
    while !stop && out.len() < p.max_new && seq.n_past + 4 < ctx {
        stats.rounds += 1;
        if let Some(cp) = copy.as_mut() {
            let o: Vec<i32> = out[indexed..].iter().map(|&t| t as i32).collect();
            cp.extend(&o);
            indexed = out.len();
            let room = ctx.saturating_sub(seq.n_past + 4);
            let cap = (slots - 1).min(TMAX - 1).min(room);
            let m = if first_round || cap < copy_min {
                None
            } else {
                cp.propose(&[], cap, false, false, |t| is_eos(t as u32), &mut copy_buf)
            };
            let k = m.map_or(0, |m| {
                cp.choose_len(&m, copy_buf.len(), copy_row_cost, copy_gain)
            });
            if let (Some(m), true) = (m, k >= copy_min) {
                // MTP KV catch-up for the last window's accepted drafts + id_last (no draft)
                let mut mt = catch.clone();
                mt.push(id_last);
                let mpos = seq.n_past - catch.len();
                for j in 0..mt.len() {
                    eng.set_hin(j, Some((j, false)))?;
                }
                eng.mtp(&mt, mpos, false)?;
                let cp_toks: Vec<u32> = copy_buf[..k].iter().map(|&t| t as u32).collect();
                let mut toks = Vec::with_capacity(k + 1);
                toks.push(id_last);
                toks.extend_from_slice(&cp_toks);
                let tw = toks.len();
                let cur = eng.st.cur_slot as i32;
                let kids: Vec<i32> = std::iter::once(cur)
                    .chain((0..slots as i32).filter(|&x| x != cur))
                    .take(tw)
                    .collect();
                if devg {
                    eng.trunk_greedy(&toks, &[], seq.n_past, &kids, tw, 2)?;
                } else {
                    eng.trunk(&toks, seq.n_past, &kids, tw)?;
                }
                let tgt = |eng: &GlmEngine, r: usize, rng: &mut Rng| -> u32 {
                    if devg {
                        eng.greedy_tok(2 + r)
                    } else {
                        let d = sample::dist(eng.logits_row(r), &p.samp, None);
                        sample::sample(&d, rng)
                    }
                };
                let mut a = 0usize;
                let mut next = 0u32;
                for (r, &dr) in cp_toks.iter().enumerate() {
                    let tg = tgt(eng, r, &mut rng);
                    if tg == dr {
                        a += 1;
                        out.push(dr);
                        if is_eos(dr) || !on_tok(dr) || out.len() >= p.max_new {
                            stop = true;
                            break;
                        }
                    } else {
                        next = tg;
                        break;
                    }
                }
                if a == k && !stop {
                    next = tgt(eng, k, &mut rng);
                }
                cp.observe(&m, k, a, false);
                stats.drafted += k;
                stats.accepted += a;
                eng.st.cur_slot = kids[a] as usize;
                seq.n_past += 1 + a;
                catch = cp_toks[..a].to_vec();
                last_row = Some(a);
                if stop {
                    break;
                }
                id_last = next;
                out.push(id_last);
                stop = is_eos(id_last) || !on_tok(id_last);
                continue;
            }
        }
        if devg && !use_mtp {
            let cur = eng.st.cur_slot as i32;
            eng.trunk_greedy(&[id_last], &[], seq.n_past, &[cur], 1, 2)?;
            seq.n_past += 1;
            last_row = Some(0);
            id_last = eng.greedy_tok(2);
            out.push(id_last);
            stop = is_eos(id_last) || !on_tok(id_last);
            continue;
        }
        if devg {
            let mut mt = catch.clone();
            mt.push(id_last);
            let mpos = seq.n_past - catch.len();
            if first_round {
                eng.put_hin_row(0, &seq.h_prev)?;
                first_round = false;
            } else {
                for j in 0..mt.len() {
                    eng.set_hin(j, Some((j, false)))?;
                }
            }
            // draft 1 -> greedy slot 0, draft 2 -> slot 1, verify argmax rows -> slots 2..5. Chained: drafted tokens
            // are embedded on the device, one sync per round; else the host feeds them (a token-id D2H per step)
            let chain = eng.dev_chain_ok();
            eng.mtp_greedy(&mt, &[], mpos, 0, !chain)?;
            eng.set_hin(0, Some((mt.len() - 1, true)))?;
            let cur = eng.st.cur_slot as i32;
            let others: Vec<i32> = (0..3).filter(|&x| x != cur).collect();
            let kids = [cur, others[0], others[1]];
            if chain {
                eng.mtp_greedy(&[id_last], &[(0, 0)], seq.n_past + 1, 1, false)?;
                eng.trunk_greedy(
                    &[id_last, id_last, id_last],
                    &[(1, 0), (2, 1)],
                    seq.n_past,
                    &kids,
                    3,
                    2,
                )?;
            } else {
                let d1 = eng.greedy_tok(0);
                eng.mtp_greedy(&[d1], &[], seq.n_past + 1, 1, true)?;
                let d2 = eng.greedy_tok(1);
                eng.trunk_greedy(&[id_last, d1, d2], &[], seq.n_past, &kids, 3, 2)?;
            }
            stats.drafted += 2;
            let (d1, d2) = (eng.greedy_tok(0), eng.greedy_tok(1));
            let mut a = 0usize;
            let mut next = 0u32;
            for (r, dr) in [d1, d2].into_iter().enumerate() {
                let tgt = eng.greedy_tok(2 + r);
                if tgt == dr {
                    a += 1;
                    out.push(dr);
                    if is_eos(dr) || !on_tok(dr) || out.len() >= p.max_new {
                        stop = true;
                        break;
                    }
                } else {
                    next = tgt;
                    break;
                }
            }
            if a == 2 && !stop {
                next = eng.greedy_tok(4);
            }
            stats.accepted += a;
            eng.st.cur_slot = kids[a] as usize;
            seq.n_past += 1 + a;
            catch = [d1, d2][..a].to_vec();
            last_row = Some(a);
            if stop {
                break;
            }
            id_last = next;
            out.push(id_last);
            stop = is_eos(id_last) || !on_tok(id_last);
            continue;
        }
        if !use_mtp {
            let cur = eng.st.cur_slot as i32;
            eng.trunk(&[id_last], seq.n_past, &[cur], 1)?;
            seq.n_past += 1;
            last_row = Some(0);
            let d = sample::dist(eng.logits_row(0), &p.samp, None);
            id_last = sample::sample(&d, &mut rng);
            out.push(id_last);
            stop = is_eos(id_last) || !on_tok(id_last);
            continue;
        }
        // --- draft step 1: catch-up rows + id_last (MTP pos = token pos)
        let mut mt = catch.clone();
        mt.push(id_last);
        let mpos = seq.n_past - catch.len();
        if first_round {
            eng.put_hin_row(0, &seq.h_prev)?;
            first_round = false;
        } else {
            for j in 0..mt.len() {
                eng.set_hin(j, Some((j, false)))?; // trunk h rows of the last verify window
            }
        }
        eng.mtp(&mt, mpos, true)?;
        let ids = std::mem::take(&mut eng.dv_ids);
        let dvi = if ids.is_empty() { None } else { Some(&ids[..]) };
        let q1 = sample::dist(eng.dlogits_row(0), &q_samp, dvi);
        let d1 = sample::sample(&q1, &mut rng);
        // --- draft step 2
        eng.set_hin(0, Some((mt.len() - 1, true)))?;
        eng.mtp(&[d1], seq.n_past + 1, true)?;
        let q2 = sample::dist(eng.dlogits_row(0), &q_samp, dvi);
        let d2 = sample::sample(&q2, &mut rng);
        eng.dv_ids = ids;
        stats.drafted += 2;
        // --- verify
        let cur = eng.st.cur_slot as i32;
        let others: Vec<i32> = (0..3).filter(|&x| x != cur).collect();
        let kids = [cur, others[0], others[1]];
        eng.trunk(&[id_last, d1, d2], seq.n_past, &kids, 3)?;
        let mut a = 0usize;
        let mut next = 0u32;
        for (r, (dr, qr)) in [(d1, &q1), (d2, &q2)].into_iter().enumerate() {
            let pr = sample::dist(eng.logits_row(r), &p.samp, None);
            match sample::accept(&pr, qr, dr, greedy, &mut rng) {
                Ok(x) => {
                    a += 1;
                    out.push(x);
                    if is_eos(x) || !on_tok(x) || out.len() >= p.max_new {
                        stop = true;
                        break;
                    }
                }
                Err(y) => {
                    next = y;
                    break;
                }
            }
        }
        if a == 2 && !stop {
            let pr = sample::dist(eng.logits_row(2), &p.samp, None);
            next = sample::sample(&pr, &mut rng);
        }
        stats.accepted += a;
        // commit: id_last + a drafts processed; KDA state = snapshot after row a
        eng.st.cur_slot = kids[a] as usize;
        seq.n_past += 1 + a;
        catch = [d1, d2][..a].to_vec();
        last_row = Some(a);
        if stop {
            break;
        }
        id_last = next;
        out.push(id_last);
        stop = is_eos(id_last) || !on_tok(id_last);
    }
    if let Some(r) = last_row {
        seq.h_prev = eng.get_h_row(r, false)?;
    }
    if let Some(cp) = copy.as_mut() {
        let s = cp.take_stats();
        if s.windows > 0 {
            eprintln!(
                "talus copy: {} windows, {} of {} copied drafts accepted",
                s.windows, s.accepted, s.proposed
            );
        }
    }
    stats.decode_ms += t0.elapsed().as_secs_f64() * 1e3;
    stats.n_gen += out.len();
    stats.finish = if out.last().map(|&t| is_eos(t)).unwrap_or(false) {
        "stop".into()
    } else {
        "length".into()
    };
    Ok(out)
}
