//! Disk tier of the prompt cache in the serve loops: Qwen3.5 / Qwen3.6 (27B) and Laguna.
//!
//! - After each reply (DONE already sent), the conversation's KV stash and its deepest prefill-produced
//!   snapshots are copied to host memory and written in the background ([`save_after_turn`]).
//! - When a request arrives and the disk holds a deeper restore point than the RAM tier, that entry is put back
//!   into the RAM tier first ([`restore_into_ram`]); the regular restore then runs unchanged, so a disk hit
//!   restores the same bytes through the same code as a RAM hit.

use std::path::Path;
use std::time::Instant;

use talus_state::disk::{
    DiskCache, DiskCacheConfig, DiskIdentity, IdentityBuilder, StageMode, ARGS_NOT_IN_IDENTITY,
};
use talus_state::{q36_disk, q36_kv_bytes, q36_state_bytes, Q36LayerStateView};

use super::{make_layer_state_view, CliConfig, Q36StdioBackend};

/// The disk settings from the flags, else the `TALUS_PCACHE_DISK*` environment.
pub(crate) fn disk_config(cfg: &CliConfig) -> Option<DiskCacheConfig> {
    let env = |k: &str| std::env::var(k).ok();
    let dir = cfg
        .prompt_cache_disk
        .clone()
        .or_else(|| env("TALUS_PCACHE_DISK"));
    let gib = cfg
        .prompt_cache_disk_gib
        .clone()
        .or_else(|| env("TALUS_PCACHE_DISK_GIB"));
    let min = cfg
        .prompt_cache_disk_min
        .clone()
        .or_else(|| env("TALUS_PCACHE_DISK_MIN"));
    DiskCacheConfig::from_values(dir.as_deref(), gib.as_deref(), min.as_deref())
}

fn identity(cfg: &CliConfig, view: &Q36LayerStateView) -> DiskIdentity {
    let mut b = IdentityBuilder::new();
    b.str("engine", "qwen35").u64("adapter", q36_disk::FORMAT);
    if let Some(m) = &cfg.model {
        let m = Path::new(m);
        b.file("model", m);
        // 4-bit prompt weights next to the GGUF (used when TALUS_Q27_W4I selects them).
        b.file("w4i", &m.with_extension("w4i"));
        b.file("w4i.fast", &m.with_extension("fast.w4i"));
    }
    for (k, p) in [
        ("mtp", &cfg.mtp_gguf),
        ("dflash", &cfg.dflash_gguf),
        ("draft_vocab", &cfg.draft_vocab),
    ] {
        if let Some(p) = p {
            b.file(k, Path::new(p));
        }
    }
    b.u64("state_bytes", q36_state_bytes(view) as u64)
        .u64("kv_bytes_256", q36_kv_bytes(view, 256, 0) as u64);
    b.process(ARGS_NOT_IN_IDENTITY);
    b.finish()
}

/// Open the disk tier when it is configured and the RAM tier is on (the disk tier extends it).
pub(crate) fn open(me: &mut Q36StdioBackend, cfg: &CliConfig) -> Option<DiskCache> {
    let dc = disk_config(cfg)?;
    if me.pc.max_snaps == 0 || me.pc.stash_budget == 0 {
        // on by default: only say so when it was asked for
        let explicit =
            cfg.prompt_cache_disk.is_some() || std::env::var_os("TALUS_PCACHE_DISK").is_some();
        if explicit {
            eprintln!(
                "talus-pcache: disk tier needs --prompt-cache > 0 and --kv-stash-mib > 0; off"
            );
        }
        return None;
    }
    let view = make_layer_state_view(&mut me.eng);
    open_with(dc, identity(cfg, &view))
}

/// Open a disk tier and log what it holds.
pub(crate) fn open_with(dc: DiskCacheConfig, id: DiskIdentity) -> Option<DiskCache> {
    let t0 = Instant::now();
    match DiskCache::open(dc, id) {
        Ok(c) => {
            // A stop (SIGTERM to the process group) right after a reply must not lose that reply's write.
            talus_state::disk::install_term_handler(std::time::Duration::from_secs(60));
            eprintln!(
                "talus-pcache: disk tier {} ({} entries, {:.1} of {:.1} GiB), identity {:016x}, opened in {:.0} ms",
                c.root().display(),
                c.len(),
                c.used_bytes() as f64 / (1u64 << 30) as f64,
                c.config().budget_bytes as f64 / (1u64 << 30) as f64,
                c.identity().hash,
                t0.elapsed().as_secs_f64() * 1000.0
            );
            Some(c)
        }
        Err(e) => {
            eprintln!("talus-pcache: disk tier off ({e})");
            None
        }
    }
}

/// Before the RAM restore: if the disk holds a deeper restore point for `ids`, load it into the RAM tier.
pub(crate) fn restore_into_ram(
    me: &mut Q36StdioBackend,
    view: &mut Q36LayerStateView,
    ids: &[i32],
    gridc: bool,
) {
    let Q36StdioBackend {
        disk,
        pc,
        live,
        live_pf,
        ..
    } = me;
    let Some(disk) = disk.as_ref() else {
        return;
    };
    let (pf, pf_only) = if gridc {
        (*live_pf, true)
    } else {
        (usize::MAX, false)
    };
    let ram_n = pc.peek_with_pf(ids, live, pf, pf_only);
    let Some((e, n)) = disk.best(|e| q36_disk::usable(e, ids, pf_only, pc, view)) else {
        return;
    };
    if n <= ram_n {
        return;
    }
    let t0 = Instant::now();
    let kv_need = talus_state::q36_kv_bytes(view, e.tokens.len() as i32, 0) as u64;
    let avail = talus_state::disk::mem_available_effective();
    let no_room = pc.acct_peer.is_none()
        && pc.mem_floor > 0
        && avail.is_some_and(|a| a < kv_need + pc.mem_floor);
    // the staged load holds the whole entry in host memory while the stash is allocated: stream when both do not
    // fit above the stash floor (found on the desktop: the staged read left too little for the stash, the import failed)
    let stream_stash = disk.should_stream_load(&e)
        || (pc.mem_floor > 0 && avail.is_some_and(|a| a < e.bytes + kv_need + pc.mem_floor));
    if no_room {
        // : no room for a RAM-tier stash (64 GB at 128K): load the entry straight into the live KV and make it
        // the live sequence; the regular restore then resumes from its snapshot without a KV copy
        match q36_disk::load_direct(pc, view, disk, &e) {
            Ok((ok, pf, bytes, read_ms)) => {
                live.clear();
                *live_pf = 0;
                if ok {
                    live.extend_from_slice(&e.tokens);
                    *live_pf = pf.min(live.len());
                }
                eprintln!(
                    "talus-pcache: disk hit {n} tokens (RAM tier had {ram_n}), {:.1} MiB streamed into the live KV in \
                     {:.1} ms (no room for a stash), {} in {:.1} ms",
                    bytes as f64 / (1u64 << 20) as f64,
                    read_ms,
                    if ok { "restored" } else { "snapshot import FAILED" },
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            Err(err) => {
                live.clear();
                *live_pf = 0;
                eprintln!("talus-pcache: disk entry skipped ({err}); the KV cells were partly overwritten: cold start");
            }
        }
        return;
    }
    if stream_stash {
        // : the KV stash goes from the file to its device buffer one chunk at a time (no host copy)
        match q36_disk::load_import_streamed(pc, view, disk, &e) {
            Ok((ok, bytes, read_ms)) => eprintln!(
                "talus-pcache: disk hit {n} tokens (RAM tier had {ram_n}), {:.1} MiB streamed in {:.1} ms, {} in {:.1} ms",
                bytes as f64 / (1u64 << 20) as f64,
                read_ms,
                if ok { "restored to the RAM tier" } else { "import FAILED" },
                t0.elapsed().as_secs_f64() * 1000.0
            ),
            Err(err) => eprintln!("talus-pcache: disk entry skipped ({err})"),
        }
        return;
    }
    match disk.load(&e) {
        Ok(l) => {
            let read_ms = l.ms;
            let ok = q36_disk::import(pc, view, &l);
            eprintln!(
                "talus-pcache: disk hit {n} tokens (RAM tier had {ram_n}), {:.1} MiB read in {:.1} ms, {} in {:.1} ms",
                l.bytes as f64 / (1u64 << 20) as f64,
                read_ms,
                if ok { "restored to the RAM tier" } else { "import FAILED" },
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Err(err) => eprintln!("talus-pcache: disk entry skipped ({err})"),
    }
}

/// After a reply: stash the conversation, then write its deepest restore point unless it is on disk already.
pub(crate) fn save_after_turn(me: &mut Q36StdioBackend) {
    let Some(disk) = me.disk.as_ref() else {
        return;
    };
    let min = disk.config().min_tokens;
    if me.live.len() < min.max(256) {
        return;
    }
    if disk.is_busy() {
        eprintln!("talus-pcache: previous disk write still running; this turn is not written");
        return;
    }
    let t0 = Instant::now();
    let view = make_layer_state_view(&mut me.eng);
    me.pc.stash_kv_with_pf(&view, &me.live, 0, me.live_pf);
    let Some(plan) = q36_disk::plan_export(&me.pc, &me.live, min) else {
        // : no stash of this conversation (it did not fit beside the floor): write from the live KV
        if me.pc.acct_peer.is_none() {
            save_live(me, &view, min, t0);
        }
        return;
    };
    let ids = &me.pc.stashes[plan.stash].ids;
    let d = plan.depth;
    let on_disk = disk
        .best(|e| {
            usize::from(e.depth as usize == d && e.tokens.len() >= d && e.tokens[..d] == ids[..d])
        })
        .is_some();
    if on_disk {
        return;
    }
    // : when the whole host copy would cross the memory floor (64 GB at 128K context), stream the entry
    let mode = match disk.stage_mode(plan.bytes, 0) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("talus-pcache: disk write skipped ({e})");
            return;
        }
    };
    // The next request comes first: if it is already waiting (or arrives while the state is copied off the GPU),
    // this turn is not written; that request's own turn writes a deeper entry.
    let waiting = talus_sched::requests_waiting;
    if waiting() {
        eprintln!("talus-pcache: disk write skipped (next request waiting)");
        return;
    }
    if mode == StageMode::Streamed {
        match q36_disk::export_streamed(&me.pc, &view, &plan, disk, &waiting) {
            Ok(e) => eprintln!(
                "talus-pcache: disk entry {} tokens, {:.1} MiB streamed in {:.0} ms (host staging {} MiB: \
                 the whole copy would leave less than the memory floor)",
                d,
                e.bytes as f64 / (1u64 << 20) as f64,
                t0.elapsed().as_secs_f64() * 1000.0,
                talus_state::disk::STREAM_CHUNK >> 20
            ),
            Err(e) => eprintln!("talus-pcache: streamed disk write skipped ({e})"),
        }
        return;
    }
    let Some(job) = q36_disk::export(&me.pc, &view, &plan, &waiting) else {
        eprintln!(
            "talus-pcache: disk write skipped ({})",
            if waiting() {
                "next request arrived while staging"
            } else {
                "copy off the GPU failed"
            }
        );
        return;
    };
    let stage_ms = t0.elapsed().as_secs_f64() * 1000.0;
    match disk.submit(job) {
        Ok(()) => eprintln!(
            "talus-pcache: staged {} tokens ({:.1} MiB) for disk in {stage_ms:.1} ms",
            d,
            plan.bytes as f64 / (1u64 << 20) as f64
        ),
        Err(e) => eprintln!("talus-pcache: disk write skipped ({e})"),
    }
}

/// The Laguna disk tier (engine-native session files, see `talus_model::laguna::serve_main`).
pub(crate) fn open_laguna(
    cfg: &CliConfig,
    model: &Path,
    dflash: Option<&str>,
    ring: usize,
    ctx16: usize,
) -> Option<DiskCache> {
    let dc = disk_config(cfg)?;
    let mut b = IdentityBuilder::new();
    b.str("engine", "laguna").u64("adapter", 1);
    b.file("model", model);
    if let Some(p) = dflash {
        b.file("dflash", Path::new(p));
    }
    b.u64("ring", ring as u64).u64("ctx16", ctx16 as u64);
    b.process(ARGS_NOT_IN_IDENTITY);
    open_with(dc, b.finish())
}

/// : [`save_after_turn`] without a RAM-tier stash: stream the entry from the live KV, one layer at a time.
fn save_live(me: &mut Q36StdioBackend, view: &Q36LayerStateView, min: usize, t0: Instant) {
    let Some(disk) = me.disk.as_ref() else {
        return;
    };
    let Some(plan) = q36_disk::plan_live(&me.pc, &me.live, me.live_pf, min) else {
        return;
    };
    let d = plan.depth;
    let live = &me.live;
    let on_disk = disk
        .best(|e| {
            usize::from(e.depth as usize == d && e.tokens.len() >= d && e.tokens[..d] == live[..d])
        })
        .is_some();
    if on_disk {
        return;
    }
    let waiting = talus_sched::requests_waiting;
    if waiting() {
        eprintln!("talus-pcache: disk write skipped (next request waiting)");
        return;
    }
    match q36_disk::export_streamed_live(&me.pc, view, live, me.live_pf, &plan, disk, &waiting) {
        Ok(e) => eprintln!(
            "talus-pcache: disk entry {} tokens, {:.1} MiB streamed from the live KV in {:.0} ms (no RAM-tier stash)",
            d,
            e.bytes as f64 / (1u64 << 20) as f64,
            t0.elapsed().as_secs_f64() * 1000.0
        ),
        Err(e) => eprintln!("talus-pcache: streamed disk write skipped ({e})"),
    }
}
