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
    DiskCache, DiskCacheConfig, DiskIdentity, IdentityBuilder, ARGS_NOT_IN_IDENTITY,
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
    view: &Q36LayerStateView,
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
    if let Err(e) = disk.can_stage(plan.bytes) {
        eprintln!("talus-pcache: disk write skipped ({e})");
        return;
    }
    // The next request comes first: if it is already waiting (or arrives while the state is copied off the GPU),
    // this turn is not written; that request's own turn writes a deeper entry.
    let waiting = talus_sched::requests_waiting;
    if waiting() {
        eprintln!("talus-pcache: disk write skipped (next request waiting)");
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
