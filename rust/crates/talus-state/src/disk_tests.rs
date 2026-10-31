//! CPU tests of the disk tier: serialization round trip, identity / geometry mismatch rejection, corrupt-file
//! handling, budget eviction, superseding, the background writer, and a Qwen3.5 end-to-end check that a restore
//! from disk leaves the engine state byte-identical to a RAM-tier hit (simulated HIP backend).

use crate::disk::*;
use crate::q36_cache::{q36_kv_bytes, q36_state_bytes, Q36LayerStateView, Q36PromptCache};
use crate::q36_disk;
use core::ffi::c_void;
use hip_sys::{
    hipFree, hipMalloc, hipMemcpy, hipMemcpyDeviceToHost, hipMemcpyHostToDevice, hipMemset,
    hipStreamCreateWithFlags, hipStreamDestroy, hipStreamNonBlocking, hipStream_t,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::{Duration, SystemTime};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let base = std::env::var_os("TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = base.join(format!(
            "talus-pcache-test-{tag}-{}-{n}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn cfg(dir: &Path, budget: u64) -> DiskCacheConfig {
    DiskCacheConfig {
        dir: dir.to_path_buf(),
        budget_bytes: budget,
        min_tokens: 1,
        mem_floor_bytes: 0,
        fs_floor: false,
    }
}

fn ident(s: &str) -> DiskIdentity {
    let mut b = IdentityBuilder::new();
    b.str("model", s);
    b.finish()
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn job(tokens: Vec<i32>, a: usize, seed: u8) -> WriteJob {
    WriteJob {
        tokens,
        meta: vec![1, 2, 3, seed],
        sections: vec![
            (tag(b"A"), pattern(a, seed)),
            (tag(b"B"), pattern(5000, seed ^ 0x55)),
        ],
        external: None,
        depth: 0,
        supersede: false,
    }
}

fn only_entry(c: &DiskCache) -> DiskEntry {
    let e = c.entries();
    assert_eq!(e.len(), 1);
    e[0].clone()
}

#[test]
fn xxh64_known_vectors_and_streaming() {
    assert_eq!(xxh64(b"", 0), 0xEF46_DB37_51D8_E999);
    assert_eq!(xxh64(b"a", 0), 0xD24E_C4F1_A98C_6E5B);
    assert_eq!(xxh64(b"abc", 0), 0x44BC_2CF5_AD77_0999);
    let data = pattern(1000, 7);
    let one = xxh64(&data, 42);
    for split in [1usize, 5, 31, 32, 33, 64, 999] {
        let mut h = Xxh64::new(42);
        for c in data.chunks(split) {
            h.update(c);
        }
        assert_eq!(h.digest(), one, "split {split}");
    }
}

#[test]
fn payload_hash_is_blockwise_and_length_sensitive() {
    let big = pattern(HASH_BLOCK * 2 + 123, 3);
    let h = payload_hash(&big);
    assert_eq!(h, payload_hash(&big.clone()));
    let mut other = big.clone();
    other[HASH_BLOCK + 5] ^= 1;
    assert_ne!(h, payload_hash(&other));
    assert_ne!(payload_hash(&big[..10]), payload_hash(&big[..11]));
    assert_ne!(payload_hash(&[]), payload_hash(&[0]));
}

#[test]
fn roundtrip_survives_reopen() {
    let d = TempDir::new("rt");
    let id = ident("m1");
    let toks: Vec<i32> = (0..3000).collect();
    {
        let c = DiskCache::open(cfg(&d.0, 1 << 30), id.clone()).unwrap();
        assert!(c.is_empty());
        let e = c.write_now(job(toks.clone(), HASH_BLOCK + 77, 9)).unwrap();
        assert_eq!(e.depth, 3000);
        assert!(fs::read_to_string(c.root().join("identity.txt"))
            .unwrap()
            .contains("model = m1"));
    }
    // A new process (server restart) sees the entry and reads back exactly what was written.
    let c = DiskCache::open(cfg(&d.0, 1 << 30), id).unwrap();
    let e = only_entry(&c);
    assert_eq!(*e.tokens, toks);
    assert_eq!(e.meta, vec![1, 2, 3, 9]);
    let l = c.load(&e).unwrap();
    assert_eq!(
        l.section(tag(b"A")).unwrap(),
        &pattern(HASH_BLOCK + 77, 9)[..]
    );
    assert_eq!(l.section(tag(b"B")).unwrap(), &pattern(5000, 9 ^ 0x55)[..]);
    assert!(l.external.is_none());
    assert_eq!(c.stats().hits.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn other_identity_is_never_read() {
    let d = TempDir::new("id");
    let a = DiskCache::open(cfg(&d.0, 1 << 30), ident("model-A")).unwrap();
    let ea = a.write_now(job(vec![1, 2, 3], 100, 1)).unwrap();
    // Another model / quant / numerics setting: its own folder, nothing visible.
    let b = DiskCache::open(cfg(&d.0, 1 << 30), ident("model-B")).unwrap();
    assert!(b.is_empty());
    assert_ne!(a.root(), b.root());
    // Even a file placed in the wrong folder is rejected by its header identity, and deleted.
    let stray = b.root().join(ea.path.file_name().unwrap());
    fs::copy(&ea.path, &stray).unwrap();
    b.rescan();
    assert!(b.is_empty());
    assert!(!stray.exists());
    // The identity hash covers every field.
    let mut x = IdentityBuilder::new();
    x.str("model", "m").u64("ctx", 1);
    let mut y = IdentityBuilder::new();
    y.str("model", "m").u64("ctx", 2);
    assert_ne!(x.finish().hash, y.finish().hash);
}

#[test]
fn identity_args_and_env_filters() {
    let args: Vec<String> = [
        "--model",
        "/m.gguf",
        "--prompt-cache-disk",
        "/x",
        "--ngram",
        "24",
        "--kv-stash-mib=4096",
        "--serve",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let kept = filter_args(&args, &["--prompt-cache-disk", "--kv-stash-mib", "--serve"]);
    assert_eq!(kept, vec!["--model", "/m.gguf", "--ngram", "24"]);
    let mut a = IdentityBuilder::new();
    a.args(&args, &["--prompt-cache-disk", "--kv-stash-mib", "--serve"]);
    let mut b = IdentityBuilder::new();
    b.args(&kept, &[]);
    assert_eq!(a.finish().hash, b.finish().hash);
}

#[test]
fn file_fingerprint_tracks_content() {
    let d = TempDir::new("fp");
    let p = d.0.join("w.gguf");
    fs::write(&p, pattern(3 << 20, 1)).unwrap();
    let a = file_fingerprint(&p).unwrap();
    let mut v = pattern(3 << 20, 1);
    v[(3 << 20) - 10] ^= 1; // the last MiB is sampled
    fs::write(&p, &v).unwrap();
    let f = fs::File::options().write(true).open(&p).unwrap();
    f.set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(1))
        .unwrap();
    let b = file_fingerprint(&p).unwrap();
    assert_ne!(a, b);
    let mut m = IdentityBuilder::new();
    m.file("model", &d.0.join("missing.gguf"));
    assert!(m.finish().desc.contains("[absent]"));
}

#[test]
fn corrupt_payload_is_detected_and_deleted() {
    let d = TempDir::new("corrupt");
    let id = ident("m");
    let c = DiskCache::open(cfg(&d.0, 1 << 30), id.clone()).unwrap();
    let e = c.write_now(job((0..500).collect(), 70_000, 4)).unwrap();
    // Flip one payload byte (index still loads: the head is intact).
    let mut raw = fs::read(&e.path).unwrap();
    let off = e.sections[0].offset as usize + 12_345;
    raw[off] ^= 0x80;
    fs::write(&e.path, &raw).unwrap();
    match c.load(&e) {
        Err(DiskError::Corrupt(_)) => {}
        other => panic!("expected Corrupt, got {other:?}"),
    }
    assert!(!e.path.exists());
    assert!(c.is_empty());
    assert_eq!(
        c.stats().corrupt.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn damaged_files_are_dropped_on_open() {
    let d = TempDir::new("damaged");
    let id = ident("m");
    let paths: Vec<PathBuf> = {
        let c = DiskCache::open(cfg(&d.0, 1 << 30), id.clone()).unwrap();
        (0..4)
            .map(|i| {
                c.write_now(job(vec![i, 1, 2, 3], 9000, i as u8))
                    .unwrap()
                    .path
            })
            .collect()
    };
    // 0: truncated (crash mid-write under the final name cannot happen, but a damaged disk can do this).
    let raw = fs::read(&paths[0]).unwrap();
    fs::write(&paths[0], &raw[..raw.len() - 100]).unwrap();
    // 1: head byte flipped (a token id).
    let mut raw = fs::read(&paths[1]).unwrap();
    raw[HEADER + 1] ^= 1;
    fs::write(&paths[1], &raw).unwrap();
    // 2: garbage.
    fs::write(&paths[2], b"not a cache file").unwrap();
    // A stale temporary from a dead writer, and an orphan payload.
    let root = paths[0].parent().unwrap().to_path_buf();
    let stale = root.join(".deadbeef-5.tpc.tmp.4000000");
    fs::write(&stale, b"partial").unwrap();
    let orphan = root.join("0000000000000000-9.tpcx");
    fs::write(&orphan, b"x").unwrap();

    let c = DiskCache::open(cfg(&d.0, 1 << 30), id).unwrap();
    let e = only_entry(&c);
    assert_eq!(e.path, paths[3]);
    for p in &paths[..3] {
        assert!(!p.exists(), "{p:?} should be deleted");
    }
    assert!(!stale.exists());
    assert!(!orphan.exists());
    assert!(c.load(&e).is_ok());
}

#[test]
fn replaced_file_is_not_trusted_from_the_index() {
    let d = TempDir::new("replaced");
    let c = DiskCache::open(cfg(&d.0, 1 << 30), ident("m")).unwrap();
    let e = c.write_now(job(vec![5, 6, 7], 100, 1)).unwrap();
    fs::remove_file(&e.path).unwrap();
    assert!(matches!(c.load(&e), Err(DiskError::Io(_))));
    assert!(c.is_empty());
}

fn set_age(p: &Path, secs_ago: u64) {
    let f = fs::File::options().write(true).open(p).unwrap();
    f.set_modified(SystemTime::now() - Duration::from_secs(secs_ago))
        .unwrap();
}

#[test]
fn budget_evicts_least_recently_used() {
    let d = TempDir::new("budget");
    let id = ident("m");
    // Each entry is a bit over 300 KiB; the budget holds two.
    let c = DiskCache::open(cfg(&d.0, 700 << 10), id).unwrap();
    let e1 = c.write_now(job(vec![1; 10], 300 << 10, 1)).unwrap();
    set_age(&e1.path, 300);
    let e2 = c.write_now(job(vec![2; 10], 300 << 10, 2)).unwrap();
    set_age(&e2.path, 200);
    // A hit refreshes e1, so e2 is now the least recently used.
    c.load(&e1).unwrap();
    let e3 = c.write_now(job(vec![3; 10], 300 << 10, 3)).unwrap();
    assert!(e1.path.exists());
    assert!(!e2.path.exists());
    assert!(e3.path.exists());
    assert_eq!(c.len(), 2);
    assert!(c.used_bytes() <= 700 << 10);
    assert_eq!(
        c.stats().evicted.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    // The budget spans identities: another model's old entries are evicted too.
    let other = DiskCache::open(cfg(&d.0, 700 << 10), ident("other")).unwrap();
    set_age(&e1.path, 100);
    set_age(&e3.path, 90);
    other.write_now(job(vec![4; 10], 300 << 10, 4)).unwrap();
    assert!(!e1.path.exists());
    assert!(e3.path.exists());
    // An entry larger than the budget is refused before any staging.
    assert!(matches!(c.can_stage(800 << 10), Err(DiskError::Budget(_))));
}

#[test]
fn supersede_keeps_the_last_two_turns_of_a_conversation() {
    let d = TempDir::new("supersede");
    let c = DiskCache::open(cfg(&d.0, 1 << 30), ident("m")).unwrap();
    let write = |tokens: Vec<i32>, depth: u64| {
        let mut j = job(tokens, 10, 1);
        j.depth = depth;
        j.supersede = true;
        c.write_now(j).unwrap()
    };
    let turn1: Vec<i32> = (0..100).collect();
    let e1 = write(turn1.clone(), 90);
    // Another conversation is never touched.
    let e_other = write((1000..1100).collect(), 100);
    // Turn 2 extends turn 1's restorable prefix: turn 1 stays (it is the previous turn).
    let mut turn2: Vec<i32> = turn1[..95].to_vec();
    turn2.extend(5000..5100);
    let e2 = write(turn2.clone(), 150);
    assert!(e1.path.exists() && e2.path.exists());
    // Turn 3: turn 2 is now the previous turn, turn 1 goes.
    let mut turn3 = turn2[..160].to_vec();
    turn3.extend(7000..7050);
    let e3 = write(turn3, 200);
    assert!(!e1.path.exists());
    assert!(e2.path.exists() && e3.path.exists() && e_other.path.exists());
    assert_eq!(c.len(), 3);
    // A branch from turn 1 (a regenerated turn-2 reply) does not supersede the deeper entries.
    let mut branch: Vec<i32> = turn1[..95].to_vec();
    branch.extend(9000..9010);
    write(branch, 100);
    assert!(e2.path.exists() && e3.path.exists());
}

#[test]
fn background_writer_and_busy_skip() {
    let d = TempDir::new("async");
    let c = DiskCache::open(cfg(&d.0, 1 << 30), ident("m")).unwrap();
    c.submit(job(vec![1, 2, 3], 4 << 20, 1)).unwrap();
    // While that write is in flight, a second one is skipped, not queued.
    let second = c.submit(job(vec![4, 5, 6], 10, 2));
    assert!(c.flush(Duration::from_secs(30)));
    assert!(!c.is_busy());
    match second {
        Ok(()) => assert_eq!(c.len(), 2), // the first write had already finished
        Err(DiskError::Busy) => assert_eq!(c.len(), 1),
        Err(e) => panic!("{e}"),
    }
    c.submit(job(vec![7, 8, 9], 10, 3)).unwrap();
    assert!(c.flush(Duration::from_secs(30)));
    assert!(c.entries().iter().any(|e| *e.tokens == vec![7, 8, 9]));
}

#[test]
fn external_payload_roundtrip_and_corruption() {
    let d = TempDir::new("ext");
    let c = DiskCache::open(cfg(&d.0, 1 << 30), ident("m")).unwrap();
    let tmp = native_tmp_path(&c, "laguna");
    fs::write(&tmp, pattern(100_000, 5)).unwrap();
    let e = c
        .write_now(WriteJob {
            tokens: vec![1, 2, 3, 4],
            meta: vec![],
            sections: vec![],
            external: Some(tmp.clone()),
            depth: 3,
            supersede: false,
        })
        .unwrap();
    assert!(!tmp.exists());
    assert!(e.has_external());
    assert_eq!(e.depth, 3);
    let l = c.load(&e).unwrap();
    let p = l.external.unwrap();
    assert_eq!(fs::read(&p).unwrap(), pattern(100_000, 5));
    // Corrupt the native payload: the entry is dropped with it.
    let mut raw = fs::read(&p).unwrap();
    raw[777] ^= 1;
    fs::write(&p, &raw).unwrap();
    assert!(matches!(c.load(&e), Err(DiskError::Corrupt(_))));
    assert!(!p.exists());
    assert!(!e.path.exists());
}

#[test]
fn config_parsing() {
    // default ON: unset / empty = the per-user folder, 32 GiB
    for v in [None, Some(""), Some("auto"), Some("1"), Some("on")] {
        let c = DiskCacheConfig::from_values(v, None, None).unwrap();
        assert!(c.dir.ends_with("talus/pcache"), "{v:?}");
        assert_eq!(c.budget_bytes, 32 << 30);
        assert!(c.fs_floor);
    }
    for v in ["0", "off", "OFF", "false", "no", "none"] {
        assert!(
            DiskCacheConfig::from_values(Some(v), None, None).is_none(),
            "{v}"
        );
    }
    assert!(DiskCacheConfig::from_values(None, Some("0"), None).is_none());
    assert_eq!(
        DiskCacheConfig::from_values(None, Some("8"), None)
            .unwrap()
            .budget_bytes,
        8 << 30
    );
    assert!(DiskCacheConfig::from_values(Some("/x"), Some("0"), None).is_none());
    let c = DiskCacheConfig::from_values(Some("/x/y"), Some("1.5"), Some("4096")).unwrap();
    assert_eq!(c.dir, PathBuf::from("/x/y"));
    assert_eq!(c.budget_bytes, 3 << 29);
    assert_eq!(c.min_tokens, 4096);
    let c = DiskCacheConfig::from_values(Some("auto"), None, None).unwrap();
    assert!(c.dir.ends_with("talus/pcache"));
    assert_eq!(c.budget_bytes, DEFAULT_BUDGET_BYTES);
    assert_eq!(c.min_tokens, DEFAULT_MIN_TOKENS);
}

#[test]
fn q36_meta_roundtrip_and_rejects() {
    let m = q36_disk::Q36DiskMeta {
        state_bytes: 10,
        kv_bytes: 20,
        pf: 30,
        snaps: vec![(30, false), (12, true)],
    };
    assert_eq!(q36_disk::Q36DiskMeta::decode(&m.encode()), Some(m.clone()));
    let mut bad = m.encode();
    bad[0] = 9; // format
    assert!(q36_disk::Q36DiskMeta::decode(&bad).is_none());
    let mut long = m.encode();
    long.push(0);
    assert!(q36_disk::Q36DiskMeta::decode(&long).is_none());
    assert_ne!(q36_disk::tag_state(0), q36_disk::tag_state(1));
}

// ---------------------------------------------------------------------------------------------------------------
// Qwen3.5 end to end on the simulated HIP backend
// ---------------------------------------------------------------------------------------------------------------

/// A tiny Q36 engine: 1 GDN layer + 1 attention layer, all buffers device-allocated.
struct MockQ36 {
    view: Q36LayerStateView,
    allocs: Vec<(*mut c_void, usize)>,
}

const CTX: i32 = 512;

impl MockQ36 {
    fn new() -> Self {
        unsafe {
            let mut st: hipStream_t = ptr::null_mut();
            assert!(hipStreamCreateWithFlags(&mut st, hipStreamNonBlocking).is_success());
            let mut allocs = Vec::new();
            let mut alloc = |n: usize| -> *mut c_void {
                let mut p: *mut c_void = ptr::null_mut();
                assert!(hipMalloc(&mut p, n).is_success());
                assert!(hipMemset(p, 0, n).is_success());
                allocs.push((p, n));
                p
            };
            let (hkv, hd) = (2usize, 8usize);
            let conv0 = alloc(3 * 8 * 4).cast::<f32>();
            let s0 = alloc(2 * 4 * 4 * 4).cast::<f32>();
            let kc1 = alloc(hkv * CTX as usize * hd * 2);
            let vc1 = alloc(hkv * CTX as usize * hd * 2);
            let vtc1 = alloc(hkv * CTX as usize * hd * 2);
            let hlast = alloc(16 * 4).cast::<f32>();
            let gctl = alloc(16).cast::<i32>();
            let view = Q36LayerStateView {
                ssm_d_conv: 4,
                ssm_conv_channels: 8,
                ssm_v_heads: 2,
                ssm_state_size: 4,
                n_head_kv: hkv as i32,
                head_dim: hd as i32,
                n_embd: 16,
                ctx: CTX,
                vt_ctx: CTX,
                has_mtp: true,
                conv: vec![conv0, ptr::null_mut()],
                s: vec![s0, ptr::null_mut()],
                kc: vec![ptr::null_mut(), kc1],
                vc: vec![ptr::null_mut(), vc1],
                vtc: vec![ptr::null_mut(), vtc1],
                hlast,
                gctl,
                gdn_pending: 0,
                pos: 0,
                st,
                dfl_k_ring: Vec::new(),
                dfl_v_ring: Vec::new(),
                dfl_ring: 0,
                dfl_keep: 0,
                dfl_kv_dim: 0,
                dfl_block: 0,
                dctx_lo: 0,
                dctx_hi: 0,
            };
            Self { view, allocs }
        }
    }

    /// Deterministic "prefill": every buffer is a function of the token prefix it represents.
    fn prefill(&mut self, ids: &[i32]) {
        let n = ids.len() as u64;
        let seed = ids
            .iter()
            .fold(0x1234_5678u64, |h, &t| h.wrapping_mul(1_000_003) ^ t as u64);
        for (i, &(p, len)) in self.allocs.iter().enumerate() {
            let host: Vec<u8> = (0..len)
                .map(|k| (seed.wrapping_add((i * 7919 + k) as u64 * 2654435761) >> 13) as u8)
                .collect();
            unsafe {
                assert!(
                    hipMemcpy(p, host.as_ptr().cast(), len, hipMemcpyHostToDevice).is_success()
                );
            }
        }
        self.view.pos = n as i64;
        self.view.gdn_pending = (n % 5) as i32;
    }

    fn dump(&self) -> Vec<Vec<u8>> {
        self.allocs
            .iter()
            .map(|&(p, len)| {
                let mut h = vec![0u8; len];
                unsafe {
                    assert!(
                        hipMemcpy(h.as_mut_ptr().cast(), p, len, hipMemcpyDeviceToHost)
                            .is_success()
                    );
                }
                h
            })
            .collect()
    }
}

impl Drop for MockQ36 {
    fn drop(&mut self) {
        unsafe {
            for &(p, _) in &self.allocs {
                let _ = hipFree(p);
            }
            let _ = hipStreamDestroy(self.view.st);
        }
    }
}

/// Turn 1 of a conversation as the 27B serve loop runs it with the grid-aligned policy: prefill to the grid end
/// (snapshot), read the tail, decode, then stash the live KV at the end of the turn.
fn run_turn1(
    eng: &mut MockQ36,
    pc: &mut Q36PromptCache,
    prompt: &[i32],
    g_end: usize,
    decoded: &[i32],
) -> Vec<i32> {
    eng.prefill(&prompt[..g_end]);
    pc.save(&mut eng.view, &prompt[..g_end], false);
    eng.prefill(prompt);
    let mut live = prompt.to_vec();
    live.extend_from_slice(decoded);
    eng.prefill(&live);
    pc.stash_kv_with_pf(&eng.view, &live, 0, prompt.len());
    live
}

#[test]
fn q36_disk_restore_is_byte_identical_to_ram_hit() {
    let d = TempDir::new("q36");
    let id = ident("qwen35-mock");
    let prompt: Vec<i32> = (0..300).map(|i| 1000 + i).collect();
    let g_end = 288; // grid-aligned prompt end
    let decoded: Vec<i32> = (0..20).map(|i| 9000 + i).collect();
    let mut turn2 = prompt.clone();
    turn2.extend(77..120); // the client re-sends the prompt with a new message (reply re-rendered differently)

    // Reference: one process, turn 2 restores from the RAM tier.
    let mut eng_a = MockQ36::new();
    let mut pc_a = Q36PromptCache::new();
    pc_a.stash_budget = 1 << 30;
    let live_a = run_turn1(&mut eng_a, &mut pc_a, &prompt, g_end, &decoded);

    // The same turn 1 in a process that writes the disk tier, then "restarts".
    {
        let mut eng = MockQ36::new();
        let mut pc = Q36PromptCache::new();
        pc.stash_budget = 1 << 30;
        let live = run_turn1(&mut eng, &mut pc, &prompt, g_end, &decoded);
        assert_eq!(live, live_a);
        let c = DiskCache::open(cfg(&d.0, 1 << 30), id.clone()).unwrap();
        let plan = q36_disk::plan_export(&pc, &live, 1).unwrap();
        assert_eq!(plan.depth, g_end);
        c.can_stage(plan.bytes).unwrap();
        assert!(
            q36_disk::export(&pc, &eng.view, &plan, &|| true).is_none(),
            "abort ignored"
        );
        let j = q36_disk::export(&pc, &eng.view, &plan, &|| false).unwrap();
        assert_eq!(j.payload_bytes(), plan.bytes);
        c.submit(j).unwrap();
        assert!(c.flush(Duration::from_secs(30)));
        // Too short for the configured minimum: nothing to write.
        assert!(q36_disk::plan_export(&pc, &live, g_end + 1).is_none());
    }

    let ram_n = {
        // Engine A goes cold (as after a request on another conversation), then turn 2 hits the RAM tier.
        for &(p, len) in &eng_a.allocs {
            unsafe {
                assert!(hipMemset(p, 0, len).is_success());
            }
        }
        eng_a.view.pos = 0;
        eng_a.view.gdn_pending = 0;
        let (n, from) = pc_a.restore_with_pf(&mut eng_a.view, &turn2, &[], 0, true);
        assert_eq!(from, "snapshot + KV stash");
        n
    };
    assert_eq!(ram_n, g_end);

    // After the restart: empty RAM tier, the disk tier has the entry.
    let mut eng_b = MockQ36::new();
    let mut pc_b = Q36PromptCache::new();
    pc_b.stash_budget = 1 << 30;
    pc_b.snap_bytes = q36_state_bytes(&eng_b.view);
    let c = DiskCache::open(cfg(&d.0, 1 << 30), id).unwrap();
    assert_eq!(pc_b.peek_with_pf(&turn2, &[], 0, true), 0);
    let (e, n) = c
        .best(|e| q36_disk::usable(e, &turn2, true, &pc_b, &eng_b.view))
        .unwrap();
    assert_eq!(n, g_end);
    let l = c.load(&e).unwrap();
    assert!(q36_disk::import(&mut pc_b, &eng_b.view, &l));
    assert_eq!(pc_b.peek_with_pf(&turn2, &[], 0, true), g_end);
    let (n_b, from_b) = pc_b.restore_with_pf(&mut eng_b.view, &turn2, &[], 0, true);
    assert_eq!((n_b, from_b), (ram_n, "snapshot + KV stash"));
    assert_eq!(eng_b.view.pos, eng_a.view.pos);
    assert_eq!(eng_b.view.gdn_pending, eng_a.view.gdn_pending);
    // Recurrent state, conv state, MTP hidden, gctl and every K/V/V^T cell: byte-identical (both engines
    // started from zeroed buffers, so cells past the restored prefix must match too).
    let (a, b) = (eng_a.dump(), eng_b.dump());
    assert!(a.iter().any(|x| x.iter().any(|&v| v != 0)));
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(x == y, "buffer {i} differs");
    }

    // A request that does not extend the conversation cannot use the entry.
    let unrelated: Vec<i32> = (0..400).collect();
    assert!(c
        .best(|e| q36_disk::usable(e, &unrelated, true, &pc_b, &eng_b.view))
        .is_none());
    // An engine with another geometry (here: a different KV head count) rejects it.
    let mut other = MockQ36::new();
    other.view.n_head_kv = 1;
    assert_eq!(q36_disk::usable(&e, &turn2, true, &pc_b, &other.view), 0);
    // A RAM stash budget too small for the entry makes it unusable instead of failing the import.
    let mut small = Q36PromptCache::new();
    small.stash_budget = q36_kv_bytes(&eng_b.view, e.tokens.len() as i32, 0) - 1;
    assert_eq!(q36_disk::usable(&e, &turn2, true, &small, &eng_b.view), 0);
}

#[test]
fn native_saver_temporaries_carry_the_writer_pid() {
    assert_eq!(tmp_pid(".deadbeef-5.tpc.tmp.4000000"), Some(4000000));
    assert_eq!(tmp_pid(".native-0.native.tmp.802240"), Some(802240));
    // the engine's own session saver appends `.tmp` to the path it is given
    assert_eq!(tmp_pid(".native-0.native.tmp.802240.tmp"), Some(802240));
    assert_eq!(tmp_pid(".x.tmp"), None);

    let d = TempDir::new("nativetmp");
    let id = ident("m");
    let root = d.0.join(format!("{:016x}", id.hash));
    fs::create_dir_all(&root).unwrap();
    // left by a killed engine (dead pid): deleted at open, however young
    let dead = root.join(".native-0.native.tmp.4000000.tmp");
    fs::write(&dead, b"partial").unwrap();
    // a live writer's (pid 1 exists): kept
    let live = root.join(".native-1.native.tmp.1.tmp");
    fs::write(&live, b"in flight").unwrap();
    let c = DiskCache::open(cfg(&d.0, 1 << 30), id).unwrap();
    assert!(c.is_empty());
    assert!(!dead.exists(), "dead writer's temporary left behind");
    assert!(live.exists(), "live writer's temporary deleted");
}

/// Child side of `term_signal_waits_for_the_turn_write` (a separate process: signal dispositions are global).
fn term_child(dir: &Path, mode: &str) {
    use std::io::Write as _;
    let c = DiskCache::open(cfg(dir, 1 << 30), ident("term")).unwrap();
    install_term_handler(Duration::from_secs(30));
    let guard = (mode == "save").then(SaveGuard::new);
    println!("READY");
    std::io::stdout().flush().unwrap();
    if guard.is_some() {
        // The stop signal arrives while the turn is still being staged (as right after DONE).
        std::thread::sleep(Duration::from_millis(400));
        c.submit(job(vec![1, 2, 3, 4], 48 << 20, 7)).unwrap();
        drop(guard);
    }
    std::thread::sleep(Duration::from_secs(30));
    std::process::exit(3);
}

#[test]
fn term_signal_waits_for_the_turn_write() {
    use std::io::BufRead as _;
    use std::process::{Command, Stdio};
    if let (Some(dir), Ok(mode)) = (
        std::env::var_os("TALUS_PCACHE_TERM_CHILD"),
        std::env::var("TALUS_PCACHE_TERM_MODE"),
    ) {
        term_child(Path::new(&dir), &mode);
        return;
    }
    let d = TempDir::new("term");
    for mode in ["idle", "save"] {
        let t0 = std::time::Instant::now();
        let mut ch = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "disk_tests::term_signal_waits_for_the_turn_write",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("TALUS_PCACHE_TERM_CHILD", &d.0)
            .env("TALUS_PCACHE_TERM_MODE", mode)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut out = std::io::BufReader::new(ch.stdout.take().unwrap());
        let mut line = String::new();
        loop {
            line.clear();
            assert!(
                out.read_line(&mut line).unwrap() > 0,
                "child ended before READY"
            );
            if line.trim_end().ends_with("READY") {
                break;
            }
        }
        let st = Command::new("kill")
            .args(["-TERM", &ch.id().to_string()])
            .status()
            .unwrap();
        assert!(st.success());
        let code = ch.wait().unwrap().code();
        assert_eq!(code, Some(128 + 15), "{mode}: exit status");
        assert!(
            t0.elapsed() < Duration::from_secs(20),
            "{mode}: exit took too long"
        );
    }
    // the write of the "save" child landed (complete and verified), nothing temporary is left
    let c = DiskCache::open(cfg(&d.0, 1 << 30), ident("term")).unwrap();
    let e = only_entry(&c);
    assert_eq!(e.tokens.as_slice(), &[1, 2, 3, 4]);
    let l = c.load(&e).unwrap();
    assert_eq!(l.section(tag(b"A")).unwrap(), &pattern(48 << 20, 7)[..]);
    let tmp: Vec<_> = fs::read_dir(c.root())
        .unwrap()
        .flatten()
        .filter(|x| x.file_name().to_string_lossy().contains(".tmp"))
        .collect();
    assert!(tmp.is_empty(), "{tmp:?}");
}

#[test]
fn open_drops_misnamed_entries_and_enforces_the_budget() {
    let d = TempDir::new("openbudget");
    let id = ident("m");
    let paths: Vec<PathBuf> = {
        let c = DiskCache::open(cfg(&d.0, 1 << 30), id.clone()).unwrap();
        (1..=3)
            .map(|k| {
                let e = c.write_now(job(vec![k; 10], 300 << 10, k as u8)).unwrap();
                set_age(&e.path, 400 - 100 * k as u64);
                e.path
            })
            .collect()
    };
    // a valid entry copied under a wrong name, and one planted in a foreign identity folder
    let root = paths[0].parent().unwrap().to_path_buf();
    let misnamed = root.join("ffffffffffffffff-1.tpc");
    fs::copy(&paths[2], &misnamed).unwrap();
    let foreign_dir = d.0.join("00000000deadbeef");
    fs::create_dir_all(&foreign_dir).unwrap();
    let foreign = foreign_dir.join(paths[2].file_name().unwrap());
    fs::copy(&paths[2], &foreign).unwrap();
    set_age(&foreign, 1000);
    // reopen with a budget that holds two entries: misnamed dropped, then the oldest files (foreign, entry 1) go
    let c = DiskCache::open(cfg(&d.0, 700 << 10), id).unwrap();
    assert!(!misnamed.exists());
    assert!(!foreign.exists());
    assert!(!paths[0].exists());
    assert!(paths[1].exists() && paths[2].exists());
    assert_eq!(c.len(), 2);
}

#[test]
fn filesystem_floor_refuses_growth() {
    let d = TempDir::new("fsfloor");
    let (total, avail) = fs_space(&d.0).expect("statvfs");
    assert!(total > 0 && avail <= total);
    assert_eq!(fs_floor(1 << 40), (1u64 << 40) / 10);
    assert_eq!(fs_floor(50 << 30), FS_FLOOR_MIN_BYTES);
    let mut c = cfg(&d.0, 1 << 40);
    c.fs_floor = true;
    let c = DiskCache::open(c, ident("m")).unwrap();
    // a write that would eat all the free space is refused; a tiny one passes when the disk has room
    assert!(matches!(c.can_stage(avail), Err(DiskError::Budget(_))));
    if avail > fs_floor(total) + (1 << 30) {
        c.space_ok(1 << 20).unwrap();
    }
}
