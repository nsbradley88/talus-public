//! Host-only tests of the conversation state: policies, buffers, and save / restore / file round trips through a
//! fake device whose "device" buffers are host memory.

use super::file::*;
use super::*;

unsafe extern "C" fn fake_layout(ctx: *mut TkQwen4ExpServeCtx, out: *mut P5Layout) -> i32 {
    *out = *(ctx as *const P5Layout);
    0
}
unsafe extern "C" fn fake_copy(
    dst: *mut u8,
    src: *const u8,
    n: usize,
    _e: *mut c_char,
    _c: usize,
) -> i32 {
    std::ptr::copy_nonoverlapping(src, dst, n);
    0
}
unsafe extern "C" fn fake_sync(_e: *mut c_char, _c: usize) -> i32 {
    0
}
unsafe extern "C" fn fake_res(
    _c: *mut TkQwen4ExpServeCtx,
    _w: i32,
    _u: i64,
    _e: *mut c_char,
    _n: usize,
) -> i32 {
    0
}
unsafe extern "C" fn fake_prev(_c: *mut TkQwen4ExpServeCtx, _a: i32, _b: i32) {}

/// A small session: 2 GDN layers, 2 QSA layers, int8 K/V, everything in host vectors.
struct Fake {
    _bufs: Vec<Vec<u8>>,
    layout: Box<P5Layout>,
}

impl Fake {
    fn new(seed: u8) -> Self {
        let mut bufs: Vec<Vec<u8>> = Vec::new();
        let mut alloc = |n: usize, k: u8| -> *mut u8 {
            let mut v: Vec<u8> = (0..n)
                .map(|i| (i as u8).wrapping_mul(31).wrapping_add(k))
                .collect();
            let p = v.as_mut_ptr();
            bufs.push(v);
            p
        };
        let mut l = P5Layout {
            geometry: [64, 4, 2, 4, 1, 2, 4, 32, 8, 2, 1, 64, 1, 16, 4, 4, 8, 16],
            layer_lo: 0,
            layer_hi: 4,
            n_gdn_layers: 2,
            n_qsa_layers: 2,
            gdn_alloc: 2,
            qsa_ord0: 0,
            qsa_alloc: 2,
            max_cells: 256,
            ssm_state_size: 4,
            ssm_v_heads: 2,
            ssm_conv_channels: 32,
            ssm_d_conv: 4,
            idx_key_dim: 16,
            n_head_kv: 1,
            head_dim: 64,
            ng_hist: 2,
            ng_hc_dim: 8,
            idx_block: 4,
            page_size: 16,
            kv_q4_bytes_per_head: 36,
            has_draft: 1,
            n_qsa: 2,
            ..Default::default()
        };
        let z = session_sizes(&l).unwrap();
        l.gdn_state = alloc(z.gdn, seed);
        l.ple_hist = alloc(z.ple, seed + 1);
        let mk = |alloc: &mut dyn FnMut(usize, u8) -> *mut u8, k: u8, index: bool| {
            let cells = 256usize;
            let mut st = P5Qsa {
                max_cells: 256,
                n_pages: 16,
                n_slots: 16,
                idx_pooled_rows: if index { 66 } else { 2 },
                kv_mode: 0,
                kv_format: 1,
                kv_int8: 1,
                ..Default::default()
            };
            st.pools[0] = alloc(cells * 64, k);
            st.pools[1] = alloc(cells * 64, k + 1);
            st.pools[2] = alloc(cells * 2, k + 2);
            st.pools[3] = alloc(cells * 2, k + 3);
            st.pools[4] = alloc(66 * 16 * 4, k + 4);
            st.idx_pooled = st.pools[4];
            st.idx_tail = alloc(3 * 16 * 4, k + 5);
            st.idx_dead = alloc(16 * 4, k + 6);
            st.idx_block_pos = alloc(4, k + 7);
            st
        };
        l.qsa[0] = mk(&mut alloc, seed + 10, true);
        l.qsa[1] = mk(&mut alloc, seed + 20, true);
        l.draft = mk(&mut alloc, seed + 30, false);
        Self {
            _bufs: bufs,
            layout: Box::new(l),
        }
    }
    fn dev(&self) -> P5Device {
        P5Device {
            ctx: (&*self.layout as *const P5Layout) as *mut TkQwen4ExpServeCtx,
            layout: fake_layout,
            copy: fake_copy,
            sync: fake_sync,
            kv_residency: fake_res,
            set_ple_prev: fake_prev,
        }
    }
}

#[test]
fn eviction_victim_matches_conv_cache() {
    // plain: root pinned, least recent of the rest
    assert_eq!(eviction_victim(&[1, 5, 3, 4], 3, None, None), 2);
    // cap < 2: the oldest (index 0) leaves
    assert_eq!(eviction_victim(&[1, 5, 3], 1, None, None), 0);
    // a stale tail (not the newest) leaves first
    assert_eq!(
        eviction_victim(&[1, 5, 3, 9], 3, Some(&[false, true, false, false]), None),
        1
    );
    // the newest tail stays
    assert_eq!(
        eviction_victim(&[1, 5, 3, 9], 3, Some(&[false, false, false, true]), None),
        2
    );
    // pins: never the pinned one
    assert_eq!(
        eviction_victim(
            &[1, 2, 3, 9],
            3,
            Some(&[false; 4]),
            Some(&[false, true, false, false])
        ),
        2
    );
    // all pinned: the oldest non-root
    assert_eq!(
        eviction_victim(&[1, 4, 3], 2, Some(&[false; 3]), Some(&[true, true, true])),
        2
    );
}

#[test]
fn conv_buffer_segments_and_visit() {
    let mut b = ConvBuffer::default();
    b.resize(ConvBuffer::SEGMENT + 10);
    assert_eq!(b.size(), ConvBuffer::SEGMENT + 10);
    let mut seen = 0;
    assert!(b.visit_mut(5, ConvBuffer::SEGMENT, |p, at| {
        for (i, x) in p.iter_mut().enumerate() {
            *x = ((at + i) % 251) as u8;
        }
        seen += p.len();
        true
    }));
    assert_eq!(seen, ConvBuffer::SEGMENT);
    let mut chk = true;
    b.visit(5, ConvBuffer::SEGMENT, |p, at| {
        chk &= p
            .iter()
            .enumerate()
            .all(|(i, &x)| x == ((at + i) % 251) as u8);
        true
    });
    assert!(chk);
    b.resize(3);
    assert_eq!(b.size(), 3);
    assert!(!b.visit(2, 2, |_, _| true));
    assert!(b.allocation_peak(1 << 20) >= 1 << 20);
}

#[test]
fn checkpoint_save_restore_round_trip() {
    let a = Fake::new(1);
    let b = Fake::new(77);
    let (da, db) = (a.dev(), b.dev());
    let mut c = Checkpoint {
        ids: (0..21).collect(),
        ..Default::default()
    };
    checkpoint_save(&mut c, &da, &a.layout).unwrap();
    checkpoint_validate(&c, &a.layout).unwrap();
    checkpoint_restore(&c, &db, &b.layout).unwrap();
    let mut c2 = Checkpoint {
        ids: c.ids.clone(),
        ..Default::default()
    };
    checkpoint_save(&mut c2, &db, &b.layout).unwrap();
    assert_eq!(c.gdn, c2.gdn);
    assert_eq!(c.tails, c2.tails);
    assert_eq!(c.dead, c2.dead);
    let bad = Checkpoint {
        gdn: vec![0; 3],
        ..c.clone()
    };
    assert!(checkpoint_validate(&bad, &a.layout).is_err());
}

#[test]
fn snapshot_and_session_file_round_trip() {
    let a = Fake::new(3);
    let da = a.dev();
    let live: Vec<i32> = (0..40).collect();
    let mut ck = Checkpoint {
        ids: live[..20].to_vec(),
        used: 7,
        ..Default::default()
    };
    checkpoint_save(&mut ck, &da, &a.layout).unwrap();
    let checks = vec![ck];
    let view = ConvView {
        ids: &live,
        images: &[],
        checkpoints: &checks,
        cvec: true,
    };
    let mut reused = 0usize;
    let img = snapshot_save(
        &view,
        &a.layout,
        &da,
        Some(&a.layout.draft),
        KvReuse::default(),
        &mut reused,
    )
    .unwrap();
    snapshot_validate(&img, &a.layout, Some(&a.layout.draft)).unwrap();
    let est = snapshot_bytes(&view, &a.layout, Some(&a.layout.draft)).unwrap();
    assert!(est > 0);

    let dir = std::env::temp_dir().join(format!("p5-sess-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let id = SessionFileIdentity {
        model: 11,
        config: 22,
    };
    // from an in-memory image and from device sources: the same bytes
    let p1 = dir.join("a.bin");
    let mut st = SessionStatus::default();
    let n1 = session_file_write(
        p1.to_str().unwrap(),
        &img,
        None,
        &id,
        Default::default(),
        &mut st,
    )
    .unwrap();
    let (meta, mut src) = snapshot_sources(&view, &a.layout, &da).unwrap();
    let p2 = dir.join("b.bin");
    let n2 = session_file_write(
        p2.to_str().unwrap(),
        &meta,
        Some(&mut src),
        &id,
        Default::default(),
        &mut st,
    )
    .unwrap();
    assert_eq!(n1, n2);
    assert_eq!(std::fs::read(&p1).unwrap(), std::fs::read(&p2).unwrap());

    let mut lim = session_read_limits(&a.layout, 256, 8).unwrap();
    let (back, size) = session_file_read(p1.to_str().unwrap(), &id, &mut lim, &mut st).unwrap();
    assert_eq!(size, n1);
    assert_eq!(back, img);
    // restore into another session and capture again: identical
    let b = Fake::new(99);
    let db = b.dev();
    let (o, e) = snapshot_restore(&back, &b.layout, &db, Some(&b.layout.draft));
    assert_eq!(o, RestoreOutcome::Restored, "{e}");
    let img2 = snapshot_save(
        &view,
        &b.layout,
        &db,
        Some(&b.layout.draft),
        KvReuse::default(),
        &mut reused,
    )
    .unwrap();
    // (the indexer's spare row is rewritten from `dead` by the restore, as in C++: K/V and running state compared)
    assert_eq!(img2.live, img.live);
    for (x, y) in img2.kv.iter().zip(img.kv.iter()) {
        assert_eq!(
            (&x.k, &x.v, &x.k_scale, &x.v_scale),
            (&y.k, &y.v, &y.k_scale, &y.v_scale)
        );
    }
    // identity mismatch and corruption are refused
    let mut lim = session_read_limits(&a.layout, 256, 8).unwrap();
    let wrong = SessionFileIdentity {
        model: 11,
        config: 23,
    };
    assert!(session_file_read(p1.to_str().unwrap(), &wrong, &mut lim, &mut st).is_err());
    let mut bytes = std::fs::read(&p1).unwrap();
    let k = bytes.len() / 2;
    bytes[k] ^= 1;
    std::fs::write(&p1, &bytes).unwrap();
    let mut lim = session_read_limits(&a.layout, 256, 8).unwrap();
    let r = session_file_read(p1.to_str().unwrap(), &id, &mut lim, &mut st);
    assert!(r.is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn cache_put_best_take_and_superseded() {
    let a = Fake::new(5);
    let da = a.dev();
    let mk = |n: i32, ck: usize| {
        let live: Vec<i32> = (0..n).collect();
        let mut c = Checkpoint {
            ids: live[..ck].to_vec(),
            ..Default::default()
        };
        checkpoint_save(&mut c, &da, &a.layout).unwrap();
        let checks = vec![c];
        let view = ConvView {
            ids: &live,
            images: &[],
            checkpoints: &checks,
            cvec: true,
        };
        let mut r = 0;
        snapshot_save(&view, &a.layout, &da, None, KvReuse::default(), &mut r).unwrap()
    };
    let mut cache = ConversationCache::new(1 << 30, 2, 0);
    assert!(cache.put(mk(40, 20), 0));
    let prompt: Vec<i64> = (0..60).collect();
    let m = cache.best(&prompt, &[], true);
    assert_eq!((m.tokens, m.live), (40, true));
    // a later turn of the same conversation supersedes it
    assert!(cache.put(mk(50, 20), 0));
    assert_eq!(cache.size(), 1);
    assert_eq!(cache.superseded(), 1);
    let got = cache.take(0);
    assert_eq!(got.live.ids.len(), 50);
    assert_eq!(cache.bytes(), 0);
}

#[test]
fn save_reclaim_keeps_selected_and_pins() {
    let mk = |n: usize, pinned: bool| Checkpoint {
        ids: vec![1; n],
        gdn: vec![0; 1000],
        pinned,
        ..Default::default()
    };
    let mut chain = vec![mk(10, false), mk(20, true), mk(30, false), mk(40, false)];
    let mut cache = ConversationCache::new(0, 0, 0);
    let live = SessionSaveLive {
        state_bytes: 1 << 20,
        tokens: 40,
        images: 0,
        kv_layers: 3,
    };
    let mut calls = 0;
    let freed = session_save_reclaim(&mut chain, &mut cache, &live, 0, || {
        calls += 1;
        Some(if calls < 4 { 1 } else { u64::MAX / 2 })
    });
    // the pinned (selected) one stays; checkpoints are dropped from the back until RAM admits
    assert!(chain.iter().any(|c| c.pinned));
    assert!(freed.checkpoints >= 1);
}
