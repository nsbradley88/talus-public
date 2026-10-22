//! Piece 1 (`qwen4exp-p1`): Flash-Next (`qwen4exp`) weight loading, GGUF / native pack layout,
//! expert arena & VRAM cache (`ExpertCache`), resident budget sizing (`host_res` / `d_res`),
//! and PLE (`PleTable`) initialization.
//!
//! ## What lives here (Rust owns it under `TALUS_FN_RUST=1`)
//! - The pack loader (`src/core/weights.cpp` `WeightTable::pool_bytes` / `load`): `<pack>/index.txt` parsing, the
//!   skip-set compaction of the arena, the per-plane segment list, the bf16 / f16 / widened-scale conversions and the
//!   upload into the device arena. [`PackIndex`], [`WeightTable`].
//! - The C ABI hook table the C++ engine calls instead of its own code when the Rust data plane is on
//!   (`tk_qwen4exp_p1_set_rust_hooks`, `src/program/qwen4exp_p1.cpp`). See [`install_rust_hooks`].
//!
//! ## Layout
//! `weights/pack.rs` (fp16 bits, index.txt, WeightTable), `weights/cache.rs` (profile, budget, ExpertCacheArena, the
//! profile fill), `weights/lend.rs` (d_res, the serve / generate loan), `weights/native.rs` (NativeDense / Head /
//! Embed), `weights/ple.rs` (PLE wiring, PLE table I/O options), `weights/plan.rs` (plan_lend, the serve scan),
//! `weights/hooks.rs` (the C ABI hook table, the process-global state, `piece1_snapshot`). This file: the
//! `Piece1Weights` handle and the tests.
//!
//! ## C++ source files being ported
//! - `src/core/weights.cpp` & `include/strata/core/weights.hpp` (`Weights`, `LayerView`, `NativeHead`)
//! - `src/core/expert_cache.cpp` & `include/strata/core/expert_cache.hpp` (`ExpertCache`, `ExpertSource`,
//!   `resident_stage_swaps`, `fill_slot_blocking`, `fill_slot_queued`, `sync_queued`)
//! - the PLE weight wiring (`ss.ple.w` from `blk.1.ple_*`)
//! - `src/program/generate.cpp` pack / GGUF opening, VRAM budget, `xcache` slot planning, `host_res` / `d_res`
//!   upload, `plan_lend` / `lend_slots`.
//!
//! ## `tk_*` C ABI entry points (declared in `include/talus_qwen4exp_kernels.h`)
//! - `tk_qwen4exp_p1_query_weights_info`, `tk_qwen4exp_p1_expert_cache_info`, `tk_qwen4exp_p1_borrow_slots`,
//!   `tk_qwen4exp_p1_refill_slots` (skeleton)
//! - `tk_qwen4exp_p1_set_rust_hooks` (qwen4exp-p1 block)
//!
//! ## Bitwise verification (`TALUS_FN_DUMP`)
//! Compare `prefill_p0_input_R.bin`, `prefill_p0_l00_R.bin` .. `prefill_p0_l47_R.bin`, and `first_logits.bin`
//! between `TALUS_FN_RUST=0 TALUS_FN_DUMP="$S/probe/fn_cpp"` and `TALUS_FN_RUST=1 TALUS_FN_DUMP="$S/probe/fn_rust"`.
//! The loader additionally prints `p1: weight table fnv=...` with `TALUS_FN_P1_HASH=1` (both paths, same code), a
//! hash over every tensor's metadata and device bytes.

use core::ffi::{c_char, c_int, c_void};
use std::collections::{BTreeMap, HashSet};
use std::ffi::{CStr, CString};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::Mutex;

use crate::qwen4exp::ffi::{Qwen4ExpKernels, TkQwen4ExpServeCtx, TkQwen4ExpWeightsInfo};
use crate::qwen4exp::skeleton::Qwen4ExpModelConfig;

pub mod cache;
pub mod hooks;
pub mod lend;
pub mod native;
pub mod pack;
pub mod plan;
pub mod ple;
// ---- qwen4exp-r1 begin ----
// the residual of Piece 1: the expert source and the PLE table reader (FileExpertSource / PleTable adopt them)
pub mod ple_reader;
pub mod r1;
pub mod source;
pub mod sys;
// ---- qwen4exp-r1 end ----

pub use cache::*;
#[allow(unused_imports)]
use cache::*;
pub use hooks::*;
#[allow(unused_imports)]
use hooks::*;
pub use lend::*;
#[allow(unused_imports)]
use lend::*;
pub use native::*;
#[allow(unused_imports)]
use native::*;
pub use pack::*;
#[allow(unused_imports)]
use pack::*;
pub use plan::*;
#[allow(unused_imports)]
use plan::*;
pub use ple::*;
#[allow(unused_imports)]
use ple::*;

// =====================================================================================================================
// Expert cache layout and the Piece 1 handle
// =====================================================================================================================

/// Expert cache & residency table layout for `48 * 512 = 24,576` experts.
#[derive(Clone, Debug)]
pub struct ExpertCacheLayout {
    pub n_layers: usize,
    pub n_expert: usize,
    pub total_slots: i32,
    pub resident_count: usize,
    pub cache_bytes: u64,
    pub max_blob_bytes: usize,
    /// Host residency table (`n_layers * n_expert`): slot index `>= 0` if in VRAM, `-1` (`kNotResident`) if on host/NVMe.
    pub host_res: Vec<i32>,
}

impl ExpertCacheLayout {
    pub fn new(n_layers: usize, n_expert: usize) -> Self {
        Self {
            n_layers,
            n_expert,
            total_slots: 0,
            resident_count: 0,
            cache_bytes: 0,
            max_blob_bytes: 0,
            host_res: vec![-1; n_layers * n_expert],
        }
    }

    /// Computes how many tail slots `[total_slots - k .. total_slots)` must be lent to carve `want_bytes` of scratch.
    pub fn plan_lend_slots(&self, want_bytes: u64) -> i32 {
        if self.total_slots <= 128 || self.max_blob_bytes == 0 || want_bytes == 0 {
            return 0;
        }
        let blob = self.max_blob_bytes as u64;
        let k = ((want_bytes + blob - 1) / blob) as i32;
        if k + 128 <= self.total_slots {
            k
        } else {
            0
        }
    }
}

/// Piece 1 state handle owned by `Qwen4ExpRustEngine`.
#[derive(Clone, Debug)]
pub struct Piece1Weights {
    pub native_pack: bool,
    pub mtp_loaded: bool,
    pub ple_ready: bool,
    pub expert_cache: ExpertCacheLayout,
    /// The dense weight table the Rust loader built (`None` on the C++ path, or before the load).
    pub weights: Option<WeightTable>,
    /// Whether the hook table is registered with `libtalus_kernels.so`.
    pub hooks_installed: bool,
    /// Everything else the Rust loaders hold (native attachments, head, embedding, PLE pointers, d_res, cache tables),
    /// refreshed by `sync_from_ctx`.
    pub snapshot: Piece1Snapshot,
}

impl Piece1Weights {
    pub fn new(cfg: &Qwen4ExpModelConfig) -> Self {
        Self {
            native_pack: false,
            mtp_loaded: false,
            ple_ready: false,
            expert_cache: ExpertCacheLayout::new(cfg.n_layers as usize, cfg.n_expert as usize),
            weights: None,
            hooks_installed: install_rust_hooks() && {
                r1::install_r1_hooks();
                true
            }, // qwen4exp-r1: + r1
            snapshot: Piece1Snapshot::default(),
        }
    }

    /// Synchronizes Piece 1 metadata from the active engine context via `tk_qwen4exp_p1_query_weights_info`,
    /// the residency table via `tk_qwen4exp_p1_expert_cache_info`, and adopts the weight table the Rust loader built.
    pub unsafe fn sync_from_ctx(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpServeCtx,
    ) -> Result<(), String> {
        let mut info = TkQwen4ExpWeightsInfo::default();
        if (kernels.p1_query_weights_info)(ctx, &mut info) < 0 {
            return Err("tk_qwen4exp_p1_query_weights_info failed".to_string());
        }
        self.native_pack = info.native_pack != 0;
        self.mtp_loaded = info.mtp_loaded != 0;
        self.ple_ready = info.ple_ready != 0;
        self.expert_cache.total_slots = info.cache_slots;
        self.expert_cache.resident_count = info.resident_experts.max(0) as usize;
        self.expert_cache.cache_bytes = info.cache_bytes;
        self.expert_cache.max_blob_bytes = info.max_blob_bytes.max(0) as usize;
        let n = (kernels.p1_expert_cache_info)(ctx, std::ptr::null_mut(), 0);
        if n > 0 {
            let mut v = vec![-1i32; n as usize];
            let got = (kernels.p1_expert_cache_info)(ctx, v.as_mut_ptr(), v.len());
            if got > 0 {
                v.truncate(got as usize);
                self.expert_cache.host_res = v;
            }
        }
        self.snapshot = piece1_snapshot();
        if self.weights.is_none() {
            self.weights = self.snapshot.weights.clone();
        }
        Ok(())
    }

    /// Borrows `want_chunk_tokens` worth of tail slots from the expert cache for batched prefill.
    pub unsafe fn borrow_slots_for_prefill(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpServeCtx,
        want_chunk_tokens: i64,
    ) -> Result<i32, String> {
        let mut lent_slots = 0i32;
        let rc = (kernels.p1_borrow_slots)(ctx, want_chunk_tokens, &mut lent_slots);
        if rc < 0 {
            return Err("tk_qwen4exp_p1_borrow_slots failed".to_string());
        }
        Ok(lent_slots)
    }

    /// Refills any expert cache slots lent to prefill and re-uploads `d_res`.
    pub unsafe fn refill_borrowed_slots(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpServeCtx,
    ) -> Result<(), String> {
        if (kernels.p1_refill_slots)(ctx) < 0 {
            return Err("tk_qwen4exp_p1_refill_slots failed".to_string());
        }
        Ok(())
    }
}

#[allow(dead_code)]
fn path_exists(p: &str) -> bool {
    Path::new(p).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_cache_plan_lend_respects_128_slot_floor() {
        let mut cache = ExpertCacheLayout::new(48, 512);
        cache.total_slots = 2048;
        cache.max_blob_bytes = 1024 * 1024; // 1 MiB per slot
        assert_eq!(cache.plan_lend_slots(512 * 1024 * 1024), 512);
        // Requesting 1950 MiB leaves < 128 slots, so lending is refused (returns 0).
        assert_eq!(cache.plan_lend_slots(1950 * 1024 * 1024), 0);
    }

    #[test]
    fn f16_round_trip_and_rne() {
        // numpy oracle values (float16 RNE)
        let cases: &[(f32, u16)] = &[
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (1.0, 0x3C00),
            (-2.0, 0xC000),
            (65504.0, 0x7BFF),
            (65520.0, 0x7C00), // rounds up to inf
            (1e30, 0x7C00),
            (f32::INFINITY, 0x7C00),
            (5.960_464_5e-8, 0x0001), // smallest subnormal
            (2.980_232_2e-8, 0x0000), // exactly half the smallest subnormal: ties to even (0)
            (6.103_515_6e-5, 0x0400), // smallest normal
            (0.1, 0x2E66),
            (1.000_976_6, 0x3C01),
            (1.000_488_3, 0x3C00), // tie, even stays
            (1.001_464_8, 0x3C02), // tie, odd rounds up
        ];
        for &(f, h) in cases {
            assert_eq!(f16_from_f32(f), h, "f16_from_f32({f})");
        }
        assert!(f32_from_f16(f16_from_f32(f32::NAN)).is_nan());
        for h in 0u16..=0xFFFF {
            let f = f32_from_f16(h);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f16_from_f32(f), h, "round trip {h:#06x}");
        }
    }

    #[test]
    fn index_parse_and_pool_bytes() {
        let text = "# strata pack index v3\n# align 64 pool 1000 tensors 3\n\
a 0 2 0 100 0 100 25 0 0 0 1 0 0 0 0 0 0 0\n\
b 0 0 0 0 128 0 2560 4 8 0 32 0 0 0 0 0 0 0\n\
c 0 4 100 8 128 8 4 0 0 0 1 0 0 0 0 0 0 0\n";
        let idx = PackIndex::parse(text).unwrap();
        assert_eq!(idx.align, 64);
        assert_eq!(idx.pool, 1000);
        assert_eq!(idx.rows.len(), 3);
        assert_eq!(idx.rows[2].kind, WeightKind::Bf16InF32);
        assert!(idx.rows[2].raw16);
        assert_eq!(PackIndex::pool_bytes_of_text(text, None).unwrap(), 1000);
        let skip: HashSet<String> = ["b".to_string()].into_iter().collect();
        // a: 100 -> 128, c: 8 -> 64
        assert_eq!(
            PackIndex::pool_bytes_of_text(text, Some(&skip)).unwrap(),
            192
        );
        let mut idx2 = idx.clone();
        let (pool, skipped) = idx2.place(Some(&skip));
        assert_eq!(pool, 192);
        assert_eq!(skipped, vec![false, true, false]);
        assert_eq!(idx2.rows[2].dst_off, 128);
        assert!(PackIndex::parse("x 0 1\n").is_err());
        assert!(PackIndex::parse("\n").is_err());
    }

    #[test]
    fn host_load_converts_planes() {
        // a pack with: an f32-promoted bf16 tensor, an f32 tensor converted to f16, and a quantized tensor with fp16
        // scales that are widened
        let dir = std::env::temp_dir().join(format!("fnp1_pack_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut dense = Vec::<u8>::new();
        // t0: bf16 in f32, 4 elems: values with known high halves
        let t0: [u32; 4] = [0x3F80_1234, 0xBF00_FFFF, 0x0000_0001, 0x7F80_0000];
        for v in t0 {
            dense.extend_from_slice(&v.to_le_bytes());
        }
        // t1: f16 in f32, 2 elems
        for v in [0.1f32, -65520.0] {
            dense.extend_from_slice(&v.to_le_bytes());
        }
        // t2: quantized: 4 code bytes, 2 fp16 scales (4 B in the pack, 8 B in the arena), 2 offset bytes
        let q_off = dense.len() as u64;
        dense.extend_from_slice(&[1, 2, 3, 4]);
        dense.extend_from_slice(&0x3C00u16.to_le_bytes());
        dense.extend_from_slice(&0xC000u16.to_le_bytes());
        dense.extend_from_slice(&[9, 8]);
        std::fs::write(dir.join("dense.bin"), &dense).unwrap();
        let idx = format!(
            "# align 64 pool 256 tensors 3\n\
t0 0 1 0 16 0 8 4 0 0 0 1 0 0 0 0 0 0 0\n\
t1 0 3 16 8 64 4 2 0 0 0 1 0 0 0 0 0 0 0\n\
t2 0 0 {q_off} 10 128 14 8 0 2 0 4 0 1 4 8 2 1 0\n"
        );
        std::fs::write(dir.join("index.txt"), idx).unwrap();
        let mut arena = HostArena(vec![0u8; 256]);
        let base = 0x1000usize as *mut u8;
        let wt = WeightTable::load_with(dir.to_str().unwrap(), base, 256, None, &mut arena, false)
            .unwrap();
        let a = &arena.0;
        assert_eq!(
            &a[0..8],
            &[0x80u8, 0x3F, 0x00, 0xBF, 0x00, 0x00, 0x80, 0x7F][..]
        );
        assert_eq!(u16::from_le_bytes([a[64], a[65]]), 0x2E66);
        assert_eq!(u16::from_le_bytes([a[66], a[67]]), 0xFC00);
        assert_eq!(&a[128..132], &[1, 2, 3, 4]);
        assert_eq!(f32::from_le_bytes([a[132], a[133], a[134], a[135]]), 1.0);
        assert_eq!(f32::from_le_bytes([a[136], a[137], a[138], a[139]]), -2.0);
        assert_eq!(&a[140..142], &[9, 8]);
        let r2 = wt.find("t2").unwrap();
        assert_eq!(r2.data as usize, 0x1000 + 128);
        assert!(r2.quantized());
        assert_eq!(wt.report.re_rounded, 2);
        assert_eq!(wt.report.bytes_saved, 8 + 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn plan_input(expert_cache: i64, profile: Vec<(i32, i32)>) -> CachePlanInput {
        CachePlanInput {
            n_layers: 4,
            n_expert: 8,
            expert_cache,
            native_pack: true,
            per_layer: false,
            multi_gpu: false,
            vram_reserve_mib: 700,
            vram_reserve_given: false,
            prefill_mib: 0,
            mtp_bind: 0,
            min_slots: 1,
            session_mib: 0,
            max_context: 4096,
            pipe_first: 0,
            max_blob: 1000,
            blob_bytes: vec![1000, 600, 300, 1000],
            profile,
            profile_path: "p".into(),
        }
    }

    #[test]
    fn cache_plan_native_profile_list() {
        let prof = vec![(0, 1), (1, 2), (2, 3), (3, 0), (1, 1)];
        // budget 3 slots of round256(1000)=1024 -> 3072 B: 1024 + 768 + 512 = 2304, + 1024 > 3072 stops
        let p = plan_cache(&plan_input(3, prof.clone()), || 1 << 40);
        assert_eq!(p.sized_slots, vec![1000, 600, 300]);
        assert_eq!(p.expert_cache, 3);
        // the free room caps it: 700 MiB reserve + 1800 B left
        let p = plan_cache(&plan_input(100, prof.clone()), || (700u64 << 20) + 1800);
        assert_eq!(p.sized_slots, vec![1000, 600]);
        // more than the profile: the profile is the ceiling
        let p = plan_cache(&plan_input(100, prof), || 1 << 40);
        assert_eq!(p.expert_cache, 5);
    }

    #[test]
    fn cache_plan_auto_and_reserve_adapt() {
        let mut inp = plan_input(-1, vec![]);
        inp.native_pack = false;
        // 700 MiB reserve, 10 000 B blobs: free = reserve + 25 000 -> 25 slots... with blob 1000 B
        let p = plan_cache(&inp, || (700u64 << 20) + 25_000);
        assert_eq!(p.expert_cache, 25);
        assert!(!p.reserve_adapted);
        // too few slots for min_slots: the reserve shrinks (to at most what leaves min_slots)
        inp.min_slots = 1000;
        let p = plan_cache(&inp, || 600u64 << 20);
        assert!(p.reserve_adapted);
        assert!(p.vram_reserve_mib >= 300 && p.vram_reserve_mib < 700);
        assert!(p.expert_cache >= 1000);
    }

    #[test]
    fn cache_admission_global_and_per_layer() {
        let mut c = ExpertCacheArena::empty();
        c.slots = 6;
        c.n_layers = 3;
        c.n_expert = 4;
        c.blob = 10;
        c.residency = vec![NOT_RESIDENT; 12];
        c.reset_layer_next();
        assert_eq!(c.admit(0, 1), 0);
        assert_eq!(c.admit(2, 3), 1);
        assert_eq!(c.admit(0, 1), 0);
        assert_eq!(c.slot_of(2, 3), 1);
        assert_eq!(c.admit(5, 0), NOT_RESIDENT);
        let mut d = ExpertCacheArena::empty();
        d.slots = 7;
        d.n_layers = 3;
        d.n_expert = 4;
        d.blob = 10;
        d.residency = vec![NOT_RESIDENT; 12];
        d.per_layer = true;
        d.reset_layer_next();
        assert_eq!(d.layer_slot_range(2), (4, 7));
        assert_eq!(d.admit(1, 0), 2);
        assert_eq!(d.admit(1, 1), 3);
        assert_eq!(d.admit(1, 2), NOT_RESIDENT);
        assert_eq!(d.admit(2, 0), 4);
        assert_eq!(d.admitted, 3);
        // sized offsets
        let mut e = ExpertCacheArena::empty();
        e.slots = 2;
        e.off = vec![0, 256, 768];
        e.base = 0x1000 as *mut u8;
        assert_eq!(e.device_slot(1) as usize, 0x1000 + 256);
        assert_eq!(e.bytes(), 768);
        e.base = std::ptr::null_mut();
    }

    #[test]
    fn profile_reader_round_trip() {
        let path = std::env::temp_dir().join(format!("fnp1_prof_{}.bin", std::process::id()));
        let mut b = Vec::new();
        b.extend_from_slice(b"STRP");
        for v in [1u32, 4, 8, 3, 3] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for (l, e) in [(0u16, 1u16), (3, 7), (2, 0)] {
            b.extend_from_slice(&l.to_le_bytes());
            b.extend_from_slice(&e.to_le_bytes());
        }
        std::fs::write(&path, &b).unwrap();
        let (r, slots) = read_expert_profile(path.to_str().unwrap(), 4, 8).unwrap();
        assert_eq!(r, vec![(0, 1), (3, 7), (2, 0)]);
        assert_eq!(slots, 3);
        assert!(read_expert_profile(path.to_str().unwrap(), 4, 9).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn lend_arithmetic_matches_cpp_rules() {
        let off = [0u64, 256, 768, 1024];
        assert_eq!(split_lend_slots(3, 1024, Some(&off), 512, 1), 1);
        assert_eq!(split_lend_slots(3, 1024, Some(&off), 512, 257), 2);
        assert_eq!(split_lend_slots(3, 1024, Some(&off), 512, 1024), 3);
        assert_eq!(split_lend_slots(3, 1024, Some(&off), 512, 5000), 3);
        assert_eq!(split_lend_slots(10, 0, None, 512, 1025), 3);
        assert_eq!(request_chunk(1, 4096), 256);
        assert_eq!(request_chunk(5000, 4096), 4096);
        assert_eq!(serve_equal_chunk(5000, 4096, false, 1024), 4096);
        // 2 chunks of 2500 -> 2560 when the last plain chunk (904) still streams everything
        assert_eq!(serve_equal_chunk(5000, 4096, true, 900), 2560);
        // the last chunk (904) is below stream_all_min: plain
        assert_eq!(serve_equal_chunk(5000, 4096, true, 1024), 4096);
        assert_eq!(serve_equal_chunk(9000, 4096, true, 128), 3072);
    }

    #[test]
    fn ple_wiring_checks_and_planes() {
        let mut t = WeightTable::default();
        let row =
            |name: &str, kind: i32, code_bits: i32, ne0: i64, ne1: i64, dst: u64, off: usize| {
                let r = PackIndexRow {
                    name: name.into(),
                    file: 0,
                    kind: WeightKind::from_index(kind),
                    src_off: 0,
                    src_bytes: dst,
                    dst_off: off as u64,
                    dst_bytes: dst,
                    ne0,
                    ne1,
                    code_bits,
                    code_bias: 0,
                    group_elems: 0,
                    codebook: 0,
                    has_offset: 0,
                    codes_bytes: if code_bits != 0 { 64 } else { 0 },
                    scales_bytes: 0,
                    offset_bytes: 0,
                    scales_fp16: 0,
                    act_kind: 0,
                    raw16: false,
                };
                WeightRef::from_row(&r, (0x10000 + off) as *const c_void, true)
            };
        t.table.insert(
            "blk.1.ple_key.weight".into(),
            row("k", 0, 8, 2560, 10240, 128, 0),
        );
        t.table.insert(
            "blk.1.ple_value.weight".into(),
            row("v", 1, 0, 4, 4, 32, 256),
        );
        t.table.insert(
            "blk.1.ple_norm_key.weight".into(),
            row("nk", 2, 0, 4, 0, 16, 512),
        );
        t.table.insert(
            "blk.1.ple_norm_query.weight".into(),
            row("nq", 2, 0, 4, 0, 16, 768),
        );
        t.table.insert(
            "blk.1.ple_norm_conv.weight".into(),
            row("nc", 2, 0, 4, 0, 16, 1024),
        );
        t.table.insert(
            "blk.1.ple_conv1d.weight".into(),
            row("c", 3, 0, 4, 4, 32, 1280),
        );
        let w = ple_wire(&t, None, false).unwrap();
        assert_eq!(w.key_codes as usize, 0x10000);
        assert_eq!(w.key_scales as usize, 0x10000 + 64);
        assert!(w.key_bf16.is_null());
        assert_eq!(w.conv1d_f16 as usize, 0x10000 + 1280);
        // the native key is required when asked
        assert!(ple_wire(&t, None, true).is_err());
        let mut nd = NativeDense::default();
        nd.attach.insert(
            "blk.1.ple_key.weight".into(),
            NativeAttach {
                native_data: 0x99,
                native_type: 42,
                native_q8_1: 0x77,
                hc_q8: 0,
            },
        );
        let w = ple_wire(&t, Some(&nd), true).unwrap();
        assert_eq!(w.key_native_type, 42);
        // a conv1d kept as f32 is refused
        t.table.insert(
            "blk.1.ple_conv1d.weight".into(),
            row("c", 2, 0, 4, 4, 64, 1280),
        );
        assert!(ple_wire(&t, None, false).is_err());
        t.table.remove("blk.1.ple_value.weight");
        assert!(ple_wire(&t, None, false).is_err());
    }

    #[test]
    fn q8_0_reference_rounding() {
        let mut x = [0f32; 32];
        for (j, v) in x.iter_mut().enumerate() {
            *v = (j as f32 - 15.5) * 0.25;
        }
        let q = q8_0_of(&x);
        let d = f32_from_f16(u16::from_le_bytes([q[0], q[1]]));
        assert!((d - 3.875 / 127.0).abs() < 1e-5);
        assert_eq!(q[2] as i8, -127);
        assert_eq!(q[33] as i8, 127);
        let bf: Vec<u16> = x.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        assert_eq!(q8_0_of_bf16(&bf), q);
    }

    /// A pricing whose buffers grow linearly with the chunk and whose ring takes `ring` max-blob slots.
    struct ModelPricing {
        per_token: u64,
        blob: u64,
        ring: i64,
        ring_cap: i64,
        calls: Vec<(i64, i64)>,
    }
    impl PrefillPricing for ModelPricing {
        fn bytes_needed(&mut self, c: i64) -> u64 {
            c as u64 * self.per_token + self.ring as u64 * self.blob
        }
        fn bytes_needed_no_ring(&mut self, c: i64) -> u64 {
            c as u64 * self.per_token
        }
        fn set_ring_budget(&mut self, room: i64, small: i64) {
            self.calls.push((room, small));
            self.ring = if room > 0 { room } else { 384 };
        }
        fn ring_cap_for(&mut self, _c: i64) -> i64 {
            self.ring_cap
        }
        fn ring_default_slots(&mut self) -> i64 {
            384
        }
        fn ring_bytes_enabled(&mut self) -> bool {
            true
        }
    }

    #[test]
    fn loan_planning_fixed_and_auto() {
        let lp = LendPlanner {
            slots: 4096,
            bytes: 4096 * 1000,
            offsets: None,
            max_blob: 1000,
            prefill_auto: false,
            prefill_chunk: 16384,
            prefill_auto_max: 16384,
            prefill_auto_cap: 0,
            max_context: 262144,
            auto_ceiling: 16384,
            lend_pct: 85,
        };
        // fixed: halved until it leaves 128 slots: 16384 tokens * 200 B = 3.3 MB + 384 ring blobs > 3968 slots?
        let mut p = ModelPricing {
            per_token: 200,
            blob: 1000,
            ring: 384,
            ring_cap: 512,
            calls: vec![],
        };
        let mut chunk = 16384;
        let k = lp.plan_lend(&mut p, &mut chunk);
        assert_eq!(chunk, 16384);
        assert_eq!(k, (16384 * 200 + 384 * 1000 + 999) / 1000);
        let mut p2 = ModelPricing {
            per_token: 300,
            ..p
        };
        let mut chunk = 16384;
        let k = lp.plan_lend(&mut p2, &mut chunk);
        assert_eq!(chunk, 8192);
        assert!(k + 128 <= 4096);
        // serve scan, fixed: the same halving, first = slots - k
        let (c, f) = lp.serve_scan(&mut p2);
        assert_eq!(c, 8192);
        assert_eq!(f, 4096 - k);
        // auto: the chunk is the largest on the grid that keeps the ring full under the 85% cap
        let lpa = LendPlanner {
            prefill_auto: true,
            ..lp.clone()
        };
        let mut p3 = ModelPricing {
            per_token: 300,
            blob: 1000,
            ring: 384,
            ring_cap: 512,
            calls: vec![],
        };
        let mut chunk = 0;
        let k = lpa.plan_lend(&mut p3, &mut chunk);
        assert!(chunk > 0 && chunk % 256 == 0 && chunk <= 16384);
        assert!(k * 100 <= 85 * 4096);
        let mut p4 = ModelPricing {
            per_token: 300,
            blob: 1000,
            ring: 384,
            ring_cap: 512,
            calls: vec![],
        };
        let (c, f) = lpa.serve_scan(&mut p4);
        assert_eq!(c, chunk);
        assert_eq!(f, 4096 - k);
        assert_eq!(biggest_chunk(1000, |t| t <= 600), 512);
        assert_eq!(biggest_chunk(100, |_| true), 0);
    }

    #[test]
    fn ple_io_options() {
        let p = ple_io_plan("direct", 256, 1 << 20, false);
        assert_eq!((p.mode, p.lock, p.io_thread), (0, 0, 1));
        let r = ple_io_plan("ram", 16, 0, true);
        assert_eq!((r.mode, r.lock, r.keepalive_ms), (1, 1, 0.0));
        assert_eq!(c_atof("12.5ms"), 12.5);
        assert_eq!(c_atof("x"), 0.0);
        assert_eq!(ple_probe_decide(20000.0, false), 0);
        assert_eq!(ple_probe_decide(0.0, false), 0);
    }

    #[test]
    fn production_index_if_present() {
        // TALUS_FN_PACK_DIR = a production native pack directory (holds index.txt); skipped when unset/absent
        let Ok(dir) = std::env::var("TALUS_FN_PACK_DIR") else {
            return;
        };
        if !path_exists(&format!("{dir}/index.txt")) {
            return;
        }
        let idx = PackIndex::open(&dir).unwrap();
        assert_eq!(idx.rows.len(), 1079);
        assert_eq!(idx.align, 64);
        assert_eq!(PackIndex::pool_bytes(&dir, None).unwrap(), idx.pool);
        // every quantized row of a native pack carries a shape only
        let skip: HashSet<String> = idx
            .rows
            .iter()
            .filter(|r| r.code_bits != 0)
            .map(|r| r.name.clone())
            .collect();
        let compact = PackIndex::pool_bytes(&dir, Some(&skip)).unwrap();
        let mut i2 = idx.clone();
        let (pool, _) = i2.place(Some(&skip));
        assert_eq!(compact, pool);
        assert_eq!(
            PackIndex::index_code_bits(&dir, "blk.1.ple_key.weight").unwrap(),
            8
        );
        assert_eq!(PackIndex::index_code_bits(&dir, "nope").unwrap(), -1);
        for r in &idx.rows {
            if r.dst_bytes > 0 {
                segments_of(r).unwrap();
            }
        }
    }
}
