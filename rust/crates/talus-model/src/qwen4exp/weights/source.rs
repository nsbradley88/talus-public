//! qwen4exp-r1: the expert source in Rust - `strata::core::FileExpertSource` (src/core/expert_source.cpp) for a native
//! pack read from its GGUF shards in place (CS-T, the Flash-Next production mode).
//!
//! What moved: the mapping of the shards (one MAP_SHARED view per file), the blob assembly into the stage-buffer pool
//! (`blob` / `blob_stable`, the #1237 copy fences of `note_async_read`), the prompt stager's copies (`copy_blob(s)`,
//! `transient`, `cached_share`), the read-ahead (`advise_pairs`, `prefetch_pairs`, `prefetch`, `warm`), the
//! unbuffered file tier (#286/#577: O_DIRECT + the kernel's AIO, `set_unbuffered` / `recheck_unbuffered`,
//! `open_prompt_direct`), the resident RAM copy (`pin_cache_complement`: the budget, the lend region, the paced and
//! partial page-locking, the copy threads, the STRATA_VERIFY_COMPLEMENT checksum) and the adaptive tier's exchanges
//! (`reserve_exchanges` .. `commit_flip`). The C++ `FileExpertSource` adopts a handle to this object and forwards
//! every method (`FileExpertSource::adopt_rust`, the `qwen4exp-r1` lines in expert_source.cpp); the C++ engine's other
//! objects keep reading it through the `ExpertSource` interface.
//!
//! What did not (the adoption is declined at creation and the C++ class runs as before): a pack with experts.bin, the
//! Linux io-prefetch / io-stats tier (STRATA_IO_PREFETCH, STRATA_IO_STATS), exchange rotation (STRATA_EXCHANGE_ROTATE),
//! the unmapped-GGUF test mode (STRATA_TEST_UNMAP_GGUF), Windows-only paths (startup unbuffered, the mapped-page
//! trims), the arena source (`ArenaExpertSource`), multi-GPU and helper-GPU tiers.
//!
//! Concurrency follows the C++ object exactly: the stage pool is guarded by one mutex + condvar (`stage_mu_`,
//! `stage_cv_`), the counters are atomics, and everything else (geometry, the views, the RAM copy and its exchange
//! tables, the unbuffered descriptors) is written only at startup or by the adaptive tier's protocol, which the C++
//! callers serialize - so it lives in an `UnsafeCell` read without a lock, as the C++ members are.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
// the C++ arithmetic is kept as written (its saturating ternaries, `s + 3 >= epoch + 1`)
#![allow(clippy::implicit_saturating_sub, clippy::int_plus_one)]

use core::ffi::{c_int, c_void};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::atomic::{
    AtomicBool, AtomicI64, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Instant;

use super::cache::{fnv1a64, STRATA_FNV_SEED};
use super::sys::*;

const RELAXED: Ordering = Ordering::Relaxed;

// =====================================================================================================================
// The expert layout (strata::kernels::cpu::expert_layout(), read through the r1 services)
// =====================================================================================================================

/// `strata::kernels::cpu::ExpertLayout` with the `NativeFmt` fields the source reads.
#[derive(Clone, Debug, Default)]
pub struct ExpertLayout {
    pub native: bool,
    pub n_layers: i64,
    pub n_expert: i64,
    pub fmt_up_off: Vec<u64>,
    pub fmt_down_off: Vec<u64>,
    pub fmt_bytes: Vec<u64>,
    pub fmt_gu_type: Vec<i32>,
    pub fmt_d_type: Vec<i32>,
    pub fmt_n_embd: Vec<i64>,
    pub fmt_n_ff: Vec<i64>,
    pub offset: Vec<u64>,
    pub bytes: Vec<u64>,
    pub gguf_off: Vec<u64>,
    pub gguf_file: Vec<String>,
    pub max_blob: u64,
    pub total: u64,
    /// `strata::kernels::cpu::BLOB` (the canonical blob size).
    pub canonical_blob: u64,
}

impl ExpertLayout {
    pub fn blob_bytes(&self, layer: i64) -> u64 {
        if self.native {
            self.bytes[layer as usize]
        } else {
            self.canonical_blob
        }
    }
}

/// `expert_gguf_file`: the GGUF file holding role `r` of layer `l` (a name beside the --native shard, or the shard).
pub fn expert_gguf_file(gguf: &str, lay: &ExpertLayout, l: i64, r: usize) -> String {
    let i = (3 * l) as usize + r;
    if lay.gguf_file.len() <= i || lay.gguf_file[i].is_empty() {
        return gguf.to_string();
    }
    let dir = match gguf.rfind(['/', '\\']) {
        Some(cut) => &gguf[..cut + 1],
        None => "",
    };
    format!("{dir}{}", lay.gguf_file[i])
}

/// `check_experts_gguf`: every (layer, role) tensor native_experts.txt names is in its file, with the pack's type,
/// shape, per-expert size and offset.
pub fn check_experts_gguf(gguf: &str, lay: &ExpertLayout) -> Result<(), String> {
    const ROLES: [&str; 3] = ["gate", "up", "down"];
    if lay.gguf_off.len() != (3 * lay.n_layers) as usize {
        return Err("native_experts.txt has no GGUF offsets (a pack older than v2): repack it with tools/iq_pack.py".into());
    }
    let mut files: HashMap<String, talus_gguf::GgufFile> = HashMap::new();
    for l in 0..lay.n_layers {
        let li = l as usize;
        let blob = lay.bytes[li];
        let per = [
            lay.fmt_up_off[li],
            lay.fmt_up_off[li],
            blob - lay.fmt_down_off[li],
        ];
        for r in 0..3 {
            let path = expert_gguf_file(gguf, lay, l, r);
            if !files.contains_key(&path) {
                let f = talus_gguf::GgufFile::open(&path)
                    .map_err(|e| format!("native experts from the GGUF: {e}"))?;
                files.insert(path.clone(), f);
            }
            let f = &files[&path];
            let name = format!("blk.{l}.ffn_{}_exps.weight", ROLES[r]);
            let want_type = if r < 2 {
                lay.fmt_gu_type[li]
            } else {
                lay.fmt_d_type[li]
            } as u64;
            let cols = if r < 2 {
                lay.fmt_n_embd[li]
            } else {
                lay.fmt_n_ff[li]
            } as u64;
            let rows = if r < 2 {
                lay.fmt_n_ff[li]
            } else {
                lay.fmt_n_embd[li]
            } as u64;
            let bytes = per[r] * lay.n_expert as u64;
            let payload = f.file_size() - f.data_start();
            let why = match f.find_tensor(&name) {
                None => "is not in it".to_string(),
                Some(t) if t.qtype.ggml_type_id() as u64 != want_type => {
                    format!("is {}, the pack says type {want_type}", t.qtype.name())
                }
                Some(t)
                    if t.shape().len() != 3
                        || t.shape()[0] != cols
                        || t.shape()[1] != rows
                        || t.shape()[2] != lay.n_expert as u64 =>
                {
                    format!("is not [{cols}, {rows}, {}]", lay.n_expert)
                }
                Some(t) if t.byte_size != bytes => format!("is not {} B per expert", per[r]),
                Some(t) if f.data_start() + t.offset != lay.gguf_off[3 * li + r] => format!(
                    "starts at byte {}, the pack says {}",
                    f.data_start() + t.offset,
                    lay.gguf_off[3 * li + r]
                ),
                Some(t) if t.offset > payload || bytes > payload - t.offset => {
                    "runs past the end of the file (a truncated shard?)".to_string()
                }
                Some(_) => String::new(),
            };
            if !why.is_empty() {
                return Err(format!(
                    "the pack's native_experts.txt does not match the model: {name} in {path} {why} - repack with \
                     tools/iq_pack.py from this model's shards"
                ));
            }
        }
    }
    Ok(())
}

// =====================================================================================================================
// The source
// =====================================================================================================================

/// One mapped GGUF shard (MAP_SHARED, as the C++ view).
struct Map {
    base: *const u8,
    bytes: u64,
    fd: c_int,
}

/// One queued fill of a stage buffer (`FileExpertSource::Fill`).
#[derive(Clone, Copy)]
struct Fill {
    v: usize,
    layer: i64,
    e: i64,
    dst: *mut u8,
}
unsafe impl Send for Fill {}
unsafe impl Sync for Fill {}

/// A staged exchange (`FileExpertSource::Exchange`).
#[derive(Clone, Copy)]
struct Exchange {
    i_in: usize,
    out: usize,
    q: i64,
    bytes: u64,
}

/// A stage buffer: pinned (hipHostMalloc, STRATA_STAGE_PIN=1) or pageable.
struct StageBuf {
    ptr: *mut u8,
    len: usize,
    pinned: bool,
}

impl Drop for StageBuf {
    fn drop(&mut self) {
        unsafe {
            if self.pinned {
                let _ = hip_sys::hipHostFree(self.ptr.cast());
            } else {
                drop(Vec::from_raw_parts(self.ptr, self.len, self.len));
            }
        }
    }
}

/// The stage pool (`stage_mu_` and everything it guards).
#[derive(Default)]
struct Stage {
    buf: Vec<StageBuf>,
    key: Vec<i64>,
    epoch: Vec<u64>,
    used: Vec<u64>,
    busy: Vec<bool>,
    of: HashMap<i64, usize>,
    idx: HashMap<usize, usize>,
    ev: Vec<hip_sys::hipEvent_t>,
    ev_live: Vec<bool>,
    blob: u64,
    seq: u64,
    epoch_now: u64,
    last_layer: i64,
    grew: bool,
    pin_said: bool,
    pin_failed_said: bool,
}

unsafe impl Send for Stage {}

/// The C++ members outside the stage lock (see the module note on concurrency).
struct Core {
    opened: bool,
    n_layers: i64,
    n_expert: i64,
    blobs: i64,
    layer_offsets: Vec<u64>,
    layer_blob_bytes: Vec<u64>,
    maps: Vec<Map>,
    paths: Vec<String>,
    role_ptr: Vec<*const u8>,
    role_bytes: Vec<u64>,
    role_file: Vec<i32>,
    direct: Vec<c_int>,
    pdirect: Vec<c_int>,
    fetch_threads: usize,
    gguf: String,
    inputs: Vec<(CString, CString)>,
    warm_stamp: Vec<AtomicU32>,
    // the resident RAM copy
    complement_arena: *mut u8,
    complement_host: *const u8,
    complement_device: *const u8,
    complement_bytes: u64,
    complement_offsets: Vec<u64>,
    complement_pinned: bool,
    complement_partial: bool,
    complement_registered: bool,
    complement_pin_limit: u64,
    complement_lock_off: u64,
    complement_ready: bool,
    complement_locked: u64,
    complement_lent_slots: i64,
    complement_aligned: bool,
    stage_lend_regions: Vec<Vec<(i32, i32)>>,
    // the adaptive tier's exchanges
    override_: Vec<*const u8>,
    staged: Vec<Exchange>,
    soft_in: Vec<usize>,
    xstage_dev: *const u8,
    xstage: *mut u8,
    xstage_pinned: bool,
    xstage_cap: i64,
    xstage_blob: u64,
    exchanges: i64,
}

impl Default for Core {
    fn default() -> Self {
        Self {
            opened: false,
            n_layers: 0,
            n_expert: 0,
            blobs: 0,
            layer_offsets: Vec::new(),
            layer_blob_bytes: Vec::new(),
            maps: Vec::new(),
            paths: Vec::new(),
            role_ptr: Vec::new(),
            role_bytes: Vec::new(),
            role_file: Vec::new(),
            direct: Vec::new(),
            pdirect: Vec::new(),
            fetch_threads: 8,
            gguf: String::new(),
            inputs: Vec::new(),
            warm_stamp: Vec::new(),
            complement_arena: std::ptr::null_mut(),
            complement_host: std::ptr::null(),
            complement_device: std::ptr::null(),
            complement_bytes: 0,
            complement_offsets: Vec::new(),
            complement_pinned: false,
            complement_partial: false,
            complement_registered: false,
            complement_pin_limit: 0,
            complement_lock_off: 0,
            complement_ready: false,
            complement_locked: 0,
            complement_lent_slots: 0,
            complement_aligned: false,
            stage_lend_regions: Vec::new(),
            override_: Vec::new(),
            staged: Vec::new(),
            soft_in: Vec::new(),
            xstage_dev: std::ptr::null(),
            xstage: std::ptr::null_mut(),
            xstage_pinned: false,
            xstage_cap: 0,
            xstage_blob: 0,
            exchanges: 0,
        }
    }
}

/// The GPU expert cache as `pin_cache_complement` reads it (valid, fills, resident, slots, slot_of per pair).
pub struct CacheView<'a> {
    pub valid: bool,
    pub fills: i64,
    pub resident: i64,
    pub slots: i64,
    pub slot_of: &'a [i32],
}

/// The ForesightSwap hold / unhold callback (C++ object, reached through the r1 services).
pub type FsHoldFn = unsafe extern "C" fn(fs: *mut c_void, layer: i64, expert: i32, hold: c_int);

/// `strata::core::FileExpertSource` for the GGUF in place.
pub struct FileSource {
    core: UnsafeCell<Core>,
    stage: Mutex<Stage>,
    stage_cv: Condvar,
    file_reads: AtomicI64,
    ram_reads: AtomicI64,
    file_read_bytes: AtomicU64,
    file_blob_bytes: AtomicU64,
    file_us: AtomicU64,
    warm_hits: AtomicI64,
    warm_count: AtomicI64,
    direct_fallbacks: AtomicI64,
    stage_fence_waits: AtomicU64,
    reads: AtomicI64,
    fs_swap: AtomicPtr<c_void>,
    fs_hold: Option<FsHoldFn>,
    layout: ExpertLayout,
}

unsafe impl Send for FileSource {}
unsafe impl Sync for FileSource {}

impl Drop for FileSource {
    fn drop(&mut self) {
        self.close();
    }
}

/// `STRATA_STAGE_PIN` (unset / 0: pageable stage buffers; anything else: try to pin them).
fn stage_pin_wanted() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        env("STRATA_STAGE_PIN")
            .map(|v| c_atoi(&v) != 0)
            .unwrap_or(false)
    })
}

fn hip_err_str(e: hip_sys::hipError_t) -> String {
    unsafe {
        let p = hip_sys::hipGetErrorString(e);
        if p.is_null() {
            format!("hip error {}", e.0)
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

fn us_since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e6
}

const MAX_STAGE_SEQ: u64 = 256;
const STAGE_AGE: u64 = 3;

impl FileSource {
    pub fn new(layout: ExpertLayout, fs_hold: Option<FsHoldFn>) -> Self {
        Self {
            core: UnsafeCell::new(Core::default()),
            stage: Mutex::new(Stage {
                last_layer: -1,
                ..Stage::default()
            }),
            stage_cv: Condvar::new(),
            file_reads: AtomicI64::new(0),
            ram_reads: AtomicI64::new(0),
            file_read_bytes: AtomicU64::new(0),
            file_blob_bytes: AtomicU64::new(0),
            file_us: AtomicU64::new(0),
            warm_hits: AtomicI64::new(0),
            warm_count: AtomicI64::new(0),
            direct_fallbacks: AtomicI64::new(0),
            stage_fence_waits: AtomicU64::new(0),
            reads: AtomicI64::new(0),
            fs_swap: AtomicPtr::new(std::ptr::null_mut()),
            fs_hold,
            layout,
        }
    }

    #[inline]
    fn c(&self) -> &Core {
        unsafe { &*self.core.get() }
    }
    /// The C++ protocol serializes the writers (see the module note).
    #[allow(clippy::mut_from_ref)]
    #[inline]
    unsafe fn cm(&self) -> &mut Core {
        &mut *self.core.get()
    }
    fn lock_stage(&self) -> MutexGuard<'_, Stage> {
        self.stage.lock().unwrap_or_else(|e| e.into_inner())
    }

    // ---- queries ----------------------------------------------------------------------------------------------------

    pub fn mapped(&self) -> bool {
        self.c().opened
    }
    pub fn gguf_mode(&self) -> bool {
        !self.c().role_ptr.is_empty()
    }
    pub fn blobs(&self) -> i64 {
        self.c().blobs
    }
    pub fn pinned_bytes(&self) -> u64 {
        let c = self.c();
        if c.complement_pinned {
            c.complement_pin_limit
        } else {
            0
        }
    }
    pub fn resident_bytes(&self) -> u64 {
        self.c().complement_bytes
    }
    pub fn complement_pinned(&self) -> bool {
        self.c().complement_pinned
    }
    pub fn complement_ready(&self) -> bool {
        self.c().complement_ready
    }
    pub fn locked_bytes(&self) -> u64 {
        self.c().complement_locked
    }
    pub fn resident_lent_slots(&self) -> i64 {
        self.c().complement_lent_slots
    }
    pub fn exchange_capacity(&self) -> i64 {
        self.c().xstage_cap
    }
    pub fn exchange_pinned(&self) -> bool {
        self.c().xstage_pinned
    }
    pub fn exchanges(&self) -> i64 {
        self.c().exchanges
    }
    pub fn file_reads(&self) -> i64 {
        self.file_reads.load(RELAXED)
    }
    pub fn ram_reads(&self) -> i64 {
        self.ram_reads.load(RELAXED)
    }
    pub fn file_read_bytes(&self) -> u64 {
        self.file_read_bytes.load(RELAXED)
    }
    pub fn file_blob_bytes(&self) -> u64 {
        self.file_blob_bytes.load(RELAXED)
    }
    pub fn file_us(&self) -> u64 {
        self.file_us.load(RELAXED)
    }
    pub fn unbuffered(&self) -> bool {
        !self.c().direct.is_empty()
    }
    pub fn warms(&self) -> bool {
        !self.c().role_ptr.is_empty() && self.c().direct.is_empty()
    }
    pub fn warmed_hits(&self) -> i64 {
        self.warm_hits.load(RELAXED)
    }
    pub fn warmed(&self) -> i64 {
        self.warm_count.load(RELAXED)
    }
    pub fn direct_fallbacks(&self) -> i64 {
        self.direct_fallbacks.load(RELAXED)
    }
    pub fn reads(&self) -> i64 {
        self.reads.load(RELAXED)
    }
    pub fn stage_fence_waits(&self) -> u64 {
        self.stage_fence_waits.load(RELAXED)
    }
    pub fn inputs(&self) -> &[(CString, CString)] {
        &self.c().inputs
    }
    pub fn expert_bytes(&self) -> u64 {
        let c = self.c();
        c.layer_blob_bytes
            .iter()
            .map(|&b| b * c.n_expert as u64)
            .sum()
    }
    pub fn set_fetch_threads(&self, n: i32) {
        unsafe { self.cm().fetch_threads = if n < 1 { 1 } else { n as usize } };
    }
    pub fn set_foresight_swap(&self, fs: *mut c_void) {
        self.fs_swap.store(fs, Ordering::Release);
    }
    pub fn set_stage_lend_regions(&self, regions: Vec<Vec<(i32, i32)>>) {
        unsafe { self.cm().stage_lend_regions = regions };
    }
    fn staged_mode(&self) -> bool {
        let c = self.c();
        !c.role_ptr.is_empty() || !c.direct.is_empty()
    }

    // ---- open / close -----------------------------------------------------------------------------------------------

    /// `FileExpertSource::open` for a native pack without experts.bin and a --native GGUF (the only mode adopted).
    pub fn open(
        &self,
        pack_dir: &str,
        gguf: &str,
        n_layers: i64,
        n_expert: i64,
    ) -> Result<(), String> {
        self.close();
        if n_layers <= 0 || n_expert <= 0 {
            return Err("FileExpertSource: the geometry is empty".into());
        }
        let layout = &self.layout;
        if layout.n_layers != n_layers || layout.n_expert != n_expert {
            return Err(
                "FileExpertSource: the requested geometry does not match the loaded expert layout"
                    .into(),
            );
        }
        if n_layers as u64 > i64::MAX as u64 / n_expert as u64 {
            return Err("FileExpertSource: the expert count overflows".into());
        }
        let blob_count = n_layers as u64 * n_expert as u64;
        let want = layout.total;
        if want == 0 {
            return Err("FileExpertSource: the loaded expert layout has an invalid size".into());
        }
        let mut layer_offsets = vec![0u64; n_layers as usize];
        let mut layer_blob_bytes = vec![0u64; n_layers as usize];
        if !layout.native {
            return Err(
                "qwen4exp-r1: a canonical (experts.bin) pack is not ported; the C++ source reads it"
                    .into(),
            );
        }
        if layout.offset.len() != n_layers as usize
            || layout.bytes.len() != n_layers as usize
            || layout.fmt_bytes.len() != n_layers as usize
        {
            return Err("FileExpertSource: the native expert layout is incomplete".into());
        }
        let mut at = 0u64;
        for l in 0..n_layers as usize {
            let bytes = layout.fmt_bytes[l];
            if layout.offset[l] != at
                || bytes == 0
                || layout.bytes[l] != bytes
                || bytes > u64::MAX / n_expert as u64
            {
                return Err(format!(
                    "FileExpertSource: the native expert layout is invalid at layer {l}"
                ));
            }
            let layer_bytes = bytes * n_expert as u64;
            if at > want || layer_bytes > want - at {
                return Err(format!("FileExpertSource: the native expert layout exceeds its declared size at layer {l}"));
            }
            layer_offsets[l] = layout.offset[l];
            layer_blob_bytes[l] = bytes;
            at += layer_bytes;
        }
        if at != want {
            return Err(
                "FileExpertSource: the native expert layout has an inconsistent size".into(),
            );
        }
        let path = format!("{pack_dir}/experts.bin");
        if gguf.is_empty() || std::path::Path::new(&path).exists() {
            return Err("qwen4exp-r1: the pack has experts.bin (or no --native GGUF); the C++ source reads it".into());
        }
        {
            let c = unsafe { self.cm() };
            c.blobs = blob_count as i64;
            c.n_layers = n_layers;
            c.n_expert = n_expert;
            c.layer_offsets = layer_offsets;
            c.layer_blob_bytes = layer_blob_bytes;
            c.gguf = gguf.to_string();
        }
        if let Err(e) = self.open_gguf() {
            self.close();
            return Err(e);
        }
        // record_inputs(path, gguf_, layout, true)
        let c = unsafe { self.cm() };
        const ROLES: [&str; 3] = ["gate", "up", "down"];
        c.inputs.clear();
        for l in 0..layout.n_layers {
            for (r, role) in ROLES.iter().enumerate() {
                c.inputs.push((
                    CString::new(format!("expert blk.{l}.ffn_{role}")).unwrap_or_default(),
                    CString::new(expert_gguf_file(gguf, layout, l, r)).unwrap_or_default(),
                ));
            }
        }
        Ok(())
    }

    fn open_gguf(&self) -> Result<(), String> {
        let lay = &self.layout;
        let gguf = self.c().gguf.clone();
        check_experts_gguf(&gguf, lay).map_err(|e| format!("FileExpertSource: {e}"))?;
        let c = unsafe { self.cm() };
        let nl = c.n_layers as usize;
        let mut index: HashMap<String, usize> = HashMap::new();
        c.role_ptr = vec![std::ptr::null(); 3 * nl];
        c.role_bytes = vec![0; 3 * nl];
        c.role_file = vec![0; 3 * nl];
        for l in 0..nl {
            let per = [
                lay.fmt_up_off[l],
                lay.fmt_up_off[l],
                lay.bytes[l] - lay.fmt_down_off[l],
            ];
            for r in 0..3 {
                let i = 3 * l + r;
                let path = expert_gguf_file(&gguf, lay, l as i64, r);
                let mi = match index.get(&path) {
                    Some(&m) => m,
                    None => {
                        let fd = open_path(&path, O_RDONLY);
                        let size = if fd >= 0 {
                            file_size(&path).unwrap_or(0)
                        } else {
                            0
                        };
                        if fd < 0 || size == 0 {
                            if fd >= 0 {
                                unsafe { close(fd) };
                            }
                            return Err(format!("FileExpertSource: cannot open {path}"));
                        }
                        let view = unsafe {
                            mmap(
                                std::ptr::null_mut(),
                                size as usize,
                                PROT_READ,
                                MAP_SHARED,
                                fd,
                                0,
                            )
                        };
                        if view == MAP_FAILED {
                            unsafe { close(fd) };
                            return Err(format!("FileExpertSource: cannot map {path}"));
                        }
                        c.maps.push(Map {
                            base: view as *const u8,
                            bytes: size,
                            fd,
                        });
                        c.paths.push(path.clone());
                        index.insert(path.clone(), c.maps.len() - 1);
                        c.maps.len() - 1
                    }
                };
                let m = &c.maps[mi];
                c.role_file[i] = mi as i32;
                let at = lay.gguf_off[i];
                let bytes = per[r] * c.n_expert as u64;
                if at > m.bytes || bytes > m.bytes - at {
                    return Err(format!(
                        "FileExpertSource: an expert span runs past the end of {path}"
                    ));
                }
                c.role_ptr[i] = unsafe { m.base.add(at as usize) };
                c.role_bytes[i] = per[r];
            }
        }
        c.opened = true;
        c.warm_stamp = (0..(c.n_layers * c.n_expert) as usize)
            .map(|_| AtomicU32::new(0))
            .collect();
        let mut st = self.lock_stage();
        for &b in &c.layer_blob_bytes {
            st.blob = st.blob.max(b);
        }
        Ok(())
    }

    /// `FileExpertSource::close`.
    pub fn close(&self) {
        let c = unsafe { self.cm() };
        c.inputs.clear();
        unsafe {
            if !c.complement_arena.is_null() {
                if c.complement_pinned && !c.complement_registered {
                    let _ = hip_sys::hipHostFree(c.complement_arena.cast());
                } else {
                    if c.complement_registered {
                        let _ = hip_sys::hipHostUnregister(c.complement_arena.cast());
                    }
                    if c.complement_locked > 0 {
                        unlock_resident(
                            c.complement_arena.add(c.complement_lock_off as usize),
                            c.complement_locked,
                        );
                    }
                    free(c.complement_arena.cast());
                }
            }
            if !c.xstage.is_null() {
                if c.xstage_pinned {
                    let _ = hip_sys::hipHostFree(c.xstage.cast());
                } else {
                    free(c.xstage.cast());
                }
            }
        }
        c.xstage = std::ptr::null_mut();
        c.xstage_pinned = false;
        c.xstage_cap = 0;
        c.xstage_blob = 0;
        c.xstage_dev = std::ptr::null();
        c.override_.clear();
        c.staged.clear();
        c.soft_in.clear();
        c.exchanges = 0;
        self.file_reads.store(0, RELAXED);
        c.complement_arena = std::ptr::null_mut();
        c.complement_host = std::ptr::null();
        c.complement_device = std::ptr::null();
        c.complement_bytes = 0;
        c.complement_offsets.clear();
        c.complement_pinned = false;
        c.complement_partial = false;
        c.complement_registered = false;
        c.complement_pin_limit = 0;
        c.complement_lock_off = 0;
        c.complement_ready = false;
        c.complement_locked = 0;
        c.complement_lent_slots = 0;
        for m in c.maps.drain(..) {
            unsafe {
                if !m.base.is_null() {
                    munmap(m.base as *mut c_void, m.bytes as usize);
                }
                if m.fd >= 0 {
                    close(m.fd);
                }
            }
        }
        c.role_ptr.clear();
        c.role_bytes.clear();
        c.role_file.clear();
        c.paths.clear();
        for fd in c.direct.drain(..).chain(c.pdirect.drain(..)) {
            unsafe { close(fd) };
        }
        {
            let mut st = self.lock_stage();
            // the buffers' last copies end before they are freed (#1237)
            if !st.ev.is_empty() {
                unsafe {
                    let _ = hip_sys::hipDeviceSynchronize();
                }
            }
            for e in st.ev.drain(..) {
                if !e.is_null() {
                    unsafe {
                        let _ = hip_sys::hipEventDestroy(e);
                    }
                }
            }
            *st = Stage {
                last_layer: -1,
                ..Stage::default()
            };
        }
        self.ram_reads.store(0, RELAXED);
        c.warm_stamp.clear();
        self.warm_hits.store(0, RELAXED);
        self.warm_count.store(0, RELAXED);
        self.file_read_bytes.store(0, RELAXED);
        self.file_blob_bytes.store(0, RELAXED);
        self.file_us.store(0, RELAXED);
        c.opened = false;
        c.blobs = 0;
        c.n_layers = 0;
        c.n_expert = 0;
        c.layer_offsets.clear();
        c.layer_blob_bytes.clear();
        self.reads.store(0, RELAXED);
    }

    // ---- the role slices ----------------------------------------------------------------------------------------------

    #[inline]
    fn in_range(&self, layer: i64, expert: i64) -> bool {
        let c = self.c();
        layer >= 0 && expert >= 0 && layer < c.n_layers && expert < c.n_expert
    }

    /// `copy_from_files`: the blob [gate | up | down] from the three role slices (unbuffered first when the tier is).
    unsafe fn copy_from_files(&self, layer: i64, expert: i64, dst: *mut u8) -> bool {
        if dst.is_null() || !self.in_range(layer, expert) {
            return false;
        }
        let c = self.c();
        if !c.direct.is_empty() {
            let f = Fill {
                v: 0,
                layer,
                e: expert,
                dst,
            };
            if self.read_direct(&[f]) {
                return true;
            }
            self.direct_fallbacks.fetch_add(1, RELAXED);
        }
        let mut at = 0u64;
        for r in 0..3 {
            let i = (3 * layer) as usize + r;
            let per = c.role_bytes[i];
            std::ptr::copy_nonoverlapping(
                c.role_ptr[i].add((expert as u64 * per) as usize),
                dst.add(at as usize),
                per as usize,
            );
            at += per;
        }
        true
    }

    /// `read_direct`: the fills' role windows read unbuffered (pdirect_ when the tier is buffered but the prompt path
    /// reads unbuffered).
    unsafe fn read_direct(&self, fills: &[Fill]) -> bool {
        const SECTOR: u64 = 4096;
        let c = self.c();
        let fds: &[c_int] = if c.direct.is_empty() {
            &c.pdirect
        } else {
            &c.direct
        };
        if fds.is_empty() {
            return false;
        }
        let mut win: Vec<DirectWindow> = Vec::with_capacity(fills.len() * 3);
        for f in fills {
            if f.dst.is_null() || !self.in_range(f.layer, f.e) {
                return false;
            }
            let mut at = 0u64;
            for r in 0..3 {
                let i = (3 * f.layer) as usize + r;
                let per = c.role_bytes[i];
                let m = &c.maps[c.role_file[i] as usize];
                let off = (c.role_ptr[i] as u64 - m.base as u64) + f.e as u64 * per;
                let a0 = off / SECTOR * SECTOR;
                let a1 = (off + per + SECTOR - 1) / SECTOR * SECTOR;
                win.push(DirectWindow {
                    file: c.role_file[i],
                    a0,
                    size: a1 - a0,
                    skip: off - a0,
                    n: per,
                    at,
                    dst: f.dst,
                    req: 0,
                    in_req: 0,
                });
                at += per;
            }
        }
        read_direct_windows(&mut win, fds)
    }

    // ---- the stage pool -----------------------------------------------------------------------------------------------

    /// `claim_stage` (stage lock held): Ok(true, v) when the blob is there or being filled; Ok(false, v) when the
    /// caller must fill buffer `v`; Err when no buffer could be allocated.
    fn claim_stage(&self, st: &mut Stage, key: i64) -> Option<(bool, usize)> {
        st.seq += 1;
        let seq = st.seq;
        if let Some(&v) = st.of.get(&key) {
            st.epoch[v] = st.epoch_now;
            st.used[v] = seq;
            return Some((true, v));
        }
        let mut v = st.buf.len();
        let mut oldest = u64::MAX;
        for i in 0..st.buf.len() {
            if !st.busy[i]
                && (st.epoch[i] + STAGE_AGE <= st.epoch_now || st.used[i] + MAX_STAGE_SEQ <= seq)
                && st.used[i] < oldest
            {
                oldest = st.used[i];
                v = i;
            }
        }
        if v == st.buf.len() {
            let want_pin = stage_pin_wanted();
            let blob = st.blob as usize;
            let mut p: *mut c_void = std::ptr::null_mut();
            let mut pin_err = hip_sys::hipSuccess;
            if want_pin {
                pin_err =
                    unsafe { hip_sys::hipHostMalloc(&mut p, blob, hip_sys::hipHostMallocDefault) };
            }
            let pinned = want_pin && pin_err == hip_sys::hipSuccess && !p.is_null();
            if !pinned {
                unsafe {
                    let _ = hip_sys::hipGetLastError();
                }
            }
            let sb = if pinned {
                StageBuf {
                    ptr: p.cast(),
                    len: blob,
                    pinned: true,
                }
            } else {
                let mut v = std::mem::ManuallyDrop::new(vec![0u8; blob]);
                StageBuf {
                    ptr: v.as_mut_ptr(),
                    len: blob,
                    pinned: false,
                }
            };
            st.idx.insert(sb.ptr as usize, st.buf.len());
            st.buf.push(sb);
            st.ev.push(std::ptr::null_mut());
            st.ev_live.push(false);
            if pinned && !st.pin_said {
                st.pin_said = true;
                eprintln!("FileExpertSource: the stage buffers are pinned (cudaHostAlloc)");
            } else if !pinned && want_pin && !st.pin_failed_said {
                st.pin_failed_said = true;
                eprintln!(
                    "FileExpertSource: cudaHostAlloc failed for a stage buffer ({}) - pageable it is",
                    hip_err_str(pin_err)
                );
            }
            st.key.push(-1);
            st.epoch.push(0);
            st.used.push(0);
            st.busy.push(false);
            if st.buf.len() == 512 && !st.grew {
                st.grew = true;
                eprintln!(
                    "FileExpertSource: {} blobs assembled from the GGUF are in use at once ({:.2} GiB)",
                    st.buf.len(),
                    st.buf.len() as f64 * st.blob as f64 / 1073741824.0
                );
            }
        } else {
            let old = st.key[v];
            st.of.remove(&old);
            // #1237: a page-locked buffer is refilled only once the last copy queued out of it has finished
            if v < st.ev_live.len() && st.ev_live[v] {
                st.ev_live[v] = false;
                let ev = st.ev[v];
                if !ev.is_null() {
                    unsafe {
                        if hip_sys::hipEventQuery(ev) == hip_sys::hipErrorNotReady {
                            self.stage_fence_waits.fetch_add(1, RELAXED);
                            let _ = hip_sys::hipEventSynchronize(ev);
                        }
                        let _ = hip_sys::hipGetLastError();
                    }
                }
            }
        }
        st.key[v] = key;
        st.epoch[v] = st.epoch_now;
        st.used[v] = seq;
        st.busy[v] = true;
        st.of.insert(key, v);
        Some((false, v))
    }

    /// `fill_stage`: fills buffer `v` outside the lock, then publishes it.
    unsafe fn fill_stage(&self, v: usize, layer: i64, expert: i64, dst: *mut u8) -> bool {
        let t0 = Instant::now();
        let ok = self.copy_from_files(layer, expert, dst);
        self.publish_stage(v, layer, ok, us_since(t0));
        ok
    }

    fn publish_stage(&self, v: usize, layer: i64, ok: bool, us: f64) {
        self.file_us.fetch_add(us as u64, RELAXED);
        if ok {
            let b = self.c().layer_blob_bytes[layer as usize];
            self.file_read_bytes.fetch_add(b, RELAXED);
            self.file_blob_bytes.fetch_add(b, RELAXED);
        }
        {
            let mut st = self.lock_stage();
            st.busy[v] = false;
            if !ok {
                let k = st.key[v];
                st.of.remove(&k);
                st.key[v] = -1;
            }
        }
        self.stage_cv.notify_all();
    }

    /// `note_async_read` (#1237): a pinned stage buffer is not refilled before the copy queued out of it on `stream`.
    pub fn note_async_read(&self, blob: *const u8, stream: *mut c_void) {
        if blob.is_null() {
            return;
        }
        let mut st = self.lock_stage();
        let Some(&v) = st.idx.get(&(blob as usize)) else {
            return;
        };
        if !st.buf[v].pinned {
            return;
        }
        let stream: hip_sys::hipStream_t = stream.cast();
        unsafe {
            if st.ev[v].is_null() {
                let mut ev: hip_sys::hipEvent_t = std::ptr::null_mut();
                if hip_sys::hipEventCreateWithFlags(&mut ev, hip_sys::hipEventDisableTiming)
                    != hip_sys::hipSuccess
                {
                    let _ = hip_sys::hipGetLastError();
                    let _ = hip_sys::hipStreamSynchronize(stream);
                    return;
                }
                st.ev[v] = ev;
            }
            if hip_sys::hipEventRecord(st.ev[v], stream) != hip_sys::hipSuccess {
                let _ = hip_sys::hipGetLastError();
                let _ = hip_sys::hipStreamSynchronize(stream);
                return;
            }
        }
        st.ev_live[v] = true;
    }

    fn staged_blob(&self, layer: i64, expert: i64) -> *const u8 {
        let key = layer * self.c().n_expert + expert;
        let (v, dst);
        {
            let mut st = self.lock_stage();
            let Some((have, vv)) = self.claim_stage(&mut st, key) else {
                return std::ptr::null();
            };
            v = vv;
            dst = st.buf[v].ptr;
            if have {
                // another thread (a prefetch, the adaptive tier) is filling it: wait for that
                let st = self
                    .stage_cv
                    .wait_while(st, |s| s.busy[v] && s.key[v] == key)
                    .unwrap_or_else(|e| e.into_inner());
                if st.key[v] != key {
                    return std::ptr::null();
                }
                return dst;
            }
        }
        if unsafe { self.fill_stage(v, layer, expert, dst) } {
            dst
        } else {
            std::ptr::null()
        }
    }

    /// `fill_many`: the fills on `fetch_threads` threads (mapped reads), or in overlapped unbuffered batches of 16.
    fn fill_many(&self, todo: &[Fill]) {
        if todo.is_empty() {
            return;
        }
        let c = self.c();
        if !c.direct.is_empty() {
            let next = AtomicUsize::new(0);
            let work = || loop {
                let at = next.fetch_add(16, RELAXED);
                if at >= todo.len() {
                    break;
                }
                let k = 16.min(todo.len() - at);
                let t0 = Instant::now();
                let ok = unsafe { self.read_direct(&todo[at..at + k]) };
                let us = us_since(t0);
                for f in &todo[at..at + k] {
                    if ok {
                        self.publish_stage(f.v, f.layer, true, us / k as f64);
                    } else {
                        unsafe {
                            let _ = self.fill_stage(f.v, f.layer, f.e, f.dst);
                        }
                    }
                }
            };
            let nt = 4.min(todo.len().div_ceil(16));
            std::thread::scope(|s| {
                for _ in 1..nt {
                    s.spawn(work);
                }
                work();
            });
            return;
        }
        let next = AtomicUsize::new(0);
        let work = || loop {
            let i = next.fetch_add(1, RELAXED);
            if i >= todo.len() {
                break;
            }
            unsafe {
                let _ = self.fill_stage(todo[i].v, todo[i].layer, todo[i].e, todo[i].dst);
            }
        };
        let nt = todo.len().min(c.fetch_threads);
        std::thread::scope(|s| {
            for _ in 1..nt {
                s.spawn(work);
            }
            work();
        });
    }

    // ---- the hand-out ------------------------------------------------------------------------------------------------

    #[inline]
    fn resident_blob_index(&self, index: usize) -> *const u8 {
        let c = self.c();
        if !c.complement_host.is_null() && index < c.complement_offsets.len() {
            let off = c.complement_offsets[index];
            if off != NO_CACHE_COMPLEMENT {
                return unsafe { c.complement_host.add(off as usize) };
            }
        }
        std::ptr::null()
    }

    #[inline]
    fn override_at(&self, index: usize) -> *const u8 {
        let c = self.c();
        if c.override_.is_empty() {
            std::ptr::null()
        } else {
            c.override_[index]
        }
    }

    /// `FileExpertSource::blob`.
    pub fn blob(&self, layer: i64, expert: i64) -> *const u8 {
        let c = self.c();
        if !c.opened || !self.in_range(layer, expert) {
            return std::ptr::null();
        }
        let index = (layer * c.n_expert + expert) as usize;
        let mut result: *const u8 = std::ptr::null();
        let mut from_files = true;
        if c.complement_ready {
            result = self.resident_blob_index(index);
            if !result.is_null() {
                self.ram_reads.fetch_add(1, RELAXED);
                from_files = false;
            } else {
                let o = self.override_at(index);
                if !o.is_null() {
                    result = o;
                    from_files = false;
                }
            }
        }
        if from_files {
            // staged() is always true for the GGUF in place
            result = self.staged_blob(layer, expert);
            if c.complement_ready && !result.is_null() {
                self.file_reads.fetch_add(1, RELAXED);
            }
        }
        if !result.is_null() {
            self.reads.fetch_add(1, RELAXED);
        }
        result
    }

    /// `FileExpertSource::blob_stable` (the io-prefetch branch is not ported: it is `blob`).
    pub fn blob_stable(&self, layer: i64, expert: i64) -> *const u8 {
        self.blob(layer, expert)
    }

    /// `FileExpertSource::copy_blob`.
    pub unsafe fn copy_blob(&self, layer: i64, expert: i64, dst: *mut u8) -> bool {
        let c = self.c();
        if !c.opened || dst.is_null() || !self.in_range(layer, expert) {
            return false;
        }
        let index = (layer * c.n_expert + expert) as usize;
        let bytes = c.layer_blob_bytes[layer as usize];
        if c.complement_ready {
            let mut held = self.resident_blob_index(index);
            if held.is_null() {
                held = self.override_at(index);
            }
            if !held.is_null() {
                std::ptr::copy_nonoverlapping(held, dst, bytes as usize);
                return true;
            }
        }
        let t0 = Instant::now();
        if !self.copy_from_files(layer, expert, dst) {
            return false;
        }
        self.file_us.fetch_add(us_since(t0) as u64, RELAXED);
        self.file_read_bytes.fetch_add(bytes, RELAXED);
        true
    }

    /// `FileExpertSource::copy_blobs` (pp-opt: one merged unbuffered batch when the reads are unbuffered).
    pub unsafe fn copy_blobs(&self, layers: &[i32], experts: &[i32], dst: &[*mut u8]) -> bool {
        let n = layers.len();
        let c = self.c();
        if (c.direct.is_empty() && c.pdirect.is_empty()) || n <= 1 {
            for i in 0..n {
                if !self.copy_blob(layers[i] as i64, experts[i] as i64, dst[i]) {
                    return false;
                }
            }
            return true;
        }
        let mut fills = Vec::with_capacity(n);
        let mut bytes = 0u64;
        for i in 0..n {
            let (l, e) = (layers[i] as i64, experts[i] as i64);
            if !c.opened || dst[i].is_null() || !self.in_range(l, e) {
                return false;
            }
            let index = (l * c.n_expert + e) as usize;
            if c.complement_ready
                && (!self.resident_blob_index(index).is_null()
                    || !self.override_at(index).is_null())
            {
                if !self.copy_blob(l, e, dst[i]) {
                    return false;
                }
                continue;
            }
            fills.push(Fill {
                v: 0,
                layer: l,
                e,
                dst: dst[i],
            });
            bytes += c.layer_blob_bytes[l as usize];
        }
        if fills.is_empty() {
            return true;
        }
        let t0 = Instant::now();
        if !self.read_direct(&fills) {
            for f in &fills {
                if !self.copy_from_files(f.layer, f.e, f.dst) {
                    return false;
                }
            }
        }
        self.file_us.fetch_add(us_since(t0) as u64, RELAXED);
        self.file_read_bytes.fetch_add(bytes, RELAXED);
        true
    }

    /// `FileExpertSource::transient`.
    pub fn transient(&self, layer: i64, expert: i64) -> bool {
        let c = self.c();
        if (c.role_ptr.is_empty() && c.direct.is_empty()) || !self.in_range(layer, expert) {
            return false;
        }
        let index = (layer * c.n_expert + expert) as usize;
        if c.complement_ready && !self.resident_blob_index(index).is_null() {
            return false;
        }
        self.override_at(index).is_null()
    }

    /// `FileExpertSource::cached_share` (#1353): mincore over an even sample of the transient experts' slices.
    pub fn cached_share(&self, samples: i64) -> f64 {
        let c = self.c();
        if c.role_ptr.is_empty() || samples <= 0 || c.n_layers <= 0 || c.n_expert <= 0 {
            return -1.0;
        }
        let total = c.n_layers * c.n_expert;
        let n = samples.min(total);
        let (mut pages, mut resident) = (0u64, 0u64);
        let mut vec: Vec<u8> = Vec::new();
        for s in 0..n {
            let idx = (total - 1).min((s * total) / n + (s * 7919) % 1.max(total / n));
            let (layer, expert) = (idx / c.n_expert, idx % c.n_expert);
            if !self.transient(layer, expert) {
                continue;
            }
            for r in 0..3 {
                let i = (3 * layer) as usize + r;
                if c.role_ptr[i].is_null() || c.role_bytes[i] == 0 {
                    continue;
                }
                let p = unsafe { c.role_ptr[i].add((expert as u64 * c.role_bytes[i]) as usize) }
                    as usize;
                let a = p & !4095usize;
                let len = p + c.role_bytes[i] as usize - a;
                let np = len.div_ceil(4096);
                vec.clear();
                vec.resize(np, 0);
                if unsafe { mincore(a as *mut c_void, len, vec.as_mut_ptr()) } != 0 {
                    return -1.0;
                }
                pages += np as u64;
                resident += vec.iter().map(|&v| (v & 1) as u64).sum::<u64>();
            }
        }
        if pages > 0 {
            resident as f64 / pages as f64
        } else {
            -1.0
        }
    }

    /// `FileExpertSource::begin_layer`: a new layer ages the stage buffers by one epoch.
    pub fn begin_layer(&self, layer: i64) {
        if !self.staged_mode() {
            return;
        }
        let mut st = self.lock_stage();
        if layer != st.last_layer {
            st.epoch_now += 1;
            st.last_layer = layer;
        }
    }

    /// `FileExpertSource::prefetch(layer, experts, n)`: the layer's experts outside the RAM copy, read together.
    pub fn prefetch(&self, layer: i64, experts: &[i64]) {
        let c = self.c();
        if !self.staged_mode() || experts.is_empty() || layer < 0 || layer >= c.n_layers {
            return;
        }
        let mut todo = Vec::new();
        {
            let mut st = self.lock_stage();
            for &e in experts {
                if e < 0 || e >= c.n_expert {
                    continue;
                }
                let index = (layer * c.n_expert + e) as usize;
                if c.complement_ready && !self.resident_blob_index(index).is_null() {
                    continue;
                }
                if !self.override_at(index).is_null() {
                    continue;
                }
                if let Some((have, v)) = self.claim_stage(&mut st, layer * c.n_expert + e) {
                    if !have {
                        todo.push(Fill {
                            v,
                            layer,
                            e,
                            dst: st.buf[v].ptr,
                        });
                        if !c.warm_stamp.is_empty() {
                            let s = c.warm_stamp[index].load(RELAXED);
                            if s != 0 && s as u64 + 3 >= st.epoch_now + 1 {
                                self.warm_hits.fetch_add(1, RELAXED);
                            }
                        }
                    }
                }
            }
        }
        self.fill_many(&todo);
    }

    /// `FileExpertSource::prefetch_pairs` (unbuffered only: up to 64 pairs staged ahead of the fill).
    pub fn prefetch_pairs(&self, pairs: &[(i32, i32)]) {
        let c = self.c();
        if c.direct.is_empty() || pairs.is_empty() {
            return;
        }
        let mut todo = Vec::new();
        {
            let mut st = self.lock_stage();
            for &(l, e) in pairs.iter().take(64) {
                let (l, e) = (l as i64, e as i64);
                if l < 0 || e < 0 || l >= c.n_layers || e >= c.n_expert {
                    continue;
                }
                if let Some((have, v)) = self.claim_stage(&mut st, l * c.n_expert + e) {
                    if !have {
                        todo.push(Fill {
                            v,
                            layer: l,
                            e,
                            dst: st.buf[v].ptr,
                        });
                    }
                }
            }
        }
        self.fill_many(&todo);
    }

    /// `FileExpertSource::advise_pairs`: MADV_WILLNEED on the pairs' slices (mapped reads only).
    pub fn advise_pairs(&self, pairs: &[(i32, i32)]) -> bool {
        let c = self.c();
        if !c.opened || !c.direct.is_empty() || !read_ahead_enabled() {
            return false;
        }
        for &(l, e) in pairs {
            let (l, e) = (l as i64, e as i64);
            if l < 0 || e < 0 || l >= c.n_layers || e >= c.n_expert {
                continue;
            }
            for r in 0..3 {
                let k = (3 * l) as usize + r;
                unsafe {
                    advise_willneed(
                        c.role_ptr[k].add((e as u64 * c.role_bytes[k]) as usize),
                        c.role_bytes[k],
                    )
                };
            }
        }
        true
    }

    /// `FileExpertSource::warm`: the predicted experts' pages asked for (MADV_WILLNEED), stamped for the hit count.
    pub fn warm(&self, layer: i64, experts: &[i64]) {
        let c = self.c();
        if c.role_ptr.is_empty() || experts.is_empty() || layer < 0 || layer >= c.n_layers {
            return;
        }
        let stamp = {
            let st = self.lock_stage();
            st.epoch_now as u32 + 1
        };
        for &e in experts {
            if e < 0 || e >= c.n_expert {
                continue;
            }
            let index = (layer * c.n_expert + e) as usize;
            if c.complement_ready && !self.resident_blob_index(index).is_null() {
                continue;
            }
            if !c.warm_stamp.is_empty() {
                c.warm_stamp[index].store(stamp, RELAXED);
            }
            self.warm_count.fetch_add(1, RELAXED);
            for r in 0..3 {
                let i = (3 * layer) as usize + r;
                let p =
                    unsafe { c.role_ptr[i].add((e as u64 * c.role_bytes[i]) as usize) } as usize;
                let a = p & !4095usize;
                unsafe {
                    let _ = madvise(
                        a as *mut c_void,
                        p + c.role_bytes[i] as usize - a,
                        MADV_WILLNEED,
                    );
                }
            }
        }
    }

    // ---- the RAM copy's residency -----------------------------------------------------------------------------------

    pub fn has_resident(&self, layer: i64, expert: i64) -> bool {
        let c = self.c();
        if !c.complement_ready || !self.in_range(layer, expert) {
            return false;
        }
        !self
            .resident_blob_index((layer * c.n_expert + expert) as usize)
            .is_null()
    }

    pub fn resident_blob(&self, layer: i64, expert: i64) -> *const u8 {
        let c = self.c();
        if !c.complement_ready || c.complement_host.is_null() || !self.in_range(layer, expert) {
            return std::ptr::null();
        }
        let index = (layer * c.n_expert + expert) as usize;
        if index >= c.complement_offsets.len() {
            return std::ptr::null();
        }
        self.resident_blob_index(index)
    }

    pub fn pinned(&self, layer: i64, expert: i64) -> bool {
        let c = self.c();
        if !c.complement_ready
            || !c.complement_pinned
            || c.complement_host.is_null()
            || !self.in_range(layer, expert)
        {
            return false;
        }
        let index = (layer * c.n_expert + expert) as usize;
        let at_index = if !c.soft_in.is_empty() && c.soft_in[index] != usize::MAX {
            c.soft_in[index]
        } else {
            index
        };
        if at_index >= c.complement_offsets.len()
            || c.complement_offsets[at_index] == NO_CACHE_COMPLEMENT
        {
            return false;
        }
        !c.complement_partial
            || c.complement_offsets[at_index] + c.layer_blob_bytes[layer as usize]
                <= c.complement_pin_limit
    }

    pub fn device_alias(&self, layer: i64, expert: i64) -> *const u8 {
        let c = self.c();
        if !self.pinned(layer, expert) || c.complement_device.is_null() {
            return std::ptr::null();
        }
        let index = (layer * c.n_expert + expert) as usize;
        if !c.soft_in.is_empty() && c.soft_in[index] != usize::MAX {
            let o = self.override_at(index);
            return if o.is_null() || c.xstage_dev.is_null() {
                std::ptr::null()
            } else {
                unsafe { c.xstage_dev.add(o as usize - c.xstage as usize) }
            };
        }
        unsafe {
            c.complement_device
                .add(c.complement_offsets[index] as usize)
        }
    }

    pub fn pcie_layer(&self, layer: i64) -> bool {
        let c = self.c();
        if c.complement_ready && c.complement_pinned && !c.complement_device.is_null() {
            return layer >= 0 && layer < c.n_layers;
        }
        !self.device_alias(layer, 0).is_null()
    }

    // ---- the unbuffered tier (#286 / #577) ----------------------------------------------------------------------------

    fn open_direct(&self, why: &mut String) -> bool {
        let c = unsafe { self.cm() };
        for path in c.paths.clone() {
            let fd = open_path(&path, O_RDONLY | O_DIRECT | O_CLOEXEC);
            if fd < 0 {
                why.push_str(&format!(
                    "; cannot open {path} with O_DIRECT ({}), read through the file cache",
                    strerror_str(errno())
                ));
                for d in c.direct.drain(..) {
                    unsafe { close(d) };
                }
                return false;
            }
            c.direct.push(fd);
        }
        true
    }

    /// `FileExpertSource::set_unbuffered`.
    pub fn set_unbuffered(&self, ram_bytes: u64, why: &mut String) -> bool {
        let c = self.c();
        if !c.opened || c.paths.is_empty() || !c.direct.is_empty() {
            *why = if !c.direct.is_empty() {
                "already unbuffered"
            } else {
                "no expert files open"
            }
            .into();
            return !c.direct.is_empty();
        }
        let experts = self.expert_bytes();
        let arena = ram_bytes.min(experts);
        if !file_tier_unbuffered(arena, experts - arena, why) {
            return false;
        }
        self.open_direct(why)
    }

    /// `FileExpertSource::recheck_unbuffered`.
    pub fn recheck_unbuffered(&self, why: &mut String) -> bool {
        if let Some(e) = env("STRATA_UNBUFFERED_LOAD") {
            if !e.is_empty() {
                *why = format!("STRATA_UNBUFFERED_LOAD={e}");
                return !self.c().direct.is_empty();
            }
        }
        let c = self.c();
        if !c.opened || c.paths.is_empty() {
            *why = "no expert files open".into();
            return false;
        }
        let experts = self.expert_bytes();
        let read = experts.saturating_sub(c.complement_bytes);
        let mut w = String::new();
        let ub = file_tier_unbuffered(0, read, &mut w);
        *why = format!(
            "re-checked with the RAM copy built ({:.2} GiB): {w}",
            c.complement_bytes as f64 / 1073741824.0
        );
        if ub == !c.direct.is_empty() {
            return ub;
        }
        if ub {
            return self.open_direct(why);
        }
        let c = unsafe { self.cm() };
        for d in c.direct.drain(..) {
            unsafe { close(d) };
        }
        false
    }

    /// `FileExpertSource::open_prompt_direct` (STRATA_PROMPT_DIRECT=1).
    pub fn open_prompt_direct(&self, why: &mut String) -> bool {
        let c = unsafe { self.cm() };
        if !c.opened || c.paths.is_empty() {
            *why = "no expert files open".into();
            return false;
        }
        if !c.direct.is_empty() {
            *why = "the file tier reads unbuffered already".into();
            return false;
        }
        if !c.pdirect.is_empty() {
            return true;
        }
        for path in c.paths.clone() {
            let fd = open_path(&path, O_RDONLY | O_DIRECT | O_CLOEXEC);
            if fd < 0 {
                *why = format!(
                    "cannot open {path} with O_DIRECT ({})",
                    strerror_str(errno())
                );
                for d in c.pdirect.drain(..) {
                    unsafe { close(d) };
                }
                return false;
            }
            c.pdirect.push(fd);
        }
        true
    }

    // ---- the resident RAM copy (pin_cache_complement) ------------------------------------------------------------------

    /// `FileExpertSource::pin_cache_complement`: the experts no GPU cache holds (by profile rank within a budget, plus
    /// the prompt path's lend region and the split stages' regions as far as RAM allows) copied into one host arena -
    /// page-locked and mapped, paced (#1250), partially registered (CS-T) or locked pageable - so no decode step reads
    /// the files for them.
    pub fn pin_cache_complement(
        &self,
        cache: &CacheView,
        pin: bool,
        additional_gpu_pairs: &[(i32, i32)],
        mut lend_from_slot: i64,
        headroom_bytes: u64,
        mut budget_bytes: u64,
        rank: Option<&[(i32, i32)]>,
    ) -> Result<(), String> {
        let pin_t0 = Instant::now();
        let (n_layers, n_expert) = (self.c().n_layers, self.c().n_expert);
        let layer_blob_bytes = self.c().layer_blob_bytes.clone();
        if !self.c().opened {
            return Err(
                "FileExpertSource: open the mapped experts before pinning a complement".into(),
            );
        }
        if self.c().complement_ready {
            return Err("FileExpertSource: the cache complement is already pinned".into());
        }
        if !cache.valid {
            return Err("FileExpertSource: the GPU expert cache is not open".into());
        }
        if cache.fills != cache.resident {
            return Err("FileExpertSource: the GPU expert cache is not fully filled".into());
        }
        let sync = unsafe { hip_sys::hipDeviceSynchronize() };
        if sync != hip_sys::hipSuccess {
            unsafe {
                let _ = hip_sys::hipGetLastError();
            }
            return Err(format!(
                "FileExpertSource: GPU expert cache is not ready: {}",
                hip_err_str(sync)
            ));
        }
        let n_slots = cache.slots;
        let mut primary_gpu_pairs = Vec::new();
        let mut pair_slot = Vec::new();
        let mut slot_bytes = vec![0u64; n_slots.max(0) as usize];
        for layer in 0..n_layers {
            for expert in 0..n_expert {
                let slot = cache.slot_of[(layer * n_expert + expert) as usize];
                if slot == -1 {
                    continue;
                }
                primary_gpu_pairs.push((layer as i32, expert as i32));
                pair_slot.push(slot);
                if slot >= 0 && (slot as i64) < n_slots {
                    slot_bytes[slot as usize] = layer_blob_bytes[layer as usize];
                }
            }
        }
        let mut offsets = Vec::new();
        let mut bytes = 0u64;
        make_cache_complement_plan(
            n_layers,
            n_expert,
            &layer_blob_bytes,
            &primary_gpu_pairs,
            additional_gpu_pairs,
            &mut offsets,
            &mut bytes,
        )?;
        let what_fits = budget_bytes == u64::MAX;
        let mut budget_physical = 0u64;
        let mut budget_commit = u64::MAX;
        if budget_bytes > 0 {
            let Some((physical, commit)) = available_memory_bytes() else {
                return Err(
                    "FileExpertSource: cannot determine available memory for --resident-budget-gib"
                        .into(),
                );
            };
            budget_commit = commit;
            budget_physical = physical;
            let available = physical.min(budget_commit);
            let limit = if budget_commit < physical {
                "commit"
            } else {
                "RAM"
            };
            let clamped =
                clamp_resident_budget(budget_bytes, physical, budget_commit, headroom_bytes);
            if budget_bytes > clamped {
                if what_fits {
                    eprintln!(
                        "FileExpertSource: room for the complement: {:.2} GiB ({:.2} GiB available {limit} minus {:.0} \
                         GiB headroom and a 0.25 GiB margin)",
                        clamped as f64 / 1073741824.0,
                        available as f64 / 1073741824.0,
                        headroom_bytes as f64 / 1073741824.0
                    );
                } else {
                    eprintln!(
                        "FileExpertSource: --resident-budget-gib {:.2} exceeds available {limit} ({:.2} GiB minus {:.0} \
                         GiB headroom and a 0.25 GiB margin): {:.2} GiB",
                        budget_bytes as f64 / 1073741824.0,
                        available as f64 / 1073741824.0,
                        headroom_bytes as f64 / 1073741824.0,
                        clamped as f64 / 1073741824.0
                    );
                }
                budget_bytes = clamped;
            }
            let mut ranked = vec![NO_CACHE_COMPLEMENT; offsets.len()];
            let mut at = 0u64;
            let mut held = 0i64;
            if let Some(rank) = rank {
                for &(l, e) in rank {
                    if l < 0 || e < 0 || l as i64 >= n_layers || e as i64 >= n_expert {
                        continue;
                    }
                    let i = l as usize * n_expert as usize + e as usize;
                    if offsets[i] == NO_CACHE_COMPLEMENT || ranked[i] != NO_CACHE_COMPLEMENT {
                        continue;
                    }
                    let b = layer_blob_bytes[l as usize];
                    if b > budget_bytes.wrapping_sub(at) {
                        continue;
                    }
                    ranked[i] = at;
                    at += b;
                    held += 1;
                }
            }
            if what_fits && held == 0 {
                return Err(
                    "FileExpertSource: available memory cannot hold any expert of the complement"
                        .into(),
                );
            }
            eprintln!(
                "FileExpertSource: RAM budget {:.2} GiB: {held} of the {:.2} GiB of experts the GPU cache does not hold, \
                 by profile rank; the rest are read from the files",
                budget_bytes as f64 / 1073741824.0,
                bytes as f64 / 1073741824.0
            );
            offsets = ranked;
            bytes = at;
            lend_from_slot = -1;
        }

        let lend =
            lend_from_slot >= 0 && lend_from_slot < n_slots && additional_gpu_pairs.is_empty();
        let mut budget = u64::MAX;
        if bytes > 0 || lend {
            let (physical, commit) = if budget_physical != 0 {
                (budget_physical, budget_commit)
            } else {
                match available_memory_bytes() {
                    Some(x) => x,
                    None => {
                        return Err("FileExpertSource: cannot determine available memory for the resident-memory \
                                    safety check"
                            .into())
                    }
                }
            };
            let available = physical.min(commit);
            budget = if available > headroom_bytes {
                available - headroom_bytes
            } else {
                0
            };
            if bytes > budget {
                let mut err = format!(
                    "FileExpertSource: resident complement {:.2} GiB exceeds available {} ({:.2} GiB) minus the {:.0} GiB \
                     safety headroom",
                    bytes as f64 / 1073741824.0,
                    if commit < physical { "commit" } else { "RAM" },
                    available as f64 / 1073741824.0,
                    headroom_bytes as f64 / 1073741824.0
                );
                if let Ok(mt) = std::fs::read_to_string("/proc/meminfo") {
                    err.push_str(&page_cache_hint(&mt));
                }
                return Err(err);
            }
        }
        let mut keep_from = n_slots;
        if lend {
            keep_from = choose_resident_keep_from(&slot_bytes, bytes, budget, lend_from_slot);
            if keep_from < 0 {
                keep_from = n_slots;
            }
            if keep_from < n_slots {
                let core: Vec<(i32, i32)> = primary_gpu_pairs
                    .iter()
                    .zip(&pair_slot)
                    .filter(|(_, &s)| (s as i64) < keep_from)
                    .map(|(p, _)| *p)
                    .collect();
                make_cache_complement_plan(
                    n_layers,
                    n_expert,
                    &layer_blob_bytes,
                    &core,
                    additional_gpu_pairs,
                    &mut offsets,
                    &mut bytes,
                )?;
            }
        }
        let mut stage_lent = 0i64;
        let mut stage_lent_bytes = 0u64;
        let stage_lend_regions = self.c().stage_lend_regions.clone();
        if !stage_lend_regions.is_empty() {
            if budget == u64::MAX {
                let (physical, commit) = if budget_physical != 0 {
                    (budget_physical, budget_commit)
                } else {
                    match available_memory_bytes() {
                        Some(x) => x,
                        None => {
                            return Err("FileExpertSource: cannot determine available memory for the stage lend \
                                        regions"
                                .into())
                        }
                    }
                };
                let available = physical.min(commit);
                budget = if available > headroom_bytes {
                    available - headroom_bytes
                } else {
                    0
                };
            }
            let mut cap = budget;
            if budget_bytes > 0 && budget_bytes != u64::MAX && budget_bytes < cap {
                cap = budget_bytes;
            }
            let before = bytes;
            stage_lent = append_stage_lend_regions(
                n_layers,
                n_expert,
                &layer_blob_bytes,
                &mut offsets,
                &mut bytes,
                cap,
                &stage_lend_regions,
            )?;
            stage_lent_bytes = bytes - before;
        }

        let mut arena: *mut u8 = std::ptr::null_mut();
        let mut device: *const u8 = std::ptr::null();
        let mut pinned_ok = false;
        let mut paced = false;
        let mut locked = 0u64;
        let mut partial_pin = 0u64;
        let mut lock_off = 0u64;
        let mut aligned = false;
        let mut note = String::new();
        let release = |arena: *mut u8,
                       pinned_ok: bool,
                       partial_pin: u64,
                       locked: u64,
                       lock_off: u64| unsafe {
            if arena.is_null() {
                return;
            }
            if pinned_ok {
                let _ = hip_sys::hipHostFree(arena.cast());
            } else {
                if partial_pin > 0 {
                    let _ = hip_sys::hipHostUnregister(arena.cast());
                }
                if locked > 0 {
                    unlock_resident(arena.add(lock_off as usize), locked);
                }
                free(arena.cast());
            }
        };
        // #1250: the cached pages of the model files this source maps, given back
        let drop_file_cache = || unsafe {
            for m in &self.c().maps {
                if !m.base.is_null() && m.bytes > 0 {
                    let _ = madvise(m.base as *mut c_void, m.bytes as usize, MADV_DONTNEED);
                }
                if m.fd >= 0 {
                    let _ = posix_fadvise(m.fd, 0, 0, POSIX_FADV_DONTNEED);
                }
            }
        };
        if bytes > 0 {
            eprintln!(
                "FileExpertSource: allocating {:.2} GiB {} cache complement",
                bytes as f64 / 1073741824.0,
                if pin {
                    "page-locked"
                } else {
                    "pageable resident"
                }
            );
            let pace_reserve = match env("STRATA_PIN_RESERVE_GIB") {
                Some(v) if c_atof(&v) > 0.0 => (c_atof(&v) * 1073741824.0) as u64,
                _ => headroom_bytes.max(2u64 << 30),
            };
            if pin {
                static GUARD: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
                let guard =
                    *GUARD.get_or_init(|| env("STRATA_PIN_GUARD").map(|v| c_atoi(&v)).unwrap_or(1));
                if guard == 2 {
                    paced = true;
                    eprintln!("FileExpertSource: STRATA_PIN_GUARD=2: page-locking the complement in steps");
                } else if guard != 0 {
                    if let Some(free_now) = free_memory_bytes() {
                        if pin_depends_on_reclaim(free_now, bytes, pace_reserve) {
                            drop_file_cache();
                            let before = free_now;
                            let mut free_after = free_now;
                            if let Some(f) = free_memory_bytes() {
                                free_after = f;
                                eprintln!(
                                    "FileExpertSource: {:.2} GiB really free for a {:.2} GiB page-locked complement; the \
                                     model files' cached pages were released: {:.2} GiB free",
                                    before as f64 / 1073741824.0,
                                    bytes as f64 / 1073741824.0,
                                    f as f64 / 1073741824.0
                                );
                            }
                            paced = pin_depends_on_reclaim(free_after, bytes, pace_reserve);
                            if paced {
                                eprintln!(
                                    "FileExpertSource: the rest comes from reclaimed cache of other files: page-locking \
                                     it in steps, as far as the RAM allows (#1250)"
                                );
                            }
                        }
                    }
                }
            }
            if pin && !paced {
                let mut p: *mut c_void = std::ptr::null_mut();
                // cudaHostAlloc(Mapped | Portable) = strata_host_alloc: a mapped allocation is made coherent on HIP
                let allocated = unsafe {
                    hip_sys::hipHostMalloc(
                        &mut p,
                        bytes as usize,
                        hip_sys::hipHostMallocMapped
                            | hip_sys::hipHostMallocPortable
                            | hip_sys::hipHostMallocCoherent,
                    )
                };
                if allocated == hip_sys::hipSuccess {
                    arena = p.cast();
                    let mut alias: *mut c_void = std::ptr::null_mut();
                    let aliased = unsafe { hip_sys::hipHostGetDevicePointer(&mut alias, p, 0) };
                    if aliased == hip_sys::hipSuccess && !alias.is_null() {
                        device = alias as *const u8;
                        pinned_ok = true;
                        note = "page-locked and mapped".into();
                    } else {
                        note = format!("no device alias ({})", hip_err_str(aliased));
                        unsafe {
                            let _ = hip_sys::hipGetLastError();
                            let _ = hip_sys::hipHostFree(p);
                        }
                        arena = std::ptr::null_mut();
                    }
                } else {
                    note = format!("page-locking refused ({})", hip_err_str(allocated));
                    unsafe {
                        let _ = hip_sys::hipGetLastError();
                    }
                    arena = std::ptr::null_mut();
                }
            }
            if arena.is_null() {
                if !note.is_empty() {
                    eprintln!("FileExpertSource: {note}; trying pageable memory");
                }
                if paced {
                    let page = 4096usize;
                    let mut a: *mut c_void = std::ptr::null_mut();
                    if unsafe {
                        posix_memalign(&mut a, page, (bytes as usize).div_ceil(page) * page)
                    } == 0
                    {
                        arena = a.cast();
                        aligned = true;
                    }
                } else {
                    arena = unsafe { malloc(bytes as usize) }.cast();
                }
                if arena.is_null() {
                    let e = errno();
                    let mut err = format!(
                        "FileExpertSource: pageable resident complement allocation failed (malloc errno {e}: {})",
                        strerror_str(e)
                    );
                    if !note.is_empty() {
                        err.push_str(&format!("; {note}"));
                    }
                    return Err(err);
                }
                if pin {
                    static PARTIAL_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    let partial_on = *PARTIAL_ON.get_or_init(|| {
                        env("STRATA_PARTIAL_PIN")
                            .map(|v| c_atoi(&v) != 0)
                            .unwrap_or(false)
                    });
                    static PIN_CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                    let pin_cap = *PIN_CAP.get_or_init(|| {
                        let g = env("STRATA_PARTIAL_PIN_GIB")
                            .map(|v| c_atof(&v))
                            .filter(|&g| g > 0.0)
                            .unwrap_or(24.0);
                        (g * 1073741824.0) as u64
                    });
                    // a blob that would straddle `want`: the registration is cut at its start
                    let cut_at = |want: u64| -> u64 {
                        for (i, &o) in offsets.iter().enumerate() {
                            if o == NO_CACHE_COMPLEMENT {
                                continue;
                            }
                            let b = layer_blob_bytes[i / n_expert as usize];
                            if o < want && o + b > want {
                                return o;
                            }
                        }
                        want
                    };
                    if paced {
                        const STEP: u64 = 1 << 30;
                        const PAGE: u64 = 4096;
                        let mut got = 0u64;
                        let mut stop: Option<&str> = None;
                        while got < bytes {
                            let step = STEP.min(bytes - got);
                            match available_memory_bytes() {
                                Some((avail_now, _))
                                    if pin_step_fits(avail_now, step, pace_reserve) => {}
                                _ => {
                                    stop = Some("the RAM left would fall under the reserve");
                                    break;
                                }
                            }
                            let mut at = got;
                            while at < got + step {
                                unsafe { std::ptr::write_volatile(arena.add(at as usize), 0) };
                                at += PAGE;
                            }
                            got += step;
                        }
                        let mut w = got;
                        if w < bytes {
                            w = cut_at(got);
                        }
                        if w > 0 {
                            unsafe {
                                if hip_sys::hipHostRegister(
                                    arena.cast(),
                                    w as usize,
                                    hip_sys::hipHostRegisterMapped
                                        | hip_sys::hipHostRegisterPortable,
                                ) == hip_sys::hipSuccess
                                {
                                    let mut alias: *mut c_void = std::ptr::null_mut();
                                    if hip_sys::hipHostGetDevicePointer(&mut alias, arena.cast(), 0)
                                        == hip_sys::hipSuccess
                                        && !alias.is_null()
                                    {
                                        device = alias as *const u8;
                                        partial_pin = w;
                                    } else {
                                        let _ = hip_sys::hipGetLastError();
                                        let _ = hip_sys::hipHostUnregister(arena.cast());
                                        if stop.is_none() {
                                            stop = Some("no device alias");
                                        }
                                    }
                                } else {
                                    let _ = hip_sys::hipGetLastError();
                                    if stop.is_none() {
                                        stop = Some("the driver refused to page-lock it");
                                    }
                                }
                            }
                        }
                        lock_off = partial_pin;
                        let msg = match stop {
                            Some(s) => format!(
                                "WARNING: only {:.2} of {:.2} GiB could be page-locked ({s}); the rest is resident but \
                                 pageable, so its copies to the GPU go through the CPU (#1250)",
                                partial_pin as f64 / 1073741824.0,
                                bytes as f64 / 1073741824.0
                            ),
                            None => format!(
                                "{:.2} GiB page-locked after taking the pages in steps",
                                partial_pin as f64 / 1073741824.0
                            ),
                        };
                        if !note.is_empty() {
                            note.push_str("; ");
                        }
                        note.push_str(&msg);
                        if stop.is_some() {
                            eprintln!("FileExpertSource: {msg}");
                        }
                    } else if budget_bytes > 0 && partial_on {
                        let step = 2u64 << 30;
                        let mut want = bytes.min(pin_cap);
                        while want >= step {
                            let w = cut_at(want);
                            if w == 0 {
                                break;
                            }
                            let ok = unsafe {
                                hip_sys::hipHostRegister(
                                    arena.cast(),
                                    w as usize,
                                    hip_sys::hipHostRegisterMapped
                                        | hip_sys::hipHostRegisterPortable,
                                ) == hip_sys::hipSuccess
                            };
                            if ok {
                                unsafe {
                                    let mut alias: *mut c_void = std::ptr::null_mut();
                                    if hip_sys::hipHostGetDevicePointer(&mut alias, arena.cast(), 0)
                                        == hip_sys::hipSuccess
                                        && !alias.is_null()
                                    {
                                        device = alias as *const u8;
                                        partial_pin = w;
                                    } else {
                                        let _ = hip_sys::hipGetLastError();
                                        let _ = hip_sys::hipHostUnregister(arena.cast());
                                    }
                                }
                                break;
                            }
                            unsafe {
                                let _ = hip_sys::hipGetLastError();
                            }
                            if want <= step {
                                break;
                            }
                            want = if want > step { want - step } else { 0 };
                        }
                        note.push_str(&format!(
                            "; {:.2} GiB of it registered for the GPU (the hottest)",
                            partial_pin as f64 / 1073741824.0
                        ));
                    }
                    if !paced {
                        lock_off = partial_pin;
                    }
                    let lr = if lock_off < bytes {
                        unsafe { lock_resident(arena.add(lock_off as usize), bytes - lock_off) }
                    } else {
                        LockResult {
                            ok: true,
                            locked_bytes: 0,
                            note: String::new(),
                        }
                    };
                    locked = lr.locked_bytes;
                    if !note.is_empty() {
                        note.push_str("; ");
                    }
                    note.push_str(&lr.note);
                }
            }
        }
        let host: *const u8 = arena;

        // Copied layer by layer on a few threads: the page faults of the mapped file are the cost, and they overlap.
        let next_layer = AtomicI64::new(0);
        let layers_done = AtomicI64::new(0);
        let copied = AtomicU64::new(0);
        let failed = AtomicBool::new(false);
        let fail_msg: Mutex<String> = Mutex::new(String::new());
        let copy_t0 = Instant::now();
        let host_addr = host as usize;
        let offsets_ref = &offsets;
        let fail = |m: &str| {
            let mut g = fail_msg.lock().unwrap_or_else(|e| e.into_inner());
            if g.is_empty() {
                *g = m.to_string();
            }
            failed.store(true, Ordering::SeqCst);
        };
        let worker = || loop {
            let layer = next_layer.fetch_add(1, Ordering::SeqCst);
            if layer >= n_layers || failed.load(Ordering::SeqCst) {
                return;
            }
            let blob_bytes = layer_blob_bytes[layer as usize];
            let mut batch: Vec<Fill> = Vec::new();
            let flush = |batch: &mut Vec<Fill>| -> bool {
                if batch.is_empty() {
                    return true;
                }
                let ok = unsafe { self.read_direct(batch) };
                if !ok {
                    for f in batch.iter() {
                        if unsafe { !self.copy_from_files(f.layer, f.e, f.dst) } {
                            return false;
                        }
                    }
                }
                copied.fetch_add(blob_bytes * batch.len() as u64, Ordering::SeqCst);
                batch.clear();
                true
            };
            let direct = !self.c().direct.is_empty();
            if !direct {
                let ahead: Vec<(i32, i32)> = (0..n_expert)
                    .filter(|&e| {
                        offsets_ref[(layer * n_expert + e) as usize] != NO_CACHE_COMPLEMENT
                    })
                    .map(|e| (layer as i32, e as i32))
                    .collect();
                let _ = self.advise_pairs(&ahead);
            }
            for expert in 0..n_expert {
                let offset = offsets_ref[(layer * n_expert + expert) as usize];
                if offset == NO_CACHE_COMPLEMENT {
                    continue;
                }
                if offset > bytes || blob_bytes > bytes - offset {
                    fail(
                        "FileExpertSource: invalid blob bounds while building the cache complement",
                    );
                    return;
                }
                let dst = (host_addr + offset as usize) as *mut u8;
                if direct {
                    batch.push(Fill {
                        v: 0,
                        layer,
                        e: expert,
                        dst,
                    });
                    if batch.len() == 32 && !flush(&mut batch) {
                        fail("FileExpertSource: an expert could not be read while building the cache complement");
                        return;
                    }
                    continue;
                }
                if unsafe { !self.copy_from_files(layer, expert, dst) } {
                    fail(
                        "FileExpertSource: invalid blob bounds while building the cache complement",
                    );
                    return;
                }
                copied.fetch_add(blob_bytes, Ordering::SeqCst);
            }
            if !flush(&mut batch) {
                fail("FileExpertSource: an expert could not be read while building the cache complement");
                return;
            }
            let done = layers_done.fetch_add(1, Ordering::SeqCst) + 1;
            if done % 8 == 0 || done == n_layers {
                eprintln!(
                    "FileExpertSource: copied cache complement through layer {done}/{n_layers} ({:.2} GiB, {:.0} s)",
                    copied.load(Ordering::SeqCst) as f64 / 1073741824.0,
                    copy_t0.elapsed().as_secs_f64()
                );
            }
        };
        {
            let threads = 1.max(6.min(n_layers)) as usize;
            std::thread::scope(|s| {
                for _ in 1..threads {
                    s.spawn(worker);
                }
                worker();
            });
        }
        if failed.load(Ordering::SeqCst) {
            release(arena, pinned_ok, partial_pin, locked, lock_off);
            return Err(fail_msg.into_inner().unwrap_or_default());
        }
        if env("STRATA_VERIFY_COMPLEMENT").is_some() && !host.is_null() && bytes > 0 {
            const PIECE: u64 = 1 << 30;
            let pieces = bytes.div_ceil(PIECE) as usize;
            let ph: Vec<AtomicU64> = (0..pieces).map(|_| AtomicU64::new(0)).collect();
            let next_piece = AtomicUsize::new(0);
            let hash_work = || loop {
                let i = next_piece.fetch_add(1, Ordering::SeqCst);
                if i >= pieces {
                    break;
                }
                let n = PIECE.min(bytes - i as u64 * PIECE) as usize;
                let s = unsafe {
                    std::slice::from_raw_parts((host_addr + i * PIECE as usize) as *const u8, n)
                };
                ph[i].store(fnv1a64(s, STRATA_FNV_SEED), Ordering::SeqCst);
            };
            std::thread::scope(|s| {
                for _ in 1..8 {
                    s.spawn(hash_work);
                }
                hash_work();
            });
            let ob: Vec<u8> = offsets.iter().flat_map(|o| o.to_le_bytes()).collect();
            let h = fnv1a64(&ob, STRATA_FNV_SEED);
            let pb: Vec<u8> = ph
                .iter()
                .flat_map(|p| p.load(Ordering::SeqCst).to_le_bytes())
                .collect();
            let h = fnv1a64(&pb, h);
            eprintln!(
                "FileExpertSource: cache complement checksum {h:016x} ({:.2} GiB, its offsets included)",
                bytes as f64 / 1073741824.0
            );
        }
        let c = unsafe { self.cm() };
        c.complement_arena = arena;
        c.complement_aligned = aligned;
        c.complement_host = host;
        c.complement_device = device;
        c.complement_bytes = bytes;
        c.complement_offsets = offsets;
        c.complement_pinned = (pinned_ok || partial_pin > 0) && bytes > 0;
        c.complement_pin_limit = if pinned_ok { bytes } else { partial_pin };
        c.complement_registered = !pinned_ok && partial_pin > 0;
        c.complement_partial = c.complement_registered && !(paced && partial_pin >= bytes);
        c.complement_locked = locked;
        c.complement_lock_off = lock_off;
        c.complement_lent_slots = if lend { n_slots - keep_from } else { 0 };
        c.complement_lent_slots += stage_lent;
        c.complement_ready = true;
        eprintln!(
            "FileExpertSource: {} cache complement ready: resident {:.2} GiB, pinned {:.2} GiB in {:.0} s{}{}",
            if c.complement_pinned {
                "mapped pinned"
            } else if pin {
                "locked resident"
            } else {
                "pageable resident"
            },
            self.resident_bytes() as f64 / 1073741824.0,
            self.pinned_bytes() as f64 / 1073741824.0,
            pin_t0.elapsed().as_secs_f64(),
            if note.is_empty() { "" } else { "; " },
            note
        );
        if lend {
            eprintln!(
                "FileExpertSource: {} of the prompt path's {} lendable slots keep their experts in RAM too{}",
                c.complement_lent_slots,
                n_slots - lend_from_slot,
                if c.complement_lent_slots < n_slots - lend_from_slot {
                    " (the others are read from the file when lent: not enough RAM for them)"
                } else {
                    ""
                }
            );
        }
        if lend && c.complement_lent_slots < n_slots - lend_from_slot {
            let lendable = n_slots - lend_from_slot;
            let missing = lendable - c.complement_lent_slots;
            eprintln!(
                "FileExpertSource: WARNING: {missing} of {lendable} lendable slots ({:.0}%) will read their experts from \
                 the pack during a long prompt; expect a slower prompt read on 50K+ token prompts (one measurement: 18% \
                 uncovered cost 26%). Free RAM or lower STRATA_RESIDENT_HEADROOM_GIB (now the RAM left free at start) \
                 and restart; the line above should read N of N.",
                100.0 * missing as f64 / lendable.max(1) as f64
            );
        }
        if !c.stage_lend_regions.is_empty() {
            let offered: i64 = c.stage_lend_regions.iter().map(|r| r.len() as i64).sum();
            eprintln!(
                "FileExpertSource: the split stages lend {offered} slots to the prompt path: {stage_lent} keep their \
                 experts in RAM too ({:.2} GiB){}",
                stage_lent_bytes as f64 / 1073741824.0,
                if stage_lent < offered {
                    " (the rest are read from the file when lent: not enough RAM for them)"
                } else {
                    ""
                }
            );
        }
        if !additional_gpu_pairs.is_empty() {
            eprintln!(
                "FileExpertSource: {} verified additional-GPU experts remain on the mmap fallback",
                additional_gpu_pairs.len()
            );
        }
        Ok(())
    }

    // ---- the adaptive tier's exchanges --------------------------------------------------------------------------------

    /// `FileExpertSource::reserve_exchanges` (the copy path; rotation is not adopted).
    pub fn reserve_exchanges(&self, n: i64) -> Result<(), String> {
        let c = unsafe { self.cm() };
        if n <= c.xstage_cap {
            return Ok(());
        }
        if !c.staged.is_empty() {
            return Err("FileExpertSource: exchange buffers are in use".into());
        }
        let blob = c.layer_blob_bytes.iter().copied().max().unwrap_or(0);
        if blob == 0 || n <= 0 {
            return Err("FileExpertSource: no expert geometry for the exchange buffers".into());
        }
        unsafe {
            if !c.xstage.is_null() {
                if c.xstage_pinned {
                    let _ = hip_sys::hipHostFree(c.xstage.cast());
                } else {
                    free(c.xstage.cast());
                }
                c.xstage = std::ptr::null_mut();
                c.xstage_cap = 0;
            }
        }
        if n as u64 > usize::MAX as u64 / blob {
            return Err("FileExpertSource: exchange allocation size overflow".into());
        }
        let total = n as usize * blob as usize;
        let mut p: *mut c_void = std::ptr::null_mut();
        unsafe {
            if hip_sys::hipHostMalloc(&mut p, total, hip_sys::hipHostMallocDefault)
                == hip_sys::hipSuccess
                && !p.is_null()
            {
                c.xstage_pinned = true;
            } else {
                let _ = hip_sys::hipGetLastError();
                p = malloc(total);
                c.xstage_pinned = false;
                if p.is_null() {
                    return Err("FileExpertSource: cannot allocate the exchange buffers".into());
                }
            }
        }
        c.xstage = p.cast();
        c.xstage_cap = n;
        c.xstage_blob = blob;
        Ok(())
    }

    pub fn exchange_buffer(&self, q: i64) -> *mut u8 {
        let c = self.c();
        if c.xstage.is_null() || q < 0 || q >= c.xstage_cap {
            return std::ptr::null_mut();
        }
        unsafe { c.xstage.add(q as usize * c.xstage_blob as usize) }
    }

    pub fn stage_exchange(&self, layer: i64, i_in: i64, out: i64, q: i64) -> bool {
        let n_expert = self.c().n_expert;
        if out < 0
            || out >= n_expert
            || !self.has_resident(layer, i_in)
            || self.has_resident(layer, out)
            || self.exchange_buffer(q).is_null()
        {
            return false;
        }
        let c = unsafe { self.cm() };
        let i_in_x = (layer * n_expert + i_in) as usize;
        let i_out = (layer * n_expert + out) as usize;
        if c.override_.is_empty() {
            c.override_ = vec![std::ptr::null(); c.blobs as usize];
        }
        if !c.override_[i_out].is_null() {
            return false;
        }
        if c.staged.iter().any(|x| x.i_in == i_in_x || x.q == q) {
            return false;
        }
        c.override_[i_out] = self.exchange_buffer(q);
        let bytes = c.layer_blob_bytes[layer as usize];
        c.staged.push(Exchange {
            i_in: i_in_x,
            out: i_out,
            q,
            bytes,
        });
        true
    }

    pub fn prepare_overrides(&self) {
        let c = unsafe { self.cm() };
        if c.override_.is_empty() {
            c.override_ = vec![std::ptr::null(); c.blobs as usize];
        }
    }

    pub fn soft_commit_exchanges(&self) -> bool {
        let c = unsafe { self.cm() };
        if c.staged.is_empty() {
            return true;
        }
        if !c.xstage_pinned
            || c.xstage.is_null()
            || !c.complement_pinned
            || c.complement_device.is_null()
        {
            return false;
        }
        if c.xstage_dev.is_null() {
            let mut d: *mut c_void = std::ptr::null_mut();
            unsafe {
                if hip_sys::hipHostGetDevicePointer(&mut d, c.xstage.cast(), 0)
                    != hip_sys::hipSuccess
                    || d.is_null()
                {
                    let _ = hip_sys::hipGetLastError();
                    return false;
                }
            }
            c.xstage_dev = d as *const u8;
        }
        if c.soft_in.is_empty() {
            c.soft_in = vec![usize::MAX; c.blobs as usize];
        }
        for x in c.staged.clone() {
            c.soft_in[x.out] = x.i_in;
        }
        true
    }

    fn fs_hold_staged(&self, hold: bool) {
        let fs = self.fs_swap.load(Ordering::Acquire);
        let c = self.c();
        let Some(f) = self.fs_hold else { return };
        if fs.is_null() || c.n_expert <= 0 {
            return;
        }
        for x in &c.staged {
            for i in [x.i_in, x.out] {
                let layer = (i / c.n_expert as usize) as i64;
                let expert = (i % c.n_expert as usize) as i32;
                unsafe { f(fs, layer, expert, hold as c_int) };
            }
        }
    }

    /// The copy of `x.out`'s bytes into `x.in`'s place when that place is in the arena; true when it was copied.
    unsafe fn exchange_copy(&self, x: &Exchange) -> bool {
        let c = self.c();
        let src = self.override_at(x.out);
        let at = c.complement_offsets[x.i_in];
        if !src.is_null()
            && at != NO_CACHE_COMPLEMENT
            && at <= c.complement_bytes
            && x.bytes <= c.complement_bytes - at
            && !c.complement_host.is_null()
        {
            std::ptr::copy_nonoverlapping(
                src,
                (c.complement_host as *mut u8).add(at as usize),
                x.bytes as usize,
            );
            return true;
        }
        false
    }

    pub fn commit_exchanges(&self) -> i64 {
        let mut n = 0i64;
        self.fs_hold_staged(true);
        let c = unsafe { self.cm() };
        let staged = c.staged.clone();
        for x in &staged {
            if !c.soft_in.is_empty() {
                c.soft_in[x.out] = usize::MAX;
            }
        }
        for x in &staged {
            if unsafe { self.exchange_copy(x) }
                && exchange_cache_complement(&mut c.complement_offsets, x.i_in, x.out)
            {
                n += 1;
            }
            c.override_[x.out] = std::ptr::null();
        }
        self.fs_hold_staged(false);
        c.staged.clear();
        c.exchanges += n;
        n
    }

    pub fn commit_copies(&self) {
        self.fs_hold_staged(true);
        let staged = self.c().staged.clone();
        for x in &staged {
            unsafe {
                let _ = self.exchange_copy(x);
            }
        }
    }

    pub fn commit_flip(&self) -> i64 {
        let mut n = 0i64;
        self.fs_hold_staged(true);
        let c = unsafe { self.cm() };
        let staged = c.staged.clone();
        for x in &staged {
            let src = self.override_at(x.out);
            let at = c.complement_offsets[x.i_in];
            if !src.is_null()
                && at != NO_CACHE_COMPLEMENT
                && at <= c.complement_bytes
                && x.bytes <= c.complement_bytes - at
                && !c.complement_host.is_null()
                && exchange_cache_complement(&mut c.complement_offsets, x.i_in, x.out)
            {
                n += 1;
            }
            c.override_[x.out] = std::ptr::null();
        }
        self.fs_hold_staged(false);
        c.staged.clear();
        c.exchanges += n;
        n
    }
}

/// Whether the Rust source can take this run (the modes it ports); `None` = yes, else why not (the C++ source runs).
pub fn decline_reason(pack_dir: &str, gguf: &str, layout: &ExpertLayout) -> Option<String> {
    let set = |k: &str| {
        env(k)
            .map(|v| !v.is_empty() && c_atoi(&v) != 0)
            .unwrap_or(false)
    };
    if !layout.native {
        return Some("a canonical pack (experts.bin)".into());
    }
    if gguf.is_empty() {
        return Some("no --native GGUF".into());
    }
    if std::path::Path::new(&format!("{pack_dir}/experts.bin")).exists() {
        return Some("the pack has experts.bin".into());
    }
    if set("STRATA_IO_PREFETCH") {
        return Some("STRATA_IO_PREFETCH (the io-prefetch tier is not ported)".into());
    }
    if set("STRATA_IO_STATS") {
        return Some("STRATA_IO_STATS (the io counters are not ported)".into());
    }
    if env("STRATA_EXCHANGE_ROTATE").as_deref() == Some("1") {
        return Some("STRATA_EXCHANGE_ROTATE=1 (exchange rotation is not ported)".into());
    }
    if set("STRATA_TEST_UNMAP_GGUF") {
        return Some("STRATA_TEST_UNMAP_GGUF".into());
    }
    None
}
