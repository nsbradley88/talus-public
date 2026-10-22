//! qwen4exp-r1: the C ABI of the Rust expert source and PLE table reader (include/talus_qwen4exp_kernels.h `qwen4exp-r1`
//! block; src/program/qwen4exp_r1.cpp). Rust registers a factory (`tk_qwen4exp_r1_set_rust_hooks`, dlsym RTLD_DEFAULT as
//! Piece 1 does); on the Rust data plane generate.cpp asks it for a source / a reader and the C++ objects adopt the
//! handle, forwarding every method to the vtables below. Kill switch: TALUS_FN_R1=0 (the C++ source and reader).

#![allow(clippy::missing_safety_doc)]

use core::ffi::{c_char, c_int, c_void};
use std::ffi::CStr;

use super::ple_reader::{PleIoOptions, PleTable, PLE_HEAD_DIM, PLE_N_HEADS};
use super::source::{decline_reason, CacheView, ExpertLayout, FileSource, FsHoldFn};
use super::{dlsym, TkQwen4ExpP1PleIo, RTLD_DEFAULT};

// =====================================================================================================================
// The ABI structs
// =====================================================================================================================

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TkR1Layout {
    pub native: i32,
    pub n_layers: i64,
    pub n_expert: i64,
    pub n_per_layer: usize,
    pub fmt_up_off: *const u64,
    pub fmt_down_off: *const u64,
    pub fmt_bytes: *const u64,
    pub fmt_gu_type: *const i32,
    pub fmt_d_type: *const i32,
    pub fmt_n_embd: *const i64,
    pub fmt_n_ff: *const i64,
    pub offset: *const u64,
    pub bytes: *const u64,
    pub gguf_off: *const u64,
    pub n_gguf_off: usize,
    pub gguf_file: *const *const c_char,
    pub n_gguf_file: usize,
    pub max_blob: u64,
    pub total: u64,
    pub canonical_blob: u64,
}

#[repr(C)]
pub struct TkR1Services {
    pub struct_size: usize,
    pub expert_layout: Option<unsafe extern "C" fn(out: *mut TkR1Layout) -> c_int>,
    pub pin_aux_thread: Option<unsafe extern "C" fn()>,
    pub fs_hold: Option<FsHoldFn>,
    pub ple_prefetch_enabled: Option<unsafe extern "C" fn() -> c_int>,
}

#[repr(C)]
pub struct TkR1CacheView {
    pub valid: i32,
    pub fills: i64,
    pub resident: i64,
    pub slots: i64,
    pub slot_of: *const i32,
    pub n_slot_of: usize,
}

pub const S_MAPPED: c_int = 0;
pub const S_BLOBS: c_int = 1;
pub const S_PINNED_BYTES: c_int = 2;
pub const S_RESIDENT_BYTES: c_int = 3;
pub const S_COMPLEMENT_PINNED: c_int = 4;
pub const S_COMPLEMENT_READY: c_int = 5;
pub const S_LOCKED_BYTES: c_int = 6;
pub const S_RESIDENT_LENT_SLOTS: c_int = 7;
pub const S_GGUF_MODE: c_int = 8;
pub const S_EXCHANGE_CAPACITY: c_int = 9;
pub const S_EXCHANGE_PINNED: c_int = 10;
pub const S_EXCHANGES: c_int = 11;
pub const S_FILE_READS: c_int = 12;
pub const S_RAM_READS: c_int = 13;
pub const S_FILE_READ_BYTES: c_int = 14;
pub const S_FILE_BLOB_BYTES: c_int = 15;
pub const S_FILE_US: c_int = 16;
pub const S_UNBUFFERED: c_int = 17;
pub const S_EXPERT_BYTES: c_int = 18;
pub const S_WARMS: c_int = 19;
pub const S_WARMED_HITS: c_int = 20;
pub const S_WARMED: c_int = 21;
pub const S_DIRECT_FALLBACKS: c_int = 22;
pub const S_READS: c_int = 23;

pub const Q_PINNED: c_int = 0;
pub const Q_TRANSIENT: c_int = 1;
pub const Q_HAS_RESIDENT: c_int = 2;
pub const Q_PCIE_LAYER: c_int = 3;
pub const P_DEVICE_ALIAS: c_int = 0;
pub const P_RESIDENT_BLOB: c_int = 1;
pub const P_EXCHANGE_BUFFER: c_int = 2;
pub const U_SET: c_int = 0;
pub const U_RECHECK: c_int = 1;
pub const U_PROMPT_DIRECT: c_int = 2;
pub const X_PREPARE_OVERRIDES: c_int = 0;
pub const X_COMMIT: c_int = 1;
pub const X_SOFT_COMMIT: c_int = 2;
pub const X_COMMIT_COPIES: c_int = 3;
pub const X_COMMIT_FLIP: c_int = 4;

#[repr(C)]
pub struct TkR1SourceVt {
    pub struct_size: usize,
    pub destroy: unsafe extern "C" fn(h: *mut c_void),
    pub open: unsafe extern "C" fn(
        *mut c_void,
        *const c_char,
        *const c_char,
        i64,
        i64,
        *mut c_char,
        usize,
    ) -> c_int,
    pub close: unsafe extern "C" fn(h: *mut c_void),
    pub stat: unsafe extern "C" fn(h: *mut c_void, what: c_int) -> u64,
    pub input:
        unsafe extern "C" fn(*mut c_void, usize, *mut *const c_char, *mut *const c_char) -> usize,
    pub blob: unsafe extern "C" fn(*mut c_void, i64, i64, c_int) -> *const u8,
    pub note_async_read: unsafe extern "C" fn(*mut c_void, *const u8, *mut c_void),
    pub query: unsafe extern "C" fn(*mut c_void, c_int, i64, i64) -> c_int,
    pub pointer: unsafe extern "C" fn(*mut c_void, c_int, i64, i64) -> *const u8,
    pub cached_share: unsafe extern "C" fn(*mut c_void, i64) -> f64,
    pub copy_blob: unsafe extern "C" fn(*mut c_void, i64, i64, *mut u8) -> c_int,
    pub copy_blobs:
        unsafe extern "C" fn(*mut c_void, *const i32, *const i32, *const *mut u8, usize) -> c_int,
    pub begin_layer: unsafe extern "C" fn(*mut c_void, i64),
    pub prefetch: unsafe extern "C" fn(*mut c_void, i64, *const i64, i64),
    pub warm: unsafe extern "C" fn(*mut c_void, i64, *const i64, i64),
    pub prefetch_pairs: unsafe extern "C" fn(*mut c_void, *const i32, i64),
    pub advise_pairs: unsafe extern "C" fn(*mut c_void, *const i32, i64) -> c_int,
    pub unbuffered: unsafe extern "C" fn(*mut c_void, c_int, u64, *mut c_char, usize) -> c_int,
    #[allow(clippy::type_complexity)]
    pub pin_cache_complement: unsafe extern "C" fn(
        *mut c_void,
        *const TkR1CacheView,
        c_int,
        *const i32,
        usize,
        i64,
        u64,
        u64,
        *const i32,
        usize,
        c_int,
        *mut c_char,
        usize,
    ) -> c_int,
    pub stage_lend_regions: unsafe extern "C" fn(*mut c_void, *const i32, *const usize, usize),
    pub reserve_exchanges: unsafe extern "C" fn(*mut c_void, i64, *mut c_char, usize) -> c_int,
    pub stage_exchange: unsafe extern "C" fn(*mut c_void, i64, i64, i64, i64) -> c_int,
    pub exchange_op: unsafe extern "C" fn(*mut c_void, c_int) -> i64,
    pub set_foresight: unsafe extern "C" fn(*mut c_void, *mut c_void),
    pub set_fetch_threads: unsafe extern "C" fn(*mut c_void, c_int),
}

pub const PLE_IS_OPEN: c_int = 0;
pub const PLE_LOCKED: c_int = 1;
pub const PLE_ROWS: c_int = 2;
pub const PLE_BYTES_READ: c_int = 3;
pub const PLE_MMAP: c_int = 4;
pub const PLE_FORMAT: c_int = 5;

#[repr(C)]
pub struct TkR1PleVt {
    pub struct_size: usize,
    pub destroy: unsafe extern "C" fn(*mut c_void),
    pub open: unsafe extern "C" fn(
        *mut c_void,
        *const c_char,
        *const TkQwen4ExpP1PleIo,
        *mut c_char,
        usize,
    ) -> c_int,
    pub close: unsafe extern "C" fn(*mut c_void),
    pub stat: unsafe extern "C" fn(*mut c_void, c_int) -> u64,
    pub issue: unsafe extern "C" fn(*mut c_void, *const u32) -> c_int,
    pub collect: unsafe extern "C" fn(*mut c_void, *mut f32, *mut c_char, usize) -> c_int,
    pub prefetch_rows: unsafe extern "C" fn(*mut c_void, *const u32),
    pub wait_prefetches: unsafe extern "C" fn(*mut c_void),
    pub gather_batch:
        unsafe extern "C" fn(*mut c_void, *const u32, usize, *mut f32, *mut c_char, usize) -> c_int,
    pub gather: unsafe extern "C" fn(*mut c_void, *const u32, *mut f32),
    pub read_row: unsafe extern "C" fn(*mut c_void, u32, *mut f32),
    pub set_injected_delay_us: unsafe extern "C" fn(*mut c_void, f64),
    pub io_report: unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> usize,
}

#[repr(C)]
pub struct TkR1RustHooks {
    pub struct_size: usize,
    pub source_create: unsafe extern "C" fn(
        *const c_char,
        *const c_char,
        *mut *const TkR1SourceVt,
        *mut c_char,
        usize,
    ) -> *mut c_void,
    pub ple_create: unsafe extern "C" fn(*mut *const TkR1PleVt, *mut c_char, usize) -> *mut c_void,
    pub release_all: unsafe extern "C" fn(),
}

// =====================================================================================================================
// Helpers
// =====================================================================================================================

unsafe fn put_str(dst: *mut c_char, cap: usize, s: &str) {
    if dst.is_null() || cap == 0 {
        return;
    }
    let b = s.as_bytes();
    let n = b.len().min(cap - 1);
    std::ptr::copy_nonoverlapping(b.as_ptr(), dst.cast::<u8>(), n);
    *dst.add(n) = 0;
}

unsafe fn cstr<'a>(p: *const c_char) -> &'a str {
    if p.is_null() {
        ""
    } else {
        CStr::from_ptr(p).to_str().unwrap_or("")
    }
}

unsafe fn pairs<'a>(p: *const i32, n: usize) -> &'a [(i32, i32)] {
    if p.is_null() || n == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p.cast::<(i32, i32)>(), n)
    }
}

fn services() -> Option<&'static TkR1Services> {
    let sym = unsafe { dlsym(RTLD_DEFAULT, c"tk_qwen4exp_r1_services".as_ptr()) };
    if sym.is_null() {
        return None;
    }
    let f: unsafe extern "C" fn() -> *const TkR1Services = unsafe { std::mem::transmute(sym) };
    let p = unsafe { f() };
    if p.is_null() || unsafe { (*p).struct_size } < std::mem::size_of::<TkR1Services>() {
        return None;
    }
    Some(unsafe { &*p })
}

/// The expert layout the C++ pack loader set (strata::kernels::cpu::expert_layout()).
pub fn read_layout(svc: &TkR1Services) -> Option<ExpertLayout> {
    let f = svc.expert_layout?;
    let mut l = std::mem::MaybeUninit::<TkR1Layout>::zeroed();
    if unsafe { f(l.as_mut_ptr()) } != 0 {
        return None;
    }
    let l = unsafe { l.assume_init() };
    let v64 = |p: *const u64, n: usize| {
        if p.is_null() || n == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
        }
    };
    let vi32 = |p: *const i32, n: usize| {
        if p.is_null() || n == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
        }
    };
    let vi64 = |p: *const i64, n: usize| {
        if p.is_null() || n == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
        }
    };
    let n = l.n_per_layer;
    let gguf_file = if l.gguf_file.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(l.gguf_file, l.n_gguf_file) }
            .iter()
            .map(|&p| unsafe { cstr(p) }.to_string())
            .collect()
    };
    Some(ExpertLayout {
        native: l.native != 0,
        n_layers: l.n_layers,
        n_expert: l.n_expert,
        fmt_up_off: v64(l.fmt_up_off, n),
        fmt_down_off: v64(l.fmt_down_off, n),
        fmt_bytes: v64(l.fmt_bytes, n),
        fmt_gu_type: vi32(l.fmt_gu_type, n),
        fmt_d_type: vi32(l.fmt_d_type, n),
        fmt_n_embd: vi64(l.fmt_n_embd, n),
        fmt_n_ff: vi64(l.fmt_n_ff, n),
        offset: v64(l.offset, n),
        bytes: v64(l.bytes, n),
        gguf_off: v64(l.gguf_off, l.n_gguf_off),
        gguf_file,
        max_blob: l.max_blob,
        total: l.total,
        canonical_blob: l.canonical_blob,
    })
}

#[inline]
unsafe fn src<'a>(h: *mut c_void) -> &'a FileSource {
    &*(h as *const FileSource)
}

// =====================================================================================================================
// The source vtable
// =====================================================================================================================

unsafe extern "C" fn s_destroy(h: *mut c_void) {
    if !h.is_null() {
        drop(Box::from_raw(h as *mut FileSource));
    }
}

unsafe extern "C" fn s_open(
    h: *mut c_void,
    pack: *const c_char,
    gguf: *const c_char,
    n_layers: i64,
    n_expert: i64,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    match src(h).open(cstr(pack), cstr(gguf), n_layers, n_expert) {
        Ok(()) => 0,
        Err(e) => {
            put_str(err, cap, &e);
            -1
        }
    }
}

unsafe extern "C" fn s_close(h: *mut c_void) {
    src(h).close();
}

unsafe extern "C" fn s_stat(h: *mut c_void, what: c_int) -> u64 {
    let s = src(h);
    match what {
        S_MAPPED => s.mapped() as u64,
        S_BLOBS => s.blobs() as u64,
        S_PINNED_BYTES => s.pinned_bytes(),
        S_RESIDENT_BYTES => s.resident_bytes(),
        S_COMPLEMENT_PINNED => s.complement_pinned() as u64,
        S_COMPLEMENT_READY => s.complement_ready() as u64,
        S_LOCKED_BYTES => s.locked_bytes(),
        S_RESIDENT_LENT_SLOTS => s.resident_lent_slots() as u64,
        S_GGUF_MODE => s.gguf_mode() as u64,
        S_EXCHANGE_CAPACITY => s.exchange_capacity() as u64,
        S_EXCHANGE_PINNED => s.exchange_pinned() as u64,
        S_EXCHANGES => s.exchanges() as u64,
        S_FILE_READS => s.file_reads() as u64,
        S_RAM_READS => s.ram_reads() as u64,
        S_FILE_READ_BYTES => s.file_read_bytes(),
        S_FILE_BLOB_BYTES => s.file_blob_bytes(),
        S_FILE_US => s.file_us(),
        S_UNBUFFERED => s.unbuffered() as u64,
        S_EXPERT_BYTES => s.expert_bytes(),
        S_WARMS => s.warms() as u64,
        S_WARMED_HITS => s.warmed_hits() as u64,
        S_WARMED => s.warmed() as u64,
        S_DIRECT_FALLBACKS => s.direct_fallbacks() as u64,
        S_READS => s.reads() as u64,
        _ => 0,
    }
}

unsafe extern "C" fn s_input(
    h: *mut c_void,
    i: usize,
    role: *mut *const c_char,
    path: *mut *const c_char,
) -> usize {
    let inputs = src(h).inputs();
    if let Some((r, p)) = inputs.get(i) {
        if !role.is_null() {
            *role = r.as_ptr();
        }
        if !path.is_null() {
            *path = p.as_ptr();
        }
    }
    inputs.len()
}

unsafe extern "C" fn s_blob(h: *mut c_void, l: i64, e: i64, stable: c_int) -> *const u8 {
    if stable != 0 {
        src(h).blob_stable(l, e)
    } else {
        src(h).blob(l, e)
    }
}

unsafe extern "C" fn s_note_async_read(h: *mut c_void, b: *const u8, stream: *mut c_void) {
    src(h).note_async_read(b, stream);
}

unsafe extern "C" fn s_query(h: *mut c_void, what: c_int, l: i64, e: i64) -> c_int {
    let s = src(h);
    (match what {
        Q_PINNED => s.pinned(l, e),
        Q_TRANSIENT => s.transient(l, e),
        Q_HAS_RESIDENT => s.has_resident(l, e),
        Q_PCIE_LAYER => s.pcie_layer(l),
        _ => false,
    }) as c_int
}

unsafe extern "C" fn s_pointer(h: *mut c_void, what: c_int, l: i64, e: i64) -> *const u8 {
    let s = src(h);
    match what {
        P_DEVICE_ALIAS => s.device_alias(l, e),
        P_RESIDENT_BLOB => s.resident_blob(l, e),
        P_EXCHANGE_BUFFER => s.exchange_buffer(l),
        _ => std::ptr::null(),
    }
}

unsafe extern "C" fn s_cached_share(h: *mut c_void, samples: i64) -> f64 {
    src(h).cached_share(samples)
}

unsafe extern "C" fn s_copy_blob(h: *mut c_void, l: i64, e: i64, dst: *mut u8) -> c_int {
    src(h).copy_blob(l, e, dst) as c_int
}

unsafe extern "C" fn s_copy_blobs(
    h: *mut c_void,
    l: *const i32,
    e: *const i32,
    dst: *const *mut u8,
    n: usize,
) -> c_int {
    if n == 0 {
        return 1;
    }
    let (l, e, d) = (
        std::slice::from_raw_parts(l, n),
        std::slice::from_raw_parts(e, n),
        std::slice::from_raw_parts(dst, n),
    );
    src(h).copy_blobs(l, e, d) as c_int
}

unsafe extern "C" fn s_begin_layer(h: *mut c_void, l: i64) {
    src(h).begin_layer(l);
}

unsafe extern "C" fn s_prefetch(h: *mut c_void, l: i64, e: *const i64, n: i64) {
    if n <= 0 || e.is_null() {
        return;
    }
    src(h).prefetch(l, std::slice::from_raw_parts(e, n as usize));
}

unsafe extern "C" fn s_warm(h: *mut c_void, l: i64, e: *const i64, n: i64) {
    if n <= 0 || e.is_null() {
        return;
    }
    src(h).warm(l, std::slice::from_raw_parts(e, n as usize));
}

unsafe extern "C" fn s_prefetch_pairs(h: *mut c_void, p: *const i32, n: i64) {
    if n <= 0 {
        return;
    }
    src(h).prefetch_pairs(pairs(p, n as usize));
}

unsafe extern "C" fn s_advise_pairs(h: *mut c_void, p: *const i32, n: i64) -> c_int {
    src(h).advise_pairs(pairs(p, n.max(0) as usize)) as c_int
}

unsafe extern "C" fn s_unbuffered(
    h: *mut c_void,
    what: c_int,
    ram: u64,
    why: *mut c_char,
    cap: usize,
) -> c_int {
    let s = src(h);
    let mut w = String::new();
    let ok = match what {
        U_SET => s.set_unbuffered(ram, &mut w),
        U_RECHECK => s.recheck_unbuffered(&mut w),
        U_PROMPT_DIRECT => s.open_prompt_direct(&mut w),
        _ => false,
    };
    put_str(why, cap, &w);
    ok as c_int
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn s_pin_cache_complement(
    h: *mut c_void,
    cache: *const TkR1CacheView,
    pin: c_int,
    add: *const i32,
    n_add: usize,
    lend_from: i64,
    headroom: u64,
    budget: u64,
    rank: *const i32,
    n_rank: usize,
    has_rank: c_int,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    let c = &*cache;
    let slot_of = if c.slot_of.is_null() {
        &[][..]
    } else {
        std::slice::from_raw_parts(c.slot_of, c.n_slot_of)
    };
    let view = CacheView {
        valid: c.valid != 0,
        fills: c.fills,
        resident: c.resident,
        slots: c.slots,
        slot_of,
    };
    let rank = if has_rank != 0 {
        Some(pairs(rank, n_rank))
    } else {
        None
    };
    match src(h).pin_cache_complement(
        &view,
        pin != 0,
        pairs(add, n_add),
        lend_from,
        headroom,
        budget,
        rank,
    ) {
        Ok(()) => 0,
        Err(e) => {
            put_str(err, cap, &e);
            -1
        }
    }
}

unsafe extern "C" fn s_stage_lend_regions(
    h: *mut c_void,
    p: *const i32,
    lens: *const usize,
    n: usize,
) {
    let mut regions = Vec::with_capacity(n);
    let mut at = 0usize;
    for i in 0..n {
        let len = *lens.add(i);
        regions.push(pairs(p.add(2 * at), len).to_vec());
        at += len;
    }
    src(h).set_stage_lend_regions(regions);
}

unsafe extern "C" fn s_reserve_exchanges(
    h: *mut c_void,
    n: i64,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    match src(h).reserve_exchanges(n) {
        Ok(()) => 0,
        Err(e) => {
            put_str(err, cap, &e);
            -1
        }
    }
}

unsafe extern "C" fn s_stage_exchange(h: *mut c_void, l: i64, i: i64, o: i64, q: i64) -> c_int {
    src(h).stage_exchange(l, i, o, q) as c_int
}

unsafe extern "C" fn s_exchange_op(h: *mut c_void, what: c_int) -> i64 {
    let s = src(h);
    match what {
        X_PREPARE_OVERRIDES => {
            s.prepare_overrides();
            0
        }
        X_COMMIT => s.commit_exchanges(),
        X_SOFT_COMMIT => s.soft_commit_exchanges() as i64,
        X_COMMIT_COPIES => {
            s.commit_copies();
            0
        }
        X_COMMIT_FLIP => s.commit_flip(),
        _ => 0,
    }
}

unsafe extern "C" fn s_set_foresight(h: *mut c_void, fs: *mut c_void) {
    src(h).set_foresight_swap(fs);
}

unsafe extern "C" fn s_set_fetch_threads(h: *mut c_void, n: c_int) {
    src(h).set_fetch_threads(n);
}

static SOURCE_VT: TkR1SourceVt = TkR1SourceVt {
    struct_size: std::mem::size_of::<TkR1SourceVt>(),
    destroy: s_destroy,
    open: s_open,
    close: s_close,
    stat: s_stat,
    input: s_input,
    blob: s_blob,
    note_async_read: s_note_async_read,
    query: s_query,
    pointer: s_pointer,
    cached_share: s_cached_share,
    copy_blob: s_copy_blob,
    copy_blobs: s_copy_blobs,
    begin_layer: s_begin_layer,
    prefetch: s_prefetch,
    warm: s_warm,
    prefetch_pairs: s_prefetch_pairs,
    advise_pairs: s_advise_pairs,
    unbuffered: s_unbuffered,
    pin_cache_complement: s_pin_cache_complement,
    stage_lend_regions: s_stage_lend_regions,
    reserve_exchanges: s_reserve_exchanges,
    stage_exchange: s_stage_exchange,
    exchange_op: s_exchange_op,
    set_foresight: s_set_foresight,
    set_fetch_threads: s_set_fetch_threads,
};

unsafe extern "C" fn h_source_create(
    pack: *const c_char,
    gguf: *const c_char,
    vt: *mut *const TkR1SourceVt,
    why: *mut c_char,
    cap: usize,
) -> *mut c_void {
    *vt = std::ptr::null();
    let Some(svc) = services() else {
        put_str(why, cap, "no tk_qwen4exp_r1_services");
        return std::ptr::null_mut();
    };
    let Some(layout) = read_layout(svc) else {
        put_str(why, cap, "the expert layout could not be read");
        return std::ptr::null_mut();
    };
    if let Some(r) = decline_reason(cstr(pack), cstr(gguf), &layout) {
        put_str(why, cap, &r);
        return std::ptr::null_mut();
    }
    *vt = &SOURCE_VT;
    Box::into_raw(Box::new(FileSource::new(layout, svc.fs_hold))) as *mut c_void
}

// =====================================================================================================================
// The PLE vtable
// =====================================================================================================================

#[inline]
unsafe fn ple<'a>(h: *mut c_void) -> &'a PleTable {
    &*(h as *const PleTable)
}

unsafe fn rows16<'a>(p: *const u32) -> &'a [u32; PLE_N_HEADS] {
    &*(p as *const [u32; PLE_N_HEADS])
}

unsafe extern "C" fn p_destroy(h: *mut c_void) {
    if !h.is_null() {
        drop(Box::from_raw(h as *mut PleTable));
    }
}

unsafe extern "C" fn p_open(
    h: *mut c_void,
    path: *const c_char,
    io: *const TkQwen4ExpP1PleIo,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    let io = &*io;
    let opt = PleIoOptions {
        mmap: io.mode == 1,
        lock: io.lock != 0,
        max_inflight: io.max_inflight,
        cache_rows: io.cache_rows,
        io_thread: io.io_thread != 0,
        keepalive_ms: io.keepalive_ms,
        keepalive_window_s: io.keepalive_window_s,
    };
    match ple(h).open(cstr(path), &opt) {
        Ok(()) => 0,
        Err(e) => {
            ple(h).close();
            put_str(err, cap, &e);
            -1
        }
    }
}

unsafe extern "C" fn p_close(h: *mut c_void) {
    ple(h).close();
}

unsafe extern "C" fn p_stat(h: *mut c_void, what: c_int) -> u64 {
    let t = ple(h);
    match what {
        PLE_IS_OPEN => t.is_open() as u64,
        PLE_LOCKED => t.locked() as u64,
        PLE_ROWS => t.rows(),
        PLE_BYTES_READ => t.bytes_read(),
        PLE_MMAP => t.mmap_mode() as u64,
        PLE_FORMAT => t.format_index() as u64,
        _ => 0,
    }
}

unsafe extern "C" fn p_issue(h: *mut c_void, rows: *const u32) -> c_int {
    ple(h).issue(rows16(rows)) as c_int
}

unsafe extern "C" fn p_collect(
    h: *mut c_void,
    out: *mut f32,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    let o = std::slice::from_raw_parts_mut(out, PLE_N_HEADS * PLE_HEAD_DIM);
    match ple(h).collect(o) {
        Ok(()) => 0,
        Err(e) => {
            put_str(err, cap, &e);
            -1
        }
    }
}

unsafe extern "C" fn p_prefetch_rows(h: *mut c_void, rows: *const u32) {
    ple(h).prefetch_rows(rows16(rows));
}

unsafe extern "C" fn p_wait_prefetches(h: *mut c_void) {
    ple(h).wait_prefetches();
}

unsafe extern "C" fn p_gather_batch(
    h: *mut c_void,
    rows: *const u32,
    n_tokens: usize,
    out: *mut f32,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    let n = n_tokens * PLE_N_HEADS;
    let r = if n == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(rows, n)
    };
    let o = if n == 0 {
        &mut [][..]
    } else {
        std::slice::from_raw_parts_mut(out, n * PLE_HEAD_DIM)
    };
    match ple(h).gather_batch(r, n_tokens, o) {
        Ok(()) => 0,
        Err(e) => {
            put_str(err, cap, &e);
            -1
        }
    }
}

unsafe extern "C" fn p_gather(h: *mut c_void, rows: *const u32, out: *mut f32) {
    ple(h).gather(
        rows16(rows),
        std::slice::from_raw_parts_mut(out, PLE_N_HEADS * PLE_HEAD_DIM),
    );
}

unsafe extern "C" fn p_read_row(h: *mut c_void, row: u32, out: *mut f32) {
    ple(h).read_row(row, std::slice::from_raw_parts_mut(out, PLE_HEAD_DIM));
}

unsafe extern "C" fn p_set_injected_delay_us(h: *mut c_void, us: f64) {
    ple(h).set_injected_delay_us(us);
}

unsafe extern "C" fn p_io_report(h: *mut c_void, out: *mut c_char, cap: usize) -> usize {
    let r = ple(h).io_report();
    put_str(out, cap, &r);
    r.len()
}

static PLE_VT: TkR1PleVt = TkR1PleVt {
    struct_size: std::mem::size_of::<TkR1PleVt>(),
    destroy: p_destroy,
    open: p_open,
    close: p_close,
    stat: p_stat,
    issue: p_issue,
    collect: p_collect,
    prefetch_rows: p_prefetch_rows,
    wait_prefetches: p_wait_prefetches,
    gather_batch: p_gather_batch,
    gather: p_gather,
    read_row: p_read_row,
    set_injected_delay_us: p_set_injected_delay_us,
    io_report: p_io_report,
};

unsafe extern "C" fn h_ple_create(
    vt: *mut *const TkR1PleVt,
    why: *mut c_char,
    cap: usize,
) -> *mut c_void {
    *vt = std::ptr::null();
    let Some(svc) = services() else {
        put_str(why, cap, "no tk_qwen4exp_r1_services");
        return std::ptr::null_mut();
    };
    *vt = &PLE_VT;
    Box::into_raw(Box::new(PleTable::new(
        svc.pin_aux_thread,
        svc.ple_prefetch_enabled,
    ))) as *mut c_void
}

unsafe extern "C" fn h_release_all() {
    super::hooks::r1_release_handed_over();
}

static HOOKS: TkR1RustHooks = TkR1RustHooks {
    struct_size: std::mem::size_of::<TkR1RustHooks>(),
    source_create: h_source_create,
    ple_create: h_ple_create,
    release_all: h_release_all,
};

/// Registers the r1 factory with libtalus_kernels (idempotent); false when the library has no r1 entry point.
pub fn install_r1_hooks() -> bool {
    static DONE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DONE.get_or_init(|| {
        let sym = unsafe { dlsym(RTLD_DEFAULT, c"tk_qwen4exp_r1_set_rust_hooks".as_ptr()) };
        if sym.is_null() {
            return false;
        }
        let set: unsafe extern "C" fn(*const TkR1RustHooks) = unsafe { std::mem::transmute(sym) };
        unsafe { set(&HOOKS) };
        true
    })
}
