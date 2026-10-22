//! Piece 1 (`qwen4exp-p1`): the C ABI hook table (src/program/qwen4exp_p1.cpp) and the process-global Piece 1 state.

#![allow(clippy::missing_safety_doc)]

use super::*;

// =====================================================================================================================
// The C ABI hook table (src/program/qwen4exp_p1.cpp, include/talus_qwen4exp_kernels.h `qwen4exp-p1` block)
// =====================================================================================================================

/// `tk_qwen4exp_p1_weight_row_t`: one row of the Rust weight table, handed to `WeightTable::adopt_rows`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpP1WeightRow {
    pub name: *const c_char,
    pub data: *const c_void,
    pub bytes: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub elements: i64,
    pub kind: i32,
    pub code_bits: i32,
    pub code_bias: i32,
    pub group_elems: i32,
    pub codebook_iq4nl: i32,
    pub has_offset: i32,
    pub codes_bytes: u64,
    pub scales_bytes: u64,
    pub offset_bytes: u64,
    pub src_off: u64,
    pub src_bytes: u64,
    pub file_id: i32,
    pub scales_fp16: i32,
    pub act_kind: i32,
    pub resident: i32,
}

/// `tk_qwen4exp_p1_load_report_t` (`strata::core::LoadReport`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TkQwen4ExpP1LoadReport {
    pub tensors: u64,
    pub arena_bytes: u64,
    pub re_rounded: u64,
    pub bytes_saved: u64,
    pub read_ms: f64,
    pub upload_ms: f64,
}

pub type FnP1PackPoolBytes = unsafe extern "C" fn(
    user: *mut c_void,
    pack_dir: *const c_char,
    skip: *const *const c_char,
    n_skip: usize,
    out_bytes: *mut u64,
    err: *mut c_char,
    err_cap: usize,
) -> c_int;

pub type FnP1PackLoad = unsafe extern "C" fn(
    user: *mut c_void,
    pack_dir: *const c_char,
    arena: *mut c_void,
    arena_bytes: u64,
    skip: *const *const c_char,
    n_skip: usize,
    out_rows: *mut *const TkQwen4ExpP1WeightRow,
    out_n_rows: *mut usize,
    out_report: *mut TkQwen4ExpP1LoadReport,
    err: *mut c_char,
    err_cap: usize,
) -> c_int;

/// `tk_qwen4exp_p1_rust_hooks_t`: what the C++ engine calls instead of its own Piece 1 code under the Rust plane.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TkQwen4ExpP1RustHooks {
    pub user_data: *mut c_void,
    pub pack_pool_bytes: Option<FnP1PackPoolBytes>,
    pub pack_load: Option<FnP1PackLoad>,
    pub profile_read: Option<FnP1ProfileRead>,
    pub cache_plan: Option<FnP1CachePlan>,
    pub cache_build: Option<FnP1CacheBuild>,
    pub res_create: Option<FnP1ResCreate>,
    pub lend_init: Option<FnP1LendInit>,
    pub lend: Option<FnP1Lend>,
    pub refill: Option<FnP1Refill>,
    pub native_served_names: Option<FnP1NativeServedNames>,
    pub native_dense_load: Option<FnP1NativeDenseLoad>,
    pub native_head_load: Option<FnP1NativeHeadLoad>,
    pub native_embed_load: Option<FnP1NativeEmbedLoad>,
    pub ple_wire: Option<FnP1PleWire>,
    pub plan_lend: Option<FnP1PlanLend>,
    pub lend_slots: Option<FnP1LendSlots>,
    pub serve_scan: Option<FnP1ServeScan>,
    pub gen_lend: Option<
        unsafe extern "C" fn(
            user: *mut c_void,
            ctx: *const TkQwen4ExpP1LendCtx,
            first: i32,
            err: *mut c_char,
            cap: usize,
        ) -> i64,
    >,
    pub gen_refill:
        Option<unsafe extern "C" fn(user: *mut c_void, err: *mut c_char, cap: usize) -> i64>,
    pub ple_io_plan: Option<FnP1PleIoPlan>,
    pub ple_probe_decide:
        Option<unsafe extern "C" fn(user: *mut c_void, rps: f64, explicit: c_int) -> c_int>,
}

pub type FnP1PleIoPlan = unsafe extern "C" fn(
    user: *mut c_void,
    ple_io: *const c_char,
    inflight: i64,
    row_cache: i64,
    sync_submit: c_int,
    out: *mut TkQwen4ExpP1PleIo,
) -> c_int;

pub type FnP1PlanLend = unsafe extern "C" fn(
    user: *mut c_void,
    planner: *const TkQwen4ExpP1Planner,
    chunk_inout: *mut i64,
) -> i64;
pub type FnP1LendSlots =
    unsafe extern "C" fn(user: *mut c_void, planner: *const TkQwen4ExpP1Planner, chunk: i64) -> i64;
pub type FnP1ServeScan = unsafe extern "C" fn(
    user: *mut c_void,
    planner: *const TkQwen4ExpP1Planner,
    out_chunk: *mut i64,
    out_first: *mut i64,
) -> c_int;

pub type FnP1PleWire = unsafe extern "C" fn(
    user: *mut c_void,
    native_ple_key: c_int,
    out: *mut TkQwen4ExpP1PleWeights,
    err: *mut c_char,
    err_cap: usize,
) -> c_int;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpP1NativeAttach {
    pub name: *const c_char,
    pub native_data: *mut c_void,
    pub native_type: i32,
    pub native_q8_1: *mut c_void,
    pub hc_q8: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpP1NativeDenseOut {
    pub weights: *const *mut c_void,
    pub n_weights: usize,
    pub packed_keys: *const *const c_void,
    pub n_packed: usize,
    pub scratch: *mut c_void,
    pub bytes: u64,
    pub allocated: u64,
    pub attach: *const TkQwen4ExpP1NativeAttach,
    pub n_attach: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpP1NativeHeadOut {
    pub weights: *mut c_void,
    pub scratch: *mut c_void,
    pub bytes: u64,
    pub n_in: i32,
    pub n_out: i32,
    pub ty: i32,
    pub device: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpP1NativeEmbedOut {
    pub host: *mut c_void,
    pub dev: *const c_void,
    pub bytes: u64,
    pub row: u64,
    pub n_embd: i64,
    pub n_vocab: i64,
    pub ty: i32,
}

pub type FnP1NativeServedNames = unsafe extern "C" fn(
    user: *mut c_void,
    shards: *const *const c_char,
    n_shards: usize,
    include_ple_key: c_int,
    out_names: *mut *const *const c_char,
    out_n: *mut usize,
    err: *mut c_char,
    err_cap: usize,
) -> c_int;
pub type FnP1NativeDenseLoad = unsafe extern "C" fn(
    user: *mut c_void,
    shards: *const *const c_char,
    n_shards: usize,
    include_ple_key: c_int,
    out: *mut TkQwen4ExpP1NativeDenseOut,
    err: *mut c_char,
    err_cap: usize,
) -> c_int;
pub type FnP1NativeHeadLoad = unsafe extern "C" fn(
    user: *mut c_void,
    shards: *const *const c_char,
    n_shards: usize,
    n_in: i64,
    n_out: i64,
    out: *mut TkQwen4ExpP1NativeHeadOut,
    err: *mut c_char,
    err_cap: usize,
) -> c_int;
pub type FnP1NativeEmbedLoad = unsafe extern "C" fn(
    user: *mut c_void,
    shards: *const *const c_char,
    n_shards: usize,
    n_embd: i64,
    n_vocab: i64,
    out: *mut TkQwen4ExpP1NativeEmbedOut,
    err: *mut c_char,
    err_cap: usize,
) -> c_int;

pub type FnP1ResCreate = unsafe extern "C" fn(
    user: *mut c_void,
    host_res: *const i32,
    n: usize,
    out_d_res: *mut *mut i32,
    out_d_hit_count: *mut *mut i32,
    err: *mut c_char,
    err_cap: usize,
) -> c_int;
pub type FnP1LendInit = unsafe extern "C" fn(
    user: *mut c_void,
    ctx: *const TkQwen4ExpP1LendCtx,
    err: *mut c_char,
    err_cap: usize,
) -> c_int;
pub type FnP1Lend = unsafe extern "C" fn(
    user: *mut c_void,
    tokens: i64,
    out_lent: *mut i32,
    err: *mut c_char,
    err_cap: usize,
) -> c_int;
pub type FnP1Refill =
    unsafe extern "C" fn(user: *mut c_void, err: *mut c_char, err_cap: usize) -> c_int;

pub type FnP1ProfileRead = unsafe extern "C" fn(
    user: *mut c_void,
    path: *const c_char,
    n_layers: i64,
    n_expert: i64,
    out_pairs: *mut *const i32,
    out_n: *mut usize,
    out_slots: *mut i64,
    err: *mut c_char,
    err_cap: usize,
) -> c_int;

pub type FnP1CachePlan = unsafe extern "C" fn(
    user: *mut c_void,
    inp: *const TkQwen4ExpP1CachePlanIn,
    out: *mut TkQwen4ExpP1CachePlanOut,
) -> c_int;

pub type FnP1CacheBuild = unsafe extern "C" fn(
    user: *mut c_void,
    inp: *const TkQwen4ExpP1CacheBuildIn,
    out: *mut TkQwen4ExpP1CacheBuildOut,
    err: *mut c_char,
    err_cap: usize,
) -> c_int;

unsafe impl Send for TkQwen4ExpP1RustHooks {}
unsafe impl Sync for TkQwen4ExpP1RustHooks {}

type FnP1SetRustHooks = unsafe extern "C" fn(hooks: *const TkQwen4ExpP1RustHooks);

/// The Rust side's process-wide Piece 1 state: what the hooks loaded, kept alive for the C++ engine's lifetime
/// (its `WeightRef`s point into `rows`' names) and adopted by [`Piece1Weights::sync_from_ctx`].
#[derive(Default)]
pub(super) struct P1Global {
    weights: Option<WeightTable>,
    row_names: Vec<CString>,
    rows: Vec<TkQwen4ExpP1WeightRow>,
    profile: Vec<i32>,
    plan_sized: Vec<i64>,
    /// the built cache's tables (the arena itself is the C++ ExpertCache's once adopted)
    cache: Option<ExpertCacheArena>,
    /// the device residency table and the hit counter (Rust-allocated, live for the process)
    d_res: usize,
    d_hit_count: usize,
    n_res: usize,
    /// the serve loan
    lend: Option<ServeLend>,
    /// generate's one-shot loan
    gen_lend: Option<ServeLend>,
    /// the native GGUF loads (the C++ objects own the device buffers; these keep the arrays handed over alive)
    served_names: Vec<CString>,
    served_ptrs: Vec<*const c_char>,
    native: Option<NativeDense>,
    native_names: Vec<CString>,
    native_attach: Vec<TkQwen4ExpP1NativeAttach>,
    native_weights: Vec<*mut c_void>,
    native_packed: Vec<*const c_void>,
    head: Option<NativeHeadBuf>,
    embed: Option<NativeEmbedBuf>,
    ple: Option<TkQwen4ExpP1PleWeights>,
}

unsafe impl Send for P1Global {}

static P1_GLOBAL: Mutex<Option<P1Global>> = Mutex::new(None);
static P1_HOOKS: Mutex<Option<Box<TkQwen4ExpP1RustHooks>>> = Mutex::new(None);

pub(super) fn with_global<R>(f: impl FnOnce(&mut P1Global) -> R) -> R {
    let mut g = P1_GLOBAL.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(P1Global::default))
}

pub(super) unsafe fn write_err(err: *mut c_char, cap: usize, msg: &str) {
    if err.is_null() || cap == 0 {
        return;
    }
    let b = msg.as_bytes();
    let n = b.len().min(cap - 1);
    std::ptr::copy_nonoverlapping(b.as_ptr().cast::<c_char>(), err, n);
    *err.add(n) = 0;
}

pub(super) unsafe fn skip_set(skip: *const *const c_char, n: usize) -> Option<HashSet<String>> {
    if skip.is_null() {
        return None;
    }
    let mut s = HashSet::with_capacity(n);
    for i in 0..n {
        let p = *skip.add(i);
        if !p.is_null() {
            s.insert(CStr::from_ptr(p).to_string_lossy().into_owned());
        }
    }
    Some(s)
}

unsafe extern "C" fn hook_pack_pool_bytes(
    _user: *mut c_void,
    pack_dir: *const c_char,
    skip: *const *const c_char,
    n_skip: usize,
    out_bytes: *mut u64,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    if pack_dir.is_null() || out_bytes.is_null() {
        return -1;
    }
    let dir = CStr::from_ptr(pack_dir).to_string_lossy().into_owned();
    let skip = skip_set(skip, n_skip);
    match PackIndex::pool_bytes(&dir, skip.as_ref()) {
        Ok(b) => {
            *out_bytes = b;
            0
        }
        Err(e) => {
            write_err(err, err_cap, &e);
            -1
        }
    }
}

unsafe extern "C" fn hook_pack_load(
    _user: *mut c_void,
    pack_dir: *const c_char,
    arena: *mut c_void,
    arena_bytes: u64,
    skip: *const *const c_char,
    n_skip: usize,
    out_rows: *mut *const TkQwen4ExpP1WeightRow,
    out_n_rows: *mut usize,
    out_report: *mut TkQwen4ExpP1LoadReport,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    if pack_dir.is_null() || out_rows.is_null() || out_n_rows.is_null() {
        return -1;
    }
    let dir = CStr::from_ptr(pack_dir).to_string_lossy().into_owned();
    let skip = skip_set(skip, n_skip);
    let wt = match WeightTable::load_device(&dir, arena, arena_bytes, skip.as_ref()) {
        Ok(t) => t,
        Err(e) => {
            write_err(err, err_cap, &e);
            return -1;
        }
    };
    with_global(|g| {
        g.row_names.clear();
        g.rows.clear();
        for (name, r) in &wt.table {
            g.row_names
                .push(CString::new(name.as_bytes()).unwrap_or_default());
            let np = g.row_names.last().unwrap().as_ptr();
            g.rows.push(TkQwen4ExpP1WeightRow {
                name: np,
                data: r.data,
                bytes: r.bytes,
                ne0: r.ne0,
                ne1: r.ne1,
                elements: r.elements,
                kind: r.kind as i32,
                code_bits: r.code_bits,
                code_bias: r.code_bias,
                group_elems: r.group_elems,
                codebook_iq4nl: r.codebook_iq4nl as i32,
                has_offset: r.has_offset as i32,
                codes_bytes: r.codes_bytes,
                scales_bytes: r.scales_bytes,
                offset_bytes: r.offset_bytes,
                src_off: r.src_off,
                src_bytes: r.src_bytes,
                file_id: r.file_id,
                scales_fp16: r.scales_fp16 as i32,
                act_kind: r.act_kind,
                resident: r.resident as i32,
            });
        }
        if !out_report.is_null() {
            *out_report = TkQwen4ExpP1LoadReport {
                tensors: wt.report.tensors as u64,
                arena_bytes: wt.report.arena_bytes,
                re_rounded: wt.report.re_rounded as u64,
                bytes_saved: wt.report.bytes_saved,
                read_ms: wt.report.read_ms,
                upload_ms: wt.report.upload_ms,
            };
        }
        *out_rows = g.rows.as_ptr();
        *out_n_rows = g.rows.len();
        g.weights = Some(wt);
    });
    0
}

unsafe extern "C" fn hook_profile_read(
    _user: *mut c_void,
    path: *const c_char,
    n_layers: i64,
    n_expert: i64,
    out_pairs: *mut *const i32,
    out_n: *mut usize,
    out_slots: *mut i64,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    if path.is_null() || out_pairs.is_null() || out_n.is_null() || out_slots.is_null() {
        return -1;
    }
    let p = CStr::from_ptr(path).to_string_lossy().into_owned();
    match read_expert_profile(&p, n_layers, n_expert) {
        Ok((ranked, slots)) => with_global(|g| {
            g.profile = ranked.iter().flat_map(|&(l, e)| [l, e]).collect();
            *out_pairs = g.profile.as_ptr();
            *out_n = ranked.len();
            *out_slots = slots;
            0
        }),
        Err(e) => {
            write_err(err, err_cap, &e);
            -1
        }
    }
}

pub(super) unsafe fn pairs_of(p: *const i32, n: usize) -> Vec<(i32, i32)> {
    if p.is_null() || n == 0 {
        return Vec::new();
    }
    let s = std::slice::from_raw_parts(p, n * 2);
    (0..n).map(|i| (s[i * 2], s[i * 2 + 1])).collect()
}

unsafe extern "C" fn hook_cache_plan(
    _user: *mut c_void,
    inp: *const TkQwen4ExpP1CachePlanIn,
    out: *mut TkQwen4ExpP1CachePlanOut,
) -> c_int {
    if inp.is_null() || out.is_null() {
        return -1;
    }
    let i = &*inp;
    if i.blob_bytes.is_null() || i.n_layers <= 0 {
        return -1;
    }
    let ci = CachePlanInput {
        n_layers: i.n_layers,
        n_expert: i.n_expert,
        expert_cache: i.expert_cache as i64,
        native_pack: i.native_pack != 0,
        per_layer: i.per_layer != 0,
        multi_gpu: i.multi_gpu != 0,
        vram_reserve_mib: i.vram_reserve_mib as i64,
        vram_reserve_given: i.vram_reserve_given != 0,
        prefill_mib: i.prefill_mib,
        mtp_bind: i.mtp_bind,
        min_slots: i.min_slots,
        session_mib: i.session_mib,
        max_context: i.max_context,
        pipe_first: i.pipe_first,
        max_blob: i.max_blob,
        blob_bytes: std::slice::from_raw_parts(i.blob_bytes, i.n_layers as usize).to_vec(),
        profile: pairs_of(i.profile, i.n_profile),
        profile_path: if i.profile_path.is_null() {
            String::new()
        } else {
            CStr::from_ptr(i.profile_path)
                .to_string_lossy()
                .into_owned()
        },
    };
    let plan = plan_cache(&ci, device_free_bytes);
    with_global(|g| {
        g.plan_sized = plan.sized_slots.clone();
        *out = TkQwen4ExpP1CachePlanOut {
            expert_cache: plan.expert_cache.clamp(0, i32::MAX as i64) as i32,
            vram_reserve_mib: plan.vram_reserve_mib as i32,
            reserve_adapted: plan.reserve_adapted as i32,
            sized_slots: g.plan_sized.as_ptr(),
            n_sized: g.plan_sized.len(),
            per_layer_bytes: plan.per_layer_bytes,
        };
    });
    0
}

unsafe extern "C" fn hook_cache_build(
    _user: *mut c_void,
    inp: *const TkQwen4ExpP1CacheBuildIn,
    out: *mut TkQwen4ExpP1CacheBuildOut,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    if inp.is_null() || out.is_null() {
        return -1;
    }
    let i = &*inp;
    if i.blob_bytes.is_null() || i.n_layers <= 0 {
        return -1;
    }
    let blob_bytes = std::slice::from_raw_parts(i.blob_bytes, i.n_layers as usize).to_vec();
    let sized = if i.sized_slots.is_null() {
        Vec::new()
    } else {
        std::slice::from_raw_parts(i.sized_slots, i.n_sized).to_vec()
    };
    let profile = pairs_of(i.profile, i.n_profile);
    let (bf, bctx) = (i.blob_fn, i.blob_ctx);
    let mut blob = |l: i32, e: i32| -> *const u8 {
        match bf {
            Some(f) => f(bctx, l, e),
            None => std::ptr::null(),
        }
    };
    let res = build_cache(
        i.n_layers,
        i.n_expert,
        i.expert_cache as i64,
        i.auto_cache != 0,
        i.per_layer != 0,
        i.vram_reserve_mib as i64,
        i.pipe_first,
        i.max_blob,
        &blob_bytes,
        sized,
        i.per_layer_bytes,
        if bf.is_some() { &profile } else { &[] },
        &mut blob,
        match (i.file_source != 0, i.advise_pairs, i.prefetch_pairs) {
            (true, Some(a), Some(p)) => Some(FileTierFill {
                unbuffered: i.unbuffered != 0,
                ctx: i.blob_ctx as usize,
                advise_pairs: a,
                prefetch_pairs: p,
                pin_aux_thread: i.pin_aux_thread,
            }),
            _ => None,
        }
        .as_ref(),
    );
    let b = match res {
        Ok(b) => b,
        Err(e) => {
            write_err(err, err_cap, &e);
            return -1;
        }
    };
    with_global(|g| {
        let mut o = TkQwen4ExpP1CacheBuildOut {
            base: std::ptr::null_mut(),
            slots: 0,
            blob: 0,
            offsets: std::ptr::null(),
            n_offsets: 0,
            residency: std::ptr::null(),
            n_residency: 0,
            layer_next: std::ptr::null(),
            n_layer_next: 0,
            next_free: 0,
            admitted: 0,
            fills: 0,
            prefilled: b.prefilled,
            expert_cache: b.expert_cache.clamp(0, i32::MAX as i64) as i32,
            sized_changed: 0,
        };
        g.plan_sized = b.sized_slots;
        if let Some(mut c) = b.cache {
            // the C++ ExpertCache owns (and frees) the arena from here; Rust keeps the tables
            c.owned = false;
            o.base = c.base;
            o.slots = c.slots;
            o.blob = c.blob;
            o.offsets = if c.off.is_empty() {
                std::ptr::null()
            } else {
                c.off.as_ptr()
            };
            o.n_offsets = c.off.len();
            o.residency = c.residency.as_ptr();
            o.n_residency = c.residency.len();
            o.layer_next = c.layer_next.as_ptr();
            o.n_layer_next = c.layer_next.len();
            o.next_free = c.next_free;
            o.admitted = c.admitted;
            o.fills = c.fills;
            g.cache = Some(c);
        } else {
            g.cache = None;
        }
        *out = o;
    });
    0
}

unsafe extern "C" fn hook_res_create(
    _user: *mut c_void,
    host_res: *const i32,
    n: usize,
    out_d_res: *mut *mut i32,
    out_d_hit_count: *mut *mut i32,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    if host_res.is_null() || out_d_res.is_null() || out_d_hit_count.is_null() || n == 0 {
        return -1;
    }
    let mut d: *mut c_void = std::ptr::null_mut();
    let mut h: *mut c_void = std::ptr::null_mut();
    if hip_sys::hipMalloc(&mut d, n * 4) != hip_sys::hipSuccess
        || hip_sys::hipMalloc(&mut h, 4) != hip_sys::hipSuccess
    {
        write_err(
            err,
            err_cap,
            "the device residency table could not be staged",
        );
        return -1;
    }
    if let Err(e) = res_put(d.cast(), host_res, n) {
        write_err(
            err,
            err_cap,
            &format!("the device residency table could not be staged ({e})"),
        );
        return -1;
    }
    *out_d_res = d.cast();
    *out_d_hit_count = h.cast();
    with_global(|g| {
        g.d_res = d as usize;
        g.d_hit_count = h as usize;
        g.n_res = n;
    });
    0
}

unsafe extern "C" fn hook_lend_init(
    _user: *mut c_void,
    ctx: *const TkQwen4ExpP1LendCtx,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    if ctx.is_null() {
        return -1;
    }
    let c = &*ctx;
    if c.active != 0 && c.first >= 0 {
        eprintln!(
            "talus qwen4exp-p1: the serve loan is Rust's (slots {}.. of {}, chunk {})",
            c.first, c.cache_slots, c.prefill_chunk
        );
    }
    match ServeLend::new(*ctx) {
        Ok(l) => with_global(|g| {
            g.lend = Some(l);
            0
        }),
        Err(e) => {
            write_err(err, err_cap, &e);
            -1
        }
    }
}

unsafe extern "C" fn hook_lend(
    _user: *mut c_void,
    tokens: i64,
    out_lent: *mut i32,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    with_global(|g| {
        let Some(l) = g.lend.as_mut() else {
            write_err(err, err_cap, "qwen4exp-p1: lend before lend_init");
            return -1;
        };
        match l.lend(tokens) {
            Ok(()) => {
                if !out_lent.is_null() {
                    *out_lent = l.lent_count() as i32;
                }
                0
            }
            Err(e) => {
                write_err(err, err_cap, &e);
                -1
            }
        }
    })
}

unsafe extern "C" fn hook_refill(_user: *mut c_void, err: *mut c_char, err_cap: usize) -> c_int {
    with_global(|g| {
        let Some(l) = g.lend.as_mut() else { return 0 };
        match l.refill() {
            Ok(()) => 0,
            Err(e) => {
                write_err(err, err_cap, &e);
                -1
            }
        }
    })
}

pub(super) unsafe fn shard_list(shards: *const *const c_char, n: usize) -> Vec<String> {
    if shards.is_null() {
        return Vec::new();
    }
    (0..n)
        .filter_map(|i| {
            let p = *shards.add(i);
            (!p.is_null()).then(|| CStr::from_ptr(p).to_string_lossy().into_owned())
        })
        .collect()
}

unsafe extern "C" fn hook_native_served_names(
    _user: *mut c_void,
    shards: *const *const c_char,
    n_shards: usize,
    include_ple_key: c_int,
    out_names: *mut *const *const c_char,
    out_n: *mut usize,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    if out_names.is_null() || out_n.is_null() {
        return -1;
    }
    match native_served_names(&shard_list(shards, n_shards), include_ple_key != 0) {
        Ok(set) => with_global(|g| {
            let mut v: Vec<String> = set.into_iter().collect();
            v.sort();
            g.served_names = v
                .iter()
                .map(|s| CString::new(s.as_bytes()).unwrap_or_default())
                .collect();
            g.served_ptrs = g.served_names.iter().map(|c| c.as_ptr()).collect();
            *out_names = g.served_ptrs.as_ptr();
            *out_n = g.served_ptrs.len();
            0
        }),
        Err(e) => {
            write_err(err, err_cap, &e);
            -1
        }
    }
}

unsafe extern "C" fn hook_native_dense_load(
    _user: *mut c_void,
    shards: *const *const c_char,
    n_shards: usize,
    include_ple_key: c_int,
    out: *mut TkQwen4ExpP1NativeDenseOut,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    if out.is_null() {
        return -1;
    }
    let shards = shard_list(shards, n_shards);
    with_global(|g| {
        let Some(table) = g.weights.as_ref() else {
            write_err(
                err,
                err_cap,
                "native dense: the Rust weight table is not loaded",
            );
            return -1;
        };
        let loaded = native_dense_load(&shards, table, include_ple_key != 0);
        gguf_release();
        let nd = match loaded {
            Ok(nd) => nd,
            Err(e) => {
                write_err(err, err_cap, &e);
                return -1;
            }
        };
        g.native_names = nd
            .attach
            .keys()
            .map(|k| CString::new(k.as_bytes()).unwrap_or_default())
            .collect();
        g.native_attach = nd
            .attach
            .values()
            .zip(g.native_names.iter())
            .map(|(a, n)| TkQwen4ExpP1NativeAttach {
                name: n.as_ptr(),
                native_data: a.native_data as *mut c_void,
                native_type: if a.native_data != 0 {
                    a.native_type
                } else {
                    -1
                },
                native_q8_1: a.native_q8_1 as *mut c_void,
                hc_q8: a.hc_q8 as *mut c_void,
            })
            .collect();
        g.native_weights = nd.weights.iter().map(|&p| p as *mut c_void).collect();
        g.native_packed = nd.packed_keys.iter().map(|&p| p as *const c_void).collect();
        *out = TkQwen4ExpP1NativeDenseOut {
            weights: g.native_weights.as_ptr(),
            n_weights: g.native_weights.len(),
            packed_keys: g.native_packed.as_ptr(),
            n_packed: g.native_packed.len(),
            scratch: nd.scratch as *mut c_void,
            bytes: nd.bytes,
            allocated: nd.allocated,
            attach: g.native_attach.as_ptr(),
            n_attach: g.native_attach.len(),
        };
        g.native = Some(nd);
        0
    })
}

unsafe extern "C" fn hook_native_head_load(
    _user: *mut c_void,
    shards: *const *const c_char,
    n_shards: usize,
    n_in: i64,
    n_out: i64,
    out: *mut TkQwen4ExpP1NativeHeadOut,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    if out.is_null() {
        return -1;
    }
    match native_head_load(&shard_list(shards, n_shards), n_in, n_out) {
        Ok(h) => {
            *out = TkQwen4ExpP1NativeHeadOut {
                weights: h.weights as *mut c_void,
                scratch: h.scratch as *mut c_void,
                bytes: h.bytes,
                n_in: h.n_in,
                n_out: h.n_out,
                ty: h.ty,
                device: h.device,
            };
            with_global(|g| g.head = Some(h));
            0
        }
        Err(e) => {
            write_err(err, err_cap, &e);
            -1
        }
    }
}

unsafe extern "C" fn hook_native_embed_load(
    _user: *mut c_void,
    shards: *const *const c_char,
    n_shards: usize,
    n_embd: i64,
    n_vocab: i64,
    out: *mut TkQwen4ExpP1NativeEmbedOut,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    if out.is_null() {
        return -1;
    }
    match native_embed_load(&shard_list(shards, n_shards), n_embd, n_vocab) {
        Ok(e) => {
            *out = TkQwen4ExpP1NativeEmbedOut {
                host: e.host as *mut c_void,
                dev: e.dev as *const c_void,
                bytes: e.bytes,
                row: e.row,
                n_embd: e.n_embd,
                n_vocab: e.n_vocab,
                ty: e.ty,
            };
            with_global(|g| g.embed = Some(e));
            0
        }
        Err(e) => {
            write_err(err, err_cap, &e);
            -1
        }
    }
}

unsafe extern "C" fn hook_ple_wire(
    _user: *mut c_void,
    native_ple_key: c_int,
    out: *mut TkQwen4ExpP1PleWeights,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    if out.is_null() {
        return -1;
    }
    with_global(|g| {
        let Some(table) = g.weights.as_ref() else {
            write_err(
                err,
                err_cap,
                "qwen4exp-p1: the Rust weight table is not loaded",
            );
            return -1;
        };
        match ple_wire(table, g.native.as_ref(), native_ple_key != 0) {
            Ok(w) => {
                *out = w;
                g.ple = Some(w);
                0
            }
            Err(e) => {
                write_err(err, err_cap, &e);
                -1
            }
        }
    })
}

unsafe extern "C" fn hook_plan_lend(
    _user: *mut c_void,
    planner: *const TkQwen4ExpP1Planner,
    chunk_inout: *mut i64,
) -> i64 {
    if planner.is_null() || chunk_inout.is_null() {
        return -1;
    }
    let lp = LendPlanner::from_c(&*planner);
    let mut pr = CPricing(*planner);
    lp.plan_lend(&mut pr, &mut *chunk_inout)
}

unsafe extern "C" fn hook_lend_slots(
    _user: *mut c_void,
    planner: *const TkQwen4ExpP1Planner,
    chunk: i64,
) -> i64 {
    if planner.is_null() {
        return -1;
    }
    let lp = LendPlanner::from_c(&*planner);
    let mut pr = CPricing(*planner);
    lp.lend_slots(&mut pr, chunk)
}

unsafe extern "C" fn hook_serve_scan(
    _user: *mut c_void,
    planner: *const TkQwen4ExpP1Planner,
    out_chunk: *mut i64,
    out_first: *mut i64,
) -> c_int {
    if planner.is_null() || out_chunk.is_null() || out_first.is_null() {
        return -1;
    }
    let lp = LendPlanner::from_c(&*planner);
    let mut pr = CPricing(*planner);
    let (c, f) = lp.serve_scan(&mut pr);
    *out_chunk = c;
    *out_first = f;
    0
}

unsafe extern "C" fn hook_ple_io_plan(
    _user: *mut c_void,
    ple_io: *const c_char,
    inflight: i64,
    row_cache: i64,
    sync_submit: c_int,
    out: *mut TkQwen4ExpP1PleIo,
) -> c_int {
    if out.is_null() {
        return -1;
    }
    let io = if ple_io.is_null() {
        String::new()
    } else {
        CStr::from_ptr(ple_io).to_string_lossy().into_owned()
    };
    *out = ple_io_plan(&io, inflight, row_cache, sync_submit != 0);
    0
}

unsafe extern "C" fn hook_ple_probe_decide(_user: *mut c_void, rps: f64, explicit: c_int) -> c_int {
    ple_probe_decide(rps, explicit != 0)
}

/// What the Rust loaders hold for the other Rust pieces (copies of the tables; the device buffers are the C++
/// objects' once adopted and live until exit).
#[derive(Clone, Debug, Default)]
pub struct Piece1Snapshot {
    /// the canonical arena table
    pub weights: Option<WeightTable>,
    /// the native GGUF attachments by canonical name (projections, hc Q8_0 copies)
    pub native: Option<NativeDense>,
    pub head: Option<NativeHeadBuf>,
    pub embed: Option<NativeEmbedBuf>,
    pub ple: Option<TkQwen4ExpP1PleWeights>,
    /// the device residency table and the hit counter
    pub d_res: usize,
    pub d_hit_count: usize,
    pub n_res: usize,
    /// the expert cache: base, slots, largest slot, sized offsets (empty: uniform), the residency it was filled with
    pub cache_base: usize,
    pub cache_slots: i64,
    pub cache_blob: i64,
    pub cache_offsets: Vec<u64>,
    pub cache_residency: Vec<i32>,
}

/// A snapshot of everything Piece 1 loaded in Rust (empty on the C++ path).
pub fn piece1_snapshot() -> Piece1Snapshot {
    with_global(|g| Piece1Snapshot {
        weights: g.weights.clone(),
        native: g.native.clone(),
        head: g.head.clone(),
        embed: g.embed.clone(),
        ple: g.ple,
        d_res: g.d_res,
        d_hit_count: g.d_hit_count,
        n_res: g.n_res,
        cache_base: g.cache.as_ref().map(|c| c.base as usize).unwrap_or(0),
        cache_slots: g.cache.as_ref().map(|c| c.slots).unwrap_or(0),
        cache_blob: g.cache.as_ref().map(|c| c.blob).unwrap_or(0),
        cache_offsets: g.cache.as_ref().map(|c| c.off.clone()).unwrap_or_default(),
        cache_residency: g
            .cache
            .as_ref()
            .map(|c| c.residency.clone())
            .unwrap_or_default(),
    })
}

unsafe extern "C" fn hook_gen_lend(
    _user: *mut c_void,
    ctx: *const TkQwen4ExpP1LendCtx,
    first: i32,
    err: *mut c_char,
    err_cap: usize,
) -> i64 {
    if ctx.is_null() {
        return -1;
    }
    let mut l = match ServeLend::new(*ctx) {
        Ok(l) => l,
        Err(e) => {
            write_err(err, err_cap, &e);
            return -1;
        }
    };
    match l.take_from(first) {
        Ok(k) => with_global(|g| {
            g.gen_lend = Some(l);
            k as i64
        }),
        Err(e) => {
            write_err(err, err_cap, &e);
            -1
        }
    }
}

unsafe extern "C" fn hook_gen_refill(_user: *mut c_void, err: *mut c_char, err_cap: usize) -> i64 {
    with_global(|g| {
        let Some(l) = g.gen_lend.as_mut() else {
            return 0;
        };
        let n = l.lent_count() as i64;
        match l.refill() {
            Ok(()) => n,
            Err(e) => {
                write_err(err, err_cap, &e);
                -1
            }
        }
    })
}

pub(super) const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();
extern "C" {
    pub(super) fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

/// Registers the Piece 1 hook table with `libtalus_kernels.so` (once per process). The C++ engine only calls the
/// hooks when the Rust data plane is on (`TALUS_FN_RUST=1` / `--rust-engine`), and `TALUS_FN_P1=0` keeps its own
/// code even then (bisection switch). No-op when the library is not loaded (unit tests).
pub fn install_rust_hooks() -> bool {
    let mut slot = P1_HOOKS.lock().unwrap_or_else(|e| e.into_inner());
    if slot.is_some() {
        return true;
    }
    let sym = unsafe { dlsym(RTLD_DEFAULT, c"tk_qwen4exp_p1_set_rust_hooks".as_ptr()) };
    if sym.is_null() {
        return false;
    }
    let set: FnP1SetRustHooks = unsafe { std::mem::transmute(sym) };
    let hooks = Box::new(TkQwen4ExpP1RustHooks {
        user_data: std::ptr::null_mut(),
        pack_pool_bytes: Some(hook_pack_pool_bytes),
        pack_load: Some(hook_pack_load),
        profile_read: Some(hook_profile_read),
        cache_plan: Some(hook_cache_plan),
        cache_build: Some(hook_cache_build),
        res_create: Some(hook_res_create),
        lend_init: Some(hook_lend_init),
        lend: Some(hook_lend),
        refill: Some(hook_refill),
        native_served_names: Some(hook_native_served_names),
        native_dense_load: Some(hook_native_dense_load),
        native_head_load: Some(hook_native_head_load),
        native_embed_load: Some(hook_native_embed_load),
        ple_wire: Some(hook_ple_wire),
        plan_lend: Some(hook_plan_lend),
        lend_slots: Some(hook_lend_slots),
        serve_scan: Some(hook_serve_scan),
        gen_lend: Some(hook_gen_lend),
        gen_refill: Some(hook_gen_refill),
        ple_io_plan: Some(hook_ple_io_plan),
        ple_probe_decide: Some(hook_ple_probe_decide),
    });
    unsafe { set(&*hooks) };
    *slot = Some(hooks);
    true
}

// ---- qwen4exp-r1 begin ----
/// qwen4exp-r1: Rust owns what Piece 1 allocated for the C++ objects that adopted it - with the r1 factory live they only
/// borrow it (`r1_borrow`) - and frees it here, once the C++ engine's objects are gone (generate.cpp's release guard,
/// through `tk_qwen4exp_r1_rust_hooks_t::release_all`): the expert cache arena, the native projections and their
/// scratch, the output head, the token embedding.
pub(super) fn r1_release_handed_over() {
    with_global(|g| unsafe {
        if let Some(c) = g.cache.as_mut() {
            if !c.owned && !c.base.is_null() {
                let _ = hip_sys::hipFree(c.base.cast());
                c.base = std::ptr::null_mut();
            }
        }
        if let Some(nd) = g.native.take() {
            for &w in &nd.weights {
                if w != 0 {
                    let _ = hip_sys::hipFree(w as *mut c_void);
                }
            }
            if nd.scratch != 0 {
                let _ = hip_sys::hipFree(nd.scratch as *mut c_void);
            }
        }
        g.native_weights.clear();
        if let Some(h) = g.head.take() {
            if h.scratch != 0 {
                let _ = hip_sys::hipFree(h.scratch as *mut c_void);
            }
            if h.weights != 0 {
                let _ = hip_sys::hipFree(h.weights as *mut c_void);
            }
        }
        if let Some(e) = g.embed.take() {
            if e.host != 0 {
                let _ = hip_sys::hipHostFree(e.host as *mut c_void);
            } else if e.dev != 0 {
                let _ = hip_sys::hipFree(e.dev as *mut c_void);
            }
        }
    });
}
// ---- qwen4exp-r1 end ----
