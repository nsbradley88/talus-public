//! qwen4exp-r2 (Piece 4 residuals): `Verifier::init`, `Verifier::set_logit_bias`, `MtpDrafter::load` and
//! `MtpDrafter::bind` in Rust.
//!
//! Rust makes every allocation, stream and event the C++ bodies made, in the same order and with the same sizes, lays
//! out the mapped staging, the GPU plan block and the device arenas (the C++ `Bump` sequence: each buffer rounded up to
//! 256 bytes), reads the drafter's files, and decides the env switches.  The C++ objects then ADOPT the result through
//! `tk_qwen4exp_r2v_*_adopt` (src/program/qwen4exp_r2v.cpp, a friend of both classes) and free it in their destructors,
//! as before.  What stays C++: the objects' registration statics (verify.cpp's live list / watchdog), the K/V state
//! carve (layer.cpp's `qsa_state_*`, called through thin wrappers), the Q4_0 conversions (ggml) and the coupled
//! sampler's setup.  Declined (the C++ body runs): layer-split stages / pipelined windows (`set_stream`), shared
//! weights, the opt-in traces (STRATA_VERIFY_TRACE / STRATA_VERIFY_PROFILE), a draft head on another device, gfx906.
//!
//! TALUS_FN_R2V=0 keeps every C++ body; TALUS_FN_R2V_CHECK=1 compares each layout against the C++ body's on a
//! scratch object (qwen4exp_r2v.cpp).

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::io::Read;
use std::os::fd::AsRawFd;
use std::sync::OnceLock;
use std::time::Instant;

use hip_sys::{
    hipDeviceSynchronize, hipEventCreateWithFlags, hipEventDisableTiming, hipEvent_t, hipFree,
    hipGetLastError, hipHostGetDevicePointer, hipHostMalloc, hipHostMallocCoherent,
    hipHostMallocMapped, hipHostMallocPortable, hipMalloc, hipMemGetInfo, hipMemcpy,
    hipMemcpyHostToDevice, hipMemset, hipStreamCreateWithFlags, hipStreamNonBlocking, hipStream_t,
    hipSuccess,
};

use super::{env_flag, hip_err_str, put_c_err};
use crate::qwen4exp::ffi::{load_sym_from, Qwen4ExpKernels};

// ============================================================================
// C mirrors (include/talus_qwen4exp_kernels.h, qwen4exp-r2v block)
// ============================================================================

const VER_MAPPED: usize = 17;
const VER_ARENA: usize = 63;
const MTP_MAPPED: usize = 7;
const MTP_ARENA: usize = 51;

#[repr(C)]
struct R2vOps {
    user: *mut c_void,
    ver_init: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_char, usize) -> c_int>,
    ver_set_logit_bias: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *const f32,
            i64,
            *mut c_char,
            usize,
        ) -> c_int,
    >,
    mtp_load:
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void, i64, *mut c_char, usize) -> c_int>,
    mtp_bind: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_char, usize) -> c_int>,
}

#[repr(C)]
struct VerFacts {
    eligible: i32,
    max_t: i32,
    layer_ok: i32,
    fused_gr_ok: i32,
    always_publish: i32,
    peer_portable: i32,
    is_gfx12: i32,
    hip_gfx906: i32,
    device: i32,
    k: i32,
    ssm_state_size: i32,
    ssm_d_conv: i32,
    n_layers: i32,
    n_expert: i32,
    n_embd: i32,
    hc: i32,
    hc_lr: i32,
    conv_ch: i32,
    value_dim: i32,
    v_heads: i32,
    n_head: i32,
    head_dim: i32,
    n_head_kv: i32,
    idx_q_heads: i32,
    idx_key_dim: i32,
    n_ff: i32,
    n_gdn: i32,
    n_qsa: i32,
    verify_max_t: i32,
    cpu_maxt: i32,
    step_count: i32,
    nonfinite_blocks: i32,
    ng_hist: i32,
    ng_hc_dim: i32,
    n_vocab: i64,
    max_cells: i64,
    idx_block: i64,
    cap: i64,
    attn_scratch_floats: i64,
    max_blob: u64,
    q8_1_bytes: u64,
    q8_1_il_bytes: u64,
    hit_grouped_scratch: u64,
    native_expert_scratch: u64,
    argmax_scratch: u64,
    d_res: *const i32,
    h_res: *const i32,
    cache_base: *const u8,
    slot_off: *const u64,
    n_slots: i64,
    blob: i64,
    why: [c_char; 256],
}

#[repr(C)]
struct VerState {
    mapped_h: [*mut c_void; VER_MAPPED],
    mapped_d: [*mut c_void; VER_MAPPED],
    plan_i32: i64,
    sink_counts: *mut i32,
    sink_start: *mut i32,
    sink_dst: *mut i32,
    sink_tok: *mut i32,
    sink_ptr: *mut u64,
    sink_ptr2: *mut u64,
    sink_start2: *mut i32,
    sink_cap: i64,
    staging_cap: i64,
    arena: *mut c_void,
    arena_bytes: u64,
    arena_ptrs: [*mut c_void; VER_ARENA],
    n_vocab: i64,
    cap: i64,
    max_blocks: i64,
    attn_scratch_floats: i64,
    le: i64,
    copy: *mut c_void,
    cs: *mut c_void,
    sh_cs: *mut c_void,
    commit_done: *mut c_void,
    ev_fork: *mut c_void,
    ev_join: *mut c_void,
    df_branch: i32,
    df_fork: *mut c_void,
    df_side: [*mut c_void; 2],
    df_join: [*mut c_void; 2],
    all_resident: i32,
    device_plan: i32,
    qcnt: *mut c_void,
    skip: *mut c_void,
    slot_off_d: *mut c_void,
}

#[repr(C)]
struct MtpFacts {
    eligible: i32,
    max_t: i32,
    device: i32,
    q4: i32,
    q4_head: i32,
    force_on: i32,
    hip: i32,
    hip_gfx906: i32,
    peer_portable: i32,
    n_embd: i32,
    hc: i32,
    hc_lr: i32,
    k: i32,
    n_head: i32,
    head_dim: i32,
    n_head_kv: i32,
    n_expert: i32,
    n_ff: i32,
    verify_max_t: i32,
    max_cells: i64,
    expert_blob: i64,
    q8_1_bytes: u64,
    gr_ws_bytes: u64,
    hit_grouped_scratch: u64,
    shared_scratch: u64,
    argmax_scratch: u64,
    top_scratch: u64,
    rt_dir: *const c_char,
}

#[repr(C)]
struct Tensor {
    name: *const c_char,
    kind: *const c_char,
    rows: i64,
    cols: i64,
    off: u64,
    bytes: u64,
}

#[repr(C)]
struct MtpState {
    mapped_h: [*mut c_void; MTP_MAPPED],
    mapped_d: [*mut c_void; MTP_MAPPED],
    state_arena: *mut c_void,
    arena: *mut c_void,
    arena_ptrs: [*mut c_void; MTP_ARENA],
    window: i64,
    cap: i64,
    attn_scratch_floats: i64,
    vram: u64,
    cs: *mut c_void,
    side: *mut c_void,
}

#[repr(C)]
struct BindFacts {
    head_loaded: i32,
    head_type: i32,
    head_device: i32,
    device: i32,
    q4_head: i32,
    coupled_env: i32,
    have_cparams: i32,
    have_dhead: i32,
    q6_packed: i32,
    max_t: i32,
    n_vocab: i64,
    head_row_bytes: u64,
    head_weights: *const c_void,
    head_logits: *mut f32,
    vocab_file: *const c_char,
}

struct R2vFfi {
    install: unsafe extern "C" fn(*const R2vOps),
    ver_facts: unsafe extern "C" fn(*mut c_void, *mut VerFacts) -> c_int,
    ver_adopt: unsafe extern "C" fn(*mut c_void, *const VerState),
    bias_view: unsafe extern "C" fn(
        *mut c_void,
        *mut *const f32,
        *mut i64,
        *mut *mut f32,
        *mut i64,
        *mut i32,
    ) -> c_int,
    bias_adopt: unsafe extern "C" fn(*mut c_void, *const f32, i64, *mut f32),
    stream_high_prio: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
    arch_is: unsafe extern "C" fn(*const c_char) -> c_int,
    mtp_facts: unsafe extern "C" fn(*mut c_void, *mut MtpFacts) -> c_int,
    attn_scratch_floats: unsafe extern "C" fn(*mut c_void, i64) -> u64,
    mtp_set_weights:
        unsafe extern "C" fn(*mut c_void, *const Tensor, i64, *mut c_void, *mut c_void, u64),
    mtp_make_q4_dense:
        unsafe extern "C" fn(*mut c_void, *const u8, u64, *mut c_char, usize) -> c_int,
    kv_flags: unsafe extern "C" fn(*mut i32, *mut i32),
    kv_set_flags: unsafe extern "C" fn(i32, i32),
    state_bytes: unsafe extern "C" fn(*mut c_void, i64, i64) -> u64,
    state_init: unsafe extern "C" fn(*mut c_void, i64, *mut c_void, i64, *mut i32) -> u64,
    state_reset: unsafe extern "C" fn(*mut c_void),
    state_zero: unsafe extern "C" fn(*mut c_void),
    mtp_adopt: unsafe extern "C" fn(*mut c_void, *const MtpState) -> u64,
    bind_facts: unsafe extern "C" fn(*mut c_void, *mut BindFacts) -> c_int,
    bind_head_logits: unsafe extern "C" fn(*mut c_void, *mut f32, i64),
    bind_dhead: unsafe extern "C" fn(*mut c_void, *mut u8, *mut i32, *const i32, i64, i32, u64),
    gather_rows: unsafe extern "C" fn(*const c_void, i64, *const i32, i64, *mut c_void),
    make_q4_head: unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> c_int,
    q6_pack: unsafe extern "C" fn(*mut c_void, c_int, c_int, *const c_char) -> c_int,
    setup_coupled: unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> c_int,
}

unsafe impl Send for R2vFfi {}
unsafe impl Sync for R2vFfi {}

static R2V_FFI: OnceLock<Option<R2vFfi>> = OnceLock::new();

fn ffi() -> Option<&'static R2vFfi> {
    R2V_FFI.get().and_then(|f| f.as_ref())
}

// ============================================================================
// Small helpers
// ============================================================================

/// C `atoi`: leading blanks, a sign, the digits; 0 when there are none.
fn c_atoi(s: &str) -> i64 {
    let t = s.trim_start();
    let (neg, rest) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    let v = digits.parse::<i64>().unwrap_or(0);
    if neg {
        -v
    } else {
        v
    }
}

/// `std::getenv(name) != nullptr && std::atoi(v) != 0`
fn env_atoi_on(name: &str) -> bool {
    std::env::var(name)
        .map(|v| c_atoi(&v) != 0)
        .unwrap_or(false)
}

/// verify.cpp `env_on`: set, not empty, not starting with '0'
fn env_on(name: &str) -> bool {
    std::env::var(name)
        .map(|v| !v.is_empty() && !v.starts_with('0'))
        .unwrap_or(false)
}

/// The C `Bump`: every take rounded up to 256 bytes; null base = count only.
struct Bump {
    base: *mut u8,
    used: u64,
}

impl Bump {
    fn take(&mut self, bytes: u64) -> *mut c_void {
        let p = if self.base.is_null() {
            std::ptr::null_mut()
        } else {
            unsafe { self.base.add(self.used as usize) }
        };
        self.used += (bytes + 255) & !255u64;
        p.cast()
    }
}

/// verify.cpp / mtp.cpp `mapped`: pinned, mapped, zeroed host memory and its device alias (coherent on AMD, as the
/// HIP shim makes every mapped allocation; portable with a peer).
fn mapped(bytes: usize, portable: bool) -> Option<(*mut c_void, *mut c_void)> {
    let mut h: *mut c_void = std::ptr::null_mut();
    let mut d: *mut c_void = std::ptr::null_mut();
    let flags = hipHostMallocMapped
        | hipHostMallocCoherent
        | if portable { hipHostMallocPortable } else { 0 };
    unsafe {
        if hipHostMalloc(&mut h, bytes, flags) != hipSuccess {
            return None;
        }
        std::ptr::write_bytes(h.cast::<u8>(), 0, bytes);
        if hipHostGetDevicePointer(&mut d, h, 0) != hipSuccess {
            return None;
        }
    }
    Some((h, d))
}

fn dmalloc(bytes: u64) -> Result<*mut c_void, hip_sys::hipError_t> {
    let mut p: *mut c_void = std::ptr::null_mut();
    let e = unsafe { hipMalloc(&mut p, bytes as usize) };
    if e == hipSuccess {
        Ok(p)
    } else {
        Err(e)
    }
}

fn stream_nb() -> Option<hipStream_t> {
    let mut s: hipStream_t = std::ptr::null_mut();
    (unsafe { hipStreamCreateWithFlags(&mut s, hipStreamNonBlocking) } == hipSuccess).then_some(s)
}

fn event_nt() -> Option<hipEvent_t> {
    let mut e: hipEvent_t = std::ptr::null_mut();
    (unsafe { hipEventCreateWithFlags(&mut e, hipEventDisableTiming) } == hipSuccess).then_some(e)
}

fn arch_is(f: &R2vFfi, prefix: &str) -> bool {
    let c = CString::new(prefix).unwrap_or_default();
    unsafe { (f.arch_is)(c.as_ptr()) != 0 }
}

/// platform `advise_willneed(fd, 0, bytes)`: POSIX_FADV_WILLNEED in 128 KiB steps (STRATA_READ_AHEAD=0: off)
fn advise_willneed(file: &std::fs::File, bytes: u64) {
    extern "C" {
        fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| {
        std::env::var("STRATA_READ_AHEAD")
            .map(|v| c_atoi(&v) != 0)
            .unwrap_or(true)
    }) {
        return;
    }
    const STEP: u64 = 128 << 10;
    let fd = file.as_raw_fd();
    let mut at = 0u64;
    while at < bytes {
        unsafe { posix_fadvise(fd, at as i64, STEP.min(bytes - at) as i64, 3) };
        at += STEP;
    }
}

/// mtp.cpp `read_file`: the whole file (with the read-ahead hint), None when it cannot be read
fn read_file(path: &str) -> Option<Vec<u8>> {
    let mut f = std::fs::File::open(path).ok()?;
    let n = f.metadata().ok()?.len();
    advise_willneed(&f, n);
    let mut out = Vec::with_capacity(n as usize);
    f.read_to_end(&mut out).ok()?;
    (out.len() as u64 == n).then_some(out)
}

const MIB: f64 = 1048576.0;

// ============================================================================
// Verifier::init
// ============================================================================

/// Verifier::init's device arena, buffer by buffer in its carve order (bytes before the 256-byte rounding).
fn ver_arena_sizes(f: &VerFacts, n_vocab: i64, plan_i32: i64) -> [u64; VER_ARENA] {
    let t = f.max_t as u64;
    let n = f.n_embd as u64;
    let hc = f.hc as u64;
    let k = f.k as u64;
    let c = f.conv_ch as u64;
    let zv = f.value_dim as u64;
    let hv = f.v_heads as u64;
    let nh = f.n_head as u64;
    let hd = f.head_dim as u64;
    let nkv = f.n_head_kv as u64;
    let iq = f.idx_q_heads as u64;
    let id = f.idx_key_dim as u64;
    let ng = f.n_gdn as u64;
    let nq = f.n_qsa as u64;
    let hs = f.ng_hist as u64 * f.ng_hc_dim as u64;
    let ts = (f.idx_block as u64 - 1) * id;
    let sc = f.step_count as u64;
    let max_blocks = (f.max_cells / f.idx_block + 2) as u64;
    let cap = f.cap as u64;
    let asf = f.attn_scratch_floats as u64;
    let nff = f.n_ff as u64;
    let (i4, f4) = (4u64, 4u64);
    [
        t * i4,                                             // tok_
        t * sc * i4,                                        // step_
        t * (nh + nkv + iq) * i4,                           // pos_
        (2 + t) * i4,                                       // commit_
        t * n * f4,                                         // ple_
        t * n * f4,                                         // emb_
        t * hc * n * f4,                                    // R_
        t * n * f4,                                         // mixed_
        t * n * f4,                                         // bo_
        t * hc * f4,                                        // inj_
        t * hc * f4,                                        // inj2_
        t * f.hc_lr as u64 * f4,                            // lo_
        t * hc * f4,                                        // rs_
        t * hc * n * f4,                                    // xn_
        f.q8_1_bytes,                                       // xq_
        f.q8_1_bytes,                                       // sh_xq_
        f.q8_1_il_bytes,                                    // xil_
        ng * t * c * f4,                                    // qkv_L_
        ng * t * c * f4,                                    // h_L_
        ng * t * hv * f4,                                   // gate_L_
        ng * t * hv * f4,                                   // beta_L_
        t * zv * f4,                                        // z_
        t * zv * f4,                                        // y_
        t * zv * f4,                                        // y_dummy_
        t * nh * 2 * hd * f4,                               // qfull_
        t * nh * hd * f4,                                   // qcur_
        t * nkv * hd * f4,                                  // kcur_
        t * nkv * hd * f4,                                  // vcur_
        nq * t * id * f4,                                   // idx_raw_L_
        t * iq * id * f4,                                   // qidx_
        t * max_blocks * f4,                                // scores_
        t * cap * i4,                                       // sel_
        t * nh * hd * f4,                                   // attn_
        t * nh * hd * f4,                                   // attn32_
        t * asf * f4,                                       // attn_scratch_
        nq * ts * f4,                                       // tail_snap_
        t * f.n_expert as u64 * f4,                         // logits_
        t * k * f4,                                         // w_
        t * k * i4,                                         // ids_
        t * n * f4,                                         // shared_
        t * k * n * f4,                                     // parts_
        t * k * n * f4,                                     // hit_out_
        t * k * i4,                                         // hit_slot_
        t * k * i4,                                         // hit_dst_
        4 * i4,                                             // hit_count_
        2 * (plan_i32 as u64 + 16) * i4,                    // plan_
        16 * f.max_blob,                                    // staging_ (kStagingBlobs)
        t * (n / 32) * 34,                                  // hit_xq_
        t * (n / 32) * f4,                                  // hit_xs_
        t * (n / 32) * 36,                                  // nat_xq_
        f.hit_grouped_scratch.max(f.native_expert_scratch), // hit_scratch_
        t * n * f4,                                         // head_mixed_
        hc * f4,                                            // head_inj_
        t * n * 2,                                          // sh_bf16_
        t * nff * f4,                                       // sh_gate_
        t * nff * f4,                                       // sh_up_
        (t + 4) * f4,                                       // sh_g_
        t * n_vocab as u64 * f4,                            // head_logits_
        f.argmax_scratch,                                   // arg_scratch_
        4 * i4,                                             // one_
        t * hs * f4,                                        // hist_snap_
        t * f.ng_hc_dim as u64 * f4,                        // ple_key_
        t * n * f4,                                         // ple_val_
    ]
}

/// One verifier's setup.  Ok(false): declined before anything was made (the C++ body runs).
unsafe fn ver_init(f: &R2vFfi, ver: *mut c_void) -> Result<bool, String> {
    let mut fa: VerFacts = std::mem::zeroed();
    if (f.ver_facts)(ver, &mut fa) != 0 || fa.eligible == 0 || fa.hip_gfx906 != 0 {
        return Ok(false);
    }
    // the opt-in diagnostics keep the C++ body (the Rust run plane declines them too)
    if env_on("STRATA_VERIFY_TRACE") || std::env::var_os("STRATA_VERIFY_PROFILE").is_some() {
        return Ok(false);
    }
    let max_t = fa.max_t;
    if max_t < 2 || max_t > fa.verify_max_t || max_t > fa.cpu_maxt {
        return Err(format!(
            "verify: the window must hold 2..{} tokens",
            fa.verify_max_t
        ));
    }
    if fa.d_res.is_null() || fa.cache_base.is_null() || fa.blob <= 0 {
        return Err("verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache); with \
                    --expert-cache auto, no VRAM was left for it - the 'no VRAM is left for the expert cache' line above \
                    says how much is short and what makes room (a smaller --max-context, --kv q4_0, setup --draft-vocab en, \
                    images on the CPU)"
            .into());
    }
    if fa.layer_ok == 0 {
        let why = CStr::from_ptr(fa.why.as_ptr()).to_string_lossy();
        return Err(format!(
            "verify: {why} (the verify window reproduces the default native decode path)"
        ));
    }
    if fa.fused_gr_ok == 0 || fa.k != 10 || fa.ssm_state_size != 128 || fa.ssm_d_conv != 4 {
        return Err("verify: geometry differs from the artifact's".into());
    }
    if fa.n_vocab < 0 {
        return Err("verify: output.weight is missing".into());
    }
    let mut st: VerState = std::mem::zeroed();
    st.le = fa.n_layers as i64;
    st.n_vocab = fa.n_vocab;
    st.cap = fa.cap;
    st.max_blocks = fa.max_cells / fa.idx_block + 2;
    st.attn_scratch_floats = fa.attn_scratch_floats;

    let t = max_t as usize;
    let (n, k) = (fa.n_embd as usize, fa.k as usize);
    let (nh, nkv, iq) = (
        fa.n_head as usize,
        fa.n_head_kv as usize,
        fa.idx_q_heads as usize,
    );
    // ---- mapped staging, in init's order (the first failure stops it)
    let sizes = [
        t * 4,
        t * fa.step_count as usize * 4,
        t * (nh + nkv + iq) * 4,
        (2 + t) * 4 + 16,
        t * n * 4,
        t * 4 + 16,
        fa.verify_max_t as usize * fa.nonfinite_blocks as usize * 4,
        t * n * 4,
        t * k * 4,
        t * k * 4,
        64,
        64,
        64,
        64,
        64,
        t * k * n * 4,
    ];
    let portable = fa.peer_portable != 0;
    for (i, &b) in sizes.iter().enumerate() {
        let Some((h, d)) = mapped(b, portable) else {
            (f.ver_adopt)(ver, &st);
            return Err("verify: mapped staging allocation failed".into());
        };
        st.mapped_h[i] = h;
        st.mapped_d[i] = d;
    }
    // the GPU plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
    let cap = (t * k) as i64;
    let i32w = 4 + (cap + 1) + cap + cap;
    let ptr_off = (i32w + 1) & !1;
    let plan_i32 = ptr_off + 4 * cap + (cap + 1) + 1;
    let Some((hp, dp)) = mapped((plan_i32 * 4 * 2 + 64) as usize, portable) else {
        (f.ver_adopt)(ver, &st);
        return Err("verify: mapped plan allocation failed".into());
    };
    st.mapped_h[16] = hp;
    st.mapped_d[16] = dp;
    st.plan_i32 = plan_i32;
    let h_plan = hp.cast::<i32>();
    st.sink_counts = h_plan;
    st.sink_start = h_plan.add(4);
    st.sink_dst = st.sink_start.add(cap as usize + 1);
    st.sink_tok = st.sink_dst.add(cap as usize);
    st.sink_ptr = h_plan.add(ptr_off as usize).cast::<u64>();
    st.sink_ptr2 = st.sink_ptr.add(cap as usize);
    st.sink_start2 = h_plan.add((ptr_off + 4 * cap) as usize);
    st.sink_cap = cap;
    st.staging_cap = 16;

    // ---- the device arena: the same sequence counted, then carved
    let sz = ver_arena_sizes(&fa, fa.n_vocab, plan_i32);
    let mut count = Bump {
        base: std::ptr::null_mut(),
        used: 0,
    };
    for &b in &sz {
        count.take(b);
    }
    let arena = match dmalloc(count.used) {
        Ok(p) => p,
        Err(_) => {
            (f.ver_adopt)(ver, &st);
            return Err(format!(
                "verify: the device arena ({} MiB) does not fit",
                count.used >> 20
            ));
        }
    };
    hipMemset(arena, 0, count.used as usize);
    st.arena = arena;
    st.arena_bytes = count.used;
    let mut real = Bump {
        base: arena.cast(),
        used: 0,
    };
    for (i, &b) in sz.iter().enumerate() {
        st.arena_ptrs[i] = real.take(b);
    }
    let one: i32 = 1;
    if hipMemcpy(
        st.arena_ptrs[59],
        (&one as *const i32).cast(),
        4,
        hipMemcpyHostToDevice,
    ) != hipSuccess
    {
        (f.ver_adopt)(ver, &st);
        return Err("verify: the arena could not be set".into());
    }
    // ---- streams and events
    // STRATA_COPY_HIP_PRIO=1 (opt-in): the copy stream at the highest priority, a hardware queue of its own
    let mut copy: *mut c_void = std::ptr::null_mut();
    let copy_made = env_atoi_on("STRATA_COPY_HIP_PRIO") && (f.stream_high_prio)(&mut copy) != 0;
    hipGetLastError();
    if !copy_made {
        match stream_nb() {
            Some(s) => copy = s.cast(),
            None => {
                (f.ver_adopt)(ver, &st);
                return Err("verify: copy stream create failed".into());
            }
        }
    }
    st.copy = copy;
    match (stream_nb(), stream_nb()) {
        (Some(a), Some(b)) => {
            st.cs = a.cast();
            st.sh_cs = b.cast();
        }
        (a, _) => {
            st.cs = a.map_or(std::ptr::null_mut(), |s| s.cast());
            (f.ver_adopt)(ver, &st);
            return Err("verify: stream create failed".into());
        }
    }
    match (event_nt(), event_nt(), event_nt()) {
        (Some(a), Some(b), Some(c)) => {
            st.commit_done = a.cast();
            st.ev_fork = b.cast();
            st.ev_join = c.cast();
        }
        _ => {
            (f.ver_adopt)(ver, &st);
            return Err("verify: event create failed".into());
        }
    }
    // STRATA_DF_BRANCH=1 (opt-in): the mixer's side branches on two more streams
    if env_atoi_on("STRATA_DF_BRANCH") {
        let mut okb = match event_nt() {
            Some(e) => {
                st.df_fork = e.cast();
                true
            }
            None => false,
        };
        for i in 0..2 {
            if !okb {
                break;
            }
            okb = match stream_nb() {
                Some(s) => {
                    st.df_side[i] = s.cast();
                    match event_nt() {
                        Some(e) => {
                            st.df_join[i] = e.cast();
                            true
                        }
                        None => false,
                    }
                }
                None => false,
            };
        }
        if okb {
            st.df_branch = 1;
        } else {
            hipGetLastError();
        }
    }
    // ---- 100% of the experts resident: the zero-doorbell graph (STRATA_VERIFY_ALL_RESIDENT=0: off)
    let mut all_res = false;
    if !fa.h_res.is_null() && !fa.d_res.is_null() && !fa.cache_base.is_null() {
        let on = std::env::var("STRATA_VERIFY_ALL_RESIDENT")
            .map(|v| c_atoi(&v) != 0)
            .unwrap_or(true);
        if on {
            let n_res = fa.n_layers as usize * fa.n_expert as usize;
            all_res = std::slice::from_raw_parts(fa.h_res, n_res)
                .iter()
                .all(|&s| s >= 0);
        }
    }
    // E-6: device-planned resident layers (STRATA_VERIFY_DEVICE_PLAN=1, opt-in)
    let mut device_plan =
        !all_res && fa.always_publish == 0 && env_atoi_on("STRATA_VERIFY_DEVICE_PLAN");
    // S26: the HC read's q8_1 group counters (STRATA_QFUSE=1; STRATA_QFUSE_HC, gfx12 default)
    let qfuse = env_atoi_on("STRATA_QFUSE");
    let qfuse_hc = match std::env::var("STRATA_QFUSE_HC") {
        Ok(v) => c_atoi(&v) != 0,
        Err(_) => fa.is_gfx12 != 0,
    };
    if qfuse || qfuse_hc {
        let bytes = 4 * (fa.n_embd / 32) as u64;
        match dmalloc(bytes) {
            Ok(p) if hipMemset(p, 0, bytes as usize) == hipSuccess => st.qcnt = p,
            Ok(p) => {
                hipGetLastError();
                hipFree(p);
            }
            Err(_) => {
                hipGetLastError();
            }
        }
    }
    if all_res || device_plan {
        let mut ok2 = true;
        if device_plan {
            ok2 = match dmalloc(64) {
                Ok(p) => {
                    st.skip = p;
                    hipMemset(p, 0, 64) == hipSuccess
                }
                Err(_) => false,
            };
        }
        if ok2 && !fa.slot_off.is_null() && fa.n_slots > 0 {
            let b = fa.n_slots as usize * 8;
            ok2 = match dmalloc(b as u64) {
                Ok(p) => {
                    st.slot_off_d = p;
                    hipMemcpy(p, fa.slot_off.cast(), b, hipMemcpyHostToDevice) == hipSuccess
                }
                Err(_) => false,
            };
        }
        if !ok2 {
            hipGetLastError();
            all_res = false;
            device_plan = false;
        }
    }
    st.all_resident = all_res as i32;
    st.device_plan = device_plan as i32;
    (f.ver_adopt)(ver, &st);
    eprintln!(
        "strata verify: window up to {} tokens, {:.1} MiB of device buffers{}",
        max_t,
        count.used as f64 / MIB,
        if all_res {
            " (100% VRAM resident: zero-doorbell graph)"
        } else {
            ""
        }
    );
    Ok(true)
}

/// Verifier::set_logit_bias.  Ok(false): declined (a split's earlier stage, another device).
unsafe fn ver_bias(f: &R2vFfi, ver: *mut c_void, bias: &[f32]) -> Result<bool, String> {
    let (mut host, mut n_host, mut dev, mut n_vocab, mut eligible) = (
        std::ptr::null::<f32>(),
        0i64,
        std::ptr::null_mut::<f32>(),
        0i64,
        0i32,
    );
    if (f.bias_view)(
        ver,
        &mut host,
        &mut n_host,
        &mut dev,
        &mut n_vocab,
        &mut eligible,
    ) != 0
        || eligible == 0
    {
        return Ok(false);
    }
    if !bias.is_empty() && bias.len() as i64 != n_vocab {
        return Err("logit_bias vocabulary size mismatch".into());
    }
    let held: &[f32] = if host.is_null() || n_host <= 0 {
        &[]
    } else {
        std::slice::from_raw_parts(host, n_host as usize)
    };
    if bias != held && !bias.is_empty() {
        let bytes = std::mem::size_of_val(bias);
        if dev.is_null() {
            match dmalloc(bytes as u64) {
                Ok(p) => dev = p.cast(),
                Err(_) => return Err("cannot upload logit_bias".into()),
            }
        }
        if hipMemcpy(
            dev.cast(),
            bias.as_ptr().cast(),
            bytes,
            hipMemcpyHostToDevice,
        ) != hipSuccess
        {
            (f.bias_adopt)(ver, held.as_ptr(), held.len() as i64, dev);
            return Err("cannot upload logit_bias".into());
        }
    }
    (f.bias_adopt)(ver, bias.as_ptr(), bias.len() as i64, dev);
    Ok(true)
}

// ============================================================================
// MtpDrafter::load
// ============================================================================

/// dense.txt: `name kind rows cols off bytes` per line (operator>> semantics: blanks between, extra words ignored)
fn parse_dense_index(text: &str) -> Result<Vec<(String, String, i64, i64, u64, u64)>, String> {
    let mut out = Vec::new();
    for line in text.split('\n') {
        if line.is_empty() {
            continue;
        }
        let w: Vec<&str> = line.split_ascii_whitespace().collect();
        let bad = || format!("mtp: malformed dense.txt line: {line}");
        if w.len() < 6 {
            return Err(bad());
        }
        let rows = w[2].parse::<i64>().map_err(|_| bad())?;
        let cols = w[3].parse::<i64>().map_err(|_| bad())?;
        let off = w[4].parse::<u64>().map_err(|_| bad())?;
        let bytes = w[5].parse::<u64>().map_err(|_| bad())?;
        out.push((w[0].to_string(), w[1].to_string(), rows, cols, off, bytes));
    }
    Ok(out)
}

fn mtp_arena_sizes(fa: &MtpFacts, cap: i64, asf: u64) -> [u64; MTP_ARENA] {
    let t = fa.max_t as u64;
    let n = fa.n_embd as u64;
    let hc = fa.hc as u64;
    let k = fa.k as u64;
    let nh = fa.n_head as u64;
    let hd = fa.head_dim as u64;
    let nkv = fa.n_head_kv as u64;
    let r2 = 2 * t;
    [
        t * 4,                      // tok_
        r2 * 4 * 4,                 // step_
        r2 * nh * 4,                // pos_
        4 * 4,                      // row_
        t * cap as u64 * 4,         // ident_
        t * hc * n * 4,             // Rin_
        t * hc * n * 4,             // R_
        t * n * 4,                  // emb_
        t * n * 4,                  // en_
        t * n * 4,                  // e2_
        t * hc * n * 4,             // hn_
        t * hc * n * 4,             // h2_
        t * n * 4,                  // mixed_
        t * hc * 4,                 // inj_
        t * hc * 4,                 // inj2_
        t * fa.hc_lr as u64 * 4,    // lo_
        t * hc * 4,                 // rs_
        t * n * 4,                  // bo_
        t * hc * n * 4,             // xn_
        fa.q8_1_bytes,              // xq_
        t * nh * 2 * hd * 4,        // qfull_
        t * nh * hd * 4,            // qcur_
        t * nkv * hd * 4,           // kcur_
        t * nkv * hd * 4,           // vcur_
        t * nh * hd * 4,            // attn_
        t * nh * hd * 4,            // attn32_
        asf * 4,                    // attn_scratch_ (one row at a time)
        t * fa.n_expert as u64 * 4, // logits_
        t * k * 4,                  // w_
        t * k * 4,                  // ids_
        t * n * 4,                  // shared_
        t * k * n * 4,              // parts_
        t * n * 4,                  // y_
        t * n * 4,                  // sample_
        fa.gr_ws_bytes,             // the GR workspace
        t * k * 4,                  // hit_slot_
        t * k * 4,                  // hit_dst_
        4 * 4,                      // hit_count_
        t * k * 8,                  // grp_ptr_
        (t * k + 1) * 4,            // grp_start_
        4 * 4,                      // grp_counts_
        t * (n / 32) * 34,          // hit_xq_
        t * (n / 32) * 4,           // hit_xs_
        fa.hit_grouped_scratch,     // hit_scratch_
        fa.shared_scratch,          // sh_scratch_
        n * 2,                      // x_bf16_
        (t + 4) * 4,                // out_ids_
        (t + 4) * 4,                // probs_
        fa.argmax_scratch,          // arg_scratch_
        fa.top_scratch,             // top_scratch_
        hc * 4,                     // dummy_inj_
    ]
}

unsafe fn mtp_load(f: &R2vFfi, mtp: *mut c_void, window: i64) -> Result<bool, String> {
    let mut fa: MtpFacts = std::mem::zeroed();
    if (f.mtp_facts)(mtp, &mut fa) != 0 || fa.eligible == 0 || fa.hip == 0 || fa.hip_gfx906 != 0 {
        return Ok(false);
    }
    let max_t = fa.max_t;
    if max_t < 1 || max_t > fa.verify_max_t {
        return Err("mtp: max_t out of range".into());
    }
    let rt_dir = CStr::from_ptr(fa.rt_dir).to_string_lossy().into_owned();
    let t_files = Instant::now();
    // ---- the index and the dense weights
    let idx = std::fs::read_to_string(format!("{rt_dir}/dense.txt"))
        .map_err(|_| format!("mtp: cannot open {rt_dir}/dense.txt (run tools/mtp_rt.py)"))?;
    let tensors = parse_dense_index(&idx)?;
    let blob = read_file(&format!("{rt_dir}/dense.bin")).ok_or("mtp: cannot read dense.bin")?;
    let dense = match dmalloc(blob.len() as u64) {
        Ok(p) => p,
        Err(e) => {
            let (mut fb, mut tb) = (0usize, 0usize);
            let info = hipMemGetInfo(&mut fb, &mut tb);
            return Err(format!(
                "mtp: dense weights allocation failed ({}), requested {} MiB, CUDA0 free {}",
                hip_err_str(e),
                blob.len() >> 20,
                if info == hipSuccess {
                    format!("{} MiB", fb >> 20)
                } else {
                    "unknown".into()
                }
            ));
        }
    };
    hipMemcpy(
        dense,
        blob.as_ptr().cast(),
        blob.len(),
        hipMemcpyHostToDevice,
    );
    let names: Vec<(CString, CString)> = tensors
        .iter()
        .map(|t| {
            (
                CString::new(t.0.as_str()).unwrap_or_default(),
                CString::new(t.1.as_str()).unwrap_or_default(),
            )
        })
        .collect();
    let table: Vec<Tensor> = tensors
        .iter()
        .zip(&names)
        .map(|(t, (nm, kd))| Tensor {
            name: nm.as_ptr(),
            kind: kd.as_ptr(),
            rows: t.2,
            cols: t.3,
            off: t.4,
            bytes: t.5,
        })
        .collect();
    (f.mtp_set_weights)(
        mtp,
        table.as_ptr(),
        table.len() as i64,
        dense,
        std::ptr::null_mut(),
        blob.len() as u64,
    );
    if fa.q4 != 0 {
        let mut eb = vec![0 as c_char; 512];
        if (f.mtp_make_q4_dense)(
            mtp,
            blob.as_ptr(),
            blob.len() as u64,
            eb.as_mut_ptr(),
            eb.len(),
        ) == 0
        {
            return Err(CStr::from_ptr(eb.as_ptr()).to_string_lossy().into_owned());
        }
    }
    drop(blob);
    // ---- the routed experts, one blob each, read in 64 MiB pieces
    let ex_bytes = fa.n_expert as u64 * fa.expert_blob as u64;
    {
        let mut file = std::fs::File::open(format!("{rt_dir}/experts.bin"))
            .map_err(|_| "mtp: cannot open experts.bin".to_string())?;
        let experts =
            dmalloc(ex_bytes).map_err(|_| "mtp: the 512 experts do not fit in VRAM".to_string())?;
        (f.mtp_set_weights)(
            mtp,
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            experts,
            ex_bytes,
        );
        advise_willneed(&file, ex_bytes);
        let mut chunk = vec![0u8; 64 << 20];
        let mut off = 0u64;
        while off < ex_bytes {
            let n = (chunk.len() as u64).min(ex_bytes - off) as usize;
            file.read_exact(&mut chunk[..n])
                .map_err(|_| "mtp: experts.bin is truncated".to_string())?;
            hipMemcpy(
                experts.cast::<u8>().add(off as usize).cast(),
                chunk.as_ptr().cast(),
                n,
                hipMemcpyHostToDevice,
            );
            off += n as u64;
        }
    }
    const REQUIRED: [&str; 9] = [
        "fc_embedding.weight",
        "fc_hidden.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
    ];
    for r in REQUIRED {
        if !tensors.iter().any(|t| t.0 == r && t.1 == "q8_0") {
            return Err(format!("mtp: {r} is missing (q8_0)"));
        }
    }
    // ---- the layer's own K/V: a ring of the window over a host copy (KV streaming), whole in VRAM when the copy
    // cannot be pinned; K8V4 never applies to the drafter (plain INT8 under --kv k8v4)
    let max_cells = fa.max_cells;
    let mut ring = if window > 0 && window < max_cells {
        window + 4 * max_t as i64 + 64
    } else {
        0
    };
    let (mut hyb_was, mut int8_was) = (0i32, 0i32);
    (f.kv_flags)(&mut hyb_was, &mut int8_was);
    (f.kv_set_flags)(0, if hyb_was != 0 { 1 } else { int8_was });
    let mut sb = (f.state_bytes)(mtp, max_cells, ring);
    let mut state_arena = dmalloc(sb).map_err(|_| "mtp: the K/V state does not fit".to_string())?;
    let mut kv_mode = 0i32;
    if (f.state_init)(mtp, max_cells, state_arena, ring, &mut kv_mode) == 0 {
        if kv_mode == 0 {
            return Err("mtp: state init failed".into());
        }
        eprintln!(
            "strata mtp: no pinned RAM left for the draft layer's K/V copy; keeping it in VRAM"
        );
        hipGetLastError();
        hipFree(state_arena);
        (f.state_reset)(mtp);
        ring = -1;
        sb = (f.state_bytes)(mtp, max_cells, ring);
        state_arena = dmalloc(sb).map_err(|_| "mtp: the K/V state does not fit".to_string())?;
        if (f.state_init)(mtp, max_cells, state_arena, ring, &mut kv_mode) == 0 {
            return Err("mtp: state init failed".into());
        }
    }
    // (the C++ order: INT8 first, then hybrid)
    (f.kv_set_flags)(hyb_was, int8_was);
    let mut st: MtpState = std::mem::zeroed();
    st.state_arena = state_arena;
    // the zeroing reads the object's st_ (written by state_init) and the arena it points into: adopt that first
    (f.state_zero)(mtp);
    hipDeviceSynchronize();
    let mut vram_add = sb;

    // ---- buffers
    let win = if window > 0 && window < max_cells {
        window
    } else {
        0
    };
    let cap = ((if win > 0 { win } else { max_cells }) + 63) / 64 * 64;
    let asf = (f.attn_scratch_floats)(mtp, cap);
    st.window = win;
    st.cap = cap;
    st.attn_scratch_floats = asf as i64;
    let t = max_t as usize;
    let nh = fa.n_head as usize;
    let r2 = 2 * t;
    let portable = fa.peer_portable != 0;
    let msz = [
        t * 4 + 64,
        r2 * 4 * 4 + 64,
        r2 * nh * 4 + 64,
        64,
        t * 4 + 64,
        t * 4 + 64,
        64,
    ];
    let n_mapped = if fa.force_on != 0 {
        MTP_MAPPED
    } else {
        MTP_MAPPED - 1
    };
    for i in 0..n_mapped {
        let Some((h, d)) = mapped(msz[i], portable) else {
            (f.mtp_adopt)(mtp, &st);
            return Err("mtp: mapped staging failed".into());
        };
        st.mapped_h[i] = h;
        st.mapped_d[i] = d;
    }
    if !st.mapped_h[6].is_null() {
        std::slice::from_raw_parts_mut(st.mapped_h[6].cast::<i32>(), 16).fill(-1);
    }
    let sz = mtp_arena_sizes(&fa, cap, asf);
    let mut count = Bump {
        base: std::ptr::null_mut(),
        used: 0,
    };
    for &b in &sz {
        count.take(b);
    }
    let arena = match dmalloc(count.used) {
        Ok(p) => p,
        Err(_) => {
            (f.mtp_adopt)(mtp, &st);
            return Err("mtp: buffers do not fit".into());
        }
    };
    hipMemset(arena, 0, count.used as usize);
    st.arena = arena;
    let mut real = Bump {
        base: arena.cast(),
        used: 0,
    };
    for (i, &b) in sz.iter().enumerate() {
        st.arena_ptrs[i] = real.take(b);
    }
    vram_add += count.used;
    {
        let capu = cap as usize;
        let mut id = vec![0i32; t * capu];
        for row in id.chunks_mut(capu) {
            for (i, v) in row.iter_mut().enumerate() {
                *v = i as i32;
            }
        }
        hipMemcpy(
            st.arena_ptrs[4],
            id.as_ptr().cast(),
            id.len() * 4,
            hipMemcpyHostToDevice,
        );
    }
    // the draft stream: highest priority with STRATA_MTP_HIP_PRIO (default on gfx1200/gfx1201), else a plain one
    let hip_prio = match std::env::var("STRATA_MTP_HIP_PRIO") {
        Ok(v) if !v.is_empty() => c_atoi(&v) != 0,
        _ => arch_is(f, "gfx12"),
    };
    let mut cs: *mut c_void = std::ptr::null_mut();
    let prio = hip_prio && (f.stream_high_prio)(&mut cs) != 0;
    hipGetLastError();
    if !prio {
        match stream_nb() {
            Some(s) => cs = s.cast(),
            None => {
                (f.mtp_adopt)(mtp, &st);
                return Err("mtp: stream".into());
            }
        }
    }
    st.cs = cs;
    // the shared expert's branch stream: gfx103x only (STRATA_MTP_SIDE_STREAM=1 / =0 forces either)
    let side_on = match std::env::var("STRATA_MTP_SIDE_STREAM") {
        Ok(v) if !v.is_empty() => c_atoi(&v) != 0,
        _ => arch_is(f, "gfx103"),
    };
    if side_on {
        match stream_nb() {
            Some(s) => st.side = s.cast(),
            None => {
                (f.mtp_adopt)(mtp, &st);
                return Err("mtp: streams".into());
            }
        }
    }
    st.vram = vram_add;
    let vram = (f.mtp_adopt)(mtp, &st);
    let files_s = t_files.elapsed().as_secs_f64();
    let ex_mib = ex_bytes as f64 / MIB;
    let dense_off = tensors.last().map_or(0, |t| t.4) as f64;
    eprintln!(
        "strata mtp: draft layer loaded, {:.0} MiB of VRAM (experts {:.0}, dense {:.0}), files read in {:.2} s ({:.0} MiB/s)",
        vram as f64 / MIB,
        ex_mib,
        dense_off / MIB,
        files_s,
        if files_s > 0.0 { (ex_bytes as f64 + dense_off) / MIB / files_s } else { 0.0 }
    );
    Ok(true)
}

// ============================================================================
// MtpDrafter::bind
// ============================================================================

/// mtp.cpp `draft_head_hint` (#474): what to change when the draft head's subset does not fit
fn draft_head_hint(n_tokens: i64, row_bytes: i64) {
    let (mut fb, mut tb) = (0usize, 0usize);
    let have_free = unsafe { hipMemGetInfo(&mut fb, &mut tb) } == hipSuccess;
    let mib = |n: i64| (n * row_bytes) as f64 / MIB;
    let free_s = if have_free {
        format!(" and {:.0} MiB is free", fb as f64 / MIB)
    } else {
        String::new()
    };
    eprintln!(
        "strata mtp: the draft head over {} tokens needs {:.0} MiB of VRAM{}.",
        n_tokens,
        mib(n_tokens),
        free_s
    );
    let smaller = [
        (
            "cyrillic",
            58963i64,
            "English, code and the Cyrillic script",
        ),
        ("en", 40525, "English and code"),
    ];
    let mut opts = String::new();
    for (name, tokens, what) in smaller {
        if tokens < n_tokens {
            opts += &format!(
                "{}--draft-vocab {} ({}, ~{:.0} MiB)",
                if opts.is_empty() { "" } else { " or " },
                name,
                what,
                mib(tokens)
            );
        }
    }
    if !opts.is_empty() {
        eprintln!(
            "strata mtp: hint: a smaller draft vocabulary needs less VRAM: {opts}. Start once with it - START-HERE.bat \
             --draft-vocab en (Windows) or ./setup.sh --draft-vocab en - and the model keeps it (\"draft_vocab\" in its \
             strata-*.json config); or a smaller --context in setup."
        );
    } else {
        eprintln!(
            "strata mtp: hint: this is already the smallest shipped draft vocabulary: a smaller --context (or closing \
             what else uses the GPU) leaves it room."
        );
    }
}

unsafe fn mtp_bind(f: &R2vFfi, mtp: *mut c_void) -> Result<bool, String> {
    let mut b: BindFacts = std::mem::zeroed();
    if (f.bind_facts)(mtp, &mut b) != 0 {
        return Ok(false);
    }
    // the head on another card (a layer split's first stage): the C++ body gathers the rows there
    if b.head_loaded != 0 && b.head_device >= 0 && b.head_device != b.device {
        return Ok(false);
    }
    if b.n_vocab < 0 {
        return Err("mtp: output.weight is missing".into());
    }
    (f.bind_head_logits)(mtp, std::ptr::null_mut(), b.n_vocab);
    if b.head_loaded == 0 {
        return Err("mtp: the draft layer needs the native head (--native)".into());
    }
    if b.head_logits.is_null() {
        let bytes = b.max_t as u64 * b.n_vocab as u64 * 4;
        let p = dmalloc(bytes).map_err(|_| "mtp: the draft logits do not fit".to_string())?;
        (f.bind_head_logits)(mtp, p.cast(), b.n_vocab);
    }
    // the draft head's token subset, when tools/draft_vocab.py wrote one (STRATA_MTP_FULL_HEAD=1: the whole head)
    let full_head = std::env::var("STRATA_MTP_FULL_HEAD")
        .map(|v| v.starts_with('1'))
        .unwrap_or(false);
    if b.have_dhead == 0 && !full_head {
        let vf = CStr::from_ptr(b.vocab_file).to_string_lossy().into_owned();
        if let Some(raw) = read_file(&vf).filter(|r| r.len() >= 4 && r.len() % 4 == 0) {
            let n = (raw.len() / 4) as i64;
            let row_bytes = b.head_row_bytes as i64;
            let dvocab = dmalloc(raw.len() as u64);
            let dhead = dvocab
                .as_ref()
                .ok()
                .map(|_| dmalloc((n * row_bytes) as u64));
            let (dvocab, dhead) = match (dvocab, dhead) {
                (Ok(v), Some(Ok(h))) => (v, h),
                (v, h) => {
                    // the C++ body keeps what it got in the object; nothing reads it after this error
                    let _ = (v, h);
                    draft_head_hint(n, row_bytes);
                    return Err("mtp: the draft head does not fit".into());
                }
            };
            hipMemcpy(
                dvocab,
                raw.as_ptr().cast(),
                raw.len(),
                hipMemcpyHostToDevice,
            );
            let ids: Vec<i32> = raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            (f.gather_rows)(b.head_weights, row_bytes, dvocab.cast(), n, dhead);
            hipDeviceSynchronize();
            (f.bind_dhead)(
                mtp,
                dhead.cast(),
                dvocab.cast(),
                ids.as_ptr(),
                n,
                b.head_type,
                (n * row_bytes) as u64 + raw.len() as u64,
            );
            eprintln!(
                "strata mtp: draft head over {} tokens ({:.1} MiB)",
                n,
                (n * row_bytes) as f64 / MIB
            );
            let mut q4 = false;
            if b.q4_head != 0 {
                let mut eb = vec![0 as c_char; 512];
                if (f.make_q4_head)(mtp, eb.as_mut_ptr(), eb.len()) == 0 {
                    return Err(CStr::from_ptr(eb.as_ptr()).to_string_lossy().into_owned());
                }
                q4 = true;
            }
            if !q4 && b.head_type == 14 && b.q6_packed != 0 {
                // STRATA_Q6_PACKED=1
                (f.q6_pack)(
                    dhead,
                    ((row_bytes / 210) * 256) as c_int,
                    n as c_int,
                    c"MTP draft head".as_ptr(),
                );
            }
        }
    }
    if b.coupled_env != 0 && b.have_cparams == 0 {
        let mut eb = vec![0 as c_char; 512];
        if (f.setup_coupled)(mtp, eb.as_mut_ptr(), eb.len()) == 0 {
            return Err(CStr::from_ptr(eb.as_ptr()).to_string_lossy().into_owned());
        }
    }
    Ok(true)
}

// ============================================================================
// The hooks C++ calls, and their installation
// ============================================================================

fn run_hook(
    err: *mut c_char,
    cap: usize,
    what: &str,
    body: impl FnOnce(&R2vFfi) -> Result<bool, String>,
) -> c_int {
    let Some(f) = ffi() else { return 0 };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(f))) {
        Ok(Ok(true)) => 1,
        Ok(Ok(false)) => 0,
        Ok(Err(e)) => {
            put_c_err(&e, err, cap);
            -1
        }
        Err(_) => {
            put_c_err(&format!("{what}: the Rust setup panicked"), err, cap);
            -1
        }
    }
}

unsafe extern "C" fn cb_ver_init(
    _u: *mut c_void,
    ver: *mut c_void,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    run_hook(err, cap, "verify", |f| unsafe { ver_init(f, ver) })
}

unsafe extern "C" fn cb_ver_bias(
    _u: *mut c_void,
    ver: *mut c_void,
    bias: *const f32,
    n: i64,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    let b: &[f32] = if bias.is_null() || n <= 0 {
        &[]
    } else {
        std::slice::from_raw_parts(bias, n as usize)
    };
    run_hook(err, cap, "logit_bias", |f| unsafe { ver_bias(f, ver, b) })
}

unsafe extern "C" fn cb_mtp_load(
    _u: *mut c_void,
    mtp: *mut c_void,
    window: i64,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    run_hook(err, cap, "mtp", |f| unsafe { mtp_load(f, mtp, window) })
}

unsafe extern "C" fn cb_mtp_bind(
    _u: *mut c_void,
    mtp: *mut c_void,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    run_hook(err, cap, "mtp", |f| unsafe { mtp_bind(f, mtp) })
}

/// Hand Verifier::init / set_logit_bias and MtpDrafter::load / bind to Rust (TALUS_FN_RUST=1; TALUS_FN_R2V=0: off).
pub fn install(kernels: &Qwen4ExpKernels) -> Result<(), String> {
    if std::env::var("TALUS_FN_R2V")
        .map(|v| v.starts_with('0'))
        .unwrap_or(false)
        || env_flag("TALUS_P4_CPP_SETUP")
    {
        return Ok(());
    }
    let f = R2V_FFI
        .get_or_init(|| unsafe {
            let h = kernels.raw_handle();
            (|| -> Result<R2vFfi, String> {
                Ok(R2vFfi {
                    install: load_sym_from(h, "tk_qwen4exp_r2v_install")?,
                    ver_facts: load_sym_from(h, "tk_qwen4exp_r2v_ver_facts")?,
                    ver_adopt: load_sym_from(h, "tk_qwen4exp_r2v_ver_adopt")?,
                    bias_view: load_sym_from(h, "tk_qwen4exp_r2v_ver_bias_view")?,
                    bias_adopt: load_sym_from(h, "tk_qwen4exp_r2v_ver_bias_adopt")?,
                    stream_high_prio: load_sym_from(h, "tk_qwen4exp_r2v_stream_high_prio")?,
                    arch_is: load_sym_from(h, "tk_qwen4exp_r2v_arch_is")?,
                    mtp_facts: load_sym_from(h, "tk_qwen4exp_r2v_mtp_facts")?,
                    attn_scratch_floats: load_sym_from(h, "tk_qwen4exp_r2v_attn_scratch_floats")?,
                    mtp_set_weights: load_sym_from(h, "tk_qwen4exp_r2v_mtp_set_weights")?,
                    mtp_make_q4_dense: load_sym_from(h, "tk_qwen4exp_r2v_mtp_make_q4_dense")?,
                    kv_flags: load_sym_from(h, "tk_qwen4exp_r2v_kv_flags")?,
                    kv_set_flags: load_sym_from(h, "tk_qwen4exp_r2v_kv_set_flags")?,
                    state_bytes: load_sym_from(h, "tk_qwen4exp_r2v_mtp_state_bytes")?,
                    state_init: load_sym_from(h, "tk_qwen4exp_r2v_mtp_state_init")?,
                    state_reset: load_sym_from(h, "tk_qwen4exp_r2v_mtp_state_reset")?,
                    state_zero: load_sym_from(h, "tk_qwen4exp_r2v_mtp_state_zero")?,
                    mtp_adopt: load_sym_from(h, "tk_qwen4exp_r2v_mtp_adopt")?,
                    bind_facts: load_sym_from(h, "tk_qwen4exp_r2v_bind_facts")?,
                    bind_head_logits: load_sym_from(h, "tk_qwen4exp_r2v_bind_head_logits")?,
                    bind_dhead: load_sym_from(h, "tk_qwen4exp_r2v_bind_dhead")?,
                    gather_rows: load_sym_from(h, "tk_qwen4exp_r2v_gather_rows")?,
                    make_q4_head: load_sym_from(h, "tk_qwen4exp_r2v_mtp_make_q4_head")?,
                    q6_pack: load_sym_from(h, "tk_qwen4exp_r2v_q6_pack")?,
                    setup_coupled: load_sym_from(h, "tk_qwen4exp_r2v_mtp_setup_coupled")?,
                })
            })()
            .map_err(|e| {
                eprintln!(
                    "talus-qwen4exp: qwen4exp-r2v: the Rust setup is off ({e}); the C++ bodies run"
                )
            })
            .ok()
        })
        .as_ref()
        .ok_or("qwen4exp-r2v: the tk_qwen4exp_r2v_* table is not available")?;
    let ops = R2vOps {
        user: std::ptr::null_mut(),
        ver_init: Some(cb_ver_init),
        ver_set_logit_bias: Some(cb_ver_bias),
        mtp_load: Some(cb_mtp_load),
        mtp_bind: Some(cb_mtp_bind),
    };
    unsafe { (f.install)(&ops) };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atoi_like_c() {
        assert_eq!(c_atoi("1"), 1);
        assert_eq!(c_atoi("  -3x"), -3);
        assert_eq!(c_atoi("abc"), 0);
        assert_eq!(c_atoi(""), 0);
    }

    #[test]
    fn bump_rounds_to_256() {
        let mut b = Bump {
            base: std::ptr::null_mut(),
            used: 0,
        };
        b.take(1);
        b.take(256);
        b.take(0);
        b.take(257);
        assert_eq!(b.used, 256 + 256 + 512);
    }

    #[test]
    fn dense_index_parses_like_operator_shift() {
        let t = parse_dense_index("a q8_0 2 32 0 68\n\nb f32  1 4 256 16 extra\n").unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[1], ("b".into(), "f32".into(), 1, 4, 256, 16));
        assert!(parse_dense_index("x q8_0 2\n").is_err());
        assert!(parse_dense_index(" \n").is_err());
    }
}
