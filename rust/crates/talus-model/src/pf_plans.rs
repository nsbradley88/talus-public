//! Shared prefill launch plans of the host  S9): the current device's tuning row and the choices derived from it.

use std::ffi::c_void;
use std::sync::OnceLock;

use talus_core_api::{DeviceId, GpuArch};
use talus_kernels::tune::{self, TpTune};
use talus_kernels::w32::RouteEnv;
use talus_kernels_sys::{
    tk_ctx, tk_moe_mmq_product, tk_qtype, tk_status, tk_stream, LibTalusKernels,
};

/// The current device's row (`tp_tune_current`); no device: the defaults.
pub fn tune_current() -> &'static TpTune {
    static T: OnceLock<&'static TpTune> = OnceLock::new();
    T.get_or_init(
        || match talus_hip::Device::new(DeviceId(0)).map(|d| d.arch()) {
            Ok(GpuArch::Gfx1151) => tune::row_for("gfx1151"),
            _ => &tune::TP_TUNE_DEFAULT,
        },
    )
}

fn i8_env() -> i32 {
    static V: OnceLock<i32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("TP_I8_VARIANT")
            .map(|v| talus_kernels::plans::c_atoi(&v))
            .unwrap_or(0)
    })
}

/// `talus_pf_i8gemm`'s kernel variant for an expansion mode.
pub fn i8gemm(mode: i32) -> i32 {
    tune::i8gemm_variant(tune_current(), i8_env(), mode)
}

/// `talus_pf_i8gemm_swiglu_q8`'s kernel variant for an expansion mode.
pub fn swq8(mode: i32) -> i32 {
    tune::swq8_variant(tune_current(), i8_env(), mode)
}

/// S10: compare this host's W32 table lookup with the library's over every (type, columns), once per process; a
/// difference is reported (the library's own lookup keeps deciding) so a drift between the two cannot pass unnoticed.
pub fn w32_startup_check(lib_plan: Option<unsafe extern "C" fn(i32, i32) -> i32>) {
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        let Some(f) = lib_plan else {
            return;
        };
        let gfx1151 = matches!(talus_hip::Device::new(DeviceId(0)).map(|d| d.arch()), Ok(GpuArch::Gfx1151));
        // SAFETY: plain table lookup, no device work.
        let diff = talus_kernels::w32::differences(gfx1151, |t, n| unsafe { f(t, n) });
        for (t, n, h, l) in diff.iter().take(8) {
            eprintln!("talus: W32 table: type {t}, {n} columns: host plan {h} differs from the library's {l}");
        }
        if diff.len() > 8 {
            eprintln!("talus: W32 table: {} differences in all", diff.len());
        }
    });
}

/// S10: `tk_mmvq` with the host's route; the legacy export on an older library.
///
/// # Safety
/// As `tk_mmvq`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn mmvq(
    lib: &LibTalusKernels,
    ctx: *mut tk_ctx,
    qtype: tk_qtype,
    w: *const c_void,
    x: *const c_void,
    y: *mut f32,
    n_in: i32,
    n_out: i32,
    ncols: i32,
    stream: tk_stream,
) -> tk_status {
    if let (Some(v), Some(on)) = (lib.tk_mmvq_v, lib.tk_mmvq_w32_on) {
        let r = talus_kernels::w32::route(&RouteEnv::get(), on() != 0, qtype as i32, n_out, ncols);
        v(ctx, qtype, w, x, y, n_in, n_out, ncols, stream, r)
    } else {
        (lib.tk_mmvq)(ctx, qtype, w, x, y, n_in, n_out, ncols, stream)
    }
}

struct MoeEnv {
    inplace_on: bool,
    ktail: bool,
    ver: i32,
    v74_min: i32,
    force_gu: bool,
    w16_gu: bool,
    w16_moe: bool,
    q2k: bool,
    q6k: bool,
}

fn moe_env() -> &'static MoeEnv {
    static E: OnceLock<MoeEnv> = OnceLock::new();
    E.get_or_init(|| {
        let int =
            |n: &str, d: i32| std::env::var(n).map_or(d, |v| talus_kernels::plans::c_atoi(&v));
        let first = |n: &str| std::env::var(n).ok().and_then(|v| v.bytes().next());
        MoeEnv {
            inplace_on: first("TALUS_PF_MOE") != Some(b'0')
                && first("STRATA_PF_INPLACE") != Some(b'0')
                && first("TP_MOE_KTAIL_SKIP") != Some(b'0'),
            ktail: first("TP_MOE_KTAIL") != Some(b'0'),
            ver: int("TP_MOE_VER", -1),
            v74_min: int("TP_MOE_V74_MIN", 0),
            force_gu: first("TALUS_PF_MOE_GU") == Some(b'2'),
            w16_gu: first("TALUS_W16_GU") != Some(b'0'),
            w16_moe: first("STRATA_PF_W16_MOE") == Some(b'1'),
            q2k: first("TP_MOE_Q2K") != Some(b'0'),
            q6k: first("TALUS_PF_I8_Q6K") != Some(b'0'),
        }
    })
}

fn moe_ver_of(p: &tk_moe_mmq_product) -> i32 {
    let e = moe_env();
    tune::moe_version(
        tune_current(),
        e.ver,
        e.v74_min,
        p.n_experts,
        p.total_rows,
        p.w_cols,
    )
}

/// S10: `tk_prefill_moe_mmq_l` with the host's tp_moe kernel version.
///
/// # Safety
/// As `tk_prefill_moe_mmq`.
pub unsafe fn moe_mmq(
    lib: &LibTalusKernels,
    ctx: *mut tk_ctx,
    p: *const tk_moe_mmq_product,
    layout: i32,
    s: tk_stream,
) -> tk_status {
    match lib.tk_prefill_moe_mmq_lv {
        Some(f) => f(
            ctx,
            p,
            layout,
            if p.is_null() { -1 } else { moe_ver_of(&*p) },
            s,
        ),
        None => lib.moe_mmq_l(ctx, p, layout, s),
    }
}

/// S10: `tk_prefill_moe_mmq_gu_l` with the host's version and fused rule (no experts: the library's).
///
/// # Safety
/// As `tk_prefill_moe_mmq_gu`.
pub unsafe fn moe_mmq_gu(
    lib: &LibTalusKernels,
    ctx: *mut tk_ctx,
    g: *const tk_moe_mmq_product,
    w_up: *const c_void,
    limit: f32,
    layout: i32,
    s: tk_stream,
) -> tk_status {
    let Some(f) = lib.tk_prefill_moe_mmq_gu_lv else {
        return lib.moe_mmq_gu_l(ctx, g, w_up, limit, layout, s);
    };
    if g.is_null() || (*g).n_experts <= 0 {
        return f(ctx, g, w_up, limit, layout, -1, -1, s);
    }
    let (p, e, row) = (&*g, moe_env(), tune_current());
    let ty = p.qtype as i32;
    let mean = p.total_rows / i64::from(p.n_experts);
    let w16_takes = e.w16_gu
        && tune::moe_w16_enabled(row, e.w16_moe, ty)
        && mean >= i64::from(row.pf_w16_moe_min_rows);
    let go = tune::moe_gu(
        row,
        e.force_gu,
        tune::moe_dot_mode(ty, e.q2k, e.q6k),
        w16_takes,
        mean,
    );
    f(ctx, g, w_up, limit, layout, moe_ver_of(p), i32::from(go), s)
}

/// S10: `tk_prefill_moe_mmq_gu_q8_l` with the host's version and fused rule.
///
/// # Safety
/// As `tk_prefill_moe_mmq_gu_q8`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_mmq_gu_q8(
    lib: &LibTalusKernels,
    ctx: *mut tk_ctx,
    g: *const tk_moe_mmq_product,
    w_up: *const c_void,
    limit: f32,
    down_qtype: i32,
    xq_out: *mut c_void,
    xq_rows: i64,
    layout: i32,
    s: tk_stream,
) -> tk_status {
    let Some(f) = lib.tk_prefill_moe_mmq_gu_q8_lv else {
        return lib.moe_mmq_gu_q8_l(ctx, g, w_up, limit, down_qtype, xq_out, xq_rows, layout, s);
    };
    if g.is_null() || (*g).n_experts <= 0 {
        return f(
            ctx, g, w_up, limit, down_qtype, xq_out, xq_rows, layout, -1, -1, s,
        );
    }
    let p = &*g;
    let go = tune::moe_gu_q8(tune_current(), p.total_rows / i64::from(p.n_experts));
    f(
        ctx,
        g,
        w_up,
        limit,
        down_qtype,
        xq_out,
        xq_rows,
        layout,
        moe_ver_of(p),
        i32::from(go),
        s,
    )
}

type FnF32MatmulV = unsafe extern "C" fn(
    *mut tk_ctx,
    *const f32,
    *const f32,
    *mut f32,
    i32,
    i32,
    i32,
    tk_stream,
    i32,
) -> tk_status;

/// S7: `tk_f32_matmul` with the host's plan; the legacy export on an older library.
///
/// # Safety
/// As `tk_f32_matmul`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn f32_matmul(
    lib: &LibTalusKernels,
    ctx: *mut tk_ctx,
    w: *const f32,
    x: *const f32,
    y: *mut f32,
    n_in: i32,
    n_out: i32,
    n_tok: i32,
    s: tk_stream,
) -> tk_status {
    static F: OnceLock<Option<usize>> = OnceLock::new();
    match *F.get_or_init(|| lib.optional_sym("tk_f32_matmul_v").map(|p| p as usize)) {
        Some(p) => {
            // SAFETY: the C ABI in include/talus_kernels.h.
            let f = std::mem::transmute::<usize, FnF32MatmulV>(p);
            f(
                ctx,
                w,
                x,
                y,
                n_in,
                n_out,
                n_tok,
                s,
                talus_kernels::plans::f32_matmul_plan(n_in, n_tok),
            )
        }
        None => (lib.tk_f32_matmul)(ctx, w, x, y, n_in, n_out, n_tok, s),
    }
}

type FnQ36AttnV = unsafe extern "C" fn(
    *mut tk_ctx,
    bool,
    i32,
    i32,
    *const f32,
    *const c_void,
    *const c_void,
    *const i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    f32,
    i32,
    i32,
    *mut f32,
    *mut f32,
    *mut f32,
    tk_stream,
    i32,
) -> tk_status;

/// S7: `tk_q36_attn_wmma_v`, when the library has it (`None`: the legacy export).
pub fn q36_attn_v(lib: &LibTalusKernels) -> Option<FnQ36AttnV> {
    static F: OnceLock<Option<usize>> = OnceLock::new();
    F.get_or_init(|| lib.optional_sym("tk_q36_attn_wmma_v").map(|p| p as usize))
        // SAFETY: the C ABI in include/talus_kernels.h.
        .map(|p| unsafe { std::mem::transmute::<usize, FnQ36AttnV>(p) })
}

/// S8c: the tp_moe kernel version of a product (the host's rule; `moe_version_for`).
pub fn moe_version(n_experts: i32, total_rows: i64, k: i64) -> i32 {
    let e = moe_env();
    tune::moe_version(tune_current(), e.ver, e.v74_min, n_experts, total_rows, k)
}

/// S8c: FN prefill's gather-free group rule for one product type and shape (`tk_qwen4exp_p2_mmq_inplace_ok`).
pub fn moe_inplace_ok(ty: i32, n: i64, k: i64) -> bool {
    let e = moe_env();
    let (n, k) = (n as i32, k as i32);
    let mode = tune::moe_dot_mode(ty, e.q2k, e.q6k);
    let supported = tune::moe_supported(mode, ty, n, k, e.ktail);
    let w16 = tune::moe_w16_enabled(tune_current(), e.w16_moe, ty);
    tune::moe_inplace_ok(
        e.inplace_on,
        supported,
        w16,
        ty,
        moe_version(16, 0, i64::from(k)),
    )
}

/// A host plan against the library's for one release (as `plan_choose` in the library): a difference is reported (8 times at
/// most) and the library's value runs, unless `TALUS_PLAN_TRUST=1`.
pub fn plan_choose(site: &str, host: i32, lib: i32) -> i32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static WARNED: AtomicU32 = AtomicU32::new(0);
    static TRUST: OnceLock<bool> = OnceLock::new();
    if host == lib {
        return host;
    }
    let trust =
        *TRUST.get_or_init(|| std::env::var("TALUS_PLAN_TRUST").is_ok_and(|v| v.starts_with('1')));
    if WARNED.fetch_add(1, Ordering::Relaxed) < 8 {
        eprintln!(
            "talus: {site}: host plan {host} differs from the library's {lib}; running the {}",
            if trust { "host's" } else { "library's" }
        );
    }
    if trust {
        host
    } else {
        lib
    }
}
