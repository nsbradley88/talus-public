//! GLM-5.3-Flash FFI: `libtalus_kernels.so` (generic `tk_*` through `talus-kernels-sys`, GLM `tkg_*` through dlsym),
//! HIP runtime helpers (through `hip-sys`), device buffers and graph capture.

#![allow(non_snake_case, clippy::too_many_arguments, clippy::missing_safety_doc)]

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;

pub use hip_sys::{hipError_t, hipGraphExec_t, hipGraph_t, hipStream_t, hipSuccess};
use talus_core_api::{CoreError, Result};
pub use talus_kernels_sys::{tk_ctx, tk_qtype, tk_status, LibTalusKernels, TK_OK};

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
const RTLD_NOW: c_int = 2;

fn dl_err() -> String {
    // SAFETY: dlerror returns a thread-local static string or NULL.
    unsafe {
        let p = dlerror();
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

pub fn kerr(msg: impl Into<String>) -> CoreError {
    CoreError::KernelError {
        code: -1,
        message: msg.into(),
    }
}

/// Fail on a nonzero `tkg_*` / hipError_t return.
fn sync_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("GLM_SYNC_CHECK")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// GLM_SYNC_CHECK=1: synchronize after every launch and attribute sticky errors to it (debug)
pub fn dbg_sync(what: &str) -> Result<()> {
    if sync_check() {
        // SAFETY: plain HIP calls.
        let e = unsafe { hip_sys::hipDeviceSynchronize() };
        let l = unsafe { hip_sys::hipGetLastError() };
        if e.0 != 0 || l.0 != 0 {
            return Err(CoreError::KernelError {
                code: if e.0 != 0 { e.0 } else { l.0 },
                message: format!(
                    "after {what}: sync {} / last {}",
                    hip_err_str(e.0),
                    hip_err_str(l.0)
                ),
            });
        }
    }
    Ok(())
}

#[inline]
pub fn ck(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        dbg_sync(what)
    } else {
        Err(CoreError::KernelError {
            code: rc,
            message: format!("{what}: {}", hip_err_str(rc)),
        })
    }
}

#[inline]
pub fn ckh(e: hipError_t, what: &str) -> Result<()> {
    ck(e.0, what)
}

pub fn hip_err_str(rc: c_int) -> String {
    // SAFETY: hipGetErrorString returns a static string.
    unsafe {
        let p = hip_sys::hipGetErrorString(hipError_t(rc));
        if p.is_null() {
            format!("hip error {rc}")
        } else {
            format!("{} ({rc})", CStr::from_ptr(p).to_string_lossy())
        }
    }
}

type S = *mut c_void; // hipStream_t as void*

/// GLM kernel entry points (`include/talus_glm_kernels.h`).
#[derive(Clone, Copy)]
pub struct Tkg {
    pub abi_version: unsafe extern "C" fn() -> c_int,
    pub hc_repeat: unsafe extern "C" fn(*const f32, *mut f32, c_int, c_int, S) -> c_int,
    pub hc_mix: unsafe extern "C" fn(
        *const c_void,
        *const f32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub hc_pre: unsafe extern "C" fn(
        *const f32,
        *const f32,
        c_int,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *mut f32,
        c_int,
        c_int,
        f32,
        f32,
        f32,
        c_int,
        *mut c_void,
        S,
    ) -> c_int,
    pub hc_post_mix: unsafe extern "C" fn(
        *const f32,
        *mut f32,
        *const f32,
        *const c_void,
        *mut f32,
        c_int,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub hc_post_mix_moe: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *const f32,
        *const c_void,
        *mut f32,
        c_int,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub hc_pre2: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *mut f32,
        c_int,
        c_int,
        f32,
        f32,
        f32,
        c_int,
        *mut c_void,
        S,
    ) -> c_int,
    pub multi_q8_0: unsafe extern "C" fn(
        c_int,
        *const *const c_void,
        *const *const f32,
        *const *mut f32,
        *const c_int,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub f32_matvec:
        unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, c_int, S) -> c_int,
    pub heads_q8_0: unsafe extern "C" fn(
        *const c_void,
        *const f32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub swiglu_clamp_q8:
        unsafe extern "C" fn(*const f32, *const f32, *mut c_void, c_int, f32, S) -> c_int,
    pub kda_gate_q8: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        *mut c_void,
        c_int,
        c_int,
        c_int,
        f32,
        S,
    ) -> c_int,
    pub kq_ids: unsafe extern "C" fn(
        c_int,
        *const c_void,
        usize,
        *const i32,
        c_int,
        c_int,
        *const f32,
        c_int,
        *mut f32,
        c_int,
        c_int,
        S,
    ) -> c_int,
    /// kq_ids for 1-2 matrices, distinct experts once (-1 = unsupported shape)
    pub kq_ids2: unsafe extern "C" fn(
        c_int,
        *const c_void,
        *const c_void,
        usize,
        *const i32,
        c_int,
        c_int,
        *const f32,
        c_int,
        *mut f32,
        *mut f32,
        c_int,
        c_int,
        S,
    ) -> c_int,
    /// device greedy argmax per row
    pub argmax_rows: unsafe extern "C" fn(
        *const f32,
        c_int,
        c_int,
        std::os::raw::c_long,
        *const i32,
        *mut i32,
        *mut i32,
        S,
    ) -> c_int,
    /// KDA conv + step + gate_q8 in one launch (-1 = no fused kernel for the layout)
    pub kda_fused: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *const i32,
        *const f32,
        *const f32,
        *mut f32,
        *const f32,
        *const f32,
        *mut c_void,
        *mut i32,
        c_int,
        c_int,
        c_int,
        c_int,
        f32,
        f32,
        f32,
        f32,
        S,
    ) -> c_int,
    /// router f32 mat-vec + routing in one launch (-1 = unsupported)
    pub f32mv_router: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        *const f32,
        *mut i32,
        *mut f32,
        c_int,
        f32,
        *mut i32,
        S,
    ) -> c_int,
    /// hc_post_mix + hc_pre2 in one launch (-1 = unsupported)
    pub hc_post_mix_pre: unsafe extern "C" fn(
        *const f32,
        *mut f32,
        *mut f32,
        *const c_void,
        *mut f32,
        c_int,
        c_int,
        c_int,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        f32,
        f32,
        f32,
        c_int,
        *mut c_void,
        *mut i32,
        S,
    ) -> c_int,
    /// hc_post_mix_moe + hc_pre2 in one launch (-1 = unsupported)
    pub hc_post_mix_moe_pre: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *mut f32,
        *const c_void,
        *mut f32,
        c_int,
        c_int,
        c_int,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        f32,
        f32,
        f32,
        c_int,
        *mut c_void,
        *mut i32,
        S,
    ) -> c_int,
    pub hc_post: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub hc_head:
        unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, f32, S) -> c_int,
    pub rms_norm: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        c_int,
        f32,
        S,
    ) -> c_int,
    pub add: unsafe extern "C" fn(*mut f32, *const f32, c_int, S) -> c_int,
    pub swiglu_clamp:
        unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, f32, S) -> c_int,
    pub embed:
        unsafe extern "C" fn(c_int, *const c_void, *const i32, *mut f32, c_int, c_int, S) -> c_int,
    pub kda_conv: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *const i32,
        c_int,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *mut f32,
        *mut f32,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub kda_gate: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        f32,
        S,
    ) -> c_int,
    pub mla_store: unsafe extern "C" fn(
        *const f32,
        *const f32,
        f32,
        *mut c_void,
        *const i32,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub idx_store: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        f32,
        *const f32,
        *mut c_void,
        *const i32,
        c_int,
        S,
    ) -> c_int,
    pub idx_pool: unsafe extern "C" fn(
        *const c_void,
        *const f32,
        *mut c_void,
        *const i32,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub dsa_select: unsafe extern "C" fn(
        *const f32,
        c_int,
        *const i32,
        c_int,
        c_int,
        c_int,
        c_int,
        *mut i32,
        S,
    ) -> c_int,
    pub router: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *mut i32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        f32,
        S,
    ) -> c_int,
    pub moe_combine: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub mmvq_ids_f32: unsafe extern "C" fn(
        c_int,
        *const c_void,
        usize,
        *const i32,
        c_int,
        c_int,
        *const f32,
        c_int,
        *mut f32,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub quantize_q8_1: unsafe extern "C" fn(*const f32, *mut c_void, c_int, S) -> c_int,
    pub glm_mmvq_ids: unsafe extern "C" fn(
        c_int,
        *const c_void,
        *const c_void,
        *const i32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub glm_mmvq_ids_pair: unsafe extern "C" fn(
        c_int,
        *const c_void,
        *const c_void,
        *const c_void,
        *const i32,
        *mut f32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub glm_kda_step: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *const i32,
        *const f32,
        *const f32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        c_int,
        f32,
        f32,
        f32,
        S,
    ) -> c_int,
    pub glm_indexer: unsafe extern "C" fn(
        *const f32,
        *const c_void,
        *const f32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub glm_sparse_fa_idx: unsafe extern "C" fn(
        *const f32,
        *const c_void,
        *const c_void,
        *const i32,
        *mut f32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        c_int,
        u32,
        u32,
        u32,
        f32,
        S,
    ) -> c_int,
    pub glm_sparse_fa_split_bytes: unsafe extern "C" fn(c_int, c_int, c_int, c_int) -> usize,
    pub gather_rows:
        unsafe extern "C" fn(*const c_void, usize, *const i32, c_int, *mut c_void, S) -> c_int,
    pub gather_q8_1:
        unsafe extern "C" fn(*const f32, *const i32, *mut c_void, c_int, c_int, S) -> c_int,
    /// routed experts of prompt windows on the fp16 kernels with the activations quantized once per token
    /// and the SwiGLU quantized inside the gate/up kernel (moe_w16.cu; optional; -1 = not covered)
    pub w16_moe_gu_q8: Option<
        unsafe extern "C" fn(
            c_int,
            *const c_void,
            *const c_void,
            usize,
            c_int,
            c_int,
            c_int,
            *const c_void,
            *const i32,
            i64,
            *const i32,
            i64,
            *mut c_void,
            f32,
            S,
        ) -> c_int,
    >,
    /// prompt-window KDA conv + step in one launch (bitwise conv then step; -1 = not covered)
    #[allow(clippy::type_complexity)]
    pub kda_step_conv_pf: Option<
        unsafe extern "C" fn(
            *const f32,
            *const f32,
            *const f32,
            *mut f32,
            *const f32,
            *const f32,
            *const f32,
            *const f32,
            *const f32,
            *mut f32,
            *const i32,
            *const f32,
            *const f32,
            *mut f32,
            *mut i32,
            c_int,
            c_int,
            c_int,
            c_int,
            f32,
            f32,
            f32,
            S,
        ) -> c_int,
    >,
    pub w16_moe_down: Option<
        unsafe extern "C" fn(
            c_int,
            *const c_void,
            usize,
            c_int,
            c_int,
            c_int,
            *const c_void,
            *const i32,
            *const i32,
            i64,
            *mut f32,
            i64,
            S,
        ) -> c_int,
    >,
    /// rb-num GLM_HC_BF16: 16-bit mHC streams on the prompt path (fmt 1 = fp16, 2 = bf16); optional symbols
    pub hc_post_mix16: Option<
        unsafe extern "C" fn(
            *const f32,
            *mut c_void,
            *const f32,
            *const c_void,
            *mut f32,
            c_int,
            c_int,
            c_int,
            c_int,
            S,
        ) -> c_int,
    >,
    pub hc_post_mix_moe16: Option<
        unsafe extern "C" fn(
            *const f32,
            *const f32,
            *const f32,
            *mut c_void,
            *const f32,
            *const c_void,
            *mut f32,
            c_int,
            c_int,
            c_int,
            c_int,
            S,
        ) -> c_int,
    >,
    pub hc_pre2_16: Option<
        unsafe extern "C" fn(
            *const c_void,
            *const f32,
            *const f32,
            *const f32,
            *const f32,
            *mut f32,
            *mut f32,
            c_int,
            c_int,
            f32,
            f32,
            f32,
            c_int,
            *mut c_void,
            c_int,
            S,
        ) -> c_int,
    >,
    pub hc_head16: Option<
        unsafe extern "C" fn(
            *const c_void,
            *const f32,
            *mut f32,
            c_int,
            c_int,
            f32,
            c_int,
            S,
        ) -> c_int,
    >,
    pub glm_mmvq_grouped: unsafe extern "C" fn(
        c_int,
        *const c_void,
        usize,
        c_int,
        *const i32,
        *const c_void,
        *const i32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        S,
    ) -> c_int,
    pub moe_route_sort: unsafe extern "C" fn(
        *const i32,
        c_int,
        c_int,
        c_int,
        *mut i32,
        *mut i32,
        *mut i32,
        S,
    ) -> c_int,
}

/// Loaded kernel library: the generic table, the GLM table and one `tk_ctx`.
pub struct GlmLib {
    /// `tk_register_mmq_q8_buf` (marks an activation buffer as MMQ layout for tk_prefill_moe_mmq)
    pub register_mmq: Option<unsafe extern "C" fn(*const c_void)>,
    pub tk: LibTalusKernels,
    pub g: Tkg,
    pub ctx: *mut tk_ctx,
    pub path: PathBuf,
}

unsafe impl Send for GlmLib {}
unsafe impl Sync for GlmLib {}

pub fn kernel_lib_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("TALUS_KERNELS_LIB") {
        if !p.is_empty() {
            v.push(PathBuf::from(p));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            v.push(d.join("libtalus_kernels.so"));
        }
    }
    v.push(PathBuf::from("libtalus_kernels.so"));
    v
}

impl GlmLib {
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        let mut cands = Vec::new();
        if let Some(p) = explicit {
            cands.push(p.to_path_buf());
        }
        cands.extend(kernel_lib_candidates());
        let mut last = String::new();
        for p in cands {
            if !p.exists() && p.components().count() > 1 {
                continue;
            }
            match Self::load_path(&p) {
                Ok(l) => return Ok(l),
                Err(e) => last = format!("{}: {e}", p.display()),
            }
        }
        Err(kerr(format!(
            "cannot load libtalus_kernels.so with GLM kernels ({last})"
        )))
    }

    fn load_path(p: &Path) -> std::result::Result<Self, String> {
        let tk = LibTalusKernels::load(p)?;
        let c = CString::new(p.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: valid NUL-terminated path; the library is already loaded (refcount bump).
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
        if h.is_null() {
            return Err(dl_err());
        }
        macro_rules! sym {
            ($n:literal) => {{
                let cn = CString::new($n).unwrap();
                // SAFETY: handle valid, name NUL-terminated.
                let f = unsafe { dlsym(h, cn.as_ptr()) };
                if f.is_null() {
                    return Err(format!("missing symbol {}", $n));
                }
                // SAFETY: the C ABI in include/talus_glm_kernels.h matches the field type.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(f)
                }
            }};
        }
        macro_rules! sym_opt {
            ($n:literal) => {{
                let cn = CString::new($n).unwrap();
                // SAFETY: handle valid, name NUL-terminated.
                let f = unsafe { dlsym(h, cn.as_ptr()) };
                if f.is_null() {
                    None
                } else {
                    // SAFETY: the C ABI in include/talus_glm_kernels.h matches the field type.
                    #[allow(clippy::missing_transmute_annotations)]
                    Some(unsafe { std::mem::transmute::<*mut c_void, _>(f) })
                }
            }};
        }
        let g = Tkg {
            abi_version: sym!("tkg_abi_version"),
            w16_moe_gu_q8: sym_opt!("tkg_w16_moe_gu_q8"),
            w16_moe_down: sym_opt!("tkg_w16_moe_down"),
            kda_step_conv_pf: sym_opt!("tkg_kda_step_conv_pf"),
            hc_post_mix16: sym_opt!("tkg_hc_post_mix16"),
            hc_post_mix_moe16: sym_opt!("tkg_hc_post_mix_moe16"),
            hc_pre2_16: sym_opt!("tkg_hc_pre2_16"),
            hc_head16: sym_opt!("tkg_hc_head16"),
            hc_repeat: sym!("tkg_hc_repeat"),
            hc_mix: sym!("tkg_hc_mix"),
            hc_pre: sym!("tkg_hc_pre"),
            f32_matvec: sym!("tkg_f32_matvec"),
            hc_post_mix: sym!("tkg_hc_post_mix"),
            hc_post_mix_moe: sym!("tkg_hc_post_mix_moe"),
            hc_pre2: sym!("tkg_hc_pre2"),
            multi_q8_0: sym!("tkg_multi_q8_0"),
            heads_q8_0: sym!("tkg_heads_q8_0"),
            swiglu_clamp_q8: sym!("tkg_swiglu_clamp_q8"),
            kda_gate_q8: sym!("tkg_kda_gate_q8"),
            kq_ids: sym!("tkg_kq_ids"),
            kq_ids2: sym!("tkg_kq_ids2"),
            hc_post_mix_pre: sym!("tkg_hc_post_mix_pre"),
            hc_post_mix_moe_pre: sym!("tkg_hc_post_mix_moe_pre"),
            f32mv_router: sym!("tkg_f32mv_router"),
            kda_fused: sym!("tkg_kda_fused"),
            argmax_rows: sym!("tkg_argmax_rows"),
            hc_post: sym!("tkg_hc_post"),
            hc_head: sym!("tkg_hc_head"),
            rms_norm: sym!("tkg_rms_norm"),
            add: sym!("tkg_add"),
            swiglu_clamp: sym!("tkg_swiglu_clamp"),
            embed: sym!("tkg_embed"),
            kda_conv: sym!("tkg_kda_conv"),
            kda_gate: sym!("tkg_kda_gate"),
            mla_store: sym!("tkg_mla_store"),
            idx_store: sym!("tkg_idx_store"),
            idx_pool: sym!("tkg_idx_pool"),
            dsa_select: sym!("tkg_dsa_select"),
            router: sym!("tkg_router"),
            moe_combine: sym!("tkg_moe_combine"),
            mmvq_ids_f32: sym!("tkg_mmvq_ids_f32"),
            quantize_q8_1: sym!("tkg_quantize_q8_1"),
            glm_mmvq_ids: sym!("tkg_glm_mmvq_ids"),
            glm_mmvq_ids_pair: sym!("tkg_glm_mmvq_ids_pair"),
            glm_kda_step: sym!("tkg_glm_kda_step"),
            glm_indexer: sym!("tkg_glm_indexer"),
            glm_sparse_fa_idx: sym!("tkg_glm_sparse_fa_idx"),
            glm_sparse_fa_split_bytes: sym!("tkg_glm_sparse_fa_split_bytes"),
            gather_rows: sym!("tkg_gather_rows"),
            moe_route_sort: sym!("tkg_moe_route_sort"),
            gather_q8_1: sym!("tkg_gather_q8_1"),
            glm_mmvq_grouped: sym!("tkg_glm_mmvq_grouped"),
        };
        // SAFETY: plain C call.
        let abi = unsafe { (g.abi_version)() };
        if abi != 7 {
            return Err(format!("tkg ABI {abi} != 7"));
        }
        // GLM defaults for the shared kernels (the caller's environment wins): K-quant MMQ for the Q4_K / Q6_K dense
        // projections, no hipBLAS / hipBLASLt handles (GLM never calls them; saves their workspace near the floor), and
        // the fp16 routed-expert kernels for GLM only (STRATA_PF_W16_MOE=0 = kill
        // switch; MiMo / other engines keep the opt-in default)
        for (k, v) in [
            ("STRATA_MMQ_KQ", "1"),
            ("STRATA_PREFILL_NO_BLAS", "1"),
            ("STRATA_PF_W16_MOE", "1"),
        ] {
            if std::env::var_os(k).is_none() {
                std::env::set_var(k, v);
            }
        }
        let mut cfg = ptr::null_mut();
        // SAFETY: tk_* C ABI.
        let st = unsafe { (tk.tk_config_create)(&mut cfg) };
        if st != TK_OK {
            return Err(format!("tk_config_create {st}"));
        }
        // SAFETY: cfg valid.
        unsafe {
            let _ = (tk.tk_config_from_env)(cfg);
        }
        let mut ctx = ptr::null_mut();
        // SAFETY: cfg valid; ctx out-pointer.
        let st = unsafe { (tk.tk_ctx_create)(0, cfg, &mut ctx) };
        // SAFETY: cfg valid.
        unsafe { (tk.tk_config_destroy)(cfg) };
        if st != TK_OK || ctx.is_null() {
            return Err(format!("tk_ctx_create {st}"));
        }
        let reg = {
            let cn = CString::new("tk_register_mmq_q8_buf").unwrap();
            // SAFETY: handle valid.
            let f = unsafe { dlsym(h, cn.as_ptr()) };
            if f.is_null() {
                None
            } else {
                // SAFETY: extern "C" void tk_register_mmq_q8_buf(const void*).
                Some(unsafe {
                    std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*const c_void)>(f)
                })
            }
        };
        Ok(Self {
            register_mmq: reg,
            tk,
            g,
            ctx,
            path: p.to_path_buf(),
        })
    }

    #[inline]
    pub fn cktk(&self, st: tk_status, what: &str) -> Result<()> {
        if st == TK_OK {
            return dbg_sync(what);
        }
        // SAFETY: ctx valid.
        let msg = unsafe {
            let p = (self.tk.tk_last_error)(self.ctx);
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Err(CoreError::KernelError {
            code: st,
            message: format!("{what}: {msg}"),
        })
    }
}

// ---------------------------------------------------------------------------------------------------------------
// HIP helpers

pub fn dev_alloc(bytes: usize) -> Result<*mut c_void> {
    let mut p = ptr::null_mut();
    // SAFETY: out-pointer valid.
    let e = unsafe { hip_sys::hipMalloc(&mut p, bytes.max(256)) };
    if e != hipSuccess || p.is_null() {
        return Err(CoreError::KernelError {
            code: e.0,
            message: format!("hipMalloc({bytes}): {}", hip_err_str(e.0)),
        });
    }
    Ok(p)
}

pub fn dev_free(p: *mut c_void) {
    if !p.is_null() {
        // SAFETY: p came from hipMalloc.
        unsafe {
            let _ = hip_sys::hipFree(p);
        }
    }
}

/// Pinned, device-mapped host memory (unified memory on Strix Halo: kernels read it directly).
pub fn host_alloc_mapped(bytes: usize) -> Result<*mut c_void> {
    let mut p = ptr::null_mut();
    // SAFETY: out-pointer valid.
    let e = unsafe {
        hip_sys::hipHostMalloc(
            &mut p,
            bytes.max(256),
            hip_sys::hipHostMallocMapped | hip_sys::hipHostMallocCoherent,
        )
    };
    if e != hipSuccess || p.is_null() {
        return Err(CoreError::KernelError {
            code: e.0,
            message: format!("hipHostMalloc({bytes}): {}", hip_err_str(e.0)),
        });
    }
    Ok(p)
}

pub fn host_alloc_pinned(bytes: usize) -> Result<*mut c_void> {
    let mut p = ptr::null_mut();
    // SAFETY: out-pointer valid.
    let e =
        unsafe { hip_sys::hipHostMalloc(&mut p, bytes.max(256), hip_sys::hipHostMallocDefault) };
    if e != hipSuccess || p.is_null() {
        return Err(CoreError::KernelError {
            code: e.0,
            message: format!("hipHostMalloc({bytes}): {}", hip_err_str(e.0)),
        });
    }
    Ok(p)
}

pub fn host_free(p: *mut c_void) {
    if !p.is_null() {
        // SAFETY: p came from hipHostMalloc.
        unsafe {
            let _ = hip_sys::hipHostFree(p);
        }
    }
}

pub fn h2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
    // SAFETY: caller provides valid ranges.
    ckh(
        unsafe { hip_sys::hipMemcpy(dst, src, bytes, hip_sys::hipMemcpyHostToDevice) },
        "hipMemcpy H2D",
    )
}

pub fn d2h(dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
    // SAFETY: caller provides valid ranges.
    ckh(
        unsafe { hip_sys::hipMemcpy(dst, src, bytes, hip_sys::hipMemcpyDeviceToHost) },
        "hipMemcpy D2H",
    )
}

pub fn d2d_async(dst: *mut c_void, src: *const c_void, bytes: usize, s: hipStream_t) -> Result<()> {
    // SAFETY: caller provides valid ranges.
    ckh(
        unsafe { hip_sys::hipMemcpyAsync(dst, src, bytes, hip_sys::hipMemcpyDeviceToDevice, s) },
        "hipMemcpyAsync D2D",
    )
}

pub fn h2d_async(dst: *mut c_void, src: *const c_void, bytes: usize, s: hipStream_t) -> Result<()> {
    // SAFETY: caller provides valid ranges.
    ckh(
        unsafe { hip_sys::hipMemcpyAsync(dst, src, bytes, hip_sys::hipMemcpyHostToDevice, s) },
        "hipMemcpyAsync H2D",
    )
}

pub fn memset(dst: *mut c_void, v: i32, bytes: usize) -> Result<()> {
    // SAFETY: caller provides valid range.
    ckh(unsafe { hip_sys::hipMemset(dst, v, bytes) }, "hipMemset")
}

pub fn memset_async(dst: *mut c_void, v: i32, bytes: usize, s: hipStream_t) -> Result<()> {
    // SAFETY: caller provides valid range.
    ckh(
        unsafe { hip_sys::hipMemsetAsync(dst, v, bytes, s) },
        "hipMemsetAsync",
    )
}

pub fn stream_create() -> Result<hipStream_t> {
    let mut s = ptr::null_mut();
    // SAFETY: out-pointer valid.
    ckh(
        unsafe { hip_sys::hipStreamCreate(&mut s) },
        "hipStreamCreate",
    )?;
    Ok(s)
}

pub fn sync(s: hipStream_t) -> Result<()> {
    // SAFETY: stream valid.
    ckh(
        unsafe { hip_sys::hipStreamSynchronize(s) },
        "hipStreamSynchronize",
    )
}

/// Capture the launches issued by `body` on `s` into an instantiated graph.
pub fn capture<F: FnOnce() -> Result<()>>(s: hipStream_t, body: F) -> Result<hipGraphExec_t> {
    // SAFETY: stream valid.
    ckh(
        unsafe { hip_sys::hipStreamBeginCapture(s, hip_sys::hipStreamCaptureModeRelaxed) },
        "hipStreamBeginCapture",
    )?;
    let r = body();
    let mut g: hipGraph_t = ptr::null_mut();
    // SAFETY: capture active on s.
    let e = unsafe { hip_sys::hipStreamEndCapture(s, &mut g) };
    r?;
    ckh(e, "hipStreamEndCapture")?;
    let mut ex: hipGraphExec_t = ptr::null_mut();
    // SAFETY: g valid.
    let e =
        unsafe { hip_sys::hipGraphInstantiate(&mut ex, g, ptr::null_mut(), ptr::null_mut(), 0) };
    // SAFETY: g valid; the exec keeps its own copy.
    unsafe {
        let _ = hip_sys::hipGraphDestroy(g);
    }
    ckh(e, "hipGraphInstantiate")?;
    Ok(ex)
}

pub fn graph_launch(ex: hipGraphExec_t, s: hipStream_t) -> Result<()> {
    // SAFETY: ex valid.
    ckh(unsafe { hip_sys::hipGraphLaunch(ex, s) }, "hipGraphLaunch")
}

pub fn graph_destroy(ex: hipGraphExec_t) {
    if !ex.is_null() {
        // SAFETY: ex valid.
        unsafe {
            let _ = hip_sys::hipGraphExecDestroy(ex);
        }
    }
}
