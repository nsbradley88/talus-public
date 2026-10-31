//! Internal FFI, HIP runtime, GGUF, and error helpers for the Qwen3.5 / Qwen3.6 (`q36`) engine.

#![allow(non_snake_case, non_camel_case_types, clippy::too_many_arguments)]

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::path::Path;
use std::ptr;
use std::sync::{Arc, OnceLock};

pub use hip_sys::{
    hipDeviceSynchronize, hipError_t, hipFree, hipGetErrorString, hipGraphDestroy,
    hipGraphExecDestroy, hipGraphExec_t, hipGraphInstantiate, hipGraphLaunch, hipGraph_t,
    hipHostFree, hipHostMalloc, hipHostMallocDefault, hipHostRegister, hipHostRegisterMapped,
    hipHostUnregister, hipMalloc, hipMemGetInfo, hipMemcpy, hipMemcpyAsync,
    hipMemcpyDeviceToDevice, hipMemcpyDeviceToHost, hipMemcpyHostToDevice, hipMemset,
    hipMemsetAsync, hipStreamBeginCapture, hipStreamCaptureModeRelaxed, hipStreamCreate,
    hipStreamDestroy, hipStreamEndCapture, hipStreamSynchronize, hipStream_t, hipSuccess,
};

pub use hip_sys::{
    hipHostMallocDefault as HIP_HOST_MALLOC_DEFAULT,
    hipHostRegisterMapped as HIP_HOST_REGISTER_MAPPED,
    hipMemcpyDeviceToDevice as HIP_MEMCPY_DEVICE_TO_DEVICE,
    hipMemcpyDeviceToHost as HIP_MEMCPY_DEVICE_TO_HOST,
    hipMemcpyHostToDevice as HIP_MEMCPY_HOST_TO_DEVICE,
    hipStreamCaptureModeRelaxed as HIP_STREAM_CAPTURE_MODE_RELAXED,
};

pub use talus_core_api::{CoreError, Result};
pub use talus_gguf::{GgufFile, GgufValue, TensorEntry as TensorInfo};
pub use talus_kernels_sys::{
    tk_config, tk_ctx, tk_mmvq_item, tk_moe_mmq_product,
    tk_qtype::{self, *},
    tk_status, tk_stream, LibTalusKernels, TK_OK,
};

/// Sampling parameters passed to `tk_q36_argmax_samp` / `tk_q36_lse` (`talus::q36::DraftSamp` in `include/talus_kernels.h`).
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct tk_q36_draft_samp {
    pub tau: f32,
    pub seed: u32,
    pub ctr: u32,
    pub lmin: f32,
    pub topp: f32,
    pub topk: i32,
}

/// Pending GDN replay buffers (`talus::q36::GdnPend` in `include/talus_kernels.h`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_q36_gdn_pend {
    pub q: [*mut f32; 2],
    pub h: [*mut f32; 2],
    pub a: [*mut f32; 2],
    pub b: [*mut f32; 2],
    pub ctl: *const i32,
    pub store: bool,
}

#[inline]
pub fn io_err(e: std::io::Error) -> CoreError {
    CoreError::IoError {
        message: e.to_string(),
    }
}

pub trait VecAssignExt<T: Clone> {
    fn assign(&mut self, count: usize, val: T);
}

impl<T: Clone> VecAssignExt<T> for Vec<T> {
    #[inline]
    fn assign(&mut self, count: usize, val: T) {
        self.clear();
        self.resize(count, val);
    }
}

#[macro_export]
macro_rules! q36_anyhow {
    ($($arg:tt)*) => {
        $crate::qwen35::ffi::CoreError::InvalidArgument {
            reason: format!($($arg)*),
        }
    };
}

#[macro_export]
macro_rules! q36_bail {
    ($($arg:tt)*) => {
        return Err($crate::qwen35::ffi::CoreError::InvalidArgument {
            reason: format!($($arg)*),
        })
    };
}

#[macro_export]
macro_rules! q36_ensure {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            return Err($crate::qwen35::ffi::CoreError::InvalidArgument {
                reason: format!($($arg)*),
            });
        }
    };
}

pub use crate::{q36_anyhow as anyhow, q36_bail as bail, q36_ensure as ensure};

/// Extension helpers on `talus_gguf::TensorEntry` (`TensorInfo`).
pub trait TensorInfoExt {
    fn name(&self) -> &str;
    fn dimensions(&self) -> &[u64];
    fn n_elements(&self) -> u64;
    fn n_bytes(&self) -> u64;
    fn ggml_type_u32(&self) -> u32;
}

impl TensorInfoExt for TensorInfo {
    #[inline]
    fn name(&self) -> &str {
        &self.name
    }

    #[inline]
    fn dimensions(&self) -> &[u64] {
        self.shape()
    }

    #[inline]
    fn n_elements(&self) -> u64 {
        self.shape().iter().copied().product::<u64>()
    }

    #[inline]
    fn n_bytes(&self) -> u64 {
        self.byte_size
    }

    #[inline]
    fn ggml_type_u32(&self) -> u32 {
        self.qtype.ggml_type_id()
    }
}

/// Extension helpers on `talus_gguf::GgufFile`.
pub trait GgufFileExt {
    fn arch(&self) -> Option<&str>;
    fn tensor(&self, name: &str) -> Option<&TensorInfo>;
}

impl GgufFileExt for GgufFile {
    #[inline]
    fn arch(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }

    #[inline]
    fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.find_tensor(name)
    }
}

// ============================================================================
// Dynamic `libtalus_kernels.so` loader & C++ W32 symbol resolution
// ============================================================================

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[cfg(target_os = "linux")]
const RTLD_NOW: c_int = 0x2;
#[cfg(target_os = "linux")]
const RTLD_LOCAL: c_int = 0;

struct LoadedKernels {
    lib: Arc<LibTalusKernels>,
    set_w32: Option<unsafe extern "C" fn(c_int, c_int)>,
    set_w32_waves: Option<unsafe extern "C" fn(c_int)>,
    w32_config: Option<unsafe extern "C" fn(c_int, c_int, *mut c_int, *mut c_int)>,
    w32_active: Option<unsafe extern "C" fn() -> bool>,
    // the opt-in W4A4 (iu4, integer group scales) prompt GEMM of src/prefill/tp_w4i.cu (optional symbols)
    w4i_pack: Option<W4iPackFn>,
    w4i_gemm: Option<W4iGemmFn>,
    // the wide verify tiles' state and their tile-plane weight copies (optional symbols)
    wide: Option<WideFns>,
}

/// `talus_mmvq_wide_*` (include/talus_kernels.h)
#[derive(Clone, Copy)]
struct WideFns {
    active: unsafe extern "C" fn() -> c_int,
    pack_mode: unsafe extern "C" fn() -> c_int,
    note: unsafe extern "C" fn(c_int, *const c_void, c_int, c_int) -> c_int,
    noted_bytes: unsafe extern "C" fn() -> u64,
    pack_noted: unsafe extern "C" fn() -> c_int,
    release: unsafe extern "C" fn(*const c_void),
    packed_bytes: unsafe extern "C" fn() -> u64,
}

/// `talus_pf_w4i_pack(x, gate, T, K, ldx, signs, wg, xq, xk, xS, stream)`
pub type W4iPackFn = unsafe extern "C" fn(
    *const f32,
    *const f32,
    c_int,
    c_int,
    i64,
    *const i8,
    c_int,
    *mut u8,
    *mut u8,
    *mut f32,
    *mut c_void,
) -> c_int;
/// `talus_pf_w4i_gemm(wq, ws, wg, N, K, xq, xk, xS, T, dst, ld, stream)`
pub type W4iGemmFn = unsafe extern "C" fn(
    *const u8,
    *const f32,
    c_int,
    c_int,
    c_int,
    *const u8,
    *const u8,
    *const f32,
    c_int,
    *mut f32,
    i64,
    *mut c_void,
) -> c_int;

unsafe impl Send for LoadedKernels {}
unsafe impl Sync for LoadedKernels {}

static GLOBAL_KERNELS: OnceLock<std::result::Result<LoadedKernels, String>> = OnceLock::new();

fn load_kernels_once() -> &'static std::result::Result<LoadedKernels, String> {
    GLOBAL_KERNELS.get_or_init(|| {
        let mut candidate_paths: Vec<String> = Vec::new();
        if let Ok(p) = std::env::var("TALUS_KERNELS_LIB") {
            if !p.is_empty() {
                candidate_paths.push(p);
            }
        }
        for var in ["TALUS_KERNELS_PREBUILT", "TALUS_KERNELS_BUILD_DIR"] {
            if let Ok(dir) = std::env::var(var) {
                if !dir.is_empty() {
                    let so = Path::new(&dir).join("libtalus_kernels.so");
                    if so.exists() {
                        candidate_paths.push(so.to_string_lossy().into_owned());
                    }
                }
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let so = dir.join("libtalus_kernels.so");
                if so.exists() {
                    candidate_paths.push(so.to_string_lossy().into_owned());
                }
            }
        }
        candidate_paths.push("libtalus_kernels.so".to_string());

        let mut last_err = String::from("no candidate paths tried");
        let mut loaded_lib: Option<(Arc<LibTalusKernels>, String)> = None;
        for path in candidate_paths {
            match LibTalusKernels::load(&path) {
                Ok(l) => {
                    loaded_lib = Some((Arc::new(l), path));
                    break;
                }
                Err(e) => {
                    last_err = e;
                }
            }
        }
        let (lib, chosen_path) = loaded_lib.ok_or(last_err)?;

        #[cfg(unix)]
        let (set_w32, set_w32_waves, w32_config, w32_active, w4i_pack, w4i_gemm, wide) = unsafe {
            let c_path = CString::new(chosen_path).ok();
            let handle = c_path
                .as_ref()
                .map(|c| dlopen(c.as_ptr(), RTLD_NOW | RTLD_LOCAL))
                .unwrap_or(ptr::null_mut());
            let resolve = |sym: &[u8]| -> *mut c_void {
                if handle.is_null() {
                    ptr::null_mut()
                } else {
                    dlsym(handle, sym.as_ptr().cast::<c_char>())
                }
            };
            let p1 = resolve(b"_ZN6strata7kernels19native_mmvq_set_w32Eii\0");
            let p2 = resolve(b"_ZN6strata7kernels25native_mmvq_set_w32_wavesEi\0");
            let p3 = resolve(b"_ZN6strata7kernels22native_mmvq_w32_configEiiPiS1_\0");
            let p4 = resolve(b"_ZN6strata7kernels22native_mmvq_w32_activeEv\0");
            let p5 = resolve(b"talus_pf_w4i_pack\0");
            let p6 = resolve(b"talus_pf_w4i_gemm\0");
            let pw = [
                resolve(b"talus_mmvq_wide_active\0"),
                resolve(b"talus_mmvq_wide_pack_mode\0"),
                resolve(b"talus_mmvq_wide_note\0"),
                resolve(b"talus_mmvq_wide_noted_bytes\0"),
                resolve(b"talus_mmvq_wide_pack_noted\0"),
                resolve(b"talus_mmvq_wide_release\0"),
                resolve(b"talus_mmvq_wide_packed_bytes\0"),
            ];
            (
                if p1.is_null() {
                    None
                } else {
                    Some(std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(c_int, c_int),
                    >(p1))
                },
                if p2.is_null() {
                    None
                } else {
                    Some(std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(c_int),
                    >(p2))
                },
                if p3.is_null() {
                    None
                } else {
                    Some(std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(c_int, c_int, *mut c_int, *mut c_int),
                    >(p3))
                },
                if p4.is_null() {
                    None
                } else {
                    Some(std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn() -> bool,
                    >(p4))
                },
                if p5.is_null() {
                    None
                } else {
                    Some(std::mem::transmute::<*mut c_void, W4iPackFn>(p5))
                },
                if p6.is_null() {
                    None
                } else {
                    Some(std::mem::transmute::<*mut c_void, W4iGemmFn>(p6))
                },
                if pw.iter().any(|p| p.is_null()) {
                    None
                } else {
                    Some(WideFns {
                        active: std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> c_int>(
                            pw[0],
                        ),
                        pack_mode: std::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn() -> c_int,
                        >(pw[1]),
                        note: std::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(c_int, *const c_void, c_int, c_int) -> c_int,
                        >(pw[2]),
                        noted_bytes: std::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn() -> u64,
                        >(pw[3]),
                        pack_noted: std::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn() -> c_int,
                        >(pw[4]),
                        release: std::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*const c_void),
                        >(pw[5]),
                        packed_bytes: std::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn() -> u64,
                        >(pw[6]),
                    })
                },
            )
        };
        #[cfg(not(unix))]
        let (set_w32, set_w32_waves, w32_config, w32_active, w4i_pack, w4i_gemm, wide) =
            (None, None, None, None, None, None, None);

        Ok(LoadedKernels {
            lib,
            set_w32,
            set_w32_waves,
            w32_config,
            w32_active,
            w4i_pack,
            w4i_gemm,
            wide,
        })
    })
}

fn wide_fns() -> Option<WideFns> {
    load_kernels_once().as_ref().ok().and_then(|k| k.wide)
}

/// the wide verify tiles are on (gfx1151 default; TALUS_MMVQ_WIDE=0 is the kill switch).
pub fn mmvq_wide_active() -> bool {
    wide_fns().is_some_and(|f| unsafe { (f.active)() } != 0)
}

/// TALUS_MMVQ_WIDE_PACK as the library read it: -1 unset (auto), 0 never, 1 force.
pub fn mmvq_wide_pack_mode() -> i32 {
    wide_fns().map_or(0, |f| unsafe { (f.pack_mode)() })
}

/// note a 2-D weight matrix as a tile-plane copy candidate (load time; nothing is allocated).
pub fn mmvq_wide_note(ggml_type: i32, w: *const c_void, n_in: i32, n_out: i32) -> bool {
    wide_fns().is_some_and(|f| unsafe { (f.note)(ggml_type, w, n_in, n_out) } != 0)
}

/// bytes the noted copies would take.
pub fn mmvq_wide_noted_bytes() -> usize {
    wide_fns().map_or(0, |f| unsafe { (f.noted_bytes)() } as usize)
}

/// make every noted copy (allocates and synchronizes: before graph capture); returns how many.
pub fn mmvq_wide_pack_noted() -> i32 {
    wide_fns().map_or(0, |f| unsafe { (f.pack_noted)() })
}

/// bytes held by the copies.
pub fn mmvq_wide_packed_bytes() -> usize {
    wide_fns().map_or(0, |f| unsafe { (f.packed_bytes)() } as usize)
}

/// forget `w` (free its copy; no-op when there is none).
pub fn mmvq_wide_release(w: *const c_void) {
    if let Some(f) = wide_fns() {
        unsafe { (f.release)(w) };
    }
}

/// true when libtalus_kernels exports the W4A4 (tp_w4i) entry points.
pub fn talus_pf_w4i_available() -> bool {
    matches!(load_kernels_once(), Ok(k) if k.w4i_pack.is_some() && k.w4i_gemm.is_some())
}

/// pack FP32 activations (gate null) or silu(gate) * x into the tp_w4i layout.  false: not available / shape.
#[allow(clippy::too_many_arguments)]
pub fn talus_pf_w4i_pack(
    x: *const f32,
    gate: *const f32,
    t: i32,
    k: i32,
    ldx: i64,
    signs: *const i8,
    wg: i32,
    xq: *mut u8,
    xk: *mut u8,
    xs: *mut f32,
    stream: *mut c_void,
) -> bool {
    match load_kernels_once() {
        Ok(LoadedKernels {
            w4i_pack: Some(f), ..
        }) => unsafe { f(x, gate, t, k, ldx, signs, wg, xq, xk, xs, stream) == 0 },
        _ => false,
    }
}

/// dst [T][ld] = the W4A4 product of packed activations with a tp_w4i weight matrix.
#[allow(clippy::too_many_arguments)]
pub fn talus_pf_w4i_gemm(
    wq: *const u8,
    ws: *const f32,
    wg: i32,
    n: i32,
    k: i32,
    xq: *const u8,
    xk: *const u8,
    xs: *const f32,
    t: i32,
    dst: *mut f32,
    ld: i64,
    stream: *mut c_void,
) -> bool {
    match load_kernels_once() {
        Ok(LoadedKernels {
            w4i_gemm: Some(f), ..
        }) => unsafe { f(wq, ws, wg, n, k, xq, xk, xs, t, dst, ld, stream) == 0 },
        _ => false,
    }
}

#[inline]
fn lib() -> &'static LibTalusKernels {
    &load_kernels_once()
        .as_ref()
        .expect("failed to load libtalus_kernels.so")
        .lib
}

#[inline]
pub fn kernels_available() -> bool {
    load_kernels_once().is_ok()
}

/// Owned wrapper around `tk_ctx*` and `LibTalusKernels`.
pub struct KernelsContext {
    lib: Arc<LibTalusKernels>,
    ctx: *mut tk_ctx,
}

unsafe impl Send for KernelsContext {}
unsafe impl Sync for KernelsContext {}

impl std::fmt::Debug for KernelsContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelsContext")
            .field("ctx", &self.ctx)
            .finish()
    }
}

impl KernelsContext {
    pub fn new() -> Result<Self> {
        let loaded = load_kernels_once()
            .as_ref()
            .map_err(|e| CoreError::KernelError {
                code: -1,
                message: format!("failed to load libtalus_kernels.so: {e}"),
            })?;
        let mut cfg: *mut tk_config = ptr::null_mut();
        let st_cfg = unsafe { (loaded.lib.tk_config_create)(&mut cfg) };
        if st_cfg != TK_OK {
            bail!("tk_config_create failed with status {st_cfg}");
        }
        let _ = unsafe { (loaded.lib.tk_config_from_env)(cfg) };
        let mut ctx: *mut tk_ctx = ptr::null_mut();
        let st = unsafe { (loaded.lib.tk_ctx_create)(0, cfg, &mut ctx) };
        unsafe {
            (loaded.lib.tk_config_destroy)(cfg);
        }
        if st != TK_OK || ctx.is_null() {
            let msg = unsafe {
                let p = (loaded.lib.tk_last_error)(ptr::null());
                if p.is_null() {
                    "tk_ctx_create failed".to_string()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            return Err(CoreError::KernelError {
                code: st,
                message: msg,
            });
        }
        Ok(Self {
            lib: Arc::clone(&loaded.lib),
            ctx,
        })
    }

    #[inline]
    pub fn from_env() -> Result<Self> {
        Self::new()
    }

    #[inline]
    pub fn as_ptr(&self) -> *mut tk_ctx {
        self.ctx
    }
}

impl Drop for KernelsContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                (self.lib.tk_ctx_destroy)(self.ctx);
            }
            self.ctx = ptr::null_mut();
        }
    }
}

// ============================================================================
// Host conversion & W32 control helpers
// ============================================================================

#[inline]
pub fn tk_f16_to_f32_row(src: *const u16, dst: *mut f32, n: i64) -> tk_status {
    if src.is_null() || dst.is_null() || n < 0 {
        return talus_kernels_sys::TK_ERR_NULL_PTR;
    }
    for i in 0..(n as usize) {
        unsafe {
            *dst.add(i) = talus_kernels::quant::f16_to_f32(*src.add(i));
        }
    }
    TK_OK
}

#[inline]
pub fn tk_bf16_to_f32_row(src: *const u16, dst: *mut f32, n: i64) -> tk_status {
    if src.is_null() || dst.is_null() || n < 0 {
        return talus_kernels_sys::TK_ERR_NULL_PTR;
    }
    for i in 0..(n as usize) {
        unsafe {
            *dst.add(i) = talus_kernels::quant::bf16_to_f32(*src.add(i));
        }
    }
    TK_OK
}

#[inline]
pub fn set_mmvq_w32(mode: i32, rows: i32) {
    if let Ok(k) = load_kernels_once() {
        if let Some(f) = k.set_w32 {
            unsafe { f(mode, rows) };
        }
    }
}

#[inline]
pub fn set_mmvq_w32_waves(w: i32) {
    if let Ok(k) = load_kernels_once() {
        if let Some(f) = k.set_w32_waves {
            unsafe { f(w) };
        }
    }
}

#[inline]
pub fn get_mmvq_w32_config(qtype: i32, ncols: i32) -> (i32, i32) {
    let mut r = 0i32;
    let mut w = 0i32;
    if let Ok(k) = load_kernels_once() {
        if let Some(f) = k.w32_config {
            unsafe { f(qtype, ncols, &mut r, &mut w) };
        }
    }
    (r, w)
}

#[inline]
pub unsafe fn tk_mmvq_w32_active() -> c_int {
    if let Ok(k) = load_kernels_once() {
        if let Some(f) = k.w32_active {
            return if unsafe { f() } { 1 } else { 0 };
        }
    }
    // Fallback: if Q8_0 packing is suppressed on an eligible shape, w32 is active
    if !tk_q36_q8_0_pack_should(8, 128, 2) && tk_q36_q8_0_pack_eligible(128, 2) {
        1
    } else {
        0
    }
}

// ============================================================================
// Dispatched `tk_*` & `tk_q36_*` wrappers
// ============================================================================

#[inline]
pub fn tk_mmvq_set_w32(_ctx: *mut tk_ctx, enabled: bool) -> tk_status {
    set_mmvq_w32(if enabled { 1 } else { 0 }, 0);
    TK_OK
}

#[inline]
pub fn tk_mmvq_set_w32_waves(_ctx: *mut tk_ctx, waves: i32) -> tk_status {
    set_mmvq_w32_waves(waves);
    TK_OK
}

#[inline]
pub fn tk_q8_1_bytes(n_in: i32, ncols: i32) -> usize {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q8_1_bytes)(n_in, ncols) }
    } else {
        talus_kernels::q8_1_bytes(n_in.max(0) as usize, ncols.max(0) as usize)
    }
}

#[inline]
pub fn tk_quantize_q8_1(
    ctx: *mut tk_ctx,
    x: *const f32,
    x_q8_1: *mut c_void,
    n_in: i32,
    ncols: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_quantize_q8_1)(ctx, x, x_q8_1, n_in, ncols, stream) }
}

#[inline]
pub fn tk_swiglu_quantize_q8_1(
    ctx: *mut tk_ctx,
    gate: *const f32,
    up: *const f32,
    x_q8_1: *mut c_void,
    n_in: i32,
    ncols: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_swiglu_quantize_q8_1)(ctx, gate, up, x_q8_1, n_in, ncols, stream) }
}

#[inline]
pub fn tk_mmvq(
    ctx: *mut tk_ctx,
    qtype: tk_qtype,
    w: *const c_void,
    x_q8_1: *const c_void,
    y: *mut f32,
    n_in: i32,
    n_out: i32,
    ncols: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_mmvq)(ctx, qtype, w, x_q8_1, y, n_in, n_out, ncols, stream) }
}

#[inline]
pub fn tk_mmvq_w32_multi(
    ctx: *mut tk_ctx,
    qtype: tk_qtype,
    n_mats: i32,
    w_ptrs: *const *const c_void,
    y_ptrs: *const *mut f32,
    n_out_per_mat: *const i32,
    x_q8_1: *const c_void,
    n_in: i32,
    ncols: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_mmvq_w32_multi)(
            ctx,
            qtype,
            n_mats,
            w_ptrs,
            y_ptrs,
            n_out_per_mat,
            x_q8_1,
            n_in,
            ncols,
            stream,
        )
    }
}

#[inline]
pub fn tk_mmvq_batch(
    ctx: *mut tk_ctx,
    items: *const tk_mmvq_item,
    n_items: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_mmvq_batch)(ctx, items, n_items, stream) }
}

#[inline]
pub fn tk_prefill_gemm(
    ctx: *mut tk_ctx,
    w_qtype: tk_qtype,
    w: *const c_void,
    x: *const c_void,
    x_qtype: tk_qtype,
    y: *mut f32,
    n_tok: i64,
    n_out: i64,
    n_in: i64,
    ldy: i64,
    ldx: i64,
    beta: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_prefill_gemm)(
            ctx, w_qtype, w, x, x_qtype, y, n_tok, n_out, n_in, ldy, ldx, beta, stream,
        )
    }
}

#[inline]
pub fn tk_prefill_mmq(
    ctx: *mut tk_ctx,
    w_qtype: tk_qtype,
    w_blocks: *const c_void,
    x_q8_1: *const c_void,
    y: *mut f32,
    n_tok: i64,
    n_out: i64,
    n_in: i64,
    ldy: i64,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_prefill_mmq)(
            ctx, w_qtype, w_blocks, x_q8_1, y, n_tok, n_out, n_in, ldy, stream,
        )
    }
}

#[inline]
pub fn tk_prefill_moe_mmq(
    ctx: *mut tk_ctx,
    product: *const tk_moe_mmq_product,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_prefill_moe_mmq)(ctx, product, stream) }
}

#[inline]
pub fn tk_attn_prefill_fa(
    ctx: *mut tk_ctx,
    q: *const f32,
    k_pool: *const c_void,
    v_pool: *const c_void,
    out: *mut f32,
    n_tok: i32,
    n_head: i32,
    n_kv: i32,
    head_dim: i32,
    kv_stride: i64,
    p0: i64,
    scale: f32,
    mask: *mut c_void,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_attn_prefill_fa)(
            ctx, q, k_pool, v_pool, out, n_tok, n_head, n_kv, head_dim, kv_stride, p0, scale, mask,
            stream,
        )
    }
}

#[inline]
pub fn tk_q36_rms(
    ctx: *mut tk_ctx,
    rows: i32,
    x: *const f32,
    w: *const f32,
    y: *mut f32,
    n: i32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    let st = unsafe { (lib().tk_q36_rms)(ctx, rows, x, w, y, n, eps, stream) };
    if st == talus_kernels_sys::TK_ERR_HIP && hip_sys::HipRuntime::global().is_simulated() {
        if rows > 0 && n > 0 && !x.is_null() && !w.is_null() && !y.is_null() {
            unsafe {
                let ws = std::slice::from_raw_parts(w, n as usize);
                for r in 0..(rows as usize) {
                    let xr = std::slice::from_raw_parts(x.add(r * n as usize), n as usize);
                    let yr = std::slice::from_raw_parts_mut(y.add(r * n as usize), n as usize);
                    let sum_sq: f32 = xr.iter().map(|v| v * v).sum();
                    let scale = 1.0f32 / (sum_sq / (n as f32) + eps).sqrt();
                    for i in 0..(n as usize) {
                        yr[i] = xr[i] * scale * ws[i];
                    }
                }
            }
            return TK_OK;
        }
    }
    st
}

#[inline]
pub fn tk_q36_rms_q8(
    ctx: *mut tk_ctx,
    rows: i32,
    x: *const f32,
    w: *const f32,
    y: *mut f32,
    q8: *mut c_void,
    n: i32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_rms_q8)(ctx, rows, x, w, y, q8, n, eps, stream) }
}

#[inline]
pub fn tk_q36_rms_ld(
    ctx: *mut tk_ctx,
    rows: i32,
    x: *const f32,
    ldx: i32,
    w: *const f32,
    out: *mut f32,
    ld_out: i32,
    n: i32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_rms_ld)(ctx, rows, x, ldx, w, out, ld_out, n, eps, stream) }
}

#[inline]
pub fn tk_q36_rms_ld_q8(
    ctx: *mut tk_ctx,
    rows: i32,
    x: *const f32,
    ldx: i32,
    w: *const f32,
    out: *mut f32,
    ld_out: i32,
    q8: *mut c_void,
    n: i32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_rms_ld_q8)(ctx, rows, x, ldx, w, out, ld_out, q8, n, eps, stream) }
}

#[inline]
pub fn tk_q36_add_rms(
    ctx: *mut tk_ctx,
    rows: i32,
    x: *mut f32,
    y: *const f32,
    w: *const f32,
    h: *mut f32,
    n: i32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_add_rms)(ctx, rows, x, y, w, h, n, eps, stream) }
}

#[inline]
pub fn tk_q36_add_rms_q8(
    ctx: *mut tk_ctx,
    rows: i32,
    x: *mut f32,
    y: *const f32,
    w: *const f32,
    h: *mut f32,
    q8: *mut c_void,
    n: i32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_add_rms_q8)(ctx, rows, x, y, w, h, q8, n, eps, stream) }
}

#[inline]
pub fn tk_q36_add_rms_exact_q8(
    ctx: *mut tk_ctx,
    rows: i32,
    x: *mut f32,
    y: *const f32,
    w: *const f32,
    h: *mut f32,
    q8: *mut c_void,
    n: i32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_add_rms_exact_q8)(ctx, rows, x, y, w, h, q8, n, eps, stream) }
}

#[inline]
pub fn tk_q36_moe_out(
    ctx: *mut tk_ctx,
    rows: i32,
    x: *mut f32,
    parts: *const f32,
    prow: *const i32,
    wts: *const f32,
    k_used: i32,
    shared: *const f32,
    sgate: *const f32,
    sg_stride: i32,
    wn: *const f32,
    h: *mut f32,
    n: i32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_moe_out)(
            ctx, rows, x, parts, prow, wts, k_used, shared, sgate, sg_stride, wn, h, n, eps, stream,
        )
    }
}

#[inline]
pub fn tk_q36_moe_out_q8(
    ctx: *mut tk_ctx,
    rows: i32,
    x: *mut f32,
    parts: *const f32,
    prow: *const i32,
    wts: *const f32,
    k_used: i32,
    shared: *const f32,
    sgate: *const f32,
    sg_stride: i32,
    wn: *const f32,
    h: *mut f32,
    q8: *mut c_void,
    n: i32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_moe_out_q8)(
            ctx, rows, x, parts, prow, wts, k_used, shared, sgate, sg_stride, wn, h, q8, n, eps,
            stream,
        )
    }
}

#[inline]
pub fn tk_q36_gdn_ab(
    ctx: *mut tk_ctx,
    t_rows: i32,
    wa: *const f32,
    wb: *const f32,
    x: *const f32,
    dt: *const f32,
    ssm_a: *const f32,
    alpha: *mut f32,
    beta: *mut f32,
    n_in: i32,
    hv: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_gdn_ab)(
            ctx, t_rows, wa, wb, x, dt, ssm_a, alpha, beta, n_in, hv, stream,
        )
    }
}

#[inline]
pub fn tk_q36_gdn_ab2(
    ctx: *mut tk_ctx,
    wa: *const f32,
    wb: *const f32,
    x: *const f32,
    dt: *const f32,
    ssm_a: *const f32,
    alpha: *mut f32,
    beta: *mut f32,
    n_in: i32,
    hv: i32,
    t_rows: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_gdn_ab2)(
            ctx, wa, wb, x, dt, ssm_a, alpha, beta, n_in, hv, t_rows, stream,
        )
    }
}

#[inline]
pub fn tk_q36_add(
    ctx: *mut tk_ctx,
    x: *mut f32,
    y: *const f32,
    n: i64,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_add)(ctx, x, y, n, stream) }
}

#[inline]
pub fn tk_q36_swiglu(
    ctx: *mut tk_ctx,
    g: *const f32,
    u: *const f32,
    h: *mut f32,
    n: i64,
    stream: tk_stream,
) -> tk_status {
    let st = unsafe { (lib().tk_q36_swiglu)(ctx, g, u, h, n, stream) };
    if st == talus_kernels_sys::TK_ERR_HIP && hip_sys::HipRuntime::global().is_simulated() {
        if n > 0 && !g.is_null() && !u.is_null() && !h.is_null() {
            unsafe {
                let gs = std::slice::from_raw_parts(g, n as usize);
                let us = std::slice::from_raw_parts(u, n as usize);
                let hs = std::slice::from_raw_parts_mut(h, n as usize);
                for i in 0..(n as usize) {
                    let gv = gs[i];
                    hs[i] = (gv / (1.0f32 + (-gv).exp())) * us[i];
                }
            }
            return TK_OK;
        }
    }
    st
}

#[inline]
pub fn tk_q36_gemv_f32(
    ctx: *mut tk_ctx,
    t_rows: i32,
    w: *const f32,
    x: *const f32,
    y: *mut f32,
    n_in: i32,
    n_out: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_gemv_f32)(ctx, t_rows, w, x, y, n_in, n_out, stream) }
}

#[inline]
pub fn tk_q36_gdn_gb(
    ctx: *mut tk_ctx,
    alpha: *mut f32,
    dt: *const f32,
    ssm_a: *const f32,
    beta: *mut f32,
    hv: i32,
    t_rows: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_gdn_gb)(ctx, alpha, dt, ssm_a, beta, hv, t_rows, stream) }
}

#[inline]
pub fn tk_q36_gdn_gb2(
    ctx: *mut tk_ctx,
    ab: *const f32,
    dt: *const f32,
    ssm_a: *const f32,
    alpha: *mut f32,
    beta: *mut f32,
    hv: i32,
    t_rows: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_gdn_gb2)(ctx, ab, dt, ssm_a, alpha, beta, hv, t_rows, stream) }
}

#[inline]
pub fn tk_q36_gate_rows(
    ctx: *mut tk_ctx,
    sh: *mut f32,
    gate: *const f32,
    n: i32,
    t_rows: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_gate_rows)(ctx, sh, gate, n, t_rows, stream) }
}

#[inline]
pub fn tk_q36_attn_prep(
    ctx: *mut tk_ctx,
    t_rows: i32,
    qg: *const f32,
    kx: *const f32,
    vx: *const f32,
    qw: *const f32,
    kw: *const f32,
    q: *mut f32,
    gate: *mut f32,
    kc_f16: *mut c_void,
    vc_f16: *mut c_void,
    h_q: i32,
    h_kv: i32,
    hd: i32,
    n_rot: i32,
    pos0: *const i32,
    ctx_cap: i32,
    rope_base: f32,
    eps: f32,
    vt_f16: *mut c_void,
    vt_ctx: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_attn_prep)(
            ctx, t_rows, qg, kx, vx, qw, kw, q, gate, kc_f16, vc_f16, h_q, h_kv, hd, n_rot, pos0,
            ctx_cap, rope_base, eps, vt_f16, vt_ctx, stream,
        )
    }
}

#[inline]
pub fn tk_q36_attn_part(
    ctx: *mut tk_ctx,
    grid_x: i32,
    grid_y: i32,
    q: *const f32,
    kc_f16: *const c_void,
    vc_f16: *const c_void,
    pos0: *const i32,
    t0: i32,
    h_q: i32,
    h_kv: i32,
    ctx_cap: i32,
    scale: f32,
    n_chunk: i32,
    pacc: *mut f32,
    pm: *mut f32,
    pl: *mut f32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_attn_part)(
            ctx, grid_x, grid_y, q, kc_f16, vc_f16, pos0, t0, h_q, h_kv, ctx_cap, scale, n_chunk,
            pacc, pm, pl, stream,
        )
    }
}

#[inline]
pub fn tk_q36_attn_merge(
    ctx: *mut tk_ctx,
    rows: i32,
    hd: i32,
    pacc: *const f32,
    pm: *const f32,
    pl: *const f32,
    gate: *const f32,
    pos0: *const i32,
    t0: i32,
    h_q: i32,
    n_chunk: i32,
    out: *mut f32,
    chunk: i32,
    q8: *mut c_void,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_attn_merge)(
            ctx, rows, hd, pacc, pm, pl, gate, pos0, t0, h_q, n_chunk, out, chunk, q8, stream,
        )
    }
}

#[inline]
pub fn tk_q36_attn_wmma(
    ctx: *mut tk_ctx,
    w8: bool,
    grid_x: i32,
    grid_y: i32,
    q: *const f32,
    kc_f16: *const c_void,
    vt_f16: *const c_void,
    pos0: *const i32,
    t_rows: i32,
    h_q: i32,
    h_kv: i32,
    ctx_cap: i32,
    vt_ctx: i32,
    scale: f32,
    chunk: i32,
    n_chunk: i32,
    pacc: *mut f32,
    pm: *mut f32,
    pl: *mut f32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_attn_wmma)(
            ctx, w8, grid_x, grid_y, q, kc_f16, vt_f16, pos0, t_rows, h_q, h_kv, ctx_cap, vt_ctx,
            scale, chunk, n_chunk, pacc, pm, pl, stream,
        )
    }
}

#[inline]
pub fn tk_q36_group(
    ctx: *mut tk_ctx,
    ids: *const i32,
    n_ent: i32,
    k_used: i32,
    n_expert: i32,
    blobs: *const u8,
    blob_stride: u64,
    grp_ptr: *mut u64,
    grp_start: *mut i32,
    n_groups: *mut i32,
    ent_dst: *mut i32,
    ent_tok: *mut i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_group)(
            ctx,
            ids,
            n_ent,
            k_used,
            n_expert,
            blobs,
            blob_stride,
            grp_ptr,
            grp_start,
            n_groups,
            ent_dst,
            ent_tok,
            stream,
        )
    }
}

#[inline]
pub fn tk_q36_group_hy(
    ctx: *mut tk_ctx,
    ids: *const i32,
    n_ent: i32,
    k_used: i32,
    n_expert: i32,
    slot_of: *const i32,
    blobs: *const u8,
    blob_stride: u64,
    grp_ptr: *mut u64,
    grp_start: *mut i32,
    n_groups: *mut i32,
    ent_dst: *mut i32,
    ent_tok: *mut i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_group_hy)(
            ctx,
            ids,
            n_ent,
            k_used,
            n_expert,
            slot_of,
            blobs,
            blob_stride,
            grp_ptr,
            grp_start,
            n_groups,
            ent_dst,
            ent_tok,
            stream,
        )
    }
}

#[inline]
pub fn tk_q36_merge_cpu(
    ctx: *mut tk_ctx,
    n_ent: i32,
    parts: *mut f32,
    cpu: *const f32,
    miss: *const i32,
    n: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_merge_cpu)(ctx, n_ent, parts, cpu, miss, n, stream) }
}

#[inline]
pub fn tk_q36_attn_softmax16(
    ctx: *mut tk_ctx,
    h_q: i32,
    s: *const f32,
    s16_f16: *mut c_void,
    ld: i32,
    t_rows: i32,
    p_cols: i32,
    pos0: i32,
    scale: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_attn_softmax16)(
            ctx, h_q, s, s16_f16, ld, t_rows, p_cols, pos0, scale, stream,
        )
    }
}

#[inline]
pub fn tk_q36_gate_mul(
    ctx: *mut tk_ctx,
    att: *mut f32,
    gate: *const f32,
    n: i64,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_gate_mul)(ctx, att, gate, n, stream) }
}

#[inline]
pub fn tk_q36_route_count(
    ctx: *mut tk_ctx,
    ids: *const i32,
    n: i32,
    counts: *mut i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_route_count)(ctx, ids, n, counts, stream) }
}

#[inline]
pub fn tk_q36_route_scan(
    ctx: *mut tk_ctx,
    cnt: *const i32,
    n_expert: i32,
    bounds: *mut i32,
    cur: *mut i32,
    max_rows: *mut i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_route_scan)(ctx, cnt, n_expert, bounds, cur, max_rows, stream) }
}

#[inline]
pub fn tk_q36_route_scatter(
    ctx: *mut tk_ctx,
    ids: *const i32,
    n_tok: i32,
    k_used: i32,
    cursors: *mut i32,
    tok_ids: *mut i32,
    exp_ids: *mut i32,
    prow: *mut i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_route_scatter)(
            ctx, ids, n_tok, k_used, cursors, tok_ids, exp_ids, prow, stream,
        )
    }
}

#[inline]
pub fn tk_q36_argmax(
    ctx: *mut tk_ctx,
    grid_x: i32,
    grid_y: i32,
    x: *const f32,
    n: i32,
    out: *mut i32,
    pv: *mut f32,
    pi: *mut i32,
    stage2: bool,
    map: *const i32,
    stream: tk_stream,
) -> tk_status {
    let st = unsafe {
        (lib().tk_q36_argmax)(ctx, grid_x, grid_y, x, n, out, pv, pi, stage2, map, stream)
    };
    if st == talus_kernels_sys::TK_ERR_HIP && hip_sys::HipRuntime::global().is_simulated() {
        let rows = grid_y.max(1) as usize;
        if n > 0 && !pv.is_null() && !pi.is_null() {
            unsafe {
                if !stage2 && !x.is_null() {
                    for r in 0..rows {
                        let xr = std::slice::from_raw_parts(x.add(r * n as usize), n as usize);
                        let mut best_v = f32::NEG_INFINITY;
                        let mut best_i = 0i32;
                        for (i, &v) in xr.iter().enumerate() {
                            if v > best_v {
                                best_v = v;
                                best_i = i as i32;
                            }
                        }
                        *pv.add(r * 16) = best_v;
                        *pi.add(r * 16) = best_i;
                    }
                    return TK_OK;
                } else if stage2 && !out.is_null() {
                    let count = (n as usize).min(16);
                    for r in 0..rows {
                        let mut best_v = f32::NEG_INFINITY;
                        let mut best_i = 0i32;
                        for c in 0..count {
                            let v = *pv.add(r * 16 + c);
                            let idx = *pi.add(r * 16 + c);
                            if v > best_v {
                                best_v = v;
                                best_i = idx;
                            }
                        }
                        if !map.is_null() && best_i >= 0 {
                            best_i = *map.add(best_i as usize);
                        }
                        *out.add(r) = best_i;
                    }
                    return TK_OK;
                }
            }
        }
    }
    st
}

#[inline]
pub fn tk_q36_argmax_samp(
    ctx: *mut tk_ctx,
    x: *const f32,
    n: i32,
    pv: *mut f32,
    pi: *mut i32,
    ds_dev: *const tk_q36_draft_samp,
    step: i32,
    thr: *const f32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_argmax_samp)(
            ctx,
            x,
            n,
            pv,
            pi,
            ds_dev.cast::<talus_kernels_sys::tk_q36_draft_samp>(),
            step,
            thr,
            stream,
        )
    }
}

#[inline]
pub fn tk_q36_lse(
    ctx: *mut tk_ctx,
    x: *const f32,
    n: i32,
    ds_dev: *const tk_q36_draft_samp,
    out: *mut f32,
    thr_out: *mut f32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_lse)(
            ctx,
            x,
            n,
            ds_dev.cast::<talus_kernels_sys::tk_q36_draft_samp>(),
            out,
            thr_out,
            stream,
        )
    }
}

#[inline]
pub fn tk_q36_ctl_end(ctx: *mut tk_ctx, ctl: *mut i32, flip: bool, stream: tk_stream) -> tk_status {
    unsafe { (lib().tk_q36_ctl_end)(ctx, ctl, flip, stream) }
}

#[inline]
pub fn tk_q36_round_prep(
    ctx: *mut tk_ctx,
    rctl: *const i32,
    tok: *mut i32,
    tok_m: *mut i32,
    pos0: *mut i32,
    pos0_m: *mut i32,
    gctl: *mut i32,
    n: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_round_prep)(ctx, rctl, tok, tok_m, pos0, pos0_m, gctl, n, stream) }
}

#[inline]
pub fn tk_q36_row_copy(
    ctx: *mut tk_ctx,
    dst: *mut f32,
    src: *const f32,
    idx: *const i32,
    n: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_row_copy)(ctx, dst, src, idx, n, stream) }
}

#[inline]
pub fn tk_q36_chain_prep(
    ctx: *mut tk_ctx,
    tok: *const i32,
    tok_m: *mut i32,
    pos0: *const i32,
    pos0_m: *mut i32,
    k: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_chain_prep)(ctx, tok, tok_m, pos0, pos0_m, k, stream) }
}

#[inline]
pub fn tk_q36_sgemm_nt(
    ctx: *mut tk_ctx,
    tm: i32,
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    w: *const f32,
    x: *const f32,
    out: *mut f32,
    k_dim: i32,
    n_out: i32,
    n_tok: i32,
    ldo: i32,
    kz: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_sgemm_nt)(
            ctx, tm, grid_x, grid_y, grid_z, w, x, out, k_dim, n_out, n_tok, ldo, kz, stream,
        )
    }
}

#[inline]
pub fn tk_q36_sum_splits(
    ctx: *mut tk_ctx,
    part: *const f32,
    out: *mut f32,
    n: i64,
    splits: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_sum_splits)(ctx, part, out, n, splits, stream) }
}

#[inline]
pub fn tk_q36_to_f16(
    ctx: *mut tk_ctx,
    x: *const f32,
    y_f16: *mut c_void,
    n: i64,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_to_f16)(ctx, x, y_f16, n, stream) }
}

#[inline]
pub fn tk_q36_conv_seq(
    ctx: *mut tk_ctx,
    hist: *mut f32,
    qkv: *const f32,
    w: *const f32,
    h: *mut f32,
    c_channels: i32,
    qk_heads: i32,
    eps: f32,
    n_tok: i32,
    write_state: bool,
    pd: *const tk_q36_gdn_pend,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_conv_seq)(
            ctx,
            hist,
            qkv,
            w,
            h,
            c_channels,
            qk_heads,
            eps,
            n_tok,
            write_state,
            pd.cast::<talus_kernels_sys::tk_q36_gdn_pend>(),
            stream,
        )
    }
}

#[inline]
pub fn tk_q36_conv_ab(
    ctx: *mut tk_ctx,
    hist: *mut f32,
    qkv: *const f32,
    w: *const f32,
    hc: *mut f32,
    c_channels: i32,
    qk_heads: i32,
    eps: f32,
    n_tok: i32,
    write_state: bool,
    pd: *const tk_q36_gdn_pend,
    wa: *const f32,
    wb: *const f32,
    x: *const f32,
    dt: *const f32,
    ssm_a: *const f32,
    alpha: *mut f32,
    beta: *mut f32,
    n_in: i32,
    hv: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_conv_ab)(
            ctx,
            hist,
            qkv,
            w,
            hc,
            c_channels,
            qk_heads,
            eps,
            n_tok,
            write_state,
            pd.cast::<talus_kernels_sys::tk_q36_gdn_pend>(),
            wa,
            wb,
            x,
            dt,
            ssm_a,
            alpha,
            beta,
            n_in,
            hv,
            stream,
        )
    }
}

#[inline]
pub fn tk_q36_conv_par(
    ctx: *mut tk_ctx,
    hist: *const f32,
    qkv: *const f32,
    w: *const f32,
    h: *mut f32,
    c_channels: i32,
    qk_heads: i32,
    eps: f32,
    n_tok: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_conv_par)(
            ctx, hist, qkv, w, h, c_channels, qk_heads, eps, n_tok, stream,
        )
    }
}

#[inline]
pub fn tk_q36_conv_hist(
    ctx: *mut tk_ctx,
    hist: *mut f32,
    qkv: *const f32,
    c_channels: i32,
    n_tok: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_conv_hist)(ctx, hist, qkv, c_channels, n_tok, stream) }
}

#[inline]
pub fn tk_q36_gdn_seq(
    ctx: *mut tk_ctx,
    state: *mut f32,
    h: *const f32,
    c_channels: i32,
    gate: *const f32,
    beta: *const f32,
    z: *const f32,
    gamma: *const f32,
    eps: f32,
    y: *mut f32,
    h_k: i32,
    h_v: i32,
    n_tok: i32,
    write_state: bool,
    pd: *const tk_q36_gdn_pend,
    yq: *mut c_void,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_gdn_seq)(
            ctx,
            state,
            h,
            c_channels,
            gate,
            beta,
            z,
            gamma,
            eps,
            y,
            h_k,
            h_v,
            n_tok,
            write_state,
            pd.cast::<talus_kernels_sys::tk_q36_gdn_pend>(),
            yq,
            stream,
        )
    }
}

#[inline]
pub fn tk_q36_gdn_seq_cols(
    ctx: *mut tk_ctx,
    state: *mut f32,
    h: *const f32,
    c_channels: i32,
    gate: *const f32,
    beta: *const f32,
    o_raw: *mut f32,
    h_k: i32,
    h_v: i32,
    n_tok: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_gdn_seq_cols)(
            ctx, state, h, c_channels, gate, beta, o_raw, h_k, h_v, n_tok, stream,
        )
    }
}

#[inline]
pub fn tk_q36_gdn_seq_warp(
    ctx: *mut tk_ctx,
    ncol: i32,
    state: *mut f32,
    h: *const f32,
    c_channels: i32,
    gate: *const f32,
    beta: *const f32,
    o_raw: *mut f32,
    h_k: i32,
    h_v: i32,
    n_tok: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_gdn_seq_warp)(
            ctx, ncol, state, h, c_channels, gate, beta, o_raw, h_k, h_v, n_tok, stream,
        )
    }
}

#[inline]
pub fn tk_q36_gdn_norm(
    ctx: *mut tk_ctx,
    n_tok: i32,
    o_raw: *const f32,
    z: *const f32,
    gamma: *const f32,
    eps: f32,
    y: *mut f32,
    h_v: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_gdn_norm)(ctx, n_tok, o_raw, z, gamma, eps, y, h_v, stream) }
}

#[inline]
pub fn tk_q36_flush_read(
    ctx: *mut tk_ctx,
    p: *const c_void,
    n: usize,
    out: *mut f32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_flush_read)(ctx, p, n, out, stream) }
}

#[inline]
pub fn tk_q36_draft_conf(
    ctx: *mut tk_ctx,
    x: *const f32,
    n: i32,
    tau: *const f32,
    lmin: *const f32,
    pick: *const i32,
    map: *const i32,
    conf: *mut f32,
    top: *mut f32,
    step: i32,
    lse: *const f32,
    thr: *const f32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_draft_conf)(
            ctx, x, n, tau, lmin, pick, map, conf, top, step, lse, thr, stream,
        )
    }
}

#[inline]
pub fn tk_q36_draft_support(
    ctx: *mut tk_ctx,
    x: *const f32,
    n: i32,
    tau: *const f32,
    lmin: *const f32,
    top_p: *const f32,
    top_k: *const i32,
    lse: *mut f32,
    thr: *mut f32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_draft_support)(ctx, x, n, tau, lmin, top_p, top_k, lse, thr, stream) }
}

#[inline]
pub fn tk_q36_dflash_attn_chunks(window: i32) -> i32 {
    unsafe { (lib().tk_q36_dflash_attn_chunks)(window) }
}

#[inline]
pub fn tk_q36_dflash_attn_scratch_floats(heads: i32, hd: i32, window: i32) -> usize {
    unsafe { (lib().tk_q36_dflash_attn_scratch_floats)(heads, hd, window) }
}

#[inline]
pub fn tk_q36_dflash_rms_rows(
    ctx: *mut tk_ctx,
    x: *const f32,
    w: *const f32,
    out: *mut f32,
    n: i32,
    rows: i32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_dflash_rms_rows)(ctx, x, w, out, n, rows, eps, stream) }
}

#[inline]
pub fn tk_q36_dflash_rms_rows_q8(
    ctx: *mut tk_ctx,
    x: *const f32,
    w: *const f32,
    out: *mut f32,
    q8: *mut c_void,
    n: i32,
    rows: i32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_dflash_rms_rows_q8)(ctx, x, w, out, q8, n, rows, eps, stream) }
}

#[inline]
pub fn tk_q36_dflash_sum5_rms(
    ctx: *mut tk_ctx,
    ys: *const *const f32,
    w: *const f32,
    out: *mut f32,
    n: i32,
    rows: i32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_dflash_sum5_rms)(ctx, ys, w, out, n, rows, eps, stream) }
}

#[inline]
pub fn tk_q36_dflash_sum5_rms_q8(
    ctx: *mut tk_ctx,
    ys: *const *const f32,
    w: *const f32,
    out: *mut f32,
    q8: *mut c_void,
    n: i32,
    rows: i32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_dflash_sum5_rms_q8)(ctx, ys, w, out, q8, n, rows, eps, stream) }
}

#[inline]
pub fn tk_q36_dflash_dyn_conv(
    ctx: *mut tk_ctx,
    x_in: *const f32,
    dyn_w: *const f32,
    base: *const f32,
    out: *mut f32,
    n: i32,
    rows: i32,
    group: i32,
    side: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_dflash_dyn_conv)(ctx, x_in, dyn_w, base, out, n, rows, group, side, stream)
    }
}

#[inline]
pub fn tk_q36_dflash_dyn_conv_q8(
    ctx: *mut tk_ctx,
    x_in: *const f32,
    dyn_w: *const f32,
    base: *const f32,
    out: *mut f32,
    q8: *mut c_void,
    n: i32,
    rows: i32,
    group: i32,
    side: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_dflash_dyn_conv_q8)(
            ctx, x_in, dyn_w, base, out, q8, n, rows, group, side, stream,
        )
    }
}

#[inline]
pub fn tk_q36_dflash_qk_norm_rope(
    ctx: *mut tk_ctx,
    x: *mut f32,
    w: *const f32,
    heads: i32,
    hd: i32,
    rows: i32,
    pos: *const i32,
    theta: f32,
    eps: f32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_dflash_qk_norm_rope)(ctx, x, w, heads, hd, rows, pos, theta, eps, stream)
    }
}

#[inline]
pub fn tk_q36_dflash_kv_store(
    ctx: *mut tk_ctx,
    k: *const f32,
    v: *const f32,
    kring: *mut f32,
    vring: *mut f32,
    width: i32,
    rows: i32,
    pos: *const i32,
    ring: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_dflash_kv_store)(ctx, k, v, kring, vring, width, rows, pos, ring, stream)
    }
}

#[inline]
pub fn tk_q36_dflash_kv_store_f16(
    ctx: *mut tk_ctx,
    k: *const f32,
    v: *const f32,
    kring_f16: *mut c_void,
    vring_f16: *mut c_void,
    k_width: i32,
    v_width: i32,
    rows: i32,
    pos: *const i32,
    ring: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_dflash_kv_store_f16)(
            ctx, k, v, kring_f16, vring_f16, k_width, v_width, rows, pos, ring, stream,
        )
    }
}

#[inline]
pub fn tk_q36_dflash_block_attn(
    ctx: *mut tk_ctx,
    q: *const f32,
    kring: *const f32,
    vring: *const f32,
    pos: *const i32,
    heads: i32,
    heads_kv: i32,
    hd: i32,
    ring: i32,
    window: i32,
    scale: f32,
    part: *mut f32,
    out: *mut f32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_dflash_block_attn)(
            ctx, q, kring, vring, pos, heads, heads_kv, hd, ring, window, scale, part, out, stream,
        )
    }
}

#[inline]
pub fn tk_q36_dflash_swiglu(
    ctx: *mut tk_ctx,
    g: *const f32,
    u: *const f32,
    out: *mut f32,
    n: i64,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_dflash_swiglu)(ctx, g, u, out, n, stream) }
}

#[inline]
pub fn tk_q36_dflash_swiglu_q8(
    ctx: *mut tk_ctx,
    g: *const f32,
    u: *const f32,
    out: *mut f32,
    q8: *mut c_void,
    n: i64,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_dflash_swiglu_q8)(ctx, g, u, out, q8, n, stream) }
}

#[inline]
pub fn tk_q36_dflash_add_inplace(
    ctx: *mut tk_ctx,
    x: *mut f32,
    y: *const f32,
    n: i64,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_dflash_add_inplace)(ctx, x, y, n, stream) }
}

#[inline]
pub fn tk_q36_dflash_block_tokens(
    ctx: *mut tk_ctx,
    anchor: *const i32,
    out: *mut i32,
    n: i32,
    mask: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_dflash_block_tokens)(ctx, anchor, out, n, mask, stream) }
}

#[inline]
pub fn tk_q36_dflash_set_i32(
    ctx: *mut tk_ctx,
    p: *mut i32,
    v: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_dflash_set_i32)(ctx, p, v, stream) }
}

#[inline]
pub fn tk_q36_dflash_top16(
    ctx: *mut tk_ctx,
    logits: *const f32,
    nv: i32,
    rows: i32,
    map: *const i32,
    cand: *mut i32,
    unary: *mut f32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_dflash_top16)(ctx, logits, nv, rows, map, cand, unary, stream) }
}

#[inline]
pub fn tk_q36_dflash_select_chain(
    ctx: *mut tk_ctx,
    cand: *const i32,
    unary: *const f32,
    gate: *const f32,
    pred: *const c_void,
    succ: *const c_void,
    anchor: *const i32,
    out: *mut i32,
    score: *mut f32,
    n: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_dflash_select_chain)(
            ctx, cand, unary, gate, pred, succ, anchor, out, score, n, stream,
        )
    }
}

#[inline]
pub fn tk_q36_dflash_select_chain_q(
    ctx: *mut tk_ctx,
    cand: *const i32,
    unary: *const f32,
    gate: *const f32,
    pred: *const c_void,
    succ: *const c_void,
    anchor: *const i32,
    out: *mut i32,
    score: *mut f32,
    q: *mut f32,
    tau: *const f32,
    seed: *const c_uint,
    ctr: *const c_uint,
    n: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_dflash_select_chain_q)(
            ctx, cand, unary, gate, pred, succ, anchor, out, score, q, tau, seed, ctr, n, stream,
        )
    }
}

#[inline]
pub fn tk_q36_mmvq_supported(ggml_type: i32) -> bool {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_mmvq_supported)(ggml_type) }
    } else {
        matches!(
            ggml_type,
            2 | 6 | 8 | 10 | 11 | 12 | 13 | 14 | 16 | 17 | 18 | 20 | 21 | 22 | 23 | 29 | 42
        )
    }
}

#[inline]
pub fn tk_q36_q8_0_pack_eligible(n_in: i64, n_out: i64) -> bool {
    n_in > 0 && (n_in % 128) == 0 && n_out > 0 && (n_out & 1) == 0
}

#[inline]
pub fn tk_q36_q8_0_pack_inplace(
    _ctx: *mut tk_ctx,
    d_weights: *mut c_void,
    n_in: i64,
    n_out: i64,
    _d_scratch: *mut c_void,
    scratch_bytes: usize,
    _stream: tk_stream,
) -> tk_status {
    if d_weights.is_null() || !tk_q36_q8_0_pack_eligible(n_in, n_out) {
        return talus_kernels_sys::TK_ERR_INVALID_ARG;
    }
    let bytes = ((n_in as usize) / 32) * (n_out as usize) * 34;
    if scratch_bytes < bytes {
        return talus_kernels_sys::TK_ERR_INVALID_ARG;
    }
    let mut host_in = vec![0u8; bytes];
    let mut host_out = vec![0u8; bytes];
    unsafe {
        if hipMemcpy(
            host_in.as_mut_ptr().cast::<c_void>(),
            d_weights,
            bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
        ) != hipSuccess
        {
            return talus_kernels_sys::TK_ERR_HIP;
        }
        tk_q36_q8_0_pack_host(
            host_in.as_ptr().cast::<c_void>(),
            host_out.as_mut_ptr().cast::<c_void>(),
            n_in as i32,
            n_out as i32,
        );
        if hipMemcpy(
            d_weights,
            host_out.as_ptr().cast::<c_void>(),
            bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
        ) != hipSuccess
        {
            return talus_kernels_sys::TK_ERR_HIP;
        }
    }
    TK_OK
}

#[inline]
pub fn tk_dequantize_q8_0_to_f16(
    src_q8_0: *const c_void,
    dst_f16: *mut u16,
    n_elems: usize,
) -> tk_status {
    if src_q8_0.is_null() || dst_f16.is_null() || !n_elems.is_multiple_of(32) {
        return talus_kernels_sys::TK_ERR_INVALID_ARG;
    }
    let n_blocks = n_elems / 32;
    let src = unsafe { std::slice::from_raw_parts(src_q8_0.cast::<u8>(), n_blocks * 34) };
    let dst = unsafe { std::slice::from_raw_parts_mut(dst_f16, n_elems) };
    for b in 0..n_blocks {
        let blk = &src[b * 34..(b + 1) * 34];
        let d = talus_kernels::quant::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let out_b = &mut dst[b * 32..(b + 1) * 32];
        for i in 0..32 {
            let q = blk[2 + i] as i8;
            out_b[i] = super::weights::fp32_to_fp16(d * (q as f32));
        }
    }
    TK_OK
}

#[inline]
pub fn tk_q36_q8_0_pack_should(ggml_type: i32, n_in: i32, n_out: i32) -> bool {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_q8_0_pack_should)(ggml_type, n_in, n_out) }
    } else {
        ggml_type == 8 && n_in > 0 && n_in % 128 == 0 && n_out > 0 && (n_out & 1) == 0
    }
}

#[inline]
pub fn tk_q36_q8_0_pack_host(gguf_blocks: *const c_void, out: *mut c_void, n_in: i32, n_out: i32) {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_q8_0_pack_host)(gguf_blocks, out, n_in, n_out) };
    }
}

#[inline]
pub fn tk_q36_q8_0_packed_register(
    gguf_weights: *const c_void,
    packed: *const c_void,
    n_in: i32,
    n_out: i32,
) {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_q8_0_packed_register)(gguf_weights, packed, n_in, n_out) };
    }
}

#[inline]
pub fn tk_q36_q8_0_packed_unregister(gguf_weights: *const c_void) {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_q8_0_packed_unregister)(gguf_weights) };
    }
}

#[inline]
pub fn tk_q36_embed_rows(
    ctx: *mut tk_ctx,
    ggml_type: i32,
    table: *const c_void,
    row_bytes: usize,
    tok_dev: *const i32,
    n_tok: i64,
    n_embd: i64,
    out: *mut f32,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_embed_rows)(
            ctx, ggml_type, table, row_bytes, tok_dev, n_tok, n_embd, out, stream,
        )
    }
}

#[inline]
pub fn tk_q36_mmq_built() -> bool {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_mmq_built)() }
    } else {
        false
    }
}

#[inline]
pub fn tk_q36_mmq_supported_fits(ggml_type: i32, w_rows: i64) -> bool {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_mmq_supported_fits)(ggml_type, w_rows) }
    } else {
        false
    }
}

#[inline]
pub fn tk_q36_mmq_q8_layout(ggml_type: i32) -> i32 {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_mmq_q8_layout)(ggml_type) }
    } else {
        0
    }
}

#[inline]
pub fn tk_q36_mmq_q8_bytes(rows: i64, cols: i64) -> usize {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_mmq_q8_bytes)(rows, cols) }
    } else {
        0
    }
}

#[inline]
pub fn tk_q36_mmq_matrix_bytes(ggml_type: i32, rows: i64, cols: i64) -> usize {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_mmq_matrix_bytes)(ggml_type, rows, cols) }
    } else {
        0
    }
}

#[inline]
pub fn tk_q36_mmq_iota(ctx: *mut tk_ctx, dst: *mut i32, n: i64, stream: tk_stream) -> tk_status {
    unsafe { (lib().tk_q36_mmq_iota)(ctx, dst, n, stream) }
}

#[inline]
pub fn tk_q36_mmq_set_bounds(
    ctx: *mut tk_ctx,
    bounds: *mut i32,
    rows: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_mmq_set_bounds)(ctx, bounds, rows, stream) }
}

#[inline]
pub fn tk_q36_mmq_quantize(
    ctx: *mut tk_ctx,
    x: *const f32,
    ids: *const i32,
    xq: *mut c_void,
    ggml_type: i32,
    cols: i64,
    ld: i64,
    rows: i64,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_mmq_quantize)(ctx, x, ids, xq, ggml_type, cols, ld, rows, stream) }
}

#[inline]
pub fn tk_q36_mmq_swiglu(
    ctx: *mut tk_ctx,
    gu: *const f32,
    h: *mut f32,
    rows: i64,
    n_ff: i64,
    interleaved: bool,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_mmq_swiglu)(ctx, gu, h, rows, n_ff, interleaved, stream) }
}

#[inline]
pub fn tk_q36_fattn_supported(head_dim: i32) -> bool {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_fattn_supported)(head_dim) }
    } else {
        head_dim == 128 || head_dim == 256
    }
}

#[inline]
pub fn tk_q36_fattn_mask_bytes(n_tok: i64, n_kv: i64) -> usize {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_fattn_mask_bytes)(n_tok, n_kv) }
    } else {
        0
    }
}

#[inline]
pub fn tk_q36_prefill_init_scratch(ctx: *mut tk_ctx, scratch_elems: i64) -> tk_status {
    unsafe { (lib().tk_q36_prefill_init_scratch)(ctx, scratch_elems) }
}

#[inline]
pub fn tk_q36_attn_blas_prefill(
    ctx: *mut tk_ctx,
    q: *const f32,
    q16: *mut c_void,
    kc: *const c_void,
    vc: *const c_void,
    att_s: *mut f32,
    att_p16: *mut c_void,
    att: *mut f32,
    gate: *const f32,
    t: i32,
    h: i32,
    hkv: i32,
    hd: i32,
    ctx_len: i32,
    p0: i64,
    score_floats: usize,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_attn_blas_prefill)(
            ctx,
            q,
            q16,
            kc,
            vc,
            att_s,
            att_p16,
            att,
            gate,
            t,
            h,
            hkv,
            hd,
            ctx_len,
            p0,
            score_floats,
            stream,
        )
    }
}

#[inline]
pub fn tk_q36_router_topk(
    ctx: *mut tk_ctx,
    logits: *const f32,
    ids: *mut i32,
    weights: *mut f32,
    n_expert: i32,
    k: i32,
    n_tok: i32,
    stream: tk_stream,
) -> tk_status {
    unsafe { (lib().tk_q36_router_topk)(ctx, logits, ids, weights, n_expert, k, n_tok, stream) }
}

#[inline]
pub fn tk_q36_expert_supported(gu_type: i32, d_type: i32, n_embd: i64, n_ff: i64) -> bool {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_expert_supported)(gu_type, d_type, n_embd, n_ff) }
    } else {
        true
    }
}

#[inline]
pub fn tk_q36_expert_layout(
    gu_type: i32,
    d_type: i32,
    n_embd: i64,
    n_ff: i64,
    out_gu_row: *mut usize,
    out_d_row: *mut usize,
    out_up_off: *mut usize,
    out_down_off: *mut usize,
    out_bytes: *mut usize,
    out_tail_off: *mut usize,
) {
    if let Ok(k) = load_kernels_once() {
        unsafe {
            (k.lib.tk_q36_expert_layout)(
                gu_type,
                d_type,
                n_embd,
                n_ff,
                out_gu_row,
                out_d_row,
                out_up_off,
                out_down_off,
                out_bytes,
                out_tail_off,
            );
        }
    }
}

#[inline]
pub fn tk_q36_expert_scratch_bytes(cap_entries: i64, n_ff: i64) -> usize {
    if let Ok(k) = load_kernels_once() {
        unsafe { (k.lib.tk_q36_expert_scratch_bytes)(cap_entries, n_ff) }
    } else {
        0
    }
}

#[inline]
pub fn tk_q36_expert_grouped(
    ctx: *mut tk_ctx,
    gu_type: i32,
    d_type: i32,
    n_embd: i64,
    n_ff: i64,
    grp_ptr: *const u64,
    grp_start: *const i32,
    n_groups: *const i32,
    ent_dst: *const i32,
    ent_tok: *const i32,
    cap_groups: i64,
    cap_entries: i64,
    x_q8_1: *const c_void,
    scratch: *mut c_void,
    out: *mut f32,
    grid_groups: i64,
    stream: tk_stream,
) -> tk_status {
    unsafe {
        (lib().tk_q36_expert_grouped)(
            ctx,
            gu_type,
            d_type,
            n_embd,
            n_ff,
            grp_ptr,
            grp_start,
            n_groups,
            ent_dst,
            ent_tok,
            cap_groups,
            cap_entries,
            x_q8_1,
            scratch,
            out,
            grid_groups,
            stream,
        )
    }
}

// ============================================================================
// 10. Talus prefill Prompt Attention, Expanded INT8 & W4A4 Wrappers
// ============================================================================

#[inline]
pub fn talus_pf_fa_prefill_ok(d: i32, h: i32, hkv: i32) -> bool {
    (d == 128 || d == 256) && h > 0 && hkv > 0 && (h % hkv) == 0 && (h / hkv) <= 16
}

#[inline]
pub fn talus_pf_fa_prefill(
    q: *const f32,
    k: *const c_void,
    vt: *const c_void,
    gate: *const f32,
    out: *mut f32,
    t: i32,
    h: i32,
    hkv: i32,
    d: i32,
    kv_stride: i64,
    vt_cells: i64,
    p0: i64,
    scale: f32,
    stream: *mut c_void,
) -> bool {
    if let Ok(kern) = load_kernels_once() {
        unsafe {
            (kern.lib.talus_pf_fa_prefill)(
                q, k, vt, gate, out, t, h, hkv, d, kv_stride, vt_cells, p0, scale, stream,
            ) == 0
        }
    } else {
        false
    }
}

#[inline]
pub const fn talus_pf_i8x_mode(ggml_type: i32) -> i32 {
    match ggml_type {
        // IQ4_XS (23), IQ4_NL (20), IQ3_S (21), IQ3_XXS (18), IQ2_XXS (16), Q8_0 (8), Q5_0 (6)
        23 | 20 | 21 | 18 | 16 | 8 | 6 => 0,
        // Q3_K (11), IQ2_XS (17), IQ2_S (22); Q6_K (14, per-16 scale d * sc, bitwise MMQ;
        // TALUS_PF_I8_Q6K=0 makes the expansion refuse it -> MMQ)
        11 | 17 | 22 | 14 => 1,
        // Q4_K (12), Q5_K (13)
        12 | 13 => 2,
        _ => -1,
    }
}

#[inline]
pub const fn talus_pf_i8x_bytes(n: i32, k: i32, mode: i32) -> Option<(usize, usize)> {
    if mode < 0 || n <= 0 || k <= 0 || (n % 128) != 0 || (k % 256) != 0 {
        return None;
    }
    let qs_bytes = (n as usize) * (k as usize);
    let mul = if mode == 0 { 1usize } else { 2usize };
    let ws_bytes = (n as usize) * ((k as usize) / 32) * mul * 4;
    Some((qs_bytes, ws_bytes))
}

#[inline]
pub fn talus_pf_i8x_expand(
    w: *const c_void,
    ggml_type: i32,
    n: i32,
    k: i32,
    qs: *mut i8,
    ws: *mut f32,
    stream: *mut c_void,
) -> bool {
    if let Ok(kern) = load_kernels_once() {
        unsafe { (kern.lib.talus_pf_i8x_expand)(w, ggml_type, n, k, qs, ws, stream) == 0 }
    } else {
        false
    }
}

#[inline]
pub fn talus_pf_i8gemm(
    qs: *const i8,
    ws: *const f32,
    n: i32,
    k: i32,
    mode: i32,
    xq: *const c_void,
    t: i32,
    dst: *mut f32,
    ld_dst: i64,
    stream: *mut c_void,
) -> bool {
    if let Ok(kern) = load_kernels_once() {
        unsafe { (kern.lib.talus_pf_i8gemm)(qs, ws, n, k, mode, xq, t, dst, ld_dst, stream) == 0 }
    } else {
        false
    }
}

/// rb-q27: up projection (expanded int8) with the SwiGLU + q8_1 quantization in its epilogue (see talus_kernels.h).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn talus_pf_i8gemm_swiglu_q8(
    qs: *const i8,
    ws: *const f32,
    n: i32,
    k: i32,
    mode: i32,
    xq: *const c_void,
    t: i32,
    g: *const f32,
    ldg: i64,
    yq: *mut c_void,
    layout: i32,
    stream: *mut c_void,
) -> bool {
    if let Ok(kern) = load_kernels_once() {
        unsafe {
            (kern.lib.talus_pf_i8gemm_swiglu_q8)(
                qs, ws, n, k, mode, xq, t, g, ldg, yq, layout, stream,
            ) == 0
        }
    } else {
        false
    }
}

#[inline]
pub fn talus_pf_swiglu_q8_mmq(
    g: *const f32,
    u: *const f32,
    xq: *mut c_void,
    ggml_type: i32,
    cols: i64,
    rows: i64,
    stream: *mut c_void,
) -> bool {
    if let Ok(kern) = load_kernels_once() {
        unsafe { (kern.lib.talus_pf_swiglu_q8_mmq)(g, u, xq, ggml_type, cols, rows, stream) == 0 }
    } else {
        false
    }
}

#[inline]
pub fn talus_pf_w4_pack(
    x: *const f32,
    t: i32,
    k: i32,
    ldx: i64,
    signs: *const i8,
    xq: *mut u8,
    xs: *mut f32,
    stream: *mut c_void,
) -> bool {
    if let Ok(kern) = load_kernels_once() {
        unsafe { (kern.lib.talus_pf_w4_pack)(x, t, k, ldx, signs, xq, xs, stream) == 0 }
    } else {
        false
    }
}

#[inline]
pub fn talus_pf_w4_gemm(
    wq: *const u8,
    ws: *const f32,
    n: i32,
    k: i32,
    xq: *const u8,
    xs: *const f32,
    t: i32,
    dst: *mut f32,
    ld_dst: i64,
    stream: *mut c_void,
) -> bool {
    if let Ok(kern) = load_kernels_once() {
        unsafe { (kern.lib.talus_pf_w4_gemm)(wq, ws, n, k, xq, xs, t, dst, ld_dst, stream) == 0 }
    } else {
        false
    }
}
