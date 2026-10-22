//! Dynamic FFI loader for `libtalus_kernels.so` and `libamdhip64.so` used by the
//! Laguna-S-2.1 (`laguna`) engine, plus POSIX file helpers for pinned weight staging
//! and host-side RoPE table construction matching `src/kernels/cuda/rope.cu`.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;

use hip_sys::{hipDeviceSynchronize, hipGetErrorString, hipGetLastError, hipStream_t};
use talus_core_api::{CoreError, Result};
use talus_kernels_sys::{tk_config, tk_ctx, tk_status, tk_stream, LibTalusKernels, TK_OK};

const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;

    pub fn pread(fd: c_int, buf: *mut c_void, count: usize, offset: i64) -> isize;
    pub fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
}

/// `POSIX_FADV_DONTNEED` on Linux (`<fcntl.h>`).
pub const POSIX_FADV_DONTNEED: c_int = 4;

/// RoPE scaling mode matching `strata::kernels::RopeScalingType::None`.
pub const TK_ROPE_SCALING_NONE: i32 = 0;
/// RoPE scaling mode matching `strata::kernels::RopeScalingType::Linear`.
pub const TK_ROPE_SCALING_LINEAR: i32 = 1;
/// RoPE scaling mode matching `strata::kernels::RopeScalingType::YaRN`.
pub const TK_ROPE_SCALING_YARN: i32 = 2;

/// RoPE scaling parameters matching `strata::kernels::RopeScalingParams` in
/// `include/strata/kernels/rope_scaling.hpp`.
#[derive(Debug, Clone, Copy)]
pub struct RopeScalingParams {
    pub scaling_type: i32,
    pub freq_base: f64,
    pub factor: f64,
    pub freq_scale_in: f64,
    pub orig_ctx: i32,
    pub ext_factor: f64,
    pub attn_factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

impl Default for RopeScalingParams {
    fn default() -> Self {
        Self {
            scaling_type: TK_ROPE_SCALING_NONE,
            freq_base: 10000.0,
            factor: 1.0,
            freq_scale_in: 0.0,
            orig_ctx: 0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

impl RopeScalingParams {
    #[inline]
    pub fn freq_scale(&self) -> f64 {
        if self.freq_scale_in > 0.0 {
            return self.freq_scale_in;
        }
        if self.scaling_type == TK_ROPE_SCALING_NONE || self.factor <= 0.0 {
            return 1.0;
        }
        1.0 / self.factor
    }

    #[inline]
    pub fn mscale(&self) -> f64 {
        if self.scaling_type != TK_ROPE_SCALING_YARN {
            return 1.0;
        }
        let fs = self.freq_scale();
        if self.ext_factor == 0.0 || fs <= 0.0 || fs >= 1.0 {
            return self.attn_factor;
        }
        self.attn_factor * (1.0 + 0.1 * (1.0 / fs).ln())
    }

    #[inline]
    pub fn corr_dims(&self, n_rot: i32) -> [f64; 2] {
        if self.scaling_type != TK_ROPE_SCALING_YARN || self.ext_factor == 0.0 || self.orig_ctx <= 0
        {
            return [0.0, -1.0];
        }
        let start =
            rope_yarn_corr_dim(n_rot, self.orig_ctx as f64, self.beta_fast, self.freq_base).floor();
        let end =
            rope_yarn_corr_dim(n_rot, self.orig_ctx as f64, self.beta_slow, self.freq_base).ceil();
        let lo = if start < 0.0 { 0.0 } else { start };
        let hi_max = (n_rot - 1) as f64;
        let hi = if end > hi_max { hi_max } else { end };
        [lo, hi]
    }
}

#[inline]
pub fn rope_yarn_corr_dim(n_dims: i32, n_ctx_orig: f64, n_rot: f64, base: f64) -> f64 {
    const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
    (n_dims as f64) * (n_ctx_orig / (n_rot * TWO_PI)).ln() / (2.0 * base.ln())
}

#[inline]
pub fn rope_yarn_ramp(low: f32, high: f32, pair: i32) -> f32 {
    let diff = high - low;
    let denom = if diff > 0.001_f32 { diff } else { 0.001_f32 };
    let y = (pair as f32 - low) / denom;
    let clamped = y.clamp(0.0_f32, 1.0_f32);
    1.0_f32 - clamped
}

/// Build `[max_pos, n_rot / 2]` RoPE cos and sin tables on the host using the exact
/// `double` (`f64`) -> `float` (`f32`) math from `src/kernels/cuda/rope.cu`.
pub fn tk_build_rope_table(
    cos_tab: &mut [f32],
    sin_tab: &mut [f32],
    max_pos: usize,
    n_rot: i32,
    sc: &RopeScalingParams,
) {
    assert!(n_rot > 0 && n_rot % 2 == 0, "n_rot must be positive even");
    let half = (n_rot / 2) as usize;
    assert!(cos_tab.len() >= max_pos * half);
    assert!(sin_tab.len() >= max_pos * half);

    if sc.scaling_type == TK_ROPE_SCALING_NONE {
        let theta = sc.freq_base;
        for p in 0..max_pos {
            for i in 0..half {
                let inv = theta.powf(-2.0 * (i as f64) / (n_rot as f64));
                let ang = (p as f64) * inv;
                cos_tab[p * half + i] = ang.cos() as f32;
                sin_tab[p * half + i] = ang.sin() as f32;
            }
        }
        return;
    }

    let fs = sc.freq_scale();
    let ms = sc.mscale();
    let cd = sc.corr_dims(n_rot);
    let correct = sc.scaling_type == TK_ROPE_SCALING_YARN && sc.ext_factor != 0.0;

    for p in 0..max_pos {
        for i in 0..half {
            let inv = sc.freq_base.powf(-2.0 * (i as f64) / (n_rot as f64));
            let extrap = (p as f64) * inv;
            let interp = fs * extrap;
            let mut ang = interp;
            if correct {
                let ramp =
                    (rope_yarn_ramp(cd[0] as f32, cd[1] as f32, i as i32) as f64) * sc.ext_factor;
                ang = interp * (1.0 - ramp) + extrap * ramp;
            }
            cos_tab[p * half + i] = (ang.cos() * ms) as f32;
            sin_tab[p * half + i] = (ang.sin() * ms) as f32;
        }
    }
}

fn last_dlerror() -> String {
    // SAFETY: dlerror returns a thread-local C string or null.
    unsafe {
        let p = dlerror();
        if p.is_null() {
            "unknown dlopen/dlsym error".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

/// Check a `tk_status` return code and convert any failure into a `CoreError`.
#[inline]
pub fn check_tk(_ctx: *const tk_ctx, st: tk_status, op: &'static str) -> Result<()> {
    if st == TK_OK {
        return Ok(());
    }
    let name = match st {
        1 => "TK_ERR_INVALID_ARG",
        2 => "TK_ERR_NULL_PTR",
        3 => "TK_ERR_UNSUPPORTED",
        4 => "TK_ERR_HIP",
        _ => "TK_ERR",
    };
    Err(CoreError::KernelError {
        code: st,
        message: format!("{op} failed: {name} ({st})"),
    })
}

/// Check a HIP runtime error code and convert any failure into a `CoreError`.
#[inline]
pub fn check_hip(err: i32, op: &'static str) -> Result<()> {
    if err == 0 {
        return Ok(());
    }
    // SAFETY: hipGetErrorString returns a static string pointer for any error code.
    let msg = unsafe {
        let ptr = hipGetErrorString(hip_sys::hipError_t(err));
        if ptr.is_null() {
            format!("{op}: HIP error {err}")
        } else {
            format!("{op}: {} ({err})", CStr::from_ptr(ptr).to_string_lossy())
        }
    };
    Err(CoreError::HipError {
        code: err,
        message: msg,
    })
}

/// Synchronize the device and check for any asynchronous HIP errors.
pub fn sync_and_check(op: &'static str) -> Result<()> {
    // SAFETY: hipDeviceSynchronize / hipGetLastError are global runtime calls.
    unsafe {
        check_hip(hipDeviceSynchronize().0, op)?;
        check_hip(hipGetLastError().0, op)
    }
}

/// Opaque HIP graph handle (`hipGraph_t`).
pub type HipGraph = *mut c_void;
/// Opaque executable HIP graph handle (`hipGraphExec_t`).
pub type HipGraphExec = *mut c_void;

/// `hipStreamCaptureModeGlobal = 0` (`cudaStreamCaptureModeGlobal`).
pub const HIP_STREAM_CAPTURE_MODE_GLOBAL: i32 = 0;
/// `hipStreamCaptureModeThreadLocal = 1` (`cudaStreamCaptureModeThreadLocal`).
pub const HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL: i32 = 1;
/// `hipStreamNonBlocking = 1` (`cudaStreamNonBlocking`).
pub const HIP_STREAM_NON_BLOCKING: u32 = 1;

/// Dynamically loaded HIP runtime functions needed by Laguna (`hipMalloc`, `hipFree`,
/// `hipHostMalloc`, `hipHostFree`, `hipMemcpy`, `hipMemcpyAsync`, `hipMemset`,
/// `hipMemsetAsync`, `hipStreamCreate`, `hipStreamCreateWithFlags`, `hipStreamSynchronize`,
/// `hipStreamDestroy`, and HIP graph capture/launch APIs).
#[allow(non_snake_case)]
pub struct HipRuntime {
    handle: *mut c_void,
    pub hipMalloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> i32,
    pub hipFree: unsafe extern "C" fn(*mut c_void) -> i32,
    pub hipHostMalloc: unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32,
    pub hipHostFree: unsafe extern "C" fn(*mut c_void) -> i32,
    pub hipMemcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32,
    pub hipMemcpyAsync:
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32, hipStream_t) -> i32,
    pub hipMemset: unsafe extern "C" fn(*mut c_void, i32, usize) -> i32,
    pub hipMemsetAsync: unsafe extern "C" fn(*mut c_void, i32, usize, hipStream_t) -> i32,
    pub hipMemset2DAsync:
        unsafe extern "C" fn(*mut c_void, usize, i32, usize, usize, hipStream_t) -> i32,
    pub hipStreamCreate: unsafe extern "C" fn(*mut hipStream_t) -> i32,
    pub hipStreamCreateWithFlags: unsafe extern "C" fn(*mut hipStream_t, u32) -> i32,
    pub hipStreamDestroy: unsafe extern "C" fn(hipStream_t) -> i32,
    pub hipStreamSynchronize: unsafe extern "C" fn(hipStream_t) -> i32,
    pub hipDeviceSynchronize: unsafe extern "C" fn() -> i32,
    pub hipStreamBeginCapture: unsafe extern "C" fn(hipStream_t, i32) -> i32,
    pub hipStreamEndCapture: unsafe extern "C" fn(hipStream_t, *mut HipGraph) -> i32,
    pub hipGraphInstantiate:
        unsafe extern "C" fn(*mut HipGraphExec, HipGraph, *mut c_void, *mut c_char, usize) -> i32,
    pub hipGraphLaunch: unsafe extern "C" fn(HipGraphExec, hipStream_t) -> i32,
    pub hipGraphExecDestroy: unsafe extern "C" fn(HipGraphExec) -> i32,
    pub hipGraphDestroy: unsafe extern "C" fn(HipGraph) -> i32,
}

// SAFETY: HIP runtime function pointers are process-global and thread-safe.
unsafe impl Send for HipRuntime {}
unsafe impl Sync for HipRuntime {}

impl HipRuntime {
    pub const MEMCPY_H2D: i32 = 1;
    pub const MEMCPY_D2H: i32 = 2;
    pub const MEMCPY_D2D: i32 = 3;

    /// Load `libamdhip64.so` dynamically.
    pub fn load() -> Result<Self> {
        let mut candidates = vec!["libamdhip64.so.7".to_string()];
        for var in ["ROCM_SDK", "ROCM_PATH"] {
            if let Ok(root) = std::env::var(var) {
                if !root.is_empty() {
                    candidates.push(format!("{root}/lib/libamdhip64.so.7"));
                    candidates.push(format!("{root}/lib/libamdhip64.so"));
                }
            }
        }
        candidates.extend([
            "/opt/rocm/lib/libamdhip64.so.7".to_string(),
            "libamdhip64.so.6".to_string(),
            "libamdhip64.so".to_string(),
            "/opt/rocm/lib/libamdhip64.so".to_string(),
        ]);
        let mut handle = std::ptr::null_mut();
        for cand in &candidates {
            let cstr = CString::new(cand.as_str()).unwrap();
            // SAFETY: dlopen with valid NUL-terminated string.
            let h = unsafe { dlopen(cstr.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
            if !h.is_null() {
                handle = h;
                break;
            }
        }
        if handle.is_null() {
            return Err(CoreError::HipError {
                code: -1,
                message: format!("failed to dlopen libamdhip64.so: {}", last_dlerror()),
            });
        }

        macro_rules! sym {
            ($name:ident, $ty:ty) => {{
                let cname = CString::new(stringify!($name)).unwrap();
                // SAFETY: handle is valid and cname is NUL-terminated.
                let ptr = unsafe { dlsym(handle, cname.as_ptr()) };
                if ptr.is_null() {
                    unsafe { dlclose(handle) };
                    return Err(CoreError::HipError {
                        code: -1,
                        message: format!(
                            "missing HIP symbol {}: {}",
                            stringify!($name),
                            last_dlerror()
                        ),
                    });
                }
                // SAFETY: symbol matches the HIP C ABI signature.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(ptr) }
            }};
        }

        Ok(Self {
            handle,
            hipMalloc: sym!(
                hipMalloc,
                unsafe extern "C" fn(*mut *mut c_void, usize) -> i32
            ),
            hipFree: sym!(hipFree, unsafe extern "C" fn(*mut c_void) -> i32),
            hipHostMalloc: sym!(
                hipHostMalloc,
                unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32
            ),
            hipHostFree: sym!(hipHostFree, unsafe extern "C" fn(*mut c_void) -> i32),
            hipMemcpy: sym!(
                hipMemcpy,
                unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32
            ),
            hipMemcpyAsync: sym!(
                hipMemcpyAsync,
                unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32, hipStream_t) -> i32
            ),
            hipMemset: sym!(
                hipMemset,
                unsafe extern "C" fn(*mut c_void, i32, usize) -> i32
            ),
            hipMemsetAsync: sym!(
                hipMemsetAsync,
                unsafe extern "C" fn(*mut c_void, i32, usize, hipStream_t) -> i32
            ),
            hipMemset2DAsync: sym!(
                hipMemset2DAsync,
                unsafe extern "C" fn(*mut c_void, usize, i32, usize, usize, hipStream_t) -> i32
            ),
            hipStreamCreate: sym!(
                hipStreamCreate,
                unsafe extern "C" fn(*mut hipStream_t) -> i32
            ),
            hipStreamCreateWithFlags: sym!(
                hipStreamCreateWithFlags,
                unsafe extern "C" fn(*mut hipStream_t, u32) -> i32
            ),
            hipStreamDestroy: sym!(hipStreamDestroy, unsafe extern "C" fn(hipStream_t) -> i32),
            hipStreamSynchronize: sym!(
                hipStreamSynchronize,
                unsafe extern "C" fn(hipStream_t) -> i32
            ),
            hipDeviceSynchronize: sym!(hipDeviceSynchronize, unsafe extern "C" fn() -> i32),
            hipStreamBeginCapture: sym!(
                hipStreamBeginCapture,
                unsafe extern "C" fn(hipStream_t, i32) -> i32
            ),
            hipStreamEndCapture: sym!(
                hipStreamEndCapture,
                unsafe extern "C" fn(hipStream_t, *mut HipGraph) -> i32
            ),
            hipGraphInstantiate: sym!(
                hipGraphInstantiate,
                unsafe extern "C" fn(
                    *mut HipGraphExec,
                    HipGraph,
                    *mut c_void,
                    *mut c_char,
                    usize,
                ) -> i32
            ),
            hipGraphLaunch: sym!(
                hipGraphLaunch,
                unsafe extern "C" fn(HipGraphExec, hipStream_t) -> i32
            ),
            hipGraphExecDestroy: sym!(
                hipGraphExecDestroy,
                unsafe extern "C" fn(HipGraphExec) -> i32
            ),
            hipGraphDestroy: sym!(hipGraphDestroy, unsafe extern "C" fn(HipGraph) -> i32),
        })
    }

    /// Allocate `bytes` of device memory. Returns null if `bytes == 0`.
    pub fn alloc_device(&self, bytes: usize) -> Result<*mut c_void> {
        if bytes == 0 {
            return Ok(std::ptr::null_mut());
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: hipMalloc writes to `ptr`.
        let err = unsafe { (self.hipMalloc)(&mut ptr, bytes) };
        if err != 0 {
            return Err(CoreError::HipError {
                code: err,
                message: format!("hipMalloc({bytes} B) failed: {err}"),
            });
        }
        Ok(ptr)
    }

    /// Allocate `bytes` of zero-initialized device memory.
    pub fn alloc_device_zeroed(&self, bytes: usize) -> Result<*mut c_void> {
        let ptr = self.alloc_device(bytes)?;
        if !ptr.is_null() {
            // SAFETY: ptr was just allocated with `bytes`.
            let err = unsafe { (self.hipMemset)(ptr, 0, bytes) };
            if err != 0 {
                unsafe { (self.hipFree)(ptr) };
                return Err(CoreError::HipError {
                    code: err,
                    message: format!("hipMemset({bytes} B) failed: {err}"),
                });
            }
        }
        Ok(ptr)
    }

    /// Allocate `bytes` of pinned host memory (`hipHostMalloc`).
    pub fn alloc_pinned(&self, bytes: usize) -> Result<*mut c_void> {
        if bytes == 0 {
            return Ok(std::ptr::null_mut());
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: hipHostMalloc writes to `ptr`.
        let err = unsafe { (self.hipHostMalloc)(&mut ptr, bytes, 0) };
        if err != 0 {
            return Err(CoreError::HipError {
                code: err,
                message: format!("hipHostMalloc({bytes} B) failed: {err}"),
            });
        }
        Ok(ptr)
    }

    /// Free device memory if non-null.
    pub fn free_device(&self, ptr: *mut c_void) {
        if !ptr.is_null() {
            // SAFETY: ptr was allocated by hipMalloc.
            unsafe {
                let _ = (self.hipFree)(ptr);
            }
        }
    }

    /// Free pinned host memory if non-null.
    pub fn free_pinned(&self, ptr: *mut c_void) {
        if !ptr.is_null() {
            // SAFETY: ptr was allocated by hipHostMalloc.
            unsafe {
                let _ = (self.hipHostFree)(ptr);
            }
        }
    }

    /// Synchronous host-to-device copy.
    pub fn copy_h2d(&self, dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: caller guarantees valid dst/src for `bytes`.
        let err = unsafe { (self.hipMemcpy)(dst, src, bytes, Self::MEMCPY_H2D) };
        if err != 0 {
            return Err(CoreError::HipError {
                code: err,
                message: format!("hipMemcpy H2D ({bytes} B) failed: {err}"),
            });
        }
        Ok(())
    }

    /// Synchronous device-to-host copy.
    pub fn copy_d2h(&self, dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: caller guarantees valid dst/src for `bytes`.
        let err = unsafe { (self.hipMemcpy)(dst, src, bytes, Self::MEMCPY_D2H) };
        if err != 0 {
            return Err(CoreError::HipError {
                code: err,
                message: format!("hipMemcpy D2H ({bytes} B) failed: {err}"),
            });
        }
        Ok(())
    }
}

impl Drop for HipRuntime {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle was opened by dlopen.
            unsafe {
                dlclose(self.handle);
            }
        }
    }
}

/// Resolve and load `libtalus_kernels.so` from `TALUS_KERNELS_LIB` or standard build paths.
pub fn load_talus_kernels(explicit_path: Option<&Path>) -> Result<(LibTalusKernels, *mut c_void)> {
    let mut candidates = Vec::new();
    if let Some(p) = explicit_path {
        candidates.push(p.to_path_buf());
    }
    if let Ok(env_p) = std::env::var("TALUS_KERNELS_LIB") {
        candidates.push(std::path::PathBuf::from(env_p));
    }
    if let Ok(env_p) = std::env::var("TALUS_KERNELS_PREBUILT") {
        candidates.push(std::path::PathBuf::from(env_p));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("libtalus_kernels.so"));
            if let Some(parent) = dir.parent() {
                candidates.push(parent.join("libtalus_kernels.so"));
                candidates.push(parent.join("build/libtalus_kernels.so"));
            }
        }
    }
    candidates.push(std::path::PathBuf::from("build/libtalus_kernels.so"));
    candidates.push(std::path::PathBuf::from("../build/libtalus_kernels.so"));
    candidates.push(std::path::PathBuf::from("../../build/libtalus_kernels.so"));
    candidates.push(std::path::PathBuf::from("libtalus_kernels.so"));

    let mut chosen = None;
    for cand in &candidates {
        if let Ok(lib) = LibTalusKernels::load(cand) {
            chosen = Some(lib);
            break;
        }
    }
    match chosen {
        Some(lib) => Ok((lib, std::ptr::null_mut())),
        None => Err(CoreError::KernelError {
            code: -1,
            message: format!(
                "failed to dlopen libtalus_kernels.so (tried {:?}): {}",
                candidates,
                last_dlerror()
            ),
        }),
    }
}

/// Helper wrapper for `f32_to_f16_bulk` (`tk_q36_to_f16` in Contract v0).
#[inline]
pub unsafe fn tk_f32_to_f16_bulk(
    tk: &LibTalusKernels,
    ctx: *mut tk_ctx,
    x: *const f32,
    y_f16: *mut c_void,
    n: i64,
    stream: tk_stream,
) -> tk_status {
    unsafe { (tk.tk_q36_to_f16)(ctx, x, y_f16, n, stream) }
}

/// Helper wrapper for `add_inplace` (`tk_q36_add` in Contract v0).
#[inline]
pub unsafe fn tk_add_inplace(
    tk: &LibTalusKernels,
    ctx: *mut tk_ctx,
    x: *mut f32,
    y: *const f32,
    n: i64,
    stream: tk_stream,
) -> tk_status {
    unsafe { (tk.tk_q36_add)(ctx, x, y, n, stream) }
}

/// RAII wrapper around `tk_ctx*` created via `LibTalusKernels`.
pub struct TkContext {
    raw: *mut tk_ctx,
    destroy_fn: unsafe extern "C" fn(*mut tk_ctx),
}

// SAFETY: tk_ctx is owned by the engine instance.
unsafe impl Send for TkContext {}
unsafe impl Sync for TkContext {}

impl TkContext {
    pub fn new(lib: &LibTalusKernels, device_id: i32, _stream: tk_stream) -> Result<Self> {
        let mut cfg_ptr: *mut tk_config = std::ptr::null_mut();
        // SAFETY: tk_config_create / tk_config_from_env initialize an opaque tk_config.
        unsafe {
            if (lib.tk_config_create)(&mut cfg_ptr) == TK_OK && !cfg_ptr.is_null() {
                let _ = (lib.tk_config_from_env)(cfg_ptr);
            }
        }
        let mut raw: *mut tk_ctx = std::ptr::null_mut();
        // SAFETY: lib.tk_ctx_create writes to `raw`.
        let st = unsafe { (lib.tk_ctx_create)(device_id, cfg_ptr as *const tk_config, &mut raw) };
        if !cfg_ptr.is_null() {
            unsafe {
                (lib.tk_config_destroy)(cfg_ptr);
            }
        }
        if st != TK_OK || raw.is_null() {
            return Err(CoreError::KernelError {
                code: st,
                message: format!("tk_ctx_create failed with status {st}"),
            });
        }
        Ok(Self {
            raw,
            destroy_fn: lib.tk_ctx_destroy,
        })
    }

    #[inline]
    pub fn as_ptr(&self) -> *mut tk_ctx {
        self.raw
    }
}

impl Drop for TkContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: raw was created by tk_ctx_create.
            unsafe {
                (self.destroy_fn)(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}
