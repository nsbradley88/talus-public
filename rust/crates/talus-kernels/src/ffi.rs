//! FFI kernel backend (`FfiKernelBackend`, `FfiConfig`, `FfiPackedWeights`) wrapping
//! `talus-kernels-sys` (`tk_ctx*`, `tk_config*`, `tk_packed_weights*`) and translating
//! `tk_status` into [`talus_core_api::Result`] using `tk_last_error`.

use core::ffi::{c_void, CStr};
use std::ffi::CString;
use std::path::Path;
use std::sync::Arc;

use talus_core_api::{CoreError, GpuArch, QuantType, Result, StreamHandle};
use talus_kernels_sys::{
    tk_arch, tk_config, tk_ctx, tk_packed_weights, tk_qtype, tk_status, tk_status_code,
    LibTalusKernels, TK_ARCH_GFX1151, TK_ARCH_GFX1201, TK_ARCH_OTHER, TK_ERR_EXCEPTION, TK_ERR_HIP,
    TK_ERR_HIPBLAS, TK_ERR_INTERNAL, TK_ERR_INVALID_ARG, TK_ERR_IO, TK_ERR_NULL_PTR,
    TK_ERR_OUT_OF_MEMORY, TK_ERR_UNSUPPORTED, TK_OK, TK_QTYPE_BF16, TK_QTYPE_F16, TK_QTYPE_F32,
    TK_QTYPE_I8, TK_QTYPE_IQ1_M, TK_QTYPE_IQ1_S, TK_QTYPE_IQ2_S, TK_QTYPE_IQ2_XS, TK_QTYPE_IQ2_XXS,
    TK_QTYPE_IQ3_S, TK_QTYPE_IQ3_XXS, TK_QTYPE_IQ4_NL, TK_QTYPE_IQ4_XS, TK_QTYPE_MXFP4,
    TK_QTYPE_NVFP4, TK_QTYPE_Q1_0, TK_QTYPE_Q2_0, TK_QTYPE_Q2_K, TK_QTYPE_Q3_K, TK_QTYPE_Q4_0,
    TK_QTYPE_Q4_1, TK_QTYPE_Q4_K, TK_QTYPE_Q5_0, TK_QTYPE_Q5_1, TK_QTYPE_Q5_K, TK_QTYPE_Q6_K,
    TK_QTYPE_Q8_0, TK_QTYPE_Q8_1, TK_QTYPE_Q8_K, TK_QTYPE_TQ1_0, TK_QTYPE_TQ2_0,
};

use crate::backend::{
    AttnDecodeArgs, AttnPrefillArgs, GdnSeqArgs, GdnStepArgs, KernelBackend, MmvqIdsArgs,
    PrefillGemmArgs, PrefillMmqArgs,
};
use crate::quant::{q8_1_bytes, row_bytes};
use crate::{invalid_arg, io_err};

/// Convert [`GpuArch`] to [`tk_arch`].
#[inline]
pub const fn gpu_arch_to_tk(arch: &GpuArch) -> tk_arch {
    match arch {
        GpuArch::Gfx1151 => TK_ARCH_GFX1151,
        GpuArch::Gfx1201 => TK_ARCH_GFX1201,
        GpuArch::Other(_) => TK_ARCH_OTHER,
    }
}

/// Convert [`tk_arch`] to [`GpuArch`].
#[inline]
pub fn tk_to_gpu_arch(arch: tk_arch) -> GpuArch {
    match arch {
        TK_ARCH_GFX1151 => GpuArch::Gfx1151,
        TK_ARCH_GFX1201 => GpuArch::Gfx1201,
        TK_ARCH_OTHER => GpuArch::Other("other".to_string()),
    }
}

/// Convert [`QuantType`] to [`tk_qtype`].
#[inline]
pub const fn quant_type_to_tk(qtype: QuantType) -> tk_qtype {
    match qtype {
        QuantType::F32 => TK_QTYPE_F32,
        QuantType::F16 => TK_QTYPE_F16,
        QuantType::Q4_0 => TK_QTYPE_Q4_0,
        QuantType::Q4_1 => TK_QTYPE_Q4_1,
        QuantType::Q5_0 => TK_QTYPE_Q5_0,
        QuantType::Q5_1 => TK_QTYPE_Q5_1,
        QuantType::Q8_0 => TK_QTYPE_Q8_0,
        QuantType::Q8_1 => TK_QTYPE_Q8_1,
        QuantType::Q2_K => TK_QTYPE_Q2_K,
        QuantType::Q3_K => TK_QTYPE_Q3_K,
        QuantType::Q4_K => TK_QTYPE_Q4_K,
        QuantType::Q5_K => TK_QTYPE_Q5_K,
        QuantType::Q6_K => TK_QTYPE_Q6_K,
        QuantType::Q8_K => TK_QTYPE_Q8_K,
        QuantType::IQ2_XXS => TK_QTYPE_IQ2_XXS,
        QuantType::IQ2_XS => TK_QTYPE_IQ2_XS,
        QuantType::IQ3_XXS => TK_QTYPE_IQ3_XXS,
        QuantType::IQ1_S => TK_QTYPE_IQ1_S,
        QuantType::IQ4_NL => TK_QTYPE_IQ4_NL,
        QuantType::IQ3_S => TK_QTYPE_IQ3_S,
        QuantType::IQ2_S => TK_QTYPE_IQ2_S,
        QuantType::IQ4_XS => TK_QTYPE_IQ4_XS,
        QuantType::I8 => TK_QTYPE_I8,
        QuantType::IQ1_M => TK_QTYPE_IQ1_M,
        QuantType::BF16 => TK_QTYPE_BF16,
        QuantType::TQ1_0 => TK_QTYPE_TQ1_0,
        QuantType::TQ2_0 => TK_QTYPE_TQ2_0,
        QuantType::MXFP4 => TK_QTYPE_MXFP4,
        QuantType::NVFP4 => TK_QTYPE_NVFP4,
        QuantType::Q1_0 => TK_QTYPE_Q1_0,
        QuantType::Q2_0 => TK_QTYPE_Q2_0,
    }
}

/// Convert [`tk_qtype`] to [`QuantType`].
#[inline]
pub const fn tk_to_quant_type(qtype: tk_qtype) -> QuantType {
    match qtype {
        TK_QTYPE_F32 => QuantType::F32,
        TK_QTYPE_F16 => QuantType::F16,
        TK_QTYPE_Q4_0 => QuantType::Q4_0,
        TK_QTYPE_Q4_1 => QuantType::Q4_1,
        TK_QTYPE_Q5_0 => QuantType::Q5_0,
        TK_QTYPE_Q5_1 => QuantType::Q5_1,
        TK_QTYPE_Q8_0 => QuantType::Q8_0,
        TK_QTYPE_Q8_1 => QuantType::Q8_1,
        TK_QTYPE_Q2_K => QuantType::Q2_K,
        TK_QTYPE_Q3_K => QuantType::Q3_K,
        TK_QTYPE_Q4_K => QuantType::Q4_K,
        TK_QTYPE_Q5_K => QuantType::Q5_K,
        TK_QTYPE_Q6_K => QuantType::Q6_K,
        TK_QTYPE_Q8_K => QuantType::Q8_K,
        TK_QTYPE_IQ2_XXS => QuantType::IQ2_XXS,
        TK_QTYPE_IQ2_XS => QuantType::IQ2_XS,
        TK_QTYPE_IQ3_XXS => QuantType::IQ3_XXS,
        TK_QTYPE_IQ1_S => QuantType::IQ1_S,
        TK_QTYPE_IQ4_NL => QuantType::IQ4_NL,
        TK_QTYPE_IQ3_S => QuantType::IQ3_S,
        TK_QTYPE_IQ2_S => QuantType::IQ2_S,
        TK_QTYPE_IQ4_XS => QuantType::IQ4_XS,
        TK_QTYPE_I8 => QuantType::I8,
        TK_QTYPE_IQ1_M => QuantType::IQ1_M,
        TK_QTYPE_BF16 => QuantType::BF16,
        TK_QTYPE_TQ1_0 => QuantType::TQ1_0,
        TK_QTYPE_TQ2_0 => QuantType::TQ2_0,
        TK_QTYPE_MXFP4 => QuantType::MXFP4,
        TK_QTYPE_NVFP4 => QuantType::NVFP4,
        TK_QTYPE_Q1_0 => QuantType::Q1_0,
        TK_QTYPE_Q2_0 => QuantType::Q2_0,
    }
}

/// Translate a `tk_status` return code and optional `tk_ctx` last-error string into [`Result<()>`].
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn translate_tk_status(
    lib: &LibTalusKernels,
    ctx: *const tk_ctx,
    status: tk_status,
) -> Result<()> {
    if status == TK_OK {
        return Ok(());
    }

    let status_name = unsafe {
        let p = (lib.tk_status_string)(status);
        if p.is_null() {
            "TK_ERR_UNKNOWN"
        } else {
            CStr::from_ptr(p).to_str().unwrap_or("TK_ERR_UNKNOWN")
        }
    };

    let detail = unsafe {
        if !ctx.is_null() {
            let err_ptr = (lib.tk_last_error)(ctx);
            if !err_ptr.is_null() {
                let s = CStr::from_ptr(err_ptr).to_string_lossy();
                if !s.is_empty() {
                    format!("{}: {}", status_name, s)
                } else {
                    status_name.to_string()
                }
            } else {
                status_name.to_string()
            }
        } else {
            status_name.to_string()
        }
    };

    match tk_status_code::from_raw(status) {
        Some(tk_status_code::TK_OK) => Ok(()),
        Some(tk_status_code::TK_ERR_INVALID_ARG | tk_status_code::TK_ERR_NULL_PTR) => {
            Err(invalid_arg(detail))
        }
        Some(tk_status_code::TK_ERR_OUT_OF_MEMORY) => Err(CoreError::OutOfMemory {
            requested_bytes: 0,
            available_bytes: 0,
        }),
        Some(tk_status_code::TK_ERR_IO) => Err(io_err(detail)),
        Some(tk_status_code::TK_ERR_HIP) => Err(CoreError::HipError {
            code: status,
            message: detail,
        }),
        Some(
            tk_status_code::TK_ERR_UNSUPPORTED
            | tk_status_code::TK_ERR_HIPBLAS
            | tk_status_code::TK_ERR_EXCEPTION
            | tk_status_code::TK_ERR_INTERNAL,
        )
        | None => Err(CoreError::KernelError {
            code: status,
            message: detail,
        }),
    }
}

/// Return true if `status` matches one of the known `TK_ERR_*` constants.
#[inline]
pub const fn is_tk_error(status: tk_status) -> bool {
    matches!(
        status,
        TK_ERR_INVALID_ARG
            | TK_ERR_NULL_PTR
            | TK_ERR_UNSUPPORTED
            | TK_ERR_OUT_OF_MEMORY
            | TK_ERR_HIP
            | TK_ERR_HIPBLAS
            | TK_ERR_IO
            | TK_ERR_EXCEPTION
            | TK_ERR_INTERNAL
    ) || status != TK_OK
}

/// Safe RAII wrapper around `tk_config*`.
pub struct FfiConfig {
    lib: Arc<LibTalusKernels>,
    ptr: *mut tk_config,
}

// Safety: `tk_config` is an owned configuration object accessed only through `&self` / `&mut self`.
unsafe impl Send for FfiConfig {}
unsafe impl Sync for FfiConfig {}

impl FfiConfig {
    /// Allocate a default `tk_config` (`tk_config_create`).
    pub fn new(lib: Arc<LibTalusKernels>) -> Result<Self> {
        let mut ptr: *mut tk_config = core::ptr::null_mut();
        let st = unsafe { (lib.tk_config_create)(&mut ptr) };
        translate_tk_status(&lib, core::ptr::null(), st)?;
        if ptr.is_null() {
            return Err(CoreError::KernelError {
                code: TK_ERR_NULL_PTR,
                message: "tk_config_create returned null pointer".to_string(),
            });
        }
        Ok(Self { lib, ptr })
    }

    /// Allocate a `tk_config` populated with defaults for `arch` (`tk_config_create_for_arch`).
    pub fn for_arch(lib: Arc<LibTalusKernels>, arch: &GpuArch) -> Result<Self> {
        let mut ptr: *mut tk_config = core::ptr::null_mut();
        let st = unsafe { (lib.tk_config_create_for_arch)(gpu_arch_to_tk(arch), &mut ptr) };
        translate_tk_status(&lib, core::ptr::null(), st)?;
        if ptr.is_null() {
            return Err(CoreError::KernelError {
                code: TK_ERR_NULL_PTR,
                message: "tk_config_create_for_arch returned null pointer".to_string(),
            });
        }
        Ok(Self { lib, ptr })
    }

    /// Populate this config from `STRATA_*` / `TALUS_*` environment variables (`tk_config_from_env`).
    pub fn load_from_env(&mut self) -> Result<()> {
        let st = unsafe { (self.lib.tk_config_from_env)(self.ptr) };
        translate_tk_status(&self.lib, core::ptr::null(), st)
    }

    /// Set a named integer tuning parameter (`tk_config_set_int`).
    pub fn set_int(&mut self, key: &str, value: i64) -> Result<()> {
        let c_key =
            CString::new(key).map_err(|_| invalid_arg("config key contains interior NUL"))?;
        let st = unsafe { (self.lib.tk_config_set_int)(self.ptr, c_key.as_ptr(), value) };
        translate_tk_status(&self.lib, core::ptr::null(), st)
    }

    /// Query a named integer tuning parameter (`tk_config_get_int`).
    pub fn get_int(&self, key: &str) -> Result<i64> {
        let c_key =
            CString::new(key).map_err(|_| invalid_arg("config key contains interior NUL"))?;
        let mut val = 0i64;
        let st = unsafe { (self.lib.tk_config_get_int)(self.ptr, c_key.as_ptr(), &mut val) };
        translate_tk_status(&self.lib, core::ptr::null(), st)?;
        Ok(val)
    }

    /// Raw `const tk_config*` pointer.
    #[inline]
    pub const fn as_ptr(&self) -> *const tk_config {
        self.ptr
    }
}

impl Drop for FfiConfig {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                (self.lib.tk_config_destroy)(self.ptr);
            }
            self.ptr = core::ptr::null_mut();
        }
    }
}

/// Safe RAII wrapper around `tk_packed_weights*` (`tk_pack_q8` / `tk_pack_q6` / `tk_packed_free`).
pub struct FfiPackedWeights {
    lib: Arc<LibTalusKernels>,
    ctx: *mut tk_ctx,
    ptr: *mut tk_packed_weights,
}

// Safety: `tk_packed_weights` is an immutable device allocation handle once created.
unsafe impl Send for FfiPackedWeights {}
unsafe impl Sync for FfiPackedWeights {}

impl FfiPackedWeights {
    /// Raw `const tk_packed_weights*` pointer.
    #[inline]
    pub const fn as_ptr(&self) -> *const tk_packed_weights {
        self.ptr
    }
}

impl Drop for FfiPackedWeights {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                (self.lib.tk_packed_free)(self.ctx, self.ptr);
            }
            self.ptr = core::ptr::null_mut();
        }
    }
}

/// Safe FFI kernel backend wrapping `LibTalusKernels` and `tk_ctx*`.
pub struct FfiKernelBackend {
    lib: Arc<LibTalusKernels>,
    ctx: *mut tk_ctx,
    device_id: i32,
    arch: GpuArch,
    stream: StreamHandle,
}

// Safety: `tk_ctx` operations are thread-safe per context/stream when synchronized through `KernelBackend`.
unsafe impl Send for FfiKernelBackend {}
unsafe impl Sync for FfiKernelBackend {}

impl FfiKernelBackend {
    /// Load `libtalus_kernels.so` from `lib_path` and create a `tk_ctx` on `device_id`.
    pub fn open(
        lib_path: impl AsRef<Path>,
        device_id: i32,
        config: Option<&FfiConfig>,
        stream: StreamHandle,
    ) -> Result<Self> {
        let lib = Arc::new(LibTalusKernels::load(lib_path.as_ref()).map_err(io_err)?);
        Self::new(lib, device_id, config, stream)
    }

    /// Load `libtalus_kernels.so` from `TALUS_KERNELS_LIB`, `TALUS_KERNELS_PREBUILT`, or the
    /// default dynamic linker search path and create a `tk_ctx` on `device_id`.
    #[cfg(unix)]
    pub fn open_default(
        device_id: i32,
        config: Option<&FfiConfig>,
        stream: StreamHandle,
    ) -> Result<Self> {
        let lib = Arc::new(LibTalusKernels::load_default().map_err(io_err)?);
        Self::new(lib, device_id, config, stream)
    }

    /// Create an [`FfiKernelBackend`] from an already-loaded [`LibTalusKernels`] table.
    pub fn new(
        lib: Arc<LibTalusKernels>,
        device_id: i32,
        config: Option<&FfiConfig>,
        stream: StreamHandle,
    ) -> Result<Self> {
        let cfg_ptr = config.map_or(core::ptr::null(), FfiConfig::as_ptr);
        let mut ctx: *mut tk_ctx = core::ptr::null_mut();
        let st = unsafe { (lib.tk_ctx_create)(device_id, cfg_ptr, &mut ctx) };
        translate_tk_status(&lib, ctx, st)?;
        if ctx.is_null() {
            return Err(CoreError::KernelError {
                code: TK_ERR_NULL_PTR,
                message: "tk_ctx_create returned null context".to_string(),
            });
        }

        let mut raw_dev = device_id;
        let st_dev = unsafe { (lib.tk_ctx_device)(ctx, &mut raw_dev) };
        if let Err(e) = translate_tk_status(&lib, ctx, st_dev) {
            unsafe { (lib.tk_ctx_destroy)(ctx) };
            return Err(e);
        }

        let mut raw_arch = TK_ARCH_GFX1151;
        let st_arch = unsafe { (lib.tk_ctx_arch)(ctx, &mut raw_arch) };
        if let Err(e) = translate_tk_status(&lib, ctx, st_arch) {
            unsafe { (lib.tk_ctx_destroy)(ctx) };
            return Err(e);
        }

        Ok(Self {
            lib,
            ctx,
            device_id: raw_dev,
            arch: tk_to_gpu_arch(raw_arch),
            stream,
        })
    }

    /// Return the underlying [`LibTalusKernels`] symbol table.
    #[inline]
    pub fn lib(&self) -> &Arc<LibTalusKernels> {
        &self.lib
    }

    /// Return the raw `tk_ctx*` handle.
    #[inline]
    pub const fn raw_ctx(&self) -> *mut tk_ctx {
        self.ctx
    }

    /// Return the HIP device ordinal bound to this context.
    #[inline]
    pub const fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Return the active [`StreamHandle`].
    #[inline]
    pub const fn stream(&self) -> StreamHandle {
        self.stream
    }

    /// Update the active [`StreamHandle`] used for subsequent kernel launches.
    #[inline]
    pub fn set_stream(&mut self, stream: StreamHandle) {
        self.stream = stream;
    }

    #[inline]
    fn raw_stream(&self) -> *mut c_void {
        self.stream.raw as *mut c_void
    }

    /// Translate a `tk_status` code using `tk_last_error(self.ctx)`.
    #[inline]
    pub fn check_status(&self, status: tk_status) -> Result<()> {
        translate_tk_status(&self.lib, self.ctx, status)
    }

    /// Launch `tk_rms_norm` on raw device pointers (`x_dev`, optional `w_dev`) using the active stream.
    ///
    /// # Safety
    /// `x_dev` must point to at least `rows * cols` valid device `f32` elements, and if non-null,
    /// `w_dev` must point to at least `cols` valid device `f32` elements on `self.device_id`.
    pub unsafe fn rms_norm_dev(
        &self,
        x_dev: *mut f32,
        w_dev: *const f32,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        let st = (self.lib.tk_rms_norm)(
            self.ctx,
            x_dev,
            w_dev,
            rows as i64,
            cols as i64,
            eps,
            self.raw_stream(),
        );
        self.check_status(st)
    }

    /// Launch out-of-place `tk_rms_norm_to` on raw device pointers (`dst_dev`, `x_dev`, `w_dev`)
    /// using the active stream.
    ///
    /// # Safety
    /// `dst_dev` and `x_dev` must point to at least `rows * cols` valid device `f32` elements,
    /// and if non-null, `w_dev` must point to at least `cols` valid device `f32` elements.
    pub unsafe fn rms_norm_to_dev(
        &self,
        dst_dev: *mut f32,
        x_dev: *const f32,
        w_dev: *const f32,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        let st = (self.lib.tk_rms_norm_to)(
            self.ctx,
            dst_dev,
            x_dev,
            w_dev,
            rows as i32,
            cols as i32,
            eps,
            self.raw_stream(),
        );
        self.check_status(st)
    }

    /// Launch `tk_silu_mul` (`gate[i] = silu(gate[i]) * up[i]`) on raw device pointers using the active stream.
    ///
    /// # Safety
    /// `gate_dev` and `up_dev` must each point to at least `n` valid device `f32` elements.
    pub unsafe fn silu_mul_dev(
        &self,
        gate_dev: *mut f32,
        up_dev: *const f32,
        n: usize,
    ) -> Result<()> {
        let st = (self.lib.tk_silu_mul)(self.ctx, gate_dev, up_dev, n as i64, self.raw_stream());
        self.check_status(st)
    }

    /// Launch `tk_axpy` (`y[i] += alpha * x[i]`) on raw device pointers using the active stream.
    ///
    /// # Safety
    /// `y_dev` and `x_dev` must each point to at least `n` valid device `f32` elements.
    pub unsafe fn axpy_dev(
        &self,
        y_dev: *mut f32,
        x_dev: *const f32,
        alpha: f32,
        n: usize,
    ) -> Result<()> {
        let st = (self.lib.tk_axpy)(self.ctx, y_dev, x_dev, alpha, n as i32, self.raw_stream());
        self.check_status(st)
    }

    /// Load a hipBLASLt tuning table into the C++ kernel context (`tk_ctx_load_tuning`).
    pub fn load_tuning(&self, path: impl AsRef<Path>) -> Result<()> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| invalid_arg("tuning path is not valid UTF-8"))?;
        let c_path =
            CString::new(path_str).map_err(|_| invalid_arg("tuning path contains interior NUL"))?;
        let st = unsafe { (self.lib.tk_ctx_load_tuning)(self.ctx, c_path.as_ptr()) };
        self.check_status(st)
    }

    /// Repack a `Q8_0` weight matrix into an explicit [`FfiPackedWeights`] handle (`tk_pack_q8`).
    pub fn pack_q8(&self, w_q8_0: &[u8], n_in: usize, n_out: usize) -> Result<FfiPackedWeights> {
        let expected = row_bytes(QuantType::Q8_0, n_in)? * n_out;
        if w_q8_0.len() < expected {
            return Err(invalid_arg(format!(
                "pack_q8: w_q8_0.len() ({}) < required {} bytes",
                w_q8_0.len(),
                expected
            )));
        }
        let mut ptr: *mut tk_packed_weights = core::ptr::null_mut();
        let st = unsafe {
            (self.lib.tk_pack_q8)(
                self.ctx,
                w_q8_0.as_ptr().cast::<c_void>(),
                n_in as i32,
                n_out as i32,
                self.raw_stream(),
                &mut ptr,
            )
        };
        self.check_status(st)?;
        Ok(FfiPackedWeights {
            lib: Arc::clone(&self.lib),
            ctx: self.ctx,
            ptr,
        })
    }

    /// Repack a `Q6_K` weight matrix into an explicit [`FfiPackedWeights`] handle (`tk_pack_q6`).
    pub fn pack_q6(&self, w_q6_k: &[u8], n_in: usize, n_out: usize) -> Result<FfiPackedWeights> {
        let expected = row_bytes(QuantType::Q6_K, n_in)? * n_out;
        if w_q6_k.len() < expected {
            return Err(invalid_arg(format!(
                "pack_q6: w_q6_k.len() ({}) < required {} bytes",
                w_q6_k.len(),
                expected
            )));
        }
        let mut ptr: *mut tk_packed_weights = core::ptr::null_mut();
        let st = unsafe {
            (self.lib.tk_pack_q6)(
                self.ctx,
                w_q6_k.as_ptr().cast::<c_void>(),
                n_in as i32,
                n_out as i32,
                self.raw_stream(),
                &mut ptr,
            )
        };
        self.check_status(st)?;
        Ok(FfiPackedWeights {
            lib: Arc::clone(&self.lib),
            ctx: self.ctx,
            ptr,
        })
    }

    /// Execute `tk_mmvq_packed` using a repacked [`FfiPackedWeights`] handle.
    pub fn mmvq_packed(
        &self,
        packed: &FfiPackedWeights,
        x_q8_1: &[u8],
        y: &mut [f32],
        ncols: usize,
    ) -> Result<()> {
        let st = unsafe {
            (self.lib.tk_mmvq_packed)(
                self.ctx,
                packed.as_ptr(),
                x_q8_1.as_ptr().cast::<c_void>(),
                y.as_mut_ptr(),
                ncols as i32,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }
}

impl Drop for FfiKernelBackend {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                (self.lib.tk_ctx_destroy)(self.ctx);
            }
            self.ctx = core::ptr::null_mut();
        }
    }
}

impl KernelBackend for FfiKernelBackend {
    #[inline]
    fn arch(&self) -> GpuArch {
        self.arch.clone()
    }

    fn rms_norm(
        &self,
        x: &mut [f32],
        w: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        if x.len() < rows.saturating_mul(cols) {
            return Err(invalid_arg("rms_norm: x buffer smaller than rows * cols"));
        }
        let w_ptr = match w {
            Some(slice) => {
                if slice.len() < cols {
                    return Err(invalid_arg("rms_norm: w buffer smaller than cols"));
                }
                slice.as_ptr()
            }
            None => core::ptr::null(),
        };
        let st = unsafe {
            (self.lib.tk_rms_norm)(
                self.ctx,
                x.as_mut_ptr(),
                w_ptr,
                rows as i64,
                cols as i64,
                eps,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn head_norm_rope(
        &self,
        x: &mut [f32],
        w: &[f32],
        rows: usize,
        eps: f32,
        n_rot: usize,
        cos_tab: &[f32],
        sin_tab: &[f32],
        pos: &[i32],
        pos_div: usize,
    ) -> Result<()> {
        let head_dim = w.len();
        if x.len() < rows.saturating_mul(head_dim) || pos.is_empty() {
            return Err(invalid_arg("head_norm_rope: invalid buffer dimensions"));
        }
        let st = unsafe {
            (self.lib.tk_head_norm_rope)(
                self.ctx,
                x.as_mut_ptr(),
                w.as_ptr(),
                rows as i32,
                eps,
                n_rot as i32,
                cos_tab.as_ptr(),
                sin_tab.as_ptr(),
                pos.as_ptr(),
                pos_div as i32,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn rope_neox(
        &self,
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        head_dim: usize,
        n_rot: usize,
        cos_tab: &[f32],
        sin_tab: &[f32],
        pos: &[i32],
    ) -> Result<()> {
        let total = rows.saturating_mul(head_dim);
        if x.len() < total || out.len() < total || pos.len() < rows {
            return Err(invalid_arg("rope_neox: invalid buffer dimensions"));
        }
        let st = unsafe {
            (self.lib.tk_rope_neox)(
                self.ctx,
                x.as_ptr(),
                out.as_mut_ptr(),
                rows as i64,
                head_dim as i32,
                n_rot as i32,
                cos_tab.as_ptr(),
                sin_tab.as_ptr(),
                pos.as_ptr(),
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn silu_mul(&self, gate: &mut [f32], up: &[f32], n: usize) -> Result<()> {
        if gate.len() < n || up.len() < n {
            return Err(invalid_arg("silu_mul: buffer length smaller than n"));
        }
        let st = unsafe {
            (self.lib.tk_silu_mul)(
                self.ctx,
                gate.as_mut_ptr(),
                up.as_ptr(),
                n as i64,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn axpy(&self, y: &mut [f32], x: &[f32], alpha: f32, n: usize) -> Result<()> {
        if y.len() < n || x.len() < n {
            return Err(invalid_arg("axpy: buffer length smaller than n"));
        }
        let st = unsafe {
            (self.lib.tk_axpy)(
                self.ctx,
                y.as_mut_ptr(),
                x.as_ptr(),
                alpha,
                n as i32,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn embed_q8_0(
        &self,
        table: &[u8],
        tokens: &[i32],
        out: &mut [f32],
        n_embd: usize,
    ) -> Result<()> {
        let n_tok = tokens.len();
        if out.len() < n_tok.saturating_mul(n_embd) {
            return Err(invalid_arg(
                "embed_q8_0: output buffer smaller than n_tok * n_embd",
            ));
        }
        let st = unsafe {
            (self.lib.tk_embed_q8_0)(
                self.ctx,
                table.as_ptr().cast::<c_void>(),
                tokens.as_ptr(),
                out.as_mut_ptr(),
                n_embd as i32,
                n_tok as i32,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn softplus_head_gate(
        &self,
        o: &mut [f32],
        g: &[f32],
        n_head: usize,
        head_dim: usize,
    ) -> Result<()> {
        if o.len() < n_head.saturating_mul(head_dim) || g.len() < n_head {
            return Err(invalid_arg("softplus_head_gate: invalid buffer dimensions"));
        }
        let st = unsafe {
            (self.lib.tk_softplus_head_gate)(
                self.ctx,
                o.as_mut_ptr(),
                g.as_ptr(),
                n_head as i32,
                head_dim as i32,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn quantize_q8_1(
        &self,
        x_f32: &[f32],
        x_q8_1: &mut [u8],
        n_in: usize,
        ncols: usize,
    ) -> Result<()> {
        let req_bytes = q8_1_bytes(n_in, ncols);
        if x_f32.len() < n_in.saturating_mul(ncols) || x_q8_1.len() < req_bytes {
            return Err(invalid_arg("quantize_q8_1: buffer length too small"));
        }
        let st = unsafe {
            (self.lib.tk_quantize_q8_1)(
                self.ctx,
                x_f32.as_ptr(),
                x_q8_1.as_mut_ptr().cast::<c_void>(),
                n_in as i32,
                ncols as i32,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn mmvq(
        &self,
        qtype: QuantType,
        w: &[u8],
        x_q8_1: &[u8],
        y: &mut [f32],
        n_in: usize,
        n_out: usize,
        ncols: usize,
    ) -> Result<()> {
        let req_w = row_bytes(qtype, n_in)?.saturating_mul(n_out);
        let req_x = q8_1_bytes(n_in, ncols);
        if w.len() < req_w || x_q8_1.len() < req_x || y.len() < n_out.saturating_mul(ncols) {
            return Err(invalid_arg("mmvq: buffer length too small"));
        }
        let st = unsafe {
            (self.lib.tk_mmvq)(
                self.ctx,
                quant_type_to_tk(qtype),
                w.as_ptr().cast::<c_void>(),
                x_q8_1.as_ptr().cast::<c_void>(),
                y.as_mut_ptr(),
                n_in as i32,
                n_out as i32,
                ncols as i32,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn mmvq_ids(&self, args: MmvqIdsArgs<'_>) -> Result<()> {
        let n_ids = args.ids.len();
        if args.y.len() < n_ids.saturating_mul(args.n_out) {
            return Err(invalid_arg(
                "mmvq_ids: output buffer y smaller than n_ids * n_out",
            ));
        }
        let (w2_ptr, y2_ptr) = match (args.w2, args.y2) {
            (Some(w2), Some(y2)) => {
                if y2.len() < n_ids.saturating_mul(args.n_out) {
                    return Err(invalid_arg(
                        "mmvq_ids: output buffer y2 smaller than n_ids * n_out",
                    ));
                }
                (w2.as_ptr().cast::<c_void>(), y2.as_mut_ptr())
            }
            (None, None) => (core::ptr::null(), core::ptr::null_mut()),
            _ => {
                return Err(invalid_arg(
                    "mmvq_ids: w2 and y2 must both be Some or both be None",
                ));
            }
        };
        let st = unsafe {
            (self.lib.tk_mmvq_ids)(
                self.ctx,
                quant_type_to_tk(args.qtype),
                args.w.as_ptr().cast::<c_void>(),
                args.expert_bytes,
                args.ids.as_ptr(),
                n_ids as i32,
                args.x_q8_1.as_ptr().cast::<c_void>(),
                args.x_per_id,
                args.y.as_mut_ptr(),
                args.n_in as i32,
                args.n_out as i32,
                w2_ptr,
                y2_ptr,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn f32_matvec(
        &self,
        w: &[f32],
        x: &[f32],
        y: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<()> {
        if w.len() < n_out.saturating_mul(n_in) || x.len() < n_in || y.len() < n_out {
            return Err(invalid_arg("f32_matvec: buffer length too small"));
        }
        let st = unsafe {
            (self.lib.tk_f32_matvec)(
                self.ctx,
                w.as_ptr(),
                x.as_ptr(),
                y.as_mut_ptr(),
                n_in as i32,
                n_out as i32,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn f32_matmul(
        &self,
        w: &[f32],
        x: &[f32],
        y: &mut [f32],
        n_in: usize,
        n_out: usize,
        n_tok: usize,
    ) -> Result<()> {
        if w.len() < n_out.saturating_mul(n_in)
            || x.len() < n_tok.saturating_mul(n_in)
            || y.len() < n_tok.saturating_mul(n_out)
        {
            return Err(invalid_arg("f32_matmul: buffer length too small"));
        }
        let st = unsafe {
            (self.lib.tk_f32_matmul)(
                self.ctx,
                w.as_ptr(),
                x.as_ptr(),
                y.as_mut_ptr(),
                n_in as i32,
                n_out as i32,
                n_tok as i32,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn route_sigmoid_topk(
        &self,
        logits: &[f32],
        bias: Option<&[f32]>,
        ids_out: &mut [i32],
        weights_out: &mut [f32],
        n_expert: usize,
        k: usize,
        scale: f32,
    ) -> Result<()> {
        if logits.len() < n_expert || ids_out.len() < k || weights_out.len() < k {
            return Err(invalid_arg("route_sigmoid_topk: buffer length too small"));
        }
        let bias_ptr = match bias {
            Some(b) => {
                if b.len() < n_expert {
                    return Err(invalid_arg(
                        "route_sigmoid_topk: bias slice smaller than n_expert",
                    ));
                }
                b.as_ptr()
            }
            None => core::ptr::null(),
        };
        let st = unsafe {
            (self.lib.tk_route_sigmoid_topk)(
                self.ctx,
                logits.as_ptr(),
                bias_ptr,
                ids_out.as_mut_ptr(),
                weights_out.as_mut_ptr(),
                n_expert as i32,
                k as i32,
                scale,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn moe_add_slots(
        &self,
        x: &mut [f32],
        shared: Option<&[f32]>,
        y: &[f32],
        map: Option<&[i32]>,
        weights: &[f32],
        n_tok: usize,
        n_embd: usize,
        k: usize,
    ) -> Result<()> {
        if x.len() < n_tok.saturating_mul(n_embd) || weights.len() < n_tok.saturating_mul(k) {
            return Err(invalid_arg("moe_add_slots: buffer length too small"));
        }
        let shared_ptr = match shared {
            Some(s) => {
                if s.len() < n_tok.saturating_mul(n_embd) {
                    return Err(invalid_arg(
                        "moe_add_slots: shared buffer smaller than n_tok * n_embd",
                    ));
                }
                s.as_ptr()
            }
            None => core::ptr::null(),
        };
        let map_ptr = match map {
            Some(m) => {
                if m.len() < n_tok.saturating_mul(k) {
                    return Err(invalid_arg(
                        "moe_add_slots: map buffer smaller than n_tok * k",
                    ));
                }
                m.as_ptr()
            }
            None => core::ptr::null(),
        };
        let st = unsafe {
            (self.lib.tk_moe_add_slots)(
                self.ctx,
                x.as_mut_ptr(),
                shared_ptr,
                y.as_ptr(),
                map_ptr,
                weights.as_ptr(),
                n_tok as i32,
                n_embd as i32,
                k as i32,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn argmax(
        &self,
        logits: &[f32],
        n_rows: usize,
        n_vocab: usize,
        out_tokens: &mut [i32],
    ) -> Result<()> {
        if logits.len() < n_rows.saturating_mul(n_vocab) || out_tokens.len() < n_rows {
            return Err(invalid_arg("argmax: buffer length too small"));
        }
        let st = unsafe {
            (self.lib.tk_argmax_rows)(
                self.ctx,
                logits.as_ptr(),
                n_rows as i32,
                n_vocab as i32,
                out_tokens.as_mut_ptr(),
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn attn_decode(&self, args: AttnDecodeArgs<'_>) -> Result<()> {
        if args.pos.is_empty() {
            return Err(invalid_arg("attn_decode: pos slice must not be empty"));
        }
        let k_scale_ptr = args.k_scale.map_or(core::ptr::null(), <[u16]>::as_ptr);
        let v_scale_ptr = args.v_scale.map_or(core::ptr::null(), <[u16]>::as_ptr);
        let scratch_ptr = args
            .scratch
            .map_or(core::ptr::null_mut(), <[f32]>::as_mut_ptr);

        if let Some(sinks) = args.sinks {
            let st = unsafe {
                (self.lib.tk_attn_swa_sinks)(
                    self.ctx,
                    args.q.as_ptr(),
                    args.k_pool.as_ptr().cast::<c_void>(),
                    args.v_pool.as_ptr().cast::<c_void>(),
                    k_scale_ptr,
                    v_scale_ptr,
                    sinks.as_ptr(),
                    args.out.as_mut_ptr(),
                    scratch_ptr,
                    args.n_tok as i32,
                    args.n_head as i32,
                    args.n_kv as i32,
                    args.head_dim as i32,
                    args.head_dim_v as i32,
                    args.window,
                    args.kv_stride as i32,
                    args.scale,
                    args.pos.as_ptr(),
                    args.n_split,
                    self.raw_stream(),
                )
            };
            return self.check_status(st);
        }

        let st = unsafe {
            (self.lib.tk_attn_decode)(
                self.ctx,
                args.q.as_ptr(),
                args.k_pool.as_ptr().cast::<c_void>(),
                args.v_pool.as_ptr().cast::<c_void>(),
                k_scale_ptr,
                v_scale_ptr,
                args.out.as_mut_ptr(),
                scratch_ptr,
                args.n_tok as i32,
                args.n_head as i32,
                args.n_kv as i32,
                args.head_dim as i32,
                args.head_dim_v as i32,
                args.kv_stride as i32,
                args.window,
                args.scale,
                args.pos.as_ptr(),
                args.n_split,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn attn_prefill(&self, args: AttnPrefillArgs<'_>) -> Result<()> {
        let st = unsafe {
            (self.lib.tk_attn_prefill_fa)(
                self.ctx,
                args.q.as_ptr(),
                args.k_pool.as_ptr().cast::<c_void>(),
                args.v_pool.as_ptr().cast::<c_void>(),
                args.out.as_mut_ptr(),
                args.n_tok as i32,
                args.n_head as i32,
                args.n_kv as i32,
                args.head_dim as i32,
                args.kv_stride as i64,
                args.p0 as i64,
                args.scale,
                core::ptr::null_mut(),
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn gdn_step(&self, args: GdnStepArgs<'_>) -> Result<()> {
        let st = unsafe {
            (self.lib.tk_gdn_step)(
                self.ctx,
                args.state.as_mut_ptr(),
                args.q.as_ptr(),
                args.k.as_ptr(),
                args.v.as_ptr(),
                args.gate.as_ptr(),
                args.beta.as_ptr(),
                args.out.as_mut_ptr(),
                args.state_size as i64,
                args.h_k as i64,
                args.h_v as i64,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn gdn_seq(&self, args: GdnSeqArgs<'_>) -> Result<()> {
        let st = unsafe {
            (self.lib.tk_gdn_seq)(
                self.ctx,
                args.state.as_mut_ptr(),
                args.q.as_ptr(),
                args.k.as_ptr(),
                args.v.as_ptr(),
                args.gate.as_ptr(),
                args.beta.as_ptr(),
                args.z.as_ptr(),
                args.gamma.as_ptr(),
                args.out.as_mut_ptr(),
                args.n_tok as i32,
                args.state_size as i64,
                args.h_k as i64,
                args.h_v as i64,
                args.norm_eps,
                args.silu_gate,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn prefill_gemm(&self, args: PrefillGemmArgs<'_>) -> Result<()> {
        let st = unsafe {
            (self.lib.tk_prefill_gemm)(
                self.ctx,
                quant_type_to_tk(args.w_qtype),
                args.w.as_ptr().cast::<c_void>(),
                args.x.as_ptr().cast::<c_void>(),
                quant_type_to_tk(args.x_qtype),
                args.y.as_mut_ptr(),
                args.n_tok as i64,
                args.n_out as i64,
                args.n_in as i64,
                args.ldy as i64,
                args.ldx as i64,
                args.beta,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }

    fn prefill_mmq(&self, args: PrefillMmqArgs<'_>) -> Result<()> {
        let st = unsafe {
            (self.lib.tk_prefill_mmq)(
                self.ctx,
                quant_type_to_tk(args.w_qtype),
                args.w_blocks.as_ptr().cast::<c_void>(),
                args.x_q8_1.as_ptr().cast::<c_void>(),
                args.y.as_mut_ptr(),
                args.n_tok as i64,
                args.n_out as i64,
                args.n_in as i64,
                args.ldy as i64,
                self.raw_stream(),
            )
        };
        self.check_status(st)
    }
}
