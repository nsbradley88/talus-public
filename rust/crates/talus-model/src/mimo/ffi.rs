//! Foreign Function Interface to `libtalus_kernels` for the MiMo-V2.6-Flash (`mimo2`) model.
//!
//! All kernel entry points are declared in `include/talus_kernels.h` (Contract v0) and
//! bound in `talus_kernels_sys::LibTalusKernels`. This module re-exports the subset required
//! by the `mimo2` decode, verification, and prefill pipelines:
//! - Hybrid attention: `tk_attn_swa_sinks` (40 SWA-128 layers with learned sinks) and
//!   `tk_attn_k192_v128` (8 full-attention K192/V128 layers).
//! - 256-expert top-8 MoE: `tk_moe_topk`, `tk_mmvq_ids` (routes `IQ2_XXS`, `IQ3_XXS`,
//!   `IQ2_XS` to `mimo_mmvq_ids` 4 rows/WG), `tk_prefill_moe_mmq` (`mimo_prefill_moe_mmq`),
//!   and `tk_moe_swiglu_down` / `tk_scale_accum`.
//! - Norms, RoPE, activations, and KV cache: `tk_rms_norm_to`, `tk_head_norm_rope`,
//!   `tk_kv_store_f16`, `tk_kv_store_q8`, `tk_swiglu`, `tk_add_inplace`, `tk_mmvq`, `tk_argmax`.

pub use crate::laguna::ffi::{
    load_talus_kernels, tk_build_rope_table, HipGraph, HipGraphExec, HipRuntime, RopeScalingParams,
    TkContext, HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL, TK_ROPE_SCALING_LINEAR, TK_ROPE_SCALING_NONE,
    TK_ROPE_SCALING_YARN,
};
pub use talus_kernels_sys::{
    tk_config, tk_ctx, tk_moe_mmq_product, tk_qtype, tk_status, tk_stream, LibTalusKernels,
    TK_ERR_HIP, TK_ERR_INVALID_ARG, TK_ERR_OUT_OF_MEMORY, TK_ERR_UNSUPPORTED, TK_OK, TK_QTYPE_BF16,
    TK_QTYPE_F16, TK_QTYPE_F32, TK_QTYPE_IQ1_M, TK_QTYPE_IQ1_S, TK_QTYPE_IQ2_S, TK_QTYPE_IQ2_XS,
    TK_QTYPE_IQ2_XXS, TK_QTYPE_IQ3_S, TK_QTYPE_IQ3_XXS, TK_QTYPE_IQ4_NL, TK_QTYPE_IQ4_XS,
    TK_QTYPE_Q2_K, TK_QTYPE_Q3_K, TK_QTYPE_Q4_0, TK_QTYPE_Q4_1, TK_QTYPE_Q4_K, TK_QTYPE_Q5_0,
    TK_QTYPE_Q5_1, TK_QTYPE_Q5_K, TK_QTYPE_Q6_K, TK_QTYPE_Q8_0, TK_QTYPE_Q8_1, TK_QTYPE_Q8_K,
};

use std::ffi::c_void;
use talus_core_api::{CoreError, Result};

/// Convert a raw `tk_status` code from `LibTalusKernels` into a `talus_core_api::Result<()>`.
#[inline]
pub fn check_tk(status: tk_status, op: &'static str) -> Result<()> {
    if status == TK_OK {
        Ok(())
    } else {
        Err(CoreError::KernelError {
            code: status,
            message: format!("{op} failed: tk_status={status}"),
        })
    }
}

/// Asynchronous host-to-device transfer from a pinned staging buffer (`hipMemcpyAsync` + stream sync).
pub fn copy_h2d_async(
    hip: &HipRuntime,
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    stream: tk_stream,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    let st = stream as hip_sys::hipStream_t;
    let err = unsafe { (hip.hipMemcpyAsync)(dst, src, count, HipRuntime::MEMCPY_H2D, st) };
    if err == 0 {
        let sync_err = unsafe { (hip.hipStreamSynchronize)(st) };
        if sync_err == 0 {
            return Ok(());
        }
    }
    hip.copy_h2d(dst, src, count)
}

/// Fill `bytes` of device memory at `dst` with `val` via `hipMemset`.
#[inline]
pub fn hip_memset(hip: &HipRuntime, dst: *mut c_void, val: i32, bytes: usize) -> Result<()> {
    if bytes == 0 || dst.is_null() {
        return Ok(());
    }
    let err = unsafe { (hip.hipMemset)(dst, val, bytes) };
    if err != 0 {
        return Err(CoreError::HipError {
            code: err,
            message: format!("hipMemset({bytes} B) failed: {err}"),
        });
    }
    Ok(())
}

/// Fill `bytes` of device memory at `dst` with `val` on `stream` via `hipMemsetAsync`.
#[inline]
pub fn hip_memset_async(
    hip: &HipRuntime,
    dst: *mut c_void,
    val: i32,
    bytes: usize,
    stream: tk_stream,
) -> Result<()> {
    if bytes == 0 || dst.is_null() {
        return Ok(());
    }
    let err = unsafe { (hip.hipMemsetAsync)(dst, val, bytes, stream as hip_sys::hipStream_t) };
    if err != 0 {
        return Err(CoreError::HipError {
            code: err,
            message: format!("hipMemsetAsync({bytes} B) failed: {err}"),
        });
    }
    Ok(())
}

/// Device-to-device memory copy via `hipMemcpy(MEMCPY_D2D)`.
#[inline]
pub fn hip_copy_d2d(
    hip: &HipRuntime,
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    let err = unsafe { (hip.hipMemcpy)(dst, src, bytes, HipRuntime::MEMCPY_D2D) };
    if err != 0 {
        return Err(CoreError::HipError {
            code: err,
            message: format!("hipMemcpy D2D ({bytes} B) failed: {err}"),
        });
    }
    Ok(())
}

/// Asynchronous device-to-device memory copy on `stream` via `hipMemcpyAsync(MEMCPY_D2D)`.
#[inline]
pub fn hip_copy_d2d_async(
    hip: &HipRuntime,
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    stream: tk_stream,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    let err = unsafe {
        (hip.hipMemcpyAsync)(
            dst,
            src,
            bytes,
            HipRuntime::MEMCPY_D2D,
            stream as hip_sys::hipStream_t,
        )
    };
    if err != 0 {
        return Err(CoreError::HipError {
            code: err,
            message: format!("hipMemcpyAsync D2D ({bytes} B) failed: {err}"),
        });
    }
    Ok(())
}

/// Build `[max_pos, n_rot / 2]` RoPE cos and sin tables on the host for `mimo2` (`n_rot = 64`).
pub fn build_mimo_rope_table(max_pos: usize, n_rot: usize, freq_base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = n_rot / 2;
    let mut cos_tab = vec![0.0f32; max_pos * half];
    let mut sin_tab = vec![0.0f32; max_pos * half];
    let sc = RopeScalingParams {
        scaling_type: TK_ROPE_SCALING_NONE,
        freq_base: f64::from(freq_base),
        ..Default::default()
    };
    tk_build_rope_table(&mut cos_tab, &mut sin_tab, max_pos, n_rot as i32, &sc);
    (cos_tab, sin_tab)
}
