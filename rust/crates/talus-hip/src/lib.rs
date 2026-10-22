//! Safe RAII HIP runtime wrappers (`Device`, `Stream`, `Event`, `Graph`, `GraphExec`,
//! `CaptureGuard`, `DeviceBuf<T>`, `HostMapped<T>`) carrying [`DeviceId`] from `talus-core-api`.

use std::ffi::{c_int, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::slice;

use hip_sys::{
    hipDeviceProp_t, hipDeviceSynchronize, hipDevice_t, hipErrorNotReady, hipError_t,
    hipEventCreateWithFlags, hipEventDefault, hipEventDestroy, hipEventDisableTiming,
    hipEventElapsedTime, hipEventQuery, hipEventRecord, hipEventSynchronize, hipEvent_t, hipFree,
    hipGetDeviceCount, hipGetDeviceProperties, hipGetLastError, hipGraphDestroy,
    hipGraphExecDestroy, hipGraphExec_t, hipGraphGetNodes, hipGraphInstantiate, hipGraphLaunch,
    hipGraph_t, hipHostFree, hipHostGetDevicePointer, hipHostMalloc, hipHostMallocMapped, hipInit,
    hipMalloc, hipMemGetInfo, hipMemcpy, hipMemcpyAsync, hipMemcpyDeviceToDevice,
    hipMemcpyDeviceToHost, hipMemcpyHostToDevice, hipMemset, hipMemsetAsync, hipSetDevice,
    hipStreamBeginCapture, hipStreamCaptureMode, hipStreamCaptureModeGlobal,
    hipStreamCaptureModeRelaxed, hipStreamCaptureModeThreadLocal, hipStreamCaptureStatus,
    hipStreamCaptureStatusActive, hipStreamCaptureStatusNone, hipStreamCreateWithFlags,
    hipStreamDefault, hipStreamDestroy, hipStreamEndCapture, hipStreamIsCapturing,
    hipStreamNonBlocking, hipStreamQuery, hipStreamSynchronize, hipStreamWaitEvent, hipStream_t,
    hipSuccess,
};
pub use talus_core_api::{ArchTraits, CoreError, DeviceId, GpuArch, StreamHandle, WmmaLayout};

/// Crate identifier.
pub const CRATE_NAME: &str = "talus-hip";

// ============================================================================
// Errors & Result
// ============================================================================

/// Error returned by safe `talus-hip` wrappers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HipError {
    /// Underlying HIP runtime call returned a non-success `hipError_t`.
    Runtime {
        code: i32,
        name: &'static str,
        message: String,
        context: &'static str,
    },
    /// Operation attempted across mismatched [`DeviceId`]s (Plan §6c).
    DeviceMismatch {
        expected: DeviceId,
        actual: DeviceId,
        operation: &'static str,
    },
    /// Slice or buffer element count mismatch in a memory transfer.
    LengthMismatch {
        expected: usize,
        actual: usize,
        operation: &'static str,
    },
    /// Caller supplied an invalid argument (e.g. zero-sized type `T` or overflow).
    InvalidArgument { reason: String },
}

impl HipError {
    /// Construct a [`HipError::Runtime`] from a raw `hipError_t` and call-site label.
    pub fn from_raw(err: hipError_t, context: &'static str) -> Self {
        Self::Runtime {
            code: err.code(),
            name: err.name_str(),
            message: hip_sys::error_string(err),
            context,
        }
    }

    /// Return the raw HIP status code if this is a [`HipError::Runtime`].
    pub const fn code(&self) -> Option<i32> {
        match self {
            Self::Runtime { code, .. } => Some(*code),
            _ => None,
        }
    }

    /// True if this error is a [`HipError::DeviceMismatch`].
    pub const fn is_device_mismatch(&self) -> bool {
        matches!(self, Self::DeviceMismatch { .. })
    }

    /// True if this error is a [`HipError::LengthMismatch`].
    pub const fn is_length_mismatch(&self) -> bool {
        matches!(self, Self::LengthMismatch { .. })
    }
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime {
                code,
                name,
                message,
                context,
            } => write!(f, "{context}: {name} (code {code}): {message}"),
            Self::DeviceMismatch {
                expected,
                actual,
                operation,
            } => write!(
                f,
                "{operation}: device mismatch (expected {:?}, got {:?})",
                expected, actual
            ),
            Self::LengthMismatch {
                expected,
                actual,
                operation,
            } => write!(
                f,
                "{operation}: length mismatch (expected {expected} elements, got {actual})"
            ),
            Self::InvalidArgument { reason } => write!(f, "invalid argument: {reason}"),
        }
    }
}

impl std::error::Error for HipError {}

impl From<HipError> for CoreError {
    fn from(err: HipError) -> Self {
        match err {
            HipError::Runtime {
                code,
                message,
                context,
                ..
            } => CoreError::HipError {
                code,
                message: format!("{context}: {message}"),
            },
            HipError::DeviceMismatch { .. }
            | HipError::LengthMismatch { .. }
            | HipError::InvalidArgument { .. } => CoreError::InvalidArgument {
                reason: err.to_string(),
            },
        }
    }
}

/// Standard result alias for `talus-hip`.
pub type Result<T> = std::result::Result<T, HipError>;

#[inline]
fn check_hip(err: hipError_t, context: &'static str) -> Result<()> {
    if err == hipSuccess {
        Ok(())
    } else {
        Err(HipError::from_raw(err, context))
    }
}

#[inline]
fn ensure_same_device(expected: DeviceId, actual: DeviceId, operation: &'static str) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(HipError::DeviceMismatch {
            expected,
            actual,
            operation,
        })
    }
}

#[inline]
fn set_active_device(device: DeviceId) -> Result<()> {
    let ordinal = c_int::try_from(device.0).map_err(|_| HipError::InvalidArgument {
        reason: format!("device id {} overflows c_int", device.0),
    })?;
    check_hip(unsafe { hipSetDevice(ordinal) }, "hipSetDevice")
}

// ============================================================================
// Device
// ============================================================================

/// Safe HIP device handle carrying [`DeviceId`], parsed [`GpuArch`], and [`ArchTraits`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    id: DeviceId,
    arch: GpuArch,
    traits: ArchTraits,
    name: String,
    gcn_arch_name: String,
    total_mem: usize,
    warp_size: u32,
    multi_processor_count: u32,
    integrated: bool,
}

impl Device {
    /// Initialize the HIP runtime on `id`, activate the device, and query its architecture and traits.
    pub fn new(id: DeviceId) -> Result<Self> {
        check_hip(unsafe { hipInit(0) }, "hipInit")?;
        set_active_device(id)?;
        let ordinal = id.0 as hipDevice_t;
        let mut prop = hipDeviceProp_t::default();
        check_hip(
            unsafe { hipGetDeviceProperties(&mut prop, ordinal) },
            "hipGetDeviceProperties",
        )?;

        let name = prop.name_str().to_string();
        let gcn_arch_name = prop.gcn_arch_name_str().to_string();
        let arch = GpuArch::from_gcn_arch_name(&gcn_arch_name);
        let traits = ArchTraits::for_arch(&arch);

        Ok(Self {
            id,
            arch,
            traits,
            name,
            gcn_arch_name,
            total_mem: prop.totalGlobalMem,
            warp_size: prop.warpSize.max(0) as u32,
            multi_processor_count: prop.multiProcessorCount.max(0) as u32,
            integrated: prop.integrated != 0,
        })
    }

    /// Query the number of available HIP devices.
    pub fn count() -> Result<usize> {
        check_hip(unsafe { hipInit(0) }, "hipInit")?;
        let mut count: c_int = 0;
        check_hip(
            unsafe { hipGetDeviceCount(&mut count) },
            "hipGetDeviceCount",
        )?;
        Ok(count.max(0) as usize)
    }

    /// Logical device identifier (`DeviceId`).
    pub const fn device_id(&self) -> DeviceId {
        self.id
    }

    /// Target GPU architecture (`GpuArch::Gfx1151`, `GpuArch::Gfx1201`, or `GpuArch::Other`).
    pub fn arch(&self) -> GpuArch {
        self.arch.clone()
    }

    /// Hardware capabilities and cache hierarchy traits (`ArchTraits`).
    pub const fn arch_traits(&self) -> ArchTraits {
        self.traits
    }

    /// Human-readable device name from `hipDeviceProp_t::name`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Raw `hipDeviceProp_t::gcnArchName` string (including target feature flags).
    pub fn gcn_arch_name(&self) -> &str {
        &self.gcn_arch_name
    }

    /// Hardware wavefront size (`warpSize`).
    pub const fn warp_size(&self) -> u32 {
        self.warp_size
    }

    /// Compute unit / workgroup processor count (`multiProcessorCount`).
    pub const fn multi_processor_count(&self) -> u32 {
        self.multi_processor_count
    }

    /// Total global memory reported by `hipDeviceProp_t::totalGlobalMem` in bytes.
    pub const fn total_memory(&self) -> usize {
        self.total_mem
    }

    /// True if this is an integrated APU sharing unified memory with the host (e.g. Strix Halo).
    pub const fn integrated(&self) -> bool {
        self.integrated
    }

    /// Bind this device as the calling thread's active HIP device (`hipSetDevice`).
    pub fn make_current(&self) -> Result<()> {
        set_active_device(self.id)
    }

    /// Block until all preceding commands on this device have completed (`hipDeviceSynchronize`).
    pub fn synchronize(&self) -> Result<()> {
        set_active_device(self.id)?;
        check_hip(unsafe { hipDeviceSynchronize() }, "hipDeviceSynchronize")
    }

    /// Query `(free_bytes, total_bytes)` on this device via `hipMemGetInfo`.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        set_active_device(self.id)?;
        let (mut free, mut total) = (0usize, 0usize);
        check_hip(
            unsafe { hipMemGetInfo(&mut free, &mut total) },
            "hipMemGetInfo",
        )?;
        Ok((free, total))
    }
}

// ============================================================================
// Stream & Event
// ============================================================================

/// Stream capture mode for [`Stream::begin_capture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CaptureMode {
    Global,
    #[default]
    ThreadLocal,
    Relaxed,
}

impl CaptureMode {
    pub const fn to_raw(self) -> hipStreamCaptureMode {
        match self {
            Self::Global => hipStreamCaptureModeGlobal,
            Self::ThreadLocal => hipStreamCaptureModeThreadLocal,
            Self::Relaxed => hipStreamCaptureModeRelaxed,
        }
    }
}

/// Safe RAII wrapper around a `hipStream_t` owned by a specific [`DeviceId`].
#[derive(Debug)]
pub struct Stream {
    raw: hipStream_t,
    device: DeviceId,
}

// SAFETY: `hipStream_t` handles are thread-safe in the HIP runtime.
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// Create a default-flags HIP stream on `device` (`hipStreamCreateWithFlags(..., hipStreamDefault)`).
    pub fn new(device: DeviceId) -> Result<Self> {
        Self::with_flags(device, hipStreamDefault)
    }

    /// Create a non-blocking HIP stream on `device` (`hipStreamCreateWithFlags(..., hipStreamNonBlocking)`).
    pub fn non_blocking(device: DeviceId) -> Result<Self> {
        Self::with_flags(device, hipStreamNonBlocking)
    }

    fn with_flags(device: DeviceId, flags: u32) -> Result<Self> {
        check_hip(unsafe { hipInit(0) }, "hipInit")?;
        set_active_device(device)?;
        let mut raw: hipStream_t = ptr::null_mut();
        check_hip(
            unsafe { hipStreamCreateWithFlags(&mut raw, flags) },
            "hipStreamCreateWithFlags",
        )?;
        Ok(Self { raw, device })
    }

    /// Device owning this stream.
    pub const fn device_id(&self) -> DeviceId {
        self.device
    }

    /// Underlying `hipStream_t` pointer.
    pub const fn raw(&self) -> hipStream_t {
        self.raw
    }

    /// Convert to a `talus_core_api::StreamHandle` for `ModelAdapter` / `StageLink` calls.
    pub fn as_core_handle(&self) -> StreamHandle {
        StreamHandle {
            raw: self.raw as usize as u64,
            device: self.device,
        }
    }

    /// Synchronize the host thread with all work queued on this stream (`hipStreamSynchronize`).
    pub fn synchronize(&self) -> Result<()> {
        check_hip(
            unsafe { hipStreamSynchronize(self.raw) },
            "hipStreamSynchronize",
        )
    }

    /// Poll whether all work queued on this stream has completed (`hipStreamQuery`).
    pub fn query(&self) -> Result<bool> {
        let rc = unsafe { hipStreamQuery(self.raw) };
        if rc == hipSuccess {
            Ok(true)
        } else if rc == hipErrorNotReady {
            Ok(false)
        } else {
            Err(HipError::from_raw(rc, "hipStreamQuery"))
        }
    }

    /// Make future work submitted to this stream wait on `event` (`hipStreamWaitEvent`).
    ///
    /// Returns [`HipError::DeviceMismatch`] if `event` belongs to a different [`DeviceId`].
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        ensure_same_device(self.device, event.device, "Stream::wait_event")?;
        check_hip(
            unsafe { hipStreamWaitEvent(self.raw, event.raw, 0) },
            "hipStreamWaitEvent",
        )
    }

    /// Query the raw `hipStreamCaptureStatus` of this stream.
    pub fn capture_status(&self) -> Result<hipStreamCaptureStatus> {
        let mut status = hipStreamCaptureStatusNone;
        check_hip(
            unsafe { hipStreamIsCapturing(self.raw, &mut status) },
            "hipStreamIsCapturing",
        )?;
        Ok(status)
    }

    /// Return `true` if this stream is currently in an active graph capture sequence.
    pub fn is_capturing(&self) -> bool {
        matches!(self.capture_status(), Ok(hipStreamCaptureStatusActive))
    }

    /// Begin HIP graph capture on this stream, returning an RAII [`CaptureGuard`].
    ///
    /// If the returned [`CaptureGuard`] is dropped without calling [`CaptureGuard::finish`],
    /// its `Drop` implementation ends capture and destroys the incomplete graph so the stream
    /// is never left in a corrupted capturing state.
    pub fn begin_capture(&self, mode: CaptureMode) -> Result<CaptureGuard<'_>> {
        set_active_device(self.device)?;
        check_hip(
            unsafe { hipStreamBeginCapture(self.raw, mode.to_raw()) },
            "hipStreamBeginCapture",
        )?;
        Ok(CaptureGuard {
            stream: self,
            active: true,
        })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = hipStreamDestroy(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}

/// Safe RAII wrapper around a `hipEvent_t` owned by a specific [`DeviceId`].
#[derive(Debug)]
pub struct Event {
    raw: hipEvent_t,
    device: DeviceId,
    timing_enabled: bool,
}

// SAFETY: `hipEvent_t` handles are thread-safe in the HIP runtime.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    /// Create a timing-enabled HIP event on `device` (`hipEventDefault`).
    pub fn new(device: DeviceId) -> Result<Self> {
        Self::with_flags(device, hipEventDefault, true)
    }

    /// Create a lightweight synchronization-only event with timing disabled (`hipEventDisableTiming`).
    pub fn disable_timing(device: DeviceId) -> Result<Self> {
        Self::with_flags(device, hipEventDisableTiming, false)
    }

    fn with_flags(device: DeviceId, flags: u32, timing_enabled: bool) -> Result<Self> {
        check_hip(unsafe { hipInit(0) }, "hipInit")?;
        set_active_device(device)?;
        let mut raw: hipEvent_t = ptr::null_mut();
        check_hip(
            unsafe { hipEventCreateWithFlags(&mut raw, flags) },
            "hipEventCreateWithFlags",
        )?;
        Ok(Self {
            raw,
            device,
            timing_enabled,
        })
    }

    /// Device owning this event.
    pub const fn device_id(&self) -> DeviceId {
        self.device
    }

    /// Underlying `hipEvent_t` pointer.
    pub const fn raw(&self) -> hipEvent_t {
        self.raw
    }

    /// True if timing was enabled when creating this event.
    pub const fn timing_enabled(&self) -> bool {
        self.timing_enabled
    }

    /// Record this event on `stream` (`hipEventRecord`).
    ///
    /// Enforces that `stream` belongs to the same [`DeviceId`].
    pub fn record(&self, stream: &Stream) -> Result<()> {
        ensure_same_device(self.device, stream.device, "Event::record")?;
        check_hip(
            unsafe { hipEventRecord(self.raw, stream.raw) },
            "hipEventRecord",
        )
    }

    /// Block the host thread until this event completes (`hipEventSynchronize`).
    pub fn synchronize(&self) -> Result<()> {
        check_hip(
            unsafe { hipEventSynchronize(self.raw) },
            "hipEventSynchronize",
        )
    }

    /// Non-blocking query returning `Ok(true)` if the event has completed (`hipEventQuery`).
    pub fn query(&self) -> Result<bool> {
        let rc = unsafe { hipEventQuery(self.raw) };
        if rc == hipSuccess {
            Ok(true)
        } else if rc == hipErrorNotReady {
            Ok(false)
        } else {
            Err(HipError::from_raw(rc, "hipEventQuery"))
        }
    }

    /// Compute elapsed time in milliseconds between `self` and `other` (`hipEventElapsedTime`).
    ///
    /// Enforces that both events belong to the same [`DeviceId`] and have timing enabled.
    pub fn elapsed_ms(&self, other: &Event) -> Result<f32> {
        ensure_same_device(self.device, other.device, "Event::elapsed_ms")?;
        let mut ms: f32 = 0.0;
        check_hip(
            unsafe { hipEventElapsedTime(&mut ms, self.raw, other.raw) },
            "hipEventElapsedTime",
        )?;
        Ok(ms.abs())
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = hipEventDestroy(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}

// ============================================================================
// Graph, GraphExec, and CaptureGuard
// ============================================================================

/// RAII stream capture guard returned by [`Stream::begin_capture`].
///
/// Calling [`Self::finish`] ends capture and returns the recorded [`Graph`].
/// Dropping the guard without calling [`Self::finish`] aborts the capture, destroys any
/// incomplete `hipGraph_t`, and clears sticky thread-local errors so the underlying [`Stream`]
/// is immediately reusable.
#[derive(Debug)]
pub struct CaptureGuard<'s> {
    stream: &'s Stream,
    active: bool,
}

impl<'s> CaptureGuard<'s> {
    /// Device owning the capturing stream.
    pub const fn device_id(&self) -> DeviceId {
        self.stream.device_id()
    }

    /// Borrow the underlying capturing [`Stream`].
    pub const fn stream(&self) -> &'s Stream {
        self.stream
    }

    /// End stream capture and return the recorded [`Graph`].
    pub fn finish(mut self) -> Result<Graph> {
        self.active = false;
        let mut raw_graph: hipGraph_t = ptr::null_mut();
        let rc = unsafe { hipStreamEndCapture(self.stream.raw, &mut raw_graph) };
        if rc != hipSuccess {
            if !raw_graph.is_null() {
                unsafe {
                    let _ = hipGraphDestroy(raw_graph);
                }
            }
            return Err(HipError::from_raw(rc, "hipStreamEndCapture"));
        }
        if raw_graph.is_null() {
            return Err(HipError::InvalidArgument {
                reason: "hipStreamEndCapture returned a null hipGraph_t".to_string(),
            });
        }
        Ok(Graph {
            raw: raw_graph,
            device: self.stream.device,
        })
    }
}

impl Drop for CaptureGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            self.active = false;
            let mut junk: hipGraph_t = ptr::null_mut();
            unsafe {
                let _ = hipStreamEndCapture(self.stream.raw, &mut junk);
                if !junk.is_null() {
                    let _ = hipGraphDestroy(junk);
                }
                let _ = hipGetLastError();
            }
        }
    }
}

/// Safe RAII wrapper around a captured `hipGraph_t` carrying [`DeviceId`].
#[derive(Debug)]
pub struct Graph {
    raw: hipGraph_t,
    device: DeviceId,
}

// SAFETY: `hipGraph_t` handles can be moved across threads.
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Graph {
    /// Device on which this graph was captured.
    pub const fn device_id(&self) -> DeviceId {
        self.device
    }

    /// Underlying `hipGraph_t` pointer.
    pub const fn raw(&self) -> hipGraph_t {
        self.raw
    }

    /// Query the number of nodes recorded in this graph (`hipGraphGetNodes`).
    pub fn node_count(&self) -> Result<usize> {
        let mut count: usize = 0;
        check_hip(
            unsafe { hipGraphGetNodes(self.raw, ptr::null_mut(), &mut count) },
            "hipGraphGetNodes",
        )?;
        Ok(count)
    }

    /// Instantiate this graph into an executable [`GraphExec`] (`hipGraphInstantiate`).
    pub fn instantiate(&self) -> Result<GraphExec> {
        set_active_device(self.device)?;
        let mut raw_exec: hipGraphExec_t = ptr::null_mut();
        check_hip(
            unsafe {
                hipGraphInstantiate(&mut raw_exec, self.raw, ptr::null_mut(), ptr::null_mut(), 0)
            },
            "hipGraphInstantiate",
        )?;
        if raw_exec.is_null() {
            return Err(HipError::InvalidArgument {
                reason: "hipGraphInstantiate returned a null hipGraphExec_t".to_string(),
            });
        }
        Ok(GraphExec {
            raw: raw_exec,
            device: self.device,
        })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = hipGraphDestroy(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}

/// Safe RAII wrapper around an instantiated executable `hipGraphExec_t` carrying [`DeviceId`].
#[derive(Debug)]
pub struct GraphExec {
    raw: hipGraphExec_t,
    device: DeviceId,
}

// SAFETY: `hipGraphExec_t` handles can be moved across threads.
unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

impl GraphExec {
    /// Device on which this executable graph was instantiated.
    pub const fn device_id(&self) -> DeviceId {
        self.device
    }

    /// Underlying `hipGraphExec_t` pointer.
    pub const fn raw(&self) -> hipGraphExec_t {
        self.raw
    }

    /// Replay this executable graph onto `stream` (`hipGraphLaunch`).
    ///
    /// Enforces that `stream` belongs to the same [`DeviceId`] as this graph.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        ensure_same_device(self.device, stream.device, "GraphExec::launch")?;
        check_hip(
            unsafe { hipGraphLaunch(self.raw, stream.raw) },
            "hipGraphLaunch",
        )
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = hipGraphExecDestroy(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}

// ============================================================================
// DeviceBuf<T>
// ============================================================================

/// Typed device memory buffer (`hipMalloc`) carrying [`DeviceId`].
#[derive(Debug)]
pub struct DeviceBuf<T> {
    ptr: *mut T,
    len: usize,
    device: DeviceId,
    _marker: PhantomData<T>,
}

// SAFETY: `DeviceBuf<T>` owns a device memory allocation when `T: Send`.
unsafe impl<T: Send> Send for DeviceBuf<T> {}
unsafe impl<T: Sync> Sync for DeviceBuf<T> {}

impl<T: Copy> DeviceBuf<T> {
    /// Allocate an uninitialized (or zeroed in simulation) device buffer of `len` elements on `device`.
    pub fn new(device: DeviceId, len: usize) -> Result<Self> {
        Self::alloc(device, len)
    }

    /// Allocate a device buffer of `len` elements on `device` (`hipMalloc`).
    pub fn alloc(device: DeviceId, len: usize) -> Result<Self> {
        if size_of::<T>() == 0 {
            return Err(HipError::InvalidArgument {
                reason: "DeviceBuf does not support zero-sized element types".to_string(),
            });
        }
        let byte_len =
            len.checked_mul(size_of::<T>())
                .ok_or_else(|| HipError::InvalidArgument {
                    reason: "DeviceBuf byte length overflow".to_string(),
                })?;

        check_hip(unsafe { hipInit(0) }, "hipInit")?;
        set_active_device(device)?;

        let mut raw: *mut c_void = ptr::null_mut();
        if byte_len > 0 {
            check_hip(unsafe { hipMalloc(&mut raw, byte_len) }, "hipMalloc")?;
        }
        Ok(Self {
            ptr: raw.cast::<T>(),
            len,
            device,
            _marker: PhantomData,
        })
    }

    /// Allocate a zero-initialized device buffer of `len` elements on `device`.
    pub fn zeros(device: DeviceId, len: usize) -> Result<Self> {
        let buf = Self::alloc(device, len)?;
        buf.zero()?;
        Ok(buf)
    }

    /// Allocate a device buffer on `device` and synchronously copy `src` into it.
    pub fn from_slice(device: DeviceId, src: &[T]) -> Result<Self> {
        let buf = Self::alloc(device, src.len())?;
        buf.copy_from_host(src)?;
        Ok(buf)
    }

    /// Logical device identifier owning this allocation.
    pub const fn device_id(&self) -> DeviceId {
        self.device
    }

    /// Number of elements of type `T`.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// True if `self.len() == 0`.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total allocation size in bytes (`self.len() * size_of::<T>()`).
    pub const fn byte_len(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Raw const device pointer (`*const T`).
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.cast_const()
    }

    /// Raw mutable device pointer (`*mut T`).
    pub const fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// 64-bit integer device address (`u64`) for `WindowIo` / `RoundIo` / `HiddenTaps`.
    pub fn device_ptr(&self) -> u64 {
        self.ptr as usize as u64
    }

    /// Synchronously zero the entire buffer (`hipMemset`).
    pub fn zero(&self) -> Result<()> {
        let bytes = self.byte_len();
        if bytes == 0 {
            return Ok(());
        }
        set_active_device(self.device)?;
        check_hip(
            unsafe { hipMemset(self.ptr.cast::<c_void>(), 0, bytes) },
            "hipMemset",
        )
    }

    /// Asynchronously zero the entire buffer on `stream` (`hipMemsetAsync`).
    ///
    /// Enforces that `stream` belongs to the same [`DeviceId`].
    pub fn zero_async(&self, stream: &Stream) -> Result<()> {
        ensure_same_device(self.device, stream.device, "DeviceBuf::zero_async")?;
        let bytes = self.byte_len();
        if bytes == 0 {
            return Ok(());
        }
        check_hip(
            unsafe { hipMemsetAsync(self.ptr.cast::<c_void>(), 0, bytes, stream.raw) },
            "hipMemsetAsync",
        )
    }

    /// Synchronously copy `src` from host memory into this device buffer (`hipMemcpyHostToDevice`).
    pub fn copy_from_host(&self, src: &[T]) -> Result<()> {
        if src.len() != self.len {
            return Err(HipError::LengthMismatch {
                expected: self.len,
                actual: src.len(),
                operation: "DeviceBuf::copy_from_host",
            });
        }
        let bytes = self.byte_len();
        if bytes == 0 {
            return Ok(());
        }
        set_active_device(self.device)?;
        check_hip(
            unsafe {
                hipMemcpy(
                    self.ptr.cast::<c_void>(),
                    src.as_ptr().cast::<c_void>(),
                    bytes,
                    hipMemcpyHostToDevice,
                )
            },
            "hipMemcpy(HostToDevice)",
        )
    }

    /// Asynchronously copy `src` from host memory into this device buffer on `stream`.
    ///
    /// Enforces that `stream` belongs to the same [`DeviceId`].
    pub fn copy_from_host_async(&self, src: &[T], stream: &Stream) -> Result<()> {
        ensure_same_device(
            self.device,
            stream.device,
            "DeviceBuf::copy_from_host_async",
        )?;
        if src.len() != self.len {
            return Err(HipError::LengthMismatch {
                expected: self.len,
                actual: src.len(),
                operation: "DeviceBuf::copy_from_host_async",
            });
        }
        let bytes = self.byte_len();
        if bytes == 0 {
            return Ok(());
        }
        check_hip(
            unsafe {
                hipMemcpyAsync(
                    self.ptr.cast::<c_void>(),
                    src.as_ptr().cast::<c_void>(),
                    bytes,
                    hipMemcpyHostToDevice,
                    stream.raw,
                )
            },
            "hipMemcpyAsync(HostToDevice)",
        )
    }

    /// Synchronously copy this device buffer into `dst` on the host (`hipMemcpyDeviceToHost`).
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            return Err(HipError::LengthMismatch {
                expected: self.len,
                actual: dst.len(),
                operation: "DeviceBuf::copy_to_host",
            });
        }
        let bytes = self.byte_len();
        if bytes == 0 {
            return Ok(());
        }
        set_active_device(self.device)?;
        check_hip(
            unsafe {
                hipMemcpy(
                    dst.as_mut_ptr().cast::<c_void>(),
                    self.ptr.cast::<c_void>(),
                    bytes,
                    hipMemcpyDeviceToHost,
                )
            },
            "hipMemcpy(DeviceToHost)",
        )
    }

    /// Asynchronously copy this device buffer into `dst` on `stream` (`hipMemcpyDeviceToHost`).
    ///
    /// Enforces that `stream` belongs to the same [`DeviceId`].
    pub fn copy_to_host_async(&self, dst: &mut [T], stream: &Stream) -> Result<()> {
        ensure_same_device(self.device, stream.device, "DeviceBuf::copy_to_host_async")?;
        if dst.len() != self.len {
            return Err(HipError::LengthMismatch {
                expected: self.len,
                actual: dst.len(),
                operation: "DeviceBuf::copy_to_host_async",
            });
        }
        let bytes = self.byte_len();
        if bytes == 0 {
            return Ok(());
        }
        check_hip(
            unsafe {
                hipMemcpyAsync(
                    dst.as_mut_ptr().cast::<c_void>(),
                    self.ptr.cast::<c_void>(),
                    bytes,
                    hipMemcpyDeviceToHost,
                    stream.raw,
                )
            },
            "hipMemcpyAsync(DeviceToHost)",
        )
    }

    /// Asynchronously copy from another `DeviceBuf<T>` on the same device (`hipMemcpyDeviceToDevice`).
    ///
    /// Enforces that `self`, `src`, and `stream` all share the same [`DeviceId`] and element length.
    pub fn copy_from_device_async(&self, src: &Self, stream: &Stream) -> Result<()> {
        ensure_same_device(self.device, src.device, "DeviceBuf::copy_from_device_async")?;
        ensure_same_device(
            self.device,
            stream.device,
            "DeviceBuf::copy_from_device_async",
        )?;
        if src.len != self.len {
            return Err(HipError::LengthMismatch {
                expected: self.len,
                actual: src.len,
                operation: "DeviceBuf::copy_from_device_async",
            });
        }
        let bytes = self.byte_len();
        if bytes == 0 {
            return Ok(());
        }
        check_hip(
            unsafe {
                hipMemcpyAsync(
                    self.ptr.cast::<c_void>(),
                    src.ptr.cast::<c_void>(),
                    bytes,
                    hipMemcpyDeviceToDevice,
                    stream.raw,
                )
            },
            "hipMemcpyAsync(DeviceToDevice)",
        )
    }

    /// Copy the contents of this device buffer into a newly allocated host `Vec<T>`.
    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Default,
    {
        let mut out = vec![T::default(); self.len];
        self.copy_to_host(&mut out)?;
        Ok(out)
    }
}

impl<T> Drop for DeviceBuf<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = hipFree(self.ptr.cast::<c_void>());
            }
            self.ptr = ptr::null_mut();
        }
    }
}

// ============================================================================
// HostMapped<T>
// ============================================================================

/// Page-locked host-mapped buffer (`hipHostMalloc` with `hipHostMallocMapped`) carrying [`DeviceId`].
///
/// Directly dereferences to `[T]` on the host while exposing a fixed device address (`device_ptr()`)
/// suitable for graph-captured control words (`pos_dev`, `tok_dev`, speculative round counters).
#[derive(Debug)]
pub struct HostMapped<T> {
    host_ptr: *mut T,
    dev_ptr: *mut T,
    len: usize,
    device: DeviceId,
    _marker: PhantomData<T>,
}

// SAFETY: `HostMapped<T>` owns its pinned host allocation.
unsafe impl<T: Send> Send for HostMapped<T> {}
unsafe impl<T: Sync> Sync for HostMapped<T> {}

impl<T: Copy> HostMapped<T> {
    /// Allocate a zero-initialized page-locked host-mapped buffer of `len` elements on `device`.
    pub fn new(device: DeviceId, len: usize) -> Result<Self> {
        Self::alloc(device, len)
    }

    /// Allocate a zero-initialized page-locked host-mapped buffer of `len` elements on `device`.
    pub fn alloc(device: DeviceId, len: usize) -> Result<Self> {
        if size_of::<T>() == 0 {
            return Err(HipError::InvalidArgument {
                reason: "HostMapped does not support zero-sized element types".to_string(),
            });
        }
        let byte_len =
            len.checked_mul(size_of::<T>())
                .ok_or_else(|| HipError::InvalidArgument {
                    reason: "HostMapped byte length overflow".to_string(),
                })?;

        check_hip(unsafe { hipInit(0) }, "hipInit")?;
        set_active_device(device)?;

        if byte_len == 0 {
            return Ok(Self {
                host_ptr: ptr::NonNull::<T>::dangling().as_ptr(),
                dev_ptr: ptr::null_mut(),
                len: 0,
                device,
                _marker: PhantomData,
            });
        }

        let mut raw_host: *mut c_void = ptr::null_mut();
        check_hip(
            unsafe { hipHostMalloc(&mut raw_host, byte_len, hipHostMallocMapped) },
            "hipHostMalloc",
        )?;

        // Zero-initialize before exposing through `as_slice` / `Deref`.
        unsafe {
            ptr::write_bytes(raw_host.cast::<u8>(), 0, byte_len);
        }

        let mut raw_dev: *mut c_void = ptr::null_mut();
        let dev_rc = unsafe { hipHostGetDevicePointer(&mut raw_dev, raw_host, 0) };
        if dev_rc != hipSuccess {
            unsafe {
                let _ = hipHostFree(raw_host);
            }
            return Err(HipError::from_raw(dev_rc, "hipHostGetDevicePointer"));
        }

        Ok(Self {
            host_ptr: raw_host.cast::<T>(),
            dev_ptr: raw_dev.cast::<T>(),
            len,
            device,
            _marker: PhantomData,
        })
    }

    /// Allocate a zero-initialized page-locked host-mapped buffer of `len` elements on `device`.
    pub fn zeros(device: DeviceId, len: usize) -> Result<Self> {
        Self::alloc(device, len)
    }

    /// Allocate a page-locked host-mapped buffer on `device` and initialize it from `src`.
    pub fn from_slice(device: DeviceId, src: &[T]) -> Result<Self> {
        let mut buf = Self::alloc(device, src.len())?;
        buf.as_mut_slice().copy_from_slice(src);
        Ok(buf)
    }

    /// Logical device identifier associated with this mapped buffer.
    pub const fn device_id(&self) -> DeviceId {
        self.device
    }

    /// Number of elements of type `T`.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// True if `self.len() == 0`.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total allocation size in bytes (`self.len() * size_of::<T>()`).
    pub const fn byte_len(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Borrow the mapped memory as a host slice `&[T]`.
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(self.host_ptr, self.len) }
        }
    }

    /// Borrow the mapped memory as a mutable host slice `&mut [T]`.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(self.host_ptr, self.len) }
        }
    }

    /// Raw host pointer (`*mut T`).
    pub const fn host_ptr(&self) -> *mut T {
        if self.len == 0 {
            ptr::null_mut()
        } else {
            self.host_ptr
        }
    }

    /// Raw mapped device pointer (`*mut T`).
    pub const fn as_device_ptr(&self) -> *mut T {
        self.dev_ptr
    }

    /// 64-bit mapped device address (`u64`) for graph-captured kernel arguments (`WindowIo`, `RoundIo`).
    pub fn device_ptr(&self) -> u64 {
        self.dev_ptr as usize as u64
    }
}

impl<T: Copy> Deref for HostMapped<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy> DerefMut for HostMapped<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Drop for HostMapped<T> {
    fn drop(&mut self) {
        if self.len > 0 && !self.host_ptr.is_null() {
            unsafe {
                let _ = hipHostFree(self.host_ptr.cast::<c_void>());
            }
            self.host_ptr = ptr::null_mut();
            self.dev_ptr = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity() {
        assert_eq!(CRATE_NAME, "talus-hip");
    }

    #[test]
    fn device_queries_arch_traits_and_mem_info() {
        let count = Device::count().expect("device count");
        assert!(count >= 1);

        let dev0 = Device::new(DeviceId(0)).expect("open device 0");
        assert_eq!(dev0.device_id(), DeviceId(0));
        assert_eq!(dev0.arch(), GpuArch::Gfx1151);

        let traits = dev0.arch_traits();
        assert_eq!(traits.wave, 32);
        assert_eq!(traits.wmma, WmmaLayout::Gfx11);
        assert!(!traits.fp8);
        assert_eq!(dev0.warp_size(), 32);
        assert!(dev0.multi_processor_count() > 0);
        assert!(dev0.total_memory() > 0);
        assert!(!dev0.name().is_empty());
        assert!(dev0.gcn_arch_name().starts_with("gfx1151"));

        dev0.make_current().expect("make_current");
        dev0.synchronize().expect("synchronize");
        let (free, total) = dev0.mem_info().expect("mem_info");
        assert!(total > 0);
        assert!(free <= total);

        // Invalid device ordinal returns HipError::Runtime
        let invalid = Device::new(DeviceId(999));
        assert!(invalid.is_err());
        let core_err: CoreError = invalid.unwrap_err().into();
        assert!(matches!(core_err, CoreError::HipError { .. }));
    }

    #[test]
    fn stream_and_event_lifecycle_and_timing() {
        let dev = DeviceId(0);
        let stream = Stream::new(dev).expect("default stream");
        let nb_stream = Stream::non_blocking(dev).expect("non-blocking stream");

        assert_eq!(stream.device_id(), dev);
        assert_eq!(nb_stream.device_id(), dev);
        assert!(!stream.raw().is_null());
        assert!(!nb_stream.raw().is_null());

        let core_handle = nb_stream.as_core_handle();
        assert_eq!(core_handle.device, dev);
        assert_eq!(core_handle.raw, nb_stream.raw() as usize as u64);
        assert!(nb_stream.query().expect("stream query"));

        let ev_start = Event::new(dev).expect("start event");
        let ev_stop = Event::new(dev).expect("stop event");
        let ev_notime = Event::disable_timing(dev).expect("disable_timing event");

        assert!(ev_start.timing_enabled());
        assert!(!ev_notime.timing_enabled());

        ev_start.record(&nb_stream).expect("record start");
        ev_stop.record(&nb_stream).expect("record stop");
        ev_notime.record(&nb_stream).expect("record notime");

        nb_stream.wait_event(&ev_stop).expect("stream wait_event");
        nb_stream.synchronize().expect("stream sync");
        ev_stop.synchronize().expect("event sync");
        assert!(ev_stop.query().expect("event query"));

        let ms = ev_start.elapsed_ms(&ev_stop).expect("elapsed_ms");
        assert!(ms >= 0.0);

        // elapsed_ms on a disable_timing event must return an error
        assert!(ev_start.elapsed_ms(&ev_notime).is_err());
        assert!(ev_notime.elapsed_ms(&ev_stop).is_err());
    }

    #[test]
    fn device_buf_alloc_transfers_and_drop() {
        let dev_id = DeviceId(0);
        let dev = Device::new(dev_id).expect("device 0");
        let stream = Stream::non_blocking(dev_id).expect("stream");

        let (free_before, _) = dev.mem_info().expect("mem_info before");

        {
            let buf = DeviceBuf::<u32>::zeros(dev_id, 64 * 1024).expect("alloc 64K u32 (256 KiB)");
            assert_eq!(buf.device_id(), dev_id);
            assert_eq!(buf.len(), 64 * 1024);
            assert!(!buf.is_empty());
            assert_eq!(buf.byte_len(), 256 * 1024);
            assert!(!buf.as_ptr().is_null());
            assert!(!buf.as_mut_ptr().is_null());
            assert_ne!(buf.device_ptr(), 0);

            let (free_during, total) = dev.mem_info().expect("mem_info during");
            assert!(total - free_during >= buf.byte_len());
            assert!(free_during < free_before);

            let initial = buf.to_vec().expect("to_vec");
            assert!(initial.iter().all(|&x| x == 0));

            let host_src: Vec<u32> = (0..(64 * 1024)).map(|i| i * 3 + 7).collect();
            buf.copy_from_host(&host_src).expect("copy_from_host");

            let mut host_dst = vec![0u32; 64 * 1024];
            buf.copy_to_host(&mut host_dst).expect("copy_to_host");
            assert_eq!(host_dst, host_src);

            // Async device-to-device copy and async zero
            let buf2 = DeviceBuf::<u32>::new(dev_id, 64 * 1024).expect("alloc buf2");
            buf2.copy_from_device_async(&buf, &stream)
                .expect("copy_from_device_async");
            stream.synchronize().expect("sync");
            assert_eq!(buf2.to_vec().expect("buf2 to_vec"), host_src);

            buf2.zero_async(&stream).expect("zero_async");
            let mut host_zeroed = vec![1u32; 64 * 1024];
            buf2.copy_to_host_async(&mut host_zeroed, &stream)
                .expect("copy_to_host_async");
            stream.synchronize().expect("sync");
            assert!(host_zeroed.iter().all(|&x| x == 0));

            // Length mismatch checks
            assert!(buf
                .copy_from_host(&[1u32, 2, 3])
                .unwrap_err()
                .is_length_mismatch());
            assert!(buf
                .copy_to_host(&mut [0u32; 4])
                .unwrap_err()
                .is_length_mismatch());
            let short_buf = DeviceBuf::<u32>::new(dev_id, 8).expect("short buf");
            assert!(buf
                .copy_from_device_async(&short_buf, &stream)
                .unwrap_err()
                .is_length_mismatch());

            let (free_with_both, _) = dev.mem_info().expect("mem_info with both");
            drop(buf2);
            let (free_after_one_drop, _) = dev.mem_info().expect("mem_info after buf2 drop");
            // Other parallel tests allocate at most a few hundred bytes; dropping 256 KiB reclaims ~256 KiB.
            assert!(free_after_one_drop >= free_with_both + 250 * 1024);
        }

        // Verify Drop freed the 256 KiB device memory
        let (free_after, total) = dev.mem_info().expect("mem_info after");
        assert!(total - free_after < 4096);

        // Zero-length DeviceBuf is valid and a no-op
        let empty = DeviceBuf::<f32>::new(dev_id, 0).expect("empty DeviceBuf");
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.byte_len(), 0);
        empty.zero().expect("zero empty");
        empty.zero_async(&stream).expect("zero_async empty");
        empty.copy_from_host(&[]).expect("copy_from_host empty");
        let mut empty_dst: [f32; 0] = [];
        empty
            .copy_to_host(&mut empty_dst)
            .expect("copy_to_host empty");
    }

    #[test]
    fn host_mapped_deref_and_device_pointer() {
        let dev = DeviceId(0);
        let mut mapped = HostMapped::<i32>::new(dev, 64).expect("HostMapped::new");
        assert_eq!(mapped.device_id(), dev);
        assert_eq!(mapped.len(), 64);
        assert!(!mapped.is_empty());
        assert_eq!(mapped.byte_len(), 256);
        assert!(!mapped.host_ptr().is_null());
        assert!(!mapped.as_device_ptr().is_null());
        assert_ne!(mapped.device_ptr(), 0);
        assert!(mapped.iter().all(|&x| x == 0));

        // Mutate through DerefMut (`&mut [i32]`)
        for (i, slot) in mapped.iter_mut().enumerate() {
            *slot = (i as i32) * 10 - 5;
        }
        assert_eq!(mapped.as_slice()[0], -5);
        assert_eq!(mapped[63], 625);

        let from_slice =
            HostMapped::from_slice(dev, &[10u64, 20, 30, 40]).expect("HostMapped::from_slice");
        assert_eq!(&*from_slice, &[10u64, 20, 30, 40]);

        // Zero-length HostMapped
        let mut empty = HostMapped::<u32>::new(dev, 0).expect("empty HostMapped");
        assert!(empty.is_empty());
        assert_eq!(empty.as_slice(), &[]);
        assert_eq!(empty.as_mut_slice(), &mut []);
        assert!(empty.host_ptr().is_null());
        assert_eq!(empty.device_ptr(), 0);
    }

    #[test]
    fn graph_capture_finish_instantiate_launch_and_drop_abort() {
        let dev = DeviceId(0);
        let stream = Stream::non_blocking(dev).expect("stream");
        let src = DeviceBuf::<u32>::from_slice(dev, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("src");
        let dst = DeviceBuf::<u32>::zeros(dev, 8).expect("dst");

        // 1. Test CaptureGuard Drop WITHOUT finish() — must cleanly abort capture and leave stream usable
        assert!(!stream.is_capturing());
        {
            let guard = stream
                .begin_capture(CaptureMode::ThreadLocal)
                .expect("begin_capture for abort");
            assert_eq!(guard.device_id(), dev);
            assert!(stream.is_capturing());
            dst.copy_from_device_async(&src, guard.stream())
                .expect("record op before abort");
            // `guard` dropped here without `.finish()`!
        }
        assert!(
            !stream.is_capturing(),
            "dropping CaptureGuard without finish() must end stream capture"
        );
        // Stream must remain usable after aborted capture
        stream.synchronize().expect("stream usable after abort");

        // 2. Test normal CaptureGuard::finish() -> Graph::instantiate() -> GraphExec::launch()
        let guard = stream
            .begin_capture(CaptureMode::ThreadLocal)
            .expect("begin_capture");
        assert!(stream.is_capturing());
        dst.zero_async(&stream).expect("capture zero_async");
        dst.copy_from_device_async(&src, &stream)
            .expect("capture copy_from_device_async");
        let graph = guard.finish().expect("finish capture");
        assert!(!stream.is_capturing());
        assert_eq!(graph.device_id(), dev);
        assert_eq!(graph.node_count().expect("node_count"), 2);

        let exec = graph.instantiate().expect("instantiate graph");
        assert_eq!(exec.device_id(), dev);

        // Update `src` after capture to verify graph replay reads live buffer contents at launch
        src.copy_from_host(&[100, 200, 300, 400, 500, 600, 700, 800])
            .expect("update src");
        exec.launch(&stream).expect("launch graph");
        stream.synchronize().expect("sync after launch");

        let out = dst.to_vec().expect("read dst");
        assert_eq!(out, vec![100, 200, 300, 400, 500, 600, 700, 800]);
    }

    #[test]
    fn cross_device_mismatch_checks() {
        let dev0 = DeviceId(0);
        let dev1 = DeviceId(1);

        let dev1_handle = Device::new(dev1).expect("open device 1");
        assert_eq!(dev1_handle.device_id(), dev1);
        assert_eq!(dev1_handle.arch(), GpuArch::Gfx1201);

        let stream0 = Stream::non_blocking(dev0).expect("stream0");
        let stream1 = Stream::non_blocking(dev1).expect("stream1");

        let ev0 = Event::new(dev0).expect("ev0");
        let ev1 = Event::new(dev1).expect("ev1");

        // Event recorded or waited on wrong device stream must fail with DeviceMismatch
        assert!(ev0.record(&stream1).unwrap_err().is_device_mismatch());
        assert!(stream0.wait_event(&ev1).unwrap_err().is_device_mismatch());
        assert!(ev0.elapsed_ms(&ev1).unwrap_err().is_device_mismatch());

        // DeviceBuf operations with wrong device stream or peer buffer must fail with DeviceMismatch
        let buf0 = DeviceBuf::<f32>::zeros(dev0, 16).expect("buf0");
        let buf1 = DeviceBuf::<f32>::zeros(dev1, 16).expect("buf1");
        let mut host = [0.0f32; 16];

        assert!(buf0.zero_async(&stream1).unwrap_err().is_device_mismatch());
        assert!(buf0
            .copy_from_host_async(&host, &stream1)
            .unwrap_err()
            .is_device_mismatch());
        assert!(buf0
            .copy_to_host_async(&mut host, &stream1)
            .unwrap_err()
            .is_device_mismatch());
        assert!(buf0
            .copy_from_device_async(&buf1, &stream0)
            .unwrap_err()
            .is_device_mismatch());
        assert!(buf0
            .copy_from_device_async(&buf0, &stream1)
            .unwrap_err()
            .is_device_mismatch());

        // GraphExec launched on wrong device stream must fail with DeviceMismatch
        let guard = stream0
            .begin_capture(CaptureMode::ThreadLocal)
            .expect("begin_capture");
        buf0.zero_async(&stream0).expect("record zero");
        let graph0 = guard.finish().expect("finish");
        let exec0 = graph0.instantiate().expect("instantiate");

        let err = exec0
            .launch(&stream1)
            .expect_err("launch on wrong device must fail");
        assert!(err.is_device_mismatch());
    }
}
