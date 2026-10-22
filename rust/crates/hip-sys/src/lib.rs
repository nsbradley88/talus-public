//! Raw HIP runtime FFI bindings, dynamic `libamdhip64.so` loader, and simulated CPU runtime for Talus.
//!
//! Both the Rust host and `libtalus_kernels.so` dynamically load
//! the same `libamdhip64.so` so stream and graph capture states are shared across the FFI boundary.
//! When no AMD GPU hardware is available (e.g. on a CPU-only build host) or when
//! [`HipBackendMode::Simulated`] is requested, [`HipRuntime`] provides an in-process simulated HIP
//! runtime supporting devices, device/host-mapped allocations, streams, events, and HIP graph
//! capture/replay.

#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    clippy::missing_safety_doc
)]

use std::alloc::{self, Layout};
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_ulonglong, c_void, CStr, CString};
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Crate identifier.
pub const CRATE_NAME: &str = "hip-sys";

// ============================================================================
// POSIX `libdl` FFI (zero external crate dependencies)
// ============================================================================

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

// ============================================================================
// Raw HIP C ABI Types & Constants (`hip/hip_runtime_api.h` ROCm 6.0 / 7.0)
// ============================================================================

/// Raw HIP error status (`hipError_t`).
///
/// Represented as a `#[repr(transparent)]` `i32` wrapper so unknown future ROCm/HSA codes never
/// cause Rust enum-discriminant UB across FFI while preserving exact C ABI layout.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct hipError_t(pub c_int);

impl hipError_t {
    pub const fn code(self) -> i32 {
        self.0
    }

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }

    pub const fn name_str(self) -> &'static str {
        match self.0 {
            0 => "hipSuccess",
            1 => "hipErrorInvalidValue",
            2 => "hipErrorOutOfMemory",
            3 => "hipErrorNotInitialized",
            4 => "hipErrorDeinitialized",
            9 => "hipErrorInvalidConfiguration",
            17 => "hipErrorInvalidDevicePointer",
            21 => "hipErrorInvalidMemcpyDirection",
            35 => "hipErrorInsufficientDriver",
            98 => "hipErrorInvalidDeviceFunction",
            100 => "hipErrorNoDevice",
            101 => "hipErrorInvalidDevice",
            200 => "hipErrorInvalidImage",
            201 => "hipErrorInvalidContext",
            205 => "hipErrorMapFailed",
            206 => "hipErrorUnmapFailed",
            208 => "hipErrorAlreadyMapped",
            209 => "hipErrorNoBinaryForGpu",
            211 => "hipErrorNotMapped",
            217 => "hipErrorPeerAccessUnsupported",
            301 => "hipErrorFileNotFound",
            400 => "hipErrorInvalidHandle",
            401 => "hipErrorIllegalState",
            500 => "hipErrorNotFound",
            600 => "hipErrorNotReady",
            700 => "hipErrorIllegalAddress",
            701 => "hipErrorLaunchOutOfResources",
            702 => "hipErrorLaunchTimeOut",
            704 => "hipErrorPeerAccessAlreadyEnabled",
            705 => "hipErrorPeerAccessNotEnabled",
            712 => "hipErrorHostMemoryAlreadyRegistered",
            713 => "hipErrorHostMemoryNotRegistered",
            719 => "hipErrorLaunchFailure",
            800 => "hipErrorNotPermitted",
            801 => "hipErrorNotSupported",
            900 => "hipErrorStreamCaptureUnsupported",
            901 => "hipErrorStreamCaptureInvalidated",
            902 => "hipErrorStreamCaptureMerge",
            903 => "hipErrorStreamCaptureUnmatched",
            904 => "hipErrorStreamCaptureUnjoined",
            905 => "hipErrorStreamCaptureIsolation",
            906 => "hipErrorStreamCaptureImplicit",
            907 => "hipErrorCapturedEvent",
            908 => "hipErrorStreamCaptureWrongThread",
            910 => "hipErrorGraphExecUpdateFailure",
            999 => "hipErrorUnknown",
            1052 => "hipErrorRuntimeMemory",
            1053 => "hipErrorRuntimeOther",
            _ => "hipErrorUnknown",
        }
    }

    pub const fn c_error_string(self) -> &'static CStr {
        match self.0 {
            0 => c"hipSuccess",
            1 => c"hipErrorInvalidValue",
            2 => c"hipErrorOutOfMemory",
            3 => c"hipErrorNotInitialized",
            4 => c"hipErrorDeinitialized",
            9 => c"hipErrorInvalidConfiguration",
            17 => c"hipErrorInvalidDevicePointer",
            21 => c"hipErrorInvalidMemcpyDirection",
            35 => c"hipErrorInsufficientDriver",
            98 => c"hipErrorInvalidDeviceFunction",
            100 => c"hipErrorNoDevice",
            101 => c"hipErrorInvalidDevice",
            200 => c"hipErrorInvalidImage",
            201 => c"hipErrorInvalidContext",
            205 => c"hipErrorMapFailed",
            206 => c"hipErrorUnmapFailed",
            208 => c"hipErrorAlreadyMapped",
            209 => c"hipErrorNoBinaryForGpu",
            211 => c"hipErrorNotMapped",
            217 => c"hipErrorPeerAccessUnsupported",
            301 => c"hipErrorFileNotFound",
            400 => c"hipErrorInvalidHandle",
            401 => c"hipErrorIllegalState",
            500 => c"hipErrorNotFound",
            600 => c"hipErrorNotReady",
            700 => c"hipErrorIllegalAddress",
            701 => c"hipErrorLaunchOutOfResources",
            702 => c"hipErrorLaunchTimeOut",
            704 => c"hipErrorPeerAccessAlreadyEnabled",
            705 => c"hipErrorPeerAccessNotEnabled",
            712 => c"hipErrorHostMemoryAlreadyRegistered",
            713 => c"hipErrorHostMemoryNotRegistered",
            719 => c"hipErrorLaunchFailure",
            800 => c"hipErrorNotPermitted",
            801 => c"hipErrorNotSupported",
            900 => c"hipErrorStreamCaptureUnsupported",
            901 => c"hipErrorStreamCaptureInvalidated",
            902 => c"hipErrorStreamCaptureMerge",
            903 => c"hipErrorStreamCaptureUnmatched",
            904 => c"hipErrorStreamCaptureUnjoined",
            905 => c"hipErrorStreamCaptureIsolation",
            906 => c"hipErrorStreamCaptureImplicit",
            907 => c"hipErrorCapturedEvent",
            908 => c"hipErrorStreamCaptureWrongThread",
            910 => c"hipErrorGraphExecUpdateFailure",
            999 => c"hipErrorUnknown",
            1052 => c"hipErrorRuntimeMemory",
            1053 => c"hipErrorRuntimeOther",
            _ => c"hipErrorUnknown",
        }
    }
}

pub const hipSuccess: hipError_t = hipError_t(0);
pub const hipErrorInvalidValue: hipError_t = hipError_t(1);
pub const hipErrorOutOfMemory: hipError_t = hipError_t(2);
pub const hipErrorMemoryAllocation: hipError_t = hipError_t(2);
pub const hipErrorNotInitialized: hipError_t = hipError_t(3);
pub const hipErrorInitializationError: hipError_t = hipError_t(3);
pub const hipErrorDeinitialized: hipError_t = hipError_t(4);
pub const hipErrorInvalidConfiguration: hipError_t = hipError_t(9);
pub const hipErrorInvalidPitchValue: hipError_t = hipError_t(12);
pub const hipErrorInvalidSymbol: hipError_t = hipError_t(13);
pub const hipErrorInvalidDevicePointer: hipError_t = hipError_t(17);
pub const hipErrorInvalidMemcpyDirection: hipError_t = hipError_t(21);
pub const hipErrorInsufficientDriver: hipError_t = hipError_t(35);
pub const hipErrorMissingConfiguration: hipError_t = hipError_t(52);
pub const hipErrorPriorLaunchFailure: hipError_t = hipError_t(53);
pub const hipErrorInvalidDeviceFunction: hipError_t = hipError_t(98);
pub const hipErrorNoDevice: hipError_t = hipError_t(100);
pub const hipErrorInvalidDevice: hipError_t = hipError_t(101);
pub const hipErrorInvalidImage: hipError_t = hipError_t(200);
pub const hipErrorInvalidContext: hipError_t = hipError_t(201);
pub const hipErrorContextAlreadyCurrent: hipError_t = hipError_t(202);
pub const hipErrorMapFailed: hipError_t = hipError_t(205);
pub const hipErrorUnmapFailed: hipError_t = hipError_t(206);
pub const hipErrorAlreadyMapped: hipError_t = hipError_t(208);
pub const hipErrorNoBinaryForGpu: hipError_t = hipError_t(209);
pub const hipErrorNotMapped: hipError_t = hipError_t(211);
pub const hipErrorPeerAccessUnsupported: hipError_t = hipError_t(217);
pub const hipErrorFileNotFound: hipError_t = hipError_t(301);
pub const hipErrorInvalidHandle: hipError_t = hipError_t(400);
pub const hipErrorInvalidResourceHandle: hipError_t = hipError_t(400);
pub const hipErrorIllegalState: hipError_t = hipError_t(401);
pub const hipErrorNotFound: hipError_t = hipError_t(500);
pub const hipErrorNotReady: hipError_t = hipError_t(600);
pub const hipErrorIllegalAddress: hipError_t = hipError_t(700);
pub const hipErrorLaunchOutOfResources: hipError_t = hipError_t(701);
pub const hipErrorLaunchTimeOut: hipError_t = hipError_t(702);
pub const hipErrorPeerAccessAlreadyEnabled: hipError_t = hipError_t(704);
pub const hipErrorPeerAccessNotEnabled: hipError_t = hipError_t(705);
pub const hipErrorHostMemoryAlreadyRegistered: hipError_t = hipError_t(712);
pub const hipErrorHostMemoryNotRegistered: hipError_t = hipError_t(713);
pub const hipErrorLaunchFailure: hipError_t = hipError_t(719);
pub const hipErrorNotPermitted: hipError_t = hipError_t(800);
pub const hipErrorNotSupported: hipError_t = hipError_t(801);
pub const hipErrorStreamCaptureUnsupported: hipError_t = hipError_t(900);
pub const hipErrorStreamCaptureInvalidated: hipError_t = hipError_t(901);
pub const hipErrorStreamCaptureMerge: hipError_t = hipError_t(902);
pub const hipErrorStreamCaptureUnmatched: hipError_t = hipError_t(903);
pub const hipErrorStreamCaptureUnjoined: hipError_t = hipError_t(904);
pub const hipErrorStreamCaptureIsolation: hipError_t = hipError_t(905);
pub const hipErrorStreamCaptureImplicit: hipError_t = hipError_t(906);
pub const hipErrorCapturedEvent: hipError_t = hipError_t(907);
pub const hipErrorStreamCaptureWrongThread: hipError_t = hipError_t(908);
pub const hipErrorGraphExecUpdateFailure: hipError_t = hipError_t(910);
pub const hipErrorUnknown: hipError_t = hipError_t(999);
pub const hipErrorRuntimeMemory: hipError_t = hipError_t(1052);
pub const hipErrorRuntimeOther: hipError_t = hipError_t(1053);

/// HIP device ordinal (`hipDevice_t`).
pub type hipDevice_t = c_int;

/// Raw untyped device pointer (`hipDeviceptr_t`).
pub type hipDeviceptr_t = *mut c_void;

/// Opaque stream structure (`ihipStream_t`).
#[repr(C)]
pub struct ihipStream_t {
    _private: [u8; 0],
}
/// HIP stream handle (`hipStream_t`).
pub type hipStream_t = *mut ihipStream_t;

/// Opaque event structure (`ihipEvent_t`).
#[repr(C)]
pub struct ihipEvent_t {
    _private: [u8; 0],
}
/// HIP event handle (`hipEvent_t`).
pub type hipEvent_t = *mut ihipEvent_t;

/// Opaque graph structure (`ihipGraph`).
#[repr(C)]
pub struct ihipGraph {
    _private: [u8; 0],
}
/// HIP graph handle (`hipGraph_t`).
pub type hipGraph_t = *mut ihipGraph;

/// Opaque graph node structure (`hipGraphNode`).
#[repr(C)]
pub struct hipGraphNode {
    _private: [u8; 0],
}
/// HIP graph node handle (`hipGraphNode_t`).
pub type hipGraphNode_t = *mut hipGraphNode;

/// Opaque executable graph structure (`hipGraphExec`).
#[repr(C)]
pub struct hipGraphExec {
    _private: [u8; 0],
}
/// HIP executable graph handle (`hipGraphExec_t`).
pub type hipGraphExec_t = *mut hipGraphExec;

/// Stream creation flags.
pub const hipStreamDefault: c_uint = 0x00;
pub const hipStreamNonBlocking: c_uint = 0x01;

/// Event creation flags.
pub const hipEventDefault: c_uint = 0x00;
pub const hipEventBlockingSync: c_uint = 0x01;
pub const hipEventDisableTiming: c_uint = 0x02;
pub const hipEventInterprocess: c_uint = 0x04;

/// Host pinned / mapped allocation flags (`hipHostMalloc`).
pub const hipHostMallocDefault: c_uint = 0x00;
pub const hipHostMallocPortable: c_uint = 0x01;
pub const hipHostMallocMapped: c_uint = 0x02;
pub const hipHostMallocWriteCombined: c_uint = 0x04;
pub const hipHostMallocUncached: c_uint = 0x1000_0000;
pub const hipHostMallocNumaUser: c_uint = 0x2000_0000;
pub const hipHostMallocCoherent: c_uint = 0x4000_0000;
pub const hipHostMallocNonCoherent: c_uint = 0x8000_0000;

/// Host memory registration flags (`hipHostRegister`).
pub const hipHostRegisterDefault: c_uint = 0x00;
pub const hipHostRegisterPortable: c_uint = 0x01;
pub const hipHostRegisterMapped: c_uint = 0x02;
pub const hipHostRegisterIoMemory: c_uint = 0x04;
pub const hipHostRegisterReadOnly: c_uint = 0x08;

/// HIP stream capture modes (`hipStreamCaptureMode`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum hipStreamCaptureMode {
    hipStreamCaptureModeGlobal = 0,
    #[default]
    hipStreamCaptureModeThreadLocal = 1,
    hipStreamCaptureModeRelaxed = 2,
}
pub use hipStreamCaptureMode::*;

/// HIP stream capture status (`hipStreamCaptureStatus`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum hipStreamCaptureStatus {
    #[default]
    hipStreamCaptureStatusNone = 0,
    hipStreamCaptureStatusActive = 1,
    hipStreamCaptureStatusInvalidated = 2,
}
pub use hipStreamCaptureStatus::*;

/// HIP memory copy direction (`hipMemcpyKind`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    #[default]
    hipMemcpyDefault = 4,
    hipMemcpyDeviceToDeviceNoCU = 1024,
}
pub use hipMemcpyKind::*;

/// HIP device attribute query enum (`hipDeviceAttribute_t`, ROCm 6.0 / 7.0).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum hipDeviceAttribute_t {
    hipDeviceAttributeEccEnabled = 0,
    hipDeviceAttributeAccessPolicyMaxWindowSize = 1,
    hipDeviceAttributeAsyncEngineCount = 2,
    hipDeviceAttributeCanMapHostMemory = 3,
    hipDeviceAttributeCanUseHostPointerForRegisteredMem = 4,
    hipDeviceAttributeClockRate = 5,
    hipDeviceAttributeComputeMode = 6,
    hipDeviceAttributeComputePreemptionSupported = 7,
    hipDeviceAttributeConcurrentKernels = 8,
    hipDeviceAttributeConcurrentManagedAccess = 9,
    hipDeviceAttributeCooperativeLaunch = 10,
    hipDeviceAttributeCooperativeMultiDeviceLaunch = 11,
    hipDeviceAttributeDeviceOverlap = 12,
    hipDeviceAttributeDirectManagedMemAccessFromHost = 13,
    hipDeviceAttributeGlobalL1CacheSupported = 14,
    hipDeviceAttributeHostNativeAtomicSupported = 15,
    hipDeviceAttributeIntegrated = 16,
    hipDeviceAttributeIsMultiGpuBoard = 17,
    hipDeviceAttributeKernelExecTimeout = 18,
    hipDeviceAttributeL2CacheSize = 19,
    hipDeviceAttributeLocalL1CacheSupported = 20,
    hipDeviceAttributeLuid = 21,
    hipDeviceAttributeLuidDeviceNodeMask = 22,
    hipDeviceAttributeComputeCapabilityMajor = 23,
    hipDeviceAttributeManagedMemory = 24,
    hipDeviceAttributeMaxBlocksPerMultiProcessor = 25,
    hipDeviceAttributeMaxBlockDimX = 26,
    hipDeviceAttributeMaxBlockDimY = 27,
    hipDeviceAttributeMaxBlockDimZ = 28,
    hipDeviceAttributeMaxGridDimX = 29,
    hipDeviceAttributeMaxGridDimY = 30,
    hipDeviceAttributeMaxGridDimZ = 31,
    hipDeviceAttributeMaxThreadsPerBlock = 51,
    hipDeviceAttributeMaxThreadsPerMultiProcessor = 52,
    hipDeviceAttributeMaxPitch = 53,
    hipDeviceAttributeMemoryBusWidth = 54,
    hipDeviceAttributeMemoryClockRate = 55,
    hipDeviceAttributeComputeCapabilityMinor = 56,
    hipDeviceAttributeMultiGpuBoardGroupID = 57,
    hipDeviceAttributeMultiprocessorCount = 58,
    hipDeviceAttributePageableMemoryAccess = 60,
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables = 61,
    hipDeviceAttributePciBusId = 62,
    hipDeviceAttributePciDeviceId = 63,
    hipDeviceAttributePciDomainId = 64,
    hipDeviceAttributePersistingL2CacheMaxSize = 65,
    hipDeviceAttributeMaxRegistersPerBlock = 66,
    hipDeviceAttributeMaxRegistersPerMultiprocessor = 67,
    hipDeviceAttributeReservedSharedMemPerBlock = 68,
    hipDeviceAttributeMaxSharedMemoryPerBlock = 69,
    hipDeviceAttributeSharedMemPerBlockOptin = 70,
    hipDeviceAttributeSharedMemPerMultiprocessor = 71,
    hipDeviceAttributeStreamPrioritiesSupported = 73,
    hipDeviceAttributeTotalConstantMemory = 77,
    hipDeviceAttributeTotalGlobalMem = 78,
    hipDeviceAttributeUnifiedAddressing = 79,
    hipDeviceAttributeWarpSize = 81,
    hipDeviceAttributeMemoryPoolsSupported = 82,
    hipDeviceAttributeVirtualMemoryManagementSupported = 83,
    hipDeviceAttributeHostRegisterSupported = 84,
    hipDeviceAttributeClockInstructionRate = 10000,
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 10002,
    hipDeviceAttributeHdpMemFlushCntl = 10005,
    hipDeviceAttributeHdpRegFlushCntl = 10006,
    hipDeviceAttributeIsLargeBar = 10011,
    hipDeviceAttributeAsicRevision = 10012,
    hipDeviceAttributeCanUseStreamWaitValue = 10013,
    hipDeviceAttributeImageSupport = 10014,
    hipDeviceAttributePhysicalMultiProcessorCount = 10015,
    hipDeviceAttributeFineGrainSupport = 10016,
    hipDeviceAttributeWallClockRate = 10017,
    hipDeviceAttributeNumberOfXccs = 10018,
}
pub use hipDeviceAttribute_t::*;

/// 16-byte device UUID (`hipUUID`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct hipUUID {
    pub bytes: [c_char; 16],
}

/// Bitfield structure `hipDeviceArch_t` (4 bytes in C ABI).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct hipDeviceArch_t {
    pub flags: u32,
}

/// HIP device properties structure (`hipDeviceProp_t` / `hipDeviceProp_tR0600` in ROCm 6.0+).
///
/// Exact 1472-byte layout verified against `hip/hip_runtime_api.h` on `x86_64-unknown-linux-gnu`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct hipDeviceProp_t {
    pub name: [c_char; 256],
    pub uuid: hipUUID,
    pub luid: [c_char; 8],
    pub luidDeviceNodeMask: c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: c_int,
    pub warpSize: c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: c_int,
    pub maxThreadsDim: [c_int; 3],
    pub maxGridSize: [c_int; 3],
    pub clockRate: c_int,
    pub totalConstMem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: c_int,
    pub multiProcessorCount: c_int,
    pub kernelExecTimeoutEnabled: c_int,
    pub integrated: c_int,
    pub canMapHostMemory: c_int,
    pub computeMode: c_int,
    pub maxTexture1D: c_int,
    pub maxTexture1DMipmap: c_int,
    pub maxTexture1DLinear: c_int,
    pub maxTexture2D: [c_int; 2],
    pub maxTexture2DMipmap: [c_int; 2],
    pub maxTexture2DLinear: [c_int; 3],
    pub maxTexture2DGather: [c_int; 2],
    pub maxTexture3D: [c_int; 3],
    pub maxTexture3DAlt: [c_int; 3],
    pub maxTextureCubemap: c_int,
    pub maxTexture1DLayered: [c_int; 2],
    pub maxTexture2DLayered: [c_int; 3],
    pub maxTextureCubemapLayered: [c_int; 2],
    pub maxSurface1D: c_int,
    pub maxSurface2D: [c_int; 2],
    pub maxSurface3D: [c_int; 3],
    pub maxSurface1DLayered: [c_int; 2],
    pub maxSurface2DLayered: [c_int; 3],
    pub maxSurfaceCubemap: c_int,
    pub maxSurfaceCubemapLayered: [c_int; 2],
    pub surfaceAlignment: usize,
    pub concurrentKernels: c_int,
    pub ECCEnabled: c_int,
    pub pciBusID: c_int,
    pub pciDeviceID: c_int,
    pub pciDomainID: c_int,
    pub tccDriver: c_int,
    pub asyncEngineCount: c_int,
    pub unifiedAddressing: c_int,
    pub memoryClockRate: c_int,
    pub memoryBusWidth: c_int,
    pub l2CacheSize: c_int,
    pub persistingL2CacheMaxSize: c_int,
    pub maxThreadsPerMultiProcessor: c_int,
    pub streamPrioritiesSupported: c_int,
    pub globalL1CacheSupported: c_int,
    pub localL1CacheSupported: c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: c_int,
    pub managedMemory: c_int,
    pub isMultiGpuBoard: c_int,
    pub multiGpuBoardGroupID: c_int,
    pub hostNativeAtomicSupported: c_int,
    pub singleToDoublePrecisionPerfRatio: c_int,
    pub pageableMemoryAccess: c_int,
    pub concurrentManagedAccess: c_int,
    pub computePreemptionSupported: c_int,
    pub canUseHostPointerForRegisteredMem: c_int,
    pub cooperativeLaunch: c_int,
    pub cooperativeMultiDeviceLaunch: c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: c_int,
    pub directManagedMemAccessFromHost: c_int,
    pub maxBlocksPerMultiProcessor: c_int,
    pub accessPolicyMaxWindowSize: c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: c_int,
    pub sparseHipArraySupported: c_int,
    pub hostRegisterReadOnlySupported: c_int,
    pub timelineSemaphoreInteropSupported: c_int,
    pub memoryPoolsSupported: c_int,
    pub gpuDirectRDMASupported: c_int,
    pub gpuDirectRDMAFlushWritesOptions: c_uint,
    pub gpuDirectRDMAWritesOrdering: c_int,
    pub memoryPoolSupportedHandleTypes: c_uint,
    pub deferredMappingHipArraySupported: c_int,
    pub ipcEventSupported: c_int,
    pub clusterLaunch: c_int,
    pub unifiedFunctionPointers: c_int,
    pub reserved: [c_int; 63],
    pub hipReserved: [c_int; 32],
    pub gcnArchName: [c_char; 256],
    pub maxSharedMemoryPerMultiProcessor: usize,
    pub clockInstructionRate: c_int,
    pub arch: hipDeviceArch_t,
    pub hdpMemFlushCntl: *mut c_uint,
    pub hdpRegFlushCntl: *mut c_uint,
    pub cooperativeMultiDeviceUnmatchedFunc: c_int,
    pub cooperativeMultiDeviceUnmatchedGridDim: c_int,
    pub cooperativeMultiDeviceUnmatchedBlockDim: c_int,
    pub cooperativeMultiDeviceUnmatchedSharedMem: c_int,
    pub isLargeBar: c_int,
    pub asicRevision: c_int,
}

pub type hipDeviceProp_tR0600 = hipDeviceProp_t;

impl Default for hipDeviceProp_t {
    fn default() -> Self {
        // SAFETY: `hipDeviceProp_t` is a plain-old-data C struct where all-zero bits are valid.
        unsafe { std::mem::zeroed() }
    }
}

impl std::fmt::Debug for hipDeviceProp_t {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("hipDeviceProp_t")
            .field("name", &self.name_str())
            .field("gcnArchName", &self.gcn_arch_name_str())
            .field("totalGlobalMem", &self.totalGlobalMem)
            .field("warpSize", &self.warpSize)
            .field("multiProcessorCount", &self.multiProcessorCount)
            .field("integrated", &self.integrated)
            .finish()
    }
}

impl hipDeviceProp_t {
    /// Extract the null-terminated device `name` as a UTF-8 string slice.
    pub fn name_str(&self) -> &str {
        c_char_array_to_str(&self.name)
    }

    /// Extract the null-terminated `gcnArchName` (e.g. `"gfx1151:sramecc-:xnack-"`) as a `&str`.
    pub fn gcn_arch_name_str(&self) -> &str {
        c_char_array_to_str(&self.gcnArchName)
    }
}

fn c_char_array_to_str(buf: &[c_char]) -> &str {
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), buf.len()) };
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..len]).unwrap_or("")
}

fn write_c_char_array(dst: &mut [c_char], s: &str) {
    dst.fill(0);
    let src = s.as_bytes();
    let n = src.len().min(dst.len().saturating_sub(1));
    for (i, &b) in src[..n].iter().enumerate() {
        dst[i] = b as c_char;
    }
}

// ============================================================================
// Optional Link-Time `extern "C"` Declarations (`--features link-hip`)
// ============================================================================

#[cfg(feature = "link-hip")]
pub mod linked_ffi {
    use super::*;

    #[link(name = "amdhip64")]
    extern "C" {
        pub fn hipInit(flags: c_uint) -> hipError_t;
        pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
        pub fn hipSetDevice(deviceId: hipDevice_t) -> hipError_t;
        pub fn hipGetDevice(deviceId: *mut hipDevice_t) -> hipError_t;
        pub fn hipDeviceSynchronize() -> hipError_t;
        pub fn hipGetDevicePropertiesR0600(
            prop: *mut hipDeviceProp_tR0600,
            deviceId: hipDevice_t,
        ) -> hipError_t;
        pub fn hipDeviceGetAttribute(
            pi: *mut c_int,
            attr: hipDeviceAttribute_t,
            deviceId: hipDevice_t,
        ) -> hipError_t;
        pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;
        pub fn hipGetErrorString(hipError: hipError_t) -> *const c_char;
        pub fn hipGetLastError() -> hipError_t;
        pub fn hipPeekAtLastError() -> hipError_t;

        pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
        pub fn hipFree(ptr: *mut c_void) -> hipError_t;
        pub fn hipMemset(dst: *mut c_void, value: c_int, sizeBytes: usize) -> hipError_t;
        pub fn hipMemsetAsync(
            dst: *mut c_void,
            value: c_int,
            sizeBytes: usize,
            stream: hipStream_t,
        ) -> hipError_t;
        pub fn hipMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            sizeBytes: usize,
            kind: hipMemcpyKind,
        ) -> hipError_t;
        pub fn hipMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            sizeBytes: usize,
            kind: hipMemcpyKind,
            stream: hipStream_t,
        ) -> hipError_t;
        pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> hipError_t;
        pub fn hipHostFree(ptr: *mut c_void) -> hipError_t;
        pub fn hipFreeHost(ptr: *mut c_void) -> hipError_t;
        pub fn hipHostRegister(hostPtr: *mut c_void, sizeBytes: usize, flags: c_uint)
            -> hipError_t;
        pub fn hipHostUnregister(hostPtr: *mut c_void) -> hipError_t;
        pub fn hipHostGetDevicePointer(
            devPtr: *mut *mut c_void,
            hstPtr: *mut c_void,
            flags: c_uint,
        ) -> hipError_t;

        pub fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
        pub fn hipStreamCreateWithFlags(stream: *mut hipStream_t, flags: c_uint) -> hipError_t;
        pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
        pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
        pub fn hipStreamQuery(stream: hipStream_t) -> hipError_t;
        pub fn hipStreamWaitEvent(
            stream: hipStream_t,
            event: hipEvent_t,
            flags: c_uint,
        ) -> hipError_t;

        pub fn hipEventCreate(event: *mut hipEvent_t) -> hipError_t;
        pub fn hipEventCreateWithFlags(event: *mut hipEvent_t, flags: c_uint) -> hipError_t;
        pub fn hipEventDestroy(event: hipEvent_t) -> hipError_t;
        pub fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> hipError_t;
        pub fn hipEventSynchronize(event: hipEvent_t) -> hipError_t;
        pub fn hipEventQuery(event: hipEvent_t) -> hipError_t;
        pub fn hipEventElapsedTime(ms: *mut f32, start: hipEvent_t, stop: hipEvent_t)
            -> hipError_t;

        pub fn hipStreamBeginCapture(stream: hipStream_t, mode: hipStreamCaptureMode)
            -> hipError_t;
        pub fn hipStreamEndCapture(stream: hipStream_t, pGraph: *mut hipGraph_t) -> hipError_t;
        pub fn hipStreamIsCapturing(
            stream: hipStream_t,
            pCaptureStatus: *mut hipStreamCaptureStatus,
        ) -> hipError_t;
        pub fn hipGraphGetNodes(
            graph: hipGraph_t,
            nodes: *mut hipGraphNode_t,
            numNodes: *mut usize,
        ) -> hipError_t;
        pub fn hipGraphInstantiate(
            pGraphExec: *mut hipGraphExec_t,
            graph: hipGraph_t,
            pErrorNode: *mut hipGraphNode_t,
            pLogBuffer: *mut c_char,
            bufferSize: usize,
        ) -> hipError_t;
        pub fn hipGraphInstantiateWithFlags(
            pGraphExec: *mut hipGraphExec_t,
            graph: hipGraph_t,
            flags: c_ulonglong,
        ) -> hipError_t;
        pub fn hipGraphLaunch(graphExec: hipGraphExec_t, stream: hipStream_t) -> hipError_t;
        pub fn hipGraphDestroy(graph: hipGraph_t) -> hipError_t;
        pub fn hipGraphExecDestroy(graphExec: hipGraphExec_t) -> hipError_t;
    }
}

// ============================================================================
// Dynamic `libamdhip64.so` Function Table
// ============================================================================

type FnHipInit = unsafe extern "C" fn(c_uint) -> hipError_t;
type FnHipGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> hipError_t;
type FnHipSetDevice = unsafe extern "C" fn(hipDevice_t) -> hipError_t;
type FnHipGetDevice = unsafe extern "C" fn(*mut hipDevice_t) -> hipError_t;
type FnHipDeviceSynchronize = unsafe extern "C" fn() -> hipError_t;
type FnHipGetDeviceProperties =
    unsafe extern "C" fn(*mut hipDeviceProp_t, hipDevice_t) -> hipError_t;
type FnHipDeviceGetAttribute =
    unsafe extern "C" fn(*mut c_int, hipDeviceAttribute_t, hipDevice_t) -> hipError_t;
type FnHipMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> hipError_t;
type FnHipGetErrorString = unsafe extern "C" fn(hipError_t) -> *const c_char;
type FnHipGetLastError = unsafe extern "C" fn() -> hipError_t;
type FnHipPeekAtLastError = unsafe extern "C" fn() -> hipError_t;

type FnHipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> hipError_t;
type FnHipFree = unsafe extern "C" fn(*mut c_void) -> hipError_t;
type FnHipMemset = unsafe extern "C" fn(*mut c_void, c_int, usize) -> hipError_t;
type FnHipMemsetAsync = unsafe extern "C" fn(*mut c_void, c_int, usize, hipStream_t) -> hipError_t;
type FnHipMemcpy =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, hipMemcpyKind) -> hipError_t;
type FnHipMemcpyAsync = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    usize,
    hipMemcpyKind,
    hipStream_t,
) -> hipError_t;
type FnHipHostMalloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> hipError_t;
type FnHipHostFree = unsafe extern "C" fn(*mut c_void) -> hipError_t;
type FnHipHostRegister = unsafe extern "C" fn(*mut c_void, usize, c_uint) -> hipError_t;
type FnHipHostUnregister = unsafe extern "C" fn(*mut c_void) -> hipError_t;
type FnHipHostGetDevicePointer =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> hipError_t;

type FnHipStreamCreate = unsafe extern "C" fn(*mut hipStream_t) -> hipError_t;
type FnHipStreamCreateWithFlags = unsafe extern "C" fn(*mut hipStream_t, c_uint) -> hipError_t;
type FnHipStreamDestroy = unsafe extern "C" fn(hipStream_t) -> hipError_t;
type FnHipStreamSynchronize = unsafe extern "C" fn(hipStream_t) -> hipError_t;
type FnHipStreamQuery = unsafe extern "C" fn(hipStream_t) -> hipError_t;
type FnHipStreamWaitEvent = unsafe extern "C" fn(hipStream_t, hipEvent_t, c_uint) -> hipError_t;

type FnHipEventCreate = unsafe extern "C" fn(*mut hipEvent_t) -> hipError_t;
type FnHipEventCreateWithFlags = unsafe extern "C" fn(*mut hipEvent_t, c_uint) -> hipError_t;
type FnHipEventDestroy = unsafe extern "C" fn(hipEvent_t) -> hipError_t;
type FnHipEventRecord = unsafe extern "C" fn(hipEvent_t, hipStream_t) -> hipError_t;
type FnHipEventSynchronize = unsafe extern "C" fn(hipEvent_t) -> hipError_t;
type FnHipEventQuery = unsafe extern "C" fn(hipEvent_t) -> hipError_t;
type FnHipEventElapsedTime = unsafe extern "C" fn(*mut f32, hipEvent_t, hipEvent_t) -> hipError_t;

type FnHipStreamBeginCapture =
    unsafe extern "C" fn(hipStream_t, hipStreamCaptureMode) -> hipError_t;
type FnHipStreamEndCapture = unsafe extern "C" fn(hipStream_t, *mut hipGraph_t) -> hipError_t;
type FnHipStreamIsCapturing =
    unsafe extern "C" fn(hipStream_t, *mut hipStreamCaptureStatus) -> hipError_t;
type FnHipGraphGetNodes =
    unsafe extern "C" fn(hipGraph_t, *mut hipGraphNode_t, *mut usize) -> hipError_t;
type FnHipGraphInstantiate = unsafe extern "C" fn(
    *mut hipGraphExec_t,
    hipGraph_t,
    *mut hipGraphNode_t,
    *mut c_char,
    usize,
) -> hipError_t;
type FnHipGraphInstantiateWithFlags =
    unsafe extern "C" fn(*mut hipGraphExec_t, hipGraph_t, c_ulonglong) -> hipError_t;
type FnHipGraphLaunch = unsafe extern "C" fn(hipGraphExec_t, hipStream_t) -> hipError_t;
type FnHipGraphDestroy = unsafe extern "C" fn(hipGraph_t) -> hipError_t;
type FnHipGraphExecDestroy = unsafe extern "C" fn(hipGraphExec_t) -> hipError_t;

/// Dynamically loaded `libamdhip64.so` symbol table.
pub struct HipDynamicTable {
    handle: *mut c_void,
    loaded_path: String,
    pub hipInit: FnHipInit,
    pub hipGetDeviceCount: FnHipGetDeviceCount,
    pub hipSetDevice: FnHipSetDevice,
    pub hipGetDevice: FnHipGetDevice,
    pub hipDeviceSynchronize: FnHipDeviceSynchronize,
    pub hipGetDeviceProperties: FnHipGetDeviceProperties,
    pub hipDeviceGetAttribute: FnHipDeviceGetAttribute,
    pub hipMemGetInfo: FnHipMemGetInfo,
    pub hipGetErrorString: FnHipGetErrorString,
    pub hipGetLastError: FnHipGetLastError,
    pub hipPeekAtLastError: FnHipPeekAtLastError,
    pub hipMalloc: FnHipMalloc,
    pub hipFree: FnHipFree,
    pub hipMemset: FnHipMemset,
    pub hipMemsetAsync: FnHipMemsetAsync,
    pub hipMemcpy: FnHipMemcpy,
    pub hipMemcpyAsync: FnHipMemcpyAsync,
    pub hipHostMalloc: FnHipHostMalloc,
    pub hipHostFree: FnHipHostFree,
    pub hipHostRegister: FnHipHostRegister,
    pub hipHostUnregister: FnHipHostUnregister,
    pub hipHostGetDevicePointer: FnHipHostGetDevicePointer,
    pub hipStreamCreate: FnHipStreamCreate,
    pub hipStreamCreateWithFlags: FnHipStreamCreateWithFlags,
    pub hipStreamDestroy: FnHipStreamDestroy,
    pub hipStreamSynchronize: FnHipStreamSynchronize,
    pub hipStreamQuery: FnHipStreamQuery,
    pub hipStreamWaitEvent: FnHipStreamWaitEvent,
    pub hipEventCreate: FnHipEventCreate,
    pub hipEventCreateWithFlags: FnHipEventCreateWithFlags,
    pub hipEventDestroy: FnHipEventDestroy,
    pub hipEventRecord: FnHipEventRecord,
    pub hipEventSynchronize: FnHipEventSynchronize,
    pub hipEventQuery: FnHipEventQuery,
    pub hipEventElapsedTime: FnHipEventElapsedTime,
    pub hipStreamBeginCapture: FnHipStreamBeginCapture,
    pub hipStreamEndCapture: FnHipStreamEndCapture,
    pub hipStreamIsCapturing: FnHipStreamIsCapturing,
    pub hipGraphGetNodes: FnHipGraphGetNodes,
    pub hipGraphInstantiate: FnHipGraphInstantiate,
    pub hipGraphInstantiateWithFlags: Option<FnHipGraphInstantiateWithFlags>,
    pub hipGraphLaunch: FnHipGraphLaunch,
    pub hipGraphDestroy: FnHipGraphDestroy,
    pub hipGraphExecDestroy: FnHipGraphExecDestroy,
}

// SAFETY: `HipDynamicTable` holds an immutable `dlopen` handle and C function pointers.
unsafe impl Send for HipDynamicTable {}
unsafe impl Sync for HipDynamicTable {}

impl std::fmt::Debug for HipDynamicTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipDynamicTable")
            .field("loaded_path", &self.loaded_path)
            .finish()
    }
}

impl HipDynamicTable {
    /// Candidate library names and ROCm SDK paths searched by `dlopen`.
    pub fn candidate_paths() -> Vec<String> {
        let mut out = Vec::new();
        if let Ok(explicit) = std::env::var("TALUS_HIP_LIB") {
            if !explicit.trim().is_empty() {
                out.push(explicit);
            }
        }
        out.push("libamdhip64.so.7".to_string());
        out.push("libamdhip64.so.6".to_string());
        out.push("libamdhip64.so".to_string());

        for env_key in ["ROCM_SDK", "ROCM_PATH"] {
            if let Ok(root) = std::env::var(env_key) {
                let lib_dir = PathBuf::from(root).join("lib");
                for soname in ["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"] {
                    out.push(lib_dir.join(soname).to_string_lossy().into_owned());
                }
            }
        }

        for soname in [
            "/opt/rocm/lib/libamdhip64.so.7",
            "/opt/rocm/lib/libamdhip64.so.6",
            "/opt/rocm/lib/libamdhip64.so",
        ] {
            out.push(soname.to_string());
        }

        out
    }

    /// Open `libamdhip64.so` and resolve all required HIP entry points.
    pub fn load() -> Result<Self, String> {
        let candidates = Self::candidate_paths();
        let mut last_err = String::new();
        for candidate in &candidates {
            let Ok(c_path) = CString::new(candidate.as_str()) else {
                continue;
            };
            // SAFETY: `c_path` is a valid null-terminated string. `RTLD_NOW | RTLD_GLOBAL` ensures
            // `libtalus_kernels.so` shares the same loaded HIP runtime instance (Plan §2.3).
            let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
            if !handle.is_null() {
                match unsafe { Self::from_handle(handle, candidate.clone()) } {
                    Ok(table) => return Ok(table),
                    Err(e) => {
                        unsafe {
                            let _ = dlclose(handle);
                        }
                        last_err = e;
                        continue;
                    }
                }
            }
            last_err = last_dlerror();
        }
        Err(format!(
            "failed to dlopen libamdhip64.so (tried {} candidates; last error: {})",
            candidates.len(),
            last_err
        ))
    }

    unsafe fn sym<T: Copy>(handle: *mut c_void, name: &CStr) -> Result<T, String> {
        let _ = dlerror();
        let raw = dlsym(handle, name.as_ptr());
        if raw.is_null() {
            return Err(format!(
                "missing HIP symbol '{}': {}",
                name.to_string_lossy(),
                last_dlerror()
            ));
        }
        Ok(std::mem::transmute_copy::<*mut c_void, T>(&raw))
    }

    unsafe fn from_handle(handle: *mut c_void, loaded_path: String) -> Result<Self, String> {
        // Prefer ROCm 6.0+ `hipGetDevicePropertiesR0600` (matches 1472-byte `hipDeviceProp_t`),
        // falling back to `hipGetDeviceProperties`.
        let get_props: FnHipGetDeviceProperties = Self::sym(handle, c"hipGetDevicePropertiesR0600")
            .or_else(|_| Self::sym(handle, c"hipGetDeviceProperties"))?;

        // Prefer `hipHostFree`, falling back to `hipFreeHost`.
        let host_free: FnHipHostFree =
            Self::sym(handle, c"hipHostFree").or_else(|_| Self::sym(handle, c"hipFreeHost"))?;

        let graph_inst_flags: Option<FnHipGraphInstantiateWithFlags> =
            Self::sym(handle, c"hipGraphInstantiateWithFlags").ok();

        Ok(Self {
            handle,
            loaded_path,
            hipInit: Self::sym(handle, c"hipInit")?,
            hipGetDeviceCount: Self::sym(handle, c"hipGetDeviceCount")?,
            hipSetDevice: Self::sym(handle, c"hipSetDevice")?,
            hipGetDevice: Self::sym(handle, c"hipGetDevice")?,
            hipDeviceSynchronize: Self::sym(handle, c"hipDeviceSynchronize")?,
            hipGetDeviceProperties: get_props,
            hipDeviceGetAttribute: Self::sym(handle, c"hipDeviceGetAttribute")?,
            hipMemGetInfo: Self::sym(handle, c"hipMemGetInfo")?,
            hipGetErrorString: Self::sym(handle, c"hipGetErrorString")?,
            hipGetLastError: Self::sym(handle, c"hipGetLastError")?,
            hipPeekAtLastError: Self::sym(handle, c"hipPeekAtLastError")?,
            hipMalloc: Self::sym(handle, c"hipMalloc")?,
            hipFree: Self::sym(handle, c"hipFree")?,
            hipMemset: Self::sym(handle, c"hipMemset")?,
            hipMemsetAsync: Self::sym(handle, c"hipMemsetAsync")?,
            hipMemcpy: Self::sym(handle, c"hipMemcpy")?,
            hipMemcpyAsync: Self::sym(handle, c"hipMemcpyAsync")?,
            hipHostMalloc: Self::sym(handle, c"hipHostMalloc")?,
            hipHostFree: host_free,
            hipHostRegister: Self::sym(handle, c"hipHostRegister")?,
            hipHostUnregister: Self::sym(handle, c"hipHostUnregister")?,
            hipHostGetDevicePointer: Self::sym(handle, c"hipHostGetDevicePointer")?,
            hipStreamCreate: Self::sym(handle, c"hipStreamCreate")?,
            hipStreamCreateWithFlags: Self::sym(handle, c"hipStreamCreateWithFlags")?,
            hipStreamDestroy: Self::sym(handle, c"hipStreamDestroy")?,
            hipStreamSynchronize: Self::sym(handle, c"hipStreamSynchronize")?,
            hipStreamQuery: Self::sym(handle, c"hipStreamQuery")?,
            hipStreamWaitEvent: Self::sym(handle, c"hipStreamWaitEvent")?,
            hipEventCreate: Self::sym(handle, c"hipEventCreate")?,
            hipEventCreateWithFlags: Self::sym(handle, c"hipEventCreateWithFlags")?,
            hipEventDestroy: Self::sym(handle, c"hipEventDestroy")?,
            hipEventRecord: Self::sym(handle, c"hipEventRecord")?,
            hipEventSynchronize: Self::sym(handle, c"hipEventSynchronize")?,
            hipEventQuery: Self::sym(handle, c"hipEventQuery")?,
            hipEventElapsedTime: Self::sym(handle, c"hipEventElapsedTime")?,
            hipStreamBeginCapture: Self::sym(handle, c"hipStreamBeginCapture")?,
            hipStreamEndCapture: Self::sym(handle, c"hipStreamEndCapture")?,
            hipStreamIsCapturing: Self::sym(handle, c"hipStreamIsCapturing")?,
            hipGraphGetNodes: Self::sym(handle, c"hipGraphGetNodes")?,
            hipGraphInstantiate: Self::sym(handle, c"hipGraphInstantiate")?,
            hipGraphInstantiateWithFlags: graph_inst_flags,
            hipGraphLaunch: Self::sym(handle, c"hipGraphLaunch")?,
            hipGraphDestroy: Self::sym(handle, c"hipGraphDestroy")?,
            hipGraphExecDestroy: Self::sym(handle, c"hipGraphExecDestroy")?,
        })
    }

    pub fn loaded_path(&self) -> &str {
        &self.loaded_path
    }
}

impl Drop for HipDynamicTable {
    fn drop(&mut self) {
        // Keep shared library loaded for the lifetime of the process if global; for local
        // test instances we intentionally do not `dlclose` a library loaded with `RTLD_GLOBAL`
        // to avoid unmapping global C++ destructors inside `libamdhip64.so`.
        let _ = self.handle;
    }
}

fn last_dlerror() -> String {
    unsafe {
        let err = dlerror();
        if err.is_null() {
            "unknown dlerror".to_string()
        } else {
            CStr::from_ptr(err).to_string_lossy().into_owned()
        }
    }
}

// ============================================================================
// In-Process Simulated HIP Runtime (GPU-less unit testing)
// ============================================================================

thread_local! {
    static SIM_CURRENT_DEVICE: Cell<hipDevice_t> = const { Cell::new(0) };
    static SIM_LAST_ERROR: Cell<hipError_t> = const { Cell::new(hipSuccess) };
}

fn sim_record_status(status: hipError_t) -> hipError_t {
    if !status.is_success() && status != hipErrorNotReady {
        SIM_LAST_ERROR.with(|e| e.set(status));
    }
    status
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AllocKind {
    Device,
    HostMapped,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SimAlloc {
    ptr: usize,
    size: usize,
    layout: Layout,
    device: hipDevice_t,
    kind: AllocKind,
    flags: c_uint,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SimRegisteredHost {
    ptr: usize,
    size: usize,
    device: hipDevice_t,
    flags: c_uint,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum SimGraphOp {
    Memset {
        dst: usize,
        value: u8,
        size_bytes: usize,
    },
    Memcpy {
        dst: usize,
        src: usize,
        size_bytes: usize,
        kind: hipMemcpyKind,
    },
    EventRecord {
        event_handle: usize,
    },
    StreamWaitEvent {
        event_handle: usize,
    },
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SimStream {
    device: hipDevice_t,
    flags: c_uint,
    capture_status: hipStreamCaptureStatus,
    capture_mode: hipStreamCaptureMode,
    captured_ops: Vec<SimGraphOp>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SimEvent {
    device: hipDevice_t,
    flags: c_uint,
    recorded_at: Option<Instant>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SimGraph {
    device: hipDevice_t,
    ops: Vec<SimGraphOp>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SimGraphExec {
    device: hipDevice_t,
    ops: Vec<SimGraphOp>,
}

#[derive(Debug, Clone)]
struct SimDeviceDesc {
    name: &'static str,
    gcn_arch_name: &'static str,
    total_mem: usize,
    shared_mem_per_block: usize,
    warp_size: c_int,
    multi_processor_count: c_int,
    major: c_int,
    minor: c_int,
    integrated: c_int,
    l2_cache_size: c_int,
}

struct SimState {
    devices: Vec<SimDeviceDesc>,
    allocs: HashMap<usize, SimAlloc>,
    registered_host: HashMap<usize, SimRegisteredHost>,
    streams: HashMap<usize, SimStream>,
    events: HashMap<usize, SimEvent>,
    graphs: HashMap<usize, SimGraph>,
    graph_execs: HashMap<usize, SimGraphExec>,
}

static NEXT_SIM_HANDLE: AtomicUsize = AtomicUsize::new(0x1000_0000);

fn next_sim_handle() -> usize {
    NEXT_SIM_HANDLE.fetch_add(0x100, Ordering::Relaxed)
}

/// In-process simulated HIP runtime for CPU-only hosts and deterministic unit tests.
pub struct SimulatedRuntime {
    state: Mutex<SimState>,
}

impl std::fmt::Debug for SimulatedRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimulatedRuntime").finish()
    }
}

impl Default for SimulatedRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulatedRuntime {
    pub fn new() -> Self {
        let devices = vec![
            SimDeviceDesc {
                name: "AMD Radeon Graphics (Simulated Strix Halo)",
                gcn_arch_name: "gfx1151:sramecc-:xnack-",
                total_mem: 128 * 1024 * 1024 * 1024, // 128 GiB unified LPDDR5X
                shared_mem_per_block: 64 * 1024,
                warp_size: 32,
                multi_processor_count: 40,
                major: 11,
                minor: 5,
                integrated: 1,
                l2_cache_size: 32 * 1024 * 1024,
            },
            SimDeviceDesc {
                name: "AMD Radeon AI PRO R9700 (Simulated RDNA4)",
                gcn_arch_name: "gfx1201:sramecc+:xnack-",
                total_mem: 32 * 1024 * 1024 * 1024, // 32 GiB VRAM
                shared_mem_per_block: 64 * 1024,
                warp_size: 32,
                multi_processor_count: 64,
                major: 12,
                minor: 0,
                integrated: 0,
                l2_cache_size: 64 * 1024 * 1024,
            },
        ];
        Self {
            state: Mutex::new(SimState {
                devices,
                allocs: HashMap::new(),
                registered_host: HashMap::new(),
                streams: HashMap::new(),
                events: HashMap::new(),
                graphs: HashMap::new(),
                graph_execs: HashMap::new(),
            }),
        }
    }

    fn current_device(&self) -> hipDevice_t {
        SIM_CURRENT_DEVICE.with(Cell::get)
    }

    fn validate_device(
        st: &SimState,
        device_id: hipDevice_t,
    ) -> Result<&SimDeviceDesc, hipError_t> {
        if device_id < 0 || (device_id as usize) >= st.devices.len() {
            Err(hipErrorInvalidDevice)
        } else {
            Ok(&st.devices[device_id as usize])
        }
    }

    fn alloc_aligned(size: usize) -> Result<(*mut u8, Layout), hipError_t> {
        let alloc_size = size.max(1);
        let layout = Layout::from_size_align(alloc_size, 256).map_err(|_| hipErrorInvalidValue)?;
        // SAFETY: `alloc_size >= 1` and alignment is a power of two (256).
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            Err(hipErrorOutOfMemory)
        } else {
            Ok((ptr, layout))
        }
    }

    unsafe fn execute_memcpy(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        _kind: hipMemcpyKind,
    ) -> hipError_t {
        if size_bytes == 0 {
            return hipSuccess;
        }
        if dst.is_null() || src.is_null() {
            return hipErrorInvalidValue;
        }
        ptr::copy(src.cast::<u8>(), dst.cast::<u8>(), size_bytes);
        hipSuccess
    }

    unsafe fn execute_memset(dst: *mut c_void, value: c_int, size_bytes: usize) -> hipError_t {
        if size_bytes == 0 {
            return hipSuccess;
        }
        if dst.is_null() {
            return hipErrorInvalidValue;
        }
        ptr::write_bytes(dst.cast::<u8>(), value as u8, size_bytes);
        hipSuccess
    }
}

impl Drop for SimulatedRuntime {
    fn drop(&mut self) {
        if let Ok(mut st) = self.state.lock() {
            for (_, alloc) in st.allocs.drain() {
                unsafe {
                    alloc::dealloc(alloc.ptr as *mut u8, alloc.layout);
                }
            }
        }
    }
}

// ============================================================================
// Unified `HipRuntime` Dispatcher (Dynamic `libamdhip64.so` or `Simulated`)
// ============================================================================

/// Execution backend mode for [`HipRuntime`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HipBackendMode {
    /// Probe `libamdhip64.so` and check for usable AMD GPU devices; automatically fall back to
    /// [`Self::Simulated`] when no GPU hardware is present.
    #[default]
    Auto,
    /// Require the real dynamically loaded `libamdhip64.so`.
    Dynamic,
    /// Use the in-process simulated CPU runtime.
    Simulated,
}

enum RuntimeInner {
    Dynamic(HipDynamicTable),
    Simulated(SimulatedRuntime),
}

/// Runtime HIP API table supporting both dynamic `libamdhip64.so` loading and in-process simulation.
pub struct HipRuntime {
    inner: RuntimeInner,
}

impl std::fmt::Debug for HipRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            RuntimeInner::Dynamic(d) => f.debug_tuple("HipRuntime::Dynamic").field(d).finish(),
            RuntimeInner::Simulated(s) => f.debug_tuple("HipRuntime::Simulated").field(s).finish(),
        }
    }
}

static GLOBAL_RUNTIME: OnceLock<HipRuntime> = OnceLock::new();

impl HipRuntime {
    /// Return the process-wide global [`HipRuntime`] instance.
    pub fn global() -> &'static Self {
        GLOBAL_RUNTIME.get_or_init(|| {
            let mode = match std::env::var("TALUS_HIP_BACKEND")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str()
            {
                "sim" | "simulated" | "mock" => HipBackendMode::Simulated,
                "dynamic" | "real" | "hardware" => HipBackendMode::Dynamic,
                _ => HipBackendMode::Auto,
            };
            Self::new(mode).unwrap_or_else(|_| Self::simulated())
        })
    }

    /// Construct a [`HipRuntime`] with the requested [`HipBackendMode`].
    pub fn new(mode: HipBackendMode) -> Result<Self, String> {
        match mode {
            HipBackendMode::Simulated => Ok(Self::simulated()),
            HipBackendMode::Dynamic => Self::load_dynamic(),
            HipBackendMode::Auto => {
                if let Ok(table) = HipDynamicTable::load() {
                    let mut count: c_int = 0;
                    // SAFETY: `table.hipInit` and `table.hipGetDeviceCount` are resolved from `libamdhip64.so`.
                    let init_rc = unsafe { (table.hipInit)(0) };
                    let cnt_rc = unsafe { (table.hipGetDeviceCount)(&mut count) };
                    if init_rc == hipSuccess && cnt_rc == hipSuccess && count > 0 {
                        return Ok(Self {
                            inner: RuntimeInner::Dynamic(table),
                        });
                    }
                    // Clear any sticky error on the dynamic runtime before falling back to Simulated.
                    unsafe {
                        let _ = (table.hipGetLastError)();
                    }
                }
                Ok(Self::simulated())
            }
        }
    }

    /// Explicitly load `libamdhip64.so` via `dlopen`/`dlsym` (even if no GPU device is attached).
    pub fn load_dynamic() -> Result<Self, String> {
        let table = HipDynamicTable::load()?;
        Ok(Self {
            inner: RuntimeInner::Dynamic(table),
        })
    }

    /// Construct an in-process simulated HIP runtime.
    pub fn simulated() -> Self {
        Self {
            inner: RuntimeInner::Simulated(SimulatedRuntime::new()),
        }
    }

    /// True if backed by a dynamically loaded `libamdhip64.so`.
    pub const fn is_dynamic(&self) -> bool {
        matches!(self.inner, RuntimeInner::Dynamic(_))
    }

    /// True if backed by the in-process simulated runtime.
    pub const fn is_simulated(&self) -> bool {
        matches!(self.inner, RuntimeInner::Simulated(_))
    }

    /// Active backend mode (`Dynamic` or `Simulated`).
    pub const fn backend_mode(&self) -> HipBackendMode {
        match self.inner {
            RuntimeInner::Dynamic(_) => HipBackendMode::Dynamic,
            RuntimeInner::Simulated(_) => HipBackendMode::Simulated,
        }
    }

    // ------------------------------------------------------------------------
    // Device Management
    // ------------------------------------------------------------------------

    pub unsafe fn hipInit(&self, flags: c_uint) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipInit)(flags),
            RuntimeInner::Simulated(_) => {
                if flags != 0 {
                    sim_record_status(hipErrorInvalidValue)
                } else {
                    hipSuccess
                }
            }
        }
    }

    pub unsafe fn hipGetDeviceCount(&self, count: *mut c_int) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipGetDeviceCount)(count),
            RuntimeInner::Simulated(s) => {
                if count.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let st = s.state.lock().unwrap();
                *count = st.devices.len() as c_int;
                hipSuccess
            }
        }
    }

    pub unsafe fn hipSetDevice(&self, device_id: hipDevice_t) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipSetDevice)(device_id),
            RuntimeInner::Simulated(s) => {
                let st = s.state.lock().unwrap();
                if let Err(e) = SimulatedRuntime::validate_device(&st, device_id) {
                    return sim_record_status(e);
                }
                SIM_CURRENT_DEVICE.with(|c| c.set(device_id));
                hipSuccess
            }
        }
    }

    pub unsafe fn hipGetDevice(&self, device_id: *mut hipDevice_t) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipGetDevice)(device_id),
            RuntimeInner::Simulated(s) => {
                if device_id.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                *device_id = s.current_device();
                hipSuccess
            }
        }
    }

    pub unsafe fn hipDeviceSynchronize(&self) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipDeviceSynchronize)(),
            RuntimeInner::Simulated(_) => hipSuccess,
        }
    }

    pub unsafe fn hipGetDeviceProperties(
        &self,
        prop: *mut hipDeviceProp_t,
        device_id: hipDevice_t,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipGetDeviceProperties)(prop, device_id),
            RuntimeInner::Simulated(s) => {
                if prop.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let st = s.state.lock().unwrap();
                let desc = match SimulatedRuntime::validate_device(&st, device_id) {
                    Ok(d) => d,
                    Err(e) => return sim_record_status(e),
                };
                let mut out = hipDeviceProp_t::default();
                write_c_char_array(&mut out.name, desc.name);
                write_c_char_array(&mut out.gcnArchName, desc.gcn_arch_name);
                out.totalGlobalMem = desc.total_mem;
                out.sharedMemPerBlock = desc.shared_mem_per_block;
                out.warpSize = desc.warp_size;
                out.maxThreadsPerBlock = 1024;
                out.maxThreadsDim = [1024, 1024, 1024];
                out.maxGridSize = [i32::MAX, 65535, 65535];
                out.major = desc.major;
                out.minor = desc.minor;
                out.multiProcessorCount = desc.multi_processor_count;
                out.integrated = desc.integrated;
                out.canMapHostMemory = 1;
                out.concurrentKernels = 1;
                out.unifiedAddressing = 1;
                out.l2CacheSize = desc.l2_cache_size;
                out.canUseHostPointerForRegisteredMem = 1;
                out.hostRegisterSupported = 1;
                out.maxSharedMemoryPerMultiProcessor = desc.shared_mem_per_block;
                *prop = out;
                hipSuccess
            }
        }
    }

    pub unsafe fn hipDeviceGetAttribute(
        &self,
        pi: *mut c_int,
        attr: hipDeviceAttribute_t,
        device_id: hipDevice_t,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipDeviceGetAttribute)(pi, attr, device_id),
            RuntimeInner::Simulated(s) => {
                if pi.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let st = s.state.lock().unwrap();
                let desc = match SimulatedRuntime::validate_device(&st, device_id) {
                    Ok(d) => d,
                    Err(e) => return sim_record_status(e),
                };
                *pi = match attr {
                    hipDeviceAttributeWarpSize => desc.warp_size,
                    hipDeviceAttributeMultiprocessorCount
                    | hipDeviceAttributePhysicalMultiProcessorCount => desc.multi_processor_count,
                    hipDeviceAttributeMaxThreadsPerBlock => 1024,
                    hipDeviceAttributeMaxSharedMemoryPerBlock
                    | hipDeviceAttributeMaxSharedMemoryPerMultiprocessor => {
                        desc.shared_mem_per_block as c_int
                    }
                    hipDeviceAttributeIntegrated => desc.integrated,
                    hipDeviceAttributeCanMapHostMemory
                    | hipDeviceAttributeCanUseHostPointerForRegisteredMem
                    | hipDeviceAttributeHostRegisterSupported
                    | hipDeviceAttributeUnifiedAddressing
                    | hipDeviceAttributeConcurrentKernels => 1,
                    hipDeviceAttributeComputeCapabilityMajor => desc.major,
                    hipDeviceAttributeComputeCapabilityMinor => desc.minor,
                    hipDeviceAttributeL2CacheSize => desc.l2_cache_size,
                    _ => 0,
                };
                hipSuccess
            }
        }
    }

    pub unsafe fn hipMemGetInfo(&self, free: *mut usize, total: *mut usize) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipMemGetInfo)(free, total),
            RuntimeInner::Simulated(s) => {
                if free.is_null() || total.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let dev = s.current_device();
                let st = s.state.lock().unwrap();
                let desc = match SimulatedRuntime::validate_device(&st, dev) {
                    Ok(d) => d,
                    Err(e) => return sim_record_status(e),
                };
                let used: usize = st
                    .allocs
                    .values()
                    .filter(|a| a.device == dev)
                    .map(|a| a.size)
                    .sum();
                *total = desc.total_mem;
                *free = desc.total_mem.saturating_sub(used);
                hipSuccess
            }
        }
    }

    pub unsafe fn hipGetErrorString(&self, hip_error: hipError_t) -> *const c_char {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipGetErrorString)(hip_error),
            RuntimeInner::Simulated(_) => hip_error.c_error_string().as_ptr(),
        }
    }

    pub unsafe fn hipGetLastError(&self) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipGetLastError)(),
            RuntimeInner::Simulated(_) => SIM_LAST_ERROR.with(|e| e.replace(hipSuccess)),
        }
    }

    pub unsafe fn hipPeekAtLastError(&self) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipPeekAtLastError)(),
            RuntimeInner::Simulated(_) => SIM_LAST_ERROR.with(Cell::get),
        }
    }

    // ------------------------------------------------------------------------
    // Memory Management
    // ------------------------------------------------------------------------

    pub unsafe fn hipMalloc(&self, ptr_out: *mut *mut c_void, size: usize) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipMalloc)(ptr_out, size),
            RuntimeInner::Simulated(s) => {
                if ptr_out.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                if size == 0 {
                    *ptr_out = ptr::null_mut();
                    return hipSuccess;
                }
                let dev = s.current_device();
                let mut st = s.state.lock().unwrap();
                let desc = match SimulatedRuntime::validate_device(&st, dev) {
                    Ok(d) => d,
                    Err(e) => return sim_record_status(e),
                };
                if size > desc.total_mem {
                    return sim_record_status(hipErrorOutOfMemory);
                }
                let (raw, layout) = match SimulatedRuntime::alloc_aligned(size) {
                    Ok(v) => v,
                    Err(e) => return sim_record_status(e),
                };
                let addr = raw as usize;
                st.allocs.insert(
                    addr,
                    SimAlloc {
                        ptr: addr,
                        size,
                        layout,
                        device: dev,
                        kind: AllocKind::Device,
                        flags: 0,
                    },
                );
                *ptr_out = raw.cast::<c_void>();
                hipSuccess
            }
        }
    }

    pub unsafe fn hipFree(&self, ptr_in: *mut c_void) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipFree)(ptr_in),
            RuntimeInner::Simulated(s) => {
                if ptr_in.is_null() {
                    return hipSuccess;
                }
                let addr = ptr_in as usize;
                let mut st = s.state.lock().unwrap();
                let Some(entry) = st.allocs.get(&addr) else {
                    return sim_record_status(hipErrorInvalidDevicePointer);
                };
                if entry.kind != AllocKind::Device {
                    return sim_record_status(hipErrorInvalidDevicePointer);
                }
                let entry = st.allocs.remove(&addr).unwrap();
                alloc::dealloc(entry.ptr as *mut u8, entry.layout);
                hipSuccess
            }
        }
    }

    pub unsafe fn hipMemset(
        &self,
        dst: *mut c_void,
        value: c_int,
        size_bytes: usize,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipMemset)(dst, value, size_bytes),
            RuntimeInner::Simulated(_) => {
                sim_record_status(SimulatedRuntime::execute_memset(dst, value, size_bytes))
            }
        }
    }

    pub unsafe fn hipMemsetAsync(
        &self,
        dst: *mut c_void,
        value: c_int,
        size_bytes: usize,
        stream: hipStream_t,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipMemsetAsync)(dst, value, size_bytes, stream),
            RuntimeInner::Simulated(s) => {
                if size_bytes > 0 && dst.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                if !stream.is_null() {
                    let mut st = s.state.lock().unwrap();
                    let Some(s_info) = st.streams.get_mut(&(stream as usize)) else {
                        return sim_record_status(hipErrorInvalidResourceHandle);
                    };
                    if s_info.capture_status == hipStreamCaptureStatusInvalidated {
                        return sim_record_status(hipErrorStreamCaptureInvalidated);
                    }
                    if s_info.capture_status == hipStreamCaptureStatusActive {
                        s_info.captured_ops.push(SimGraphOp::Memset {
                            dst: dst as usize,
                            value: value as u8,
                            size_bytes,
                        });
                        return hipSuccess;
                    }
                }
                sim_record_status(SimulatedRuntime::execute_memset(dst, value, size_bytes))
            }
        }
    }

    pub unsafe fn hipMemcpy(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipMemcpy)(dst, src, size_bytes, kind),
            RuntimeInner::Simulated(_) => {
                sim_record_status(SimulatedRuntime::execute_memcpy(dst, src, size_bytes, kind))
            }
        }
    }

    pub unsafe fn hipMemcpyAsync(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipMemcpyAsync)(dst, src, size_bytes, kind, stream),
            RuntimeInner::Simulated(s) => {
                if size_bytes > 0 && (dst.is_null() || src.is_null()) {
                    return sim_record_status(hipErrorInvalidValue);
                }
                if !stream.is_null() {
                    let mut st = s.state.lock().unwrap();
                    let Some(s_info) = st.streams.get_mut(&(stream as usize)) else {
                        return sim_record_status(hipErrorInvalidResourceHandle);
                    };
                    if s_info.capture_status == hipStreamCaptureStatusInvalidated {
                        return sim_record_status(hipErrorStreamCaptureInvalidated);
                    }
                    if s_info.capture_status == hipStreamCaptureStatusActive {
                        s_info.captured_ops.push(SimGraphOp::Memcpy {
                            dst: dst as usize,
                            src: src as usize,
                            size_bytes,
                            kind,
                        });
                        return hipSuccess;
                    }
                }
                sim_record_status(SimulatedRuntime::execute_memcpy(dst, src, size_bytes, kind))
            }
        }
    }

    pub unsafe fn hipHostMalloc(
        &self,
        ptr_out: *mut *mut c_void,
        size: usize,
        flags: c_uint,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipHostMalloc)(ptr_out, size, flags),
            RuntimeInner::Simulated(s) => {
                if ptr_out.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                if size == 0 {
                    *ptr_out = ptr::null_mut();
                    return hipSuccess;
                }
                let dev = s.current_device();
                let mut st = s.state.lock().unwrap();
                let (raw, layout) = match SimulatedRuntime::alloc_aligned(size) {
                    Ok(v) => v,
                    Err(e) => return sim_record_status(e),
                };
                let addr = raw as usize;
                st.allocs.insert(
                    addr,
                    SimAlloc {
                        ptr: addr,
                        size,
                        layout,
                        device: dev,
                        kind: AllocKind::HostMapped,
                        flags,
                    },
                );
                *ptr_out = raw.cast::<c_void>();
                hipSuccess
            }
        }
    }

    pub unsafe fn hipHostFree(&self, ptr_in: *mut c_void) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipHostFree)(ptr_in),
            RuntimeInner::Simulated(s) => {
                if ptr_in.is_null() {
                    return hipSuccess;
                }
                let addr = ptr_in as usize;
                let mut st = s.state.lock().unwrap();
                let Some(entry) = st.allocs.get(&addr) else {
                    return sim_record_status(hipErrorInvalidValue);
                };
                if entry.kind != AllocKind::HostMapped {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let entry = st.allocs.remove(&addr).unwrap();
                alloc::dealloc(entry.ptr as *mut u8, entry.layout);
                hipSuccess
            }
        }
    }

    pub unsafe fn hipHostRegister(
        &self,
        host_ptr: *mut c_void,
        size_bytes: usize,
        flags: c_uint,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipHostRegister)(host_ptr, size_bytes, flags),
            RuntimeInner::Simulated(s) => {
                if host_ptr.is_null() || size_bytes == 0 {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let addr = host_ptr as usize;
                let dev = s.current_device();
                let mut st = s.state.lock().unwrap();
                if st.registered_host.contains_key(&addr) {
                    return sim_record_status(hipErrorHostMemoryAlreadyRegistered);
                }
                st.registered_host.insert(
                    addr,
                    SimRegisteredHost {
                        ptr: addr,
                        size: size_bytes,
                        device: dev,
                        flags,
                    },
                );
                hipSuccess
            }
        }
    }

    pub unsafe fn hipHostUnregister(&self, host_ptr: *mut c_void) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipHostUnregister)(host_ptr),
            RuntimeInner::Simulated(s) => {
                if host_ptr.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let addr = host_ptr as usize;
                let mut st = s.state.lock().unwrap();
                if st.registered_host.remove(&addr).is_none() {
                    return sim_record_status(hipErrorHostMemoryNotRegistered);
                }
                hipSuccess
            }
        }
    }

    pub unsafe fn hipHostGetDevicePointer(
        &self,
        dev_ptr: *mut *mut c_void,
        hst_ptr: *mut c_void,
        flags: c_uint,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipHostGetDevicePointer)(dev_ptr, hst_ptr, flags),
            RuntimeInner::Simulated(s) => {
                if dev_ptr.is_null() || hst_ptr.is_null() || flags != 0 {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let addr = hst_ptr as usize;
                let st = s.state.lock().unwrap();
                let in_host_alloc = st.allocs.values().any(|a| {
                    a.kind == AllocKind::HostMapped && addr >= a.ptr && addr < a.ptr + a.size
                });
                let in_registered = st
                    .registered_host
                    .values()
                    .any(|r| addr >= r.ptr && addr < r.ptr + r.size);
                if !in_host_alloc && !in_registered {
                    return sim_record_status(hipErrorInvalidValue);
                }
                *dev_ptr = hst_ptr;
                hipSuccess
            }
        }
    }

    // ------------------------------------------------------------------------
    // Streams & Events
    // ------------------------------------------------------------------------

    pub unsafe fn hipStreamCreate(&self, stream: *mut hipStream_t) -> hipError_t {
        self.hipStreamCreateWithFlags(stream, hipStreamDefault)
    }

    pub unsafe fn hipStreamCreateWithFlags(
        &self,
        stream: *mut hipStream_t,
        flags: c_uint,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipStreamCreateWithFlags)(stream, flags),
            RuntimeInner::Simulated(s) => {
                if stream.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                if flags != hipStreamDefault && flags != hipStreamNonBlocking {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let dev = s.current_device();
                let mut st = s.state.lock().unwrap();
                if let Err(e) = SimulatedRuntime::validate_device(&st, dev) {
                    return sim_record_status(e);
                }
                let id = next_sim_handle();
                st.streams.insert(
                    id,
                    SimStream {
                        device: dev,
                        flags,
                        capture_status: hipStreamCaptureStatusNone,
                        capture_mode: hipStreamCaptureModeGlobal,
                        captured_ops: Vec::new(),
                    },
                );
                *stream = id as hipStream_t;
                hipSuccess
            }
        }
    }

    pub unsafe fn hipStreamDestroy(&self, stream: hipStream_t) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipStreamDestroy)(stream),
            RuntimeInner::Simulated(s) => {
                if stream.is_null() {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                }
                let mut st = s.state.lock().unwrap();
                if st.streams.remove(&(stream as usize)).is_none() {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                }
                hipSuccess
            }
        }
    }

    pub unsafe fn hipStreamSynchronize(&self, stream: hipStream_t) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipStreamSynchronize)(stream),
            RuntimeInner::Simulated(s) => {
                if stream.is_null() {
                    return hipSuccess;
                }
                let mut st = s.state.lock().unwrap();
                let Some(s_info) = st.streams.get_mut(&(stream as usize)) else {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                };
                if s_info.capture_status != hipStreamCaptureStatusNone {
                    s_info.capture_status = hipStreamCaptureStatusInvalidated;
                    return sim_record_status(hipErrorStreamCaptureUnsupported);
                }
                hipSuccess
            }
        }
    }

    pub unsafe fn hipStreamQuery(&self, stream: hipStream_t) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipStreamQuery)(stream),
            RuntimeInner::Simulated(s) => {
                if stream.is_null() {
                    return hipSuccess;
                }
                let mut st = s.state.lock().unwrap();
                let Some(s_info) = st.streams.get_mut(&(stream as usize)) else {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                };
                if s_info.capture_status != hipStreamCaptureStatusNone {
                    s_info.capture_status = hipStreamCaptureStatusInvalidated;
                    return sim_record_status(hipErrorStreamCaptureUnsupported);
                }
                hipSuccess
            }
        }
    }

    pub unsafe fn hipStreamWaitEvent(
        &self,
        stream: hipStream_t,
        event: hipEvent_t,
        flags: c_uint,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipStreamWaitEvent)(stream, event, flags),
            RuntimeInner::Simulated(s) => {
                if event.is_null() || flags != 0 {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let mut st = s.state.lock().unwrap();
                if !st.events.contains_key(&(event as usize)) {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                }
                if !stream.is_null() {
                    let Some(s_info) = st.streams.get_mut(&(stream as usize)) else {
                        return sim_record_status(hipErrorInvalidResourceHandle);
                    };
                    if s_info.capture_status == hipStreamCaptureStatusInvalidated {
                        return sim_record_status(hipErrorStreamCaptureInvalidated);
                    }
                    if s_info.capture_status == hipStreamCaptureStatusActive {
                        s_info.captured_ops.push(SimGraphOp::StreamWaitEvent {
                            event_handle: event as usize,
                        });
                    }
                }
                hipSuccess
            }
        }
    }

    pub unsafe fn hipEventCreate(&self, event: *mut hipEvent_t) -> hipError_t {
        self.hipEventCreateWithFlags(event, hipEventDefault)
    }

    pub unsafe fn hipEventCreateWithFlags(
        &self,
        event: *mut hipEvent_t,
        flags: c_uint,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipEventCreateWithFlags)(event, flags),
            RuntimeInner::Simulated(s) => {
                if event.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let dev = s.current_device();
                let mut st = s.state.lock().unwrap();
                if let Err(e) = SimulatedRuntime::validate_device(&st, dev) {
                    return sim_record_status(e);
                }
                let id = next_sim_handle();
                st.events.insert(
                    id,
                    SimEvent {
                        device: dev,
                        flags,
                        recorded_at: None,
                    },
                );
                *event = id as hipEvent_t;
                hipSuccess
            }
        }
    }

    pub unsafe fn hipEventDestroy(&self, event: hipEvent_t) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipEventDestroy)(event),
            RuntimeInner::Simulated(s) => {
                if event.is_null() {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                }
                let mut st = s.state.lock().unwrap();
                if st.events.remove(&(event as usize)).is_none() {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                }
                hipSuccess
            }
        }
    }

    pub unsafe fn hipEventRecord(&self, event: hipEvent_t, stream: hipStream_t) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipEventRecord)(event, stream),
            RuntimeInner::Simulated(s) => {
                if event.is_null() {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                }
                let mut st = s.state.lock().unwrap();
                if !st.events.contains_key(&(event as usize)) {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                }
                if !stream.is_null() {
                    let Some(s_info) = st.streams.get_mut(&(stream as usize)) else {
                        return sim_record_status(hipErrorInvalidResourceHandle);
                    };
                    if s_info.capture_status == hipStreamCaptureStatusInvalidated {
                        return sim_record_status(hipErrorStreamCaptureInvalidated);
                    }
                    if s_info.capture_status == hipStreamCaptureStatusActive {
                        s_info.captured_ops.push(SimGraphOp::EventRecord {
                            event_handle: event as usize,
                        });
                    }
                }
                if let Some(ev) = st.events.get_mut(&(event as usize)) {
                    ev.recorded_at = Some(Instant::now());
                }
                hipSuccess
            }
        }
    }

    pub unsafe fn hipEventSynchronize(&self, event: hipEvent_t) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipEventSynchronize)(event),
            RuntimeInner::Simulated(s) => {
                if event.is_null() {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                }
                let st = s.state.lock().unwrap();
                if !st.events.contains_key(&(event as usize)) {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                }
                hipSuccess
            }
        }
    }

    pub unsafe fn hipEventQuery(&self, event: hipEvent_t) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipEventQuery)(event),
            RuntimeInner::Simulated(s) => {
                if event.is_null() {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                }
                let st = s.state.lock().unwrap();
                if !st.events.contains_key(&(event as usize)) {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                }
                hipSuccess
            }
        }
    }

    pub unsafe fn hipEventElapsedTime(
        &self,
        ms: *mut f32,
        start: hipEvent_t,
        stop: hipEvent_t,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipEventElapsedTime)(ms, start, stop),
            RuntimeInner::Simulated(s) => {
                if ms.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                if start.is_null() || stop.is_null() {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                }
                let st = s.state.lock().unwrap();
                let (Some(ev_start), Some(ev_stop)) = (
                    st.events.get(&(start as usize)),
                    st.events.get(&(stop as usize)),
                ) else {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                };
                if (ev_start.flags & hipEventDisableTiming) != 0
                    || (ev_stop.flags & hipEventDisableTiming) != 0
                {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                }
                let (Some(t0), Some(t1)) = (ev_start.recorded_at, ev_stop.recorded_at) else {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                };
                let dur = if t1 >= t0 {
                    t1.duration_since(t0)
                } else {
                    t0.duration_since(t1)
                };
                *ms = dur.as_secs_f32() * 1000.0;
                hipSuccess
            }
        }
    }

    // ------------------------------------------------------------------------
    // HIP Graphs
    // ------------------------------------------------------------------------

    pub unsafe fn hipStreamBeginCapture(
        &self,
        stream: hipStream_t,
        mode: hipStreamCaptureMode,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipStreamBeginCapture)(stream, mode),
            RuntimeInner::Simulated(s) => {
                if stream.is_null() {
                    return sim_record_status(hipErrorStreamCaptureUnsupported);
                }
                let mut st = s.state.lock().unwrap();
                let Some(s_info) = st.streams.get_mut(&(stream as usize)) else {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                };
                if s_info.capture_status != hipStreamCaptureStatusNone {
                    return sim_record_status(hipErrorIllegalState);
                }
                s_info.capture_status = hipStreamCaptureStatusActive;
                s_info.capture_mode = mode;
                s_info.captured_ops.clear();
                hipSuccess
            }
        }
    }

    pub unsafe fn hipStreamEndCapture(
        &self,
        stream: hipStream_t,
        p_graph: *mut hipGraph_t,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipStreamEndCapture)(stream, p_graph),
            RuntimeInner::Simulated(s) => {
                if p_graph.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                *p_graph = ptr::null_mut();
                if stream.is_null() {
                    return sim_record_status(hipErrorIllegalState);
                }
                let mut st = s.state.lock().unwrap();
                let Some(s_info) = st.streams.get_mut(&(stream as usize)) else {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                };
                match s_info.capture_status {
                    hipStreamCaptureStatusNone => sim_record_status(hipErrorIllegalState),
                    hipStreamCaptureStatusInvalidated => {
                        s_info.capture_status = hipStreamCaptureStatusNone;
                        s_info.captured_ops.clear();
                        sim_record_status(hipErrorStreamCaptureInvalidated)
                    }
                    hipStreamCaptureStatusActive => {
                        s_info.capture_status = hipStreamCaptureStatusNone;
                        let ops = std::mem::take(&mut s_info.captured_ops);
                        let dev = s_info.device;
                        let graph_id = next_sim_handle();
                        st.graphs.insert(graph_id, SimGraph { device: dev, ops });
                        *p_graph = graph_id as hipGraph_t;
                        hipSuccess
                    }
                }
            }
        }
    }

    pub unsafe fn hipStreamIsCapturing(
        &self,
        stream: hipStream_t,
        p_capture_status: *mut hipStreamCaptureStatus,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipStreamIsCapturing)(stream, p_capture_status),
            RuntimeInner::Simulated(s) => {
                if p_capture_status.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                if stream.is_null() {
                    *p_capture_status = hipStreamCaptureStatusNone;
                    return hipSuccess;
                }
                let st = s.state.lock().unwrap();
                let Some(s_info) = st.streams.get(&(stream as usize)) else {
                    return sim_record_status(hipErrorInvalidResourceHandle);
                };
                *p_capture_status = s_info.capture_status;
                hipSuccess
            }
        }
    }

    pub unsafe fn hipGraphGetNodes(
        &self,
        graph: hipGraph_t,
        nodes: *mut hipGraphNode_t,
        num_nodes: *mut usize,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipGraphGetNodes)(graph, nodes, num_nodes),
            RuntimeInner::Simulated(s) => {
                if graph.is_null() || num_nodes.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let st = s.state.lock().unwrap();
                let Some(g_info) = st.graphs.get(&(graph as usize)) else {
                    return sim_record_status(hipErrorInvalidValue);
                };
                *num_nodes = g_info.ops.len();
                if !nodes.is_null() {
                    for i in 0..g_info.ops.len() {
                        *nodes.add(i) = (i + 1) as hipGraphNode_t;
                    }
                }
                hipSuccess
            }
        }
    }

    pub unsafe fn hipGraphInstantiate(
        &self,
        p_graph_exec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        p_error_node: *mut hipGraphNode_t,
        p_log_buffer: *mut c_char,
        buffer_size: usize,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipGraphInstantiate)(
                p_graph_exec,
                graph,
                p_error_node,
                p_log_buffer,
                buffer_size,
            ),
            RuntimeInner::Simulated(s) => {
                if p_graph_exec.is_null() || graph.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                *p_graph_exec = ptr::null_mut();
                let mut st = s.state.lock().unwrap();
                let Some(g_info) = st.graphs.get(&(graph as usize)).cloned() else {
                    return sim_record_status(hipErrorInvalidValue);
                };
                let exec_id = next_sim_handle();
                st.graph_execs.insert(
                    exec_id,
                    SimGraphExec {
                        device: g_info.device,
                        ops: g_info.ops,
                    },
                );
                *p_graph_exec = exec_id as hipGraphExec_t;
                hipSuccess
            }
        }
    }

    pub unsafe fn hipGraphInstantiateWithFlags(
        &self,
        p_graph_exec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        flags: c_ulonglong,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => {
                if let Some(f) = d.hipGraphInstantiateWithFlags {
                    f(p_graph_exec, graph, flags)
                } else {
                    (d.hipGraphInstantiate)(
                        p_graph_exec,
                        graph,
                        ptr::null_mut(),
                        ptr::null_mut(),
                        0,
                    )
                }
            }
            RuntimeInner::Simulated(_) => {
                let _ = flags;
                self.hipGraphInstantiate(p_graph_exec, graph, ptr::null_mut(), ptr::null_mut(), 0)
            }
        }
    }

    pub unsafe fn hipGraphLaunch(
        &self,
        graph_exec: hipGraphExec_t,
        stream: hipStream_t,
    ) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipGraphLaunch)(graph_exec, stream),
            RuntimeInner::Simulated(s) => {
                if graph_exec.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let ops = {
                    let mut st = s.state.lock().unwrap();
                    if !stream.is_null() && !st.streams.contains_key(&(stream as usize)) {
                        return sim_record_status(hipErrorInvalidResourceHandle);
                    }
                    let Some(exec_info) = st.graph_execs.get(&(graph_exec as usize)) else {
                        return sim_record_status(hipErrorInvalidValue);
                    };
                    let ops = exec_info.ops.clone();
                    for op in &ops {
                        if let SimGraphOp::EventRecord { event_handle } = op {
                            if let Some(ev) = st.events.get_mut(event_handle) {
                                ev.recorded_at = Some(Instant::now());
                            }
                        }
                    }
                    ops
                };

                for op in ops {
                    match op {
                        SimGraphOp::Memset {
                            dst,
                            value,
                            size_bytes,
                        } => {
                            let rc = SimulatedRuntime::execute_memset(
                                dst as *mut c_void,
                                value as c_int,
                                size_bytes,
                            );
                            if !rc.is_success() {
                                return sim_record_status(rc);
                            }
                        }
                        SimGraphOp::Memcpy {
                            dst,
                            src,
                            size_bytes,
                            kind,
                        } => {
                            let rc = SimulatedRuntime::execute_memcpy(
                                dst as *mut c_void,
                                src as *const c_void,
                                size_bytes,
                                kind,
                            );
                            if !rc.is_success() {
                                return sim_record_status(rc);
                            }
                        }
                        SimGraphOp::EventRecord { .. } | SimGraphOp::StreamWaitEvent { .. } => {}
                    }
                }
                hipSuccess
            }
        }
    }

    pub unsafe fn hipGraphDestroy(&self, graph: hipGraph_t) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipGraphDestroy)(graph),
            RuntimeInner::Simulated(s) => {
                if graph.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let mut st = s.state.lock().unwrap();
                if st.graphs.remove(&(graph as usize)).is_none() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                hipSuccess
            }
        }
    }

    pub unsafe fn hipGraphExecDestroy(&self, graph_exec: hipGraphExec_t) -> hipError_t {
        match &self.inner {
            RuntimeInner::Dynamic(d) => (d.hipGraphExecDestroy)(graph_exec),
            RuntimeInner::Simulated(s) => {
                if graph_exec.is_null() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                let mut st = s.state.lock().unwrap();
                if st.graph_execs.remove(&(graph_exec as usize)).is_none() {
                    return sim_record_status(hipErrorInvalidValue);
                }
                hipSuccess
            }
        }
    }
}

// ============================================================================
// Standard Top-Level `extern "C"` HIP Runtime Functions
// ============================================================================

pub unsafe extern "C" fn hipInit(flags: c_uint) -> hipError_t {
    HipRuntime::global().hipInit(flags)
}

pub unsafe extern "C" fn hipGetDeviceCount(count: *mut c_int) -> hipError_t {
    HipRuntime::global().hipGetDeviceCount(count)
}

pub unsafe extern "C" fn hipSetDevice(deviceId: hipDevice_t) -> hipError_t {
    HipRuntime::global().hipSetDevice(deviceId)
}

pub unsafe extern "C" fn hipGetDevice(deviceId: *mut hipDevice_t) -> hipError_t {
    HipRuntime::global().hipGetDevice(deviceId)
}

pub unsafe extern "C" fn hipDeviceSynchronize() -> hipError_t {
    HipRuntime::global().hipDeviceSynchronize()
}

pub unsafe extern "C" fn hipGetDeviceProperties(
    prop: *mut hipDeviceProp_t,
    deviceId: hipDevice_t,
) -> hipError_t {
    HipRuntime::global().hipGetDeviceProperties(prop, deviceId)
}

pub unsafe extern "C" fn hipGetDevicePropertiesR0600(
    prop: *mut hipDeviceProp_tR0600,
    deviceId: hipDevice_t,
) -> hipError_t {
    HipRuntime::global().hipGetDeviceProperties(prop, deviceId)
}

pub unsafe extern "C" fn hipDeviceGetAttribute(
    pi: *mut c_int,
    attr: hipDeviceAttribute_t,
    deviceId: hipDevice_t,
) -> hipError_t {
    HipRuntime::global().hipDeviceGetAttribute(pi, attr, deviceId)
}

pub unsafe extern "C" fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t {
    HipRuntime::global().hipMemGetInfo(free, total)
}

pub unsafe extern "C" fn hipGetErrorString(hipError: hipError_t) -> *const c_char {
    HipRuntime::global().hipGetErrorString(hipError)
}

pub unsafe extern "C" fn hipGetLastError() -> hipError_t {
    HipRuntime::global().hipGetLastError()
}

pub unsafe extern "C" fn hipPeekAtLastError() -> hipError_t {
    HipRuntime::global().hipPeekAtLastError()
}

pub unsafe extern "C" fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t {
    HipRuntime::global().hipMalloc(ptr, size)
}

pub unsafe extern "C" fn hipFree(ptr: *mut c_void) -> hipError_t {
    HipRuntime::global().hipFree(ptr)
}

pub unsafe extern "C" fn hipMemset(dst: *mut c_void, value: c_int, sizeBytes: usize) -> hipError_t {
    HipRuntime::global().hipMemset(dst, value, sizeBytes)
}

pub unsafe extern "C" fn hipMemsetAsync(
    dst: *mut c_void,
    value: c_int,
    sizeBytes: usize,
    stream: hipStream_t,
) -> hipError_t {
    HipRuntime::global().hipMemsetAsync(dst, value, sizeBytes, stream)
}

pub unsafe extern "C" fn hipMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    sizeBytes: usize,
    kind: hipMemcpyKind,
) -> hipError_t {
    HipRuntime::global().hipMemcpy(dst, src, sizeBytes, kind)
}

pub unsafe extern "C" fn hipMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    sizeBytes: usize,
    kind: hipMemcpyKind,
    stream: hipStream_t,
) -> hipError_t {
    HipRuntime::global().hipMemcpyAsync(dst, src, sizeBytes, kind, stream)
}

pub unsafe extern "C" fn hipHostMalloc(
    ptr: *mut *mut c_void,
    size: usize,
    flags: c_uint,
) -> hipError_t {
    HipRuntime::global().hipHostMalloc(ptr, size, flags)
}

pub unsafe extern "C" fn hipHostFree(ptr: *mut c_void) -> hipError_t {
    HipRuntime::global().hipHostFree(ptr)
}

pub unsafe extern "C" fn hipFreeHost(ptr: *mut c_void) -> hipError_t {
    HipRuntime::global().hipHostFree(ptr)
}

pub unsafe extern "C" fn hipHostRegister(
    hostPtr: *mut c_void,
    sizeBytes: usize,
    flags: c_uint,
) -> hipError_t {
    HipRuntime::global().hipHostRegister(hostPtr, sizeBytes, flags)
}

pub unsafe extern "C" fn hipHostUnregister(hostPtr: *mut c_void) -> hipError_t {
    HipRuntime::global().hipHostUnregister(hostPtr)
}

pub unsafe extern "C" fn hipHostGetDevicePointer(
    devPtr: *mut *mut c_void,
    hstPtr: *mut c_void,
    flags: c_uint,
) -> hipError_t {
    HipRuntime::global().hipHostGetDevicePointer(devPtr, hstPtr, flags)
}

pub unsafe extern "C" fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t {
    HipRuntime::global().hipStreamCreate(stream)
}

pub unsafe extern "C" fn hipStreamCreateWithFlags(
    stream: *mut hipStream_t,
    flags: c_uint,
) -> hipError_t {
    HipRuntime::global().hipStreamCreateWithFlags(stream, flags)
}

pub unsafe extern "C" fn hipStreamDestroy(stream: hipStream_t) -> hipError_t {
    HipRuntime::global().hipStreamDestroy(stream)
}

pub unsafe extern "C" fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t {
    HipRuntime::global().hipStreamSynchronize(stream)
}

pub unsafe extern "C" fn hipStreamQuery(stream: hipStream_t) -> hipError_t {
    HipRuntime::global().hipStreamQuery(stream)
}

pub unsafe extern "C" fn hipStreamWaitEvent(
    stream: hipStream_t,
    event: hipEvent_t,
    flags: c_uint,
) -> hipError_t {
    HipRuntime::global().hipStreamWaitEvent(stream, event, flags)
}

pub unsafe extern "C" fn hipEventCreate(event: *mut hipEvent_t) -> hipError_t {
    HipRuntime::global().hipEventCreate(event)
}

pub unsafe extern "C" fn hipEventCreateWithFlags(
    event: *mut hipEvent_t,
    flags: c_uint,
) -> hipError_t {
    HipRuntime::global().hipEventCreateWithFlags(event, flags)
}

pub unsafe extern "C" fn hipEventDestroy(event: hipEvent_t) -> hipError_t {
    HipRuntime::global().hipEventDestroy(event)
}

pub unsafe extern "C" fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> hipError_t {
    HipRuntime::global().hipEventRecord(event, stream)
}

pub unsafe extern "C" fn hipEventSynchronize(event: hipEvent_t) -> hipError_t {
    HipRuntime::global().hipEventSynchronize(event)
}

pub unsafe extern "C" fn hipEventQuery(event: hipEvent_t) -> hipError_t {
    HipRuntime::global().hipEventQuery(event)
}

pub unsafe extern "C" fn hipEventElapsedTime(
    ms: *mut f32,
    start: hipEvent_t,
    stop: hipEvent_t,
) -> hipError_t {
    HipRuntime::global().hipEventElapsedTime(ms, start, stop)
}

pub unsafe extern "C" fn hipStreamBeginCapture(
    stream: hipStream_t,
    mode: hipStreamCaptureMode,
) -> hipError_t {
    HipRuntime::global().hipStreamBeginCapture(stream, mode)
}

pub unsafe extern "C" fn hipStreamEndCapture(
    stream: hipStream_t,
    pGraph: *mut hipGraph_t,
) -> hipError_t {
    HipRuntime::global().hipStreamEndCapture(stream, pGraph)
}

pub unsafe extern "C" fn hipStreamIsCapturing(
    stream: hipStream_t,
    pCaptureStatus: *mut hipStreamCaptureStatus,
) -> hipError_t {
    HipRuntime::global().hipStreamIsCapturing(stream, pCaptureStatus)
}

pub unsafe extern "C" fn hipGraphGetNodes(
    graph: hipGraph_t,
    nodes: *mut hipGraphNode_t,
    numNodes: *mut usize,
) -> hipError_t {
    HipRuntime::global().hipGraphGetNodes(graph, nodes, numNodes)
}

pub unsafe extern "C" fn hipGraphInstantiate(
    pGraphExec: *mut hipGraphExec_t,
    graph: hipGraph_t,
    pErrorNode: *mut hipGraphNode_t,
    pLogBuffer: *mut c_char,
    bufferSize: usize,
) -> hipError_t {
    HipRuntime::global().hipGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize)
}

pub unsafe extern "C" fn hipGraphInstantiateWithFlags(
    pGraphExec: *mut hipGraphExec_t,
    graph: hipGraph_t,
    flags: c_ulonglong,
) -> hipError_t {
    HipRuntime::global().hipGraphInstantiateWithFlags(pGraphExec, graph, flags)
}

pub unsafe extern "C" fn hipGraphLaunch(
    graphExec: hipGraphExec_t,
    stream: hipStream_t,
) -> hipError_t {
    HipRuntime::global().hipGraphLaunch(graphExec, stream)
}

pub unsafe extern "C" fn hipGraphDestroy(graph: hipGraph_t) -> hipError_t {
    HipRuntime::global().hipGraphDestroy(graph)
}

pub unsafe extern "C" fn hipGraphExecDestroy(graphExec: hipGraphExec_t) -> hipError_t {
    HipRuntime::global().hipGraphExecDestroy(graphExec)
}

/// Query the `gcnArchName` string for `device_id` via `hipGetDeviceProperties`.
pub fn device_gcn_arch_name(device_id: hipDevice_t) -> Result<String, hipError_t> {
    let mut prop = hipDeviceProp_t::default();
    let rc = unsafe { hipGetDeviceProperties(&mut prop, device_id) };
    if !rc.is_success() {
        return Err(rc);
    }
    Ok(prop.gcn_arch_name_str().to_string())
}

/// Convert a `hipError_t` to a human-readable `String` using `hipGetErrorString`.
pub fn error_string(err: hipError_t) -> String {
    unsafe {
        let ptr = hipGetErrorString(err);
        if ptr.is_null() {
            err.name_str().to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn crate_identity() {
        assert_eq!(CRATE_NAME, "hip-sys");
    }

    #[test]
    fn device_prop_r0600_abi_layout() {
        assert_eq!(size_of::<hipDeviceArch_t>(), 4);
        assert_eq!(size_of::<hipUUID>(), 16);
        assert_eq!(size_of::<hipDeviceProp_t>(), 1472);
        assert_eq!(offset_of!(hipDeviceProp_t, name), 0);
        assert_eq!(offset_of!(hipDeviceProp_t, uuid), 256);
        assert_eq!(offset_of!(hipDeviceProp_t, luid), 272);
        assert_eq!(offset_of!(hipDeviceProp_t, luidDeviceNodeMask), 280);
        assert_eq!(offset_of!(hipDeviceProp_t, totalGlobalMem), 288);
        assert_eq!(offset_of!(hipDeviceProp_t, sharedMemPerBlock), 296);
        assert_eq!(offset_of!(hipDeviceProp_t, warpSize), 308);
        assert_eq!(offset_of!(hipDeviceProp_t, major), 360);
        assert_eq!(offset_of!(hipDeviceProp_t, minor), 364);
        assert_eq!(offset_of!(hipDeviceProp_t, multiProcessorCount), 388);
        assert_eq!(offset_of!(hipDeviceProp_t, integrated), 396);
        assert_eq!(offset_of!(hipDeviceProp_t, canMapHostMemory), 400);
        assert_eq!(offset_of!(hipDeviceProp_t, l2CacheSize), 616);
        assert_eq!(offset_of!(hipDeviceProp_t, gcnArchName), 1160);
        assert_eq!(
            offset_of!(hipDeviceProp_t, maxSharedMemoryPerMultiProcessor),
            1416
        );
        assert_eq!(offset_of!(hipDeviceProp_t, asicRevision), 1468);
    }

    #[test]
    fn dynamic_rocm_sdk_symbol_resolution() {
        // If `libamdhip64.so` is present in the ROCm SDK or system, verify `dlopen` + `dlsym`
        // resolves every required HIP symbol and calls `hipGetErrorString`.
        if let Ok(rt) = HipRuntime::load_dynamic() {
            assert!(rt.is_dynamic());
            assert_eq!(rt.backend_mode(), HipBackendMode::Dynamic);
            unsafe {
                let s = CStr::from_ptr(rt.hipGetErrorString(hipSuccess)).to_string_lossy();
                assert!(s.contains("hipSuccess") || s.contains("no error"));
            }
        }
    }

    #[test]
    fn simulated_device_and_memory_lifecycle() {
        let rt = HipRuntime::simulated();
        assert!(rt.is_simulated());
        assert_eq!(rt.backend_mode(), HipBackendMode::Simulated);

        unsafe {
            assert_eq!(rt.hipInit(0), hipSuccess);
            assert_eq!(rt.hipInit(99), hipErrorInvalidValue);
            assert_eq!(rt.hipGetLastError(), hipErrorInvalidValue);
            assert_eq!(rt.hipPeekAtLastError(), hipSuccess);

            let mut count = 0;
            assert_eq!(rt.hipGetDeviceCount(&mut count), hipSuccess);
            assert!(count >= 2);

            assert_eq!(rt.hipSetDevice(0), hipSuccess);
            let mut active = -1;
            assert_eq!(rt.hipGetDevice(&mut active), hipSuccess);
            assert_eq!(active, 0);
            assert_eq!(rt.hipSetDevice(99), hipErrorInvalidDevice);

            let mut prop = hipDeviceProp_t::default();
            assert_eq!(rt.hipGetDeviceProperties(&mut prop, 0), hipSuccess);
            assert!(prop.gcn_arch_name_str().starts_with("gfx1151"));
            assert_eq!(prop.warpSize, 32);

            let mut warp = 0;
            assert_eq!(
                rt.hipDeviceGetAttribute(&mut warp, hipDeviceAttributeWarpSize, 0),
                hipSuccess
            );
            assert_eq!(warp, 32);

            let (mut free0, mut total0) = (0usize, 0usize);
            assert_eq!(rt.hipMemGetInfo(&mut free0, &mut total0), hipSuccess);
            assert_eq!(free0, total0);

            let mut dptr: *mut c_void = ptr::null_mut();
            assert_eq!(rt.hipMalloc(&mut dptr, 4096), hipSuccess);
            assert!(!dptr.is_null());

            let (mut free1, mut total1) = (0usize, 0usize);
            assert_eq!(rt.hipMemGetInfo(&mut free1, &mut total1), hipSuccess);
            assert_eq!(total1, total0);
            assert_eq!(free0 - free1, 4096);

            assert_eq!(rt.hipMemset(dptr, 0xAB, 4096), hipSuccess);
            let mut host_buf = vec![0u8; 4096];
            assert_eq!(
                rt.hipMemcpy(
                    host_buf.as_mut_ptr().cast(),
                    dptr,
                    4096,
                    hipMemcpyDeviceToHost
                ),
                hipSuccess
            );
            assert!(host_buf.iter().all(|&b| b == 0xAB));

            assert_eq!(rt.hipFree(dptr), hipSuccess);
            assert_eq!(rt.hipFree(dptr), hipErrorInvalidDevicePointer);

            // Host mapped & registered memory
            let mut hptr: *mut c_void = ptr::null_mut();
            assert_eq!(
                rt.hipHostMalloc(&mut hptr, 1024, hipHostMallocMapped),
                hipSuccess
            );
            let mut mapped_dptr: *mut c_void = ptr::null_mut();
            assert_eq!(
                rt.hipHostGetDevicePointer(&mut mapped_dptr, hptr, 0),
                hipSuccess
            );
            assert_eq!(mapped_dptr, hptr);
            assert_eq!(rt.hipHostFree(hptr), hipSuccess);

            let mut reg_backing = [42u64; 16];
            let reg_ptr = reg_backing.as_mut_ptr().cast::<c_void>();
            assert_eq!(
                rt.hipHostRegister(reg_ptr, 128, hipHostRegisterMapped),
                hipSuccess
            );
            assert_eq!(
                rt.hipHostRegister(reg_ptr, 128, hipHostRegisterMapped),
                hipErrorHostMemoryAlreadyRegistered
            );
            assert_eq!(
                rt.hipHostGetDevicePointer(&mut mapped_dptr, reg_ptr, 0),
                hipSuccess
            );
            assert_eq!(mapped_dptr, reg_ptr);
            assert_eq!(rt.hipHostUnregister(reg_ptr), hipSuccess);
            assert_eq!(
                rt.hipHostUnregister(reg_ptr),
                hipErrorHostMemoryNotRegistered
            );
        }
    }

    #[test]
    fn simulated_streams_events_and_graphs() {
        let rt = HipRuntime::simulated();
        unsafe {
            assert_eq!(rt.hipSetDevice(0), hipSuccess);

            let mut stream: hipStream_t = ptr::null_mut();
            assert_eq!(
                rt.hipStreamCreateWithFlags(&mut stream, hipStreamNonBlocking),
                hipSuccess
            );
            assert!(!stream.is_null());

            let (mut ev0, mut ev1, mut ev_no_time) =
                (ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
            assert_eq!(rt.hipEventCreate(&mut ev0), hipSuccess);
            assert_eq!(rt.hipEventCreate(&mut ev1), hipSuccess);
            assert_eq!(
                rt.hipEventCreateWithFlags(&mut ev_no_time, hipEventDisableTiming),
                hipSuccess
            );

            assert_eq!(rt.hipEventRecord(ev0, stream), hipSuccess);
            assert_eq!(rt.hipEventRecord(ev1, stream), hipSuccess);
            assert_eq!(rt.hipEventRecord(ev_no_time, stream), hipSuccess);
            assert_eq!(rt.hipEventSynchronize(ev1), hipSuccess);

            let mut ms = -1.0f32;
            assert_eq!(rt.hipEventElapsedTime(&mut ms, ev0, ev1), hipSuccess);
            assert!(ms >= 0.0);
            assert_eq!(
                rt.hipEventElapsedTime(&mut ms, ev0, ev_no_time),
                hipErrorInvalidResourceHandle
            );

            // Graph capture + replay of async memset and memcpy
            let (mut a, mut b) = (ptr::null_mut(), ptr::null_mut());
            assert_eq!(rt.hipMalloc(&mut a, 16), hipSuccess);
            assert_eq!(rt.hipMalloc(&mut b, 16), hipSuccess);

            assert_eq!(
                rt.hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
                hipSuccess
            );
            let mut status = hipStreamCaptureStatusNone;
            assert_eq!(rt.hipStreamIsCapturing(stream, &mut status), hipSuccess);
            assert_eq!(status, hipStreamCaptureStatusActive);

            assert_eq!(
                rt.hipMemcpyAsync(b, a, 16, hipMemcpyDeviceToDevice, stream),
                hipSuccess
            );
            let mut graph: hipGraph_t = ptr::null_mut();
            assert_eq!(rt.hipStreamEndCapture(stream, &mut graph), hipSuccess);
            assert!(!graph.is_null());

            let mut num_nodes = 0usize;
            assert_eq!(
                rt.hipGraphGetNodes(graph, ptr::null_mut(), &mut num_nodes),
                hipSuccess
            );
            assert_eq!(num_nodes, 1);

            let mut exec: hipGraphExec_t = ptr::null_mut();
            assert_eq!(
                rt.hipGraphInstantiate(&mut exec, graph, ptr::null_mut(), ptr::null_mut(), 0),
                hipSuccess
            );
            assert!(!exec.is_null());

            // Populate `a` with 0x5A after capture and verify graph launch copies `a -> b`.
            assert_eq!(rt.hipMemset(a, 0x5A, 16), hipSuccess);
            assert_eq!(rt.hipMemset(b, 0x00, 16), hipSuccess);
            assert_eq!(rt.hipGraphLaunch(exec, stream), hipSuccess);
            assert_eq!(rt.hipStreamSynchronize(stream), hipSuccess);

            let mut out = [0u8; 16];
            assert_eq!(
                rt.hipMemcpy(out.as_mut_ptr().cast(), b, 16, hipMemcpyDeviceToHost),
                hipSuccess
            );
            assert_eq!(out, [0x5Au8; 16]);

            assert_eq!(rt.hipGraphExecDestroy(exec), hipSuccess);
            assert_eq!(rt.hipGraphDestroy(graph), hipSuccess);
            assert_eq!(rt.hipFree(a), hipSuccess);
            assert_eq!(rt.hipFree(b), hipSuccess);
            assert_eq!(rt.hipEventDestroy(ev0), hipSuccess);
            assert_eq!(rt.hipEventDestroy(ev1), hipSuccess);
            assert_eq!(rt.hipEventDestroy(ev_no_time), hipSuccess);
            assert_eq!(rt.hipStreamDestroy(stream), hipSuccess);
        }
    }
}
