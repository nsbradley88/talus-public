//! WP-1.3 foundation verification & Strix Halo (`gfx1151`) real GPU FFI + HIP Graph smoke test.
//!
//! - Non-ignored tests run on CPU in `rust/ci.sh` to stress-test cross-crate invariants
//!   across `hip-sys`, `talus-hip`, `talus-gguf`, `talus-mem`, `talus-kernels-sys`, and
//!   `talus-kernels`.
//! - Ignored test `strix_ffi_hip_graph_capture_and_replay_smoke` runs on a `gfx1151` GPU
//!   against the real `libamdhip64.so` and `libtalus_kernels.so`.

use std::sync::Arc;
use talus_core_api::{
    DeviceId, GpuArch, MemoryModel, PlacementKind, QuantType, TensorClass, WmmaLayout,
};
use talus_hip::{CaptureMode, Device, DeviceBuf, Event, HostMapped, Stream};
use talus_kernels::ffi::{FfiConfig, FfiKernelBackend};
use talus_kernels::{CpuMockBackend, KernelBackend, KernelOp, KernelSet};
use talus_kernels_sys::LibTalusKernels;
use talus_mem::{
    Budget, BudgetCategory, HostMemProbe, LinuxProcMemProbe, MemError, MockMemProbe,
    StrixUnifiedMemory, DEFAULT_FLOOR_BYTES, GIB,
};

#[test]
fn cross_crate_foundation_invariants_and_adversarial_edge_cases() {
    // 1. Verify Device / Stream / Event / CaptureGuard drop-abort & multi-replay on current runtime
    let dev_id = DeviceId(0);
    let dev = Device::new(dev_id).expect("Device::new(0)");
    assert_eq!(dev.arch(), GpuArch::Gfx1151);
    assert_eq!(dev.arch_traits().wave, 32);
    assert_eq!(dev.arch_traits().wmma, WmmaLayout::Gfx11);

    let stream = Stream::non_blocking(dev_id).expect("Stream::non_blocking");
    let stream_handle = stream.as_core_handle();
    assert_eq!(stream_handle.device, dev_id);
    assert_eq!(stream_handle.raw, stream.raw() as usize as u64);

    // Nested begin_capture while already capturing must fail cleanly and not corrupt stream state
    {
        let _guard = stream
            .begin_capture(CaptureMode::ThreadLocal)
            .expect("outer begin_capture");
        assert!(stream.is_capturing());
        let nested = stream.begin_capture(CaptureMode::ThreadLocal);
        assert!(
            nested.is_err(),
            "nested begin_capture on the same stream must fail"
        );
        // Drop `_guard` without `.finish()` -> must abort capture cleanly
    }
    assert!(
        !stream.is_capturing(),
        "CaptureGuard drop must leave stream out of capture mode"
    );

    // Stream remains usable for a subsequent valid graph capture + multi-replay
    let src = DeviceBuf::<f32>::from_slice(dev_id, &[1.5, -2.5, 3.25, 4.0]).expect("src");
    let dst = DeviceBuf::<f32>::zeros(dev_id, 4).expect("dst");
    let guard = stream
        .begin_capture(CaptureMode::ThreadLocal)
        .expect("begin_capture after abort");
    dst.copy_from_device_async(&src, guard.stream())
        .expect("copy_from_device_async in capture");
    let graph = guard.finish().expect("finish capture");
    assert_eq!(graph.node_count().expect("node_count"), 1);
    let exec = graph.instantiate().expect("instantiate");
    exec.launch(&stream).expect("launch 1");
    stream.synchronize().expect("sync 1");
    assert_eq!(dst.to_vec().expect("to_vec 1"), vec![1.5, -2.5, 3.25, 4.0]);

    // Update src in-place and replay without re-instantiating
    src.copy_from_host(&[10.0, 20.0, 30.0, 40.0])
        .expect("update src");
    dst.zero().expect("zero dst");
    exec.launch(&stream).expect("launch 2");
    stream.synchronize().expect("sync 2");
    assert_eq!(
        dst.to_vec().expect("to_vec 2"),
        vec![10.0, 20.0, 30.0, 40.0]
    );

    // 2. Verify talus-mem Budget floor enforcement and Unified placement invariants
    let unified = StrixUnifiedMemory::new(dev_id);
    let emb_place = unified.place(TensorClass::Embedding, 4096);
    let dense_place = unified.place(TensorClass::DenseWeight, 4096);
    assert!(unified.host_visible(&dense_place));
    assert!(StrixUnifiedMemory::is_zero_copy(&emb_place));
    assert_eq!(emb_place.kind, PlacementKind::MmapFile);
    assert_eq!(
        unified.place(TensorClass::HostMappedControl, 64).kind,
        PlacementKind::HostMapped
    );
    assert_eq!(dense_place.kind, PlacementKind::DeviceLocal);

    // Exact boundary at DEFAULT_FLOOR_BYTES (8 GiB)
    let avail = 12 * GIB;
    let exact_headroom = avail - DEFAULT_FLOOR_BYTES;
    let budget = Budget::new(MockMemProbe::new(avail));
    assert_eq!(budget.floor_bytes(), DEFAULT_FLOOR_BYTES);

    // Allocating exactly max headroom succeeds; 1 byte more fails with FloorViolation
    let guard_res = budget
        .try_reserve_guard(BudgetCategory::Scratch, "arena_exact", exact_headroom)
        .expect("exact headroom reservation");
    let over = budget.try_reserve(BudgetCategory::Weights, "one_byte_over", 1);
    assert!(matches!(over, Err(MemError::FloorViolation { .. })));
    drop(guard_res);
    assert_eq!(budget.total_reserved_bytes(), 0);

    // 3. Verify KernelSet + CpuMockBackend adversarial validation
    let kid = KernelSet::select(&dev.arch(), KernelOp::Mmvq, QuantType::Q4_K, 1)
        .expect("KernelSet::select");
    assert_eq!(kid.arch, GpuArch::Gfx1151);
    assert_eq!(kid.op, KernelOp::Mmvq);

    let mock = CpuMockBackend::new();
    // Non-multiple-of-32 must be rejected by quantize_q8_1 and mmvq
    let mut q8_buf = vec![0u8; 72];
    assert!(mock
        .quantize_q8_1(&[1.0f32; 30], &mut q8_buf, 30, 1)
        .is_err());
    // Buffer underflow checks
    let mut short_x = [1.0f32; 3];
    assert!(mock.rms_norm(&mut short_x, None, 2, 4, 1e-5).is_err());
    assert!(mock
        .rms_norm(&mut [1.0f32; 8], Some(&[1.0f32; 2]), 2, 4, 1e-5)
        .is_err());
    assert!(mock.silu_mul(&mut [1.0f32; 2], &[1.0f32; 4], 4).is_err());
    assert!(mock.axpy(&mut [1.0f32; 2], &[1.0f32; 4], 1.0, 4).is_err());
}

/// Real GPU smoke test on Strix Halo (`gfx1151`).
///
/// Allocates device memory via `talus-hip`, loads `libtalus_kernels.so` via `talus-kernels`,
/// captures a HIP graph with real GPU kernels (`tk_rms_norm_to`, `tk_silu_mul`, `tk_axpy`)
/// launched across the C FFI boundary onto `stream.raw()`, instantiates `GraphExec`, zeroes
/// output memory, replays the captured graph twice (including with updated device inputs at
/// fixed device addresses), and verifies the GPU output matches `CpuMockBackend` reference
/// `f32` values.
#[test]
#[ignore = "requires gfx1151 GPU + libamdhip64.so + libtalus_kernels.so"]
fn strix_ffi_hip_graph_capture_and_replay_smoke() {
    // 1. Verify we are running against the real dynamic ROCm HIP runtime (`libamdhip64.so`), not simulated!
    let hip_rt = hip_sys::HipRuntime::global();
    assert!(
        hip_rt.is_dynamic(),
        "expected dynamic libamdhip64.so runtime on gfx1151, got {:?}",
        hip_rt.backend_mode()
    );

    let dev_id = DeviceId(0);
    let device = Device::new(dev_id).expect("open HIP DeviceId(0) on gfx1151");
    assert_eq!(
        device.arch(),
        GpuArch::Gfx1151,
        "gfx1151 DeviceId(0) must be Gfx1151, got {:?} ({})",
        device.arch(),
        device.gcn_arch_name()
    );
    assert_eq!(device.warp_size(), 32);
    assert!(device.gcn_arch_name().starts_with("gfx1151"));

    let (free_bytes, total_bytes) = device.mem_info().expect("hipMemGetInfo on gfx1151");
    println!(
        "[gfx1151-smoke] HIP device 0: name='{}' arch='{}' cu={} free={:.2} GiB / total={:.2} GiB",
        device.name(),
        device.gcn_arch_name(),
        device.multi_processor_count(),
        free_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    // Also verify talus-mem live /proc/meminfo budget check enforces >= 8 GiB floor
    let proc_probe = LinuxProcMemProbe::new();
    let strix_mem_avail = proc_probe
        .mem_available_bytes()
        .expect("read /proc/meminfo");
    assert!(
        strix_mem_avail >= DEFAULT_FLOOR_BYTES,
        "Strix Halo MemAvailable ({strix_mem_avail}) must remain >= 8 GiB floor"
    );
    let strix_budget = Budget::strix_default();
    assert_eq!(strix_budget.floor_bytes(), DEFAULT_FLOOR_BYTES);
    let _smoke_res = strix_budget
        .try_reserve_guard(BudgetCategory::Scratch, "strix_graph_smoke_bufs", 64 * 1024)
        .expect("reserve 64 KiB in Strix Halo Budget");

    // 2. Create non-blocking HIP stream and timing events
    let stream = Stream::non_blocking(dev_id).expect("create non-blocking HIP stream");
    let ev_start = Event::new(dev_id).expect("create start event");
    let ev_stop = Event::new(dev_id).expect("create stop event");

    // 3. Load `libtalus_kernels.so` via LibTalusKernels & create FfiKernelBackend on device 0
    let lib = Arc::new(
        LibTalusKernels::load_default()
            .expect("load libtalus_kernels.so via TALUS_KERNELS_LIB / LD_LIBRARY_PATH"),
    );
    let mut cfg =
        FfiConfig::for_arch(Arc::clone(&lib), &GpuArch::Gfx1151).expect("FfiConfig::for_arch");
    cfg.load_from_env().expect("FfiConfig::load_from_env");

    let backend = FfiKernelBackend::new(
        Arc::clone(&lib),
        dev_id.0 as i32,
        Some(&cfg),
        stream.as_core_handle(),
    )
    .expect("FfiKernelBackend::new on device 0");
    assert_eq!(backend.device_id(), 0);
    assert_eq!(backend.arch(), GpuArch::Gfx1151);

    // 4. Allocate DeviceBuf<f32> buffers on device 0
    let rows = 4usize;
    let cols = 128usize;
    let n = rows * cols;
    let eps = 1e-5f32;
    let alpha = 0.375f32;

    let x_host_1: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.03125 - 7.5).sin() * 2.25)
        .collect();
    let w_host: Vec<f32> = (0..cols)
        .map(|c| 0.5 + ((c as f32) / (cols as f32)) * 1.25)
        .collect();
    let up_host: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.07 - 3.0).cos() * 1.5)
        .collect();
    let residual_host: Vec<f32> = (0..n).map(|i| ((i as i32 - 256) as f32) * 0.01).collect();

    let x_dev = DeviceBuf::<f32>::from_slice(dev_id, &x_host_1).expect("alloc x_dev");
    let w_dev = DeviceBuf::<f32>::from_slice(dev_id, &w_host).expect("alloc w_dev");
    let up_dev = DeviceBuf::<f32>::from_slice(dev_id, &up_host).expect("alloc up_dev");
    let residual_dev =
        DeviceBuf::<f32>::from_slice(dev_id, &residual_host).expect("alloc residual_dev");
    let out_dev = DeviceBuf::<f32>::zeros(dev_id, n).expect("alloc out_dev");
    stream
        .synchronize()
        .expect("sync initial host->device setup");

    // 5. Begin HIP stream graph capture and launch real kernels from libtalus_kernels.so across FFI
    let guard = stream
        .begin_capture(CaptureMode::ThreadLocal)
        .expect("stream.begin_capture(ThreadLocal)");
    assert!(stream.is_capturing());

    unsafe {
        // Kernel 1: out_dev = rms_norm(x_dev, w_dev, rows=4, cols=128, eps=1e-5)
        backend
            .rms_norm_to_dev(
                out_dev.as_mut_ptr(),
                x_dev.as_ptr(),
                w_dev.as_ptr(),
                rows,
                cols,
                eps,
            )
            .expect("capture tk_rms_norm_to");

        // Kernel 2: out_dev = silu(out_dev) * up_dev
        backend
            .silu_mul_dev(out_dev.as_mut_ptr(), up_dev.as_ptr(), n)
            .expect("capture tk_silu_mul");

        // Kernel 3: out_dev += alpha * residual_dev
        backend
            .axpy_dev(out_dev.as_mut_ptr(), residual_dev.as_ptr(), alpha, n)
            .expect("capture tk_axpy");
    }

    let graph = guard
        .finish()
        .expect("finish HIP stream capture into Graph");
    assert!(!stream.is_capturing());
    let node_count = graph.node_count().expect("query captured graph node_count");
    println!("[gfx1151-smoke] Captured HIP graph with {node_count} kernel node(s)");
    assert!(
        node_count >= 3,
        "expected >= 3 kernel nodes in captured HIP graph, got {node_count}"
    );

    let graph_exec = graph
        .instantiate()
        .expect("hipGraphInstantiate captured kernel graph");

    // 6. Replay #1: zero output buffer, replay graph on stream, synchronize, copy back, compare with CPU reference
    out_dev.zero().expect("zero out_dev before replay #1");
    let pre_replay = out_dev.to_vec().expect("verify zeroed out_dev");
    assert!(pre_replay.iter().all(|&v| v == 0.0));

    ev_start.record(&stream).expect("record ev_start");
    graph_exec
        .launch(&stream)
        .expect("graph_exec.launch replay #1");
    ev_stop.record(&stream).expect("record ev_stop");
    stream.synchronize().expect("stream.synchronize replay #1");
    let replay1_ms = ev_start.elapsed_ms(&ev_stop).expect("elapsed_ms replay #1");

    let gpu_out_1 = out_dev
        .to_vec()
        .expect("copy out_dev to host after replay #1");

    let cpu_ref = CpuMockBackend::new();
    let mut expected_1 = x_host_1.clone();
    cpu_ref
        .rms_norm(&mut expected_1, Some(&w_host), rows, cols, eps)
        .expect("cpu rms_norm #1");
    cpu_ref
        .silu_mul(&mut expected_1, &up_host, n)
        .expect("cpu silu_mul #1");
    cpu_ref
        .axpy(&mut expected_1, &residual_host, alpha, n)
        .expect("cpu axpy #1");

    let mut max_diff_1 = 0.0f32;
    for (i, (&got, &exp)) in gpu_out_1.iter().zip(expected_1.iter()).enumerate() {
        let diff = (got - exp).abs();
        if diff > max_diff_1 {
            max_diff_1 = diff;
        }
        assert!(
            diff < 1e-4,
            "replay #1 mismatch at index {i}: gpu={got} vs expected={exp} (diff={diff})"
        );
    }
    println!(
        "[gfx1151-smoke] Replay #1 PASS: n={n} max_abs_diff={max_diff_1:.3e} elapsed={replay1_ms:.4} ms sample=[{:.5}, {:.5}, {:.5}, {:.5}]",
        gpu_out_1[0], gpu_out_1[1], gpu_out_1[2], gpu_out_1[3]
    );

    // 7. Replay #2: mutate x_dev in-place at the same device address, zero out_dev, and replay
    //    the already-instantiated GraphExec without re-capturing!
    let x_host_2: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.11 + 1.25).cos() * 3.1)
        .collect();
    x_dev
        .copy_from_host(&x_host_2)
        .expect("update x_dev for replay #2");
    out_dev.zero().expect("zero out_dev before replay #2");

    graph_exec
        .launch(&stream)
        .expect("graph_exec.launch replay #2");
    stream.synchronize().expect("stream.synchronize replay #2");

    let gpu_out_2 = out_dev
        .to_vec()
        .expect("copy out_dev to host after replay #2");
    assert_ne!(
        gpu_out_1, gpu_out_2,
        "replay #2 with new input must produce distinct output"
    );

    let mut expected_2 = x_host_2.clone();
    cpu_ref
        .rms_norm(&mut expected_2, Some(&w_host), rows, cols, eps)
        .expect("cpu rms_norm #2");
    cpu_ref
        .silu_mul(&mut expected_2, &up_host, n)
        .expect("cpu silu_mul #2");
    cpu_ref
        .axpy(&mut expected_2, &residual_host, alpha, n)
        .expect("cpu axpy #2");

    let mut max_diff_2 = 0.0f32;
    for (i, (&got, &exp)) in gpu_out_2.iter().zip(expected_2.iter()).enumerate() {
        let diff = (got - exp).abs();
        if diff > max_diff_2 {
            max_diff_2 = diff;
        }
        assert!(
            diff < 1e-4,
            "replay #2 mismatch at index {i}: gpu={got} vs expected={exp} (diff={diff})"
        );
    }
    println!(
        "[gfx1151-smoke] Replay #2 PASS: n={n} max_abs_diff={max_diff_2:.3e} sample=[{:.5}, {:.5}, {:.5}, {:.5}]",
        gpu_out_2[0], gpu_out_2[1], gpu_out_2[2], gpu_out_2[3]
    );

    // 8. Also verify zero-copy HostMapped<f32> through the safe KernelBackend trait on unified memory
    let mut mapped_x = HostMapped::from_slice(dev_id, &x_host_1[0..cols]).expect("HostMapped x");
    let mapped_w = HostMapped::from_slice(dev_id, &w_host).expect("HostMapped w");
    backend
        .rms_norm(&mut mapped_x, Some(&mapped_w), 1, cols, eps)
        .expect("FfiKernelBackend::rms_norm on HostMapped unified memory");
    stream
        .synchronize()
        .expect("sync HostMapped rms_norm kernel");

    let mut expected_row0 = x_host_1[0..cols].to_vec();
    cpu_ref
        .rms_norm(&mut expected_row0, Some(&w_host), 1, cols, eps)
        .expect("cpu rms_norm row0");
    for (i, (&got, &exp)) in mapped_x.iter().zip(expected_row0.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(
            diff < 1e-4,
            "HostMapped unified memory rms_norm mismatch at {i}: gpu={got} vs expected={exp}"
        );
    }
    println!("[gfx1151-smoke] HostMapped unified-memory KernelBackend::rms_norm PASS");
}
