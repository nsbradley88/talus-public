//! Unified memory placement (`StrixUnifiedMemory`), host memory probing (`HostMemProbe`),
//! single `Budget` accountant (`MemAvailable >= 8 GiB` / `11 GiB` floor enforcement),
//! and 256-byte aligned offset/region planners (`ScratchArenaPlan`, `SlabPlanner`).

pub mod arena;
pub mod budget;
pub mod error;
pub mod placement;
pub mod probe;

pub use arena::{
    align_up, ArenaMark, KvLayerRegions, KvPoolPlan, PlannedRegion, RecurrentSnapshotPoolPlan,
    RegionId, ScratchArenaPlan, SlabGroup, SlabPlanner, TypedRegion, DEFAULT_ARENA_ALIGN,
};
pub use budget::{
    Budget, BudgetCategory, BudgetConfig, BudgetSummary, Reservation, ReservationGuard,
    ReservationId, DEFAULT_FLOOR_BYTES, GIB, KIB, MIB, MIMO_FLOOR_BYTES,
};
pub use error::{MemError, Result};
pub use placement::{
    DeviceId, MemoryModel, Placement, PlacementKind, StrixUnifiedMemory, TensorClass,
    UnifiedMemoryModel, STRIX_UNIFIED_TOTAL_BYTES,
};
pub use probe::{
    conversation_mem_available, conversation_memory_admit, meminfo_kb_bytes, parse_meminfo,
    sample_host_memory, HostMemProbe, HostMemoryReading, LinuxProcMemProbe, MeminfoStats,
    MemoryProbePaths, MemorySource, MockMemProbe,
};

/// Crate identifier.
pub const CRATE_NAME: &str = "talus-mem";

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use talus_core_api::{CoreError, KvCacheFormat, KvLayout, RecurrentLayout};

    static FIXTURE_SEQ: AtomicUsize = AtomicUsize::new(0);

    /// Scratch fixture directory placed under the workspace's `target/test-scratch` (never `/tmp`).
    struct Fixture {
        dir: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let seq = FIXTURE_SEQ.fetch_add(1, Ordering::Relaxed);
            let dir = PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../target/test-scratch"
            ))
            .join("talus-mem-fixtures")
            .join(format!("fixture_{}_{}", std::process::id(), seq));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(dir.join("cgroup")).expect("create fixture cgroup dir");
            Self { dir }
        }

        fn put_str(&self, rel: &str, text: &str) {
            let p = self.dir.join(rel);
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent).expect("create parent dir");
            }
            fs::write(p, text.as_bytes()).expect("write fixture file");
        }

        fn put_u64(&self, rel: &str, n: u64) {
            self.put_str(rel, &format!("{n}\n"));
        }

        fn meminfo(&self, total: usize, available: usize) {
            self.put_str(
                "meminfo",
                &format!(
                    "MemTotal:       {} kB\nMemFree: 1024 kB\nMemAvailable:   {} kB\n",
                    total / 1024,
                    available / 1024
                ),
            );
        }

        fn self_cgroup(&self, text: &str) {
            self.put_str("self_cgroup", &format!("{text}\n"));
        }

        fn paths(&self) -> MemoryProbePaths {
            MemoryProbePaths::from_fixture_dir(&self.dir)
        }

        fn probe(&self) -> LinuxProcMemProbe {
            LinuxProcMemProbe::from_fixture_dir(&self.dir)
        }

        fn sample(&self, explicit_limit_bytes: usize) -> Option<HostMemoryReading> {
            sample_host_memory(&self.paths(), explicit_limit_bytes)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn crate_identity() {
        assert_eq!(CRATE_NAME, "talus-mem");
        assert_eq!(DEFAULT_FLOOR_BYTES, 8 * GIB);
        assert_eq!(MIMO_FLOOR_BYTES, 11 * GIB);
        assert_eq!(STRIX_UNIFIED_TOTAL_BYTES, 128 * GIB);
        assert_eq!(DEFAULT_ARENA_ALIGN, 256);
    }

    // ========================================================================
    // 1. StrixUnifiedMemory placement tests for every TensorClass
    // ========================================================================

    #[test]
    fn strix_unified_memory_places_every_tensor_class() {
        let dev = DeviceId(0);
        let strix_mmap = StrixUnifiedMemory::new(dev);
        assert_eq!(strix_mmap.device(), dev);
        assert_eq!(strix_mmap.embedding_kind(), PlacementKind::MmapFile);
        assert_eq!(strix_mmap.total_unified_bytes(), 128 * GIB);

        let device_local_classes = [
            TensorClass::DenseWeight,
            TensorClass::RoutedExpert,
            TensorClass::SharedExpert,
            TensorClass::OutputHead,
            TensorClass::RecurrentState,
            TensorClass::KvCache,
            TensorClass::ActivationScratch,
        ];

        for class in device_local_classes {
            let p = strix_mmap.place(class, 4096);
            assert_eq!(
                p,
                Placement {
                    device: dev,
                    kind: PlacementKind::DeviceLocal,
                },
                "expected DeviceLocal for {class:?}"
            );
            assert!(strix_mmap.host_visible(&p));
            assert!(!StrixUnifiedMemory::is_zero_copy(&p));
        }

        // Embedding and PLE with default MmapFile policy.
        let emb_p = strix_mmap.place(TensorClass::Embedding, 64 * MIB);
        assert_eq!(
            emb_p,
            Placement {
                device: dev,
                kind: PlacementKind::MmapFile,
            }
        );
        assert!(strix_mmap.host_visible(&emb_p));
        assert!(StrixUnifiedMemory::is_zero_copy(&emb_p));
        assert_eq!(strix_mmap.place_ple(128 * MIB), emb_p);
        assert_eq!(strix_mmap.place_embedding(64 * MIB), emb_p);

        // HostMappedControl always uses HostMapped.
        let ctl_p = strix_mmap.place(TensorClass::HostMappedControl, 64);
        assert_eq!(
            ctl_p,
            Placement {
                device: dev,
                kind: PlacementKind::HostMapped,
            }
        );
        assert!(strix_mmap.host_visible(&ctl_p));
        assert!(StrixUnifiedMemory::is_zero_copy(&ctl_p));

        // HostMapped embedding / PLE mode on a secondary device ID.
        let dev1 = DeviceId(1);
        let strix_host = StrixUnifiedMemory::with_host_mapped_embeddings(dev1);
        assert_eq!(strix_host.device(), dev1);
        assert_eq!(strix_host.embedding_kind(), PlacementKind::HostMapped);
        let emb_host = strix_host.place(TensorClass::Embedding, 64 * MIB);
        assert_eq!(
            emb_host,
            Placement {
                device: dev1,
                kind: PlacementKind::HostMapped,
            }
        );
        assert_eq!(strix_host.place_ple(128 * MIB), emb_host);
        assert!(StrixUnifiedMemory::is_zero_copy(&emb_host));

        for class in device_local_classes {
            let p = strix_host.place(class, 1024);
            assert_eq!(
                p,
                Placement {
                    device: dev1,
                    kind: PlacementKind::DeviceLocal,
                }
            );
        }

        // Reject DeviceLocal as zero-copy embedding mode.
        assert!(StrixUnifiedMemory::try_with_embedding_kind(dev, PlacementKind::MmapFile).is_ok());
        assert!(
            StrixUnifiedMemory::try_with_embedding_kind(dev, PlacementKind::HostMapped).is_ok()
        );
        assert!(
            StrixUnifiedMemory::try_with_embedding_kind(dev, PlacementKind::DeviceLocal).is_err()
        );
    }

    #[test]
    fn tensor_class_to_budget_category_mapping() {
        assert_eq!(
            BudgetCategory::from_tensor_class(TensorClass::DenseWeight),
            BudgetCategory::Weights
        );
        assert_eq!(
            BudgetCategory::from_tensor_class(TensorClass::RoutedExpert),
            BudgetCategory::Weights
        );
        assert_eq!(
            BudgetCategory::from_tensor_class(TensorClass::SharedExpert),
            BudgetCategory::Weights
        );
        assert_eq!(
            BudgetCategory::from_tensor_class(TensorClass::Embedding),
            BudgetCategory::Weights
        );
        assert_eq!(
            BudgetCategory::from_tensor_class(TensorClass::OutputHead),
            BudgetCategory::Weights
        );
        assert_eq!(
            BudgetCategory::from_tensor_class(TensorClass::KvCache),
            BudgetCategory::KvCache
        );
        assert_eq!(
            BudgetCategory::from_tensor_class(TensorClass::RecurrentState),
            BudgetCategory::RecurrentState
        );
        assert_eq!(
            BudgetCategory::from_tensor_class(TensorClass::ActivationScratch),
            BudgetCategory::Scratch
        );
        assert_eq!(
            BudgetCategory::from_tensor_class(TensorClass::HostMappedControl),
            BudgetCategory::Scratch
        );
    }

    // ========================================================================
    // 2. /proc/meminfo & cgroup v1/v2 fixture parsing (port of memory_guard_test.cpp)
    // ========================================================================

    #[test]
    fn real_proc_meminfo_on_host() {
        let probe = LinuxProcMemProbe::new();
        let reading = probe
            .sample()
            .expect("real /proc/meminfo on Linux host must parse");
        assert!(reading.mem_available > 0);
        assert!(reading.available > 0);
        assert!(reading.mem_total.expect("MemTotal present") >= reading.mem_available);
        assert!(reading.mem_free.expect("MemFree present") > 0);
        assert!(probe.mem_available_bytes().unwrap() > 0);

        // With a 1-byte explicit operator cap, available memory saturates to 0 with source Flag.
        let capped = LinuxProcMemProbe::new().with_explicit_limit_bytes(1);
        let r_capped = capped.sample().expect("sample with 1-byte cap");
        assert_eq!(r_capped.available, 0);
        assert_eq!(r_capped.source, MemorySource::Flag);
    }

    #[test]
    fn cgroup_v2_max_everywhere_is_plain_memavailable() {
        let f = Fixture::new();
        f.meminfo(120 * GIB, 25 * GIB);
        f.self_cgroup("0::/a/b");
        f.put_str("cgroup/memory.max", "max\n");
        f.put_u64("cgroup/memory.current", (90 * GIB) as u64);
        f.put_str("cgroup/a/memory.max", "max\n");
        f.put_u64("cgroup/a/memory.current", (60 * GIB) as u64);
        f.put_str("cgroup/a/b/memory.max", "max\n");
        f.put_u64("cgroup/a/b/memory.current", (50 * GIB) as u64);

        let r = f.sample(0).expect("sample should succeed");
        assert_eq!(r.available, 25 * GIB);
        assert_eq!(r.source, MemorySource::Meminfo);
        assert_eq!(r.limit, 120 * GIB);
        assert_eq!(r.current, 95 * GIB);
        assert_eq!(r.mem_available, 25 * GIB);
        assert_eq!(r.mem_total, Some(120 * GIB));
        assert_eq!(r.mem_free, Some(1024 * KIB));
    }

    #[test]
    fn cgroup_v2_nested_limits_and_memory_high() {
        let f = Fixture::new();
        f.meminfo(120 * GIB, 80 * GIB);
        f.self_cgroup("0::/a/b");
        f.put_u64("cgroup/memory.max", (100 * GIB) as u64);
        f.put_u64("cgroup/memory.current", (90 * GIB) as u64); // 10 GiB room
        f.put_u64("cgroup/a/memory.max", (60 * GIB) as u64);
        f.put_u64("cgroup/a/memory.current", (58 * GIB) as u64); // 2 GiB room
        f.put_str("cgroup/a/b/memory.max", "max\n");
        f.put_u64("cgroup/a/b/memory.current", (50 * GIB) as u64);

        let r = f.sample(0).unwrap();
        assert_eq!(r.available, 2 * GIB);
        assert_eq!(r.source, MemorySource::Cgroup);
        assert_eq!(r.limit, 60 * GIB);
        assert_eq!(r.current, 58 * GIB);

        f.put_u64("cgroup/a/memory.current", (59 * GIB + 512 * MIB) as u64);
        let r = f.sample(0).unwrap();
        assert_eq!(r.available, 512 * MIB);

        f.put_u64("cgroup/a/memory.current", (61 * GIB) as u64);
        let r = f.sample(0).unwrap();
        assert_eq!(r.available, 0);
        assert_eq!(r.source, MemorySource::Cgroup);

        f.put_u64("cgroup/a/memory.current", (10 * GIB) as u64);
        let r = f.sample(0).unwrap();
        assert_eq!(r.available, 10 * GIB);
        assert_eq!(r.limit, 100 * GIB);
        assert_eq!(r.source, MemorySource::Cgroup);

        // MemAvailable bounds a roomy cgroup.
        f.meminfo(120 * GIB, 4 * GIB);
        let r = f.sample(0).unwrap();
        assert_eq!(r.available, 4 * GIB);
        assert_eq!(r.source, MemorySource::Meminfo);

        // memory.high below memory.max counts, and "max" does not.
        let f2 = Fixture::new();
        f2.meminfo(120 * GIB, 80 * GIB);
        f2.self_cgroup("0::/g");
        f2.put_u64("cgroup/g/memory.max", (100 * GIB) as u64);
        f2.put_u64("cgroup/g/memory.high", (99 * GIB + 900 * MIB) as u64);
        f2.put_u64("cgroup/g/memory.current", (99 * GIB) as u64);
        let r = f2.sample(0).unwrap();
        assert_eq!(r.available, 900 * MIB);
        assert_eq!(r.limit, 99 * GIB + 900 * MIB);

        f2.put_str("cgroup/g/memory.high", "max\n");
        let r = f2.sample(0).unwrap();
        assert_eq!(r.available, GIB);
        assert_eq!(r.limit, 100 * GIB);

        f2.put_str("cgroup/g/memory.max", "max\n");
        f2.put_u64("cgroup/g/memory.high", (90 * GIB) as u64);
        let r = f2.sample(0).unwrap();
        assert_eq!(r.available, 0);
    }

    #[test]
    fn operator_cap_and_tie_breaking() {
        let f = Fixture::new();
        f.meminfo(120 * GIB, 25 * GIB);
        f.self_cgroup("0::/system.slice/strata.service");
        f.put_str("cgroup/memory.max", "max\n");
        f.put_str("cgroup/memory.high", "max\n");
        f.put_u64("cgroup/memory.current", (99 * GIB + 724 * MIB) as u64);
        f.put_str("cgroup/system.slice/strata.service/memory.max", "max\n");
        f.put_u64(
            "cgroup/system.slice/strata.service/memory.current",
            (97 * GIB) as u64,
        );

        let r0 = f.sample(0).unwrap();
        assert_eq!(r0.available, 25 * GIB);
        assert_eq!(r0.source, MemorySource::Meminfo);

        let r = f.sample(100 * GIB).unwrap();
        assert_eq!(r.available, 300 * MIB);
        assert_eq!(r.source, MemorySource::Flag);
        assert_eq!(r.limit, 100 * GIB);
        assert_eq!(r.current, 99 * GIB + 724 * MIB);

        let r = f.sample(99 * GIB).unwrap();
        assert_eq!(r.available, 0);
        assert_eq!(r.source, MemorySource::Flag);

        let r = f.sample(200 * GIB).unwrap();
        assert_eq!(r.available, 25 * GIB);
        assert_eq!(r.source, MemorySource::Meminfo);

        // Visible cgroup limit vs operator cap.
        f.put_u64("cgroup/system.slice/memory.max", (98 * GIB) as u64);
        f.put_u64(
            "cgroup/system.slice/memory.current",
            (97 * GIB + 512 * MIB) as u64,
        );
        let r = f.sample(100 * GIB).unwrap();
        assert_eq!(r.available, 300 * MIB);
        assert_eq!(r.source, MemorySource::Flag);

        let r = f.sample(150 * GIB).unwrap();
        assert_eq!(r.available, 512 * MIB);
        assert_eq!(r.source, MemorySource::Cgroup);

        // Tie goes to operator's cap.
        let r = f.sample(99 * GIB + 724 * MIB + 512 * MIB).unwrap();
        assert_eq!(r.available, 512 * MIB);
        assert_eq!(r.source, MemorySource::Flag);

        // Without root memory.current, meminfo-derived usage (MemTotal - MemAvailable) is used.
        let _ = fs::remove_file(f.dir.join("cgroup/system.slice/memory.max"));
        let _ = fs::remove_file(f.dir.join("cgroup/memory.current"));
        let r = f.sample(100 * GIB).unwrap();
        assert_eq!(r.available, 5 * GIB);
        assert_eq!(r.source, MemorySource::Flag);
        assert_eq!(r.current, 95 * GIB);

        // Without MemTotal either, usage under a cap is unknown (None).
        f.put_str("meminfo", "MemAvailable: 1024 kB\n");
        assert!(f.sample(100 * GIB).is_none());
        assert!(f.sample(0).is_some());
    }

    #[test]
    fn malformed_meminfo_and_cgroup_files() {
        let bad_values = [
            "garbage\n",
            "",
            "\n",
            "-5\n",
            "+5\n",
            "12x\n",
            "1 2\n",
            "18446744073709551616\n",
            "0x10\n",
            "MAX\n",
        ];
        for file in ["memory.max", "memory.high", "memory.current"] {
            for text in bad_values {
                let f = Fixture::new();
                f.meminfo(120 * GIB, 25 * GIB);
                f.self_cgroup("0::/g");
                f.put_u64("cgroup/g/memory.max", (100 * GIB) as u64);
                f.put_str("cgroup/g/memory.high", "max\n");
                f.put_u64("cgroup/g/memory.current", (50 * GIB) as u64);
                f.put_str(&format!("cgroup/g/{file}"), text);

                // Without operator cap: falls back to MemAvailable.
                let r = f.sample(0).unwrap();
                assert_eq!(r.available, 25 * GIB);
                assert_eq!(r.source, MemorySource::Meminfo);

                // With operator cap: fails closed (None).
                assert!(f.sample(100 * GIB).is_none());
            }
        }

        let f = Fixture::new();
        f.self_cgroup("0::/g");
        f.put_u64("cgroup/g/memory.max", (100 * GIB) as u64);
        f.put_u64("cgroup/g/memory.current", (50 * GIB) as u64);
        f.meminfo(120 * GIB, 25 * GIB);
        assert_eq!(f.sample(0).unwrap().available, 25 * GIB);

        // Finite limit without memory.current beside it.
        let _ = fs::remove_file(f.dir.join("cgroup/g/memory.current"));
        assert_eq!(f.sample(0).unwrap().source, MemorySource::Meminfo);
        assert!(f.sample(100 * GIB).is_none());

        f.put_str("cgroup/g/memory.max", "max\n");
        assert_eq!(f.sample(0).unwrap().available, 25 * GIB);

        f.put_str("cgroup/g/memory.current", "max\n");
        assert_eq!(f.sample(0).unwrap().source, MemorySource::Meminfo);
        assert!(f.sample(100 * GIB).is_none());

        f.put_u64("cgroup/g/memory.current", 1);
        for bad_meminfo in [
            "",
            "MemAvailable: 12 MB\n",
            "MemAvailable: 1x kB\n",
            "MemTotal: 5 kB\n",
            "MemAvailable: 5 kB\nMemAvailable: 6 kB\n",
        ] {
            f.put_str("meminfo", bad_meminfo);
            assert!(f.sample(0).is_none());
            assert!(f.probe().mem_available_bytes().is_err());
        }

        let _ = fs::remove_file(f.dir.join("meminfo"));
        assert!(f.sample(0).is_none());

        // Path traversal ".." in cgroup path is rejected.
        f.meminfo(120 * GIB, 25 * GIB);
        f.self_cgroup("0::/a/../g");
        assert_eq!(f.sample(0).unwrap().source, MemorySource::Meminfo);
        assert!(f.sample(100 * GIB).is_none());
    }

    #[test]
    fn cgroup_v1_hybrid_and_reclaimable_inactive_file_credit() {
        // cgroup v1 hierarchy and unlimited sentinel.
        let f = Fixture::new();
        f.meminfo(120 * GIB, 80 * GIB);
        f.self_cgroup("12:cpu,cpuacct:/x\n4:memory:/m/n\n1:name=systemd:/x");
        f.put_u64(
            "cgroup/memory/memory.limit_in_bytes",
            9_223_372_036_854_771_712u64,
        );
        f.put_u64("cgroup/memory/memory.usage_in_bytes", (90 * GIB) as u64);
        f.put_u64("cgroup/memory/m/memory.limit_in_bytes", (64 * GIB) as u64);
        f.put_u64("cgroup/memory/m/memory.usage_in_bytes", (60 * GIB) as u64);
        f.put_u64(
            "cgroup/memory/m/n/memory.limit_in_bytes",
            9_223_372_036_854_771_712u64,
        );
        f.put_u64("cgroup/memory/m/n/memory.usage_in_bytes", GIB as u64);

        let r = f.sample(0).unwrap();
        assert_eq!(r.available, 4 * GIB);
        assert_eq!(r.source, MemorySource::Cgroup);
        assert_eq!(r.limit, 64 * GIB);

        // Reclaimable inactive_file cache credit in cgroup v2.
        let stat = |inactive: usize, active: usize, shmem: usize, dirty: usize, wb: usize| {
            format!(
                "anon 1\nfile 2\nshmem {shmem}\ninactive_file {inactive}\n\
                 active_file {active}\nfile_dirty {dirty}\nfile_writeback {wb}\nunevictable 5\n"
            )
        };
        let f2 = Fixture::new();
        f2.meminfo(120 * GIB, 25 * GIB);
        f2.self_cgroup("0::/system.slice/strata.service");
        f2.put_str("cgroup/memory.max", "max\n");
        f2.put_str("cgroup/memory.high", "max\n");
        f2.put_u64("cgroup/memory.current", (98 * GIB + 324 * MIB) as u64);
        f2.put_str(
            "cgroup/memory.stat",
            &stat(3 * GIB + 800 * MIB, 27 * GIB, 65 * GIB, 0, 0),
        );
        f2.put_str("cgroup/system.slice/strata.service/memory.max", "max\n");
        f2.put_u64(
            "cgroup/system.slice/strata.service/memory.current",
            (90 * GIB) as u64,
        );

        let r = f2.sample(99 * GIB).unwrap();
        assert_eq!(r.source, MemorySource::Flag);
        assert_eq!(r.credit, 3 * GIB + 800 * MIB);
        assert_eq!(
            r.available,
            99 * GIB - (98 * GIB + 324 * MIB - (3 * GIB + 800 * MIB))
        );

        // Dirty + writeback pages reduce credit.
        f2.put_str("cgroup/memory.stat", &stat(3 * GIB, 0, 0, GIB, 512 * MIB));
        let r = f2.sample(99 * GIB).unwrap();
        assert_eq!(r.credit, GIB + 512 * MIB);
        assert_eq!(r.available, 700 * MIB + GIB + 512 * MIB);

        // Dirty + writeback exceeding inactive_file leaves 0 credit.
        f2.put_str("cgroup/memory.stat", &stat(3 * GIB, 0, 0, 2 * GIB, 2 * GIB));
        let r = f2.sample(99 * GIB).unwrap();
        assert_eq!(r.credit, 0);
        assert_eq!(r.available, 700 * MIB);

        // inactive_file larger than current usage stops credit at usage.
        f2.put_str("cgroup/memory.stat", &stat(500 * GIB, 0, 0, 0, 0));
        f2.meminfo(300 * GIB, 200 * GIB);
        let r = f2.sample(99 * GIB).unwrap();
        assert_eq!(r.credit, 98 * GIB + 324 * MIB);
        assert_eq!(r.available, 99 * GIB);
        assert_eq!(r.source, MemorySource::Flag);

        // Missing or unparsable memory.stat gives 0 credit without failing the sample.
        let _ = fs::remove_file(f2.dir.join("cgroup/memory.stat"));
        let r = f2.sample(99 * GIB).unwrap();
        assert_eq!(r.available, 700 * MIB);
        assert_eq!(r.credit, 0);
        for bad_stat in [
            "",
            "garbage\n",
            "inactive_file\n",
            "inactive_file -1\n",
            "inactive_file 12x\n",
            "inactive_file 1\ninactive_file 2\n",
            "inactive_file 99999999999999999999\n",
            "inactive_file 4096\nfile_dirty zz\n",
        ] {
            f2.put_str("cgroup/memory.stat", bad_stat);
            let r = f2.sample(99 * GIB).unwrap();
            assert_eq!(r.available, 700 * MIB);
            assert_eq!(r.credit, 0);
        }

        // cgroup v1 total_inactive_file vs non-total inactive_file.
        let g = Fixture::new();
        g.meminfo(120 * GIB, 80 * GIB);
        g.self_cgroup("4:memory:/m");
        g.put_u64("cgroup/memory/m/memory.limit_in_bytes", (64 * GIB) as u64);
        g.put_u64("cgroup/memory/m/memory.usage_in_bytes", (62 * GIB) as u64);
        g.put_u64(
            "cgroup/memory/memory.limit_in_bytes",
            9_223_372_036_854_771_712u64,
        );
        g.put_u64("cgroup/memory/memory.usage_in_bytes", (62 * GIB) as u64);
        g.put_str(
            "cgroup/memory/m/memory.stat",
            &format!(
                "cache 1\ninactive_file 7\ntotal_inactive_file {}\ntotal_dirty {}\ntotal_writeback 0\n",
                3 * GIB,
                GIB
            ),
        );
        let r = g.sample(0).unwrap();
        assert_eq!(r.available, 4 * GIB);
        assert_eq!(r.credit, 2 * GIB);
        assert_eq!(r.source, MemorySource::Cgroup);

        g.put_str(
            "cgroup/memory/m/memory.stat",
            &format!("inactive_file {}\n", 3 * GIB),
        );
        let r = g.sample(0).unwrap();
        assert_eq!(r.available, 2 * GIB);
        assert_eq!(r.credit, 0);
    }

    #[test]
    fn cgroup_invisible_subpath_missing_self_cgroup_and_hybrid_fallback() {
        // Invisible container subpath falls back to deepest visible prefix.
        let f = Fixture::new();
        f.meminfo(120 * GIB, 25 * GIB);
        f.self_cgroup("0::/lxc/105/ns/deep");
        f.put_u64("cgroup/memory.max", (100 * GIB) as u64);
        f.put_u64("cgroup/memory.current", (99 * GIB) as u64);
        let r = f.sample(0).unwrap();
        assert_eq!(r.available, GIB);
        assert_eq!(r.source, MemorySource::Cgroup);

        f.put_u64("cgroup/lxc/memory.max", (100 * GIB) as u64);
        f.put_u64("cgroup/lxc/memory.current", (99 * GIB + 256 * MIB) as u64);
        let r = f.sample(0).unwrap();
        assert_eq!(r.available, GIB - 256 * MIB);

        // No /proc/self/cgroup file -> reads root cgroup; no cgroup mount -> MemAvailable.
        let f2 = Fixture::new();
        f2.meminfo(120 * GIB, 25 * GIB);
        f2.put_u64("cgroup/memory.max", (100 * GIB) as u64);
        f2.put_u64("cgroup/memory.current", (98 * GIB) as u64);
        let r = f2.sample(0).unwrap();
        assert_eq!(r.available, 2 * GIB);

        let _ = fs::remove_dir_all(f2.dir.join("cgroup"));
        let r = f2.sample(0).unwrap();
        assert_eq!(r.available, 25 * GIB);
        assert_eq!(r.source, MemorySource::Meminfo);
        let r_cap = f2.sample(100 * GIB).unwrap();
        assert_eq!(r_cap.available, 5 * GIB);
        assert_eq!(r_cap.source, MemorySource::Flag);

        // Hybrid v2 + v1 where memory controller is only in v1 tree.
        let f3 = Fixture::new();
        f3.meminfo(120 * GIB, 80 * GIB);
        f3.self_cgroup("5:memory:/g\n0::/g");
        f3.put_str("cgroup/cgroup.controllers", "cpu\n");
        f3.put_str("cgroup/g/cpu.max", "max 100000\n");
        f3.put_u64("cgroup/memory/g/memory.limit_in_bytes", (10 * GIB) as u64);
        f3.put_u64("cgroup/memory/g/memory.usage_in_bytes", (9 * GIB) as u64);
        let r = f3.sample(0).unwrap();
        assert_eq!(r.available, GIB);
        assert_eq!(r.source, MemorySource::Cgroup);

        // When v2 memory files exist, v2 wins over v1.
        f3.put_str("cgroup/memory.max", "max\n");
        f3.put_u64("cgroup/memory.current", GIB as u64);
        let r = f3.sample(0).unwrap();
        assert_eq!(r.available, 80 * GIB);

        // conversation_memory_admit checks.
        assert!(!conversation_memory_admit(None, 0, 8 * GIB));
        assert!(!conversation_memory_admit(Some(7 * GIB), 0, 8 * GIB));
        assert!(conversation_memory_admit(Some(10 * GIB), 2 * GIB, 8 * GIB));
        assert!(!conversation_memory_admit(
            Some(10 * GIB),
            2 * GIB + 1,
            8 * GIB
        ));
    }

    // ========================================================================
    // 3. Budget floor enforcement (8 GiB default, 11 GiB MiMo, categories, RAII)
    // ========================================================================

    #[test]
    fn budget_enforces_8gib_default_floor_and_category_accounting() {
        // Host starts with 30 GiB MemAvailable; default floor is 8 GiB -> 22 GiB headroom.
        let probe = MockMemProbe::new(30 * GIB);
        let budget = Budget::new(probe.clone());
        assert_eq!(budget.floor_bytes(), 8 * GIB);
        assert_eq!(budget.headroom_bytes(), 22 * GIB);
        assert!(budget.can_fit(22 * GIB));
        assert!(!budget.can_fit(22 * GIB + 1));

        // Reserve across multiple categories.
        let r_weights = budget
            .try_reserve(BudgetCategory::Weights, "qwen35.weights", 14 * GIB)
            .expect("reserve weights");
        let r_kv = budget
            .try_reserve(BudgetCategory::KvCache, "qwen35.kv", 4 * GIB)
            .expect("reserve kv");
        let r_gdn = budget
            .try_reserve(BudgetCategory::RecurrentState, "qwen35.gdn", GIB)
            .expect("reserve gdn");
        let r_snap = budget
            .try_reserve(BudgetCategory::PromptSnapshots, "qwen35.snaps", GIB)
            .expect("reserve snaps");
        let r_stash = budget
            .try_reserve(BudgetCategory::KvStash, "qwen35.stash", 512 * MIB)
            .expect("reserve stash");
        let r_drafter = budget
            .try_reserve(BudgetCategory::Drafter, "qwen35.mtp", 512 * MIB)
            .expect("reserve drafter");
        let r_scratch = budget
            .try_reserve(BudgetCategory::Scratch, "qwen35.scratch", 512 * MIB)
            .expect("reserve scratch");

        // Total reserved = 21.5 GiB; remaining headroom above 8 GiB floor = 512 MiB.
        assert_eq!(budget.total_reserved_bytes(), 21 * GIB + 512 * MIB);
        assert_eq!(budget.headroom_bytes(), 512 * MIB);
        assert!(budget.can_fit(512 * MIB));
        assert!(!budget.can_fit(512 * MIB + 1));

        // Attempting to reserve 1 GiB would leave 7.5 GiB (< 8 GiB floor) -> FloorViolation.
        let err = budget
            .try_reserve(BudgetCategory::KvStash, "overflow_stash", GIB)
            .expect_err("must reject allocation that violates 8 GiB floor");
        assert!(err.is_floor_violation());
        match &err {
            MemError::FloorViolation {
                category,
                label,
                requested_bytes,
                mem_available_bytes,
                unmaterialized_reserved_bytes,
                effective_available_bytes,
                floor_bytes,
                headroom_bytes,
                total_reserved_bytes,
                category_totals,
                ..
            } => {
                assert_eq!(*category, BudgetCategory::KvStash);
                assert_eq!(label, "overflow_stash");
                assert_eq!(*requested_bytes, GIB);
                assert_eq!(*mem_available_bytes, 30 * GIB);
                assert_eq!(*unmaterialized_reserved_bytes, 21 * GIB + 512 * MIB);
                assert_eq!(*effective_available_bytes, 8 * GIB + 512 * MIB);
                assert_eq!(*floor_bytes, 8 * GIB);
                assert_eq!(*headroom_bytes, 512 * MIB);
                assert_eq!(*total_reserved_bytes, 21 * GIB + 512 * MIB);
                assert_eq!(category_totals[&BudgetCategory::Weights], 14 * GIB);
                assert_eq!(category_totals[&BudgetCategory::KvCache], 4 * GIB);
                assert_eq!(category_totals[&BudgetCategory::RecurrentState], GIB);
                assert_eq!(category_totals[&BudgetCategory::PromptSnapshots], GIB);
                assert_eq!(category_totals[&BudgetCategory::KvStash], 512 * MIB);
                assert_eq!(category_totals[&BudgetCategory::Drafter], 512 * MIB);
                assert_eq!(category_totals[&BudgetCategory::Scratch], 512 * MIB);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }

        // Convert MemError to CoreError::OutOfMemory.
        let core_err: CoreError = err.into();
        assert_eq!(
            core_err,
            CoreError::OutOfMemory {
                requested_bytes: GIB as u64,
                available_bytes: (512 * MIB) as u64,
            }
        );

        // Release KvCache (4 GiB) and verify headroom increases while peak usage is preserved.
        let peak_before = budget.peak_reserved_bytes();
        assert_eq!(budget.release(r_kv), 4 * GIB);
        assert_eq!(budget.category_bytes(BudgetCategory::KvCache), 0);
        assert_eq!(budget.category_peak_bytes(BudgetCategory::KvCache), 4 * GIB);
        assert_eq!(budget.peak_reserved_bytes(), peak_before);
        assert_eq!(budget.headroom_bytes(), 4 * GIB + 512 * MIB);

        // Reuse the released space with an RAII ReservationGuard.
        {
            let guard = budget
                .try_reserve_guard(BudgetCategory::KvStash, "reused_stash", 4 * GIB)
                .expect("reuse freed headroom");
            assert_eq!(guard.category(), BudgetCategory::KvStash);
            assert_eq!(guard.bytes(), 4 * GIB);
            assert_eq!(budget.headroom_bytes(), 512 * MIB);
        }
        // Guard dropped -> 4 GiB automatically returned.
        assert_eq!(budget.headroom_bytes(), 4 * GIB + 512 * MIB);
        assert_eq!(budget.category_bytes(BudgetCategory::KvStash), 512 * MIB);
        assert_eq!(
            budget.category_peak_bytes(BudgetCategory::KvStash),
            4 * GIB + 512 * MIB
        );

        let summary = budget.summary();
        assert_eq!(summary.floor_bytes, 8 * GIB);
        assert_eq!(summary.peak_reserved_bytes, 21 * GIB + 512 * MIB);
        assert_eq!(summary.total_reserved_bytes, 17 * GIB + 512 * MIB);
        assert_eq!(summary.headroom_bytes, 4 * GIB + 512 * MIB);
        assert_eq!(summary.active_reservation_count, 6);

        budget.release(r_weights);
        budget.release(r_gdn);
        budget.release(r_snap);
        budget.release(r_stash);
        budget.release(r_drafter);
        budget.release(r_scratch);
        assert_eq!(budget.total_reserved_bytes(), 0);
        assert_eq!(budget.active_reservation_count(), 0);
    }

    #[test]
    fn budget_enforces_11gib_mimo_floor_and_budget_cap() {
        let probe = MockMemProbe::new(20 * GIB);
        let budget =
            Budget::with_config(probe.clone(), BudgetConfig::mimo().with_max_budget(7 * GIB));
        assert_eq!(budget.floor_bytes(), 11 * GIB);
        assert_eq!(budget.max_budget_bytes(), Some(7 * GIB));

        // Floor headroom is 20 - 11 = 9 GiB, but max_budget_bytes is 7 GiB -> headroom is 7 GiB.
        assert_eq!(budget.headroom_bytes(), 7 * GIB);
        assert!(budget.can_fit(7 * GIB));
        assert!(!budget.can_fit(7 * GIB + 1));

        // Requesting 8 GiB passes the 9 GiB floor headroom check, but fails the 7 GiB budget cap.
        let cap_err = budget
            .try_reserve(BudgetCategory::Weights, "mimo_experts", 8 * GIB)
            .expect_err("must exceed 7 GiB cap");
        assert!(cap_err.is_cap_exceeded());

        // Requesting 10 GiB violates the 11 GiB floor (20 - 10 = 10 < 11 GiB).
        let floor_err = budget
            .try_reserve(BudgetCategory::Weights, "mimo_too_big", 10 * GIB)
            .expect_err("must violate 11 GiB MiMo floor");
        assert!(floor_err.is_floor_violation());

        // Materialized reservation transitions accounting when OS MemAvailable drops.
        let res = budget
            .try_reserve(BudgetCategory::Weights, "mimo_chunk", 5 * GIB)
            .expect("reserve 5 GiB");
        assert_eq!(budget.unmaterialized_reserved_bytes(), 5 * GIB);
        probe.sub_available_bytes(5 * GIB); // OS now reports 15 GiB MemAvailable
        budget.mark_materialized(res.id).expect("mark materialized");
        assert_eq!(budget.unmaterialized_reserved_bytes(), 0);
        // Effective available is 15 GiB, floor is 11 GiB -> 4 GiB floor headroom, 2 GiB cap headroom.
        assert_eq!(budget.headroom_bytes(), 2 * GIB);
    }

    #[test]
    fn budget_with_synthetic_low_memory_fixture() {
        let f = Fixture::new();
        // Synthetic host with 10 GiB MemAvailable out of 128 GiB total.
        f.meminfo(128 * GIB, 10 * GIB);
        let budget = Budget::new(f.probe());
        assert_eq!(budget.headroom_bytes(), 2 * GIB);

        let r1 = budget
            .try_reserve(BudgetCategory::Weights, "weights", 2 * GIB)
            .expect("2 GiB fits exactly above 8 GiB floor");
        assert_eq!(budget.headroom_bytes(), 0);
        assert!(!budget.can_fit(1));

        let err = budget
            .try_reserve(BudgetCategory::Scratch, "extra_byte", 1)
            .expect_err("1 byte past 8 GiB floor must fail");
        assert!(err.is_floor_violation());

        budget.release(r1);
        assert_eq!(budget.headroom_bytes(), 2 * GIB);
    }

    // ========================================================================
    // 4. ScratchArenaPlan & SlabPlanner 256-byte alignment and peak reuse math
    // ========================================================================

    #[test]
    fn scratch_arena_plan_256_byte_alignment_and_peak_reuse() {
        let mut plan = ScratchArenaPlan::new();
        assert_eq!(plan.base_alignment(), 256);

        // Persistent activation buffers with odd element counts.
        let x: TypedRegion<f32> = plan.alloc::<f32>("x", 100).expect("alloc x");
        assert_eq!(x.offset(), 0);
        assert_eq!(x.size_bytes(), 400);
        assert_eq!(x.aligned_bytes(), 512);
        assert_eq!(x.offset() % 256, 0);

        let h: TypedRegion<f32> = plan.alloc::<f32>("h", 65).expect("alloc h");
        assert_eq!(h.offset(), 512);
        assert_eq!(h.size_bytes(), 260);
        assert_eq!(h.aligned_bytes(), 512);
        assert_eq!(h.offset() % 256, 0);

        assert_eq!(plan.current_bytes(), 1024);
        assert_eq!(plan.peak_bytes(), 1024);

        // Scoped Phase 1: Attention scratch (q, k, v, att).
        let (q_off, att_end) = plan
            .with_scope(|s| {
                let q = s.alloc::<f32>("attn_q", 4096)?; // 16384 B
                let k = s.alloc::<f32>("attn_k", 1024)?; // 4096 B
                let v = s.alloc::<f32>("attn_v", 1024)?; // 4096 B
                let att = s.alloc_bytes("attn_scores", 1000)?; // 1000 B -> 1024 B aligned
                assert_eq!(q.offset(), 1024);
                assert_eq!(k.offset(), 1024 + 16384);
                assert_eq!(v.offset(), 1024 + 16384 + 4096);
                assert_eq!(att.offset, 1024 + 16384 + 4096 + 4096);
                assert_eq!(att.offset % 256, 0);
                Ok((q.offset(), att.aligned_end_offset()))
            })
            .expect("attn scope");

        assert_eq!(q_off, 1024);
        assert_eq!(att_end, 1024 + 16384 + 4096 + 4096 + 1024); // 26624 B
                                                                // Cursor unwound to 1024 after scope exit, while peak_bytes recorded 26624!
        assert_eq!(plan.current_bytes(), 1024);
        assert_eq!(plan.peak_bytes(), 26624);

        // Scoped Phase 2: MoE FFN scratch reuses [1024..].
        let moe_end = plan
            .with_scope(|s| {
                let eparts = s.alloc::<f32>("moe_eparts", 8192)?; // 32768 B
                let moe_out = s.alloc::<f32>("moe_out", 2048)?; // 8192 B
                assert_eq!(eparts.offset(), 1024);
                assert_eq!(moe_out.offset(), 1024 + 32768);
                Ok(moe_out.region().aligned_end_offset())
            })
            .expect("moe scope");

        assert_eq!(moe_end, 1024 + 32768 + 8192); // 41984 B
        assert_eq!(plan.current_bytes(), 1024);
        assert_eq!(plan.peak_bytes(), 41984);
        assert_eq!(plan.total_requested_bytes(), 1024 + 25600 + 40960);
        assert_eq!(plan.saved_bytes(), 25600);

        // Explicit free_region and best-fit coalescing reuse.
        let tmp1 = plan.alloc_bytes("tmp1", 500).unwrap(); // [1024..1536) (512 B)
        let tmp2 = plan.alloc_bytes("tmp2", 500).unwrap(); // [1536..2048) (512 B)
        let pin = plan.alloc_bytes("pin", 200).unwrap(); // [2048..2304) (256 B)
        assert_eq!(tmp1.offset, 1024);
        assert_eq!(tmp2.offset, 1536);
        assert_eq!(pin.offset, 2048);

        // Free tmp1 and tmp2 -> they coalesce into a single [1024..2048) 1024-byte hole.
        plan.free_region(tmp1.id).unwrap();
        plan.free_region(tmp2.id).unwrap();
        let reused = plan.alloc_bytes("reused_1k", 900).unwrap(); // 900 B -> 1024 B aligned
        assert_eq!(reused.offset, 1024);
        assert_eq!(reused.aligned_bytes, 1024);
        // Peak was not increased by any of these smaller allocations.
        assert_eq!(plan.peak_bytes(), 41984);

        // Verify every planned region has 256-byte aligned offset and aligned_bytes.
        for r in plan.regions() {
            assert_eq!(r.offset % 256, 0, "region '{}' unaligned offset", r.label);
            assert_eq!(
                r.aligned_bytes % 256,
                0,
                "region '{}' unaligned size",
                r.label
            );
            let ptr = r
                .resolve_ptr(0x1000_0000)
                .expect("resolve aligned base ptr");
            assert_eq!(ptr % 256, 0);
        }
        assert!(x.resolve_ptr(0x1000_0080).is_err());
    }

    #[test]
    fn slab_planner_kv_and_recurrent_snapshot_pools() {
        let mut planner = SlabPlanner::new();

        // 1. Plan KV pools for 4 layers (F16 with vt_tile_stride = 16 and kv_pad_stride = 256).
        let kv_layout = KvLayout {
            num_layers: 4,
            n_head_kv: 4,
            head_dim_k: 256,
            head_dim_v: 256,
            max_seq_len: 1000, // padded to 1024 by kv_pad_stride=256
            ring_window: None,
            kv_pad_stride: 256,
            vt_tile_stride: Some(16),
            format: KvCacheFormat::F16,
        };
        let kv_plan = planner.plan_kv_pools(&kv_layout).expect("plan kv pools");
        assert_eq!(kv_plan.layers.len(), 4);
        for layer in &kv_plan.layers {
            assert_eq!(layer.padded_positions, 1024);
            assert_eq!(layer.k_pool.offset % 256, 0);
            assert_eq!(layer.v_pool.offset % 256, 0);
            let vt = layer.vt_pool.as_ref().expect("vt pool present");
            assert_eq!(vt.offset % 256, 0);
            assert!(!layer.k_pool.overlaps(&layer.v_pool));
            assert!(!layer.v_pool.overlaps(vt));
        }

        // 2. Plan GDN recurrent snapshot pool (12 snapshots: 8 turn + 4 message-start).
        let rec_layout = RecurrentLayout {
            num_layers: 48,
            state_size: 128,
            h_k: 16,
            h_v: 48,
            d_conv: 4,
            conv_channels: 10240,
        };
        let rec_plan = planner
            .plan_recurrent_snapshots(&rec_layout, 5120, 12)
            .expect("plan recurrent snapshots");
        assert_eq!(rec_plan.slab.slot_count, 12);
        assert_eq!(rec_plan.slab.base_offset % 256, 0);
        assert_eq!(rec_plan.slab.slot_stride_bytes % 256, 0);
        assert_eq!(rec_plan.hlast_bytes, 5120 * 4);
        assert_eq!(rec_plan.conv_bytes_per_layer, 10240 * 3 * 4);
        assert_eq!(rec_plan.ssm_bytes_per_layer, 48 * 128 * 128 * 4);

        for slot_idx in 0..12 {
            let off = rec_plan.slab.slot_offset(slot_idx).unwrap();
            assert_eq!(off % 256, 0);
            let reg = rec_plan.slab.slot_region(slot_idx).unwrap();
            assert_eq!(reg.offset, off);
        }
        assert!(rec_plan.slab.slot_offset(12).is_err());
        assert_eq!(planner.total_bytes() % 256, 0);

        // 3. Plan quantized SWA-128 ring KV pools (Q8_0 and Q4_0) and slice into a backing buffer.
        let mut swa_planner = SlabPlanner::new();
        let q8_layout = KvLayout {
            num_layers: 2,
            n_head_kv: 8,
            head_dim_k: 128,
            head_dim_v: 128,
            max_seq_len: 32768,
            ring_window: Some(128),
            kv_pad_stride: 256,
            vt_tile_stride: None,
            format: KvCacheFormat::Q8_0,
        };
        let q8_plan = swa_planner.plan_kv_pools(&q8_layout).unwrap();
        assert_eq!(q8_plan.layers[0].padded_positions, 256);
        assert!(q8_plan.layers[0].vt_pool.is_none());

        let q4_layout = KvLayout {
            format: KvCacheFormat::Q4_0,
            ..q8_layout
        };
        let q4_plan = swa_planner.plan_kv_pools(&q4_layout).unwrap();
        assert!(q4_plan.total_bytes < q8_plan.total_bytes);

        let mut backing = vec![0u8; swa_planner.total_bytes()];
        let k0_slice = q8_plan.layers[0].k_pool.slice_mut(&mut backing).unwrap();
        k0_slice[0] = 0x5A;
        assert_eq!(q8_plan.layers[0].k_pool.slice(&backing).unwrap()[0], 0x5A);

        // 4. Failing probe behavior in Budget.
        let failing_budget = Budget::new(MockMemProbe::failing("simulated failure"));
        assert_eq!(failing_budget.headroom_bytes(), 0);
        assert!(!failing_budget.can_fit(1));
        assert!(failing_budget
            .try_reserve(BudgetCategory::Scratch, "x", 1)
            .is_err());
    }
}
