//! padded copies of the f32 router rows (`talus_rowpad_*` in libtalus_kernels, include/strata/kernels/rowpad.hpp).
//!
//! Router / gate weights are read one block (or wave) per 12-16 KiB row, rows a power of two apart, so every block hits
//! the same DRAM channel at once. With `TALUS_ROUTER_PAD=1` the library keeps a copy whose rows start 512 B further
//! apart; the decode / verify router kernels read it (same bytes per row: the same bits). The engines register their
//! router tensors at load and release them on drop; without the switch (or with an older library) this is a no-op.
use std::ffi::c_void;
use talus_kernels_sys::LibTalusKernels;

type PrepareFn = unsafe extern "C" fn(*const c_void, i64, i64, *mut c_void) -> i32;
type ReleaseFn = unsafe extern "C" fn(*const c_void);
type BytesFn = unsafe extern "C" fn() -> i64;

/// The router tensors one engine registered; dropping it releases them.
#[derive(Default)]
pub struct RowPad {
    release: Option<ReleaseFn>,
    held: Vec<usize>,
    bytes: i64,
}

impl RowPad {
    /// Register `rows` rows of `row_bytes` at each pointer (device memory). Returns the guard; a no-op when padding is
    /// off or the library has no `talus_rowpad_prepare`.
    ///
    /// `default_on`  ): the engine pads unless `TALUS_ROUTER_PAD` says otherwise. MiMo (streaming f32 router
    /// kernel on the padded rows) and Laguna (served decode through verify(1)) pass true: exact and faster. GLM keeps
    /// it opt-in (serving context near the floor).
    pub fn register(
        lib: &LibTalusKernels,
        tensors: &[(*const c_void, i64, i64)],
        default_on: bool,
    ) -> Self {
        let mut me = Self::default();
        if default_on && std::env::var_os("TALUS_ROUTER_PAD").is_none() {
            // read once by the library's pad_bytes() at the first prepare: set before it
            std::env::set_var("TALUS_ROUTER_PAD", "1");
        }
        let (Some(p), Some(r)) = (
            lib.optional_sym("talus_rowpad_prepare"),
            lib.optional_sym("talus_rowpad_release"),
        ) else {
            return me;
        };
        // The copy costs host-visible memory (UMA): skip it unless MemAvailable stays >= 10 GiB after it (the GLM
        // engine exits at its 8 GiB floor; rdpad tier1 showed +200 MiB tipping a 9.3 GiB GLM load over during a 32K
        // prompt). TALUS_ROUTER_PAD_MIN_AVAIL_GIB overrides the margin.
        let need: i64 = tensors
            .iter()
            .filter(|t| !t.0.is_null())
            .map(|t| t.1 * (t.2 + 512))
            .sum();
        let margin_gib: f64 = std::env::var("TALUS_ROUTER_PAD_MIN_AVAIL_GIB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10.0);
        if std::env::var("TALUS_ROUTER_PAD")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
        {
            if let Some(avail) = mem_available_bytes() {
                if (avail as f64) - (need as f64) < margin_gib * 1073741824.0 {
                    eprintln!(
                        "talus: TALUS_ROUTER_PAD skipped: MemAvailable {:.2} GiB - {:.1} MiB copy < {margin_gib} GiB",
                        avail as f64 / 1073741824.0,
                        need as f64 / 1048576.0
                    );
                    return me;
                }
            }
        }
        // SAFETY: the C ABI in include/strata/kernels/rowpad.hpp matches these types.
        let prepare: PrepareFn = unsafe { std::mem::transmute::<*mut c_void, PrepareFn>(p) };
        me.release = Some(unsafe { std::mem::transmute::<*mut c_void, ReleaseFn>(r) });
        for &(w, rows, row_bytes) in tensors {
            if w.is_null() {
                continue;
            }
            // SAFETY: w is a live device allocation of rows * row_bytes bytes.
            let rc = unsafe { prepare(w, rows, row_bytes, std::ptr::null_mut()) };
            if rc == 0 {
                break; // padding off: nothing else will qualify either
            }
            if rc > 0 {
                me.held.push(w as usize);
            }
        }
        if !me.held.is_empty() {
            if let Some(b) = lib.optional_sym("talus_rowpad_bytes") {
                // SAFETY: matches the C ABI.
                me.bytes = unsafe { std::mem::transmute::<*mut c_void, BytesFn>(b)() };
            }
            eprintln!(
                "talus: router rows padded (TALUS_ROUTER_PAD): {} tensors, {:.1} MiB",
                me.held.len(),
                me.bytes as f64 / 1048576.0
            );
        }
        me
    }
    /// Number of registered tensors.
    pub fn len(&self) -> usize {
        self.held.len()
    }
    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }
}

/// MemAvailable from /proc/meminfo, bytes.
fn mem_available_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let l = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = l.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

impl Drop for RowPad {
    fn drop(&mut self) {
        if let Some(r) = self.release {
            for &w in &self.held {
                // SAFETY: w was registered by this guard.
                unsafe { r(w as *const c_void) };
            }
        }
    }
}
