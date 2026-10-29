//! Piece 1 (`qwen4exp-p1`): the expert profile, the VRAM budget and the expert cache (expert_cache.cpp, generate.cpp sizing + fill).

#![allow(clippy::missing_safety_doc)]

use super::*;

// =====================================================================================================================
// The expert profile, the VRAM budget, and the expert cache (src/core/expert_cache.cpp, generate.cpp's sizing + fill)
// =====================================================================================================================

/// `strata::core::kNotResident`.
pub const NOT_RESIDENT: i32 = -1;

/// `strata::core::read_expert_profile`: `STRP`, `version n_layers n_expert slots n_ranked` (u32), then `n_ranked`
/// `(layer, expert)` u16 pairs ranked hottest first. Returns `(ranked, slots)`.
pub fn read_expert_profile(
    path: &str,
    n_layers: i64,
    n_expert: i64,
) -> Result<(Vec<(i32, i32)>, i64), String> {
    let mut f = File::open(path).map_err(|_| format!("read_expert_profile: cannot open {path}"))?;
    use std::io::Read;
    let mut head = [0u8; 24];
    if f.read_exact(&mut head).is_err() {
        return Err(format!(
            "read_expert_profile: {path} is too short to hold a header"
        ));
    }
    if &head[0..4] != b"STRP" {
        return Err(format!(
            "read_expert_profile: {path} does not start with STRP"
        ));
    }
    let u = |i: usize| {
        u32::from_le_bytes([
            head[4 + i * 4],
            head[5 + i * 4],
            head[6 + i * 4],
            head[7 + i * 4],
        ])
    };
    let (version, nl, ne, want, n_ranked) = (u(0), u(1), u(2), u(3), u(4));
    if version != 1 {
        return Err(format!(
            "read_expert_profile: {path} is a version {version} profile but this engine reads version 1 - it was \
             written by a different release's tools/make_profile.py or --expert-profile-save"
        ));
    }
    if nl as i64 != n_layers || ne as i64 != n_expert {
        return Err(format!(
            "read_expert_profile: {path} is {nl}x{ne} but this model is {n_layers}x{n_expert} - it is a profile for a \
             different artifact"
        ));
    }
    if n_ranked > want {
        return Err("read_expert_profile: the header claims more ranked pairs than slots".into());
    }
    let mut raw = vec![0u8; n_ranked as usize * 4];
    if n_ranked > 0 && f.read_exact(&mut raw).is_err() {
        return Err("read_expert_profile: the ranked list is truncated".into());
    }
    let mut ranked = Vec::with_capacity(n_ranked as usize);
    for i in 0..n_ranked as usize {
        let l = u16::from_le_bytes([raw[i * 4], raw[i * 4 + 1]]) as i32;
        let e = u16::from_le_bytes([raw[i * 4 + 2], raw[i * 4 + 3]]) as i32;
        if l as i64 >= n_layers || e as i64 >= n_expert {
            return Err(format!(
                "read_expert_profile: pair {i} is (layer {l}, expert {e}), out of range"
            ));
        }
        ranked.push((l, e));
    }
    Ok((ranked, want as i64))
}

pub(super) fn meminfo_kb(key: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            let mut t = rest.split_ascii_whitespace();
            let v: u64 = t.next()?.parse().ok()?;
            if t.next()? == "kB" {
                return Some(v);
            }
        }
    }
    None
}

/// `STRATA_UMA_HEADROOM_GIB` (0..1024, default 6), read once.
pub(super) fn uma_headroom_gib() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("STRATA_UMA_HEADROOM_GIB") {
        Err(_) => 6,
        Ok(h) => match h.parse::<i64>() {
            Ok(v) if (0..=1024).contains(&v) => v as u64,
            _ => {
                eprintln!(
                    "talus: STRATA_UMA_HEADROOM_GIB={h} is not a whole number of GiB (0-1024): using 6"
                );
                6
            }
        },
    })
}

/// Whether the current device is an APU (unified memory), cached per device.
pub(super) fn device_integrated() -> bool {
    static CACHE: Mutex<[u8; 64]> = Mutex::new([0u8; 64]);
    let mut dev: c_int = 0;
    if unsafe { hip_sys::hipGetDevice(&mut dev) } != hip_sys::hipSuccess || !(0..64).contains(&dev)
    {
        return false;
    }
    let mut c = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if c[dev as usize] == 0 {
        let mut v: c_int = 0;
        let ok = unsafe {
            hip_sys::hipDeviceGetAttribute(
                &mut v,
                hip_sys::hipDeviceAttribute_t::hipDeviceAttributeIntegrated,
                dev,
            )
        } == hip_sys::hipSuccess;
        c[dev as usize] = if ok && v != 0 { 1 } else { 2 };
    }
    c[dev as usize] == 1
}

/// `strata::core::device_free_bytes`: `hipMemGetInfo`'s free figure, or on an APU MemAvailable less the UMA headroom
/// when that is larger.
pub fn device_free_bytes() -> u64 {
    let (mut free_b, mut total_b) = (0usize, 0usize);
    unsafe {
        let _ = hip_sys::hipMemGetInfo(&mut free_b, &mut total_b);
    }
    let mut free_b = free_b as u64;
    if device_integrated() {
        if let Some(kb) = meminfo_kb("MemAvailable:") {
            let head = uma_headroom_gib() << 30;
            let avail = kb << 10;
            if avail > head && avail - head > free_b {
                free_b = avail - head;
            }
        }
    }
    free_b
}

#[inline]
pub(super) fn round256(b: u64) -> u64 {
    (b + 255) / 256 * 256
}

/// Inputs of the cache sizing (`tk_qwen4exp_p1_cache_plan_in_t`): the CLI's numbers and what other modules priced.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpP1CachePlanIn {
    pub n_layers: i64,
    pub n_expert: i64,
    /// `--expert-cache` as it stands (-1: auto)
    pub expert_cache: i32,
    pub native_pack: i32,
    pub per_layer: i32,
    pub multi_gpu: i32,
    pub vram_reserve_mib: i32,
    pub vram_reserve_given: i32,
    /// auto only: the prompt path's own buffers (MiB), the draft heads' bind bytes, the cache floor (slots)
    pub prefill_mib: i64,
    pub mtp_bind: i64,
    pub min_slots: i64,
    /// auto only, for the "no VRAM left" message: the session's MiB at `max_context`
    pub session_mib: i64,
    pub max_context: i64,
    /// --pipeline-windows' CUDA0 bytes kept out of the cache
    pub pipe_first: i64,
    pub max_blob: u64,
    /// `n_layers` entries: each layer's expert blob bytes (`expert_layout().blob_bytes(l)`)
    pub blob_bytes: *const u64,
    /// `2 * n_profile` int32: the ranked (layer, expert) pairs
    pub profile: *const i32,
    pub n_profile: usize,
    pub profile_path: *const c_char,
}

/// Result of the cache sizing (`tk_qwen4exp_p1_cache_plan_out_t`); `sized_slots` stays valid until the next plan.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpP1CachePlanOut {
    pub expert_cache: i32,
    pub vram_reserve_mib: i32,
    pub reserve_adapted: i32,
    pub sized_slots: *const i64,
    pub n_sized: usize,
    pub per_layer_bytes: u64,
}

/// The cache sizing, owned by Rust: what `generate.cpp` does from `const bool auto_cache` to the slot list.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CachePlan {
    pub expert_cache: i64,
    pub vram_reserve_mib: i64,
    pub reserve_adapted: bool,
    pub sized_slots: Vec<i64>,
    pub per_layer_bytes: u64,
}

/// The sizing inputs in Rust form (unit-testable without a device).
#[derive(Clone, Debug)]
pub struct CachePlanInput {
    pub n_layers: i64,
    pub n_expert: i64,
    pub expert_cache: i64,
    pub native_pack: bool,
    pub per_layer: bool,
    pub multi_gpu: bool,
    pub vram_reserve_mib: i64,
    pub vram_reserve_given: bool,
    pub prefill_mib: i64,
    pub mtp_bind: i64,
    pub min_slots: i64,
    pub session_mib: i64,
    pub max_context: i64,
    pub pipe_first: i64,
    pub max_blob: u64,
    pub blob_bytes: Vec<u64>,
    pub profile: Vec<(i32, i32)>,
    pub profile_path: String,
}

/// `kSmallReserveMib`: the smallest reserve the auto sizing lowers to (#496).
pub(super) const SMALL_RESERVE_MIB: i64 = 300;

/// The cache sizing (`generate.cpp`, `--expert-cache` auto / the per-layer list / the native-pack profile list).
/// `free_bytes` is read where the C++ reads `device_free_bytes()` (each branch reads it once).
pub fn plan_cache(inp: &CachePlanInput, mut free_bytes: impl FnMut() -> u64) -> CachePlan {
    let auto_cache = inp.expert_cache < 0;
    let mut p = CachePlan {
        expert_cache: inp.expert_cache,
        vram_reserve_mib: inp.vram_reserve_mib,
        ..Default::default()
    };
    let blob = inp.max_blob as i64;
    if p.expert_cache < 0 {
        let free_b = free_bytes() as i64;
        // (the WDDM reserve floor is Windows HIP only)
        let prefill_mib = inp.prefill_mib;
        let mtp_bind = inp.mtp_bind;
        let reserve = ((p.vram_reserve_mib + prefill_mib) << 20) + mtp_bind + inp.pipe_first;
        let mut slots = if blob > 0 {
            (free_b - reserve) / blob
        } else {
            0
        };
        if !inp.profile.is_empty() {
            slots = slots.min(inp.profile.len() as i64);
        }
        p.expert_cache = slots.max(0).min(i32::MAX as i64);
        eprintln!(
            "talus generate: expert cache auto: {:.2} GiB free, {} MiB reserved (+{} MiB for the draft head) -> {} slots",
            free_b as f64 / 1073741824.0,
            p.vram_reserve_mib,
            mtp_bind >> 20,
            p.expert_cache
        );
        if device_integrated() {
            if let Some(kb) = meminfo_kb("MemAvailable:") {
                let host_headroom: i64 = 4 << 30;
                let cache_b = p.expert_cache * blob;
                let avail = (kb << 10) as i64;
                if cache_b > avail - host_headroom {
                    eprintln!(
                        "talus generate: WARNING: this GPU shares system RAM, and the cache above ({:.2} GiB) is more than \
                         the host memory available now ({:.2} GiB) less 4 GiB for the OS: the system may run out of \
                         memory; --expert-cache N sets the cache by hand",
                        cache_b as f64 / 1073741824.0,
                        avail as f64 / 1073741824.0
                    );
                }
            }
        }
        let min_slots = inp.min_slots;
        if p.expert_cache < min_slots
            && !inp.vram_reserve_given
            && p.vram_reserve_mib > SMALL_RESERVE_MIB
        {
            let fit_mib =
                (free_b - mtp_bind - inp.pipe_first - min_slots * blob) / (1 << 20) - prefill_mib;
            if fit_mib >= SMALL_RESERVE_MIB {
                let r = fit_mib.min(p.vram_reserve_mib);
                let mut s2 =
                    (free_b - (((r + prefill_mib) << 20) + mtp_bind + inp.pipe_first)) / blob;
                if !inp.profile.is_empty() {
                    s2 = s2.min(inp.profile.len() as i64);
                }
                eprintln!(
                    "talus generate: expert cache auto: the {} MiB reserve leaves too few slots on this card (a working \
                     cache needs {}): a {} MiB reserve instead -> {} slots",
                    p.vram_reserve_mib, min_slots, r, s2
                );
                p.vram_reserve_mib = r;
                p.expert_cache = s2;
                p.reserve_adapted = true;
            }
        }
        if p.expert_cache == 0 {
            let at_reserve = if inp.vram_reserve_given {
                p.vram_reserve_mib
            } else {
                p.vram_reserve_mib.min(SMALL_RESERVE_MIB)
            };
            let need_b =
                ((at_reserve + prefill_mib) << 20) + mtp_bind + inp.pipe_first + min_slots * blob;
            let short_mib = 1i64.max((need_b - free_b + (1 << 20) - 1) >> 20);
            let reserve_tip = if inp.vram_reserve_given && p.vram_reserve_mib > SMALL_RESERVE_MIB {
                format!(
                    ", a smaller --vram-reserve-mib ({} now; {} is enough on a small card)",
                    p.vram_reserve_mib, SMALL_RESERVE_MIB
                )
            } else {
                String::new()
            };
            eprintln!(
                "talus generate: no VRAM is left for the expert cache: it needs at least {} slots ({} MiB), about {} MiB \
                 more than this card has free. To make room: a smaller --max-context (the session, mostly its KV cache, \
                 takes {} MiB at {} tokens), --kv q4_0, the English draft subset (setup --draft-vocab en; the draft head \
                 takes {} MiB now){}, images on the CPU, or close other programs that use the GPU",
                min_slots,
                (min_slots * blob) >> 20,
                short_mib,
                inp.session_mib,
                inp.max_context,
                mtp_bind >> 20,
                reserve_tip
            );
        }
    } else if inp.multi_gpu && p.expert_cache > 0 {
        let free_b = free_bytes() as i64;
        let reserve = ((p.vram_reserve_mib + inp.prefill_mib) << 20) + inp.pipe_first;
        let fit = ((free_b - reserve) / blob).max(0);
        if p.expert_cache > fit {
            eprintln!(
                "talus generate: WARNING: layer split: --expert-cache {} leaves no room for the prompt path's buffers \
                 ({} MiB) and the {} MiB reserve on CUDA0: {} slots instead (a smaller --vram-reserve-mib leaves more of \
                 them)",
                p.expert_cache, inp.prefill_mib, p.vram_reserve_mib, fit
            );
            p.expert_cache = fit;
        }
    }
    if inp.native_pack && p.expert_cache > 0 && inp.per_layer {
        let free_b = free_bytes();
        let asked = p.expert_cache;
        let budget = asked as u64 * inp.max_blob;
        let keep = (p.vram_reserve_mib as u64) << 20;
        let free_room = free_b.saturating_sub(keep);
        let cap = budget.min(free_room);
        let sum: u64 = (0..inp.n_layers as usize)
            .map(|l| round256(inp.blob_bytes[l]))
            .sum();
        let mut q = if sum > 0 { (cap / sum) as i64 } else { 0 };
        if q > inp.n_expert {
            q = inp.n_expert;
        }
        if q > 0 {
            p.per_layer_bytes = sum;
            p.sized_slots.reserve((q * inp.n_layers) as usize);
            for l in 0..inp.n_layers as usize {
                for _ in 0..q {
                    p.sized_slots.push(inp.blob_bytes[l] as i64);
                }
            }
            p.expert_cache = p.sized_slots.len() as i64;
            eprintln!(
                "talus generate: per-layer slots use each layer's blob: {} uniform slots ({:.2} GiB) -> {} slots, {} per \
                 layer ({:.2} GiB)",
                asked,
                budget as f64 / 1073741824.0,
                p.expert_cache,
                q,
                (q as u64 * sum) as f64 / 1073741824.0
            );
        }
    } else if inp.native_pack && p.expert_cache > 0 && !inp.profile.is_empty() {
        let free_b = free_bytes();
        let budget = p.expert_cache as u64 * round256(inp.max_blob);
        let mut used = 0u64;
        let keep_free = ((p.vram_reserve_mib as u64) << 20) + inp.pipe_first as u64;
        let free_room = free_b.saturating_sub(keep_free);
        let cap = budget.min(free_room);
        for &(l, _e) in &inp.profile {
            let b = round256(inp.blob_bytes[l as usize]);
            if used + b > cap {
                break;
            }
            used += b;
            p.sized_slots.push(inp.blob_bytes[l as usize] as i64);
        }
        if !auto_cache
            && p.expert_cache > 0
            && p.sized_slots.len() == inp.profile.len()
            && p.expert_cache as usize > inp.profile.len()
        {
            eprintln!(
                "talus generate: WARNING: --expert-cache {} is more than the {} pairs of the profile {}: {} slots (the \
                 profile is the ceiling on a native pack; a profile that ranks every (layer, expert) pair lifts it - \
                 tools/make_profile.py)",
                p.expert_cache,
                inp.profile.len(),
                inp.profile_path,
                inp.profile.len()
            );
        }
        p.expert_cache = p.sized_slots.len() as i64;
    }
    p
}

/// `fnv1a64` (pinned.hpp): FNV-1a with the standard basis, continuing from `seed`.
pub fn fnv1a64(bytes: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
pub const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// The seed `strata::core::fnv1a64` defaults to (pinned.hpp): 1469598103934665603, one digit short of the FNV-1a
/// basis 14695981039346656037 - the expert cache checksum is defined with it, so it is kept.
pub const STRATA_FNV_SEED: u64 = 1_469_598_103_934_665_603;

/// The VRAM expert tier (`strata::core::ExpertCache`, the non-segmented HIP form): one `hipMalloc` arena of uniform or
/// sized slots, the `(layer, expert) -> slot` table, admission in arrival order (or per layer), blocking fills through
/// a pinned staging buffer, and the read-back check.
pub struct ExpertCacheArena {
    pub base: *mut u8,
    pub slots: i64,
    pub n_layers: i64,
    pub n_expert: i64,
    pub blob: i64,
    /// sized slots: the `slots + 1` byte offsets (empty: uniform `blob` slots)
    pub off: Vec<u64>,
    pub residency: Vec<i32>,
    pub next_free: i64,
    pub fills: i64,
    pub admitted: i64,
    pub per_layer: bool,
    pub layer_next: Vec<i32>,
    staging: *mut u8,
    staging_bytes: usize,
    /// false once the arena has been handed to the C++ `ExpertCache` (it frees it)
    pub(super) owned: bool,
}

unsafe impl Send for ExpertCacheArena {}

impl Drop for ExpertCacheArena {
    fn drop(&mut self) {
        self.close();
    }
}

impl ExpertCacheArena {
    pub(super) fn empty() -> Self {
        Self {
            base: std::ptr::null_mut(),
            slots: 0,
            n_layers: 0,
            n_expert: 0,
            blob: 0,
            off: Vec::new(),
            residency: Vec::new(),
            next_free: 0,
            fills: 0,
            admitted: 0,
            per_layer: false,
            layer_next: Vec::new(),
            staging: std::ptr::null_mut(),
            staging_bytes: 0,
            owned: true,
        }
    }

    /// `ExpertCache::close`.
    pub fn close(&mut self) {
        unsafe {
            if !self.staging.is_null() {
                let _ = hip_sys::hipHostFree(self.staging.cast());
            }
            if self.owned && !self.base.is_null() {
                let _ = hip_sys::hipFree(self.base.cast());
            }
        }
        self.staging = std::ptr::null_mut();
        self.staging_bytes = 0;
        self.base = std::ptr::null_mut();
        self.off.clear();
        self.residency.clear();
        self.slots = 0;
        self.blob = 0;
        self.next_free = 0;
        self.fills = 0;
        self.admitted = 0;
        self.layer_next.clear();
    }

    fn ensure_staging(&mut self, bytes: usize) -> Result<(), String> {
        if bytes <= self.staging_bytes {
            return Ok(());
        }
        let mut p: *mut c_void = std::ptr::null_mut();
        let e = unsafe { hip_sys::hipHostMalloc(&mut p, bytes, hip_sys::hipHostMallocDefault) };
        if e != hip_sys::hipSuccess || p.is_null() {
            return Err("ExpertCache: HIP blocking staging allocation failed".into());
        }
        if !self.staging.is_null() {
            unsafe {
                let _ = hip_sys::hipHostFree(self.staging.cast());
            }
        }
        self.staging = p.cast();
        self.staging_bytes = bytes;
        Ok(())
    }

    /// `ExpertCache::open`: `n_slots * blob_bytes` of device memory, checked against what is free, zeroed.
    pub fn open(
        n_slots: i64,
        n_layers: i64,
        n_expert: i64,
        blob_bytes: i64,
    ) -> Result<Self, String> {
        if n_slots <= 0 {
            return Err("ExpertCache: n_slots must be positive".into());
        }
        if n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0 {
            return Err(
                "ExpertCache: n_layers, n_expert and blob_bytes must all be positive".into(),
            );
        }
        let want = n_slots as u64 * blob_bytes as u64;
        let (mut fb, mut tb) = (0usize, 0usize);
        if unsafe { hip_sys::hipMemGetInfo(&mut fb, &mut tb) } == hip_sys::hipSuccess {
            let free_b = device_free_bytes();
            if free_b < want {
                return Err(format!(
                    "ExpertCache: {} slots x {} B = {:.2} GiB, but only {:.2} GiB of VRAM is free ({:.2} GiB of {:.2} GiB \
                     total). Lower --expert-cache.",
                    n_slots,
                    blob_bytes,
                    want as f64 / 1073741824.0,
                    free_b as f64 / 1073741824.0,
                    (tb as f64 - free_b as f64) / 1073741824.0,
                    tb as f64 / 1073741824.0
                ));
            }
        }
        let mut c = Self::empty();
        let mut p: *mut c_void = std::ptr::null_mut();
        if unsafe { hip_sys::hipMalloc(&mut p, want as usize) } != hip_sys::hipSuccess
            || p.is_null()
        {
            let msg = unsafe {
                let e = hip_sys::hipGetLastError();
                let s = hip_sys::hipGetErrorString(e);
                if s.is_null() {
                    "unknown".to_string()
                } else {
                    CStr::from_ptr(s).to_string_lossy().into_owned()
                }
            };
            return Err(format!(
                "ExpertCache: cudaMalloc({:.2} GiB) failed: {msg}",
                want as f64 / 1073741824.0
            ));
        }
        c.base = p.cast();
        if unsafe { hip_sys::hipMemset(p, 0, want as usize) } != hip_sys::hipSuccess {
            return Err("ExpertCache: cudaMemset of the slot arena failed".into());
        }
        c.residency = vec![NOT_RESIDENT; (n_layers * n_expert) as usize];
        c.slots = n_slots;
        c.n_layers = n_layers;
        c.n_expert = n_expert;
        c.blob = blob_bytes;
        c.ensure_staging(c.blob as usize)?;
        c.reset_layer_next();
        Ok(c)
    }

    /// `ExpertCache::open_sized`: slots of the given sizes back to back, each rounded up to 256 B.
    pub fn open_sized(slot_bytes: &[i64], n_layers: i64, n_expert: i64) -> Result<Self, String> {
        if slot_bytes.is_empty() {
            return Err("ExpertCache: no slots".into());
        }
        let mut off = vec![0u64; slot_bytes.len() + 1];
        let mut mx = 0i64;
        for (i, &b) in slot_bytes.iter().enumerate() {
            off[i + 1] = off[i] + round256(b as u64);
            mx = mx.max(b);
        }
        let mut c = Self::open(*off.last().unwrap() as i64, n_layers, n_expert, 1)?;
        c.slots = slot_bytes.len() as i64;
        c.blob = mx;
        c.off = off;
        c.ensure_staging(c.blob as usize)?;
        c.reset_layer_next();
        Ok(c)
    }

    pub(super) fn reset_layer_next(&mut self) {
        self.layer_next = vec![0; self.n_layers.max(0) as usize];
        for l in 0..self.n_layers {
            let (lo, _) = self.layer_slot_range(l);
            self.layer_next[l as usize] = lo as i32;
        }
    }

    /// `ExpertCache::layer_slot_range`.
    pub fn layer_slot_range(&self, layer: i64) -> (i64, i64) {
        if self.n_layers <= 0 || self.slots <= 0 || layer < 0 || layer >= self.n_layers {
            return (0, 0);
        }
        let q = self.slots / self.n_layers;
        let lo = layer * q;
        let hi = if layer == self.n_layers - 1 {
            self.slots
        } else {
            (layer + 1) * q
        };
        (lo, hi)
    }

    /// The bytes of all slots (`ExpertCache::bytes`).
    pub fn bytes(&self) -> i64 {
        if self.off.is_empty() {
            self.slots * self.blob
        } else {
            self.off[self.slots as usize] as i64
        }
    }

    pub fn slot_of(&self, layer: i64, expert: i64) -> i32 {
        if layer < 0 || layer >= self.n_layers || expert < 0 || expert >= self.n_expert {
            return NOT_RESIDENT;
        }
        self.residency[(layer * self.n_expert + expert) as usize]
    }

    /// `ExpertCache::admit`: the next free slot (or the layer's next under per-layer admission); never evicts.
    pub fn admit(&mut self, layer: i64, expert: i64) -> i32 {
        if layer < 0 || layer >= self.n_layers || expert < 0 || expert >= self.n_expert {
            return NOT_RESIDENT;
        }
        let at = (layer * self.n_expert + expert) as usize;
        if self.residency[at] != NOT_RESIDENT {
            return self.residency[at];
        }
        if self.per_layer {
            if self.layer_next.is_empty() {
                return NOT_RESIDENT;
            }
            let (_, hi) = self.layer_slot_range(layer);
            if self.layer_next[layer as usize] as i64 >= hi {
                return NOT_RESIDENT;
            }
            self.residency[at] = self.layer_next[layer as usize];
            self.layer_next[layer as usize] += 1;
            self.admitted += 1;
            return self.residency[at];
        }
        if self.next_free >= self.slots {
            return NOT_RESIDENT;
        }
        self.residency[at] = self.next_free as i32;
        self.next_free += 1;
        (self.next_free - 1) as i32
    }

    pub fn device_slot(&self, slot: i32) -> *mut u8 {
        if slot < 0 || slot as i64 >= self.slots {
            return std::ptr::null_mut();
        }
        let o = if self.off.is_empty() {
            slot as u64 * self.blob as u64
        } else {
            self.off[slot as usize]
        };
        unsafe { self.base.add(o as usize) }
    }

    fn slot_len(&self, bytes: i64) -> usize {
        (if bytes > 0 && bytes <= self.blob {
            bytes
        } else {
            self.blob
        }) as usize
    }

    /// `ExpertCache::fill_slot_blocking` (HIP form): the blob through the pinned staging buffer, one blocking copy.
    pub fn fill_slot_blocking(
        &mut self,
        slot: i32,
        host_blob: *const u8,
        bytes: i64,
    ) -> Result<(), String> {
        let n = self.slot_len(bytes);
        let dst = self.device_slot(slot);
        if dst.is_null() {
            return Err("ExpertCache::fill_slot_blocking: slot outside the arena".into());
        }
        if host_blob.is_null() {
            return Err("ExpertCache::fill_slot_blocking: the host blob is null".into());
        }
        if self.staging.is_null() || n > self.staging_bytes {
            return Err("ExpertCache::fill_slot_blocking: HIP staging buffer is too small".into());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(host_blob, self.staging, n);
            if hip_sys::hipMemcpy(
                dst.cast(),
                self.staging.cast(),
                n,
                hip_sys::hipMemcpyHostToDevice,
            ) != hip_sys::hipSuccess
            {
                return Err("ExpertCache::fill_slot_blocking: hipMemcpy failed".into());
            }
        }
        self.fills += 1;
        Ok(())
    }

    fn read_back(&self, slot: i32, n: usize) -> Result<Vec<u8>, String> {
        let src = self.device_slot(slot);
        if src.is_null() {
            return Err("ExpertCache::verify_slot: slot outside the arena".into());
        }
        let mut got = vec![0u8; n];
        if unsafe {
            hip_sys::hipMemcpy(
                got.as_mut_ptr().cast(),
                src.cast(),
                n,
                hip_sys::hipMemcpyDeviceToHost,
            )
        } != hip_sys::hipSuccess
        {
            return Err("ExpertCache::verify_slot: hipMemcpy failed".into());
        }
        Ok(got)
    }

    /// `ExpertCache::verify_slot`: the slot read back and compared byte for byte with the host blob.
    pub fn verify_slot(&self, slot: i32, host_blob: *const u8, bytes: i64) -> Result<(), String> {
        let nb = self.slot_len(bytes);
        let got = self.read_back(slot, nb)?;
        let want = unsafe { std::slice::from_raw_parts(host_blob, nb) };
        if got != want {
            let first = got.iter().zip(want).take_while(|(a, b)| a == b).count();
            return Err(format!(
                "ExpertCache::verify_slot: slot {slot} differs from the arena at byte {first} (of {})",
                self.blob
            ));
        }
        Ok(())
    }
}

/// `tk_qwen4exp_p1_blob_fn`: the host address of `(layer, expert)`'s blob in the expert source (`blob_stable`).
pub type FnP1Blob = unsafe extern "C" fn(ctx: *mut c_void, layer: i32, expert: i32) -> *const u8;

/// Inputs of the cache build (`tk_qwen4exp_p1_cache_build_in_t`): the planned size and the profile fill.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpP1CacheBuildIn {
    pub n_layers: i64,
    pub n_expert: i64,
    pub expert_cache: i32,
    pub auto_cache: i32,
    pub per_layer: i32,
    pub vram_reserve_mib: i32,
    pub pipe_first: i64,
    pub max_blob: u64,
    pub blob_bytes: *const u64,
    pub sized_slots: *const i64,
    pub n_sized: usize,
    pub per_layer_bytes: u64,
    pub profile: *const i32,
    pub n_profile: usize,
    pub blob_fn: Option<FnP1Blob>,
    pub blob_ctx: *mut c_void,
    pub file_source: i32,
    pub unbuffered: i32,
    pub advise_pairs:
        Option<unsafe extern "C" fn(ctx: *mut c_void, pairs: *const i32, n: i64) -> c_int>,
    pub prefetch_pairs: Option<unsafe extern "C" fn(ctx: *mut c_void, pairs: *const i32, n: i64)>,
    pub pin_aux_thread: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
}

/// The mapped file tier's read-ahead during the profile fill (`FileExpertSource::advise_pairs` / `prefetch_pairs`).
pub struct FileTierFill {
    pub unbuffered: bool,
    pub ctx: usize,
    pub advise_pairs: unsafe extern "C" fn(ctx: *mut c_void, pairs: *const i32, n: i64) -> c_int,
    pub prefetch_pairs: unsafe extern "C" fn(ctx: *mut c_void, pairs: *const i32, n: i64),
    pub pin_aux_thread: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
}

/// The built cache, handed to `ExpertCache::adopt` (the C++ object then owns and frees the arena). The arrays stay
/// valid until the next build.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpP1CacheBuildOut {
    pub base: *mut u8,
    pub slots: i64,
    pub blob: i64,
    pub offsets: *const u64,
    pub n_offsets: usize,
    pub residency: *const i32,
    pub n_residency: usize,
    pub layer_next: *const i32,
    pub n_layer_next: usize,
    pub next_free: i64,
    pub admitted: i64,
    pub fills: i64,
    pub prefilled: i64,
    pub expert_cache: i32,
    pub sized_changed: i32,
}

/// What the cache build did, in Rust form.
pub struct CacheBuild {
    pub cache: Option<ExpertCacheArena>,
    pub expert_cache: i64,
    pub sized_slots: Vec<i64>,
    pub prefilled: i64,
}

/// `generate.cpp` from `if (o.expert_cache > 0)` (open, shrinking an auto cache that fails or leaves the reserve
/// short) through the profile fill, its slot-0 check and the STRATA_VERIFY_COMPLEMENT checksum.
#[allow(clippy::too_many_arguments)]
pub fn build_cache(
    n_layers: i64,
    n_expert: i64,
    mut expert_cache: i64,
    auto_cache: bool,
    per_layer: bool,
    vram_reserve_mib: i64,
    pipe_first: i64,
    max_blob: u64,
    blob_bytes: &[u64],
    mut sized_slots: Vec<i64>,
    per_layer_bytes: u64,
    profile: &[(i32, i32)],
    blob: &mut dyn FnMut(i32, i32) -> *const u8,
    file_tier: Option<&FileTierFill>,
) -> Result<CacheBuild, String> {
    let mut cache: Option<ExpertCacheArena> = None;
    if expert_cache > 0 {
        // keep the first `keep_bytes` of the cache (the profile's hottest experts first); false when nothing is left
        let shrink_to = |keep_bytes: i64, expert_cache: &mut i64, sized: &mut Vec<i64>| -> bool {
            if keep_bytes <= 0 {
                *expert_cache = 0;
                sized.clear();
                return false;
            }
            if per_layer_bytes > 0 {
                let mut q = keep_bytes / per_layer_bytes as i64;
                if q > n_expert {
                    q = n_expert;
                }
                sized.clear();
                if q <= 0 {
                    *expert_cache = 0;
                    return false;
                }
                for l in 0..n_layers as usize {
                    for _ in 0..q {
                        sized.push(blob_bytes[l] as i64);
                    }
                }
                *expert_cache = sized.len() as i64;
                return true;
            }
            if !sized.is_empty() {
                let mut used = 0i64;
                let mut keep = 0usize;
                while keep < sized.len() && used + round256(sized[keep] as u64) as i64 <= keep_bytes
                {
                    used += round256(sized[keep] as u64) as i64;
                    keep += 1;
                }
                sized.truncate(keep);
                *expert_cache = keep as i64;
            } else {
                *expert_cache = keep_bytes / max_blob as i64;
            }
            if *expert_cache <= 0 {
                *expert_cache = 0;
                sized.clear();
                return false;
            }
            true
        };
        let cache_bytes = |expert_cache: i64, sized: &Vec<i64>| -> i64 {
            if sized.is_empty() {
                expert_cache * max_blob as i64
            } else {
                sized.iter().map(|&s| round256(s as u64) as i64).sum()
            }
        };
        let mut fake_fails: i64 = std::env::var("STRATA_TEST_CACHE_FAIL")
            .ok()
            .map(|v| v.trim().parse().unwrap_or(0))
            .unwrap_or(0);
        let mut failed = 0i64;
        let mut zero_reads = 0i64;
        let mut attempt = 0i64;
        loop {
            let opened: Result<ExpertCacheArena, String> = if fake_fails > 0 {
                fake_fails -= 1;
                Err("ExpertCache: cudaMalloc failed: out of memory (STRATA_TEST_CACHE_FAIL)".into())
            } else if sized_slots.is_empty() {
                ExpertCacheArena::open(expert_cache, n_layers, n_expert, max_blob as i64)
            } else {
                ExpertCacheArena::open_sized(&sized_slots, n_layers, n_expert)
            };
            match opened {
                Err(err) => {
                    unsafe {
                        let _ = hip_sys::hipGetLastError();
                    }
                    if auto_cache && failed < 8 {
                        let keep = cache_bytes(expert_cache, &sized_slots) / 4 * 3;
                        if shrink_to(keep, &mut expert_cache, &mut sized_slots) {
                            failed += 1;
                            eprintln!(
                                "talus generate: {err}; trying a smaller expert cache: {expert_cache} slots"
                            );
                            attempt += 1;
                            continue;
                        }
                    }
                    return Err(err);
                }
                Ok(c) => cache = Some(c),
            }
            if !auto_cache || attempt - failed >= 6 {
                break;
            }
            let c = cache.as_ref().unwrap();
            unsafe {
                let _ = hip_sys::hipMemset(c.device_slot(0).cast(), 0, c.bytes() as usize);
                let _ = hip_sys::hipDeviceSynchronize();
            }
            let free_b = device_free_bytes() as i64;
            let want = (vram_reserve_mib << 20) + pipe_first;
            if free_b >= want - (64 << 20) {
                break;
            }
            let mut give = want - free_b + (64 << 20);
            if free_b < (16 << 20) {
                zero_reads += 1;
                give = give.max(if zero_reads <= 2 {
                    1 << 30
                } else {
                    c.bytes() / 4
                });
            }
            let keep_bytes = c.bytes() - give;
            eprintln!(
                "talus generate: only {} MiB free once the slots are written (reserve {} MiB); shrinking the expert cache",
                free_b >> 20,
                vram_reserve_mib
            );
            cache = None; // close
            if !shrink_to(keep_bytes, &mut expert_cache, &mut sized_slots) {
                break;
            }
            attempt += 1;
        }
        if failed > 0 && expert_cache > 0 {
            eprintln!(
                "talus generate: expert cache: {} slots ({:.2} GiB) after {} smaller tries - a bigger page file lets it \
                 use more of the free VRAM",
                expert_cache,
                cache.as_ref().map(|c| c.bytes()).unwrap_or(0) as f64 / 1073741824.0,
                failed
            );
        }
    }
    let mut prefilled = 0i64;
    if let Some(c) = cache.as_mut() {
        c.per_layer = per_layer;
        if !profile.is_empty() {
            let want = if per_layer {
                profile.len() as i64
            } else {
                (profile.len() as i64).min(c.slots)
            };
            // the flat (layer, expert) int32 pairs the file tier's read-ahead takes
            let flat: Vec<i32> = profile.iter().flat_map(|&(l, e)| [l, e]).collect();
            let ft = file_tier.filter(|_| !per_layer);
            let mut fill_ahead = 0i64;
            if let Some(f) = ft.filter(|f| !f.unbuffered) {
                fill_ahead = match std::env::var("STRATA_FILL_AHEAD") {
                    Err(_) => 256,
                    Ok(v) => v.trim().parse::<i64>().unwrap_or(0).max(0),
                };
                if fill_ahead > 0
                    && unsafe {
                        (f.advise_pairs)(f.ctx as *mut c_void, flat.as_ptr(), fill_ahead.min(want))
                    } == 0
                {
                    fill_ahead = 0;
                }
                eprintln!("talus generate: the profile fill asks for {fill_ahead} pairs ahead (STRATA_FILL_AHEAD)");
            }
            // unbuffered file tier: the pairs read in batches of 64, the next batch while this one is copied
            let read_batch = |f: &FileTierFill, at: i64| unsafe {
                (f.prefetch_pairs)(
                    f.ctx as *mut c_void,
                    flat.as_ptr().add(at as usize * 2),
                    64.min(want - at),
                );
            };
            let mut ahead: Option<std::thread::JoinHandle<()>> = None;
            let t0 = std::time::Instant::now();
            let mut fill_bytes = 0u64;
            for i in 0..want as usize {
                if let Some(f) = ft.filter(|f| f.unbuffered) {
                    if i % 64 == 0 {
                        match ahead.take() {
                            Some(h) => {
                                let _ = h.join();
                            }
                            None => read_batch(f, i as i64),
                        }
                        if (i as i64) + 64 < want {
                            let (ctx, pf, pin) = (f.ctx, f.prefetch_pairs, f.pin_aux_thread);
                            let base = flat.as_ptr() as usize;
                            let (at, n) = (i as i64 + 64, 64.min(want - (i as i64 + 64)));
                            ahead = Some(std::thread::spawn(move || unsafe {
                                if let Some(p) = pin {
                                    p(ctx as *mut c_void);
                                }
                                pf(
                                    ctx as *mut c_void,
                                    (base as *const i32).add(at as usize * 2),
                                    n,
                                );
                            }));
                        }
                    }
                }
                if fill_ahead > 0 && (i as i64) + fill_ahead < want {
                    if let Some(f) = ft {
                        unsafe {
                            (f.advise_pairs)(
                                f.ctx as *mut c_void,
                                flat.as_ptr().add((i + fill_ahead as usize) * 2),
                                1,
                            )
                        };
                    }
                }
                let (l, e) = profile[i];
                let slot = c.admit(l as i64, e as i64);
                if slot == NOT_RESIDENT {
                    if per_layer {
                        continue;
                    }
                    break;
                }
                let b = blob(l, e);
                let nb = blob_bytes[l as usize] as i64;
                if b.is_null() {
                    return Err(format!(
                        "the profile fill failed at pair {i}: ExpertCache::fill_slot_blocking: the host blob is null"
                    ));
                }
                c.fill_slot_blocking(slot, b, nb)
                    .map_err(|e| format!("the profile fill failed at pair {i}: {e}"))?;
                fill_bytes += nb as u64;
                prefilled += 1;
            }
            if let Some(h) = ahead.take() {
                let _ = h.join();
            }
            let fill_s = t0.elapsed().as_secs_f64();
            if prefilled > 0 {
                let (l0, e0) = profile[0];
                c.verify_slot(
                    c.slot_of(l0 as i64, e0 as i64),
                    blob(l0, e0),
                    blob_bytes[l0 as usize] as i64,
                )?;
            }
            eprintln!(
                "talus generate: pre-filled {} of {} slots from the profile in {:.1} s ({:.0} MB/s); slot 0 verified \
                 (qwen4exp-p1: Rust)",
                prefilled,
                if per_layer { c.slots } else { want },
                fill_s,
                if fill_s > 0.0 {
                    fill_bytes as f64 / 1e6 / fill_s
                } else {
                    0.0
                }
            );
            if std::env::var_os("STRATA_VERIFY_COMPLEMENT").is_some() {
                let mut sl: Vec<(i32, i64)> = Vec::new();
                for l in 0..n_layers {
                    for e in 0..n_expert {
                        let s = c.slot_of(l, e);
                        if s >= 0 {
                            sl.push((s, blob_bytes[l as usize] as i64));
                        }
                    }
                }
                sl.sort();
                // each slot read back and hashed on 8 threads (as the C++), then the hashes in slot order
                let addrs: Vec<(usize, usize)> = sl
                    .iter()
                    .map(|&(s, nb)| (c.device_slot(s) as usize, nb as usize))
                    .collect();
                let mut sh = vec![0u64; sl.len()];
                let bad = std::sync::atomic::AtomicBool::new(false);
                let per = sl.len().div_ceil(8).max(1);
                std::thread::scope(|sc| {
                    for (ci, out) in sh.chunks_mut(per).enumerate() {
                        let addrs = &addrs;
                        let bad = &bad;
                        sc.spawn(move || {
                            let mut hb = Vec::new();
                            for (k, o) in out.iter_mut().enumerate() {
                                let (a, n) = addrs[ci * per + k];
                                hb.resize(n, 0u8);
                                let rc = unsafe {
                                    hip_sys::hipMemcpy(
                                        hb.as_mut_ptr().cast(),
                                        a as *const c_void,
                                        n,
                                        hip_sys::hipMemcpyDeviceToHost,
                                    )
                                };
                                if rc != hip_sys::hipSuccess {
                                    bad.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                                *o = fnv1a64(&hb, STRATA_FNV_SEED);
                            }
                        });
                    }
                });
                let ok = !bad.load(std::sync::atomic::Ordering::Relaxed);
                let mut h = STRATA_FNV_SEED;
                for (idx, &(s, nb)) in sl.iter().enumerate() {
                    let hs = sh[idx];
                    let mut v = [0u8; 16];
                    v[..8].copy_from_slice(&(s as i64).to_le_bytes());
                    v[8..].copy_from_slice(&nb.to_le_bytes());
                    h = fnv1a64(&v, h);
                    h = fnv1a64(&hs.to_le_bytes(), h);
                }
                eprintln!(
                    "talus generate: expert cache checksum {h:016x} ({} slots{})",
                    sl.len(),
                    if ok { "" } else { ", A READ-BACK FAILED" }
                );
            }
        }
    }
    Ok(CacheBuild {
        cache,
        expert_cache,
        sized_slots,
        prefilled,
    })
}
