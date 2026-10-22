//! W4A4 prompt GEMMs for the dense 27B FFN (`src/prefill/tp_w4i.cu`: iu4 WMMA, 32-value activation groups
//! with integer scales, int4 weights from a GPTQ sidecar in the rotated basis).  Numerics change: quality-gated
//! (release default: FFN of layers 48-63, gate = KL vs llama.cpp CPU on 4K prompts <= x1.35 of the default, needle,
//! agentic; docs/METHODOLOGY.md).  Prefill chunks only (T > gemm_t and T >= `TALUS_Q27_W4I_MIN`, default 64);
//! decode / verify / MTP stay on the exact paths.
//!
//! `TALUS_Q27_W4I`:
//!   unset / `0` / `off` - off (exact int8 prompt path; the release default since 2026-10-28: on a 2K prompt the
//!                    changed prompt numerics lower draft acceptance, which outweighs the prompt
//!                    gain for multi-turn use);
//!   `auto` / `on`  - opt-in: the sidecar next to the GGUF (`<stem>.w4i`, e.g. Qwen3.8-27B-UD-IQ4_XS.w4i), planes
//!                    ffn:48-63 (gated; a missing sidecar logs one line and the prompt path stays exact);
//!   `fast`         - `<stem>.fast.w4i`, planes ffn:0-63 (opt-in "fast" profile: FAILED the quality gate);
//!   a path         - that sidecar, every FFN matrix it holds.
//! `TALUS_Q27_W4I_PLANES` overrides the planes: comma (or '+') list of `in:A-B` (or `in_A-B`) (ffn_gate + ffn_up of
//! layers A..B), `dn:A-B` (ffn_down), `ffn:A-B` (both), single layers `in:7`.
//!
//! Sidecar format (`tools/w4i/w4i_convert.py`): little-endian
//!   "W4I1" u32 version (1 | 2) u32 n_mats u32 0 | v2: { src_name[96] u64 src_size u64 0 u64 0 u64 0 } |
//!   n_mats x { name[64] u32 n u32 k u32 wg u32 0 u64 off_wq u64 off_ws u64 off_signs } |
//!   data: wq [n/16][k/32][16][16 B], ws f32 [n] (wg 0) or [n][k/256] (wg 256), signs i8 [k]
//! v2's src_size must equal the loaded GGUF's file size (else: not used).
use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ptr;

use super::ffi::{hipFree, hipMemcpy, HIP_MEMCPY_HOST_TO_DEVICE};
use super::weights::hip_malloc_bytes;

#[derive(Debug, Clone, Copy)]
pub struct W4iMat {
    pub wq: *mut u8,
    pub ws: *mut f32,
    pub signs: *mut i8,
    pub n: i32,
    pub k: i32,
    pub wg: i32,
}

#[derive(Default)]
pub struct W4iCache {
    pub mats: HashMap<usize, W4iMat>,
    pub bytes: usize,
    pub min_rows: i32,
    /// activation buffers for max_batch rows x the widest K
    pub xq: *mut u8,
    pub xk: *mut u8,
    pub xs: *mut f32,
    pub cap_rows: i32,
    pub cap_k: i32,
    pub budget_guard: Option<talus_mem::ReservationGuard>,
}

unsafe impl Send for W4iCache {}
unsafe impl Sync for W4iCache {}

impl W4iCache {
    #[inline]
    pub fn find(&self, w_ptr: *const u8) -> Option<W4iMat> {
        if self.mats.is_empty() {
            return None;
        }
        self.mats.get(&(w_ptr as usize)).copied()
    }

    pub fn clear(&mut self) {
        for (_, m) in self.mats.drain() {
            unsafe {
                let _ = hipFree(m.wq.cast::<c_void>());
                let _ = hipFree(m.ws.cast::<c_void>());
                let _ = hipFree(m.signs.cast::<c_void>());
            }
        }
        for p in [
            self.xq.cast::<c_void>(),
            self.xk.cast::<c_void>(),
            self.xs.cast::<c_void>(),
        ] {
            if !p.is_null() {
                unsafe {
                    let _ = hipFree(p);
                }
            }
        }
        self.xq = ptr::null_mut();
        self.xk = ptr::null_mut();
        self.xs = ptr::null_mut();
        self.bytes = 0;
        self.budget_guard = None;
    }
}

impl Drop for W4iCache {
    fn drop(&mut self) {
        self.clear();
    }
}

/// which (layer, plane) pairs `TALUS_Q27_W4I_PLANES` selects; plane 0 = ffn_gate / ffn_up, 1 = ffn_down
pub fn plane_selected(spec: &str, layer: usize, plane: usize) -> bool {
    // items separated by ',' or '+', kind and range by ':' or '_' (job runners that split on ':' / ',')
    for item in spec
        .split([',', '+'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let (kind, range) = item.split_once([':', '_']).unwrap_or((item, "0-9999"));
        let hit_kind = match kind {
            "in" => plane == 0,
            "dn" => plane == 1,
            "ffn" | "all" => true,
            _ => false,
        };
        if !hit_kind {
            continue;
        }
        let (a, b) = match range.split_once('-') {
            Some((a, b)) => (
                a.parse::<usize>().unwrap_or(usize::MAX),
                b.parse::<usize>().unwrap_or(0),
            ),
            None => {
                let v = range.parse::<usize>().unwrap_or(usize::MAX);
                (v, v)
            }
        };
        if layer >= a && layer <= b {
            return true;
        }
    }
    false
}

pub struct W4iEntry {
    pub name: String,
    pub n: u32,
    pub k: u32,
    pub wg: u32,
    pub off_wq: u64,
    pub off_ws: u64,
    pub off_signs: u64,
}

/// What a sidecar was built from (format v2; v1 files have none).
#[derive(Debug, Clone, Default)]
pub struct W4iSource {
    pub name: String,
    pub size: u64,
}

/// the sidecar's matrix index and (v2) its source GGUF
pub fn read_index(f: &mut File) -> Result<(Vec<W4iEntry>, Option<W4iSource>), String> {
    let mut h = [0u8; 16];
    f.read_exact(&mut h).map_err(|e| e.to_string())?;
    if &h[0..4] != b"W4I1" {
        return Err("not a W4I1 sidecar".into());
    }
    let ver = u32::from_le_bytes(h[4..8].try_into().unwrap());
    let n = u32::from_le_bytes(h[8..12].try_into().unwrap()) as usize;
    let mut src = None;
    if ver >= 2 {
        let mut m = [0u8; 128];
        f.read_exact(&mut m).map_err(|e| e.to_string())?;
        let end = m[..96].iter().position(|&c| c == 0).unwrap_or(96);
        src = Some(W4iSource {
            name: String::from_utf8_lossy(&m[..end]).into_owned(),
            size: u64::from_le_bytes(m[96..104].try_into().unwrap()),
        });
    } else if ver != 1 {
        return Err(format!("unknown W4I1 version {ver}"));
    }
    let mut out = Vec::with_capacity(n);
    let mut e = [0u8; 104];
    for _ in 0..n {
        f.read_exact(&mut e).map_err(|e| e.to_string())?;
        let name_end = e[..64].iter().position(|&c| c == 0).unwrap_or(64);
        let u32at = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap());
        let u64at = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap());
        out.push(W4iEntry {
            name: String::from_utf8_lossy(&e[..name_end]).into_owned(),
            n: u32at(64),
            k: u32at(68),
            wg: u32at(72),
            off_wq: u64at(80),
            off_ws: u64at(88),
            off_signs: u64at(96),
        });
    }
    Ok((out, src))
}

/// `TALUS_Q27_W4I` resolved against the model file: (sidecar path, default planes, mode label), or None (off).
pub fn resolve(
    env: Option<&str>,
    gguf: Option<&std::path::Path>,
) -> Option<(std::path::PathBuf, &'static str, &'static str)> {
    let v = env.map(str::trim).unwrap_or("0");
    let next_to = |suffix: &str| -> Option<std::path::PathBuf> {
        let g = gguf?;
        let stem = g.file_stem()?.to_string_lossy().into_owned();
        Some(g.with_file_name(format!("{stem}{suffix}")))
    };
    match v {
        "" | "0" | "off" | "OFF" | "none" => None,
        "auto" | "1" | "on" => next_to(".w4i").map(|p| (p, "ffn:48-63", "auto")),
        "fast" => next_to(".fast.w4i").map(|p| (p, "ffn:0-63", "fast")),
        p => Some((std::path::PathBuf::from(p), "ffn:0-9999", "path")),
    }
}

/// upload `bytes` at `off` of the file into a fresh device buffer
pub fn upload(f: &mut File, off: u64, bytes: usize, host: &mut Vec<u8>) -> Result<*mut u8, String> {
    host.resize(bytes, 0);
    f.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
    f.read_exact(&mut host[..]).map_err(|e| e.to_string())?;
    let d = hip_malloc_bytes(bytes).map_err(|e| format!("{e}"))?;
    let rc = unsafe {
        hipMemcpy(
            d.cast::<c_void>(),
            host.as_ptr().cast::<c_void>(),
            bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
    };
    if rc != super::ffi::hipSuccess {
        unsafe {
            let _ = hipFree(d.cast::<c_void>());
        }
        return Err(format!("hipMemcpy {rc:?}"));
    }
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::plane_selected;
    #[test]
    fn plane_specs() {
        assert!(
            plane_selected("ffn:48-63", 48, 0)
                && plane_selected("ffn:48-63", 63, 1)
                && !plane_selected("ffn:48-63", 47, 0)
        );
        assert!(
            plane_selected("ffn_48-63+in_40-47", 40, 0)
                && !plane_selected("ffn_48-63+in_40-47", 40, 1)
        );
        assert!(plane_selected("dn:7", 7, 1) && !plane_selected("dn:7", 7, 0));
    }
    #[test]
    fn resolve_modes() {
        use super::resolve;
        use std::path::Path;
        let g = Path::new("/m/Qwen3.8-27B-UD-IQ4_XS.gguf");
        // off by default (2026-10-28); `auto` is the gated opt-in
        assert!(resolve(None, Some(g)).is_none() && resolve(Some(""), Some(g)).is_none());
        let a = resolve(Some("auto"), Some(g)).unwrap();
        assert_eq!(a.0, Path::new("/m/Qwen3.8-27B-UD-IQ4_XS.w4i"));
        assert_eq!(a.1, "ffn:48-63");
        let f = resolve(Some("fast"), Some(g)).unwrap();
        assert_eq!(f.0, Path::new("/m/Qwen3.8-27B-UD-IQ4_XS.fast.w4i"));
        assert_eq!(f.1, "ffn:0-63");
        assert!(resolve(Some("0"), Some(g)).is_none() && resolve(Some("off"), Some(g)).is_none());
        assert_eq!(
            resolve(Some("/x/y.w4i"), Some(g)).unwrap().0,
            Path::new("/x/y.w4i")
        );
        assert!(resolve(None, None).is_none());
    }
}
