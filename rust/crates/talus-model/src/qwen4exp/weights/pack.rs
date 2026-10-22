//! Piece 1 (`qwen4exp-p1`): fp16 bits, the pack index and the weight table (src/core/weights.cpp).

#![allow(clippy::missing_safety_doc)]

use super::*;

// =====================================================================================================================
// fp16 bits (include/strata/kernels/f16_bits.hpp), bit-exact ports
// =====================================================================================================================

/// Round-to-nearest-even f32 -> fp16 bits (`strata::kernels::f16_from_f32`): inf/NaN stay inf/NaN, a finite
/// overflow saturates to inf, subnormals round at the moved rounding point.
#[inline]
pub fn f16_from_f32(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = (x >> 16) & 0x8000;
    let rawexp = (x >> 23) & 0xFF;
    let exp = rawexp as i32 - 127 + 15;
    let mut man = x & 0x7F_FFFF;
    if rawexp == 0xFF {
        return (sign | 0x7C00 | if man != 0 { 0x200 } else { 0 }) as u16;
    }
    if exp >= 31 {
        return (sign | 0x7C00) as u16;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign as u16;
        }
        man |= 0x80_0000;
        let sh = (14 - exp) as u32;
        let mut h = (man >> sh) & 0x3FF;
        let rem = man & ((1u32 << sh) - 1);
        let half = 1u32 << (sh - 1);
        if rem > half || (rem == half && (h & 1) != 0) {
            h += 1;
        }
        return (sign | h) as u16;
    }
    let mut h = (sign | ((exp as u32) << 10) | (man >> 13)) as u16;
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) != 0) {
        h = h.wrapping_add(1);
    }
    h
}

/// Exact fp16 bits -> f32 (`strata::kernels::f32_from_f16`), subnormals included.
#[inline]
pub fn f32_from_f16(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let ex = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let out = if ex == 0 {
        if man == 0 {
            sign
        } else {
            let mut m = man;
            let mut s = 0u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                s += 1;
            }
            sign | ((113 - s) << 23) | ((m & 0x3FF) << 13)
        }
    } else if ex == 31 {
        sign | 0x7F80_0000 | (man << 13)
    } else {
        sign | ((ex + 112) << 23) | (man << 13)
    };
    f32::from_bits(out)
}

// =====================================================================================================================
// The pack index (`<pack>/index.txt`, written by tools/pack_index.py / tools/iq_pack.py)
// =====================================================================================================================

/// What the loader does to a tensor (`strata::core::WeightKind`); index kinds 4 / 5 (raw 16-bit, native packs) map
/// to `Bf16InF32` / `F16InF32` with `raw16` set.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightKind {
    Verbatim = 0,
    Bf16InF32 = 1,
    F32 = 2,
    F16InF32 = 3,
}

impl WeightKind {
    pub(super) fn from_index(k: i32) -> Self {
        match k {
            1 => Self::Bf16InF32,
            2 => Self::F32,
            3 => Self::F16InF32,
            _ => Self::Verbatim,
        }
    }
}

/// One parsed row of `index.txt` (19 fields).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackIndexRow {
    pub name: String,
    pub file: i32,
    pub kind: WeightKind,
    pub src_off: u64,
    pub src_bytes: u64,
    pub dst_off: u64,
    pub dst_bytes: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub code_bits: i32,
    pub code_bias: i32,
    pub group_elems: i32,
    pub codebook: i32,
    pub has_offset: i32,
    pub codes_bytes: u64,
    pub scales_bytes: u64,
    pub offset_bytes: u64,
    pub scales_fp16: i32,
    pub act_kind: i32,
    /// Plan v0.3 P6: already 16-bit in the file (index kinds 4 / 5).
    pub raw16: bool,
}

/// `<pack>/index.txt`: the `# align A pool P tensors T` header and the rows, in file order.
#[derive(Clone, Debug, Default)]
pub struct PackIndex {
    pub align: i32,
    pub pool: u64,
    pub rows: Vec<PackIndexRow>,
}

/// Whitespace tokens of a line the way `sscanf("%s %d ...")` sees them.
pub(super) fn tokens(line: &str) -> std::str::SplitAsciiWhitespace<'_> {
    line.split_ascii_whitespace()
}

/// `sscanf("# align %d pool %llu tensors %d")` on a comment line: `Some((align, pool))` when all three matched.
pub(super) fn parse_header(line: &str) -> Option<(i32, u64)> {
    let rest = line.strip_prefix('#')?;
    let mut t = tokens(rest);
    if t.next()? != "align" {
        return None;
    }
    let a: i32 = t.next()?.parse().ok()?;
    if t.next()? != "pool" {
        return None;
    }
    let p: u64 = t.next()?.parse().ok()?;
    if t.next()? != "tensors" {
        return None;
    }
    let _n: i32 = t.next()?.parse().ok()?;
    Some((a, p))
}

/// The number of leading fields of a row that parse (`sscanf`'s return value for the 19-field format), and the row.
pub(super) fn parse_row(line: &str) -> (usize, Option<PackIndexRow>) {
    let mut t = tokens(line);
    let mut n = 0usize;
    macro_rules! field {
        ($ty:ty) => {{
            match t.next().and_then(|s| s.parse::<$ty>().ok()) {
                Some(v) => {
                    n += 1;
                    v
                }
                None => return (n, None),
            }
        }};
    }
    let name = match t.next() {
        Some(s) => {
            n += 1;
            // %255s: sscanf truncates the token at 255 chars and the rest becomes the next field (never happens in a
            // valid pack; such a row then fails to parse its integers, as in C++)
            if s.len() > 255 {
                return (n, None);
            }
            s.to_string()
        }
        None => return (0, None),
    };
    let file = field!(i32);
    let kind_raw = field!(i32);
    let src_off = field!(u64);
    let src_bytes = field!(u64);
    let dst_off = field!(u64);
    let dst_bytes = field!(u64);
    let ne0 = field!(i64);
    let ne1 = field!(i64);
    let code_bits = field!(i32);
    let code_bias = field!(i32);
    let group_elems = field!(i32);
    let codebook = field!(i32);
    let has_offset = field!(i32);
    let codes_bytes = field!(u64);
    let scales_bytes = field!(u64);
    let offset_bytes = field!(u64);
    let scales_fp16 = field!(i32);
    let act_kind = field!(i32);
    let (kind, raw16) = if kind_raw == 4 || kind_raw == 5 {
        (
            if kind_raw == 4 {
                WeightKind::Bf16InF32
            } else {
                WeightKind::F16InF32
            },
            true,
        )
    } else {
        (WeightKind::from_index(kind_raw), false)
    };
    (
        n,
        Some(PackIndexRow {
            name,
            file,
            kind,
            src_off,
            src_bytes,
            dst_off,
            dst_bytes,
            ne0,
            ne1,
            code_bits,
            code_bias,
            group_elems,
            codebook,
            has_offset,
            codes_bytes,
            scales_bytes,
            offset_bytes,
            scales_fp16,
            act_kind,
            raw16,
        }),
    )
}

/// The pack's data file for an index `file` id.
pub fn pack_file_name(id: i32) -> Option<&'static str> {
    match id {
        0 => Some("dense.bin"),
        1 => Some("embd.bin"),
        2 => Some("experts.bin"),
        3 => Some("extra.bin"),
        _ => None,
    }
}

pub(super) fn read_index_text(pack_dir: &str) -> Result<String, String> {
    let path = format!("{pack_dir}/index.txt");
    let bytes = std::fs::read(&path).map_err(|_| format!("cannot open {path}"))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

impl PackIndex {
    /// Parses `index.txt` text with `WeightTable::load`'s rules: every non-comment line must have all 19 fields.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut out = PackIndex::default();
        for line in text.split_inclusive('\n') {
            if line.starts_with('#') {
                if let Some((a, p)) = parse_header(line) {
                    out.align = a;
                    out.pool = p;
                }
                continue;
            }
            let (n, row) = parse_row(line);
            match row {
                Some(r) => out.rows.push(r),
                None => {
                    // sscanf on an empty line returns EOF (-1)
                    let shown = if n == 0 && tokens(line).next().is_none() {
                        -1
                    } else {
                        n as i64
                    };
                    return Err(format!(
                        "index.txt: could not parse a row ({shown} of 19 fields)"
                    ));
                }
            }
        }
        Ok(out)
    }

    pub fn open(pack_dir: &str) -> Result<Self, String> {
        let path = format!("{pack_dir}/index.txt");
        let text = std::fs::read(&path)
            .map_err(|_| format!("cannot open {path} (run tools/pack_index.py)"))?;
        Self::parse(&String::from_utf8_lossy(&text))
    }

    #[inline]
    fn align_bytes(align: i32) -> u64 {
        if align > 0 {
            align as u64
        } else {
            256
        }
    }

    /// `WeightTable::pool_bytes`: the header's pool, or with `skip` the compacted size of every row not skipped.
    /// Lenient like the C++ (it only needs the first 7 fields of a row and ignores lines that lack them).
    pub fn pool_bytes(pack_dir: &str, skip: Option<&HashSet<String>>) -> Result<u64, String> {
        let text = read_index_text(pack_dir)?;
        Self::pool_bytes_of_text(&text, skip).map_err(|e| {
            if e.is_empty() {
                format!("no '# align ... pool ...' header in {pack_dir}/index.txt")
            } else {
                e
            }
        })
    }

    /// `pool_bytes` over the index text; an empty error string means "no header".
    pub fn pool_bytes_of_text(text: &str, skip: Option<&HashSet<String>>) -> Result<u64, String> {
        let mut pool = 0u64;
        let mut align = 0i32;
        let mut compact = 0u64;
        for line in text.split_inclusive('\n') {
            if line.starts_with('#') {
                if let Some((a, p)) = parse_header(line) {
                    pool = p;
                    align = a;
                }
                continue;
            }
            let Some(skip) = skip else { continue };
            let mut t = tokens(line);
            let Some(name) = t.next() else { continue };
            let mut ok = true;
            let mut dst_bytes = 0u64;
            for i in 0..6 {
                let Some(tok) = t.next() else {
                    ok = false;
                    break;
                };
                if i < 2 {
                    if tok.parse::<i32>().is_err() {
                        ok = false;
                        break;
                    }
                } else {
                    match tok.parse::<u64>() {
                        Ok(v) => {
                            if i == 5 {
                                dst_bytes = v;
                            }
                        }
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            if !ok || skip.contains(name) {
                continue;
            }
            // the C++ reads `align` as it stands at this row (the header comes first in every pack)
            let a = Self::align_bytes(align);
            compact += (dst_bytes + a - 1) / a * a;
        }
        if pool == 0 {
            return Err(String::new());
        }
        Ok(if skip.is_some() { compact } else { pool })
    }

    /// `WeightTable::index_code_bits`: the `code_bits` of row `name` (-1: no such row).
    pub fn index_code_bits(pack_dir: &str, name: &str) -> Result<i32, String> {
        let text = read_index_text(pack_dir)?;
        for line in text.split_inclusive('\n') {
            if line.starts_with('#') {
                continue;
            }
            let mut t = tokens(line);
            let Some(row) = t.next() else { continue };
            let mut vals = [0i64; 9];
            let mut ok = true;
            for v in vals.iter_mut() {
                match t.next().and_then(|s| s.parse::<i64>().ok()) {
                    Some(x) => *v = x,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && row == name {
                return Ok(vals[8] as i32);
            }
        }
        Ok(-1)
    }

    /// The arena placement `WeightTable::load` uses: with `skip`, the kept rows re-placed in index order at the
    /// index's alignment (skipped rows get no bytes) and the compacted pool; without, the index's own offsets.
    /// Returns `(pool, skipped[row])`.
    pub fn place(&mut self, skip: Option<&HashSet<String>>) -> (u64, Vec<bool>) {
        let mut skipped = vec![false; self.rows.len()];
        let mut pool = self.pool;
        if let Some(skip) = skip {
            let a = Self::align_bytes(self.align);
            let mut at = 0u64;
            for (i, r) in self.rows.iter_mut().enumerate() {
                if skip.contains(&r.name) {
                    skipped[i] = true;
                    continue;
                }
                r.dst_off = at;
                at += (r.dst_bytes + a - 1) / a * a;
            }
            pool = at;
        }
        (pool, skipped)
    }
}

// =====================================================================================================================
// The weight table (`strata::core::WeightRef` / `WeightTable`)
// =====================================================================================================================

/// One loaded tensor (`strata::core::WeightRef`, the canonical part; the native GGUF fields are NativeDense's).
#[derive(Clone, Debug)]
pub struct WeightRef {
    /// Device pointer inside the arena (null when skipped: served natively).
    pub data: *const c_void,
    pub bytes: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub elements: i64,
    pub kind: WeightKind,
    pub code_bits: i32,
    pub code_bias: i32,
    pub group_elems: i32,
    pub codebook_iq4nl: bool,
    pub has_offset: bool,
    pub codes_bytes: u64,
    pub scales_bytes: u64,
    pub offset_bytes: u64,
    pub src_off: u64,
    pub src_bytes: u64,
    pub file_id: i32,
    pub scales_fp16: bool,
    pub act_kind: i32,
    pub resident: bool,
}

unsafe impl Send for WeightRef {}
unsafe impl Sync for WeightRef {}

impl WeightRef {
    pub(super) fn from_row(r: &PackIndexRow, data: *const c_void, resident: bool) -> Self {
        Self {
            data,
            bytes: r.dst_bytes,
            ne0: r.ne0,
            ne1: r.ne1,
            elements: r.ne0 * if r.ne1 > 0 { r.ne1 } else { 1 },
            kind: r.kind,
            code_bits: r.code_bits,
            code_bias: r.code_bias,
            group_elems: r.group_elems,
            codebook_iq4nl: r.codebook != 0,
            has_offset: r.has_offset != 0,
            codes_bytes: r.codes_bytes,
            scales_bytes: r.scales_bytes,
            offset_bytes: r.offset_bytes,
            src_off: r.src_off,
            src_bytes: r.src_bytes,
            file_id: r.file,
            scales_fp16: r.scales_fp16 != 0,
            act_kind: r.act_kind,
            resident,
        }
    }

    #[inline]
    pub fn quantized(&self) -> bool {
        self.code_bits != 0
    }
}

/// `strata::core::LoadReport`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadReport {
    pub tensors: usize,
    pub arena_bytes: u64,
    pub re_rounded: usize,
    pub bytes_saved: u64,
    pub read_ms: f64,
    pub upload_ms: f64,
}

/// The dense weights in the device arena: name -> [`WeightRef`] (`strata::core::WeightTable`).
#[derive(Clone, Debug, Default)]
pub struct WeightTable {
    pub table: BTreeMap<String, WeightRef>,
    pub report: LoadReport,
    pub pack_dir: String,
    pub arena: usize,
    pub arena_bytes: u64,
}

/// How a segment's bytes change from the pack to the arena (`weights.cpp` `Conv`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Conv {
    Copy,
    Bf16High,
    ToF16,
    WidenF16,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Seg {
    src_off: u64,
    src_bytes: u64,
    dst_off: u64,
    dst_bytes: u64,
    conv: Conv,
}

/// The segment list of one row (codes / scales / offset planes of a quantized tensor, or one run).
pub(super) fn segments_of(r: &PackIndexRow) -> Result<Vec<Seg>, String> {
    let mut segs = Vec::with_capacity(3);
    let mut err: Option<String> = None;
    if r.codes_bytes != 0 {
        segs.push(Seg {
            src_off: 0,
            src_bytes: r.codes_bytes,
            dst_off: 0,
            dst_bytes: r.codes_bytes,
            conv: Conv::Copy,
        });
        if r.scales_bytes != 0 {
            let fp16 = r.scales_fp16 != 0;
            let src_scale_bytes = if fp16 {
                r.scales_bytes / 2
            } else {
                r.scales_bytes
            };
            if fp16 && (r.scales_bytes % 2 != 0 || src_scale_bytes * 2 != r.scales_bytes) {
                err = Some(format!(
                    "{}: an fp16 scale plane of an odd byte count",
                    r.name
                ));
            }
            segs.push(Seg {
                src_off: r.codes_bytes,
                src_bytes: src_scale_bytes,
                dst_off: r.codes_bytes,
                dst_bytes: r.scales_bytes,
                conv: if fp16 { Conv::WidenF16 } else { Conv::Copy },
            });
        }
        if r.offset_bytes != 0 {
            let after = r.codes_bytes
                + if r.scales_fp16 != 0 {
                    r.scales_bytes / 2
                } else {
                    r.scales_bytes
                };
            segs.push(Seg {
                src_off: after,
                src_bytes: r.offset_bytes,
                dst_off: r.codes_bytes + r.scales_bytes,
                dst_bytes: r.offset_bytes,
                conv: Conv::Copy,
            });
        }
    } else {
        let c = if r.raw16 {
            Conv::Copy
        } else if r.kind == WeightKind::Bf16InF32 {
            Conv::Bf16High
        } else if r.kind == WeightKind::F16InF32 {
            Conv::ToF16
        } else {
            Conv::Copy
        };
        let elems = (r.ne0 as u64).wrapping_mul(if r.ne1 > 0 { r.ne1 as u64 } else { 1 });
        if c != Conv::Copy && (r.src_bytes != elems * 4 || r.dst_bytes != elems * 2) {
            err = Some(format!(
                "{}: a promoted tensor of {} elements is {} B in and {} B out, not {}/{}",
                r.name,
                elems,
                r.src_bytes,
                r.dst_bytes,
                elems * 4,
                elems * 2
            ));
        }
        segs.push(Seg {
            src_off: 0,
            src_bytes: r.src_bytes,
            dst_off: 0,
            dst_bytes: r.dst_bytes,
            conv: c,
        });
    }
    if let Some(e) = err {
        return Err(e);
    }
    let sum_src: u64 = segs.iter().map(|s| s.src_bytes).sum();
    let sum_dst: u64 = segs.iter().map(|s| s.dst_bytes).sum();
    if sum_src != r.src_bytes || sum_dst != r.dst_bytes {
        return Err(format!(
            "{}: the segments cover {}/{} B but the index says {}/{} - the plane layout this loader built is not the \
             one the index describes",
            r.name, sum_src, sum_dst, r.src_bytes, r.dst_bytes
        ));
    }
    Ok(segs)
}

/// Converts `src` (one staged chunk of a segment) into `out`, returning `(out_bytes, out_at_rel_to_segment)` where the
/// relative offset is computed from `done` (the bytes of the segment's source consumed before this chunk).
pub(super) fn convert_chunk(
    conv: Conv,
    src: &[u8],
    out: &mut [u8],
    done: u64,
) -> (usize, u64, bool) {
    match conv {
        Conv::Copy => (src.len(), done, false),
        Conv::Bf16High => {
            let elems = src.len() / 4;
            for i in 0..elems {
                out[i * 2] = src[i * 4 + 2];
                out[i * 2 + 1] = src[i * 4 + 3];
            }
            (elems * 2, done / 2, true)
        }
        Conv::ToF16 => {
            let elems = src.len() / 4;
            for i in 0..elems {
                let v = f32::from_le_bytes([
                    src[i * 4],
                    src[i * 4 + 1],
                    src[i * 4 + 2],
                    src[i * 4 + 3],
                ]);
                let h = f16_from_f32(v).to_le_bytes();
                out[i * 2] = h[0];
                out[i * 2 + 1] = h[1];
            }
            (elems * 2, done / 2, true)
        }
        Conv::WidenF16 => {
            let elems = src.len() / 2;
            for i in 0..elems {
                let h = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                let f = f32_from_f16(h).to_le_bytes();
                out[i * 4..i * 4 + 4].copy_from_slice(&f);
            }
            (elems * 4, done * 2, true)
        }
    }
}

/// 8 MiB of SOURCE per staging round (`weights.cpp` CHUNK).
pub(super) const LOAD_CHUNK: u64 = 8 << 20;
pub(super) const ADVISE_STEP: u64 = 128 << 10;
pub(super) const POSIX_FADV_WILLNEED: c_int = 3;

extern "C" {
    fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
}

pub(super) fn read_ahead_enabled() -> bool {
    std::env::var("STRATA_READ_AHEAD")
        .ok()
        .map(|v| v.trim().parse::<i64>().unwrap_or(0) != 0)
        .unwrap_or(true)
}

/// `strata::platform::advise_willneed(fd, off, bytes)`.
pub(super) fn advise_willneed(f: &File, offset: u64, bytes: u64) {
    if !read_ahead_enabled() {
        return;
    }
    let fd = f.as_raw_fd();
    let mut at = 0u64;
    while at < bytes {
        let n = ADVISE_STEP.min(bytes - at);
        unsafe {
            let _ = posix_fadvise(fd, (offset + at) as i64, n as i64, POSIX_FADV_WILLNEED);
        }
        at += ADVISE_STEP;
    }
}

/// Where converted bytes go: the device arena (`hipMemcpy` H2D, the engine), or a host buffer (tests).
pub trait ArenaSink {
    fn write(&mut self, dst_off: u64, src: &[u8]) -> Result<(), String>;
    fn finish(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// A host `Vec<u8>` standing in for the arena (unit tests, the CPU-side cross check).
pub struct HostArena(pub Vec<u8>);

impl ArenaSink for HostArena {
    fn write(&mut self, dst_off: u64, src: &[u8]) -> Result<(), String> {
        let a = dst_off as usize;
        let b = a + src.len();
        if b > self.0.len() {
            return Err("host arena overflow".into());
        }
        self.0[a..b].copy_from_slice(src);
        Ok(())
    }
}

/// The device arena: one synchronous `hipMemcpy` per staged chunk, as `WeightTable::load` does.
pub struct DeviceArena {
    pub base: *mut u8,
    pub bytes: u64,
    pub upload_ms: f64,
}

impl ArenaSink for DeviceArena {
    fn write(&mut self, dst_off: u64, src: &[u8]) -> Result<(), String> {
        if dst_off + src.len() as u64 > self.bytes {
            return Err("device arena overflow".into());
        }
        let t0 = std::time::Instant::now();
        let e = unsafe {
            hip_sys::hipMemcpy(
                self.base.add(dst_off as usize).cast(),
                src.as_ptr().cast(),
                src.len(),
                hip_sys::hipMemcpyHostToDevice,
            )
        };
        self.upload_ms += t0.elapsed().as_secs_f64() * 1e3;
        if e != hip_sys::hipSuccess {
            return Err("hipMemcpy failed".into());
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), String> {
        if unsafe { hip_sys::hipDeviceSynchronize() } != hip_sys::hipSuccess {
            return Err("cudaDeviceSynchronize after the load failed".into());
        }
        Ok(())
    }
}

/// Pinned (`hipHostMalloc`) staging buffer, or a plain Vec when HIP is not available (tests).
pub(super) struct Staging {
    ptr: *mut u8,
    len: usize,
    pinned: bool,
    vec: Vec<u8>,
}

impl Staging {
    fn new(len: usize, pinned: bool) -> Result<Self, String> {
        if pinned {
            let mut p: *mut c_void = std::ptr::null_mut();
            let e = unsafe { hip_sys::hipHostMalloc(&mut p, len, hip_sys::hipHostMallocDefault) };
            if e != hip_sys::hipSuccess || p.is_null() {
                return Err("cudaHostAlloc for the staging buffers failed".into());
            }
            Ok(Self {
                ptr: p.cast(),
                len,
                pinned: true,
                vec: Vec::new(),
            })
        } else {
            let mut vec = vec![0u8; len];
            Ok(Self {
                ptr: vec.as_mut_ptr(),
                len,
                pinned: false,
                vec,
            })
        }
    }

    fn slice_mut(&mut self) -> &mut [u8] {
        if self.pinned {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        } else {
            &mut self.vec[..]
        }
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        if self.pinned && !self.ptr.is_null() {
            unsafe {
                let _ = hip_sys::hipHostFree(self.ptr.cast());
            }
        }
    }
}

impl WeightTable {
    /// `WeightTable::load`: every row of `<pack_dir>/index.txt` into the arena through `sink`, `arena_base` being the
    /// device address the arena's offset 0 has (the `WeightRef::data` pointers are built from it).
    pub fn load_with<S: ArenaSink>(
        pack_dir: &str,
        arena_base: *mut u8,
        arena_bytes: u64,
        skip: Option<&HashSet<String>>,
        sink: &mut S,
        pinned: bool,
    ) -> Result<Self, String> {
        let mut idx = PackIndex::open(pack_dir)?;
        if idx.pool == 0 || idx.rows.is_empty() {
            return Err("index.txt has no header or no rows".into());
        }
        let (pool, skipped) = idx.place(skip);
        if arena_bytes < pool {
            return Err(format!(
                "arena is {arena_bytes} B but the index needs {pool}"
            ));
        }
        let mut stage_in = Staging::new(LOAD_CHUNK as usize, pinned)?;
        let mut stage_out = Staging::new((LOAD_CHUNK * 2) as usize, pinned)?;

        let mut out = WeightTable {
            pack_dir: pack_dir.to_string(),
            arena: arena_base as usize,
            arena_bytes,
            ..Default::default()
        };
        let t_read0 = std::time::Instant::now();
        let mut upload_s = 0.0f64;
        let mut cur: Option<File> = None;
        let mut cur_file = -1i32;
        for (row_i, r) in idx.rows.iter().enumerate() {
            if skipped[row_i] {
                out.table.insert(
                    r.name.clone(),
                    WeightRef::from_row(r, std::ptr::null(), false),
                );
                out.report.tensors += 1;
                continue;
            }
            if r.code_bits != 0 && r.dst_bytes == 0 {
                return Err(format!(
                    "{}: this pack holds the tensor only in its GGUF form (run with --native SHARD1)",
                    r.name
                ));
            }
            if r.file != cur_file {
                let p = format!("{pack_dir}/{}", pack_file_name(r.file).unwrap_or("?"));
                let f = File::open(&p).map_err(|_| format!("cannot open {p}"))?;
                // ask for this file's remaining rows up front so their reads overlap
                for (j, rj) in idx.rows.iter().enumerate().skip(row_i) {
                    if rj.file == r.file && !skipped[j] {
                        advise_willneed(&f, rj.src_off, rj.src_bytes);
                    }
                }
                cur = Some(f);
                cur_file = r.file;
            }
            let f = cur.as_ref().unwrap();
            let segs = segments_of(r)?;
            for s in &segs {
                let widening = s.conv == Conv::WidenF16;
                if widening && (s.src_bytes & 1) != 0 {
                    return Err(format!(
                        "{}: an fp16 plane with an odd source byte count",
                        r.name
                    ));
                }
                let mut done = 0u64;
                while done < s.src_bytes {
                    let mut n = LOAD_CHUNK.min(s.src_bytes - done);
                    if widening && (n & 1) != 0 {
                        if n == 1 {
                            return Err(format!(
                                "{}: an fp16 plane ending on a half element",
                                r.name
                            ));
                        }
                        n -= 1;
                    }
                    let inb = &mut stage_in.slice_mut()[..n as usize];
                    f.read_exact_at(inb, r.src_off + s.src_off + done)
                        .map_err(|_| format!("{}: short read", r.name))?;
                    let inb: &[u8] = &stage_in.slice_mut()[..n as usize];
                    // SAFETY: stage_in and stage_out are distinct buffers.
                    let inb: &[u8] = unsafe { std::slice::from_raw_parts(inb.as_ptr(), inb.len()) };
                    let outb = stage_out.slice_mut();
                    let (out_bytes, rel, converted) = convert_chunk(s.conv, inb, outb, done);
                    let host_src: &[u8] = if converted { &outb[..out_bytes] } else { inb };
                    let t0 = std::time::Instant::now();
                    sink.write(r.dst_off + s.dst_off + rel, host_src)
                        .map_err(|e| format!("cudaMemcpy failed for {} ({e})", r.name))?;
                    upload_s += t0.elapsed().as_secs_f64();
                    done += n;
                }
            }
            let data = unsafe { arena_base.add(r.dst_off as usize) } as *const c_void;
            out.table
                .insert(r.name.clone(), WeightRef::from_row(r, data, true));
            out.report.tensors += 1;
            if (r.kind == WeightKind::Bf16InF32 || r.kind == WeightKind::F16InF32) && !r.raw16 {
                out.report.re_rounded += 1;
                out.report.bytes_saved += r.src_bytes - r.dst_bytes;
            }
        }
        drop(cur);
        out.report.arena_bytes = pool;
        let total_ms = t_read0.elapsed().as_secs_f64() * 1e3;
        out.report.upload_ms = upload_s * 1e3;
        out.report.read_ms = total_ms - out.report.upload_ms;
        sink.finish()?;
        Ok(out)
    }

    /// Loads into a device arena (`hipMalloc`ed by the caller, at least `pool_bytes` long).
    pub fn load_device(
        pack_dir: &str,
        arena_base: *mut c_void,
        arena_bytes: u64,
        skip: Option<&HashSet<String>>,
    ) -> Result<Self, String> {
        let mut sink = DeviceArena {
            base: arena_base.cast(),
            bytes: arena_bytes,
            upload_ms: 0.0,
        };
        Self::load_with(
            pack_dir,
            arena_base.cast(),
            arena_bytes,
            skip,
            &mut sink,
            true,
        )
    }

    pub fn find(&self, name: &str) -> Option<&WeightRef> {
        self.table.get(name)
    }
}
