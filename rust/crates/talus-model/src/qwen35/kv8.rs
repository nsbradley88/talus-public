//! CPU reference of the 27B int8 KV copy (`src/qwen35/q36_kv8.cu`, `TALUS_Q27_KV8=1`) and of the
//! turbo6-style codec considered for stage 2 (design only; no GPU kernel for it yet).
//!
//! Int8 copy (the Flash-Next `kv_q8.hpp` rule, offset-binary code): per cell, KV head and group of 64 values,
//! `scale = fp16(max|x| / 127)`, `c = clamp(rint(x / scale), -127, 127)`, stored as `u = c + 128`.  `x` is the f16
//! value of the f16 cache, so the copy is a pure function of the f16 cache (a prompt-cache restore rebuilds it).
//! Layouts per attention layer: `kq u8 [kvh][ctx][256]`, `ks f16 [kvh][ctx][4]`, `vq u8 [kvh][vt_ctx/16][256][16]`,
//! `vs f16 [kvh][4][vt_ctx]`.
//!
//! turbo6-style codec (format credit: turbo6 by JakeATX, TheTom/llama-cpp-turboquant PR #381, MIT; idea via tubo,
//! BonkusCheemus/tubo, MIT; implemented here from the format description only): 128-value blocks, randomized
//! Hadamard rotation (sign flip, FWHT, sign flip), a 64-entry Lloyd-Max codebook for N(0, 1) on the rotated and
//! normalized values, one fp16 norm per block corrected so the decoded block keeps the block's L2 norm.
//! 98 bytes per block = 6.125 bits per value.

use std::sync::OnceLock;

/// Head dim the int8 copy supports.
pub const KV8_HD: usize = 256;
/// Values per scale.
pub const KV8_GROUP: usize = 64;
/// Scales per cell and KV head.
pub const KV8_GROUPS: usize = KV8_HD / KV8_GROUP;
/// Bytes per cell and KV head (K + V codes and scales).
pub const KV8_CELL_BYTES: usize = 2 * KV8_HD + 2 * KV8_GROUPS * 2;
/// Format version (must equal `talus_q36_kv8_version()` of the kernel library).
pub const KV8_VERSION: i32 = 1;

/// `TALUS_Q27_KV8=1` (default off): decode / verify attention reads the int8 copy.
pub fn kv8_requested() -> bool {
    std::env::var("TALUS_Q27_KV8")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("q8"))
        .unwrap_or(false)
}

/// `TALUS_Q27_KV8_OFS=1`: the reader variant with offset operands and per-row corrections (fewer VALU ops; numerics
/// differ from variant 0 at rounding level).  0 = exact codes (default).
pub fn kv8_variant() -> i32 {
    std::env::var("TALUS_Q27_KV8_OFS")
        .map(|v| i32::from(v == "1"))
        .unwrap_or(0)
}

// ------------------------------------------------------------------------------------------------ fp16 helpers

/// f32 -> IEEE binary16 bits, round to nearest even (as `__float2half`).
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && (r & 1) != 0) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) != 0) {
        r += 1;
    }
    sign | r as u16
}

/// IEEE binary16 bits -> f32 (exact).
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 113u32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

// ------------------------------------------------------------------------------------------------ int8 copy

/// One group: (scale bits, offset codes), the kernel's `kv8_code` for every value.
pub fn kv8_quant_group(x: &[f32]) -> (u16, Vec<u8>) {
    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let sbits = f32_to_f16_bits(amax / 127.0);
    let sf = f16_bits_to_f32(sbits);
    let codes = x
        .iter()
        .map(|&v| {
            let q = if sf > 0.0 {
                ((v / sf).round_ties_even() as i32).clamp(-127, 127)
            } else {
                0
            };
            (q + 128) as u8
        })
        .collect();
    (sbits, codes)
}

/// Decoded value of one code.
pub fn kv8_dequant(code: u8, sbits: u16) -> f32 {
    (code as i32 - 128) as f32 * f16_bits_to_f32(sbits)
}

/// `vt_index` of `include/strata/qwen35/q36_kernels.hpp` (also the `vq` byte index).
pub fn kv8_vt_index(kvh: usize, cell: usize, d: usize, vt_ctx: usize) -> usize {
    ((kvh * (vt_ctx / 16) + cell / 16) * KV8_HD + d) * 16 + cell % 16
}
/// `kq` byte index.
pub fn kv8_kq_index(kvh: usize, cell: usize, d: usize, ctx: usize) -> usize {
    (kvh * ctx + cell) * KV8_HD + d
}
/// `ks` element index.
pub fn kv8_ks_index(kvh: usize, cell: usize, g: usize, ctx: usize) -> usize {
    (kvh * ctx + cell) * KV8_GROUPS + g
}
/// `vs` element index.
pub fn kv8_vs_index(kvh: usize, cell: usize, g: usize, vt_ctx: usize) -> usize {
    (kvh * KV8_GROUPS + g) * vt_ctx + cell
}

/// Element counts of one layer's buffers: (kq bytes, ks halves, vq bytes, vs halves).
pub fn kv8_layer_elems(hkv: usize, ctx: usize, vt_ctx: usize) -> (usize, usize, usize, usize) {
    (
        hkv * ctx * KV8_HD,
        hkv * ctx * KV8_GROUPS,
        hkv * vt_ctx * KV8_HD,
        hkv * KV8_GROUPS * vt_ctx,
    )
}

/// One layer's int8 buffers on the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kv8Layer {
    pub kq: Vec<u8>,
    pub ks: Vec<u16>,
    pub vq: Vec<u8>,
    pub vs: Vec<u16>,
}

impl Kv8Layer {
    pub fn zeroed(hkv: usize, ctx: usize, vt_ctx: usize) -> Self {
        let (a, b, c, d) = kv8_layer_elems(hkv, ctx, vt_ctx);
        Self {
            kq: vec![0; a],
            ks: vec![0; b],
            vq: vec![0; c],
            vs: vec![0; d],
        }
    }
}

/// CPU mirror of `k_kv8_quant`: cells `[c0, c0 + n)` of one layer from its f16 K cache (`[kvh][ctx][256]` bits) and
/// vt copy (`kv8_vt_index` bits).
#[allow(clippy::too_many_arguments)]
pub fn kv8_quant_cells(
    kc: &[u16],
    vt: &[u16],
    hkv: usize,
    ctx: usize,
    vt_ctx: usize,
    c0: usize,
    n: usize,
    out: &mut Kv8Layer,
) {
    let mut gk = [0.0f32; KV8_GROUP];
    let mut gv = [0.0f32; KV8_GROUP];
    for kvh in 0..hkv {
        for cell in c0..(c0 + n).min(ctx).min(vt_ctx) {
            for g in 0..KV8_GROUPS {
                for j in 0..KV8_GROUP {
                    let d = g * KV8_GROUP + j;
                    gk[j] = f16_bits_to_f32(kc[(kvh * ctx + cell) * KV8_HD + d]);
                    gv[j] = f16_bits_to_f32(vt[kv8_vt_index(kvh, cell, d, vt_ctx)]);
                }
                let (sk, ck) = kv8_quant_group(&gk);
                let (sv, cv) = kv8_quant_group(&gv);
                for j in 0..KV8_GROUP {
                    let d = g * KV8_GROUP + j;
                    out.kq[kv8_kq_index(kvh, cell, d, ctx)] = ck[j];
                    out.vq[kv8_vt_index(kvh, cell, d, vt_ctx)] = cv[j];
                }
                out.ks[kv8_ks_index(kvh, cell, g, ctx)] = sk;
                out.vs[kv8_vs_index(kvh, cell, g, vt_ctx)] = sv;
            }
        }
    }
}

/// Decoded K row (256 values) of one cell and KV head.
pub fn kv8_k_row(l: &Kv8Layer, kvh: usize, cell: usize, ctx: usize) -> Vec<f32> {
    (0..KV8_HD)
        .map(|d| {
            kv8_dequant(
                l.kq[kv8_kq_index(kvh, cell, d, ctx)],
                l.ks[kv8_ks_index(kvh, cell, d / KV8_GROUP, ctx)],
            )
        })
        .collect()
}

/// Decoded V row (256 values) of one cell and KV head.
pub fn kv8_v_row(l: &Kv8Layer, kvh: usize, cell: usize, vt_ctx: usize) -> Vec<f32> {
    (0..KV8_HD)
        .map(|d| {
            kv8_dequant(
                l.vq[kv8_vt_index(kvh, cell, d, vt_ctx)],
                l.vs[kv8_vs_index(kvh, cell, d / KV8_GROUP, vt_ctx)],
            )
        })
        .collect()
}

/// Reference gated attention for T rows over decoded K/V rows (`k(cell)`, `v(cell)`): row (t, h) sees cells
/// `[0, p0 + t]`; returns `[T][H][256]` like `tk_q36_attn_merge` (times sigmoid(gate)).
#[allow(clippy::too_many_arguments)]
pub fn ref_attention(
    q: &[f32],
    gate: &[f32],
    t_rows: usize,
    h: usize,
    hkv: usize,
    p0: usize,
    scale: f32,
    k: &dyn Fn(usize, usize) -> Vec<f32>,
    v: &dyn Fn(usize, usize) -> Vec<f32>,
) -> Vec<f32> {
    let g = h / hkv;
    let mut out = vec![0.0f32; t_rows * h * KV8_HD];
    for t in 0..t_rows {
        let n = p0 + t + 1;
        for hh in 0..h {
            let kvh = hh / g;
            let qr = &q[(t * h + hh) * KV8_HD..(t * h + hh + 1) * KV8_HD];
            let s: Vec<f64> = (0..n)
                .map(|c| {
                    let kr = k(kvh, c);
                    qr.iter()
                        .zip(&kr)
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum::<f64>()
                        * scale as f64
                })
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|&x| (x - m).exp()).collect();
            let l: f64 = e.iter().sum();
            let mut acc = vec![0.0f64; KV8_HD];
            for (c, &w) in e.iter().enumerate() {
                let vr = v(kvh, c);
                for d in 0..KV8_HD {
                    acc[d] += w * vr[d] as f64;
                }
            }
            for d in 0..KV8_HD {
                let gi = (t * h + hh) * KV8_HD + d;
                let sg = 1.0 / (1.0 + (-(gate[gi] as f64)).exp());
                out[gi] = (acc[d] / l * sg) as f32;
            }
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ turbo6-style codec

/// Values per turbo6 block.
pub const T6_BLOCK: usize = 128;
/// Codebook entries.
pub const T6_LEVELS: usize = 64;
/// Bytes per block: 128 x 6-bit indices + an fp16 norm.
pub const T6_BLOCK_BYTES: usize = T6_BLOCK * 6 / 8 + 2;

/// In-place unnormalized fast Walsh-Hadamard transform (length a power of two).
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut hlen = 1;
    while hlen < n {
        for i in (0..n).step_by(2 * hlen) {
            for j in i..i + hlen {
                let (a, b) = (x[j], x[j + hlen]);
                x[j] = a + b;
                x[j + hlen] = a - b;
            }
        }
        hlen *= 2;
    }
}

/// The two fixed sign vectors of the randomized Hadamard rotation (xorshift from fixed seeds; any fixed choice works,
/// the reader only needs the same ones).
pub fn t6_signs() -> &'static ([f32; T6_BLOCK], [f32; T6_BLOCK]) {
    static S: OnceLock<([f32; T6_BLOCK], [f32; T6_BLOCK])> = OnceLock::new();
    S.get_or_init(|| {
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            if st & 1 == 0 {
                1.0f32
            } else {
                -1.0f32
            }
        };
        let mut a = [0.0f32; T6_BLOCK];
        let mut b = [0.0f32; T6_BLOCK];
        for v in &mut a {
            *v = next();
        }
        for v in &mut b {
            *v = next();
        }
        (a, b)
    })
}

/// Orthonormal randomized Hadamard rotation `R x = s2 . H (s1 . x) / sqrt(128)` and its inverse.
pub fn t6_rotate(x: &[f32], inverse: bool) -> Vec<f32> {
    let (s1, s2) = t6_signs();
    let (first, second) = if inverse { (s2, s1) } else { (s1, s2) };
    let mut y: Vec<f32> = x.iter().zip(first.iter()).map(|(a, b)| a * b).collect();
    fwht(&mut y);
    let k = 1.0 / (T6_BLOCK as f32).sqrt();
    y.iter()
        .zip(second.iter())
        .map(|(a, b)| a * b * k)
        .collect()
}

/// The 64-level Lloyd-Max quantizer of N(0, 1): reconstruction levels, ascending, symmetric.  Computed once by Lloyd
/// iteration over a dense grid of the density (deterministic).
pub fn t6_codebook() -> &'static [f32; T6_LEVELS] {
    static CB: OnceLock<[f32; T6_LEVELS]> = OnceLock::new();
    CB.get_or_init(|| {
        const GRID: usize = 1 << 17;
        const LIM: f64 = 7.0;
        let dx = 2.0 * LIM / GRID as f64;
        let xs: Vec<f64> = (0..GRID).map(|i| -LIM + (i as f64 + 0.5) * dx).collect();
        let ws: Vec<f64> = xs.iter().map(|&x| (-0.5 * x * x).exp()).collect();
        let mut lv: Vec<f64> = (0..T6_LEVELS)
            .map(|i| -2.7 + 5.4 * (i as f64 + 0.5) / T6_LEVELS as f64)
            .collect();
        for _ in 0..400 {
            let th: Vec<f64> = (0..T6_LEVELS - 1)
                .map(|i| 0.5 * (lv[i] + lv[i + 1]))
                .collect();
            let mut num = vec![0.0f64; T6_LEVELS];
            let mut den = vec![0.0f64; T6_LEVELS];
            let mut k = 0;
            for (&x, &w) in xs.iter().zip(&ws) {
                while k < T6_LEVELS - 1 && x > th[k] {
                    k += 1;
                }
                num[k] += w * x;
                den[k] += w;
            }
            for i in 0..T6_LEVELS {
                if den[i] > 0.0 {
                    lv[i] = num[i] / den[i];
                }
            }
            for i in 0..T6_LEVELS / 2 {
                // keep it exactly symmetric
                let m = 0.5 * (lv[T6_LEVELS - 1 - i] - lv[i]);
                lv[i] = -m;
                lv[T6_LEVELS - 1 - i] = m;
            }
        }
        let mut out = [0.0f32; T6_LEVELS];
        for (o, l) in out.iter_mut().zip(&lv) {
            *o = *l as f32;
        }
        out
    })
}

fn t6_nearest(z: f32, cb: &[f32; T6_LEVELS]) -> u8 {
    // ascending levels: binary search on the midpoints
    let (mut lo, mut hi) = (0usize, T6_LEVELS - 1);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if z > 0.5 * (cb[mid] + cb[mid + 1]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo as u8
}

/// One encoded block: 6-bit indices (one per byte here; the packed form is 96 bytes) and the fp16 corrected norm.
#[derive(Debug, Clone, PartialEq)]
pub struct T6Block {
    pub idx: Vec<u8>,
    pub norm: u16,
}

/// Encode one 128-value block.
pub fn t6_encode(x: &[f32]) -> T6Block {
    assert_eq!(x.len(), T6_BLOCK);
    let cb = t6_codebook();
    let norm = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
    if norm == 0.0 {
        return T6Block {
            idx: vec![t6_nearest(0.0, cb); T6_BLOCK],
            norm: 0,
        };
    }
    let y = t6_rotate(x, false);
    let k = (T6_BLOCK as f64).sqrt() / norm;
    let idx: Vec<u8> = y
        .iter()
        .map(|&v| t6_nearest((v as f64 * k) as f32, cb))
        .collect();
    // corrected norm: the decoded block norm' * |cb[idx]| / sqrt(128) equals the block's L2 norm
    let cn = idx
        .iter()
        .map(|&i| cb[i as usize] as f64 * cb[i as usize] as f64)
        .sum::<f64>()
        .sqrt();
    let corr = if cn > 0.0 {
        norm * (T6_BLOCK as f64).sqrt() / cn
    } else {
        0.0
    };
    T6Block {
        idx,
        norm: f32_to_f16_bits(corr as f32),
    }
}

/// Decode one block (back to the original basis).
pub fn t6_decode(b: &T6Block) -> Vec<f32> {
    let cb = t6_codebook();
    let k = f16_bits_to_f32(b.norm) / (T6_BLOCK as f32).sqrt();
    let y: Vec<f32> = b.idx.iter().map(|&i| cb[i as usize] * k).collect();
    t6_rotate(&y, true)
}

/// Pack 128 6-bit indices into 96 bytes (4 indices per 3 bytes, little-endian bit order).
pub fn t6_pack(idx: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(idx.len() * 6 / 8);
    for c in idx.chunks(4) {
        let w = (c[0] as u32 & 63)
            | (c[1] as u32 & 63) << 6
            | (c[2] as u32 & 63) << 12
            | (c[3] as u32 & 63) << 18;
        out.extend_from_slice(&[w as u8, (w >> 8) as u8, (w >> 16) as u8]);
    }
    out
}

/// Unpack [`t6_pack`].
pub fn t6_unpack(p: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(p.len() * 8 / 6);
    for c in p.chunks(3) {
        let w = c[0] as u32 | (c[1] as u32) << 8 | (c[2] as u32) << 16;
        for j in 0..4 {
            out.push(((w >> (6 * j)) & 63) as u8);
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn gauss(&mut self) -> f32 {
            let (a, b) = (self.u().max(1e-300), self.u());
            ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
        }
    }

    fn f16r(x: f32) -> f32 {
        f16_bits_to_f32(f32_to_f16_bits(x))
    }

    #[test]
    fn f16_conversion_matches_known_bits() {
        let cases: &[(f32, u16)] = &[
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (65504.0, 0x7bff),
            (65520.0, 0x7c00), // rounds to inf
            (1024.0, 0x6400),
            (1.0 / 127.0, 0x2008),
            (5.960_464_5e-8, 0x0001), // smallest subnormal
            (2.980_232_2e-8, 0x0000), // tie to even -> 0
            (f32::INFINITY, 0x7c00),
        ];
        for &(x, h) in cases {
            assert_eq!(f32_to_f16_bits(x), h, "{x}");
        }
        for h in 0u16..=0xffff {
            if (h & 0x7c00) == 0x7c00 && (h & 0x3ff) != 0 {
                continue; // NaN payloads
            }
            assert_eq!(f32_to_f16_bits(f16_bits_to_f32(h)), h, "{h:#x}");
        }
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
    }

    #[test]
    fn decoder_trick_is_exact() {
        // the kernel's byte splice: f16 bits 0x6400 | u == 1024 + u, minus 1152 == u - 128, for every code
        for u in 0u16..=255 {
            let v = f16_bits_to_f32(0x6400 | u) - 1152.0;
            assert_eq!(v, u as f32 - 128.0);
            assert_eq!(f16r(v), v);
        }
    }

    #[test]
    fn kv8_round_trip_error_bound() {
        let mut r = Rng(0x1234_5678_9abc_def1);
        for trial in 0..2000 {
            let s = [1e-3f32, 0.05, 1.0, 30.0, 400.0][trial % 5];
            let mut x: Vec<f32> = (0..KV8_GROUP).map(|_| f16r(r.gauss() * s)).collect();
            if trial % 7 == 0 {
                x[trial % KV8_GROUP] = f16r(s * 25.0); // an outlier channel
            }
            let (sb, codes) = kv8_quant_group(&x);
            let sf = f16_bits_to_f32(sb);
            let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            // the stored scale is within fp16 rounding of amax / 127, so no value clips
            assert!((sf - amax / 127.0).abs() <= amax / 127.0 * 4.9e-4 + 6e-8);
            for (&v, &c) in x.iter().zip(&codes) {
                assert!((1..=255).contains(&c));
                let e = (v - kv8_dequant(c, sb)).abs();
                assert!(
                    e <= 0.5 * sf * (1.0 + 1e-6) + 1e-12,
                    "x={v} err={e} scale={sf}"
                );
            }
        }
    }

    #[test]
    fn kv8_edge_groups() {
        let z = vec![0.0f32; KV8_GROUP];
        let (sb, c) = kv8_quant_group(&z);
        assert_eq!(sb, 0);
        assert!(c.iter().all(|&u| u == 128));
        let mut x = vec![0.0f32; KV8_GROUP];
        x[5] = -3.0;
        x[9] = 1.5;
        let (sb, c) = kv8_quant_group(&x);
        assert_eq!(c[5], 1); // -127 + 128: the max magnitude maps to the end code
        assert_eq!(kv8_dequant(c[5], sb), -127.0 * f16_bits_to_f32(sb));
        assert!((kv8_dequant(c[9], sb) - 1.5).abs() <= 0.5 * f16_bits_to_f32(sb));
        // a group so small its scale underflows fp16: every code is zero
        let tiny = vec![1e-9f32; KV8_GROUP];
        let (sb, c) = kv8_quant_group(&tiny);
        assert_eq!(sb, 0);
        assert!(c.iter().all(|&u| u == 128));
    }

    #[test]
    fn kv8_relative_error_gaussian() {
        // relative RMS error of the int8 copy on N(0,1) groups (reported in the design note): ~0.6%
        let mut r = Rng(42);
        let (mut se, mut ss) = (0.0f64, 0.0f64);
        for _ in 0..4000 {
            let x: Vec<f32> = (0..KV8_GROUP).map(|_| f16r(r.gauss())).collect();
            let (sb, c) = kv8_quant_group(&x);
            for (&v, &u) in x.iter().zip(&c) {
                let d = (v - kv8_dequant(u, sb)) as f64;
                se += d * d;
                ss += v as f64 * v as f64;
            }
        }
        let rel = (se / ss).sqrt();
        eprintln!("kv8 N(0,1) group rel rms {rel:.5}");
        assert!(rel > 0.003 && rel < 0.008, "rel rms {rel}");
    }

    #[test]
    fn kv8_layout_and_bytes() {
        let (hkv, ctx, vt_ctx) = (4usize, 96usize, 128usize);
        let (a, b, c, d) = kv8_layer_elems(hkv, ctx, vt_ctx);
        assert_eq!(a + 2 * b, hkv * ctx * (KV8_HD + 2 * KV8_GROUPS));
        assert_eq!(c + 2 * d, hkv * vt_ctx * (KV8_HD + 2 * KV8_GROUPS));
        assert_eq!(KV8_CELL_BYTES, 528);
        // every index is in range and hit exactly once
        let mut seen = vec![0u8; c];
        let mut seen_s = vec![0u8; d];
        for kvh in 0..hkv {
            for cell in 0..vt_ctx {
                for dd in 0..KV8_HD {
                    seen[kv8_vt_index(kvh, cell, dd, vt_ctx)] += 1;
                }
                for g in 0..KV8_GROUPS {
                    seen_s[kv8_vs_index(kvh, cell, g, vt_ctx)] += 1;
                }
            }
        }
        assert!(seen.iter().all(|&s| s == 1) && seen_s.iter().all(|&s| s == 1));
        // the 16 cells of one dim are contiguous (one 16-byte WMMA operand), and one group's 16 cell scales too
        assert_eq!(
            kv8_vt_index(1, 37, 5, vt_ctx) + 1,
            kv8_vt_index(1, 38, 5, vt_ctx)
        );
        assert_eq!(
            kv8_vs_index(2, 64, 3, vt_ctx) + 15,
            kv8_vs_index(2, 79, 3, vt_ctx)
        );
        // 27B at 32K: 16 attention layers x 4 KV heads; f16 K + vt read per round vs the int8 copy
        let f16_bytes: f64 = 16.0 * 4.0 * 32768.0 * 2.0 * 512.0;
        let kv8_bytes: f64 = 16.0 * 4.0 * 32768.0 * KV8_CELL_BYTES as f64;
        assert!((f16_bytes / 1e9 - 2.147).abs() < 0.01);
        assert!((kv8_bytes / 1e9 - 1.107).abs() < 0.01);
    }

    #[test]
    fn kv8_quant_cells_touches_only_the_range_and_round_trips() {
        let (hkv, ctx, vt_ctx) = (2usize, 80usize, 96usize);
        let mut r = Rng(7);
        let mut kc = vec![0u16; hkv * ctx * KV8_HD];
        let mut vt = vec![0u16; hkv * vt_ctx * KV8_HD];
        for kvh in 0..hkv {
            for cell in 0..ctx {
                for d in 0..KV8_HD {
                    let k = r.gauss() * if d == 17 { 20.0 } else { 1.0 };
                    kc[kv8_kq_index(kvh, cell, d, ctx)] = f32_to_f16_bits(k);
                    vt[kv8_vt_index(kvh, cell, d, vt_ctx)] = f32_to_f16_bits(r.gauss() * 0.3);
                }
            }
        }
        let mut l = Kv8Layer::zeroed(hkv, ctx, vt_ctx);
        kv8_quant_cells(&kc, &vt, hkv, ctx, vt_ctx, 19, 13, &mut l);
        for kvh in 0..hkv {
            for cell in 0..ctx {
                let inside = (19..32).contains(&cell);
                let kr = kv8_k_row(&l, kvh, cell, ctx);
                let vr = kv8_v_row(&l, kvh, cell, vt_ctx);
                for d in 0..KV8_HD {
                    let kq = l.kq[kv8_kq_index(kvh, cell, d, ctx)];
                    if !inside {
                        assert_eq!(kq, 0);
                        continue;
                    }
                    let g = d / KV8_GROUP;
                    let sk = f16_bits_to_f32(l.ks[kv8_ks_index(kvh, cell, g, ctx)]);
                    let sv = f16_bits_to_f32(l.vs[kv8_vs_index(kvh, cell, g, vt_ctx)]);
                    let k = f16_bits_to_f32(kc[kv8_kq_index(kvh, cell, d, ctx)]);
                    let v = f16_bits_to_f32(vt[kv8_vt_index(kvh, cell, d, vt_ctx)]);
                    assert!((k - kr[d]).abs() <= 0.5 * sk * (1.0 + 1e-6));
                    assert!((v - vr[d]).abs() <= 0.5 * sv * (1.0 + 1e-6));
                }
            }
        }
        // a second pass over the same f16 cells gives identical bytes (the restore rebuild)
        let mut l2 = l.clone();
        kv8_quant_cells(&kc, &vt, hkv, ctx, vt_ctx, 19, 13, &mut l2);
        assert_eq!(l, l2);
    }

    #[test]
    fn kv8_attention_error_is_small() {
        // gated attention over 300 cells: int8 vs f16 values, relative output error well under 1%
        let (h, hkv, n) = (6usize, 1usize, 300usize);
        let ctx = n;
        let vt_ctx = 304;
        let mut r = Rng(99);
        let mut kc = vec![0u16; hkv * ctx * KV8_HD];
        let mut vt = vec![0u16; hkv * vt_ctx * KV8_HD];
        for cell in 0..n {
            for d in 0..KV8_HD {
                kc[kv8_kq_index(0, cell, d, ctx)] =
                    f32_to_f16_bits(r.gauss() * if d < 4 { 6.0 } else { 1.0 });
                vt[kv8_vt_index(0, cell, d, vt_ctx)] = f32_to_f16_bits(r.gauss());
            }
        }
        let mut l = Kv8Layer::zeroed(hkv, ctx, vt_ctx);
        kv8_quant_cells(&kc, &vt, hkv, ctx, vt_ctx, 0, n, &mut l);
        let q: Vec<f32> = (0..2 * h * KV8_HD).map(|_| r.gauss() * 0.5).collect();
        let gate: Vec<f32> = (0..2 * h * KV8_HD).map(|_| r.gauss()).collect();
        let sc = 1.0 / 16.0;
        let kf = |_: usize, c: usize| -> Vec<f32> {
            (0..KV8_HD)
                .map(|d| f16_bits_to_f32(kc[kv8_kq_index(0, c, d, ctx)]))
                .collect()
        };
        let vf = |_: usize, c: usize| -> Vec<f32> {
            (0..KV8_HD)
                .map(|d| f16_bits_to_f32(vt[kv8_vt_index(0, c, d, vt_ctx)]))
                .collect()
        };
        let k8 = |_: usize, c: usize| kv8_k_row(&l, 0, c, ctx);
        let v8 = |_: usize, c: usize| kv8_v_row(&l, 0, c, vt_ctx);
        let a = ref_attention(&q, &gate, 2, h, hkv, n - 2, sc, &kf, &vf);
        let b = ref_attention(&q, &gate, 2, h, hkv, n - 2, sc, &k8, &v8);
        let (mut se, mut ss) = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(&b) {
            se += ((x - y) as f64).powi(2);
            ss += (*x as f64).powi(2);
        }
        let rel = (se / ss).sqrt();
        eprintln!("kv8 attention output rel rms {rel:.5}");
        assert!(rel < 0.01, "rel {rel}");
    }

    #[test]
    fn t6_rotation_is_orthonormal_and_inverts() {
        let mut r = Rng(5);
        let x: Vec<f32> = (0..T6_BLOCK).map(|_| r.gauss()).collect();
        let y = t6_rotate(&x, false);
        let n0: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum();
        let n1: f64 = y.iter().map(|&v| (v as f64).powi(2)).sum();
        assert!((n0 - n1).abs() / n0 < 1e-5);
        let z = t6_rotate(&y, true);
        for (a, b) in x.iter().zip(&z) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn t6_codebook_is_lloyd_max() {
        let cb = t6_codebook();
        for i in 0..T6_LEVELS - 1 {
            assert!(cb[i] < cb[i + 1]);
        }
        for i in 0..T6_LEVELS {
            assert!((cb[i] + cb[T6_LEVELS - 1 - i]).abs() < 1e-6);
        }
        // the 6-bit Gaussian Lloyd-Max distortion is about 0.00065 (vs 0.00080 for a uniform 64-level grid)
        let mut r = Rng(11);
        let mut se = 0.0f64;
        let n = 400_000;
        for _ in 0..n {
            let z = r.gauss();
            let q = cb[t6_nearest(z, cb) as usize];
            se += ((z - q) as f64).powi(2);
        }
        let mse = se / n as f64;
        eprintln!("t6 codebook N(0,1) mse {mse:.6}");
        assert!(mse > 0.0005 && mse < 0.0008, "mse {mse}");
    }

    #[test]
    fn t6_round_trip_keeps_norm_and_bounds_error() {
        let mut r = Rng(77);
        let (mut se, mut ss) = (0.0f64, 0.0f64);
        for trial in 0..500 {
            let mut x: Vec<f32> = (0..T6_BLOCK).map(|_| r.gauss()).collect();
            if trial % 3 == 0 {
                x[3] = 12.0; // outlier channel: the rotation spreads it
            }
            let b = t6_encode(&x);
            assert_eq!(b.idx.len(), T6_BLOCK);
            let packed = t6_pack(&b.idx);
            assert_eq!(packed.len() + 2, T6_BLOCK_BYTES);
            assert_eq!(t6_unpack(&packed), b.idx);
            let y = t6_decode(&b);
            let n0: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
            let n1: f64 = y.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
            assert!((n0 - n1).abs() / n0 < 1.5e-3, "norm {n0} vs {n1}"); // fp16 norm rounding
            for (a, c) in x.iter().zip(&y) {
                se += ((a - c) as f64).powi(2);
                ss += (*a as f64).powi(2);
            }
        }
        let rel = (se / ss).sqrt();
        eprintln!("t6 block rel rms {rel:.5}");
        // ~2.6% relative RMS: about 5x the int8 copy's error (design note)
        assert!(rel > 0.015 && rel < 0.04, "rel {rel}");
        assert_eq!(T6_BLOCK_BYTES * 8, 784); // 6.125 bits per value
        let zero = t6_encode(&[0.0; T6_BLOCK]);
        assert!(t6_decode(&zero)
            .iter()
            .all(|&v| v == 0.0 || v.abs() < 1e-30));
    }
}
