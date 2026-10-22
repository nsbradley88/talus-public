//! Pure-Rust CPU reference quantization, dequantization, `f16`/`bf16` conversion,
//! and RoPE table builders used by [`crate::CpuMockBackend`] and unit tests.

use talus_core_api::{QuantType, Result};

use crate::{invalid_arg, unsupported_quant};

/// Convert an IEEE 754 binary16 (`f16`) bit pattern to `f32`.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal f16 -> normalized f32
            let mut m = mant;
            let mut e = 127u32 - 14u32;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        let e = exp + (127 - 15);
        sign | (e << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Convert an `f32` value to an IEEE 754 binary16 (`f16`) bit pattern (round-to-nearest-even).
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        if mant != 0 {
            return sign | 0x7E00;
        }
        return sign | 0x7C00;
    }

    let unbiased = exp - 127;
    if unbiased < -24 {
        return sign;
    }
    if unbiased < -14 {
        let full_mant = mant | 0x0080_0000;
        let shift = (-14 - unbiased) as u32 + 13;
        let round_bit = 1u32 << (shift - 1);
        let mask = round_bit - 1;
        let mut q = full_mant >> shift;
        let rem = full_mant & (round_bit | mask);
        if rem > round_bit || (rem == round_bit && (q & 1) != 0) {
            q += 1;
        }
        return sign | (q as u16);
    }
    if unbiased > 15 {
        return sign | 0x7C00;
    }

    let h_exp = ((unbiased + 15) as u16) << 10;
    let mut h_mant = (mant >> 13) as u16;
    let rem = mant & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (h_mant & 1) != 0) {
        h_mant += 1;
        if h_mant >= 0x0400 {
            let next_exp = ((unbiased + 16) as u16) << 10;
            if unbiased + 16 > 30 {
                return sign | 0x7C00;
            }
            return sign | next_exp;
        }
    }
    sign | h_exp | h_mant
}

/// Convert a `bf16` bit pattern to `f32`.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert an `f32` value to a `bf16` bit pattern (round-to-nearest-even).
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    if (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Return the byte size required for `ncols` columns of `n_in` Q8_1 activations (`36 * (n_in / 32) * ncols`).
#[inline]
pub const fn q8_1_bytes(n_in: usize, ncols: usize) -> usize {
    (n_in / 32) * 36 * ncols
}

/// Return the byte size of one row of `n_in` elements in `qtype`.
pub fn row_bytes(qtype: QuantType, n_in: usize) -> Result<usize> {
    let (bs, type_size) = qtype.block_geometry();
    if n_in == 0 || !n_in.is_multiple_of(bs) {
        return Err(invalid_arg(format!(
            "n_in ({}) must be a positive multiple of {} block_size ({})",
            n_in,
            qtype.name(),
            bs
        )));
    }
    Ok((n_in / bs) * type_size)
}

/// Quantize a contiguous `f32` slice (`n_in` must be a multiple of 32) into `Q8_1` blocks (36 bytes/block).
///
/// Each 36-byte block stores:
/// - `bytes[0..2]`: `f16 d = amax / 127.0`
/// - `bytes[2..4]`: `f16 s = d * sum_{i=0..31} qs[i]`
/// - `bytes[4..36]`: `qs[0..32]` (`i8`)
pub fn quantize_row_q8_1(src: &[f32], dst: &mut [u8]) -> Result<()> {
    if src.is_empty() || !src.len().is_multiple_of(32) {
        return Err(invalid_arg(format!(
            "quantize_row_q8_1: src.len() ({}) must be a positive multiple of 32",
            src.len()
        )));
    }
    let n_blocks = src.len() / 32;
    let expected_bytes = n_blocks * 36;
    if dst.len() < expected_bytes {
        return Err(invalid_arg(format!(
            "quantize_row_q8_1: dst.len() ({}) < required {} bytes",
            dst.len(),
            expected_bytes
        )));
    }

    for (blk_idx, chunk) in src.as_chunks::<32>().0.iter().enumerate() {
        let mut amax = 0.0f32;
        for &v in chunk {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }
        let d = amax / 127.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
        let d_bits = f32_to_f16(d);
        let d_eff = f16_to_f32(d_bits);

        let out_blk = &mut dst[blk_idx * 36..(blk_idx + 1) * 36];
        let mut sum_q = 0i32;
        for (i, &v) in chunk.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            sum_q += i32::from(q);
            out_blk[4 + i] = q as u8;
        }
        let s_bits = f32_to_f16(d_eff * (sum_q as f32));
        out_blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
        out_blk[2..4].copy_from_slice(&s_bits.to_le_bytes());
    }
    Ok(())
}

/// Dequantize a `Q8_1` row (`n_blocks * 36` bytes) into `dst` (`n_blocks * 32` floats).
pub fn dequantize_row_q8_1(src: &[u8], dst: &mut [f32]) -> Result<()> {
    if dst.is_empty() || !dst.len().is_multiple_of(32) {
        return Err(invalid_arg(format!(
            "dequantize_row_q8_1: dst.len() ({}) must be a positive multiple of 32",
            dst.len()
        )));
    }
    let n_blocks = dst.len() / 32;
    let expected_bytes = n_blocks * 36;
    if src.len() < expected_bytes {
        return Err(invalid_arg(format!(
            "dequantize_row_q8_1: src.len() ({}) < required {} bytes",
            src.len(),
            expected_bytes
        )));
    }

    for blk_idx in 0..n_blocks {
        let blk = &src[blk_idx * 36..(blk_idx + 1) * 36];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let out_chunk = &mut dst[blk_idx * 32..(blk_idx + 1) * 32];
        for i in 0..32 {
            let q = blk[4 + i] as i8;
            out_chunk[i] = d * (q as f32);
        }
    }
    Ok(())
}

/// Quantize a contiguous `f32` slice (`n_in` multiple of 32) into `Q8_0` blocks (34 bytes/block).
pub fn quantize_row_q8_0(src: &[f32], dst: &mut [u8]) -> Result<()> {
    if src.is_empty() || !src.len().is_multiple_of(32) {
        return Err(invalid_arg(format!(
            "quantize_row_q8_0: src.len() ({}) must be a positive multiple of 32",
            src.len()
        )));
    }
    let n_blocks = src.len() / 32;
    let expected_bytes = n_blocks * 34;
    if dst.len() < expected_bytes {
        return Err(invalid_arg(format!(
            "quantize_row_q8_0: dst.len() ({}) < required {} bytes",
            dst.len(),
            expected_bytes
        )));
    }

    for (blk_idx, chunk) in src.as_chunks::<32>().0.iter().enumerate() {
        let mut amax = 0.0f32;
        for &v in chunk {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }
        let d = amax / 127.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
        let d_bits = f32_to_f16(d);
        let out_blk = &mut dst[blk_idx * 34..(blk_idx + 1) * 34];
        out_blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
        for (i, &v) in chunk.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            out_blk[2 + i] = q as u8;
        }
    }
    Ok(())
}

/// Dequantize a `Q8_0` row (`n_blocks * 34` bytes) into `dst` (`n_blocks * 32` floats).
pub fn dequantize_row_q8_0(src: &[u8], dst: &mut [f32]) -> Result<()> {
    if dst.is_empty() || !dst.len().is_multiple_of(32) {
        return Err(invalid_arg(format!(
            "dequantize_row_q8_0: dst.len() ({}) must be a positive multiple of 32",
            dst.len()
        )));
    }
    let n_blocks = dst.len() / 32;
    let expected_bytes = n_blocks * 34;
    if src.len() < expected_bytes {
        return Err(invalid_arg(format!(
            "dequantize_row_q8_0: src.len() ({}) < required {} bytes",
            src.len(),
            expected_bytes
        )));
    }

    for blk_idx in 0..n_blocks {
        let blk = &src[blk_idx * 34..(blk_idx + 1) * 34];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let out_chunk = &mut dst[blk_idx * 32..(blk_idx + 1) * 32];
        for i in 0..32 {
            let q = blk[2 + i] as i8;
            out_chunk[i] = d * (q as f32);
        }
    }
    Ok(())
}

/// Quantize a contiguous `f32` slice (`n_in` multiple of 32) into `Q4_0` blocks (18 bytes/block).
pub fn quantize_row_q4_0(src: &[f32], dst: &mut [u8]) -> Result<()> {
    if src.is_empty() || !src.len().is_multiple_of(32) {
        return Err(invalid_arg(format!(
            "quantize_row_q4_0: src.len() ({}) must be a positive multiple of 32",
            src.len()
        )));
    }
    let n_blocks = src.len() / 32;
    let expected_bytes = n_blocks * 18;
    if dst.len() < expected_bytes {
        return Err(invalid_arg(format!(
            "quantize_row_q4_0: dst.len() ({}) < required {} bytes",
            dst.len(),
            expected_bytes
        )));
    }

    for (blk_idx, chunk) in src.as_chunks::<32>().0.iter().enumerate() {
        let mut amax = 0.0f32;
        let mut max_val = 0.0f32;
        for &v in chunk {
            let av = v.abs();
            if av > amax {
                amax = av;
                max_val = v;
            }
        }
        let d = max_val / -8.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
        let d_bits = f32_to_f16(d);
        let out_blk = &mut dst[blk_idx * 18..(blk_idx + 1) * 18];
        out_blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
        for i in 0..16 {
            let x0 = (chunk[i] * id + 8.5f32).floor().clamp(0.0, 15.0) as u8;
            let x1 = (chunk[i + 16] * id + 8.5f32).floor().clamp(0.0, 15.0) as u8;
            out_blk[2 + i] = x0 | (x1 << 4);
        }
    }
    Ok(())
}

/// Dequantize a `Q4_0` row (`n_blocks * 18` bytes) into `dst` (`n_blocks * 32` floats).
pub fn dequantize_row_q4_0(src: &[u8], dst: &mut [f32]) -> Result<()> {
    if dst.is_empty() || !dst.len().is_multiple_of(32) {
        return Err(invalid_arg(format!(
            "dequantize_row_q4_0: dst.len() ({}) must be a positive multiple of 32",
            dst.len()
        )));
    }
    let n_blocks = dst.len() / 32;
    let expected_bytes = n_blocks * 18;
    if src.len() < expected_bytes {
        return Err(invalid_arg(format!(
            "dequantize_row_q4_0: src.len() ({}) < required {} bytes",
            src.len(),
            expected_bytes
        )));
    }

    for blk_idx in 0..n_blocks {
        let blk = &src[blk_idx * 18..(blk_idx + 1) * 18];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let out_chunk = &mut dst[blk_idx * 32..(blk_idx + 1) * 32];
        for i in 0..16 {
            let b = blk[2 + i];
            let q0 = (b & 0x0F) as i32 - 8;
            let q1 = (b >> 4) as i32 - 8;
            out_chunk[i] = (q0 as f32) * d;
            out_chunk[i + 16] = (q1 as f32) * d;
        }
    }
    Ok(())
}

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (sc, m)
    }
}

/// Dequantize a `Q4_K` row (`(dst.len() / 256) * 144` bytes) into `dst`.
pub fn dequantize_row_q4_k(src: &[u8], dst: &mut [f32]) -> Result<()> {
    if dst.is_empty() || !dst.len().is_multiple_of(256) {
        return Err(invalid_arg(format!(
            "dequantize_row_q4_k: dst.len() ({}) must be a positive multiple of 256",
            dst.len()
        )));
    }
    let n_blocks = dst.len() / 256;
    let expected_bytes = n_blocks * 144;
    if src.len() < expected_bytes {
        return Err(invalid_arg(format!(
            "dequantize_row_q4_k: src.len() ({}) < required {} bytes",
            src.len(),
            expected_bytes
        )));
    }

    for blk_idx in 0..n_blocks {
        let blk = &src[blk_idx * 144..(blk_idx + 1) * 144];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&blk[4..16]);
        let qs = &blk[16..144];
        let out_blk = &mut dst[blk_idx * 256..(blk_idx + 1) * 256];

        let mut is = 0usize;
        for chunk_idx in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, &scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
            let d1 = d * (sc1 as f32);
            let min1 = dmin * (m1 as f32);
            let d2 = d * (sc2 as f32);
            let min2 = dmin * (m2 as f32);

            let q_slice = &qs[chunk_idx * 32..(chunk_idx + 1) * 32];
            let out_sub = &mut out_blk[chunk_idx * 64..(chunk_idx + 1) * 64];
            for l in 0..32 {
                out_sub[l] = d1 * ((q_slice[l] & 0x0F) as f32) - min1;
                out_sub[32 + l] = d2 * ((q_slice[l] >> 4) as f32) - min2;
            }
            is += 2;
        }
    }
    Ok(())
}

/// Dequantize one row of `dst.len()` elements from `src` according to `qtype`.
pub fn dequantize_row(qtype: QuantType, src: &[u8], dst: &mut [f32]) -> Result<()> {
    let n = dst.len();
    let expected = row_bytes(qtype, n)?;
    if src.len() < expected {
        return Err(invalid_arg(format!(
            "dequantize_row({}): src.len() ({}) < required {} bytes for {} elements",
            qtype.name(),
            src.len(),
            expected,
            n
        )));
    }
    match qtype {
        QuantType::F32 => {
            for (i, out) in dst.iter_mut().enumerate() {
                let off = i * 4;
                *out = f32::from_le_bytes([src[off], src[off + 1], src[off + 2], src[off + 3]]);
            }
            Ok(())
        }
        QuantType::F16 => {
            for (i, out) in dst.iter_mut().enumerate() {
                let off = i * 2;
                *out = f16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]));
            }
            Ok(())
        }
        QuantType::BF16 => {
            for (i, out) in dst.iter_mut().enumerate() {
                let off = i * 2;
                *out = bf16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]));
            }
            Ok(())
        }
        QuantType::Q8_0 => dequantize_row_q8_0(&src[..expected], dst),
        QuantType::Q8_1 => dequantize_row_q8_1(&src[..expected], dst),
        QuantType::Q4_0 => dequantize_row_q4_0(&src[..expected], dst),
        QuantType::Q4_K => dequantize_row_q4_k(&src[..expected], dst),
        other => Err(unsupported_quant(
            "cpu_mock",
            "dequantize_row",
            other.name(),
        )),
    }
}

/// Build the `(cos_tab, sin_tab)` lookup tables of shape `[max_pos, n_rot / 2]` in `f64`
/// matching `strata::kernels::build_rope_table` in `src/kernels/cuda/rope.cu`.
pub fn build_rope_table(n_rot: usize, theta: f64, max_pos: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    if n_rot == 0 || !n_rot.is_multiple_of(2) {
        return Err(invalid_arg(format!(
            "build_rope_table: n_rot ({}) must be a positive even integer",
            n_rot
        )));
    }
    let half = n_rot / 2;
    let mut cos_tab = vec![0.0f32; max_pos * half];
    let mut sin_tab = vec![0.0f32; max_pos * half];
    for p in 0..max_pos {
        for i in 0..half {
            let inv = theta.powf(-2.0 * (i as f64) / (n_rot as f64));
            let ang = (p as f64) * inv;
            cos_tab[p * half + i] = ang.cos() as f32;
            sin_tab[p * half + i] = ang.sin() as f32;
        }
    }
    Ok((cos_tab, sin_tab))
}
