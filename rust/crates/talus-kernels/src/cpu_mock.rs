//! Deterministic CPU reference mock backend (`CpuMockBackend`) implementing [`KernelBackend`].
//!
//! Executes exact host `f32` reference math matching the Strata/Talus GPU kernels:
//! - Weighted RMSNorm (`rms_norm`)
//! - Per-head RMSNorm + NEOX partial RoPE (`head_norm_rope`) and standalone NEOX RoPE (`rope_neox`)
//! - SwiGLU (`silu_mul`), `axpy`, `Q8_0` embedding gather (`embed_q8_0`), softplus head gate (`softplus_head_gate`)
//! - `Q8_1` activation quantization (`quantize_q8_1`), quantized `mmvq` and MoE `mmvq_ids`
//! - `f32_matvec`, `f32_matmul`, top-k sigmoid MoE router (`route_sigmoid_topk`), MoE slot combine (`moe_add_slots`), `argmax`
//! - Causal GQA attention (`attn_decode`, `attn_prefill`) with sliding window (`window`), attention sinks (`sinks`), and query gate (`q_gate`)
//! - Gated DeltaNet recurrent step (`gdn_step`) and multi-token sequence step with per-head RMSNorm + gate (`gdn_seq`)
//! - Dense `prefill_gemm` and quantized `prefill_mmq`

use talus_core_api::{GpuArch, QuantType, Result};

use crate::backend::{
    AttnDecodeArgs, AttnPrefillArgs, GdnSeqArgs, GdnStepArgs, KernelBackend, MmvqIdsArgs,
    PrefillGemmArgs, PrefillMmqArgs,
};
use crate::invalid_arg;
use crate::quant::{
    dequantize_row, dequantize_row_q8_0, dequantize_row_q8_1, f16_to_f32, q8_1_bytes,
    quantize_row_q8_1, row_bytes,
};

/// Minimum router denominator clamp matching `src/kernels/cuda/moe_route.cu` (`6.103515625e-5`, i.e. `2^-14`).
#[allow(clippy::excessive_precision)]
pub const ROUTER_MIN_DENOM: f32 = 6.103_515_625e-5;

/// Deterministic CPU reference mock backend implementing [`KernelBackend`] on host slices.
#[derive(Debug, Clone)]
pub struct CpuMockBackend {
    arch: GpuArch,
}

impl Default for CpuMockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuMockBackend {
    /// Create a [`CpuMockBackend`] reporting [`GpuArch::Gfx1151`] as its target architecture.
    #[inline]
    pub const fn new() -> Self {
        Self {
            arch: GpuArch::Gfx1151,
        }
    }

    /// Create a [`CpuMockBackend`] configured to report `arch`.
    #[inline]
    pub const fn with_arch(arch: GpuArch) -> Self {
        Self { arch }
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0f32 {
        x
    } else {
        (1.0f32 + x.exp()).ln()
    }
}

#[inline]
fn read_kv_elem(pool: &[u8], kv_f16: bool, idx: usize) -> Result<f32> {
    if kv_f16 {
        let off = idx * 2;
        if off + 2 > pool.len() {
            return Err(invalid_arg(format!(
                "KV f16 pool out of bounds at element {} (byte offset {}, len {})",
                idx,
                off,
                pool.len()
            )));
        }
        Ok(f16_to_f32(u16::from_le_bytes([pool[off], pool[off + 1]])))
    } else {
        let off = idx * 4;
        if off + 4 > pool.len() {
            return Err(invalid_arg(format!(
                "KV f32 pool out of bounds at element {} (byte offset {}, len {})",
                idx,
                off,
                pool.len()
            )));
        }
        Ok(f32::from_le_bytes([
            pool[off],
            pool[off + 1],
            pool[off + 2],
            pool[off + 3],
        ]))
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_causal_gqa_attention(
    q: &[f32],
    k_pool: &[u8],
    v_pool: &[u8],
    kv_f16: bool,
    out: &mut [f32],
    n_tok: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    head_dim_v: usize,
    kv_stride: usize,
    p0: usize,
    window: i32,
    scale: f32,
    sinks: Option<&[f32]>,
    q_gate: Option<&[f32]>,
) -> Result<()> {
    if n_head == 0 || n_kv == 0 || !n_head.is_multiple_of(n_kv) {
        return Err(invalid_arg(format!(
            "attention: invalid head counts n_head={} n_kv={}",
            n_head, n_kv
        )));
    }
    if head_dim == 0 || head_dim_v == 0 {
        return Err(invalid_arg(
            "attention: head_dim and head_dim_v must be > 0",
        ));
    }
    if q.len() < n_tok * n_head * head_dim || out.len() < n_tok * n_head * head_dim_v {
        return Err(invalid_arg("attention: q or out buffer too small"));
    }
    if let Some(s) = sinks {
        if s.len() < n_head {
            return Err(invalid_arg("attention: sinks slice smaller than n_head"));
        }
    }
    if let Some(g) = q_gate {
        if g.len() < n_tok * n_head * head_dim_v {
            return Err(invalid_arg(
                "attention: q_gate slice smaller than n_tok * n_head * head_dim_v",
            ));
        }
    }

    let gqa_ratio = n_head / n_kv;

    for t in 0..n_tok {
        let q_pos = p0 + t;
        if q_pos >= kv_stride {
            return Err(invalid_arg(format!(
                "attention: query position {} exceeds kv_stride {}",
                q_pos, kv_stride
            )));
        }
        let start_pos = if window > 0 {
            let w = window as usize;
            (q_pos + 1).saturating_sub(w)
        } else {
            0
        };

        for h in 0..n_head {
            let kv_h = h / gqa_ratio;
            let q_row = &q[(t * n_head + h) * head_dim..(t * n_head + h + 1) * head_dim];
            let out_row =
                &mut out[(t * n_head + h) * head_dim_v..(t * n_head + h + 1) * head_dim_v];

            let span = q_pos - start_pos + 1;
            let mut scores = Vec::with_capacity(span);
            let mut max_score = f32::NEG_INFINITY;

            for kv_pos in start_pos..=q_pos {
                let k_base = (kv_pos * n_kv + kv_h) * head_dim;
                let mut dot = 0.0f32;
                for (d, &qv) in q_row.iter().enumerate() {
                    let kv = read_kv_elem(k_pool, kv_f16, k_base + d)?;
                    dot += qv * kv;
                }
                let s = dot * scale;
                if s > max_score {
                    max_score = s;
                }
                scores.push(s);
            }

            if let Some(sink_vals) = sinks {
                let sink_logit = sink_vals[h];
                if sink_logit > max_score {
                    max_score = sink_logit;
                }
            }

            let mut denom = 0.0f32;
            for s in &mut scores {
                let w = (*s - max_score).exp();
                *s = w;
                denom += w;
            }
            if let Some(sink_vals) = sinks {
                denom += (sink_vals[h] - max_score).exp();
            }

            let inv_denom = if denom > 0.0 { 1.0f32 / denom } else { 0.0f32 };
            out_row.fill(0.0f32);

            for (idx, kv_pos) in (start_pos..=q_pos).enumerate() {
                let prob = scores[idx] * inv_denom;
                let v_base = (kv_pos * n_kv + kv_h) * head_dim_v;
                for (d, out_val) in out_row.iter_mut().enumerate() {
                    let vv = read_kv_elem(v_pool, kv_f16, v_base + d)?;
                    *out_val += prob * vv;
                }
            }

            if let Some(gate_slice) = q_gate {
                let gate_row =
                    &gate_slice[(t * n_head + h) * head_dim_v..(t * n_head + h + 1) * head_dim_v];
                for (out_val, &gv) in out_row.iter_mut().zip(gate_row.iter()) {
                    *out_val *= sigmoid(gv);
                }
            }
        }
    }

    Ok(())
}

impl KernelBackend for CpuMockBackend {
    #[inline]
    fn arch(&self) -> GpuArch {
        self.arch.clone()
    }

    fn rms_norm(
        &self,
        x: &mut [f32],
        w: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        if cols == 0 || x.len() < rows.saturating_mul(cols) {
            return Err(invalid_arg(format!(
                "rms_norm: invalid shape rows={} cols={} x.len()={}",
                rows,
                cols,
                x.len()
            )));
        }
        if let Some(weights) = w {
            if weights.len() < cols {
                return Err(invalid_arg(format!(
                    "rms_norm: w.len() ({}) < cols ({})",
                    weights.len(),
                    cols
                )));
            }
        }

        for r in 0..rows {
            let row = &mut x[r * cols..(r + 1) * cols];
            let mut sum_sq = 0.0f32;
            for &v in row.iter() {
                sum_sq += v * v;
            }
            let mean = sum_sq / (cols as f32);
            let scale = 1.0f32 / (mean + eps).sqrt();
            match w {
                Some(weights) => {
                    for (val, &wv) in row.iter_mut().zip(weights.iter()) {
                        *val = *val * scale * wv;
                    }
                }
                None => {
                    for val in row.iter_mut() {
                        *val *= scale;
                    }
                }
            }
        }
        Ok(())
    }

    fn head_norm_rope(
        &self,
        x: &mut [f32],
        w: &[f32],
        rows: usize,
        eps: f32,
        n_rot: usize,
        cos_tab: &[f32],
        sin_tab: &[f32],
        pos: &[i32],
        pos_div: usize,
    ) -> Result<()> {
        let head_dim = w.len();
        if head_dim == 0 || !n_rot.is_multiple_of(2) || n_rot > head_dim {
            return Err(invalid_arg(format!(
                "head_norm_rope: invalid head_dim={} n_rot={}",
                head_dim, n_rot
            )));
        }
        if x.len() < rows.saturating_mul(head_dim) || pos.is_empty() {
            return Err(invalid_arg("head_norm_rope: x or pos buffer too small"));
        }

        let half = n_rot / 2;
        for r in 0..rows {
            let row = &mut x[r * head_dim..(r + 1) * head_dim];
            let mut sum_sq = 0.0f32;
            for &v in row.iter() {
                sum_sq += v * v;
            }
            let scale = 1.0f32 / (sum_sq / (head_dim as f32) + eps).sqrt();
            for (val, &wv) in row.iter_mut().zip(w.iter()) {
                *val = *val * scale * wv;
            }

            if half > 0 {
                let pos_idx = r.checked_div(pos_div).unwrap_or(0);
                if pos_idx >= pos.len() || pos[pos_idx] < 0 {
                    return Err(invalid_arg(format!(
                        "head_norm_rope: invalid pos index {} (pos.len()={})",
                        pos_idx,
                        pos.len()
                    )));
                }
                let p = pos[pos_idx] as usize;
                let tab_off = p * half;
                if tab_off + half > cos_tab.len() || tab_off + half > sin_tab.len() {
                    return Err(invalid_arg(format!(
                        "head_norm_rope: position {} exceeds cos/sin table capacity",
                        p
                    )));
                }
                for d in 0..half {
                    let a = row[d];
                    let b = row[d + half];
                    let c = cos_tab[tab_off + d];
                    let sn = sin_tab[tab_off + d];
                    row[d] = a * c - b * sn;
                    row[d + half] = a * sn + b * c;
                }
            }
        }
        Ok(())
    }

    fn rope_neox(
        &self,
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        head_dim: usize,
        n_rot: usize,
        cos_tab: &[f32],
        sin_tab: &[f32],
        pos: &[i32],
    ) -> Result<()> {
        if head_dim == 0 || !n_rot.is_multiple_of(2) || n_rot > head_dim {
            return Err(invalid_arg(format!(
                "rope_neox: invalid head_dim={} n_rot={}",
                head_dim, n_rot
            )));
        }
        let total = rows.saturating_mul(head_dim);
        if x.len() < total || out.len() < total || pos.len() < rows {
            return Err(invalid_arg("rope_neox: buffer length too small"));
        }

        let half = n_rot / 2;
        for r in 0..rows {
            if pos[r] < 0 {
                return Err(invalid_arg(format!(
                    "rope_neox: negative position {} at row {}",
                    pos[r], r
                )));
            }
            let p = pos[r] as usize;
            let tab_off = p * half;
            if tab_off + half > cos_tab.len() || tab_off + half > sin_tab.len() {
                return Err(invalid_arg(format!(
                    "rope_neox: position {} exceeds cos/sin table capacity",
                    p
                )));
            }
            let xr = &x[r * head_dim..(r + 1) * head_dim];
            let orow = &mut out[r * head_dim..(r + 1) * head_dim];
            for i in 0..half {
                let a = xr[i];
                let b = xr[half + i];
                let c = cos_tab[tab_off + i];
                let s = sin_tab[tab_off + i];
                orow[i] = a * c - b * s;
                orow[half + i] = a * s + b * c;
            }
            orow[n_rot..head_dim].copy_from_slice(&xr[n_rot..head_dim]);
        }
        Ok(())
    }

    fn silu_mul(&self, gate: &mut [f32], up: &[f32], n: usize) -> Result<()> {
        if gate.len() < n || up.len() < n {
            return Err(invalid_arg("silu_mul: buffer length smaller than n"));
        }
        for i in 0..n {
            gate[i] = silu(gate[i]) * up[i];
        }
        Ok(())
    }

    fn axpy(&self, y: &mut [f32], x: &[f32], alpha: f32, n: usize) -> Result<()> {
        if y.len() < n || x.len() < n {
            return Err(invalid_arg("axpy: buffer length smaller than n"));
        }
        for i in 0..n {
            y[i] += alpha * x[i];
        }
        Ok(())
    }

    fn embed_q8_0(
        &self,
        table: &[u8],
        tokens: &[i32],
        out: &mut [f32],
        n_embd: usize,
    ) -> Result<()> {
        let rbytes = row_bytes(QuantType::Q8_0, n_embd)?;
        let n_tok = tokens.len();
        if out.len() < n_tok.saturating_mul(n_embd) {
            return Err(invalid_arg(
                "embed_q8_0: out buffer smaller than n_tok * n_embd",
            ));
        }
        for (t, &tok) in tokens.iter().enumerate() {
            if tok < 0 {
                return Err(invalid_arg(format!(
                    "embed_q8_0: negative token id {}",
                    tok
                )));
            }
            let row_off = (tok as usize).saturating_mul(rbytes);
            if row_off + rbytes > table.len() {
                return Err(invalid_arg(format!(
                    "embed_q8_0: token id {} out of range for table length {}",
                    tok,
                    table.len()
                )));
            }
            dequantize_row_q8_0(
                &table[row_off..row_off + rbytes],
                &mut out[t * n_embd..(t + 1) * n_embd],
            )?;
        }
        Ok(())
    }

    fn softplus_head_gate(
        &self,
        o: &mut [f32],
        g: &[f32],
        n_head: usize,
        head_dim: usize,
    ) -> Result<()> {
        if o.len() < n_head.saturating_mul(head_dim) || g.len() < n_head {
            return Err(invalid_arg("softplus_head_gate: buffer length too small"));
        }
        for (h, &gv) in g.iter().take(n_head).enumerate() {
            let sp = softplus(gv);
            let head_slice = &mut o[h * head_dim..(h + 1) * head_dim];
            for val in head_slice.iter_mut() {
                *val *= sp;
            }
        }
        Ok(())
    }

    fn quantize_q8_1(
        &self,
        x_f32: &[f32],
        x_q8_1: &mut [u8],
        n_in: usize,
        ncols: usize,
    ) -> Result<()> {
        if n_in == 0 || !n_in.is_multiple_of(32) || ncols == 0 {
            return Err(invalid_arg(format!(
                "quantize_q8_1: invalid n_in={} ncols={}",
                n_in, ncols
            )));
        }
        let col_bytes = q8_1_bytes(n_in, 1);
        if x_f32.len() < n_in * ncols || x_q8_1.len() < col_bytes * ncols {
            return Err(invalid_arg(
                "quantize_q8_1: input or output buffer too small",
            ));
        }
        for col in 0..ncols {
            quantize_row_q8_1(
                &x_f32[col * n_in..(col + 1) * n_in],
                &mut x_q8_1[col * col_bytes..(col + 1) * col_bytes],
            )?;
        }
        Ok(())
    }

    fn mmvq(
        &self,
        qtype: QuantType,
        w: &[u8],
        x_q8_1: &[u8],
        y: &mut [f32],
        n_in: usize,
        n_out: usize,
        ncols: usize,
    ) -> Result<()> {
        let w_row_bytes = row_bytes(qtype, n_in)?;
        let x_col_bytes = q8_1_bytes(n_in, 1);
        if w.len() < w_row_bytes * n_out
            || x_q8_1.len() < x_col_bytes * ncols
            || y.len() < n_out * ncols
        {
            return Err(invalid_arg(
                "mmvq: weight, activation, or output buffer too small",
            ));
        }

        let mut x_dequant = vec![0.0f32; n_in * ncols];
        for col in 0..ncols {
            dequantize_row_q8_1(
                &x_q8_1[col * x_col_bytes..(col + 1) * x_col_bytes],
                &mut x_dequant[col * n_in..(col + 1) * n_in],
            )?;
        }

        let mut w_row = vec![0.0f32; n_in];
        for row in 0..n_out {
            dequantize_row(
                qtype,
                &w[row * w_row_bytes..(row + 1) * w_row_bytes],
                &mut w_row,
            )?;
            for col in 0..ncols {
                let x_col = &x_dequant[col * n_in..(col + 1) * n_in];
                let mut acc = 0.0f64;
                for (&wv, &xv) in w_row.iter().zip(x_col.iter()) {
                    acc += f64::from(wv) * f64::from(xv);
                }
                y[col * n_out + row] = acc as f32;
            }
        }
        Ok(())
    }

    fn mmvq_ids(&self, mut args: MmvqIdsArgs<'_>) -> Result<()> {
        let n_ids = args.ids.len();
        let w_row_bytes = row_bytes(args.qtype, args.n_in)?;
        let min_expert_bytes = w_row_bytes * args.n_out;
        if args.expert_bytes < min_expert_bytes {
            return Err(invalid_arg(format!(
                "mmvq_ids: expert_bytes ({}) < required {} bytes",
                args.expert_bytes, min_expert_bytes
            )));
        }
        if args.y.len() < n_ids * args.n_out {
            return Err(invalid_arg(
                "mmvq_ids: output buffer y smaller than n_ids * n_out",
            ));
        }
        let x_col_bytes = q8_1_bytes(args.n_in, 1);
        let req_x_cols = if args.x_per_id { n_ids } else { 1 };
        if args.x_q8_1.len() < x_col_bytes * req_x_cols {
            return Err(invalid_arg("mmvq_ids: x_q8_1 buffer too small"));
        }

        let mut x_col = vec![0.0f32; args.n_in];
        let mut w_row = vec![0.0f32; args.n_in];

        for (slot, &expert_id) in args.ids.iter().enumerate() {
            let y_slice = &mut args.y[slot * args.n_out..(slot + 1) * args.n_out];
            if expert_id < 0 {
                y_slice.fill(0.0f32);
                if let Some(ref mut y2) = args.y2 {
                    y2[slot * args.n_out..(slot + 1) * args.n_out].fill(0.0f32);
                }
                continue;
            }
            let e = expert_id as usize;
            let e_off = e * args.expert_bytes;
            if e_off + min_expert_bytes > args.w.len() {
                return Err(invalid_arg(format!(
                    "mmvq_ids: expert {} out of bounds for w.len()={}",
                    e,
                    args.w.len()
                )));
            }

            let x_idx = if args.x_per_id { slot } else { 0 };
            dequantize_row_q8_1(
                &args.x_q8_1[x_idx * x_col_bytes..(x_idx + 1) * x_col_bytes],
                &mut x_col,
            )?;

            let w_expert = &args.w[e_off..e_off + min_expert_bytes];
            for (row, out_val) in y_slice.iter_mut().enumerate() {
                dequantize_row(
                    args.qtype,
                    &w_expert[row * w_row_bytes..(row + 1) * w_row_bytes],
                    &mut w_row,
                )?;
                let mut acc = 0.0f64;
                for (&wv, &xv) in w_row.iter().zip(x_col.iter()) {
                    acc += f64::from(wv) * f64::from(xv);
                }
                *out_val = acc as f32;
            }

            if let (Some(w2), Some(y2)) = (args.w2, args.y2.as_deref_mut()) {
                if e_off + min_expert_bytes > w2.len() || y2.len() < n_ids * args.n_out {
                    return Err(invalid_arg("mmvq_ids: w2 or y2 buffer too small"));
                }
                let w2_expert = &w2[e_off..e_off + min_expert_bytes];
                let y2_slice = &mut y2[slot * args.n_out..(slot + 1) * args.n_out];
                for (row, out2_val) in y2_slice.iter_mut().enumerate() {
                    dequantize_row(
                        args.qtype,
                        &w2_expert[row * w_row_bytes..(row + 1) * w_row_bytes],
                        &mut w_row,
                    )?;
                    let mut acc = 0.0f64;
                    for (&wv, &xv) in w_row.iter().zip(x_col.iter()) {
                        acc += f64::from(wv) * f64::from(xv);
                    }
                    *out2_val = acc as f32;
                }
            }
        }

        Ok(())
    }

    fn f32_matvec(
        &self,
        w: &[f32],
        x: &[f32],
        y: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<()> {
        if w.len() < n_out.saturating_mul(n_in) || x.len() < n_in || y.len() < n_out {
            return Err(invalid_arg("f32_matvec: buffer length too small"));
        }
        let x_slice = &x[..n_in];
        for (o, out_val) in y.iter_mut().take(n_out).enumerate() {
            let w_row = &w[o * n_in..(o + 1) * n_in];
            let mut sum = 0.0f64;
            for (&wv, &xv) in w_row.iter().zip(x_slice.iter()) {
                sum += f64::from(wv) * f64::from(xv);
            }
            *out_val = sum as f32;
        }
        Ok(())
    }

    fn f32_matmul(
        &self,
        w: &[f32],
        x: &[f32],
        y: &mut [f32],
        n_in: usize,
        n_out: usize,
        n_tok: usize,
    ) -> Result<()> {
        if w.len() < n_out.saturating_mul(n_in)
            || x.len() < n_tok.saturating_mul(n_in)
            || y.len() < n_tok.saturating_mul(n_out)
        {
            return Err(invalid_arg("f32_matmul: buffer length too small"));
        }
        for t in 0..n_tok {
            let x_row = &x[t * n_in..(t + 1) * n_in];
            let y_row = &mut y[t * n_out..(t + 1) * n_out];
            for (o, out_val) in y_row.iter_mut().enumerate() {
                let w_row = &w[o * n_in..(o + 1) * n_in];
                let mut sum = 0.0f64;
                for (&wv, &xv) in w_row.iter().zip(x_row.iter()) {
                    sum += f64::from(wv) * f64::from(xv);
                }
                *out_val = sum as f32;
            }
        }
        Ok(())
    }

    fn route_sigmoid_topk(
        &self,
        logits: &[f32],
        bias: Option<&[f32]>,
        ids_out: &mut [i32],
        weights_out: &mut [f32],
        n_expert: usize,
        k: usize,
        scale: f32,
    ) -> Result<()> {
        if k == 0
            || k > n_expert
            || logits.len() < n_expert
            || ids_out.len() < k
            || weights_out.len() < k
        {
            return Err(invalid_arg(format!(
                "route_sigmoid_topk: invalid n_expert={} k={}",
                n_expert, k
            )));
        }
        if let Some(b) = bias {
            if b.len() < n_expert {
                return Err(invalid_arg(
                    "route_sigmoid_topk: bias slice smaller than n_expert",
                ));
            }
        }

        let mut probs = vec![0.0f32; n_expert];
        let mut sel_scores = vec![0.0f32; n_expert];
        for e in 0..n_expert {
            let p = sigmoid(logits[e]);
            probs[e] = p;
            sel_scores[e] = p + bias.map_or(0.0f32, |b| b[e]);
        }

        let mut used = vec![false; n_expert];
        let mut sum_p = 0.0f32;
        for out_id in ids_out.iter_mut().take(k) {
            let mut best_e = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for e in 0..n_expert {
                if !used[e] && sel_scores[e] > best_score {
                    best_score = sel_scores[e];
                    best_e = e;
                }
            }
            used[best_e] = true;
            *out_id = best_e as i32;
            sum_p += probs[best_e];
        }

        let denom = sum_p.max(ROUTER_MIN_DENOM);
        for slot in 0..k {
            let e = ids_out[slot] as usize;
            weights_out[slot] = (probs[e] / denom) * scale;
        }
        Ok(())
    }

    fn moe_add_slots(
        &self,
        x: &mut [f32],
        shared: Option<&[f32]>,
        y: &[f32],
        map: Option<&[i32]>,
        weights: &[f32],
        n_tok: usize,
        n_embd: usize,
        k: usize,
    ) -> Result<()> {
        if x.len() < n_tok.saturating_mul(n_embd) || weights.len() < n_tok.saturating_mul(k) {
            return Err(invalid_arg("moe_add_slots: x or weights buffer too small"));
        }
        if let Some(s) = shared {
            if s.len() < n_tok.saturating_mul(n_embd) {
                return Err(invalid_arg(
                    "moe_add_slots: shared buffer smaller than n_tok * n_embd",
                ));
            }
        }
        if let Some(m) = map {
            if m.len() < n_tok.saturating_mul(k) {
                return Err(invalid_arg(
                    "moe_add_slots: map buffer smaller than n_tok * k",
                ));
            }
        }

        for t in 0..n_tok {
            let x_row = &mut x[t * n_embd..(t + 1) * n_embd];
            if let Some(s) = shared {
                let s_row = &s[t * n_embd..(t + 1) * n_embd];
                for (xv, &sv) in x_row.iter_mut().zip(s_row.iter()) {
                    *xv += sv;
                }
            }
            for j in 0..k {
                let idx = t * k + j;
                let w = weights[idx];
                let slot_row = match map {
                    Some(m) => {
                        if m[idx] < 0 {
                            continue;
                        }
                        m[idx] as usize
                    }
                    None => idx,
                };
                let y_off = slot_row * n_embd;
                if y_off + n_embd > y.len() {
                    return Err(invalid_arg(format!(
                        "moe_add_slots: slot row {} out of range for y.len()={}",
                        slot_row,
                        y.len()
                    )));
                }
                let y_row = &y[y_off..y_off + n_embd];
                for (xv, &yv) in x_row.iter_mut().zip(y_row.iter()) {
                    *xv += w * yv;
                }
            }
        }
        Ok(())
    }

    fn argmax(
        &self,
        logits: &[f32],
        n_rows: usize,
        n_vocab: usize,
        out_tokens: &mut [i32],
    ) -> Result<()> {
        if n_vocab == 0
            || logits.len() < n_rows.saturating_mul(n_vocab)
            || out_tokens.len() < n_rows
        {
            return Err(invalid_arg("argmax: invalid dimensions or buffer length"));
        }
        for (r, out_tok) in out_tokens.iter_mut().take(n_rows).enumerate() {
            let row = &logits[r * n_vocab..(r + 1) * n_vocab];
            let mut best_idx = 0usize;
            let mut best_val = row[0];
            for (i, &v) in row.iter().enumerate().skip(1) {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            *out_tok = best_idx as i32;
        }
        Ok(())
    }

    fn attn_decode(&self, args: AttnDecodeArgs<'_>) -> Result<()> {
        if args.pos.is_empty() || args.pos[0] < 0 {
            return Err(invalid_arg("attn_decode: pos[0] must be non-negative"));
        }
        cpu_causal_gqa_attention(
            args.q,
            args.k_pool,
            args.v_pool,
            args.kv_f16,
            args.out,
            args.n_tok,
            args.n_head,
            args.n_kv,
            args.head_dim,
            args.head_dim_v,
            args.kv_stride,
            args.pos[0] as usize,
            args.window,
            args.scale,
            args.sinks,
            args.q_gate,
        )
    }

    fn attn_prefill(&self, args: AttnPrefillArgs<'_>) -> Result<()> {
        cpu_causal_gqa_attention(
            args.q,
            args.k_pool,
            args.v_pool,
            args.kv_f16,
            args.out,
            args.n_tok,
            args.n_head,
            args.n_kv,
            args.head_dim,
            args.head_dim_v,
            args.kv_stride,
            args.p0,
            args.window,
            args.scale,
            args.sinks,
            args.q_gate,
        )
    }

    fn gdn_step(&self, args: GdnStepArgs<'_>) -> Result<()> {
        let s = args.state_size;
        let h_k = args.h_k;
        let h_v = args.h_v;
        if s == 0 || h_k == 0 || h_v == 0 || !h_v.is_multiple_of(h_k) {
            return Err(invalid_arg(format!(
                "gdn_step: invalid dimensions state_size={} h_k={} h_v={}",
                s, h_k, h_v
            )));
        }
        if args.state.len() < s * h_v * s
            || args.q.len() < h_k * s
            || args.k.len() < h_k * s
            || args.v.len() < h_v * s
            || args.gate.len() < h_v
            || args.beta.len() < h_v
            || args.out.len() < h_v * s
        {
            return Err(invalid_arg("gdn_step: buffer length too small"));
        }

        // Matches `gdn_step_kernel` in `src/kernels/cuda/gdn.cu`:
        // `qh = hv % h_k` (modular head mapping) and `q_scale = 1 / sqrt(S)`.
        let q_scale = 1.0f32 / (s as f32).sqrt();

        for hv in 0..h_v {
            let qh = hv % h_k;
            let q_head = &args.q[qh * s..(qh + 1) * s];
            let k_head = &args.k[qh * s..(qh + 1) * s];
            let v_head = &args.v[hv * s..(hv + 1) * s];
            let out_head = &mut args.out[hv * s..(hv + 1) * s];
            let g = args.gate[hv].exp();
            let b = args.beta[hv];

            for c in 0..s {
                let mut kv_col = 0.0f32;
                for (r, &kr) in k_head.iter().enumerate() {
                    let st_idx = (r * h_v + hv) * s + c;
                    kv_col += args.state[st_idx] * kr;
                }
                let delta = (v_head[c] - g * kv_col) * b;
                let mut o_acc = 0.0f32;
                for r in 0..s {
                    let st_idx = (r * h_v + hv) * s + c;
                    let s_new = g * args.state[st_idx] + k_head[r] * delta;
                    args.state[st_idx] = s_new;
                    o_acc += s_new * q_head[r];
                }
                out_head[c] = o_acc * q_scale;
            }
        }
        Ok(())
    }

    fn gdn_seq(&self, args: GdnSeqArgs<'_>) -> Result<()> {
        let s = args.state_size;
        let h_k = args.h_k;
        let h_v = args.h_v;
        let n_tok = args.n_tok;
        if s == 0 || h_k == 0 || h_v == 0 || !h_v.is_multiple_of(h_k) {
            return Err(invalid_arg(format!(
                "gdn_seq: invalid dimensions state_size={} h_k={} h_v={}",
                s, h_k, h_v
            )));
        }
        if args.state.len() < s * h_v * s
            || args.q.len() < n_tok * h_k * s
            || args.k.len() < n_tok * h_k * s
            || args.v.len() < n_tok * h_v * s
            || args.gate.len() < n_tok * h_v
            || args.beta.len() < n_tok * h_v
            || args.z.len() < n_tok * h_v * s
            || (args.gamma.len() != s && args.gamma.len() < h_v * s)
            || args.out.len() < n_tok * h_v * s
        {
            return Err(invalid_arg("gdn_seq: buffer length too small"));
        }

        let mut raw_step = vec![0.0f32; h_v * s];
        for t in 0..n_tok {
            self.gdn_step(GdnStepArgs {
                state: args.state,
                q: &args.q[t * h_k * s..(t + 1) * h_k * s],
                k: &args.k[t * h_k * s..(t + 1) * h_k * s],
                v: &args.v[t * h_v * s..(t + 1) * h_v * s],
                gate: &args.gate[t * h_v..(t + 1) * h_v],
                beta: &args.beta[t * h_v..(t + 1) * h_v],
                out: &mut raw_step,
                state_size: s,
                h_k,
                h_v,
            })?;

            let z_tok = &args.z[t * h_v * s..(t + 1) * h_v * s];
            let out_tok = &mut args.out[t * h_v * s..(t + 1) * h_v * s];
            let per_head_gamma = args.gamma.len() >= h_v * s;

            for hv in 0..h_v {
                let raw_h = &raw_step[hv * s..(hv + 1) * s];
                let z_h = &z_tok[hv * s..(hv + 1) * s];
                let out_h = &mut out_tok[hv * s..(hv + 1) * s];
                let gamma_h = if per_head_gamma {
                    &args.gamma[hv * s..(hv + 1) * s]
                } else {
                    &args.gamma[..s]
                };

                let mut sum_sq = 0.0f32;
                for &v in raw_h {
                    sum_sq += v * v;
                }
                let rstd = 1.0f32 / (sum_sq / (s as f32) + args.norm_eps).sqrt();

                for c in 0..s {
                    let normed = raw_h[c] * rstd * gamma_h[c];
                    let gate_val = if args.silu_gate {
                        silu(z_h[c])
                    } else {
                        sigmoid(z_h[c])
                    };
                    out_h[c] = normed * gate_val;
                }
            }
        }
        Ok(())
    }

    fn prefill_gemm(&self, args: PrefillGemmArgs<'_>) -> Result<()> {
        if args.ldy < args.n_out || args.ldx < args.n_in {
            return Err(invalid_arg("prefill_gemm: ldy < n_out or ldx < n_in"));
        }
        let w_row_bytes = row_bytes(args.w_qtype, args.n_in)?;
        let x_elem_bytes = row_bytes(args.x_qtype, 1)?;
        let x_stride_bytes = args.ldx * x_elem_bytes;
        if args.w.len() < w_row_bytes * args.n_out
            || args.x.len() < x_stride_bytes * args.n_tok
            || args.y.len() < args.ldy * args.n_tok
        {
            return Err(invalid_arg("prefill_gemm: buffer length too small"));
        }

        let mut x_dequant = vec![0.0f32; args.n_tok * args.n_in];
        let x_row_bytes = args.n_in * x_elem_bytes;
        for t in 0..args.n_tok {
            let src_off = t * x_stride_bytes;
            dequantize_row(
                args.x_qtype,
                &args.x[src_off..src_off + x_row_bytes],
                &mut x_dequant[t * args.n_in..(t + 1) * args.n_in],
            )?;
        }

        let mut w_row = vec![0.0f32; args.n_in];
        for o in 0..args.n_out {
            dequantize_row(
                args.w_qtype,
                &args.w[o * w_row_bytes..(o + 1) * w_row_bytes],
                &mut w_row,
            )?;
            for t in 0..args.n_tok {
                let x_row = &x_dequant[t * args.n_in..(t + 1) * args.n_in];
                let mut sum = 0.0f64;
                for (&wv, &xv) in w_row.iter().zip(x_row.iter()) {
                    sum += f64::from(wv) * f64::from(xv);
                }
                let y_idx = t * args.ldy + o;
                if args.beta == 0.0 {
                    args.y[y_idx] = sum as f32;
                } else {
                    args.y[y_idx] = (sum as f32) + args.beta * args.y[y_idx];
                }
            }
        }
        Ok(())
    }

    fn prefill_mmq(&self, args: PrefillMmqArgs<'_>) -> Result<()> {
        if args.ldy < args.n_out {
            return Err(invalid_arg("prefill_mmq: ldy < n_out"));
        }
        let w_row_bytes = row_bytes(args.w_qtype, args.n_in)?;
        let x_col_bytes = q8_1_bytes(args.n_in, 1);
        if args.w_blocks.len() < w_row_bytes * args.n_out
            || args.x_q8_1.len() < x_col_bytes * args.n_tok
            || args.y.len() < args.ldy * args.n_tok
        {
            return Err(invalid_arg("prefill_mmq: buffer length too small"));
        }

        let mut x_dequant = vec![0.0f32; args.n_tok * args.n_in];
        for t in 0..args.n_tok {
            dequantize_row_q8_1(
                &args.x_q8_1[t * x_col_bytes..(t + 1) * x_col_bytes],
                &mut x_dequant[t * args.n_in..(t + 1) * args.n_in],
            )?;
        }

        let mut w_row = vec![0.0f32; args.n_in];
        for o in 0..args.n_out {
            dequantize_row(
                args.w_qtype,
                &args.w_blocks[o * w_row_bytes..(o + 1) * w_row_bytes],
                &mut w_row,
            )?;
            for t in 0..args.n_tok {
                let x_row = &x_dequant[t * args.n_in..(t + 1) * args.n_in];
                let mut sum = 0.0f64;
                for (&wv, &xv) in w_row.iter().zip(x_row.iter()) {
                    sum += f64::from(wv) * f64::from(xv);
                }
                args.y[t * args.ldy + o] = sum as f32;
            }
        }
        Ok(())
    }
}
