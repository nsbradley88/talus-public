//! GLM prefill seam: every window above TMAX tokens runs its dense projections and routed experts through these two
//! methods, so the GLM adapter can switch to the shared gfx1151 prefill subsystem (int8 WMMA dense
//! GEMM, generic grouped MoE GEMM) by replacing their bodies.
//!
//! Dense: `tk_prefill_mmq` (one launch per projection). Routed experts: `tk_prefill_moe_mmq` (grouped-expert tp_moe
//! grouped GEMM over all 288 experts, one launch per product: gate, up, down), MMQ-layout activations from
//! `tk_q36_mmq_quantize` gathered through the device route sort, no host syncs.

use std::ptr;

use talus_core_api::Result;

use super::super::ffi::{self, ck};
use super::super::weights::{FfnW, Tensor};
use super::GlmEngine;

impl GlmEngine {
    /// Dense projection y[t][n_out] = W x[t] for a prefill window (t > TMAX).
    pub(super) fn prefill_dense(
        &self,
        w: &Tensor,
        x: *const f32,
        y: *mut f32,
        t: usize,
    ) -> Result<()> {
        self.mmq_quant(
            x,
            ptr::null(),
            w.ty,
            w.n_in() as usize,
            w.n_in() as usize,
            t,
        )?;
        self.prefill_dense_q(w, y, t)
    }

    /// The GEMM half of `prefill_dense`: x already quantized (MMQ layout for `w.ty`) in mmq_xq.
    pub(super) fn prefill_dense_q(&self, w: &Tensor, y: *mut f32, t: usize) -> Result<()> {
        // SAFETY: tk_* C ABI.
        let stv = unsafe {
            (self.lib.tk.tk_prefill_mmq)(
                self.lib.ctx,
                std::mem::transmute::<i32, ffi::tk_qtype>(w.ty as i32),
                w.p(),
                self.mmq_xq.p,
                y,
                t as i64,
                w.n_out() as i64,
                w.n_in() as i64,
                w.n_out() as i64,
                self.stream as _,
            )
        };
        self.lib.cktk(stv, "tk_prefill_mmq")
    }

    /// Routed experts of a prefill window: route sort, gate/up products, clamped SwiGLU, down product into a.ed
    /// (rows in natural (token, slot) order). Expert types: those `tk_prefill_moe_mmq` takes (IQ2_S / IQ3_S trunk,
    /// Q2_K / Q3_K MTP layer).
    pub(super) fn prefill_experts(
        &self,
        f: &FfnW,
        t: usize,
        k: usize,
        nf: usize,
        clamp_e: f32,
    ) -> Result<()> {
        let hp = &self.w.hp;
        let a = self.a;
        let e = hp.n_embd;
        let gate_t = f.gate_exps.ty;
        let down_t = f.down_exps.ty;
        let rows = t * k;
        // SAFETY: tkg C ABI.
        unsafe {
            ck(
                (self.lib.g.moe_route_sort)(
                    a.eids.i(),
                    t as i32,
                    k as i32,
                    hp.n_expert as i32,
                    self.rowsrc.i(),
                    self.rownat.i(),
                    self.bounds.i(),
                    self.s(),
                ),
                "tkg_moe_route_sort",
            )?;
        }
        // shared prefill engine (tp_moe grouped GEMM; llama MMQ for the MTP layer's Q2_K / Q3_K): bounds stay on the
        // device, max_rows = t is an upper bound (an expert takes at most one row per token; empty blocks exit).
        // eact = eg (in-place SwiGLU) and ed overlays eg + eu: the down product only reads the quantized activations.
        let max_rows = t;
        // the fp16 kernels with the token rows quantized once (the gate/up kernel reads row rowsrc[r]) and the
        // SwiGLU quantized in its epilogue straight into the down product's activations (bitwise the path below)
        if let (Some(gu), Some(dn)) = (self.lib.g.w16_moe_gu_q8, self.lib.g.w16_moe_down) {
            let up = &f.up_exps;
            if up.ty == gate_t
                && up.expert_bytes() == f.gate_exps.expert_bytes()
                && up.n_out() == f.gate_exps.n_out()
            {
                // SAFETY: plain C queries.
                let (xb, hb) = unsafe {
                    (
                        (self.lib.tk.tk_q36_mmq_q8_bytes)(t as i64, e as i64),
                        (self.lib.tk.tk_q36_mmq_q8_bytes)(rows as i64, nf as i64),
                    )
                };
                let off = (xb + 4095) / 4096 * 4096;
                if off + hb <= self.mmq_xq.bytes {
                    self.mmq_quant(a.y.f(), ptr::null(), gate_t, e, e, t)?;
                    let hq = self.mmq_xq.at(off);
                    // SAFETY: tkg C ABI; xq holds t rows, hq rows x nf fits behind them (checked).
                    let r = unsafe {
                        gu(
                            gate_t as i32,
                            f.gate_exps.p(),
                            up.p(),
                            f.gate_exps.expert_bytes(),
                            nf as i32,
                            e as i32,
                            hp.n_expert as i32,
                            self.mmq_xq.p,
                            self.rowsrc.i(),
                            t as i64,
                            self.bounds.i(),
                            rows as i64,
                            hq,
                            clamp_e,
                            self.s(),
                        )
                    };
                    if r > 0 {
                        ck(r, "tkg_w16_moe_gu_q8")?;
                    }
                    if r == 0 {
                        // SAFETY: tkg C ABI (a.ed holds rows x e).
                        let r2 = unsafe {
                            dn(
                                down_t as i32,
                                f.down_exps.p(),
                                f.down_exps.expert_bytes(),
                                e as i32,
                                nf as i32,
                                hp.n_expert as i32,
                                hq,
                                self.bounds.i(),
                                self.rownat.i(),
                                rows as i64,
                                a.ed.f(),
                                e as i64,
                                self.s(),
                            )
                        };
                        if r2 == 0 {
                            return Ok(());
                        }
                        // (the down entry covers what the gate/up entry covers: same thresholds, IQ2_S / IQ3_S)
                        return ck(if r2 > 0 { r2 } else { 1 }, "tkg_w16_moe_down");
                    }
                }
            }
        }
        self.mmq_quant(a.y.f(), self.rowsrc.i(), gate_t, e, e, rows)?;
        // gate, up and SwiGLU in one tp_moe launch when covered (bitwise the separate path below)
        let fused = self.moe_product_gu(
            &f.gate_exps,
            &f.up_exps,
            a.eg.f(),
            nf,
            rows,
            max_rows,
            clamp_e,
        )?;
        if !fused {
            self.moe_product(&f.gate_exps, a.eg.f(), nf, rows, max_rows)?;
            if f.up_exps.ty != gate_t {
                self.mmq_quant(a.y.f(), self.rowsrc.i(), f.up_exps.ty, e, e, rows)?;
            }
            self.moe_product(&f.up_exps, a.eu.f(), nf, rows, max_rows)?;
            // SAFETY: tkg C ABI (in place on eg).
            unsafe {
                ck(
                    (self.lib.g.swiglu_clamp)(
                        a.eg.f(),
                        a.eu.f(),
                        a.eg.f(),
                        (rows * nf) as i32,
                        clamp_e,
                        self.s(),
                    ),
                    "tkg_swiglu_clamp exp",
                )?;
            }
        }
        self.mmq_quant(a.eg.f(), self.rownat.i(), down_t, nf, nf, rows)?;
        self.moe_product(&f.down_exps, a.ed.f(), e, rows, max_rows)?;
        Ok(())
    }
}
