//! the opt-in 16-bit weight (BF16 / F16) path of the dense qwen35 engine, so a BF16 GGUF (e.g. the
//! Qwen3.8-27B full-precision reference) runs in-engine: Talus as its own reference for quality (KL) and speed.
//!
//! * Kernels: `src/prefill/tp_w16.cu` in libtalus_kernels, resolved as optional symbols (`talus_w16_*`, see
//!   `ffi.rs`). Mat-vec for 1..16 rows with FP32 activations (every row of a window bitwise the 1-row result), WMMA
//!   GEMM for prompts with the activations rounded once to bf16 (llama.cpp's BF16 prompt GEMM does the same).
//! * Routing: [`Q36Engine::mm`] sends BF16 matrices (and F16 at decode width) here; nothing else changes, so every
//!   quantized model takes exactly the code it took before.
//! * Load-time checks: [`check_decode_types`] rejects a model with a matrix no kernel can multiply (before rd-bf16 a
//!   BF16/F16 matrix at decode width left its output unwritten, silently), and [`W16MemPlan`] refuses a load that
//!   would take MemAvailable below the 8 GiB floor.
//! * KL tooling: [`Q36Engine::score_rows`] produces teacher-forced logits of every position of a sequence (the
//!   `talus-q27-ppl` binary streams them to `tools/kldstream.py`).

#![allow(non_snake_case)]

use std::ffi::c_void;

use super::engine::Q36Engine;
use super::ffi::*;
use super::weights::{
    describe_gguf, hip_check, LayerW, Mat, ModelGeometry, Q36Model, Q36Options, GGML_TYPE_BF16,
    GGML_TYPE_F16, GGML_TYPE_F32,
};

const GIB: u64 = 1 << 30;

/// GGML BF16 / F16: the weight types of this path.
#[inline]
pub fn is_w16_type(raw_qt: i32) -> bool {
    raw_qt == GGML_TYPE_BF16 as i32 || raw_qt == GGML_TYPE_F16 as i32
}

/// The kernel family that multiplies a weight of type `raw_qt` at decode width (1..16 rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeKernel {
    /// fp32 weights: `f32mm` / `tk_q36_gemv_f32`
    F32,
    /// quantized weights: `tk_mmvq` (native mat-vec on q8_1 activations)
    Mmvq,
    /// BF16 / F16 weights: `talus_w16_gemv`
    W16,
    /// nothing: the model must not load
    Unsupported,
}

/// Which kernel takes a `raw_qt` weight at decode width, given whether the native mat-vec supports the type and
/// whether the 16-bit kernels are in the loaded libtalus_kernels.
pub fn decode_kernel(raw_qt: i32, mmvq_supported: bool, w16_available: bool) -> DecodeKernel {
    if raw_qt == GGML_TYPE_F32 as i32 {
        DecodeKernel::F32
    } else if mmvq_supported {
        DecodeKernel::Mmvq
    } else if is_w16_type(raw_qt) && w16_available {
        DecodeKernel::W16
    } else {
        DecodeKernel::Unsupported
    }
}

/// Fails on the first named matrix whose type has no decode-width kernel (`mmvq_supported(raw_qt)` is the native
/// mat-vec's type list, `w16_ok(raw_qt, n_out, n_in)` the 16-bit kernels' shape check).
pub fn check_decode_types<'a, I, S, W>(mats: I, mmvq_supported: S, w16_ok: W) -> Result<()>
where
    I: IntoIterator<Item = (String, &'a Mat)>,
    S: Fn(i32) -> bool,
    W: Fn(i32, i32, i32) -> bool,
{
    for (name, m) in mats {
        if m.is_null() {
            continue;
        }
        let qt = m.raw_qtype();
        let w16 = is_w16_type(qt) && w16_ok(qt, m.n_out, m.n_in);
        if decode_kernel(qt, mmvq_supported(qt), w16) == DecodeKernel::Unsupported {
            let why = if is_w16_type(qt) {
                "the 16-bit weight kernels (talus_w16_*, src/prefill/tp_w16.cu) are not in libtalus_kernels or \
                 do not take this shape"
            } else {
                "no decode mat-vec kernel for this weight type"
            };
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "qwen35: matrix {name} (GGML type {qt}, {} x {}) cannot be multiplied at decode width: {why}",
                    m.n_out, m.n_in
                ),
            });
        }
    }
    Ok(())
}

fn layer_mats<'a>(tag: &str, L: &'a LayerW, out: &mut Vec<(String, &'a Mat)>) {
    let named: [(&str, &'a Mat); 13] = [
        ("attn_q", &L.wq_mat),
        ("attn_k", &L.wk_mat),
        ("attn_v", &L.wv_mat),
        ("attn_output", &L.wo_mat),
        ("attn_qkv", &L.wqkv),
        ("attn_gate", &L.wgate),
        ("ssm_out", &L.ssm_out_mat),
        ("ffn_gate", &L.ffn_gate_mat),
        ("ffn_up", &L.ffn_up_mat),
        ("ffn_down", &L.ffn_down_mat),
        ("ffn_gate_shexp", &L.sh_gate),
        ("ffn_up_shexp", &L.sh_up),
        ("ffn_down_shexp", &L.sh_down),
    ];
    for (n, m) in named {
        if !m.is_null() {
            out.push((format!("{tag}.{n}"), m));
        }
    }
}

/// Every matrix the dense / MoE layers, the heads and the MTP block multiply through `Q36Engine::mm` / `tk_mmvq`.
pub fn model_mats(model: &Q36Model) -> Vec<(String, &Mat)> {
    let mut v = Vec::new();
    for (i, L) in model.layers.iter().enumerate() {
        layer_mats(&format!("blk.{i}"), L, &mut v);
    }
    if model.has_mtp {
        layer_mats("mtp", &model.mtp, &mut v);
        v.push(("mtp.eh_proj".to_string(), &model.eh_proj));
        v.push(("mtp.head".to_string(), &model.mtp_head));
    }
    v.push(("output".to_string(), &model.head));
    v.push(("output.draft_vocab".to_string(), &model.dv_head));
    v
}

/// [`check_decode_types`] over a loaded model with the loaded kernel library.
pub fn check_model_decode_types(model: &Q36Model) -> Result<()> {
    check_decode_types(
        model_mats(model),
        tk_q36_mmvq_supported,
        |qt, n_out, n_in| talus_w16_supported(qt, n_out as i64, n_in as i64),
    )
}

/// True when the GGUF's projection matrices include BF16 / F16 ones (the model takes this path).
pub fn model_uses_w16(model: &Q36Model) -> bool {
    model_mats(model)
        .iter()
        .any(|(_, m)| is_w16_type(m.raw_qtype()))
}

/// rd-bf16 memory plan for a 16-bit weight load at `ctx` tokens of context (estimates, GiB-level):
/// weights + the attention caches + context-scaled scratch + a fixed engine overhead, against the MemAvailable floor.
/// The constants come from the shipped UD-IQ4_XS engine measured on the development laptop (MemAvailable at ready 72.6 GiB
/// at 64K, 51.8 GiB at 256K, idle 122.8 GiB): used 50.2 / 71.0 GiB = weights 13.27 + int8 prompt copy 26.05 + MTP
/// sidecar 1.28 + DFlash2 1.06 + KV 102 KiB/token + 2.17 GiB fixed + ~8.9 KB/token of context-scaled scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct W16MemPlan {
    pub ctx: u64,
    pub weights: u64,
    pub kv: u64,
    pub scratch: u64,
    pub fixed: u64,
    pub floor: u64,
}

impl W16MemPlan {
    /// Engine overhead independent of the context (buffers sized by the 512-row batch, contexts, the draft head).
    pub const FIXED: u64 = 2 * GIB + 175 * (1 << 20);
    /// Context-scaled scratch per token beyond the caches (score rows, prompt mask), measured 64K -> 256K.
    pub const SCRATCH_PER_TOKEN: u64 = 8_900;
    /// The MemAvailable floor (agent rules: > 8 GiB always).
    pub const FLOOR: u64 = 8 * GIB;

    /// `weights` = device bytes of the model's tensors; KV = (attention layers + MTP layer) x 3 fp16 copies (K, V,
    /// transposed V) x kv_dim per token.
    pub fn new(g: &ModelGeometry, ctx: u64, weights: u64, has_mtp: bool) -> Self {
        let states = g.n_full_attn_layers() as u64 + u64::from(has_mtp);
        let kv = states * 3 * (g.kv_dim() as u64) * 2 * ctx;
        Self {
            ctx,
            weights,
            kv,
            scratch: Self::SCRATCH_PER_TOKEN * ctx,
            fixed: Self::FIXED,
            floor: Self::FLOOR,
        }
    }

    /// Bytes the engine holds at ready.
    pub fn total(&self) -> u64 {
        self.weights + self.kv + self.scratch + self.fixed
    }

    /// MemAvailable left at ready, from MemAvailable before the load (negative: does not fit).
    pub fn left(&self, mem_available: u64) -> i64 {
        mem_available as i64 - self.total() as i64
    }

    /// Room above the floor (negative: the load would break the floor).
    pub fn headroom(&self, mem_available: u64) -> i64 {
        self.left(mem_available) - self.floor as i64
    }

    pub fn describe(&self, mem_available: u64) -> String {
        let g = |b: u64| (b as f64) / (GIB as f64);
        format!(
            "ctx {}: weights {:.2} + KV {:.2} + scratch {:.2} + fixed {:.2} = {:.2} GiB; MemAvailable {:.1} -> {:.1} GiB \
             at ready (floor {:.0} GiB, headroom {:.1} GiB)",
            self.ctx,
            g(self.weights),
            g(self.kv),
            g(self.scratch),
            g(self.fixed),
            g(self.total()),
            g(mem_available),
            (self.left(mem_available) as f64) / (GIB as f64),
            g(self.floor),
            (self.headroom(mem_available) as f64) / (GIB as f64)
        )
    }
}

/// MemAvailable from /proc/meminfo (bytes), if readable.
pub fn mem_available_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    talus_mem::conversation_mem_available(&s).map(|b| b as u64)
}

/// Before uploading a GGUF whose projections are BF16 / F16: the [`W16MemPlan`] at `opts.max_seq` against
/// MemAvailable now; refuses a load that would leave less than the 8 GiB floor (TALUS_W16_MEMCHECK=0: warn only).
/// Models without 16-bit projections return at once (no change for the quantized models).
pub fn check_gguf_memory(f: &GgufFile, side: Option<&GgufFile>, opts: &Q36Options) -> Result<()> {
    let is16 = |t: &TensorInfo| {
        let id = t.ggml_type_u32();
        t.n_dims >= 2 && (id == GGML_TYPE_BF16 || id == GGML_TYPE_F16) && t.name.starts_with("blk.")
    };
    if !f.tensors().iter().any(is16) {
        return Ok(());
    }
    let g = describe_gguf(f)?;
    let mut weights: u64 = f.tensors().iter().map(|t| t.byte_size).sum();
    if let Some(s) = side {
        weights += s.tensors().iter().map(|t| t.byte_size).sum::<u64>();
    }
    let has_mtp = !opts.no_mtp
        && (side.is_some()
            || f.find_tensor(&format!("blk.{}.nextn.eh_proj.weight", g.n_layer))
                .is_some());
    if has_mtp && !opts.no_draft_vocab {
        // the draft-vocab head: up to 65536 rows copied from the output head
        if let Some(o) = f
            .find_tensor("output.weight")
            .or_else(|| f.find_tensor("token_embd.weight"))
        {
            let rows = o.dims[1].max(1);
            weights += o.byte_size / rows * rows.min(65536);
        }
    }
    let plan = W16MemPlan::new(&g, opts.max_seq.max(0) as u64, weights, has_mtp);
    let Some(avail) = mem_available_bytes() else {
        eprintln!("talus-cli: MemAvailable unreadable; memory plan not checked");
        return Ok(());
    };
    eprintln!(
        "talus-cli: rd-bf16 16-bit weights: {}",
        plan.describe(avail)
    );
    if plan.headroom(avail) < 0 {
        let msg = format!(
            "qwen35: 16-bit weights need more memory than MemAvailable allows above the 8 GiB floor ({}); lower \
             --max-context or free memory (TALUS_W16_MEMCHECK=0 to override)",
            plan.describe(avail)
        );
        if std::env::var("TALUS_W16_MEMCHECK")
            .map(|v| v == "0")
            .unwrap_or(false)
        {
            eprintln!("talus-cli: WARNING: {msg}");
        } else {
            return Err(CoreError::InvalidArgument { reason: msg });
        }
    }
    Ok(())
}

impl Q36Engine {
    /// 16-bit weight product for `Q36Engine::mm` (decode width: mat-vec on the fp32 activations; prompt width: the
    /// WMMA GEMM with `x_f16` as the bf16 / f16 activation scratch). false: not taken (no kernels / shape).
    pub fn w16_mm(&mut self, w: &Mat, x: *const f32, y: *mut f32, n_tok: i32, ldy: i32) -> bool {
        let qt = w.raw_qtype();
        if !is_w16_type(qt) || !talus_w16_available() {
            return false;
        }
        let ldo = if ldy > 0 { ldy } else { w.n_out } as i64;
        let stream = self.s_main.cast::<c_void>();
        let rc = if n_tok > self.gemm_t && !self.x_f16.is_null() {
            talus_w16_gemm(
                qt,
                w.d.cast::<c_void>(),
                w.n_out,
                w.n_in,
                x,
                w.n_in as i64,
                y,
                ldo,
                n_tok,
                self.x_f16.cast::<c_void>(),
                (n_tok as i64) * (w.n_in as i64),
                stream,
            )
        } else {
            talus_w16_gemv(
                qt,
                w.d.cast::<c_void>(),
                w.n_out,
                w.n_in,
                x,
                w.n_in as i64,
                y,
                ldo,
                n_tok,
                stream,
            )
        };
        rc == 0
    }

    /// Teacher-forced logits for KL / PPL: the engine is reset, `tokens[..first]` is prefilled, then the logits of
    /// every position `first .. tokens.len() - 1` (the rows that predict `tokens[first + 1 ..]`, llama-perplexity's
    /// scored half with `first = n_ctx / 2`) are computed in passes of at most `rows` positions (each pass one
    /// prompt forward plus the output head over all its rows) and handed to `sink` one row (n_vocab floats) at a
    /// time, in order. Passes are kept above the decode width so every row takes the prompt kernels.
    pub fn score_rows<F: FnMut(&[f32])>(
        &mut self,
        tokens: &[i32],
        first: usize,
        rows: usize,
        mut sink: F,
    ) -> Result<()> {
        let n = tokens.len();
        ensure!(
            n >= 2 && first >= 1 && first < n - 1,
            "score_rows: need 1 <= first < len - 1"
        );
        ensure!(
            n as i32 <= self.ctx_len,
            "score_rows: sequence longer than the context"
        );
        let v = self.n_vocab() as usize;
        let e = self.d_model() as usize;
        let gt = (self.gemm_t.max(1) + 1) as usize;
        let rows = rows.clamp(gt, self.max_batch.max(gt as i32) as usize);
        let total = n - 1 - first;
        let dbuf = super::engine::hip_alloc_zeroed::<f32>(rows * v, &mut self.allocs)?;
        let mut host = vec![0.0f32; rows * v];

        self.reset();
        self.process(&tokens[..first], false);
        self.flush_gdn();
        let mut done = 0usize;
        while done < total {
            let mut t = (total - done).min(rows);
            if total - done - t > 0 && total - done - t < gt {
                t -= gt; // keep the last pass above the decode width too
            }
            let off = first + done;
            let p0 = self.n_past;
            unsafe {
                let _ = hipStreamSynchronize(self.s_main);
                *self.hpin.add(1) = p0;
                let _ = hipMemcpyAsync(
                    self.tok_d.cast::<c_void>(),
                    tokens.as_ptr().add(off).cast::<c_void>(),
                    t * 4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    self.s_main,
                );
                let _ = hipMemcpyAsync(
                    self.pos0_d.cast::<c_void>(),
                    self.hpin.add(1).cast::<c_void>(),
                    4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    self.s_main,
                );
            }
            self.cur_p0_h = p0;
            self.forward_full(self.tok_d, t as i32, self.pos0_d, 0, false);
            self.cur_p0_h = -1;
            self.head_rows(self.h_last, t as i32, dbuf);
            // the head runs on s_main (a non-blocking stream); a plain hipMemcpy does not wait for it, which read the
            // previous pass's logits (and zeros on the first pass): sync s_main first.
            hip_check(
                unsafe { hipStreamSynchronize(self.s_main) },
                "hipStreamSynchronize(score_rows)",
            )?;
            hip_check(
                unsafe {
                    hipMemcpy(
                        host.as_mut_ptr().cast::<c_void>(),
                        dbuf.cast::<c_void>(),
                        t * v * 4,
                        HIP_MEMCPY_DEVICE_TO_HOST,
                    )
                },
                "hipMemcpy(score_rows)",
            )?;
            for r in 0..t {
                sink(&host[r * v..(r + 1) * v]);
            }
            self.n_past += t as i32;
            done += t;
        }
        let _ = e;
        // the scratch buffer stays in self.allocs (freed with the engine); one per call is fine for a tool
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(qt: u32, n_out: i32, n_in: i32) -> Mat {
        Mat {
            d: 16usize as *const u8,
            qtype: qt as i32,
            n_in,
            n_out,
            w_scale: 1.0,
            ne2: 1,
            bytes: 0,
        }
    }

    // the native mat-vec's type list (tk_q36_mmvq_supported's CPU fallback in ffi.rs)
    fn mmvq(qt: i32) -> bool {
        matches!(
            qt,
            2 | 6 | 8 | 10 | 11 | 12 | 13 | 14 | 16 | 17 | 18 | 20 | 21 | 22 | 23 | 29 | 42
        )
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mem_available_reads_proc_meminfo() {
        let a = mem_available_bytes().expect("MemAvailable from /proc/meminfo");
        assert!(a > (1 << 20));
    }

    #[test]
    fn decode_kernel_routes_every_type() {
        assert_eq!(decode_kernel(0, false, false), DecodeKernel::F32);
        assert_eq!(decode_kernel(23, true, false), DecodeKernel::Mmvq);
        assert_eq!(decode_kernel(30, false, true), DecodeKernel::W16);
        assert_eq!(decode_kernel(1, false, true), DecodeKernel::W16);
        // before rd-bf16 these fell through Q36Engine::mm without writing the output
        assert_eq!(decode_kernel(30, false, false), DecodeKernel::Unsupported);
        assert_eq!(decode_kernel(1, false, false), DecodeKernel::Unsupported);
        assert_eq!(decode_kernel(3, false, true), DecodeKernel::Unsupported); // Q4_1: no kernel at all
        for qt in 0..64 {
            let k = decode_kernel(qt, mmvq(qt), true);
            if qt != 0 && !mmvq(qt) && !is_w16_type(qt) {
                assert_eq!(k, DecodeKernel::Unsupported, "type {qt}");
            } else {
                assert_ne!(k, DecodeKernel::Unsupported, "type {qt}");
            }
        }
    }

    #[test]
    fn load_check_rejects_bf16_without_kernels_and_accepts_with() {
        let q = mat(23, 5120, 5120);
        let b = mat(30, 17408, 5120);
        let h = mat(1, 1024, 5120);
        let null = Mat::null();
        let list = || {
            vec![
                ("blk.0.attn_qkv".to_string(), &q),
                ("blk.0.ffn_up".to_string(), &b),
                ("mtp.attn_k".to_string(), &h),
                ("output.draft_vocab".to_string(), &null),
            ]
        };
        // no 16-bit kernels in the library: the BF16 matrix is named in the error
        let err = check_decode_types(list(), mmvq, |_, _, _| false).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("blk.0.ffn_up") && msg.contains("type 30"),
            "{msg}"
        );
        // kernels present: loads
        assert!(check_decode_types(list(), mmvq, |_, _, k| k % 8 == 0).is_ok());
        // kernels present but the shape is refused (n_in not a multiple of 8)
        let odd = mat(30, 64, 5121);
        let err = check_decode_types(vec![("x".to_string(), &odd)], mmvq, |_, _, k| k % 8 == 0)
            .unwrap_err();
        assert!(format!("{err}").contains("5121"));
        // quantized-only models are unaffected
        assert!(check_decode_types(vec![("q".to_string(), &q)], mmvq, |_, _, _| false).is_ok());
    }

    fn geom_27b() -> ModelGeometry {
        super::super::weights::reference_dense()
    }

    #[test]
    fn mem_plan_27b_bf16_64k_and_256k() {
        let g = geom_27b();
        assert_eq!(g.n_full_attn_layers(), 16);
        assert_eq!(g.kv_dim(), 1024);
        // unsloth BF16 shards: 49,986,159,616 + 4,671,576,000 bytes (headers included; tensors ~ the same)
        let w = 49_986_159_616u64 + 4_671_576_000u64 + 625 * (1 << 20); // + the 64K-row bf16 draft-vocab head
        let idle = (122.8 * GIB as f64) as u64; // MemAvailable with nothing loaded (development laptop)
        let p64 = W16MemPlan::new(&g, 65536, w, true);
        let p256 = W16MemPlan::new(&g, 262144, w, true);
        // KV: 17 states x 3 x 1024 x 2 B = 102 KiB per token (rd-q27fp's breakdown)
        assert_eq!(p64.kv, 102 * 1024 * 65536);
        assert_eq!(p256.kv, 102 * 1024 * 262144);
        let h64 = p64.headroom(idle) as f64 / GIB as f64;
        let h256 = p256.headroom(idle) as f64 / GIB as f64;
        // the floor holds with a wide margin at 64K and still at 256K
        assert!(
            h64 > 40.0,
            "64K headroom {h64:.1} GiB: {}",
            p64.describe(idle)
        );
        assert!(
            h256 > 20.0,
            "256K headroom {h256:.1} GiB: {}",
            p256.describe(idle)
        );
        assert!(h64 - h256 > 19.0 && h64 - h256 < 22.0);
        // and refuses when it does not fit
        let tight = W16MemPlan::new(&g, 65536, w, true);
        assert!(tight.headroom(60 * GIB) < 0);
    }
}
