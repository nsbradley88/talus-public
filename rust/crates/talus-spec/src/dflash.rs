//! DFlash / DFlash2 block-diffusion drafter (`DFlash`, `DFlashTargetInfo`, `DFlashTargetView`)
//! ported from `src/qwen35/q36_dflash_model.{hpp,cpp}` and implementing [`DeviceDrafter`].

use core::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use talus_core_api::{
    CoreError, DeviceDrafter, DeviceId, DraftVocab as CoreDraftVocab, GraphBuilder, HiddenTaps,
    QuantType, Result, RoundIo, SeqState, StreamHandle,
};
use talus_gguf::{GgufFile, GgufValue};
use talus_hip::{DeviceBuf, HostMapped};
use talus_kernels::ffi::{quant_type_to_tk, translate_tk_status};
use talus_kernels_sys::{tk_ctx, LibTalusKernels};

#[inline]
fn inv_arg(reason: impl Into<String>) -> CoreError {
    CoreError::InvalidArgument {
        reason: reason.into(),
    }
}

/// Number of tapped target-model layers fused into the DFlash conditioning input.
pub const DFLASH_TAPS: usize = 5;

/// Number of top-K candidates per step used in DFlash2 dynamic programming chain selection.
pub const DFLASH_TOP_K: usize = 16;

/// Target model tensor pointers and geometry shared with [`DFlash`] without coupling to
/// `talus-model`'s concrete `Model` struct.
#[derive(Debug, Clone, Copy)]
pub struct DFlashTargetInfo {
    /// Target vocabulary size (`n_vocab`).
    pub n_vocab: i32,
    /// Target hidden dimension (`d_model`).
    pub d_model: i32,
    /// Target number of transformer layers (`n_layer`).
    pub n_layer: i32,
    /// Device pointer to target `token_embd.weight`.
    pub tok_embd_ptr: *const c_void,
    /// GGML quantization type ID of `token_embd.weight`.
    pub tok_embd_qtype: u32,
    /// Device pointer to target `output_norm.weight` (`f32`), if needed.
    pub output_norm_ptr: *const f32,
    /// Device pointer to target `output.weight`.
    pub output_ptr: *const c_void,
    /// GGML quantization type ID of `output.weight`.
    pub output_qtype: u32,
}

// Safety: `DFlashTargetInfo` holds immutable device pointers owned by the target model.
unsafe impl Send for DFlashTargetInfo {}
unsafe impl Sync for DFlashTargetInfo {}

/// Alias matching the WP-2b specification (`DFlashTargetView`).
pub type DFlashTargetView = DFlashTargetInfo;

/// Stochastic sampling parameters for DFlash2 chain selection (`DFlash::Samp`).
#[derive(Debug, Clone, Copy)]
pub struct DFlashSamp {
    /// Device pointer to current sampling temperature (`f32`).
    pub tau: *const f32,
    /// Device pointer to Philox RNG seed (`u32`).
    pub seed: *const u32,
    /// Device pointer to Philox round counter (`u32`).
    pub ctr: *const u32,
}

/// Device weight matrix descriptor (`DFlash::W` in `q36_dflash_model.hpp`).
#[derive(Debug)]
pub struct DFlashWeight {
    /// Device buffer holding the tensor bytes.
    pub buf: DeviceBuf<u8>,
    /// Quantization type of the matrix.
    pub qtype: QuantType,
    /// Input feature dimension (`ne[0]`).
    pub cols: i32,
    /// Output feature dimension (`ne[1]`).
    pub rows: i32,
}

impl DFlashWeight {
    /// Raw device pointer to the matrix weights.
    #[inline]
    pub fn as_ptr(&self) -> *const c_void {
        self.buf.as_ptr() as *const c_void
    }
}

/// Dynamic depthwise convolution block (`DFlash::DynConv` in `q36_dflash_model.hpp`).
#[derive(Debug)]
pub struct DFlashDynConv {
    /// Dynamic kernel projection weight (`d_m -> group * kernel_size`).
    pub weight: DFlashWeight,
    /// Static base depthwise kernel (`f32`).
    pub base: DeviceBuf<f32>,
    /// Number of channels per group (`group = d_m / head`).
    pub group: i32,
    /// Half-width `(kernel_size - 1) / 2`.
    pub side: i32,
}

/// Per-layer weights and KV ring buffers for a DFlash block-diffusion layer (`DFlash::Layer`).
#[derive(Debug)]
pub struct DFlashLayer {
    /// Attention pre-norm weights (`f32`, `[d_m]`).
    pub attn_norm: DeviceBuf<f32>,
    /// Post-attention norm weights (`f32`, `[d_m]`), present on DFlash2 models.
    pub attn_post_norm: Option<DeviceBuf<f32>>,
    /// Optional dynamic convolution before Q projection (DFlash2).
    pub attn_conv: Option<DFlashDynConv>,
    /// Query projection (`[d_m, n_q * d_h]`).
    pub wq: DFlashWeight,
    /// Key projection (`[d_m, n_kv * d_h]`).
    pub wk: DFlashWeight,
    /// Value projection (`[d_m, n_kv * d_h]`).
    pub wv: DFlashWeight,
    /// Output projection (`[n_q * d_h, d_m]`).
    pub wo: DFlashWeight,
    /// Query per-head RMSNorm weights (`f32`, `[d_h]`).
    pub q_norm: DeviceBuf<f32>,
    /// Key per-head RMSNorm weights (`f32`, `[d_h]`).
    pub k_norm: DeviceBuf<f32>,
    /// FFN pre-norm weights (`f32`, `[d_m]`).
    pub ffn_norm: DeviceBuf<f32>,
    /// Post-FFN norm weights (`f32`, `[d_m]`), present on DFlash2 models.
    pub ffn_post_norm: Option<DeviceBuf<f32>>,
    /// Gate projection (`[d_m, d_ff]`).
    pub gate: DFlashWeight,
    /// Up projection (`[d_m, d_ff]`).
    pub up: DFlashWeight,
    /// Optional dynamic convolution inside FFN (DFlash2).
    pub ffn_conv: Option<DFlashDynConv>,
    /// Down projection (`[d_ff, d_m]`).
    pub down: DFlashWeight,
    /// Key cache ring buffer (`f32`, `[max_kv * n_kv * d_h]`).
    pub kring: DeviceBuf<f32>,
    /// Value cache ring buffer (`f32`, `[max_kv * n_kv * d_h]`).
    pub vring: DeviceBuf<f32>,
}

/// Full DFlash / DFlash2 block-diffusion drafter (`struct DFlash` in `q36_dflash_model.hpp`).
#[allow(non_snake_case)]
pub struct DFlash {
    /// Number of drafter layers (`n_L`).
    pub n_L: i32,
    /// Hidden dimension (`d_m`).
    pub d_m: i32,
    /// Feed-forward intermediate dimension (`d_ff`).
    pub d_ff: i32,
    /// Number of query attention heads (`n_q`).
    pub n_q: i32,
    /// Number of key/value attention heads (`n_kv`).
    pub n_kv: i32,
    /// Head dimension (`d_h`).
    pub d_h: i32,
    /// Target vocabulary size (`n_v`).
    pub n_v: i32,
    /// Draft block size (`B`, e.g. 16 -> `B - 1` draft tokens per block).
    pub B: i32,
    /// Ring buffer / sliding window KV capacity (`max_kv`).
    pub max_kv: i32,
    /// Attention sliding window size (`window_`).
    pub window: i32,
    /// RMSNorm epsilon (`eps`).
    pub eps: f32,
    /// RoPE frequency base (`rope_base`).
    pub rope_base: f32,
    /// Indices of tapped target layers (`target_layers`, up to 5 layers).
    pub target_layers: Vec<i32>,
    /// Mask token ID (`dflash.mask_token_id`).
    pub mask_token_id: i32,
    /// Current KV cache length (`kv_len`).
    pub kv_len: i32,
    /// Total speculation rounds executed.
    pub n_rounds: i64,
    /// Total tokens drafted across all rounds.
    pub n_drafted: i64,
    /// Total drafted tokens accepted by the target model.
    pub n_accepted: i64,

    /// Target model view (shared `token_embd` and `output` pointers).
    pub target: DFlashTargetInfo,
    /// Target `token_embd` row stride in bytes.
    tok_embd_row_bytes: usize,
    /// Target `output` quantization type.
    output_quant_type: QuantType,

    /// Logical HIP device identifier.
    device: DeviceId,
    /// Dynamically loaded `libtalus_kernels.so` function table.
    lib: Arc<LibTalusKernels>,
    /// Owned `tk_ctx` handle for kernel invocations.
    ctx: *mut tk_ctx,

    /// Per-tap target feature projection matrices (`fc.{0..4}.weight`).
    fc: Vec<DFlashWeight>,
    /// Fused target feature encoder RMSNorm (`enc_norm.weight`).
    enc_norm: DeviceBuf<f32>,
    /// Final drafter output RMSNorm (`output_norm.weight`).
    out_norm: DeviceBuf<f32>,
    /// DFlash layers (`blk.0..n_L`).
    layers: Vec<DFlashLayer>,

    /// Optional DFlash2 chain-selection hidden gate (`select_hidden.weight`).
    sel_hidden: Option<DFlashWeight>,
    /// Optional DFlash2 predecessor bilinear embedding (`select_pred.weight`, `F16`).
    pred: Option<DeviceBuf<u8>>,
    /// Optional DFlash2 successor bilinear embedding (`select_succ.weight`, `F16`).
    succ: Option<DeviceBuf<u8>>,
    /// Optional DFlash2 transition vocabulary mapping (`select_map.weight`, `I32`).
    vmap: Option<DeviceBuf<i32>>,
    /// Size of the DFlash2 transition vocabulary (`n_trans_`).
    n_trans: i32,

    /// Device pointer array to `taps[0..5]` buffers (`taps_dev_`).
    taps_: TapBuffers,
    /// Scratch buffers for block-diffusion execution (`alloc`).
    y_norm: Option<DeviceBuf<f32>>,
    y_proj: Option<DeviceBuf<f32>>,
    x: Option<DeviceBuf<f32>>,
    r_norm: Option<DeviceBuf<f32>>,
    conv: Option<DeviceBuf<f32>>,
    dyn_w: Option<DeviceBuf<f32>>,
    q: Option<DeviceBuf<f32>>,
    k: Option<DeviceBuf<f32>>,
    v: Option<DeviceBuf<f32>>,
    attn_part: Option<DeviceBuf<f32>>,
    attn_out: Option<DeviceBuf<f32>>,
    gate_buf: Option<DeviceBuf<f32>>,
    up_buf: Option<DeviceBuf<f32>>,
    logits: Option<DeviceBuf<f32>>,
    q8: Option<DeviceBuf<u8>>,
    pos: Option<DeviceBuf<i32>>,
    pos_zero: Option<DeviceBuf<i32>>,
    blk_tok: Option<DeviceBuf<i32>>,
    cand: Option<DeviceBuf<i32>>,
    unary: Option<DeviceBuf<f32>>,
    sel_gate: Option<DeviceBuf<f32>>,
    score: Option<DeviceBuf<f32>>,
    q_sel: Option<DeviceBuf<f32>>,
    hq: Option<HostMapped<f32>>,
    hcand: Option<HostMapped<i32>>,

    /// Optional core draft vocabulary handle for `DeviceDrafter::vocab()`.
    draft_vocab: Option<CoreDraftVocab>,
}

// Helper struct for tapped hidden state buffers and their device pointer table.
struct TapBuffers {
    bufs: Vec<DeviceBuf<f32>>,
    ptrs: DeviceBuf<u64>,
}

// Safety: `DFlash` owns its device buffers and `tk_ctx` and is accessed single-threaded per stream.
unsafe impl Send for DFlash {}
unsafe impl Sync for DFlash {}

#[allow(non_snake_case)]
impl DFlash {
    /// Load a DFlash / DFlash2 drafter GGUF model and bind it to `target`'s shared
    /// `token_embd` and `output` weights.
    pub fn load(
        gguf_path: &Path,
        target: &DFlashTargetInfo,
        block_size: i32,
        max_kv_len: i32,
    ) -> Result<Self> {
        let gguf = GgufFile::open(gguf_path).map_err(|e| CoreError::IoError {
            message: e.to_string(),
        })?;
        let device = DeviceId(0);
        let lib = Arc::new(load_kernels_lib()?);

        let ctx = unsafe {
            let mut c: *mut tk_ctx = core::ptr::null_mut();
            let st = (lib.tk_ctx_create)(device.0 as i32, core::ptr::null(), &mut c);
            translate_tk_status(&lib, core::ptr::null(), st)?;
            c
        };

        let load_res = Self::load_inner(
            &gguf,
            target,
            block_size,
            max_kv_len,
            device,
            Arc::clone(&lib),
            ctx,
        );
        match load_res {
            Ok(dflash) => Ok(dflash),
            Err(e) => {
                unsafe {
                    (lib.tk_ctx_destroy)(ctx);
                }
                Err(e)
            }
        }
    }

    fn load_inner(
        gguf: &GgufFile,
        target: &DFlashTargetInfo,
        block_size: i32,
        max_kv_len: i32,
        device: DeviceId,
        lib: Arc<LibTalusKernels>,
        ctx: *mut tk_ctx,
    ) -> Result<Self> {
        let arch = gguf.get_str("general.architecture").unwrap_or("dflash");
        if arch != "dflash" {
            return Err(inv_arg(format!(
                "DFlash::load expected general.architecture == 'dflash', got '{arch}'"
            )));
        }

        let d_m = get_i32_meta(gguf, "dflash.embedding_length")?;
        let d_ff = get_i32_meta(gguf, "dflash.feed_forward_length")?;
        let n_L = get_i32_meta(gguf, "dflash.block_count")?;
        let n_q = get_i32_meta(gguf, "dflash.attention.head_count")?;
        let n_kv = get_i32_meta(gguf, "dflash.attention.head_count_kv")?;
        let d_h = get_i32_meta(gguf, "dflash.attention.key_length")?;
        let eps = get_f32_meta(gguf, "dflash.attention.layer_norm_rms_epsilon")?;
        let rope_base = get_f32_meta(gguf, "dflash.rope.freq_base")?;
        let gguf_block_size = get_i32_meta(gguf, "dflash.block_size").unwrap_or(16);
        let B = if block_size > 0 {
            block_size
        } else {
            gguf_block_size
        };
        let mask_token_id = get_i32_meta(gguf, "dflash.mask_token_id")?;
        let window_meta = get_i32_meta(gguf, "dflash.sliding_window").unwrap_or(512);
        let max_kv = if max_kv_len > 0 {
            max_kv_len
        } else {
            window_meta
        };
        let window = window_meta.min(max_kv);

        if d_m != target.d_model {
            return Err(inv_arg(format!(
                "DFlash embedding_length ({d_m}) != target d_model ({})",
                target.d_model
            )));
        }

        let target_layers = get_i32_array_meta(gguf, "dflash.target_layer_ids")?;
        if target_layers.len() != DFLASH_TAPS {
            return Err(inv_arg(format!(
                "DFlash expected {DFLASH_TAPS} target_layer_ids, got {}",
                target_layers.len()
            )));
        }
        for &tl in &target_layers {
            if tl < 0 || tl >= target.n_layer {
                return Err(inv_arg(format!(
                    "DFlash target_layer_id {tl} out of range 0..{}",
                    target.n_layer
                )));
            }
        }

        let tok_embd_qt = QuantType::from_ggml_type_id(target.tok_embd_qtype).ok_or_else(|| {
            inv_arg(format!(
                "unsupported target tok_embd qtype {}",
                target.tok_embd_qtype
            ))
        })?;
        let (blck, ts) = tok_embd_qt.block_geometry();
        let tok_embd_row_bytes = ((d_m as usize) / blck) * ts;
        let output_quant_type =
            QuantType::from_ggml_type_id(target.output_qtype).ok_or_else(|| {
                inv_arg(format!(
                    "unsupported target output qtype {}",
                    target.output_qtype
                ))
            })?;

        let mut fc = Vec::with_capacity(DFLASH_TAPS);
        for t in 0..DFLASH_TAPS {
            fc.push(upload_weight(
                gguf,
                device,
                &format!("fc.{t}.weight"),
                d_m,
                d_m,
            )?);
        }
        let enc_norm = upload_f32_vec(gguf, device, "enc_norm.weight", d_m as usize)?;
        let out_norm = upload_f32_vec(gguf, device, "output_norm.weight", d_m as usize)?;

        let mut layers = Vec::with_capacity(n_L as usize);
        for l in 0..(n_L as usize) {
            let pfx = format!("blk.{l}.");
            let attn_norm = upload_f32_vec(
                gguf,
                device,
                &format!("{pfx}attn_norm.weight"),
                d_m as usize,
            )?;
            let attn_post_norm = upload_f32_vec_opt(
                gguf,
                device,
                &format!("{pfx}attn_post_norm.weight"),
                d_m as usize,
            )?;
            let attn_conv =
                upload_dyn_conv_opt(gguf, device, &format!("{pfx}attn_conv"), d_m, d_m)?;
            let wq = upload_weight(gguf, device, &format!("{pfx}attn_q.weight"), d_m, n_q * d_h)?;
            let wk = upload_weight(
                gguf,
                device,
                &format!("{pfx}attn_k.weight"),
                d_m,
                n_kv * d_h,
            )?;
            let wv = upload_weight(
                gguf,
                device,
                &format!("{pfx}attn_v.weight"),
                d_m,
                n_kv * d_h,
            )?;
            let wo = upload_weight(
                gguf,
                device,
                &format!("{pfx}attn_output.weight"),
                n_q * d_h,
                d_m,
            )?;
            let q_norm = upload_f32_vec(
                gguf,
                device,
                &format!("{pfx}attn_q_norm.weight"),
                d_h as usize,
            )?;
            let k_norm = upload_f32_vec(
                gguf,
                device,
                &format!("{pfx}attn_k_norm.weight"),
                d_h as usize,
            )?;
            let ffn_norm =
                upload_f32_vec(gguf, device, &format!("{pfx}ffn_norm.weight"), d_m as usize)?;
            let ffn_post_norm = upload_f32_vec_opt(
                gguf,
                device,
                &format!("{pfx}ffn_post_norm.weight"),
                d_m as usize,
            )?;
            let gate = upload_weight(gguf, device, &format!("{pfx}ffn_gate.weight"), d_m, d_ff)?;
            let up = upload_weight(gguf, device, &format!("{pfx}ffn_up.weight"), d_m, d_ff)?;
            let ffn_conv =
                upload_dyn_conv_opt(gguf, device, &format!("{pfx}ffn_conv"), d_ff, d_ff)?;
            let down = upload_weight(gguf, device, &format!("{pfx}ffn_down.weight"), d_ff, d_m)?;

            let kv_elems = (max_kv as usize) * (n_kv as usize) * (d_h as usize);
            let kring = DeviceBuf::<f32>::zeros(device, kv_elems)?;
            let vring = DeviceBuf::<f32>::zeros(device, kv_elems)?;

            layers.push(DFlashLayer {
                attn_norm,
                attn_post_norm,
                attn_conv,
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                ffn_norm,
                ffn_post_norm,
                gate,
                up,
                ffn_conv,
                down,
                kring,
                vring,
            });
        }

        // Optional DFlash2 chain-selection weights.
        let (sel_hidden, pred, succ, vmap, n_trans) =
            if let Some(ti) = gguf.find_tensor("select_pred.weight") {
                let r_sel = ti.dims[0] as i32;
                let n_tr = ti.dims[1] as i32;
                if r_sel != 64 {
                    return Err(inv_arg(format!(
                        "DFlash select_pred rank must be 64, got {r_sel}"
                    )));
                }
                let sh = upload_weight(gguf, device, "select_hidden.weight", d_m, r_sel)?;
                let pr = upload_raw_tensor(gguf, device, "select_pred.weight")?;
                let su = upload_raw_tensor(gguf, device, "select_succ.weight")?;
                let vm = if gguf.find_tensor("select_map.weight").is_some() {
                    Some(upload_i32_vec(
                        gguf,
                        device,
                        "select_map.weight",
                        target.n_vocab as usize,
                    )?)
                } else {
                    None
                };
                (Some(sh), Some(pr), Some(su), vm, n_tr)
            } else {
                (None, None, None, None, 0)
            };

        let mut tap_bufs = Vec::with_capacity(DFLASH_TAPS);
        let mut tap_ptrs_host = [0u64; DFLASH_TAPS];
        for i in 0..DFLASH_TAPS {
            let b = DeviceBuf::<f32>::zeros(device, d_m as usize)?;
            tap_ptrs_host[i] = b.device_ptr();
            tap_bufs.push(b);
        }
        let tap_ptrs = DeviceBuf::<u64>::from_slice(device, &tap_ptrs_host)?;

        let mut dflash = Self {
            n_L,
            d_m,
            d_ff,
            n_q,
            n_kv,
            d_h,
            n_v: target.n_vocab,
            B,
            max_kv,
            window,
            eps,
            rope_base,
            target_layers,
            mask_token_id,
            kv_len: 0,
            n_rounds: 0,
            n_drafted: 0,
            n_accepted: 0,
            target: *target,
            tok_embd_row_bytes,
            output_quant_type,
            device,
            lib,
            ctx,
            fc,
            enc_norm,
            out_norm,
            layers,
            sel_hidden,
            pred,
            succ,
            vmap,
            n_trans,
            taps_: TapBuffers {
                bufs: tap_bufs,
                ptrs: tap_ptrs,
            },
            y_norm: None,
            y_proj: None,
            x: None,
            r_norm: None,
            conv: None,
            dyn_w: None,
            q: None,
            k: None,
            v: None,
            attn_part: None,
            attn_out: None,
            gate_buf: None,
            up_buf: None,
            logits: None,
            q8: None,
            pos: None,
            pos_zero: None,
            blk_tok: None,
            cand: None,
            unary: None,
            sel_gate: None,
            score: None,
            q_sel: None,
            hq: None,
            hcand: None,
            draft_vocab: None,
        };

        dflash.alloc(max_kv)?;
        Ok(dflash)
    }

    /// Allocate or resize internal scratch and KV ring buffers for `max_kv_len`.
    pub fn alloc(&mut self, max_kv_len: i32) -> Result<()> {
        if max_kv_len > 0 && max_kv_len != self.max_kv {
            self.max_kv = max_kv_len;
            self.window = self.window.min(max_kv_len);
            let kv_elems = (self.max_kv as usize) * (self.n_kv as usize) * (self.d_h as usize);
            for layer in &mut self.layers {
                layer.kring = DeviceBuf::<f32>::zeros(self.device, kv_elems)?;
                layer.vring = DeviceBuf::<f32>::zeros(self.device, kv_elems)?;
            }
        }

        let b = self.B as usize;
        let dm = self.d_m as usize;
        let dff = self.d_ff as usize;
        let q_dim = (self.n_q * self.d_h) as usize;
        let kv_dim = (self.n_kv * self.d_h) as usize;
        let nv = self.n_v as usize;

        self.y_norm = Some(DeviceBuf::<f32>::zeros(self.device, dm)?);
        self.y_proj = Some(DeviceBuf::<f32>::zeros(self.device, dm)?);
        self.x = Some(DeviceBuf::<f32>::zeros(self.device, b * dm)?);
        self.r_norm = Some(DeviceBuf::<f32>::zeros(self.device, b * dm)?);
        self.conv = Some(DeviceBuf::<f32>::zeros(self.device, b * dm.max(dff))?);

        let mut max_dyn = 0usize;
        for layer in &self.layers {
            if let Some(ac) = &layer.attn_conv {
                max_dyn = max_dyn.max(ac.weight.rows as usize);
            }
            if let Some(fc) = &layer.ffn_conv {
                max_dyn = max_dyn.max(fc.weight.rows as usize);
            }
        }
        if max_dyn > 0 {
            self.dyn_w = Some(DeviceBuf::<f32>::zeros(self.device, b * max_dyn)?);
        }

        self.q = Some(DeviceBuf::<f32>::zeros(self.device, b * q_dim)?);
        self.k = Some(DeviceBuf::<f32>::zeros(self.device, b * kv_dim)?);
        self.v = Some(DeviceBuf::<f32>::zeros(self.device, b * kv_dim)?);

        let attn_scratch = unsafe {
            (self.lib.tk_q36_dflash_attn_scratch_floats)(self.n_q, self.d_h, self.window)
        };
        self.attn_part = Some(DeviceBuf::<f32>::zeros(
            self.device,
            b * attn_scratch.max(1),
        )?);
        self.attn_out = Some(DeviceBuf::<f32>::zeros(self.device, b * q_dim)?);
        self.gate_buf = Some(DeviceBuf::<f32>::zeros(self.device, b * dff)?);
        self.up_buf = Some(DeviceBuf::<f32>::zeros(self.device, b * dff)?);
        self.logits = Some(DeviceBuf::<f32>::zeros(self.device, (b - 1) * nv)?);

        let q8_bytes = unsafe { (self.lib.tk_q8_1_bytes)(self.d_m.max(self.d_ff), self.B) };
        self.q8 = Some(DeviceBuf::<u8>::zeros(self.device, q8_bytes.max(64))?);
        self.pos = Some(DeviceBuf::<i32>::zeros(self.device, 1)?);
        self.pos_zero = Some(DeviceBuf::<i32>::zeros(self.device, 1)?);
        self.blk_tok = Some(DeviceBuf::<i32>::zeros(self.device, b)?);

        if self.sel_hidden.is_some() {
            self.cand = Some(DeviceBuf::<i32>::zeros(
                self.device,
                (b - 1) * DFLASH_TOP_K,
            )?);
            self.unary = Some(DeviceBuf::<f32>::zeros(
                self.device,
                (b - 1) * DFLASH_TOP_K,
            )?);
            self.sel_gate = Some(DeviceBuf::<f32>::zeros(self.device, (b - 1) * 64)?);
            self.score = Some(DeviceBuf::<f32>::zeros(self.device, b - 1)?);
            self.q_sel = Some(DeviceBuf::<f32>::zeros(self.device, b - 1)?);
            self.hq = Some(HostMapped::<f32>::alloc(self.device, b - 1)?);
            self.hcand = Some(HostMapped::<i32>::alloc(
                self.device,
                (b - 1) * DFLASH_TOP_K,
            )?);
        }

        Ok(())
    }

    /// Reset KV cache state for a new sequence / request.
    #[inline]
    pub fn reset(&mut self) {
        self.kv_len = 0;
    }

    /// Truncate the drafter KV cache length to `new_kv_len`.
    #[inline]
    pub fn truncate_to(&mut self, new_kv_len: i32) {
        self.kv_len = new_kv_len.clamp(0, self.max_kv);
    }

    /// Trim KV length if adding `extra_tokens` would exceed `max_kv`.
    #[inline]
    pub fn trim_if_needed(&mut self, extra_tokens: i32) {
        if self.kv_len + extra_tokens > self.max_kv {
            self.kv_len = (self.max_kv - extra_tokens).max(0);
        }
    }

    /// Copy `row_bytes` from `src_d` into `dst_d` on `stream` via `tk_gather_rows` at index 0.
    fn copy_row_d2d(
        &self,
        dst_d: *mut f32,
        src_d: *const f32,
        row_bytes: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let zero_idx = self
            .pos_zero
            .as_ref()
            .ok_or_else(|| inv_arg("DFlash scratch not allocated"))?
            .as_ptr();
        unsafe {
            let st = (self.lib.tk_gather_rows)(
                self.ctx,
                src_d as *const c_void,
                dst_d as *mut c_void,
                zero_idx,
                1,
                row_bytes,
                stream,
            );
            translate_tk_status(&self.lib, self.ctx, st)
        }
    }

    /// Copy the last row of a tapped target hidden layer (`layer_idx`) into the corresponding
    /// DFlash tap slot (`0..5`). No-op if `layer_idx` is not one of `self.target_layers`.
    pub fn tap(
        &mut self,
        layer_idx: i32,
        hidden_d: *const f32,
        n_rows: i32,
        stream: *mut c_void,
    ) -> Result<()> {
        if n_rows <= 0 || hidden_d.is_null() {
            return Ok(());
        }
        let mut match_slot = None;
        for (t, &tl) in self.target_layers.iter().enumerate() {
            if tl == layer_idx {
                match_slot = Some(t);
                break;
            }
        }
        if let Some(t) = match_slot {
            let src = unsafe { hidden_d.add(((n_rows - 1) as usize) * (self.d_m as usize)) };
            let bytes = (self.d_m as usize) * core::mem::size_of::<f32>();
            let dst = self.taps_.bufs[t].as_mut_ptr();
            self.copy_row_d2d(dst, src, bytes, stream)?;
        }
        Ok(())
    }

    /// Run a quantized matrix-vector / small-batch projection `out = w * x` via
    /// `tk_quantize_q8_1` + `tk_mmvq`.
    fn gemv(
        &self,
        w: &DFlashWeight,
        x_d: *const f32,
        out_d: *mut f32,
        n_rows: i32,
        stream: *mut c_void,
    ) -> Result<()> {
        let q8_ptr = self
            .q8
            .as_ref()
            .ok_or_else(|| inv_arg("DFlash scratch not allocated"))?
            .as_mut_ptr() as *mut c_void;

        unsafe {
            let st1 = (self.lib.tk_quantize_q8_1)(self.ctx, x_d, q8_ptr, w.cols, n_rows, stream);
            translate_tk_status(&self.lib, self.ctx, st1)?;

            let st2 = (self.lib.tk_mmvq)(
                self.ctx,
                quant_type_to_tk(w.qtype),
                w.as_ptr(),
                q8_ptr,
                out_d,
                w.cols,
                w.rows,
                n_rows,
                stream,
            );
            translate_tk_status(&self.lib, self.ctx, st2)
        }
    }

    /// Run optional DFlash2 dynamic depthwise convolution on `x_in` (`n_rows` rows of width `dim`).
    fn run_dyn_conv(
        &self,
        dc: &DFlashDynConv,
        x_in: *const f32,
        out: *mut f32,
        dim: i32,
        n_rows: i32,
        stream: *mut c_void,
    ) -> Result<()> {
        let dyn_w_ptr = self
            .dyn_w
            .as_ref()
            .ok_or_else(|| inv_arg("DFlash dyn_w not allocated"))?
            .as_mut_ptr();
        self.gemv(&dc.weight, x_in, dyn_w_ptr, n_rows, stream)?;
        unsafe {
            let st = (self.lib.tk_q36_dflash_dyn_conv)(
                self.ctx,
                x_in,
                dyn_w_ptr,
                dc.base.as_ptr(),
                out,
                dim,
                n_rows,
                dc.group,
                dc.side,
                stream,
            );
            translate_tk_status(&self.lib, self.ctx, st)
        }
    }

    /// Encode the 5 tapped target hidden states and write K/V projections at device position `pos_d`.
    pub fn inject(&mut self, pos_d: *const i32, stream: *mut c_void) -> Result<()> {
        let y_norm_ptr = self.y_norm.as_mut().unwrap().as_mut_ptr();
        let y_proj_ptr = self.y_proj.as_mut().unwrap().as_mut_ptr();
        let k_ptr = self.k.as_mut().unwrap().as_mut_ptr();
        let v_ptr = self.v.as_mut().unwrap().as_mut_ptr();

        unsafe {
            for t in 0..DFLASH_TAPS {
                let st = (self.lib.tk_q36_dflash_rms_rows)(
                    self.ctx,
                    self.taps_.bufs[t].as_ptr(),
                    self.enc_norm.as_ptr(),
                    y_norm_ptr,
                    self.d_m,
                    1,
                    self.eps,
                    stream,
                );
                translate_tk_status(&self.lib, self.ctx, st)?;

                let out_tap = self.taps_.bufs[t].as_mut_ptr();
                self.gemv(&self.fc[t], y_norm_ptr, out_tap, 1, stream)?;
            }

            let st = (self.lib.tk_q36_dflash_sum5_rms)(
                self.ctx,
                self.taps_.ptrs.as_ptr() as *const *const f32,
                self.enc_norm.as_ptr(),
                y_proj_ptr,
                self.d_m,
                1,
                self.eps,
                stream,
            );
            translate_tk_status(&self.lib, self.ctx, st)?;

            let kv_width = self.n_kv * self.d_h;
            for l in 0..(self.n_L as usize) {
                self.gemv(&self.layers[l].wk, y_proj_ptr, k_ptr, 1, stream)?;
                self.gemv(&self.layers[l].wv, y_proj_ptr, v_ptr, 1, stream)?;

                let st_qk = (self.lib.tk_q36_dflash_qk_norm_rope)(
                    self.ctx,
                    k_ptr,
                    self.layers[l].k_norm.as_ptr(),
                    self.n_kv,
                    self.d_h,
                    1,
                    self.pos_zero.as_ref().unwrap().as_ptr(),
                    self.rope_base,
                    self.eps,
                    stream,
                );
                translate_tk_status(&self.lib, self.ctx, st_qk)?;

                let st_kv = (self.lib.tk_q36_dflash_kv_store)(
                    self.ctx,
                    k_ptr,
                    v_ptr,
                    self.layers[l].kring.as_mut_ptr(),
                    self.layers[l].vring.as_mut_ptr(),
                    kv_width,
                    1,
                    pos_d,
                    self.max_kv,
                    stream,
                );
                translate_tk_status(&self.lib, self.ctx, st_kv)?;
            }
        }
        Ok(())
    }

    /// Inject current tapped target features at host integer position `pos`.
    pub fn inject_at(&mut self, pos: i32, stream: *mut c_void) -> Result<()> {
        let pos_ptr = self.pos.as_mut().unwrap().as_mut_ptr();
        unsafe {
            let st = (self.lib.tk_q36_dflash_set_i32)(self.ctx, pos_ptr, pos, stream);
            translate_tk_status(&self.lib, self.ctx, st)?;
        }
        self.inject(pos_ptr, stream)
    }

    /// Append target conditioning features (`target_feats_d` of shape `[n_tokens, 5 * d_m]` or
    /// `[5, d_m]` per token) into the DFlash KV ring cache.
    pub fn cache_append_target(
        &mut self,
        target_feats_d: *const f32,
        _token_ids_d: *const i32,
        n_tokens: i32,
        stream: *mut c_void,
    ) -> Result<()> {
        if n_tokens <= 0 || target_feats_d.is_null() {
            return Ok(());
        }
        self.trim_if_needed(n_tokens);
        let dm = self.d_m as usize;
        let bytes = dm * core::mem::size_of::<f32>();

        for tok in 0..(n_tokens as usize) {
            for t in 0..DFLASH_TAPS {
                let src = unsafe { target_feats_d.add((tok * DFLASH_TAPS + t) * dm) };
                let dst = self.taps_.bufs[t].as_mut_ptr();
                self.copy_row_d2d(dst, src, bytes, stream)?;
            }
            let pos = self.kv_len;
            self.inject_at(pos, stream)?;
            self.kv_len = (self.kv_len + 1).min(self.max_kv);
        }
        Ok(())
    }

    /// Run the full `B`-row block-diffusion forward pass on device starting from `anchor_d` at
    /// position `pos_d`, writing `B - 1` drafted token IDs to `out_d`.
    pub fn draft_device(
        &mut self,
        anchor_d: *const i32,
        pos_d: *const i32,
        out_d: *mut i32,
        samp: Option<&DFlashSamp>,
        stream: *mut c_void,
    ) -> Result<()> {
        let blk_tok_ptr = self.blk_tok.as_mut().unwrap().as_mut_ptr();
        let x_ptr = self.x.as_mut().unwrap().as_mut_ptr();
        let r_norm_ptr = self.r_norm.as_mut().unwrap().as_mut_ptr();
        let conv_ptr = self.conv.as_mut().unwrap().as_mut_ptr();
        let q_ptr = self.q.as_mut().unwrap().as_mut_ptr();
        let k_ptr = self.k.as_mut().unwrap().as_mut_ptr();
        let v_ptr = self.v.as_mut().unwrap().as_mut_ptr();
        let attn_part_ptr = self.attn_part.as_mut().unwrap().as_mut_ptr();
        let attn_out_ptr = self.attn_out.as_mut().unwrap().as_mut_ptr();
        let gate_ptr = self.gate_buf.as_mut().unwrap().as_mut_ptr();
        let up_ptr = self.up_buf.as_mut().unwrap().as_mut_ptr();
        let logits_ptr = self.logits.as_mut().unwrap().as_mut_ptr();
        let q8_ptr = self.q8.as_mut().unwrap().as_mut_ptr() as *mut c_void;

        unsafe {
            // 1. Fill block tokens: [anchor, mask, mask, ...]
            let st_bt = (self.lib.tk_q36_dflash_block_tokens)(
                self.ctx,
                anchor_d,
                blk_tok_ptr,
                self.B,
                self.mask_token_id,
                stream,
            );
            translate_tk_status(&self.lib, self.ctx, st_bt)?;

            // 2. Embed B tokens using target's token_embd table.
            let st_emb = (self.lib.tk_q36_embed_rows)(
                self.ctx,
                self.target.tok_embd_qtype as i32,
                self.target.tok_embd_ptr,
                self.tok_embd_row_bytes,
                blk_tok_ptr,
                self.B as i64,
                self.d_m as i64,
                x_ptr,
                stream,
            );
            translate_tk_status(&self.lib, self.ctx, st_emb)?;

            let kv_width = self.n_kv * self.d_h;
            let scale = 1.0f32 / (self.d_h as f32).sqrt();
            let b_dm = (self.B as i64) * (self.d_m as i64);
            let b_dff = (self.B as i64) * (self.d_ff as i64);

            for l in 0..(self.n_L as usize) {
                let st_an = (self.lib.tk_q36_dflash_rms_rows)(
                    self.ctx,
                    x_ptr,
                    self.layers[l].attn_norm.as_ptr(),
                    r_norm_ptr,
                    self.d_m,
                    self.B,
                    self.eps,
                    stream,
                );
                translate_tk_status(&self.lib, self.ctx, st_an)?;

                let q_in = if let Some(ac) = &self.layers[l].attn_conv {
                    self.run_dyn_conv(ac, r_norm_ptr, conv_ptr, self.d_m, self.B, stream)?;
                    conv_ptr as *const f32
                } else {
                    r_norm_ptr as *const f32
                };

                self.gemv(&self.layers[l].wq, q_in, q_ptr, self.B, stream)?;
                self.gemv(&self.layers[l].wk, r_norm_ptr, k_ptr, self.B, stream)?;
                self.gemv(&self.layers[l].wv, r_norm_ptr, v_ptr, self.B, stream)?;

                let st_qr = (self.lib.tk_q36_dflash_qk_norm_rope)(
                    self.ctx,
                    q_ptr,
                    self.layers[l].q_norm.as_ptr(),
                    self.n_q,
                    self.d_h,
                    self.B,
                    pos_d,
                    self.rope_base,
                    self.eps,
                    stream,
                );
                translate_tk_status(&self.lib, self.ctx, st_qr)?;

                let st_kr = (self.lib.tk_q36_dflash_qk_norm_rope)(
                    self.ctx,
                    k_ptr,
                    self.layers[l].k_norm.as_ptr(),
                    self.n_kv,
                    self.d_h,
                    self.B,
                    pos_d,
                    self.rope_base,
                    self.eps,
                    stream,
                );
                translate_tk_status(&self.lib, self.ctx, st_kr)?;

                let st_kvs = (self.lib.tk_q36_dflash_kv_store)(
                    self.ctx,
                    k_ptr,
                    v_ptr,
                    self.layers[l].kring.as_mut_ptr(),
                    self.layers[l].vring.as_mut_ptr(),
                    kv_width,
                    self.B,
                    pos_d,
                    self.max_kv,
                    stream,
                );
                translate_tk_status(&self.lib, self.ctx, st_kvs)?;

                let st_ba = (self.lib.tk_q36_dflash_block_attn)(
                    self.ctx,
                    q_ptr,
                    self.layers[l].kring.as_ptr(),
                    self.layers[l].vring.as_ptr(),
                    pos_d,
                    self.n_q,
                    self.n_kv,
                    self.d_h,
                    self.max_kv,
                    self.window,
                    scale,
                    attn_part_ptr,
                    attn_out_ptr,
                    stream,
                );
                translate_tk_status(&self.lib, self.ctx, st_ba)?;

                self.gemv(&self.layers[l].wo, attn_out_ptr, r_norm_ptr, self.B, stream)?;

                if let Some(apn) = &self.layers[l].attn_post_norm {
                    let st_apn = (self.lib.tk_q36_dflash_rms_rows)(
                        self.ctx,
                        r_norm_ptr,
                        apn.as_ptr(),
                        r_norm_ptr,
                        self.d_m,
                        self.B,
                        self.eps,
                        stream,
                    );
                    translate_tk_status(&self.lib, self.ctx, st_apn)?;
                }

                let st_add1 =
                    (self.lib.tk_q36_dflash_add_inplace)(self.ctx, x_ptr, r_norm_ptr, b_dm, stream);
                translate_tk_status(&self.lib, self.ctx, st_add1)?;

                let st_fn = (self.lib.tk_q36_dflash_rms_rows)(
                    self.ctx,
                    x_ptr,
                    self.layers[l].ffn_norm.as_ptr(),
                    r_norm_ptr,
                    self.d_m,
                    self.B,
                    self.eps,
                    stream,
                );
                translate_tk_status(&self.lib, self.ctx, st_fn)?;

                self.gemv(&self.layers[l].gate, r_norm_ptr, gate_ptr, self.B, stream)?;
                self.gemv(&self.layers[l].up, r_norm_ptr, up_ptr, self.B, stream)?;

                let st_sw = (self.lib.tk_q36_dflash_swiglu)(
                    self.ctx, gate_ptr, up_ptr, gate_ptr, b_dff, stream,
                );
                translate_tk_status(&self.lib, self.ctx, st_sw)?;

                let down_in = if let Some(fc) = &self.layers[l].ffn_conv {
                    self.run_dyn_conv(fc, gate_ptr, conv_ptr, self.d_ff, self.B, stream)?;
                    conv_ptr as *const f32
                } else {
                    gate_ptr as *const f32
                };

                self.gemv(&self.layers[l].down, down_in, r_norm_ptr, self.B, stream)?;

                if let Some(fpn) = &self.layers[l].ffn_post_norm {
                    let st_fpn = (self.lib.tk_q36_dflash_rms_rows)(
                        self.ctx,
                        r_norm_ptr,
                        fpn.as_ptr(),
                        r_norm_ptr,
                        self.d_m,
                        self.B,
                        self.eps,
                        stream,
                    );
                    translate_tk_status(&self.lib, self.ctx, st_fpn)?;
                }

                let st_add2 =
                    (self.lib.tk_q36_dflash_add_inplace)(self.ctx, x_ptr, r_norm_ptr, b_dm, stream);
                translate_tk_status(&self.lib, self.ctx, st_add2)?;
            }

            // Rows 1..B-1 predict the B-1 draft tokens.
            let x_tail = x_ptr.add(self.d_m as usize);
            let st_on = (self.lib.tk_q36_dflash_rms_rows)(
                self.ctx,
                x_tail,
                self.out_norm.as_ptr(),
                r_norm_ptr,
                self.d_m,
                self.B - 1,
                self.eps,
                stream,
            );
            translate_tk_status(&self.lib, self.ctx, st_on)?;

            // LM head projection using target's output weight matrix.
            let st_q8 = (self.lib.tk_quantize_q8_1)(
                self.ctx,
                r_norm_ptr,
                q8_ptr,
                self.d_m,
                self.B - 1,
                stream,
            );
            translate_tk_status(&self.lib, self.ctx, st_q8)?;

            let st_lm = (self.lib.tk_mmvq)(
                self.ctx,
                quant_type_to_tk(self.output_quant_type),
                self.target.output_ptr,
                q8_ptr,
                logits_ptr,
                self.d_m,
                self.n_v,
                self.B - 1,
                stream,
            );
            translate_tk_status(&self.lib, self.ctx, st_lm)?;

            if let Some(sh) = &self.sel_hidden {
                let cand_ptr = self.cand.as_mut().unwrap().as_mut_ptr();
                let unary_ptr = self.unary.as_mut().unwrap().as_mut_ptr();
                let sel_gate_ptr = self.sel_gate.as_mut().unwrap().as_mut_ptr();
                let score_ptr = self.score.as_mut().unwrap().as_mut_ptr();
                let q_sel_ptr = self.q_sel.as_mut().unwrap().as_mut_ptr();
                let pred_ptr = self.pred.as_ref().unwrap().as_ptr() as *const c_void;
                let succ_ptr = self.succ.as_ref().unwrap().as_ptr() as *const c_void;
                let vmap_ptr = self.vmap.as_ref().map_or(core::ptr::null(), |m| m.as_ptr());

                let st_sg = (self.lib.tk_mmvq)(
                    self.ctx,
                    quant_type_to_tk(sh.qtype),
                    sh.as_ptr(),
                    q8_ptr,
                    sel_gate_ptr,
                    sh.cols,
                    sh.rows,
                    self.B - 1,
                    stream,
                );
                translate_tk_status(&self.lib, self.ctx, st_sg)?;

                let st_top = (self.lib.tk_q36_dflash_top16)(
                    self.ctx,
                    logits_ptr,
                    self.n_v,
                    self.B - 1,
                    vmap_ptr,
                    cand_ptr,
                    unary_ptr,
                    stream,
                );
                translate_tk_status(&self.lib, self.ctx, st_top)?;

                if let Some(s) = samp {
                    let st_sc = (self.lib.tk_q36_dflash_select_chain_q)(
                        self.ctx,
                        cand_ptr,
                        unary_ptr,
                        sel_gate_ptr,
                        pred_ptr,
                        succ_ptr,
                        anchor_d,
                        out_d,
                        score_ptr,
                        q_sel_ptr,
                        s.tau,
                        s.seed,
                        s.ctr,
                        self.B - 1,
                        stream,
                    );
                    translate_tk_status(&self.lib, self.ctx, st_sc)?;
                } else {
                    let st_sc = (self.lib.tk_q36_dflash_select_chain)(
                        self.ctx,
                        cand_ptr,
                        unary_ptr,
                        sel_gate_ptr,
                        pred_ptr,
                        succ_ptr,
                        anchor_d,
                        out_d,
                        score_ptr,
                        self.B - 1,
                        stream,
                    );
                    translate_tk_status(&self.lib, self.ctx, st_sc)?;
                }
            } else {
                let st_arg = (self.lib.tk_argmax_rows)(
                    self.ctx,
                    logits_ptr,
                    self.B - 1,
                    self.n_v,
                    out_d,
                    stream,
                );
                translate_tk_status(&self.lib, self.ctx, st_arg)?;
            }
        }

        Ok(())
    }

    /// Host-callable draft entry point: drafts up to `out_drafts.len().min(B - 1)` tokens
    /// starting from `anchor_token` at current `self.kv_len`.
    pub fn draft(
        &mut self,
        anchor_token: i32,
        out_drafts: &mut [i32],
        stream: *mut c_void,
    ) -> Result<usize> {
        let max_d = out_drafts.len().min((self.B - 1).max(0) as usize);
        if max_d == 0 {
            return Ok(0);
        }

        let anchor_dev = DeviceBuf::<i32>::from_slice(self.device, &[anchor_token])?;
        let out_dev = DeviceBuf::<i32>::zeros(self.device, (self.B - 1) as usize)?;
        let pos_ptr = self.pos.as_mut().unwrap().as_mut_ptr();

        unsafe {
            let st = (self.lib.tk_q36_dflash_set_i32)(self.ctx, pos_ptr, self.kv_len, stream);
            translate_tk_status(&self.lib, self.ctx, st)?;
        }

        self.draft_device(
            anchor_dev.as_ptr(),
            pos_ptr,
            out_dev.as_mut_ptr(),
            None,
            stream,
        )?;

        let mut host_out = vec![0i32; (self.B - 1) as usize];
        out_dev.copy_to_host(&mut host_out)?;
        out_drafts[..max_d].copy_from_slice(&host_out[..max_d]);

        self.n_rounds += 1;
        self.n_drafted += max_d as i64;
        Ok(max_d)
    }

    /// Commit `n_accepted` accepted draft tokens and the bonus token's target features into
    /// the DFlash KV cache.
    pub fn accept_commit(
        &mut self,
        n_accepted: i32,
        _bonus_token: i32,
        target_feats_d: *const f32,
        stream: *mut c_void,
    ) -> Result<()> {
        let acc = n_accepted.max(0);
        self.n_accepted += acc as i64;
        // Target features cover the accepted draft positions plus the bonus token (acc + 1).
        if !target_feats_d.is_null() {
            self.cache_append_target(target_feats_d, core::ptr::null(), acc + 1, stream)?;
        } else {
            self.kv_len = (self.kv_len + acc + 1).min(self.max_kv);
        }
        Ok(())
    }

    /// Copy DFlash2 proposal probabilities `q_sel` and candidates `cand` to host-mapped memory.
    pub fn q_to_host(&mut self, _stream: *mut c_void) -> Result<()> {
        if let (Some(q_sel), Some(hq), Some(cand), Some(hcand)) =
            (&self.q_sel, &mut self.hq, &self.cand, &mut self.hcand)
        {
            q_sel.copy_to_host(hq.as_mut_slice())?;
            cand.copy_to_host(hcand.as_mut_slice())?;
        }
        Ok(())
    }

    /// Query the proposal probability `q(token_id)` at chain step `step` after [`Self::q_to_host`].
    #[inline]
    pub fn q_host(&self, step: usize, token_id: i32) -> f64 {
        let b_m1 = (self.B - 1).max(0) as usize;
        if step >= b_m1 {
            return 0.0;
        }
        if let (Some(hq), Some(hcand)) = (&self.hq, &self.hcand) {
            let q_val = hq.as_slice()[step] as f64;
            let row = &hcand.as_slice()[step * DFLASH_TOP_K..(step + 1) * DFLASH_TOP_K];
            for &c in row {
                if c == token_id {
                    return q_val;
                }
            }
            // Outside top-16 candidates: `q = 0` so target tokens outside top-16 are never
            // under-sampled.
            1e-30
        } else {
            0.0
        }
    }

    /// Device pointer to per-step chain selection confidence scores (`B - 1` floats), if DFlash2.
    #[inline]
    pub fn scores_ptr(&self) -> *const f32 {
        self.score
            .as_ref()
            .map_or(core::ptr::null(), |s| s.as_ptr())
    }

    /// Return a human-readable summary matching `DFlash::describe()` in `q36_dflash_model.cpp`.
    pub fn describe(&self) -> String {
        let mut s = format!(
            "dflash: L={} dim={} ffn={} heads={}/{}x{} block={} window={} mask={} taps=[",
            self.n_L,
            self.d_m,
            self.d_ff,
            self.n_q,
            self.n_kv,
            self.d_h,
            self.B,
            self.window,
            self.mask_token_id
        );
        for (i, &tl) in self.target_layers.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&tl.to_string());
        }
        s.push(']');
        if self
            .layers
            .first()
            .and_then(|l| l.attn_conv.as_ref())
            .is_some()
        {
            s.push_str(" +dynconv");
        }
        if self.sel_hidden.is_some() {
            s.push_str(&format!(" +select({})", self.n_trans));
        }
        s
    }

    /// Print DFlash acceptance statistics to stderr.
    pub fn print_stats(&self) {
        let rate = if self.n_drafted > 0 {
            100.0 * (self.n_accepted as f64) / (self.n_drafted as f64)
        } else {
            0.0
        };
        let mean_acc = if self.n_rounds > 0 {
            (self.n_accepted as f64) / (self.n_rounds as f64)
        } else {
            0.0
        };
        eprintln!(
            "[dflash] rounds={} drafted={} accepted={} ({:.1}%, {:.2} tok/round)",
            self.n_rounds, self.n_drafted, self.n_accepted, rate, mean_acc
        );
    }
}

impl Drop for DFlash {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                (self.lib.tk_ctx_destroy)(self.ctx);
            }
            self.ctx = core::ptr::null_mut();
        }
    }
}

impl DeviceDrafter for DFlash {
    fn seed(&mut self, st: &SeqState, taps: &HiddenTaps, s: &StreamHandle) -> Result<()> {
        self.kv_len = st.pos as i32;
        if !taps.device_ptrs.is_empty() {
            let n_taps = taps.device_ptrs.len().min(DFLASH_TAPS);
            let dm = self.d_m as usize;
            let bytes = dm * core::mem::size_of::<f32>();
            let stream_ptr = s.raw as usize as *mut c_void;
            for t in 0..n_taps {
                let src = taps.device_ptrs[t] as usize as *const f32;
                if !src.is_null() {
                    let dst = self.taps_.bufs[t].as_mut_ptr();
                    self.copy_row_d2d(dst, src, bytes, stream_ptr)?;
                }
            }
            self.inject_at(self.kv_len.saturating_sub(1), stream_ptr)?;
        }
        Ok(())
    }

    fn capture_chain(&mut self, d: usize, io: &RoundIo, _g: &mut GraphBuilder) -> Result<()> {
        let _ = (d, io);
        Ok(())
    }

    fn vocab(&self) -> Option<&CoreDraftVocab> {
        self.draft_vocab.as_ref()
    }
}

// =========================================================================
// Phase C2 / Phase H Model Profiles & Mock/Callback DeviceDrafter
// =========================================================================

use crate::draft_vocab::DraftVocab;
use crate::ngram::NgramMod;
use crate::pause::{AdaptivePausePolicy, NgramEditProfile};
use crate::prompt_lookup::PromptLookupDrafter;
use talus_core_api::{HiddenTapKind, RoundPlan, RoundStats, SpecController};

/// Model family for DFlash / DFlash2 block-diffusion drafters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DFlashFamily {
    /// Qwen3.8-27B DFlash2: 5 hidden taps, 16-row block, 64K draft vocab (`qwen-dv-65536.bin`).
    Qwen27BDFlash2,
    /// Laguna-S-2.1 DFlash (Phase C2): 1024-token SWA window, 100,352 target vocab, optional 64K draft vocab, adaptive pause 1.2 + `ngram-mod` edit fallback.
    LagunaDFlash,
    /// MiMo-V2-Flash DFlash (Phase H): 152K (`152064`) Q8_0 drafter head + 64K draft vocab (`mimo-dv-65536.bin`), `n4` (`max_draft = 4`).
    MimoDFlash,
}

/// Output LM-head projection mode for a DFlash drafter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlashHeadMode {
    /// Shares the target model's output LM head (optionally filtered via draft vocab).
    SharedTargetLmHead,
    /// Uses a dedicated Q8_0 drafter LM head (e.g. MiMo-V2-Flash 152,064-row Q8_0 head).
    DedicatedQ8_0Head { head_vocab: usize },
}

/// Architecture and runtime profile for a DFlash / DFlash2 block-diffusion drafter
/// (`Qwen3.8-27B`, `Laguna-S-2.1`, `MiMo-V2-Flash`).
#[derive(Debug, Clone, PartialEq)]
pub struct DFlashModelProfile {
    /// Drafter model family.
    pub family: DFlashFamily,
    /// Canonical profile name.
    pub name: &'static str,
    /// Number of transformer layers in the drafter (`5` for Qwen/MiMo, `6` for Laguna).
    pub n_layers: usize,
    /// Number of hidden-state taps extracted from the target model (`5`).
    pub n_taps: usize,
    /// 0-based target layer indices tapped for conditioning.
    pub tap_layers: Vec<u32>,
    /// Drafter hidden dimension (`d_model`).
    pub d_model: usize,
    /// Drafter FFN intermediate dimension (`d_ff`).
    pub d_ff: usize,
    /// Number of query attention heads (`n_q`).
    pub n_q: usize,
    /// Number of key/value attention heads (`n_kv`).
    pub n_kv: usize,
    /// Head dimension (`d_head`).
    pub d_head: usize,
    /// Full target model vocabulary size.
    pub target_vocab: usize,
    /// Drafter head vocabulary size (`152064` for MiMo Q8_0 head).
    pub head_vocab: usize,
    /// Active draft vocabulary subset size (e.g. `Some(65536)`), or `None` if projecting full vocab.
    pub draft_vocab_size: Option<usize>,
    /// Default draft vocabulary `.bin` filename (`qwen-dv-65536.bin`, `mimo-dv-65536.bin`).
    pub default_draft_vocab_file: Option<&'static str>,
    /// Block-diffusion forward rows `B` (`1` anchor row + `B - 1` mask/draft rows).
    pub block_size: usize,
    /// Maximum draft tokens proposed per round (`B - 1`, e.g. `15` for Qwen, `4` (`n4`) for Laguna/MiMo).
    pub max_draft: usize,
    /// Sliding-window attention (SWA) KV cache capacity in tokens (`1024` for Laguna/MiMo, `2048` for Qwen).
    pub swa_window: usize,
    /// True if the drafter uses DFlash2 dynamic conv + bilinear chain selector (`n_trans`).
    pub is_dflash2: bool,
    /// True if the drafter uses Laguna's `enc.aux_norm` + per-layer context norm + softplus `attn_gate`.
    pub has_attn_gate_softplus: bool,
    /// True if the drafter uses MiMo's per-head attention sinks (`attn_sinks.weight`).
    pub has_attn_sinks: bool,
    /// Output LM head mode.
    pub head_mode: DFlashHeadMode,
    /// Adaptive pause threshold (`1.2`).
    pub adaptive_pause_ratio: f32,
    /// Coordinated host `ngram-mod` / edit profile.
    pub edit_profile: NgramEditProfile,
}

impl DFlashModelProfile {
    /// **Qwen3.8-27B DFlash2** profile:
    /// 5 hidden taps, 16-row block (`max_draft = 15`), 248,320 target vocab, 64K draft vocab (`qwen-dv-65536.bin`).
    pub fn qwen38_27b_dflash2() -> Self {
        Self {
            family: DFlashFamily::Qwen27BDFlash2,
            name: "qwen3.8-27b-dflash2",
            n_layers: 5,
            n_taps: DFLASH_TAPS,
            tap_layers: vec![1, 16, 32, 48, 63],
            d_model: 5120,
            d_ff: 17408,
            n_q: 32,
            n_kv: 8,
            d_head: 128,
            target_vocab: 248_320,
            head_vocab: 248_320,
            draft_vocab_size: Some(65_536),
            default_draft_vocab_file: Some("qwen-dv-65536.bin"),
            block_size: 16,
            max_draft: 15,
            swa_window: 2048,
            is_dflash2: true,
            has_attn_gate_softplus: false,
            has_attn_sinks: false,
            head_mode: DFlashHeadMode::SharedTargetLmHead,
            adaptive_pause_ratio: 1.2,
            edit_profile: NgramEditProfile::code(),
        }
    }

    /// Alias for [`Self::qwen38_27b_dflash2`].
    pub fn qwen27b() -> Self {
        Self::qwen38_27b_dflash2()
    }

    /// **Laguna-S-2.1 DFlash (Phase C2)** profile:
    /// 6 layers, 5 hidden taps, 1024-token SWA window (`swa_window = 1024`), 100,352 target vocab,
    /// optional 64K draft vocab, adaptive pause `1.2`, and `ngram-mod` edit fallback.
    pub fn laguna_s_2_1_dflash() -> Self {
        Self {
            family: DFlashFamily::LagunaDFlash,
            name: "laguna-s-2.1-dflash",
            n_layers: 6,
            n_taps: DFLASH_TAPS,
            tap_layers: vec![1, 8, 15, 22, 29],
            d_model: 3072,
            d_ff: 12288,
            n_q: 72,
            n_kv: 8,
            d_head: 128,
            target_vocab: 100_352,
            head_vocab: 100_352,
            draft_vocab_size: Some(65_536),
            default_draft_vocab_file: None,
            block_size: 5,
            max_draft: 4,
            swa_window: 1024,
            is_dflash2: false,
            has_attn_gate_softplus: true,
            has_attn_sinks: false,
            head_mode: DFlashHeadMode::SharedTargetLmHead,
            adaptive_pause_ratio: 1.2,
            edit_profile: NgramEditProfile::edit(),
        }
    }

    /// Alias for [`Self::laguna_s_2_1_dflash`].
    pub fn laguna() -> Self {
        Self::laguna_s_2_1_dflash()
    }

    /// **MiMo-V2-Flash DFlash (Phase H)** profile:
    /// 5 layers, 5 hidden taps, 152K (`152064`) Q8_0 drafter head + 64K draft vocab (`mimo-dv-65536.bin`),
    /// `n4` (`max_draft = 4`, `block_size = 5`) block-diffusion drafting, 1024 SWA window, adaptive pause `1.2`.
    pub fn mimo_v2_flash_dflash() -> Self {
        Self {
            family: DFlashFamily::MimoDFlash,
            name: "mimo-v2-flash-dflash",
            n_layers: 5,
            n_taps: DFLASH_TAPS,
            tap_layers: vec![1, 12, 24, 36, 47],
            d_model: 4096,
            d_ff: 16384,
            n_q: 64,
            n_kv: 8,
            d_head: 128,
            target_vocab: 152_064,
            head_vocab: 152_064,
            draft_vocab_size: Some(65_536),
            default_draft_vocab_file: Some("mimo-dv-65536.bin"),
            block_size: 5,
            max_draft: 4,
            swa_window: 1024,
            is_dflash2: false,
            has_attn_gate_softplus: false,
            has_attn_sinks: true,
            head_mode: DFlashHeadMode::DedicatedQ8_0Head {
                head_vocab: 152_064,
            },
            adaptive_pause_ratio: 1.2,
            edit_profile: NgramEditProfile::mimo_dflash(),
        }
    }

    /// Alias for [`Self::mimo_v2_flash_dflash`].
    pub fn mimo() -> Self {
        Self::mimo_v2_flash_dflash()
    }

    /// Builder: overrides `draft_vocab_size` (`None` disables draft vocab subset mapping).
    pub fn with_draft_vocab_size(mut self, draft_vocab_size: Option<usize>) -> Self {
        self.draft_vocab_size = draft_vocab_size;
        self
    }

    /// Builder: overrides `swa_window`.
    pub fn with_swa_window(mut self, swa_window: usize) -> Self {
        self.swa_window = swa_window.max(1);
        self
    }

    /// Builder: overrides `edit_profile`.
    pub fn with_edit_profile(mut self, edit_profile: NgramEditProfile) -> Self {
        self.edit_profile = edit_profile;
        self
    }

    /// Constructs the [`HiddenTaps`] descriptor expected from the target model for this profile.
    pub fn expected_hidden_taps(&self) -> HiddenTaps {
        HiddenTaps {
            taps: self
                .tap_layers
                .iter()
                .map(|&l| HiddenTapKind::LayerPostFfn(l))
                .collect(),
            n_embd: self.d_model as u32,
            device_ptrs: vec![0; self.n_taps],
        }
    }
}

/// Callback signature for custom / mock CPU block-diffusion draft token generation.
///
/// Arguments: `(&DFlashModelProfile, swa_window_tokens, requested_k, Option<&DraftVocab>) -> Vec<u32>`.
pub type DFlashDraftFn =
    Box<dyn FnMut(&DFlashModelProfile, &[u32], usize, Option<&DraftVocab>) -> Vec<u32> + Send>;

/// Rust-side DFlash block-diffusion drafter specification & mock/callback execution engine
/// (`DFlashDrafterSpec`), implementing [`talus_core_api::DeviceDrafter`] and [`talus_core_api::SpecController`].
///
/// Supports Qwen3.8-27B DFlash2, Laguna-S-2.1 DFlash (1024 SWA window + adaptive pause 1.2 +
/// `ngram-mod` edit fallback), and MiMo-V2-Flash DFlash (152K Q8_0 head + 64K draft vocab + `n4`).
pub struct DFlashDrafterSpec {
    /// Model architecture profile.
    pub profile: DFlashModelProfile,
    /// Optional rich `DraftVocab` (`draft_id <-> target_id`).
    pub draft_vocab: Option<DraftVocab>,
    /// `talus-core-api` `DraftVocab` view returned by [`DeviceDrafter::vocab`].
    core_draft_vocab: Option<CoreDraftVocab>,
    /// Phase C2 adaptive pause policy (`1.2`).
    pub pause_policy: AdaptivePausePolicy,
    /// Coordinated `ngram-mod` / prompt-lookup edit profile.
    pub edit_profile: NgramEditProfile,
    /// Host 3-gram drafter for edit fallback.
    pub host_ngram: NgramMod,
    /// Host prompt-lookup copy drafter for edit fallback.
    pub host_lookup: PromptLookupDrafter,
    /// Sliding-window KV context buffer capped at `profile.swa_window`.
    pub swa_tokens: Vec<u32>,
    /// Total tokens ever committed (may exceed `swa_tokens.len()` once SWA wraps).
    pub total_committed_tokens: usize,
    /// Number of times `DeviceDrafter::seed` was invoked.
    pub seed_count: u64,
    /// Number of times `DeviceDrafter::capture_chain` was invoked.
    pub captured_chains: u64,
    /// Number of device draft rounds executed.
    pub device_rounds: u64,
    /// Optional custom draft generator callback for deterministic/mock CPU testing.
    draft_fn: Option<DFlashDraftFn>,
}

impl core::fmt::Debug for DFlashDrafterSpec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DFlashDrafterSpec")
            .field("profile", &self.profile)
            .field("has_draft_vocab", &self.draft_vocab.is_some())
            .field("pause_policy", &self.pause_policy)
            .field("edit_profile", &self.edit_profile)
            .field("swa_len", &self.swa_tokens.len())
            .field("total_committed_tokens", &self.total_committed_tokens)
            .field("seed_count", &self.seed_count)
            .field("captured_chains", &self.captured_chains)
            .field("device_rounds", &self.device_rounds)
            .finish()
    }
}

impl DFlashDrafterSpec {
    /// Creates a new `DFlashDrafterSpec` from a [`DFlashModelProfile`].
    ///
    /// If `profile.draft_vocab_size` is `Some(n_dv)`, builds a default identity/stride
    /// [`DraftVocab`] of `n_dv` entries over `profile.target_vocab` (which can be replaced
    /// with an exact `.bin` vocabulary via [`Self::with_draft_vocab`]).
    pub fn new(profile: DFlashModelProfile) -> Self {
        let edit_profile = profile.edit_profile.clone();
        let pause_policy = AdaptivePausePolicy::new(profile.adaptive_pause_ratio)
            .with_max_draft(profile.max_draft as u32)
            .with_fallback_to_host(
                edit_profile.enable_on_device_pause && edit_profile.host_enabled(),
            );
        let host_ngram = edit_profile.build_ngram_mod(profile.target_vocab);
        let host_lookup = edit_profile.build_prompt_lookup();

        let (draft_vocab, core_draft_vocab) = if let Some(n_dv) = profile.draft_vocab_size {
            let n_clamped = n_dv.min(profile.target_vocab);
            let ids: Vec<i32> = (0..n_clamped as i32).collect();
            let dv = DraftVocab::from_ids(&ids, profile.target_vocab as i32).ok();
            let core_dv = dv
                .as_ref()
                .map(|d| CoreDraftVocab::new(d.ids.iter().map(|&id| id as u32).collect()));
            (dv, core_dv)
        } else {
            (None, None)
        };

        Self {
            profile,
            draft_vocab,
            core_draft_vocab,
            pause_policy,
            edit_profile,
            host_ngram,
            host_lookup,
            swa_tokens: Vec::new(),
            total_committed_tokens: 0,
            seed_count: 0,
            captured_chains: 0,
            device_rounds: 0,
            draft_fn: None,
        }
    }

    /// Creates a Qwen3.8-27B DFlash2 drafter spec.
    pub fn qwen38_27b() -> Self {
        Self::new(DFlashModelProfile::qwen38_27b_dflash2())
    }

    /// Creates a Laguna-S-2.1 DFlash (Phase C2) drafter spec.
    pub fn laguna_s_2_1() -> Self {
        Self::new(DFlashModelProfile::laguna_s_2_1_dflash())
    }

    /// Creates a MiMo-V2-Flash DFlash (Phase H) drafter spec.
    pub fn mimo_v2_flash() -> Self {
        Self::new(DFlashModelProfile::mimo_v2_flash_dflash())
    }

    /// Attaches an explicit [`DraftVocab`] (e.g. loaded from `qwen-dv-65536.bin` or `mimo-dv-65536.bin`).
    pub fn with_draft_vocab(mut self, dv: DraftVocab) -> Result<Self> {
        if dv.n_vocab as usize != self.profile.target_vocab {
            return Err(inv_arg(format!(
                "DraftVocab n_vocab ({}) != profile target_vocab ({})",
                dv.n_vocab, self.profile.target_vocab
            )));
        }
        let core_dv = CoreDraftVocab::new(dv.ids.iter().map(|&id| id as u32).collect());
        self.profile.draft_vocab_size = Some(dv.n_draft as usize);
        self.draft_vocab = Some(dv);
        self.core_draft_vocab = Some(core_dv);
        Ok(self)
    }

    /// Installs a custom mock/callback draft generator for CPU testing.
    pub fn with_draft_callback<F>(mut self, f: F) -> Self
    where
        F: FnMut(&DFlashModelProfile, &[u32], usize, Option<&DraftVocab>) -> Vec<u32>
            + Send
            + 'static,
    {
        self.draft_fn = Some(Box::new(f));
        self
    }

    /// Returns the current number of tokens in the drafter's SWA window (`<= profile.swa_window`).
    #[inline]
    pub fn swa_len(&self) -> usize {
        self.swa_tokens.len()
    }

    /// Appends tokens into the SWA ring window (retaining at most the last `profile.swa_window` tokens)
    /// and updates the host fallback drafters (`host_ngram` and `host_lookup`).
    pub fn push_context(&mut self, tokens: &[u32]) {
        if tokens.is_empty() {
            return;
        }
        self.total_committed_tokens += tokens.len();
        self.swa_tokens.extend_from_slice(tokens);
        let win = self.profile.swa_window.max(1);
        if self.swa_tokens.len() > win {
            let drop_n = self.swa_tokens.len() - win;
            self.swa_tokens.drain(0..drop_n);
        }
        for &t in tokens {
            self.host_lookup.push(t as i32);
        }
        let swa_i32: Vec<i32> = self.swa_tokens.iter().map(|&t| t as i32).collect();
        self.host_ngram.add(&swa_i32);
    }

    /// Resets the drafter's SWA cache, adaptive pause policy, and host fallback drafters.
    pub fn reset(&mut self) {
        self.swa_tokens.clear();
        self.total_committed_tokens = 0;
        self.pause_policy.reset();
        self.host_ngram.reset();
        self.host_lookup.reset(&[]);
    }

    /// Maps a draft vocabulary index (`0..n_draft`) to a target vocabulary token ID (`0..target_vocab`).
    #[inline]
    pub fn map_draft_to_target(&self, draft_idx: usize) -> Option<u32> {
        if let Some(dv) = &self.core_draft_vocab {
            dv.map_to_target(draft_idx)
        } else if draft_idx < self.profile.target_vocab {
            Some(draft_idx as u32)
        } else {
            None
        }
    }

    /// Proposes up to `max_draft` tokens using the device block-diffusion drafter (or callback).
    pub fn propose_device_block(&mut self, max_draft: usize) -> Vec<u32> {
        let k = max_draft
            .min(self.profile.max_draft)
            .min(self.profile.block_size.saturating_sub(1));
        if k == 0 {
            return Vec::new();
        }
        self.device_rounds += 1;

        let raw = if let Some(cb) = self.draft_fn.as_mut() {
            cb(
                &self.profile,
                &self.swa_tokens,
                k,
                self.draft_vocab.as_ref(),
            )
        } else {
            // Default deterministic mock block-diffusion proposal:
            // Uses draft-vocab indices derived from trailing SWA token context and maps to target vocab.
            let last_tok = self.swa_tokens.last().copied().unwrap_or(0) as usize;
            let dv_cap = self
                .profile
                .draft_vocab_size
                .unwrap_or(self.profile.target_vocab)
                .max(1);
            let mut out = Vec::with_capacity(k);
            for step in 0..k {
                let draft_id = (last_tok.wrapping_add(step + 1)) % dv_cap;
                let target_id = self
                    .map_draft_to_target(draft_id)
                    .unwrap_or((draft_id % self.profile.target_vocab) as u32);
                out.push(target_id);
            }
            out
        };

        let tv = self.profile.target_vocab as u32;
        raw.into_iter().take(k).map(|t| t % tv).collect()
    }

    /// Proposes up to `max_draft` tokens from the coordinated host fallback drafters
    /// (`PromptLookupDrafter` + `NgramMod`).
    pub fn propose_host_fallback(&mut self, max_draft: usize) -> Vec<u32> {
        let mut out = Vec::new();
        self.edit_profile.propose_coordinated(
            &self.swa_tokens,
            max_draft,
            Some(&mut self.host_lookup),
            None,
            Some(&mut self.host_ngram),
            &mut out,
        );
        out
    }
}

impl DeviceDrafter for DFlashDrafterSpec {
    fn seed(&mut self, st: &SeqState, taps: &HiddenTaps, _s: &StreamHandle) -> Result<()> {
        if !taps.taps.is_empty() && taps.taps.len() != self.profile.n_taps {
            return Err(inv_arg(format!(
                "{} expected {} hidden taps, got {}",
                self.profile.name,
                self.profile.n_taps,
                taps.taps.len()
            )));
        }
        if taps.n_embd > 0 && taps.n_embd as usize != self.profile.d_model {
            return Err(inv_arg(format!(
                "{} expected tap n_embd={}, got {}",
                self.profile.name, self.profile.d_model, taps.n_embd
            )));
        }
        let win = self.profile.swa_window.max(1);
        if self.swa_tokens.len() > win {
            let drop_n = self.swa_tokens.len() - win;
            self.swa_tokens.drain(0..drop_n);
        }
        let _ = st.pos;
        self.seed_count += 1;
        Ok(())
    }

    fn capture_chain(&mut self, d: usize, io: &RoundIo, g: &mut GraphBuilder) -> Result<()> {
        let k = d
            .min(io.draft_depth)
            .min(self.profile.max_draft)
            .min(self.profile.block_size.saturating_sub(1));
        g.record_launch();
        self.captured_chains += 1;

        if io.device_draft_tokens_ptr != 0 && k > 0 {
            let drafts = self.propose_device_block(k);
            let out_ptr = io.device_draft_tokens_ptr as usize as *mut u32;
            unsafe {
                for (i, &tok) in drafts.iter().enumerate() {
                    *out_ptr.add(i + 1) = tok;
                }
            }
        }
        Ok(())
    }

    fn vocab(&self) -> Option<&CoreDraftVocab> {
        self.core_draft_vocab.as_ref()
    }
}

impl SpecController for DFlashDrafterSpec {
    fn plan(&mut self, last: &RoundStats) -> RoundPlan {
        self.edit_profile
            .plan_with_pause(&mut self.pause_policy, last)
    }
}

// =========================================================================
// Internal GGUF & kernel loading helpers
// =========================================================================

fn load_kernels_lib() -> Result<LibTalusKernels> {
    if let Ok(dir) = std::env::var("TALUS_KERNELS_BUILD_DIR") {
        if !dir.is_empty() {
            let candidate = Path::new(&dir).join("libtalus_kernels.so");
            if candidate.exists() {
                return LibTalusKernels::load(&candidate)
                    .map_err(|message| CoreError::KernelError { code: -1, message });
            }
        }
    }
    LibTalusKernels::load_default().map_err(|message| CoreError::KernelError { code: -1, message })
}

fn get_i32_meta(gguf: &GgufFile, key: &str) -> Result<i32> {
    if let Some(v) = gguf.get_i64(key) {
        return Ok(v as i32);
    }
    Err(inv_arg(format!(
        "missing or non-integer GGUF metadata key '{key}'"
    )))
}

fn get_f32_meta(gguf: &GgufFile, key: &str) -> Result<f32> {
    if let Some(v) = gguf.get_f64(key) {
        return Ok(v as f32);
    }
    Err(inv_arg(format!(
        "missing or non-float GGUF metadata key '{key}'"
    )))
}

fn get_i32_array_meta(gguf: &GgufFile, key: &str) -> Result<Vec<i32>> {
    match gguf.get_metadata(key) {
        Some(GgufValue::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                if let Some(v) = item.as_i64() {
                    out.push(v as i32);
                } else {
                    return Err(inv_arg(format!(
                        "GGUF metadata array '{key}' element has unexpected type: {item:?}"
                    )));
                }
            }
            Ok(out)
        }
        Some(other) => Err(inv_arg(format!(
            "GGUF metadata '{key}' is not an array: {other:?}"
        ))),
        None => Err(inv_arg(format!("missing GGUF metadata key '{key}'"))),
    }
}

fn upload_raw_tensor(gguf: &GgufFile, device: DeviceId, name: &str) -> Result<DeviceBuf<u8>> {
    let bytes = gguf
        .tensor_bytes(name)
        .map_err(|_| inv_arg(format!("DFlash missing GGUF tensor '{name}'")))?;
    Ok(DeviceBuf::<u8>::from_slice(device, bytes)?)
}

fn upload_weight(
    gguf: &GgufFile,
    device: DeviceId,
    name: &str,
    cols: i32,
    rows: i32,
) -> Result<DFlashWeight> {
    let ti = gguf
        .find_tensor(name)
        .ok_or_else(|| inv_arg(format!("DFlash missing GGUF tensor '{name}'")))?;
    if ti.dims[0] as i32 != cols || ti.dims[1] as i32 != rows {
        return Err(inv_arg(format!(
            "DFlash tensor '{name}' shape mismatch: expected [{cols}, {rows}], got {:?}",
            ti.shape()
        )));
    }
    let bytes = gguf
        .tensor_bytes(name)
        .map_err(|e| inv_arg(e.to_string()))?;
    let buf = DeviceBuf::<u8>::from_slice(device, bytes)?;
    Ok(DFlashWeight {
        buf,
        qtype: ti.qtype,
        cols,
        rows,
    })
}

fn upload_f32_vec(
    gguf: &GgufFile,
    device: DeviceId,
    name: &str,
    len: usize,
) -> Result<DeviceBuf<f32>> {
    upload_f32_vec_opt(gguf, device, name, len)?
        .ok_or_else(|| inv_arg(format!("DFlash missing F32 GGUF tensor '{name}'")))
}

fn upload_f32_vec_opt(
    gguf: &GgufFile,
    device: DeviceId,
    name: &str,
    len: usize,
) -> Result<Option<DeviceBuf<f32>>> {
    let Some(ti) = gguf.find_tensor(name) else {
        return Ok(None);
    };
    if ti.qtype != QuantType::F32 || ti.elements() as usize != len {
        return Err(inv_arg(format!(
            "DFlash F32 tensor '{name}' mismatch: qtype={:?}, elems={}, expected {len}",
            ti.qtype,
            ti.elements()
        )));
    }
    let bytes = gguf
        .tensor_bytes(name)
        .map_err(|e| inv_arg(e.to_string()))?;
    let mut host = vec![0.0f32; len];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        host[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(Some(DeviceBuf::<f32>::from_slice(device, &host)?))
}

fn upload_i32_vec(
    gguf: &GgufFile,
    device: DeviceId,
    name: &str,
    len: usize,
) -> Result<DeviceBuf<i32>> {
    let ti = gguf
        .find_tensor(name)
        .ok_or_else(|| inv_arg(format!("DFlash missing I32 GGUF tensor '{name}'")))?;
    if ti.elements() as usize != len || ti.byte_size != (len * 4) as u64 {
        return Err(inv_arg(format!(
            "DFlash I32 tensor '{name}' length mismatch: elems={}, expected {len}",
            ti.elements()
        )));
    }
    let bytes = gguf
        .tensor_bytes(name)
        .map_err(|e| inv_arg(e.to_string()))?;
    let mut host = vec![0i32; len];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        host[i] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(DeviceBuf::<i32>::from_slice(device, &host)?)
}

fn upload_dyn_conv_opt(
    gguf: &GgufFile,
    device: DeviceId,
    prefix: &str,
    in_dim: i32,
    ch_dim: i32,
) -> Result<Option<DFlashDynConv>> {
    let w_name = format!("{prefix}.weight");
    let b_name = format!("{prefix}.base");
    let Some(w_ti) = gguf.find_tensor(&w_name) else {
        return Ok(None);
    };
    let b_ti = gguf
        .find_tensor(&b_name)
        .ok_or_else(|| inv_arg(format!("DFlash missing '{b_name}' alongside '{w_name}'")))?;
    let k = b_ti.dims[0] as i32;
    let head = (w_ti.dims[1] as i32) / k;
    if (k & 1) == 0 || head <= 0 || (ch_dim % head) != 0 || b_ti.dims[1] as i32 != ch_dim {
        return Err(inv_arg(format!("DFlash bad dyn_conv shape on '{prefix}'")));
    }
    let weight = upload_weight(gguf, device, &w_name, in_dim, head * k)?;
    let base = upload_f32_vec(gguf, device, &b_name, (k * ch_dim) as usize)?;
    Ok(Some(DFlashDynConv {
        weight,
        base,
        group: ch_dim / head,
        side: (k - 1) / 2,
    }))
}
