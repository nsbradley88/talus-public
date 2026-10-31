//! Laguna-S-2.1 DFlash drafter (poolside's 6-layer block drafter, `laguna-s-2.1-dflash-Q8_0.gguf`),
//! implementing block speculative drafting for Laguna.
//!
//! Architecture (GGUF `dflash.*`, `decoder_arch = laguna`): 6 dense layers (d 3072, 72 heads x 128, 8 KV heads,
//! per-head q/k RMSNorm, NEOX RoPE 128 dims theta 5e5, softplus per-head output gate, SwiGLU FFN 12288), the
//! target's `token_embd` and `output` head. Context: the target's residual stream entering layers
//! 2, 11, 20, 30, 39 and leaving the last layer (`TAP_LAYERS`), each RMS-normalised with its own `enc.aux_norm` row,
//! concatenated (18432), projected by `fc`, normalised by `enc.output_norm`; every drafter layer projects that through
//! its own `attn_norm`, `attn_k`, `attn_v` (+ k_norm, RoPE at the target position) into its KV ring. A draft is one
//! causal block `[anchor, mask x k]` at positions `P .. P + k` over a 1024-token sliding window (halo: DFLASH_SWA=1024);
//! rows 1 .. k of the head's argmax are the drafts.
//!
//! Every launch reuses the target's Laguna kernels (`tk_qkv_norm_rope_store`, `tk_laguna_decode_attn`,
//! `tk_attn_merge_gate_quant` = the softplus gate, `tk_mmvq_batch`, ...): no new device code. The drafter only
//! proposes tokens; the target verifies them, so greedy output does not depend on it.

#![allow(non_snake_case)]

use super::*;

use talus_gguf::GgufFile;

/// Drafter sliding window (halo `DFLASH_SWA=1024`).
pub const DF_WINDOW: usize = 1024;
/// Ring positions per drafter layer: the window, a 16-aligned tile, a block and a verify's injection fit.
const DF_STRIDE: usize = DF_WINDOW + 64 + K_MAX_VERIFY;
/// Largest block (anchor + drafts): the trained block size.
pub const DF_MAX_BLOCK: usize = 16;
/// Rows per context-injection chunk (the dense mat-vec's column path, expert_major = 2, takes up to 64).
const DF_INJ: usize = 32;
const DF_LAYERS: usize = 6;
const DF_HEADS: usize = 72;
const DF_FF: usize = 12288;
const DF_FEAT: usize = TAP_LAYERS.len() * K_EMBD;

struct DLayer {
    attn_norm: DevTensor,
    wq: DevTensor,
    wk: DevTensor,
    wv: DevTensor,
    wo: DevTensor,
    q_norm: DevTensor,
    k_norm: DevTensor,
    gate: DevTensor,
    ffn_norm: DevTensor,
    f_gate: DevTensor,
    f_up: DevTensor,
    f_down: DevTensor,
    kpool: *mut u16,
    vpool: *mut u16,
}

/// The Laguna DFlash drafter: weights, KV rings and scratch. Runs on the target engine's stream.
pub struct LagunaDFlash {
    aux_norm: DevTensor,
    enc_norm: DevTensor,
    fc: DevTensor,
    out_norm: DevTensor,
    layers: Vec<DLayer>,
    pub mask_id: i32,
    cos: *mut f32,
    sin: *mut f32,
    max_pos: usize,
    // context injection scratch (DF_INJ rows)
    itmp: *mut f32,
    ifeat: *mut f32,
    iq8: *mut c_void,
    ige: *mut f32,
    igin: *mut f32,
    icq8: *mut c_void,
    ik: *mut f32,
    iv: *mut f32,
    iqd: *mut f32,
    iidx: *mut i32,
    // block scratch (DF_MAX_BLOCK rows)
    bx: *mut f32,
    bq8: *mut c_void,
    bq8b: *mut c_void,
    bq: *mut f32,
    bk: *mut f32,
    bv: *mut f32,
    bg: *mut f32,
    ba: *mut f32,
    bf1: *mut f32,
    bf2: *mut f32,
    bpart: *mut f32,
    bml: *mut f32,
    blogits: *mut f32,
    bout: *mut i32,
    btok: *mut i32,
    pos: *mut i32,
    // top-16 candidates (ids, logits) per block row for sampled requests (DF_MAX_BLOCK x 16)
    bcand: *mut i32,
    bunary: *mut f32,
    // confident-prefix verify (talus_spec::ConfPrefix): tk_q36_draft_conf scalars ([0] tau 0, [1] lmin -inf,
    // [2] pick -1 as i32 bits) and outputs ([16, 32) conf, [32, 48) per-row top-1 probability); null when off
    conf_dev: *mut f32,
    conf_tops: Vec<f32>,
    q8_row: usize,
    // pinned host staging: [0, 16) block tokens, [16, 32) drafts out, [32, 32 + DF_POS_SLOTS) chunk positions
    host: *mut i32,
    // HIP graphs: the block forward per block size, the context injection per row count (rows 0 .. n of the taps)
    g_draft: [HipGraphExec; DF_MAX_BLOCK + 1],
    g_inj: [HipGraphExec; DF_INJ + 1],
    use_graph: bool,
    allocs: Vec<*mut c_void>,
    hip: Arc<HipRuntime>,
    /// Rounds drafted / tokens proposed (stats).
    pub n_rounds: u64,
    pub n_drafted: u64,
}

unsafe impl Send for LagunaDFlash {}
unsafe impl Sync for LagunaDFlash {}

impl Drop for LagunaDFlash {
    fn drop(&mut self) {
        for ex in self.g_draft.iter().chain(self.g_inj.iter()) {
            if !ex.is_null() {
                unsafe {
                    let _ = (self.hip.hipGraphExecDestroy)(*ex);
                }
            }
        }
        self.hip.free_pinned(self.host as *mut c_void);
        for &p in &self.allocs {
            self.hip.free_device(p);
        }
    }
}

/// Pinned position slots (one per injection chunk of a prefill).
const DF_POS_SLOTS: usize = 256;

fn df_tensor(
    g: &GgufFile,
    hip: &HipRuntime,
    allocs: &mut Vec<*mut c_void>,
    name: &str,
    expect: &[i64],
) -> Result<DevTensor> {
    let info = g
        .find_tensor(name)
        .ok_or_else(|| CoreError::MissingTensor {
            name: name.to_string(),
        })?;
    let bytes = g.tensor_data(info).map_err(|e| CoreError::IoError {
        message: format!("{name}: {e}"),
    })?;
    let n_in = info.dims[0] as i64;
    let n_out = if info.n_dims > 1 {
        info.dims[1] as i64
    } else {
        1
    };
    let (bs, ts) = talus_gguf::block_geometry(info.qtype);
    let t = DevTensor {
        dev: hip.alloc_device(bytes.len())?,
        qtype: info.qtype,
        ne: [
            info.dims[0] as i64,
            info.dims[1] as i64,
            info.dims[2] as i64,
            info.dims[3] as i64,
        ],
        n_dims: info.n_dims as i32,
        nbytes: bytes.len(),
        row_bytes: (n_in as usize / bs as usize) * ts as usize,
        n_in,
        n_out,
    };
    allocs.push(t.dev);
    hip.copy_h2d(t.dev, bytes.as_ptr() as *const c_void, bytes.len())?;
    t.check(name, expect)?;
    Ok(t)
}

impl LagunaDFlash {
    /// Load the drafter GGUF and allocate its rings / scratch for target positions `< max_pos`.
    pub fn load(path: &Path, eng: &Engine, max_pos: usize) -> Result<Self> {
        if eng.taps.is_null() {
            return Err(CoreError::InvalidArgument {
                reason: "laguna dflash: the engine was created without taps (Options::taps)"
                    .to_string(),
            });
        }
        let g = GgufFile::open(path).map_err(|e| CoreError::IoError {
            message: format!("{}: {e}", path.display()),
        })?;
        if g.get_str("general.architecture") != Some("dflash") {
            return Err(CoreError::InvalidArgument {
                reason: format!("{}: not a dflash GGUF", path.display()),
            });
        }
        if g.get_u64("dflash.block_count") != Some(DF_LAYERS as u64)
            || g.get_u64("dflash.embedding_length") != Some(K_EMBD as u64)
        {
            return Err(CoreError::InvalidArgument {
                reason: "laguna dflash: expected 6 layers of 3072".to_string(),
            });
        }
        let mask_id = g.get_u64("tokenizer.ggml.mask_token_id").unwrap_or(12) as i32;
        let hip = Arc::clone(&eng.hip);
        let mut allocs: Vec<*mut c_void> = Vec::new();
        let res = (|| -> Result<Self> {
            let e = K_EMBD as i64;
            let kvw = (K_HEAD_KV * K_HEAD_DIM) as i64;
            let qw = (DF_HEADS * K_HEAD_DIM) as i64;
            let aux_norm = df_tensor(&g, &hip, &mut allocs, "enc.aux_norm.weight", &[e, 6])?;
            let enc_norm = df_tensor(&g, &hip, &mut allocs, "enc.output_norm.weight", &[e])?;
            let fc = df_tensor(&g, &hip, &mut allocs, "fc.weight", &[DF_FEAT as i64, e])?;
            let out_norm = df_tensor(&g, &hip, &mut allocs, "output_norm.weight", &[e])?;
            let mut layers = Vec::with_capacity(DF_LAYERS);
            for il in 0..DF_LAYERS {
                let t = |allocs: &mut Vec<*mut c_void>, n: &str, x: &[i64]| {
                    df_tensor(&g, &hip, allocs, &format!("blk.{il}.{n}.weight"), x)
                };
                let pool_bytes = DF_STRIDE * K_HEAD_KV * K_HEAD_DIM * 2;
                let kpool = hip.alloc_device_zeroed(pool_bytes)?;
                allocs.push(kpool);
                let vpool = hip.alloc_device_zeroed(pool_bytes)?;
                allocs.push(vpool);
                layers.push(DLayer {
                    attn_norm: t(&mut allocs, "attn_norm", &[e])?,
                    wq: t(&mut allocs, "attn_q", &[e, qw])?,
                    wk: t(&mut allocs, "attn_k", &[e, kvw])?,
                    wv: t(&mut allocs, "attn_v", &[e, kvw])?,
                    wo: t(&mut allocs, "attn_output", &[qw, e])?,
                    q_norm: t(&mut allocs, "attn_q_norm", &[K_HEAD_DIM as i64])?,
                    k_norm: t(&mut allocs, "attn_k_norm", &[K_HEAD_DIM as i64])?,
                    gate: t(&mut allocs, "attn_gate", &[e, DF_HEADS as i64])?,
                    ffn_norm: t(&mut allocs, "ffn_norm", &[e])?,
                    f_gate: t(&mut allocs, "ffn_gate", &[e, DF_FF as i64])?,
                    f_up: t(&mut allocs, "ffn_up", &[e, DF_FF as i64])?,
                    f_down: t(&mut allocs, "ffn_down", &[DF_FF as i64, e])?,
                    kpool: kpool as *mut u16,
                    vpool: vpool as *mut u16,
                });
            }

            let mut alloc = |bytes: usize| -> Result<*mut c_void> {
                let p = hip.alloc_device_zeroed(bytes)?;
                allocs.push(p);
                Ok(p)
            };
            // RoPE: 128 dims, theta 5e5, no scaling (the drafter GGUF carries no rope scaling keys)
            let cos = alloc(max_pos * 64 * 4)? as *mut f32;
            let sin = alloc(max_pos * 64 * 4)? as *mut f32;
            {
                let mut hc = vec![0.0_f32; max_pos * 64];
                let mut hs = vec![0.0_f32; max_pos * 64];
                let p = RopeScalingParams {
                    scaling_type: TK_ROPE_SCALING_NONE,
                    freq_base: 500_000.0,
                    ..Default::default()
                };
                tk_build_rope_table(&mut hc, &mut hs, max_pos, 128, &p);
                hip.copy_h2d(
                    cos as *mut c_void,
                    hc.as_ptr() as *const c_void,
                    hc.len() * 4,
                )?;
                hip.copy_h2d(
                    sin as *mut c_void,
                    hs.as_ptr() as *const c_void,
                    hs.len() * 4,
                )?;
            }
            let tk = &eng.tk;
            let q8b = |n: usize, rows: usize| unsafe { (tk.tk_q8_1_bytes)(n as i32, rows as i32) };
            let itmp = alloc(TAP_LAYERS.len() * DF_INJ * K_EMBD * 4)? as *mut f32;
            let ifeat = alloc(DF_INJ * DF_FEAT * 4)? as *mut f32;
            let iq8 = alloc(q8b(DF_FEAT, DF_INJ))?;
            let ige = alloc(DF_INJ * K_EMBD * 4)? as *mut f32;
            let igin = alloc(DF_INJ * K_EMBD * 4)? as *mut f32;
            let icq8 = alloc(q8b(K_EMBD, DF_INJ))?;
            let ik = alloc(DF_INJ * K_HEAD_KV * K_HEAD_DIM * 4)? as *mut f32;
            let iv = alloc(DF_INJ * K_HEAD_KV * K_HEAD_DIM * 4)? as *mut f32;
            let iqd = alloc(DF_INJ * K_HEAD_KV * K_HEAD_DIM * 4)? as *mut f32;
            let iidx = alloc(TAP_LAYERS.len() * DF_INJ * 4)? as *mut i32;
            {
                // feature row t * 6 + k <- normalised tap k, row t (itmp is [6][DF_INJ][K_EMBD])
                let idx: Vec<i32> = (0..TAP_LAYERS.len() * DF_INJ)
                    .map(|r| ((r % TAP_LAYERS.len()) * DF_INJ + r / TAP_LAYERS.len()) as i32)
                    .collect();
                hip.copy_h2d(
                    iidx as *mut c_void,
                    idx.as_ptr() as *const c_void,
                    idx.len() * 4,
                )?;
            }
            let nb = DF_MAX_BLOCK;
            let bx = alloc(nb * K_EMBD * 4)? as *mut f32;
            let bq8 = alloc(q8b(DF_FF, nb))?;
            let bq8b = alloc(q8b(DF_FF, nb))?;
            let bq = alloc(nb * DF_HEADS * K_HEAD_DIM * 4)? as *mut f32;
            let bk = alloc(nb * K_HEAD_KV * K_HEAD_DIM * 4)? as *mut f32;
            let bv = alloc(nb * K_HEAD_KV * K_HEAD_DIM * 4)? as *mut f32;
            let bg = alloc(nb * DF_HEADS * 4)? as *mut f32;
            let ba = alloc(nb * K_EMBD * 4)? as *mut f32;
            let bf1 = alloc(nb * DF_FF * 4)? as *mut f32;
            let bf2 = alloc(nb * DF_FF * 4)? as *mut f32;
            let bpart = alloc(nb * DF_HEADS * (K_SPLIT as usize) * K_HEAD_DIM * 4)? as *mut f32;
            let bml = alloc(nb * DF_HEADS * (K_SPLIT as usize) * 2 * 4)? as *mut f32;
            let blogits = alloc(nb * K_VOCAB * 4)? as *mut f32;
            let bout = alloc(nb * 4)? as *mut i32;
            let btok = alloc(nb * 4)? as *mut i32;
            let pos = alloc(16)? as *mut i32;
            let bcand = alloc(nb * 16 * 4)? as *mut i32;
            let bunary = alloc(nb * 16 * 4)? as *mut f32;
            let conf_dev = if talus_spec::ConfPrefix::from_env("LAGUNA").active() {
                let p = alloc(64 * 4)? as *mut f32;
                let mut init = [0f32; 64];
                init[1] = f32::NEG_INFINITY;
                init[2] = f32::from_bits((-1i32) as u32);
                hip.copy_h2d(p as *mut c_void, init.as_ptr() as *const c_void, 64 * 4)?;
                p
            } else {
                std::ptr::null_mut()
            };
            let q8_row = q8b(K_EMBD, 1);
            let host = hip.alloc_pinned((32 + DF_POS_SLOTS) * 4)? as *mut i32;
            Ok(Self {
                aux_norm,
                enc_norm,
                fc,
                out_norm,
                layers,
                mask_id,
                cos,
                sin,
                max_pos,
                itmp,
                ifeat,
                iq8,
                ige,
                igin,
                icq8,
                ik,
                iv,
                iqd,
                iidx,
                bx,
                bq8,
                bq8b,
                bq,
                bk,
                bv,
                bg,
                ba,
                bf1,
                bf2,
                bpart,
                bml,
                blogits,
                bout,
                btok,
                pos,
                bcand,
                bunary,
                conf_dev,
                conf_tops: Vec::new(),
                q8_row,
                host,
                g_draft: [std::ptr::null_mut(); DF_MAX_BLOCK + 1],
                g_inj: [std::ptr::null_mut(); DF_INJ + 1],
                use_graph: eng.use_graph
                    && std::env::var("TALUS_LAGUNA_DF_GRAPH")
                        .map(|v| v != "0")
                        .unwrap_or(true),
                allocs: std::mem::take(&mut allocs),
                hip: Arc::clone(&hip),
                n_rounds: 0,
                n_drafted: 0,
            })
        })();
        if res.is_err() {
            for p in allocs {
                hip.free_device(p);
            }
        }
        res
    }

    /// `*self.pos = p` on the stream, from pinned slot `slot` (a slot is not rewritten before the stream has passed
    /// its copy: every public call ends with a stream synchronisation).
    fn set_pos(&self, eng: &Engine, p: i32, slot: usize) -> Result<()> {
        assert!(slot < DF_POS_SLOTS);
        unsafe {
            *self.host.add(32 + slot) = p;
        }
        check_hip(
            unsafe {
                (eng.hip.hipMemcpyAsync)(
                    self.pos as *mut c_void,
                    self.host.add(32 + slot) as *const c_void,
                    4,
                    HipRuntime::MEMCPY_H2D,
                    eng.s,
                )
            },
            "hipMemcpyAsync(dflash pos)",
        )
    }

    /// Run `body` on the stream, through a graph captured on first use when `slot` is set.
    fn run_graph<F>(eng: &Engine, slot: Option<&mut HipGraphExec>, body: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        let Some(ex) = slot else {
            return body();
        };
        Self::capture(eng, ex, body)?;
        check_hip(
            unsafe { (eng.hip.hipGraphLaunch)(*ex, eng.s) },
            "hipGraphLaunch(dflash)",
        )
    }

    /// Capture `body` into `*ex` unless it holds a graph already (nothing runs).
    fn capture<F>(eng: &Engine, ex: &mut HipGraphExec, body: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        if ex.is_null() {
            check_hip(
                unsafe { (eng.hip.hipStreamSynchronize)(eng.s) },
                "sync(before dflash capture)",
            )?;
            check_hip(
                unsafe {
                    (eng.hip.hipStreamBeginCapture)(eng.s, HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                },
                "hipStreamBeginCapture(dflash)",
            )?;
            let r = body();
            let mut g: HipGraph = std::ptr::null_mut();
            let end = unsafe { (eng.hip.hipStreamEndCapture)(eng.s, &mut g) };
            r?;
            check_hip(end, "hipStreamEndCapture(dflash)")?;
            let mut e: HipGraphExec = std::ptr::null_mut();
            let inst = check_hip(
                unsafe {
                    (eng.hip.hipGraphInstantiate)(
                        &mut e,
                        g,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0,
                    )
                },
                "hipGraphInstantiate(dflash)",
            );
            unsafe {
                let _ = (eng.hip.hipGraphDestroy)(g);
            }
            inst?;
            *ex = e;
        }
        Ok(())
    }

    /// Capture the block-forward graphs of block sizes `2 ..= block_max` and the decode / verify injection
    /// graphs of `1 ..= inj_max` rows up front (serving then never captures one). Returns the number of graphs.
    pub fn capture_graphs(
        &mut self,
        eng: &Engine,
        block_max: usize,
        inj_max: usize,
    ) -> Result<usize> {
        if !self.use_graph {
            return Ok(0);
        }
        let mut n_graphs = 0usize;
        for n in 2..=block_max.min(DF_MAX_BLOCK) {
            let mut ex = std::mem::replace(&mut self.g_draft[n], std::ptr::null_mut());
            let me = &*self;
            let r = Self::capture(eng, &mut ex, || me.draft_body(eng, n));
            self.g_draft[n] = ex;
            r?;
            n_graphs += 1;
        }
        for c in 1..=inj_max.min(DF_INJ) {
            let mut ex = std::mem::replace(&mut self.g_inj[c], std::ptr::null_mut());
            let me = &*self;
            let r = Self::capture(eng, &mut ex, || me.inject_chunk(eng, 0, c));
            self.g_inj[c] = ex;
            r?;
            n_graphs += 1;
        }
        Ok(n_graphs)
    }

    fn dense(&self, eng: &Engine, items: &[tk_mmvq_item], what: &'static str) -> Result<()> {
        // the engine's dense path (tk_mmvq / wave32 when enabled, else tk_mmvq_batch)
        eng.run_items(items, what)
    }

    #[inline]
    fn item(eng: &Engine, t: &DevTensor, xq: *const c_void, y: *mut f32, n: usize) -> tk_mmvq_item {
        let mut it = eng.mmvq_item(t, xq, y, eng.vzero, n as i32, 0, true);
        it.expert_major = 2; // every column through the column path (8 at a time)
        it
    }

    /// Inject the target context of `n` positions `p0 .. p0 + n` from the engine's tap rows `r0 .. r0 + n` (the last
    /// prefill chunk, decode step or verify). Positions older than the window are skipped by the caller.
    pub fn inject(&mut self, eng: &Engine, r0: usize, p0: usize, n: usize) -> Result<()> {
        assert!(r0 + n <= eng.tap_rows && p0 + n <= self.max_pos);
        let mut done = 0usize;
        let mut slot = 0usize;
        while done < n {
            let c = (n - done).min(DF_INJ);
            self.set_pos(eng, (p0 + done) as i32, slot)?;
            slot += 1;
            if slot == DF_POS_SLOTS {
                check_hip(
                    unsafe { (eng.hip.hipStreamSynchronize)(eng.s) },
                    "sync(dflash pos slots)",
                )?;
                slot = 0;
            }
            let r = r0 + done;
            // decode / verify injections (rows 0 .. c of the taps) replay a graph per row count
            let mut gslot = if r == 0 && self.use_graph {
                Some(std::mem::replace(&mut self.g_inj[c], std::ptr::null_mut()))
            } else {
                None
            };
            let me = &*self;
            let res = Self::run_graph(eng, gslot.as_mut(), || me.inject_chunk(eng, r, c));
            if let Some(ex) = gslot {
                self.g_inj[c] = ex;
            }
            res?;
            done += c;
        }
        check_hip(
            unsafe { (eng.hip.hipStreamSynchronize)(eng.s) },
            "sync(dflash inject)",
        )
    }

    /// One injection chunk: tap rows `r .. r + c` -> the rings at positions `*self.pos ..`.
    fn inject_chunk(&self, eng: &Engine, r: usize, c: usize) -> Result<()> {
        let ctx = eng.tk_ctx.as_ptr();
        let st = eng.s as tk_stream;
        let e = K_EMBD;
        {
            for k in 0..TAP_LAYERS.len() {
                check_tk(
                    ctx,
                    unsafe {
                        (eng.tk.tk_rms_norm_to)(
                            ctx,
                            self.itmp.add(k * DF_INJ * e),
                            eng.taps.add((k * eng.tap_rows + r) * e),
                            (self.aux_norm.dev as *const f32).add(k * e),
                            c as i32,
                            e as i32,
                            K_EPS,
                            st,
                        )
                    },
                    "tk_rms_norm_to(dflash aux)",
                )?;
            }
            check_tk(
                ctx,
                unsafe {
                    (eng.tk.tk_gather_rows)(
                        ctx,
                        self.itmp as *const c_void,
                        self.ifeat as *mut c_void,
                        self.iidx,
                        (TAP_LAYERS.len() * c) as i32,
                        e * 4,
                        st,
                    )
                },
                "tk_gather_rows(dflash feat)",
            )?;
            check_tk(
                ctx,
                unsafe {
                    (eng.tk.tk_quantize_q8_1)(
                        ctx,
                        self.ifeat,
                        self.iq8,
                        DF_FEAT as i32,
                        c as i32,
                        st,
                    )
                },
                "tk_quantize_q8_1(dflash feat)",
            )?;
            self.dense(
                eng,
                &[Self::item(eng, &self.fc, self.iq8, self.ige, c)],
                "mmvq(dflash fc)",
            )?;
            check_tk(
                ctx,
                unsafe {
                    (eng.tk.tk_rms_norm_to)(
                        ctx,
                        self.igin,
                        self.ige,
                        self.enc_norm.as_f32(),
                        c as i32,
                        e as i32,
                        K_EPS,
                        st,
                    )
                },
                "tk_rms_norm_to(dflash enc)",
            )?;
            for L in &self.layers {
                check_tk(
                    ctx,
                    unsafe {
                        (eng.tk.tk_add_norm_quant)(
                            ctx,
                            self.igin,
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            0,
                            L.attn_norm.as_f32(),
                            std::ptr::null_mut(),
                            self.icq8,
                            e as i32,
                            K_EPS,
                            c as i32,
                            st,
                        )
                    },
                    "tk_add_norm_quant(dflash ctx)",
                )?;
                self.dense(
                    eng,
                    &[
                        Self::item(eng, &L.wk, self.icq8, self.ik, c),
                        Self::item(eng, &L.wv, self.icq8, self.iv, c),
                    ],
                    "mmvq(dflash ctx kv)",
                )?;
                check_tk(
                    ctx,
                    unsafe {
                        (eng.tk.tk_qkv_norm_rope_store)(
                            ctx,
                            self.iqd,
                            self.ik,
                            self.iv,
                            L.k_norm.as_f32(),
                            L.k_norm.as_f32(),
                            K_HEAD_KV as i32,
                            K_HEAD_KV as i32,
                            K_EPS,
                            128,
                            self.cos,
                            self.sin,
                            self.pos,
                            L.kpool as *mut c_void,
                            L.vpool as *mut c_void,
                            DF_STRIDE as i32,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            c as i32,
                            st,
                        )
                    },
                    "tk_qkv_norm_rope_store(dflash ctx)",
                )?;
            }
        }
        Ok(())
    }

    /// Draft `k` (1 ..= 15) tokens after `anchor` at position `p` (the context must be injected up to `p - 1`).
    pub fn draft(
        &mut self,
        eng: &Engine,
        anchor: i32,
        p: usize,
        k: usize,
        out: &mut [i32],
    ) -> Result<usize> {
        let k = k.min(DF_MAX_BLOCK - 1).min(out.len());
        if k == 0 || p + k + 1 > self.max_pos {
            return Ok(0);
        }
        let n = k + 1;
        unsafe {
            *self.host = anchor;
            for i in 1..n {
                *self.host.add(i) = self.mask_id;
            }
        }
        check_hip(
            unsafe {
                (eng.hip.hipMemcpyAsync)(
                    self.btok as *mut c_void,
                    self.host as *const c_void,
                    n * 4,
                    HipRuntime::MEMCPY_H2D,
                    eng.s,
                )
            },
            "hipMemcpyAsync(dflash block)",
        )?;
        self.set_pos(eng, p as i32, 0)?;
        let mut gslot = if self.use_graph {
            Some(std::mem::replace(
                &mut self.g_draft[n],
                std::ptr::null_mut(),
            ))
        } else {
            None
        };
        let me = &*self;
        let res = Self::run_graph(eng, gslot.as_mut(), || me.draft_body(eng, n));
        if let Some(ex) = gslot {
            self.g_draft[n] = ex;
        }
        res?;
        // the drafter's top-1 probability per draft row (blogits row i = draft i), outside the graph
        self.conf_tops.clear();
        if !self.conf_dev.is_null() {
            let ctx = eng.tk_ctx.as_ptr();
            let cd = self.conf_dev;
            for i in 0..k {
                check_tk(
                    ctx,
                    unsafe {
                        (eng.tk.tk_q36_draft_conf)(
                            ctx,
                            self.blogits.add(i * K_VOCAB) as *const f32,
                            K_VOCAB as i32,
                            cd as *const f32,
                            cd.add(1) as *const f32,
                            cd.add(2) as *const i32,
                            std::ptr::null(),
                            cd.add(16),
                            cd.add(32),
                            i as i32,
                            std::ptr::null(),
                            std::ptr::null(),
                            eng.s as tk_stream,
                        )
                    },
                    "tk_q36_draft_conf(laguna dflash)",
                )?;
            }
            self.conf_tops.resize(k, 0.0);
            check_hip(
                unsafe {
                    (eng.hip.hipMemcpyAsync)(
                        self.conf_tops.as_mut_ptr() as *mut c_void,
                        cd.add(32) as *const c_void,
                        k * 4,
                        HipRuntime::MEMCPY_D2H,
                        eng.s,
                    )
                },
                "hipMemcpyAsync(dflash conf)",
            )?;
        }
        check_hip(
            unsafe {
                (eng.hip.hipMemcpyAsync)(
                    self.host.add(16) as *mut c_void,
                    self.bout as *const c_void,
                    k * 4,
                    HipRuntime::MEMCPY_D2H,
                    eng.s,
                )
            },
            "hipMemcpyAsync(dflash drafts)",
        )?;
        check_hip(
            unsafe { (eng.hip.hipStreamSynchronize)(eng.s) },
            "sync(dflash drafts)",
        )?;
        for (i, o) in out.iter_mut().enumerate().take(k) {
            *o = unsafe { *self.host.add(16 + i) };
        }
        self.n_rounds += 1;
        self.n_drafted += k as u64;
        Ok(k)
    }

    /// the drafter's top-1 probability of each row of the last [`Self::draft`] (empty unless a
    /// confident-prefix switch or its trace is on; talus_spec::ConfPrefix).
    pub fn last_tops(&self) -> &[f32] {
        &self.conf_tops
    }

    /// the top-16 head candidates `(token id, logit)` of each of the `k` rows the last [`Self::draft`]
    /// drafted (`tk_q36_dflash_top16` over `blogits`, descending logit, ties to the lower id), for the proposal
    /// distribution q of exact speculative sampling. Call right after `draft(.., k, ..)`.
    pub fn draft_cands(&mut self, eng: &Engine, k: usize) -> Result<Vec<Vec<(i32, f32)>>> {
        let k = k.min(DF_MAX_BLOCK - 1);
        if k == 0 {
            return Ok(Vec::new());
        }
        let ctx = eng.tk_ctx.as_ptr();
        let st = eng.s as tk_stream;
        // top16 reads rows 1 .. rows-1 at `logits + r * nv` and writes cand / unary rows 1 .. rows-1: point its
        // base one row before `blogits` (row 0 is never read), so row r is `blogits` row r - 1 = draft r - 1
        let base = self.blogits.wrapping_sub(K_VOCAB) as *const f32;
        check_tk(
            ctx,
            unsafe {
                (eng.tk.tk_q36_dflash_top16)(
                    ctx,
                    base,
                    K_VOCAB as i32,
                    (k + 1) as i32,
                    std::ptr::null(),
                    self.bcand,
                    self.bunary,
                    st,
                )
            },
            "tk_q36_dflash_top16(laguna dflash)",
        )?;
        let n = (k + 1) * 16;
        let mut ids = vec![0i32; n];
        let mut lg = vec![0f32; n];
        check_hip(
            unsafe {
                (eng.hip.hipMemcpyAsync)(
                    ids.as_mut_ptr() as *mut c_void,
                    self.bcand as *const c_void,
                    n * 4,
                    HipRuntime::MEMCPY_D2H,
                    eng.s,
                )
            },
            "hipMemcpyAsync(dflash cands)",
        )?;
        check_hip(
            unsafe {
                (eng.hip.hipMemcpyAsync)(
                    lg.as_mut_ptr() as *mut c_void,
                    self.bunary as *const c_void,
                    n * 4,
                    HipRuntime::MEMCPY_D2H,
                    eng.s,
                )
            },
            "hipMemcpyAsync(dflash cand logits)",
        )?;
        check_hip(
            unsafe { (eng.hip.hipStreamSynchronize)(eng.s) },
            "sync(dflash cands)",
        )?;
        Ok((1..=k)
            .map(|r| {
                (r * 16..(r + 1) * 16)
                    .filter(|&j| lg[j].is_finite() && ids[j] >= 0 && (ids[j] as usize) < K_VOCAB)
                    .map(|j| (ids[j], lg[j]))
                    .collect()
            })
            .collect())
    }

    /// The block forward of `n` tokens (`btok`, at `*self.pos`): drafts for rows 1 .. n into `bout`.
    fn draft_body(&self, eng: &Engine, n: usize) -> Result<()> {
        let k = n - 1;
        let ctx = eng.tk_ctx.as_ptr();
        let st = eng.s as tk_stream;
        let e = K_EMBD;
        check_tk(
            ctx,
            unsafe {
                (eng.tk.tk_embed_q8_0)(
                    ctx,
                    eng.m.emb.dev as *const c_void,
                    self.btok,
                    self.bx,
                    e as i32,
                    n as i32,
                    st,
                )
            },
            "tk_embed_q8_0(dflash)",
        )?;
        check_tk(
            ctx,
            unsafe {
                (eng.tk.tk_add_norm_quant)(
                    ctx,
                    self.bx,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    self.layers[0].attn_norm.as_f32(),
                    std::ptr::null_mut(),
                    self.bq8,
                    e as i32,
                    K_EPS,
                    n as i32,
                    st,
                )
            },
            "tk_add_norm_quant(dflash in)",
        )?;
        for (il, L) in self.layers.iter().enumerate() {
            let next_norm = if il + 1 < DF_LAYERS {
                self.layers[il + 1].attn_norm.as_f32()
            } else {
                self.out_norm.as_f32()
            };
            self.dense(
                eng,
                &[
                    Self::item(eng, &L.wq, self.bq8, self.bq, n),
                    Self::item(eng, &L.wk, self.bq8, self.bk, n),
                    Self::item(eng, &L.wv, self.bq8, self.bv, n),
                    Self::item(eng, &L.gate, self.bq8, self.bg, n),
                ],
                "mmvq(dflash qkvg)",
            )?;
            check_tk(
                ctx,
                unsafe {
                    (eng.tk.tk_qkv_norm_rope_store)(
                        ctx,
                        self.bq,
                        self.bk,
                        self.bv,
                        L.q_norm.as_f32(),
                        L.k_norm.as_f32(),
                        DF_HEADS as i32,
                        K_HEAD_KV as i32,
                        K_EPS,
                        128,
                        self.cos,
                        self.sin,
                        self.pos,
                        L.kpool as *mut c_void,
                        L.vpool as *mut c_void,
                        DF_STRIDE as i32,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        n as i32,
                        st,
                    )
                },
                "tk_qkv_norm_rope_store(dflash)",
            )?;
            check_tk(
                ctx,
                unsafe {
                    (eng.tk.tk_laguna_decode_attn)(
                        ctx,
                        self.bq,
                        L.kpool,
                        L.vpool,
                        std::ptr::null(),
                        std::ptr::null(),
                        DF_HEADS as i32,
                        K_HEAD_KV as i32,
                        DF_WINDOW as i32,
                        DF_STRIDE as i32,
                        self.pos,
                        K_SPLIT,
                        self.bpart,
                        self.bml,
                        n as i32,
                        std::ptr::null(),
                        0,
                        false,
                        16,
                        st,
                    )
                },
                "tk_laguna_decode_attn(dflash)",
            )?;
            check_tk(
                ctx,
                unsafe {
                    (eng.tk.tk_attn_merge_gate_quant)(
                        ctx,
                        self.bpart,
                        self.bml,
                        self.bg,
                        self.bq8,
                        DF_HEADS as i32,
                        K_SPLIT,
                        n as i32,
                        st,
                    )
                },
                "tk_attn_merge_gate_quant(dflash)",
            )?;
            self.dense(
                eng,
                &[Self::item(eng, &L.wo, self.bq8, self.ba, n)],
                "mmvq(dflash wo)",
            )?;
            check_tk(
                ctx,
                unsafe {
                    (eng.tk.tk_add_norm_quant)(
                        ctx,
                        self.bx,
                        self.ba,
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        L.ffn_norm.as_f32(),
                        std::ptr::null_mut(),
                        self.bq8,
                        e as i32,
                        K_EPS,
                        n as i32,
                        st,
                    )
                },
                "tk_add_norm_quant(dflash ffn)",
            )?;
            self.dense(
                eng,
                &[
                    Self::item(eng, &L.f_gate, self.bq8, self.bf1, n),
                    Self::item(eng, &L.f_up, self.bq8, self.bf2, n),
                ],
                "mmvq(dflash gate+up)",
            )?;
            check_tk(
                ctx,
                unsafe {
                    (eng.tk.tk_swiglu_quantize_q8_1)(
                        ctx,
                        self.bf1,
                        self.bf2,
                        self.bq8b,
                        DF_FF as i32,
                        n as i32,
                        st,
                    )
                },
                "tk_swiglu_quantize_q8_1(dflash)",
            )?;
            self.dense(
                eng,
                &[Self::item(eng, &L.f_down, self.bq8b, self.ba, n)],
                "mmvq(dflash down)",
            )?;
            check_tk(
                ctx,
                unsafe {
                    (eng.tk.tk_add_norm_quant)(
                        ctx,
                        self.bx,
                        self.ba,
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        next_norm,
                        std::ptr::null_mut(),
                        self.bq8,
                        e as i32,
                        K_EPS,
                        n as i32,
                        st,
                    )
                },
                "tk_add_norm_quant(dflash out)",
            )?;
        }
        // the target head over block rows 1 .. k (the masks predict themselves)
        let head_x = unsafe { (self.bq8 as *const u8).add(self.q8_row) as *const c_void };
        self.dense(
            eng,
            &[Self::item(eng, &eng.m.output, head_x, self.blogits, k)],
            "mmvq(dflash head)",
        )?;
        check_tk(
            ctx,
            unsafe {
                (eng.tk.tk_argmax_rows)(ctx, self.blogits, k as i32, K_VOCAB as i32, self.bout, st)
            },
            "tk_argmax_rows(dflash)",
        )?;
        Ok(())
    }
}
