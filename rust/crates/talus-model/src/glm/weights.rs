//! GLM-5.3-Flash weights: hyper-parameters from the GGUF header, the per-layer tensor table, and the one-pass
//! loader (pread -> pinned stage -> device, page cache dropped behind the reads, = llama.cpp `--load-mode none`).

use std::ffi::c_void;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::ptr;

use talus_core_api::{CoreError, Result};
use talus_gguf::{GgufModel, GgufValue, TensorEntry};
use talus_mem::{Budget, BudgetCategory};

use super::ffi::{self, dev_alloc};

pub const ARCH_NAME: &str = "glm5-next";

/// ggml type ids used by GLM-5.3-Flash.
pub mod gt {
    pub const F32: u32 = 0;
    pub const Q8_0: u32 = 8;
    pub const Q2_K: u32 = 10;
    pub const Q3_K: u32 = 11;
    pub const Q4_K: u32 = 12;
    pub const Q5_K: u32 = 13;
    pub const Q6_K: u32 = 14;
    pub const IQ3_S: u32 = 21;
    pub const IQ2_S: u32 = 22;
}

#[derive(Debug, Clone)]
pub struct GlmHparams {
    pub n_layer_all: usize, // 46 incl. NextN
    pub n_layer: usize,     // 45 trunk
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub head_kda: usize,
    pub d_conv: usize,
    pub kda_lb: f32,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub head_k_mla: usize,
    pub head_v_mla: usize,
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub idx_topk: usize,
    pub kpool: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff: usize,
    pub n_ff_exp: usize,
    pub n_dense_lead: usize,
    pub w_scale: f32,
    pub w_norm: bool,
    pub clamp_exp: Vec<f32>,
    pub clamp_shexp: Vec<f32>,
    pub hc_mult: usize,
    pub hc_iters: usize,
    pub hc_eps: f32,
    pub rms_eps: f32,
    pub norm_eps: f32,
    pub is_recr: Vec<bool>,
    pub ctx_train: usize,
    pub eos: Vec<u32>,
    pub bos: u32,
}

fn md_u(g: &GgufModel, k: &str) -> Result<usize> {
    g.get_metadata(k)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| CoreError::InvalidArgument {
            reason: format!("missing GGUF key {k}"),
        })
}
fn md_f(g: &GgufModel, k: &str) -> Result<f32> {
    g.get_metadata(k)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .ok_or_else(|| CoreError::InvalidArgument {
            reason: format!("missing GGUF key {k}"),
        })
}
fn md_arr_f(g: &GgufModel, k: &str, n: usize, def: f32) -> Vec<f32> {
    match g.get_metadata(k) {
        Some(GgufValue::Array(a)) => a
            .iter()
            .map(|v| v.as_f64().unwrap_or(def as f64) as f32)
            .collect(),
        Some(v) => vec![v.as_f64().unwrap_or(def as f64) as f32; n],
        None => vec![def; n],
    }
}

impl GlmHparams {
    pub fn from_gguf(g: &GgufModel) -> Result<Self> {
        let a = ARCH_NAME;
        let arch = g
            .get_metadata("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if arch != a {
            return Err(CoreError::InvalidArgument {
                reason: format!("not a GLM-5.3-Flash GGUF (architecture {arch:?})"),
            });
        }
        let k = |s: &str| format!("{a}.{s}");
        let n_layer_all = md_u(g, &k("block_count"))?;
        let nextn = md_u(g, &k("nextn_predict_layers")).unwrap_or(0);
        let n_layer = n_layer_all - nextn;
        let hkv: Vec<u64> = match g.get_metadata(&k("attention.head_count_kv")) {
            Some(GgufValue::Array(v)) => v.iter().map(|x| x.as_u64().unwrap_or(0)).collect(),
            _ => vec![0; n_layer_all],
        };
        let is_recr = (0..n_layer_all)
            .map(|i| hkv.get(i).copied().unwrap_or(0) == 0)
            .collect();
        let mut eos = vec![];
        for key in [
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.eom_token_id",
        ] {
            if let Some(v) = g.get_metadata(key).and_then(|v| v.as_u64()) {
                if !eos.contains(&(v as u32)) {
                    eos.push(v as u32);
                }
            }
        }
        let hp = Self {
            n_layer_all,
            n_layer,
            n_embd: md_u(g, &k("embedding_length"))?,
            n_vocab: md_u(g, &k("vocab_size"))?,
            n_head: md_u(g, &k("attention.head_count"))?,
            head_kda: md_u(g, &k("kda.head_dim"))?,
            d_conv: md_u(g, &k("ssm.conv_kernel"))?,
            kda_lb: md_f(g, &k("kda.gate_lower_bound")).unwrap_or(f32::NEG_INFINITY),
            q_lora: md_u(g, &k("attention.q_lora_rank"))?,
            kv_lora: md_u(g, &k("attention.kv_lora_rank"))?,
            head_k_mla: md_u(g, &k("attention.key_length_mla"))?,
            head_v_mla: md_u(g, &k("attention.value_length_mla"))?,
            idx_heads: md_u(g, &k("attention.indexer.head_count"))?,
            idx_dim: md_u(g, &k("attention.indexer.key_length"))?,
            idx_topk: md_u(g, &k("attention.indexer.top_k"))?,
            kpool: md_u(g, &k("attention.indexer.kpool"))?,
            n_expert: md_u(g, &k("expert_count"))?,
            n_expert_used: md_u(g, &k("expert_used_count"))?,
            n_ff: md_u(g, &k("feed_forward_length"))?,
            n_ff_exp: md_u(g, &k("expert_feed_forward_length"))?,
            n_dense_lead: md_u(g, &k("leading_dense_block_count")).unwrap_or(0),
            w_scale: md_f(g, &k("expert_weights_scale")).unwrap_or(1.0),
            w_norm: g
                .get_metadata(&k("expert_weights_norm"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            clamp_exp: md_arr_f(g, &k("swiglu_clamp_exp"), n_layer_all, 0.0),
            clamp_shexp: md_arr_f(g, &k("swiglu_clamp_shexp"), n_layer_all, 0.0),
            hc_mult: md_u(g, &k("hyper_connection.count"))?,
            hc_iters: md_u(g, &k("hyper_connection.sinkhorn_iterations"))?,
            hc_eps: md_f(g, &k("hyper_connection.epsilon"))?,
            rms_eps: md_f(g, &k("attention.layer_norm_rms_epsilon"))?,
            norm_eps: md_f(g, &k("attention.layer_norm_epsilon")).unwrap_or(1e-6),
            is_recr,
            ctx_train: md_u(g, &k("context_length")).unwrap_or(131072),
            eos,
            bos: g
                .get_metadata("tokenizer.ggml.bos_token_id")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
        };
        if hp.hc_mult != 4
            || hp.head_kda != 128
            || hp.kv_lora != 512
            || hp.idx_dim != 128
            || hp.kpool != 4
        {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "unsupported GLM geometry (hc {}, kda head {}, kv_lora {}, idx dim {}, kpool {})",
                    hp.hc_mult, hp.head_kda, hp.kv_lora, hp.idx_dim, hp.kpool
                ),
            });
        }
        Ok(hp)
    }
}

/// One device tensor (weights in native GGUF block layout).
/// Rows of a GGUF tensor read on demand with pread (one row = one token's embedding block row).
pub struct HostRows {
    pub file: File,
    pub off: u64,
    pub row_bytes: usize,
    pub n_rows: usize,
}

impl HostRows {
    /// Copy row `r` into `dst` (row_bytes long).
    pub fn read_row(&self, r: usize, dst: &mut [u8]) -> Result<()> {
        if r >= self.n_rows {
            return Err(CoreError::InvalidArgument {
                reason: format!("token id {r} >= vocab {}", self.n_rows),
            });
        }
        self.file
            .read_exact_at(dst, self.off + (r * self.row_bytes) as u64)
            .map_err(|e| CoreError::IoError {
                message: format!("pread token_embd row {r}: {e}"),
            })
    }
}

/// Whether the token embedding stays in the GGUF (host rows) rather than on the device.
pub fn embed_on_host() -> bool {
    std::env::var("GLM_EMBED_DEVICE")
        .map(|v| v != "1")
        .unwrap_or(true)
}

/// Device bytes of the token embedding (left out of the device budget when it stays on the host).
pub fn tok_embd_bytes(g: &GgufModel) -> u64 {
    g.find_tensor("token_embd.weight")
        .map(|(e, _)| e.byte_size)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy)]
pub struct Tensor {
    pub ptr: *mut c_void,
    pub ty: u32,
    pub ne: [u64; 4],
    pub bytes: usize,
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            ptr: ptr::null_mut(),
            ty: 0,
            ne: [0; 4],
            bytes: 0,
        }
    }
}

impl Tensor {
    #[inline]
    pub fn f(&self) -> *const f32 {
        self.ptr as *const f32
    }
    #[inline]
    pub fn p(&self) -> *const c_void {
        self.ptr as *const c_void
    }
    #[inline]
    pub fn n_in(&self) -> i32 {
        self.ne[0] as i32
    }
    #[inline]
    pub fn n_out(&self) -> i32 {
        self.ne[1] as i32
    }
    #[inline]
    pub fn expert_bytes(&self) -> usize {
        if self.ne[2] > 1 {
            self.bytes / self.ne[2] as usize
        } else {
            self.bytes
        }
    }
    pub fn is_some(&self) -> bool {
        !self.ptr.is_null()
    }
}

#[derive(Debug, Default, Clone)]
pub struct HcW {
    pub fn_: Tensor,
    pub base: Tensor,
    pub scale: Tensor,
}

#[derive(Debug, Default, Clone)]
pub struct KdaW {
    pub conv_q: Tensor,
    pub conv_k: Tensor,
    pub conv_v: Tensor,
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub f_a: Tensor,
    pub f_b: Tensor,
    pub beta: Tensor,
    pub a: Tensor,
    pub dt_b: Tensor,
    pub g_a: Tensor,
    pub g_b: Tensor,
    pub o_norm: Tensor,
    pub wo: Tensor,
}

#[derive(Debug, Default, Clone)]
pub struct MlaW {
    pub q_a_norm: Tensor,
    pub kv_a_norm: Tensor,
    pub wq_a: Tensor,
    pub wq_b: Tensor,
    pub wkv_a: Tensor,
    pub wk_b: Tensor,
    pub wv_b: Tensor,
    pub wo: Tensor,
    pub idx_k_norm: Tensor,
    pub idx_k_norm_b: Tensor,
    pub idx_proj: Tensor, // F32, pre-scaled by 1/sqrt(idx_dim * idx_heads) at load
    pub idx_k: Tensor,
    pub idx_q_b: Tensor,
    pub idx_gate: Tensor,
    pub idx_ape: Tensor,
}

#[derive(Debug, Default, Clone)]
pub struct FfnW {
    // dense (layers < n_dense_lead) or shared expert
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
    // MoE
    pub gate_inp: Tensor,
    pub probs_b: Tensor,
    pub gate_exps: Tensor,
    pub up_exps: Tensor,
    pub down_exps: Tensor,
}

#[derive(Debug, Default, Clone)]
pub struct LayerW {
    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    pub hc_attn: HcW,
    pub hc_ffn: HcW,
    pub recr: bool,
    pub moe: bool,
    pub kda: KdaW,
    pub mla: MlaW,
    pub ffn: FfnW,
    // NextN
    pub eh_proj: Tensor,
    pub enorm: Tensor,
    pub hnorm: Tensor,
    pub head_norm: Tensor,
}

pub struct GlmWeights {
    pub hp: GlmHparams,
    pub tok_embd: Tensor,
    /// Token embedding rows read from the GGUF per window (page cache, reclaimable) instead of a device copy
    /// (`tok_embd.ptr` is null then); GLM_EMBED_DEVICE=1 keeps the device table.
    pub tok_embd_rows: Option<HostRows>,
    pub output_norm: Tensor,
    pub output: Tensor,
    pub layers: Vec<LayerW>,
    pub allocs: Vec<*mut c_void>,
    pub bytes: usize,
}

unsafe impl Send for GlmWeights {}

impl Drop for GlmWeights {
    fn drop(&mut self) {
        for p in self.allocs.drain(..) {
            ffi::dev_free(p);
        }
    }
}

unsafe extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}
const POSIX_FADV_DONTNEED: i32 = 4;
const STAGE: usize = 64 << 20;

/// Streams tensors from the shard files into device memory.
struct Uploader<'a> {
    g: &'a GgufModel,
    files: Vec<File>,
    stage: [*mut c_void; 2],
    events: [hip_sys::hipEvent_t; 2],
    stream: hip_sys::hipStream_t,
    which: usize,
    allocs: Vec<*mut c_void>,
    bytes: usize,
    names: Vec<String>,
    total: u64,
    floor: u64,
    t_read: f64,
    t_alloc: f64,
    t_sync: f64,
}

impl<'a> Uploader<'a> {
    fn new(g: &'a GgufModel) -> Result<Self> {
        let mut files = vec![];
        for s in g.shards() {
            let p = s.path().ok_or_else(|| CoreError::InvalidArgument {
                reason: "GGUF shard without a path".into(),
            })?;
            files.push(File::open(p).map_err(|e| CoreError::IoError {
                message: format!("open {}: {e}", p.display()),
            })?);
        }
        let stage = [
            ffi::host_alloc_pinned(STAGE)?,
            ffi::host_alloc_pinned(STAGE)?,
        ];
        let mut events = [ptr::null_mut(); 2];
        for e in events.iter_mut() {
            // SAFETY: out-pointer valid.
            ffi::ckh(unsafe { hip_sys::hipEventCreate(e) }, "hipEventCreate")?;
        }
        Ok(Self {
            g,
            files,
            stage,
            events,
            stream: ffi::stream_create()?,
            which: 0,
            allocs: vec![],
            bytes: 0,
            names: vec![],
            total: total_weight_bytes(g),
            floor: 8 << 30,
            t_read: 0.0,
            t_alloc: 0.0,
            t_sync: 0.0,
        })
    }

    fn entry(&self, name: &str) -> Option<(TensorEntry, usize, u64)> {
        let (e, loc) = self.g.find_tensor(name)?;
        let shard = &self.g.shards()[loc.shard_idx];
        Some((e.clone(), loc.shard_idx, shard.data_start() + e.offset))
    }

    /// Tensor `name` left in the file: a pread handle for its rows (dims [n_cols, n_rows]).
    fn host_rows(&mut self, name: &str) -> Result<(Tensor, HostRows)> {
        let (e, si, off) = self
            .entry(name)
            .ok_or_else(|| CoreError::MissingTensor { name: name.into() })?;
        let n_rows = e.dims[1] as usize;
        let file = self.files[si]
            .try_clone()
            .map_err(|err| CoreError::IoError {
                message: format!("dup {name}: {err}"),
            })?;
        self.total = self.total.saturating_sub(e.byte_size);
        Ok((
            Tensor {
                ptr: ptr::null_mut(),
                ty: ggml_type_id(&e),
                ne: e.dims,
                bytes: 0,
            },
            HostRows {
                file,
                off,
                row_bytes: e.byte_size as usize / n_rows,
                n_rows,
            },
        ))
    }

    fn load(&mut self, name: &str) -> Result<Tensor> {
        self.load_opt(name)?
            .ok_or_else(|| CoreError::MissingTensor { name: name.into() })
    }

    fn load_opt(&mut self, name: &str) -> Result<Option<Tensor>> {
        let Some((e, si, off)) = self.entry(name) else {
            return Ok(None);
        };
        let bytes = e.byte_size as usize;
        // live check: what is still to load must fit above the floor (another process may have taken memory)
        let remaining = self.total.saturating_sub(self.bytes as u64);
        let avail = mem_available_bytes();
        if avail < remaining + self.floor {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "load aborted at {name}: MemAvailable {:.1} GiB < {:.1} GiB still to load + 8 GiB floor",
                    avail as f64 / (1u64 << 30) as f64,
                    remaining as f64 / (1u64 << 30) as f64
                ),
            });
        }
        let ta = std::time::Instant::now();
        let dev = dev_alloc(bytes)?;
        self.t_alloc += ta.elapsed().as_secs_f64();
        self.allocs.push(dev);
        let fd = self.files[si].as_raw_fd();
        let mut done = 0usize;
        while done < bytes {
            let n = (bytes - done).min(STAGE);
            let st = self.stage[self.which];
            let ev = self.events[self.which];
            // the stage is free once its previous copy completed
            let ts = std::time::Instant::now();
            // SAFETY: event valid.
            ffi::ckh(
                unsafe { hip_sys::hipEventSynchronize(ev) },
                "hipEventSynchronize",
            )?;
            self.t_sync += ts.elapsed().as_secs_f64();
            // SAFETY: stage has STAGE bytes.
            let buf = unsafe { std::slice::from_raw_parts_mut(st as *mut u8, n) };
            let tr = std::time::Instant::now();
            self.files[si]
                .read_exact_at(buf, off + done as u64)
                .map_err(|err| CoreError::IoError {
                    message: format!("pread {name}: {err}"),
                })?;
            self.t_read += tr.elapsed().as_secs_f64();
            // SAFETY: dev has `bytes` bytes.
            let dst = unsafe { (dev as *mut u8).add(done) } as *mut c_void;
            ffi::h2d_async(dst, st, n, self.stream)?;
            // SAFETY: event/stream valid.
            ffi::ckh(
                unsafe { hip_sys::hipEventRecord(ev, self.stream) },
                "hipEventRecord",
            )?;
            // SAFETY: fd valid; advisory.
            unsafe {
                posix_fadvise(
                    fd,
                    (off + done as u64) as i64,
                    n as i64,
                    POSIX_FADV_DONTNEED,
                );
            }
            self.which ^= 1;
            done += n;
        }
        self.bytes += bytes;
        self.names.push(name.to_string());
        Ok(Some(Tensor {
            ptr: dev,
            ty: ggml_type_id(&e),
            ne: e.dims,
            bytes,
        }))
    }

    fn finish(mut self) -> Result<(Vec<*mut c_void>, usize)> {
        ffi::sync(self.stream)?;
        for f in &self.files {
            // SAFETY: advisory.
            unsafe {
                posix_fadvise(f.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED);
            }
        }
        for s in self.stage {
            ffi::host_free(s);
        }
        for e in self.events {
            // SAFETY: event valid.
            unsafe {
                let _ = hip_sys::hipEventDestroy(e);
            }
        }
        // SAFETY: stream valid.
        unsafe {
            let _ = hip_sys::hipStreamDestroy(self.stream);
        }
        Ok((std::mem::take(&mut self.allocs), self.bytes))
    }
}

/// ggml type id of a tensor entry (talus-core-api QuantType carries the ggml id).
pub fn ggml_type_id(e: &TensorEntry) -> u32 {
    e.qtype as u32
}

/// Sum of the tensor payloads (for the Budget, before any allocation).
pub fn total_weight_bytes(g: &GgufModel) -> u64 {
    g.tensors().map(|(_, e)| e.byte_size).sum()
}

impl GlmWeights {
    /// Load every tensor once into device memory. `budget` must already hold a Weights reservation (the caller
    /// reserves the whole engine before calling, so a refusal happens before any allocation).
    pub fn load(g: &GgufModel, hp: &GlmHparams, load_mtp: bool, verbose: bool) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let mut up = Uploader::new(g)?;
        let (tok_embd, tok_embd_rows) = if embed_on_host() {
            let (t, r) = up.host_rows("token_embd.weight")?;
            (t, Some(r))
        } else {
            (up.load("token_embd.weight")?, None)
        };
        let output_norm = up.load("output_norm.weight")?;
        let output = up.load("output.weight")?;
        let n_load = if load_mtp { hp.n_layer_all } else { hp.n_layer };
        let mut layers = Vec::with_capacity(n_load);
        for il in 0..n_load {
            let b = |s: &str| format!("blk.{il}.{s}");
            let mut l = LayerW {
                attn_norm: up.load(&b("attn_norm.weight"))?,
                ffn_norm: up.load(&b("ffn_norm.weight"))?,
                recr: hp.is_recr[il],
                moe: il >= hp.n_dense_lead,
                ..Default::default()
            };
            if il < hp.n_layer {
                l.hc_attn = HcW {
                    fn_: up.load(&b("hc_attn_fn.weight"))?,
                    base: up.load(&b("hc_attn_base.weight"))?,
                    scale: up.load(&b("hc_attn_scale.weight"))?,
                };
                l.hc_ffn = HcW {
                    fn_: up.load(&b("hc_ffn_fn.weight"))?,
                    base: up.load(&b("hc_ffn_base.weight"))?,
                    scale: up.load(&b("hc_ffn_scale.weight"))?,
                };
            }
            if l.recr {
                l.kda = KdaW {
                    conv_q: up.load(&b("ssm_conv1d_q.weight"))?,
                    conv_k: up.load(&b("ssm_conv1d_k.weight"))?,
                    conv_v: up.load(&b("ssm_conv1d_v.weight"))?,
                    wq: up.load(&b("attn_q.weight"))?,
                    wk: up.load(&b("attn_k.weight"))?,
                    wv: up.load(&b("attn_v.weight"))?,
                    f_a: up.load(&b("ssm_f_a.weight"))?,
                    f_b: up.load(&b("ssm_f_b.weight"))?,
                    beta: up.load(&b("ssm_beta.weight"))?,
                    a: up.load(&b("ssm_a"))?,
                    dt_b: up.load(&b("ssm_dt.bias"))?,
                    g_a: up.load(&b("ssm_g_a.weight"))?,
                    g_b: up.load(&b("ssm_g_b.weight"))?,
                    o_norm: up.load(&b("ssm_norm.weight"))?,
                    wo: up.load(&b("attn_output.weight"))?,
                };
            } else {
                l.mla = MlaW {
                    q_a_norm: up.load(&b("attn_q_a_norm.weight"))?,
                    kv_a_norm: up.load(&b("attn_kv_a_norm.weight"))?,
                    wq_a: up.load(&b("attn_q_a.weight"))?,
                    wq_b: up.load(&b("attn_q_b.weight"))?,
                    wkv_a: up.load(&b("attn_kv_a_mqa.weight"))?,
                    wk_b: up.load(&b("attn_k_b.weight"))?,
                    wv_b: up.load(&b("attn_v_b.weight"))?,
                    wo: up.load(&b("attn_output.weight"))?,
                    idx_k_norm: up.load(&b("indexer.k_norm.weight"))?,
                    idx_k_norm_b: up.load(&b("indexer.k_norm.bias"))?,
                    idx_proj: up.load(&b("indexer.proj.weight"))?,
                    idx_k: up.load(&b("indexer.attn_k.weight"))?,
                    idx_q_b: up.load(&b("indexer.attn_q_b.weight"))?,
                    idx_gate: up.load(&b("indexer_compressor_gate.weight"))?,
                    idx_ape: up.load(&b("indexer_compressor_ape.weight"))?,
                };
            }
            if l.moe {
                l.ffn = FfnW {
                    gate: up.load(&b("ffn_gate_shexp.weight"))?,
                    up: up.load(&b("ffn_up_shexp.weight"))?,
                    down: up.load(&b("ffn_down_shexp.weight"))?,
                    gate_inp: up.load(&b("ffn_gate_inp.weight"))?,
                    probs_b: up.load(&b("exp_probs_b.bias"))?,
                    gate_exps: up.load(&b("ffn_gate_exps.weight"))?,
                    up_exps: up.load(&b("ffn_up_exps.weight"))?,
                    down_exps: up.load(&b("ffn_down_exps.weight"))?,
                };
            } else {
                l.ffn = FfnW {
                    gate: up.load(&b("ffn_gate.weight"))?,
                    up: up.load(&b("ffn_up.weight"))?,
                    down: up.load(&b("ffn_down.weight"))?,
                    ..Default::default()
                };
            }
            if il >= hp.n_layer {
                l.eh_proj = up.load(&b("nextn.eh_proj.weight"))?;
                l.enorm = up.load(&b("nextn.enorm.weight"))?;
                l.hnorm = up.load(&b("nextn.hnorm.weight"))?;
                l.head_norm = up
                    .load_opt(&b("nextn.shared_head_norm.weight"))?
                    .unwrap_or(output_norm);
            }
            if verbose && (il % 4 == 3 || il + 1 == n_load) {
                eprintln!(
                    "glm: loaded layer {il} ({:.1} GiB, {:.1} s; read {:.1} s, alloc {:.1} s, sync {:.1} s; MemAvailable {:.1} GiB)",
                    up.bytes as f64 / (1u64 << 30) as f64,
                    t0.elapsed().as_secs_f64(),
                    up.t_read,
                    up.t_alloc,
                    up.t_sync,
                    mem_available_bytes() as f64 / (1u64 << 30) as f64
                );
            }
            layers.push(l);
        }
        let (allocs, bytes) = up.finish()?;
        let w = Self {
            hp: hp.clone(),
            tok_embd,
            tok_embd_rows,
            output_norm,
            output,
            layers,
            allocs,
            bytes,
        };
        // fold the indexer weight scale 1/sqrt(idx_dim * idx_heads) (= 1/64, exact) into idx_proj
        let sc = 1.0 / ((hp.idx_dim * hp.idx_heads) as f32).sqrt();
        for l in &w.layers {
            if !l.recr {
                scale_f32_tensor(&l.mla.idx_proj, sc)?;
            }
        }
        if verbose {
            eprintln!(
                "glm: weights {:.2} GiB in {:.1} s",
                w.bytes as f64 / (1u64 << 30) as f64,
                t0.elapsed().as_secs_f64()
            );
        }
        Ok(w)
    }
}

/// Live /proc/meminfo MemAvailable (0 if unreadable).
pub fn mem_available_bytes() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for l in s.lines() {
        if let Some(r) = l.strip_prefix("MemAvailable:") {
            let kb: u64 = r.trim().trim_end_matches("kB").trim().parse().unwrap_or(0);
            return kb * 1024;
        }
    }
    0
}

/// Background MemAvailable watchdog: exits the process (freeing its GPU memory) when MemAvailable stays below
/// `floor` for two consecutive samples (200 ms apart).
pub fn spawn_watchdog(floor: u64) {
    unsafe extern "C" {
        fn _exit(code: i32) -> !;
    }
    std::thread::spawn(move || {
        let mut low = 0;
        loop {
            std::thread::sleep(std::time::Duration::from_millis(200));
            let a = mem_available_bytes();
            if a != 0 && a < floor {
                low += 1;
                if low >= 2 {
                    eprintln!(
                        "glm: WATCHDOG MemAvailable {:.2} GiB < floor {:.2} GiB: exiting",
                        a as f64 / (1u64 << 30) as f64,
                        floor as f64 / (1u64 << 30) as f64
                    );
                    // SAFETY: immediate process exit; the driver releases GPU memory.
                    unsafe { _exit(3) };
                }
            } else {
                low = 0;
            }
        }
    });
}

fn scale_f32_tensor(t: &Tensor, s: f32) -> Result<()> {
    let n = t.bytes / 4;
    let mut h = vec![0f32; n];
    ffi::d2h(h.as_mut_ptr() as *mut c_void, t.p(), t.bytes)?;
    for v in h.iter_mut() {
        *v *= s;
    }
    ffi::h2d(t.ptr, h.as_ptr() as *const c_void, t.bytes)
}

/// Reserve the whole engine in the Budget before loading: refuses configurations projecting MemAvailable below the
/// floor (8 GiB). Returns the reservation ids to mark materialized.
pub fn reserve(
    budget: &Budget,
    items: &[(BudgetCategory, &str, u64)],
) -> Result<Vec<talus_mem::ReservationId>> {
    let mut ids = vec![];
    for (cat, label, bytes) in items {
        let r = budget
            .try_reserve(*cat, *label, *bytes as usize)
            .map_err(|e| CoreError::InvalidArgument {
                reason: format!(
                    "memory budget refused {label} ({:.2} GiB): {e}",
                    *bytes as f64 / (1u64 << 30) as f64
                ),
            })?;
        ids.push(r.id);
    }
    Ok(ids)
}
