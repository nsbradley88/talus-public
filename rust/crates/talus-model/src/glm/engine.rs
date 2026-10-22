//! GLM-5.3-Flash forward pass over a window of T tokens (T <= TMAX), state (MLA latent / indexer caches, KDA
//! recurrent + conv state with rollback slots), and the MTP block.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

use talus_core_api::{CoreError, Result};

use super::ffi::{self, ck, dev_alloc, GlmLib};
use super::weights::{gt, GlmWeights, LayerW, Tensor};

// prefill seam (dense GEMM + grouped routed experts for windows above TMAX)
#[path = "prefill.rs"]
mod prefill;

/// Max tokens per window: glm_mmvq_ids takes <= 64 (token, expert) pairs at top-8.
pub const TMAX: usize = 8;
/// Prefill chunk: windows larger than TMAX run the MMQ (int8 GEMM) paths; chunks are aligned to multiples of PCH.
pub const PCH: usize = 2048;
/// the largest run-time prefill chunk (GLM_PCH up to 4096: the routed experts then see ~114 rows per
/// expert instead of ~57, where the fp16 expert kernels (STRATA_PF_W16_MOE=1) pay off); the static buffers (control
/// block, embedding staging, row ids) are sized for it.
pub const PCH_MAX: usize = 4096;

/// Prefill chunk used at run time (GLM_PCH, default PCH_MAX = 4096 GLM_PCH=2048 = the previous
/// default; at most PCH_MAX; a multiple of TMAX).
pub fn pch() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let v = std::env::var("GLM_PCH")
            .ok()
            .and_then(|x| x.parse::<usize>().ok())
            .unwrap_or(PCH_MAX);
        (v.clamp(2 * TMAX, PCH_MAX) / TMAX) * TMAX
    })
}
/// KDA rollback slots per layer for the MTP rounds (current + 2 for a 3-token verify).
pub const KDA_SLOTS: usize = 3;

/// KDA rollback slots per layer at run time: GLM_KDA_SLOTS (3..=TMAX), else TMAX while the long-span copy
/// proposer is on (a copy window of up to TMAX tokens snapshots every row; +5 slots of KDA + conv state,
/// ~4 MiB per KDA layer each), else KDA_SLOTS.
pub fn kda_slots() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GLM_KDA_SLOTS")
            .ok()
            .and_then(|x| x.parse::<usize>().ok())
            .unwrap_or(if super::spec::copy_on() {
                TMAX
            } else {
                KDA_SLOTS
            })
            .clamp(KDA_SLOTS, TMAX)
    })
}
/// Context bucket baked into the indexer / selection launches (pools = bucket / kpool).
pub const KV_BUCKET: usize = 1024;
/// key|gate ring rows (TKG_KG_RING in talus_glm_kernels.h)
pub const KG_RING: usize = 4096;
/// Control-block staging slots.
const RING: usize = 64;
/// Control block (int32 offsets): pos0 at CTL_POS, KDA slot ids at CTL_KDA, MTP tokens after them, window tokens
/// from CTL_TOK (up to PCH). Only the header and the window's tokens are copied per window.
const CTL_POS: usize = 0;
const CTL_KDA: usize = 8;
/// sub-window positions (pos0 + k * ASUB) for the attention blocks of a big window
const CTL_SUB: usize = 24;
const CTL_TOK: usize = 32;
/// Attention sub-window: the KDA / MLA blocks of a prefill window run in slices of ASUB rows (their scratch is
/// sized for ASUB), the dense projections, routed experts and mHC run over the whole window.
pub const ASUB: usize = 512;
/// Attention sub-window at run time (GLM_ASUB, default ASUB; a multiple of 64 between pch() / 8 and pch(): at most 8
/// sub-windows a window, the CTL_SUB slots).
pub fn asub() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let v = std::env::var("GLM_ASUB")
            .ok()
            .and_then(|x| x.parse::<usize>().ok())
            .unwrap_or(ASUB);
        let p = pch();
        (v.clamp(p / (CTL_TOK - CTL_SUB), p) / 64) * 64
    })
}
const CTL_BYTES: usize = (CTL_TOK + PCH_MAX) * 4;

#[derive(Debug, Clone)]
pub struct GlmConfig {
    pub ctx: usize,
    pub mtp: bool,
    pub verbose: bool,
}

/// Device buffer (bytes) with a typed pointer helper.
#[derive(Clone, Copy)]
pub struct Buf {
    pub p: *mut c_void,
    pub bytes: usize,
}
impl Buf {
    #[inline]
    pub fn f(&self) -> *mut f32 {
        self.p as *mut f32
    }
    #[inline]
    pub fn i(&self) -> *mut i32 {
        self.p as *mut i32
    }
    #[inline]
    pub fn v(&self) -> *mut c_void {
        self.p
    }
    #[inline]
    pub fn at(&self, off_bytes: usize) -> *mut c_void {
        // SAFETY: caller keeps the offset inside the buffer.
        unsafe { (self.p as *mut u8).add(off_bytes) as *mut c_void }
    }
}

#[inline]
fn q8b(n: usize) -> usize {
    n / 32 * 36
}

/// rb-num GLM_HC_BF16: unset / 0 = fp32 mHC streams (default); 1 / fp16 = fp16 (saturating, the recommended 16-bit
/// format: the streams stay below ~5e3, fp16 keeps 3 more mantissa bits than bf16); bf16 = bf16.
fn hc16_env(g: &super::ffi::Tkg) -> i32 {
    let f = match std::env::var("GLM_HC_BF16").as_deref() {
        Ok("1") | Ok("fp16") | Ok("f16") => 1,
        Ok("bf16") => 2,
        _ => 0,
    };
    if f == 0 {
        return 0;
    }
    if g.hc_post_mix16.is_none()
        || g.hc_post_mix_moe16.is_none()
        || g.hc_pre2_16.is_none()
        || g.hc_head16.is_none()
    {
        eprintln!("glm: GLM_HC_BF16 ignored (libtalus_kernels has no tkg_hc_*16 entries)");
        return 0;
    }
    eprintln!(
        "glm: GLM_HC_BF16 active: prompt-window mHC streams stored as {} (opt-in numerics)",
        if f == 1 { "fp16" } else { "bf16" }
    );
    f
}

/// Activation arena: every per-window buffer, sized for TMAX tokens.
#[derive(Clone, Copy)]
struct Acts {
    xhc: Buf,
    parts: Buf,
    pc: Buf,
    y: Buf,
    yq: Buf,
    blk_out: Buf,
    // KDA
    qr: Buf,
    kr: Buf,
    vr: Buf,
    qc: Buf,
    kc: Buf,
    vc: Buf,
    fa: Buf,
    faq: Buf,
    graw: Buf,
    beta: Buf,
    ga: Buf,
    gaq: Buf,
    g2: Buf,
    o: Buf,
    og: Buf,
    ogq: Buf,
    // MLA / DSA
    qa: Buf,
    qan: Buf,
    qaq: Buf,
    q: Buf,
    qq: Buf,
    qabs: Buf,
    kvr: Buf,
    iq: Buf,
    ik: Buf,
    ig: Buf,
    iw: Buf,
    scores: Buf,
    sel: Buf,
    lat: Buf,
    latq: Buf,
    vb: Buf,
    vbq: Buf,
    fa_split: Buf,
    head_ids: Buf,
    // FFN / MoE
    fg: Buf,
    fu: Buf,
    fact: Buf,
    factq: Buf,
    sh_out: Buf,
    /// the shared expert's buffers when it runs on the side stream (decode windows), disjoint from the routed
    /// experts' group (the FFN groups alias each other)
    sfg: Buf,
    sfu: Buf,
    sfact: Buf,
    sfactq: Buf,
    rlog: Buf,
    eids: Buf,
    ew: Buf,
    eg: Buf,
    eu: Buf,
    eact: Buf,
    eactq: Buf,
    ed: Buf,
    #[allow(dead_code)]
    ffn_out: Buf,
    // head
    h: Buf,
    hq: Buf,
    // MTP
    memb: Buf,
    mcat: Buf,
    mcatq: Buf,
    mres: Buf,
    hin: Buf,  // MTP h input rows [t][n_embd]
    hout: Buf, // MTP h output rows (shared_head_norm) [t][n_embd]
}

/// Per-layer state.
pub struct State {
    pub mla_kv: Vec<Buf>,     // [ctx][512] f16, one per DSA layer (trunk + MTP)
    pub keygate: Vec<Buf>,    // [ctx][256] f16
    pub pooled: Vec<Buf>,     // [ctx/4][128] f16
    pub kda_state: Vec<Buf>,  // [KDA_SLOTS][64][128][128] f32
    pub conv_state: Vec<Buf>, // [KDA_SLOTS][3*8192][3] f32
    pub zero_mask: Buf,       // [ctx] f16 zeros
    pub cur_slot: usize,      // KDA/conv slot holding the current state
}

/// Mapped host control block shared by every launch (unified memory: kernels read it directly).
struct Ctl {
    base: *mut c_void, // device control block (1 KiB)
    ring: *mut c_void, // pinned host staging ring: RING slots of 1 KiB, copied stream-ordered into `base`
    ev: Vec<hip_sys::hipEvent_t>,
    slot: usize,
    pos0: usize,
    tok: *mut i32,     // [TMAX]
    ctl: *mut i32,     // [8]: pos0
    kda_ids: *mut i32, // [4]
    #[allow(dead_code)]
    mtp_tok: *mut i32, // [TMAX]
    /// host embedding rows: device rows [PCH][row_bytes] after the control block, staged through the ring (decode
    /// windows: TMAX rows per slot) or `big` (larger windows; `big_ev` guards its reuse); `iota` = 0..PCH row ids
    rows_dev: *mut c_void,
    row_bytes: usize,
    slot_bytes: usize,
    big: *mut c_void,
    big_ev: hip_sys::hipEvent_t,
    iota: *mut i32,
}

pub struct GlmEngine {
    /// prompt-cache snapshot inside a sliced prefill window: (absolute position, destination); the KDA / conv state
    /// of every layer is copied when its attention slice ending there completes (`snap_save` layout)
    pub snap_mid: Option<(usize, Buf)>,
    pub lib: GlmLib,
    pub w: GlmWeights,
    pub cfg: GlmConfig,
    pub st: State,
    a: Acts,
    c: Ctl,
    pub stream: ffi::hipStream_t,
    /// main stream (self.stream is temporarily the side stream inside `side()`)
    main: ffi::hipStream_t,
    side_stream: ffi::hipStream_t,
    ev_fork: hip_sys::hipEvent_t,
    ev_join: hip_sys::hipEvent_t,
    pub use_side: bool,
    /// gate + up in one launch (routed experts, dense / shared FFN); GLM_PAIR=0 = separate launches
    pub pair_ids: bool,
    /// fuse tkg_moe_combine + tkg_hc_post_mix<0> on trunk MoE layers; GLM_FUSE_MOE_POST=0 = separate launches
    pub fuse_moe_post: bool,
    /// MTP-layer K-quant experts through tkg_kq_ids2 (gate+up one launch, x staged in LDS; bitwise);
    /// GLM_KQ_IDS2=0 = tkg_kq_ids per matrix
    pub kq_ids2: bool,
    /// greedy argmax on the device (D2H of token ids instead of logits, no host argmax); GLM_DEV_GREEDY=0 =
    /// logits to the host and host argmax after every window
    pub dev_greedy: bool,
    /// chain the greedy MTP round on the device (drafted tokens embedded on the GPU, one host sync per round).
    /// Host-row embedding mode needs the draft vocabulary's embedding rows on the device (+151 MB): GLM_DEV_CHAIN=1.
    /// Always on with GLM_EMBED_DEVICE=1.
    pub dev_chain: bool,
    /// KDA conv + step + output gate in one launch on decode windows (bitwise); GLM_KDA_FUSED=0 = 3 launches
    pub kda_fused: bool,
    kda_cnt: Buf,
    /// router logits + routing in one launch on decode windows (bitwise); opt-in GLM_FUSE_ROUTER=1 (slower in
    /// the decode graph: the last block's routing tail costs more than the launch it saves)
    pub fuse_router: bool,
    router_cnt: Buf,
    /// each mHC post-mix launch also runs the next sub-block's hc_pre2 (bitwise); opt-in GLM_FUSE_HC_PRE=1 (slower
    /// in the decode graph: the head's last block does all 8 slices)
    pub fuse_hc_pre: bool,
    /// the fused mHC post-mix + next pre2 for prompt windows (bitwise the two launches; the token's xhc rows
    /// are re-read from L2 instead of DRAM); GLM_FUSE_HC_PRE_PF=0 = two launches
    pub fuse_hc_pre_pf: bool,
    hc_cnt: Buf,
    /// MTP windows without logits (the prompt pass) only write the MTP layer's caches (latent KV, indexer
    /// keys / pools): q path, attention, wo, FFN and head norm are skipped - their outputs are never read (the first
    /// draft round re-runs the MTP block from the trunk h). Caches bitwise; GLM_MTP_KV_ONLY=0 = the full block.
    pub mtp_kv_only: bool,
    /// set while an MTP prompt window runs its attention block in KV-only form
    kv_only: bool,
    /// Prompt-chunk projections sharing an input quantize it once (bitwise); GLM_SHARE_QUANT=0 = once each
    pub share_quant: bool,
    /// rb-glm E4: windows above ASUB rows run the KDA projections (q/k/v/o, f_a/g_a/beta) over the whole window
    /// (bitwise: the GEMM rows are independent); GLM_E4_KDA=0 / GLM_E4=0 = per slice
    pub e4_kda: bool,
    /// rb-glm E4: ... and the MLA output projection (wo) over the whole window; GLM_E4_MLA=0 / GLM_E4=0 = per slice
    pub e4_mla: bool,
    /// the next hc_pre already ran inside the last post-mix launch
    pre_done: bool,
    /// rb-num (opt-in, numerics): GLM_HC_BF16=1 (or fp16) / bf16 stores the 4 mHC streams of prompt windows
    /// (t > TMAX) in 16 bits (1 = fp16 saturating, 2 = bf16); math stays fp32. 0 = off (default, fp32 streams)
    pub hc16: i32,
    /// the format of the current window's streams (0 = fp32)
    hc16_win: i32,
    /// device greedy results: [0..16) tokens, [16..32) draft-vocabulary indices (slots: see `greedy_tok`)
    gdev: Buf,
    gpin: *mut i32,
    /// draft vocabulary ids on the device (index -> token) and, in host-row embedding mode, their embedding rows
    dv_ids_dev: *mut i32,
    dv_embd: *mut c_void,
    allocs: Vec<*mut c_void>,
    pub logits_host: Vec<f32>,
    logits_dev: Buf,
    logits_pin: *mut f32,
    dsa_index: Vec<i32>, // layer -> DSA cache index or -1
    kda_index: Vec<i32>, // layer -> KDA state index or -1
    pub n_pool_cap: usize,
    pub n_kv_max: usize,
    pub dump: Option<Dump>,
    pub state_bytes: usize,
    pub act_bytes: usize,
    /// draft-vocabulary head: output.weight rows of `dv_ids` (gathered at load), its logits buffers
    dv_head: Option<Tensor>,
    pub dv_ids: Vec<u32>,
    dlog_dev: Buf,
    dlog_pin: *mut f32,
    mmq_xq: Buf,
    rowsrc: Buf,
    rownat: Buf,
    bounds: Buf,
    pub use_graphs: bool,
    graphs: HashMap<GKey, ffi::hipGraphExec_t>,
    /// last use of each graph (LRU eviction above GLM_MAX_GRAPHS: context growth leaves old KV buckets behind)
    graph_used: HashMap<GKey, u64>,
    graph_clock: u64,
    seen: std::collections::HashSet<GKey>,
}

unsafe impl Send for GlmEngine {}

/// Graph cache key: everything the launches bake in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GKey {
    pub mtp: bool,
    pub t: u16,
    pub k: u8,
    pub want: bool,
    pub bucket: u32,
}

/// Per-layer activation dumps for the tier-0 check (host copies, keyed "stage-layer").
#[derive(Default)]
pub struct Dump {
    pub tensors: HashMap<String, Vec<f32>>,
    pub order: Vec<String>,
    /// layer-isolated check: reference l_out of layer il-1 ([n_tok][4][n_embd]) injected before layer il
    pub inject: HashMap<usize, Vec<f32>>,
    /// first token of the current window (row offset into `inject`)
    pub row0: usize,
    /// only dump these layers (GLM_DUMP_LAYERS=3,7)
    pub layers: Option<Vec<usize>>,
}

impl Dump {
    fn push(&mut self, name: String, data: &[f32]) {
        if !self.tensors.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.tensors
            .entry(name)
            .or_default()
            .extend_from_slice(data);
    }
}

impl Drop for GlmEngine {
    fn drop(&mut self) {
        for (_, g) in self.graphs.drain() {
            ffi::graph_destroy(g);
        }
        for p in self.allocs.drain(..) {
            ffi::dev_free(p);
        }
        ffi::host_free(self.c.ring);
        if !self.c.big.is_null() {
            ffi::host_free(self.c.big);
            // SAFETY: event valid.
            unsafe {
                let _ = hip_sys::hipEventDestroy(self.c.big_ev);
            }
        }
        for e in self.c.ev.drain(..) {
            // SAFETY: event valid.
            unsafe {
                let _ = hip_sys::hipEventDestroy(e);
            }
        }
        ffi::host_free(self.logits_pin as *mut c_void);
        ffi::host_free(self.dlog_pin as *mut c_void);
        ffi::host_free(self.gpin as *mut c_void);
        if !self.dv_ids_dev.is_null() {
            ffi::dev_free(self.dv_ids_dev as *mut c_void);
        }
        if !self.dv_embd.is_null() {
            ffi::dev_free(self.dv_embd);
        }
    }
}

/// Bytes the engine allocates beyond the weights (for the Budget).
pub fn state_bytes(hp: &super::weights::GlmHparams, ctx: usize, mtp: bool) -> (usize, usize) {
    let n_dsa = (0..hp.n_layer).filter(|&i| !hp.is_recr[i]).count() + usize::from(mtp);
    let n_kda = (0..hp.n_layer).filter(|&i| hp.is_recr[i]).count();
    let kv = n_dsa * (ctx * (hp.kv_lora + hp.idx_dim / hp.kpool) + KG_RING * 256) * 2;
    let kda = n_kda
        * kda_slots()
        * (hp.n_head * hp.head_kda * hp.head_kda + 3 * hp.n_head * hp.head_kda * 3)
        * 4;
    (kv + ctx * 2, kda)
}

impl GlmEngine {
    pub fn new(lib: GlmLib, w: GlmWeights, cfg: GlmConfig) -> Result<Self> {
        let hp = w.hp.clone();
        let ctx = cfg.ctx.div_ceil(KV_BUCKET) * KV_BUCKET;
        let mut allocs = vec![];
        let mut alloc = |bytes: usize| -> Result<Buf> {
            let p = dev_alloc(bytes)?;
            allocs.push(p);
            Ok(Buf { p, bytes })
        };
        let t = pch();
        let e = hp.n_embd;
        let dk = hp.n_head * hp.head_kda; // 8192
        let nh = hp.n_head;
        let n_pool_cap = ctx / hp.kpool;
        let n_kv_max = hp.idx_topk + hp.kpool - 1; // 2051
        let lat = hp.kv_lora;
        let ff = hp.n_ff.max(hp.n_ff_exp);
        let k = hp.n_expert_used;
        let mut act_bytes = 0usize;
        let mut al = |bytes: usize| -> Result<Buf> {
            act_bytes += bytes;
            alloc(bytes)
        };
        // phase-aliased groups: KDA-attention (0), MLA/DSA-attention (1), dense FFN / shared expert (2) and routed
        // experts (3) are never live at the same time (blocks run one after another; the shared expert finishes before
        // routing), so they share one region sized for the largest. GLM_SIDE_STREAM=1 overlaps the shared expert with
        // routing in decode windows: then groups 2 and 3 get disjoint space.
        let ta = t.min(asub()); // attention groups hold one ASUB-row slice
                                // the dense FFN / shared expert (2) and the MTP block (5) run in slices of at most PCH rows (rows
                                // are independent: bitwise the whole-window result), so a 4096-row chunk only grows what the routed experts and
                                // the trunk streams need
        let tp = t.min(PCH);
        // rb-glm E4: the KDA q/k/v/o and f_a/g_a/beta projections and the MLA output projection of a window above
        // ASUB rows run over the whole window (the recurrence / sparse attention stay in ASUB slices): those
        // buffers then hold the whole window. They are laid out last in their phase-aliased groups (0 / 1) and each
        // part only turns on when its group still fits the region the engine would have without it (at the default
        // 4096-row chunk and 64K context both do: groups 0 and 1 stay inside the routed experts' group 3), so E4
        // never adds memory. GLM_E4=0 (or GLM_E4_KDA=0 / GLM_E4_MLA=0) = per-slice projections and the previous
        // buffer sizes; GLM_E4=2 = on even where the region would grow.
        let env_on = |k: &str| std::env::var(k).map(|v| v != "0").unwrap_or(true);
        let e4_force = std::env::var("GLM_E4").map(|v| v == "2").unwrap_or(false);
        let mut e4_kda = env_on("GLM_E4") && env_on("GLM_E4_KDA") && t > ta;
        let mut e4_mla = env_on("GLM_E4") && env_on("GLM_E4_MLA") && t > ta;
        let nul = || Buf {
            p: ptr::null_mut(),
            bytes: 0,
        };
        let mut goff = [0usize; 6];
        let mut gl = |g: usize, bytes: usize| -> Result<Buf> {
            let o = goff[g];
            goff[g] += bytes.div_ceil(256) * 256;
            Ok(Buf {
                p: o as *mut c_void,
                bytes,
            })
        };
        let fa_split = {
            // the split-K buffer is largest for small windows (more splits per token)
            let mut b = 256usize;
            // every window size can occur (prefill tails between TMAX and the chunk): take the max over all of them
            for n in 1..=pch().min(asub()) {
                // SAFETY: plain C query.
                b = b.max(unsafe {
                    (lib.g.glm_sparse_fa_split_bytes)(
                        lat as i32,
                        nh as i32,
                        n as i32,
                        n_kv_max as i32,
                    )
                });
            }
            b
        };
        let region_bytes: usize;
        let mut a = Acts {
            xhc: gl(4, t * 4 * e * 4)?,
            parts: gl(4, t * 16 * 64 * 4)?,
            pc: gl(4, t * 32 * 4)?,
            y: al(t * e * 4)?,
            yq: al(t * q8b(e))?,
            blk_out: al(t * e * 4)?,
            qr: nul(), // whole-window capable: laid out last in group 0 (below)
            kr: nul(),
            vr: nul(),
            qc: gl(0, ta * dk * 4)?,
            kc: gl(0, ta * dk * 4)?,
            vc: gl(0, ta * dk * 4)?,
            fa: nul(),
            faq: gl(0, ta * q8b(hp.head_kda))?,
            graw: gl(0, ta * dk * 4)?,
            beta: nul(),
            ga: nul(),
            gaq: gl(0, ta * q8b(hp.head_kda))?,
            g2: gl(0, ta * dk * 4)?,
            o: gl(0, ta * dk * 4)?,
            og: gl(0, ta * dk * 4)?,
            ogq: gl(0, ta * q8b(dk))?,
            qa: gl(1, ta * hp.q_lora * 4)?,
            qan: gl(1, ta * hp.q_lora * 4)?,
            qaq: gl(1, ta * q8b(hp.q_lora))?,
            q: gl(1, ta * nh * hp.head_k_mla * 4)?,
            qq: gl(1, ta * q8b(nh * hp.head_k_mla))?,
            qabs: gl(1, ta * nh * lat * 4)?,
            kvr: gl(1, ta * lat * 4)?,
            iq: gl(1, ta * hp.idx_heads * hp.idx_dim * 4)?,
            ik: gl(1, ta * hp.idx_dim * 4)?,
            ig: gl(1, ta * hp.idx_dim * 4)?,
            iw: gl(1, ta * hp.idx_heads * 4)?,
            scores: gl(1, ta * n_pool_cap * 4)?,
            sel: gl(1, ta * n_kv_max * 4)?,
            lat: gl(1, ta * nh * lat * 4)?,
            latq: gl(1, ta * q8b(nh * lat))?,
            vb: nul(), // whole-window capable: laid out last in group 1 (below)
            vbq: gl(1, ta * q8b(nh * hp.head_v_mla))?,
            fa_split: al(fa_split)?,
            head_ids: al(t * nh * 4)?,
            fg: gl(2, tp * ff * 4)?,
            fu: gl(2, tp * ff * 4)?,
            fact: gl(2, tp * ff * 4)?,
            factq: gl(2, tp * q8b(ff))?,
            sh_out: al(t * e * 4)?,
            sfg: al(TMAX * ff * 4)?,
            sfu: al(TMAX * ff * 4)?,
            sfact: al(TMAX * ff * 4)?,
            sfactq: al(TMAX * q8b(ff))?,
            rlog: gl(3, t * hp.n_expert * 4)?,
            eids: gl(3, t * k * 4)?,
            ew: gl(3, t * k * 4)?,
            eg: gl(3, t * k * hp.n_ff_exp * 4)?,
            eu: gl(3, t * k * hp.n_ff_exp * 4)?,
            eact: al(TMAX * k * hp.n_ff_exp * 4)?, // decode K-quant path only (prefill: in place on eg)
            eactq: al(TMAX * k * q8b(hp.n_ff_exp))?, // decode paths only (prefill quantizes into mmq_xq)
            ed: Buf {
                p: ptr::null_mut(),
                bytes: t * k * e * 4,
            }, // overlays eg + eu, set below
            ffn_out: Buf {
                p: ptr::null_mut(),
                bytes: 0,
            }, // unused
            h: al(t * e * 4)?,
            hq: al(t * q8b(e))?,
            memb: gl(5, tp * e * 4)?,
            mcat: gl(5, tp * 2 * e * 4)?,
            mcatq: gl(5, tp * q8b(2 * e))?,
            mres: gl(5, tp * e * 4)?,
            hin: gl(5, tp * e * 4)?,
            hout: gl(5, tp * e * 4)?,
        };
        {
            // rb-glm E4 guard and the whole-window capable buffers (zero-byte gl() calls read a group's end)
            let a256 = |b: usize| b.div_ceil(256) * 256;
            let hkd = hp.head_kda;
            let kda_rows = |r: usize| {
                [
                    r * dk * 4,
                    r * dk * 4,
                    r * dk * 4,
                    r * hkd * 4,
                    r * nh * 4,
                    r * hkd * 4,
                ]
            };
            let vbw = nh * hp.head_v_mla;
            let g0 = gl(0, 0)?.p as usize;
            let g1 = gl(1, 0)?.p as usize;
            let g0_end = |r: usize| g0 + kda_rows(r).iter().map(|&b| a256(b)).sum::<usize>();
            let g1_end = |r: usize| g1 + a256(r * vbw * 4);
            let base = g0_end(ta)
                .max(g1_end(ta))
                .max(gl(2, 0)?.p as usize)
                .max(gl(3, 0)?.p as usize);
            e4_kda &= e4_force || g0_end(t) <= base;
            e4_mla &= e4_force || g1_end(t) <= base;
            let tq = if e4_kda { t } else { ta };
            let [bq, bk, bv, bfa, bbeta, bga] = kda_rows(tq);
            a.qr = gl(0, bq)?;
            a.kr = gl(0, bk)?;
            a.vr = gl(0, bv)?;
            a.fa = gl(0, bfa)?;
            a.beta = gl(0, bbeta)?;
            a.ga = gl(0, bga)?;
            a.vb = gl(1, (if e4_mla { t } else { ta }) * vbw * 4)?;
        }
        {
            let side = false; // the side-stream shared expert has its own buffers (sfg..)
            let ffn = if side {
                goff[2] + goff[3]
            } else {
                goff[2].max(goff[3])
            };
            let region = al(goff[0].max(goff[1]).max(ffn).max(256))?;
            region_bytes = region.bytes;
            let shift3 = if side { goff[2] } else { 0 };
            let base = region.p as usize;
            for b in [
                &mut a.qr,
                &mut a.kr,
                &mut a.vr,
                &mut a.qc,
                &mut a.kc,
                &mut a.vc,
                &mut a.fa,
                &mut a.faq,
                &mut a.graw,
                &mut a.beta,
                &mut a.ga,
                &mut a.gaq,
                &mut a.g2,
                &mut a.o,
                &mut a.og,
                &mut a.ogq,
                &mut a.qa,
                &mut a.qan,
                &mut a.qaq,
                &mut a.q,
                &mut a.qq,
                &mut a.qabs,
                &mut a.kvr,
                &mut a.iq,
                &mut a.ik,
                &mut a.ig,
                &mut a.iw,
                &mut a.scores,
                &mut a.sel,
                &mut a.lat,
                &mut a.latq,
                &mut a.vb,
                &mut a.vbq,
                &mut a.fg,
                &mut a.fu,
                &mut a.fact,
                &mut a.factq,
            ] {
                b.p = (base + b.p as usize) as *mut c_void;
            }
            for b in [&mut a.rlog, &mut a.eids, &mut a.ew, &mut a.eg, &mut a.eu] {
                b.p = (base + shift3 + b.p as usize) as *mut c_void;
            }
            // ed overlays eg + eu (consecutive in group 3, same total size): written only after both are consumed
            debug_assert_eq!(
                a.eu.p as usize,
                a.eg.p as usize + a.eg.bytes.div_ceil(256) * 256
            );
            a.ed.p = a.eg.p;
        }
        {
            // trunk-only mHC streams (4) and the MTP block's buffers (5) are never live together: the MTP block runs
            // between trunk windows and reads the trunk's h, not its streams
            let region2 = al(goff[4].max(goff[5]).max(256))?;
            let base = region2.p as usize;
            for b in [
                &mut a.xhc,
                &mut a.parts,
                &mut a.pc,
                &mut a.memb,
                &mut a.mcat,
                &mut a.mcatq,
                &mut a.mres,
                &mut a.hin,
                &mut a.hout,
            ] {
                b.p = (base + b.p as usize) as *mut c_void;
            }
        }
        // head ids 0..63 repeated per token (per-head batched Q8_0 mat-vecs)
        let hids: Vec<i32> = (0..t * nh).map(|i| (i % nh) as i32).collect();
        ffi::h2d(
            a.head_ids.v(),
            hids.as_ptr() as *const c_void,
            hids.len() * 4,
        )?;

        // state
        let mut dsa_index = vec![-1i32; w.layers.len()];
        let mut kda_index = vec![-1i32; w.layers.len()];
        let (mut nd, mut nk) = (0, 0);
        for (il, l) in w.layers.iter().enumerate() {
            if l.recr {
                kda_index[il] = nk;
                nk += 1;
            } else {
                dsa_index[il] = nd;
                nd += 1;
            }
        }
        let mut state_bytes = 0usize;
        let mut sa = |bytes: usize| -> Result<Buf> {
            state_bytes += bytes;
            let b = alloc(bytes)?;
            ffi::memset(b.p, 0, bytes)?;
            Ok(b)
        };
        let mut st = State {
            mla_kv: vec![],
            keygate: vec![],
            pooled: vec![],
            kda_state: vec![],
            conv_state: vec![],
            zero_mask: sa(ctx * 2)?,
            cur_slot: 0,
        };
        for _ in 0..nd {
            st.mla_kv.push(sa(ctx * lat * 2)?);
            st.keygate.push(sa(KG_RING * 2 * hp.idx_dim * 2)?); // ring (TKG_KG_RING rows)
            st.pooled.push(sa(n_pool_cap * hp.idx_dim * 2)?);
        }
        for _ in 0..nk {
            st.kda_state
                .push(sa(kda_slots() * nh * hp.head_kda * hp.head_kda * 4)?);
            st.conv_state.push(sa(kda_slots() * 3 * dk * 3 * 4)?);
        }
        let row_bytes = w.tok_embd_rows.as_ref().map(|r| r.row_bytes).unwrap_or(0);
        let ctl_dev_bytes = CTL_BYTES.div_ceil(256) * 256;
        let base = alloc(ctl_dev_bytes + pch() * row_bytes)?.p;
        ffi::memset(base, 0, CTL_BYTES)?;
        let slot_bytes = ctl_dev_bytes + TMAX * row_bytes;
        let ring = ffi::host_alloc_pinned(RING * slot_bytes)?;
        let (big, big_ev, iota) = if row_bytes > 0 {
            let big = ffi::host_alloc_pinned(pch() * row_bytes)?;
            let mut ev = ptr::null_mut();
            // SAFETY: out-pointer valid; 2 = hipEventDisableTiming.
            ffi::ckh(
                unsafe { hip_sys::hipEventCreateWithFlags(&mut ev, 2) },
                "hipEventCreate",
            )?;
            let iota = alloc(pch() * 4)?.p as *mut i32;
            let ids: Vec<i32> = (0..pch() as i32).collect();
            ffi::h2d(
                iota as *mut c_void,
                ids.as_ptr() as *const c_void,
                pch() * 4,
            )?;
            (big, ev, iota)
        } else {
            (ptr::null_mut(), ptr::null_mut(), ptr::null_mut())
        };
        let mut ev = vec![];
        for _ in 0..RING {
            let mut e = ptr::null_mut();
            // SAFETY: out-pointer valid.
            ffi::ckh(
                unsafe { hip_sys::hipEventCreateWithFlags(&mut e, 2) },
                "hipEventCreate",
            )?; // disable timing
            ev.push(e);
        }
        // SAFETY: base has 1024 bytes.
        let c = unsafe {
            let b = base as *mut i32;
            Ctl {
                base,
                ring,
                ev,
                slot: 0,
                pos0: 0,
                tok: b.add(CTL_TOK),
                ctl: b.add(CTL_POS),
                kda_ids: b.add(CTL_KDA),
                mtp_tok: b.add(CTL_KDA + 8),
                rows_dev: (base as *mut u8).add(ctl_dev_bytes) as *mut c_void,
                row_bytes,
                slot_bytes,
                big,
                big_ev,
                iota,
            }
        };
        let stream = ffi::stream_create()?;
        let side_stream = ffi::stream_create()?;
        let mut ev_fork = ptr::null_mut();
        let mut ev_join = ptr::null_mut();
        // SAFETY: out-pointers valid; 2 = hipEventDisableTiming.
        unsafe {
            ffi::ckh(
                hip_sys::hipEventCreateWithFlags(&mut ev_fork, 2),
                "hipEventCreate",
            )?;
            ffi::ckh(
                hip_sys::hipEventCreateWithFlags(&mut ev_join, 2),
                "hipEventCreate",
            )?;
        }
        let dlog_dev = alloc(TMAX * hp.n_vocab * 4)?;
        let dlog_pin = ffi::host_alloc_pinned(TMAX * hp.n_vocab * 4)? as *mut f32;
        let logits_dev = alloc(TMAX * hp.n_vocab * 4)?;
        let logits_pin = ffi::host_alloc_pinned(TMAX * hp.n_vocab * 4)? as *mut f32;
        let gdev = alloc(32 * 4)?;
        let kda_cnt = alloc(hp.n_head.max(64) * 4)?;
        let router_cnt = alloc(64)?;
        ffi::memset(router_cnt.p, 0, 64)?;
        // one counter per window token (prompt windows too: fuse_hc_pre_pf)
        let hc_cnt = alloc(PCH_MAX * 4)?;
        ffi::memset(hc_cnt.p, 0, PCH_MAX * 4)?;
        ffi::memset(kda_cnt.p, 0, hp.n_head.max(64) * 4)?;
        let gpin = ffi::host_alloc_pinned(32 * 4)? as *mut i32;
        // MMQ (prefill) activation buffer + routing buffers
        // SAFETY: plain C queries.
        let mmq_bytes = unsafe {
            (lib.tk.tk_q36_mmq_q8_bytes)((pch() * k) as i64, e.max(hp.n_ff_exp) as i64).max((lib
                .tk
                .tk_q36_mmq_q8_bytes)(
                pch() as i64,
                (nh * hp.head_v_mla).max(hp.n_ff) as i64,
            ))
        };
        let mmq_xq = alloc(mmq_bytes)?;
        if let Some(reg) = lib.register_mmq {
            // SAFETY: registers the buffer as MMQ-layout activations (tk_prefill_moe_mmq dispatch).
            unsafe { reg(mmq_xq.p) };
        }
        let rowsrc = alloc(pch() * k * 4)?;
        let rownat = alloc(pch() * k * 4)?;
        let bounds = alloc((hp.n_expert + 1) * 4)?;
        if cfg.verbose {
            eprintln!(
                "glm: engine ctx {ctx}, state {:.2} GiB, activations {:.1} MiB (phase region {:.1} MiB, E4 kda {e4_kda} mla {e4_mla}), {nd} DSA / {nk} KDA layers",
                state_bytes as f64 / (1u64 << 30) as f64,
                act_bytes as f64 / (1u64 << 20) as f64,
                region_bytes as f64 / (1u64 << 20) as f64
            );
        }
        let hc16 = hc16_env(&lib.g);
        Ok(Self {
            snap_mid: None,
            lib,
            w,
            cfg: GlmConfig { ctx, ..cfg },
            st,
            a,
            c,
            stream,
            main: stream,
            side_stream,
            ev_fork,
            ev_join,
            // on by default (KDA gate projections next to q/k/v, the shared expert next to routing, the DSA
            // indexer next to the latent projections: the same kernels, disjoint buffers, bitwise); GLM_SIDE_STREAM=0
            use_side: std::env::var("GLM_SIDE_STREAM")
                .map(|v| v != "0")
                .unwrap_or(true),
            pair_ids: std::env::var("GLM_PAIR").map(|v| v != "0").unwrap_or(true),
            kq_ids2: std::env::var("GLM_KQ_IDS2")
                .map(|v| v != "0")
                .unwrap_or(true),
            dev_greedy: std::env::var("GLM_DEV_GREEDY")
                .map(|v| v != "0")
                .unwrap_or(true),
            dev_chain: std::env::var("GLM_DEV_CHAIN")
                .map(|v| v == "1")
                .unwrap_or(false),
            kda_fused: std::env::var("GLM_KDA_FUSED")
                .map(|v| v != "0")
                .unwrap_or(true),
            kda_cnt,
            // opt-in: in the decode graph the two separate launches have lower latency than the fused launch
            fuse_router: std::env::var("GLM_FUSE_ROUTER")
                .map(|v| v == "1")
                .unwrap_or(false),
            router_cnt,
            // opt-in: the last block's serial pre2 tail costs more than the launch gap it saves
            // in the multi-token decode graph
            fuse_hc_pre: std::env::var("GLM_FUSE_HC_PRE")
                .map(|v| v == "1")
                .unwrap_or(false),
            hc_cnt,
            fuse_hc_pre_pf: std::env::var("GLM_FUSE_HC_PRE_PF")
                .map(|v| v != "0")
                .unwrap_or(true),
            mtp_kv_only: std::env::var("GLM_MTP_KV_ONLY")
                .map(|v| v != "0")
                .unwrap_or(true),
            kv_only: false,
            share_quant: std::env::var("GLM_SHARE_QUANT")
                .map(|v| v != "0")
                .unwrap_or(true),
            e4_kda,
            e4_mla,
            pre_done: false,
            hc16,
            hc16_win: 0,
            gdev,
            gpin,
            dv_ids_dev: ptr::null_mut(),
            dv_embd: ptr::null_mut(),
            fuse_moe_post: std::env::var("GLM_FUSE_MOE_POST")
                .map(|v| v != "0")
                .unwrap_or(true),
            allocs,
            logits_host: vec![],
            logits_dev,
            logits_pin,
            dsa_index,
            kda_index,
            n_pool_cap,
            n_kv_max,
            dump: None,
            state_bytes,
            act_bytes,
            dv_head: None,
            dv_ids: vec![],
            dlog_dev,
            dlog_pin,
            mmq_xq,
            rowsrc,
            rownat,
            bounds,
            graph_used: HashMap::new(),
            graph_clock: 0,
            use_graphs: std::env::var("GLM_GRAPHS")
                .map(|v| v != "0")
                .unwrap_or(true),
            graphs: HashMap::new(),
            seen: Default::default(),
        })
    }

    #[inline]
    /// Run `f` on the side stream, concurrently with what the main stream issues next; `join()` waits for it.
    fn side<F: FnOnce(&mut Self) -> Result<()>>(&mut self, f: F) -> Result<()> {
        if !self.use_side {
            return f(self);
        }
        // SAFETY: valid events/streams; captured as graph dependencies while capturing.
        unsafe {
            ffi::ckh(
                hip_sys::hipEventRecord(self.ev_fork, self.main),
                "hipEventRecord fork",
            )?;
            ffi::ckh(
                hip_sys::hipStreamWaitEvent(self.side_stream, self.ev_fork, 0),
                "hipStreamWaitEvent fork",
            )?;
        }
        self.stream = self.side_stream;
        let r = f(self);
        self.stream = self.main;
        r?;
        // SAFETY: as above.
        unsafe {
            ffi::ckh(
                hip_sys::hipEventRecord(self.ev_join, self.side_stream),
                "hipEventRecord join",
            )?;
        }
        Ok(())
    }

    /// The main stream waits for the last `side()` work.
    fn join(&mut self) -> Result<()> {
        if !self.use_side {
            return Ok(());
        }
        // SAFETY: valid event/stream.
        unsafe {
            ffi::ckh(
                hip_sys::hipStreamWaitEvent(self.main, self.ev_join, 0),
                "hipStreamWaitEvent join",
            )
        }
    }

    fn s(&self) -> *mut c_void {
        self.stream as *mut c_void
    }

    fn dump_dev(&mut self, name: &str, il: usize, p: *const f32, n: usize) -> Result<()> {
        let Some(d) = self.dump.as_ref() else {
            return Ok(());
        };
        if let Some(f) = &d.layers {
            if !f.contains(&il) {
                return Ok(());
            }
        }
        ffi::sync(self.stream)?;
        let mut h = vec![0f32; n];
        ffi::d2h(h.as_mut_ptr() as *mut c_void, p as *const c_void, n * 4)?;
        let key = format!("{name}-{il}");
        self.dump.as_mut().unwrap().push(key, &h);
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------------------
    // kernel helpers

    fn mmvq(&self, w: &Tensor, xq: *const c_void, y: *mut f32, t: usize) -> Result<()> {
        // SAFETY: tk_* C ABI; pointers are device buffers sized for t columns.
        let stv = unsafe {
            (self.lib.tk.tk_mmvq)(
                self.lib.ctx,
                std::mem::transmute::<i32, ffi::tk_qtype>(w.ty as i32),
                w.p(),
                xq,
                y,
                w.n_in(),
                w.n_out(),
                t as i32,
                self.stream as _,
            )
        };
        self.lib.cktk(stv, "tk_mmvq")
    }

    /// Dense projection: q8_1 mat-vec for decode windows (t <= TMAX, `xq` = q8_1 of `x`), MMQ GEMM above.
    fn proj(
        &self,
        w: &Tensor,
        x: *const f32,
        xq: *const c_void,
        y: *mut f32,
        t: usize,
    ) -> Result<()> {
        if t <= TMAX {
            return self.mmvq(w, xq, y, t);
        }
        self.prefill_dense(w, x, y, t)
    }

    /// Several projections sharing the input (same type and n_in): one multi-matrix mat-vec launch for decode
    /// windows, separate MMQ GEMMs for prefill chunks.
    fn proj_multi(
        &self,
        ws: &[&Tensor],
        x: *const f32,
        xq: *const c_void,
        ys: &[*mut f32],
        t: usize,
    ) -> Result<()> {
        let same = ws
            .iter()
            .all(|w| w.ty == ws[0].ty && w.n_in() == ws[0].n_in());
        if t > TMAX && same && self.share_quant {
            // Prompt chunk: one MMQ-layout quantization of x for all the projections (same type and n_in:
            // the same q8_1 rows each separate call would produce), then one GEMM each
            self.mmq_quant(
                x,
                ptr::null(),
                ws[0].ty,
                ws[0].n_in() as usize,
                ws[0].n_in() as usize,
                t,
            )?;
            for (w, y) in ws.iter().zip(ys) {
                self.prefill_dense_q(w, *y, t)?;
            }
            return Ok(());
        }
        if t > TMAX || !same || ws.len() > 3 {
            for (w, y) in ws.iter().zip(ys) {
                self.proj(w, x, xq, *y, t)?;
            }
            return Ok(());
        }
        let wp: Vec<*const c_void> = ws.iter().map(|w| w.p()).collect();
        let no: Vec<i32> = ws.iter().map(|w| w.n_out()).collect();
        // SAFETY: tk_* C ABI; arrays read at launch.
        let stv = unsafe {
            (self.lib.tk.tk_mmvq_w32_multi)(
                self.lib.ctx,
                std::mem::transmute::<i32, ffi::tk_qtype>(ws[0].ty as i32),
                ws.len() as i32,
                wp.as_ptr(),
                ys.as_ptr(),
                no.as_ptr(),
                xq,
                ws[0].n_in(),
                t as i32,
                self.stream as _,
            )
        };
        self.lib.cktk(stv, "tk_mmvq_w32_multi")
    }

    /// MMQ-layout q8_1 of `rows` rows of `x` (gathered through `ids` when non-null) for weights of type `ty`.
    fn mmq_quant(
        &self,
        x: *const f32,
        ids: *const i32,
        ty: u32,
        cols: usize,
        ld: usize,
        rows: usize,
    ) -> Result<()> {
        // SAFETY: tk_* C ABI.
        let stv = unsafe {
            (self.lib.tk.tk_q36_mmq_quantize)(
                self.lib.ctx,
                x,
                ids,
                self.mmq_xq.p,
                ty as i32,
                cols as i64,
                ld as i64,
                rows as i64,
                self.stream as _,
            )
        };
        self.lib.cktk(stv, "tk_q36_mmq_quantize")
    }

    /// Grouped expert MMQ over the routed rows (sorted by expert in `bounds`, destination rows `rownat`).
    fn moe_product(
        &self,
        w: &Tensor,
        dst: *mut f32,
        ld: usize,
        rows: usize,
        t: usize,
    ) -> Result<()> {
        let pr = talus_kernels_sys::tk_moe_mmq_product {
            w: w.p(),
            qtype: unsafe { std::mem::transmute::<i32, ffi::tk_qtype>(w.ty as i32) },
            w_rows: w.n_out() as i64,
            w_cols: w.n_in() as i64,
            expert_bytes: w.expert_bytes(),
            n_experts: w.ne[2] as i32,
            x_q8_1: self.mmq_xq.p,
            bounds_dev: self.bounds.i(),
            ids_dev: self.rownat.i(),
            total_rows: rows as i64,
            max_rows: t as i64,
            dst,
            ld_dst: ld as i64,
            y_scale_dev: ptr::null(),
        };
        // SAFETY: tk_* C ABI; the descriptor is read at launch.
        let stv = unsafe { (self.lib.tk.tk_prefill_moe_mmq)(self.lib.ctx, &pr, self.stream as _) };
        self.lib.cktk(stv, "tk_prefill_moe_mmq")
    }

    /// fused grouped gate/up + SwiGLU (`tk_prefill_moe_mmq_gu`) into `dst` (ld `ld`); false = not covered
    /// (nothing launched: run the separate gate / up / SwiGLU path).  Bitwise that path.
    #[allow(clippy::too_many_arguments)]
    fn moe_product_gu(
        &self,
        gate: &Tensor,
        up: &Tensor,
        dst: *mut f32,
        ld: usize,
        rows: usize,
        t: usize,
        limit: f32,
    ) -> Result<bool> {
        if up.ty != gate.ty
            || up.expert_bytes() != gate.expert_bytes()
            || up.n_out() != gate.n_out()
        {
            return Ok(false);
        }
        let pr = talus_kernels_sys::tk_moe_mmq_product {
            w: gate.p(),
            qtype: unsafe { std::mem::transmute::<i32, ffi::tk_qtype>(gate.ty as i32) },
            w_rows: gate.n_out() as i64,
            w_cols: gate.n_in() as i64,
            expert_bytes: gate.expert_bytes(),
            n_experts: gate.ne[2] as i32,
            x_q8_1: self.mmq_xq.p,
            bounds_dev: self.bounds.i(),
            ids_dev: self.rownat.i(),
            total_rows: rows as i64,
            max_rows: t as i64,
            dst,
            ld_dst: ld as i64,
            y_scale_dev: ptr::null(),
        };
        // SAFETY: tk_* C ABI; the descriptor is read at launch.
        let stv = unsafe {
            (self.lib.tk.tk_prefill_moe_mmq_gu)(self.lib.ctx, &pr, up.p(), limit, self.stream as _)
        };
        if stv == talus_kernels_sys::TK_ERR_UNSUPPORTED {
            return Ok(false);
        }
        self.lib.cktk(stv, "tk_prefill_moe_mmq_gu")?;
        Ok(true)
    }

    fn quant(&self, x: *const f32, xq: *mut c_void, n_in: usize, t: usize) -> Result<()> {
        // SAFETY: tk_* C ABI.
        let stv = unsafe {
            (self.lib.tk.tk_quantize_q8_1)(
                self.lib.ctx,
                x,
                xq,
                n_in as i32,
                t as i32,
                self.stream as _,
            )
        };
        self.lib.cktk(stv, "tk_quantize_q8_1")
    }

    #[allow(dead_code)]
    fn quant_glm(&self, x: *const f32, xq: *mut c_void, n: usize) -> Result<()> {
        // SAFETY: tkg C ABI.
        ck(
            unsafe { (self.lib.g.quantize_q8_1)(x, xq, n as i32, self.s()) },
            "tkg_quantize_q8_1",
        )
    }

    fn f32mv(&self, w: &Tensor, x: *const f32, y: *mut f32, t: usize) -> Result<()> {
        let (ni, no) = (w.n_in() as usize, w.n_out() as usize);
        // Prompt chunks in one call (the kernel library tiles them; bitwise the 8-token tiles)
        for t0 in (0..t).step_by(if t > TMAX { t } else { TMAX }) {
            let tt = if t > TMAX { t } else { (t - t0).min(TMAX) };
            // SAFETY: tkg C ABI; tile offsets stay inside the [t][..] buffers.
            ck(
                unsafe {
                    (self.lib.g.f32_matvec)(
                        w.f(),
                        x.add(t0 * ni),
                        y.add(t0 * no),
                        ni as i32,
                        no as i32,
                        tt as i32,
                        self.s(),
                    )
                },
                "tkg_f32_matvec",
            )?;
        }
        Ok(())
    }

    /// up to 3 Q8_0 mat-vecs (same n_in) with f32 activations in one launch
    fn multi_q8(&self, mats: &[(&Tensor, *mut f32, *mut f32)], t: usize) -> Result<()> {
        // Prompt chunks go to the library in one call (it tiles them; bitwise the 8-token tiles)
        let w: Vec<*const c_void> = mats.iter().map(|m| m.0.p()).collect();
        let x: Vec<*const f32> = mats.iter().map(|m| m.1 as *const f32).collect();
        let y: Vec<*mut f32> = mats.iter().map(|m| m.2).collect();
        let n: Vec<i32> = mats.iter().map(|m| m.0.n_out()).collect();
        // SAFETY: tkg C ABI; the arrays are read at launch time.
        ck(
            unsafe {
                (self.lib.g.multi_q8_0)(
                    mats.len() as i32,
                    w.as_ptr(),
                    x.as_ptr(),
                    y.as_ptr(),
                    n.as_ptr(),
                    mats[0].0.n_in(),
                    t as i32,
                    self.s(),
                )
            },
            "tkg_multi_q8_0",
        )
    }

    /// per-head Q8_0 mat-vec with f32 activations x [t][H][n_in]
    fn heads(&self, w: &Tensor, x: *const f32, y: *mut f32, t: usize) -> Result<()> {
        let h = w.ne[2] as usize;
        let (ni, no) = (w.n_in() as usize, w.n_out() as usize);
        // Prompt chunks in one call (the kernel library tiles them; bitwise the 8-token tiles)
        for t0 in (0..t).step_by(if t > TMAX { t } else { TMAX }) {
            let tt = if t > TMAX { t } else { (t - t0).min(TMAX) };
            // SAFETY: tkg C ABI; tile offsets inside [t][H][..].
            ck(
                unsafe {
                    (self.lib.g.heads_q8_0)(
                        w.p(),
                        x.add(t0 * h * ni),
                        y.add(t0 * h * no),
                        h as i32,
                        ni as i32,
                        no as i32,
                        tt as i32,
                        self.s(),
                    )
                },
                "tkg_heads_q8_0",
            )?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn mmvq_heads(&self, w: &Tensor, xq: *const c_void, y: *mut f32, t: usize) -> Result<()> {
        let nh = w.ne[2] as i32;
        // SAFETY: tk_* C ABI.
        let stv = unsafe {
            (self.lib.tk.tk_mmvq_ids)(
                self.lib.ctx,
                std::mem::transmute::<i32, ffi::tk_qtype>(w.ty as i32),
                w.p(),
                w.expert_bytes(),
                self.a.head_ids.i(),
                nh * t as i32,
                xq,
                true,
                y,
                w.n_in(),
                w.n_out(),
                ptr::null(),
                ptr::null_mut(),
                self.stream as _,
            )
        };
        self.lib.cktk(stv, "tk_mmvq_ids(heads)")
    }

    fn rms(
        &self,
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        // SAFETY: tkg C ABI.
        ck(
            unsafe {
                (self.lib.g.rms_norm)(
                    x,
                    w,
                    y,
                    rows as i32,
                    cols as i32,
                    cols as i32,
                    cols as i32,
                    eps,
                    self.s(),
                )
            },
            "tkg_rms_norm",
        )
    }

    /// mHC pre (v2): Sinkhorn / collapse / rmsnorm from the partial mixes + Gram of the last post_mix -> a.y, a.yq, a.pc
    fn hc_pre(&mut self, hc: &super::weights::HcW, norm: &Tensor, t: usize) -> Result<()> {
        if std::mem::take(&mut self.pre_done) {
            return Ok(());
        }
        let hp = &self.w.hp;
        if self.hc16_win != 0 {
            let f = self.lib.g.hc_pre2_16.unwrap();
            // SAFETY: tkg C ABI (a.xhc holds 16-bit streams in this window).
            return ck(
                unsafe {
                    f(
                        self.a.xhc.v(),
                        self.a.parts.f(),
                        hc.scale.f(),
                        hc.base.f(),
                        norm.f(),
                        self.a.y.f(),
                        self.a.pc.f(),
                        t as i32,
                        hp.n_embd as i32,
                        hp.rms_eps,
                        hp.rms_eps,
                        hp.hc_eps,
                        hp.hc_iters as i32,
                        self.a.yq.v(),
                        self.hc16_win,
                        self.s(),
                    )
                },
                "tkg_hc_pre2_16",
            );
        }
        // SAFETY: tkg C ABI.
        ck(
            unsafe {
                (self.lib.g.hc_pre2)(
                    self.a.xhc.f(),
                    self.a.parts.f(),
                    hc.scale.f(),
                    hc.base.f(),
                    norm.f(),
                    self.a.y.f(),
                    self.a.pc.f(),
                    t as i32,
                    hp.n_embd as i32,
                    hp.rms_eps,
                    hp.rms_eps,
                    hp.hc_eps,
                    hp.hc_iters as i32,
                    self.a.yq.v(),
                    self.s(),
                )
            },
            "tkg_hc_pre2",
        )
    }

    /// mHC post (mode 0: from a.blk_out / a.pc; 1: a.y repeated; 2: mixes only) + partial mixes for `next`.
    fn hc_post_mix(
        &mut self,
        mode: i32,
        next: Option<&super::weights::HcW>,
        pre: Option<&Tensor>,
        t: usize,
    ) -> Result<()> {
        let w = next.map(|h| h.fn_.p()).unwrap_or(ptr::null());
        let xo = if mode == 1 {
            self.a.y.f()
        } else {
            self.a.blk_out.f()
        };
        if let (Some(h), Some(norm)) = (next, pre) {
            let fuse = if t <= TMAX {
                self.fuse_hc_pre
            } else {
                self.fuse_hc_pre_pf && self.hc16_win == 0
            };
            if fuse && mode != 2 && self.dump.is_none() {
                let hp = &self.w.hp;
                // SAFETY: tkg C ABI.
                let r = unsafe {
                    (self.lib.g.hc_post_mix_pre)(
                        xo,
                        self.a.xhc.f(),
                        self.a.pc.f(),
                        w,
                        self.a.parts.f(),
                        t as i32,
                        hp.n_embd as i32,
                        mode,
                        h.scale.f(),
                        h.base.f(),
                        norm.f(),
                        self.a.y.f(),
                        hp.rms_eps,
                        hp.rms_eps,
                        hp.hc_eps,
                        hp.hc_iters as i32,
                        self.a.yq.v(),
                        self.hc_cnt.i(),
                        self.s(),
                    )
                };
                if r > 0 {
                    ck(r, "tkg_hc_post_mix_pre")?;
                }
                if r == 0 {
                    self.pre_done = true;
                    return Ok(());
                }
            }
        }
        if self.hc16_win != 0 {
            let f = self.lib.g.hc_post_mix16.unwrap();
            // SAFETY: tkg C ABI (a.xhc holds 16-bit streams in this window).
            return ck(
                unsafe {
                    f(
                        xo,
                        self.a.xhc.v(),
                        self.a.pc.f(),
                        w,
                        self.a.parts.f(),
                        t as i32,
                        self.w.hp.n_embd as i32,
                        mode,
                        self.hc16_win,
                        self.s(),
                    )
                },
                "tkg_hc_post_mix16",
            );
        }
        // SAFETY: tkg C ABI.
        ck(
            unsafe {
                (self.lib.g.hc_post_mix)(
                    xo,
                    self.a.xhc.f(),
                    self.a.pc.f(),
                    w,
                    self.a.parts.f(),
                    t as i32,
                    self.w.hp.n_embd as i32,
                    mode,
                    self.s(),
                )
            },
            "tkg_hc_post_mix",
        )
    }

    /// Fused MoE combine + mHC post (mode 0) + partial mixes for `next`.
    fn hc_post_mix_moe(
        &mut self,
        next: Option<&super::weights::HcW>,
        pre: Option<&Tensor>,
        t: usize,
    ) -> Result<()> {
        let w = next.map(|h| h.fn_.p()).unwrap_or(ptr::null());
        if let (Some(h), Some(norm)) = (next, pre) {
            let fuse = if t <= TMAX {
                self.fuse_hc_pre
            } else {
                self.fuse_hc_pre_pf && self.hc16_win == 0
            };
            if fuse && self.dump.is_none() {
                let hp = &self.w.hp;
                // SAFETY: tkg C ABI.
                let r = unsafe {
                    (self.lib.g.hc_post_mix_moe_pre)(
                        self.a.ed.f(),
                        self.a.ew.f(),
                        self.a.sh_out.f(),
                        self.a.xhc.f(),
                        self.a.pc.f(),
                        w,
                        self.a.parts.f(),
                        t as i32,
                        hp.n_expert_used as i32,
                        hp.n_embd as i32,
                        h.scale.f(),
                        h.base.f(),
                        norm.f(),
                        self.a.y.f(),
                        hp.rms_eps,
                        hp.rms_eps,
                        hp.hc_eps,
                        hp.hc_iters as i32,
                        self.a.yq.v(),
                        self.hc_cnt.i(),
                        self.s(),
                    )
                };
                if r > 0 {
                    ck(r, "tkg_hc_post_mix_moe_pre")?;
                }
                if r == 0 {
                    self.pre_done = true;
                    return Ok(());
                }
            }
        }
        if self.hc16_win != 0 {
            let f = self.lib.g.hc_post_mix_moe16.unwrap();
            // SAFETY: tkg C ABI (a.xhc holds 16-bit streams in this window).
            return ck(
                unsafe {
                    f(
                        self.a.ed.f(),
                        self.a.ew.f(),
                        self.a.sh_out.f(),
                        self.a.xhc.v(),
                        self.a.pc.f(),
                        w,
                        self.a.parts.f(),
                        t as i32,
                        self.w.hp.n_expert_used as i32,
                        self.w.hp.n_embd as i32,
                        self.hc16_win,
                        self.s(),
                    )
                },
                "tkg_hc_post_mix_moe16",
            );
        }
        // SAFETY: tkg C ABI.
        ck(
            unsafe {
                (self.lib.g.hc_post_mix_moe)(
                    self.a.ed.f(),
                    self.a.ew.f(),
                    self.a.sh_out.f(),
                    self.a.xhc.f(),
                    self.a.pc.f(),
                    w,
                    self.a.parts.f(),
                    t as i32,
                    self.w.hp.n_expert_used as i32,
                    self.w.hp.n_embd as i32,
                    self.s(),
                )
            },
            "tkg_hc_post_mix_moe",
        )
    }

    // ------------------------------------------------------------------------------------------------------------
    // blocks (input: a.y normalized [t][n_embd] + a.yq; output: a.blk_out)

    fn kda_block(&mut self, l: &LayerW, il: usize, t: usize, k_snap: usize) -> Result<()> {
        let hp = self.w.hp.clone();
        let a = self.a;
        let kw = &l.kda;
        let ki = self.kda_index[il] as usize;
        let dk = hp.n_head * hp.head_kda;
        // gate projections on the side stream while q/k/v + conv run
        let gate_proj = |me: &mut Self| -> Result<()> {
            me.multi_q8(
                &[
                    (&kw.f_a, a.y.f(), a.fa.f()),
                    (&kw.g_a, a.y.f(), a.ga.f()),
                    (&kw.beta, a.y.f(), a.beta.f()),
                ],
                t,
            )?;
            me.multi_q8(
                &[
                    (&kw.f_b, a.fa.f(), a.graw.f()),
                    (&kw.g_b, a.ga.f(), a.g2.f()),
                ],
                t,
            )
        };
        if t <= TMAX {
            self.side(gate_proj)?;
        } else {
            gate_proj(self)?;
        }
        self.proj_multi(
            &[&kw.wq, &kw.wk, &kw.wv],
            a.y.f(),
            a.yq.v(),
            &[a.qr.f(), a.kr.f(), a.vr.f()],
            t,
        )?;
        if t <= TMAX && self.kda_fused && self.dump.is_none() {
            self.join()?;
            // SAFETY: tkg C ABI, buffers sized for TMAX.
            let r = unsafe {
                (self.lib.g.kda_fused)(
                    a.qr.f(),
                    a.kr.f(),
                    a.vr.f(),
                    self.st.conv_state[ki].f(),
                    kw.conv_q.f(),
                    kw.conv_k.f(),
                    kw.conv_v.f(),
                    a.graw.f(),
                    a.beta.f(),
                    self.st.kda_state[ki].f(),
                    self.c.kda_ids,
                    kw.dt_b.f(),
                    kw.a.f(),
                    a.o.f(),
                    a.g2.f(),
                    kw.o_norm.f(),
                    a.ogq.v(),
                    self.kda_cnt.i(),
                    hp.n_head as i32,
                    hp.head_kda as i32,
                    t as i32,
                    k_snap as i32,
                    1.0 / (hp.head_kda as f32).sqrt(),
                    hp.kda_lb,
                    1e-6,
                    hp.rms_eps,
                    self.s(),
                )
            };
            if r > 0 {
                ck(r, "tkg_kda_fused")?;
            }
            if r == 0 {
                return self.proj(&kw.wo, a.og.f(), a.ogq.v(), a.blk_out.f(), t);
            }
        }
        // prompt windows: conv + step in one launch (bitwise; -1 = not covered -> the two launches)
        let mut fused_cs = false;
        if let (true, Some(f)) = (t > TMAX, self.lib.g.kda_step_conv_pf) {
            // SAFETY: tkg C ABI; kda_cnt holds n_head zeros.
            let r = unsafe {
                f(
                    a.qr.f(),
                    a.kr.f(),
                    a.vr.f(),
                    self.st.conv_state[ki].f(),
                    kw.conv_q.f(),
                    kw.conv_k.f(),
                    kw.conv_v.f(),
                    a.graw.f(),
                    a.beta.f(),
                    self.st.kda_state[ki].f(),
                    self.c.kda_ids,
                    kw.dt_b.f(),
                    kw.a.f(),
                    a.o.f(),
                    self.kda_cnt.i(),
                    hp.n_head as i32,
                    hp.head_kda as i32,
                    t as i32,
                    k_snap as i32,
                    1.0 / (hp.head_kda as f32).sqrt(),
                    hp.kda_lb,
                    1e-6,
                    self.s(),
                )
            };
            if r > 0 {
                ck(r, "tkg_kda_step_conv_pf")?;
            }
            fused_cs = r == 0;
        }
        if !fused_cs {
            // SAFETY: tkg C ABI, buffers sized for TMAX.
            unsafe {
                ck(
                    (self.lib.g.kda_conv)(
                        a.qr.f(),
                        a.kr.f(),
                        a.vr.f(),
                        self.st.conv_state[ki].f(),
                        self.c.kda_ids,
                        k_snap as i32,
                        kw.conv_q.f(),
                        kw.conv_k.f(),
                        kw.conv_v.f(),
                        a.qc.f(),
                        a.kc.f(),
                        a.vc.f(),
                        t as i32,
                        dk as i32,
                        self.s(),
                    ),
                    "tkg_kda_conv",
                )?;
            }
            if t <= TMAX {
                self.join()?;
            }
            // SAFETY: tkg C ABI.
            unsafe {
                ck(
                    (self.lib.g.glm_kda_step)(
                        a.qc.f(),
                        a.kc.f(),
                        a.vc.f(),
                        a.graw.f(),
                        a.beta.f(),
                        self.st.kda_state[ki].f(),
                        self.c.kda_ids,
                        kw.dt_b.f(),
                        kw.a.f(),
                        a.o.f(),
                        hp.n_head as i32,
                        hp.head_kda as i32,
                        t as i32,
                        k_snap as i32,
                        1.0 / (hp.head_kda as f32).sqrt(),
                        hp.kda_lb,
                        1e-6,
                        self.s(),
                    ),
                    "tkg_glm_kda_step",
                )?;
            }
        }
        // SAFETY: tkg C ABI.
        unsafe {
            if t <= TMAX {
                ck(
                    (self.lib.g.kda_gate_q8)(
                        a.o.f(),
                        a.g2.f(),
                        kw.o_norm.f(),
                        a.ogq.v(),
                        t as i32,
                        hp.n_head as i32,
                        hp.head_kda as i32,
                        hp.rms_eps,
                        self.s(),
                    ),
                    "tkg_kda_gate_q8",
                )?;
            } else {
                ck(
                    (self.lib.g.kda_gate)(
                        a.o.f(),
                        a.g2.f(),
                        kw.o_norm.f(),
                        a.og.f(),
                        t as i32,
                        hp.n_head as i32,
                        hp.head_kda as i32,
                        hp.rms_eps,
                        self.s(),
                    ),
                    "tkg_kda_gate",
                )?;
            }
        }
        self.proj(&kw.wo, a.og.f(), a.ogq.v(), a.blk_out.f(), t)?;
        Ok(())
    }

    /// Attention block (KDA or MLA/DSA) of a window: windows above ASUB rows run in ASUB-row slices, in order (the
    /// recurrent state, conv state and caches advance slice by slice exactly as with ASUB-row windows).
    /// rb-glm E4: with `e4_kda` / `e4_mla` the projections of such a window run over the whole window instead (q/k/v,
    /// f_a/g_a/beta and o for KDA, wo for MLA; one GEMM instead of one per slice: the GEMM's output rows are
    /// independent of the row count, so bitwise), the recurrence / attention still per slice. A last slice of at
    /// most TMAX rows keeps its own decode-kernel window, as before (its projections are mat-vecs there).
    fn attn_block(&mut self, l: &LayerW, il: usize, t: usize, k_snap: usize) -> Result<()> {
        if t <= asub() {
            return if l.recr {
                self.kda_block(l, il, t, k_snap)
            } else {
                self.dsa_block(l, il, t)
            };
        }
        let e4 = if l.recr {
            self.e4_kda
        } else {
            self.e4_mla && !self.kv_only
        };
        let tail = t % asub();
        let tw = if tail != 0 && tail <= TMAX {
            t - tail
        } else {
            t
        };
        let from = if e4 && tw > asub() {
            if l.recr {
                self.kda_e4(l, il, tw, k_snap)?;
            } else {
                self.dsa_e4(l, il, tw)?;
            }
            tw
        } else {
            0
        };
        if from == t {
            return Ok(());
        }
        self.for_slices(l, il, from, t, |me, _r0, tt| {
            if l.recr {
                me.kda_block(l, il, tt, k_snap)
            } else {
                me.dsa_block(l, il, tt)
            }
        })
    }

    /// Run `f(self, r0, tt)` on the ASUB slices [r0, r0 + tt) of rows [0, to) that start at or after `from`, with
    /// a.y / a.yq / a.blk_out / the control block / pos0 pointing at the slice (restored afterwards); after a KDA
    /// layer's slice ending at the prompt-cache point `snap_mid`, that layer's state is copied out.
    fn for_slices<F: FnMut(&mut Self, usize, usize) -> Result<()>>(
        &mut self,
        l: &LayerW,
        il: usize,
        from: usize,
        to: usize,
        mut f: F,
    ) -> Result<()> {
        let e = self.w.hp.n_embd;
        let (y0, yq0, bo0, ctl0, pos00) =
            (self.a.y, self.a.yq, self.a.blk_out, self.c.ctl, self.c.pos0);
        let mut r = Ok(());
        for (k, r0) in (0..to).step_by(asub()).enumerate() {
            if r0 < from {
                continue;
            }
            let tt = (to - r0).min(asub());
            self.a.y.p = y0.at(r0 * e * 4);
            self.a.yq.p = yq0.at(r0 * q8b(e));
            self.a.blk_out.p = bo0.at(r0 * e * 4);
            // SAFETY: the control block holds CTL_TOK - CTL_SUB sub-window positions (PCH / ASUB of them).
            self.c.ctl = unsafe { (self.c.base as *mut i32).add(CTL_SUB + k) };
            self.c.pos0 = pos00 + r0;
            r = f(self, r0, tt);
            if r.is_err() {
                break;
            }
            if l.recr {
                if let Some((p, dst)) = self.snap_mid {
                    if p == pos00 + r0 + tt {
                        r = self.snap_layer(il, &dst);
                        if r.is_err() {
                            break;
                        }
                    }
                }
            }
        }
        self.a.y = y0;
        self.a.yq = yq0;
        self.a.blk_out = bo0;
        self.c.ctl = ctl0;
        self.c.pos0 = pos00;
        r
    }

    /// rb-glm E4: KDA block of rows [0, tw) of a prompt window (tw > ASUB, every slice above TMAX rows): f_a / g_a
    /// / beta and q / k / v over the whole rows, then per ASUB slice f_b / g_b, conv, recurrence and the output gate
    /// (written over the slice's raw q rows, which its conv has consumed), then o over the whole rows. Per slice the
    /// same kernels on the same inputs as `kda_block` above TMAX rows.
    fn kda_e4(&mut self, l: &LayerW, il: usize, tw: usize, k_snap: usize) -> Result<()> {
        let hp = self.w.hp.clone();
        let a = self.a;
        let kw = &l.kda;
        let ki = self.kda_index[il] as usize;
        let (nh, hk) = (hp.n_head, hp.head_kda);
        let dk = nh * hk;
        self.multi_q8(
            &[
                (&kw.f_a, a.y.f(), a.fa.f()),
                (&kw.g_a, a.y.f(), a.ga.f()),
                (&kw.beta, a.y.f(), a.beta.f()),
            ],
            tw,
        )?;
        self.proj_multi(
            &[&kw.wq, &kw.wk, &kw.wv],
            a.y.f(),
            a.yq.v(),
            &[a.qr.f(), a.kr.f(), a.vr.f()],
            tw,
        )?;
        self.for_slices(l, il, 0, tw, |me, r0, tt| {
            // SAFETY: row offsets r0 + tt <= tw inside the whole-window buffers (sized for pch() rows with E4).
            let (q, k, v, fa, ga, beta) = unsafe {
                (
                    a.qr.f().add(r0 * dk),
                    a.kr.f().add(r0 * dk),
                    a.vr.f().add(r0 * dk),
                    a.fa.f().add(r0 * hk),
                    a.ga.f().add(r0 * hk),
                    a.beta.f().add(r0 * nh),
                )
            };
            me.multi_q8(&[(&kw.f_b, fa, a.graw.f()), (&kw.g_b, ga, a.g2.f())], tt)?;
            // conv + step in one launch (bitwise the two below; -1 = not covered)
            let mut fused_cs = false;
            if let Some(f) = me.lib.g.kda_step_conv_pf {
                // SAFETY: tkg C ABI; kda_cnt holds n_head zeros.
                let r = unsafe {
                    f(
                        q,
                        k,
                        v,
                        me.st.conv_state[ki].f(),
                        kw.conv_q.f(),
                        kw.conv_k.f(),
                        kw.conv_v.f(),
                        a.graw.f(),
                        beta,
                        me.st.kda_state[ki].f(),
                        me.c.kda_ids,
                        kw.dt_b.f(),
                        kw.a.f(),
                        a.o.f(),
                        me.kda_cnt.i(),
                        nh as i32,
                        hk as i32,
                        tt as i32,
                        k_snap as i32,
                        1.0 / (hk as f32).sqrt(),
                        hp.kda_lb,
                        1e-6,
                        me.s(),
                    )
                };
                if r > 0 {
                    ck(r, "tkg_kda_step_conv_pf")?;
                }
                fused_cs = r == 0;
            }
            // SAFETY: tkg C ABI; slice buffers sized for ASUB rows.
            unsafe {
                if !fused_cs {
                    ck(
                        (me.lib.g.kda_conv)(
                            q,
                            k,
                            v,
                            me.st.conv_state[ki].f(),
                            me.c.kda_ids,
                            k_snap as i32,
                            kw.conv_q.f(),
                            kw.conv_k.f(),
                            kw.conv_v.f(),
                            a.qc.f(),
                            a.kc.f(),
                            a.vc.f(),
                            tt as i32,
                            dk as i32,
                            me.s(),
                        ),
                        "tkg_kda_conv",
                    )?;
                    ck(
                        (me.lib.g.glm_kda_step)(
                            a.qc.f(),
                            a.kc.f(),
                            a.vc.f(),
                            a.graw.f(),
                            beta,
                            me.st.kda_state[ki].f(),
                            me.c.kda_ids,
                            kw.dt_b.f(),
                            kw.a.f(),
                            a.o.f(),
                            nh as i32,
                            hk as i32,
                            tt as i32,
                            k_snap as i32,
                            1.0 / (hk as f32).sqrt(),
                            hp.kda_lb,
                            1e-6,
                            me.s(),
                        ),
                        "tkg_glm_kda_step",
                    )?;
                }
                ck(
                    (me.lib.g.kda_gate)(
                        a.o.f(),
                        a.g2.f(),
                        kw.o_norm.f(),
                        q,
                        tt as i32,
                        nh as i32,
                        hk as i32,
                        hp.rms_eps,
                        me.s(),
                    ),
                    "tkg_kda_gate",
                )?;
            }
            Ok(())
        })?;
        // the gated outputs of every slice now sit in the q rows
        self.proj(&kw.wo, a.qr.f(), a.ogq.v(), a.blk_out.f(), tw)
    }

    /// rb-glm E4: MLA/DSA block of rows [0, tw) of a prompt window (tw > ASUB, every slice above TMAX rows): per
    /// ASUB slice everything up to wv_b (into the slice's rows of the whole-window vb), then wo over the whole rows.
    fn dsa_e4(&mut self, l: &LayerW, il: usize, tw: usize) -> Result<()> {
        let a = self.a;
        let vbw = self.w.hp.n_head * self.w.hp.head_v_mla;
        self.for_slices(l, il, 0, tw, |me, r0, tt| {
            // SAFETY: row offset inside the whole-window vb (sized for pch() rows with E4).
            let vb = unsafe { a.vb.f().add(r0 * vbw) };
            me.dsa_core(l, il, tt, vb)
        })?;
        self.proj(&l.mla.wo, a.vb.f(), a.vbq.v(), a.blk_out.f(), tw)
    }

    fn dsa_block(&mut self, l: &LayerW, il: usize, t: usize) -> Result<()> {
        if self.kv_only {
            return self.dsa_kv_store(l, il, t);
        }
        let hp = self.w.hp.clone();
        let a = self.a;
        self.dsa_core(l, il, t, a.vb.f())?;
        if t <= TMAX {
            self.quant(a.vb.f(), a.vbq.v(), hp.n_head * hp.head_v_mla, t)?;
        }
        self.proj(&l.mla.wo, a.vb.f(), a.vbq.v(), a.blk_out.f(), t)
    }

    /// MLA/DSA block up to the value projection: q path, indexer + selection, latent cache, sparse attention,
    /// wv_b into `vb` ([t][n_head][head_v_mla]).
    fn dsa_core(&mut self, l: &LayerW, il: usize, t: usize, vb: *mut f32) -> Result<()> {
        let hp = self.w.hp.clone();
        let a = self.a;
        let m = &l.mla;
        let j = self.dsa_index[il] as usize;
        let nh = hp.n_head;
        let lat = hp.kv_lora;
        // q path
        self.proj(&m.wq_a, a.y.f(), a.yq.v(), a.qa.f(), t)?;
        self.rms(
            a.qa.f(),
            m.q_a_norm.f(),
            a.qan.f(),
            t,
            hp.q_lora,
            hp.rms_eps,
        )?;
        if t <= TMAX {
            self.quant(a.qan.f(), a.qaq.v(), hp.q_lora, t)?;
        }
        let n_pool = self.n_pool_bucket(t);
        // selection width: never more rows than the bucket holds (short contexts launch a narrower sparse FA)
        let nkv = self.n_kv_max.min(n_pool * hp.kpool);
        let ctl = self.c.ctl;
        let st_kg = self.st.keygate[j];
        let st_pool = self.st.pooled[j];
        let st_kv = self.st.mla_kv[j];
        // indexer branch (projections, key store, pooling, scores, top-k selection) on the side stream
        // idx_k, idx_gate and wkv_a read the same yq (Q8_0, n_in n_embd): one multi-matrix launch for
        // decode windows on a single stream (rows bitwise the separate launches); kvr is only read after the indexer
        // Prompt chunks run the three y projections together too (one shared quantization of y)
        let fused3 =
            (self.pair_ids && !self.use_side && t <= TMAX) || (t > TMAX && self.share_quant);
        if fused3 {
            self.proj_multi(
                &[&m.idx_k, &m.idx_gate, &m.wkv_a],
                a.y.f(),
                a.yq.v(),
                &[a.ik.f(), a.ig.f(), a.kvr.f()],
                t,
            )?;
        }
        let indexer = |me: &mut Self| -> Result<()> {
            me.proj(&m.idx_q_b, a.qan.f(), a.qaq.v(), a.iq.f(), t)?;
            if !fused3 {
                me.proj(&m.idx_k, a.y.f(), a.yq.v(), a.ik.f(), t)?;
                me.proj(&m.idx_gate, a.y.f(), a.yq.v(), a.ig.f(), t)?;
            }
            me.f32mv(&m.idx_proj, a.y.f(), a.iw.f(), t)?;
            let s = me.s();
            // SAFETY: tkg C ABI.
            unsafe {
                ck(
                    (me.lib.g.idx_store)(
                        a.ik.f(),
                        m.idx_k_norm.f(),
                        m.idx_k_norm_b.f(),
                        hp.norm_eps,
                        a.ig.f(),
                        st_kg.v(),
                        ctl,
                        t as i32,
                        s,
                    ),
                    "tkg_idx_store",
                )?;
                ck(
                    (me.lib.g.idx_pool)(
                        st_kg.v(),
                        m.idx_ape.f(),
                        st_pool.v(),
                        ctl,
                        t as i32,
                        hp.kpool as i32,
                        s,
                    ),
                    "tkg_idx_pool",
                )?;
                ck(
                    (me.lib.g.glm_indexer)(
                        a.iq.f(),
                        st_pool.v(),
                        a.iw.f(),
                        a.scores.f(),
                        hp.idx_dim as i32,
                        hp.idx_heads as i32,
                        n_pool as i32,
                        t as i32,
                        s,
                    ),
                    "tkg_glm_indexer",
                )?;
                ck(
                    (me.lib.g.dsa_select)(
                        a.scores.f(),
                        n_pool as i32,
                        ctl,
                        t as i32,
                        hp.kpool as i32,
                        (hp.idx_topk / hp.kpool) as i32,
                        nkv as i32,
                        a.sel.i(),
                        s,
                    ),
                    "tkg_dsa_select",
                )?;
            }
            Ok(())
        };
        let side = t <= TMAX;
        if side {
            self.side(indexer)?;
        } else {
            indexer(self)?;
        }
        // q absorption and the latent cache on the main stream
        self.proj(&m.wq_b, a.qan.f(), a.qaq.v(), a.q.f(), t)?;
        self.heads(&m.wk_b, a.q.f(), a.qabs.f(), t)?;
        if !fused3 {
            self.proj(&m.wkv_a, a.y.f(), a.yq.v(), a.kvr.f(), t)?;
        }
        // SAFETY: tkg C ABI.
        unsafe {
            ck(
                (self.lib.g.mla_store)(
                    a.kvr.f(),
                    m.kv_a_norm.f(),
                    hp.rms_eps,
                    st_kv.v(),
                    ctl,
                    t as i32,
                    lat as i32,
                    self.s(),
                ),
                "tkg_mla_store",
            )?;
        }
        if side {
            self.join()?;
        }
        // SAFETY: tkg C ABI.
        unsafe {
            ck(
                (self.lib.g.glm_sparse_fa_idx)(
                    a.qabs.f(),
                    st_kv.v(),
                    self.st.zero_mask.v(),
                    a.sel.i(),
                    a.fa_split.f(),
                    a.lat.f(),
                    lat as i32,
                    nh as i32,
                    t as i32,
                    nkv as i32,
                    (nh * lat) as u32,
                    lat as u32,
                    lat as u32,
                    1.0 / (hp.head_k_mla as f32).sqrt(),
                    self.s(),
                ),
                "tkg_glm_sparse_fa_idx",
            )?;
        }
        if self.dump.is_some() {
            self.dump_dev("idx_q", il, a.iq.f(), t * hp.idx_heads * hp.idx_dim)?;
            self.dump_dev("idx_w", il, a.iw.f(), t * hp.idx_heads)?;
            self.dump_dev("idx_scores", il, a.scores.f(), t * n_pool)?;
            let sel = {
                ffi::sync(self.stream)?;
                let mut v = vec![0i32; t * nkv];
                ffi::d2h(v.as_mut_ptr() as *mut c_void, a.sel.v(), v.len() * 4)?;
                v.iter().map(|&x| x as f32).collect::<Vec<f32>>()
            };
            if self
                .dump
                .as_ref()
                .unwrap()
                .layers
                .as_ref()
                .map(|f| f.contains(&il))
                .unwrap_or(true)
            {
                let mut full = vec![-1f32; t * self.n_kv_max];
                for r in 0..t {
                    full[r * self.n_kv_max..r * self.n_kv_max + nkv]
                        .copy_from_slice(&sel[r * nkv..(r + 1) * nkv]);
                }
                self.dump
                    .as_mut()
                    .unwrap()
                    .push(format!("dsa_sel-{il}"), &full);
            }
            self.dump_dev("q_abs", il, a.qabs.f(), t * nh * lat)?;
            self.dump_dev("lat", il, a.lat.f(), t * nh * lat)?;
        }
        self.heads(&m.wv_b, a.lat.f(), vb, t)
    }

    /// The cache writes of `dsa_block` only (indexer key store + pools, latent KV store), with the same
    /// launches / inputs as the full block (so the caches are bitwise the full block's).
    fn dsa_kv_store(&mut self, l: &LayerW, il: usize, t: usize) -> Result<()> {
        let hp = self.w.hp.clone();
        let a = self.a;
        let m = &l.mla;
        let j = self.dsa_index[il] as usize;
        let ctl = self.c.ctl;
        let st_kg = self.st.keygate[j];
        let st_pool = self.st.pooled[j];
        let st_kv = self.st.mla_kv[j];
        let fused3 = self.pair_ids && !self.use_side && t <= TMAX;
        if fused3 || t > TMAX {
            self.proj_multi(
                &[&m.idx_k, &m.idx_gate, &m.wkv_a],
                a.y.f(),
                a.yq.v(),
                &[a.ik.f(), a.ig.f(), a.kvr.f()],
                t,
            )?;
        } else {
            self.proj(&m.idx_k, a.y.f(), a.yq.v(), a.ik.f(), t)?;
            self.proj(&m.idx_gate, a.y.f(), a.yq.v(), a.ig.f(), t)?;
            self.proj(&m.wkv_a, a.y.f(), a.yq.v(), a.kvr.f(), t)?;
        }
        let s = self.s();
        // SAFETY: tkg C ABI.
        unsafe {
            ck(
                (self.lib.g.idx_store)(
                    a.ik.f(),
                    m.idx_k_norm.f(),
                    m.idx_k_norm_b.f(),
                    hp.norm_eps,
                    a.ig.f(),
                    st_kg.v(),
                    ctl,
                    t as i32,
                    s,
                ),
                "tkg_idx_store",
            )?;
            ck(
                (self.lib.g.idx_pool)(
                    st_kg.v(),
                    m.idx_ape.f(),
                    st_pool.v(),
                    ctl,
                    t as i32,
                    hp.kpool as i32,
                    s,
                ),
                "tkg_idx_pool",
            )?;
            ck(
                (self.lib.g.mla_store)(
                    a.kvr.f(),
                    m.kv_a_norm.f(),
                    hp.rms_eps,
                    st_kv.v(),
                    ctl,
                    t as i32,
                    hp.kv_lora as i32,
                    s,
                ),
                "tkg_mla_store",
            )?;
        }
        Ok(())
    }

    /// Dense FFN / shared expert: clamp-SwiGLU MLP of a.y into `out` (windows above PCH rows in PCH-row slices: the
    /// scratch is sized for PCH rows; rows are independent, so the slices are bitwise the whole window).
    fn dense_ffn(
        &mut self,
        f: &super::weights::FfnW,
        clamp: f32,
        out: *mut f32,
        t: usize,
    ) -> Result<()> {
        if t <= PCH {
            return self.dense_ffn_rows(f, clamp, out, t);
        }
        let e = self.w.hp.n_embd;
        let (y0, yq0) = (self.a.y, self.a.yq);
        let mut r = Ok(());
        for r0 in (0..t).step_by(PCH) {
            let tt = (t - r0).min(PCH);
            self.a.y.p = y0.at(r0 * e * 4);
            self.a.yq.p = yq0.at(r0 * q8b(e));
            // SAFETY: out holds t rows of e floats.
            r = self.dense_ffn_rows(f, clamp, unsafe { out.add(r0 * e) }, tt);
            if r.is_err() {
                break;
            }
        }
        self.a.y = y0;
        self.a.yq = yq0;
        r
    }

    fn dense_ffn_rows(
        &mut self,
        f: &super::weights::FfnW,
        clamp: f32,
        out: *mut f32,
        t: usize,
    ) -> Result<()> {
        let mut a = self.a;
        let ffd = f.gate.n_out() as usize;
        if self.stream == self.side_stream && t <= TMAX {
            // on the side stream (shared expert next to the routed experts): own buffers
            a.fg = a.sfg;
            a.fu = a.sfu;
            a.fact = a.sfact;
            a.factq = a.sfactq;
        }
        // gate + up as one multi-matrix launch for decode windows (rows bitwise the separate launches)
        if self.pair_ids {
            self.proj_multi(
                &[&f.gate, &f.up],
                a.y.f(),
                a.yq.v(),
                &[a.fg.f(), a.fu.f()],
                t,
            )?;
        } else {
            self.proj(&f.gate, a.y.f(), a.yq.v(), a.fg.f(), t)?;
            self.proj(&f.up, a.y.f(), a.yq.v(), a.fu.f(), t)?;
        }
        // SAFETY: tkg C ABI.
        unsafe {
            if t <= TMAX {
                ck(
                    (self.lib.g.swiglu_clamp_q8)(
                        a.fg.f(),
                        a.fu.f(),
                        a.factq.v(),
                        (t * ffd) as i32,
                        clamp,
                        self.s(),
                    ),
                    "tkg_swiglu_clamp_q8",
                )?;
            } else {
                ck(
                    (self.lib.g.swiglu_clamp)(
                        a.fg.f(),
                        a.fu.f(),
                        a.fact.f(),
                        (t * ffd) as i32,
                        clamp,
                        self.s(),
                    ),
                    "tkg_swiglu_clamp",
                )?;
            }
        }
        self.proj(&f.down, a.fact.f(), a.factq.v(), out, t)
    }

    fn ffn_block(&mut self, l: &LayerW, il: usize, t: usize) -> Result<()> {
        let hp = self.w.hp.clone();
        let a = self.a;
        let f = &l.ffn;
        let e = hp.n_embd;
        let clamp_sh = hp.clamp_shexp.get(il).copied().unwrap_or(0.0);
        if !l.moe {
            return self.dense_ffn(f, clamp_sh, a.blk_out.f(), t);
        }
        // the shared expert runs on the side stream, concurrently with routing and the routed experts
        if t <= TMAX {
            self.side(|me| me.dense_ffn(f, clamp_sh, a.sh_out.f(), t))?;
        } else {
            self.dense_ffn(f, clamp_sh, a.sh_out.f(), t)?;
        }
        let k = hp.n_expert_used;
        let nf = hp.n_ff_exp;
        let clamp_e = hp.clamp_exp.get(il).copied().unwrap_or(0.0);
        // SAFETY: tkg C ABI.
        let fused_r = t <= TMAX
            && self.fuse_router
            && unsafe {
                let r = (self.lib.g.f32mv_router)(
                    f.gate_inp.f(),
                    a.y.f(),
                    a.rlog.f(),
                    f.gate_inp.n_in() as i32,
                    hp.n_expert as i32,
                    t as i32,
                    f.probs_b.f(),
                    a.eids.i(),
                    a.ew.f(),
                    k as i32,
                    hp.w_scale,
                    self.router_cnt.i(),
                    self.s(),
                );
                if r > 0 {
                    ck(r, "tkg_f32mv_router")?;
                }
                r == 0
            };
        if !fused_r {
            self.f32mv(&f.gate_inp, a.y.f(), a.rlog.f(), t)?;
            // SAFETY: tkg C ABI.
            unsafe {
                ck(
                    (self.lib.g.router)(
                        a.rlog.f(),
                        f.probs_b.f(),
                        a.eids.i(),
                        a.ew.f(),
                        t as i32,
                        hp.n_expert as i32,
                        k as i32,
                        hp.w_scale,
                        self.s(),
                    ),
                    "tkg_router",
                )?;
            }
        }
        let gate_t = f.gate_exps.ty;
        let down_t = f.down_exps.ty;
        let glm_ok = |ty: u32| ty == gt::IQ2_S || ty == gt::IQ3_S;
        let pf_ok = |ty: u32| glm_ok(ty) || ty == gt::Q2_K || ty == gt::Q3_K;
        let prefill = t > TMAX && pf_ok(gate_t) && pf_ok(f.up_exps.ty) && pf_ok(down_t);
        if !prefill && t > TMAX {
            // (eactq is sized for decode windows; the trunk / MTP expert types always take the prefill seam)
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "glm: routed experts of types {gate_t}/{down_t} in a {t}-row window"
                ),
            });
        }
        if prefill {
            // prefill: routed experts through the prefill seam (prefill.rs)
            self.prefill_experts(f, t, k, nf, clamp_e)?;
        } else if glm_ok(gate_t) && glm_ok(f.up_exps.ty) {
            // gate + up in one launch when both have one type (rows bitwise the two-launch result);
            // -1 = no pair kernel for the config: two launches below
            let paired = gate_t == f.up_exps.ty && self.pair_ids && {
                // SAFETY: tkg C ABI.
                let r = unsafe {
                    (self.lib.g.glm_mmvq_ids_pair)(
                        gate_t as i32,
                        f.gate_exps.p(),
                        f.up_exps.p(),
                        a.yq.v(),
                        a.eids.i(),
                        a.eg.f(),
                        a.eu.f(),
                        nf as i32,
                        e as i32,
                        k as i32,
                        t as i32,
                        1,
                        k as i32,
                        self.s(),
                    )
                };
                if r > 0 {
                    ck(r, "glm_mmvq_ids_pair")?;
                }
                r == 0
            };
            // SAFETY: tkg C ABI.
            if !paired {
                unsafe {
                    ck(
                        (self.lib.g.glm_mmvq_ids)(
                            gate_t as i32,
                            f.gate_exps.p(),
                            a.yq.v(),
                            a.eids.i(),
                            a.eg.f(),
                            nf as i32,
                            e as i32,
                            k as i32,
                            t as i32,
                            1,
                            k as i32,
                            self.s(),
                        ),
                        "glm_mmvq_ids gate",
                    )?;
                    ck(
                        (self.lib.g.glm_mmvq_ids)(
                            f.up_exps.ty as i32,
                            f.up_exps.p(),
                            a.yq.v(),
                            a.eids.i(),
                            a.eu.f(),
                            nf as i32,
                            e as i32,
                            k as i32,
                            t as i32,
                            1,
                            k as i32,
                            self.s(),
                        ),
                        "glm_mmvq_ids up",
                    )?;
                }
            }
        } else {
            // MTP layer: Q2_K gate/up via the f32-activation ids path. one launch for both, activations
            // staged in LDS (bitwise); GLM_KQ_IDS2=0 = one tkg_kq_ids launch per matrix below
            let both = self.kq_ids2
                && f.gate_exps.ty == f.up_exps.ty
                && f.gate_exps.expert_bytes() == f.up_exps.expert_bytes()
                && {
                    // SAFETY: tkg C ABI.
                    let r = unsafe {
                        (self.lib.g.kq_ids2)(
                            f.gate_exps.ty as i32,
                            f.gate_exps.p(),
                            f.up_exps.p(),
                            f.gate_exps.expert_bytes(),
                            a.eids.i(),
                            (t * k) as i32,
                            k as i32,
                            a.y.f(),
                            0,
                            a.eg.f(),
                            a.eu.f(),
                            e as i32,
                            nf as i32,
                            self.s(),
                        )
                    };
                    if r > 0 {
                        ck(r, "tkg_kq_ids2 gate/up")?;
                    }
                    r == 0
                };
            if !both {
                for (w, y) in [(&f.gate_exps, a.eg), (&f.up_exps, a.eu)] {
                    // SAFETY: tkg C ABI.
                    unsafe {
                        ck(
                            (self.lib.g.kq_ids)(
                                w.ty as i32,
                                w.p(),
                                w.expert_bytes(),
                                a.eids.i(),
                                (t * k) as i32,
                                k as i32,
                                a.y.f(),
                                0,
                                y.f(),
                                e as i32,
                                nf as i32,
                                self.s(),
                            ),
                            "tkg_mmvq_ids_f32 gate/up",
                        )?;
                    }
                }
            }
        }
        if prefill {
            // done above
        } else if glm_ok(down_t) {
            // SAFETY: tkg C ABI.
            unsafe {
                ck(
                    (self.lib.g.swiglu_clamp_q8)(
                        a.eg.f(),
                        a.eu.f(),
                        a.eactq.v(),
                        (t * k * nf) as i32,
                        clamp_e,
                        self.s(),
                    ),
                    "tkg_swiglu_clamp_q8 exp",
                )?;
            }
            // SAFETY: tkg C ABI.
            unsafe {
                ck(
                    (self.lib.g.glm_mmvq_ids)(
                        down_t as i32,
                        f.down_exps.p(),
                        a.eactq.v(),
                        a.eids.i(),
                        a.ed.f(),
                        e as i32,
                        nf as i32,
                        k as i32,
                        t as i32,
                        k as i32,
                        k as i32,
                        self.s(),
                    ),
                    "glm_mmvq_ids down",
                )?;
            }
        } else {
            // SAFETY: tkg C ABI.
            unsafe {
                ck(
                    (self.lib.g.swiglu_clamp)(
                        a.eg.f(),
                        a.eu.f(),
                        a.eact.f(),
                        (t * k * nf) as i32,
                        clamp_e,
                        self.s(),
                    ),
                    "tkg_swiglu_clamp exp",
                )?;
                let r = if self.kq_ids2 {
                    (self.lib.g.kq_ids2)(
                        down_t as i32,
                        f.down_exps.p(),
                        std::ptr::null(),
                        f.down_exps.expert_bytes(),
                        a.eids.i(),
                        (t * k) as i32,
                        k as i32,
                        a.eact.f(),
                        1,
                        a.ed.f(),
                        std::ptr::null_mut(),
                        nf as i32,
                        e as i32,
                        self.s(),
                    )
                } else {
                    -1
                };
                if r > 0 {
                    ck(r, "tkg_kq_ids2 down")?;
                }
                if r != 0 {
                    ck(
                        (self.lib.g.kq_ids)(
                            down_t as i32,
                            f.down_exps.p(),
                            f.down_exps.expert_bytes(),
                            a.eids.i(),
                            (t * k) as i32,
                            k as i32,
                            a.eact.f(),
                            1,
                            a.ed.f(),
                            nf as i32,
                            e as i32,
                            self.s(),
                        ),
                        "tkg_mmvq_ids_f32 down",
                    )?;
                }
            }
        }
        if t <= TMAX {
            self.join()?;
        }
        if il < hp.n_layer && self.fuse_moe_post && self.dump.is_none() {
            return Ok(());
        }
        // SAFETY: tkg C ABI.
        unsafe {
            ck(
                (self.lib.g.moe_combine)(
                    a.ed.f(),
                    a.ew.f(),
                    a.sh_out.f(),
                    a.blk_out.f(),
                    t as i32,
                    k as i32,
                    e as i32,
                    self.s(),
                ),
                "tkg_moe_combine",
            )?;
        }
        Ok(())
    }

    /// Pools covered by the indexer launch for this window (bucketed so graphs can be reused).
    fn n_pool_bucket(&self, t: usize) -> usize {
        let pos_end = self.c.pos0 + t;
        let b = pos_end.div_ceil(KV_BUCKET).max(1) * KV_BUCKET;
        (b / self.w.hp.kpool).min(self.n_pool_cap)
    }

    // ------------------------------------------------------------------------------------------------------------
    // trunk forward

    /// Stage the window inputs (tokens, pos0, KDA slot ids) and copy them stream-ordered into the device control
    /// block, so the host can run ahead of the GPU without overwriting inputs of windows still in flight.
    /// Embedding table and row ids for the embed kernel: the window's staged rows (host-row mode) or the device table.
    fn embed_table(&self) -> *const c_void {
        if self.w.tok_embd_rows.is_some() {
            self.c.rows_dev
        } else {
            self.w.tok_embd.p()
        }
    }

    fn embed_ids(&self) -> *const i32 {
        if self.w.tok_embd_rows.is_some() {
            self.c.iota
        } else {
            self.c.tok
        }
    }

    fn set_window(&mut self, toks: &[u32], pos0: usize, kda_ids: &[i32]) -> Result<()> {
        let k = self.c.slot % RING;
        self.c.slot += 1;
        // SAFETY: event valid; the slot's previous copy must have completed before it is rewritten.
        ffi::ckh(
            unsafe { hip_sys::hipEventSynchronize(self.c.ev[k]) },
            "hipEventSynchronize ctl",
        )?;
        // SAFETY: slot k has CTL_BYTES of pinned memory.
        unsafe {
            let h = (self.c.ring as *mut u8).add(k * self.c.slot_bytes) as *mut i32;
            for (i, &tk) in toks.iter().enumerate() {
                *h.add(CTL_TOK + i) = tk as i32;
            }
            *h.add(CTL_POS) = pos0 as i32;
            for k in 0..(CTL_TOK - CTL_SUB) {
                *h.add(CTL_SUB + k) = (pos0 + k * asub()) as i32;
            }
            for i in 0..8 {
                *h.add(CTL_KDA + i) = kda_ids.get(i).copied().unwrap_or(0);
            }
            ffi::h2d_async(
                self.c.base,
                h as *const c_void,
                (CTL_TOK + toks.len()) * 4,
                self.stream,
            )?;
            if let Some(er) = self.w.tok_embd_rows.as_ref() {
                // this window's embedding rows: pread from the GGUF (page cache) into pinned staging, then to the
                // device rows area the embed kernel reads (row i = token i)
                let rb = self.c.row_bytes;
                let stage = if toks.len() <= TMAX {
                    (h as *mut u8).add(CTL_BYTES.div_ceil(256) * 256)
                } else {
                    ffi::ckh(
                        hip_sys::hipEventSynchronize(self.c.big_ev),
                        "hipEventSynchronize embed",
                    )?;
                    self.c.big as *mut u8
                };
                for (i, &tk) in toks.iter().enumerate() {
                    let dst = std::slice::from_raw_parts_mut(stage.add(i * rb), rb);
                    er.read_row(tk as usize, dst)?;
                }
                ffi::h2d_async(
                    self.c.rows_dev,
                    stage as *const c_void,
                    toks.len() * rb,
                    self.stream,
                )?;
                if toks.len() > TMAX {
                    ffi::ckh(
                        hip_sys::hipEventRecord(self.c.big_ev, self.stream),
                        "hipEventRecord embed",
                    )?;
                }
            }
            ffi::ckh(
                hip_sys::hipEventRecord(self.c.ev[k], self.stream),
                "hipEventRecord ctl",
            )?;
        }
        self.c.pos0 = pos0;
        Ok(())
    }

    /// Trunk forward of one window (t <= TMAX tokens at positions pos0..). KDA reads slot kda_ids[0] and writes the
    /// last k_snap tokens' states to kda_ids[..k_snap]. Leaves logits of every row in a.h -> logits (if want_logits).
    fn forward_window(&mut self, t: usize, k_snap: usize, want_logits: bool) -> Result<()> {
        let hp = self.w.hp.clone();
        let e = hp.n_embd;
        let s = self.s();
        // SAFETY: tkg C ABI.
        unsafe {
            ck(
                (self.lib.g.embed)(
                    self.w.tok_embd.ty as i32,
                    self.embed_table(),
                    self.embed_ids(),
                    self.a.y.f(),
                    t as i32,
                    e as i32,
                    s,
                ),
                "tkg_embed",
            )?;
        }
        // rb-num GLM_HC_BF16: 16-bit streams for prompt windows only (the streams live inside one window: written by
        // the first post-mix, read last by the head); decode / verify windows and layer dumps keep fp32
        self.hc16_win = if t > TMAX && self.dump.is_none() {
            self.hc16
        } else {
            0
        };
        let layers = std::mem::take(&mut self.w.layers);
        let r = (|| -> Result<()> {
            self.pre_done = false;
            self.hc_post_mix(1, Some(&layers[0].hc_attn), Some(&layers[0].attn_norm), t)?;
            for il in 0..hp.n_layer {
                let l = &layers[il];
                if il > 0 {
                    let inj = self.dump.as_ref().and_then(|d| {
                        d.inject.get(&(il - 1)).map(|v| {
                            let n = 4 * e;
                            v[d.row0 * n..(d.row0 + t) * n].to_vec()
                        })
                    });
                    if let Some(v) = inj {
                        ffi::sync(self.stream)?;
                        ffi::h2d(self.a.xhc.v(), v.as_ptr() as *const c_void, v.len() * 4)?;
                        self.hc_post_mix(2, Some(&l.hc_attn), None, t)?;
                    }
                }
                self.hc_pre(&l.hc_attn, &l.attn_norm, t)?;
                self.dump_dev("attn_norm", il, self.a.y.f(), t * e)?;
                self.attn_block(l, il, t, k_snap)?;
                self.dump_dev("blk_out", il, self.a.blk_out.f(), t * e)?;
                self.hc_post_mix(0, Some(&l.hc_ffn), Some(&l.ffn_norm), t)?;
                self.dump_dev("hc_attn_post", il, self.a.xhc.f(), t * 4 * e)?;
                self.hc_pre(&l.hc_ffn, &l.ffn_norm, t)?;
                self.dump_dev("ffn_norm", il, self.a.y.f(), t * e)?;
                self.ffn_block(l, il, t)?;
                let next_l = layers.get(il + 1).filter(|_| il + 1 < hp.n_layer);
                let next_hc = next_l.map(|n| &n.hc_attn);
                let next_norm = next_l.map(|n| &n.attn_norm);
                if l.moe && self.fuse_moe_post && self.dump.is_none() {
                    self.hc_post_mix_moe(next_hc, next_norm, t)?;
                } else {
                    self.dump_dev("ffn_out", il, self.a.blk_out.f(), t * e)?;
                    self.hc_post_mix(0, next_hc, next_norm, t)?;
                }
                self.dump_dev("l_out", il, self.a.xhc.f(), t * 4 * e)?;
            }
            Ok(())
        })();
        self.w.layers = layers;
        r?;
        if self.hc16_win != 0 {
            let f = self.lib.g.hc_head16.unwrap();
            // SAFETY: tkg C ABI (a.xhc holds 16-bit streams in this window).
            ck(
                unsafe {
                    f(
                        self.a.xhc.v(),
                        self.w.output_norm.f(),
                        self.a.h.f(),
                        t as i32,
                        e as i32,
                        hp.rms_eps,
                        self.hc16_win,
                        s,
                    )
                },
                "tkg_hc_head16",
            )?;
            self.hc16_win = 0;
        } else {
            // SAFETY: tkg C ABI.
            unsafe {
                ck(
                    (self.lib.g.hc_head)(
                        self.a.xhc.f(),
                        self.w.output_norm.f(),
                        self.a.h.f(),
                        t as i32,
                        e as i32,
                        hp.rms_eps,
                        s,
                    ),
                    "tkg_hc_head",
                )?;
            }
        }
        self.dump_dev("result_norm", 0, self.a.h.f(), t * e)?;
        if want_logits {
            let out = self.w.output;
            if t <= TMAX {
                self.quant(self.a.h.f(), self.a.hq.v(), e, t)?;
                self.mmvq(&out, self.a.hq.v(), self.logits_dev.f(), t)?;
            } else {
                // prefill chunk: the last row only (row 0 of the logits buffer)
                // SAFETY: row t-1 inside the activation buffer.
                let hrow = unsafe { self.a.h.f().add((t - 1) * e) };
                self.quant(hrow, self.a.hq.v(), e, 1)?;
                self.mmvq(&out, self.a.hq.v(), self.logits_dev.f(), 1)?;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------------------
    // public API

    /// Zero the recurrent state (KDA + conv, every slot); caches need no clearing (rows are only read below pos).
    pub fn reset(&mut self) -> Result<()> {
        ffi::sync(self.stream)?;
        for b in self.st.kda_state.iter().chain(self.st.conv_state.iter()) {
            ffi::memset(b.p, 0, b.bytes)?;
        }
        self.st.cur_slot = 0;
        Ok(())
    }

    pub fn n_vocab(&self) -> usize {
        self.w.hp.n_vocab
    }

    /// Run `toks` at positions pos0.. through the trunk in windows of TMAX (prefill / teacher forcing).
    /// Returns the logits of every position if `all_logits`, else of the last one.
    pub fn eval(&mut self, toks: &[u32], pos0: usize, all_logits: bool) -> Result<Vec<f32>> {
        self.eval_last(toks, pos0, if all_logits { toks.len() } else { 1 })
    }

    /// Like `eval`, returning the logits of the last `n_last` positions.
    pub fn eval_last(&mut self, toks: &[u32], pos0: usize, n_last: usize) -> Result<Vec<f32>> {
        let nv = self.w.hp.n_vocab;
        let first_row = toks.len().saturating_sub(n_last.max(1));
        let mut out = Vec::with_capacity((toks.len() - first_row) * nv);
        let mut i = 0;
        while i < toks.len() {
            // prefill chunks aligned to PCH; decode-size windows where several logits rows are wanted
            let pos = pos0 + i;
            let left = toks.len() - i;
            let mut t = left.min(pch() - pos % pch());
            if t < 64 {
                t = t.min(TMAX - pos % TMAX);
            }
            if toks.len() - first_row > 1 {
                if i >= first_row {
                    t = left.min(TMAX - pos % TMAX);
                } else if i + t > first_row {
                    t = first_row - i;
                }
            }
            let cur = self.st.cur_slot as i32;
            self.set_window(&toks[i..i + t], pos0 + i, &[cur])?;
            if let Some(d) = self.dump.as_mut() {
                d.row0 = i;
            }
            let want = i + t > first_row;
            self.run(GKey {
                mtp: false,
                t: t as u16,
                k: 1,
                want,
                bucket: self.n_pool_bucket(t) as u32,
            })?;
            if want {
                // big windows return their last row only (in row 0)
                let (first, rows) = if t <= TMAX {
                    (first_row.saturating_sub(i), t - first_row.saturating_sub(i))
                } else {
                    (0, 1)
                };
                // SAFETY: device buffer holds t rows; pinned buffer TMAX rows.
                unsafe {
                    ffi::ckh(
                        hip_sys::hipMemcpyAsync(
                            self.logits_pin as *mut c_void,
                            (self.logits_dev.f()).add(first * nv) as *const c_void,
                            rows * nv * 4,
                            hip_sys::hipMemcpyDeviceToHost,
                            self.stream,
                        ),
                        "logits D2H",
                    )?;
                }
                ffi::sync(self.stream)?;
                // SAFETY: pinned buffer filled with rows * nv floats.
                out.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(self.logits_pin, rows * nv)
                });
            }
            i += t;
        }
        ffi::sync(self.stream)?;
        Ok(out)
    }

    // ------------------------------------------------------------------------------------------------------------
    // speculative decoding building blocks

    /// Build the draft-vocabulary head (rows of output.weight for `ids`), as llama.cpp `--spec-draft-vocab`.
    pub fn set_draft_vocab(&mut self, ids: &[u32]) -> Result<()> {
        let out = self.w.output;
        let row_bytes = out.bytes / out.ne[1] as usize;
        let n = ids.len();
        let dst = dev_alloc(n * row_bytes)?;
        self.allocs.push(dst);
        let idd = dev_alloc(n * 4)?;
        let iv: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        ffi::h2d(idd, iv.as_ptr() as *const c_void, n * 4)?;
        // SAFETY: tkg C ABI.
        let rc = unsafe {
            (self.lib.g.gather_rows)(
                out.p(),
                row_bytes,
                idd as *const i32,
                n as i32,
                dst,
                self.s(),
            )
        };
        ffi::sync(self.stream)?;
        ck(rc, "tkg_gather_rows")?;
        if !self.dv_ids_dev.is_null() {
            ffi::dev_free(self.dv_ids_dev as *mut c_void);
        }
        self.dv_ids_dev = idd as *mut i32;
        if !self.dv_embd.is_null() {
            ffi::dev_free(self.dv_embd);
            self.dv_embd = ptr::null_mut();
        }
        if self.dev_greedy && self.dev_chain {
            if let Some(er) = self.w.tok_embd_rows.as_ref() {
                // embedding rows of the draft vocabulary (device greedy chain: drafted tokens are embedded from here)
                let rb = er.row_bytes;
                let mut host = vec![0u8; n * rb];
                for (i, &tk) in ids.iter().enumerate() {
                    er.read_row(tk as usize, &mut host[i * rb..(i + 1) * rb])?;
                }
                let d = dev_alloc(n * rb)?;
                ffi::h2d(d, host.as_ptr() as *const c_void, n * rb)?;
                self.dv_embd = d;
            }
        }
        let mut t = out;
        t.ptr = dst;
        t.ne[1] = n as u64;
        t.bytes = n * row_bytes;
        self.dv_head = Some(t);
        self.dv_ids = ids.to_vec();
        Ok(())
    }

    /// the device greedy chain is usable (draft vocabulary set; its embedding rows on the device in host-row
    /// mode).
    pub fn dev_greedy_ok(&self) -> bool {
        self.dev_greedy_plain_ok() && self.dv_head.is_some() && !self.dv_ids_dev.is_null()
    }

    /// drafted tokens can be embedded on the device (no host round trip inside a greedy MTP round).
    pub fn dev_chain_ok(&self) -> bool {
        self.dev_greedy_ok() && (self.w.tok_embd_rows.is_none() || !self.dv_embd.is_null())
    }

    /// plain greedy decode can take the argmax on the device (GLM_DEV_GREEDY=0: host argmax).
    pub fn dev_greedy_plain_ok(&self) -> bool {
        self.dev_greedy && self.dump.is_none()
    }

    /// Window row `row` takes the drafted token of greedy slot `slot` (0 / 1), stream-ordered after set_window.
    fn place_dev_tok(&mut self, row: usize, slot: usize) -> Result<()> {
        // SAFETY: slot < 16, row < TMAX; tkg C ABI.
        unsafe {
            if self.w.tok_embd_rows.is_some() {
                let rb = self.c.row_bytes;
                ck(
                    (self.lib.g.gather_rows)(
                        self.dv_embd,
                        rb,
                        self.gdev.i().add(16 + slot) as *const i32,
                        1,
                        (self.c.rows_dev as *mut u8).add(row * rb) as *mut c_void,
                        self.s(),
                    ),
                    "tkg_gather_rows dv",
                )
            } else {
                ffi::d2d_async(
                    self.c.tok.add(row) as *mut c_void,
                    self.gdev.i().add(slot) as *const c_void,
                    4,
                    self.stream,
                )
            }
        }
    }

    /// Greedy argmax of `rows` rows (stride ld) into greedy slots slot0.. (tokens via `map` when given).
    fn argmax_to(
        &self,
        x: *const f32,
        n: usize,
        rows: usize,
        ld: usize,
        map: *const i32,
        slot0: usize,
    ) -> Result<()> {
        // SAFETY: slot0 + rows <= 16; tkg C ABI.
        unsafe {
            ck(
                (self.lib.g.argmax_rows)(
                    x,
                    n as i32,
                    rows as i32,
                    ld as std::os::raw::c_long,
                    map,
                    self.gdev.i().add(slot0),
                    self.gdev.i().add(16 + slot0),
                    self.s(),
                ),
                "tkg_argmax_rows",
            )
        }
    }

    /// MTP window for the device greedy chain: rows `dev` = (row, slot) take drafted tokens from greedy slots;
    /// the draft argmax lands in greedy slot `out`. `sync`: copy the greedy slots to the host and wait (else no sync).
    pub fn mtp_greedy(
        &mut self,
        toks: &[u32],
        dev: &[(usize, usize)],
        pos0: usize,
        out: usize,
        sync: bool,
    ) -> Result<()> {
        let t = toks.len();
        if self.w.layers.len() <= self.w.hp.n_layer {
            return Err(CoreError::InvalidArgument {
                reason: "MTP block not loaded".into(),
            });
        }
        self.set_window(toks, pos0, &[])?;
        for &(r, sl) in dev {
            self.place_dev_tok(r, sl)?;
        }
        self.run(GKey {
            mtp: true,
            t: t as u16,
            k: 0,
            want: true,
            bucket: self.n_pool_bucket(t) as u32,
        })?;
        let n = self
            .dv_head
            .map(|h| h.ne[1] as usize)
            .unwrap_or(self.w.hp.n_vocab);
        self.argmax_to(self.dlog_dev.f(), n, 1, n, self.dv_ids_dev, out)?;
        if sync {
            self.greedy_d2h()?;
        }
        Ok(())
    }

    fn greedy_d2h(&mut self) -> Result<()> {
        // SAFETY: 32 ints device and pinned.
        unsafe {
            ffi::ckh(
                hip_sys::hipMemcpyAsync(
                    self.gpin as *mut c_void,
                    self.gdev.f() as *const c_void,
                    32 * 4,
                    hip_sys::hipMemcpyDeviceToHost,
                    self.stream,
                ),
                "greedy D2H",
            )?;
        }
        ffi::sync(self.stream)
    }

    /// trunk window for the device greedy chain: rows `dev` take drafted tokens from greedy slots, the argmax
    /// of every row lands in greedy slots slot0..slot0 + t; then the greedy slots are copied to the host (one sync).
    /// Host logits are not filled.
    pub fn trunk_greedy(
        &mut self,
        toks: &[u32],
        dev: &[(usize, usize)],
        pos0: usize,
        kda_ids: &[i32],
        k_snap: usize,
        slot0: usize,
    ) -> Result<()> {
        let t = toks.len();
        self.set_window(toks, pos0, kda_ids)?;
        for &(r, sl) in dev {
            self.place_dev_tok(r, sl)?;
        }
        self.run(GKey {
            mtp: false,
            t: t as u16,
            k: k_snap as u8,
            want: true,
            bucket: self.n_pool_bucket(t) as u32,
        })?;
        let nv = self.w.hp.n_vocab;
        self.argmax_to(self.logits_dev.f(), nv, t, nv, ptr::null(), slot0)?;
        self.greedy_d2h()
    }

    /// Token of greedy slot `i` (valid after `trunk_greedy`).
    pub fn greedy_tok(&self, i: usize) -> u32 {
        // SAFETY: i < 16, pinned buffer of 32 ints.
        unsafe { *self.gpin.add(i) as u32 }
    }

    /// Trunk window with rollback snapshots: toks at pos0.., KDA reads kda_ids[0], the last k_snap tokens write
    /// kda_ids[..k_snap]. Logits of all rows land in the host buffer (`logits_row`).
    pub fn trunk(
        &mut self,
        toks: &[u32],
        pos0: usize,
        kda_ids: &[i32],
        k_snap: usize,
    ) -> Result<()> {
        let t = toks.len();
        self.set_window(toks, pos0, kda_ids)?;
        self.run(GKey {
            mtp: false,
            t: t as u16,
            k: k_snap as u8,
            want: true,
            bucket: self.n_pool_bucket(t) as u32,
        })?;
        let nv = self.w.hp.n_vocab;
        // SAFETY: t rows in the device buffer, pinned buffer TMAX rows.
        unsafe {
            ffi::ckh(
                hip_sys::hipMemcpyAsync(
                    self.logits_pin as *mut c_void,
                    self.logits_dev.f() as *const c_void,
                    if t <= TMAX { t } else { 1 } * nv * 4,
                    hip_sys::hipMemcpyDeviceToHost,
                    self.stream,
                ),
                "logits D2H",
            )?;
        }
        ffi::sync(self.stream)
    }

    pub fn logits_row(&self, r: usize) -> &[f32] {
        let nv = self.w.hp.n_vocab;
        // SAFETY: pinned buffer holds TMAX rows.
        unsafe { std::slice::from_raw_parts(self.logits_pin.add(r * nv), nv) }
    }

    pub fn dlogits_row(&self, r: usize) -> &[f32] {
        let n = self
            .dv_head
            .map(|t| t.ne[1] as usize)
            .unwrap_or(self.w.hp.n_vocab);
        // SAFETY: pinned buffer holds TMAX rows of n_vocab >= n.
        unsafe { std::slice::from_raw_parts(self.dlog_pin.add(r * n), n) }
    }

    /// MTP input row `dst` of the next MTP window := trunk h of window row `src` (or MTP h if `from_mtp`), or zeros.
    pub fn set_hin(&mut self, dst: usize, src: Option<(usize, bool)>) -> Result<()> {
        let e = self.w.hp.n_embd;
        // SAFETY: rows inside TMAX.
        let d = unsafe { self.a.hin.f().add(dst * e) } as *mut c_void;
        match src {
            None => ffi::memset_async(d, 0, e * 4, self.stream),
            Some((r, from_mtp)) => {
                let b = if from_mtp { self.a.hout } else { self.a.h };
                // SAFETY: rows inside TMAX.
                let sp = unsafe { b.f().add(r * e) } as *const c_void;
                ffi::d2d_async(d, sp, e * 4, self.stream)
            }
        }
    }

    /// Save / restore one h row (host copy) across windows (e.g. the last trunk h of a prompt).
    pub fn get_h_row(&self, r: usize, from_mtp: bool) -> Result<Vec<f32>> {
        let e = self.w.hp.n_embd;
        ffi::sync(self.stream)?;
        let mut v = vec![0f32; e];
        let b = if from_mtp { self.a.hout } else { self.a.h };
        // SAFETY: row inside TMAX.
        ffi::d2h(
            v.as_mut_ptr() as *mut c_void,
            unsafe { b.f().add(r * e) } as *const c_void,
            e * 4,
        )?;
        Ok(v)
    }

    pub fn put_hin_row(&mut self, dst: usize, v: &[f32]) -> Result<()> {
        let e = self.w.hp.n_embd;
        ffi::sync(self.stream)?;
        // SAFETY: row inside TMAX.
        ffi::h2d(
            unsafe { self.a.hin.f().add(dst * e) } as *mut c_void,
            v.as_ptr() as *const c_void,
            e * 4,
        )
    }

    /// MTP (NextN) window: token toks[i] at pos0 + i paired with hin row i. Writes the MTP KV / indexer caches,
    /// leaves h_mtp in hout and the draft logits of the last row (draft vocabulary if set) in `dlogits_row(0)`.
    pub fn mtp(&mut self, toks: &[u32], pos0: usize, want_logits: bool) -> Result<()> {
        let hp = self.w.hp.clone();
        let t = toks.len();
        let il = hp.n_layer;
        if self.w.layers.len() <= il {
            return Err(CoreError::InvalidArgument {
                reason: "MTP block not loaded".into(),
            });
        }
        self.set_window(toks, pos0, &[])?;
        self.run(GKey {
            mtp: true,
            t: t as u16,
            k: 0,
            want: want_logits,
            bucket: self.n_pool_bucket(t) as u32,
        })?;
        if want_logits {
            let n = self.dv_head.map(|h| h.ne[1] as usize).unwrap_or(hp.n_vocab);
            // SAFETY: device buffer holds n floats; pinned too.
            unsafe {
                ffi::ckh(
                    hip_sys::hipMemcpyAsync(
                        self.dlog_pin as *mut c_void,
                        self.dlog_dev.f() as *const c_void,
                        n * 4,
                        hip_sys::hipMemcpyDeviceToHost,
                        self.stream,
                    ),
                    "dlogits D2H",
                )?;
            }
        }
        ffi::sync(self.stream)
    }

    /// MTP block body (device work only; graph-capturable).
    fn mtp_body(&mut self, t: usize, want_logits: bool) -> Result<()> {
        let hp = self.w.hp.clone();
        let e = hp.n_embd;
        let il = hp.n_layer;
        let s = self.s();
        let layers = std::mem::take(&mut self.w.layers);
        let r = (|| -> Result<()> {
            let l = &layers[il];
            let a = self.a;
            // SAFETY: tkg C ABI.
            unsafe {
                ck(
                    (self.lib.g.embed)(
                        self.w.tok_embd.ty as i32,
                        self.embed_table(),
                        self.embed_ids(),
                        a.memb.f(),
                        t as i32,
                        e as i32,
                        s,
                    ),
                    "tkg_embed",
                )?;
                // [enorm(e) | hnorm(h)] rows of 2*e
                ck(
                    (self.lib.g.rms_norm)(
                        a.memb.f(),
                        l.enorm.f(),
                        a.mcat.f(),
                        t as i32,
                        e as i32,
                        e as i32,
                        (2 * e) as i32,
                        hp.rms_eps,
                        s,
                    ),
                    "enorm",
                )?;
                ck(
                    (self.lib.g.rms_norm)(
                        a.hin.f(),
                        l.hnorm.f(),
                        a.mcat.f().add(e),
                        t as i32,
                        e as i32,
                        e as i32,
                        (2 * e) as i32,
                        hp.rms_eps,
                        s,
                    ),
                    "hnorm",
                )?;
            }
            if t <= TMAX {
                self.quant(a.mcat.f(), a.mcatq.v(), 2 * e, t)?;
            }
            self.proj(&l.eh_proj, a.mcat.f(), a.mcatq.v(), a.mres.f(), t)?;
            self.rms(a.mres.f(), l.attn_norm.f(), a.y.f(), t, e, hp.rms_eps)?;
            if t <= TMAX {
                self.quant(a.y.f(), a.yq.v(), e, t)?;
            }
            if !want_logits && self.mtp_kv_only && self.dump.is_none() {
                // Prompt pass: only the MTP caches are kept
                self.kv_only = true;
                let r = self.attn_block(l, il, t, 1);
                self.kv_only = false;
                return r;
            }
            self.attn_block(l, il, t, 1)?;
            // SAFETY: tkg C ABI.
            unsafe {
                ck(
                    (self.lib.g.add)(a.mres.f(), a.blk_out.f(), (t * e) as i32, s),
                    "add",
                )?;
            }
            self.rms(a.mres.f(), l.ffn_norm.f(), a.y.f(), t, e, hp.rms_eps)?;
            if t <= TMAX {
                self.quant(a.y.f(), a.yq.v(), e, t)?;
            }
            self.ffn_block(l, il, t)?;
            // SAFETY: tkg C ABI.
            unsafe {
                ck(
                    (self.lib.g.add)(a.blk_out.f(), a.mres.f(), (t * e) as i32, s),
                    "add",
                )?;
            }
            self.rms(a.blk_out.f(), l.head_norm.f(), a.hout.f(), t, e, hp.rms_eps)?;
            if want_logits {
                // SAFETY: row t-1 inside TMAX.
                let hrow = unsafe { a.hout.f().add((t - 1) * e) };
                self.quant(hrow, a.hq.v(), e, 1)?;
                let head = self.dv_head.unwrap_or(self.w.output);
                self.mmvq(&head, a.hq.v(), self.dlog_dev.f(), 1)?;
            }
            Ok(())
        })();
        self.w.layers = layers;
        r
    }

    fn exec(&mut self, k: GKey) -> Result<()> {
        if k.mtp {
            self.mtp_body(k.t as usize, k.want)
        } else {
            self.forward_window(k.t as usize, k.k as usize, k.want)
        }
    }

    /// Run a window: eagerly the first time a key is seen (lazy kernel init), from a captured graph afterwards.
    fn run(&mut self, k: GKey) -> Result<()> {
        // prefill chunks run eagerly (MMQ grids sized from routing read back on the host)
        if self.dump.is_some() || !self.use_graphs || k.t as usize > TMAX {
            return self.exec(k);
        }
        self.graph_clock += 1;
        if let Some(&g) = self.graphs.get(&k) {
            self.graph_used.insert(k, self.graph_clock);
            return ffi::graph_launch(g, self.stream);
        }
        if self.seen.insert(k) {
            return self.exec(k);
        }
        let max_graphs = std::env::var("GLM_MAX_GRAPHS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(32);
        while self.graphs.len() >= max_graphs.max(1) {
            let old = *self
                .graph_used
                .iter()
                .min_by_key(|(_, &u)| u)
                .map(|(k, _)| k)
                .unwrap();
            ffi::sync(self.stream)?; // no launch of it may still be in flight
            if let Some(g) = self.graphs.remove(&old) {
                ffi::graph_destroy(g);
            }
            self.graph_used.remove(&old);
            self.seen.remove(&old);
        }
        let s = self.stream;
        let g = ffi::capture(s, || self.exec(k))?;
        self.graphs.insert(k, g);
        self.graph_used.insert(k, self.graph_clock);
        ffi::graph_launch(g, s)
    }

    pub fn n_graphs(&self) -> usize {
        self.graphs.len()
    }

    // ------------------------------------------------------------------------------------------------------------
    // recurrent-state snapshots (prompt cache)

    /// Bytes of one snapshot (KDA + conv state of every layer, one slot).
    pub fn snap_bytes(&self) -> usize {
        self.st
            .kda_state
            .iter()
            .chain(self.st.conv_state.iter())
            .map(|b| b.bytes / kda_slots())
            .sum()
    }

    /// Allocate a snapshot buffer (caller accounts it in the Budget).
    pub fn snap_alloc(&mut self) -> Result<Buf> {
        let n = self.snap_bytes();
        let p = dev_alloc(n)?;
        self.allocs.push(p);
        Ok(Buf { p, bytes: n })
    }

    /// Copy one KDA layer's current recurrent state (KDA + conv, slot cur) into its place in a snapshot buffer.
    fn snap_layer(&mut self, il: usize, dst: &Buf) -> Result<()> {
        let ki = self.kda_index[il] as usize;
        let nk = self.st.kda_state.len();
        let cur = self.st.cur_slot;
        let kn = self.st.kda_state[ki].bytes / kda_slots();
        let cn = self.st.conv_state[ki].bytes / kda_slots();
        ffi::d2d_async(
            dst.at(ki * kn),
            self.st.kda_state[ki].at(cur * kn),
            kn,
            self.stream,
        )?;
        ffi::d2d_async(
            dst.at(nk * kn + ki * cn),
            self.st.conv_state[ki].at(cur * cn),
            cn,
            self.stream,
        )
    }

    /// Copy the current recurrent state (slot cur) into `dst` (stream-ordered).
    pub fn snap_save(&mut self, dst: &Buf) -> Result<()> {
        let mut off = 0usize;
        let cur = self.st.cur_slot;
        for b in self.st.kda_state.iter().chain(self.st.conv_state.iter()) {
            let n = b.bytes / kda_slots();
            ffi::d2d_async(dst.at(off), b.at(cur * n), n, self.stream)?;
            off += n;
        }
        Ok(())
    }

    /// Restore the recurrent state from `src` into slot 0 and make it current.
    pub fn snap_restore(&mut self, src: &Buf) -> Result<()> {
        let mut off = 0usize;
        for b in self.st.kda_state.iter().chain(self.st.conv_state.iter()) {
            let n = b.bytes / kda_slots();
            ffi::d2d_async(b.at(0), src.at(off), n, self.stream)?;
            off += n;
        }
        self.st.cur_slot = 0;
        ffi::sync(self.stream)
    }
}
