//! Piece 1 (`qwen4exp-p1`): the native GGUF tensors (NativeDense / NativeHead / NativeEmbed).

#![allow(clippy::missing_safety_doc)]

use super::*;

// =====================================================================================================================
// The native GGUF tensors: projections (NativeDense), the output head (NativeHead), the token embedding (NativeEmbed)
// =====================================================================================================================

/// `tk_qwen4exp_p1_services_t`: the kernel-side facts and helpers the loaders need from `libtalus_kernels.so`
/// (byte counts of the GPU formats, the architecture check, the optional repacks). Launch-free except the Q8_0
/// quantizer of a resident bf16 projection.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TkQwen4ExpP1Services {
    pub native_mmvq_supported: unsafe extern "C" fn(ty: c_int) -> c_int,
    pub native_mmvq_weight_bytes: unsafe extern "C" fn(ty: c_int, n_in: c_int, n_out: c_int) -> u64,
    pub native_q8_1_bytes: unsafe extern "C" fn(n_in: c_int, ncols: c_int) -> u64,
    pub check_model: unsafe extern "C" fn(
        paths: *const *const c_char,
        n: usize,
        allow_embd: c_int,
        err: *mut c_char,
        cap: usize,
    ) -> c_int,
    pub check_shard_arch:
        unsafe extern "C" fn(path: *const c_char, err: *mut c_char, cap: usize) -> c_int,
    pub q8_0_packed_enabled: unsafe extern "C" fn() -> c_int,
    pub q8_0_packed_eligible: unsafe extern "C" fn(n_in: c_int, n_out: c_int) -> c_int,
    pub q8_0_pack_host:
        unsafe extern "C" fn(src: *const c_void, dst: *mut c_void, n_in: c_int, n_out: c_int),
    pub q8_0_packed_register:
        unsafe extern "C" fn(w: *const c_void, packed: *const c_void, n_in: c_int, n_out: c_int),
    pub hc_q8_requested: unsafe extern "C" fn() -> c_int,
    pub fused_gr_quantize_q8_0:
        unsafe extern "C" fn(bf16: *const u16, out: *mut u8, n_blocks: i64, stream: *mut c_void),
    pub hc_scratch_install: unsafe extern "C" fn(
        elems: u64,
        out4: *mut *mut c_void,
        err: *mut c_char,
        cap: usize,
    ) -> c_int,
    pub embed_type_supported: unsafe extern "C" fn(ty: c_int) -> c_int,
    pub iq_row_bytes: unsafe extern "C" fn(ty: c_int, n: i64) -> u64,
    pub q6_k_packed_enabled: unsafe extern "C" fn() -> c_int,
    pub q6_k_pack: unsafe extern "C" fn(
        w: *const c_void,
        n_in: c_int,
        n_out: c_int,
        what: *const c_char,
    ) -> c_int,
}

pub(super) fn services() -> Option<&'static TkQwen4ExpP1Services> {
    static S: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let p = *S.get_or_init(|| unsafe {
        let sym = dlsym(RTLD_DEFAULT, c"tk_qwen4exp_p1_services".as_ptr());
        if sym.is_null() {
            return 0;
        }
        let f: unsafe extern "C" fn() -> *const TkQwen4ExpP1Services = std::mem::transmute(sym);
        f() as usize
    });
    if p == 0 {
        None
    } else {
        Some(unsafe { &*(p as *const TkQwen4ExpP1Services) })
    }
}

pub(super) fn cerr(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// `strata::block_geometry` for the types a GGUF of this engine holds: (elements, bytes) per block.
pub fn block_geometry(t: u32) -> Option<(u64, u64)> {
    Some(match t {
        0 => (1, 4),
        1 | 30 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 36),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        29 => (256, 56),
        39 => (32, 17),
        42 => (64, 18),
        _ => return None,
    })
}

/// `eligible` (native_dense.cpp): the GDN / QSA / shared-expert projections, and the PLE key when asked.
pub(super) fn native_eligible(name: &str, ty: u32, include_ple_key: bool) -> bool {
    if !name.starts_with("blk.") {
        return false;
    }
    if name == "blk.1.ple_key.weight" {
        return include_ple_key && matches!(ty, 42 | 18 | 23 | 8);
    }
    const SUFFIXES: [&str; 10] = [
        ".attn_qkv.weight",
        ".attn_gate.weight",
        ".ssm_out.weight",
        ".attn_q.weight",
        ".attn_k.weight",
        ".attn_v.weight",
        ".attn_output.weight",
        ".ffn_gate_shexp.weight",
        ".ffn_up_shexp.weight",
        ".ffn_down_shexp.weight",
    ];
    SUFFIXES.iter().any(|s| name.ends_with(s))
}

/// `hc_q8_name`: the hyper-connection projections (and the final mixer's) the STRATA_HC_Q8 read takes as Q8_0.
pub(super) fn hc_q8_name(name: &str) -> bool {
    if name == "output_hc_down.weight" || name == "output_hc_up.weight" {
        return true;
    }
    if !name.starts_with("blk.") {
        return false;
    }
    const SUFFIXES: [&str; 6] = [
        ".hc_attn_down.weight",
        ".hc_attn_up.weight",
        ".hc_attn_inject.weight",
        ".hc_ffn_down.weight",
        ".hc_ffn_up.weight",
        ".hc_ffn_inject.weight",
    ];
    SUFFIXES.iter().any(|s| name.ends_with(s))
}

/// ggml's reference Q8_0 of 32-value blocks (`q8_0_of`): d = amax / 127 as fp16, codes `lround(x / d)`.
pub fn q8_0_of(x: &[f32]) -> Vec<u8> {
    let nb = x.len() / 32;
    let mut out = vec![0u8; nb * 34];
    for b in 0..nb {
        q8_0_block(&x[b * 32..b * 32 + 32], &mut out[b * 34..b * 34 + 34]);
    }
    out
}

#[inline]
pub(super) fn q8_0_block(v: &[f32], o: &mut [u8]) {
    let mut amax = 0.0f32;
    for &x in v {
        amax = amax.max(x.abs());
    }
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    o[0..2].copy_from_slice(&f16_from_f32(d).to_le_bytes());
    for j in 0..32 {
        // std::lround: half away from zero
        o[2 + j] = ((v[j] * id).round() as i64 as i8) as u8;
    }
}

/// `q8_0_of_bf16`: the same quantizer over bf16 values (blocks independent, so threaded like the C++).
pub fn q8_0_of_bf16(x: &[u16]) -> Vec<u8> {
    let nb = x.len() / 32;
    let mut out = vec![0u8; nb * 34];
    let nt = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 8);
    let per = nb.div_ceil(nt.max(1)).max(1);
    std::thread::scope(|sc| {
        for (ci, chunk) in out.chunks_mut(per * 34).enumerate() {
            let b0 = ci * per;
            sc.spawn(move || {
                let mut v = [0f32; 32];
                for (k, o) in chunk.chunks_mut(34).enumerate() {
                    let b = b0 + k;
                    for j in 0..32 {
                        v[j] = f32::from_bits((x[b * 32 + j] as u32) << 16);
                    }
                    q8_0_block(&v, o);
                }
            });
        }
    });
    out
}

impl WeightTable {
    /// `WeightTable::read_bf16_host`: a bf16 row's values read back from the pack file (raw 16-bit, or the high
    /// halves of the promoted f32).
    pub fn read_bf16_host(&self, name: &str) -> Result<Vec<u16>, String> {
        let r = self
            .find(name)
            .ok_or_else(|| format!("{name}: no such tensor"))?;
        let fname = pack_file_name(r.file_id);
        if fname.is_none() || r.kind != WeightKind::Bf16InF32 {
            return Err(format!("{name}: not a bf16 row of a pack file"));
        }
        let elems = r.elements as u64;
        let raw16 = r.src_bytes == elems * 2;
        if !raw16 && r.src_bytes != elems * 4 {
            return Err(format!("{name}: unexpected source size"));
        }
        let p = format!("{}/{}", self.pack_dir, fname.unwrap());
        let f = File::open(&p).map_err(|_| format!("cannot open {p}"))?;
        let mut out = vec![0u16; elems as usize];
        if raw16 {
            let mut b = vec![0u8; (elems * 2) as usize];
            f.read_exact_at(&mut b, r.src_off)
                .map_err(|_| format!("{name}: short read"))?;
            for i in 0..elems as usize {
                out[i] = u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]);
            }
        } else {
            let mut b = vec![0u8; (elems * 4) as usize];
            f.read_exact_at(&mut b, r.src_off)
                .map_err(|_| format!("{name}: short read"))?;
            for i in 0..elems as usize {
                out[i] = u16::from_le_bytes([b[i * 4 + 2], b[i * 4 + 3]]);
            }
        }
        Ok(out)
    }
}

/// What a native projection load attaches to one canonical row (`WeightRef::native_*`, `hc_q8`).
#[derive(Clone, Debug, Default)]
pub struct NativeAttach {
    pub native_data: usize,
    pub native_type: i32,
    pub native_q8_1: usize,
    pub hc_q8: usize,
}

/// `strata::core::NativeDense` after `load`: the device buffers (freed by the C++ object once adopted) and what each
/// row gets.
#[derive(Clone, Debug, Default)]
pub struct NativeDense {
    pub weights: Vec<usize>,
    pub packed_keys: Vec<usize>,
    pub scratch: usize,
    pub bytes: u64,
    pub allocated: u64,
    pub attach: BTreeMap<String, NativeAttach>,
}

/// `alloc_bytes`: the 2 MiB granule rule a stage's footprint is priced with.
pub(super) fn native_alloc_bytes(bytes: u64) -> u64 {
    const GRANULE: u64 = 2 << 20;
    if bytes < (1 << 20) {
        bytes
    } else {
        bytes.div_ceil(GRANULE) * GRANULE
    }
}

pub(super) unsafe fn dev_alloc_copy(src: *const u8, bytes: usize) -> Result<*mut c_void, String> {
    let mut p: *mut c_void = std::ptr::null_mut();
    if hip_sys::hipMalloc(&mut p, bytes) != hip_sys::hipSuccess || p.is_null() {
        return Err("hipMalloc failed".into());
    }
    if !src.is_null()
        && hip_sys::hipMemcpy(p, src.cast(), bytes, hip_sys::hipMemcpyHostToDevice)
            != hip_sys::hipSuccess
    {
        let _ = hip_sys::hipFree(p);
        return Err("hipMemcpy failed".into());
    }
    Ok(p)
}

pub(super) fn advise_slice(s: &[u8]) {
    if !read_ahead_enabled() || s.is_empty() {
        return;
    }
    extern "C" {
        fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    }
    const MADV_WILLNEED: c_int = 3;
    let pg = 4096usize;
    let start = s.as_ptr() as usize & !(pg - 1);
    let end = s.as_ptr() as usize + s.len();
    let mut a = start;
    while a < end {
        let n = (ADVISE_STEP as usize).min(end - a);
        unsafe {
            let _ = madvise(a as *mut c_void, n, MADV_WILLNEED);
        }
        a += ADVISE_STEP as usize;
    }
}

type GgufCache = BTreeMap<String, std::sync::Arc<talus_gguf::GgufFile>>;
static GGUF_CACHE: Mutex<GgufCache> = Mutex::new(BTreeMap::new());

/// The GGUF shards the native loads read, opened (mapped and parsed) once per path between `served_names` and the
/// projection load; `gguf_release` unmaps them once the loads are done.
pub(super) fn gguf_open(path: &str) -> Result<std::sync::Arc<talus_gguf::GgufFile>, String> {
    let mut c = GGUF_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(g) = c.get(path) {
        return Ok(g.clone());
    }
    let g = std::sync::Arc::new(talus_gguf::GgufFile::open(path).map_err(|e| e.to_string())?);
    c.insert(path.to_string(), g.clone());
    Ok(g)
}

/// Drops the cached shard mappings (the projections are in VRAM by then).
pub fn gguf_release() {
    GGUF_CACHE.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

pub(super) fn gguf_u64(g: &talus_gguf::GgufFile, key: &str) -> Option<u64> {
    g.get_u64(key)
}

/// `NativeDense::served_names`: the canonical rows a native projection serves (they are skipped from the arena).
pub fn native_served_names(
    shards: &[String],
    include_ple_key: bool,
) -> Result<HashSet<String>, String> {
    let svc = services().ok_or("native dense: the kernel services are not loaded")?;
    let mut out = HashSet::new();
    for path in shards {
        let g = gguf_open(path).map_err(|e| format!("native dense: {e}"))?;
        for t in g.tensors() {
            let ty = t.qtype.ggml_type_id();
            if native_eligible(&t.name, ty, include_ple_key)
                && unsafe { (svc.native_mmvq_supported)(ty as c_int) } != 0
                && t.n_dims == 2
            {
                out.insert(t.name.clone());
            }
        }
    }
    Ok(out)
}

/// `NativeDense::load` (one GPU: every layer): the GGUF projections uploaded one buffer per matrix, the optional Q8_0
/// repack, the STRATA_HC_Q8 hyper-connection copies, and the shared Q8_1 scratch.
pub fn native_dense_load(
    shards: &[String],
    table: &WeightTable,
    include_ple_key: bool,
) -> Result<NativeDense, String> {
    let svc = services().ok_or("native dense: the kernel services are not loaded")?;
    if shards.is_empty() {
        return Err("native dense: at least one GGUF shard is required".into());
    }
    let mut nd = NativeDense::default();
    // on any error the buffers made so far are freed
    struct Guard<'a>(&'a mut Vec<usize>, bool);
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            if !self.1 {
                for &p in self.0.iter() {
                    unsafe {
                        let _ = hip_sys::hipFree(p as *mut c_void);
                    }
                }
            }
        }
    }
    struct Pending {
        name: String,
        ty: i32,
        bytes: u64,
        data: usize,
        packed: usize,
        ne0: i64,
        ne1: i64,
    }
    let mut owned: Vec<usize> = Vec::new();
    let mut guard = Guard(&mut owned, false);
    let mut pending: Vec<Pending> = Vec::new();
    let mut seen = HashSet::new();
    let mut max_in = 0i64;
    let (mut total, mut hc_q8_bytes, mut allocated) = (0u64, 0u64, 0u64);
    let (mut split_count, mut split_tensors) = (0u64, 0u64);
    let mut split_numbers = HashSet::new();
    let mut have_architecture = false;
    let hc_q8 = unsafe { (svc.hc_q8_requested)() } != 0;
    let q8_inject = std::env::var("STRATA_HC_Q8_INJECT")
        .map(|v| v.starts_with('1'))
        .unwrap_or(false);
    let mut hc_set: HashSet<String> = HashSet::new();
    let mut ebuf = [0 as c_char; 512];
    for path in shards {
        let g = gguf_open(path).map_err(|e| format!("native dense: {e}"))?;
        let count = gguf_u64(&g, "split.count");
        let number = gguf_u64(&g, "split.no");
        let tensors = gguf_u64(&g, "split.tensors.count");
        if g.get_metadata("general.architecture").is_some()
            && number.map(|n| n == 0).unwrap_or(true)
        {
            let cp = CString::new(path.as_bytes()).unwrap_or_default();
            if unsafe { (svc.check_shard_arch)(cp.as_ptr(), ebuf.as_mut_ptr(), ebuf.len()) } != 0 {
                return Err(cerr(&ebuf));
            }
            have_architecture = true;
            if let (Some(c), Some(0), Some(t)) = (count, number, tensors) {
                if c > 1 {
                    split_count = c;
                    split_tensors = t;
                }
            }
        } else if !have_architecture
            || split_count == 0
            || count != Some(split_count)
            || number.map(|n| n == 0 || n >= split_count).unwrap_or(true)
            || tensors != Some(split_tensors)
        {
            return Err(
                "native dense: additional shard must match the architecture-validated first shard's split metadata"
                    .into(),
            );
        }
        if let Some(n) = number {
            if !split_numbers.insert(n) {
                return Err("native dense: duplicate split shard number".into());
            }
        }
        let mut offsets: Vec<u64> = g.tensors().iter().map(|t| t.offset).collect();
        offsets.sort_unstable();
        if offsets.windows(2).any(|w| w[0] == w[1]) {
            return Err("native dense: tensor payload offsets overlap".into());
        }
        for t in g.tensors() {
            let ty = t.qtype.ggml_type_id();
            let Some((be, bb)) = block_geometry(ty) else {
                return Err(format!("native dense: invalid block geometry {}", t.name));
            };
            if t.n_dims == 0 || t.dims[0] % be != 0 {
                return Err(format!("native dense: invalid block geometry {}", t.name));
            }
            let bytes = t.elements() / be * bb;
            let next = offsets.partition_point(|&o| o <= t.offset);
            if next < offsets.len() && bytes > offsets[next] - t.offset {
                return Err(format!("native dense: overlapping payload {}", t.name));
            }
            if native_eligible(&t.name, ty, include_ple_key)
                && unsafe { (svc.native_mmvq_supported)(ty as c_int) } != 0
            {
                if let Ok(d) = g.tensor_data(t) {
                    advise_slice(d);
                }
            }
        }
        for t in g.tensors() {
            let ty = t.qtype.ggml_type_id();
            if !native_eligible(&t.name, ty, include_ple_key) {
                continue;
            }
            if !seen.insert(t.name.clone()) {
                return Err(format!("native dense: duplicate tensor {}", t.name));
            }
            let Some(r) = table.find(&t.name) else {
                return Err(format!(
                    "native dense: tensor absent from canonical table: {}",
                    t.name
                ));
            };
            if unsafe { (svc.native_mmvq_supported)(ty as c_int) } == 0 {
                continue;
            }
            if t.name == "blk.1.ple_key.weight" && !r.quantized() {
                continue;
            }
            if !r.quantized()
                || t.n_dims != 2
                || r.ne0 <= 0
                || r.ne0 > i32::MAX as i64
                || r.ne1 <= 0
                || r.ne1 > i32::MAX as i64
                || t.dims[0] != r.ne0 as u64
                || t.dims[1] != r.ne1 as u64
            {
                return Err(format!("native dense: incompatible matrix {}", t.name));
            }
            let bytes = unsafe {
                (svc.native_mmvq_weight_bytes)(ty as c_int, r.ne0 as c_int, r.ne1 as c_int)
            };
            let src = g
                .tensor_data(t)
                .map_err(|e| format!("native dense upload {}: {e}", t.name))?;
            if (src.len() as u64) < bytes {
                return Err(format!("native dense upload {}: truncated payload", t.name));
            }
            let data = unsafe { dev_alloc_copy(src.as_ptr(), bytes as usize) }
                .map_err(|e| format!("native dense upload {}: {e}", t.name))?;
            guard.0.push(data as usize);
            max_in = max_in.max(r.ne0);
            total += bytes;
            allocated += native_alloc_bytes(bytes);
            let mut packed = 0usize;
            if ty == 8
                && t.name != "blk.1.ple_key.weight"
                && unsafe { (svc.q8_0_packed_enabled)() } != 0
                && unsafe { (svc.q8_0_packed_eligible)(r.ne0 as c_int, r.ne1 as c_int) } != 0
            {
                let mut host = vec![0u8; bytes as usize];
                unsafe {
                    (svc.q8_0_pack_host)(
                        src.as_ptr().cast(),
                        host.as_mut_ptr().cast(),
                        r.ne0 as c_int,
                        r.ne1 as c_int,
                    )
                };
                let pk = unsafe { dev_alloc_copy(host.as_ptr(), bytes as usize) }
                    .map_err(|e| format!("native dense packed upload {}: {e}", t.name))?;
                guard.0.push(pk as usize);
                packed = pk as usize;
            }
            pending.push(Pending {
                name: t.name.clone(),
                ty: ty as i32,
                bytes,
                data: data as usize,
                packed,
                ne0: r.ne0,
                ne1: r.ne1,
            });
        }
        if hc_q8 {
            for t in g.tensors() {
                let ty = t.qtype.ggml_type_id();
                let f32_inject = q8_inject && ty == 0 && t.name.ends_with("_inject.weight");
                if (ty != 8 && !f32_inject) || !hc_q8_name(&t.name) || t.n_dims != 2 {
                    continue;
                }
                let Some(r) = table.find(&t.name) else {
                    continue;
                };
                if hc_set.contains(&t.name) {
                    continue;
                }
                if t.dims[0] != r.ne0 as u64 || t.dims[1] != r.ne1 as u64 || r.ne0 % 32 != 0 {
                    return Err(format!(
                        "native dense (STRATA_HC_Q8): incompatible matrix {}",
                        t.name
                    ));
                }
                let bytes = (r.ne0 as u64 / 32 * 34 * r.ne1 as u64) as usize;
                let src = g
                    .tensor_data(t)
                    .map_err(|e| format!("native dense (STRATA_HC_Q8) upload {}: {e}", t.name))?;
                let q8;
                let from: *const u8 = if f32_inject {
                    let vals: Vec<f32> = src
                        .chunks_exact(4)
                        .take((r.ne0 * r.ne1) as usize)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    q8 = q8_0_of(&vals);
                    q8.as_ptr()
                } else {
                    src.as_ptr()
                };
                let p = unsafe { dev_alloc_copy(from, bytes) }
                    .map_err(|_| format!("native dense (STRATA_HC_Q8) upload {}", t.name))?;
                guard.0.push(p as usize);
                nd.weights.push(p as usize);
                nd.attach.entry(t.name.clone()).or_default().hc_q8 = p as usize;
                hc_set.insert(t.name.clone());
                hc_q8_bytes += bytes as u64;
            }
        }
    }
    if hc_q8 {
        let mut made = 0u64;
        for (name, r) in &table.table {
            if hc_set.contains(name)
                || !name.starts_with("blk.")
                || !hc_q8_name(name)
                || !(name.ends_with("_down.weight") || name.ends_with("_up.weight"))
            {
                continue;
            }
            let skipped = r.data.is_null();
            if skipped && r.resident {
                continue;
            }
            if r.ne0 % 32 != 0
                || r.elements != r.ne0 * r.ne1
                || (!skipped && r.bytes != r.elements as u64 * 2)
            {
                return Err(format!(
                    "native dense (STRATA_HC_Q8): {name} is not a bf16 matrix"
                ));
            }
            let nb = r.elements as u64 / 32;
            let bytes = (nb * 34) as usize;
            let p = unsafe { dev_alloc_copy(std::ptr::null(), bytes) }
                .map_err(|_| format!("native dense (STRATA_HC_Q8) alloc {name}"))?;
            guard.0.push(p as usize);
            nd.weights.push(p as usize);
            if skipped {
                let host = table.read_bf16_host(name)?;
                let q8 = q8_0_of_bf16(&host);
                if unsafe {
                    hip_sys::hipMemcpy(p, q8.as_ptr().cast(), bytes, hip_sys::hipMemcpyHostToDevice)
                } != hip_sys::hipSuccess
                {
                    return Err(format!("native dense (STRATA_HC_Q8) upload {name}"));
                }
            } else {
                unsafe {
                    (svc.fused_gr_quantize_q8_0)(
                        r.data.cast(),
                        p.cast(),
                        nb as i64,
                        std::ptr::null_mut(),
                    )
                };
            }
            nd.attach.entry(name.clone()).or_default().hc_q8 = p as usize;
            hc_set.insert(name.clone());
            hc_q8_bytes += bytes as u64;
            allocated += native_alloc_bytes(bytes as u64);
            made += 1;
        }
        if made > 0 {
            let ok = unsafe {
                hip_sys::hipGetLastError() == hip_sys::hipSuccess
                    && hip_sys::hipDeviceSynchronize() == hip_sys::hipSuccess
            };
            if !ok {
                return Err(
                    "native dense (STRATA_HC_Q8): quantizing the projections failed".into(),
                );
            }
        }
        // the scratches of hc_bf16_view (rows without a bf16 copy), installed now and not on first use
        let mut elems = 0u64;
        for (name, r) in &table.table {
            if hc_set.contains(name) && r.data.is_null() {
                elems = elems.max(r.elements as u64);
            }
        }
        if elems != 0 {
            let mut bufs = [std::ptr::null_mut::<c_void>(); 4];
            let rc = unsafe {
                (svc.hc_scratch_install)(elems, bufs.as_mut_ptr(), ebuf.as_mut_ptr(), ebuf.len())
            };
            if rc < 0 {
                return Err(cerr(&ebuf));
            }
            if rc == 0 {
                for b in bufs {
                    guard.0.push(b as usize);
                    nd.weights.push(b as usize);
                }
            }
        }
    }
    if pending.is_empty() {
        return Err("native dense: no supported GDN/QSA matrices in supplied shards".into());
    }
    let sb = unsafe { (svc.native_q8_1_bytes)(max_in as c_int, 1) } as usize;
    let scratch = unsafe { dev_alloc_copy(std::ptr::null(), sb) }
        .map_err(|e| format!("native dense scratch: {e}"))?;
    guard.0.push(scratch as usize);
    let mut packed_bytes = 0u64;
    let mut packed_count = 0usize;
    for item in &pending {
        let a = nd.attach.entry(item.name.clone()).or_default();
        a.native_data = item.data;
        a.native_type = item.ty;
        a.native_q8_1 = scratch as usize;
        if item.packed != 0 {
            unsafe {
                (svc.q8_0_packed_register)(
                    item.data as *const c_void,
                    item.packed as *const c_void,
                    item.ne0 as c_int,
                    item.ne1 as c_int,
                )
            };
            nd.packed_keys.push(item.data);
            packed_bytes += item.bytes;
            packed_count += 1;
            nd.weights.push(item.packed);
        }
        nd.weights.push(item.data);
    }
    if packed_count > 0 {
        eprintln!(
            "native dense: STRATA_Q8_PACKED=1 packed {packed_count} Q8_0 matrices for decode (+{:.1} MiB)",
            packed_bytes as f64 / 1048576.0
        );
    }
    nd.scratch = scratch as usize;
    nd.bytes = total + hc_q8_bytes;
    nd.allocated = allocated;
    if hc_q8 {
        eprintln!(
            "strata: STRATA_HC_Q8=1: {:.2} GiB of Q8_0 hyper-connection projections for the verify read",
            hc_q8_bytes as f64 / 1073741824.0
        );
    }
    guard.1 = true; // published: the C++ NativeDense frees them from here
    drop(guard);
    Ok(nd)
}

/// `strata::core::NativeHead` after `load`.
#[derive(Clone, Debug, Default)]
pub struct NativeHeadBuf {
    pub weights: usize,
    pub scratch: usize,
    pub bytes: u64,
    pub n_in: i32,
    pub n_out: i32,
    pub ty: i32,
    pub device: i32,
}

pub(super) fn open_model(
    shards: &[String],
    allow_embd: bool,
) -> Result<talus_gguf::GgufModel, String> {
    let svc = services().ok_or("the kernel services are not loaded")?;
    let cs: Vec<CString> = shards
        .iter()
        .map(|s| CString::new(s.as_bytes()).unwrap_or_default())
        .collect();
    let ps: Vec<*const c_char> = cs.iter().map(|c| c.as_ptr()).collect();
    let mut ebuf = [0 as c_char; 512];
    if unsafe {
        (svc.check_model)(
            ps.as_ptr(),
            ps.len(),
            allow_embd as c_int,
            ebuf.as_mut_ptr(),
            ebuf.len(),
        )
    } != 0
    {
        return Err(cerr(&ebuf));
    }
    talus_gguf::GgufModel::open_paths(shards.iter()).map_err(|e| e.to_string())
}

/// `NativeHead::load`: `output.weight` from whichever shard holds it, in VRAM with its Q8_1 scratch.
pub fn native_head_load(shards: &[String], n_in: i64, n_out: i64) -> Result<NativeHeadBuf, String> {
    let svc = services().ok_or("native head: the kernel services are not loaded")?;
    if n_in <= 0
        || n_out <= 0
        || n_in > i32::MAX as i64
        || n_out > i32::MAX as i64
        || n_in % 256 != 0
    {
        return Err(
            "native head requires positive int32 dimensions and whole 256-value rows".into(),
        );
    }
    let model = open_model(shards, false).map_err(|e| format!("native head: {e}"))?;
    let found = model.find_tensor("output.weight");
    let ok = match found {
        Some((t, _)) => {
            (unsafe { (svc.native_mmvq_supported)(t.qtype.ggml_type_id() as c_int) }) != 0
                && t.n_dims == 2
                && t.dims[0] == n_in as u64
                && t.dims[1] == n_out as u64
        }
        None => false,
    };
    if !ok {
        return Err(
            "native head: expected a natively supported output.weight with the canonical head dimensions".into(),
        );
    }
    let (t, loc) = found.unwrap();
    let ty = t.qtype.ggml_type_id() as c_int;
    let bytes = unsafe { (svc.native_mmvq_weight_bytes)(ty, n_in as c_int, n_out as c_int) };
    let g = &model.shards()[loc.shard_idx];
    let src = g
        .tensor_data(t)
        .map_err(|_| "native head: truncated output.weight payload".to_string())?;
    if (src.len() as u64) < bytes {
        return Err("native head: truncated output.weight payload".into());
    }
    let scratch = unsafe {
        dev_alloc_copy(
            std::ptr::null(),
            (svc.native_q8_1_bytes)(n_in as c_int, 1) as usize,
        )
    }
    .map_err(|e| format!("native head upload: {e}"))?;
    advise_slice(&src[..bytes as usize]);
    let weights = match unsafe { dev_alloc_copy(src.as_ptr(), bytes as usize) } {
        Ok(w) => w,
        Err(e) => {
            unsafe {
                let _ = hip_sys::hipFree(scratch);
            }
            return Err(format!("native head upload: {e}"));
        }
    };
    let mut dev: c_int = -1;
    unsafe {
        let _ = hip_sys::hipGetDevice(&mut dev);
    }
    if ty == 14 && unsafe { (svc.q6_k_packed_enabled)() } != 0 {
        unsafe {
            (svc.q6_k_pack)(
                weights,
                n_in as c_int,
                n_out as c_int,
                c"output head".as_ptr(),
            )
        };
    }
    Ok(NativeHeadBuf {
        weights: weights as usize,
        scratch: scratch as usize,
        bytes,
        n_in: n_in as i32,
        n_out: n_out as i32,
        ty,
        device: dev,
    })
}

/// `strata::core::NativeEmbed` after `load`: the token embedding in mapped pinned host memory (or VRAM).
#[derive(Clone, Debug, Default)]
pub struct NativeEmbedBuf {
    pub host: usize,
    pub dev: usize,
    pub bytes: u64,
    pub row: u64,
    pub n_embd: i64,
    pub n_vocab: i64,
    pub ty: i32,
}

/// `NativeEmbed::load`.
pub fn native_embed_load(
    shards: &[String],
    n_embd: i64,
    n_vocab: i64,
) -> Result<NativeEmbedBuf, String> {
    let svc = services().ok_or("native embedding: the kernel services are not loaded")?;
    let model = open_model(shards, true).map_err(|e| format!("native embedding: {e}"))?;
    let found = model.find_tensor("token_embd.weight");
    let ok = match found {
        Some((t, _)) => {
            t.n_dims == 2
                && t.dims[0] == n_embd as u64
                && t.dims[1] == n_vocab as u64
                && unsafe { (svc.embed_type_supported)(t.qtype.ggml_type_id() as c_int) } != 0
                && n_embd % 256 == 0
        }
        None => false,
    };
    if !ok {
        return Err("native embedding: token_embd.weight is absent, of another shape, or of a type without a GPU \
                    dequantizer"
            .into());
    }
    let (t, loc) = found.unwrap();
    let ty = t.qtype.ggml_type_id() as c_int;
    let row = unsafe { (svc.iq_row_bytes)(ty, n_embd) };
    let bytes = row * n_vocab as u64;
    let g = &model.shards()[loc.shard_idx];
    let payload_ok = block_geometry(ty as u32)
        .map(|(be, bb)| t.dims[0] % be == 0 && t.elements() / be * bb == bytes)
        .unwrap_or(false);
    let src = match g.tensor_data(t) {
        Ok(s) if payload_ok && s.len() as u64 == bytes => s,
        _ => {
            return Err(format!(
                "native embedding: token_embd.weight's payload is truncated or not {bytes} B ({})",
                shards[loc.shard_idx.min(shards.len() - 1)]
            ))
        }
    };
    advise_slice(src);
    let mut out = NativeEmbedBuf {
        host: 0,
        dev: 0,
        bytes,
        row,
        n_embd,
        n_vocab,
        ty,
    };
    let mut host: *mut c_void = std::ptr::null_mut();
    let e = unsafe {
        hip_sys::hipHostMalloc(
            &mut host,
            bytes as usize,
            hip_sys::hipHostMallocMapped | hip_sys::hipHostMallocPortable,
        )
    };
    if e != hip_sys::hipSuccess || host.is_null() {
        unsafe {
            let _ = hip_sys::hipGetLastError();
        }
        let d = unsafe { dev_alloc_copy(src.as_ptr(), bytes as usize) }.map_err(|_| {
            format!(
                "native embedding: cannot pin {} MiB, nor place it in VRAM",
                bytes >> 20
            )
        })?;
        eprintln!(
            "strata: native embedding: cannot pin {} MiB, kept in VRAM instead",
            bytes >> 20
        );
        out.dev = d as usize;
        return Ok(out);
    }
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), host.cast::<u8>(), bytes as usize) };
    let mut d: *mut c_void = std::ptr::null_mut();
    if unsafe { hip_sys::hipHostGetDevicePointer(&mut d, host, 0) } != hip_sys::hipSuccess
        || d.is_null()
    {
        unsafe {
            let _ = hip_sys::hipGetLastError();
        }
        let dd = unsafe { dev_alloc_copy(src.as_ptr(), bytes as usize) };
        unsafe {
            let _ = hip_sys::hipHostFree(host);
        }
        let dd = dd.map_err(|_| {
            "native embedding: no device alias for the mapped table, and no VRAM to copy it into".to_string()
        })?;
        eprintln!("strata: native embedding: no device alias for the mapped table, kept in VRAM");
        out.dev = dd as usize;
        return Ok(out);
    }
    out.host = host as usize;
    out.dev = d as usize;
    Ok(out)
}
