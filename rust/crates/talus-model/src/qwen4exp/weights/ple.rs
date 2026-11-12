//! Piece 1 (`qwen4exp-p1`): the PLE weights wiring and the PLE table's I/O options.

#![allow(clippy::missing_safety_doc)]

use super::*;

// =====================================================================================================================
// The PLE weights (layer 1's blk.1.ple_* tensors wired into the PLE block, generate.cpp "THE PLE")
// =====================================================================================================================

/// `strata::kernels::PleWeights` (the weight part: the six `blk.1.ple_*` tensors; `pre_key` / `pre_value` are the
/// batch path's per-window inputs, not weights).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpP1PleWeights {
    pub key_codes: *const u8,
    pub key_scales: *const f32,
    pub value_bf16: *const u16,
    pub norm_key: *const f32,
    pub norm_query: *const f32,
    pub norm_conv: *const f32,
    pub conv1d_f16: *const u16,
    pub key_native_data: *const c_void,
    pub key_native_type: i32,
    pub key_native_q8_1: *mut c_void,
    pub key_bf16: *const u16,
}

impl Default for TkQwen4ExpP1PleWeights {
    fn default() -> Self {
        Self {
            key_codes: std::ptr::null(),
            key_scales: std::ptr::null(),
            value_bf16: std::ptr::null(),
            norm_key: std::ptr::null(),
            norm_query: std::ptr::null(),
            norm_conv: std::ptr::null(),
            conv1d_f16: std::ptr::null(),
            key_native_data: std::ptr::null(),
            key_native_type: -1,
            key_native_q8_1: std::ptr::null_mut(),
            key_bf16: std::ptr::null(),
        }
    }
}

unsafe impl Send for TkQwen4ExpP1PleWeights {}
unsafe impl Sync for TkQwen4ExpP1PleWeights {}

/// The PLE wiring: which planes of which tensors the PLE block reads, checked (a missing tensor is a different
/// model, LEDGER L123; a conv1d that is not F16 would be read as the low halves of f32 words).
pub fn ple_wire(
    table: &WeightTable,
    native: Option<&NativeDense>,
    native_ple_key: bool,
) -> Result<TkQwen4ExpP1PleWeights, String> {
    let (Some(wk), Some(wv), Some(wnk), Some(wnq), Some(wnc), Some(wc)) = (
        table.find("blk.1.ple_key.weight"),
        table.find("blk.1.ple_value.weight"),
        table.find("blk.1.ple_norm_key.weight"),
        table.find("blk.1.ple_norm_query.weight"),
        table.find("blk.1.ple_norm_conv.weight"),
        table.find("blk.1.ple_conv1d.weight"),
    ) else {
        return Err("the pack has no blk.1.ple_* tensors, so the PLE cannot be wired - and running without it is a \
                    DIFFERENT MODEL (LEDGER L123)"
            .into());
    };
    let mut w = TkQwen4ExpP1PleWeights::default();
    if !wk.quantized() {
        w.key_bf16 = wk.data.cast();
    } else if !wk.data.is_null() {
        w.key_codes = wk.data.cast();
        w.key_scales = unsafe { wk.data.cast::<u8>().add(wk.codes_bytes as usize) }.cast();
    }
    if native_ple_key && wk.quantized() {
        let a = native.and_then(|n| n.attach.get("blk.1.ple_key.weight"));
        match a {
            Some(a)
                if a.native_data != 0
                    && matches!(a.native_type, 42 | 18 | 23 | 8)
                    && a.native_q8_1 != 0 =>
            {
                w.key_native_data = a.native_data as *const c_void;
                w.key_native_type = a.native_type;
                w.key_native_q8_1 = a.native_q8_1 as *mut c_void;
            }
            _ => return Err("native PLE key is absent or incompatible".into()),
        }
    }
    w.value_bf16 = wv.data.cast();
    w.norm_key = wnk.data.cast();
    w.norm_query = wnq.data.cast();
    w.norm_conv = wnc.data.cast();
    let f16 =
        wc.kind == WeightKind::F16InF32 || (wc.kind == WeightKind::Verbatim && wc.code_bits == 0);
    if !f16 || wc.bytes != wc.elements as u64 * 2 {
        return Err(format!(
            "blk.1.ple_conv1d.weight is not stored as F16 (pack index kind {}, {} B for {} values); the PLE conv1d \
             kernel reads F16 - repack with tools/iq_pack.py",
            wc.kind as i32, wc.bytes, wc.elements
        ));
    }
    w.conv1d_f16 = wc.data.cast();
    Ok(w)
}

// =====================================================================================================================
// The PLE table (the n-gram rows the PLE block gathers): its I/O options and the slow-drive decision
// =====================================================================================================================

/// C `atof`: the longest numeric prefix (0 when there is none).
pub(super) fn c_atof(s: &str) -> f64 {
    let t = s.trim_start();
    let b = t.as_bytes();
    let mut end = 0;
    let mut best = 0.0;
    while end < b.len() {
        end += 1;
        if let Ok(v) = t[..end].parse::<f64>() {
            best = v;
        } else if !matches!(b[end - 1], b'+' | b'-' | b'.' | b'e' | b'E' | b'0'..=b'9') {
            break;
        }
    }
    best
}

/// `tk_qwen4exp_p1_ple_io_t`: `strata::kernels::PleIoOptions` as the engine opens the PLE table with it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TkQwen4ExpP1PleIo {
    /// 0 = PleIo::Direct, 1 = PleIo::Mmap
    pub mode: i32,
    pub lock: i32,
    pub max_inflight: u32,
    pub cache_rows: u64,
    pub io_thread: i32,
    pub keepalive_ms: f64,
    pub keepalive_window_s: f64,
    /// the caller sets STRATA_IO_THREADS=64 (not overwriting) before the open (Linux, direct reads)
    pub set_io_threads: i32,
}

/// generate.cpp's PleIoOptions: `--ple-io direct|mmap|ram`, `--ple-inflight`, `--ple-row-cache`, `--ple-sync-submit`,
/// STRATA_SSD_KEEPALIVE / _WINDOW, and the 64-thread default for direct reads.
pub fn ple_io_plan(
    ple_io: &str,
    inflight: i64,
    row_cache: i64,
    sync_submit: bool,
) -> TkQwen4ExpP1PleIo {
    let mut p = TkQwen4ExpP1PleIo {
        mode: if ple_io == "mmap" || ple_io == "ram" {
            1
        } else {
            0
        },
        lock: (ple_io == "ram") as i32,
        max_inflight: inflight as u32,
        cache_rows: row_cache as u64,
        io_thread: (!sync_submit) as i32,
        ..Default::default()
    };
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    p.keepalive_ms = env("STRATA_SSD_KEEPALIVE")
        .map(|v| c_atof(&v).clamp(0.0, 10000.0))
        .unwrap_or(100.0);
    p.keepalive_window_s = env("STRATA_SSD_KEEPALIVE_WINDOW")
        .map(|v| c_atof(&v).clamp(1.0, 86400.0))
        .unwrap_or(60.0);
    if p.mode != 0 || p.io_thread == 0 {
        p.keepalive_ms = 0.0;
    }
    p.set_io_threads = (p.mode == 0 && std::env::var_os("STRATA_IO_THREADS").is_none()) as i32;
    p
}

/// What to do after the direct-read probe: 0 nothing (fine, or no probe), 1 switch to mapped reads, 2 warn.
pub fn ple_probe_decide(rps: f64, explicit: bool) -> i32 {
    const SLOW_ROWS_PER_S: f64 = 15000.0;
    if rps > 0.0 && rps < SLOW_ROWS_PER_S {
        let asked = std::env::var("STRATA_PLE_AUTO_MMAP")
            .ok()
            .map(|v| c_atof(&v) as i64 != 0)
            .unwrap_or(false);
        return if !explicit && asked { 1 } else { 2 };
    }
    0
}

// ===========================================================================================================
// The n-gram row hash  S8; `strata::kernels::ngram_rows`, qwen4exp.cpp L1092-1124)
// ===========================================================================================================

/// `PleConsts`: the artifact's own constants, verbatim from the GGUF dump (ngram.cpp `ple_artifact_consts`), not derived:
/// `head_offsets` is its own metadata array, so deriving it would hide a mismatch; the test checks the two agree.
pub const PLE_MULT: [u64; 3] = [23703573157769, 20109073645365, 8052911324071];
pub const PLE_VOCAB: [u64; 16] = [
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
    20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
];
pub const PLE_OFFSET: [u64; 16] = [
    0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374,
    180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
];
const PLE_EOS_TOKEN_ID: i32 = 248044;

/// Sixteen row indices of one token from its two predecessors `prev`, OLDEST FIRST (negative: none). The window is built
/// once and only its prefixes are used: heads 0..8 bigrams, 8..16 trigrams. The cut is evaluated BEFORE the value is stored,
/// so the position whose predecessor was EOS is itself EOS; products wrap mod 2^64.
pub fn ngram_rows_one(token: i32, prev: [i32; 2], out: &mut [u32]) {
    let mut ctx = [i64::from(token), 0, 0];
    let mut cut = false;
    for s in 1..3 {
        let t = if cut { -1 } else { prev[2 - s] };
        cut = cut || t < 0 || t == PLE_EOS_TOKEN_ID;
        ctx[s] = i64::from(if cut { PLE_EOS_TOKEN_ID } else { t });
    }
    let mut mixed = 0u64;
    for n in 0..3 {
        let term = (ctx[n] as u64).wrapping_mul(PLE_MULT[n]);
        mixed = if n == 0 { term } else { mixed ^ term };
        if n == 0 {
            continue;
        }
        for g in 0..8 {
            let h = (n - 1) * 8 + g;
            out[h] = (mixed % PLE_VOCAB[h] + PLE_OFFSET[h]) as u32;
        }
    }
}

/// [`ngram_rows_one`] over `tokens`, sliding `prev` (`out`: 16 rows per token).
pub fn ngram_rows_advance(tokens: &[i32], prev: &mut [i32; 2], out: &mut [u32]) {
    for (tok, rows) in tokens.iter().zip(out.chunks_exact_mut(16)) {
        ngram_rows_one(*tok, *prev, rows);
        *prev = [prev[1], *tok];
    }
}

#[cfg(test)]
mod ngram_tests {
    use super::*;

    const ORACLE: &str = include_str!("../../../../../../src/kernels/ple_oracle_vectors.inc");

    fn array(name: &str) -> Vec<i64> {
        let key = format!(" {name}[");
        let at = ORACLE
            .find(&key)
            .unwrap_or_else(|| panic!("{name} missing"));
        let open = at + ORACLE[at..].find('{').unwrap();
        let close = open + ORACLE[open..].find('}').unwrap();
        ORACLE[open + 1..close]
            .split(',')
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(|v| v.parse().unwrap())
            .collect()
    }

    #[test]
    fn matches_the_oracle_vectors() {
        for case in 0..6 {
            let tokens = array(&format!("case{case}_tokens"));
            let prev = array(&format!("case{case}_prev"));
            let rows = array(&format!("case{case}_rows"));
            assert_eq!(rows.len(), tokens.len() * 16);
            for (t, tok) in tokens.iter().enumerate() {
                let mut got = [0u32; 16];
                ngram_rows_one(
                    *tok as i32,
                    [prev[t * 2] as i32, prev[t * 2 + 1] as i32],
                    &mut got,
                );
                let want: Vec<u32> = rows[t * 16..(t + 1) * 16]
                    .iter()
                    .map(|v| *v as u32)
                    .collect();
                assert_eq!(got.as_slice(), want.as_slice(), "case {case} token {t}");
            }
        }
    }

    #[test]
    fn offsets_are_the_running_sum() {
        let mut run = 0;
        for h in 0..16 {
            assert_eq!(PLE_OFFSET[h], run);
            run += PLE_VOCAB[h];
        }
    }

    #[test]
    fn advance_slides_the_window() {
        let (mut a, mut b) = ([0u32; 32], [0u32; 16]);
        let mut prev = [-1, 7];
        ngram_rows_advance(&[8, 9], &mut prev, &mut a);
        assert_eq!(prev, [8, 9]);
        ngram_rows_one(9, [7, 8], &mut b);
        assert_eq!(&a[16..], &b[..]);
    }
}
