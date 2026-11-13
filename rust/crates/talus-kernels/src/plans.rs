//! Host-side launch choices for shared / Laguna kernel exports  S6). Each mirrors the library's own rule
//! (named below); the `*_v` exports compare the host's value with the library's for one release and report a
//! difference (`plan_mismatch`), so a divergence here cannot change output.

use std::sync::OnceLock;

/// C `atoi`: optional whitespace and sign, then leading digits; no digits = 0.
pub fn c_atoi(s: &str) -> i32 {
    let t = s.trim_start();
    let (neg, rest) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let mut v: i64 = 0;
    for b in rest.bytes() {
        if !b.is_ascii_digit() {
            break;
        }
        v = (v * 10 + i64::from(b - b'0')).min(i64::from(i32::MAX) + 1);
    }
    let v = if neg { -v } else { v };
    v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn env_int(name: &str, default: i32) -> i32 {
    std::env::var(name).map(|v| c_atoi(&v)).unwrap_or(default)
}

/// `moe_add_slots_variant` (laguna_kernels.cu): 1 = float4 kernel when TALUS_MOE_ADD4 is not "0...", `n_embd % 4 == 0`
/// and x / shared / y (their addresses) are 16-byte aligned; else 0. Shared by Laguna and MiMo.
pub fn moe_add_slots_variant(x: usize, shared: usize, y: usize, n_embd: i32) -> i32 {
    static V4: OnceLock<bool> = OnceLock::new();
    let v4 = *V4.get_or_init(|| {
        std::env::var_os("TALUS_MOE_ADD4")
            .map(|v| v.as_encoded_bytes().first() != Some(&b'0'))
            .unwrap_or(true)
    });
    let al = (x | shared | y).is_multiple_of(16);
    i32::from(v4 && n_embd % 4 == 0 && al)
}

/// `f32_matmul_plan` (laguna_kernels.cu), S7: kind * 1000 + stream mode * 10 + config. Windows up to 16 tokens with n_in % 64 == 0,
/// n_in <= 8192: the few-row family (TALUS_F32FEW_STREAM, TALUS_F32FEW_CFG; whether a padded copy exists stays with the rowpad
/// registry); else 2 v2 (opt-in TALUS_F32MM_V2=1), 3 the base kernel; TALUS_F32MM_OLD=1: the base kernel always.
pub fn f32_matmul_plan(n_in: i32, n_tok: i32) -> i32 {
    #[derive(Clone, Copy)]
    struct E {
        old_only: bool,
        stream_mode: i32,
        cfg: i32,
        v2: bool,
    }
    static ENV: OnceLock<E> = OnceLock::new();
    let first = |n: &str| std::env::var(n).ok().and_then(|v| v.bytes().next());
    let e = *ENV.get_or_init(|| E {
        old_only: first("TALUS_F32MM_OLD") == Some(b'1'),
        stream_mode: env_int("TALUS_F32FEW_STREAM", 1),
        cfg: env_int("TALUS_F32FEW_CFG", 5),
        v2: first("TALUS_F32MM_V2") == Some(b'1'),
    });
    f32_matmul_with(e.old_only, e.stream_mode, e.cfg, e.v2, n_in, n_tok)
}

fn f32_matmul_with(
    old_only: bool,
    stream_mode: i32,
    cfg: i32,
    v2: bool,
    n_in: i32,
    n_tok: i32,
) -> i32 {
    if !old_only && n_tok <= 16 && n_in % 64 == 0 && n_in <= 8192 {
        let m = if stream_mode >= 2 {
            2
        } else if stream_mode == 1 {
            1
        } else {
            0
        };
        return 1000 + m * 10 + if (1..=6).contains(&cfg) { cfg } else { 5 };
    }
    if v2 && !old_only && n_in % 16 == 0 {
        2000
    } else {
        3000
    }
}

/// `attn_wmma_plan` (q36_kernels.cu), S7: the 27B f16 reader's launch, red * 10000 + shape: 1..10 the prefetching reader's variant
/// (TALUS_ATTN_PF, default 11 = 5 for 3-6 rows, else 1), 100 groups of 1 tile, 200 / 300 / 400 RT 1 / 2 / 3, 500 groups of 2 tiles, 600 a launch
/// per group. Graph-frozen.
/// S7 kv8: the int8 decode / verify reader's launch (`talus_q36_kv8_plan` in q36_kv8.cu): `ng * 10 + variant`,
/// ng = the 16-row groups per chunk, ceil(T * G / 16) with G = H / HKV query heads per KV head; variant 0..=2 picks the kernel
/// instance (0 the prefetching reader, 1 the offset-operand reader, 2 the rd-kv27 reader). -1 = a launch the library refuses.
pub fn q36_kv8_plan(t: i32, h: i32, hkv: i32, variant: i32) -> i32 {
    if hkv <= 0 || h % hkv != 0 || t <= 0 || !(0..=2).contains(&variant) {
        return -1;
    }
    ((t * (h / hkv) + 15) / 16) * 10 + variant
}

pub fn q36_attn_plan(t: i32, h: i32, hkv: i32) -> i32 {
    static ENV: OnceLock<(i32, i32, i32, bool, bool)> = OnceLock::new();
    let (red, pf, rt1g, rt3, split) = *ENV.get_or_init(|| {
        let red = env_int("TALUS_ATTN_RED", 3);
        let pf = env_int("TALUS_ATTN_PF", 11);
        (
            if (0..=3).contains(&red) { red } else { 3 },
            if (0..=11).contains(&pf) { pf } else { 11 },
            env_int("TALUS_ATTN_RT1G", 1),
            std::env::var_os("STRATA_ATTN_RT3").is_some(),
            std::env::var("STRATA_ATTN_SPLIT")
                .map_or(true, |v| v.as_bytes().first() != Some(&b'0')),
        )
    });
    q36_attn_with(red, pf, rt1g != 0, rt3, split, t, h, hkv)
}

#[allow(clippy::too_many_arguments)]
fn q36_attn_with(
    red: i32,
    pf: i32,
    rt1g: bool,
    rt3: bool,
    split: bool,
    t: i32,
    h: i32,
    hkv: i32,
) -> i32 {
    let rt = (t * (h / hkv) + 15) / 16;
    let shape = if red == 3 && pf > 0 && (rt == 1 || rt1g) {
        if pf != 11 {
            pf
        } else if (3..=6).contains(&t) {
            5
        } else {
            1
        }
    } else if rt > 1 && rt1g {
        100
    } else if rt == 1 {
        200
    } else if rt == 2 {
        300
    } else if rt3 && rt == 3 {
        400
    } else if split {
        500
    } else {
        600
    };
    red * 10000 + shape
}

/// `decode_attn_plan` (laguna_kernels.cu), S7. The sum of: (nt * 10 + cap) * 1000 for the grouped verify kernel
/// (nt tokens per block, cap resident units) from LAGUNA_DEC_GRP tokens (default 2; dense ranges only, 32-key tiles, G <= 9;
/// 8 tokens per block for G <= 6 from 8 tokens, else 4; LAGUNA_DEC_GRP_NT = cap * 10 + nt); 100 for 64-key tiles
/// (LAGUNA_DEC_KT = 64); 1 for the token-major grid (more than one token; LAGUNA_DEC_TOKMAJOR, default on: full layers at
/// any width, sliding window up to 16 tokens). Graph-frozen: a captured window keeps this shape.
pub fn laguna_decode_attn_plan(
    n_head: i32,
    n_kv: i32,
    window: i32,
    n_tok: i32,
    has_sel: bool,
) -> i32 {
    #[derive(Clone, Copy)]
    struct E {
        tokmajor: i32,
        kt64: bool,
        grp_min: i32,
        grp_nt: i32,
    }
    static ENV: OnceLock<E> = OnceLock::new();
    let e = *ENV.get_or_init(|| E {
        tokmajor: env_int("LAGUNA_DEC_TOKMAJOR", 1),
        kt64: env_int("LAGUNA_DEC_KT", 0) == 64,
        grp_min: env_int("LAGUNA_DEC_GRP", 2),
        grp_nt: env_int("LAGUNA_DEC_GRP_NT", 0),
    });
    decode_attn_plan_with(
        e.tokmajor, e.kt64, e.grp_min, e.grp_nt, n_head, n_kv, window, n_tok, has_sel,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_attn_plan_with(
    tokmajor: i32,
    kt64: bool,
    grp_min: i32,
    grp_nt: i32,
    n_head: i32,
    n_kv: i32,
    window: i32,
    n_tok: i32,
    has_sel: bool,
) -> i32 {
    let g = n_head / n_kv;
    let tm = n_tok > 1 && tokmajor != 0 && (window == 0 || n_tok <= 16);
    let mut grp = 0;
    if grp_min > 0 && n_tok >= grp_min && !kt64 && g <= 9 && !has_sel {
        let nt = if grp_nt > 0 {
            if grp_nt % 10 == 8 && g <= 6 {
                8
            } else {
                4
            }
        } else if g <= 6 && n_tok >= 8 {
            8
        } else {
            4
        };
        let cap = if grp_nt >= 10 && grp_nt / 10 != 2 {
            4
        } else {
            2
        };
        grp = nt * 10 + cap;
    }
    grp * 1000 + if kt64 { 100 } else { 0 } + i32::from(tm)
}

/// `prompt_attn_variant` (laguna_kernels.cu): 100 for one query token (QT 1 / KT 32), else LAGUNA_PA_SHAPE (default 1).
pub fn laguna_prompt_attn_variant(nq: i32) -> i32 {
    static SHAPE: OnceLock<i32> = OnceLock::new();
    if nq == 1 {
        return 100;
    }
    *SHAPE.get_or_init(|| env_int("LAGUNA_PA_SHAPE", 1))
}

/// `prompt_attn_wmma_variant` (laguna_kernels.cu): 1 split decode (n_split > 1); with TALUS_LAGUNA_ATTN (default 6 =
/// 4 with conflict-free LDS rows, routed as 4; 4 is the kill switch) at least 2 and (q8 KV, a device position, more
/// than 4096 positions, or version 3 or 5): 2 (version 4+, shared tiles transposed) or 3 (shared tiles); else 4
/// (per-wave tiles).
pub fn laguna_prompt_attn_wmma_variant(
    nq: i32,
    q_pos0: i32,
    n_split: i32,
    has_ksc: bool,
    has_pos_dev: bool,
) -> i32 {
    static VER: OnceLock<i32> = OnceLock::new();
    let pv = *VER.get_or_init(|| env_int("TALUS_LAGUNA_ATTN", 6));
    wmma_variant_with(pv, nq, q_pos0, n_split, has_ksc, has_pos_dev)
}

fn wmma_variant_with(
    pv: i32,
    nq: i32,
    q_pos0: i32,
    n_split: i32,
    has_ksc: bool,
    has_pos_dev: bool,
) -> i32 {
    if n_split > 1 {
        return 1;
    }
    if pv >= 2 && (has_ksc || has_pos_dev || q_pos0 + nq > 4096 || pv == 3 || pv == 5) {
        return if pv >= 4 { 2 } else { 3 };
    }
    4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_matmul_plans() {
        let f = |n, t| f32_matmul_with(false, 1, 5, false, n, t);
        assert_eq!(f(4096, 1), 1015);
        assert_eq!(f(4096, 16), 1015);
        assert_eq!(f(4096, 17), 3000);
        assert_eq!(f(4112, 4), 3000);
        assert_eq!(f32_matmul_with(false, 0, 5, false, 4096, 1), 1005);
        assert_eq!(f32_matmul_with(false, 2, 2, false, 4096, 1), 1022);
        assert_eq!(f32_matmul_with(false, 1, 9, false, 4096, 1), 1015);
        assert_eq!(f32_matmul_with(false, 1, 5, true, 4096, 17), 2000);
        assert_eq!(f32_matmul_with(true, 1, 5, true, 4096, 1), 3000);
    }

    #[test]
    fn q36_kv8_plans() {
        // 27B: H 24, HKV 4 (G 6): T 1 -> 1 group, T 3 -> 18 rows -> 2, T 8 -> 48 -> 3, T 32 -> 192 -> 12
        assert_eq!(q36_kv8_plan(1, 24, 4, 0), 10);
        assert_eq!(q36_kv8_plan(3, 24, 4, 0), 20);
        assert_eq!(q36_kv8_plan(8, 24, 4, 2), 32);
        assert_eq!(q36_kv8_plan(32, 24, 4, 1), 121);
        assert_eq!(q36_kv8_plan(1, 24, 5, 0), -1); // H % HKV != 0
        assert_eq!(q36_kv8_plan(0, 24, 4, 0), -1);
        assert_eq!(q36_kv8_plan(1, 24, 4, 3), -1); // NG variant 3 is not a shipped reader (w2-27b-all, FAIL)
    }

    #[test]
    fn q36_attn_plans() {
        let d = |t| q36_attn_with(3, 11, true, false, true, t, 16, 8);
        assert_eq!(d(1), 30001);
        assert_eq!(d(3), 30005);
        assert_eq!(d(6), 30005);
        assert_eq!(d(8), 30001);
        assert_eq!(q36_attn_with(3, 2, true, false, true, 8, 16, 8), 30002);
        assert_eq!(q36_attn_with(3, 0, true, false, true, 16, 16, 8), 30100);
        assert_eq!(q36_attn_with(1, 11, true, false, true, 1, 16, 8), 10200);
        assert_eq!(q36_attn_with(3, 11, false, false, true, 8, 16, 8), 30001);
        assert_eq!(q36_attn_with(3, 11, false, false, true, 16, 16, 8), 30300);
        assert_eq!(q36_attn_with(3, 11, false, true, true, 24, 16, 8), 30400);
        assert_eq!(q36_attn_with(3, 11, false, false, true, 40, 16, 8), 30500);
        assert_eq!(q36_attn_with(3, 11, false, false, false, 40, 16, 8), 30600);
    }

    #[test]
    fn decode_attn_plan_shapes() {
        let f = |w, n, sel| decode_attn_plan_with(1, false, 2, 0, 48, 8, w, n, sel);
        assert_eq!(f(0, 1, false), 0); // one token: per-token kernel, split-major grid
        assert_eq!(f(0, 2, false), 42_001); // G 6, 2 tokens: 4 per block, 2 resident
        assert_eq!(f(0, 8, false), 82_001);
        assert_eq!(f(0, 8, true), 1); // sparse selection: token-major per-token
        assert_eq!(f(128, 32, false), 82_000); // sliding window past 16 tokens: no token-major grid
        assert_eq!(
            decode_attn_plan_with(1, true, 2, 0, 48, 8, 0, 8, false),
            101
        );
        assert_eq!(decode_attn_plan_with(1, false, 0, 0, 48, 8, 0, 8, false), 1);
        assert_eq!(
            decode_attn_plan_with(0, false, 2, 44, 48, 8, 0, 8, false),
            44_000
        );
        assert_eq!(
            decode_attn_plan_with(1, false, 2, 0, 72, 8, 0, 8, false),
            42_001
        ); // G 9: 4 per block
    }

    #[test]
    fn wmma_variants() {
        assert_eq!(wmma_variant_with(4, 64, 0, 4, false, false), 1);
        assert_eq!(wmma_variant_with(4, 2048, 0, 1, false, false), 4);
        assert_eq!(wmma_variant_with(4, 2048, 4000, 1, false, false), 2);
        assert_eq!(wmma_variant_with(4, 8, 0, 1, true, false), 2);
        assert_eq!(wmma_variant_with(2, 8, 0, 1, false, true), 3);
        assert_eq!(wmma_variant_with(3, 8, 0, 1, false, false), 3);
        for (nq, p0, ns, ksc, pd) in [
            (64, 0, 4, false, false),
            (2048, 0, 1, false, false),
            (2048, 4000, 1, false, false),
            (8, 0, 1, true, false),
            (8, 0, 1, false, true),
        ] {
            assert_eq!(
                wmma_variant_with(6, nq, p0, ns, ksc, pd),
                wmma_variant_with(4, nq, p0, ns, ksc, pd)
            );
        }
        assert_eq!(wmma_variant_with(5, 8, 0, 1, false, false), 2);
        assert_eq!(wmma_variant_with(1, 8, 9000, 1, true, true), 4);
    }

    #[test]
    fn add_slots_alignment() {
        let a = 0x1000usize;
        let b = 0x1004usize;
        // (env TALUS_MOE_ADD4 unset in the test process: on)
        if std::env::var_os("TALUS_MOE_ADD4").is_none() {
            assert_eq!(moe_add_slots_variant(a, a, a, 4096), 1);
            assert_eq!(moe_add_slots_variant(a, b, a, 4096), 0);
            assert_eq!(moe_add_slots_variant(a, a, a, 4094), 0);
        }
        assert_eq!(c_atoi(" 3x"), 3);
    }
}
