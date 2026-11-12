//! Launch plan of MiMo's full-attention kernel `tk_attn_k192_v128`  S6).
//!
//! The same rules, in the same order, as `mimo_attn_k192_v128_plan` in `src/kernels/cuda/mimo_kernels.cu`:
//! - prompt windows (`n_tok >= 8`, no split request, f16 KV, GQA ratio a multiple of 8): kind 1, with the
//!   `TALUS_MIMO_FA` switch and the raw `TALUS_MIMO_FA_VER` (default 7) the kernel family is picked from;
//! - otherwise decode / verify: split-K count from the explicit request (`n_split > 1`, at most 32) or, when the caller
//!   passes no request (`n_split <= 0`) and has a scratch buffer, from the KV stride (16 from 16384 cells, 8 from 2048,
//!   4 from 512, else 1) - the stride (allocation), not the live KV length, so the reduction order depends on the
//!   context size the engine was started with; then the v3 kernel (`TALUS_MIMO_ATT3`, shape `TALUS_MIMO_ATT3_SHAPE` or
//!   4:4 for one token / 2:8), else v2 (`TALUS_MIMO_ATT2`, shape `TALUS_MIMO_ATT2_FULL` or 2:8:4), when the shape is
//!   one the library instantiates and divides the GQA ratio, else v1 (4 heads per wave when the ratio allows, else 1).
//!
//! The library still computes its own plan and compares (one release): a mismatch is reported and the library's plan
//! runs, so a divergence here cannot change output.

use std::sync::OnceLock;
use talus_kernels_sys::tk_mimo_k192_plan;

/// C `atoi`: optional whitespace and sign, then leading digits; anything else stops; no digits = 0.
fn c_atoi(s: &str) -> i32 {
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

/// `sscanf(e, "%d:%d...")`: the leading integers separated by ':' (stops at the first field that fails).
fn scan_ints(s: &str, n: usize) -> Vec<i32> {
    let mut out = Vec::new();
    let mut rest = s;
    for i in 0..n {
        if i > 0 {
            match rest.strip_prefix(':') {
                Some(r) => rest = r,
                None => break,
            }
        }
        let t = rest.trim_start();
        let bytes = t.as_bytes();
        let mut j = 0;
        if j < bytes.len() && (bytes[j] == b'-' || bytes[j] == b'+') {
            j += 1;
        }
        let d0 = j;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j == d0 {
            break;
        }
        out.push(c_atoi(&t[..j]));
        rest = &t[j..];
    }
    out
}

fn env_on(name: &str) -> bool {
    // `e == nullptr || e[0] != '0'`
    match std::env::var_os(name) {
        None => true,
        Some(v) => v.as_encoded_bytes().first() != Some(&b'0'),
    }
}

struct Env {
    use_wmma: bool,
    fa_ver: i32,
    att3: bool,
    att2: bool,
    att3_shape: (i32, i32),
    att2_shape: Option<(i32, i32, i32)>,
    swa_ver: i32,
    att2_draft: bool,
    att2_swa_shape: Option<(i32, i32, i32)>,
}

fn env() -> &'static Env {
    static E: OnceLock<Env> = OnceLock::new();
    E.get_or_init(|| Env {
        use_wmma: env_on("TALUS_MIMO_FA"),
        fa_ver: std::env::var("TALUS_MIMO_FA_VER")
            .map(|v| c_atoi(&v))
            .unwrap_or(7),
        att3: env_on("TALUS_MIMO_ATT3"),
        att2: env_on("TALUS_MIMO_ATT2"),
        att3_shape: std::env::var("TALUS_MIMO_ATT3_SHAPE")
            .map(|v| {
                let f = scan_ints(&v, 2);
                (
                    f.first().copied().unwrap_or(0),
                    f.get(1).copied().unwrap_or(0),
                )
            })
            .unwrap_or((0, 0)),
        att2_shape: std::env::var("TALUS_MIMO_ATT2_FULL").ok().and_then(|v| {
            let f = scan_ints(&v, 3);
            (f.len() == 3).then(|| (f[0], f[1], f[2]))
        }),
        swa_ver: std::env::var("TALUS_MIMO_SWA_VER")
            .map(|v| c_atoi(&v))
            .unwrap_or(2),
        att2_draft: env_on("TALUS_MIMO_ATT2_DRAFT"),
        att2_swa_shape: std::env::var("TALUS_MIMO_ATT2_SWA").ok().and_then(|v| {
            let f = scan_ints(&v, 3);
            (f.len() == 3).then(|| (f[0], f[1], f[2]))
        }),
    })
}

/// v3 shapes the library instantiates (`X(4, 4) X(2, 8) X(2, 4) X(1, 4) X(1, 8)`).
const ATT3_SHAPES: [(i32, i32); 5] = [(4, 4), (2, 8), (2, 4), (1, 4), (1, 8)];
/// v2 shapes the library instantiates (`ATT2_SHAPES`).
const ATT2_SHAPES: [(i32, i32, i32); 12] = [
    (1, 4, 4),
    (1, 8, 4),
    (2, 4, 4),
    (2, 8, 4),
    (4, 4, 4),
    (1, 4, 2),
    (2, 8, 2),
    (2, 4, 2),
    (1, 4, 8),
    (2, 8, 8),
    (2, 4, 8),
    (4, 4, 2),
];

/// The plan for these arguments (see the module doc). `has_scratch`: a non-null split-K scratch buffer.
pub fn k192_plan(
    q8_kv: bool,
    has_scratch: bool,
    n_tok: i32,
    n_head: i32,
    n_kv: i32,
    stride: i32,
    n_split: i32,
) -> tk_mimo_k192_plan {
    plan_with(
        env(),
        q8_kv,
        has_scratch,
        n_tok,
        n_head,
        n_kv,
        stride,
        n_split,
    )
}

#[allow(clippy::too_many_arguments)]
fn plan_with(
    e: &Env,
    q8_kv: bool,
    has_scratch: bool,
    n_tok: i32,
    n_head: i32,
    n_kv: i32,
    stride: i32,
    n_split: i32,
) -> tk_mimo_k192_plan {
    let mut p = tk_mimo_k192_plan {
        splits: 1,
        ..Default::default()
    };
    let g_ratio = if n_kv > 0 { n_head / n_kv } else { 0 };
    if !q8_kv && n_tok >= 8 && n_split <= 1 && g_ratio % 8 == 0 {
        p.kind = 1;
        p.a = i32::from(e.use_wmma);
        p.b = e.fa_ver;
        return p;
    }
    let mut splits = 1;
    if has_scratch {
        if n_split > 1 {
            splits = n_split.min(32);
        } else if n_split <= 0 {
            splits = if stride >= 16384 {
                16
            } else if stride >= 2048 {
                8
            } else if stride >= 512 {
                4
            } else {
                1
            };
        }
    }
    p.splits = splits;
    if !q8_kv && e.att3 {
        let def = if n_tok == 1 { (4, 4) } else { (2, 8) };
        let (hg, nw) = if e.att3_shape.0 > 0 {
            e.att3_shape
        } else {
            def
        };
        if hg > 0 && nw > 0 && g_ratio % (hg * nw) == 0 && ATT3_SHAPES.contains(&(hg, nw)) {
            p.kind = 2;
            p.a = hg;
            p.b = nw;
            return p;
        }
    }
    if !q8_kv && e.att2 {
        let (hq, w, u) = match e.att2_shape {
            Some(s) if s.0 > 0 => s,
            _ => (2, 8, 4),
        };
        if hq > 0 && g_ratio % hq == 0 && ATT2_SHAPES.contains(&(hq, w, u)) {
            p.kind = 3;
            p.a = hq;
            p.b = w;
            p.c = u;
            return p;
        }
    }
    p.kind = 4;
    p.a = if g_ratio % 4 == 0 { 4 } else { 1 };
    p
}

/// The sliding-window attention plan (`tk_attn_swa_sinks*`), the same rules and order as `mimo_attn_swa_sinks_plan`
/// in `mimo_kernels.cu`: prefill v2 (`TALUS_MIMO_SWA_VER` 2, causal, head_dim 192, f16 KV, >= 8 tokens, GQA 8), prefill
/// v1 (GQA a multiple of 8), v2 decode (`TALUS_MIMO_ATT2`, shape `TALUS_MIMO_ATT2_SWA` or 1:4:4), the v2 drafter
/// form (head_dim 128, also `TALUS_MIMO_ATT2_DRAFT`), else v1. `causal` = `n_split != -2`.
pub fn swa_plan(
    q8_kv: bool,
    causal: bool,
    head_dim: i32,
    n_tok: i32,
    n_head: i32,
    n_kv: i32,
) -> talus_kernels_sys::tk_mimo_swa_plan {
    swa_plan_with(env(), q8_kv, causal, head_dim, n_tok, n_head, n_kv)
}

fn swa_plan_with(
    e: &Env,
    q8_kv: bool,
    causal: bool,
    head_dim: i32,
    n_tok: i32,
    n_head: i32,
    n_kv: i32,
) -> talus_kernels_sys::tk_mimo_swa_plan {
    let mut p = talus_kernels_sys::tk_mimo_swa_plan::default();
    let g_ratio = if n_kv > 0 { n_head / n_kv } else { 0 };
    if e.swa_ver == 2 && causal && head_dim == 192 && !q8_kv && n_tok >= 8 && g_ratio == 8 {
        p.kind = 1;
        return p;
    }
    if causal && head_dim == 192 && !q8_kv && n_tok >= 8 && g_ratio % 8 == 0 {
        p.kind = 2;
        return p;
    }
    let shape = match e.att2_swa_shape {
        Some(s) if s.0 > 0 => s,
        _ => (1, 4, 4),
    };
    let ok = shape.0 > 0 && g_ratio % shape.0 == 0 && ATT2_SHAPES.contains(&shape);
    if causal && head_dim == 192 && !q8_kv && e.att2 && ok {
        p.kind = 3;
        (p.a, p.b, p.c) = shape;
        return p;
    }
    if head_dim == 128 && !q8_kv && e.att2 && e.att2_draft && ok {
        p.kind = 4;
        (p.a, p.b, p.c) = shape;
        return p;
    }
    p.kind = 5;
    p.a = if g_ratio % 4 == 0 { 4 } else { 1 };
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> Env {
        Env {
            use_wmma: true,
            fa_ver: 7,
            att3: true,
            att2: true,
            att3_shape: (0, 0),
            att2_shape: None,
            swa_ver: 2,
            att2_draft: true,
            att2_swa_shape: None,
        }
    }

    #[test]
    fn swa_plans() {
        let e = defaults();
        let sp = |kind, a, b, c| talus_kernels_sys::tk_mimo_swa_plan { kind, a, b, c };
        // MiMo SWA layers: 64 query heads over 8 KV heads (the 2K gate trace: prefill_v2, att2 1:4:4 decode)
        assert_eq!(
            swa_plan_with(&e, false, true, 192, 2048, 64, 8),
            sp(1, 0, 0, 0)
        );
        assert_eq!(
            swa_plan_with(&e, false, true, 192, 1, 64, 8),
            sp(3, 1, 4, 4)
        );
        assert_eq!(swa_plan_with(&e, true, true, 192, 1, 64, 8), sp(5, 4, 0, 0));
        // the DFlash drafter: head_dim 128, non-causal
        assert_eq!(
            swa_plan_with(&e, false, false, 128, 16, 32, 4),
            sp(4, 1, 4, 4)
        );
        let mut e2 = defaults();
        e2.swa_ver = 1;
        assert_eq!(
            swa_plan_with(&e2, false, true, 192, 2048, 64, 8),
            sp(2, 0, 0, 0)
        );
        e2.att2_draft = false;
        assert_eq!(
            swa_plan_with(&e2, false, false, 128, 16, 32, 4),
            sp(5, 4, 0, 0)
        );
    }

    fn pl(kind: i32, splits: i32, a: i32, b: i32, c: i32) -> tk_mimo_k192_plan {
        tk_mimo_k192_plan {
            kind,
            splits,
            a,
            b,
            c,
        }
    }

    #[test]
    fn mimo_shapes_with_defaults() {
        let e = defaults();
        // the 2K gate trace (rport-s1, mimo): prompt fa_ver 7, decode att3 4:4 with 8 splits (GQA ratio 16)
        assert_eq!(
            plan_with(&e, false, true, 2048, 64, 4, 2048, 0),
            pl(1, 1, 1, 7, 0)
        );
        assert_eq!(
            plan_with(&e, false, true, 1, 64, 4, 2048, 0),
            pl(2, 8, 4, 4, 0)
        );
        assert_eq!(
            plan_with(&e, false, true, 5, 64, 4, 196608, 0),
            pl(2, 16, 2, 8, 0)
        );
        assert_eq!(
            plan_with(&e, false, true, 1, 64, 4, 511, 0),
            pl(2, 1, 4, 4, 0)
        );
        assert_eq!(
            plan_with(&e, false, false, 1, 64, 4, 65536, 0),
            pl(2, 1, 4, 4, 0)
        );
        assert_eq!(
            plan_with(&e, false, true, 1, 64, 4, 4096, 40),
            pl(2, 32, 4, 4, 0)
        );
        // q8 KV: v1 only
        assert_eq!(
            plan_with(&e, true, true, 1, 64, 4, 4096, 0),
            pl(4, 8, 4, 0, 0)
        );
        // prompt branch needs n_split <= 1 and f16 KV
        assert_eq!(
            plan_with(&e, false, true, 16, 64, 4, 4096, 2),
            pl(2, 2, 2, 8, 0)
        );
    }

    #[test]
    fn env_overrides() {
        let mut e = defaults();
        e.att3 = false;
        assert_eq!(
            plan_with(&e, false, true, 1, 64, 4, 2048, 0),
            pl(3, 8, 2, 8, 4)
        );
        e.att2_shape = Some((4, 4, 2));
        assert_eq!(
            plan_with(&e, false, true, 1, 64, 4, 2048, 0),
            pl(3, 8, 4, 4, 2)
        );
        e.att2_shape = Some((3, 4, 4)); // not instantiated: v1
        assert_eq!(
            plan_with(&e, false, true, 1, 64, 4, 2048, 0),
            pl(4, 8, 4, 0, 0)
        );
        let mut e = defaults();
        e.att3_shape = (8, 0); // sscanf read only one field: invalid shape, v2
        assert_eq!(
            plan_with(&e, false, true, 1, 64, 4, 2048, 0),
            pl(3, 8, 2, 8, 4)
        );
        e.use_wmma = false;
        assert_eq!(
            plan_with(&e, false, true, 64, 64, 4, 2048, 0),
            pl(1, 1, 0, 7, 0)
        );
    }

    #[test]
    fn c_parsers() {
        assert_eq!(c_atoi("7"), 7);
        assert_eq!(c_atoi(" 12x"), 12);
        assert_eq!(c_atoi("x"), 0);
        assert_eq!(c_atoi("-3"), -3);
        assert_eq!(scan_ints("2:8:4", 3), vec![2, 8, 4]);
        assert_eq!(scan_ints("2:8", 3), vec![2, 8]);
        assert_eq!(scan_ints("2x8", 2), vec![2]);
        assert_eq!(scan_ints("", 2), Vec::<i32>::new());
    }
}
