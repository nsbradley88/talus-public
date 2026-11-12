//! The per-architecture prefill tuning table  S9): the source of truth for the host's launch plans. The library's
//! `include/strata/prefill/tp_tune.hpp` must hold the same rows (a test compares them field by field, so a retune has to
//! change both), and for one release its exports cross-check the host's plan against its own (`plan_choose`).

/// One row of `TpTune` (tp_tune.hpp has the field notes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpTune {
    pub arch: &'static str,
    pub fa_variant: i32,
    pub i8_variant: i32,
    pub w4_bm: i32,
    pub w4_bn: i32,
    pub w4_min_rows: i32,
    pub i8_mode_variant: [i32; 3],
    pub moe_ver: i32,
    pub moe_gu: i32,
    pub moe_gu_min_rows: i32,
    pub moe_gu_mode1_max_rows: i32,
    pub pf_w16: i32,
    pub pf_w16_blocks: i32,
    pub pf_w16_bm_gu: i32,
    pub pf_w16_bm_d: i32,
    pub pf_w16_blocks128: i32,
    pub pf_w16_moe_min_rows: i32,
    pub pf_w16_min_rows: i32,
}

pub const TP_TUNE_DEFAULT: TpTune = TpTune {
    arch: "default",
    fa_variant: 3,
    i8_variant: 6,
    w4_bm: 128,
    w4_bn: 128,
    w4_min_rows: 64,
    i8_mode_variant: [0, 0, 0],
    moe_ver: 0,
    moe_gu: 0,
    moe_gu_min_rows: 0,
    moe_gu_mode1_max_rows: 0,
    pf_w16: 0,
    pf_w16_blocks: 2,
    pf_w16_bm_gu: 0,
    pf_w16_bm_d: 0,
    pf_w16_blocks128: 2,
    pf_w16_moe_min_rows: 32,
    pf_w16_min_rows: 16,
};

/// gfx1151 (Radeon 8060S, 20 WGPs), the one measured architecture.
pub const TP_TUNE: [TpTune; 1] = [TpTune {
    arch: "gfx1151",
    fa_variant: 3,
    i8_variant: 6,
    w4_bm: 128,
    w4_bn: 128,
    w4_min_rows: 64,
    i8_mode_variant: [116, 116, 116],
    moe_ver: 73,
    moe_gu: 1,
    moe_gu_min_rows: 8,
    moe_gu_mode1_max_rows: 32,
    pf_w16: 2,
    pf_w16_blocks: 2,
    pf_w16_bm_gu: 0,
    pf_w16_bm_d: 0,
    pf_w16_blocks128: 0,
    pf_w16_moe_min_rows: 32,
    pf_w16_min_rows: 16,
}];

/// The row for `gcn` ("gfx1151" or "gfx1151:sramecc-:xnack-"), else the defaults (`tp_tune`).
pub fn row_for(gcn: &str) -> &'static TpTune {
    TP_TUNE
        .iter()
        .find(|t| {
            gcn.strip_prefix(t.arch)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with(':'))
        })
        .unwrap_or(&TP_TUNE_DEFAULT)
}

/// `i8gemm`'s variant (tp_i8gemm.cu): `TP_I8_VARIANT` (`env`, 0 = unset), else the row's per-mode variant, else its `i8_variant`;
/// the expansion modes 1 / 2 only have the LDS kernels (at least 6).
pub fn i8gemm_variant(row: &TpTune, env: i32, mode: i32) -> i32 {
    let mv = row.i8_mode_variant[(mode % 3) as usize];
    let v = if env != 0 {
        env
    } else if mv != 0 {
        mv
    } else {
        row.i8_variant
    };
    if mode != 0 && v < 6 {
        6
    } else {
        v
    }
}

/// The fused up GEMM's variant (`launch_swq8`): `TP_I8_VARIANT`, else the row's per-mode variant.
pub fn swq8_variant(row: &TpTune, env: i32, mode: i32) -> i32 {
    if env != 0 {
        env
    } else {
        row.i8_mode_variant[(mode % 3) as usize]
    }
}

/// `moe_version_for` (tp_moe.cu): `TP_MOE_VER` (`env_ver`, < 0 = unset), else the row's version for grouped products (one expert: v6's
/// slot 0); 73 becomes 74 from `TP_MOE_V74_MIN` routed rows per expert; a K tail always takes v6.
pub fn moe_version(
    row: &TpTune,
    env_ver: i32,
    v74_min: i32,
    n_experts: i32,
    total_rows: i64,
    k: i64,
) -> i32 {
    let mut v = if env_ver >= 0 {
        env_ver
    } else if n_experts > 1 {
        row.moe_ver
    } else {
        0
    };
    if env_ver < 0
        && v == 73
        && v74_min > 0
        && n_experts > 1
        && total_rows >= i64::from(v74_min) * i64::from(n_experts)
    {
        v = 74;
    }
    if k % 256 != 0 {
        6
    } else {
        v
    }
}

/// `moe_mode` / `i8x_mode`: the dot mode of a tp_moe type (Q2_K 3, or refused with `TP_MOE_Q2K=0`; Q6_K mode 1, or refused with
/// `TALUS_PF_I8_Q6K=0`), -1 = none.
pub fn moe_dot_mode(ty: i32, q2k_on: bool, q6k_on: bool) -> i32 {
    match ty {
        10 => {
            if q2k_on {
                3
            } else {
                -1
            }
        }
        23 | 20 | 21 | 18 | 16 | 8 | 6 => 0,
        11 | 17 | 22 => 1,
        14 => {
            if q6k_on {
                1
            } else {
                -1
            }
        }
        12 | 13 => 2,
        _ => -1,
    }
}

/// `moe_supported` (tp_moe.cu): a dot mode, n % 128 == 0, k > 0, k % 256 == 0 or a K tail type (IQ4_NL, Q8_0, Q5_0) with k % 32 == 0
/// unless `TP_MOE_KTAIL=0`.
pub fn moe_supported(dot_mode: i32, ty: i32, n: i32, k: i32, ktail_on: bool) -> bool {
    if dot_mode < 0 || n % 128 != 0 || k <= 0 {
        return false;
    }
    k % 256 == 0 || (ktail_on && matches!(ty, 20 | 8 | 6) && k % 32 == 0)
}

/// `moe_inplace_ok` + the FN export's `TALUS_PF_MOE`  , gather-free groups): `on` = TALUS_PF_MOE, STRATA_PF_INPLACE and
/// TP_MOE_KTAIL_SKIP not "0..."; tp_moe covers the product, the fp16 kernels do not take it, not Q2_K, and the version of a
/// 16-expert product is one of the k_moe5 bodies.
pub fn moe_inplace_ok(on: bool, supported: bool, w16: bool, ty: i32, ver: i32) -> bool {
    on && supported && !w16 && ty != 10 && matches!(ver, 0 | 5 | 6 | 7 | 63 | 72 | 73 | 74)
}

/// `moe_w16_enabled`: the tp_moe contract on the fp16 kernels (`STRATA_PF_W16_MOE=1`, listed types, allowed on this GPU).
pub fn moe_w16_enabled(row: &TpTune, env_on: bool, ty: i32) -> bool {
    env_on && matches!(ty, 22 | 21 | 20 | 16 | 18) && row.pf_w16 != 0
}

/// The fused gate/up rule of `tk_prefill_moe_mmq_gu`: from `moe_gu_min_rows` mean rows per expert, for dot mode 1 also at most
/// `moe_gu_mode1_max_rows` unless the fp16 kernels take the product; `force` = `TALUS_PF_MOE_GU=2`, `w16_takes`: their rule.
pub fn moe_gu(row: &TpTune, force: bool, dot_mode: i32, w16_takes: bool, mean: i64) -> bool {
    force
        || !(row.moe_gu == 0
            || mean < i64::from(row.moe_gu_min_rows)
            || (dot_mode == 1 && mean > i64::from(row.moe_gu_mode1_max_rows) && !w16_takes))
}

/// `tk_prefill_moe_mmq_gu_q8`'s rule.
pub fn moe_gu_q8(row: &TpTune, mean: i64) -> bool {
    row.moe_gu != 0 && mean >= i64::from(row.moe_gu_min_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inplace_rules() {
        assert!(moe_supported(0, 23, 1024, 4096, true));
        assert!(!moe_supported(0, 23, 1000, 4096, true));
        assert!(moe_supported(0, 8, 1024, 4128, true) && !moe_supported(0, 8, 1024, 4128, false));
        assert!(!moe_supported(0, 23, 1024, 4128, true));
        assert!(!moe_supported(-1, 10, 1024, 4096, true));
        assert!(moe_inplace_ok(true, true, false, 23, 73));
        assert!(!moe_inplace_ok(true, true, false, 23, 70));
        assert!(!moe_inplace_ok(true, true, true, 23, 73));
        assert!(!moe_inplace_ok(true, true, false, 10, 73));
        assert!(!moe_inplace_ok(false, true, false, 23, 73));
    }

    #[test]
    fn moe_rules() {
        let r = row_for("gfx1151");
        assert_eq!(moe_version(r, -1, 0, 288, 8000, 4096), 73);
        assert_eq!(moe_version(r, -1, 0, 1, 8000, 4096), 0);
        assert_eq!(moe_version(r, -1, 0, 288, 8000, 4128), 6);
        assert_eq!(moe_version(r, 72, 0, 288, 8000, 4096), 72);
        assert_eq!(moe_version(r, -1, 10, 288, 2880, 4096), 74);
        assert_eq!(moe_version(r, -1, 10, 288, 2879, 4096), 73);
        assert!(!moe_gu(r, false, 0, false, 7));
        assert!(moe_gu(r, false, 0, false, 8));
        assert!(moe_gu(r, false, 1, false, 32));
        assert!(!moe_gu(r, false, 1, false, 33));
        assert!(moe_gu(r, false, 1, true, 33));
        assert!(moe_gu(r, true, 1, false, 0));
        assert!(!moe_gu(&TP_TUNE_DEFAULT, false, 0, false, 100));
        assert!(moe_gu_q8(r, 8) && !moe_gu_q8(r, 7));
        assert_eq!(moe_dot_mode(10, false, true), -1);
        assert_eq!(moe_dot_mode(14, true, true), 1);
        assert!(
            moe_w16_enabled(r, true, 22)
                && !moe_w16_enabled(r, false, 22)
                && !moe_w16_enabled(r, true, 23)
        );
    }

    const HPP: &str = include_str!("../../../../include/strata/prefill/tp_tune.hpp");

    fn flat(t: &TpTune) -> Vec<i32> {
        let m = t.i8_mode_variant;
        vec![
            t.fa_variant,
            t.i8_variant,
            t.w4_bm,
            t.w4_bn,
            t.w4_min_rows,
            m[0],
            m[1],
            m[2],
            t.moe_ver,
            t.moe_gu,
            t.moe_gu_min_rows,
            t.moe_gu_mode1_max_rows,
            t.pf_w16,
            t.pf_w16_blocks,
            t.pf_w16_bm_gu,
            t.pf_w16_bm_d,
            t.pf_w16_blocks128,
            t.pf_w16_moe_min_rows,
            t.pf_w16_min_rows,
        ]
    }

    /// The numbers of the row literal `{"arch", ...}`; a field the literal leaves out keeps the struct's initializer, which the
    /// last two (and, for short rows, `pf_w16_blocks128`) use.
    fn hpp_row(arch: &str) -> Vec<i32> {
        let key = format!("{{\"{arch}\",");
        let at = HPP
            .find(&key)
            .unwrap_or_else(|| panic!("{arch} row missing"))
            + key.len();
        let end = at
            + HPP[at..]
                .find("};")
                .or_else(|| HPP[at..].find("},\n"))
                .unwrap();
        HPP[at..end]
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap())
            .collect()
    }

    #[test]
    fn rows_match_the_library_header() {
        for t in TP_TUNE.iter().chain([&TP_TUNE_DEFAULT]) {
            let lit = hpp_row(t.arch);
            let rs = flat(t);
            assert!(lit.len() <= rs.len(), "{}: {lit:?}", t.arch);
            assert_eq!(&rs[..lit.len()], &lit[..], "{}", t.arch);
        }
        // the struct's initializers for the fields the literals leave out
        assert!(HPP.contains("int pf_w16_moe_min_rows = 32;"));
        assert!(HPP.contains("int pf_w16_min_rows = 16;"));
        assert!(HPP.contains("pf_w16_blocks128 = 2;"));
        assert_eq!(
            HPP.matches("{\"gfx").count(),
            TP_TUNE.len(),
            "a new row in tp_tune.hpp"
        );
    }

    #[test]
    fn lookup_and_variants() {
        assert_eq!(row_for("gfx1151").arch, "gfx1151");
        assert_eq!(row_for("gfx1151:sramecc-:xnack-").arch, "gfx1151");
        assert_eq!(row_for("gfx11510").arch, "default");
        assert_eq!(row_for("gfx1150").arch, "default");
        let r = row_for("gfx1151");
        assert_eq!(i8gemm_variant(r, 0, 0), 116);
        assert_eq!(i8gemm_variant(r, 0, 1), 116);
        assert_eq!(i8gemm_variant(r, 3, 0), 3);
        assert_eq!(i8gemm_variant(r, 3, 2), 6);
        assert_eq!(i8gemm_variant(&TP_TUNE_DEFAULT, 0, 1), 6);
        assert_eq!(swq8_variant(r, 0, 2), 116);
        assert_eq!(swq8_variant(&TP_TUNE_DEFAULT, 0, 2), 0);
    }
}
