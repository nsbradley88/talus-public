//! The wave32 expert mat-vec's configuration  S10): rows per wave, wave cap, column groups and schedule by (type,
//! column count), `w32_cfg` of native_mmvq.cu. The rule table has ONE source, `include/strata/kernels/w32_rules_*.inc`, compiled
//! into the library (its runtime lookup and the tsum twins' compile-time lookup) and parsed here, so they cannot drift; the
//! host also compares the two lookups over every (type, columns) at startup (`tk_mmvq_w32_table_plan`).

use std::sync::OnceLock;

use crate::plans::c_atoi;

const GENERIC: &str = include_str!("../../../../include/strata/kernels/w32_rules_generic.inc");
const GFX1151: &str = include_str!("../../../../include/strata/kernels/w32_rules_gfx1151.inc");

/// The types with a wave32 kernel (`native_mmvq_w32_supported`).
pub const W32_TYPES: [i32; 17] = [
    2, 6, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 42,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct W32Rule {
    pub ty: i32,
    pub n0: i32,
    pub n1: i32,
    pub r: i32,
    pub w: i32,
    pub cg: i32,
    pub sg: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct W32Cfg {
    pub r: i32,
    pub w: i32,
    pub cg: i32,
    pub sg: i32,
}

impl W32Cfg {
    /// `native_mmvq_w32_table_plan`'s encoding.
    pub fn plan(self) -> i32 {
        self.r * 1_000_000 + self.w * 10_000 + (self.cg + 8) * 100 + self.sg
    }
}

/// The `{type, n0, n1, r, w[, cg[, sg]]}` rows of a rules file (comments stripped).
pub fn parse_rules(src: &str) -> Vec<W32Rule> {
    let text: String = src
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join(" ");
    text.split('{')
        .skip(1)
        .map(|row| {
            let body = row.split('}').next().unwrap_or("");
            let f: Vec<i32> = body
                .split(',')
                .map(|v| v.trim().parse().expect("w32 rules: not a number"))
                .collect();
            assert!((5..=7).contains(&f.len()), "w32 rules: {body}");
            W32Rule {
                ty: f[0],
                n0: f[1],
                n1: f[2],
                r: f[3],
                w: f[4],
                cg: f.get(5).copied().unwrap_or(0),
                sg: f.get(6).copied().unwrap_or(0),
            }
        })
        .collect()
}

/// `STRATA_MMVQ_W32_TAB="type:cols:R:W[:CG[:SG]],..."`: type or cols `*`, cols also `lo-hi`; a missing type is any.
pub fn parse_env_rules(v: &str) -> Vec<W32Rule> {
    v.split(',')
        .filter(|it| !it.is_empty())
        .map(|it| {
            let mut f = [-1, -1, 1, 0, 0, 0];
            let (mut n0, mut n1) = (1, 16);
            for (k, t) in it.split(':').take(6).enumerate() {
                if k == 1 {
                    if t != "*" {
                        n0 = c_atoi(t);
                        n1 = t.find('-').map_or(n0, |d| c_atoi(&t[d + 1..]));
                    }
                } else if t != "*" {
                    f[k] = c_atoi(t);
                }
            }
            W32Rule {
                ty: f[0],
                n0,
                n1,
                r: f[2],
                w: f[3],
                cg: f[4],
                sg: f[5],
            }
        })
        .collect()
}

fn valid(n: i32, r: i32, w: i32) -> bool {
    match w {
        0 => n <= 8,
        12 => (5..=8).contains(&n),
        8 => n >= 2 && (n <= 8 || r <= 2),
        6 => n >= 11 && r == 2,
        _ => false,
    }
}

/// `w32_sanitize`.
fn sanitize(n: i32, mut c: W32Cfg) -> W32Cfg {
    if n <= 8 || (c.cg != 8 && c.cg != -8) {
        c.cg = 0;
    }
    if c.r != 1 && c.r != 2 && c.r != 4 {
        c.r = 1;
    }
    if n > 8 && c.r == 4 {
        c.r = 2;
    }
    if !matches!(c.w, 0 | 6 | 8 | 12) {
        c.w = 0;
    }
    if !valid(n, c.r, c.w) {
        c.w = if n > 8 {
            8
        } else if n >= 5 && c.w == 12 {
            12
        } else if n >= 2 && c.w != 0 {
            8
        } else {
            0
        };
    }
    if !valid(n, c.r, c.w) {
        c.w = if n > 8 { 8 } else { 0 };
    }
    c
}

fn find(rules: &[W32Rule], ty: i32, n: i32) -> Option<W32Cfg> {
    rules
        .iter()
        .find(|q| (q.ty < 0 || q.ty == ty) && n >= q.n0 && n <= q.n1 && q.w >= 0)
        .map(|q| W32Cfg {
            r: q.r,
            w: q.w,
            cg: q.cg,
            sg: q.sg,
        })
}

/// The rule lookup, before the process overrides (`native_mmvq_set_w32` / `_waves`).
pub fn table_cfg(
    env: &[W32Rule],
    rtab: bool,
    gfx1151: &[W32Rule],
    generic: &[W32Rule],
    ty: i32,
    n: i32,
) -> W32Cfg {
    let hit = find(env, ty, n)
        .or_else(|| if rtab { find(gfx1151, ty, n) } else { None })
        .or_else(|| find(generic, ty, n))
        .unwrap_or(W32Cfg {
            r: 1,
            w: 0,
            cg: 0,
            sg: 0,
        });
    sanitize(n, hit)
}

struct Tables {
    env: Vec<W32Rule>,
    rtab: bool,
    gfx1151: Vec<W32Rule>,
    generic: Vec<W32Rule>,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| Tables {
        env: std::env::var("STRATA_MMVQ_W32_TAB")
            .map_or_else(|_| Vec::new(), |v| parse_env_rules(&v)),
        rtab: std::env::var("STRATA_MMVQ_W32_RTAB").map_or(1, |v| c_atoi(&v)) != 0,
        gfx1151: parse_rules(GFX1151),
        generic: parse_rules(GENERIC),
    })
}

/// This process's table choice for (type, columns) on a device with (`is_gfx1151`) or without the tuned rows.
pub fn cfg(is_gfx1151: bool, ty: i32, n: i32) -> W32Cfg {
    let t = tables();
    table_cfg(&t.env, t.rtab && is_gfx1151, &t.gfx1151, &t.generic, ty, n)
}

/// The startup equality check: every (type, columns) against the library's lookup; the differences (type, columns, host, library).
pub fn differences(
    is_gfx1151: bool,
    lib_plan: impl Fn(i32, i32) -> i32,
) -> Vec<(i32, i32, i32, i32)> {
    let mut out = Vec::new();
    for ty in W32_TYPES {
        for n in 1..=16 {
            let (h, l) = (cfg(is_gfx1151, ty, n).plan(), lib_plan(ty, n));
            if h != l {
                out.push((ty, n, h, l));
            }
        }
    }
    out
}

/// The environment of [`route`], read once.
#[derive(Clone, Copy, Debug)]
pub struct RouteEnv {
    pub xt_min: i32,
    pub xt_q8_min: i32,
    pub xt_q6_min: i32,
    pub xt2_cfg0: i32,
    pub lds_min: i32,
    pub kq8: bool,
}

impl RouteEnv {
    pub fn get() -> RouteEnv {
        static E: OnceLock<RouteEnv> = OnceLock::new();
        *E.get_or_init(|| {
            let int = |n: &str, d: i32| std::env::var(n).map_or(d, |v| c_atoi(&v));
            RouteEnv {
                xt_min: int("STRATA_MMVQ_XT", 0),
                xt_q8_min: int("STRATA_MMVQ_XT_Q8", 16),
                xt_q6_min: int("STRATA_MMVQ_XT_Q6", 32),
                xt2_cfg0: std::env::var("STRATA_MMVQ_XT2_CFG").map_or(0, |v| {
                    10 * c_atoi(&v) + v.find(':').map_or(8, |c| c_atoi(&v[c + 1..]))
                }),
                lds_min: int("STRATA_MMVQ_LDS", 0),
                kq8: int("STRATA_MMVQ_KQ8", 1) != 0,
            }
        })
    }
}

/// `native_mmvq_route_plan`: which family takes a `tk_mmvq` call, kind * 1000 + the XT2 config: 1 XT2 tiles, 2 XT KQ8 tiles, 3 the
/// LDS-staged kernel, 4 wave32 / wide-tile chunks. A verify window's kernels are frozen in its graph, so this decides capture.
pub fn route(e: &RouteEnv, w32_on: bool, ty: i32, n_out: i32, ncols: i32) -> i32 {
    if e.xt_min > 0 && ncols >= e.xt_min && ncols <= 64 && w32_on {
        if ty == 14 || ty == 8 {
            let tmin = if ty == 14 { e.xt_q6_min } else { e.xt_q8_min };
            if tmin > 0 && ncols >= tmin {
                let cfg = if e.xt2_cfg0 != 0 {
                    e.xt2_cfg0
                } else if ty == 8 && n_out >= 8192 {
                    88
                } else {
                    48
                };
                return 1000 + cfg;
            }
        } else if e.kq8 && (ty == 12 || ty == 13) {
            return 2000;
        }
    }
    if e.lds_min > 0
        && ncols >= e.lds_min
        && w32_on
        && W32_TYPES.contains(&ty)
        && (((ty == 12 || ty == 13) && e.kq8) || ty == 14 || ty == 8)
    {
        return 3000;
    }
    4000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes() {
        let mut e = RouteEnv {
            xt_min: 0,
            xt_q8_min: 16,
            xt_q6_min: 32,
            xt2_cfg0: 0,
            lds_min: 0,
            kq8: true,
        };
        assert_eq!(route(&e, true, 12, 1024, 8), 4000);
        e.xt_min = 8; // Laguna serve mode
        assert_eq!(route(&e, true, 12, 1024, 8), 2000);
        assert_eq!(route(&e, true, 12, 1024, 7), 4000);
        assert_eq!(route(&e, false, 12, 1024, 8), 4000);
        assert_eq!(route(&e, true, 8, 100_352, 16), 1088);
        assert_eq!(route(&e, true, 8, 72, 16), 1048);
        assert_eq!(route(&e, true, 8, 72, 15), 4000);
        assert_eq!(route(&e, true, 14, 1024, 16), 4000);
        assert_eq!(route(&e, true, 14, 1024, 32), 1048);
        e.lds_min = 9;
        assert_eq!(route(&e, true, 14, 1024, 16), 3000);
        assert_eq!(route(&e, true, 23, 1024, 16), 4000);
        e.kq8 = false;
        assert_eq!(route(&e, true, 12, 1024, 16), 4000);
    }

    fn at(ty: i32, n: i32) -> (i32, i32, i32) {
        let g = parse_rules(GFX1151);
        let d = parse_rules(GENERIC);
        let c = table_cfg(&[], true, &g, &d, ty, n);
        (c.r, c.w, c.cg)
    }

    #[test]
    fn rules_parse() {
        assert_eq!(parse_rules(GENERIC).len(), 27);
        let g = parse_rules(GFX1151);
        assert_eq!(
            g[0],
            W32Rule {
                ty: 12,
                n0: 4,
                n1: 5,
                r: 1,
                w: 0,
                cg: 0,
                sg: 0
            }
        );
        assert_eq!(g.last().unwrap().cg, 8);
    }

    #[test]
    fn gfx1151_choices() {
        assert_eq!(at(12, 4), (1, 0, 0)); // MiMo DFlash verify
        assert_eq!(at(22, 4), (4, 8, 0));
        assert_eq!(at(11, 8), (4, 0, 0));
        assert_eq!(at(23, 4), (1, 8, 0));
        assert_eq!(at(14, 7), (2, 8, 0));
        assert_eq!(at(23, 16), (2, 6, 0));
        assert_eq!(at(14, 16), (2, 8, 8));
        assert_eq!(at(14, 12), (2, 6, 0));
        // generic rows behind the tuned ones (one column)
        assert_eq!(at(10, 1), (2, 0, 0));
        assert_eq!(at(12, 1), (1, 0, 0));
    }

    #[test]
    fn sanitize_rules() {
        // R 4 past 8 columns falls to 2; W 6 needs at least 11 columns
        let s = sanitize(
            12,
            W32Cfg {
                r: 4,
                w: 6,
                cg: 0,
                sg: 0,
            },
        );
        assert_eq!((s.r, s.w), (2, 6));
        let s = sanitize(
            6,
            W32Cfg {
                r: 2,
                w: 6,
                cg: 8,
                sg: 0,
            },
        );
        assert_eq!((s.w, s.cg), (8, 0));
        assert_eq!(
            sanitize(
                1,
                W32Cfg {
                    r: 3,
                    w: 8,
                    cg: 0,
                    sg: 0
                }
            )
            .w,
            0
        );
    }

    #[test]
    fn env_rules_parse_like_the_library() {
        let r = parse_env_rules("18:4:2:0,*:2-3:4:8:8:1,12");
        assert_eq!(
            r[0],
            W32Rule {
                ty: 18,
                n0: 4,
                n1: 4,
                r: 2,
                w: 0,
                cg: 0,
                sg: 0
            }
        );
        assert_eq!(
            r[1],
            W32Rule {
                ty: -1,
                n0: 2,
                n1: 3,
                r: 4,
                w: 8,
                cg: 8,
                sg: 1
            }
        );
        assert_eq!(
            r[2],
            W32Rule {
                ty: 12,
                n0: 1,
                n1: 16,
                r: 1,
                w: 0,
                cg: 0,
                sg: 0
            }
        );
    }
}
