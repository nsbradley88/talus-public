//! GLM launch plans computed by the Rust host  S6e), the same rules and order as the kernel library's
//! `glm_heads_plan` / `glm_multi_plan` / `wide_q8_variant` (src/glm/glm_kernels.cu) and `glm_kda_step_plan` /
//! `glm_lightning_indexer_plan` / `glm_sparse_fa_idx_plan` (glm_spike_kernels.cu). For one release the `tkg_*_v` exports
//! compare the plan with the library's own, report a difference and run the library's (`TALUS_PLAN_TRUST=1`: the
//! host's), so a divergence here cannot change output.
//!
//! S7 adds the expert mat-vec table (`mmvq_pick`) and the fused KDA layout (`tkg_kda_fused`).
//!
//! Output-changing defaults decided here: GLM_N2 (wide WMMA), GLM_N3 (indexer WMMA), GLM_SPARSE_2P; each only on gfx11.

use std::sync::OnceLock;

use talus_core_api::{DeviceId, GpuArch};
use talus_kernels::plans::c_atoi;

/// `e == nullptr || e[0] != '0'`
fn on_unless_0(name: &str) -> bool {
    std::env::var_os(name)
        .map(|v| v.as_encoded_bytes().first() != Some(&b'0'))
        .unwrap_or(true)
}

/// `!e || atoi(e) != 0`
fn on_unless_atoi0(name: &str) -> bool {
    std::env::var(name).map(|v| c_atoi(&v) != 0).unwrap_or(true)
}

fn atoi_or(name: &str, default: i32) -> i32 {
    std::env::var(name).map(|v| c_atoi(&v)).unwrap_or(default)
}

/// Everything the plans read once: the environment and the device.
#[derive(Clone, Debug)]
pub struct Env {
    pub gfx11: bool,
    pub gfx1151: bool,
    pub num_cus: i32,
    pub mv_wide: bool,
    pub n2: bool,
    pub n3: bool,
    pub mv_tile: bool,
    pub mv_tile_cfg: i32,
    pub mv_heads: i32,
    pub kda_cpw: i32,
    pub kda_v3: bool,
    pub li_cfg: Option<(i32, i32, i32)>,
    pub sparse_2p: bool,
    pub sparse_2p_min: i32,
    pub sparse_wmma: i32,
    pub fa_wgs_per_cu: i32,
    pub f32_wide: bool,
    pub f32_pf: bool,
    /// `GLM_MMVQ_CFG=variant:R:NW:WPE[:pers]`
    pub mmvq_cfg: Option<i32>,
}

fn env() -> &'static Env {
    static E: OnceLock<Env> = OnceLock::new();
    E.get_or_init(|| {
        let dev = talus_hip::Device::new(DeviceId(0)).ok();
        let arch = dev.as_ref().map(|d| d.arch());
        let w = atoi_or("GLM_SPARSE_WMMA", 0);
        Env {
            gfx11: arch.as_ref().is_some_and(GpuArch::has_gfx11_wmma),
            gfx1151: arch == Some(GpuArch::Gfx1151),
            num_cus: dev
                .as_ref()
                .map_or(20, |d| d.multi_processor_count() as i32),
            mv_wide: on_unless_atoi0("GLM_MV_WIDE"),
            n2: on_unless_0("GLM_N2"),
            n3: on_unless_0("GLM_N3"),
            mv_tile: on_unless_atoi0("GLM_MV_TILE"),
            mv_tile_cfg: atoi_or("GLM_MV_TILE", 1),
            mv_heads: atoi_or("GLM_MV_HEADS", 0),
            kda_cpw: atoi_or("GLM_KDA_CPW", 0),
            kda_v3: on_unless_0("GLM_KDA_V3"),
            li_cfg: std::env::var("GLM_LI_CFG").ok().and_then(|v| {
                let f: Vec<&str> = v.split(':').collect();
                let num = |s: &str| {
                    !s.is_empty()
                        && s.trim_start_matches(['+', '-'])
                            .starts_with(|c: char| c.is_ascii_digit())
                };
                (f.len() >= 3 && f[..3].iter().all(|s| num(s)))
                    .then(|| (c_atoi(f[0]), c_atoi(f[1]), c_atoi(f[2])))
            }),
            sparse_2p: !std::env::var("GLM_SPARSE_2P").is_ok_and(|v| c_atoi(&v) == 0),
            sparse_2p_min: std::env::var("GLM_SPARSE_2P_MIN").map_or(9, |v| c_atoi(&v).max(1)),
            sparse_wmma: if w == 1 || w == 2 { w } else { 0 },
            fa_wgs_per_cu: atoi_or("GLM_FA_WGS_PER_CU", 2),
            f32_wide: std::env::var("GLM_MV_WIDE").is_ok_and(|v| c_atoi(&v) == 2),
            f32_pf: on_unless_0("GLM_F32MV_PF"),
            mmvq_cfg: std::env::var("GLM_MMVQ_CFG")
                .ok()
                .and_then(|v| parse_mmvq_cfg(&v)),
        }
    })
}

/// 0 = not covered, 1 = N2 WMMA, 2 = two tokens per wave, 3 = KS 1 / RW 8, 4 = KS 2 / RW 4.
fn wide_q8_variant(e: &Env, n_out_max: i32, n_in: i32) -> i32 {
    if !e.mv_wide || n_in % 32 != 0 || n_in > 512 {
        return 0;
    }
    if e.n2 && e.gfx11 && n_out_max % 64 == 0 {
        return 1;
    }
    if n_in <= 128 && e.mv_tile {
        return 2;
    }
    if n_in <= 256 {
        3
    } else {
        4
    }
}

fn heads_with(e: &Env, n_in: i32, n_out: i32, t: i32) -> i32 {
    if t <= 8 || n_in % 256 != 0 {
        return 0;
    }
    if e.mv_heads > 0 && e.mv_tile {
        return 100
            + if e.mv_heads == 2 || e.mv_heads == 3 {
                e.mv_heads
            } else {
                1
            };
    }
    match wide_q8_variant(e, n_out, n_in) {
        0 => 300,
        w => 200 + w,
    }
}

fn multi_with(e: &Env, n_mats: i32, mx: i32, n_in: i32, t: i32) -> i32 {
    if t <= 8 || !(1..=3).contains(&n_mats) || n_in % 32 != 0 {
        return 0;
    }
    let w = wide_q8_variant(e, mx, n_in);
    if w != 0 {
        return 200 + w;
    }
    if e.mv_tile && mx >= 1 && n_in > 512 {
        let cc = if e.mv_tile_cfg == 2 || e.mv_tile_cfg == 3 {
            e.mv_tile_cfg
        } else {
            1
        };
        if n_in % ((if cc == 1 { 2 } else { 1 }) * 256) == 0 {
            return 400 + cc;
        }
    }
    300
}

fn kda_with(e: &Env, t: i32) -> i32 {
    let cpw = if e.kda_cpw > 0 {
        e.kda_cpw
    } else if e.gfx1151 {
        if t <= 1 {
            11601
        } else {
            10808
        }
    } else {
        2
    };
    cpw * 10 + i32::from(e.kda_v3)
}

/// `li_pick`: (n_kv upper bound, tokens) -> (variant, nwv, tf).
fn li_pick(e: &Env, n_kv: i32, t: i32) -> (i32, i32, i32) {
    if let Some(c) = e.li_cfg {
        return c;
    }
    if !e.gfx1151 {
        return (0, 1, 1);
    }
    if n_kv <= 4096 {
        (1, 4, 1)
    } else if t == 1 {
        if n_kv <= 12288 {
            (1, 2, 1)
        } else {
            (1, 4, 1)
        }
    } else {
        (0, 1, 1)
    }
}

fn indexer_with(e: &Env, h: i32, n_kv: i32, t: i32) -> i32 {
    if e.n3 && h == 32 && t > 8 && e.gfx11 {
        return 1_000_000;
    }
    let (v, n, f) = li_pick(e, n_kv, t);
    v * 10000 + n * 100 + f
}

/// The split-K count of the sparse attention (`glm_sparse_fa_split_plan`): decides the reduction order. Windows up to the decode
/// tile (8 rows, up to GLM-5.3-Flash's largest launch of 2051 keys) all use one partition, the blocks per split a single row gets
/// at that largest launch, at least two splits (7bdae4d3: a token's logits must not depend on its window). Larger windows keep
/// their own count.
fn split_k(e: &Env, h: i32, n: i32, n_kv_max: i32) -> u32 {
    const HG: u32 = 32;
    const BC: u32 = 32;
    const MAX_SLOTS: u32 = 2304;
    const DECODE_ROWS: u32 = 8;
    const DECODE_KV: u32 = 2051;
    let n_kv = n_kv_max as u32;
    let total_blocks = (n_kv + BC - 1) / BC;
    let target_wgs = (e.num_cus as u32).wrapping_mul(e.fa_wgs_per_cu as u32);
    if n as u32 <= DECODE_ROWS && n_kv <= DECODE_KV {
        let plan_blocks = (DECODE_KV + BC - 1) / BC;
        let plan_k = (target_wgs / (h as u32 / HG).max(1))
            .max(1)
            .max((DECODE_KV + MAX_SLOTS - 1) / MAX_SLOTS)
            .min(plan_blocks);
        let per = (plan_blocks + plan_k - 1) / plan_k;
        return ((total_blocks + per - 1) / per).max(2);
    }
    let base_wgs = (h as u32 / HG) * n as u32;
    let k = (target_wgs / base_wgs.max(1))
        .max(1)
        .max((n_kv + MAX_SLOTS - 1) / MAX_SLOTS);
    let k = k.min(total_blocks);
    let per = (total_blocks + k - 1) / k;
    (total_blocks + per - 1) / per
}

#[allow(clippy::too_many_arguments)]
fn sparse_with(
    e: &Env,
    q: usize,
    kv: usize,
    dst: usize,
    d: i32,
    h: i32,
    n: i32,
    n_kv_max: i32,
    q_nb1: u32,
    q_nb2: u32,
    k_stride: u32,
) -> i32 {
    let strides = k_stride % 8 == 0 && q_nb1 % 4 == 0 && q_nb2 % 4 == 0;
    if e.sparse_2p
        && n >= e.sparse_2p_min
        && d == 512
        && h % 64 == 0
        && n_kv_max <= 2080
        && e.gfx11
        && strides
        && (kv | q | dst) & 15 == 0
    {
        return 100_001;
    }
    let k = split_k(e, h, n, n_kv_max);
    if e.sparse_wmma != 0 && k == 1 && d == 512 && h % 64 == 0 && e.gfx11 && strides {
        return if e.sparse_wmma == 2 { 300_001 } else { 200_001 };
    }
    400_000 + k as i32
}

const fn mmvq_plan(variant: i32, r: i32, nw: i32, wpe: i32, pers: i32) -> i32 {
    variant * 100_000_000 + r * 1_000_000 + nw * 10_000 + wpe * 100 + pers
}

/// `sscanf("%d:%d:%d:%d:%d") >= 4`, the fifth field 0 when absent.
fn parse_mmvq_cfg(v: &str) -> Option<i32> {
    let mut f = [0i32; 5];
    let mut n = 0;
    for s in v.split(':').take(5) {
        let t = s.trim_start();
        let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
        if !digits.starts_with(|c: char| c.is_ascii_digit()) {
            break;
        }
        f[n] = c_atoi(t);
        n += 1;
        // sscanf stops at the first character after the number that is not the ':' separator
        if !digits.chars().all(|c| c.is_ascii_digit()) {
            break;
        }
    }
    (n >= 4).then(|| mmvq_plan(f[0], f[1], f[2], f[3], f[4]))
}

/// (qtype, K) -1 = any; first match wins. gfx1151's sweep-selected table, then the generic 4 x 4 layout
/// (`kMmvqGfx1151`, `kMmvqGeneric` in glm_spike_kernels.cu; the library's own copy cross-checks for one release).
const MMVQ_GFX1151: [(i32, i32, i32, i32, i32); 6] = [
    (22, -1, 1, 1, mmvq_plan(1, 4, 4, 8, 16)),
    (21, -1, 1, 1, mmvq_plan(1, 4, 2, 8, 16)),
    (22, 2048, 2, 64, mmvq_plan(1, 4, 4, 8, 16)),
    (22, -1, 2, 64, mmvq_plan(1, 4, 4, 8, 0)),
    (21, 4096, 2, 64, mmvq_plan(1, 2, 8, 8, 0)),
    (21, -1, 2, 64, mmvq_plan(1, 4, 8, 8, 0)),
];
const MMVQ_GENERIC: i32 = mmvq_plan(1, 4, 4, 8, 0);

fn mmvq_with(e: &Env, qtype: i32, k: i32, t: i32) -> i32 {
    if let Some(p) = e.mmvq_cfg {
        return p;
    }
    if e.gfx1151 {
        for (q, kk, t0, t1, p) in MMVQ_GFX1151 {
            if (q < 0 || q == qtype) && (kk < 0 || kk == k) && t >= t0 && t <= t1 {
                return p;
            }
        }
    }
    if (1..=64).contains(&t) {
        MMVQ_GENERIC
    } else {
        0
    }
}

/// The fused KDA step's layout: `GLM_KDA_CPW`, else gfx1151's table, else none (the three launches).
fn kda_fused_with(e: &Env, t: i32) -> i32 {
    if e.kda_cpw > 0 {
        e.kda_cpw
    } else if e.gfx1151 {
        if t <= 1 {
            11601
        } else {
            10808
        }
    } else {
        0
    }
}

/// `glm_f32_matvec_plan`: kind * 1000 + config. 1 prompt tiles (the GLM_MV_TILE config), 2 wide f32 (opt-in GLM_MV_WIDE=2), 3 tiles of 8
/// tokens driven by the host (each then planned itself), 4 the decode kernel (config 1: batched loads, T <= 4, n_in >= 1024).
fn f32_with(e: &Env, n_in: i32, n_out: i32, t: i32) -> i32 {
    if t > 8 {
        let c = e.mv_tile_cfg;
        let stage = if c == 3 { 128 } else { 256 };
        if e.mv_tile && n_out >= 1 && n_in % stage == 0 {
            return 1000 + c;
        }
        if n_in % 4 == 0 && n_in <= 4096 && e.f32_wide {
            return 2000;
        }
        return 3000;
    }
    if e.f32_pf && t <= 4 && n_in >= 1024 {
        4001
    } else {
        4000
    }
}

/// The two-pass sparse attention's scratch slots when it can run on this device (gfx11, GLM_SPARSE_2P): 4 per CU, as the
/// library's own lazy allocation. `None`: the two-pass kernel cannot be selected, no pool needed.
pub fn sparse_2p_slots() -> Option<i32> {
    let e = env();
    (e.sparse_2p && e.gfx11).then(|| 4 * e.num_cus.max(1))
}

pub fn f32_matvec(n_in: i32, n_out: i32, t: i32) -> i32 {
    f32_with(env(), n_in, n_out, t)
}
pub fn mmvq(qtype: i32, k: i32, t: i32) -> i32 {
    mmvq_with(env(), qtype, k, t)
}
pub fn kda_fused(t: i32) -> i32 {
    kda_fused_with(env(), t)
}
pub fn heads(n_in: i32, n_out: i32, t: i32) -> i32 {
    heads_with(env(), n_in, n_out, t)
}
pub fn multi(n_mats: i32, mx: i32, n_in: i32, t: i32) -> i32 {
    multi_with(env(), n_mats, mx, n_in, t)
}
pub fn kda_step(t: i32) -> i32 {
    kda_with(env(), t)
}
pub fn indexer(h: i32, n_kv: i32, t: i32) -> i32 {
    indexer_with(env(), h, n_kv, t)
}
#[allow(clippy::too_many_arguments)]
pub fn sparse(
    q: usize,
    kv: usize,
    dst: usize,
    d: i32,
    h: i32,
    n: i32,
    n_kv_max: i32,
    q_nb1: u32,
    q_nb2: u32,
    k_stride: u32,
) -> i32 {
    sparse_with(env(), q, kv, dst, d, h, n, n_kv_max, q_nb1, q_nb2, k_stride)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_plans() {
        let e = gfx1151();
        assert_eq!(f32_with(&e, 4096, 288, 512), 1001);
        assert_eq!(f32_with(&e, 4096, 288, 8), 4000);
        assert_eq!(f32_with(&e, 4096, 288, 4), 4001);
        assert_eq!(f32_with(&e, 512, 288, 4), 4000);
        assert_eq!(f32_with(&e, 4100, 288, 512), 3000);
        let mut o = gfx1151();
        o.mv_tile = false;
        assert_eq!(f32_with(&o, 4096, 288, 512), 3000);
        o.f32_wide = true;
        assert_eq!(f32_with(&o, 4096, 288, 512), 2000);
        let mut c3 = gfx1151();
        c3.mv_tile_cfg = 3;
        assert_eq!(f32_with(&c3, 128, 288, 512), 1003);
    }

    #[test]
    fn split_decode_partition_is_one_for_every_window() {
        let e = gfx1151(); // 20 CUs, 2 groups per CU
                           // 65 plan blocks, 20 splits at H 64: 4 blocks per split at every size, at least two splits
        assert_eq!(split_k(&e, 64, 1, 2048), 16);
        assert_eq!(split_k(&e, 64, 8, 2048), 16);
        assert_eq!(split_k(&e, 64, 3, 2051), 17);
        assert_eq!(split_k(&e, 64, 1, 8), 2);
        // past the decode tile or the largest launch: their own count
        assert_eq!(split_k(&e, 64, 9, 2048), 2);
        assert_eq!(split_k(&e, 64, 8, 4096), 2);
    }

    #[test]
    fn mmvq_table_matches_the_library_rows() {
        let e = gfx1151();
        assert_eq!(mmvq_with(&e, 21, 2048, 1), 104_020_816); // 21 T=1: 4 x 2 x 8, persistent 16
        assert_eq!(mmvq_with(&e, 22, 4096, 1), 104_040_816);
        assert_eq!(mmvq_with(&e, 22, 2048, 8), 104_040_816);
        assert_eq!(mmvq_with(&e, 22, 4096, 8), 104_040_800);
        assert_eq!(mmvq_with(&e, 21, 4096, 8), 102_080_800);
        assert_eq!(mmvq_with(&e, 21, 2048, 8), 104_080_800);
        let mut o = gfx1151();
        o.gfx1151 = false;
        assert_eq!(mmvq_with(&o, 21, 2048, 1), 104_040_800);
        assert_eq!(mmvq_with(&o, 21, 2048, 65), 0);
        assert_eq!(kda_fused_with(&e, 1), 11601);
        assert_eq!(kda_fused_with(&e, 4), 10808);
        assert_eq!(kda_fused_with(&o, 4), 0);
    }

    #[test]
    fn mmvq_cfg_parses_like_sscanf() {
        assert_eq!(parse_mmvq_cfg("1:4:4:8"), Some(104_040_800));
        assert_eq!(parse_mmvq_cfg("1:4:4:8:16"), Some(104_040_816));
        assert_eq!(parse_mmvq_cfg("1:4:4"), None);
        assert_eq!(parse_mmvq_cfg(""), None);
        assert_eq!(parse_mmvq_cfg("1:4:4:8:x"), Some(104_040_800));
        assert_eq!(parse_mmvq_cfg("1:4:4x:8"), None);
    }

    fn gfx1151() -> Env {
        Env {
            gfx11: true,
            gfx1151: true,
            num_cus: 20,
            mv_wide: true,
            n2: true,
            n3: true,
            mv_tile: true,
            mv_tile_cfg: 1,
            mv_heads: 0,
            kda_cpw: 0,
            kda_v3: true,
            li_cfg: None,
            sparse_2p: true,
            sparse_2p_min: 9,
            sparse_wmma: 0,
            fa_wgs_per_cu: 2,
            f32_wide: false,
            f32_pf: true,
            mmvq_cfg: None,
        }
    }

    #[test]
    fn mv_cascades() {
        let e = gfx1151();
        assert_eq!(heads_with(&e, 256, 512, 8), 0);
        assert_eq!(heads_with(&e, 256, 512, 2048), 201); // N2
        assert_eq!(heads_with(&e, 512, 250, 2048), 204);
        assert_eq!(multi_with(&e, 3, 4096, 4096, 2048), 401); // KDA f_a / g_a: tile
        assert_eq!(multi_with(&e, 2, 128, 128, 2048), 201);
        assert_eq!(multi_with(&e, 1, 100, 128, 2048), 202);
        assert_eq!(multi_with(&e, 1, 100, 256, 2048), 203);
        assert_eq!(multi_with(&e, 1, 100, 544, 2048), 300);
        let mut off = gfx1151();
        off.gfx11 = false; // D4: no N2 off gfx11
        assert_eq!(multi_with(&off, 2, 128, 128, 2048), 202);
    }

    #[test]
    fn kda_indexer_sparse() {
        let e = gfx1151();
        assert_eq!(kda_with(&e, 1), 116011);
        assert_eq!(kda_with(&e, 5), 108081);
        assert_eq!(indexer_with(&e, 32, 2000, 2048), 1_000_000);
        assert_eq!(indexer_with(&e, 32, 2000, 1), 10401);
        assert_eq!(indexer_with(&e, 32, 8000, 1), 10201);
        assert_eq!(indexer_with(&e, 32, 20000, 1), 10401);
        assert_eq!(indexer_with(&e, 32, 20000, 4), 101);
        // 2P on prompts; decode tiles on fdot2 with the CU-based split
        assert_eq!(
            sparse_with(&e, 16, 16, 16, 512, 64, 2048, 2048, 4, 4, 8),
            100_001
        );
        assert_eq!(split_k(&e, 64, 1, 2048), 16);
        assert_eq!(
            sparse_with(&e, 16, 16, 16, 512, 64, 1, 2048, 4, 4, 8),
            400_016
        );
        assert_eq!(split_k(&e, 64, 8, 4096), 2);
    }
}
