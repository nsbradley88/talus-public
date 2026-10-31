//! exact tail-row pruning for prefill (an exact cousin of DeepInfer's "decoder SWA bounded replay").
//!
//! DeepSeek V4.1 (YoCo: layers 20-39 are consumers of one shared global KV) runs its consumer layers over only the
//! last `window + backtracking` prompt rows. That is NOT exact there: the consumer layers' SWA keys / values come from
//! each layer's own hidden state, so the replayed rows see truncated windows (gufo measured NLL +-0.008, top-1 254/256).
//!
//! For a non-YoCo stack (MiMo, Laguna, Flash-Next) the same idea is only valid where nothing downstream reads every row:
//! after the LAST layer that keeps state for every row (a full-attention layer's KV, a recurrent layer's state). This
//! module computes, by a backward receptive-field pass, which rows each layer must still produce, so that skipping the
//! rest is bitwise exact for every later consumer: the last row's logits, drafter taps, decode / verify queries and
//! resumes at or after `min_query`.
//!
//! Per layer `L` (absolute positions, prefill rows `[.., final_end)`):
//! - `kv_from[L]`: the rows whose KV (or recurrent state contribution) must be written. Full / recurrent layers: 0.
//!   A window-`W` layer: `min(min_query, out_from[L]) - (W - 1)` (a query at `q` reads keys `q - W + 1 ..= q`).
//! - `out_from[L]`: the rows whose output (the residual after the layer) a later layer, a tap or the head consumes.
//!   Last layer: `final_end - out_rows` (and taps); layer `L - 1`: `min(kv_from[L], out_from[L], tap rows of L - 1)`.
//!
//! The engine then runs, per chunk, layer `L`'s KV part for rows `>= kv_from[L]` and its attention / output / FFN part
//! for rows `>= out_from[L]` only. To keep the GPU kernels' per-row arithmetic identical (query groups / tiles are laid
//! out from the chunk's first row; small batches take other kernels), [`chunk_tail_offset`] rounds the kept suffix of a
//! chunk to `align` rows from the chunk start and never keeps fewer than `min_rows` (or skips the whole chunk).

/// How a layer's mixer depends on earlier rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerAttn {
    /// Causal attention over every earlier row (its KV is read by every later query).
    Full,
    /// Sliding-window attention: a query at `q` reads keys `q - (w - 1) ..= q`.
    Window(usize),
    /// A recurrent mixer (GDN / KDA / SSM): the carried state depends on every row.
    Recurrent,
}

impl LayerAttn {
    /// True when every row of this layer is needed by any later query (full KV or a recurrent state).
    pub fn keeps_all_rows(self) -> bool {
        !matches!(self, LayerAttn::Window(_))
    }
}

/// What a prefill must leave exact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TailRequest {
    /// One past the last row of the whole prompt (absolute position), across all chunks / calls.
    pub final_end: usize,
    /// Trailing rows whose LAST-layer output is consumed (1 = the logits row; + a drafter's context window).
    pub out_rows: usize,
    /// Lowest absolute position a later query may sit at (decode starts at `final_end`; a serve resume or a rollback
    /// can go lower). Clamped to `final_end`.
    pub min_query: usize,
    /// Extra output taps: `(layer, rows)` = layer `layer`'s output is consumed for the last `rows` rows.
    pub taps: Vec<(usize, usize)>,
}

impl TailRequest {
    /// Plain prefill: only the last row's logits, decode continues at `final_end`.
    pub fn logits_only(final_end: usize) -> Self {
        Self {
            final_end,
            out_rows: 1,
            min_query: final_end,
            taps: Vec::new(),
        }
    }
}

/// Rows a layer must produce (absolute positions; `0` = every row).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerRows {
    /// First row whose KV / state must be written.
    pub kv_from: usize,
    /// First row whose output (attention + FFN + residual) must be computed.
    pub out_from: usize,
}

impl LayerRows {
    /// First row whose input hidden state is needed (= the previous layer's `out_from` bound).
    pub fn in_from(&self) -> usize {
        self.kv_from.min(self.out_from)
    }
}

/// The prefill's chunk schedule and the kernels' row-grouping constraints. Rows a layer must produce are rounded to
/// what the engine really computes: in the chunk holding the bound, the kept suffix starts on a multiple of `align`
/// rows from the chunk start, has at least `min_rows` rows, and chunks of `<= small_nb` rows are never split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// First row of this prefill (rows before it come from earlier calls and are assumed exact).
    pub start: usize,
    /// Chunk size (the engine's ubatch): chunks are `[start + k * ubatch, ..)`.
    pub ubatch: usize,
    pub align: usize,
    pub min_rows: usize,
    pub small_nb: usize,
}

impl Grid {
    /// Row-exact grid (no rounding), for analysis.
    pub fn exact(start: usize, ubatch: usize) -> Self {
        Self {
            start,
            ubatch,
            align: 1,
            min_rows: 1,
            small_nb: 0,
        }
    }

    /// The first row the engine computes when rows `>= x` are needed (`<= x`).
    pub fn round(&self, x: usize, end: usize) -> usize {
        if x <= self.start || x >= end {
            return x;
        }
        let ub = self.ubatch.max(1);
        let cs = self.start + (x - self.start) / ub * ub;
        let nb = (end - cs).min(ub);
        cs + chunk_tail_offset(cs, nb, x, self.align, self.min_rows, self.small_nb)
    }
}

/// Backward receptive-field pass: the rows each layer must produce for `req` to stay bitwise exact (row-exact bounds).
pub fn plan(layers: &[LayerAttn], req: &TailRequest) -> Vec<LayerRows> {
    plan_on_grid(layers, req, None)
}

/// [`plan`] with every bound rounded to the rows the engine really computes on `grid` (a rounded-down output bound
/// computes extra rows, whose attention then needs extra keys: the rounding feeds the next bound).
pub fn plan_on_grid(
    layers: &[LayerAttn],
    req: &TailRequest,
    grid: Option<&Grid>,
) -> Vec<LayerRows> {
    let n = layers.len();
    let mut out = vec![
        LayerRows {
            kv_from: 0,
            out_from: 0
        };
        n
    ];
    if n == 0 {
        return out;
    }
    let end = req.final_end;
    let min_q = req.min_query.min(end);
    let round = |x: usize| -> usize { grid.map_or(x, |g| g.round(x, end)) };
    let tap_from = |il: usize| -> usize {
        req.taps
            .iter()
            .filter(|&&(l, _)| l == il)
            .map(|&(_, r)| end.saturating_sub(r.max(1)))
            .min()
            .unwrap_or(usize::MAX)
    };
    // rows of the last layer's output: the head (+ taps on the last layer)
    let mut need_out = end.saturating_sub(req.out_rows.max(1)).min(tap_from(n - 1));
    for il in (0..n).rev() {
        let out_from = round(need_out);
        let kv_from = match layers[il] {
            LayerAttn::Full | LayerAttn::Recurrent => 0,
            LayerAttn::Window(w) => {
                let w1 = w.max(1) - 1;
                round(min_q.saturating_sub(w1).min(out_from.saturating_sub(w1)))
            }
        };
        out[il] = LayerRows { kv_from, out_from };
        if il > 0 {
            need_out = out[il].in_from().min(tap_from(il - 1));
        }
    }
    out
}

/// The first layer from which any row can be skipped (layers before it compute every row), or `None`.
pub fn first_prunable_layer(rows: &[LayerRows]) -> Option<usize> {
    rows.iter().position(|r| r.out_from > 0)
}

/// Offset (0 ..= nb) inside the chunk `[chunk_start, chunk_start + nb)` from which rows `>= need_from` are computed:
/// `0` = every row (no pruning), `nb` = the whole chunk is skipped. The kept suffix starts on a multiple of `align`
/// rows from the chunk start and has at least `min_rows` rows; chunks of `nb <= small_nb` rows (decode / verify kernels)
/// are never split.
pub fn chunk_tail_offset(
    chunk_start: usize,
    nb: usize,
    need_from: usize,
    align: usize,
    min_rows: usize,
    small_nb: usize,
) -> usize {
    let align = align.max(1);
    if nb == 0 || need_from <= chunk_start {
        return 0;
    }
    if need_from >= chunk_start + nb {
        return nb;
    }
    if nb <= small_nb {
        return 0;
    }
    let mut ks = (need_from - chunk_start) / align * align;
    if nb - ks < min_rows {
        ks = nb.saturating_sub(min_rows) / align * align;
    }
    ks
}

/// Row-level work count for a plan under a chunk schedule: `(full_rows, kv_only_rows, skipped_rows)` summed over the
/// layers (rows of layer `L` computed fully, KV-only, or not at all). For estimating the gain from a profile.
pub fn work_rows(
    layers: &[LayerAttn],
    rows: &[LayerRows],
    start: usize,
    end: usize,
    ubatch: usize,
    align: usize,
    min_rows: usize,
    small_nb: usize,
) -> Vec<(usize, usize, usize)> {
    let mut acc = vec![(0usize, 0usize, 0usize); layers.len()];
    let mut cs = start;
    while cs < end {
        let nb = (end - cs).min(ubatch.max(1));
        for (il, r) in rows.iter().enumerate() {
            let ko = chunk_tail_offset(cs, nb, r.out_from, align, min_rows, small_nb);
            let kk = if layers[il].keeps_all_rows() {
                0
            } else {
                chunk_tail_offset(cs, nb, r.kv_from, align, min_rows, small_nb).min(ko)
            };
            acc[il].0 += nb - ko;
            acc[il].1 += ko - kk;
            acc[il].2 += kk;
        }
        cs += nb;
    }
    acc
}

/// MiMo-V2.6-Flash main stack (48 layers; full attention at 0, 5, 11, 17, 23, 29, 35, 41, 47, SWA-128 elsewhere;
/// `mimo2.attention.sliding_window_pattern` of the a4 / MOPD GGUFs).
pub fn mimo_layers() -> Vec<LayerAttn> {
    (0..48)
        .map(|il| {
            if il == 0 || il % 6 == 5 {
                LayerAttn::Full
            } else {
                LayerAttn::Window(128)
            }
        })
        .collect()
}

/// Laguna-S-2.1 (48 layers; full at `il % 4 == 0`, SWA-512 elsewhere).
pub fn laguna_layers() -> Vec<LayerAttn> {
    (0..48)
        .map(|il| {
            if il % 4 == 0 {
                LayerAttn::Full
            } else {
                LayerAttn::Window(512)
            }
        })
        .collect()
}

/// Qwen3.8-Flash-Next (48 layers; softmax + DSA at `il % 4 == 3`, gated-delta-net elsewhere).
pub fn flash_next_layers() -> Vec<LayerAttn> {
    (0..48)
        .map(|il| {
            if il % 4 == 3 {
                LayerAttn::Full
            } else {
                LayerAttn::Recurrent
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------------------------------------
    // A toy transformer whose per-row arithmetic is independent of which other rows a batch holds (as the GPU
    // kernels are once the kept suffix keeps their row grouping). Every KV cell is an Option: reading a cell that
    // a pruned prefill skipped panics, so the tests prove both bitwise equality AND that nothing reads skipped rows.
    // ---------------------------------------------------------------------------------------------------------
    const D: usize = 4;
    type Row = [f32; D];

    fn mix(seed: u32, i: usize, j: usize) -> f32 {
        let mut h = seed
            .wrapping_mul(2654435761)
            .wrapping_add((i as u32).wrapping_mul(40503))
            .wrapping_add((j as u32).wrapping_mul(9973));
        h ^= h >> 13;
        h = h.wrapping_mul(0x5bd1e995);
        h ^= h >> 15;
        ((h % 2001) as f32 - 1000.0) / 2500.0
    }

    fn matvec(seed: u32, x: &Row) -> Row {
        let mut y = [0f32; D];
        for i in 0..D {
            let mut s = 0f32;
            for j in 0..D {
                s += mix(seed, i, j) * x[j];
            }
            y[i] = s;
        }
        y
    }

    fn embed(tok: i32) -> Row {
        let mut r = [0f32; D];
        for (i, v) in r.iter_mut().enumerate() {
            *v = mix(7 + tok as u32, i, 3) * 2.0;
        }
        r
    }

    struct Toy {
        layers: Vec<LayerAttn>,
        k: Vec<Vec<Option<Row>>>,
        v: Vec<Vec<Option<Row>>>,
        // recurrent layers: state after row r (Some once computed)
        st: Vec<Vec<Option<Row>>>,
        // last-layer outputs per row (taps / head)
        last_out: Vec<Option<Row>>,
        n_pos: usize,
        // work counters
        full_rows: usize,
        kv_rows: usize,
    }

    impl Toy {
        fn new(layers: Vec<LayerAttn>, cap: usize) -> Self {
            let n = layers.len();
            Self {
                layers,
                k: vec![vec![None; cap]; n],
                v: vec![vec![None; cap]; n],
                st: vec![vec![None; cap]; n],
                last_out: vec![None; cap],
                n_pos: 0,
                full_rows: 0,
                kv_rows: 0,
            }
        }

        fn rollback(&mut self, pos: usize) {
            assert!(pos <= self.n_pos);
            self.n_pos = pos;
        }

        /// One chunk `tokens` at `self.n_pos` with an optional plan (`None` = every row of every layer).
        fn chunk(
            &mut self,
            tokens: &[i32],
            plan: Option<&[LayerRows]>,
            align: usize,
            min_rows: usize,
        ) -> Option<Row> {
            let p0 = self.n_pos;
            let nb = tokens.len();
            let mut h: Vec<Option<Row>> = tokens.iter().map(|&t| Some(embed(t))).collect();
            for il in 0..self.layers.len() {
                let (ko, kk) = match plan {
                    None => (0, 0),
                    Some(p) => {
                        let ko = chunk_tail_offset(p0, nb, p[il].out_from, align, min_rows, 0);
                        let kk = if self.layers[il].keeps_all_rows() {
                            0
                        } else {
                            chunk_tail_offset(p0, nb, p[il].kv_from, align, min_rows, 0).min(ko)
                        };
                        (ko, kk)
                    }
                };
                // KV / state for rows kk..nb
                for r in kk..nb {
                    let x = h[r].expect("input row of a needed KV row was skipped");
                    let pos = p0 + r;
                    match self.layers[il] {
                        LayerAttn::Recurrent => {
                            let prev = if pos == 0 {
                                [0f32; D]
                            } else {
                                self.st[il][pos - 1].expect("recurrent state of a skipped row read")
                            };
                            let u = matvec(100 + il as u32, &x);
                            let mut s = [0f32; D];
                            for i in 0..D {
                                s[i] = 0.9 * prev[i] + 0.1 * u[i].tanh();
                            }
                            self.st[il][pos] = Some(s);
                        }
                        _ => {
                            self.k[il][pos] = Some(matvec(200 + il as u32, &x));
                            self.v[il][pos] = Some(matvec(300 + il as u32, &x));
                        }
                    }
                }
                self.kv_rows += nb - kk;
                // outputs for rows ko..nb; rows before ko become unknown
                let mut nh: Vec<Option<Row>> = vec![None; nb];
                for r in ko..nb {
                    let x = h[r].expect("input row of a needed output row was skipped");
                    let pos = p0 + r;
                    let a: Row = match self.layers[il] {
                        LayerAttn::Recurrent => self.st[il][pos].unwrap(),
                        LayerAttn::Full | LayerAttn::Window(_) => {
                            let lo = match self.layers[il] {
                                LayerAttn::Window(w) => (pos + 1).saturating_sub(w),
                                _ => 0,
                            };
                            let q = matvec(400 + il as u32, &x);
                            // sequential online softmax over keys lo..=pos in ascending order (one fixed order)
                            let mut m = f32::NEG_INFINITY;
                            let mut l = 0f32;
                            let mut acc = [0f32; D];
                            for p in lo..=pos {
                                let kp = self.k[il][p].expect("attention read a skipped key");
                                let vp = self.v[il][p].expect("attention read a skipped value");
                                let mut s = 0f32;
                                for i in 0..D {
                                    s += q[i] * kp[i];
                                }
                                let mn = m.max(s);
                                let al = (m - mn).exp();
                                let w = (s - mn).exp();
                                l = l * al + w;
                                for i in 0..D {
                                    acc[i] = acc[i] * al + w * vp[i];
                                }
                                m = mn;
                            }
                            let mut o = [0f32; D];
                            for i in 0..D {
                                o[i] = acc[i] / l;
                            }
                            o
                        }
                    };
                    let f = matvec(500 + il as u32, &a);
                    let mut y = x;
                    for i in 0..D {
                        y[i] += f[i].tanh();
                    }
                    nh[r] = Some(y);
                }
                self.full_rows += nb - ko;
                h = nh;
            }
            for r in 0..nb {
                self.last_out[p0 + r] = h[r];
            }
            self.n_pos += nb;
            h[nb - 1].map(|x| matvec(999, &x))
        }

        fn prefill(
            &mut self,
            tokens: &[i32],
            ubatch: usize,
            plan: Option<&[LayerRows]>,
            align: usize,
            min_rows: usize,
        ) -> Row {
            let mut last = None;
            for c in tokens.chunks(ubatch) {
                last = self.chunk(c, plan, align, min_rows);
            }
            last.expect("last row's logits")
        }
    }

    fn toks(n: usize, seed: i32) -> Vec<i32> {
        (0..n as i32).map(|i| (i * 37 + seed * 11) % 97).collect()
    }

    fn bits(r: &Row) -> [u32; D] {
        let mut b = [0u32; D];
        for i in 0..D {
            b[i] = r[i].to_bits();
        }
        b
    }

    /// Exact vs pruned: last logits, tap rows, then `n_dec` decode steps, then a resume at `resume` (>= min_query)
    /// with fresh tokens; all bitwise. Returns (full_rows exact, full_rows pruned).
    #[allow(clippy::too_many_arguments)]
    fn check(
        layers: &[LayerAttn],
        n: usize,
        ubatch: usize,
        out_rows: usize,
        slack: usize,
        align: usize,
        min_rows: usize,
        tweak: impl Fn(&mut Vec<LayerRows>),
    ) -> (usize, usize) {
        let prompt = toks(n, 3);
        let cap = n + 64;
        let mut a = Toy::new(layers.to_vec(), cap);
        let mut b = Toy::new(layers.to_vec(), cap);
        let req = TailRequest {
            final_end: n,
            out_rows,
            min_query: n.saturating_sub(slack),
            taps: Vec::new(),
        };
        let grid = Grid {
            start: 0,
            ubatch,
            align,
            min_rows,
            small_nb: 0,
        };
        let mut p = plan_on_grid(layers, &req, Some(&grid));
        tweak(&mut p);
        let la = a.prefill(&prompt, ubatch, None, align, min_rows);
        let lb = b.prefill(&prompt, ubatch, Some(&p), align, min_rows);
        assert_eq!(bits(&la), bits(&lb), "last-row logits differ");
        for r in n - out_rows.min(n)..n {
            assert_eq!(
                bits(&a.last_out[r].unwrap()),
                bits(&b.last_out[r].expect("tap row skipped")),
                "tap row {r} differs"
            );
        }
        let (fa, fb) = (a.full_rows, b.full_rows);
        for t in 0..6 {
            let x = a.chunk(&[t * 5 + 1], None, align, min_rows).unwrap();
            let y = b.chunk(&[t * 5 + 1], None, align, min_rows).unwrap();
            assert_eq!(bits(&x), bits(&y), "decode step {t} differs");
        }
        // resume inside the prompt tail (serve prefix reuse / rollback) and re-prefill fresh tokens
        let resume = n.saturating_sub(slack);
        a.rollback(resume);
        b.rollback(resume);
        let more = toks(5, 9);
        let x = a.prefill(&more, ubatch, None, align, min_rows);
        let y = b.prefill(&more, ubatch, None, align, min_rows);
        assert_eq!(bits(&x), bits(&y), "resume at {resume} differs");
        (fa, fb)
    }

    fn small_mimo() -> Vec<LayerAttn> {
        // MiMo's shape in miniature: full first / every 6th, window elsewhere, the last layer full
        (0..12)
            .map(|il| {
                if il == 0 || il % 6 == 5 {
                    LayerAttn::Full
                } else {
                    LayerAttn::Window(6)
                }
            })
            .collect()
    }

    fn small_laguna() -> Vec<LayerAttn> {
        // Laguna's shape: full every 4th from 0, so the last 3 layers are windowed
        (0..12)
            .map(|il| {
                if il % 4 == 0 {
                    LayerAttn::Full
                } else {
                    LayerAttn::Window(5)
                }
            })
            .collect()
    }

    #[test]
    fn plan_shapes_of_the_talus_models() {
        // MiMo: every layer but the last computes every row; the last keeps its (full) KV for all rows
        let req = TailRequest::logits_only(32768);
        let p = plan(&mimo_layers(), &req);
        assert!(p[..47].iter().all(|r| r.out_from == 0 && r.kv_from == 0));
        assert_eq!(
            p[47],
            LayerRows {
                kv_from: 0,
                out_from: 32767
            }
        );
        assert_eq!(first_prunable_layer(&p), Some(47));

        // Laguna: the last full layer is 44; layers 45-47 are SWA-512
        let p = plan(&laguna_layers(), &req);
        assert!(p[..44].iter().all(|r| r.out_from == 0));
        // E = 32768: the last row's query reads keys E-512 ..= E-1, so each SWA layer reaches 511 rows further back
        assert_eq!(p[47].out_from, 32767);
        assert_eq!(p[47].kv_from, 32767 - 511);
        assert_eq!(p[46].out_from, 32767 - 511);
        assert_eq!(p[46].kv_from, 32767 - 2 * 511);
        assert_eq!(p[45].out_from, 32767 - 2 * 511);
        assert_eq!(p[45].kv_from, 32767 - 3 * 511);
        assert_eq!(
            p[44],
            LayerRows {
                kv_from: 0,
                out_from: 32767 - 3 * 511
            }
        );
        assert_eq!(first_prunable_layer(&p), Some(44));

        // Flash-Next: recurrent layers keep every row; only the last (full) layer's output is prunable
        let p = plan(&flash_next_layers(), &req);
        assert_eq!(first_prunable_layer(&p), Some(47));
        assert_eq!(p[47].out_from, 32767);
    }

    #[test]
    fn plan_with_drafter_window_and_resume_slack() {
        // MiMo DFlash: the drafter taps the last layer's output over a 1024-row window
        let req = TailRequest {
            final_end: 32768,
            out_rows: 1024 + 128,
            min_query: 32768 - 2048,
            taps: Vec::new(),
        };
        let p = plan(&mimo_layers(), &req);
        assert_eq!(p[47].out_from, 32768 - 1152);
        // Laguna DFlash (window 1024) + a 2048-row resume slack
        let p = plan(&laguna_layers(), &req);
        // the resume slack (min_query = E - 2048) dominates the drafter tail (E - 1152)
        assert_eq!(p[47].kv_from, 32768 - 2048 - 511);
        assert_eq!(p[44].out_from, 32768 - 2048 - 3 * 511);
        // a tap on a mid layer keeps that layer's rows
        let req = TailRequest {
            final_end: 4096,
            out_rows: 1,
            min_query: 4096,
            taps: vec![(46, 3000)],
        };
        let p = plan(&laguna_layers(), &req);
        assert!(p[46].out_from <= 4096 - 3000);
    }

    #[test]
    fn chunk_offsets() {
        // whole chunk before the tail: skip
        assert_eq!(chunk_tail_offset(0, 2048, 30000, 256, 128, 8), 2048);
        // tail inside the chunk: rounded down to 256 from the chunk start
        assert_eq!(chunk_tail_offset(30720, 2048, 31616, 256, 128, 8), 768);
        // never fewer than min_rows
        assert_eq!(chunk_tail_offset(0, 2048, 2047, 256, 128, 8), 1792);
        assert_eq!(chunk_tail_offset(0, 300, 299, 256, 128, 8), 0);
        // need_from at / before the chunk start: everything
        assert_eq!(chunk_tail_offset(4096, 2048, 4096, 256, 128, 8), 0);
        // decode / verify sized chunks are never split
        assert_eq!(chunk_tail_offset(100, 5, 103, 1, 1, 8), 0);
        assert_eq!(chunk_tail_offset(100, 5, 200, 1, 1, 8), 5);
    }

    #[test]
    fn toy_exact_mimo_like() {
        for &(n, ub) in &[(37usize, 8usize), (64, 16), (100, 32), (129, 64), (23, 100)] {
            for &(out_rows, slack) in &[(1usize, 0usize), (9, 0), (9, 12)] {
                for &(align, min_rows) in &[(1usize, 1usize), (4, 4), (8, 3)] {
                    let (fa, fb) = check(
                        &small_mimo(),
                        n,
                        ub,
                        out_rows,
                        slack,
                        align,
                        min_rows,
                        |_| {},
                    );
                    assert!(fb <= fa);
                    if n >= 64 && align == 1 && out_rows == 1 {
                        assert!(fb < fa, "nothing pruned at n {n}");
                    }
                }
            }
        }
    }

    #[test]
    fn toy_exact_laguna_like() {
        for &(n, ub) in &[(40usize, 8usize), (77, 16), (128, 32), (19, 64)] {
            for &(out_rows, slack) in &[(1usize, 0usize), (7, 0), (7, 9)] {
                for &(align, min_rows) in &[(1usize, 1usize), (4, 4), (8, 3)] {
                    let (fa, fb) = check(
                        &small_laguna(),
                        n,
                        ub,
                        out_rows,
                        slack,
                        align,
                        min_rows,
                        |_| {},
                    );
                    assert!(fb <= fa);
                    if n >= 40 && align == 1 && out_rows == 1 {
                        assert!(fb < fa);
                    }
                }
            }
        }
    }

    #[test]
    fn toy_exact_recurrent_like() {
        let layers: Vec<LayerAttn> = (0..8)
            .map(|il| {
                if il % 4 == 3 {
                    LayerAttn::Full
                } else {
                    LayerAttn::Recurrent
                }
            })
            .collect();
        for &(n, ub) in &[(50usize, 16usize), (33, 8)] {
            let (fa, fb) = check(&layers, n, ub, 1, 0, 1, 1, |_| {});
            assert!(fb < fa);
        }
    }

    #[test]
    #[should_panic]
    fn toy_off_by_one_window_is_caught() {
        // pruning one key row more than the window allows must be detected
        let _ = check(&small_laguna(), 64, 16, 1, 0, 1, 1, |p| {
            p[11].kv_from += 1; // the last layer (window 5)
        });
    }

    #[test]
    #[should_panic]
    fn toy_off_by_one_output_is_caught() {
        let _ = check(&small_laguna(), 64, 16, 1, 0, 1, 1, |p| {
            p[10].out_from += 1;
        });
    }

    #[test]
    #[should_panic]
    fn toy_missing_resume_slack_is_caught() {
        // a resume below the planned min_query reads keys the pruned prefill never wrote
        let layers = small_laguna();
        let n = 64;
        let prompt = toks(n, 3);
        let mut b = Toy::new(layers.clone(), n + 64);
        let p = plan(&layers, &TailRequest::logits_only(n));
        b.prefill(&prompt, 16, Some(&p), 1, 1);
        b.rollback(n - 8);
        b.prefill(&toks(3, 9), 16, None, 1, 1);
    }

    #[test]
    fn work_estimates() {
        // MiMo, 32K, ubatch 2048, served DFlash tail (1024 + 128 rows): layer 47 computes ~1.3K of 32K rows
        let req = TailRequest {
            final_end: 32768,
            out_rows: 1152,
            min_query: 32768,
            taps: Vec::new(),
        };
        let g = |_end: usize| Grid {
            start: 0,
            ubatch: 2048,
            align: 256,
            min_rows: 128,
            small_nb: 8,
        };
        let l = mimo_layers();
        let p = plan_on_grid(&l, &req, Some(&g(32768)));
        let w = work_rows(&l, &p, 0, 32768, 2048, 256, 128, 8);
        assert_eq!(w[46], (32768, 0, 0));
        assert_eq!(w[47].0, 1280);
        assert_eq!(w[47].1, 32768 - 1280);
        // Laguna 2K with the same tail: nothing below the window budget to skip
        let l = laguna_layers();
        let p = plan_on_grid(&l, &req.clone_with_end(2048), Some(&g(2048)));
        let w = work_rows(&l, &p, 0, 2048, 2048, 256, 128, 8);
        assert!(w[47].0 >= 1152);
    }

    impl TailRequest {
        fn clone_with_end(&self, end: usize) -> Self {
            Self {
                final_end: end,
                min_query: end,
                ..self.clone()
            }
        }
    }
}
