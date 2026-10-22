//! Piece 6 (`qwen4exp-p5`, was `qwen4exp-p6`): Flash-Next (`qwen4exp`) logits diagnostics owned by the Rust
//! host — the teacher-forced `--dump-logits` writer, the `STRATA_DUMP_FIRST_LOGITS` first-window dumps that
//! the first-token KLD gate reads, Rust-side teacher-forced KLD (`--logits-out` / `--kld-in` / `--kld-top`) — plus
//! the disposition of the C++-only niche flags.
//!
//! ## What the C++ engine really has (2026-10-21, `generate.cpp` at `efabf057`)
//! - There is no `--logits-out` / `--kld-in` / `--kld-top` in the C++ engine. The teacher-forced logits mode is
//!   `--dump-logits PATH [--logits-stride N]` on the per-token generate path: a file of
//!   `int32 n_vocab, int32 n_rows` followed by `n_rows` rows of `n_vocab` f32 logits, one row per selected
//!   position (`logits_selection.hpp`: multiples of the stride plus the last position; the header is written with
//!   the first row). External KL comparison tools read that file.
//! - The first-window KL gate does not use either: it drives `--serve` with
//!   `STRATA_DUMP_FIRST_LOGITS=<dir>/first`, which writes each request's first-window logits row to
//!   `<dir>/first.<n>` (raw f32, no header), and computes full-vocabulary KL(ref || run) in numpy.
//!
//! ## Ported here (Rust owns it under `TALUS_FN_RUST=1`)
//! - [`LogitsDumpWriter`]: the `--dump-logits` writer (row selection, lazy header, non-finite check, close);
//!   the C++ step loop hands the open file over through `tk_qwen4exp_p6_take_logits_dump` and Rust reads each
//!   step's row with `tk_qwen4exp_p6_step_logits` (a D2H copy of `d_logits`; the sampler does not modify it).
//! - [`FirstLogitsDump`]: `STRATA_DUMP_FIRST_LOGITS` for `--serve` (`<base>.<n>`) and generate (`<base>`).
//! - Rust-only flags, stripped from the C++ argv by [`split_rust_only_flags`]: `--logits-out PATH` (alias of
//!   `--dump-logits`), `--kld-in REF` (teacher-forced KL(REF || this run) per selected row against a
//!   `--dump-logits`-format reference, summary printed at the end), `--kld-top K` (score against the reference's
//!   top-K only, [`compute_sparse_topk_kld`]; 0 = full vocabulary, the full-vocabulary KLD formula).
//! - [`compare_first_logits_dirs`]: first-window KL directory comparison statistics in Rust.
//!
//! ## Niche flag disposition
//! See [`classify_niche_flag`] and [`NICHE_FLAGS`]: everything the production / gate configs use is ported or owned
//! by another piece; multi-device, remote, batch-slot, vision and oracle/bisect diagnostics are proposed for drop
//! with Group B (the Rust serve/generate loops already refuse them by falling back to the C++ loops).

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::qwen4exp::ffi::{P5Kernels, Qwen4ExpKernels, TkQwen4ExpGenCtx, TkQwen4ExpKldReport};
use crate::qwen4exp::skeleton::Qwen4ExpModelConfig;

// ---------------------------------------------------------------------------------------------------------------
// Niche flag disposition
// ---------------------------------------------------------------------------------------------------------------

/// Classification of legacy C++ CLI flags for Group B retirement readiness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NicheFlagDisposition {
    /// Ported to the Rust Flash-Next engine (`qwen4exp`), or a Rust-only flag.
    PortedToRust,
    /// Proposed to drop when Group B C++ code is deleted (single-APU Strix Halo scope).
    DropWithGroupB,
}

/// One niche flag, its disposition and the reason.
pub struct NicheFlag {
    pub flag: &'static str,
    pub disposition: NicheFlagDisposition,
    pub reason: &'static str,
}

const P: NicheFlagDisposition = NicheFlagDisposition::PortedToRust;
const D: NicheFlagDisposition = NicheFlagDisposition::DropWithGroupB;

/// The niche (non-production) flags of the C++ Flash-Next engine and what happens to them.
pub const NICHE_FLAGS: &[NicheFlag] = &[
    NicheFlag {
        flag: "--dump-logits",
        disposition: P,
        reason: "teacher-forced logits rows; Rust writer (kld_flags.rs)",
    },
    NicheFlag {
        flag: "--logits-stride",
        disposition: P,
        reason: "row selection of --dump-logits; Rust (kld_flags.rs)",
    },
    NicheFlag {
        flag: "--logits-out",
        disposition: P,
        reason: "Rust-only alias of --dump-logits",
    },
    NicheFlag {
        flag: "--kld-in",
        disposition: P,
        reason: "Rust-only teacher-forced KL against a --dump-logits reference",
    },
    NicheFlag {
        flag: "--kld-top",
        disposition: P,
        reason: "Rust-only sparse top-K KL for --kld-in",
    },
    NicheFlag {
        flag: "--dump-first-logits",
        disposition: P,
        reason: "env STRATA_DUMP_FIRST_LOGITS (first-token KLD gate); Rust (kld_flags.rs)",
    },
    NicheFlag {
        flag: "--expert-profile",
        disposition: P,
        reason: "production config: startup placement (Piece 1)",
    },
    NicheFlag {
        flag: "--check-logits",
        disposition: D,
        reason: "stream-token finite scan; the Rust writer checks the rows it emits",
    },
    NicheFlag {
        flag: "--layer-split",
        disposition: D,
        reason: "multi-GPU stage split; Strix Halo is one unified-memory APU",
    },
    NicheFlag {
        flag: "--split-device",
        disposition: D,
        reason: "multi-GPU stage split",
    },
    NicheFlag {
        flag: "--split-skip-if-fits",
        disposition: D,
        reason: "multi-GPU stage split",
    },
    NicheFlag {
        flag: "--gpu-stages",
        disposition: D,
        reason: "multi-GPU stage split",
    },
    NicheFlag {
        flag: "--peer-experts",
        disposition: D,
        reason: "second-GPU expert offload (P2P)",
    },
    NicheFlag {
        flag: "--peer-device",
        disposition: D,
        reason: "second-GPU expert offload (P2P)",
    },
    NicheFlag {
        flag: "--peer-slots",
        disposition: D,
        reason: "second-GPU expert offload (P2P)",
    },
    NicheFlag {
        flag: "--peer-reserve-mib",
        disposition: D,
        reason: "second-GPU expert offload (P2P)",
    },
    NicheFlag {
        flag: "--peer-prefill-rows",
        disposition: D,
        reason: "second-GPU expert offload (P2P)",
    },
    NicheFlag {
        flag: "--peer-adapt-swaps",
        disposition: D,
        reason: "second-GPU expert offload (P2P)",
    },
    NicheFlag {
        flag: "--expert-cache-device1",
        disposition: D,
        reason: "extra-GPU expert caches",
    },
    NicheFlag {
        flag: "--expert-cache-device2",
        disposition: D,
        reason: "extra-GPU expert caches",
    },
    NicheFlag {
        flag: "--expert-cache-device3",
        disposition: D,
        reason: "extra-GPU expert caches",
    },
    NicheFlag {
        flag: "--remote-experts",
        disposition: D,
        reason: "TCP remote expert worker",
    },
    NicheFlag {
        flag: "--remote-expert-opt",
        disposition: D,
        reason: "TCP remote expert worker",
    },
    NicheFlag {
        flag: "--expert-cache-remote-placement",
        disposition: D,
        reason: "TCP remote expert worker",
    },
    NicheFlag {
        flag: "--expert-profile-save",
        disposition: D,
        reason: "live heat-map writer; production reads a fixed profile",
    },
    NicheFlag {
        flag: "--expert-profile-save-every",
        disposition: D,
        reason: "live heat-map writer",
    },
    NicheFlag {
        flag: "--expert-profile-save-min",
        disposition: D,
        reason: "live heat-map writer",
    },
    NicheFlag {
        flag: "--batch",
        disposition: D,
        reason:
            "multi-slot batched serve; the Rust serve loop is single-slot (talus-server queues)",
    },
    NicheFlag {
        flag: "--batch-groups",
        disposition: D,
        reason: "multi-slot batched serve",
    },
    NicheFlag {
        flag: "--batch-mtp",
        disposition: D,
        reason: "multi-slot batched serve",
    },
    NicheFlag {
        flag: "--pipeline-windows",
        disposition: D,
        reason: "batched-serve window pipelining",
    },
    NicheFlag {
        flag: "--vision",
        disposition: D,
        reason: "vision projector; Qwen4-Exp text packs only",
    },
    NicheFlag {
        flag: "--spec-oracle",
        disposition: D,
        reason: "offline draft-oracle bisect tool",
    },
    NicheFlag {
        flag: "--spec-follow",
        disposition: D,
        reason: "offline draft-oracle bisect tool",
    },
    NicheFlag {
        flag: "--window-hashes",
        disposition: D,
        reason: "offline window bisect tool",
    },
    NicheFlag {
        flag: "--cpu-oracle-q8-0",
        disposition: D,
        reason: "CPU oracle bisect tool",
    },
    NicheFlag {
        flag: "--dump-layers",
        disposition: D,
        reason: "C1 oracle residual dump; superseded by TALUS_FN_DUMP",
    },
    NicheFlag {
        flag: "--dump-halves",
        disposition: D,
        reason: "bisect dump; superseded by TALUS_FN_DUMP",
    },
    NicheFlag {
        flag: "--dump-mixed",
        disposition: D,
        reason: "bisect dump; superseded by TALUS_FN_DUMP",
    },
    NicheFlag {
        flag: "--dump-residual",
        disposition: D,
        reason: "bisect dump; superseded by TALUS_FN_DUMP",
    },
    NicheFlag {
        flag: "--dump-final-r",
        disposition: D,
        reason: "bisect dump; superseded by TALUS_FN_DUMP",
    },
    NicheFlag {
        flag: "--dump-routing",
        disposition: D,
        reason: "routing trace for profile building (offline)",
    },
    NicheFlag {
        flag: "--dump-model-description",
        disposition: D,
        reason: "metadata print; talus-gguf covers it",
    },
];

/// Returns the disposition of a Flash-Next niche CLI flag (`None` = a regular engine flag).
pub fn classify_niche_flag(flag: &str) -> Option<NicheFlagDisposition> {
    NICHE_FLAGS
        .iter()
        .find(|f| f.flag == flag)
        .map(|f| f.disposition)
}

// ---------------------------------------------------------------------------------------------------------------
// Rust-only flags
// ---------------------------------------------------------------------------------------------------------------

/// Rust-only teacher-forced KLD options (`--kld-in`, `--kld-top`; `--logits-out` becomes `--dump-logits`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KldCliOpts {
    pub kld_in: Option<PathBuf>,
    pub kld_top: usize,
}

/// Rewrites `--logits-out P` to `--dump-logits P` and removes `--kld-in R` / `--kld-top K`, which the C++
/// argument parser does not know. Returns the argv for the C++ side and the Rust options.
pub fn split_rust_only_flags<S: AsRef<str>>(
    argv: &[S],
) -> Result<(Vec<String>, KldCliOpts), String> {
    let mut out = Vec::with_capacity(argv.len());
    let mut opts = KldCliOpts::default();
    let mut i = 0usize;
    while i < argv.len() {
        let a = argv[i].as_ref();
        let val = |i: usize| -> Result<String, String> {
            argv.get(i + 1)
                .map(|v| v.as_ref().to_string())
                .ok_or_else(|| format!("{a} needs a value"))
        };
        match a {
            "--logits-out" => {
                out.push("--dump-logits".to_string());
                out.push(val(i)?);
                i += 2;
            }
            "--kld-in" => {
                opts.kld_in = Some(PathBuf::from(val(i)?));
                i += 2;
            }
            "--kld-top" => {
                let v = val(i)?;
                opts.kld_top = v
                    .parse()
                    .map_err(|_| format!("--kld-top: bad count '{v}'"))?;
                i += 2;
            }
            _ => {
                out.push(a.to_string());
                i += 1;
            }
        }
    }
    Ok((out, opts))
}

// ---------------------------------------------------------------------------------------------------------------
// --dump-logits row selection (include/strata/program/logits_selection.hpp)
// ---------------------------------------------------------------------------------------------------------------

/// Positive decimal stride, no sign, whitespace or suffix (`logits_selection::parse_stride`).
pub fn parse_logits_stride(text: &str) -> Option<i64> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse::<i64>().ok().filter(|&v| v > 0)
}

/// Zero-based multiples of `stride`, then the final position once (`logits_selection::selected`).
pub const fn logits_row_selected(position: i64, total: i64, stride: i64) -> bool {
    total > 0
        && stride > 0
        && position >= 0
        && position < total
        && (position % stride == 0 || position == total - 1)
}

/// Rows written for `total` positions (`logits_selection::row_count`).
pub const fn logits_row_count(total: i64, stride: i64) -> i64 {
    if total > 0 && stride > 0 {
        (total - 1) / stride + 1 + if (total - 1) % stride != 0 { 1 } else { 0 }
    } else {
        0
    }
}

fn f32_bytes(row: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; the slice covers exactly `len * 4` initialized bytes.
    unsafe { std::slice::from_raw_parts(row.as_ptr().cast::<u8>(), std::mem::size_of_val(row)) }
}

/// The `--dump-logits` file writer: header `[n_vocab, n_rows]` (int32) written with the first row, then rows.
pub struct LogitsDumpWriter {
    path: String,
    file: Option<BufWriter<File>>,
    hdr: [i32; 2],
    hdr_written: bool,
    pub stride: i64,
    pub positions: i64,
    pub rows: i64,
}

impl LogitsDumpWriter {
    /// Creates (truncates) `path` like the C++ `fopen(path, "wb")` at startup.
    pub fn create(path: &str, n_vocab: i64, stride: i64, positions: i64) -> Result<Self, String> {
        if positions > i32::MAX as i64 || n_vocab > i32::MAX as i64 {
            return Err("strata generate: logits dump dimensions exceed int32".to_string());
        }
        let f = File::create(path).map_err(|_| format!("strata generate: cannot write {path}"))?;
        Ok(Self {
            path: path.to_string(),
            file: Some(BufWriter::with_capacity(1 << 20, f)),
            hdr: [n_vocab as i32, logits_row_count(positions, stride) as i32],
            hdr_written: false,
            stride,
            positions,
            rows: 0,
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn selected(&self, pos: i64) -> bool {
        self.file.is_some() && logits_row_selected(pos, self.positions, self.stride)
    }

    /// Writes one row (header first). Errors carry the C++ messages.
    pub fn write_row(&mut self, pos: i64, row: &[f32]) -> Result<(), String> {
        let Some(f) = self.file.as_mut() else {
            return Ok(());
        };
        if !self.hdr_written {
            let h = [self.hdr[0].to_ne_bytes(), self.hdr[1].to_ne_bytes()].concat();
            if f.write_all(&h).is_err() {
                self.file = None;
                return Err("strata generate: cannot write logits header".to_string());
            }
            self.hdr_written = true;
        }
        if f.write_all(f32_bytes(row)).is_err() {
            self.file = None;
            return Err(format!(
                "strata generate: cannot write logits at position {pos}"
            ));
        }
        self.rows += 1;
        Ok(())
    }

    /// Flushes and closes the file (`fclose` at finish).
    pub fn finish(&mut self) -> Result<(), String> {
        if let Some(f) = self.file.take() {
            f.into_inner()
                .map_err(|_| "strata generate: cannot finish logits dump".to_string())?
                .sync_all()
                .ok();
        }
        Ok(())
    }
}

/// Counts the non-finite entries of a logits row (the C++ `--check-logits` / readback check).
pub fn count_nonfinite(row: &[f32]) -> usize {
    row.iter().filter(|v| !v.is_finite()).count()
}

// ---------------------------------------------------------------------------------------------------------------
// STRATA_DUMP_FIRST_LOGITS (first-token KLD gate)
// ---------------------------------------------------------------------------------------------------------------

/// `STRATA_DUMP_FIRST_LOGITS=<base>`: the logits row of each request's first window (raw f32, no header).
/// `--serve` writes `<base>.<n>` with a process-wide counter; generate writes `<base>`.
#[derive(Clone, Debug, Default)]
pub struct FirstLogitsDump {
    pub base: Option<String>,
    pub seq: usize,
}

impl FirstLogitsDump {
    pub fn from_env() -> Self {
        Self {
            base: std::env::var("STRATA_DUMP_FIRST_LOGITS").ok(),
            seq: 0,
        }
    }

    pub fn enabled(&self) -> bool {
        self.base.is_some()
    }

    /// Writes the first-window row; `copy` fills the row (`serve_copy_logits` / `gen_spec_copy_logits`, row 0)
    /// and returns false when no row is available. `serve` selects `<base>.<n>` naming and the message prefix.
    pub fn write(&mut self, serve: bool, n_vocab: usize, copy: impl FnOnce(&mut [f32]) -> bool) {
        let Some(base) = self.base.as_ref() else {
            return;
        };
        let path = if serve {
            let p = format!("{base}.{}", self.seq);
            self.seq += 1;
            p
        } else {
            base.clone()
        };
        let mut row = vec![0.0f32; n_vocab];
        let ok = copy(&mut row) && std::fs::write(&path, f32_bytes(&row)).is_ok();
        if !ok {
            eprintln!(
                "strata {}: STRATA_DUMP_FIRST_LOGITS: cannot write {path}",
                if serve { "serve" } else { "generate" }
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// KL divergence
// ---------------------------------------------------------------------------------------------------------------

/// Log-softmax in f64 (subtract the max, then the log of the sum of exponentials).
pub fn log_softmax_f64(x: &[f32]) -> Vec<f64> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut out: Vec<f64> = x.iter().map(|&v| v as f64 - m).collect();
    let z: f64 = out.iter().map(|v| v.exp()).sum::<f64>().ln();
    for v in &mut out {
        *v -= z;
    }
    out
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

/// Full-vocabulary KL(ref || run) and whether the top-1 tokens agree.
pub fn compute_full_kld(ref_logits: &[f32], run_logits: &[f32]) -> (f64, bool) {
    let lr = log_softmax_f64(ref_logits);
    let la = log_softmax_f64(run_logits);
    let kl = lr
        .iter()
        .zip(la.iter())
        .map(|(&r, &a)| r.exp() * (r - a))
        .sum::<f64>();
    (kl, argmax(ref_logits) == argmax(run_logits))
}

/// Pure-Rust sparse top-K KLD: reference top-K `(token_id, log_prob)` pairs (descending), the student's full
/// logits; the reference tail mass is matched against the student's mass outside the top-K.
/// `KL(P_ref || P_stu) = sum_i p_ref[i] * (log_p_ref[i] - log_p_stu[id_i]) + p_tail * log(p_tail / q_tail)`.
pub fn compute_sparse_topk_kld(
    ref_ids: &[i32],
    ref_logprobs: &[f32],
    student_logits: &[f32],
) -> (f64, bool) {
    if ref_ids.is_empty() || ref_ids.len() != ref_logprobs.len() || student_logits.is_empty() {
        return (0.0, false);
    }
    let max_logit = student_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f64;
    for &x in student_logits {
        sum_exp += ((x - max_logit) as f64).exp();
    }
    let log_z = (max_logit as f64) + sum_exp.ln();
    let top1_match = (argmax(student_logits) as i32) == ref_ids[0];

    let mut kld = 0.0f64;
    let mut p_sum = 0.0f64;
    let mut q_sum = 0.0f64;
    for (&id, &lp_ref) in ref_ids.iter().zip(ref_logprobs.iter()) {
        if id < 0 || (id as usize) >= student_logits.len() {
            continue;
        }
        let p = (lp_ref as f64).exp();
        let lp_stu = (student_logits[id as usize] as f64) - log_z;
        p_sum += p;
        q_sum += lp_stu.exp();
        if p > 0.0 {
            kld += p * ((lp_ref as f64) - lp_stu);
        }
    }
    let p_tail = (1.0 - p_sum).max(0.0);
    let q_tail = (1.0 - q_sum).max(1e-12);
    if p_tail > 1e-9 {
        kld += p_tail * (p_tail / q_tail).ln();
    }
    (kld.max(0.0), top1_match)
}

/// Reference top-K `(ids, log-probs)` of a full logits row, descending.
pub fn reference_topk(ref_logits: &[f32], k: usize) -> (Vec<i32>, Vec<f32>) {
    let lp = log_softmax_f64(ref_logits);
    let mut idx: Vec<usize> = (0..ref_logits.len()).collect();
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
        lp[b]
            .partial_cmp(&lp[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.sort_by(|&a, &b| {
        lp[b]
            .partial_cmp(&lp[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    (
        idx.iter().map(|&i| i as i32).collect(),
        idx.iter().map(|&i| lp[i] as f32).collect(),
    )
}

/// Per-position KL values and top-1 agreement summary
/// (`p99 = sorted[int(0.99 * (n - 1))]`; median and p90 the same way).
#[derive(Clone, Debug, Default)]
pub struct KldSummary {
    pub kls: Vec<f64>,
    pub agree: usize,
}

impl KldSummary {
    pub fn push(&mut self, kl: f64, agree: bool) {
        self.kls.push(kl);
        self.agree += agree as usize;
    }

    pub fn report(&self) -> TkQwen4ExpKldReport {
        let n = self.kls.len();
        if n == 0 {
            return TkQwen4ExpKldReport::default();
        }
        let mut k = self.kls.clone();
        k.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q = |f: f64| k[(f * (n - 1) as f64) as usize];
        let top1 = self.agree as f64 / n as f64;
        TkQwen4ExpKldReport {
            positions_scored: n as i64,
            mean_kld: k.iter().sum::<f64>() / n as f64,
            median_kld: q(0.5),
            p90_kld: q(0.9),
            p99_kld: q(0.99),
            max_kld: k[n - 1],
            top1_agreement: top1,
            flip_rate: 1.0 - top1,
        }
    }
}

/// Format a [`TkQwen4ExpKldReport`] summary line.
pub fn format_kld_report(name: &str, r: &TkQwen4ExpKldReport) -> String {
    format!(
        "{name}: {} positions, KL mean {:.5} median {:.5} p90 {:.5} p99 {:.5} max {:.5}, top-1 agree {:.1}%",
        r.positions_scored,
        r.mean_kld,
        r.median_kld,
        r.p90_kld,
        r.p99_kld,
        r.max_kld,
        100.0 * r.top1_agreement
    )
}

fn read_f32_file(path: &Path) -> Option<Vec<f32>> {
    let b = std::fs::read(path).ok()?;
    Some(
        b.chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Compare `first.<i>` logits dumps in `ref_dir` vs `run_dir` in Rust: KL(REF || RUN) over matching rows.
pub fn compare_first_logits_dirs(
    ref_dir: &Path,
    run_dir: &Path,
) -> Result<TkQwen4ExpKldReport, String> {
    let mut s = KldSummary::default();
    let mut i = 0usize;
    let mut misses = 0usize;
    while misses < 64 {
        let (r, a) = (
            ref_dir.join(format!("first.{i}")),
            run_dir.join(format!("first.{i}")),
        );
        i += 1;
        let (Some(lr), Some(la)) = (read_f32_file(&r), read_f32_file(&a)) else {
            misses += 1;
            continue;
        };
        misses = 0;
        if lr.len() != la.len() || lr.is_empty() {
            return Err(format!(
                "first.{}: row sizes differ ({} vs {})",
                i - 1,
                lr.len(),
                la.len()
            ));
        }
        let (kl, agree) = compute_full_kld(&lr, &la);
        s.push(kl, agree);
    }
    Ok(s.report())
}

/// Streams rows of a `--dump-logits`-format reference file for `--kld-in`.
pub struct KldReference {
    file: std::io::BufReader<File>,
    pub n_vocab: usize,
    pub n_rows: usize,
    read: usize,
    top_k: usize,
    pub summary: KldSummary,
    row: Vec<f32>,
    bytes: Vec<u8>,
}

impl KldReference {
    pub fn open(path: &Path, n_vocab: usize, top_k: usize) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("--kld-in {}: {e}", path.display()))?;
        let mut file = std::io::BufReader::with_capacity(1 << 20, f);
        let mut h = [0u8; 8];
        file.read_exact(&mut h)
            .map_err(|e| format!("--kld-in {}: header: {e}", path.display()))?;
        let nv = i32::from_ne_bytes([h[0], h[1], h[2], h[3]]);
        let nr = i32::from_ne_bytes([h[4], h[5], h[6], h[7]]);
        if nv as i64 != n_vocab as i64 || nr < 0 {
            return Err(format!(
                "--kld-in {}: header says {nv} x {nr}, the model's vocabulary is {n_vocab}",
                path.display()
            ));
        }
        Ok(Self {
            file,
            n_vocab,
            n_rows: nr as usize,
            read: 0,
            top_k,
            summary: KldSummary::default(),
            row: vec![0.0; n_vocab],
            bytes: vec![0u8; n_vocab * 4],
        })
    }

    /// Scores the next row of this run against the next reference row; false once the reference is exhausted.
    pub fn score_next(&mut self, run_row: &[f32]) -> Result<bool, String> {
        if self.read >= self.n_rows {
            return Ok(false);
        }
        self.file
            .read_exact(&mut self.bytes)
            .map_err(|e| format!("--kld-in: row {}: {e}", self.read))?;
        for (d, c) in self.row.iter_mut().zip(self.bytes.chunks_exact(4)) {
            *d = f32::from_ne_bytes([c[0], c[1], c[2], c[3]]);
        }
        self.read += 1;
        let (kl, agree) = if self.top_k > 0 {
            let (ids, lps) = reference_topk(&self.row, self.top_k);
            compute_sparse_topk_kld(&ids, &lps, run_row)
        } else {
            compute_full_kld(&self.row, run_row)
        };
        self.summary.push(kl, agree);
        Ok(true)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Piece 6 state
// ---------------------------------------------------------------------------------------------------------------

/// Piece 6 logits diagnostics owned by `Qwen4ExpRustEngine`.
pub struct Piece6KldFlags {
    pub kld_top_k: i32,
    pub opts: KldCliOpts,
    pub first_logits: FirstLogitsDump,
    pub dump: Option<LogitsDumpWriter>,
    pub kld_ref: Option<KldReference>,
    /// Positions/stride scored by `--kld-in` when no `--dump-logits` file is open.
    kld_positions: i64,
    kld_stride: i64,
    n_vocab: usize,
    row: Vec<f32>,
    pub last_report: TkQwen4ExpKldReport,
}

impl Clone for Piece6KldFlags {
    fn clone(&self) -> Self {
        Self {
            kld_top_k: self.kld_top_k,
            opts: self.opts.clone(),
            first_logits: self.first_logits.clone(),
            dump: None,
            kld_ref: None,
            kld_positions: 0,
            kld_stride: 1,
            n_vocab: self.n_vocab,
            row: Vec::new(),
            last_report: self.last_report,
        }
    }
}

impl std::fmt::Debug for Piece6KldFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Piece6KldFlags")
            .field("opts", &self.opts)
            .field("first_logits", &self.first_logits)
            .field("dump", &self.dump.as_ref().map(|d| d.path().to_string()))
            .field("last_report", &self.last_report)
            .finish()
    }
}

impl Piece6KldFlags {
    pub fn new(_cfg: &Qwen4ExpModelConfig) -> Self {
        Self {
            kld_top_k: 0,
            opts: KldCliOpts::default(),
            first_logits: FirstLogitsDump::from_env(),
            dump: None,
            kld_ref: None,
            kld_positions: 0,
            kld_stride: 1,
            n_vocab: 0,
            row: Vec::new(),
            last_report: TkQwen4ExpKldReport::default(),
        }
    }

    /// Stores the Rust-only options parsed by [`split_rust_only_flags`].
    pub fn set_opts(&mut self, opts: KldCliOpts) {
        self.kld_top_k = opts.kld_top as i32;
        self.opts = opts;
    }

    /// Generate start (before the prompt): takes the `--dump-logits` file from the C++ step loop and opens the
    /// `--kld-in` reference. `n_prompt` / `max_new` give the scored positions when no dump file is open.
    pub unsafe fn gen_begin(
        &mut self,
        p5k: Option<&P5Kernels>,
        ctx: *mut TkQwen4ExpGenCtx,
        n_vocab: i64,
        n_prompt: i64,
        max_new: i64,
    ) -> Result<(), String> {
        self.n_vocab = n_vocab.max(0) as usize;
        let Some(p5k) = p5k else {
            return Ok(());
        };
        let mut path = vec![0u8; 4096];
        let (mut stride, mut positions) = (1i64, 0i64);
        let rc = (p5k.p6_take_logits_dump)(
            ctx,
            path.as_mut_ptr().cast(),
            path.len(),
            &mut stride,
            &mut positions,
        );
        if rc < 0 {
            return Err("tk_qwen4exp_p6_take_logits_dump failed".to_string());
        }
        if rc == 1 {
            let end = path.iter().position(|&b| b == 0).unwrap_or(path.len());
            let p = String::from_utf8_lossy(&path[..end]).into_owned();
            self.dump = Some(LogitsDumpWriter::create(&p, n_vocab, stride, positions)?);
        }
        if let Some(r) = self.opts.kld_in.clone() {
            self.kld_ref = Some(KldReference::open(&r, self.n_vocab, self.opts.kld_top)?);
            self.kld_positions = (n_prompt - 1 + max_new).max(0);
            self.kld_stride = 1;
        }
        Ok(())
    }

    fn wants_row(&self, pos: i64) -> bool {
        match (&self.dump, &self.kld_ref) {
            (Some(d), _) => d.selected(pos),
            (None, Some(_)) => logits_row_selected(pos, self.kld_positions, self.kld_stride),
            (None, None) => false,
        }
    }

    /// After each single-token step at `pos`: reads the row when selected, checks it, writes and scores it.
    pub unsafe fn gen_after_step(
        &mut self,
        p5k: Option<&P5Kernels>,
        ctx: *mut TkQwen4ExpGenCtx,
        pos: i64,
    ) -> Result<(), String> {
        if !self.wants_row(pos) {
            return Ok(());
        }
        let Some(p5k) = p5k else {
            return Ok(());
        };
        self.row.resize(self.n_vocab, 0.0);
        if (p5k.p6_step_logits)(ctx, self.row.as_mut_ptr()) < 0 {
            return Err("strata generate: reading the logits back failed".to_string());
        }
        let bad = count_nonfinite(&self.row);
        if bad != 0 {
            return Err(format!(
                "strata generate: {bad} of {} logits are not finite at position {pos}",
                self.n_vocab
            ));
        }
        if let Some(d) = self.dump.as_mut() {
            d.write_row(pos, &self.row)?;
        }
        if let Some(r) = self.kld_ref.as_mut() {
            r.score_next(&self.row)?;
        }
        Ok(())
    }

    /// Generate end: closes the dump (before the C++ finish prints "logits dumped") and reports `--kld-in`.
    pub fn gen_finish(&mut self) -> Result<(), String> {
        if let Some(mut d) = self.dump.take() {
            d.finish()?;
        }
        Ok(())
    }

    /// Prints the `--kld-in` summary (after the C++ finish table).
    pub fn gen_report(&mut self) {
        if let Some(r) = self.kld_ref.take() {
            let rep = r.summary.report();
            self.last_report = rep;
            let name = format!(
                "kld vs {}{}",
                self.opts
                    .kld_in
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default(),
                if self.opts.kld_top > 0 {
                    format!(" (top-{})", self.opts.kld_top)
                } else {
                    String::new()
                }
            );
            if rep.positions_scored == 0 {
                eprintln!("strata generate: --kld-in scored no positions (a native pack or batched prefill skips the per-token path)");
            }
            println!("{}", format_kld_report(&name, &rep));
        }
    }

    /// Legacy coarse ABI (`tk_qwen4exp_p6_kld_eval`, never wired on the C++ side): returns the last `--kld-in`
    /// report computed in Rust.
    pub unsafe fn run_kld_eval(
        &mut self,
        _kernels: &Qwen4ExpKernels,
        _ctx: *mut TkQwen4ExpGenCtx,
    ) -> Result<TkQwen4ExpKldReport, String> {
        Ok(self.last_report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_topk_kld_is_zero_on_identical_distribution() {
        let logits = vec![2.0f32, 1.0, 0.0, -1.0];
        let (ids, lps) = reference_topk(&logits, 4);
        assert_eq!(ids, vec![0, 1, 2, 3]);
        let (kld, top1) = compute_sparse_topk_kld(&ids, &lps, &logits);
        assert!(top1);
        assert!(kld.abs() < 1e-6, "expected ~0 KLD, got {kld}");
        let (kl2, _) = compute_sparse_topk_kld(&ids[..2], &lps[..2], &logits);
        assert!(kl2.abs() < 1e-6, "top-2 with matched tail, got {kl2}");
    }

    #[test]
    fn full_kld_matches_closed_form() {
        let a = vec![0.0f32, 0.0];
        let b = vec![(3.0f32).ln(), 0.0]; // q = [3/4, 1/4]
        let (kl, agree) = compute_full_kld(&a, &b);
        let want = 0.5 * (0.5f64 / 0.75).ln() + 0.5 * (0.5f64 / 0.25).ln();
        assert!((kl - want).abs() < 1e-6, "{kl} vs {want}");
        assert!(!agree || argmax(&a) == argmax(&b));
        assert_eq!(compute_full_kld(&b, &b).0, 0.0);
    }

    #[test]
    fn logits_selection_matches_cpp() {
        // stride 1: every position; the C++ count for 5-token prompt + max_new 6 is 10 rows
        assert_eq!(logits_row_count(10, 1), 10);
        assert_eq!(logits_row_count(10, 4), 4); // 0, 4, 8, 9
        assert_eq!(logits_row_count(9, 4), 3); // 0, 4, 8
        let sel: Vec<i64> = (0..10).filter(|&p| logits_row_selected(p, 10, 4)).collect();
        assert_eq!(sel, vec![0, 4, 8, 9]);
        assert_eq!(parse_logits_stride("4"), Some(4));
        assert_eq!(parse_logits_stride("+4"), None);
        assert_eq!(parse_logits_stride("0"), None);
        assert_eq!(parse_logits_stride("4x"), None);
    }

    #[test]
    fn summary_percentiles_follow_first_token_kld() {
        let mut s = KldSummary::default();
        for i in 0..101 {
            s.push(i as f64, i % 2 == 0);
        }
        let r = s.report();
        assert_eq!(r.positions_scored, 101);
        assert_eq!(r.p99_kld, 99.0);
        assert_eq!(r.median_kld, 50.0);
        assert_eq!(r.max_kld, 100.0);
        assert!((r.mean_kld - 50.0).abs() < 1e-12);
    }

    #[test]
    fn rust_only_flags_are_split() {
        let argv = [
            "talus-cli",
            "--pack",
            "x",
            "--logits-out",
            "a.bin",
            "--kld-in",
            "r.bin",
            "--kld-top",
            "64",
        ];
        let (out, o) = split_rust_only_flags(&argv).unwrap();
        assert_eq!(
            out,
            vec!["talus-cli", "--pack", "x", "--dump-logits", "a.bin"]
        );
        assert_eq!(o.kld_in, Some(PathBuf::from("r.bin")));
        assert_eq!(o.kld_top, 64);
        assert!(split_rust_only_flags(&["x", "--kld-in"]).is_err());
    }

    #[test]
    fn dump_writer_and_reference_round_trip() {
        let dir = std::env::temp_dir().join(format!("p6-dump-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("l.bin");
        let mut w = LogitsDumpWriter::create(p.to_str().unwrap(), 3, 1, 2).unwrap();
        w.write_row(0, &[1.0, 2.0, 3.0]).unwrap();
        w.write_row(1, &[0.0, 0.0, 9.0]).unwrap();
        w.finish().unwrap();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(b.len(), 8 + 2 * 12);
        let mut r = KldReference::open(&p, 3, 0).unwrap();
        assert!(r.score_next(&[1.0, 2.0, 3.0]).unwrap());
        assert!(r.score_next(&[0.0, 0.0, 9.0]).unwrap());
        assert!(!r.score_next(&[0.0, 0.0, 9.0]).unwrap());
        let rep = r.summary.report();
        assert_eq!(rep.positions_scored, 2);
        assert_eq!(rep.max_kld, 0.0);
        assert_eq!(rep.top1_agreement, 1.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn niche_flag_classification_matches_group_b_plan() {
        assert_eq!(
            classify_niche_flag("--logits-out"),
            Some(NicheFlagDisposition::PortedToRust)
        );
        assert_eq!(
            classify_niche_flag("--kld-in"),
            Some(NicheFlagDisposition::PortedToRust)
        );
        assert_eq!(
            classify_niche_flag("--dump-logits"),
            Some(NicheFlagDisposition::PortedToRust)
        );
        assert_eq!(
            classify_niche_flag("--layer-split"),
            Some(NicheFlagDisposition::DropWithGroupB)
        );
        assert_eq!(
            classify_niche_flag("--remote-experts"),
            Some(NicheFlagDisposition::DropWithGroupB)
        );
        assert_eq!(classify_niche_flag("--spec"), None);
    }
}
