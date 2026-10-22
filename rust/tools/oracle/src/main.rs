//! Oracle verification, diff harness, and bitflip injection CLI (`talus-oracle` / `oracle`).

#![forbid(unsafe_code)]
#![allow(clippy::too_many_lines)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use oracle::{
    assemble_strix_recordings, diff_dump_dirs, inject_bitflip, verify_dump_dir, BitflipOptions,
    BitflipTarget, DiffConfig, DiffMode, OracleError, EXIT_ERROR, EXIT_PASS,
};

const USAGE: &str = "\
Usage:
  talus-oracle verify <dump-dir> [--json]
  talus-oracle diff <ref-dir> <cand-dir> [options]
  talus-oracle --golden <ref-dir> --candidate <cand-dir> [options]
  talus-oracle inject-bitflip <src-dir> <dst-dir> --target <tokens|logits|residuals> [options]
  talus-oracle assemble <raw-dir> <oracle-dir>

Diff options:
  --prompt <ID>          Filter to a single prompt ID
  --mode <MODE>          Comparison mode: tokens|logits|residuals|all (default: all)
  --ulp, --max-ulp <N>   Max ULP difference allowed (default: 0)
  --abs-tol, --max-abs <F>
                         Max absolute f32 difference allowed (default: 0.0)
  --rel-tol <F>          Max relative f32 difference allowed (default: 0.0)
  --max-rel-l2 <F>       Max relative L2 norm error allowed per row/snapshot
  --min-cosine <F>       Minimum cosine similarity allowed per row/snapshot
  --report <PATH>        Write JSON report to file
  --json                 Print JSON report to stdout

Inject-bitflip options:
  --target <TARGET>      Artifact to mutate: tokens|logits|residuals
  --prompt <ID>          Prompt ID to mutate (default: first prompt in manifest)
  --step <N>             Generated token step index for --target tokens (default: 0)
  --row <N>              Logit row index for --target logits (default: 0)
  --vocab <N>            Vocab index for --target logits (default: 0)
  --snapshot <N>         Snapshot index for --target residuals
  --pos <N>              Token position filter for --target residuals
  --layer <N>            Layer index filter for --target residuals
  --tap <TAP>            Tap filter (embd_out|attn_out|ffn_out|final_norm_out)
  --dim, --index <N>     Dimension/element index (default: 0)
  --bit <0..31>          Bit position to XOR (default: 0)
  --json                 Print JSON summary to stdout
";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match run_cli(&args) {
        Ok(code) => ExitCode::from(u8::try_from(code).unwrap_or(2)),
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(u8::try_from(EXIT_ERROR).unwrap_or(2))
        }
    }
}

fn next_arg<'a>(args: &'a [String], idx: &mut usize, flag: &str) -> Result<&'a str, OracleError> {
    *idx += 1;
    args.get(*idx)
        .map(String::as_str)
        .ok_or_else(|| OracleError::InvalidArg(format!("{flag} requires a value")))
}

fn parse_num<T: std::str::FromStr>(val: &str, flag: &str) -> Result<T, OracleError>
where
    T::Err: std::fmt::Display,
{
    val.parse::<T>()
        .map_err(|e| OracleError::InvalidArg(format!("invalid value {val:?} for {flag}: {e}")))
}

pub fn run_cli(args: &[String]) -> Result<i32, OracleError> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(if args.is_empty() {
            EXIT_ERROR
        } else {
            EXIT_PASS
        });
    }

    match args[0].as_str() {
        "verify" => run_verify(&args[1..]),
        "diff" => run_diff(&args[1..]),
        "inject-bitflip" => run_inject_bitflip(&args[1..]),
        "assemble" => run_assemble(&args[1..]),
        first if first.starts_with("--") => run_diff(args),
        other => Err(OracleError::InvalidArg(format!(
            "unknown subcommand {other:?}\n\n{USAGE}"
        ))),
    }
}

fn run_assemble(args: &[String]) -> Result<i32, OracleError> {
    if args.len() != 2 {
        return Err(OracleError::InvalidArg(
            "assemble requires <raw-dir> <oracle-dir>".into(),
        ));
    }
    let raw_dir = PathBuf::from(&args[0]);
    let oracle_dir = PathBuf::from(&args[1]);
    assemble_strix_recordings(&raw_dir, &oracle_dir)?;
    println!("OK: assembled golden dumps in {}", oracle_dir.display());
    Ok(EXIT_PASS)
}

fn run_verify(args: &[String]) -> Result<i32, OracleError> {
    let mut dump_dir: Option<PathBuf> = None;
    let mut json_out = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => json_out = true,
            "--golden" | "--dir" => {
                let flag = args[i].as_str();
                dump_dir = Some(PathBuf::from(next_arg(args, &mut i, flag)?));
            }
            arg if !arg.starts_with('-') && dump_dir.is_none() => {
                dump_dir = Some(PathBuf::from(arg));
            }
            other => {
                return Err(OracleError::InvalidArg(format!(
                    "unexpected verify argument {other:?}"
                )));
            }
        }
        i += 1;
    }
    let dir =
        dump_dir.ok_or_else(|| OracleError::InvalidArg("verify requires <dump-dir>".into()))?;
    let summary = verify_dump_dir(&dir)?;
    if json_out {
        let s = serde_json::to_string_pretty(&summary).map_err(|e| OracleError::json(&dir, e))?;
        println!("{s}");
    } else {
        println!(
            "VERIFY: PASS  dir={}  model={}  prompts={}  checksums={}  tokens={}  logit_rows={}  residual_snapshots={}",
            summary.dump_dir,
            summary.model_id,
            summary.prompts_verified,
            summary.checksums_verified,
            summary.total_tokens_verified,
            summary.total_logit_rows_verified,
            summary.total_residual_snapshots_verified
        );
    }
    Ok(EXIT_PASS)
}

fn run_diff(args: &[String]) -> Result<i32, OracleError> {
    let mut ref_dir: Option<PathBuf> = None;
    let mut cand_dir: Option<PathBuf> = None;
    let mut cfg = DiffConfig::default();
    let mut json_out = false;
    let mut report_path: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--golden" | "--ref" => {
                ref_dir = Some(PathBuf::from(next_arg(args, &mut i, flag)?));
            }
            "--candidate" | "--cand" => {
                cand_dir = Some(PathBuf::from(next_arg(args, &mut i, flag)?));
            }
            "--prompt" => {
                cfg.prompt_filter = Some(next_arg(args, &mut i, flag)?.to_string());
            }
            "--mode" => {
                cfg.mode = DiffMode::parse(next_arg(args, &mut i, flag)?)?;
            }
            "--ulp" | "--max-ulp" => {
                cfg.max_ulp = parse_num(next_arg(args, &mut i, flag)?, flag)?;
                cfg.ulp_explicit = true;
            }
            "--abs-tol" | "--max-abs" => {
                cfg.abs_tol = parse_num(next_arg(args, &mut i, flag)?, flag)?;
            }
            "--rel-tol" => {
                cfg.rel_tol = parse_num(next_arg(args, &mut i, flag)?, flag)?;
            }
            "--max-rel-l2" => {
                cfg.max_rel_l2 = Some(parse_num(next_arg(args, &mut i, flag)?, flag)?);
            }
            "--min-cosine" | "--cosine-min" => {
                cfg.min_cosine = Some(parse_num(next_arg(args, &mut i, flag)?, flag)?);
            }
            "--report" => {
                report_path = Some(PathBuf::from(next_arg(args, &mut i, flag)?));
            }
            "--json" => {
                json_out = true;
            }
            arg if !arg.starts_with('-') => {
                if ref_dir.is_none() {
                    ref_dir = Some(PathBuf::from(arg));
                } else if cand_dir.is_none() {
                    cand_dir = Some(PathBuf::from(arg));
                } else {
                    return Err(OracleError::InvalidArg(format!(
                        "unexpected positional argument {arg:?}"
                    )));
                }
            }
            other => {
                return Err(OracleError::InvalidArg(format!(
                    "unknown diff option {other:?}"
                )));
            }
        }
        i += 1;
    }

    let golden = ref_dir
        .ok_or_else(|| OracleError::InvalidArg("diff requires <ref-dir> (or --golden)".into()))?;
    let candidate = cand_dir.ok_or_else(|| {
        OracleError::InvalidArg("diff requires <cand-dir> (or --candidate)".into())
    })?;

    let report = diff_dump_dirs(&golden, &candidate, &cfg)?;
    if let Some(rp) = &report_path {
        let json_str =
            serde_json::to_string_pretty(&report).map_err(|e| OracleError::json(rp, e))?;
        fs::write(rp, format!("{json_str}\n")).map_err(|e| OracleError::io(rp, e))?;
    }
    if json_out {
        let json_str =
            serde_json::to_string_pretty(&report).map_err(|e| OracleError::json(&golden, e))?;
        println!("{json_str}");
    } else {
        print!("{}", report.format_human());
    }
    Ok(report.exit_code())
}

fn run_inject_bitflip(args: &[String]) -> Result<i32, OracleError> {
    let mut src_dir: Option<PathBuf> = None;
    let mut dst_dir: Option<PathBuf> = None;
    let mut target: Option<BitflipTarget> = None;
    let mut opts = BitflipOptions::default();
    let mut json_out = false;

    let mut i = 0usize;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--target" => {
                let t = BitflipTarget::parse(next_arg(args, &mut i, flag)?)?;
                target = Some(t);
                opts.target = t;
            }
            "--prompt" => {
                opts.prompt_id = Some(next_arg(args, &mut i, flag)?.to_string());
            }
            "--step" => {
                opts.step = parse_num(next_arg(args, &mut i, flag)?, flag)?;
            }
            "--row" => {
                opts.row_idx = parse_num(next_arg(args, &mut i, flag)?, flag)?;
            }
            "--vocab" => {
                opts.vocab_idx = parse_num(next_arg(args, &mut i, flag)?, flag)?;
            }
            "--snapshot" => {
                opts.snapshot_idx = Some(parse_num(next_arg(args, &mut i, flag)?, flag)?);
            }
            "--pos" => {
                opts.token_pos = Some(parse_num(next_arg(args, &mut i, flag)?, flag)?);
            }
            "--layer" => {
                opts.layer_idx = Some(parse_num(next_arg(args, &mut i, flag)?, flag)?);
            }
            "--tap" => {
                opts.tap = Some(next_arg(args, &mut i, flag)?.to_string());
            }
            "--dim" | "--index" => {
                let val: usize = parse_num(next_arg(args, &mut i, flag)?, flag)?;
                opts.dim_idx = val;
                opts.vocab_idx = val;
                opts.step = val;
            }
            "--bit" => {
                opts.bit = parse_num(next_arg(args, &mut i, flag)?, flag)?;
            }
            "--json" => {
                json_out = true;
            }
            arg if !arg.starts_with('-') => {
                if src_dir.is_none() {
                    src_dir = Some(PathBuf::from(arg));
                } else if dst_dir.is_none() {
                    dst_dir = Some(PathBuf::from(arg));
                } else {
                    return Err(OracleError::InvalidArg(format!(
                        "unexpected positional argument {arg:?}"
                    )));
                }
            }
            other => {
                return Err(OracleError::InvalidArg(format!(
                    "unknown inject-bitflip option {other:?}"
                )));
            }
        }
        i += 1;
    }

    let src = src_dir
        .ok_or_else(|| OracleError::InvalidArg("inject-bitflip requires <src-dir>".into()))?;
    let dst = dst_dir
        .ok_or_else(|| OracleError::InvalidArg("inject-bitflip requires <dst-dir>".into()))?;
    if target.is_none() {
        return Err(OracleError::InvalidArg(
            "inject-bitflip requires --target <tokens|logits|residuals>".into(),
        ));
    }

    let summary = inject_bitflip(&src, &dst, &opts)?;
    if json_out {
        let s = serde_json::to_string_pretty(&summary).map_err(|e| OracleError::json(&dst, e))?;
        println!("{s}");
    } else {
        println!(
            "INJECT_BITFLIP: target={:?} prompt={} bit={} {}",
            summary.target, summary.prompt_id, summary.bit, summary.detail
        );
    }
    Ok(EXIT_PASS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_help_exits_zero() {
        let code = run_cli(&["--help".to_string()]).expect("help");
        assert_eq!(code, EXIT_PASS);
    }
}
