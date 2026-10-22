//! CLI / GGUF inspection and `--serve` protocol line validator for Flash-Next (`qwen4exp`).

use std::path::Path;

pub const ARCH_NAME: &str = "qwen4exp";

/// Checks whether a GGUF file at `path` has `general.architecture == "qwen4exp"`.
pub fn is_qwen4exp_model(path: &Path) -> bool {
    if let Ok(gguf) = talus_gguf::GgufFile::open(path) {
        if gguf.get_str("general.architecture") == Some(ARCH_NAME) {
            return true;
        }
    }
    if let Ok(model) = talus_gguf::GgufModel::open(path) {
        if model.meta_shard().get_str("general.architecture") == Some(ARCH_NAME) {
            return true;
        }
    }
    false
}

/// Checks whether the argument list invokes a Flash-Next pack (`--pack <DIR>`).
pub fn has_pack_flag<S: AsRef<str>>(argv: &[S]) -> bool {
    argv.iter().any(|a| a.as_ref() == "--pack")
}

/// Checks whether the argument list invokes `--fn-dump-compare <lhs_dir> <rhs_dir>`.
pub fn parse_fn_dump_compare_args<S: AsRef<str>>(argv: &[S]) -> Option<(String, String)> {
    let mut i = 1usize;
    while i + 2 < argv.len() {
        if argv[i].as_ref() == "--fn-dump-compare" {
            return Some((
                argv[i + 1].as_ref().to_string(),
                argv[i + 2].as_ref().to_string(),
            ));
        }
        i += 1;
    }
    None
}

/// Checks whether the argument list passes `--model` / `--native` / `-m` pointing to a `qwen4exp` GGUF,
/// or passes `--pack` or `--fn-dump-compare`.
pub fn is_qwen4exp_invocation<S: AsRef<str>>(argv: &[S]) -> bool {
    if has_pack_flag(argv) || parse_fn_dump_compare_args(argv).is_some() {
        return true;
    }
    let mut i = 1usize;
    while i + 1 < argv.len() {
        let flag = argv[i].as_ref();
        if flag == "--model" || flag == "--native" || flag == "-m" {
            let path = Path::new(argv[i + 1].as_ref());
            if is_qwen4exp_model(path) {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Parses a comma-separated list of `i64` token IDs (`parse_i64_list` in `generate.cpp`).
pub fn parse_i64_list(s: &str) -> Result<Vec<i64>, String> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for part in s.split(',') {
        if part.is_empty() {
            continue;
        }
        let val = part
            .trim_start()
            .parse::<i64>()
            .map_err(|_| format!("bad integer in list: '{part}'"))?;
        out.push(val);
    }
    Ok(out)
}

/// Checks whether the pure-Rust Flash-Next data plane is selected (default unless `TALUS_FN_RUST=0`,
/// `TALUS_FN_CPP=1`, or `--cpp-engine`).
pub fn use_rust_engine<S: AsRef<str>>(argv: &[S]) -> bool {
    if argv.iter().any(|a| a.as_ref() == "--cpp-engine") {
        return false;
    }
    if let Ok(v) = std::env::var("TALUS_FN_CPP") {
        let trimmed = v.trim();
        if !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false") {
            return false;
        }
    }
    if argv.iter().any(|a| a.as_ref() == "--rust-engine") {
        return true;
    }
    if let Ok(v) = std::env::var("TALUS_FN_RUST") {
        let trimmed = v.trim();
        if trimmed == "0" || trimmed.eq_ignore_ascii_case("false") {
            return false;
        }
        if !trimmed.is_empty() {
            return true;
        }
    }
    true
}

/// Returns `true` when the C++ Flash-Next data plane is selected (pure-Rust is the default unless `TALUS_FN_RUST=0`,
/// `TALUS_FN_CPP=1`, or `--cpp-engine`).
pub fn use_cpp_engine<S: AsRef<str>>(argv: &[S]) -> bool {
    !use_rust_engine(argv)
}

/// Parsed `GEN <max_new> [key=val ...] <id,id,...>` request on the `--serve` line protocol.
#[derive(Debug, Clone)]
pub struct ParsedServeGenRequest {
    pub max_new: i64,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub seed: u64,
    pub pcie_frac: f64,
    pub spec_min_p: f64,
    pub aux_cpus: i32,
    pub req_cvec: i32,
    pub req_ckpt: i32,
    pub req_pin: i64,
    pub req_ignore_eos: bool,
    pub req_cache: bool,
    pub bias_text: Option<String>,
    pub ids: Vec<i64>,
}

/// Parses a `GEN <max_new> [key=val ...] <id,id,...>` protocol line.
pub fn parse_serve_gen_line(
    line: &str,
    default_pcie_frac: f64,
    default_spec_min_p: f64,
) -> Result<ParsedServeGenRequest, String> {
    let trimmed = line.trim_end_matches(['\r', '\n']);
    if trimmed.starts_with("GENI ") {
        return Err("ERR this engine was started without --vision".to_string());
    }
    let Some(rest) = trimmed.strip_prefix("GEN ") else {
        return Err(
            "ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>"
                .to_string(),
        );
    };
    let rest = rest.trim_start_matches(' ');
    let Some(sp_idx) = rest.find(' ') else {
        return Err("ERR bad request: max_new".to_string());
    };
    let max_new_str = &rest[..sp_idx];
    let max_new = max_new_str
        .parse::<i64>()
        .map_err(|_| "ERR bad request: max_new".to_string())?;
    if max_new < 1 {
        return Err("ERR bad request: max_new".to_string());
    }

    let mut req = ParsedServeGenRequest {
        max_new,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 20,
        min_p: 0.0,
        penalty_last_n: 0,
        penalty_repeat: 1.0,
        penalty_freq: 0.0,
        penalty_present: 0.0,
        seed: 0,
        pcie_frac: default_pcie_frac,
        spec_min_p: default_spec_min_p,
        aux_cpus: -1,
        req_cvec: 1,
        req_ckpt: 1,
        req_pin: 0,
        req_ignore_eos: false,
        req_cache: true,
        bias_text: None,
        ids: Vec::new(),
    };

    let mut cursor = &rest[sp_idx..];
    loop {
        cursor = cursor.trim_start_matches(' ');
        if cursor.is_empty() {
            break;
        }
        let tok_end = cursor.find(' ').unwrap_or(cursor.len());
        let tok = &cursor[..tok_end];
        let Some(eq) = tok.find('=') else {
            break;
        };
        let key = &tok[..eq];
        let val = &tok[eq + 1..];
        let fv = val.parse::<f32>().unwrap_or(0.0);
        let iv = val.parse::<i32>().unwrap_or(0);
        match key {
            "logit_bias" => req.bias_text = Some(val.to_string()),
            "cvec" => req.req_cvec = iv,
            "ckpt" => req.req_ckpt = if iv != 0 { 1 } else { 0 },
            "ignore_eos" => req.req_ignore_eos = iv != 0,
            "cache" => req.req_cache = iv != 0,
            "pin" => req.req_pin = val.parse::<i64>().unwrap_or(0).max(0),
            "temperature" => req.temperature = fv,
            "top_p" => req.top_p = fv,
            "top_k" => req.top_k = iv,
            "min_p" => req.min_p = fv,
            "penalty_last_n" => req.penalty_last_n = iv,
            "penalty_repeat" => req.penalty_repeat = fv,
            "penalty_freq" => req.penalty_freq = fv,
            "penalty_present" => req.penalty_present = fv,
            "seed" => req.seed = val.parse::<u64>().unwrap_or(0),
            "pcie_frac" => req.pcie_frac = (fv as f64).clamp(0.0, 1.0),
            "spec_min_p" => req.spec_min_p = (fv as f64).clamp(0.0, 1.0),
            "aux_cpus" => req.aux_cpus = if iv != 0 { 1 } else { 0 },
            _ => {}
        }
        cursor = &cursor[tok_end..];
    }

    let ids_str = cursor.trim();
    if ids_str.is_empty() {
        return Err("ERR bad request: empty token list".to_string());
    }
    let ids = parse_i64_list(ids_str).map_err(|e| format!("ERR bad request: {e}"))?;
    if ids.is_empty() {
        return Err("ERR bad request: empty token list".to_string());
    }
    req.ids = ids;
    Ok(req)
}

/// High-level command kind on the `--serve` line protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServeCommandKind {
    Empty,
    Ping,
    Quit,
    Stop,
    Warmup,
    Pin,
    Unpin,
    CheckpointSave,
    CheckpointLoad,
    CheckpointInfo,
    Batch,
    Gen,
    Unknown(String),
}

/// Classifies a single `--serve` stdin protocol line (stripping trailing `\r`/`\n`).
pub fn classify_serve_line(line: &str) -> ServeCommandKind {
    let trimmed = line.trim_end_matches(['\r', '\n']);
    if trimmed.is_empty() {
        return ServeCommandKind::Empty;
    }
    if trimmed == "PING" {
        return ServeCommandKind::Ping;
    }
    if trimmed == "QUIT" {
        return ServeCommandKind::Quit;
    }
    if trimmed == "STOP" {
        return ServeCommandKind::Stop;
    }
    if trimmed == "WARMUP" || trimmed.starts_with("WARMUP ") {
        return ServeCommandKind::Warmup;
    }
    if trimmed == "PIN" || trimmed.starts_with("PIN ") {
        return ServeCommandKind::Pin;
    }
    if trimmed == "UNPIN" {
        return ServeCommandKind::Unpin;
    }
    if trimmed == "CHECKPOINT_SAVE" || trimmed.starts_with("CHECKPOINT_SAVE ") {
        return ServeCommandKind::CheckpointSave;
    }
    if trimmed == "CHECKPOINT_LOAD" || trimmed.starts_with("CHECKPOINT_LOAD ") {
        return ServeCommandKind::CheckpointLoad;
    }
    if trimmed == "CHECKPOINT_INFO" || trimmed.starts_with("CHECKPOINT_INFO ") {
        return ServeCommandKind::CheckpointInfo;
    }
    if trimmed == "BATCH" || trimmed.starts_with("BATCH ") {
        return ServeCommandKind::Batch;
    }
    if trimmed.starts_with("GEN ") {
        return ServeCommandKind::Gen;
    }
    let first = trimmed.split_whitespace().next().unwrap_or("");
    ServeCommandKind::Unknown(first.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_serve_lines_and_parse_ids() {
        assert_eq!(classify_serve_line("\r\n"), ServeCommandKind::Empty);
        assert_eq!(classify_serve_line("PING\n"), ServeCommandKind::Ping);
        assert_eq!(classify_serve_line("STOP\r\n"), ServeCommandKind::Stop);
        assert_eq!(classify_serve_line("QUIT"), ServeCommandKind::Quit);
        assert_eq!(
            classify_serve_line("GEN max_new=32 1,2,3\n"),
            ServeCommandKind::Gen
        );
        assert_eq!(
            classify_serve_line("BATCH 2 max_new=16\n"),
            ServeCommandKind::Batch
        );
        assert_eq!(parse_i64_list("10,20,30").unwrap(), vec![10, 20, 30]);
        assert!(parse_i64_list("10,bad,30").is_err());

        let parsed = parse_serve_gen_line(
            "GEN 64 temperature=0.7 top_k=40 ckpt=0 pin=128 logit_bias=10:1.5 100,200,300\n",
            0.25,
            0.1,
        )
        .unwrap();
        assert_eq!(parsed.max_new, 64);
        assert!((parsed.temperature - 0.7).abs() < 1e-5);
        assert_eq!(parsed.top_k, 40);
        assert_eq!(parsed.req_ckpt, 0);
        assert_eq!(parsed.req_pin, 128);
        assert_eq!(parsed.bias_text.as_deref(), Some("10:1.5"));
        assert_eq!(parsed.ids, vec![100, 200, 300]);
        assert!(use_cpp_engine(&["talus-cli", "--cpp-engine"]));
        assert!(use_rust_engine(&["talus-cli", "--serve"]));
        assert!(!use_cpp_engine(&["talus-cli", "--serve"]));
        assert!(use_rust_engine(&["talus-cli", "--rust-engine", "--serve"]));
        assert!(!use_cpp_engine(&["talus-cli", "--rust-engine", "--serve"]));
        assert_eq!(
            parse_fn_dump_compare_args(&["talus-cli", "--fn-dump-compare", "/a", "/b"]),
            Some(("/a".to_string(), "/b".to_string()))
        );
    }
}
