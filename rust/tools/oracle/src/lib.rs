//! Oracle recording verification, diff harness, and bitflip injection library.
//!
//! Implements Contract v0 (`docs/golden-dump-format.md`) for comparing C++ golden
//! reference dumps against candidate dumps (tokens, teacher-forced logits rows,
//! and per-layer residual snapshots).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::too_many_lines
)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const CRATE_NAME: &str = "oracle";
pub const SCHEMA_VERSION: &str = "0.1.0";
pub const DUMP_FORMAT: &str = "talus-golden-dump-v0";
pub const TOKENS_MAGIC: [u8; 8] = *b"TLSTOK00";
pub const TOKENS_BIN_VERSION: u32 = 1;
pub const TOKENS_HEADER_BYTES: usize = 32;

/// Exit code when all checks pass within configured tolerances.
pub const EXIT_PASS: i32 = 0;
/// Exit code when one or more divergence checks fail.
pub const EXIT_DIFF: i32 = 1;
/// Exit code when arguments, files, schema, or SHA-256 checksums are invalid.
pub const EXIT_ERROR: i32 = 2;

/// Errors returned by oracle verification, diffing, or bitflip injection.
#[derive(Debug, Error)]
pub enum OracleError {
    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("JSON parse error in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("schema validation failed in {path}: {message}")]
    Schema { path: PathBuf, message: String },
    #[error("SHA-256 mismatch for {path}: expected {expected}, actual {actual}")]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("invalid argument: {0}")]
    InvalidArg(String),
}

impl OracleError {
    pub fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub fn json(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        Self::Json {
            path: path.into(),
            source,
        }
    }

    fn schema(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self::Schema {
            path: path.into(),
            message: message.into(),
        }
    }
}

// ============================================================================
// Pure-Rust FIPS 180-4 SHA-256
// ============================================================================

pub mod sha256 {
    use super::{fs, OracleError, Path, Read};
    use std::fmt::Write;

    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    const H0: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    /// Streaming SHA-256 state.
    #[derive(Clone, Debug)]
    pub struct Sha256 {
        state: [u32; 8],
        buf: [u8; 64],
        buf_len: usize,
        total_bytes: u64,
    }

    impl Default for Sha256 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Sha256 {
        #[must_use]
        pub const fn new() -> Self {
            Self {
                state: H0,
                buf: [0u8; 64],
                buf_len: 0,
                total_bytes: 0,
            }
        }

        pub fn update(&mut self, mut data: &[u8]) {
            self.total_bytes = self.total_bytes.wrapping_add(data.len() as u64);
            if self.buf_len > 0 {
                let take = (64 - self.buf_len).min(data.len());
                self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
                self.buf_len += take;
                data = &data[take..];
                if self.buf_len == 64 {
                    let block = self.buf;
                    self.compress_block(&block);
                    self.buf_len = 0;
                }
            }
            while data.len() >= 64 {
                let mut block = [0u8; 64];
                block.copy_from_slice(&data[..64]);
                self.compress_block(&block);
                data = &data[64..];
            }
            if !data.is_empty() {
                self.buf[..data.len()].copy_from_slice(data);
                self.buf_len = data.len();
            }
        }

        #[must_use]
        pub fn finalize(mut self) -> [u8; 32] {
            let bit_len = self.total_bytes.wrapping_mul(8);
            self.buf[self.buf_len] = 0x80;
            self.buf_len += 1;
            if self.buf_len > 56 {
                self.buf[self.buf_len..64].fill(0);
                let block = self.buf;
                self.compress_block(&block);
                self.buf_len = 0;
            }
            self.buf[self.buf_len..56].fill(0);
            self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
            let block = self.buf;
            self.compress_block(&block);

            let mut out = [0u8; 32];
            for (i, word) in self.state.iter().enumerate() {
                out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
            }
            out
        }

        #[must_use]
        pub fn finalize_hex(self) -> String {
            let digest = self.finalize();
            let mut s = String::with_capacity(64);
            for b in digest {
                let _ = write!(s, "{b:02x}");
            }
            s
        }

        fn compress_block(&mut self, block: &[u8; 64]) {
            let mut w = [0u32; 64];
            for (i, chunk) in block.as_chunks::<4>().0.iter().enumerate() {
                w[i] = u32::from_be_bytes(*chunk);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }

            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let temp1 = h
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(maj);

                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }

            self.state[0] = self.state[0].wrapping_add(a);
            self.state[1] = self.state[1].wrapping_add(b);
            self.state[2] = self.state[2].wrapping_add(c);
            self.state[3] = self.state[3].wrapping_add(d);
            self.state[4] = self.state[4].wrapping_add(e);
            self.state[5] = self.state[5].wrapping_add(f);
            self.state[6] = self.state[6].wrapping_add(g);
            self.state[7] = self.state[7].wrapping_add(h);
        }
    }

    #[must_use]
    pub fn sha256_hex(data: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(data);
        h.finalize_hex()
    }

    pub fn sha256_file(path: &Path) -> Result<String, OracleError> {
        let mut f = fs::File::open(path).map_err(|e| OracleError::io(path, e))?;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = f.read(&mut buf).map_err(|e| OracleError::io(path, e))?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        Ok(h.finalize_hex())
    }
}

// ============================================================================
// Contract v0 Schema Types
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub schema_version: String,
    pub format: String,
    pub model_id: String,
    #[serde(default)]
    pub gguf_files: Vec<String>,
    #[serde(default)]
    pub gguf_sha256: Vec<String>,
    pub quant_scheme: String,
    pub cpp_build: CppBuildInfo,
    pub recorded_at_utc: String,
    pub recorded_host: String,
    pub hparams: ModelHparams,
    pub prompts: Vec<PromptEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CppBuildInfo {
    pub git_commit: String,
    pub git_branch: String,
    pub binary: String,
    pub rocm_version: String,
    pub gpu_arch: String,
    pub hip_flags: String,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelHparams {
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub n_embd: u32,
    pub n_layer: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_ff: u32,
    pub n_rot: u32,
    pub rope_freq_base: f64,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub n_expert: u32,
    #[serde(default)]
    pub n_expert_used: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PromptEntry {
    pub prompt_id: String,
    pub category: String,
    pub prompt_tokens_len: u32,
    pub generated_tokens_len: u32,
    pub seed: u64,
    pub temperature: f64,
    pub top_k: u32,
    pub top_p: f64,
    pub min_p: f64,
    pub tokens_json: String,
    pub logits_json: String,
    pub residuals_json: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TokensMeta {
    pub schema_version: String,
    pub prompt_id: String,
    pub bin_file: String,
    pub bin_sha256: String,
    pub prompt_tokens_len: u32,
    pub generated_tokens_len: u32,
    pub eos_token_id: u32,
    pub stop_reason: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokensBinData {
    pub version: u32,
    pub flags: u32,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LogitsMeta {
    pub schema_version: String,
    pub prompt_id: String,
    pub bin_file: String,
    pub bin_sha256: String,
    pub dtype: String,
    pub endianness: String,
    pub n_rows: u32,
    pub n_vocab: u32,
    pub row_stride_bytes: u64,
    pub total_bytes: u64,
    pub positions: Vec<u32>,
    pub input_token_ids: Vec<u32>,
    pub rows_summary: Vec<LogitRowSummary>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LogitRowSummary {
    pub row_idx: u32,
    pub token_pos: u32,
    pub input_token_id: u32,
    pub argmax_token_id: u32,
    pub max_logit: f32,
    pub min_logit: f32,
    pub l2_norm: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResidualsMeta {
    pub schema_version: String,
    pub prompt_id: String,
    pub bin_file: String,
    pub bin_sha256: String,
    pub dtype: String,
    pub endianness: String,
    pub n_embd: u32,
    pub n_layer: u32,
    pub snapshots: Vec<ResidualSnapshotMeta>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResidualSnapshotMeta {
    pub snapshot_idx: u32,
    pub token_pos: u32,
    pub layer_idx: i32,
    pub tap: String,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub l2_norm: f64,
    pub rms: f64,
}

/// Canonical residual tap points defined by Contract v0 (`docs/golden-dump-format.md` §4.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResidualTap {
    EmbdOut = 0,
    AttnOut = 1,
    FfnOut = 2,
    FinalNormOut = 3,
}

impl ResidualTap {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "embd_out" => Some(Self::EmbdOut),
            "attn_out" => Some(Self::AttnOut),
            "ffn_out" => Some(Self::FfnOut),
            "final_norm_out" => Some(Self::FinalNormOut),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EmbdOut => "embd_out",
            Self::AttnOut => "attn_out",
            Self::FfnOut => "ffn_out",
            Self::FinalNormOut => "final_norm_out",
        }
    }

    #[must_use]
    pub const fn order(self) -> u8 {
        self as u8
    }
}

// ============================================================================
// Binary Encoding / Decoding Helpers
// ============================================================================

/// Encode `.tokens.bin` in Contract v0 format (`TLSTOK00` 32-byte header + `u32` payload).
#[must_use]
pub fn encode_tokens_bin(prompt_ids: &[u32], generated_ids: &[u32]) -> Vec<u8> {
    let total_len = TOKENS_HEADER_BYTES + 4 * (prompt_ids.len() + generated_ids.len());
    let mut buf = Vec::with_capacity(total_len);
    buf.extend_from_slice(&TOKENS_MAGIC);
    buf.extend_from_slice(&TOKENS_BIN_VERSION.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // flags = 0
    buf.extend_from_slice(&(prompt_ids.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(generated_ids.len() as u32).to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // reserved = 0
    for &id in prompt_ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    for &id in generated_ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    buf
}

/// Parse `.tokens.bin` in Contract v0 format (`TLSTOK00` 32-byte header + `u32` payload).
pub fn parse_tokens_bin(path: &Path, bytes: &[u8]) -> Result<TokensBinData, OracleError> {
    if bytes.len() < TOKENS_HEADER_BYTES {
        return Err(OracleError::schema(
            path,
            format!(
                "tokens.bin too short: {} bytes (< {})",
                bytes.len(),
                TOKENS_HEADER_BYTES
            ),
        ));
    }
    if bytes[0..8] != TOKENS_MAGIC {
        return Err(OracleError::schema(
            path,
            format!("invalid magic in tokens.bin: {:?}", &bytes[0..8]),
        ));
    }
    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if version != TOKENS_BIN_VERSION {
        return Err(OracleError::schema(
            path,
            format!("unsupported tokens.bin version {version} (expected {TOKENS_BIN_VERSION})"),
        ));
    }
    let flags = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    if flags != 0 {
        return Err(OracleError::schema(
            path,
            format!("non-zero tokens.bin flags {flags:#x}"),
        ));
    }
    let prompt_len = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
    let gen_len = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]) as usize;
    let reserved = u64::from_le_bytes([
        bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
    ]);
    if reserved != 0 {
        return Err(OracleError::schema(
            path,
            format!("non-zero tokens.bin reserved field {reserved:#x}"),
        ));
    }
    let expected_bytes = TOKENS_HEADER_BYTES + 4 * (prompt_len + gen_len);
    if bytes.len() != expected_bytes {
        return Err(OracleError::schema(
            path,
            format!(
                "tokens.bin size mismatch: got {} bytes, expected {expected_bytes}",
                bytes.len()
            ),
        ));
    }
    let payload = &bytes[TOKENS_HEADER_BYTES..];
    let mut prompt_token_ids = Vec::with_capacity(prompt_len);
    for chunk in payload[..4 * prompt_len].as_chunks::<4>().0 {
        prompt_token_ids.push(u32::from_le_bytes(*chunk));
    }
    let mut generated_token_ids = Vec::with_capacity(gen_len);
    for chunk in payload[4 * prompt_len..].as_chunks::<4>().0 {
        generated_token_ids.push(u32::from_le_bytes(*chunk));
    }
    Ok(TokensBinData {
        version,
        flags,
        prompt_token_ids,
        generated_token_ids,
    })
}

/// Decode a raw little-endian `f32` byte slice into `Vec<f32>`.
pub fn decode_f32_le(path: &Path, bytes: &[u8]) -> Result<Vec<f32>, OracleError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(OracleError::schema(
            path,
            format!(
                "binary f32 buffer length {} is not a multiple of 4",
                bytes.len()
            ),
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*chunk));
    }
    Ok(out)
}

/// Encode a slice of `f32` values into little-endian bytes.
#[must_use]
pub fn encode_f32_le(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

// ============================================================================
// Floating-Point Comparison & Formatting Metrics (Contract v0 §5.3 & §5.4)
// ============================================================================

/// Map IEEE-754 `f32` bit pattern into lexicographical order (`docs/golden-dump-format.md` §5.3).
#[must_use]
pub const fn float_lex_order(bits: u32) -> u32 {
    if (bits & 0x8000_0000) != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

/// Compute ULP distance between two `f32` values (`u64::MAX` if either is `NaN`).
#[must_use]
pub fn ulp_diff_f32(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let oa = u64::from(float_lex_order(a.to_bits()));
    let ob = u64::from(float_lex_order(b.to_bits()));
    oa.abs_diff(ob)
}

/// Format an `f32` in C99 `%a` hexadecimal floating-point notation (6 hex fraction digits).
#[must_use]
pub fn format_f32_hex(v: f32) -> String {
    let bits = v.to_bits();
    let sign = if (bits >> 31) != 0 { "-" } else { "" };
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        if mant != 0 {
            return "nan".to_string();
        }
        return format!("{sign}inf");
    }
    if exp == 0 && mant == 0 {
        return format!("{sign}0x0.000000p+0");
    }
    if exp == 0 {
        return format!("{sign}0x0.{:06x}p-126", mant << 1);
    }
    let e = exp - 127;
    format!("{sign}0x1.{:06x}p{e:+}", mant << 1)
}

/// Compute `(l2_norm, rms)` in `f64` for a slice of `f32`.
#[must_use]
pub fn slice_l2_and_rms(v: &[f32]) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let sum_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let l2 = sum_sq.sqrt();
    let rms = (sum_sq / (v.len() as f64)).sqrt();
    (l2, rms)
}

/// Compute `LogitRowSummary` for a single logit row.
#[must_use]
pub fn summarize_logit_row(
    row_idx: u32,
    token_pos: u32,
    input_token_id: u32,
    row: &[f32],
) -> LogitRowSummary {
    let mut max_logit = f32::NEG_INFINITY;
    let mut min_logit = f32::INFINITY;
    let mut argmax_idx = 0u32;
    let mut sum_sq = 0.0f64;
    for (idx, &v) in row.iter().enumerate() {
        if v > max_logit {
            max_logit = v;
            argmax_idx = idx as u32;
        }
        if v < min_logit {
            min_logit = v;
        }
        sum_sq += f64::from(v) * f64::from(v);
    }
    LogitRowSummary {
        row_idx,
        token_pos,
        input_token_id,
        argmax_token_id: argmax_idx,
        max_logit,
        min_logit,
        l2_norm: sum_sq.sqrt(),
    }
}

/// Aggregate comparison statistics over two equal-length `f32` slices.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorComparison {
    pub first_failing_idx: Option<u32>,
    pub max_diff_idx: u32,
    pub max_abs_diff: f32,
    pub max_ulp_diff: u64,
    pub l2_rel_err: f64,
    pub cosine_similarity: f64,
    pub argmax_expected: u32,
    pub argmax_actual: u32,
    pub has_non_finite: bool,
}

#[must_use]
pub fn compare_f32_vectors(expected: &[f32], actual: &[f32], cfg: &DiffConfig) -> VectorComparison {
    let mut first_failing_idx = None;
    let mut max_diff_idx = 0u32;
    let mut max_abs_diff = 0.0f32;
    let mut max_ulp_diff = 0u64;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_exp = 0.0f64;
    let mut sum_sq_act = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_exp = f32::NEG_INFINITY;
    let mut max_act = f32::NEG_INFINITY;
    let mut argmax_expected = 0u32;
    let mut argmax_actual = 0u32;
    let mut has_non_finite = false;

    for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        let idx = i as u32;
        if !e.is_finite() || !a.is_finite() {
            has_non_finite = true;
            if first_failing_idx.is_none() {
                first_failing_idx = Some(idx);
            }
        }
        if e > max_exp {
            max_exp = e;
            argmax_expected = idx;
        }
        if a > max_act {
            max_act = a;
            argmax_actual = idx;
        }
        let abs_diff = (e - a).abs();
        let ulp = ulp_diff_f32(e, a);
        if ulp > max_ulp_diff || (ulp == max_ulp_diff && abs_diff > max_abs_diff) {
            max_ulp_diff = ulp;
            max_abs_diff = abs_diff;
            max_diff_idx = idx;
        } else if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }

        if !cfg.element_passes(e, a, abs_diff, ulp) && first_failing_idx.is_none() {
            first_failing_idx = Some(idx);
        }

        let ef = f64::from(e);
        let af = f64::from(a);
        let df = ef - af;
        sum_sq_diff += df * df;
        sum_sq_exp += ef * ef;
        sum_sq_act += af * af;
        dot += ef * af;
    }

    let l2_diff = sum_sq_diff.sqrt();
    let l2_exp = sum_sq_exp.sqrt();
    let l2_act = sum_sq_act.sqrt();
    let l2_rel_err = l2_diff / l2_exp.max(1e-30);
    let denom = (l2_exp * l2_act).max(1e-30);
    let cosine_similarity = if sum_sq_exp == 0.0 && sum_sq_act == 0.0 {
        1.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    };

    VectorComparison {
        first_failing_idx,
        max_diff_idx,
        max_abs_diff,
        max_ulp_diff,
        l2_rel_err,
        cosine_similarity,
        argmax_expected,
        argmax_actual,
        has_non_finite,
    }
}

// ============================================================================
// Verification (`verify <dump-dir>`)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerifySummary {
    pub dump_dir: String,
    pub model_id: String,
    pub prompts_verified: usize,
    pub checksums_verified: usize,
    pub total_tokens_verified: usize,
    pub total_logit_rows_verified: usize,
    pub total_residual_snapshots_verified: usize,
}

pub fn load_manifest(dump_dir: &Path) -> Result<Manifest, OracleError> {
    let manifest_path = dump_dir.join("manifest.json");
    let raw = fs::read_to_string(&manifest_path).map_err(|e| OracleError::io(&manifest_path, e))?;
    let manifest: Manifest =
        serde_json::from_str(&raw).map_err(|e| OracleError::json(&manifest_path, e))?;
    if manifest.schema_version != SCHEMA_VERSION {
        return Err(OracleError::schema(
            &manifest_path,
            format!(
                "unsupported schema_version {:?} (expected {:?})",
                manifest.schema_version, SCHEMA_VERSION
            ),
        ));
    }
    if manifest.format != DUMP_FORMAT {
        return Err(OracleError::schema(
            &manifest_path,
            format!(
                "unsupported format {:?} (expected {:?})",
                manifest.format, DUMP_FORMAT
            ),
        ));
    }
    if manifest.prompts.is_empty() {
        return Err(OracleError::schema(
            &manifest_path,
            "manifest.prompts must not be empty",
        ));
    }
    Ok(manifest)
}

/// Parse `SHA256SUMS` lines of the form `<64-hex>  <rel_path>` or `<64-hex> *<rel_path>`.
pub fn parse_sha256sums(
    sums_path: &Path,
    content: &str,
) -> Result<Vec<(String, String)>, OracleError> {
    let mut entries = Vec::new();
    for (line_no, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (hash, rest) = line.split_once(char::is_whitespace).ok_or_else(|| {
            OracleError::schema(
                sums_path,
                format!("malformed SHA256SUMS line {}: {line:?}", line_no + 1),
            )
        })?;
        if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(OracleError::schema(
                sums_path,
                format!(
                    "invalid SHA-256 hex digest on line {}: {hash:?}",
                    line_no + 1
                ),
            ));
        }
        let rel = rest.trim_start().trim_start_matches('*');
        if rel.is_empty() {
            return Err(OracleError::schema(
                sums_path,
                format!("empty path in SHA256SUMS line {}", line_no + 1),
            ));
        }
        entries.push((hash.to_ascii_lowercase(), rel.to_string()));
    }
    if entries.is_empty() {
        return Err(OracleError::schema(sums_path, "SHA256SUMS is empty"));
    }
    Ok(entries)
}

/// Verify all entries in `<dump_dir>/SHA256SUMS`.
pub fn verify_sha256sums(dump_dir: &Path) -> Result<usize, OracleError> {
    let sums_path = dump_dir.join("SHA256SUMS");
    let content = fs::read_to_string(&sums_path).map_err(|e| OracleError::io(&sums_path, e))?;
    let entries = parse_sha256sums(&sums_path, &content)?;
    for (expected, rel) in &entries {
        let file_path = dump_dir.join(rel);
        let actual = sha256::sha256_file(&file_path)?;
        if &actual != expected {
            return Err(OracleError::ChecksumMismatch {
                path: file_path,
                expected: expected.clone(),
                actual,
            });
        }
    }
    Ok(entries.len())
}

/// Write `<dump_dir>/SHA256SUMS` covering all regular files under `dump_dir` (except `SHA256SUMS` itself).
pub fn write_sha256sums(dump_dir: &Path) -> Result<usize, OracleError> {
    let mut rel_files = Vec::new();
    collect_relative_files(dump_dir, dump_dir, &mut rel_files)?;
    rel_files.sort();
    let mut out = String::new();
    let mut count = 0usize;
    for rel in rel_files {
        if rel == "SHA256SUMS" {
            continue;
        }
        let full = dump_dir.join(&rel);
        let digest = sha256::sha256_file(&full)?;
        let _ = writeln!(out, "{digest}  {rel}");
        count += 1;
    }
    let sums_path = dump_dir.join("SHA256SUMS");
    fs::write(&sums_path, out).map_err(|e| OracleError::io(&sums_path, e))?;
    Ok(count)
}

fn collect_relative_files(
    root: &Path,
    cur: &Path,
    out: &mut Vec<String>,
) -> Result<(), OracleError> {
    let entries = fs::read_dir(cur).map_err(|e| OracleError::io(cur, e))?;
    for entry in entries {
        let entry = entry.map_err(|e| OracleError::io(cur, e))?;
        let path = entry.path();
        let ft = entry.file_type().map_err(|e| OracleError::io(&path, e))?;
        if ft.is_dir() {
            collect_relative_files(root, &path, out)?;
        } else if ft.is_file() {
            let rel = path
                .strip_prefix(root)
                .map_err(|e| OracleError::schema(&path, e.to_string()))?;
            let rel_str = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            out.push(rel_str);
        }
    }
    Ok(())
}

fn resolve_companion_bin(json_path: &Path, dump_dir: &Path, bin_file: &str) -> PathBuf {
    let candidate = json_path.parent().unwrap_or(dump_dir).join(bin_file);
    if candidate.exists() {
        candidate
    } else {
        dump_dir.join(bin_file)
    }
}

pub fn load_and_verify_tokens(
    dump_dir: &Path,
    rel_json: &str,
    expected_prompt_id: &str,
    expected_prompt_len: u32,
    expected_gen_len: u32,
) -> Result<(TokensMeta, TokensBinData), OracleError> {
    let json_path = dump_dir.join(rel_json);
    let raw = fs::read_to_string(&json_path).map_err(|e| OracleError::io(&json_path, e))?;
    let meta: TokensMeta =
        serde_json::from_str(&raw).map_err(|e| OracleError::json(&json_path, e))?;
    if meta.schema_version != SCHEMA_VERSION {
        return Err(OracleError::schema(
            &json_path,
            format!("unsupported schema_version {:?}", meta.schema_version),
        ));
    }
    if meta.prompt_id != expected_prompt_id {
        return Err(OracleError::schema(
            &json_path,
            format!(
                "prompt_id mismatch: got {:?}, expected {:?}",
                meta.prompt_id, expected_prompt_id
            ),
        ));
    }
    if meta.prompt_tokens_len != expected_prompt_len
        || meta.generated_tokens_len != expected_gen_len
    {
        return Err(OracleError::schema(
            &json_path,
            format!(
                "token counts ({}, {}) do not match manifest ({expected_prompt_len}, {expected_gen_len})",
                meta.prompt_tokens_len, meta.generated_tokens_len
            ),
        ));
    }
    if meta.prompt_token_ids.len() != expected_prompt_len as usize {
        return Err(OracleError::schema(
            &json_path,
            format!(
                "prompt_token_ids.len() == {} != prompt_tokens_len == {expected_prompt_len}",
                meta.prompt_token_ids.len()
            ),
        ));
    }
    if meta.generated_token_ids.len() != expected_gen_len as usize {
        return Err(OracleError::schema(
            &json_path,
            format!(
                "generated_token_ids.len() == {} != generated_tokens_len == {expected_gen_len}",
                meta.generated_token_ids.len()
            ),
        ));
    }

    let bin_path = resolve_companion_bin(&json_path, dump_dir, &meta.bin_file);
    let bin_bytes = fs::read(&bin_path).map_err(|e| OracleError::io(&bin_path, e))?;
    let actual_sha = sha256::sha256_hex(&bin_bytes);
    if actual_sha != meta.bin_sha256.to_ascii_lowercase() {
        return Err(OracleError::ChecksumMismatch {
            path: bin_path,
            expected: meta.bin_sha256,
            actual: actual_sha,
        });
    }
    let bin_data = parse_tokens_bin(&bin_path, &bin_bytes)?;
    if bin_data.prompt_token_ids != meta.prompt_token_ids {
        return Err(OracleError::schema(
            &bin_path,
            "prompt_token_ids in .tokens.bin do not match .tokens.json",
        ));
    }
    if bin_data.generated_token_ids != meta.generated_token_ids {
        return Err(OracleError::schema(
            &bin_path,
            "generated_token_ids in .tokens.bin do not match .tokens.json",
        ));
    }
    Ok((meta, bin_data))
}

pub fn load_and_verify_logits(
    dump_dir: &Path,
    rel_json: &str,
    expected_prompt_id: &str,
    expected_n_vocab: u32,
) -> Result<(LogitsMeta, Vec<f32>), OracleError> {
    let json_path = dump_dir.join(rel_json);
    let raw = fs::read_to_string(&json_path).map_err(|e| OracleError::io(&json_path, e))?;
    let meta: LogitsMeta =
        serde_json::from_str(&raw).map_err(|e| OracleError::json(&json_path, e))?;
    if meta.schema_version != SCHEMA_VERSION {
        return Err(OracleError::schema(
            &json_path,
            format!("unsupported schema_version {:?}", meta.schema_version),
        ));
    }
    if meta.prompt_id != expected_prompt_id {
        return Err(OracleError::schema(
            &json_path,
            format!(
                "prompt_id mismatch: got {:?}, expected {:?}",
                meta.prompt_id, expected_prompt_id
            ),
        ));
    }
    if meta.dtype != "f32" || meta.endianness != "little" {
        return Err(OracleError::schema(
            &json_path,
            format!(
                "unsupported logits dtype/endianness ({:?}/{:?})",
                meta.dtype, meta.endianness
            ),
        ));
    }
    if meta.n_vocab != expected_n_vocab {
        return Err(OracleError::schema(
            &json_path,
            format!(
                "n_vocab {} does not match manifest hparams.n_vocab {expected_n_vocab}",
                meta.n_vocab
            ),
        ));
    }
    let expected_stride = u64::from(meta.n_vocab) * 4;
    if meta.row_stride_bytes != expected_stride {
        return Err(OracleError::schema(
            &json_path,
            format!(
                "row_stride_bytes {} != n_vocab * 4 ({expected_stride})",
                meta.row_stride_bytes
            ),
        ));
    }
    let expected_total = u64::from(meta.n_rows) * expected_stride;
    if meta.total_bytes != expected_total {
        return Err(OracleError::schema(
            &json_path,
            format!(
                "total_bytes {} != n_rows * row_stride_bytes ({expected_total})",
                meta.total_bytes
            ),
        ));
    }
    let n_rows = meta.n_rows as usize;
    if meta.positions.len() != n_rows
        || meta.input_token_ids.len() != n_rows
        || meta.rows_summary.len() != n_rows
    {
        return Err(OracleError::schema(
            &json_path,
            format!(
                "positions/input_token_ids/rows_summary lengths ({}/{}/{}) must all equal n_rows ({n_rows})",
                meta.positions.len(),
                meta.input_token_ids.len(),
                meta.rows_summary.len()
            ),
        ));
    }

    let bin_path = resolve_companion_bin(&json_path, dump_dir, &meta.bin_file);
    let bin_bytes = fs::read(&bin_path).map_err(|e| OracleError::io(&bin_path, e))?;
    if bin_bytes.len() as u64 != expected_total {
        return Err(OracleError::schema(
            &bin_path,
            format!(
                "logits binary size {} != expected {expected_total}",
                bin_bytes.len()
            ),
        ));
    }
    let actual_sha = sha256::sha256_hex(&bin_bytes);
    if actual_sha != meta.bin_sha256.to_ascii_lowercase() {
        return Err(OracleError::ChecksumMismatch {
            path: bin_path,
            expected: meta.bin_sha256,
            actual: actual_sha,
        });
    }
    let values = decode_f32_le(&bin_path, &bin_bytes)?;
    if let Some((idx, val)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(OracleError::schema(
            &bin_path,
            format!("non-finite logit value {val} at flat index {idx}"),
        ));
    }
    Ok((meta, values))
}

pub fn load_and_verify_residuals(
    dump_dir: &Path,
    rel_json: &str,
    expected_prompt_id: &str,
    expected_n_embd: u32,
) -> Result<(ResidualsMeta, Vec<u8>), OracleError> {
    let json_path = dump_dir.join(rel_json);
    let raw = fs::read_to_string(&json_path).map_err(|e| OracleError::io(&json_path, e))?;
    let meta: ResidualsMeta =
        serde_json::from_str(&raw).map_err(|e| OracleError::json(&json_path, e))?;
    if meta.schema_version != SCHEMA_VERSION {
        return Err(OracleError::schema(
            &json_path,
            format!("unsupported schema_version {:?}", meta.schema_version),
        ));
    }
    if meta.prompt_id != expected_prompt_id {
        return Err(OracleError::schema(
            &json_path,
            format!(
                "prompt_id mismatch: got {:?}, expected {:?}",
                meta.prompt_id, expected_prompt_id
            ),
        ));
    }
    if meta.dtype != "f32" || meta.endianness != "little" {
        return Err(OracleError::schema(
            &json_path,
            format!(
                "unsupported residuals dtype/endianness ({:?}/{:?})",
                meta.dtype, meta.endianness
            ),
        ));
    }
    if meta.n_embd != expected_n_embd {
        return Err(OracleError::schema(
            &json_path,
            format!(
                "n_embd {} does not match manifest hparams.n_embd {expected_n_embd}",
                meta.n_embd
            ),
        ));
    }
    if meta.snapshots.is_empty() {
        return Err(OracleError::schema(
            &json_path,
            "residuals snapshots list must not be empty",
        ));
    }

    let bin_path = resolve_companion_bin(&json_path, dump_dir, &meta.bin_file);
    let bin_bytes = fs::read(&bin_path).map_err(|e| OracleError::io(&bin_path, e))?;
    let actual_sha = sha256::sha256_hex(&bin_bytes);
    if actual_sha != meta.bin_sha256.to_ascii_lowercase() {
        return Err(OracleError::ChecksumMismatch {
            path: bin_path,
            expected: meta.bin_sha256,
            actual: actual_sha,
        });
    }

    let expected_vec_bytes = u64::from(meta.n_embd) * 4;
    for snap in &meta.snapshots {
        if ResidualTap::parse(&snap.tap).is_none() {
            return Err(OracleError::schema(
                &json_path,
                format!("invalid residual tap {:?}", snap.tap),
            ));
        }
        if snap.byte_len != expected_vec_bytes {
            return Err(OracleError::schema(
                &json_path,
                format!(
                    "snapshot {} byte_len {} != n_embd * 4 ({expected_vec_bytes})",
                    snap.snapshot_idx, snap.byte_len
                ),
            ));
        }
        let end = snap.byte_offset.saturating_add(snap.byte_len);
        if end > bin_bytes.len() as u64 {
            return Err(OracleError::schema(
                &bin_path,
                format!(
                    "snapshot {} range {}..{end} exceeds file length {}",
                    snap.snapshot_idx,
                    snap.byte_offset,
                    bin_bytes.len()
                ),
            ));
        }
        let slice = &bin_bytes[snap.byte_offset as usize..end as usize];
        let vals = decode_f32_le(&bin_path, slice)?;
        if let Some((dim, val)) = vals.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(OracleError::schema(
                &bin_path,
                format!(
                    "non-finite residual value {val} in snapshot {} dim {dim}",
                    snap.snapshot_idx
                ),
            ));
        }
    }
    Ok((meta, bin_bytes))
}

/// Extract a single snapshot's `Vec<f32>` from raw residual bytes.
pub fn extract_residual_snapshot(
    bin_path: &Path,
    bin_bytes: &[u8],
    snap: &ResidualSnapshotMeta,
) -> Result<Vec<f32>, OracleError> {
    let start = snap.byte_offset as usize;
    let end = start + snap.byte_len as usize;
    if end > bin_bytes.len() {
        return Err(OracleError::schema(
            bin_path,
            format!(
                "snapshot {} out of bounds ({start}..{end} > {})",
                snap.snapshot_idx,
                bin_bytes.len()
            ),
        ));
    }
    decode_f32_le(bin_path, &bin_bytes[start..end])
}

/// Verify an entire golden/candidate dump directory against Contract v0.
pub fn verify_dump_dir(dump_dir: &Path) -> Result<VerifySummary, OracleError> {
    let checksums_verified = verify_sha256sums(dump_dir)?;
    let manifest = load_manifest(dump_dir)?;
    let mut total_tokens = 0usize;
    let mut total_logit_rows = 0usize;
    let mut total_snapshots = 0usize;

    for prompt in &manifest.prompts {
        let (tok_meta, _) = load_and_verify_tokens(
            dump_dir,
            &prompt.tokens_json,
            &prompt.prompt_id,
            prompt.prompt_tokens_len,
            prompt.generated_tokens_len,
        )?;
        total_tokens += tok_meta.prompt_token_ids.len() + tok_meta.generated_token_ids.len();

        let (log_meta, _) = load_and_verify_logits(
            dump_dir,
            &prompt.logits_json,
            &prompt.prompt_id,
            manifest.hparams.n_vocab,
        )?;
        total_logit_rows += log_meta.n_rows as usize;

        let (res_meta, _) = load_and_verify_residuals(
            dump_dir,
            &prompt.residuals_json,
            &prompt.prompt_id,
            manifest.hparams.n_embd,
        )?;
        total_snapshots += res_meta.snapshots.len();

        // If flat mirror files exist under tokens/, logits/, residuals/, verify them too.
        let flat_tok = format!("tokens/{}.tokens.json", prompt.prompt_id);
        if dump_dir.join(&flat_tok).exists() && flat_tok != prompt.tokens_json {
            load_and_verify_tokens(
                dump_dir,
                &flat_tok,
                &prompt.prompt_id,
                prompt.prompt_tokens_len,
                prompt.generated_tokens_len,
            )?;
        }
        let flat_log = format!("logits/{}.logits.json", prompt.prompt_id);
        if dump_dir.join(&flat_log).exists() && flat_log != prompt.logits_json {
            load_and_verify_logits(
                dump_dir,
                &flat_log,
                &prompt.prompt_id,
                manifest.hparams.n_vocab,
            )?;
        }
        let flat_res = format!("residuals/{}.resid.json", prompt.prompt_id);
        if dump_dir.join(&flat_res).exists() && flat_res != prompt.residuals_json {
            load_and_verify_residuals(
                dump_dir,
                &flat_res,
                &prompt.prompt_id,
                manifest.hparams.n_embd,
            )?;
        }
    }

    Ok(VerifySummary {
        dump_dir: dump_dir.display().to_string(),
        model_id: manifest.model_id,
        prompts_verified: manifest.prompts.len(),
        checksums_verified,
        total_tokens_verified: total_tokens,
        total_logit_rows_verified: total_logit_rows,
        total_residual_snapshots_verified: total_snapshots,
    })
}

// ============================================================================
// Diff Harness (`diff <ref-dir> <cand-dir>`)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiffMode {
    All,
    Tokens,
    Logits,
    Residuals,
}

impl DiffMode {
    pub fn parse(s: &str) -> Result<Self, OracleError> {
        match s {
            "all" => Ok(Self::All),
            "tokens" => Ok(Self::Tokens),
            "logits" => Ok(Self::Logits),
            "residuals" => Ok(Self::Residuals),
            _ => Err(OracleError::InvalidArg(format!(
                "invalid mode {s:?} (expected tokens|logits|residuals|all)"
            ))),
        }
    }

    #[must_use]
    pub const fn checks_tokens(self) -> bool {
        matches!(self, Self::All | Self::Tokens)
    }

    #[must_use]
    pub const fn checks_logits(self) -> bool {
        matches!(self, Self::All | Self::Logits)
    }

    #[must_use]
    pub const fn checks_residuals(self) -> bool {
        matches!(self, Self::All | Self::Residuals)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiffConfig {
    pub prompt_filter: Option<String>,
    pub mode: DiffMode,
    pub max_ulp: u64,
    pub ulp_explicit: bool,
    pub abs_tol: f32,
    pub rel_tol: f32,
    pub max_rel_l2: Option<f64>,
    pub min_cosine: Option<f64>,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            prompt_filter: None,
            mode: DiffMode::All,
            max_ulp: 0,
            ulp_explicit: false,
            abs_tol: 0.0,
            rel_tol: 0.0,
            max_rel_l2: None,
            min_cosine: None,
        }
    }
}

impl DiffConfig {
    #[must_use]
    pub fn element_passes(&self, expected: f32, actual: f32, abs_diff: f32, ulp_diff: u64) -> bool {
        if !expected.is_finite() || !actual.is_finite() {
            return false;
        }
        let has_float_tol = self.abs_tol > 0.0 || self.rel_tol > 0.0;
        if !has_float_tol {
            return ulp_diff <= self.max_ulp;
        }
        let allowed_abs = self.abs_tol + self.rel_tol * expected.abs();
        if self.ulp_explicit {
            ulp_diff <= self.max_ulp && abs_diff <= allowed_abs
        } else {
            ulp_diff <= self.max_ulp || abs_diff <= allowed_abs
        }
    }

    #[must_use]
    pub fn vector_passes(&self, cmp: &VectorComparison) -> bool {
        if cmp.has_non_finite || cmp.first_failing_idx.is_some() {
            return false;
        }
        if let Some(max_l2) = self.max_rel_l2 {
            if cmp.l2_rel_err > max_l2 {
                return false;
            }
        }
        if let Some(min_cos) = self.min_cosine {
            if cmp.cosine_similarity < min_cos {
                return false;
            }
        }
        true
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffStatus {
    #[serde(rename = "PASS")]
    Pass,
    #[serde(rename = "FAIL")]
    Fail,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FirstTokenDiff {
    pub prompt_id: String,
    pub is_prompt_token: bool,
    pub step: u32,
    pub token_pos: u32,
    pub expected_token_id: u32,
    pub actual_token_id: u32,
    pub golden_token_id: u32,
    pub candidate_token_id: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FirstLogitRowDiff {
    pub prompt_id: String,
    pub row_idx: u32,
    pub pos: u32,
    pub token_pos: u32,
    pub vocab_idx: u32,
    pub dim_idx: u32,
    pub max_diff_vocab_idx: u32,
    pub expected: f32,
    pub expected_hex: String,
    pub actual: f32,
    pub actual_hex: String,
    pub abs_diff: f32,
    pub ulp_diff: u64,
    pub max_abs_diff: f32,
    pub max_ulp_diff: u64,
    pub l2_rel_err: f64,
    pub cosine_similarity: f64,
    pub argmax_expected: u32,
    pub argmax_actual: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FirstResidualLayerDiff {
    pub prompt_id: String,
    pub snapshot_idx: u32,
    pub token_pos: u32,
    pub layer_idx: i32,
    pub tap: String,
    pub dim_idx: u32,
    pub max_diff_dim_idx: u32,
    pub expected: f32,
    pub expected_hex: String,
    pub actual: f32,
    pub actual_hex: String,
    pub abs_diff: f32,
    pub ulp_diff: u64,
    pub max_abs_diff: f32,
    pub max_ulp_diff: u64,
    pub l2_rel_err: f64,
    pub cosine_similarity: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptDiffReport {
    pub prompt_id: String,
    pub status: DiffStatus,
    pub tokens_matched: bool,
    pub logit_rows_checked: usize,
    pub logit_rows_matched: bool,
    pub residual_snapshots_checked: usize,
    pub residuals_matched: bool,
    pub first_token_diff: Option<FirstTokenDiff>,
    pub first_logit_row_diff: Option<FirstLogitRowDiff>,
    pub first_residual_layer_diff: Option<FirstResidualLayerDiff>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiffReport {
    pub status: DiffStatus,
    pub golden_dir: String,
    pub candidate_dir: String,
    pub model_id: String,
    pub prompts_checked: usize,
    pub prompts_passed: usize,
    pub prompts_failed: usize,
    pub first_diverging_prompt: Option<String>,
    pub first_diverging_token: Option<FirstTokenDiff>,
    pub first_diverging_logit_row: Option<FirstLogitRowDiff>,
    pub first_diverging_layer: Option<FirstResidualLayerDiff>,
    pub prompts: Vec<PromptDiffReport>,
}

impl DiffReport {
    #[must_use]
    pub const fn is_pass(&self) -> bool {
        matches!(self.status, DiffStatus::Pass)
    }

    #[must_use]
    pub const fn exit_code(&self) -> i32 {
        if self.is_pass() {
            EXIT_PASS
        } else {
            EXIT_DIFF
        }
    }

    /// Format the human-readable report specified by Contract v0 (`docs/golden-dump-format.md` §5.4).
    #[must_use]
    pub fn format_human(&self) -> String {
        let mut s = String::new();
        let status_str = match self.status {
            DiffStatus::Pass => "PASS",
            DiffStatus::Fail => "FAIL",
        };
        let _ = writeln!(s, "RESULT: {status_str}");
        let _ = writeln!(s, "model_id: {}", self.model_id);
        let _ = writeln!(
            s,
            "prompts_checked: {} (passed: {}, failed: {})",
            self.prompts_checked, self.prompts_passed, self.prompts_failed
        );
        if let Some(prompt_id) = &self.first_diverging_prompt {
            let _ = writeln!(s, "first_diverging_prompt: {prompt_id}");
        }
        if let Some(tok) = &self.first_diverging_token {
            let _ = writeln!(
                s,
                "first_diverging_token: step={} token_pos={} expected_token_id={} actual_token_id={}",
                tok.step, tok.token_pos, tok.expected_token_id, tok.actual_token_id
            );
        }
        if let Some(row) = &self.first_diverging_logit_row {
            let _ = writeln!(
                s,
                "first_diverging_logit_row: row_idx={} pos={} vocab_idx={} expected={:.8} ({}) actual={:.8} ({}) abs_diff={:.6e} ulp_diff={} max_abs_diff={:.6e} l2_rel_err={:.6e} cosine_sim={:.12} argmax_expected={} argmax_actual={}",
                row.row_idx,
                row.pos,
                row.vocab_idx,
                row.expected,
                row.expected_hex,
                row.actual,
                row.actual_hex,
                row.abs_diff,
                row.ulp_diff,
                row.max_abs_diff,
                row.l2_rel_err,
                row.cosine_similarity,
                row.argmax_expected,
                row.argmax_actual
            );
        }
        if let Some(res) = &self.first_diverging_layer {
            let _ = writeln!(
                s,
                "first_diverging_layer: token_pos={} layer_idx={} tap={}",
                res.token_pos, res.layer_idx, res.tap
            );
            let _ = writeln!(s, "dim_idx: {}", res.dim_idx);
            let _ = writeln!(s, "golden_val: {:.8} ({})", res.expected, res.expected_hex);
            let _ = writeln!(s, "candidate_val: {:.8} ({})", res.actual, res.actual_hex);
            let _ = writeln!(
                s,
                "abs_diff: {:.6e}  ulp_diff: {}  l2_rel_err: {:.6e}  cosine_sim: {:.12}",
                res.abs_diff, res.ulp_diff, res.l2_rel_err, res.cosine_similarity
            );
        }
        s
    }
}

/// Compare a golden dump directory and a candidate dump directory according to `cfg`.
pub fn diff_dump_dirs(
    golden_dir: &Path,
    candidate_dir: &Path,
    cfg: &DiffConfig,
) -> Result<DiffReport, OracleError> {
    verify_sha256sums(golden_dir)?;
    verify_sha256sums(candidate_dir)?;
    let golden_manifest = load_manifest(golden_dir)?;
    let cand_manifest = load_manifest(candidate_dir)?;

    if golden_manifest.model_id != cand_manifest.model_id {
        return Err(OracleError::schema(
            candidate_dir.join("manifest.json"),
            format!(
                "model_id mismatch: golden {:?} vs candidate {:?}",
                golden_manifest.model_id, cand_manifest.model_id
            ),
        ));
    }

    let cand_prompts: BTreeMap<&str, &PromptEntry> = cand_manifest
        .prompts
        .iter()
        .map(|p| (p.prompt_id.as_str(), p))
        .collect();

    let mut prompt_reports = Vec::new();
    let mut first_diverging_prompt = None;
    let mut first_diverging_token = None;
    let mut first_diverging_logit_row = None;
    let mut first_diverging_layer = None;
    let mut prompts_passed = 0usize;
    let mut prompts_failed = 0usize;

    for g_prompt in &golden_manifest.prompts {
        if let Some(filter) = &cfg.prompt_filter {
            if &g_prompt.prompt_id != filter {
                continue;
            }
        }
        let c_prompt = cand_prompts
            .get(g_prompt.prompt_id.as_str())
            .copied()
            .ok_or_else(|| {
                OracleError::schema(
                    candidate_dir.join("manifest.json"),
                    format!(
                        "missing prompt {:?} in candidate manifest",
                        g_prompt.prompt_id
                    ),
                )
            })?;

        let mut p_tok_diff = None;
        let mut p_log_diff = None;
        let mut p_res_diff = None;
        let mut logit_rows_checked = 0usize;
        let mut residual_snapshots_checked = 0usize;

        if cfg.mode.checks_tokens() {
            let (_, g_tok) = load_and_verify_tokens(
                golden_dir,
                &g_prompt.tokens_json,
                &g_prompt.prompt_id,
                g_prompt.prompt_tokens_len,
                g_prompt.generated_tokens_len,
            )?;
            let (_, c_tok) = load_and_verify_tokens(
                candidate_dir,
                &c_prompt.tokens_json,
                &c_prompt.prompt_id,
                c_prompt.prompt_tokens_len,
                c_prompt.generated_tokens_len,
            )?;
            p_tok_diff = diff_tokens(&g_prompt.prompt_id, &g_tok, &c_tok);
        }

        if cfg.mode.checks_logits() {
            let (g_log_meta, g_log_vals) = load_and_verify_logits(
                golden_dir,
                &g_prompt.logits_json,
                &g_prompt.prompt_id,
                golden_manifest.hparams.n_vocab,
            )?;
            let (c_log_meta, c_log_vals) = load_and_verify_logits(
                candidate_dir,
                &c_prompt.logits_json,
                &c_prompt.prompt_id,
                cand_manifest.hparams.n_vocab,
            )?;
            logit_rows_checked = g_log_meta.n_rows as usize;
            p_log_diff = diff_logits(
                &g_prompt.prompt_id,
                &g_log_meta,
                &g_log_vals,
                &c_log_meta,
                &c_log_vals,
                cfg,
            )?;
        }

        if cfg.mode.checks_residuals() {
            let (g_res_meta, g_res_bytes) = load_and_verify_residuals(
                golden_dir,
                &g_prompt.residuals_json,
                &g_prompt.prompt_id,
                golden_manifest.hparams.n_embd,
            )?;
            let (c_res_meta, c_res_bytes) = load_and_verify_residuals(
                candidate_dir,
                &c_prompt.residuals_json,
                &c_prompt.prompt_id,
                cand_manifest.hparams.n_embd,
            )?;
            residual_snapshots_checked = g_res_meta.snapshots.len();
            p_res_diff = diff_residuals(
                &g_prompt.prompt_id,
                golden_dir,
                &g_res_meta,
                &g_res_bytes,
                candidate_dir,
                &c_res_meta,
                &c_res_bytes,
                cfg,
            )?;
        }

        let tokens_matched = p_tok_diff.is_none();
        let logit_rows_matched = p_log_diff.is_none();
        let residuals_matched = p_res_diff.is_none();
        let passed = tokens_matched && logit_rows_matched && residuals_matched;
        if passed {
            prompts_passed += 1;
        } else {
            prompts_failed += 1;
            if first_diverging_prompt.is_none() {
                first_diverging_prompt = Some(g_prompt.prompt_id.clone());
            }
            if first_diverging_token.is_none() {
                first_diverging_token.clone_from(&p_tok_diff);
            }
            if first_diverging_logit_row.is_none() {
                first_diverging_logit_row.clone_from(&p_log_diff);
            }
            if first_diverging_layer.is_none() {
                first_diverging_layer.clone_from(&p_res_diff);
            }
        }

        prompt_reports.push(PromptDiffReport {
            prompt_id: g_prompt.prompt_id.clone(),
            status: if passed {
                DiffStatus::Pass
            } else {
                DiffStatus::Fail
            },
            tokens_matched,
            logit_rows_checked,
            logit_rows_matched,
            residual_snapshots_checked,
            residuals_matched,
            first_token_diff: p_tok_diff,
            first_logit_row_diff: p_log_diff,
            first_residual_layer_diff: p_res_diff,
        });
    }

    if prompt_reports.is_empty() {
        return Err(OracleError::InvalidArg(format!(
            "no prompts matched filter {:?}",
            cfg.prompt_filter
        )));
    }

    let status = if prompts_failed == 0 {
        DiffStatus::Pass
    } else {
        DiffStatus::Fail
    };

    Ok(DiffReport {
        status,
        golden_dir: golden_dir.display().to_string(),
        candidate_dir: candidate_dir.display().to_string(),
        model_id: golden_manifest.model_id,
        prompts_checked: prompt_reports.len(),
        prompts_passed,
        prompts_failed,
        first_diverging_prompt,
        first_diverging_token,
        first_diverging_logit_row,
        first_diverging_layer,
        prompts: prompt_reports,
    })
}

fn diff_tokens(
    prompt_id: &str,
    golden: &TokensBinData,
    cand: &TokensBinData,
) -> Option<FirstTokenDiff> {
    let min_p = golden
        .prompt_token_ids
        .len()
        .min(cand.prompt_token_ids.len());
    for i in 0..min_p {
        let g = golden.prompt_token_ids[i];
        let c = cand.prompt_token_ids[i];
        if g != c {
            return Some(FirstTokenDiff {
                prompt_id: prompt_id.to_string(),
                is_prompt_token: true,
                step: 0,
                token_pos: i as u32,
                expected_token_id: g,
                actual_token_id: c,
                golden_token_id: g,
                candidate_token_id: c,
            });
        }
    }
    if golden.prompt_token_ids.len() != cand.prompt_token_ids.len() {
        let g = golden
            .prompt_token_ids
            .get(min_p)
            .copied()
            .unwrap_or(u32::MAX);
        let c = cand
            .prompt_token_ids
            .get(min_p)
            .copied()
            .unwrap_or(u32::MAX);
        return Some(FirstTokenDiff {
            prompt_id: prompt_id.to_string(),
            is_prompt_token: true,
            step: 0,
            token_pos: min_p as u32,
            expected_token_id: g,
            actual_token_id: c,
            golden_token_id: g,
            candidate_token_id: c,
        });
    }

    let prompt_len = golden.prompt_token_ids.len() as u32;
    let min_g = golden
        .generated_token_ids
        .len()
        .min(cand.generated_token_ids.len());
    for step in 0..min_g {
        let g = golden.generated_token_ids[step];
        let c = cand.generated_token_ids[step];
        if g != c {
            let step_u32 = step as u32;
            return Some(FirstTokenDiff {
                prompt_id: prompt_id.to_string(),
                is_prompt_token: false,
                step: step_u32,
                token_pos: prompt_len + step_u32,
                expected_token_id: g,
                actual_token_id: c,
                golden_token_id: g,
                candidate_token_id: c,
            });
        }
    }
    if golden.generated_token_ids.len() != cand.generated_token_ids.len() {
        let step_u32 = min_g as u32;
        let g = golden
            .generated_token_ids
            .get(min_g)
            .copied()
            .unwrap_or(u32::MAX);
        let c = cand
            .generated_token_ids
            .get(min_g)
            .copied()
            .unwrap_or(u32::MAX);
        return Some(FirstTokenDiff {
            prompt_id: prompt_id.to_string(),
            is_prompt_token: false,
            step: step_u32,
            token_pos: prompt_len + step_u32,
            expected_token_id: g,
            actual_token_id: c,
            golden_token_id: g,
            candidate_token_id: c,
        });
    }
    None
}

fn diff_logits(
    prompt_id: &str,
    g_meta: &LogitsMeta,
    g_vals: &[f32],
    c_meta: &LogitsMeta,
    c_vals: &[f32],
    cfg: &DiffConfig,
) -> Result<Option<FirstLogitRowDiff>, OracleError> {
    if g_meta.n_rows != c_meta.n_rows || g_meta.n_vocab != c_meta.n_vocab {
        return Err(OracleError::InvalidArg(format!(
            "logits shape mismatch for {prompt_id}: golden ({}, {}) vs candidate ({}, {})",
            g_meta.n_rows, g_meta.n_vocab, c_meta.n_rows, c_meta.n_vocab
        )));
    }
    let n_vocab = g_meta.n_vocab as usize;
    for r in 0..g_meta.n_rows as usize {
        let start = r * n_vocab;
        let end = start + n_vocab;
        let g_row = &g_vals[start..end];
        let c_row = &c_vals[start..end];
        let cmp = compare_f32_vectors(g_row, c_row, cfg);
        if !cfg.vector_passes(&cmp) {
            let vocab_idx = cmp.first_failing_idx.unwrap_or(cmp.max_diff_idx);
            let expected = g_row[vocab_idx as usize];
            let actual = c_row[vocab_idx as usize];
            let abs_diff = (expected - actual).abs();
            let ulp_diff = ulp_diff_f32(expected, actual);
            let pos = g_meta.positions[r];
            return Ok(Some(FirstLogitRowDiff {
                prompt_id: prompt_id.to_string(),
                row_idx: r as u32,
                pos,
                token_pos: pos,
                vocab_idx,
                dim_idx: vocab_idx,
                max_diff_vocab_idx: cmp.max_diff_idx,
                expected,
                expected_hex: format_f32_hex(expected),
                actual,
                actual_hex: format_f32_hex(actual),
                abs_diff,
                ulp_diff,
                max_abs_diff: cmp.max_abs_diff,
                max_ulp_diff: cmp.max_ulp_diff,
                l2_rel_err: cmp.l2_rel_err,
                cosine_similarity: cmp.cosine_similarity,
                argmax_expected: cmp.argmax_expected,
                argmax_actual: cmp.argmax_actual,
            }));
        }
    }
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn diff_residuals(
    prompt_id: &str,
    golden_dir: &Path,
    g_meta: &ResidualsMeta,
    g_bytes: &[u8],
    cand_dir: &Path,
    c_meta: &ResidualsMeta,
    c_bytes: &[u8],
    cfg: &DiffConfig,
) -> Result<Option<FirstResidualLayerDiff>, OracleError> {
    if g_meta.n_embd != c_meta.n_embd || g_meta.snapshots.len() != c_meta.snapshots.len() {
        return Err(OracleError::InvalidArg(format!(
            "residuals metadata mismatch for {prompt_id}: golden (n_embd={}, snaps={}) vs candidate (n_embd={}, snaps={})",
            g_meta.n_embd,
            g_meta.snapshots.len(),
            c_meta.n_embd,
            c_meta.snapshots.len()
        )));
    }

    let mut c_map: BTreeMap<(u32, i32, u8), &ResidualSnapshotMeta> = BTreeMap::new();
    for snap in &c_meta.snapshots {
        let tap_ord = ResidualTap::parse(&snap.tap).map_or(255, ResidualTap::order);
        c_map.insert((snap.token_pos, snap.layer_idx, tap_ord), snap);
    }

    let mut g_sorted: Vec<&ResidualSnapshotMeta> = g_meta.snapshots.iter().collect();
    g_sorted.sort_by_key(|s| {
        let tap_ord = ResidualTap::parse(&s.tap).map_or(255, ResidualTap::order);
        (s.token_pos, s.layer_idx, tap_ord, s.snapshot_idx)
    });

    let g_bin_path = golden_dir.join(&g_meta.bin_file);
    let c_bin_path = cand_dir.join(&c_meta.bin_file);

    for g_snap in g_sorted {
        let tap_ord = ResidualTap::parse(&g_snap.tap).map_or(255, ResidualTap::order);
        let c_snap = c_map
            .get(&(g_snap.token_pos, g_snap.layer_idx, tap_ord))
            .copied()
            .ok_or_else(|| {
                OracleError::InvalidArg(format!(
                    "candidate missing residual snapshot (token_pos={}, layer_idx={}, tap={}) for {prompt_id}",
                    g_snap.token_pos, g_snap.layer_idx, g_snap.tap
                ))
            })?;
        let g_vec = extract_residual_snapshot(&g_bin_path, g_bytes, g_snap)?;
        let c_vec = extract_residual_snapshot(&c_bin_path, c_bytes, c_snap)?;
        let cmp = compare_f32_vectors(&g_vec, &c_vec, cfg);
        if !cfg.vector_passes(&cmp) {
            let dim_idx = cmp.first_failing_idx.unwrap_or(cmp.max_diff_idx);
            let expected = g_vec[dim_idx as usize];
            let actual = c_vec[dim_idx as usize];
            let abs_diff = (expected - actual).abs();
            let ulp_diff = ulp_diff_f32(expected, actual);
            return Ok(Some(FirstResidualLayerDiff {
                prompt_id: prompt_id.to_string(),
                snapshot_idx: g_snap.snapshot_idx,
                token_pos: g_snap.token_pos,
                layer_idx: g_snap.layer_idx,
                tap: g_snap.tap.clone(),
                dim_idx,
                max_diff_dim_idx: cmp.max_diff_idx,
                expected,
                expected_hex: format_f32_hex(expected),
                actual,
                actual_hex: format_f32_hex(actual),
                abs_diff,
                ulp_diff,
                max_abs_diff: cmp.max_abs_diff,
                max_ulp_diff: cmp.max_ulp_diff,
                l2_rel_err: cmp.l2_rel_err,
                cosine_similarity: cmp.cosine_similarity,
            }));
        }
    }
    Ok(None)
}

// ============================================================================
// Bitflip Injection (`inject-bitflip <src-dir> <dst-dir>`)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BitflipTarget {
    Tokens,
    Logits,
    Residuals,
}

impl BitflipTarget {
    pub fn parse(s: &str) -> Result<Self, OracleError> {
        match s {
            "tokens" => Ok(Self::Tokens),
            "logits" => Ok(Self::Logits),
            "residuals" => Ok(Self::Residuals),
            _ => Err(OracleError::InvalidArg(format!(
                "invalid bitflip target {s:?} (expected tokens|logits|residuals)"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BitflipOptions {
    pub prompt_id: Option<String>,
    pub target: BitflipTarget,
    pub step: usize,
    pub row_idx: usize,
    pub vocab_idx: usize,
    pub snapshot_idx: Option<usize>,
    pub token_pos: Option<u32>,
    pub layer_idx: Option<i32>,
    pub tap: Option<String>,
    pub dim_idx: usize,
    pub bit: u32,
}

impl Default for BitflipOptions {
    fn default() -> Self {
        Self {
            prompt_id: None,
            target: BitflipTarget::Logits,
            step: 0,
            row_idx: 0,
            vocab_idx: 0,
            snapshot_idx: None,
            token_pos: None,
            layer_idx: None,
            tap: None,
            dim_idx: 0,
            bit: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BitflipSummary {
    pub prompt_id: String,
    pub target: BitflipTarget,
    pub bit: u32,
    pub detail: String,
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), OracleError> {
    fs::create_dir_all(dst).map_err(|e| OracleError::io(dst, e))?;
    let entries = fs::read_dir(src).map_err(|e| OracleError::io(src, e))?;
    for entry in entries {
        let entry = entry.map_err(|e| OracleError::io(src, e))?;
        let s_path = entry.path();
        let d_path = dst.join(entry.file_name());
        let ft = entry.file_type().map_err(|e| OracleError::io(&s_path, e))?;
        if ft.is_dir() {
            copy_dir_recursive(&s_path, &d_path)?;
        } else if ft.is_file() {
            fs::copy(&s_path, &d_path).map_err(|e| OracleError::io(&d_path, e))?;
        }
    }
    Ok(())
}

/// Copy `src_dir` to `dst_dir`, inject a 1-bit flip in the target artifact, and update checksums.
pub fn inject_bitflip(
    src_dir: &Path,
    dst_dir: &Path,
    opts: &BitflipOptions,
) -> Result<BitflipSummary, OracleError> {
    if opts.bit >= 32 {
        return Err(OracleError::InvalidArg(format!(
            "bit index {} out of range 0..32",
            opts.bit
        )));
    }
    if dst_dir.exists() {
        fs::remove_dir_all(dst_dir).map_err(|e| OracleError::io(dst_dir, e))?;
    }
    copy_dir_recursive(src_dir, dst_dir)?;

    let manifest = load_manifest(dst_dir)?;
    let prompt = if let Some(pid) = &opts.prompt_id {
        manifest
            .prompts
            .iter()
            .find(|p| &p.prompt_id == pid)
            .ok_or_else(|| {
                OracleError::InvalidArg(format!("prompt_id {pid:?} not found in manifest"))
            })?
    } else {
        &manifest.prompts[0]
    };

    let mask = 1u32 << opts.bit;
    let detail = match opts.target {
        BitflipTarget::Tokens => {
            let (mut meta, _) = load_and_verify_tokens(
                dst_dir,
                &prompt.tokens_json,
                &prompt.prompt_id,
                prompt.prompt_tokens_len,
                prompt.generated_tokens_len,
            )?;
            if opts.step >= meta.generated_token_ids.len() {
                return Err(OracleError::InvalidArg(format!(
                    "step {} >= generated_token_ids.len() ({})",
                    opts.step,
                    meta.generated_token_ids.len()
                )));
            }
            let before = meta.generated_token_ids[opts.step];
            let after = before ^ mask;
            meta.generated_token_ids[opts.step] = after;
            let new_bin = encode_tokens_bin(&meta.prompt_token_ids, &meta.generated_token_ids);
            meta.bin_sha256 = sha256::sha256_hex(&new_bin);

            let json_path = dst_dir.join(&prompt.tokens_json);
            let bin_path = resolve_companion_bin(&json_path, dst_dir, &meta.bin_file);
            fs::write(&bin_path, &new_bin).map_err(|e| OracleError::io(&bin_path, e))?;
            let json_text = serde_json::to_string_pretty(&meta)
                .map_err(|e| OracleError::json(&json_path, e))?;
            fs::write(&json_path, format!("{json_text}\n"))
                .map_err(|e| OracleError::io(&json_path, e))?;

            // Also update flat mirror in tokens/ if present.
            let flat_json = dst_dir.join(format!("tokens/{}.tokens.json", prompt.prompt_id));
            if flat_json.exists() && flat_json != json_path {
                let flat_bin = resolve_companion_bin(&flat_json, dst_dir, &meta.bin_file);
                if flat_bin != bin_path && flat_bin.exists() {
                    fs::write(&flat_bin, &new_bin).map_err(|e| OracleError::io(&flat_bin, e))?;
                }
                fs::write(&flat_json, format!("{json_text}\n"))
                    .map_err(|e| OracleError::io(&flat_json, e))?;
            }

            format!(
                "step={} token_pos={} before={} after={}",
                opts.step,
                meta.prompt_tokens_len + opts.step as u32,
                before,
                after
            )
        }
        BitflipTarget::Logits => {
            let (mut meta, mut vals) = load_and_verify_logits(
                dst_dir,
                &prompt.logits_json,
                &prompt.prompt_id,
                manifest.hparams.n_vocab,
            )?;
            let n_rows = meta.n_rows as usize;
            let n_vocab = meta.n_vocab as usize;
            if opts.row_idx >= n_rows || opts.vocab_idx >= n_vocab {
                return Err(OracleError::InvalidArg(format!(
                    "logit index (row={}, vocab={}) out of bounds ({n_rows}, {n_vocab})",
                    opts.row_idx, opts.vocab_idx
                )));
            }
            let flat_idx = opts.row_idx * n_vocab + opts.vocab_idx;
            let before = vals[flat_idx];
            let after = f32::from_bits(before.to_bits() ^ mask);
            vals[flat_idx] = after;
            let row_slice = &vals[opts.row_idx * n_vocab..(opts.row_idx + 1) * n_vocab];
            meta.rows_summary[opts.row_idx] = summarize_logit_row(
                opts.row_idx as u32,
                meta.positions[opts.row_idx],
                meta.input_token_ids[opts.row_idx],
                row_slice,
            );
            let new_bin = encode_f32_le(&vals);
            meta.bin_sha256 = sha256::sha256_hex(&new_bin);

            let json_path = dst_dir.join(&prompt.logits_json);
            let bin_path = resolve_companion_bin(&json_path, dst_dir, &meta.bin_file);
            fs::write(&bin_path, &new_bin).map_err(|e| OracleError::io(&bin_path, e))?;
            let json_text = serde_json::to_string_pretty(&meta)
                .map_err(|e| OracleError::json(&json_path, e))?;
            fs::write(&json_path, format!("{json_text}\n"))
                .map_err(|e| OracleError::io(&json_path, e))?;

            let flat_json = dst_dir.join(format!("logits/{}.logits.json", prompt.prompt_id));
            if flat_json.exists() && flat_json != json_path {
                let mut flat_meta = meta.clone();
                let flat_f32 = dst_dir.join(format!("logits/{}.logits.f32", prompt.prompt_id));
                if flat_f32.exists() {
                    fs::write(&flat_f32, &new_bin).map_err(|e| OracleError::io(&flat_f32, e))?;
                    flat_meta.bin_file = format!("{}.logits.f32", prompt.prompt_id);
                }
                let flat_text = serde_json::to_string_pretty(&flat_meta)
                    .map_err(|e| OracleError::json(&flat_json, e))?;
                fs::write(&flat_json, format!("{flat_text}\n"))
                    .map_err(|e| OracleError::io(&flat_json, e))?;
            }

            format!(
                "row_idx={} pos={} vocab_idx={} before={} ({}) after={} ({})",
                opts.row_idx,
                meta.positions[opts.row_idx],
                opts.vocab_idx,
                before,
                format_f32_hex(before),
                after,
                format_f32_hex(after)
            )
        }
        BitflipTarget::Residuals => {
            let (mut meta, mut bin_bytes) = load_and_verify_residuals(
                dst_dir,
                &prompt.residuals_json,
                &prompt.prompt_id,
                manifest.hparams.n_embd,
            )?;
            let n_embd = meta.n_embd as usize;
            if opts.dim_idx >= n_embd {
                return Err(OracleError::InvalidArg(format!(
                    "dim_idx {} >= n_embd ({n_embd})",
                    opts.dim_idx
                )));
            }
            let snap_pos = if let Some(sidx) = opts.snapshot_idx {
                meta.snapshots
                    .iter()
                    .position(|s| s.snapshot_idx as usize == sidx)
                    .ok_or_else(|| {
                        OracleError::InvalidArg(format!("snapshot_idx {sidx} not found"))
                    })?
            } else {
                meta.snapshots
                    .iter()
                    .position(|s| {
                        opts.token_pos.is_none_or(|p| s.token_pos == p)
                            && opts.layer_idx.is_none_or(|l| s.layer_idx == l)
                            && opts.tap.as_deref().is_none_or(|t| s.tap == t)
                    })
                    .ok_or_else(|| {
                        OracleError::InvalidArg(
                            "no matching residual snapshot found for filter".to_string(),
                        )
                    })?
            };

            let snap = &mut meta.snapshots[snap_pos];
            let byte_off = snap.byte_offset as usize + opts.dim_idx * 4;
            let before_bits = u32::from_le_bytes([
                bin_bytes[byte_off],
                bin_bytes[byte_off + 1],
                bin_bytes[byte_off + 2],
                bin_bytes[byte_off + 3],
            ]);
            let after_bits = before_bits ^ mask;
            bin_bytes[byte_off..byte_off + 4].copy_from_slice(&after_bits.to_le_bytes());

            let vec_start = snap.byte_offset as usize;
            let vec_end = vec_start + snap.byte_len as usize;
            let updated_vec =
                decode_f32_le(Path::new(&meta.bin_file), &bin_bytes[vec_start..vec_end])?;
            let (l2, rms) = slice_l2_and_rms(&updated_vec);
            snap.l2_norm = l2;
            snap.rms = rms;
            let token_pos = snap.token_pos;
            let layer_idx = snap.layer_idx;
            let tap = snap.tap.clone();

            meta.bin_sha256 = sha256::sha256_hex(&bin_bytes);

            let json_path = dst_dir.join(&prompt.residuals_json);
            let bin_path = resolve_companion_bin(&json_path, dst_dir, &meta.bin_file);
            fs::write(&bin_path, &bin_bytes).map_err(|e| OracleError::io(&bin_path, e))?;
            let json_text = serde_json::to_string_pretty(&meta)
                .map_err(|e| OracleError::json(&json_path, e))?;
            fs::write(&json_path, format!("{json_text}\n"))
                .map_err(|e| OracleError::io(&json_path, e))?;

            let flat_json = dst_dir.join(format!("residuals/{}.resid.json", prompt.prompt_id));
            if flat_json.exists() && flat_json != json_path {
                let mut flat_meta = meta.clone();
                let flat_f32 = dst_dir.join(format!("residuals/{}.resid.f32", prompt.prompt_id));
                if flat_f32.exists() {
                    fs::write(&flat_f32, &bin_bytes).map_err(|e| OracleError::io(&flat_f32, e))?;
                    flat_meta.bin_file = format!("{}.resid.f32", prompt.prompt_id);
                }
                let flat_text = serde_json::to_string_pretty(&flat_meta)
                    .map_err(|e| OracleError::json(&flat_json, e))?;
                fs::write(&flat_json, format!("{flat_text}\n"))
                    .map_err(|e| OracleError::io(&flat_json, e))?;
            }

            let before = f32::from_bits(before_bits);
            let after = f32::from_bits(after_bits);
            format!(
                "token_pos={token_pos} layer_idx={layer_idx} tap={tap} dim_idx={} before={} ({}) after={} ({})",
                opts.dim_idx,
                before,
                format_f32_hex(before),
                after,
                format_f32_hex(after)
            )
        }
    };

    write_sha256sums(dst_dir)?;

    Ok(BitflipSummary {
        prompt_id: prompt.prompt_id.clone(),
        target: opts.target,
        bit: opts.bit,
        detail,
    })
}

// ============================================================================
// Synthetic / Programmatic Dump Writer (used by tests & recording tools)
// ============================================================================

/// Input data for writing one prompt's golden dump files.
#[derive(Clone, Debug)]
pub struct PromptDumpInput {
    pub prompt_id: String,
    pub category: String,
    pub eos_token_id: u32,
    pub stop_reason: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub logit_positions: Vec<u32>,
    pub logit_input_token_ids: Vec<u32>,
    pub n_vocab: u32,
    pub logits_f32: Vec<f32>,
    pub n_embd: u32,
    pub n_layer: u32,
    /// `(token_pos, layer_idx, tap, vector_f32)`
    pub residual_snapshots: Vec<(u32, i32, ResidualTap, Vec<f32>)>,
}

/// Write a complete Contract v0 dump directory (`manifest.json`, `prompts/<id>/...`,
/// plus flat `tokens/`, `logits/`, `residuals/`, and `SHA256SUMS`).
pub fn write_golden_dump_dir(
    dump_dir: &Path,
    mut manifest: Manifest,
    prompts_data: &[PromptDumpInput],
) -> Result<(), OracleError> {
    fs::create_dir_all(dump_dir).map_err(|e| OracleError::io(dump_dir, e))?;
    let tok_flat_dir = dump_dir.join("tokens");
    let log_flat_dir = dump_dir.join("logits");
    let res_flat_dir = dump_dir.join("residuals");
    fs::create_dir_all(&tok_flat_dir).map_err(|e| OracleError::io(&tok_flat_dir, e))?;
    fs::create_dir_all(&log_flat_dir).map_err(|e| OracleError::io(&log_flat_dir, e))?;
    fs::create_dir_all(&res_flat_dir).map_err(|e| OracleError::io(&res_flat_dir, e))?;

    manifest.prompts.clear();

    for p in prompts_data {
        let p_dir = dump_dir.join("prompts").join(&p.prompt_id);
        fs::create_dir_all(&p_dir).map_err(|e| OracleError::io(&p_dir, e))?;

        // 1. Tokens
        let tok_bin_bytes = encode_tokens_bin(&p.prompt_token_ids, &p.generated_token_ids);
        let tok_bin_sha = sha256::sha256_hex(&tok_bin_bytes);
        let tok_bin_name = format!("{}.tokens.bin", p.prompt_id);
        let tok_bin_path = p_dir.join(&tok_bin_name);
        fs::write(&tok_bin_path, &tok_bin_bytes).map_err(|e| OracleError::io(&tok_bin_path, e))?;
        let flat_tok_bin_path = tok_flat_dir.join(&tok_bin_name);
        fs::write(&flat_tok_bin_path, &tok_bin_bytes)
            .map_err(|e| OracleError::io(&flat_tok_bin_path, e))?;

        let tok_meta = TokensMeta {
            schema_version: SCHEMA_VERSION.to_string(),
            prompt_id: p.prompt_id.clone(),
            bin_file: tok_bin_name,
            bin_sha256: tok_bin_sha,
            prompt_tokens_len: p.prompt_token_ids.len() as u32,
            generated_tokens_len: p.generated_token_ids.len() as u32,
            eos_token_id: p.eos_token_id,
            stop_reason: p.stop_reason.clone(),
            prompt_token_ids: p.prompt_token_ids.clone(),
            generated_token_ids: p.generated_token_ids.clone(),
        };
        let tok_json_text = format!(
            "{}\n",
            serde_json::to_string_pretty(&tok_meta)
                .map_err(|e| OracleError::json(&tok_bin_path, e))?
        );
        let tok_json_path = p_dir.join(format!("{}.tokens.json", p.prompt_id));
        fs::write(&tok_json_path, &tok_json_text)
            .map_err(|e| OracleError::io(&tok_json_path, e))?;
        let flat_tok_json_path = tok_flat_dir.join(format!("{}.tokens.json", p.prompt_id));
        fs::write(&flat_tok_json_path, &tok_json_text)
            .map_err(|e| OracleError::io(&flat_tok_json_path, e))?;

        // 2. Logits
        let n_rows = p.logit_positions.len();
        let n_vocab = p.n_vocab as usize;
        if p.logits_f32.len() != n_rows * n_vocab {
            return Err(OracleError::InvalidArg(format!(
                "logits_f32.len() ({}) != n_rows ({n_rows}) * n_vocab ({n_vocab})",
                p.logits_f32.len()
            )));
        }
        let log_bin_bytes = encode_f32_le(&p.logits_f32);
        let log_bin_sha = sha256::sha256_hex(&log_bin_bytes);
        let log_bin_name = format!("{}.logits.bin", p.prompt_id);
        let log_bin_path = p_dir.join(&log_bin_name);
        fs::write(&log_bin_path, &log_bin_bytes).map_err(|e| OracleError::io(&log_bin_path, e))?;
        let flat_log_f32_name = format!("{}.logits.f32", p.prompt_id);
        let flat_log_f32_path = log_flat_dir.join(&flat_log_f32_name);
        fs::write(&flat_log_f32_path, &log_bin_bytes)
            .map_err(|e| OracleError::io(&flat_log_f32_path, e))?;

        let mut rows_summary = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let row_slice = &p.logits_f32[r * n_vocab..(r + 1) * n_vocab];
            rows_summary.push(summarize_logit_row(
                r as u32,
                p.logit_positions[r],
                p.logit_input_token_ids[r],
                row_slice,
            ));
        }
        let row_stride_bytes = u64::from(p.n_vocab) * 4;
        let total_bytes = (n_rows as u64) * row_stride_bytes;
        let log_meta = LogitsMeta {
            schema_version: SCHEMA_VERSION.to_string(),
            prompt_id: p.prompt_id.clone(),
            bin_file: log_bin_name,
            bin_sha256: log_bin_sha,
            dtype: "f32".to_string(),
            endianness: "little".to_string(),
            n_rows: n_rows as u32,
            n_vocab: p.n_vocab,
            row_stride_bytes,
            total_bytes,
            positions: p.logit_positions.clone(),
            input_token_ids: p.logit_input_token_ids.clone(),
            rows_summary,
        };
        let log_json_path = p_dir.join(format!("{}.logits.json", p.prompt_id));
        let log_json_text = format!(
            "{}\n",
            serde_json::to_string_pretty(&log_meta)
                .map_err(|e| OracleError::json(&log_json_path, e))?
        );
        fs::write(&log_json_path, &log_json_text)
            .map_err(|e| OracleError::io(&log_json_path, e))?;

        let mut flat_log_meta = log_meta.clone();
        flat_log_meta.bin_file = flat_log_f32_name;
        let flat_log_json_path = log_flat_dir.join(format!("{}.logits.json", p.prompt_id));
        let flat_log_json_text = format!(
            "{}\n",
            serde_json::to_string_pretty(&flat_log_meta)
                .map_err(|e| OracleError::json(&flat_log_json_path, e))?
        );
        fs::write(&flat_log_json_path, &flat_log_json_text)
            .map_err(|e| OracleError::io(&flat_log_json_path, e))?;

        // 3. Residuals (sorted in topological (token_pos, layer_idx, tap) order)
        let mut sorted_snaps = p.residual_snapshots.clone();
        sorted_snaps.sort_by_key(|(pos, layer, tap, _)| (*pos, *layer, tap.order()));

        let vec_bytes = u64::from(p.n_embd) * 4;
        let mut res_bin_bytes = Vec::with_capacity(sorted_snaps.len() * (vec_bytes as usize));
        let mut snap_metas = Vec::with_capacity(sorted_snaps.len());
        for (idx, (token_pos, layer_idx, tap, vec)) in sorted_snaps.iter().enumerate() {
            if vec.len() != p.n_embd as usize {
                return Err(OracleError::InvalidArg(format!(
                    "residual snapshot {idx} len {} != n_embd {}",
                    vec.len(),
                    p.n_embd
                )));
            }
            let byte_offset = res_bin_bytes.len() as u64;
            res_bin_bytes.extend_from_slice(&encode_f32_le(vec));
            let (l2_norm, rms) = slice_l2_and_rms(vec);
            snap_metas.push(ResidualSnapshotMeta {
                snapshot_idx: idx as u32,
                token_pos: *token_pos,
                layer_idx: *layer_idx,
                tap: tap.as_str().to_string(),
                byte_offset,
                byte_len: vec_bytes,
                l2_norm,
                rms,
            });
        }
        let res_bin_sha = sha256::sha256_hex(&res_bin_bytes);
        let res_bin_name = format!("{}.residuals.bin", p.prompt_id);
        let res_bin_path = p_dir.join(&res_bin_name);
        fs::write(&res_bin_path, &res_bin_bytes).map_err(|e| OracleError::io(&res_bin_path, e))?;
        let flat_res_f32_name = format!("{}.resid.f32", p.prompt_id);
        let flat_res_f32_path = res_flat_dir.join(&flat_res_f32_name);
        fs::write(&flat_res_f32_path, &res_bin_bytes)
            .map_err(|e| OracleError::io(&flat_res_f32_path, e))?;

        let res_meta = ResidualsMeta {
            schema_version: SCHEMA_VERSION.to_string(),
            prompt_id: p.prompt_id.clone(),
            bin_file: res_bin_name,
            bin_sha256: res_bin_sha,
            dtype: "f32".to_string(),
            endianness: "little".to_string(),
            n_embd: p.n_embd,
            n_layer: p.n_layer,
            snapshots: snap_metas,
        };
        let res_json_path = p_dir.join(format!("{}.residuals.json", p.prompt_id));
        let res_json_text = format!(
            "{}\n",
            serde_json::to_string_pretty(&res_meta)
                .map_err(|e| OracleError::json(&res_json_path, e))?
        );
        fs::write(&res_json_path, &res_json_text)
            .map_err(|e| OracleError::io(&res_json_path, e))?;

        let mut flat_res_meta = res_meta;
        flat_res_meta.bin_file = flat_res_f32_name;
        let flat_res_json_path = res_flat_dir.join(format!("{}.resid.json", p.prompt_id));
        let flat_res_json_text = format!(
            "{}\n",
            serde_json::to_string_pretty(&flat_res_meta)
                .map_err(|e| OracleError::json(&flat_res_json_path, e))?
        );
        fs::write(&flat_res_json_path, &flat_res_json_text)
            .map_err(|e| OracleError::io(&flat_res_json_path, e))?;

        manifest.prompts.push(PromptEntry {
            prompt_id: p.prompt_id.clone(),
            category: p.category.clone(),
            prompt_tokens_len: p.prompt_token_ids.len() as u32,
            generated_tokens_len: p.generated_token_ids.len() as u32,
            seed: 0,
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            tokens_json: format!("prompts/{0}/{0}.tokens.json", p.prompt_id),
            logits_json: format!("prompts/{0}/{0}.logits.json", p.prompt_id),
            residuals_json: format!("prompts/{0}/{0}.residuals.json", p.prompt_id),
        });
    }

    let manifest_path = dump_dir.join("manifest.json");
    let manifest_text = format!(
        "{}\n",
        serde_json::to_string_pretty(&manifest)
            .map_err(|e| OracleError::json(&manifest_path, e))?
    );
    fs::write(&manifest_path, manifest_text).map_err(|e| OracleError::io(&manifest_path, e))?;

    write_sha256sums(dump_dir)?;
    Ok(())
}

fn parse_u32_tokens(text: &str) -> Vec<u32> {
    text.split(|c: char| c.is_ascii_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<u32>().ok())
        .collect()
}

fn parse_gen_line(text: &str, prefix: &str) -> Result<Vec<u32>, OracleError> {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(prefix) {
            return Ok(parse_u32_tokens(rest));
        }
    }
    Err(OracleError::InvalidArg(format!(
        "missing {prefix:?} line in output"
    )))
}

#[allow(clippy::too_many_arguments)]
fn assemble_qwen27b_build(
    raw_root: &Path,
    raw_subdir: &str,
    oracle_root: &Path,
    build_subdir: &str,
    git_commit: &str,
    git_branch: &str,
    binary_desc: &str,
    gguf_files: Vec<String>,
    gguf_sha256: Vec<String>,
    recorded_at_utc: &str,
    prompt_defs: &[(&str, &str)],
) -> Result<(), OracleError> {
    let qwen_out = oracle_root.join("qwen27b").join(build_subdir);
    if qwen_out.exists() {
        fs::remove_dir_all(&qwen_out).map_err(|e| OracleError::io(&qwen_out, e))?;
    }
    let mut qwen_env = BTreeMap::new();
    qwen_env.insert("CUDA_MODULE_LOADING".to_string(), "EAGER".to_string());
    qwen_env.insert("STRATA_Q8_PACKED".to_string(), "1".to_string());
    qwen_env.insert("STRATA_MMQ_KQ".to_string(), "1".to_string());
    if build_subdir != "ref-27b-v1" {
        qwen_env.insert("TALUS_PF_FA".to_string(), "1".to_string());
    }
    let qwen_manifest = Manifest {
        schema_version: SCHEMA_VERSION.to_string(),
        format: DUMP_FORMAT.to_string(),
        model_id: "qwen27b".to_string(),
        gguf_files,
        gguf_sha256,
        quant_scheme: "UD-IQ4_XS".to_string(),
        cpp_build: CppBuildInfo {
            git_commit: git_commit.to_string(),
            git_branch: git_branch.to_string(),
            binary: binary_desc.to_string(),
            rocm_version: "10.2.0a20261014".to_string(),
            gpu_arch: "gfx1151".to_string(),
            hip_flags: "-O3 --offload-arch=gfx1151".to_string(),
            env: qwen_env,
        },
        recorded_at_utc: recorded_at_utc.to_string(),
        recorded_host: "gfx1151-host".to_string(),
        hparams: ModelHparams {
            n_vocab: 248_320,
            n_ctx_train: 131_072,
            n_embd: 5120,
            n_layer: 64,
            n_head: 64,
            n_head_kv: 8,
            n_ff: 25_600,
            n_rot: 128,
            rope_freq_base: 1_000_000.0,
            rms_norm_eps: 1e-6,
            n_expert: 0,
            n_expert_used: 0,
        },
        prompts: Vec::new(),
    };

    let qwen_vocab = 248_320usize;
    let qwen_embd = 5120usize;
    let qwen_layers = 64usize;
    let qwen_kverify = 16usize;
    let mut qwen_prompts = Vec::new();
    for &(pid, cat) in prompt_defs {
        let ids_path = raw_root
            .join("prompts")
            .join("qwen27b")
            .join(format!("{pid}.ids"));
        let ids_text = fs::read_to_string(&ids_path).map_err(|e| OracleError::io(&ids_path, e))?;
        let prompt_token_ids = parse_u32_tokens(&ids_text);

        let greedy_path = raw_root.join(raw_subdir).join(format!("{pid}.greedy.out"));
        let greedy_text =
            fs::read_to_string(&greedy_path).map_err(|e| OracleError::io(&greedy_path, e))?;
        let generated_token_ids = parse_gen_line(&greedy_text, "gen:")?;

        let logits_path = raw_root.join(raw_subdir).join(format!("{pid}.logits.f32"));
        let logits_bytes = fs::read(&logits_path).map_err(|e| OracleError::io(&logits_path, e))?;
        let all_logits = decode_f32_le(&logits_path, &logits_bytes)?;
        let n_keep_rows = if pid == "bench27-chat" {
            8usize
        } else {
            4usize
        };
        let logits_f32 = all_logits[..n_keep_rows * qwen_vocab].to_vec();
        let p_len = prompt_token_ids.len() as u32;
        let row0_argmax = summarize_logit_row(0, 0, 0, &logits_f32[..qwen_vocab]).argmax_token_id;
        let (logit_positions, logit_input_token_ids) = if !generated_token_ids.is_empty()
            && row0_argmax == generated_token_ids[0]
        {
            let positions: Vec<u32> = (0..n_keep_rows as u32).map(|r| (p_len - 1) + r).collect();
            let mut inputs = vec![*prompt_token_ids.last().unwrap_or(&0)];
            inputs.extend_from_slice(&generated_token_ids[..(n_keep_rows - 1)]);
            (positions, inputs)
        } else {
            let positions: Vec<u32> = (0..n_keep_rows as u32).map(|r| p_len + r).collect();
            let inputs: Vec<u32> = generated_token_ids[..n_keep_rows].to_vec();
            (positions, inputs)
        };

        let resid_path = raw_root
            .join(raw_subdir)
            .join(format!("{pid}.resid_raw.bin"));
        let resid_bytes = fs::read(&resid_path).map_err(|e| OracleError::io(&resid_path, e))?;
        let all_resid = decode_f32_le(&resid_path, &resid_bytes)?;
        let row_indices: &[usize] = if pid == "bench27-chat" { &[0, 7] } else { &[0] };
        let mut residual_snapshots = Vec::new();
        for &t in row_indices {
            let token_pos = p_len + t as u32;
            for l in 0..qwen_layers {
                let start = (l * qwen_kverify + t) * qwen_embd;
                let vec = all_resid[start..start + qwen_embd].to_vec();
                residual_snapshots.push((token_pos, l as i32, ResidualTap::FfnOut, vec));
            }
        }

        qwen_prompts.push(PromptDumpInput {
            prompt_id: pid.to_string(),
            category: cat.to_string(),
            eos_token_id: 248_046,
            stop_reason: "length".to_string(),
            prompt_token_ids,
            generated_token_ids,
            logit_positions,
            logit_input_token_ids,
            n_vocab: qwen_vocab as u32,
            logits_f32,
            n_embd: qwen_embd as u32,
            n_layer: qwen_layers as u32,
            residual_snapshots,
        });
    }
    write_golden_dump_dir(&qwen_out, qwen_manifest, &qwen_prompts)
}

/// Assemble the Qwen 27B (`qwen27b/ref-27b-v1`, `qwen27b/ref-27b-mtp`,
/// `qwen27b/ref-27b-dflash`) and Laguna (`laguna/ref-laguna`)
/// Contract v0 golden dump directories from raw Strix Halo recordings.
pub fn assemble_strix_recordings(raw_root: &Path, oracle_root: &Path) -> Result<(), OracleError> {
    let prompt_defs = [
        ("bench27-chat", "bench27-chat"),
        ("bench27-code", "bench27-code"),
        ("bench27-edit", "bench27-edit"),
        ("needle", "needle"),
        ("depth-2k", "depth-2k"),
    ];

    let mut assembled_any = false;

    // 1a. Qwen 27B (ref-27b-v1)
    if raw_root.join("qwen27b-r1").is_dir() {
        assemble_qwen27b_build(
            raw_root,
            "qwen27b-r1",
            oracle_root,
            "ref-27b-v1",
            "688a9f48",
            "ref-27b-v1",
            "talus-qwen35 (sha256:83f687ad9826707367ac8abe64336a6953a01d3cd3d3cc281fa4583e4d230c66)",
            vec!["<models-dir>/gguf/Qwen3.8-27B/Qwen3.8-27B-UD-IQ4_XS.gguf".to_string()],
            vec!["177ae5e70ef0d340fdf2d1d75539cf992d69eecdd8d244e36e33456cdcb07cfa".to_string()],
            "2026-10-16T17:31:40Z",
            &prompt_defs,
        )?;
        assembled_any = true;
    }

    // 1b. Qwen 27B (ref-27b-mtp default MTP profile)
    if raw_root.join("qwen27b-42-r1").is_dir() {
        assemble_qwen27b_build(
            raw_root,
            "qwen27b-42-r1",
            oracle_root,
            "ref-27b-mtp",
            "7cd579b4",
            "talus-27b",
            "talus-qwen35 (sha256:4f6c2146edee6e8a334a5c42aea1a415cff51c83c657d16de9384d48d3d9ed32)",
            vec![
                "<models-dir>/gguf/Qwen3.8-27B/Qwen3.8-27B-UD-IQ4_XS.gguf".to_string(),
                "<models-dir>/gguf/Qwen3.8-27B/MTP/mtp-Qwen3.8-27B-Q4_0.gguf".to_string(),
            ],
            vec![
                "177ae5e70ef0d340fdf2d1d75539cf992d69eecdd8d244e36e33456cdcb07cfa".to_string(),
                "50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e".to_string(),
            ],
            "2026-10-17T05:30:00Z",
            &prompt_defs,
        )?;
        assembled_any = true;
    }

    // 1c. Qwen 27B (ref-27b-dflash DFlash2 profile)
    if raw_root.join("qwen27b-42-dflash-r1").is_dir() {
        assemble_qwen27b_build(
            raw_root,
            "qwen27b-42-dflash-r1",
            oracle_root,
            "ref-27b-dflash",
            "7cd579b4",
            "talus-27b",
            "talus-qwen35 (sha256:4f6c2146edee6e8a334a5c42aea1a415cff51c83c657d16de9384d48d3d9ed32)",
            vec![
                "<models-dir>/gguf/Qwen3.8-27B/Qwen3.8-27B-UD-IQ4_XS.gguf".to_string(),
                "<models-dir>/gguf/Qwen3.8-27B/MTP/mtp-Qwen3.8-27B-Q4_0.gguf".to_string(),
                "<models-dir>/gguf/Qwen3.8-27B/DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf".to_string(),
            ],
            vec![
                "177ae5e70ef0d340fdf2d1d75539cf992d69eecdd8d244e36e33456cdcb07cfa".to_string(),
                "50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e".to_string(),
                "1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd".to_string(),
            ],
            "2026-10-17T05:30:00Z",
            &prompt_defs,
        )?;
        assembled_any = true;
    }

    // 2. Laguna (`laguna/ref-laguna`)
    if !raw_root.join("laguna-r1").is_dir() {
        if !assembled_any {
            return Err(OracleError::schema(
                raw_root,
                "no recognized recording directories (qwen27b-r1, qwen27b-42-r1, qwen27b-42-dflash-r1, laguna-r1) found",
            ));
        }
        return Ok(());
    }
    let lag_out = oracle_root.join("laguna").join("ref-laguna");
    if lag_out.exists() {
        fs::remove_dir_all(&lag_out).map_err(|e| OracleError::io(&lag_out, e))?;
    }
    let mut lag_env = BTreeMap::new();
    lag_env.insert("ROCM_PATH".to_string(), "/opt/rocm".to_string());
    lag_env.insert("STRATA_MMQ_KQ".to_string(), "1".to_string());
    let lag_manifest = Manifest {
        schema_version: SCHEMA_VERSION.to_string(),
        format: DUMP_FORMAT.to_string(),
        model_id: "laguna".to_string(),
        gguf_files: vec![
            "<models-dir>/gguf/Laguna-S-2.1-Q4-lean/Laguna-S-2.1-Q4-lean-00001-of-00003.gguf"
                .to_string(),
        ],
        gguf_sha256: vec![
            "f5db6a96eee7d133246419864325514311ee0b532ef87f0921f861b73d2fb0cc".to_string(),
        ],
        quant_scheme: "Q4_K_M-lean".to_string(),
        cpp_build: CppBuildInfo {
            git_commit: "089c326e".to_string(),
            git_branch: "rust/1.1".to_string(),
            binary: "talus-laguna (sha256:f0f0b5b9815fb6aa64d1651b04ee63fe83ec7b7d2a9ff416a13005a038cf848e)".to_string(),
            rocm_version: "10.2.0a20261014".to_string(),
            gpu_arch: "gfx1151".to_string(),
            hip_flags: "-O3 --offload-arch=gfx1151".to_string(),
            env: lag_env,
        },
        recorded_at_utc: "2026-10-16T17:35:47Z".to_string(),
        recorded_host: "gfx1151-host".to_string(),
        hparams: ModelHparams {
            n_vocab: 100_352,
            n_ctx_train: 262_144,
            n_embd: 3072,
            n_layer: 48,
            n_head: 64,
            n_head_kv: 8,
            n_ff: 12_288,
            n_rot: 128,
            rope_freq_base: 500_000.0,
            rms_norm_eps: 1e-6,
            n_expert: 256,
            n_expert_used: 8,
        },
        prompts: Vec::new(),
    };

    let lag_vocab = 100_352usize;
    let lag_embd = 3072usize;
    let lag_layers = 48usize;
    let mut lag_prompts = Vec::new();
    for (pid, cat) in prompt_defs {
        let ids_path = raw_root
            .join("prompts")
            .join("laguna")
            .join(format!("{pid}.ids"));
        let ids_text = fs::read_to_string(&ids_path).map_err(|e| OracleError::io(&ids_path, e))?;
        let prompt_token_ids = parse_u32_tokens(&ids_text);

        let greedy_path = raw_root.join("laguna-r1").join(format!("{pid}.greedy.out"));
        let greedy_text =
            fs::read_to_string(&greedy_path).map_err(|e| OracleError::io(&greedy_path, e))?;
        let generated_token_ids = parse_gen_line(&greedy_text, "generated:")?;

        let logits_path = raw_root.join("laguna-r1").join(format!("{pid}.logits.f32"));
        let logits_bytes = fs::read(&logits_path).map_err(|e| OracleError::io(&logits_path, e))?;
        let logits_f32 = decode_f32_le(&logits_path, &logits_bytes)?;
        let n_force = 4usize;
        let p_len = prompt_token_ids.len();
        let logit_positions: Vec<u32> = ((p_len - n_force)..p_len).map(|i| i as u32).collect();
        let logit_input_token_ids: Vec<u32> = prompt_token_ids[(p_len - n_force)..p_len].to_vec();

        let resid_path = raw_root
            .join("laguna-r1")
            .join(format!("{pid}.resid_raw.bin"));
        let resid_bytes = fs::read(&resid_path).map_err(|e| OracleError::io(&resid_path, e))?;
        let all_resid = decode_f32_le(&resid_path, &resid_bytes)?;
        let recorded_nl = if pid == "bench27-chat" {
            48usize
        } else {
            4usize
        };
        let recorded_nt = 8usize;
        let tok_indices: &[usize] = if pid == "bench27-chat" {
            &[0, 1, 7]
        } else {
            &[0, 1, 2, 3, 4, 5, 6, 7]
        };
        let mut residual_snapshots = Vec::new();
        for &t in tok_indices {
            for il in 0..recorded_nl {
                let start = (il * recorded_nt + t) * lag_embd;
                let vec = all_resid[start..start + lag_embd].to_vec();
                residual_snapshots.push((t as u32, il as i32, ResidualTap::FfnOut, vec));
            }
        }

        lag_prompts.push(PromptDumpInput {
            prompt_id: pid.to_string(),
            category: cat.to_string(),
            eos_token_id: 2,
            stop_reason: "length".to_string(),
            prompt_token_ids,
            generated_token_ids,
            logit_positions,
            logit_input_token_ids,
            n_vocab: lag_vocab as u32,
            logits_f32,
            n_embd: lag_embd as u32,
            n_layer: lag_layers as u32,
            residual_snapshots,
        });
    }
    write_golden_dump_dir(&lag_out, lag_manifest, &lag_prompts)?;
    Ok(())
}

// ============================================================================
// Unit & Integration Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn scratch_test_dir(label: &str) -> PathBuf {
        let base = std::env::var_os("TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../target/test-scratch"
                ))
            });
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = base.join(format!("oracle-test-{label}-{}-{id}", std::process::id()));
        if dir.exists() {
            let _ = fs::remove_dir_all(&dir);
        }
        fs::create_dir_all(&dir).expect("create test scratch dir");
        dir
    }

    fn sample_manifest() -> Manifest {
        let mut env = BTreeMap::new();
        env.insert("HSA_ENABLE_SDMA".to_string(), "0".to_string());
        Manifest {
            schema_version: SCHEMA_VERSION.to_string(),
            format: DUMP_FORMAT.to_string(),
            model_id: "qwen27b".to_string(),
            gguf_files: vec!["Qwen3.8-27B-UD-IQ4_XS.gguf".to_string()],
            gguf_sha256: vec!["0".repeat(64)],
            quant_scheme: "IQ4_XS".to_string(),
            cpp_build: CppBuildInfo {
                git_commit: "089c326e".to_string(),
                git_branch: "main".to_string(),
                binary: "talus-qwen35".to_string(),
                rocm_version: "7.14.1".to_string(),
                gpu_arch: "gfx1151".to_string(),
                hip_flags: "-O3 --offload-arch=gfx1151".to_string(),
                env,
            },
            recorded_at_utc: "2026-10-16T00:00:00Z".to_string(),
            recorded_host: "gfx1151-host".to_string(),
            hparams: ModelHparams {
                n_vocab: 16,
                n_ctx_train: 4096,
                n_embd: 8,
                n_layer: 2,
                n_head: 4,
                n_head_kv: 2,
                n_ff: 32,
                n_rot: 4,
                rope_freq_base: 10000.0,
                rms_norm_eps: 1e-6,
                n_expert: 0,
                n_expert_used: 0,
            },
            prompts: Vec::new(),
        }
    }

    fn sample_prompt_input() -> PromptDumpInput {
        let n_vocab = 16u32;
        let n_embd = 8u32;
        let n_layer = 2u32;
        let mut logits_f32 = Vec::new();
        for r in 0..3 {
            for v in 0..n_vocab as usize {
                let val = (r as f32) * 0.5 + (v as f32) * 0.125 - 0.75;
                logits_f32.push(val);
            }
        }
        let make_vec = |seed: f32| -> Vec<f32> {
            (0..n_embd as usize)
                .map(|i| seed + (i as f32) * 0.0625)
                .collect()
        };
        let residual_snapshots = vec![
            (3, -1, ResidualTap::EmbdOut, make_vec(0.125)),
            (3, 0, ResidualTap::AttnOut, make_vec(0.25)),
            (3, 0, ResidualTap::FfnOut, make_vec(0.375)),
            (3, 1, ResidualTap::AttnOut, make_vec(0.5)),
            (3, 1, ResidualTap::FfnOut, make_vec(0.625)),
            (3, 2, ResidualTap::FinalNormOut, make_vec(0.75)),
        ];

        PromptDumpInput {
            prompt_id: "bench27-chat".to_string(),
            category: "chat".to_string(),
            eos_token_id: 248_046,
            stop_reason: "length".to_string(),
            prompt_token_ids: vec![248_045, 846, 198, 7734],
            generated_token_ids: vec![264, 2716, 3255, 318],
            logit_positions: vec![3, 4, 5],
            logit_input_token_ids: vec![7734, 264, 2716],
            n_vocab,
            logits_f32,
            n_embd,
            n_layer,
            residual_snapshots,
        }
    }

    #[test]
    fn crate_identity() {
        assert_eq!(CRATE_NAME, "oracle");
    }

    #[test]
    fn sha256_nist_vectors() {
        assert_eq!(
            sha256::sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256::sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256::sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn ulp_and_hex_float_contract() {
        let a = 0.125_f32;
        let b = f32::from_bits(a.to_bits() ^ 1);
        assert_eq!(ulp_diff_f32(a, a), 0);
        assert_eq!(ulp_diff_f32(a, b), 1);
        assert_eq!(format_f32_hex(a), "0x1.000000p-3");
        assert_eq!(format_f32_hex(b), "0x1.000002p-3");
    }

    #[test]
    fn bitwise_identical_dumps_pass() {
        let root = scratch_test_dir("identical");
        let ref_dir = root.join("ref");
        let cand_dir = root.join("cand");
        write_golden_dump_dir(&ref_dir, sample_manifest(), &[sample_prompt_input()])
            .expect("write ref");
        write_golden_dump_dir(&cand_dir, sample_manifest(), &[sample_prompt_input()])
            .expect("write cand");

        let summary = verify_dump_dir(&ref_dir).expect("verify ref");
        assert_eq!(summary.prompts_verified, 1);

        let report =
            diff_dump_dirs(&ref_dir, &cand_dir, &DiffConfig::default()).expect("diff identical");
        assert!(report.is_pass());
        assert_eq!(report.exit_code(), EXIT_PASS);
        assert!(report.first_diverging_token.is_none());
        assert!(report.first_diverging_logit_row.is_none());
        assert!(report.first_diverging_layer.is_none());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn bitflip_in_logit_row_caught_at_exact_coordinates() {
        let root = scratch_test_dir("logit-bitflip");
        let ref_dir = root.join("ref");
        let cand_dir = root.join("cand");
        write_golden_dump_dir(&ref_dir, sample_manifest(), &[sample_prompt_input()])
            .expect("write ref");

        let opts = BitflipOptions {
            prompt_id: Some("bench27-chat".to_string()),
            target: BitflipTarget::Logits,
            row_idx: 2,
            vocab_idx: 11,
            bit: 0,
            ..BitflipOptions::default()
        };
        inject_bitflip(&ref_dir, &cand_dir, &opts).expect("inject logit bitflip");

        let report =
            diff_dump_dirs(&ref_dir, &cand_dir, &DiffConfig::default()).expect("diff logit flip");
        assert!(!report.is_pass());
        assert_eq!(report.exit_code(), EXIT_DIFF);
        assert!(report.first_diverging_token.is_none());
        assert!(report.first_diverging_layer.is_none());

        let row_diff = report
            .first_diverging_logit_row
            .as_ref()
            .expect("expected logit row diff");
        assert_eq!(row_diff.prompt_id, "bench27-chat");
        assert_eq!(row_diff.row_idx, 2);
        assert_eq!(row_diff.pos, 5);
        assert_eq!(row_diff.vocab_idx, 11);
        assert_eq!(row_diff.ulp_diff, 1);

        // With --ulp 1 tolerance, the 1-ULP difference passes.
        let tolerant_cfg = DiffConfig {
            max_ulp: 1,
            ..Default::default()
        };
        let pass_report =
            diff_dump_dirs(&ref_dir, &cand_dir, &tolerant_cfg).expect("diff tolerant");
        assert!(pass_report.is_pass());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn bitflip_in_residual_layer_caught_at_exact_coordinates() {
        let root = scratch_test_dir("resid-bitflip");
        let ref_dir = root.join("ref");
        let cand_dir = root.join("cand");
        write_golden_dump_dir(&ref_dir, sample_manifest(), &[sample_prompt_input()])
            .expect("write ref");

        let opts = BitflipOptions {
            prompt_id: Some("bench27-chat".to_string()),
            target: BitflipTarget::Residuals,
            token_pos: Some(3),
            layer_idx: Some(1),
            tap: Some("ffn_out".to_string()),
            dim_idx: 5,
            bit: 0,
            ..BitflipOptions::default()
        };
        inject_bitflip(&ref_dir, &cand_dir, &opts).expect("inject residual bitflip");

        let report =
            diff_dump_dirs(&ref_dir, &cand_dir, &DiffConfig::default()).expect("diff resid flip");
        assert!(!report.is_pass());
        assert_eq!(report.exit_code(), EXIT_DIFF);
        let res_diff = report
            .first_diverging_layer
            .as_ref()
            .expect("expected residual diff");
        assert_eq!(res_diff.prompt_id, "bench27-chat");
        assert_eq!(res_diff.token_pos, 3);
        assert_eq!(res_diff.layer_idx, 1);
        assert_eq!(res_diff.tap, "ffn_out");
        assert_eq!(res_diff.dim_idx, 5);
        assert_eq!(res_diff.ulp_diff, 1);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn bitflip_in_tokens_caught_at_exact_step_and_pos() {
        let root = scratch_test_dir("tok-bitflip");
        let ref_dir = root.join("ref");
        let cand_dir = root.join("cand");
        write_golden_dump_dir(&ref_dir, sample_manifest(), &[sample_prompt_input()])
            .expect("write ref");

        let opts = BitflipOptions {
            prompt_id: Some("bench27-chat".to_string()),
            target: BitflipTarget::Tokens,
            step: 2,
            bit: 0,
            ..BitflipOptions::default()
        };
        inject_bitflip(&ref_dir, &cand_dir, &opts).expect("inject token bitflip");

        let report =
            diff_dump_dirs(&ref_dir, &cand_dir, &DiffConfig::default()).expect("diff token flip");
        assert!(!report.is_pass());
        assert_eq!(report.exit_code(), EXIT_DIFF);
        let tok_diff = report
            .first_diverging_token
            .as_ref()
            .expect("expected token diff");
        assert_eq!(tok_diff.prompt_id, "bench27-chat");
        assert!(!tok_diff.is_prompt_token);
        assert_eq!(tok_diff.step, 2);
        assert_eq!(tok_diff.token_pos, 6);
        assert_eq!(tok_diff.expected_token_id, 3255);
        assert_eq!(tok_diff.actual_token_id, 3255 ^ 1);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn corrupted_binary_detected_by_sha256_verification() {
        let root = scratch_test_dir("corrupt");
        let ref_dir = root.join("ref");
        write_golden_dump_dir(&ref_dir, sample_manifest(), &[sample_prompt_input()])
            .expect("write ref");

        let bin_path = ref_dir.join("prompts/bench27-chat/bench27-chat.logits.bin");
        let mut bytes = fs::read(&bin_path).expect("read bin");
        bytes[0] ^= 0xff;
        fs::write(&bin_path, bytes).expect("corrupt bin");

        let err = verify_dump_dir(&ref_dir).expect_err("expected checksum error");
        assert!(matches!(err, OracleError::ChecksumMismatch { .. }));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn real_recorded_qwen27b_and_laguna_dumps_verify_and_pass_diff() {
        let Some(oracle_root) = std::env::var_os("TALUS_ORACLE_DIR").map(PathBuf::from) else {
            return;
        };
        let qwen_dir = oracle_root.join("qwen27b/ref-27b-v1");
        if qwen_dir.exists() {
            let summary = verify_dump_dir(&qwen_dir).expect("verify real qwen27b dump");
            assert_eq!(summary.model_id, "qwen27b");
            assert_eq!(summary.prompts_verified, 5);
            assert_eq!(summary.checksums_verified, 61);

            let cfg = DiffConfig {
                prompt_filter: Some("bench27-chat".to_string()),
                ..Default::default()
            };
            let report = diff_dump_dirs(&qwen_dir, &qwen_dir, &cfg).expect("self-diff qwen27b");
            assert!(report.is_pass());
        }

        let qwen42_dir = oracle_root.join("qwen27b/ref-27b-mtp");
        if qwen42_dir.exists() {
            let summary = verify_dump_dir(&qwen42_dir).expect("verify real ref-27b-mtp dump");
            assert_eq!(summary.model_id, "qwen27b");
            assert_eq!(summary.prompts_verified, 5);
            assert_eq!(summary.checksums_verified, 61);

            let cfg = DiffConfig {
                prompt_filter: Some("bench27-chat".to_string()),
                ..Default::default()
            };
            let report =
                diff_dump_dirs(&qwen42_dir, &qwen42_dir, &cfg).expect("self-diff ref-27b-mtp");
            assert!(report.is_pass());

            let qwen42_df_dir = oracle_root.join("qwen27b/ref-27b-dflash");
            if qwen42_df_dir.exists() {
                let df_summary =
                    verify_dump_dir(&qwen42_df_dir).expect("verify real ref-27b-dflash dump");
                assert_eq!(df_summary.model_id, "qwen27b");
                assert_eq!(df_summary.prompts_verified, 5);
                assert_eq!(df_summary.checksums_verified, 61);

                let cross_report =
                    diff_dump_dirs(&qwen42_dir, &qwen42_df_dir, &DiffConfig::default())
                        .expect("diff ref-27b-mtp vs ref-27b-dflash");
                assert!(cross_report.is_pass());
            }

            let root = scratch_test_dir("qwen42-bitflip");
            let cand_dir = root.join("cand");
            let opts = BitflipOptions {
                prompt_id: Some("bench27-chat".to_string()),
                target: BitflipTarget::Residuals,
                token_pos: Some(37),
                layer_idx: Some(12),
                tap: Some("ffn_out".to_string()),
                dim_idx: 42,
                bit: 0,
                ..BitflipOptions::default()
            };
            inject_bitflip(&qwen42_dir, &cand_dir, &opts).expect("inject bitflip on ref-27b-mtp");
            let flip_report =
                diff_dump_dirs(&qwen42_dir, &cand_dir, &cfg).expect("diff bitflip on ref-27b-mtp");
            assert!(!flip_report.is_pass());
            assert_eq!(flip_report.exit_code(), EXIT_DIFF);
            let res_diff = flip_report
                .first_diverging_layer
                .as_ref()
                .expect("expected residual diff on ref-27b-mtp");
            assert_eq!(res_diff.prompt_id, "bench27-chat");
            assert_eq!(res_diff.token_pos, 37);
            assert_eq!(res_diff.layer_idx, 12);
            assert_eq!(res_diff.tap, "ffn_out");
            assert_eq!(res_diff.dim_idx, 42);
            assert_eq!(res_diff.ulp_diff, 1);
            let _ = fs::remove_dir_all(&root);
        }

        let lag_dir = oracle_root.join("laguna/ref-laguna");
        if lag_dir.exists() {
            let summary = verify_dump_dir(&lag_dir).expect("verify real laguna dump");
            assert_eq!(summary.model_id, "laguna");
            assert_eq!(summary.prompts_verified, 5);
            assert_eq!(summary.checksums_verified, 61);

            let cfg = DiffConfig {
                prompt_filter: Some("bench27-chat".to_string()),
                ..Default::default()
            };
            let report = diff_dump_dirs(&lag_dir, &lag_dir, &cfg).expect("self-diff laguna");
            assert!(report.is_pass());
        }
    }
}
