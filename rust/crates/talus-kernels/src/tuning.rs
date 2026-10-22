//! hipBLASLt prefill GEMM tuning table loader and selector (`TuningTables`).
//!
//! Mirrors `src/prefill/hipblaslt_tuning.hpp` and the commit `5c7e9867` runtime
//! version-sibling fallback rule from `src/prefill/gemm.cu:207-225`:
//! - Parses `STRATA_HIPBLASLT_TUNING_V1 <arch> <hipblaslt_version>` headers and
//!   `<type: f16|bf16> <n> <k> <ldy> <t_bucket> <solution_id>` rows.
//! - Validates architecture family compatibility (exact match, RDNA4 `gfx1200`/`gfx1201`,
//!   or RDNA3 `gfx1100`/`gfx1101`/`gfx1102`) and exact `hipblaslt_version`.
//! - When a configured `<path>` fails to load for `(expected_arch, runtime_version)`
//!   (for example due to a ROCm upgrade changing `hipblaslt_version`), automatically
//!   falls back to `<dir>/<expected_arch>-hipblaslt-<runtime_version>.txt` in the same
//!   directory.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use talus_core_api::{GpuArch, QuantType, Result};

use crate::{invalid_arg, io_err};

/// Format header token required at the start of every tuning table file.
pub const TUNING_HEADER_MAGIC: &str = "STRATA_HIPBLASLT_TUNING_V1";

/// Matrix element type in a hipBLASLt tuning table row (`f16` or `bf16`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TuningDtype {
    /// IEEE 754 binary16 (`f16`).
    F16,
    /// Brain floating-point 16 (`bf16`).
    Bf16,
}

impl TuningDtype {
    /// Parse `"f16"` or `"bf16"`.
    pub fn parse(token: &str) -> Option<Self> {
        match token {
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Convert from [`QuantType`] (`F16` or `BF16`).
    pub const fn from_quant_type(qtype: QuantType) -> Option<Self> {
        match qtype {
            QuantType::F16 => Some(Self::F16),
            QuantType::BF16 => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Convert to [`QuantType`].
    #[inline]
    pub const fn to_quant_type(self) -> QuantType {
        match self {
            Self::F16 => QuantType::F16,
            Self::Bf16 => QuantType::BF16,
        }
    }

    /// Canonical token string (`"f16"` or `"bf16"`).
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

/// A single tuned `(dtype, n, k, ldy, t_bucket) -> solution_id` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuningRow {
    /// Precision (`f16` or `bf16`).
    pub dtype: TuningDtype,
    /// Output rows `N` (`n_out`).
    pub n: i32,
    /// Inner dimension `K` (`n_in`).
    pub k: i32,
    /// Output leading dimension `ldy` (`>= n`).
    pub ldy: i32,
    /// Tuned token-count bucket `T`.
    pub t_bucket: i32,
    /// hipBLASLt solution index (`>= 0`).
    pub solution_id: i32,
}

/// Strip ROCm feature suffixes past `':'` (e.g. `"gfx1151:sramecc-:xnack-"` -> `"gfx1151"`).
#[inline]
pub fn strip_arch_features(arch: &str) -> &str {
    match arch.find(':') {
        Some(idx) => &arch[..idx],
        None => arch,
    }
}

#[inline]
fn is_rdna4_arch(arch: &str) -> bool {
    matches!(arch, "gfx1200" | "gfx1201")
}

#[inline]
fn is_rdna3_arch(arch: &str) -> bool {
    matches!(arch, "gfx1100" | "gfx1101" | "gfx1102")
}

/// Check whether `runtime_arch` is compatible with `table_arch` per
/// `src/prefill/hipblaslt_tuning.hpp`.
///
/// Exact match is always accepted; same-ISA RDNA4 (`gfx1200`/`gfx1201`) and
/// RDNA3 (`gfx1100`/`gfx1101`/`gfx1102`) families are also compatible.
/// RDNA3.5 (`gfx1151`) requires an exact match.
pub fn arch_matches_table(runtime_arch: &str, table_arch: &str) -> bool {
    let a = strip_arch_features(runtime_arch);
    let b = strip_arch_features(table_arch);
    if a == b {
        return true;
    }
    if is_rdna4_arch(a) && is_rdna4_arch(b) {
        return true;
    }
    if is_rdna3_arch(a) && is_rdna3_arch(b) {
        return true;
    }
    false
}

/// Parsed hipBLASLt tuning table (`<arch>-hipblaslt-<ver>.txt`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningTables {
    arch: String,
    hipblaslt_version: i32,
    rows: Vec<TuningRow>,
    source_path: Option<PathBuf>,
}

/// Alias for [`TuningTables`].
pub type TuningTable = TuningTables;

impl TuningTables {
    /// Architecture string recorded in the table header (stripped of `:feature` suffixes).
    #[inline]
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// hipBLASLt version integer recorded in the table header (e.g. `101100` or `102000`).
    #[inline]
    pub fn hipblaslt_version(&self) -> i32 {
        self.hipblaslt_version
    }

    /// All parsed tuning rows in file order.
    #[inline]
    pub fn rows(&self) -> &[TuningRow] {
        &self.rows
    }

    /// Number of tuning rows in the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the tuning table contains zero rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// File path from which this table was loaded, if loaded from disk.
    #[inline]
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    /// Parse a tuning table from an in-memory string, optionally validating
    /// `expected_arch` and `expected_version`.
    pub fn parse_str(
        content: &str,
        expected_arch: Option<&str>,
        expected_version: Option<i32>,
    ) -> Result<Self> {
        let mut header_seen = false;
        let mut arch = String::new();
        let mut hipblaslt_version = 0i32;
        let mut rows = Vec::new();
        let mut seen_keys = HashSet::new();

        for (line_idx, raw_line) in content.lines().enumerate() {
            let line_no = line_idx + 1;
            let without_comment = match raw_line.find('#') {
                Some(pos) => &raw_line[..pos],
                None => raw_line,
            };
            let trimmed = without_comment.trim();
            if trimmed.is_empty() {
                continue;
            }

            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            if !header_seen {
                if tokens.len() != 3 || tokens[0] != TUNING_HEADER_MAGIC {
                    return Err(invalid_arg(format!(
                        "line {}: expected '{} <arch> <version>' header",
                        line_no, TUNING_HEADER_MAGIC
                    )));
                }
                let parsed_arch = strip_arch_features(tokens[1]).to_string();
                let parsed_ver: i32 = tokens[2].parse().map_err(|_| {
                    invalid_arg(format!(
                        "line {}: invalid hipblaslt version '{}'",
                        line_no, tokens[2]
                    ))
                })?;
                if parsed_arch.is_empty() || parsed_ver <= 0 {
                    return Err(invalid_arg(format!(
                        "line {}: invalid tuning header arch/version",
                        line_no
                    )));
                }
                if let Some(exp_arch) = expected_arch {
                    if !exp_arch.is_empty() && !arch_matches_table(exp_arch, &parsed_arch) {
                        return Err(invalid_arg(format!(
                            "arch mismatch (table='{}', runtime='{}')",
                            parsed_arch,
                            strip_arch_features(exp_arch)
                        )));
                    }
                }
                if let Some(exp_ver) = expected_version {
                    if exp_ver > 0 && parsed_ver != exp_ver {
                        return Err(invalid_arg(format!(
                            "hipblaslt version mismatch (table={}, runtime={})",
                            parsed_ver, exp_ver
                        )));
                    }
                }
                arch = parsed_arch;
                hipblaslt_version = parsed_ver;
                header_seen = true;
                continue;
            }

            if tokens.len() != 6 {
                return Err(invalid_arg(format!(
                    "line {}: malformed tuning row (expected 6 columns, got {})",
                    line_no,
                    tokens.len()
                )));
            }

            let dtype = TuningDtype::parse(tokens[0]).ok_or_else(|| {
                invalid_arg(format!(
                    "line {}: unknown type '{}' (expected 'f16' or 'bf16')",
                    line_no, tokens[0]
                ))
            })?;
            let n: i32 = tokens[1]
                .parse()
                .map_err(|_| invalid_arg(format!("line {}: invalid n '{}'", line_no, tokens[1])))?;
            let k: i32 = tokens[2]
                .parse()
                .map_err(|_| invalid_arg(format!("line {}: invalid k '{}'", line_no, tokens[2])))?;
            let ldy: i32 = tokens[3].parse().map_err(|_| {
                invalid_arg(format!("line {}: invalid ldy '{}'", line_no, tokens[3]))
            })?;
            let t_bucket: i32 = tokens[4].parse().map_err(|_| {
                invalid_arg(format!(
                    "line {}: invalid t_bucket '{}'",
                    line_no, tokens[4]
                ))
            })?;
            let solution_id: i32 = tokens[5].parse().map_err(|_| {
                invalid_arg(format!(
                    "line {}: invalid solution_id '{}'",
                    line_no, tokens[5]
                ))
            })?;

            if n <= 0 || k <= 0 || ldy < n || t_bucket <= 0 || solution_id < 0 {
                return Err(invalid_arg(format!(
                    "line {}: invalid tuning row values (n={}, k={}, ldy={}, t_bucket={}, solution_id={})",
                    line_no, n, k, ldy, t_bucket, solution_id
                )));
            }

            let key = (dtype, n, k, ldy, t_bucket);
            if !seen_keys.insert(key) {
                return Err(invalid_arg(format!(
                    "line {}: duplicate tuning row for ({}, n={}, k={}, ldy={}, t_bucket={})",
                    line_no,
                    dtype.as_str(),
                    n,
                    k,
                    ldy,
                    t_bucket
                )));
            }

            rows.push(TuningRow {
                dtype,
                n,
                k,
                ldy,
                t_bucket,
                solution_id,
            });
        }

        if !header_seen {
            return Err(invalid_arg("empty tuning file (missing header)"));
        }

        Ok(Self {
            arch,
            hipblaslt_version,
            rows,
            source_path: None,
        })
    }

    /// Load a tuning table from `path`, optionally validating `expected_arch` and `expected_version`.
    pub fn load_file(
        path: impl AsRef<Path>,
        expected_arch: Option<&str>,
        expected_version: Option<i32>,
    ) -> Result<Self> {
        let path_ref = path.as_ref();
        let content = fs::read_to_string(path_ref).map_err(|e| {
            io_err(format!(
                "cannot open tuning file '{}': {}",
                path_ref.display(),
                e
            ))
        })?;
        let mut table = Self::parse_str(&content, expected_arch, expected_version)?;
        table.source_path = Some(path_ref.to_path_buf());
        Ok(table)
    }

    /// Load a tuning table from `path` for `(expected_arch, runtime_version)`,
    /// applying the commit `5c7e9867` version-sibling fallback rule (`src/prefill/gemm.cu:207-225`).
    ///
    /// If `path` fails to load (for example because `path` targets an older or newer
    /// `hipblaslt` version than `runtime_version`), this method probes the sibling file
    /// `<parent_dir>/<expected_arch>-hipblaslt-<runtime_version>.txt`.
    pub fn load_with_fallback(
        path: impl AsRef<Path>,
        expected_arch: &str,
        runtime_version: i32,
    ) -> Result<Self> {
        let path_ref = path.as_ref();
        match Self::load_file(path_ref, Some(expected_arch), Some(runtime_version)) {
            Ok(table) => Ok(table),
            Err(primary_err) => {
                let clean_arch = strip_arch_features(expected_arch);
                if !clean_arch.is_empty() && runtime_version > 0 {
                    let sibling_name = format!("{}-hipblaslt-{}.txt", clean_arch, runtime_version);
                    let sibling_path = match path_ref.parent() {
                        Some(parent) if !parent.as_os_str().is_empty() => {
                            parent.join(&sibling_name)
                        }
                        _ => PathBuf::from(&sibling_name),
                    };
                    if sibling_path != path_ref && sibling_path.is_file() {
                        if let Ok(table) =
                            Self::load_file(&sibling_path, Some(clean_arch), Some(runtime_version))
                        {
                            return Ok(table);
                        }
                    }
                }
                Err(primary_err)
            }
        }
    }

    /// Load the tuning table for `(arch, runtime_version)` from a directory (such as `tools/hip/`).
    pub fn load_from_dir(
        dir: impl AsRef<Path>,
        arch: &GpuArch,
        runtime_version: i32,
    ) -> Result<Self> {
        let dir_ref = dir.as_ref();
        let arch_str = strip_arch_features(arch.as_str());
        let direct = dir_ref.join(format!("{}-hipblaslt-{}.txt", arch_str, runtime_version));
        Self::load_with_fallback(direct, arch_str, runtime_version)
    }

    /// Load all `*-hipblaslt-*.txt` tuning tables present in `dir`, sorted by filename.
    pub fn load_all_in_dir(dir: impl AsRef<Path>) -> Result<Vec<Self>> {
        let dir_ref = dir.as_ref();
        let entries = fs::read_dir(dir_ref).map_err(|e| {
            io_err(format!(
                "failed to read tuning directory '{}': {}",
                dir_ref.display(),
                e
            ))
        })?;
        let mut paths = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| io_err(e.to_string()))?;
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.contains("-hipblaslt-") && name.ends_with(".txt") {
                        paths.push(path);
                    }
                }
            }
        }
        paths.sort();

        let mut tables = Vec::with_capacity(paths.len());
        for path in paths {
            tables.push(Self::load_file(&path, None, None)?);
        }
        Ok(tables)
    }

    /// Find the closest tuned row for `(dtype, n, k, ldy)` at token count `t`.
    ///
    /// Matches `Table::closest` in `src/prefill/hipblaslt_tuning.hpp`:
    /// minimizes `|row.t_bucket - t|` and breaks ties toward the smaller `t_bucket`.
    pub fn closest(
        &self,
        dtype: TuningDtype,
        n: i32,
        k: i32,
        ldy: i32,
        t: i32,
    ) -> Option<&TuningRow> {
        let mut best: Option<&TuningRow> = None;
        let mut best_dist = i64::MAX;
        for row in &self.rows {
            if row.dtype != dtype || row.n != n || row.k != k || row.ldy != ldy {
                continue;
            }
            let dist = (i64::from(row.t_bucket) - i64::from(t)).abs();
            match best {
                None => {
                    best = Some(row);
                    best_dist = dist;
                }
                Some(prev) => {
                    if dist < best_dist || (dist == best_dist && row.t_bucket < prev.t_bucket) {
                        best = Some(row);
                        best_dist = dist;
                    }
                }
            }
        }
        best
    }

    /// Return the tuned `solution_id` for `(dtype, n, k, ldy)` at token count `t`, if present.
    #[inline]
    pub fn select_solution(
        &self,
        dtype: TuningDtype,
        n: i32,
        k: i32,
        ldy: i32,
        t: i32,
    ) -> Option<i32> {
        self.closest(dtype, n, k, ldy, t).map(|r| r.solution_id)
    }
}
