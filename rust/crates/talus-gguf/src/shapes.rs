//! Shape-catalog loader and validator for `*.json` and `hdr/*.hdr` shape catalogs.
//!
//! Includes a self-contained JSON parser (zero external crate dependencies) and cross-validates
//! `talus-gguf`'s [`QuantType`] block geometry, byte-size calculations, split-shard combiner,
//! and binary GGUF header dumps across all 5 target models and their drafters.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use talus_core_api::QuantType;

use crate::error::{GgufError, Result};
use crate::gguf::{compute_tensor_byte_size, quant_type_from_name};
use crate::split::GgufModel;

/// The 5 canonical model shape catalog filenames in `TALUS_SHAPES_DIR`.
pub const MATRIX_SHAPE_CATALOG_FILES: &[&str] = &[
    "qwen3.8-27b.json",
    "laguna-s-2.1.json",
    "mimo-v2.6-flash.json",
    "glm-5.3-flash.json",
    "qwen3.8-flash-next.json",
];

/// One tensor entry from a JSON shape catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogTensor {
    pub name: String,
    pub type_name: String,
    pub qtype: QuantType,
    pub ne: Vec<u64>,
    pub bytes: u64,
}

/// One model component (target model or speculative drafter) in a JSON shape catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentCatalog {
    pub role: String,
    pub files: Vec<String>,
    pub arch: String,
    pub tensor_count: usize,
    pub total_bytes: u64,
    pub type_histogram: BTreeMap<String, usize>,
    pub tensors: Vec<CatalogTensor>,
}

/// Parsed JSON shape catalog for a model family (`qwen3.8-27b`, `laguna-s-2.1`, etc.).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeCatalog {
    pub model: String,
    pub components: Vec<ComponentCatalog>,
}

/// Summary of validating a [`ShapeCatalog`] (and optionally its binary `.gguf.hdr` headers).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogValidationSummary {
    pub model: String,
    pub components_checked: usize,
    pub shards_checked: usize,
    pub tensors_checked: usize,
    pub total_bytes_verified: u64,
}

impl ShapeCatalog {
    /// Opens and parses a JSON shape catalog at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let text = fs::read_to_string(path_ref).map_err(|e| GgufError::Io {
            path: Some(path_ref.display().to_string()),
            message: e.to_string(),
        })?;
        Self::from_json_str(&text)
    }

    /// Parses a JSON shape catalog from a string.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let root = JsonParser::new(text).parse_root()?;
        let obj = root.as_object().ok_or_else(|| GgufError::ShapeValidation {
            message: "catalog root is not a JSON object".to_string(),
        })?;

        let model = obj
            .get("model")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| GgufError::ShapeValidation {
                message: "catalog missing string field 'model'".to_string(),
            })?
            .to_string();

        let comps_arr = obj
            .get("components")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| GgufError::ShapeValidation {
                message: format!("model '{model}': missing array field 'components'"),
            })?;

        let mut components = Vec::with_capacity(comps_arr.len());
        for (c_idx, comp_val) in comps_arr.iter().enumerate() {
            let c_obj = comp_val
                .as_object()
                .ok_or_else(|| GgufError::ShapeValidation {
                    message: format!("model '{model}': component[{c_idx}] is not an object"),
                })?;

            let role = c_obj
                .get("role")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| GgufError::ShapeValidation {
                    message: format!("model '{model}': component[{c_idx}] missing 'role'"),
                })?
                .to_string();

            let arch = c_obj
                .get("arch")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| GgufError::ShapeValidation {
                    message: format!("model '{model}' ({role}): missing 'arch'"),
                })?
                .to_string();

            let files_arr = c_obj
                .get("files")
                .and_then(JsonValue::as_array)
                .ok_or_else(|| GgufError::ShapeValidation {
                    message: format!("model '{model}' ({role}): missing 'files'"),
                })?;
            let mut files = Vec::with_capacity(files_arr.len());
            for f in files_arr {
                let s = f.as_str().ok_or_else(|| GgufError::ShapeValidation {
                    message: format!("model '{model}' ({role}): non-string entry in 'files'"),
                })?;
                files.push(s.to_string());
            }

            let tensors_arr = c_obj
                .get("tensors")
                .and_then(JsonValue::as_array)
                .ok_or_else(|| GgufError::ShapeValidation {
                    message: format!("model '{model}' ({role}): missing 'tensors'"),
                })?;

            let mut tensors = Vec::with_capacity(tensors_arr.len());
            let mut computed_total_bytes = 0u64;
            let mut computed_histogram: BTreeMap<String, usize> = BTreeMap::new();
            for (t_idx, t_val) in tensors_arr.iter().enumerate() {
                let t_obj = t_val
                    .as_object()
                    .ok_or_else(|| GgufError::ShapeValidation {
                        message: format!(
                            "model '{model}' ({role}): tensor[{t_idx}] is not an object"
                        ),
                    })?;
                let name = t_obj
                    .get("name")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| GgufError::ShapeValidation {
                        message: format!(
                            "model '{model}' ({role}): tensor[{t_idx}] missing 'name'"
                        ),
                    })?
                    .to_string();
                let type_name = t_obj
                    .get("type")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| GgufError::ShapeValidation {
                        message: format!(
                            "model '{model}' ({role}): tensor '{name}' missing 'type'"
                        ),
                    })?
                    .to_string();
                let qtype = quant_type_from_name(&type_name).ok_or_else(|| {
                    GgufError::ShapeValidation {
                        message: format!(
                            "model '{model}' ({role}): tensor '{name}' has unknown type '{type_name}'"
                        ),
                    }
                })?;
                let ne_arr = t_obj
                    .get("ne")
                    .and_then(JsonValue::as_array)
                    .ok_or_else(|| GgufError::ShapeValidation {
                        message: format!("model '{model}' ({role}): tensor '{name}' missing 'ne'"),
                    })?;
                let mut ne = Vec::with_capacity(ne_arr.len());
                for d in ne_arr {
                    let dim = d.as_u64().ok_or_else(|| GgufError::ShapeValidation {
                        message: format!(
                            "model '{model}' ({role}): tensor '{name}' has non-integer dim in 'ne'"
                        ),
                    })?;
                    ne.push(dim);
                }
                let bytes = t_obj
                    .get("bytes")
                    .and_then(JsonValue::as_u64)
                    .ok_or_else(|| GgufError::ShapeValidation {
                        message: format!(
                            "model '{model}' ({role}): tensor '{name}' missing 'bytes'"
                        ),
                    })?;

                computed_total_bytes += bytes;
                *computed_histogram.entry(type_name.clone()).or_insert(0) += 1;
                tensors.push(CatalogTensor {
                    name,
                    type_name,
                    qtype,
                    ne,
                    bytes,
                });
            }

            let tensor_count = match c_obj.get("tensor_count").and_then(JsonValue::as_u64) {
                Some(n) => n as usize,
                None => tensors.len(),
            };

            let total_bytes = c_obj
                .get("total_bytes")
                .and_then(JsonValue::as_u64)
                .unwrap_or(computed_total_bytes);

            let type_histogram = if let Some(hist_obj) =
                c_obj.get("type_histogram").and_then(JsonValue::as_object)
            {
                let mut h = BTreeMap::new();
                for (k, v) in hist_obj {
                    let cnt = v.as_u64().ok_or_else(|| GgufError::ShapeValidation {
                        message: format!(
                            "model '{model}' ({role}): invalid count in type_histogram['{k}']"
                        ),
                    })? as usize;
                    h.insert(k.clone(), cnt);
                }
                h
            } else {
                computed_histogram
            };

            components.push(ComponentCatalog {
                role,
                files,
                arch,
                tensor_count,
                total_bytes,
                type_histogram,
                tensors,
            });
        }

        Ok(Self { model, components })
    }

    /// Validates all tensor shapes, `QuantType` block sizes, computed byte sizes, total byte sums,
    /// and type histograms in this catalog.
    pub fn validate(&self) -> Result<CatalogValidationSummary> {
        let mut tensors_checked = 0usize;
        let mut total_bytes_verified = 0u64;
        let mut shards_checked = 0usize;

        for comp in &self.components {
            shards_checked += comp.files.len();
            if comp.tensors.len() != comp.tensor_count {
                return Err(GgufError::ShapeValidation {
                    message: format!(
                        "model '{}' ({}): tensor_count={} != tensors.len()={}",
                        self.model,
                        comp.role,
                        comp.tensor_count,
                        comp.tensors.len()
                    ),
                });
            }

            let mut sum_bytes = 0u64;
            let mut computed_hist: BTreeMap<String, usize> = BTreeMap::new();
            let path_ctx = Some(format!("{} ({})", self.model, comp.role));

            for t in &comp.tensors {
                let computed = compute_tensor_byte_size(t.qtype, &t.ne, &path_ctx, &t.name)?;
                if computed != t.bytes {
                    return Err(GgufError::ShapeValidation {
                        message: format!(
                            "model '{}' ({}) tensor '{}': computed {} B for {:?} {:?}, but catalog says {} B",
                            self.model, comp.role, t.name, computed, t.qtype, t.ne, t.bytes
                        ),
                    });
                }
                sum_bytes += computed;
                *computed_hist.entry(t.type_name.clone()).or_insert(0) += 1;
                tensors_checked += 1;
            }

            if sum_bytes != comp.total_bytes {
                return Err(GgufError::ShapeValidation {
                    message: format!(
                        "model '{}' ({}): sum of tensor bytes {sum_bytes} != total_bytes {}",
                        self.model, comp.role, comp.total_bytes
                    ),
                });
            }

            if computed_hist != comp.type_histogram {
                return Err(GgufError::ShapeValidation {
                    message: format!(
                        "model '{}' ({}): computed type histogram {:?} != catalog {:?}",
                        self.model, comp.role, computed_hist, comp.type_histogram
                    ),
                });
            }

            total_bytes_verified += sum_bytes;
        }

        Ok(CatalogValidationSummary {
            model: self.model.clone(),
            components_checked: self.components.len(),
            shards_checked,
            tensors_checked,
            total_bytes_verified,
        })
    }

    /// Validates this JSON catalog AND cross-checks every component and tensor against the binary
    /// `.gguf.hdr` (+ `.gguf.size`) header dumps in `hdr_dir`.
    pub fn validate_with_headers(
        &self,
        hdr_dir: impl AsRef<Path>,
    ) -> Result<CatalogValidationSummary> {
        let summary = self.validate()?;
        let hdr_dir_ref = hdr_dir.as_ref();

        for comp in &self.components {
            if comp.files.is_empty() {
                return Err(GgufError::ShapeValidation {
                    message: format!("model '{}' ({}) has no files", self.model, comp.role),
                });
            }

            // Open via automatic split-shard discovery from the first `.hdr` file, and verify
            // that the discovered shard count matches `comp.files.len()`.
            let first_hdr = hdr_dir_ref.join(format!("{}.hdr", comp.files[0]));
            let gguf_model = GgufModel::open(&first_hdr)?;
            if gguf_model.shard_count() != comp.files.len() {
                return Err(GgufError::ShapeValidation {
                    message: format!(
                        "model '{}' ({}): opened {} shards from {}, expected {}",
                        self.model,
                        comp.role,
                        gguf_model.shard_count(),
                        first_hdr.display(),
                        comp.files.len()
                    ),
                });
            }

            // Verify offset brackets across all shards.
            gguf_model.check_brackets()?;

            // Verify architecture and total tensor count via GgufView.
            let view = gguf_model.to_gguf_view();
            if view.architecture != comp.arch {
                return Err(GgufError::ShapeValidation {
                    message: format!(
                        "model '{}' ({}): GGUF header arch '{}' != catalog arch '{}'",
                        self.model, comp.role, view.architecture, comp.arch
                    ),
                });
            }
            if view.tensor_count != comp.tensor_count {
                return Err(GgufError::ShapeValidation {
                    message: format!(
                        "model '{}' ({}): GGUF header tensor_count {} != catalog {}",
                        self.model, comp.role, view.tensor_count, comp.tensor_count
                    ),
                });
            }

            // Verify every catalog tensor matches the parsed binary GGUF header entry 1-to-1.
            for ct in &comp.tensors {
                let (entry, loc) =
                    gguf_model
                        .find_tensor(&ct.name)
                        .ok_or_else(|| GgufError::ShapeValidation {
                            message: format!(
                                "model '{}' ({}): tensor '{}' missing in GGUF headers",
                                self.model, comp.role, ct.name
                            ),
                        })?;
                if entry.qtype != ct.qtype {
                    return Err(GgufError::ShapeValidation {
                        message: format!(
                            "model '{}' ({}) tensor '{}': GGUF qtype {:?} != catalog {:?}",
                            self.model, comp.role, ct.name, entry.qtype, ct.qtype
                        ),
                    });
                }
                if entry.shape() != ct.ne.as_slice() {
                    return Err(GgufError::ShapeValidation {
                        message: format!(
                            "model '{}' ({}) tensor '{}': GGUF shape {:?} != catalog {:?}",
                            self.model,
                            comp.role,
                            ct.name,
                            entry.shape(),
                            ct.ne
                        ),
                    });
                }
                if entry.byte_size != ct.bytes || loc.byte_size != ct.bytes {
                    return Err(GgufError::ShapeValidation {
                        message: format!(
                            "model '{}' ({}) tensor '{}': GGUF byte_size {} != catalog {}",
                            self.model, comp.role, ct.name, entry.byte_size, ct.bytes
                        ),
                    });
                }
            }
        }

        Ok(summary)
    }
}

/// Loads and validates all 5 canonical model shape catalogs in `shapes_dir`, including their
/// binary `.gguf.hdr` + `.gguf.size` header dumps in `<shapes_dir>/hdr` if present.
pub fn validate_matrix_shapes_dir(
    shapes_dir: impl AsRef<Path>,
) -> Result<Vec<CatalogValidationSummary>> {
    let dir = shapes_dir.as_ref();
    let hdr_dir = dir.join("hdr");
    let has_hdrs = hdr_dir.is_dir();

    let mut summaries = Vec::with_capacity(MATRIX_SHAPE_CATALOG_FILES.len());
    for &fname in MATRIX_SHAPE_CATALOG_FILES {
        let json_path = dir.join(fname);
        let catalog = ShapeCatalog::open(&json_path)?;
        let summary = if has_hdrs {
            catalog.validate_with_headers(&hdr_dir)?
        } else {
            catalog.validate()?
        };
        summaries.push(summary);
    }
    Ok(summaries)
}

// ============================================================================
// Self-contained JSON parser (RFC 8259)
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
enum JsonValue {
    Null,
    Bool(bool),
    Uint(u64),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::Uint(u) => Some(u),
            Self::Int(i) if i >= 0 => Some(i as u64),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(o) => Some(o),
            _ => None,
        }
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            pos: 0,
        }
    }

    fn err(&self, msg: impl Into<String>) -> GgufError {
        GgufError::ShapeValidation {
            message: format!("JSON parse error at byte {}: {}", self.pos, msg.into()),
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_root(&mut self) -> Result<JsonValue> {
        self.skip_ws();
        let v = self.parse_value()?;
        self.skip_ws();
        if self.pos != self.bytes.len() {
            return Err(self.err("trailing characters after JSON root value"));
        }
        Ok(v)
    }

    fn parse_value(&mut self) -> Result<JsonValue> {
        self.skip_ws();
        let b = self.peek().ok_or_else(|| self.err("unexpected EOF"))?;
        match b {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => self.parse_string().map(JsonValue::String),
            b't' => {
                self.expect_literal(b"true")?;
                Ok(JsonValue::Bool(true))
            }
            b'f' => {
                self.expect_literal(b"false")?;
                Ok(JsonValue::Bool(false))
            }
            b'n' => {
                self.expect_literal(b"null")?;
                Ok(JsonValue::Null)
            }
            b'-' | b'0'..=b'9' => self.parse_number(),
            _ => Err(self.err(format!("unexpected byte '{}'", b as char))),
        }
    }

    fn expect_literal(&mut self, lit: &[u8]) -> Result<()> {
        if self.bytes.get(self.pos..self.pos + lit.len()) == Some(lit) {
            self.pos += lit.len();
            Ok(())
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue> {
        self.pos += 1; // '{'
        self.skip_ws();
        let mut map = BTreeMap::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected string key in object"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(self.err("expected ':' after object key"));
            }
            self.pos += 1;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(map));
                }
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue> {
        self.pos += 1; // '['
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            let val = self.parse_value()?;
            items.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.pos += 1; // '"'
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = *self
                        .bytes
                        .get(self.pos)
                        .ok_or_else(|| self.err("unexpected EOF in string escape"))?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hex = self
                                .bytes
                                .get(self.pos..self.pos + 4)
                                .ok_or_else(|| self.err("truncated \\uXXXX escape"))?;
                            self.pos += 4;
                            let s = std::str::from_utf8(hex)
                                .map_err(|_| self.err("invalid hex in \\u escape"))?;
                            let code = u32::from_str_radix(s, 16)
                                .map_err(|_| self.err("invalid hex digits in \\u escape"))?;
                            let ch = char::from_u32(code).unwrap_or(char::REPLACEMENT_CHARACTER);
                            out.push(ch);
                        }
                        _ => return Err(self.err("invalid escape character")),
                    }
                }
                0x00..=0x1F => return Err(self.err("unescaped control character in string")),
                _ => {
                    // Fast path for UTF-8 sequences: back up 1 byte and decode next char from slice.
                    self.pos -= 1;
                    let rest = std::str::from_utf8(&self.bytes[self.pos..])
                        .map_err(|_| self.err("invalid UTF-8 in JSON string"))?;
                    let ch = rest.chars().next().unwrap();
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
        Err(self.err("unterminated JSON string"))
    }

    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.pos;
        let mut is_float = false;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b'0'..=b'9') = self.peek() {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.err("invalid UTF-8 in number"))?;
        if !is_float {
            if let Ok(u) = s.parse::<u64>() {
                return Ok(JsonValue::Uint(u));
            }
            if let Ok(i) = s.parse::<i64>() {
                return Ok(JsonValue::Int(i));
            }
        }
        let f = s.parse::<f64>().map_err(|_| self.err("invalid number"))?;
        Ok(JsonValue::Float(f))
    }
}
