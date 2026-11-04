//! Minimal, zero-dependency RFC 8259 JSON parser, serializer, base64, and ID helpers.
//!
//! Preserves insertion order in JSON objects (like Python's `dict`) and distinguishes
//! integers (`JsonValue::Int`) from floats (`JsonValue::Float`) and booleans (`JsonValue::Bool`)
//! so request sampling validation (`check_request_sampling`) matches `serve/server.py` exactly.

use std::fmt::{self, Write as _};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn obj(entries: Vec<(&str, JsonValue)>) -> Self {
        JsonValue::Object(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }

    pub fn empty_obj() -> Self {
        JsonValue::Object(Vec::new())
    }

    pub fn empty_arr() -> Self {
        JsonValue::Array(Vec::new())
    }

    pub fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, JsonValue::Bool(_))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, JsonValue::Int(_))
    }

    pub fn is_float(&self) -> bool {
        matches!(self, JsonValue::Float(_))
    }

    pub fn is_number(&self) -> bool {
        matches!(self, JsonValue::Int(_) | JsonValue::Float(_))
    }

    pub fn is_str(&self) -> bool {
        matches!(self, JsonValue::Str(_))
    }

    pub fn is_array(&self) -> bool {
        matches!(self, JsonValue::Array(_))
    }

    pub fn is_object(&self) -> bool {
        matches!(self, JsonValue::Object(_))
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Int(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Int(n) => Some(*n as f64),
            JsonValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<JsonValue>> {
        match self {
            JsonValue::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(o) => Some(o.as_slice()),
            _ => None,
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut Vec<(String, JsonValue)>> {
        match self {
            JsonValue::Object(o) => Some(o),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut JsonValue> {
        match self {
            JsonValue::Object(entries) => {
                entries.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v)
            }
            _ => None,
        }
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_str())
    }

    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(|v| v.as_i64())
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get_i64(key).and_then(|n| u64::try_from(n).ok())
    }

    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.get_i64(key).and_then(|n| usize::try_from(n).ok())
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_f64())
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(|v| v.as_bool())
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn insert(&mut self, key: impl Into<String>, val: impl Into<JsonValue>) {
        let key = key.into();
        let val = val.into();
        if let JsonValue::Object(entries) = self {
            if let Some((_, existing)) = entries.iter_mut().find(|(k, _)| *k == key) {
                *existing = val;
            } else {
                entries.push((key, val));
            }
        }
    }

    pub fn remove(&mut self, key: &str) -> Option<JsonValue> {
        if let JsonValue::Object(entries) = self {
            if let Some(pos) = entries.iter().position(|(k, _)| k == key) {
                return Some(entries.remove(pos).1);
            }
        }
        None
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            JsonValue::Null => false,
            JsonValue::Bool(b) => *b,
            JsonValue::Int(n) => *n != 0,
            JsonValue::Float(f) => *f != 0.0,
            JsonValue::Str(s) => !s.is_empty(),
            JsonValue::Array(a) => !a.is_empty(),
            JsonValue::Object(o) => !o.is_empty(),
        }
    }

    pub fn parse_bytes(bytes: &[u8]) -> Result<JsonValue, String> {
        let s = std::str::from_utf8(bytes).map_err(|e| format!("invalid UTF-8: {e}"))?;
        let s = s.strip_prefix('\u{feff}').unwrap_or(s);
        Self::parse(s)
    }

    pub fn parse(input: &str) -> Result<JsonValue, String> {
        let mut parser = Parser {
            bytes: input.as_bytes(),
            pos: 0,
            depth: 0,
        };
        let val = parser.parse_value()?;
        parser.skip_ws();
        if parser.pos < parser.bytes.len() {
            return Err(format!("trailing characters at byte {}", parser.pos));
        }
        Ok(val)
    }

    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(128);
        self.write_compact(&mut out);
        out
    }

    pub fn to_json_pretty(&self) -> String {
        let mut out = String::with_capacity(256);
        self.write_pretty(&mut out, 0);
        out
    }

    fn write_compact(&self, out: &mut String) {
        match self {
            JsonValue::Null => out.push_str("null"),
            JsonValue::Bool(true) => out.push_str("true"),
            JsonValue::Bool(false) => out.push_str("false"),
            JsonValue::Int(n) => {
                let _ = write!(out, "{n}");
            }
            JsonValue::Float(f) => {
                if !f.is_finite() {
                    out.push_str("null");
                } else if f.fract() == 0.0 && f.abs() < 1e15 {
                    let _ = write!(out, "{f:.1}");
                } else {
                    let _ = write!(out, "{f}");
                }
            }
            JsonValue::Str(s) => write_escaped_str(out, s),
            JsonValue::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    item.write_compact(out);
                }
                out.push(']');
            }
            JsonValue::Object(entries) => {
                out.push('{');
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_escaped_str(out, k);
                    out.push_str(": ");
                    v.write_compact(out);
                }
                out.push('}');
            }
        }
    }

    fn write_pretty(&self, out: &mut String, indent: usize) {
        match self {
            JsonValue::Array(items) if items.is_empty() => out.push_str("[]"),
            JsonValue::Array(items) => {
                out.push_str("[\n");
                for (i, item) in items.iter().enumerate() {
                    for _ in 0..(indent + 2) {
                        out.push(' ');
                    }
                    item.write_pretty(out, indent + 2);
                    if i + 1 < items.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                for _ in 0..indent {
                    out.push(' ');
                }
                out.push(']');
            }
            JsonValue::Object(entries) if entries.is_empty() => out.push_str("{}"),
            JsonValue::Object(entries) => {
                out.push_str("{\n");
                for (i, (k, v)) in entries.iter().enumerate() {
                    for _ in 0..(indent + 2) {
                        out.push(' ');
                    }
                    write_escaped_str(out, k);
                    out.push_str(": ");
                    v.write_pretty(out, indent + 2);
                    if i + 1 < entries.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                for _ in 0..indent {
                    out.push(' ');
                }
                out.push('}');
            }
            _ => self.write_compact(out),
        }
    }
}

impl fmt::Display for JsonValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_json_string())
    }
}

impl From<bool> for JsonValue {
    fn from(v: bool) -> Self {
        JsonValue::Bool(v)
    }
}

impl From<i64> for JsonValue {
    fn from(v: i64) -> Self {
        JsonValue::Int(v)
    }
}

impl From<i32> for JsonValue {
    fn from(v: i32) -> Self {
        JsonValue::Int(i64::from(v))
    }
}

impl From<usize> for JsonValue {
    fn from(v: usize) -> Self {
        JsonValue::Int(v as i64)
    }
}

impl From<u64> for JsonValue {
    fn from(v: u64) -> Self {
        JsonValue::Int(v as i64)
    }
}

impl From<u32> for JsonValue {
    fn from(v: u32) -> Self {
        JsonValue::Int(i64::from(v))
    }
}

impl From<f64> for JsonValue {
    fn from(v: f64) -> Self {
        JsonValue::Float(v)
    }
}

impl From<&str> for JsonValue {
    fn from(v: &str) -> Self {
        JsonValue::Str(v.to_string())
    }
}

impl From<String> for JsonValue {
    fn from(v: String) -> Self {
        JsonValue::Str(v)
    }
}

impl<T: Into<JsonValue>> From<Vec<T>> for JsonValue {
    fn from(v: Vec<T>) -> Self {
        JsonValue::Array(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<JsonValue>> From<Option<T>> for JsonValue {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(x) => x.into(),
            None => JsonValue::Null,
        }
    }
}

pub fn write_escaped_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Deepest array/object nesting [`JsonValue::parse`] accepts (serde_json's default). The parser recurses once per
/// level, so the depth is bounded.
pub const MAX_JSON_DEPTH: usize = 128;

/// Objects with more keys than this look duplicates up in a hash index instead of a linear scan (a body of a
/// million distinct keys was O(n^2): minutes of CPU per request).
const OBJECT_INDEX_AFTER: usize = 16;

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        let b = self
            .peek()
            .ok_or_else(|| "unexpected end of JSON".to_string())?;
        match b {
            b'n' => self.parse_literal(b"null", JsonValue::Null),
            b't' => self.parse_literal(b"true", JsonValue::Bool(true)),
            b'f' => self.parse_literal(b"false", JsonValue::Bool(false)),
            b'"' => self.parse_string().map(JsonValue::Str),
            b'[' => self.parse_array(),
            b'{' => self.parse_object(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            _ => Err(format!(
                "unexpected character {:?} at byte {}",
                b as char, self.pos
            )),
        }
    }

    fn parse_literal(&mut self, lit: &[u8], val: JsonValue) -> Result<JsonValue, String> {
        if self.bytes.get(self.pos..self.pos + lit.len()) == Some(lit) {
            self.pos += lit.len();
            Ok(val)
        } else {
            Err(format!("invalid literal at byte {}", self.pos))
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err(format!("expected '\"' at byte {}", self.pos));
        }
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                if self.pos >= self.bytes.len() {
                    return Err("unterminated escape sequence".to_string());
                }
                let esc = self.bytes[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\x08'),
                    b'f' => out.push('\x0c'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let u1 = self.parse_hex4()?;
                        if (0xD800..=0xDBFF).contains(&u1) {
                            if self.bytes.get(self.pos..self.pos + 2) != Some(b"\\u") {
                                return Err("unpaired high surrogate in JSON string".to_string());
                            }
                            self.pos += 2;
                            let u2 = self.parse_hex4()?;
                            if !(0xDC00..=0xDFFF).contains(&u2) {
                                return Err("invalid low surrogate in JSON string".to_string());
                            }
                            let cp =
                                0x10000 + (((u1 as u32 - 0xD800) << 10) | (u2 as u32 - 0xDC00));
                            let ch = char::from_u32(cp)
                                .ok_or_else(|| "invalid unicode codepoint".to_string())?;
                            out.push(ch);
                        } else if (0xDC00..=0xDFFF).contains(&u1) {
                            return Err("unpaired low surrogate in JSON string".to_string());
                        } else {
                            let ch = char::from_u32(u1 as u32)
                                .ok_or_else(|| "invalid unicode codepoint".to_string())?;
                            out.push(ch);
                        }
                    }
                    _ => return Err(format!("invalid escape '\\{}'", esc as char)),
                }
            } else if b < 0x20 {
                return Err(format!("unescaped control character at byte {}", self.pos));
            } else {
                // Copy the whole run of plain bytes up to the next quote, backslash or control byte at once.
                // These delimiters are ASCII, so the run ends on a char boundary; validating just the run keeps the
                // string linear (validating the rest of the buffer per character was O(n^2): ~0.6 s for a 256 KB
                // prompt, talus#51).
                let end = self.bytes[self.pos..]
                    .iter()
                    .position(|&c| c == b'"' || c == b'\\' || c < 0x20)
                    .map_or(self.bytes.len(), |n| self.pos + n);
                let run = std::str::from_utf8(&self.bytes[self.pos..end])
                    .map_err(|_| "invalid UTF-8 in string".to_string())?;
                out.push_str(run);
                self.pos = end;
            }
        }
        Err("unterminated string".to_string())
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        let slice = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| "incomplete \\u escape".to_string())?;
        let s = std::str::from_utf8(slice).map_err(|_| "invalid \\u escape".to_string())?;
        let v = u16::from_str_radix(s, 16).map_err(|_| "invalid hex in \\u escape".to_string())?;
        self.pos += 4;
        Ok(v)
    }

    /// Enter one array/object level; an error past [`MAX_JSON_DEPTH`].
    fn descend(&mut self) -> Result<(), String> {
        if self.depth >= MAX_JSON_DEPTH {
            return Err(format!(
                "JSON nested too deeply (more than {MAX_JSON_DEPTH} levels) at byte {}",
                self.pos
            ));
        }
        self.depth += 1;
        Ok(())
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.descend()?;
        let v = self.parse_array_items();
        self.depth -= 1;
        v
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.descend()?;
        let v = self.parse_object_entries();
        self.depth -= 1;
        v
    }

    fn parse_array_items(&mut self) -> Result<JsonValue, String> {
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
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.pos)),
            }
        }
    }

    fn parse_object_entries(&mut self) -> Result<JsonValue, String> {
        self.pos += 1; // '{'
        self.skip_ws();
        let mut entries: Vec<(String, JsonValue)> = Vec::new();
        // key -> position in `entries`, built once the object outgrows a linear scan
        let mut index: Option<std::collections::HashMap<String, usize>> = None;
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at byte {}", self.pos));
            }
            self.pos += 1;
            let val = self.parse_value()?;
            // a repeated key keeps its first position and takes the last value
            let found = match index.as_ref() {
                Some(ix) => ix.get(&key).copied(),
                None => entries.iter().position(|(k, _)| *k == key),
            };
            if let Some(i) = found {
                entries[i].1 = val;
            } else {
                if let Some(ix) = index.as_mut() {
                    ix.insert(key.clone(), entries.len());
                }
                entries.push((key, val));
                if index.is_none() && entries.len() > OBJECT_INDEX_AFTER {
                    index = Some(
                        entries
                            .iter()
                            .enumerate()
                            .map(|(i, (k, _))| (k.clone(), i))
                            .collect(),
                    );
                }
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(entries));
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.pos)),
            }
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
            }
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(format!("invalid number at byte {}", self.pos)),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!("invalid decimal in number at byte {}", self.pos));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!("invalid exponent in number at byte {}", self.pos));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let raw = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid UTF-8 in number".to_string())?;
        if !is_float {
            if let Ok(n) = raw.parse::<i64>() {
                return Ok(JsonValue::Int(n));
            }
        }
        let f = raw
            .parse::<f64>()
            .map_err(|_| format!("invalid float {raw:?}"))?;
        if !f.is_finite() {
            return Err(format!("non-finite number {raw:?}"));
        }
        Ok(JsonValue::Float(f))
    }
}

static ID_COUNTER: AtomicU64 = AtomicU64::new(1);

pub fn next_hex_id(len: usize) -> String {
    let seq = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut state = nanos
        .wrapping_mul(6364136223846793005)
        .wrapping_add(seq.wrapping_mul(1442695040888963407));
    let mut out = String::with_capacity(len);
    while out.len() < len {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let word = state.wrapping_mul(0x2545F4914F6CDD1D);
        let _ = write!(out, "{word:016x}");
    }
    out.truncate(len);
    out
}

pub fn unix_timestamp_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

const B64_STD_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

const B64_URL_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

pub fn base64_std_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_STD_ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(B64_STD_ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_STD_ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_STD_ALPHABET[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

pub fn base64_url_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_URL_ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(B64_URL_ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_URL_ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_URL_ALPHABET[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

pub fn base64_url_decode(s: &str) -> Result<Vec<u8>, String> {
    let clean: Vec<u8> = s
        .bytes()
        .filter(|b| !matches!(b, b'=' | b'\r' | b'\n' | b' '))
        .collect();
    if clean.len() % 4 == 1 {
        return Err("invalid base64 length".to_string());
    }
    let decode_val = |b: u8| -> Result<u32, String> {
        match b {
            b'A'..=b'Z' => Ok((b - b'A') as u32),
            b'a'..=b'z' => Ok((b - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((b - b'0' + 52) as u32),
            b'-' | b'+' => Ok(62),
            b'_' | b'/' => Ok(63),
            _ => Err(format!("invalid base64 character {:?}", b as char)),
        }
    };
    let mut out = Vec::with_capacity(clean.len() * 3 / 4);
    for chunk in clean.chunks(4) {
        let v0 = decode_val(chunk[0])?;
        let v1 = decode_val(chunk[1])?;
        let v2 = if chunk.len() > 2 {
            decode_val(chunk[2])?
        } else {
            0
        };
        let v3 = if chunk.len() > 3 {
            decode_val(chunk[3])?
        } else {
            0
        };
        let triple = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
        out.push(((triple >> 16) & 0xFF) as u8);
        if chunk.len() > 2 {
            out.push(((triple >> 8) & 0xFF) as u8);
        }
        if chunk.len() > 3 {
            out.push((triple & 0xFF) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod string_tests {
    use super::JsonValue;

    #[test]
    fn strings_with_multibyte_and_escapes() {
        let v = JsonValue::parse(r#"{"a":"héllo 〈|EOS|〉 \"q\" \\ \n é 😀 end","b":""}"#).unwrap();
        assert_eq!(v.get_str("a"), Some("héllo 〈|EOS|〉 \"q\" \\ \n é 😀 end"));
        assert_eq!(v.get_str("b"), Some(""));
        assert!(JsonValue::parse("\"abc").is_err());
        assert!(JsonValue::parse("\"a\u{1}b\"").is_err());
    }

    #[test]
    fn long_string_parses_in_linear_time() {
        // talus#51: a 32K-token prompt (~128 KB) cost ~0.15 s and 256 KB ~0.6 s with the per-character re-validation;
        // 4 MB would take minutes. Linear: well under a second even unoptimized.
        let body: String = "def f(x):\n    return x  # é〈\n".repeat(4 << 20 >> 5);
        let json = format!(
            "{{\"prompt\":{}}}",
            JsonValue::from(body.as_str()).to_json_string()
        );
        let t = std::time::Instant::now();
        let v = JsonValue::parse(&json).unwrap();
        assert_eq!(v.get_str("prompt"), Some(body.as_str()));
        assert!(
            t.elapsed().as_secs_f64() < 3.0,
            "parse took {:?}",
            t.elapsed()
        );
    }

    #[test]
    fn nesting_limit() {
        use super::MAX_JSON_DEPTH;
        let deep = |n: usize| format!("{}{}", "[".repeat(n), "]".repeat(n));
        assert!(JsonValue::parse(&deep(MAX_JSON_DEPTH)).is_ok());
        let e = JsonValue::parse(&deep(MAX_JSON_DEPTH + 1)).unwrap_err();
        assert!(e.contains("nested too deeply"), "{e}");
        // mixed objects and arrays count together; siblings do not add depth
        let mixed = format!(
            "{}0{}",
            "{\"a\":[".repeat(MAX_JSON_DEPTH / 2),
            "]}".repeat(MAX_JSON_DEPTH / 2)
        );
        assert!(JsonValue::parse(&mixed).is_ok());
        let wide = format!("[{}]", vec!["[[1]]"; 10_000].join(","));
        assert!(JsonValue::parse(&wide).is_ok());
        // past the limit is an error, not a stack overflow, even far past it
        assert!(JsonValue::parse(&deep(1_000_000)).is_err());
    }

    #[test]
    fn duplicate_keys_last_wins_first_position_kept() {
        for n in [3usize, 40] {
            let mut body: Vec<String> = (0..n).map(|i| format!("\"k{i}\":{i}")).collect();
            body.push("\"k1\":\"late\"".to_string());
            let v = JsonValue::parse(&format!("{{{}}}", body.join(","))).unwrap();
            let o = v.as_object().unwrap();
            assert_eq!(o.len(), n);
            assert_eq!(o[1].0, "k1");
            assert_eq!(v.get_str("k1"), Some("late"));
            assert_eq!(v.get_i64(&format!("k{}", n - 1)), Some(n as i64 - 1));
        }
    }

    #[test]
    fn wide_object_parses_in_linear_time() {
        // 200k distinct keys: the per-key linear duplicate scan was ~2e10 compares
        let body: Vec<String> = (0..200_000).map(|i| format!("\"k{i}\":0")).collect();
        let json = format!("{{{}}}", body.join(","));
        let t = std::time::Instant::now();
        let v = JsonValue::parse(&json).unwrap();
        assert_eq!(v.as_object().unwrap().len(), 200_000);
        assert!(
            t.elapsed().as_secs_f64() < 5.0,
            "parse took {:?}",
            t.elapsed()
        );
    }
}
