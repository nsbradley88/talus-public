//! Grammar-constrained JSON output (server extension E6, `response_format: json_schema`).
//!
//! A JSON Schema subset is compiled to a small node graph; a byte-level pushdown recogniser walks it (a set of
//! configurations, so `anyOf`/`oneOf`/type unions are followed in parallel); per decode step the set of tokens
//! whose bytes keep at least one configuration alive is computed by walking a byte trie of the vocabulary, and
//! memoised by recogniser state (inside a free string the state does not change, so a long string costs one mask).
//!
//! Supported keywords (the talus-bench validator's subset, `talus-bench-core/src/jsonschema.rs`): `type` (string or
//! list), `enum`, `const`, `properties`, `required`, `additionalProperties` (bool or schema), `items`, `minItems`,
//! `maxItems`, `uniqueItems`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `minLength`,
//! `maxLength`, `anyOf`, `oneOf`, `allOf` (merged), and the annotations `title`, `description`, `$schema`,
//! `$comment`. Anything else is rejected, so a request never gets half-understood constraints.
//!
//! Deliberate restrictions of the generated JSON (it stays valid JSON that validates): no exponent in numbers, at
//! most [`WS_MAX`] whitespace bytes between tokens, no `\u` escapes of UTF-16 surrogates, keys of a closed object
//! match the declared names byte for byte, nothing after the top-level value but an end token. `oneOf` is treated
//! as `anyOf` (exact when the branches are disjoint, e.g. `const` discriminators); `uniqueItems` compares the raw
//! item text (exact for strings without escapes and for integers).

use serde_json::{Map, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// Whitespace bytes allowed between two JSON tokens (bounds the "endless newlines" failure of constrained decoding).
pub const WS_MAX: u8 = 16;
const NUM_MAX: usize = 24;
const MAX_CFGS: usize = 64;
const MAX_PROPS: usize = 64;
const MAX_DEPTH: usize = 32;
const NONE: u32 = u32::MAX;
/// Masks kept per compiled schema (each is n_vocab bits).
const MEMO_CAP: usize = 128;

const KEYWORDS: &[&str] = &[
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "anyOf",
    "allOf",
    "oneOf",
    "title",
    "description",
    "$schema",
    "$comment",
];
const ANNOTATIONS: &[&str] = &["title", "description", "$schema", "$comment"];

#[derive(Clone, Debug)]
enum Node {
    Never,
    Union(Vec<u32>),
    Lit(Vec<Vec<u8>>),
    Str {
        min: u32,
        max: u32,
    },
    Num {
        int: bool,
        lo: f64,
        lo_ex: bool,
        hi: f64,
        hi_ex: bool,
    },
    Obj {
        props: Vec<(Vec<u8>, u32)>,
        required: u64,
        additional: u32,
    },
    Arr {
        items: u32,
        min: u32,
        max: u32,
        unique: bool,
    },
}

/// A compiled schema plus its per-state mask memo.
pub struct Grammar {
    nodes: Vec<Node>,
    root: u32,
    memo: Mutex<HashMap<State, Arc<Vec<u64>>>>,
}

impl std::fmt::Debug for Grammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grammar")
            .field("nodes", &self.nodes.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Schema preparation and compilation
// ---------------------------------------------------------------------------------------------------------------

/// Checks `schema` and returns its compact form without annotations: what the server forwards to the engine.
pub fn prepare(schema: &Value) -> Result<String, String> {
    let stripped = strip_annotations(schema);
    compile_value(&stripped)?;
    serde_json::to_string(&stripped).map_err(|e| e.to_string())
}

/// Compiles a schema given as JSON text (the engine side of [`prepare`]).
pub fn compile(schema_json: &str) -> Result<Grammar, String> {
    let v: Value =
        serde_json::from_str(schema_json).map_err(|e| format!("json_schema: not JSON: {e}"))?;
    compile_value(&v)
}

fn strip_annotations(s: &Value) -> Value {
    match s {
        Value::Object(o) => {
            let mut out = Map::new();
            for (k, v) in o {
                if ANNOTATIONS.contains(&k.as_str()) {
                    continue;
                }
                let nv = match k.as_str() {
                    "properties" => match v {
                        Value::Object(p) => Value::Object(
                            p.iter()
                                .map(|(pk, pv)| (pk.clone(), strip_annotations(pv)))
                                .collect(),
                        ),
                        _ => v.clone(),
                    },
                    "items" | "additionalProperties" => strip_annotations(v),
                    "anyOf" | "oneOf" | "allOf" => match v {
                        Value::Array(a) => Value::Array(a.iter().map(strip_annotations).collect()),
                        _ => v.clone(),
                    },
                    _ => v.clone(),
                };
                out.insert(k.clone(), nv);
            }
            Value::Object(out)
        }
        _ => s.clone(),
    }
}

fn compile_value(v: &Value) -> Result<Grammar, String> {
    let mut c = Compiler {
        nodes: Vec::new(),
        any: None,
    };
    let root = c.schema(v, 0)?;
    if matches!(c.nodes[root as usize], Node::Never) {
        return Err("json_schema: the schema admits no value".into());
    }
    Ok(Grammar {
        nodes: c.nodes,
        root,
        memo: Mutex::new(HashMap::new()),
    })
}

struct Compiler {
    nodes: Vec<Node>,
    any: Option<u32>,
}

fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| json_eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| json_eq(v, w)))
        }
        _ => a == b,
    }
}

fn type_matches(t: &str, v: &Value) -> bool {
    match t {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        "number" => v.is_number(),
        "integer" => match v {
            Value::Number(n) => {
                n.is_i64() || n.is_u64() || n.as_f64().is_some_and(|f| f.fract() == 0.0)
            }
            _ => false,
        },
        _ => false,
    }
}

const ALL_TYPES: &[&str] = &["object", "array", "string", "number", "boolean", "null"];

fn type_list(v: &Value) -> Result<Vec<String>, String> {
    let ok = |s: &str| ALL_TYPES.contains(&s) || s == "integer";
    match v {
        Value::String(s) if ok(s) => Ok(vec![s.clone()]),
        Value::Array(a) if !a.is_empty() => a
            .iter()
            .map(|t| match t.as_str() {
                Some(s) if ok(s) => Ok(s.to_string()),
                _ => Err(format!("json_schema: bad type {t}")),
            })
            .collect(),
        _ => Err(format!("json_schema: bad type {v}")),
    }
}

/// allOf-style merge of two schemas (the subset that has a clean intersection).
fn merge(a: &Value, b: &Value) -> Result<Value, String> {
    let (ao, bo) = match (a, b) {
        (Value::Bool(true), x) | (x, Value::Bool(true)) => return Ok(x.clone()),
        (Value::Bool(false), _) | (_, Value::Bool(false)) => return Ok(Value::Bool(false)),
        (Value::Object(ao), Value::Object(bo)) => (ao, bo),
        _ => return Err("json_schema: a schema must be an object or a boolean".into()),
    };
    let mut out = ao.clone();
    for (k, vb) in bo {
        let Some(va) = ao.get(k) else {
            out.insert(k.clone(), vb.clone());
            continue;
        };
        let nv = match k.as_str() {
            "type" => {
                let ta = type_list(va)?;
                let tb = type_list(vb)?;
                let mut t = Vec::new();
                for x in &ta {
                    for y in &tb {
                        let r = match (x.as_str(), y.as_str()) {
                            (p, q) if p == q => Some(p),
                            ("integer", "number") | ("number", "integer") => Some("integer"),
                            _ => None,
                        };
                        if let Some(r) = r {
                            if !t.contains(&Value::from(r)) {
                                t.push(Value::from(r));
                            }
                        }
                    }
                }
                if t.is_empty() {
                    return Ok(Value::Bool(false));
                }
                Value::Array(t)
            }
            "enum" => {
                let (Some(ea), Some(eb)) = (va.as_array(), vb.as_array()) else {
                    return Err("json_schema: enum must be a list".into());
                };
                Value::Array(
                    ea.iter()
                        .filter(|x| eb.iter().any(|y| json_eq(x, y)))
                        .cloned()
                        .collect(),
                )
            }
            "const" => {
                if !json_eq(va, vb) {
                    return Ok(Value::Bool(false));
                }
                va.clone()
            }
            "properties" => {
                let (Some(pa), Some(pb)) = (va.as_object(), vb.as_object()) else {
                    return Err("json_schema: properties must be an object".into());
                };
                let mut m = pa.clone();
                for (pk, pv) in pb {
                    let nv = match pa.get(pk) {
                        Some(x) => merge(x, pv)?,
                        None => pv.clone(),
                    };
                    m.insert(pk.clone(), nv);
                }
                Value::Object(m)
            }
            "required" => {
                let mut r = va.as_array().cloned().unwrap_or_default();
                for x in vb.as_array().into_iter().flatten() {
                    if !r.contains(x) {
                        r.push(x.clone());
                    }
                }
                Value::Array(r)
            }
            "items" => merge(va, vb)?,
            "additionalProperties" => {
                if json_eq(va, vb) {
                    va.clone()
                } else {
                    return Err(
                        "json_schema: allOf with different additionalProperties is not supported"
                            .into(),
                    );
                }
            }
            "minItems" | "minLength" | "minimum" | "exclusiveMinimum" => {
                let (x, y) = (va.as_f64().unwrap_or(0.0), vb.as_f64().unwrap_or(0.0));
                if x >= y {
                    va.clone()
                } else {
                    vb.clone()
                }
            }
            "maxItems" | "maxLength" | "maximum" | "exclusiveMaximum" => {
                let (x, y) = (va.as_f64().unwrap_or(0.0), vb.as_f64().unwrap_or(0.0));
                if x <= y {
                    va.clone()
                } else {
                    vb.clone()
                }
            }
            "uniqueItems" => Value::Bool(va == &Value::Bool(true) || vb == &Value::Bool(true)),
            "anyOf" | "oneOf" | "allOf" => {
                // keep both: a nested allOf of the two combinators
                let mut m = Map::new();
                m.insert(k.clone(), vb.clone());
                let prev = out.remove("allOf");
                let mut all = match prev {
                    Some(Value::Array(a)) => a,
                    _ => Vec::new(),
                };
                all.push(Value::Object(m));
                out.insert("allOf".into(), Value::Array(all));
                continue;
            }
            _ => va.clone(),
        };
        out.insert(k.clone(), nv);
    }
    Ok(Value::Object(out))
}

fn as_u32(v: Option<&Value>, dflt: u32) -> Result<u32, String> {
    match v {
        None => Ok(dflt),
        Some(x) => x
            .as_u64()
            .map(|n| n.min(u32::MAX as u64 - 1) as u32)
            .ok_or_else(|| format!("json_schema: expected a non-negative integer, got {x}")),
    }
}

impl Compiler {
    fn push(&mut self, n: Node) -> u32 {
        self.nodes.push(n);
        (self.nodes.len() - 1) as u32
    }

    fn any(&mut self) -> u32 {
        if let Some(a) = self.any {
            return a;
        }
        let a = self.push(Node::Never);
        self.any = Some(a);
        let o = self.push(Node::Obj {
            props: Vec::new(),
            required: 0,
            additional: a,
        });
        let r = self.push(Node::Arr {
            items: a,
            min: 0,
            max: NONE,
            unique: false,
        });
        let s = self.push(Node::Str { min: 0, max: NONE });
        let n = self.push(Node::Num {
            int: false,
            lo: f64::NEG_INFINITY,
            lo_ex: false,
            hi: f64::INFINITY,
            hi_ex: false,
        });
        let l = self.push(Node::Lit(vec![
            b"true".to_vec(),
            b"false".to_vec(),
            b"null".to_vec(),
        ]));
        self.nodes[a as usize] = Node::Union(vec![o, r, s, n, l]);
        a
    }

    fn union(&mut self, alts: Vec<u32>) -> u32 {
        let mut flat = Vec::new();
        for a in alts {
            match &self.nodes[a as usize] {
                Node::Never => {}
                Node::Union(xs) if Some(a) != self.any => flat.extend(xs.iter().copied()),
                _ => flat.push(a),
            }
        }
        flat.dedup();
        match flat.len() {
            0 => self.push(Node::Never),
            1 => flat[0],
            _ => self.push(Node::Union(flat)),
        }
    }

    fn schema(&mut self, s: &Value, depth: usize) -> Result<u32, String> {
        if depth > MAX_DEPTH {
            return Err("json_schema: nested too deeply".into());
        }
        let o = match s {
            Value::Bool(true) => return Ok(self.any()),
            Value::Bool(false) => return Ok(self.push(Node::Never)),
            Value::Object(o) => o,
            _ => return Err("json_schema: a schema must be an object or a boolean".into()),
        };
        for k in o.keys() {
            if !KEYWORDS.contains(&k.as_str()) {
                return Err(format!("json_schema: unsupported schema keyword {k:?}"));
            }
        }
        if let Some(all) = o.get("allOf") {
            let arr = all
                .as_array()
                .filter(|a| !a.is_empty())
                .ok_or("json_schema: allOf must be a non-empty list")?;
            let mut base = o.clone();
            base.remove("allOf");
            let mut m = Value::Object(base);
            for x in arr {
                m = merge(&m, x)?;
            }
            return self.schema(&m, depth + 1);
        }
        for comb in ["anyOf", "oneOf"] {
            if let Some(alts) = o.get(comb) {
                let arr = alts
                    .as_array()
                    .filter(|a| !a.is_empty())
                    .ok_or(format!("json_schema: {comb} must be a non-empty list"))?;
                let mut base = o.clone();
                base.remove(comb);
                let base = Value::Object(base);
                let mut ids = Vec::new();
                for a in arr {
                    let m = merge(&base, a)?;
                    ids.push(self.schema(&m, depth + 1)?);
                }
                return Ok(self.union(ids));
            }
        }
        self.typed(o, depth)
    }

    fn typed(&mut self, o: &Map<String, Value>, depth: usize) -> Result<u32, String> {
        let mut types: Vec<String> = match o.get("type") {
            Some(t) => type_list(t)?,
            None => ALL_TYPES.iter().map(|s| s.to_string()).collect(),
        };
        if types.iter().any(|t| t == "number") {
            types.retain(|t| t != "integer");
        }
        let num = |k: &str| o.get(k).and_then(Value::as_f64);
        let (mut lo, mut lo_ex) = (f64::NEG_INFINITY, false);
        if let Some(m) = num("minimum") {
            lo = m;
        }
        if let Some(m) = num("exclusiveMinimum") {
            if m >= lo {
                lo = m;
                lo_ex = true;
            }
        }
        let (mut hi, mut hi_ex) = (f64::INFINITY, false);
        if let Some(m) = num("maximum") {
            hi = m;
        }
        if let Some(m) = num("exclusiveMaximum") {
            if m <= hi {
                hi = m;
                hi_ex = true;
            }
        }
        let smin = as_u32(o.get("minLength"), 0)?;
        let smax = as_u32(o.get("maxLength"), NONE)?;

        // enum / const: a literal set, filtered by the declared type(s) and the scalar bounds
        let mut lits: Option<Vec<Value>> = None;
        if let Some(e) = o.get("enum") {
            lits = Some(
                e.as_array()
                    .ok_or("json_schema: enum must be a list")?
                    .clone(),
            );
        }
        if let Some(c) = o.get("const") {
            lits = Some(match lits {
                Some(l) => l.into_iter().filter(|x| json_eq(x, c)).collect(),
                None => vec![c.clone()],
            });
        }
        if let Some(l) = lits {
            let mut out = Vec::new();
            for v in l {
                let t_ok = types.iter().any(|t| type_matches(t, &v));
                if !t_ok {
                    continue;
                }
                if let Some(n) = v.as_f64().filter(|_| v.is_number()) {
                    if n < lo || n > hi || (lo_ex && n <= lo) || (hi_ex && n >= hi) {
                        continue;
                    }
                }
                if let Some(s) = v.as_str() {
                    let n = s.chars().count() as u64;
                    if n < smin as u64 || (smax != NONE && n > smax as u64) {
                        continue;
                    }
                }
                let enc = serde_json::to_vec(&v).map_err(|e| e.to_string())?;
                if !out.contains(&enc) {
                    out.push(enc);
                }
            }
            return Ok(if out.is_empty() {
                self.push(Node::Never)
            } else {
                self.push(Node::Lit(out))
            });
        }

        let mut ids = Vec::new();
        for t in &types {
            let id = match t.as_str() {
                "string" => {
                    if smax != NONE && smin > smax {
                        continue;
                    }
                    self.push(Node::Str {
                        min: smin,
                        max: smax,
                    })
                }
                "number" | "integer" => {
                    if lo > hi {
                        continue;
                    }
                    self.push(Node::Num {
                        int: t == "integer",
                        lo,
                        lo_ex,
                        hi,
                        hi_ex,
                    })
                }
                "boolean" => self.push(Node::Lit(vec![b"true".to_vec(), b"false".to_vec()])),
                "null" => self.push(Node::Lit(vec![b"null".to_vec()])),
                "object" => self.object(o, depth)?,
                "array" => {
                    let items = match o.get("items") {
                        Some(i) => self.schema(i, depth + 1)?,
                        None => self.any(),
                    };
                    let min = as_u32(o.get("minItems"), 0)?;
                    let max = as_u32(o.get("maxItems"), NONE)?;
                    if max != NONE && min > max {
                        continue;
                    }
                    if min > 0 && matches!(self.nodes[items as usize], Node::Never) {
                        continue;
                    }
                    self.push(Node::Arr {
                        items,
                        min,
                        max,
                        unique: o.get("uniqueItems") == Some(&Value::Bool(true)),
                    })
                }
                _ => continue,
            };
            ids.push(id);
        }
        Ok(self.union(ids))
    }

    fn object(&mut self, o: &Map<String, Value>, depth: usize) -> Result<u32, String> {
        let additional = match o.get("additionalProperties") {
            None | Some(Value::Bool(true)) => self.any(),
            Some(Value::Bool(false)) => NONE,
            Some(s) => {
                let a = self.schema(s, depth + 1)?;
                if matches!(self.nodes[a as usize], Node::Never) {
                    NONE
                } else {
                    a
                }
            }
        };
        let mut props: Vec<(Vec<u8>, u32)> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        if let Some(p) = o.get("properties") {
            let p = p
                .as_object()
                .ok_or("json_schema: properties must be an object")?;
            for (k, v) in p {
                let id = self.schema(v, depth + 1)?;
                names.push(k.clone());
                props.push((key_bytes(k), id));
            }
        }
        let mut required = 0u64;
        if let Some(r) = o.get("required") {
            let r = r
                .as_array()
                .ok_or("json_schema: required must be a list of strings")?;
            for x in r {
                let k = x
                    .as_str()
                    .ok_or("json_schema: required must be a list of strings")?;
                let i = match names.iter().position(|n| n == k) {
                    Some(i) => i,
                    None => {
                        if additional == NONE {
                            return Ok(self.push(Node::Never));
                        }
                        names.push(k.to_string());
                        props.push((key_bytes(k), additional));
                        names.len() - 1
                    }
                };
                if matches!(self.nodes[props[i].1 as usize], Node::Never) {
                    return Ok(self.push(Node::Never));
                }
                required |= 1u64 << i.min(63);
            }
        }
        if props.len() > MAX_PROPS {
            return Err(format!(
                "json_schema: at most {MAX_PROPS} properties per object"
            ));
        }
        Ok(self.push(Node::Obj {
            props,
            required,
            additional,
        }))
    }
}

/// The JSON-encoded form of a key, without the quotes.
fn key_bytes(k: &str) -> Vec<u8> {
    let enc = serde_json::to_vec(&Value::String(k.to_string())).unwrap_or_default();
    enc[1..enc.len() - 1].to_vec()
}

// ---------------------------------------------------------------------------------------------------------------
// Byte-level recogniser
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
enum Frame {
    /// Expecting a value of `node` (after `ws` whitespace bytes).
    Val { node: u32, ws: u8 },
    /// Inside one of a literal set; `alts` index the node's literals still matching `pos` bytes.
    Lit { node: u32, alts: Vec<u16>, pos: u16 },
    /// Inside a string: `n` characters so far (saturated when unbounded), escape and UTF-8 progress.
    Str { node: u32, n: u32, esc: u8, utf: u8 },
    /// Inside a number: its text so far.
    Num { node: u32, text: Vec<u8> },
    /// Object: phase 0 after `{`, 1 in key, 2 after key, 3 after value, 4 after `,`.
    Obj {
        node: u32,
        ph: u8,
        ws: u8,
        seen: u64,
        key: Vec<u8>,
        kesc: u8,
        cur: u32,
        extra: Vec<Vec<u8>>,
    },
    /// Array: phase 0 after `[`, 1 in or after an item, 2 after `,`.
    Arr {
        node: u32,
        ph: u8,
        ws: u8,
        count: u32,
        cur: Vec<u8>,
        items: Vec<Vec<u8>>,
    },
}

type Cfg = Vec<Frame>;

/// Recogniser state: the live configurations, sorted and deduplicated.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct State(Vec<Cfg>);

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\r' | b'\t')
}

fn is_hex(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

/// Number text is a valid prefix of a JSON number without exponent (and without a fraction for integers).
fn num_prefix_ok(t: &[u8], int: bool) -> bool {
    let mut i = 0;
    if t.first() == Some(&b'-') {
        i = 1;
    }
    let digits = &t[i..];
    if digits.is_empty() {
        return true;
    }
    let dot = digits.iter().position(|&b| b == b'.');
    let (ip, fp) = match dot {
        Some(d) => (&digits[..d], Some(&digits[d + 1..])),
        None => (digits, None),
    };
    if ip.is_empty() || !ip.iter().all(u8::is_ascii_digit) {
        return false;
    }
    if ip.len() > 1 && ip[0] == b'0' {
        return false;
    }
    match fp {
        None => true,
        Some(f) => !int && f.iter().all(u8::is_ascii_digit),
    }
}

fn num_complete(t: &[u8]) -> bool {
    matches!(t.last(), Some(b) if b.is_ascii_digit())
}

fn in_bounds(v: f64, lo: f64, lo_ex: bool, hi: f64, hi_ex: bool) -> bool {
    !(v < lo || v > hi || (lo_ex && v <= lo) || (hi_ex && v >= hi))
}

/// Some completion of the number prefix `t` can land within the bounds.
fn num_feasible(t: &[u8], int: bool, lo: f64, lo_ex: bool, hi: f64, hi_ex: bool) -> bool {
    if lo == f64::NEG_INFINITY && hi == f64::INFINITY {
        return true;
    }
    let neg = t.first() == Some(&b'-');
    let body = if neg { &t[1..] } else { t };
    // magnitude intervals [a, b] reachable from this prefix
    let mut ivs: Vec<(f64, f64)> = Vec::new();
    let room = NUM_MAX.saturating_sub(t.len());
    if body.is_empty() {
        ivs.push((0.0, f64::INFINITY));
    } else if let Some(d) = body.iter().position(|&b| b == b'.') {
        let s = std::str::from_utf8(&body[..d]).unwrap_or("0");
        let f = &body[d + 1..];
        let base: f64 = format!("{s}.{}", std::str::from_utf8(f).unwrap_or(""))
            .trim_end_matches('.')
            .parse()
            .unwrap_or(0.0);
        let w = 10f64.powi(-(f.len() as i32));
        ivs.push((base, base + w));
    } else {
        let ip: f64 = std::str::from_utf8(body)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        if body == b"0" {
            ivs.push((0.0, if int { 0.0 } else { 1.0 }));
        } else {
            let mut p = 1.0f64;
            for _ in 0..=room.min(20) {
                let a = ip * p;
                let b = if int {
                    (ip + 1.0) * p - 1.0
                } else {
                    (ip + 1.0) * p
                };
                ivs.push((a, b));
                p *= 10.0;
            }
        }
    }
    ivs.iter().any(|&(a, b)| {
        let (x, y) = if neg { (-b, -a) } else { (a, b) };
        // [x, y] intersects the bound set (open ends treated loosely; the exact check runs at termination)
        let lo_ok = if lo_ex { y > lo } else { y >= lo };
        let hi_ok = if hi_ex { x < hi } else { x <= hi };
        lo_ok && hi_ok
    })
}

impl Grammar {
    fn lits(&self, node: u32) -> &[Vec<u8>] {
        match &self.nodes[node as usize] {
            Node::Lit(l) => l,
            _ => &[],
        }
    }

    /// The initial recogniser state.
    pub fn start(&self) -> State {
        State(vec![vec![Frame::Val {
            node: self.root,
            ws: 0,
        }]])
    }

    /// One configuration consumes byte `b`; successors (with the index of the frame that consumed it) go to `out`.
    fn step(&self, mut c: Cfg, b: u8, out: &mut Vec<(Cfg, usize)>) {
        loop {
            let ti = match c.len() {
                0 => return,
                n => n - 1,
            };
            let top = c.last_mut().unwrap();
            match top {
                Frame::Val { node, ws } => {
                    let node = *node;
                    if is_ws(b) {
                        if *ws < WS_MAX {
                            *ws += 1;
                            out.push((c, ti));
                        }
                        return;
                    }
                    let w = *ws;
                    match &self.nodes[node as usize] {
                        Node::Never => {}
                        Node::Union(alts) => {
                            for &a in alts {
                                let mut c2 = c.clone();
                                *c2.last_mut().unwrap() = Frame::Val { node: a, ws: w };
                                self.step(c2, b, out);
                            }
                        }
                        Node::Lit(ls) => {
                            let alts: Vec<u16> = (0..ls.len() as u16)
                                .filter(|&i| ls[i as usize][0] == b)
                                .collect();
                            if !alts.is_empty() {
                                *top = Frame::Lit { node, alts, pos: 1 };
                                self.lit_settle(c, ti, out);
                            }
                        }
                        Node::Str { .. } => {
                            if b == b'"' {
                                *top = Frame::Str {
                                    node,
                                    n: 0,
                                    esc: 0,
                                    utf: 0,
                                };
                                out.push((c, ti));
                            }
                        }
                        Node::Num {
                            int,
                            lo,
                            lo_ex,
                            hi,
                            hi_ex,
                        } => {
                            if b == b'-' || b.is_ascii_digit() {
                                let t = vec![b];
                                if num_feasible(&t, *int, *lo, *lo_ex, *hi, *hi_ex) {
                                    *top = Frame::Num { node, text: t };
                                    out.push((c, ti));
                                }
                            }
                        }
                        Node::Obj { .. } => {
                            if b == b'{' {
                                *top = Frame::Obj {
                                    node,
                                    ph: 0,
                                    ws: 0,
                                    seen: 0,
                                    key: Vec::new(),
                                    kesc: 0,
                                    cur: NONE,
                                    extra: Vec::new(),
                                };
                                out.push((c, ti));
                            }
                        }
                        Node::Arr { .. } => {
                            if b == b'[' {
                                *top = Frame::Arr {
                                    node,
                                    ph: 0,
                                    ws: 0,
                                    count: 0,
                                    cur: Vec::new(),
                                    items: Vec::new(),
                                };
                                out.push((c, ti));
                            }
                        }
                    }
                    return;
                }
                Frame::Lit { node, alts, pos } => {
                    let ls = self.lits(*node);
                    let p = *pos as usize;
                    let complete = alts.iter().any(|&i| ls[i as usize].len() == p);
                    let ext: Vec<u16> = alts
                        .iter()
                        .copied()
                        .filter(|&i| ls[i as usize].len() > p && ls[i as usize][p] == b)
                        .collect();
                    if complete {
                        let mut c2 = c.clone();
                        c2.pop();
                        self.step(c2, b, out);
                    }
                    if !ext.is_empty() {
                        if let Some(Frame::Lit { alts, pos, .. }) = c.last_mut() {
                            *alts = ext;
                            *pos += 1;
                        }
                        self.lit_settle(c, ti, out);
                    }
                    return;
                }
                Frame::Str { node, n, esc, utf } => {
                    let (min, max) = match self.nodes[*node as usize] {
                        Node::Str { min, max } => (min, max),
                        _ => (0, NONE),
                    };
                    let bump = |n: &mut u32| {
                        if max == NONE {
                            *n = (*n + 1).min(min);
                        } else {
                            *n += 1;
                        }
                    };
                    let room = max == NONE || *n < max;
                    match *esc {
                        1 => match b {
                            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                                *esc = 0;
                                out.push((c, ti));
                            }
                            b'u' => {
                                *esc = 2;
                                out.push((c, ti));
                            }
                            _ => {}
                        },
                        2 => {
                            if is_hex(b) {
                                // \uD800-\uDFFF (surrogates) are not produced
                                *esc = if b == b'd' || b == b'D' { 6 } else { 3 };
                                out.push((c, ti));
                            }
                        }
                        6 => {
                            if (b'0'..=b'7').contains(&b) {
                                *esc = 4;
                                out.push((c, ti));
                            }
                        }
                        3 | 4 => {
                            if is_hex(b) {
                                *esc += 1;
                                out.push((c, ti));
                            }
                        }
                        5 => {
                            if is_hex(b) {
                                *esc = 0;
                                out.push((c, ti));
                            }
                        }
                        _ if *utf > 0 => {
                            if (0x80..=0xBF).contains(&b) {
                                *utf -= 1;
                                out.push((c, ti));
                            }
                        }
                        _ => match b {
                            b'"' => {
                                if *n >= min {
                                    c.pop();
                                    out.push((c, ti));
                                }
                            }
                            b'\\' if room => {
                                bump(n);
                                *esc = 1;
                                out.push((c, ti));
                            }
                            0x20..=0x7F if b != b'\\' && room => {
                                bump(n);
                                out.push((c, ti));
                            }
                            0xC2..=0xF4 if room => {
                                bump(n);
                                *utf = match b {
                                    0xC2..=0xDF => 1,
                                    0xE0..=0xEF => 2,
                                    _ => 3,
                                };
                                out.push((c, ti));
                            }
                            _ => {}
                        },
                    }
                    return;
                }
                Frame::Num { node, text } => {
                    let Node::Num {
                        int,
                        lo,
                        lo_ex,
                        hi,
                        hi_ex,
                    } = self.nodes[*node as usize]
                    else {
                        return;
                    };
                    if b.is_ascii_digit() || b == b'.' {
                        let mut t = text.clone();
                        t.push(b);
                        if t.len() <= NUM_MAX
                            && num_prefix_ok(&t, int)
                            && num_feasible(&t, int, lo, lo_ex, hi, hi_ex)
                        {
                            *text = t;
                            out.push((c, ti));
                        }
                        return;
                    }
                    if !num_complete(text) {
                        return;
                    }
                    let v: f64 = std::str::from_utf8(text)
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(f64::NAN);
                    if !in_bounds(v, lo, lo_ex, hi, hi_ex) {
                        return;
                    }
                    c.pop();
                    continue; // the parent takes the delimiter
                }
                Frame::Obj {
                    node,
                    ph,
                    ws,
                    seen,
                    key,
                    kesc,
                    cur,
                    extra,
                } => {
                    let Node::Obj {
                        props,
                        required,
                        additional,
                    } = &self.nodes[*node as usize]
                    else {
                        return;
                    };
                    let unseen = props
                        .iter()
                        .enumerate()
                        .any(|(i, _)| *seen & (1u64 << i) == 0);
                    let key_possible = *additional != NONE || unseen;
                    match *ph {
                        0 | 3 | 4 => {
                            if is_ws(b) {
                                if *ws < WS_MAX {
                                    *ws += 1;
                                    out.push((c, ti));
                                }
                                return;
                            }
                            match (b, *ph) {
                                (b'"', 0 | 4) if key_possible => {
                                    *ph = 1;
                                    *ws = 0;
                                    key.clear();
                                    *kesc = 0;
                                    out.push((c, ti));
                                }
                                (b'}', 0 | 3) if *seen & required == *required => {
                                    c.pop();
                                    out.push((c, ti));
                                }
                                (b',', 3) if key_possible => {
                                    *ph = 4;
                                    *ws = 0;
                                    out.push((c, ti));
                                }
                                _ => {}
                            }
                        }
                        1 => {
                            let open = *additional != NONE;
                            if *kesc > 0 {
                                // inside an escape of a free-form key
                                let ok = match *kesc {
                                    1 => matches!(
                                        b,
                                        b'"' | b'\\'
                                            | b'/'
                                            | b'b'
                                            | b'f'
                                            | b'n'
                                            | b'r'
                                            | b't'
                                            | b'u'
                                    ),
                                    _ => is_hex(b),
                                };
                                if !ok {
                                    return;
                                }
                                *kesc = match (*kesc, b) {
                                    (1, b'u') => 2,
                                    (1, _) => 0,
                                    (5, _) => 0,
                                    (k, _) => k + 1,
                                };
                                key.push(b);
                                if !open && !props.iter().any(|(p, _)| p.starts_with(key)) {
                                    return;
                                }
                                out.push((c, ti));
                                return;
                            }
                            if b == b'"' {
                                let hit = props.iter().position(|(p, _)| p == key);
                                match hit {
                                    Some(i) => {
                                        if *seen & (1u64 << i) != 0 {
                                            return;
                                        }
                                        *cur = i as u32;
                                    }
                                    None => {
                                        if !open || extra.contains(key) {
                                            return;
                                        }
                                        extra.push(key.clone());
                                        *cur = NONE;
                                    }
                                }
                                *ph = 2;
                                *ws = 0;
                                out.push((c, ti));
                                return;
                            }
                            if b < 0x20 {
                                return;
                            }
                            key.push(b);
                            if b == b'\\' {
                                *kesc = 1;
                            }
                            if !open
                                && !props.iter().enumerate().any(|(i, (p, _))| {
                                    *seen & (1u64 << i) == 0 && p.starts_with(key)
                                })
                            {
                                return;
                            }
                            out.push((c, ti));
                        }
                        2 => {
                            if is_ws(b) {
                                if *ws < WS_MAX {
                                    *ws += 1;
                                    out.push((c, ti));
                                }
                                return;
                            }
                            if b == b':' {
                                let child = if *cur == NONE {
                                    *additional
                                } else {
                                    *seen |= 1u64 << *cur;
                                    props[*cur as usize].1
                                };
                                *ph = 3;
                                *ws = 0;
                                key.clear();
                                c.push(Frame::Val { node: child, ws: 0 });
                                out.push((c, ti));
                            }
                        }
                        _ => {}
                    }
                    return;
                }
                Frame::Arr {
                    node,
                    ph,
                    ws,
                    count,
                    cur,
                    items,
                } => {
                    let Node::Arr {
                        items: inode,
                        min,
                        max,
                        unique,
                    } = self.nodes[*node as usize]
                    else {
                        return;
                    };
                    if unique && *ph == 1 && !cur.is_empty() {
                        let it = trim_ws(cur).to_vec();
                        if items.contains(&it) {
                            return;
                        }
                        items.push(it);
                        cur.clear();
                    }
                    if is_ws(b) {
                        if *ws < WS_MAX {
                            *ws += 1;
                            out.push((c, ti));
                        }
                        return;
                    }
                    match (*ph, b) {
                        (0, b']') | (1, b']') => {
                            if *count >= min {
                                c.pop();
                                out.push((c, ti));
                            }
                        }
                        (1, b',') => {
                            if max == NONE || *count < max {
                                *ph = 2;
                                *ws = 0;
                                out.push((c, ti));
                            }
                        }
                        (0 | 2, _) => {
                            if max != NONE && *count >= max {
                                return;
                            }
                            *count += 1;
                            *ph = 1;
                            *ws = 0;
                            c.push(Frame::Val { node: inode, ws: 0 });
                            continue; // the new value frame takes this byte
                        }
                        _ => {}
                    }
                    return;
                }
            }
        }
    }

    /// After a literal frame advanced: a literal that is fully matched and cannot be extended and does not end in a
    /// digit (strings, true/false/null, objects) closes right away; numbers wait for their delimiter.
    fn lit_settle(&self, mut c: Cfg, ti: usize, out: &mut Vec<(Cfg, usize)>) {
        if let Some(Frame::Lit { node, alts, pos }) = c.last() {
            let ls = self.lits(*node);
            let p = *pos as usize;
            let all_done = alts.iter().all(|&i| ls[i as usize].len() == p);
            let last_digit = alts
                .iter()
                .any(|&i| ls[i as usize].last().is_some_and(u8::is_ascii_digit));
            if all_done && !last_digit {
                c.pop();
            }
        }
        out.push((c, ti));
    }

    /// Post-step bookkeeping of one successor: raw item capture for `uniqueItems` arrays below the consumer, and the
    /// duplicate check when an item just closed. Returns false when the configuration dies.
    fn post(&self, c: &mut Cfg, consumer: usize, b: u8) -> bool {
        let lim = consumer.min(c.len());
        for f in c[..lim].iter_mut() {
            if let Frame::Arr {
                node, ph: 1, cur, ..
            } = f
            {
                if matches!(self.nodes[*node as usize], Node::Arr { unique: true, .. }) {
                    cur.push(b);
                }
            }
        }
        let top_i = c.len().saturating_sub(1);
        if let Some(Frame::Arr {
            node,
            ph: 1,
            cur,
            items,
            ..
        }) = c.last_mut()
        {
            if consumer >= top_i
                && !cur.is_empty()
                && matches!(self.nodes[*node as usize], Node::Arr { unique: true, .. })
            {
                let it = trim_ws(cur).to_vec();
                if items.contains(&it) {
                    return false;
                }
                items.push(it);
                cur.clear();
            }
        }
        true
    }

    /// The state after byte `b` (empty = dead).
    pub fn feed(&self, s: &State, b: u8) -> State {
        let mut succ = Vec::new();
        for c in &s.0 {
            self.step(c.clone(), b, &mut succ);
        }
        let mut out: Vec<Cfg> = Vec::with_capacity(succ.len());
        for (mut c, ci) in succ {
            if self.post(&mut c, ci, b) {
                out.push(c);
            }
        }
        out.sort();
        out.dedup();
        out.truncate(MAX_CFGS);
        State(out)
    }

    /// The state after `bytes` (empty = dead).
    pub fn feed_bytes(&self, s: &State, bytes: &[u8]) -> State {
        let mut st = s.clone();
        for &b in bytes {
            st = self.feed(&st, b);
            if st.0.is_empty() {
                break;
            }
        }
        st
    }

    /// Some configuration has a complete top-level value (an end token is allowed now).
    pub fn can_finish(&self, s: &State) -> bool {
        s.0.iter().any(|c| match c.as_slice() {
            [] => true,
            [Frame::Lit { node, alts, pos }] => {
                let ls = self.lits(*node);
                alts.iter().any(|&i| ls[i as usize].len() == *pos as usize)
            }
            [Frame::Num { node, text }] => match self.nodes[*node as usize] {
                Node::Num {
                    lo,
                    lo_ex,
                    hi,
                    hi_ex,
                    ..
                } => {
                    num_complete(text)
                        && std::str::from_utf8(text)
                            .ok()
                            .and_then(|s| s.parse::<f64>().ok())
                            .is_some_and(|v| in_bounds(v, lo, lo_ex, hi, hi_ex))
                }
                _ => false,
            },
            _ => false,
        })
    }

    /// Every configuration is complete with nothing left to read: the only continuation is an end token.
    pub fn finished(&self, s: &State) -> bool {
        !s.0.is_empty() && s.0.iter().all(Vec::is_empty)
    }

    /// Characters still allowed when every configuration sits in the body of a free string (a plain token, one
    /// without quote, backslash or control bytes and with complete UTF-8, then needs only that many characters).
    fn absorbing_room(&self, s: &State) -> Option<u32> {
        let mut room = 0u32;
        for c in &s.0 {
            let r = match c.last()? {
                Frame::Str {
                    node,
                    n,
                    esc: 0,
                    utf: 0,
                } => match self.nodes[*node as usize] {
                    Node::Str { max, .. } if max == NONE => NONE,
                    Node::Str { max, .. } => max.saturating_sub(*n),
                    _ => return None,
                },
                Frame::Obj {
                    node,
                    ph: 1,
                    kesc: 0,
                    ..
                } => match &self.nodes[*node as usize] {
                    Node::Obj { additional, .. } if *additional != NONE => NONE,
                    _ => return None,
                },
                _ => return None,
            };
            room = room.max(r);
        }
        Some(room)
    }
}

fn trim_ws(v: &[u8]) -> &[u8] {
    let s = v.iter().position(|&b| !is_ws(b)).unwrap_or(v.len());
    let e = v.iter().rposition(|&b| !is_ws(b)).map_or(s, |e| e + 1);
    &v[s..e]
}

// ---------------------------------------------------------------------------------------------------------------
// Vocabulary and token masks
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct TNode {
    byte: u8,
    child: (u32, u32),
    term: (u32, u32),
}

#[derive(Debug, Default)]
struct Trie {
    nodes: Vec<TNode>,
    term: Vec<u32>,
}

impl Trie {
    fn build(mut items: Vec<(Vec<u8>, u32)>) -> Trie {
        items.sort();
        let mut t = Trie {
            nodes: vec![TNode {
                byte: 0,
                child: (0, 0),
                term: (0, 0),
            }],
            term: Vec::new(),
        };
        let mut q = VecDeque::new();
        q.push_back((0usize, 0usize, items.len(), 0usize));
        while let Some((ni, lo, hi, d)) = q.pop_front() {
            let mut i = lo;
            let ts = t.term.len() as u32;
            while i < hi && items[i].0.len() == d {
                t.term.push(items[i].1);
                i += 1;
            }
            t.nodes[ni].term = (ts, t.term.len() as u32);
            let cs = t.nodes.len() as u32;
            while i < hi {
                let b = items[i].0[d];
                let mut j = i;
                while j < hi && items[j].0[d] == b {
                    j += 1;
                }
                let ci = t.nodes.len();
                t.nodes.push(TNode {
                    byte: b,
                    child: (0, 0),
                    term: (0, 0),
                });
                q.push_back((ci, i, j, d + 1));
                i = j;
            }
            t.nodes[ni].child = (cs, t.nodes.len() as u32);
        }
        t
    }
}

/// The served vocabulary as byte strings, ready for mask computation.
#[derive(Debug)]
pub struct Vocab {
    n: usize,
    bytes: Vec<Vec<u8>>,
    text: Vec<bool>,
    eos: Vec<u32>,
    /// plain tokens (allowed anywhere in a free string body), sorted by character count
    plain: Vec<(u32, u32)>,
    all: Trie,
    nonplain: Trie,
}

impl Vocab {
    /// `n` = logits width; `bytes(id)` = the token's bytes for ordinary text tokens, `None` for special/control
    /// tokens (never produced inside JSON); `eos` = the end tokens allowed once the value is complete (they are
    /// never treated as text).
    pub fn new(n: usize, bytes: impl Fn(u32) -> Option<Vec<u8>>, eos: &[u32]) -> Vocab {
        let mut all = Vec::new();
        let mut nonplain = Vec::new();
        let mut plain = Vec::new();
        let mut bs = vec![Vec::new(); n];
        let mut text = vec![false; n];
        for id in 0..n as u32 {
            if eos.contains(&id) {
                continue;
            }
            let Some(b) = bytes(id).filter(|b| !b.is_empty()) else {
                continue;
            };
            text[id as usize] = true;
            let is_plain = b.iter().all(|&x| x >= 0x20 && x != b'"' && x != b'\\');
            match std::str::from_utf8(&b) {
                Ok(s) if is_plain => plain.push((s.chars().count() as u32, id)),
                _ => nonplain.push((b.clone(), id)),
            }
            all.push((b.clone(), id));
            bs[id as usize] = b;
        }
        plain.sort();
        Vocab {
            n,
            bytes: bs,
            text,
            eos: eos.iter().copied().filter(|&e| (e as usize) < n).collect(),
            plain: plain.into_iter().map(|(c, id)| (id, c)).collect(),
            all: Trie::build(all),
            nonplain: Trie::build(nonplain),
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    fn walk(&self, g: &Grammar, t: &Trie, ni: usize, s: &State, mask: &mut [u64]) {
        let (a, b) = t.nodes[ni].child;
        for ci in a..b {
            let node = t.nodes[ci as usize];
            let s2 = g.feed(s, node.byte);
            if s2.0.is_empty() {
                continue;
            }
            for &id in &t.term[node.term.0 as usize..node.term.1 as usize] {
                mask[(id >> 6) as usize] |= 1u64 << (id & 63);
            }
            if node.child.0 < node.child.1 {
                self.walk(g, t, ci as usize, &s2, mask);
            }
        }
    }

    fn compute(&self, g: &Grammar, s: &State) -> Vec<u64> {
        let mut mask = vec![0u64; self.n.div_ceil(64)];
        if g.finished(s) {
            // nothing but an end token
        } else if let Some(room) = g.absorbing_room(s) {
            for &(id, nc) in &self.plain {
                if nc > room {
                    break;
                }
                mask[(id >> 6) as usize] |= 1u64 << (id & 63);
            }
            self.walk(g, &self.nonplain, 0, s, &mut mask);
        } else {
            self.walk(g, &self.all, 0, s, &mut mask);
        }
        if g.can_finish(s) {
            for &e in &self.eos {
                mask[(e >> 6) as usize] |= 1u64 << (e & 63);
            }
        }
        mask
    }
}

/// Per-request matcher: the recogniser state of one constrained generation.
pub struct Matcher {
    g: Arc<Grammar>,
    v: Arc<Vocab>,
    state: State,
    /// masks computed / served from the memo, and the time spent computing (ms)
    stats: std::cell::Cell<(u32, u32, f64)>,
}

impl Matcher {
    pub fn new(g: Arc<Grammar>, v: Arc<Vocab>) -> Self {
        let state = g.start();
        Self {
            g,
            v,
            state,
            stats: std::cell::Cell::new((0, 0, 0.0)),
        }
    }

    /// The allowed-token bitmask (bit `id`) for the next token; memoised per state in the grammar.
    pub fn mask(&self) -> Arc<Vec<u64>> {
        let (c, h, ms) = self.stats.get();
        if let Some(m) = self.g.memo.lock().unwrap().get(&self.state) {
            self.stats.set((c, h + 1, ms));
            return m.clone();
        }
        let t0 = std::time::Instant::now();
        let m = Arc::new(self.v.compute(&self.g, &self.state));
        self.stats
            .set((c + 1, h, ms + t0.elapsed().as_secs_f64() * 1000.0));
        let mut memo = self.g.memo.lock().unwrap();
        if memo.len() >= MEMO_CAP {
            memo.clear();
        }
        memo.insert(self.state.clone(), m.clone());
        m
    }

    /// Advances over a sampled token. False = the token is not allowed here (end tokens included).
    pub fn advance(&mut self, id: u32) -> bool {
        let i = id as usize;
        if i >= self.v.n || !self.v.text[i] {
            return false;
        }
        let s = self.g.feed_bytes(&self.state, &self.v.bytes[i]);
        if s.0.is_empty() {
            return false;
        }
        self.state = s;
        true
    }

    /// The value is complete and nothing but an end token may follow.
    pub fn finished(&self) -> bool {
        self.g.finished(&self.state)
    }

    pub fn can_finish(&self) -> bool {
        self.g.can_finish(&self.state)
    }

    /// (masks computed, masks reused from the memo, ms spent computing masks).
    pub fn stats(&self) -> (u32, u32, f64) {
        self.stats.get()
    }
}

/// Bit `id` of a mask.
#[inline]
pub fn mask_has(mask: &[u64], id: usize) -> bool {
    mask.get(id >> 6)
        .is_some_and(|w| w & (1u64 << (id & 63)) != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn accepts(schema: &Value, text: &str) -> bool {
        let g = compile_value(schema).unwrap();
        let s = g.feed_bytes(&g.start(), text.as_bytes());
        !s.0.is_empty() && g.can_finish(&s)
    }

    fn prefix_ok(schema: &Value, text: &str) -> bool {
        let g = compile_value(schema).unwrap();
        !g.feed_bytes(&g.start(), text.as_bytes()).0.is_empty()
    }

    #[test]
    fn objects_required_and_closed() {
        let s = json!({"type":"object","additionalProperties":false,
            "properties":{"city":{"type":"string"},"day":{"enum":["today","tomorrow"],"type":"string"}},
            "required":["city","day"]});
        assert!(accepts(&s, r#"{"city": "Lyon", "day": "tomorrow"}"#));
        assert!(accepts(&s, "{\n  \"day\":\"today\",\n  \"city\":\"x\"\n}"));
        assert!(!accepts(&s, r#"{"city": "Lyon"}"#));
        assert!(!prefix_ok(&s, r#"{"city": "Lyon", "note"#));
        assert!(!prefix_ok(&s, r#"{"city": "Lyon", "city""#));
        assert!(!prefix_ok(&s, r#"{"city": "Lyon", "day": "yes"#));
        assert!(!prefix_ok(&s, "```"));
        assert!(!accepts(&s, r#"{"city": "Lyon", "day": "today"} "#));
    }

    #[test]
    fn numbers_and_bounds() {
        let s = json!({"type":"integer","minimum":0,"maximum":100});
        assert!(accepts(&s, "0"));
        assert!(accepts(&s, "100"));
        assert!(!accepts(&s, "101"));
        assert!(!prefix_ok(&s, "101"));
        assert!(!prefix_ok(&s, "-1"));
        assert!(!prefix_ok(&s, "1."));
        assert!(!prefix_ok(&s, "01"));
        let s = json!({"type":"number","exclusiveMinimum":0});
        assert!(!accepts(&s, "0"));
        assert!(accepts(&s, "0.5"));
        assert!(!prefix_ok(&s, "-"));
        let s = json!({"type":"number","minimum":0,"maximum":1});
        assert!(accepts(&s, "0.93"));
        assert!(accepts(&s, "1"));
        assert!(!prefix_ok(&s, "2"));
        assert!(!prefix_ok(&s, "1.5"));
        let s = json!({"type":"integer","minimum":1,"maximum":4294967295u64});
        assert!(accepts(&s, "65001"));
    }

    #[test]
    fn strings_lengths_and_escapes() {
        let s = json!({"type":"string","minLength":2,"maxLength":2});
        assert!(accepts(&s, "\"FR\""));
        assert!(!prefix_ok(&s, "\"F\""));
        assert!(!prefix_ok(&s, "\"FRA"));
        assert!(accepts(&s, "\"é\\n\""));
        let s = json!({"type":"string"});
        assert!(accepts(&s, "\"a\\u00e9\\\"\""));
        assert!(!prefix_ok(&s, "\"\\ud8"));
        assert!(!prefix_ok(&s, "\"a\nb"));
        assert!(!prefix_ok(&s, "\"\\x"));
    }

    #[test]
    fn unions_null_and_oneof() {
        let s = json!({"type":"object","additionalProperties":false,
            "properties":{"tool":{"enum":["search_flights","book_hotel",null],"type":["string","null"]},
                          "call_tool":{"type":"boolean"}},"required":["call_tool","tool"]});
        assert!(accepts(&s, r#"{"call_tool": false, "tool": null}"#));
        assert!(accepts(&s, r#"{"call_tool": true, "tool": "book_hotel"}"#));
        assert!(!prefix_ok(&s, r#"{"call_tool": true, "tool": "x"#));
        let s = json!({"oneOf":[
            {"type":"object","additionalProperties":false,"properties":{"type":{"const":"card"},"last4":{"type":"string","minLength":4,"maxLength":4}},"required":["type","last4"]},
            {"type":"object","additionalProperties":false,"properties":{"type":{"const":"sepa"},"iban":{"type":"string"}},"required":["type","iban"]}]});
        assert!(accepts(&s, r#"{"type":"card","last4":"1234"}"#));
        assert!(accepts(&s, r#"{"type":"sepa","iban":"DE00"}"#));
        assert!(!prefix_ok(&s, r#"{"type":"card","iban""#));
    }

    #[test]
    fn arrays_counts_and_unique() {
        let s = json!({"type":"array","items":{"type":"string","minLength":1},"minItems":3,"maxItems":3,"uniqueItems":true});
        assert!(accepts(&s, r#"["a", "b", "c"]"#));
        assert!(!prefix_ok(&s, r#"["a", "a""#));
        assert!(!prefix_ok(&s, r#"["a", "b"]"#));
        assert!(!prefix_ok(&s, r#"["a", "b", "c","#));
        let s = json!({"type":"array","items":{"type":"integer"},"uniqueItems":true});
        assert!(accepts(&s, "[1, 12, 2]"));
        assert!(!prefix_ok(&s, "[1, 1,"));
        assert!(prefix_ok(&s, "[1, 1"));
        let s = json!({"type":"array","items":{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2},"minItems":2,"maxItems":2});
        assert!(accepts(&s, "[[1, 2], [3, 4]]"));
        assert!(!prefix_ok(&s, "[[1]"));
    }

    #[test]
    fn open_objects_and_any() {
        let s = json!({"type":"object","properties":{"headers":{"type":"object","additionalProperties":{"type":"string"}}},"required":["headers"],"additionalProperties":false});
        assert!(accepts(&s, r#"{"headers": {"Accept": "x", "X-Y": "z"}}"#));
        assert!(!prefix_ok(&s, r#"{"headers": {"Accept": 1"#));
        assert!(!prefix_ok(&s, r#"{"headers": {"A": "x", "A""#));
        assert!(accepts(
            &json!(true),
            r#"{"a": [1, {"b": null}], "c": "d"}"#
        ));
        assert!(accepts(&json!({}), "[true, -0.5, \"x\"]"));
        assert!(accepts(&json!({"type":"object"}), "{}"));
    }

    #[test]
    fn rejects_unsupported() {
        assert!(prepare(&json!({"type":"string","pattern":"^a"})).is_err());
        assert!(prepare(&json!({"$ref":"#/x"})).is_err());
        assert!(prepare(&json!(false)).is_err());
        let p = prepare(&json!({"title":"T","type":"object","properties":{"a":{"description":"d","type":"integer"}}})).unwrap();
        assert_eq!(
            p,
            r#"{"type":"object","properties":{"a":{"type":"integer"}}}"#
        );
    }

    #[test]
    fn allof_merges() {
        let s = json!({"allOf":[{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]},
                                {"properties":{"a":{"minimum":5}}}]});
        assert!(accepts(&s, r#"{"a": 7}"#));
        assert!(!prefix_ok(&s, r#"{"a": 3}"#));
    }

    fn toy_vocab() -> (Vocab, Vec<&'static str>) {
        let toks = vec![
            "{", "}", "\"", "\":", " ", "\n", "city", "Lyon", "\"Lyon\"", ",", ":", "day", "today",
            "tomorrow", "\"}", "```", "ab\"", "é", "1", "12", "x", "\"c", "ity", "<eos>",
        ];
        let n = toks.len();
        let t2 = toks.clone();
        let v = Vocab::new(
            n,
            move |id| {
                let s = t2[id as usize];
                (s != "<eos>").then(|| s.as_bytes().to_vec())
            },
            &[(n - 1) as u32],
        );
        (v, toks)
    }

    #[test]
    fn masks_match_bytewise_feeding() {
        let schema = json!({"type":"object","additionalProperties":false,
            "properties":{"city":{"type":"string"},"day":{"enum":["today","tomorrow"]}},"required":["city","day"]});
        let g = Arc::new(compile_value(&schema).unwrap());
        let (v, toks) = toy_vocab();
        let v = Arc::new(v);
        let mut m = Matcher::new(g.clone(), v.clone());
        // follow a path, checking at each state that the mask equals the brute-force answer
        let path = [
            "{", "\"c", "ity", "\":", " ", "\"", "Lyon", "ab\"", ",", "\"", "day", "\":", "\"",
            "today", "\"}",
        ];
        for p in path {
            let mask = m.mask();
            for (id, t) in toks.iter().enumerate() {
                let brute = if *t == "<eos>" {
                    g.can_finish(&m.state)
                } else {
                    !g.feed_bytes(&m.state, t.as_bytes()).0.is_empty()
                };
                assert_eq!(mask_has(&mask, id), brute, "token {t:?} before {p:?}");
            }
            let id = toks.iter().position(|t| *t == p).unwrap() as u32;
            assert!(m.advance(id), "{p}");
        }
        assert!(m.finished());
        let mask = m.mask();
        assert!(mask_has(&mask, toks.len() - 1));
        assert_eq!(mask.iter().map(|w| w.count_ones()).sum::<u32>(), 1);
    }
}
