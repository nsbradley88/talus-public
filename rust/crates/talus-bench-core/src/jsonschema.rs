//! A native JSON Schema validator for the subset the bench's tasks use (draft 2020-12 semantics for these
//! keywords). A schema that uses any other keyword is rejected up front (`check_schema`), so a case can never pass
//! because its schema was only half understood.
//!
//! Supported: type (string or list; "integer" = number with no fraction), enum, const, properties, required,
//! additionalProperties (bool or schema), items (schema), minItems, maxItems, uniqueItems, minimum, maximum,
//! exclusiveMinimum, exclusiveMaximum, minLength, maxLength (in Unicode scalar values), anyOf, allOf, oneOf,
//! plus the annotations title, description, $schema, $comment.

use serde_json::Value;

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

/// One validation failure, with a JSON-pointer-like path.
#[derive(Clone, Debug, PartialEq)]
pub struct Violation {
    pub path: String,
    pub message: String,
    /// The violation is a property not allowed by `additionalProperties: false`.
    pub extra_field: bool,
}

/// Reject schemas with keywords this validator does not implement.
pub fn check_schema(schema: &Value) -> Result<(), String> {
    fn walk(s: &Value, path: &str) -> Result<(), String> {
        let Some(obj) = s.as_object() else {
            return match s {
                Value::Bool(_) => Ok(()),
                _ => Err(format!("{path}: a schema must be an object or a boolean")),
            };
        };
        for (k, v) in obj {
            if !KEYWORDS.contains(&k.as_str()) {
                return Err(format!("{path}: unsupported schema keyword {k:?}"));
            }
            match k.as_str() {
                "properties" => {
                    for (pk, pv) in v
                        .as_object()
                        .ok_or(format!("{path}/properties: not an object"))?
                    {
                        walk(pv, &format!("{path}/properties/{pk}"))?;
                    }
                }
                "items" | "additionalProperties" => walk(v, &format!("{path}/{k}"))?,
                "anyOf" | "allOf" | "oneOf" => {
                    let arr = v.as_array().ok_or(format!("{path}/{k}: not an array"))?;
                    if arr.is_empty() {
                        return Err(format!("{path}/{k}: empty"));
                    }
                    for (i, x) in arr.iter().enumerate() {
                        walk(x, &format!("{path}/{k}/{i}"))?;
                    }
                }
                "type" => {
                    let ok = |t: &Value| {
                        matches!(
                            t.as_str(),
                            Some(
                                "object"
                                    | "array"
                                    | "string"
                                    | "number"
                                    | "integer"
                                    | "boolean"
                                    | "null"
                            )
                        )
                    };
                    let valid = match v {
                        Value::Array(a) => !a.is_empty() && a.iter().all(ok),
                        t => ok(t),
                    };
                    if !valid {
                        return Err(format!("{path}/type: bad type {v}"));
                    }
                }
                "required" if !v.as_array().is_some_and(|a| a.iter().all(Value::is_string)) => {
                    return Err(format!("{path}/required: must be a list of strings"));
                }
                _ => {}
            }
        }
        Ok(())
    }
    walk(schema, "#")
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

/// JSON equality with numbers compared by value (1 == 1.0).
pub fn json_eq(a: &Value, b: &Value) -> bool {
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

/// Validate `v` against `schema` (which should have passed `check_schema`). Empty = valid.
pub fn validate(schema: &Value, v: &Value) -> Vec<Violation> {
    let mut out = Vec::new();
    walk(schema, v, "", &mut out);
    out
}

fn push(out: &mut Vec<Violation>, path: &str, message: String) {
    out.push(Violation {
        path: if path.is_empty() {
            "/".into()
        } else {
            path.into()
        },
        message,
        extra_field: false,
    });
}

fn walk(s: &Value, v: &Value, path: &str, out: &mut Vec<Violation>) {
    let obj = match s {
        Value::Bool(true) => return,
        Value::Bool(false) => return push(out, path, "no value is allowed here".into()),
        Value::Object(o) => o,
        _ => return push(out, path, "invalid schema".into()),
    };
    if let Some(t) = obj.get("type") {
        let ok = match t {
            Value::String(t) => type_matches(t, v),
            Value::Array(ts) => ts
                .iter()
                .filter_map(Value::as_str)
                .any(|t| type_matches(t, v)),
            _ => false,
        };
        if !ok {
            push(out, path, format!("expected type {t}, got {}", kind(v)));
            return; // other keywords would only add noise
        }
    }
    if let Some(Value::Array(opts)) = obj.get("enum") {
        if !opts.iter().any(|o| json_eq(o, v)) {
            push(
                out,
                path,
                format!("{v} is not one of {}", Value::Array(opts.clone())),
            );
        }
    }
    if let Some(c) = obj.get("const") {
        if !json_eq(c, v) {
            push(out, path, format!("expected the constant {c}"));
        }
    }
    if let Some(n) = v.as_f64().filter(|_| v.is_number()) {
        let num = |k: &str| obj.get(k).and_then(Value::as_f64);
        if let Some(m) = num("minimum").filter(|m| n < *m) {
            push(out, path, format!("{n} < minimum {m}"));
        }
        if let Some(m) = num("maximum").filter(|m| n > *m) {
            push(out, path, format!("{n} > maximum {m}"));
        }
        if let Some(m) = num("exclusiveMinimum").filter(|m| n <= *m) {
            push(out, path, format!("{n} <= exclusiveMinimum {m}"));
        }
        if let Some(m) = num("exclusiveMaximum").filter(|m| n >= *m) {
            push(out, path, format!("{n} >= exclusiveMaximum {m}"));
        }
    }
    if let Some(sv) = v.as_str() {
        let len = sv.chars().count() as u64;
        if let Some(m) = obj
            .get("minLength")
            .and_then(Value::as_u64)
            .filter(|m| len < *m)
        {
            push(out, path, format!("string shorter than {m}"));
        }
        if let Some(m) = obj
            .get("maxLength")
            .and_then(Value::as_u64)
            .filter(|m| len > *m)
        {
            push(out, path, format!("string longer than {m}"));
        }
    }
    if let Some(arr) = v.as_array() {
        let n = arr.len() as u64;
        if let Some(m) = obj
            .get("minItems")
            .and_then(Value::as_u64)
            .filter(|m| n < *m)
        {
            push(out, path, format!("fewer than {m} items"));
        }
        if let Some(m) = obj
            .get("maxItems")
            .and_then(Value::as_u64)
            .filter(|m| n > *m)
        {
            push(out, path, format!("more than {m} items"));
        }
        if obj.get("uniqueItems") == Some(&Value::Bool(true)) {
            for i in 0..arr.len() {
                if arr[..i].iter().any(|p| json_eq(p, &arr[i])) {
                    push(out, &format!("{path}/{i}"), "duplicate item".into());
                }
            }
        }
        if let Some(items) = obj.get("items") {
            for (i, x) in arr.iter().enumerate() {
                walk(items, x, &format!("{path}/{i}"), out);
            }
        }
    }
    if let Some(map) = v.as_object() {
        let props = obj.get("properties").and_then(Value::as_object);
        if let Some(Value::Array(req)) = obj.get("required") {
            for r in req.iter().filter_map(Value::as_str) {
                if !map.contains_key(r) {
                    push(out, path, format!("missing required property {r:?}"));
                }
            }
        }
        for (k, x) in map {
            let sub = format!("{path}/{k}");
            match props.and_then(|p| p.get(k)) {
                Some(ps) => walk(ps, x, &sub, out),
                None => match obj.get("additionalProperties") {
                    Some(Value::Bool(false)) => out.push(Violation {
                        path: sub,
                        message: format!("property {k:?} is not allowed"),
                        extra_field: true,
                    }),
                    Some(ap @ Value::Object(_)) => walk(ap, x, &sub, out),
                    _ => {}
                },
            }
        }
    }
    if let Some(Value::Array(all)) = obj.get("allOf") {
        for s in all {
            walk(s, v, path, out);
        }
    }
    if let Some(Value::Array(any)) = obj.get("anyOf") {
        if !any.iter().any(|s| validate(s, v).is_empty()) {
            push(out, path, "matches none of anyOf".into());
        }
    }
    if let Some(Value::Array(one)) = obj.get("oneOf") {
        let n = one.iter().filter(|s| validate(s, v).is_empty()).count();
        if n != 1 {
            push(out, path, format!("matches {n} of oneOf (need exactly 1)"));
        }
    }
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn person() -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 5},
                "age": {"type": "integer", "minimum": 0, "exclusiveMaximum": 150},
                "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "uniqueItems": true, "maxItems": 2},
                "kind": {"const": "person"},
                "email": {"type": ["string", "null"]}
            },
            "required": ["name", "age"],
            "additionalProperties": false
        })
    }

    #[test]
    fn valid_documents_pass() {
        let s = person();
        check_schema(&s).unwrap();
        assert!(validate(&s, &json!({"name": "Ann", "age": 3})).is_empty());
        assert!(validate(
            &s,
            &json!({"name": "Ann", "age": 3.0, "tags": ["a", "b"], "kind": "person", "email": null})
        )
        .is_empty());
    }

    #[test]
    fn each_rule_catches_its_violation() {
        let s = person();
        let cases = [
            (json!({"name": "Ann"}), "missing required property \"age\""),
            (json!({"name": "", "age": 1}), "shorter"),
            (json!({"name": "Annabel", "age": 1}), "longer"),
            (json!({"name": "A", "age": 1.5}), "expected type"),
            (json!({"name": "A", "age": -1}), "minimum"),
            (json!({"name": "A", "age": 150}), "exclusiveMaximum"),
            (
                json!({"name": "A", "age": 1, "tags": ["a", "a"]}),
                "duplicate",
            ),
            (json!({"name": "A", "age": 1, "tags": ["c"]}), "not one of"),
            (
                json!({"name": "A", "age": 1, "tags": ["a", "b", "a"]}),
                "more than 2",
            ),
            (json!({"name": "A", "age": 1, "kind": "dog"}), "constant"),
            (json!({"name": "A", "age": 1, "email": 3}), "expected type"),
            (json!([1]), "expected type"),
        ];
        for (doc, want) in cases {
            let v = validate(&s, &doc);
            assert!(
                v.iter().any(|x| x.message.contains(want)),
                "{doc}: wanted {want:?}, got {v:?}"
            );
        }
        let extra = validate(&s, &json!({"name": "A", "age": 1, "x": 1}));
        assert_eq!(extra.len(), 1);
        assert!(extra[0].extra_field);
        assert_eq!(extra[0].path, "/x");
    }

    #[test]
    fn combinators() {
        let s = json!({"anyOf": [{"type": "string"}, {"type": "integer"}]});
        assert!(validate(&s, &json!("x")).is_empty());
        assert!(!validate(&s, &json!(1.5)).is_empty());
        let s = json!({"oneOf": [{"type": "number"}, {"type": "integer"}]});
        assert!(!validate(&s, &json!(1)).is_empty(), "1 matches both");
        assert!(validate(&s, &json!(1.5)).is_empty());
        let s = json!({"allOf": [{"minimum": 1}, {"maximum": 2}]});
        assert!(validate(&s, &json!(1.5)).is_empty());
        assert!(!validate(&s, &json!(3)).is_empty());
    }

    #[test]
    fn unsupported_keywords_are_rejected() {
        assert!(check_schema(&json!({"type": "string", "pattern": "^a"}))
            .unwrap_err()
            .contains("pattern"));
        assert!(check_schema(&json!({"properties": {"a": {"$ref": "#/x"}}})).is_err());
        assert!(check_schema(&json!({"type": "date"})).is_err());
        assert!(check_schema(&json!({"anyOf": []})).is_err());
        assert!(check_schema(&json!(true)).is_ok());
    }
}
