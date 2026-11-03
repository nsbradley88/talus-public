//! `bfcl`: Berkeley Function Calling Leaderboard v3, the single-turn Python categories `simple`, `multiple`,
//! `parallel` and `irrelevance` (gorilla-llm, Apache-2.0), in function-calling mode: the functions go out as
//! OpenAI `tools` (converted as the official FC handler does: `.` in names becomes `_`, gorilla types map to
//! OpenAPI types, floats get `format: float`), and the model's `tool_calls` are graded by a reimplementation of
//! the official AST checker (gorilla commit 6ea57973, `bfcl_eval/eval_checker/ast_eval/ast_checker.py`), Python
//! equality semantics included (1 == 1.0 == True). `irrelevance` passes when the model makes no call.
//!
//! Smoke: the first rows of an embedded interleaved slice (stratified by category). Full: all 1,040 rows,
//! combined from the seven pinned upstream files (`talus-bench datasets fetch bfcl-<file>`).

use super::{take_limit, Task, TaskContext};
use crate::case::{Case, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::{self, DatasetPin};
use serde_json::{json, Map, Value};

pub struct BfclTask;

const CATEGORIES: [&str; 4] = ["simple", "multiple", "parallel", "irrelevance"];

impl Task for BfclTask {
    fn id(&self) -> &str {
        "bfcl"
    }

    fn scorer_version(&self) -> &'static str {
        "bfcl-ast/1"
    }

    fn datasets(&self, tier: Tier) -> Vec<&'static DatasetPin> {
        match tier {
            Tier::Smoke => vec![datasets::pin("bfcl-v3-smoke").expect("pinned")],
            _ => [
                "bfcl-simple",
                "bfcl-answers-simple",
                "bfcl-multiple",
                "bfcl-answers-multiple",
                "bfcl-parallel",
                "bfcl-answers-parallel",
                "bfcl-irrelevance",
            ]
            .iter()
            .map(|id| datasets::pin(id).expect("pinned"))
            .collect(),
        }
    }

    fn default_limit(&self, tier: Tier) -> Option<usize> {
        (tier == Tier::Smoke).then_some(40)
    }

    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String> {
        let pins = self.datasets(ctx.tier);
        let rows: Vec<Value> = if ctx.tier == Tier::Smoke {
            let t = datasets::load(pins[0], ctx.cache)?;
            t.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str(l).map_err(|e| format!("bfcl smoke: {e}")))
                .collect::<Result<_, _>>()?
        } else {
            combine_upstream(ctx.cache)?
        };
        let revision = pins
            .iter()
            .map(|p| &p.sha256[..8])
            .collect::<Vec<_>>()
            .join("+");
        let mut out = Vec::new();
        for r in rows {
            let id = r["id"].as_str().ok_or("bfcl row without id")?;
            let messages = r["question"][0]
                .as_array()
                .cloned()
                .ok_or(format!("bfcl {id}: no question"))?;
            let funcs = r["function"].as_array().cloned().unwrap_or_default();
            out.push(Case {
                task: self.id().into(),
                case_id: format!("bfcl/{id}"),
                dataset: pins[0].id.into(),
                dataset_revision: format!("{}@{revision}", pins[0].revision),
                messages,
                expected: json!({"category": r["category"], "ground_truth": r["ground_truth"]}),
                meta: json!({"function": funcs}),
                critical: false,
                max_tokens: Some(1024),
                tools: Value::Array(funcs.iter().map(to_tool).collect()),
                request: Value::Null,
            });
        }
        Ok(take_limit(out, ctx.limit))
    }

    fn score(&self, case: &Case, out: &RawOutput) -> Score {
        let category = case.expected["category"].as_str().unwrap_or("simple");
        let calls = match decode_calls(&out.tool_calls) {
            Ok(c) => c,
            Err(e) => {
                return Score::binary(false, Value::Null, json!({"stage": "decode", "error": e}))
            }
        };
        let parsed = Value::Array(calls.iter().map(|(n, a)| json!({n.clone(): a})).collect());
        if category == "irrelevance" {
            let pass = calls.is_empty();
            return Score::binary(
                pass,
                parsed,
                json!({"stage": if pass { "pass" } else { "called_irrelevant_function" }}),
            );
        }
        if calls.is_empty() {
            return Score::binary(false, parsed, json!({"stage": "no_function_call"}));
        }
        let funcs = case.meta["function"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let gt = case.expected["ground_truth"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let res = ast_checker(&funcs, &calls, &gt, category);
        Score::binary(
            res.is_ok(),
            parsed,
            match res {
                Ok(()) => json!({"stage": "pass"}),
                Err((t, e)) => json!({"stage": t, "error": e}),
            },
        )
    }
}

/// Rows of the four categories combined and interleaved exactly like the embedded smoke slice.
fn combine_upstream(cache: &std::path::Path) -> Result<Vec<Value>, String> {
    let load = |id: &str| -> Result<Vec<Value>, String> {
        let t = datasets::load(datasets::pin(id).expect("pinned"), cache)?;
        t.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).map_err(|e| format!("{id}: {e}")))
            .collect()
    };
    let mut cats: Vec<Vec<Value>> = Vec::new();
    for c in CATEGORIES {
        let q = load(&format!("bfcl-{c}"))?;
        let answers: Map<String, Value> = if c == "irrelevance" {
            Map::new()
        } else {
            load(&format!("bfcl-answers-{c}"))?
                .into_iter()
                .filter_map(|a| Some((a["id"].as_str()?.to_string(), a["ground_truth"].clone())))
                .collect()
        };
        cats.push(
            q.into_iter()
                .map(|x| {
                    let id = x["id"].as_str().unwrap_or_default().to_string();
                    json!({"id": id, "category": c, "question": x["question"], "function": x["function"],
                           "ground_truth": answers.get(&id).cloned().unwrap_or(Value::Null)})
                })
                .collect(),
        );
    }
    let max = cats.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = Vec::new();
    for i in 0..max {
        for c in &cats {
            if let Some(r) = c.get(i) {
                out.push(r.clone());
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------
// Tool conversion (bfcl_eval/model_handler/utils.py convert_to_tool + _cast_to_openai_type, GORILLA_TO_OPENAPI)

fn openapi_type(t: &str) -> &'static str {
    match t {
        "integer" | "byte" | "short" | "long" => "integer",
        "number" | "float" | "double" => "number",
        "string" | "any" | "char" | "String" => "string",
        "boolean" | "bool" => "boolean",
        "array" | "list" | "tuple" | "ArrayList" | "Array" => "array",
        "dict" | "object" | "HashMap" | "Hashtable" | "Queue" | "Stack" => "object",
        _ => "string",
    }
}

fn cast_properties(props: &mut Map<String, Value>) {
    for (_, v) in props.iter_mut() {
        let Some(o) = v.as_object_mut() else { continue };
        match o.get("type").and_then(Value::as_str).map(str::to_string) {
            None => {
                o.insert("type".into(), json!("string"));
            }
            Some(t) => {
                if t == "float" {
                    o.insert("format".into(), json!("float"));
                    let d = o
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    o.insert(
                        "description".into(),
                        json!(format!("{d} This is a float type value.")),
                    );
                }
                o.insert("type".into(), json!(openapi_type(&t)));
            }
        }
        let ty = o["type"].as_str().unwrap_or_default().to_string();
        if ty == "array" || ty == "object" {
            if let Some(Value::Object(p)) = o.get_mut("properties") {
                cast_properties(p);
            } else if let Some(Value::Object(items)) = o.get_mut("items") {
                if let Some(it) = items
                    .get("type")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                {
                    items.insert("type".into(), json!(openapi_type(&it)));
                }
                let it = items
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if it == "array" {
                    if let Some(Value::Object(inner)) = items.get_mut("items") {
                        if let Some(t2) = inner
                            .get("type")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                        {
                            inner.insert("type".into(), json!(openapi_type(&t2)));
                        }
                    }
                } else if it == "object" {
                    if let Some(Value::Object(p)) = items.get_mut("properties") {
                        cast_properties(p);
                    }
                }
            }
        }
    }
}

/// One BFCL function description as an OpenAI tool.
pub fn to_tool(f: &Value) -> Value {
    let mut f = f.clone();
    let name = f["name"].as_str().unwrap_or_default().replace('.', "_");
    f["name"] = json!(name);
    if let Some(p) = f.get_mut("parameters").and_then(Value::as_object_mut) {
        p.insert("type".into(), json!("object"));
        if let Some(Value::Object(props)) = p.get_mut("properties") {
            cast_properties(props);
        }
    }
    json!({"type": "function", "function": f})
}

/// tool_calls -> [(name, arguments object)]
pub fn decode_calls(tool_calls: &Value) -> Result<Vec<(String, Value)>, String> {
    let Some(arr) = tool_calls.as_array() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for c in arr {
        let f = &c["function"];
        let name = f["name"]
            .as_str()
            .ok_or("tool call without a name")?
            .to_string();
        let args = match &f["arguments"] {
            Value::String(s) if s.trim().is_empty() => json!({}),
            Value::String(s) => serde_json::from_str::<Value>(s)
                .map_err(|e| format!("{name}: arguments are not JSON: {e}"))?,
            Value::Object(_) => f["arguments"].clone(),
            Value::Null => json!({}),
            other => return Err(format!("{name}: arguments {other} are not an object")),
        };
        if !args.is_object() {
            return Err(format!("{name}: arguments are not an object"));
        }
        out.push((name, args));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------
// AST checker (Python semantics)

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PyType {
    Int,
    Float,
    Bool,
    Str,
    List,
    Dict,
    None,
}

fn py_type(v: &Value) -> PyType {
    match v {
        Value::Null => PyType::None,
        Value::Bool(_) => PyType::Bool,
        Value::Number(n) if n.is_f64() => PyType::Float,
        Value::Number(_) => PyType::Int,
        Value::String(_) => PyType::Str,
        Value::Array(_) => PyType::List,
        Value::Object(_) => PyType::Dict,
    }
}

fn python_type(t: &str) -> PyType {
    match t {
        "integer" => PyType::Int,
        "float" => PyType::Float,
        "boolean" => PyType::Bool,
        "array" | "tuple" => PyType::List,
        "dict" => PyType::Dict,
        _ => PyType::Str, // "string", "any"
    }
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

/// Python `==` for JSON-decoded values.
pub fn py_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w)))
        }
        _ => match (num(a), num(b)) {
            (Some(x), Some(y)) => x == y,
            _ => a == b,
        },
    }
}

fn py_in(v: &Value, list: &[Value]) -> bool {
    list.iter().any(|x| py_eq(v, x))
}

fn as_list(v: &Value) -> Vec<Value> {
    v.as_array().cloned().unwrap_or_default()
}

fn possible_answer_type(pa: &[Value]) -> Option<PyType> {
    pa.iter().find(|a| **a != json!("")).map(py_type)
}

pub fn standardize_string(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, ' ' | ',' | '.' | '/' | '-' | '_' | '*' | '^'))
        .collect::<String>()
        .to_lowercase()
        .replace('\'', "\"")
}

fn std_value(v: &Value) -> Value {
    match v {
        Value::String(s) => Value::String(standardize_string(s)),
        x => x.clone(),
    }
}

type CheckErr = (&'static str, String);

/// type_checker: Ok(is_variable) or the error.
fn type_checker(
    param: &str,
    value: &Value,
    pa: &[Value],
    expected: PyType,
    nested: Option<PyType>,
) -> Result<bool, CheckErr> {
    let pat = possible_answer_type(pa);
    let mut is_variable = false;
    if let Some(t) = pat {
        if t != expected {
            is_variable = true;
        }
    }
    if py_type(value) == expected {
        match nested {
            None => return Ok(is_variable),
            Some(n) => {
                for item in pa {
                    let mut flag = true;
                    if let Value::Array(inner_pa) = item {
                        for vi in value.as_array().into_iter().flatten() {
                            if type_checker(param, vi, inner_pa, n, None).is_err() {
                                flag = false;
                                break;
                            }
                        }
                    }
                    if flag {
                        return Ok(is_variable);
                    }
                }
                // the official code falls through to its "variable" branch here, which returns the same
                // invalid result whichever way it goes
                return Err((
                    "type_error:nested",
                    format!("nested type check failed for {param:?}"),
                ));
            }
        }
    }
    if let Some(t) = pat {
        if py_type(value) == t {
            return Ok(true);
        }
    }
    Err((
        "type_error:simple",
        format!("incorrect type for parameter {param:?}: {value}"),
    ))
}

fn string_checker(param: &str, v: &str, pa: &[Value]) -> Result<(), CheckErr> {
    let std: Vec<String> = pa
        .iter()
        .filter_map(|x| x.as_str().map(standardize_string))
        .collect();
    if std.contains(&standardize_string(v)) {
        Ok(())
    } else {
        Err((
            "value_error:string",
            format!("invalid value for {param:?}: {v:?}"),
        ))
    }
}

fn list_checker(param: &str, v: &[Value], pa: &[Value]) -> Result<(), CheckErr> {
    let model = Value::Array(v.iter().map(std_value).collect());
    let options: Vec<Value> = pa
        .iter()
        .map(|p| match p {
            Value::Array(a) => Value::Array(a.iter().map(std_value).collect()),
            _ => Value::Array(Vec::new()), // "" (optional) iterates no characters
        })
        .collect();
    if py_in(&model, &options) {
        Ok(())
    } else {
        Err((
            "value_error:list/tuple",
            format!("invalid value for {param:?}"),
        ))
    }
}

fn dict_checker(param: &str, model: &Map<String, Value>, pas: &[Value]) -> Result<(), CheckErr> {
    let mut last: CheckErr = (
        "dict_checker:unclear",
        format!("no possible answer for {param:?}"),
    );
    for pa in pas {
        if *pa == json!("") {
            continue;
        }
        let Some(pa) = pa.as_object() else { continue };
        let mut ok = true;
        for (k, v) in model {
            let Some(opts) = pa.get(k) else {
                last = ("value_error:dict_key", format!("unexpected dict key {k:?}"));
                ok = false;
                break;
            };
            let std_opts: Vec<Value> = as_list(opts).iter().map(std_value).collect();
            if !py_in(&std_value(v), &std_opts) {
                last = (
                    "value_error:dict_value",
                    format!("invalid value for dict key {k:?}"),
                );
                ok = false;
                break;
            }
        }
        if ok {
            for (k, opts) in pa {
                if !model.contains_key(k) && !py_in(&json!(""), &as_list(opts)) {
                    last = ("value_error:dict_key", format!("missing dict key {k:?}"));
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return Ok(());
        }
    }
    Err(last)
}

fn list_dict_checker(param: &str, model: &[Value], pas: &[Value]) -> Result<(), CheckErr> {
    let mut last: CheckErr = (
        "list_dict_checker:unclear",
        format!("no possible answer for {param:?}"),
    );
    for pa in pas {
        let pa = as_list(pa);
        if model.len() != pa.len() {
            last = (
                "value_error:list_dict_count",
                "wrong number of dictionaries".into(),
            );
            continue;
        }
        let mut ok = true;
        for (m, p) in model.iter().zip(&pa) {
            let empty = Map::new();
            if let Err(e) = dict_checker(
                param,
                m.as_object().unwrap_or(&empty),
                std::slice::from_ref(p),
            ) {
                last = e;
                ok = false;
                break;
            }
        }
        if ok {
            return Ok(());
        }
    }
    Err(last)
}

fn find_description<'a>(funcs: &'a [Value], name: &str) -> Option<&'a Value> {
    funcs.iter().find(|f| f["name"] == name)
}

/// simple_function_checker. `call` = (name, args) from the model, `answer` = {func_name: {param: [options]}}.
fn simple_function_checker(
    desc: &Value,
    call: &(String, Value),
    answer: &Value,
) -> Result<(), CheckErr> {
    let pa_map = answer
        .as_object()
        .and_then(|m| m.values().next())
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let func_name = desc["name"].as_str().unwrap_or_default().replace('.', "_");
    let props = desc["parameters"]["properties"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let required = as_list(&desc["parameters"]["required"]);
    if call.0 != func_name {
        return Err((
            "simple_function_checker:wrong_func_name",
            format!("function {func_name:?} not found in model output"),
        ));
    }
    let params = call.1.as_object().cloned().unwrap_or_default();
    for r in required.iter().filter_map(Value::as_str) {
        if !params.contains_key(r) {
            return Err((
                "simple_function_checker:missing_required",
                format!("missing required parameter {r:?}"),
            ));
        }
    }
    for (param, value) in &params {
        let (Some(detail), Some(pa)) = (props.get(param), pa_map.get(param)) else {
            return Err((
                "simple_function_checker:unexpected_param",
                format!("unexpected parameter {param:?}"),
            ));
        };
        let pa = as_list(pa);
        let type_desc = detail["type"].as_str().unwrap_or("string");
        let expected = python_type(type_desc);
        let nested = if type_desc == "array" || type_desc == "tuple" {
            Some(python_type(
                detail["items"]["type"].as_str().unwrap_or("string"),
            ))
        } else {
            None
        };
        let mut value = value.clone();
        if type_desc == "float" && py_type(&value) == PyType::Int {
            value = json!(value.as_f64().unwrap_or(0.0));
        }
        let is_variable = type_checker(param, &value, &pa, expected, nested)?;
        if !is_variable {
            match (expected, nested) {
                (PyType::Dict, _) => {
                    let empty = Map::new();
                    dict_checker(param, value.as_object().unwrap_or(&empty), &pa)?;
                    continue;
                }
                (PyType::List, Some(PyType::Dict)) => {
                    list_dict_checker(
                        param,
                        value.as_array().map(Vec::as_slice).unwrap_or(&[]),
                        &pa,
                    )?;
                    continue;
                }
                (PyType::Str, _) => {
                    string_checker(param, value.as_str().unwrap_or_default(), &pa)?;
                    continue;
                }
                (PyType::List, _) => {
                    list_checker(
                        param,
                        value.as_array().map(Vec::as_slice).unwrap_or(&[]),
                        &pa,
                    )?;
                    continue;
                }
                _ => {}
            }
        }
        if !py_in(&value, &pa) {
            return Err((
                "value_error:others",
                format!("invalid value for {param:?}: {value}"),
            ));
        }
    }
    for (param, opts) in &pa_map {
        if !params.contains_key(param) && !py_in(&json!(""), &as_list(opts)) {
            return Err((
                "simple_function_checker:missing_optional",
                format!("optional parameter {param:?} not provided and not marked optional"),
            ));
        }
    }
    Ok(())
}

/// ast_checker for the Python categories.
pub fn ast_checker(
    funcs: &[Value],
    calls: &[(String, Value)],
    answers: &[Value],
    category: &str,
) -> Result<(), CheckErr> {
    let expected_name = |a: &Value| {
        a.as_object()
            .and_then(|m| m.keys().next().cloned())
            .unwrap_or_default()
    };
    if category.contains("parallel") {
        if calls.len() != answers.len() {
            return Err((
                "parallel_function_checker_no_order:wrong_count",
                "wrong number of functions".into(),
            ));
        }
        let mut matched: Vec<usize> = Vec::new();
        for a in answers {
            let Some(desc) = find_description(funcs, &expected_name(a)) else {
                return Err((
                    "parallel_function_checker_no_order:cannot_find_match",
                    "no description".into(),
                ));
            };
            let mut found = false;
            for (i, c) in calls.iter().enumerate() {
                if matched.contains(&i) {
                    continue;
                }
                if simple_function_checker(desc, c, a).is_ok() {
                    matched.push(i);
                    found = true;
                    break;
                }
            }
            if !found {
                return Err((
                    "parallel_function_checker_no_order:cannot_find_match",
                    format!("no model call matches {}", expected_name(a)),
                ));
            }
        }
        Ok(())
    } else if category.contains("multiple") {
        if calls.len() != answers.len() {
            return Err((
                "multiple_function_checker:wrong_count",
                "wrong number of functions".into(),
            ));
        }
        let a = answers.first().cloned().unwrap_or(Value::Null);
        let desc = find_description(funcs, &expected_name(&a))
            .cloned()
            .unwrap_or(Value::Null);
        simple_function_checker(&desc, &calls[0], &a)
    } else {
        if calls.len() != 1 {
            return Err((
                "simple_function_checker:wrong_count",
                "wrong number of functions".into(),
            ));
        }
        simple_function_checker(
            &funcs[0],
            &calls[0],
            answers.first().unwrap_or(&Value::Null),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: Value) -> Value {
        json!([{"id": "c1", "type": "function", "function": {"name": name, "arguments": args.to_string()}}])
    }

    fn out(tc: Value) -> RawOutput {
        RawOutput {
            tool_calls: tc,
            finish_reason: "tool_calls".into(),
            ..Default::default()
        }
    }

    fn cases() -> Vec<Case> {
        BfclTask
            .cases(&TaskContext::for_tests(Tier::Smoke))
            .unwrap()
    }

    #[test]
    fn simple_case_rules() {
        let t = BfclTask;
        let cs = cases();
        assert_eq!(cs.len(), 120);
        let c = cs.iter().find(|c| c.case_id == "bfcl/simple_0").unwrap();
        // tool conversion
        assert_eq!(c.tools[0]["function"]["parameters"]["type"], "object");
        assert_eq!(c.tools[0]["function"]["name"], "calculate_triangle_area");
        // gt: base [10], height [5], unit ["units", ""]
        assert!(
            t.score(
                c,
                &out(call(
                    "calculate_triangle_area",
                    json!({"base": 10, "height": 5})
                ))
            )
            .pass
        );
        assert!(
            t.score(
                c,
                &out(call(
                    "calculate_triangle_area",
                    json!({"base": 10, "height": 5, "unit": "Units"})
                ))
            )
            .pass,
            "strings standardized"
        );
        let s = t.score(
            c,
            &out(call(
                "calculate_triangle_area",
                json!({"base": 10, "height": 6}),
            )),
        );
        assert_eq!(s.details["stage"], "value_error:others");
        let s = t.score(
            c,
            &out(call(
                "calculate_triangle_area",
                json!({"base": "10", "height": 5}),
            )),
        );
        assert_eq!(s.details["stage"], "type_error:simple");
        let s = t.score(
            c,
            &out(call("calculate_triangle_area", json!({"base": 10}))),
        );
        assert_eq!(
            s.details["stage"],
            "simple_function_checker:missing_required"
        );
        let s = t.score(
            c,
            &out(call(
                "calculate_triangle_area",
                json!({"base": 10, "height": 5, "color": "red"}),
            )),
        );
        assert_eq!(
            s.details["stage"],
            "simple_function_checker:unexpected_param"
        );
        let s = t.score(c, &out(call("area", json!({"base": 10, "height": 5}))));
        assert_eq!(
            s.details["stage"],
            "simple_function_checker:wrong_func_name"
        );
        let s = t.score(c, &out(Value::Null));
        assert_eq!(s.details["stage"], "no_function_call");
        // dotted name -> underscore (simple_1: math.factorial)
        let c1 = cs.iter().find(|c| c.case_id == "bfcl/simple_1").unwrap();
        assert_eq!(c1.tools[0]["function"]["name"], "math_factorial");
        assert!(
            t.score(c1, &out(call("math_factorial", json!({"number": 5}))))
                .pass
        );
    }

    #[test]
    fn parallel_and_irrelevance() {
        let t = BfclTask;
        let cs = cases();
        let p = cs.iter().find(|c| c.case_id == "bfcl/parallel_0").unwrap();
        let two = json!([
            {"function": {"name": "spotify_play", "arguments": "{\"artist\": \"Maroon 5\", \"duration\": 15}"}},
            {"function": {"name": "spotify_play", "arguments": "{\"artist\": \"Taylor Swift\", \"duration\": 20}"}}
        ]);
        assert!(t.score(p, &out(two)).pass, "order does not matter");
        let one = call(
            "spotify_play",
            json!({"artist": "Taylor Swift", "duration": 20}),
        );
        assert_eq!(
            t.score(p, &out(one)).details["stage"],
            "parallel_function_checker_no_order:wrong_count"
        );
        let irr = cs
            .iter()
            .find(|c| c.expected["category"] == "irrelevance")
            .unwrap();
        assert!(t.score(irr, &out(Value::Null)).pass);
        assert!(!t.score(irr, &out(call("anything", json!({})))).pass);
        let bad = json!([{"function": {"name": "f", "arguments": "{not json"}}]);
        assert_eq!(t.score(irr, &out(bad)).details["stage"], "decode");
    }

    #[test]
    fn python_equality_and_types() {
        assert!(py_eq(&json!(1), &json!(1.0)));
        assert!(py_eq(&json!(true), &json!(1)));
        assert!(!py_eq(&json!("1"), &json!(1)));
        assert!(py_eq(&json!([1, {"a": 2.0}]), &json!([1.0, {"a": 2}])));
        assert_eq!(standardize_string("April 1, 2024's"), "april12024\"s");
        // int -> float auto conversion, float param
        let desc = json!({"name": "f", "parameters": {"type": "dict", "properties": {"x": {"type": "float"}}, "required": ["x"]}});
        let ans = json!({"f": {"x": [2.5, 3.0]}});
        assert!(simple_function_checker(&desc, &("f".into(), json!({"x": 3})), &ans).is_ok());
        // variable: possible answer typed str while param is integer
        let desc = json!({"name": "f", "parameters": {"type": "dict", "properties": {"n": {"type": "integer"}}, "required": ["n"]}});
        let ans = json!({"f": {"n": ["count"]}});
        assert!(simple_function_checker(&desc, &("f".into(), json!({"n": "count"})), &ans).is_ok());
        // array of strings, standardized elementwise
        let desc = json!({"name": "f", "parameters": {"type": "dict", "properties": {"l": {"type": "array", "items": {"type": "string"}}}, "required": ["l"]}});
        let ans = json!({"f": {"l": [["New York", "LA"]]}});
        assert!(simple_function_checker(
            &desc,
            &("f".into(), json!({"l": ["new york", "la"]})),
            &ans
        )
        .is_ok());
        assert!(simple_function_checker(
            &desc,
            &("f".into(), json!({"l": ["la", "new york"]})),
            &ans
        )
        .is_err());
        // dict parameter
        let desc = json!({"name": "f", "parameters": {"type": "dict", "properties": {"d": {"type": "dict"}}, "required": ["d"]}});
        let ans = json!({"f": {"d": [{"city": ["Paris"], "days": [3, ""]}]}});
        assert!(simple_function_checker(
            &desc,
            &("f".into(), json!({"d": {"city": "paris"}})),
            &ans
        )
        .is_ok());
        assert!(simple_function_checker(
            &desc,
            &("f".into(), json!({"d": {"city": "Rome"}})),
            &ans
        )
        .is_err());
    }
}
