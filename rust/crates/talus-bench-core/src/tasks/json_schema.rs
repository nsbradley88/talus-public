//! `json_schema`: structured-output reliability. The model gets a task and a JSON Schema and must answer with a
//! single JSON value. A case passes only when the reply, trimmed of surrounding whitespace, parses as JSON as is
//! (no code fence, no prose), validates against the schema, and meets the case's semantic checks (JSON pointer ->
//! expected value). Fenced replies, extra fields and each failure stage are recorded separately.
//!
//! No `response_format` is sent, so this measures prompt-following JSON, not grammar-constrained decoding; the
//! grammar-constrained variant is `json_schema_strict` (server extension E6).

use super::{take_limit, Task, TaskContext};
use crate::case::{Case, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::{self, DatasetPin};
use crate::jsonschema::{check_schema, json_eq, validate};
use serde_json::{json, Value};

pub struct JsonSchemaTask;

const SYSTEM: &str = "You produce machine-readable output. Reply with a single JSON value only: no prose, no explanation, no Markdown code fences.";

impl Task for JsonSchemaTask {
    fn id(&self) -> &str {
        "json_schema"
    }

    fn scorer_version(&self) -> &'static str {
        "json_schema/1"
    }

    fn datasets(&self, _tier: Tier) -> Vec<&'static DatasetPin> {
        vec![datasets::pin("talus-json-schema-v1").expect("pinned")]
    }

    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String> {
        let d = self.datasets(ctx.tier)[0];
        let text = datasets::load(d, ctx.cache)?;
        let mut out = Vec::new();
        for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            let row: Value =
                serde_json::from_str(line).map_err(|e| format!("{} row {i}: {e}", d.id))?;
            let id = row["id"]
                .as_str()
                .ok_or(format!("{} row {i}: no id", d.id))?;
            let schema = row["schema"].clone();
            check_schema(&schema).map_err(|e| format!("{} {id}: {e}", d.id))?;
            let user = format!(
                "{}\n\nThe JSON must conform to this JSON Schema:\n{}",
                row["instruction"].as_str().unwrap_or_default(),
                serde_json::to_string_pretty(&schema).unwrap_or_default()
            );
            out.push(Case {
                task: self.id().into(),
                case_id: format!("json_schema/{id}"),
                dataset: d.id.into(),
                dataset_revision: format!("{}@{}", d.revision, &d.sha256[..12]),
                messages: vec![
                    json!({"role": "system", "content": SYSTEM}),
                    json!({"role": "user", "content": user}),
                ],
                expected: json!({"must": row["must"]}),
                meta: json!({"schema": schema}),
                critical: row["critical"].as_bool().unwrap_or(false),
                max_tokens: Some(512),
                tools: Value::Null,
                request: Value::Null,
            });
        }
        Ok(take_limit(out, ctx.limit))
    }

    fn score(&self, case: &Case, out: &RawOutput) -> Score {
        let text = out.content.trim();
        let schema = &case.meta["schema"];
        let parsed: Option<Value> = serde_json::from_str(text).ok();
        let fenced_ok = parsed.is_none()
            && strip_fence(text)
                .and_then(|t| serde_json::from_str::<Value>(t).ok())
                .is_some();
        let Some(doc) = parsed else {
            return Score::binary(
                false,
                Value::Null,
                json!({"stage": "parse", "parse_ok": false, "fenced": fenced_ok, "schema_ok": false,
                       "semantic_ok": false, "extra_fields": 0}),
            );
        };
        let violations = validate(schema, &doc);
        let extra = violations.iter().filter(|v| v.extra_field).count();
        let schema_ok = violations.is_empty();
        let mut failed_checks = Vec::new();
        if let Some(must) = case.expected["must"].as_object() {
            for (ptr, want) in must {
                match doc.pointer(ptr) {
                    Some(got) if json_eq(got, want) => {}
                    got => {
                        failed_checks.push(json!({"pointer": ptr, "expected": want, "got": got}))
                    }
                }
            }
        }
        let semantic_ok = failed_checks.is_empty();
        let stage = if !schema_ok {
            "schema"
        } else if !semantic_ok {
            "semantic"
        } else {
            "pass"
        };
        Score::binary(
            schema_ok && semantic_ok,
            doc,
            json!({
                "stage": stage, "parse_ok": true, "fenced": false, "schema_ok": schema_ok,
                "semantic_ok": semantic_ok, "extra_fields": extra,
                "violations": violations.iter().take(10).map(|v| format!("{}: {}", v.path, v.message)).collect::<Vec<_>>(),
                "failed_checks": failed_checks,
            }),
        )
    }
}

/// The body of a single ```...``` block that makes up the whole reply.
fn strip_fence(t: &str) -> Option<&str> {
    let t = t.strip_prefix("```")?.strip_suffix("```")?;
    let body = t
        .split_once('\n')
        .map(|(lang, rest)| {
            if lang.trim().chars().all(|c| c.is_ascii_alphanumeric()) {
                rest
            } else {
                t
            }
        })
        .unwrap_or(t);
    Some(body.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> TaskContext<'static> {
        TaskContext {
            limit: None,
            ..TaskContext::for_tests(Tier::Smoke)
        }
    }

    fn out(s: &str) -> RawOutput {
        RawOutput {
            content: s.into(),
            finish_reason: "stop".into(),
            ..Default::default()
        }
    }

    #[test]
    fn cases_load_and_are_stable() {
        let t = JsonSchemaTask;
        let cs = t.cases(&ctx()).unwrap();
        assert_eq!(cs.len(), 20);
        assert_eq!(cs[0].case_id, "json_schema/order-extract");
        assert!(cs.iter().any(|c| c.critical));
        let again = t.cases(&ctx()).unwrap();
        assert_eq!(cs[3].prompt_hash(), again[3].prompt_hash());
        let ids: std::collections::BTreeSet<_> = cs.iter().map(|c| &c.case_id).collect();
        assert_eq!(ids.len(), cs.len(), "case ids are unique");
    }

    #[test]
    fn scoring_stages() {
        let t = JsonSchemaTask;
        let cs = t.cases(&ctx()).unwrap();
        let weather = cs
            .iter()
            .find(|c| c.case_id == "json_schema/weather-tool")
            .unwrap();
        let good = r#"{"city": "Lyon", "day": "tomorrow", "unit": "celsius"}"#;
        let s = t.score(weather, &out(&format!("  {good}\n")));
        assert!(s.pass, "{:?}", s.details);
        let s = t.score(weather, &out(&format!("```json\n{good}\n```")));
        assert!(!s.pass);
        assert_eq!(s.details["stage"], "parse");
        assert_eq!(s.details["fenced"], true);
        let s = t.score(
            weather,
            &out(r#"{"city": "Lyon", "day": "tomorrow", "unit": "celsius", "note": "x"}"#),
        );
        assert!(!s.pass);
        assert_eq!(s.details["stage"], "schema");
        assert_eq!(s.details["extra_fields"], 1);
        let s = t.score(
            weather,
            &out(r#"{"city": "Paris", "day": "tomorrow", "unit": "celsius"}"#),
        );
        assert!(!s.pass);
        assert_eq!(s.details["stage"], "semantic");
        let s = t.score(weather, &out("Sure! Here it is"));
        assert_eq!(s.details["parse_ok"], false);
        assert_eq!(s.details["fenced"], false);
    }

    #[test]
    fn fence_stripping() {
        assert_eq!(strip_fence("```json\n{}\n```"), Some("{}"));
        assert_eq!(strip_fence("```\n[1]\n```"), Some("[1]"));
        assert_eq!(strip_fence("{}"), None);
    }
}
