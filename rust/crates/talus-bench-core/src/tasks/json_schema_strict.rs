//! `json_schema_strict`: the `json_schema` cases sent with `response_format: {"type": "json_schema", ...}`, i.e.
//! grammar-constrained decoding (talus-server extension E6). Parse and schema compliance should then be 100% by
//! construction; the semantic checks still measure whether the model filled the structure correctly. Constrained
//! sampling is not the model's own distribution, so this task is never a quality gate (server-extensions.md E6).
//! An engine without support answers HTTP 400 "json_schema not supported by this engine".

use super::json_schema::JsonSchemaTask;
use super::{Task, TaskContext};
use crate::case::{Case, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::DatasetPin;
use serde_json::json;

pub struct JsonSchemaStrictTask;

impl Task for JsonSchemaStrictTask {
    fn id(&self) -> &str {
        "json_schema_strict"
    }

    fn scorer_version(&self) -> &'static str {
        "json_schema_strict/1"
    }

    fn datasets(&self, tier: Tier) -> Vec<&'static DatasetPin> {
        JsonSchemaTask.datasets(tier)
    }

    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String> {
        let mut out = JsonSchemaTask.cases(ctx)?;
        for c in &mut out {
            let id = c
                .case_id
                .strip_prefix("json_schema/")
                .unwrap_or(&c.case_id)
                .to_string();
            c.task = self.id().into();
            c.case_id = format!("json_schema_strict/{id}");
            c.request = json!({"response_format": {"type": "json_schema", "json_schema": {
                "name": id.replace('-', "_"), "strict": true, "schema": c.meta["schema"].clone()}}});
        }
        Ok(out)
    }

    fn score(&self, case: &Case, out: &RawOutput) -> Score {
        JsonSchemaTask.score(case, out)
    }

    fn caveats(&self) -> Vec<String> {
        vec!["json_schema_strict is grammar-constrained (response_format json_schema, server extension E6): parse/schema validity is by construction and the output is not the model's own distribution; informational, never a quality gate".into()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_carry_response_format() {
        let ctx = TaskContext {
            limit: None,
            ..TaskContext::for_tests(Tier::Smoke)
        };
        let cs = JsonSchemaStrictTask.cases(&ctx).unwrap();
        assert_eq!(cs.len(), 20);
        assert_eq!(cs[0].case_id, "json_schema_strict/order-extract");
        assert_eq!(cs[0].task, "json_schema_strict");
        let rf = &cs[0].request["response_format"];
        assert_eq!(rf["type"], "json_schema");
        assert_eq!(rf["json_schema"]["schema"], cs[0].meta["schema"]);
        assert_eq!(rf["json_schema"]["name"], "order_extract");
        let plain = JsonSchemaTask.cases(&ctx).unwrap();
        assert_ne!(
            cs[0].prompt_hash(),
            plain[0].prompt_hash(),
            "the request is part of the hash"
        );
    }
}
