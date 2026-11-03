//! `mmlu_pro`: MMLU-Pro (TIGER-Lab, dataset MIT) with the official chain-of-thought protocol of
//! `evaluate_from_api.py` (TIGER-AI-Lab/MMLU-Pro commit f418b116): the category header, the five validation
//! examples of the question's category with their CoT, then the question, all in one user message; temperature
//! from the run settings (the official script uses 0); `**` removed from the reply before extraction; extraction
//! in three steps (`answer is (X)`, then `Answer: X`, then the last standalone letter A-J).
//!
//! Deviation, on purpose: the official script scores a reply with no extractable letter by a *random* guess;
//! here it is a failure (`no_answer`), so paired comparisons stay deterministic. Options equal to "N/A" are
//! dropped as in the official preprocessing.
//!
//! Smoke: the first 28 rows of the embedded stratified subset (2 per category). Full: the 280-row stratified
//! subset; the whole 12,032-row test split is used when `benchmarks.params.mmlu_pro.split: full` and the
//! converted file was imported (`talus-bench datasets import mmlu-pro-test FILE`).

use super::{take_limit, Task, TaskContext};
use crate::case::{Case, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::{self, DatasetPin};
use regex::Regex;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub struct MmluProTask;

const LETTERS: &str = "ABCDEFGHIJ";

fn options(row: &Value) -> Vec<String> {
    row["options"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|o| *o != "N/A")
        .map(str::to_string)
        .collect()
}

/// The official `format_example`.
pub fn format_example(question: &str, opts: &[String], cot: &str) -> String {
    let mut cot = if cot.is_empty() {
        "Let's think step by step."
    } else {
        cot
    };
    if let Some(c) = cot.strip_prefix("A: ") {
        cot = c;
    }
    let mut s = format!("Question: {question}\nOptions: ");
    for (i, o) in opts.iter().enumerate() {
        s.push_str(&format!("{}. {o}\n", &LETTERS[i..i + 1]));
    }
    s.push_str(&format!("Answer: {cot}\n\n"));
    s
}

/// The official three-step extraction (after `**` removal). None = no letter found.
pub fn extract_answer(text: &str) -> Option<String> {
    let text = text.replace("**", "");
    let r1 = Regex::new(r"answer is \(?([A-J])\)?").expect("re");
    if let Some(c) = r1.captures(&text) {
        return Some(c[1].to_string());
    }
    let r2 = Regex::new(r".*[aA]nswer:\s*([A-J])").expect("re");
    if let Some(c) = r2.captures(&text) {
        return Some(c[1].to_string());
    }
    // \b[A-J]\b(?!.*\b[A-J]\b) with DOTALL == the last standalone letter
    let r3 = Regex::new(r"\b[A-J]\b").expect("re");
    r3.find_iter(&text).last().map(|m| m.as_str().to_string())
}

impl Task for MmluProTask {
    fn id(&self) -> &str {
        "mmlu_pro"
    }

    fn scorer_version(&self) -> &'static str {
        "mmlu_pro-cot5/1"
    }

    fn datasets(&self, _tier: Tier) -> Vec<&'static DatasetPin> {
        vec![
            datasets::pin("mmlu-pro-strat280").expect("pinned"),
            datasets::pin("mmlu-pro-validation").expect("pinned"),
        ]
    }

    fn default_limit(&self, tier: Tier) -> Option<usize> {
        (tier == Tier::Smoke).then_some(28)
    }

    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String> {
        let full = ctx.params["split"].as_str() == Some("full");
        let test_pin = if full {
            datasets::pin("mmlu-pro-test").expect("pinned")
        } else {
            self.datasets(ctx.tier)[0]
        };
        let val_pin = datasets::pin("mmlu-pro-validation").expect("pinned");
        let parse = |t: String, id: &str| -> Result<Vec<Value>, String> {
            t.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str(l).map_err(|e| format!("{id}: {e}")))
                .collect()
        };
        let test = parse(datasets::load(test_pin, ctx.cache)?, test_pin.id)?;
        let val = parse(datasets::load(val_pin, ctx.cache)?, val_pin.id)?;
        let mut shots: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
        for v in &val {
            shots
                .entry(v["category"].as_str().unwrap_or_default().to_string())
                .or_default()
                .push(v);
        }
        let mut out = Vec::new();
        for row in &test {
            let cat = row["category"].as_str().unwrap_or_default();
            let qid = row["question_id"]
                .as_i64()
                .ok_or("mmlu_pro row without question_id")?;
            let mut prompt = format!(
                "The following are multiple choice questions (with answers) about {cat}. Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n"
            );
            for ex in shots.get(cat).map(Vec::as_slice).unwrap_or(&[]) {
                prompt.push_str(&format_example(
                    ex["question"].as_str().unwrap_or_default(),
                    &options(ex),
                    ex["cot_content"].as_str().unwrap_or_default(),
                ));
            }
            prompt.push_str(&format_example(
                row["question"].as_str().unwrap_or_default(),
                &options(row),
                "",
            ));
            out.push(Case {
                task: self.id().into(),
                case_id: format!("mmlu_pro/{qid}"),
                dataset: test_pin.id.into(),
                dataset_revision: format!(
                    "{}@{}+{}",
                    test_pin.revision,
                    &test_pin.sha256[..12],
                    &val_pin.sha256[..12]
                ),
                messages: vec![json!({"role": "user", "content": prompt})],
                expected: json!({"answer": row["answer"], "category": cat}),
                meta: Value::Null,
                critical: false,
                max_tokens: Some(4000),
                tools: Value::Null,
                request: Value::Null,
            });
        }
        Ok(take_limit(out, ctx.limit))
    }

    fn score(&self, case: &Case, out: &RawOutput) -> Score {
        let got = extract_answer(&out.content);
        let pass = got.is_some() && got.as_deref() == case.expected["answer"].as_str();
        Score::binary(
            pass,
            got.clone().map(Value::String).unwrap_or(Value::Null),
            json!({"stage": if pass { "pass" } else if got.is_some() { "wrong_answer" } else { "no_answer" },
                   "gold": case.expected["answer"], "category": case.expected["category"]}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extraction_steps() {
        assert_eq!(
            extract_answer("so **the answer is (C)**."),
            Some("C".into())
        );
        assert_eq!(extract_answer("The answer is B"), Some("B".into()));
        assert_eq!(extract_answer("blah\nAnswer: D\nmore"), Some("D".into()));
        assert_eq!(
            extract_answer("I pick E, not F. Final: G"),
            Some("G".into())
        );
        assert_eq!(extract_answer("none here"), None);
        assert_eq!(
            extract_answer("answer is (K)"),
            None,
            "K is not an option letter"
        );
    }

    #[test]
    fn prompt_and_scoring() {
        let t = MmluProTask;
        let cs = t.cases(&TaskContext::for_tests(Tier::Full)).unwrap();
        assert_eq!(cs.len(), 280);
        let cats: std::collections::BTreeSet<_> = cs[..28]
            .iter()
            .map(|c| c.expected["category"].to_string())
            .collect();
        assert_eq!(cats.len(), 14, "the first 28 cover every category twice");
        let p = cs[0].messages[0]["content"].as_str().unwrap();
        assert!(p.starts_with("The following are multiple choice questions (with answers) about "));
        assert_eq!(p.matches("Question: ").count(), 6, "5 shots + the question");
        assert!(p.ends_with("Answer: Let's think step by step.\n\n"));
        assert!(!p.contains("N/A\n"));
        let gold = cs[0].expected["answer"].as_str().unwrap();
        let ok = RawOutput {
            content: format!("Reasoning...\nThe answer is ({gold})."),
            ..Default::default()
        };
        assert!(t.score(&cs[0], &ok).pass);
        let none = RawOutput {
            content: "unclear, no letter here.".into(),
            ..Default::default()
        };
        assert_eq!(t.score(&cs[0], &none).details["stage"], "no_answer");
    }
}
