//! `gsm8k`: grade-school math with an exact numeric answer (dataset: openai/grade-school-math, MIT; pinned).
//!
//! Prompt: zero-shot, "end with `#### <number>`" (the dataset's own answer convention). Scoring is on the answer
//! content only (never `reasoning_content`):
//! - strict: the number after the last `####` in the reply;
//! - flexible: else the last number in the reply (the lm-evaluation-harness "flexible-extract" idea);
//!   the case passes if the extracted number equals the reference. The extraction method is recorded, and the
//!   strict-only result is kept in the details so both rates can be reported.
//!
//! Numbers are compared by value after removing thousands separators, a leading `$` and a trailing `.`.

use super::{take_limit, Task, TaskContext};
use crate::case::{Case, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::{self, DatasetPin};
use serde_json::{json, Value};

pub struct Gsm8kTask;

const INSTRUCTION: &str = "Solve the following math problem. Reason step by step, then write the final answer on its own last line in the form \"#### <number>\" (digits only, no units).";

impl Task for Gsm8kTask {
    fn id(&self) -> &str {
        "gsm8k"
    }

    fn scorer_version(&self) -> &'static str {
        "gsm8k/1"
    }

    fn datasets(&self, tier: Tier) -> Vec<&'static DatasetPin> {
        vec![match tier {
            Tier::Smoke => datasets::pin("gsm8k-test-slice50"),
            Tier::Full | Tier::Qualification => datasets::pin("gsm8k-test"),
        }
        .expect("pinned")]
    }

    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String> {
        let d = self.datasets(ctx.tier)[0];
        let text = datasets::load(d, ctx.cache)?;
        let mut out = Vec::new();
        for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            let row: Value =
                serde_json::from_str(line).map_err(|e| format!("{} row {i}: {e}", d.id))?;
            let q = row["question"]
                .as_str()
                .ok_or(format!("{} row {i}: no question", d.id))?;
            let a = row["answer"]
                .as_str()
                .ok_or(format!("{} row {i}: no answer", d.id))?;
            let gold = reference_answer(a).ok_or(format!("{} row {i}: no #### answer", d.id))?;
            out.push(Case {
                task: self.id().into(),
                // index in the full test split (the slice is its prefix, so ids agree across tiers)
                case_id: format!("gsm8k/test-{i:04}"),
                dataset: d.id.into(),
                dataset_revision: format!("{}@{}", &d.revision[..12], &d.sha256[..12]),
                messages: vec![json!({"role": "user", "content": format!("{INSTRUCTION}\n\n{q}")})],
                expected: json!({"answer": gold}),
                meta: Value::Null,
                critical: false,
                max_tokens: None,
                tools: Value::Null,
                request: Value::Null,
            });
        }
        Ok(take_limit(out, ctx.limit))
    }

    fn score(&self, case: &Case, out: &RawOutput) -> Score {
        let gold = case.expected["answer"].as_str().and_then(parse_number);
        let strict = strict_extract(&out.content);
        let (got, method) = match strict.clone() {
            Some(s) => (Some(s), "strict"),
            None => (last_number(&out.content), "flexible"),
        };
        let eq = |x: &Option<String>| match (x.as_deref().and_then(parse_number), gold) {
            (Some(a), Some(b)) => (a - b).abs() <= 1e-6 * b.abs().max(1.0),
            _ => false,
        };
        let pass = eq(&got);
        Score::binary(
            pass,
            got.clone().map(Value::String).unwrap_or(Value::Null),
            json!({"stage": if pass { "pass" } else if got.is_some() { "wrong_answer" } else { "no_answer" },
                   "extraction": if got.is_some() { method } else { "none" }, "strict_pass": eq(&strict), "gold": case.expected["answer"]}),
        )
    }
}

/// The reference answer: the text after `####`, without thousands separators.
pub fn reference_answer(a: &str) -> Option<String> {
    let tail = a.rsplit_once("####")?.1.trim();
    let n = tail.replace(',', "");
    parse_number(&n).map(|_| n)
}

/// The first number after the last `####`.
pub fn strict_extract(s: &str) -> Option<String> {
    let tail = s.rsplit_once("####")?.1;
    numbers(tail).into_iter().next()
}

pub fn last_number(s: &str) -> Option<String> {
    numbers(s).pop()
}

/// Numbers in reading order: optional '-', digits with optional ',' groups, optional decimal part.
pub fn numbers(s: &str) -> Vec<String> {
    let b: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let neg = i > 0 && b[i - 1] == '-' && (i < 2 || !b[i - 2].is_ascii_alphanumeric());
            let start = i;
            // a ',' is a thousands separator only when exactly three digits follow it
            let group = |i: usize| {
                i + 3 < b.len() + 1
                    && b[i + 1..i + 4].iter().all(|c| c.is_ascii_digit())
                    && b.get(i + 4).is_none_or(|c| !c.is_ascii_digit())
            };
            while i < b.len() && (b[i].is_ascii_digit() || (b[i] == ',' && group(i))) {
                i += 1;
            }
            if i + 1 < b.len() && b[i] == '.' && b[i + 1].is_ascii_digit() {
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let mut n: String = b[start..i].iter().filter(|c| **c != ',').collect();
            if neg {
                n.insert(0, '-');
            }
            out.push(n);
        } else {
            i += 1;
        }
    }
    out
}

pub fn parse_number(s: &str) -> Option<f64> {
    let t = s
        .trim()
        .trim_start_matches('$')
        .trim_end_matches('.')
        .replace(',', "");
    t.parse::<f64>().ok().filter(|x| x.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out(s: &str) -> RawOutput {
        RawOutput {
            content: s.into(),
            finish_reason: "stop".into(),
            ..Default::default()
        }
    }

    #[test]
    fn number_extraction() {
        assert_eq!(
            numbers("I paid $1,234.50 for 3 items, -7 left"),
            vec!["1234.50", "3", "-7"]
        );
        assert_eq!(numbers("x-5 is not negative"), vec!["5"]);
        assert_eq!(
            numbers("pick 3,4 or 12,345,678"),
            vec!["3", "4", "12345678"]
        );
        assert_eq!(numbers("version 2."), vec!["2"]);
        assert_eq!(strict_extract("blah 4 #### 18\n"), Some("18".into()));
        assert_eq!(
            strict_extract("#### 1 then #### 2,000"),
            Some("2000".into())
        );
        assert_eq!(last_number("so the answer is 42."), Some("42".into()));
        assert_eq!(reference_answer("... \n#### 1,000"), Some("1000".into()));
        assert_eq!(parse_number("$18."), Some(18.0));
    }

    #[test]
    fn scoring() {
        let t = Gsm8kTask;
        let cs = t
            .cases(&TaskContext {
                limit: Some(5),
                ..TaskContext::for_tests(Tier::Smoke)
            })
            .unwrap();
        assert_eq!(cs.len(), 5);
        assert_eq!(cs[0].case_id, "gsm8k/test-0000");
        assert_eq!(cs[0].expected["answer"], "18"); // Janet's ducks
        let s = t.score(&cs[0], &out("16-3-4=9, 9*2=18\n#### 18"));
        assert!(s.pass);
        assert_eq!(s.details["extraction"], "strict");
        let s = t.score(&cs[0], &out("She makes $18.00 per day."));
        assert!(s.pass);
        assert_eq!(s.details["extraction"], "flexible");
        assert_eq!(s.details["strict_pass"], false);
        let s = t.score(&cs[0], &out("#### 19"));
        assert!(!s.pass);
        let s = t.score(&cs[0], &out(""));
        assert!(!s.pass);
        assert_eq!(s.details["extraction"], "none");
        // strict wins over a later stray number
        let s = t.score(&cs[0], &out("#### 18\n(checked 2 times)"));
        assert!(s.pass);
    }

    #[test]
    fn full_tier_needs_the_cache() {
        let e = Gsm8kTask
            .cases(&TaskContext {
                limit: None,
                ..TaskContext::for_tests(Tier::Full)
            })
            .unwrap_err();
        assert!(e.contains("datasets fetch gsm8k-test"), "{e}");
    }
}
