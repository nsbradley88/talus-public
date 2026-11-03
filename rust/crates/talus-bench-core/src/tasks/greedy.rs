//! `greedy_identity`: the greedy byte-identity gate of docs/METHODOLOGY.md. The cases are the bench27 `code`
//! mix (6 prompts) and `edit` mix (3 whole-file C++ edits of bench/editsrc, first 9000 characters), sent exactly
//! as bench/bench27.py sends them: temperature 0, thinking off, seed = the prompt's index in its mix, 700 / 1500
//! output tokens. Each case "passes" for itself when an answer came back; the gate compares the candidate's
//! answer text and reasoning with the reference's byte for byte (the identity gate), which is the check for
//! changes that claim to keep numerics unchanged.

use super::{take_limit, Task, TaskContext};
use crate::case::{Case, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::DatasetPin;
use crate::util::sha256_hex;
use serde_json::{json, Value};

pub struct GreedyTask;

pub const CODE: [&str; 6] = [
    "Write a Python class LRUCache with get/put in O(1) using a dict and a doubly linked list, with docstrings and pytest tests.",
    "Implement a Go function that merges overlapping intervals ([][2]int), sorts by start, and include table-driven tests.",
    "Write a TypeScript function debounce<T extends (...args:any[])=>void>(fn:T, ms:number) with cancel() and flush(), plus Jest tests.",
    "Write a Rust function that parses a CSV line with quoted fields (commas and escaped quotes inside quotes) into Vec<String>, with unit tests.",
    "Refactor this Python into a dataclass-based design with type hints and explain each change:\nclass P:\n def __init__(s,n,a): s.n=n; s.a=a\n def older(s,o): return s.a>o.a\n def __repr__(s): return 'P('+s.n+','+str(s.a)+')'",
    "Write a SQL schema for orders, order_items and products, then a query returning the top 5 products by revenue in the last 30 days, with indexes.",
];

const EDIT_SRC: [(&str, &str); 3] = [
    (
        "sampling.cpp",
        include_str!("../../../../../bench/editsrc/sampling.cpp"),
    ),
    (
        "speculative.cpp",
        include_str!("../../../../../bench/editsrc/speculative.cpp"),
    ),
    (
        "llama-batch.cpp",
        include_str!("../../../../../bench/editsrc/llama-batch.cpp"),
    ),
];

const EDIT_ASK: [&str; 3] = [
    "Add a short doc comment above every function that lacks one. Output the complete updated file, nothing else.",
    "Rename every local variable named `i` to `idx` and every `n` to `count`. Output the complete updated file, nothing else.",
    "Replace every use of `fprintf(stderr, ...)` or LOG_* macro message text so it starts with the function name. Output the complete updated file, nothing else.",
];

/// The edit prompts as bench27.py builds them.
pub fn edit_prompts() -> Vec<String> {
    EDIT_SRC
        .iter()
        .zip(EDIT_ASK)
        .map(|((_, src), ask)| {
            let head: String = src.chars().take(9000).collect();
            format!("Here is a C++ source file:\n```cpp\n{head}\n```\n{ask}")
        })
        .collect()
}

impl Task for GreedyTask {
    fn id(&self) -> &str {
        "greedy_identity"
    }

    fn scorer_version(&self) -> &'static str {
        "greedy_identity/1"
    }

    fn datasets(&self, _tier: Tier) -> Vec<&'static DatasetPin> {
        Vec::new()
    }

    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String> {
        let src_sha = sha256_hex(
            EDIT_SRC
                .iter()
                .map(|(_, s)| *s)
                .collect::<String>()
                .as_bytes(),
        );
        let rev = format!("bench27+editsrc@{}", &src_sha[..12]);
        let mut out = Vec::new();
        let mixes: [(&str, Vec<String>, u32); 2] = [
            ("code", CODE.iter().map(|s| s.to_string()).collect(), 700),
            ("edit", edit_prompts(), 1500),
        ];
        for (mix, prompts, max) in mixes {
            for (i, p) in prompts.into_iter().enumerate() {
                out.push(Case {
                    task: self.id().into(),
                    case_id: format!("greedy_identity/{mix}-{i}"),
                    dataset: "bench27".into(),
                    dataset_revision: rev.clone(),
                    messages: vec![json!({"role": "user", "content": p})],
                    expected: json!({"mix": mix, "i": i}),
                    meta: Value::Null,
                    critical: false,
                    max_tokens: Some(max),
                    tools: Value::Null,
                    request: json!({"seed": i, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": false}}),
                });
            }
        }
        Ok(take_limit(out, ctx.limit))
    }

    fn score(&self, _case: &Case, out: &RawOutput) -> Score {
        let has = !out.content.is_empty() || !out.reasoning_content.is_empty();
        Score::binary(
            has,
            Value::Null,
            json!({"stage": if has { "pass" } else { "empty" },
                   "sha256": sha256_hex(format!("{}\u{0}{}", out.content, out.reasoning_content).as_bytes())}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_match_bench27() {
        let cs = GreedyTask
            .cases(&TaskContext::for_tests(Tier::Smoke))
            .unwrap();
        assert_eq!(cs.len(), 9);
        assert_eq!(cs[0].request["seed"], 0);
        assert_eq!(cs[8].max_tokens, Some(1500));
        let e = edit_prompts();
        // python3: the three prompts bench27.py builds, joined with NUL
        assert_eq!(
            sha256_hex(e.join("\u{0}").as_bytes()),
            "a4b4e38fc283bf5b4e7e3489f0efb3622596ce5026815462c9d4c59eef869305"
        );
        assert!(e[0].starts_with("Here is a C++ source file:\n```cpp\n"));
        assert!(e[2].ends_with("Output the complete updated file, nothing else."));
        for p in &e {
            let body = p
                .split("```cpp\n")
                .nth(1)
                .unwrap()
                .split("\n```\n")
                .next()
                .unwrap();
            assert!(body.chars().count() <= 9000);
        }
    }
}
