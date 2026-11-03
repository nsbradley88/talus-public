//! `ifeval`: IFEval instruction following (google/IFEval, Apache-2.0). The checkers reimplement the official
//! rules of google-research `instruction_following_eval` (Apache-2.0, commit 0f585cca) instruction by
//! instruction; the case passes when every instruction of the prompt is followed in strict mode (the official
//! prompt-level strict accuracy). Loose mode (the official variants: asterisks removed, first and/or last line
//! dropped) and per-instruction results are kept in the details.
//!
//! Five of the 25 instruction types depend on external models in the official code (nltk punkt sentence
//! splitting, nltk word_tokenize, langdetect) and cannot be reproduced exactly: `length_constraints:
//! number_sentences`, `change_case:capital_word_frequency`, `change_case:english_capital`,
//! `change_case:english_lowercase`, `language:response_language`. Prompts that use any of them are excluded
//! from the pinned subset (387 of 541 prompts) instead of being scored approximately.
//!
//! Patterns the official code builds from data (keywords, section splitters, postscript markers) are used as
//! regular expressions, as there; Python and Rust regex syntax agree for the strings in the dataset.

use super::{take_limit, Task, TaskContext};
use crate::case::{Case, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::{self, DatasetPin};
use regex::{Regex, RegexBuilder};
use serde_json::{json, Value};

pub struct IfevalTask;

pub const SUPPORTED: &[&str] = &[
    "combination:repeat_prompt",
    "combination:two_responses",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:constrained_response",
    "detectable_format:json_format",
    "detectable_format:multiple_sections",
    "detectable_format:number_bullet_lists",
    "detectable_format:number_highlighted_sections",
    "detectable_format:title",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "keywords:letter_frequency",
    "length_constraints:nth_paragraph_first_word",
    "length_constraints:number_paragraphs",
    "length_constraints:number_words",
    "punctuation:no_comma",
    "startend:end_checker",
    "startend:quotation",
];

impl Task for IfevalTask {
    fn id(&self) -> &str {
        "ifeval"
    }

    fn scorer_version(&self) -> &'static str {
        "ifeval/1"
    }

    fn datasets(&self, _tier: Tier) -> Vec<&'static DatasetPin> {
        vec![datasets::pin("ifeval-supported").expect("pinned")]
    }

    fn default_limit(&self, tier: Tier) -> Option<usize> {
        (tier == Tier::Smoke).then_some(30)
    }

    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String> {
        let d = self.datasets(ctx.tier)[0];
        let text = datasets::load(d, ctx.cache)?;
        let mut out = Vec::new();
        for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            let row: Value =
                serde_json::from_str(line).map_err(|e| format!("{} row {i}: {e}", d.id))?;
            let key = row["key"]
                .as_i64()
                .ok_or(format!("{} row {i}: no key", d.id))?;
            let prompt = row["prompt"].as_str().unwrap_or_default();
            for id in row["instruction_id_list"].as_array().into_iter().flatten() {
                let id = id.as_str().unwrap_or_default();
                if !SUPPORTED.contains(&id) {
                    return Err(format!("{} key {key}: unsupported instruction {id}", d.id));
                }
            }
            out.push(Case {
                task: self.id().into(),
                case_id: format!("ifeval/{key}"),
                dataset: d.id.into(),
                dataset_revision: format!("{}@{}", d.revision, &d.sha256[..12]),
                messages: vec![json!({"role": "user", "content": prompt})],
                expected: json!({"instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"]}),
                meta: Value::Null,
                critical: false,
                max_tokens: Some(1280),
                tools: Value::Null,
                request: Value::Null,
            });
        }
        Ok(take_limit(out, ctx.limit))
    }

    fn score(&self, case: &Case, out: &RawOutput) -> Score {
        let prompt = case.messages[0]["content"].as_str().unwrap_or_default();
        let ids = case.expected["instruction_id_list"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let kwargs = case.expected["kwargs"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let resp = out.content.as_str();
        let variants = loose_variants(resp);
        let mut strict = Vec::new();
        let mut loose = Vec::new();
        let mut errors = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            let id = id.as_str().unwrap_or_default();
            let kw = kwargs.get(i).cloned().unwrap_or(Value::Null);
            let check = |r: &str| -> Result<bool, String> {
                Ok(!r.trim().is_empty() && check_following(id, &kw, prompt, r)?)
            };
            match check(resp) {
                Ok(s) => strict.push(s),
                Err(e) => {
                    errors.push(format!("{id}: {e}"));
                    strict.push(false);
                }
            }
            loose.push(variants.iter().any(|v| check(v).unwrap_or(false)));
        }
        let pass = !strict.is_empty() && strict.iter().all(|x| *x);
        Score::binary(
            pass,
            Value::Null,
            json!({
                "stage": if pass { "pass" } else { "not_followed" },
                "instructions": ids, "strict": strict, "loose": loose,
                "loose_pass": !loose.is_empty() && loose.iter().all(|x| *x),
                "errors": errors,
            }),
        )
    }
}

/// The official loose-mode variants of a response.
pub fn loose_variants(r: &str) -> Vec<String> {
    // Python: r = response.split("\n"); the dropped-line variants are .strip()ped
    let lines: Vec<&str> = r.split('\n').collect();
    let n = lines.len();
    let remove_first = lines[1.min(n)..].join("\n").trim().to_string();
    let remove_last = lines[..n.saturating_sub(1)].join("\n").trim().to_string();
    let remove_both = if n >= 2 {
        lines[1..n - 1].join("\n").trim().to_string()
    } else {
        String::new()
    };
    vec![
        r.to_string(),
        r.replace('*', ""),
        remove_first.clone(),
        remove_last.clone(),
        remove_both.clone(),
        remove_first.replace('*', ""),
        remove_last.replace('*', ""),
        remove_both.replace('*', ""),
    ]
}

fn re_ci(p: &str) -> Result<Regex, String> {
    RegexBuilder::new(p)
        .case_insensitive(true)
        .build()
        .map_err(|e| format!("pattern {p:?}: {e}"))
}

fn re(p: &str) -> Result<Regex, String> {
    Regex::new(p).map_err(|e| format!("pattern {p:?}: {e}"))
}

fn kw_i64(kw: &Value, k: &str) -> Result<i64, String> {
    kw[k]
        .as_i64()
        .ok_or(format!("missing integer argument {k}"))
}

fn kw_str<'a>(kw: &'a Value, k: &str) -> Result<&'a str, String> {
    kw[k].as_str().ok_or(format!("missing string argument {k}"))
}

fn relation(kw: &Value, k: &str, n: i64, threshold: i64) -> Result<bool, String> {
    match kw_str(kw, k)? {
        "less than" => Ok(n < threshold),
        "at least" => Ok(n >= threshold),
        r => Err(format!("unknown relation {r:?}")),
    }
}

/// Python str.isupper/islower semantics are not needed (those types are excluded); helpers below mirror the
/// Python operations each supported checker uses.
fn py_strip(s: &str) -> &str {
    s.trim()
}

/// Number of non-overlapping matches.
fn count(re: &Regex, s: &str) -> usize {
    re.find_iter(s).count()
}

/// One instruction's strict check (the official `check_following`).
pub fn check_following(id: &str, kw: &Value, prompt: &str, value: &str) -> Result<bool, String> {
    Ok(match id {
        "punctuation:no_comma" => !value.contains(','),
        "detectable_content:number_placeholders" => {
            count(&re(r"\[.*?\]")?, value) as i64 >= kw_i64(kw, "num_placeholders")?
        }
        "detectable_format:number_bullet_lists" => {
            let a = count(&re(r"(?m)^\s*\*[^\*].*$")?, value);
            let b = count(&re(r"(?m)^\s*-.*$")?, value);
            (a + b) as i64 == kw_i64(kw, "num_bullets")?
        }
        "detectable_format:constrained_response" => {
            let v = py_strip(value);
            [
                "My answer is yes.",
                "My answer is no.",
                "My answer is maybe.",
            ]
            .iter()
            .any(|o| v.contains(o))
        }
        "detectable_format:number_highlighted_sections" => {
            let mut n = 0i64;
            for m in re(r"\*[^\n\*]*\*")?.find_iter(value) {
                if !m.as_str().trim_matches('*').trim().is_empty() {
                    n += 1;
                }
            }
            for m in re(r"\*\*[^\n\*]*\*\*")?.find_iter(value) {
                let t = m.as_str();
                let inner = t.strip_prefix("**").unwrap_or(t);
                let inner = inner.strip_suffix("**").unwrap_or(inner);
                if !inner.trim().is_empty() {
                    n += 1;
                }
            }
            n >= kw_i64(kw, "num_highlights")?
        }
        "detectable_format:multiple_sections" => {
            let sp = kw_str(kw, "section_spliter")?.trim();
            let pat = re(&format!(r"\s?{sp}\s?\d+\s?"))?;
            // len(re.split(...)) - 1 == number of matches
            count(&pat, value) as i64 >= kw_i64(kw, "num_sections")?
        }
        "length_constraints:number_paragraphs" => {
            let parts: Vec<&str> = re(r"\s?\*\*\*\s?")?.split(value).collect();
            let mut n = parts.len() as i64;
            for (i, p) in parts.iter().enumerate() {
                if p.trim().is_empty() {
                    if i == 0 || i == parts.len() - 1 {
                        n -= 1;
                    } else {
                        return Ok(false);
                    }
                }
            }
            n == kw_i64(kw, "num_paragraphs")?
        }
        "detectable_content:postscript" => {
            let marker = kw_str(kw, "postscript_marker")?.trim();
            let v = value.to_lowercase();
            let pat = match marker {
                "P.P.S" => r"(?m)\s*p\.\s?p\.\s?s.*$".to_string(),
                "P.S." => r"(?m)\s*p\.\s?s\..*$".to_string(),
                m => format!(r"(?m)\s*{}.*$", m.to_lowercase()),
            };
            re(&pat)?.is_match(&v)
        }
        "keywords:existence" => {
            for k in kw["keywords"].as_array().ok_or("missing keywords")? {
                if !re_ci(k.as_str().unwrap_or_default())?.is_match(value) {
                    return Ok(false);
                }
            }
            true
        }
        "keywords:frequency" => {
            let k = kw_str(kw, "keyword")?.trim();
            let n = count(&re_ci(k)?, value) as i64;
            relation(kw, "relation", n, kw_i64(kw, "frequency")?)?
        }
        "keywords:forbidden_words" => {
            for w in kw["forbidden_words"]
                .as_array()
                .ok_or("missing forbidden_words")?
            {
                let w = w.as_str().unwrap_or_default();
                if re_ci(&format!(r"\b{w}\b"))?.is_match(value) {
                    return Ok(false);
                }
            }
            true
        }
        "keywords:letter_frequency" => {
            let letter = kw_str(kw, "letter")?.trim().to_lowercase();
            let c = letter.chars().next().ok_or("empty letter")?;
            let n = value.to_lowercase().chars().filter(|x| *x == c).count() as i64;
            relation(kw, "let_relation", n, kw_i64(kw, "let_frequency")?)?
        }
        "length_constraints:number_words" => {
            // nltk RegexpTokenizer(r"\w+")
            let n = count(&re(r"\w+")?, value) as i64;
            relation(kw, "relation", n, kw_i64(kw, "num_words")?)?
        }
        "detectable_format:json_format" => {
            let v = py_strip(value);
            let v = v.strip_prefix("```json").unwrap_or(v);
            let v = v.strip_prefix("```Json").unwrap_or(v);
            let v = v.strip_prefix("```JSON").unwrap_or(v);
            let v = v.strip_prefix("```").unwrap_or(v);
            let v = v.strip_suffix("```").unwrap_or(v);
            serde_json::from_str::<Value>(v.trim()).is_ok()
        }
        "length_constraints:nth_paragraph_first_word" => {
            let paragraphs: Vec<&str> = value.split("\n\n").collect();
            let num = paragraphs.iter().filter(|p| !p.trim().is_empty()).count() as i64;
            let nth = kw_i64(kw, "nth_paragraph")?;
            if nth > num || nth < 1 {
                return Ok(false);
            }
            let paragraph = paragraphs[(nth - 1) as usize].trim();
            if paragraph.is_empty() {
                return Ok(false);
            }
            let word = paragraph
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .trim();
            let word = word.trim_start_matches('\'').trim_start_matches('"');
            let mut first = String::new();
            for ch in word.chars() {
                if ['.', ',', '?', '!', '\'', '"'].contains(&ch) {
                    break;
                }
                first.extend(ch.to_lowercase());
            }
            num == kw_i64(kw, "num_paragraphs")?
                && first == kw_str(kw, "first_word")?.to_lowercase()
        }
        "combination:two_responses" => {
            let parts: Vec<&str> = value.split("******").collect();
            let mut valid = Vec::new();
            for (i, p) in parts.iter().enumerate() {
                if p.trim().is_empty() {
                    if i != 0 && i != parts.len() - 1 {
                        return Ok(false);
                    }
                } else {
                    valid.push(*p);
                }
            }
            valid.len() == 2 && valid[0].trim() != valid[1].trim()
        }
        "combination:repeat_prompt" => {
            let p = kw["prompt_to_repeat"].as_str().unwrap_or(prompt);
            value
                .trim()
                .to_lowercase()
                .starts_with(&p.trim().to_lowercase())
        }
        "startend:end_checker" => {
            let v = value.trim().trim_matches('"').to_lowercase();
            v.ends_with(&kw_str(kw, "end_phrase")?.trim().to_lowercase())
        }
        "detectable_format:title" => re(r"<<[^\n]+>>")?.find_iter(value).any(|m| {
            !m.as_str()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .trim()
                .is_empty()
        }),
        "startend:quotation" => {
            let v = value.trim();
            v.chars().count() > 1 && v.starts_with('"') && v.ends_with('"')
        }
        other => return Err(format!("instruction {other} is not supported")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ck(id: &str, kw: Value, v: &str) -> bool {
        check_following(id, &kw, "PROMPT", v).unwrap()
    }

    #[test]
    fn checkers_follow_the_official_rules() {
        assert!(ck("punctuation:no_comma", json!({}), "no commas here"));
        assert!(!ck("punctuation:no_comma", json!({}), "a, b"));
        assert!(ck(
            "detectable_content:number_placeholders",
            json!({"num_placeholders": 2}),
            "[a] and [b]"
        ));
        assert!(!ck(
            "detectable_content:number_placeholders",
            json!({"num_placeholders": 3}),
            "[a] and [b]"
        ));
        assert!(ck(
            "detectable_format:number_bullet_lists",
            json!({"num_bullets": 3}),
            "* one\n* two\n- three"
        ));
        assert!(!ck(
            "detectable_format:number_bullet_lists",
            json!({"num_bullets": 2}),
            "* one\n* two\n- three"
        ));
        assert!(!ck(
            "detectable_format:number_bullet_lists",
            json!({"num_bullets": 1}),
            "**bold** line"
        ));
        assert!(ck(
            "detectable_format:constrained_response",
            json!({}),
            "  My answer is maybe. "
        ));
        // *a* counts once as single; **b** counts as double and (inner *b*?) the single regex sees "**" pairs
        assert!(ck(
            "detectable_format:number_highlighted_sections",
            json!({"num_highlights": 2}),
            "*a* and *b*"
        ));
        assert!(!ck(
            "detectable_format:number_highlighted_sections",
            json!({"num_highlights": 2}),
            "*a* and ** "
        ));
        assert!(ck(
            "detectable_format:multiple_sections",
            json!({"section_spliter": "Section", "num_sections": 2}),
            "Section 1\nx\nSection 2\ny"
        ));
        assert!(!ck(
            "detectable_format:multiple_sections",
            json!({"section_spliter": "SECTION", "num_sections": 3}),
            "SECTION 1 x SECTION 2 y"
        ));
        assert!(ck(
            "length_constraints:number_paragraphs",
            json!({"num_paragraphs": 3}),
            "a\n***\nb\n***\nc"
        ));
        assert!(!ck(
            "length_constraints:number_paragraphs",
            json!({"num_paragraphs": 3}),
            "a\n***\n\n***\nc"
        ));
        assert!(ck(
            "length_constraints:number_paragraphs",
            json!({"num_paragraphs": 2}),
            "***\na\n***\nb\n***"
        ));
        assert!(ck(
            "detectable_content:postscript",
            json!({"postscript_marker": "P.S."}),
            "Hi\nP.S. bye"
        ));
        assert!(ck(
            "detectable_content:postscript",
            json!({"postscript_marker": "P.P.S"}),
            "Hi\np. p. s bye"
        ));
        assert!(!ck(
            "detectable_content:postscript",
            json!({"postscript_marker": "P.S."}),
            "Hi"
        ));
        assert!(ck(
            "keywords:existence",
            json!({"keywords": ["Rust", "safety"]}),
            "rust gives SAFETY"
        ));
        assert!(!ck(
            "keywords:existence",
            json!({"keywords": ["Rust", "speed"]}),
            "rust gives safety"
        ));
        assert!(ck(
            "keywords:frequency",
            json!({"keyword": "cat", "frequency": 2, "relation": "at least"}),
            "cat Cat dog"
        ));
        assert!(ck(
            "keywords:frequency",
            json!({"keyword": "cat", "frequency": 2, "relation": "less than"}),
            "cat dog"
        ));
        assert!(!ck(
            "keywords:forbidden_words",
            json!({"forbidden_words": ["bad"]}),
            "this is Bad."
        ));
        assert!(ck(
            "keywords:forbidden_words",
            json!({"forbidden_words": ["bad"]}),
            "badge is fine"
        ));
        assert!(ck(
            "keywords:letter_frequency",
            json!({"letter": "A", "let_frequency": 3, "let_relation": "at least"}),
            "banana"
        ));
        assert!(ck(
            "length_constraints:number_words",
            json!({"num_words": 3, "relation": "less than"}),
            "two words"
        ));
        assert!(ck(
            "length_constraints:number_words",
            json!({"num_words": 4, "relation": "at least"}),
            "it's a well-known fact"
        ));
        assert!(ck(
            "detectable_format:json_format",
            json!({}),
            "```json\n{\"a\": 1}\n```"
        ));
        assert!(!ck("detectable_format:json_format", json!({}), "{a: 1}"));
        assert!(ck(
            "length_constraints:nth_paragraph_first_word",
            json!({"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "Then"}),
            "First.\n\n\"Then, more."
        ));
        assert!(!ck(
            "length_constraints:nth_paragraph_first_word",
            json!({"num_paragraphs": 3, "nth_paragraph": 2, "first_word": "then"}),
            "First.\n\nThen."
        ));
        assert!(ck(
            "combination:two_responses",
            json!({}),
            "one\n******\ntwo"
        ));
        assert!(!ck(
            "combination:two_responses",
            json!({}),
            "one\n******\none"
        ));
        assert!(ck(
            "combination:repeat_prompt",
            json!({"prompt_to_repeat": "Say hi."}),
            "say hi. Hi!"
        ));
        assert!(ck(
            "startend:end_checker",
            json!({"end_phrase": "Any other questions?"}),
            "Done. Any other questions?\""
        ));
        assert!(ck(
            "detectable_format:title",
            json!({}),
            "<<My Title>>\ntext"
        ));
        assert!(!ck("detectable_format:title", json!({}), "<< >>\ntext"));
        assert!(ck("startend:quotation", json!({}), " \"quoted\" "));
        assert!(!ck("startend:quotation", json!({}), "\""));
        assert!(check_following("language:response_language", &json!({}), "", "x").is_err());
    }

    #[test]
    fn loose_mode_and_cases() {
        let t = IfevalTask;
        let cs = t.cases(&TaskContext::for_tests(Tier::Full)).unwrap();
        assert_eq!(cs.len(), 387);
        let ids: std::collections::BTreeSet<_> = cs.iter().map(|c| &c.case_id).collect();
        assert_eq!(ids.len(), cs.len());
        // a case with no_comma: strict fails on a leading line with a comma, loose (first line removed) passes
        let c = cs
            .iter()
            .find(|c| c.expected["instruction_id_list"] == json!(["punctuation:no_comma"]))
            .expect("a no_comma-only case");
        let out = RawOutput {
            content: "Sure, here it is:\nno commas at all".into(),
            finish_reason: "stop".into(),
            ..Default::default()
        };
        let s = t.score(c, &out);
        assert!(!s.pass);
        assert_eq!(s.details["loose_pass"], true);
        let empty = RawOutput {
            content: "  ".into(),
            ..Default::default()
        };
        assert!(
            !t.score(c, &empty).pass,
            "an empty response follows nothing"
        );
        assert_eq!(
            t.cases(&TaskContext {
                limit: t.default_limit(Tier::Smoke),
                ..TaskContext::for_tests(Tier::Smoke)
            })
            .unwrap()
            .len(),
            30
        );
    }
}
