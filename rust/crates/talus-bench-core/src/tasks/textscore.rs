//! `perplexity` and `kld_topk`: teacher-forced scoring of pinned text through talus-server's
//! `POST /v1/talus/score` (server extension E2, bench-design/server-extensions.md). Teacher forcing makes both
//! endpoints see exactly the same contexts, so no sampling noise enters (E1's sampled logprobs are not used).
//!
//! Cases (built per endpoint, `needs_server`): the text (the embedded public-domain prose `pd-prose-v1`, or the
//! file in `params.text`) is tokenized once with the server's `/tokenize`; window `i` is the prompt
//! `tokens[i*M .. i*M+P]` and the continuation `tokens[i*M+P .. i*M+P+M]`, so every scored token has P tokens of
//! context and no token is scored twice. The ids go into the case's request, so the prompt hash covers them: two
//! endpoints whose tokenizers differ produce cases that never pair (a PPL ratio over different tokens would be
//! meaningless). Before the cases are built, one tiny probe request checks that the endpoint can score: a 501
//! ("scoring not supported by this engine"), a 400 "... not supported by this engine" or a 404 (no such route)
//! make the task NOT_SUPPORTED, not NOT_RUN.
//!
//! - `perplexity`: per case the NLL sum, mean NLL per token and PPL = exp(mean NLL); the gate compares the
//!   candidate's token-weighted PPL with the reference's (ratio, `quality_gates.ppl_ratio`).
//! - `kld_topk`: the same windows scored with `top_k` = K (default 20); each row keeps every position's forced
//!   token, logprob and top-k list, and the gate computes the screening KL(reference || candidate) per position
//!   from the two lists with the tail mass in one bucket (`kl::topk_kl`), top-1 agreement and |Δlogprob| of the
//!   forced token (`quality_gates.kld_topk`, WARN-only by default). The id `kld` stays the full-vocabulary KL,
//!   which HTTP cannot provide (GPU-host mode `kld_teacher_forced`).
//!
//! Params (`benchmarks.params.perplexity` / `.kld_topk`): `windows` (default perplexity 32 / 64 / 96 and
//! kld_topk 16 / 32 / 64 for smoke / full / qualification; fewer when the text is shorter), `prompt_tokens`
//! (P, default 256), `continuation_tokens` (M, default 256, at most 4096), `top_k` (kld_topk only, 1..=64,
//! default 20), `text` (a UTF-8 file to score instead of the embedded text), `timeout_s` (per request, default
//! 600).

use super::{take_limit, ExecEnv, ExecOutcome, Task, TaskContext, NOT_SUPPORTED};
use crate::case::{Case, CaseStatus, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::{self, DatasetPin};
use crate::http::{error_message, unsupported_reason, Client, HttpError};
use crate::util::sha256_hex;
use serde_json::{json, Value};

pub struct PerplexityTask;
pub struct KldTopkTask;

/// The embedded public-domain text.
pub const TEXT_PIN: &str = "pd-prose-v1";
pub const DEFAULT_PROMPT_TOKENS: usize = 256;
pub const DEFAULT_CONTINUATION_TOKENS: usize = 256;
pub const DEFAULT_TOP_K: usize = 20;
/// talus-server's E2 limits.
pub const MAX_TOP_K: usize = 64;
pub const MAX_CONTINUATION: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Plan {
    prompt: usize,
    cont: usize,
    windows: usize,
    top_k: usize,
}

fn param_usize(params: &Value, key: &str) -> Result<Option<usize>, String> {
    match &params[key] {
        Value::Null => Ok(None),
        v => v
            .as_u64()
            .map(|n| Some(n as usize))
            .ok_or_else(|| format!("params.{key} must be a non-negative integer")),
    }
}

fn plan(task: &str, ctx: &TaskContext) -> Result<Plan, String> {
    let kld = task == "kld_topk";
    let windows = param_usize(ctx.params, "windows")?.unwrap_or(match (kld, ctx.tier) {
        (false, Tier::Smoke) => 32,
        (false, Tier::Full) => 64,
        (false, Tier::Qualification) => 96,
        (true, Tier::Smoke) => 16,
        (true, Tier::Full) => 32,
        (true, Tier::Qualification) => 64,
    });
    let prompt = param_usize(ctx.params, "prompt_tokens")?.unwrap_or(DEFAULT_PROMPT_TOKENS);
    let cont =
        param_usize(ctx.params, "continuation_tokens")?.unwrap_or(DEFAULT_CONTINUATION_TOKENS);
    let top_k = if kld {
        param_usize(ctx.params, "top_k")?.unwrap_or(DEFAULT_TOP_K)
    } else {
        0
    };
    if windows == 0 || prompt == 0 || cont == 0 {
        return Err(format!(
            "{task}: windows, prompt_tokens and continuation_tokens must be > 0"
        ));
    }
    if cont > MAX_CONTINUATION {
        return Err(format!(
            "{task}: continuation_tokens {cont} exceeds the server's scoring limit {MAX_CONTINUATION}"
        ));
    }
    if kld && !(1..=MAX_TOP_K).contains(&top_k) {
        return Err(format!("{task}: top_k must be in 1..={MAX_TOP_K}"));
    }
    Ok(Plan {
        prompt,
        cont,
        windows,
        top_k,
    })
}

/// The text to score: (text, dataset id, revision, source).
fn text(ctx: &TaskContext) -> Result<(String, String, String, String), String> {
    if let Some(p) = ctx.params["text"].as_str() {
        let t = std::fs::read_to_string(p).map_err(|e| format!("text {p}: {e}"))?;
        let sha = sha256_hex(t.as_bytes());
        return Ok((t, "file".into(), format!("file@{}", &sha[..12]), p.into()));
    }
    let pin = datasets::pin(TEXT_PIN).ok_or("pd-prose-v1 pin missing")?;
    let t = datasets::load(pin, ctx.cache)?;
    Ok((
        t,
        TEXT_PIN.into(),
        format!("{TEXT_PIN}@{}", &pin.sha256[..12]),
        pin.source.into(),
    ))
}

/// Probe whether the endpoint can score at all (one 2-token request). Err starts with NOT_SUPPORTED when the
/// server or engine lacks the feature.
pub fn probe(task: &str, client: &Client, a: u64, b: u64) -> Result<(), String> {
    let r = client
        .score(
            &json!({"prompt_ids": [a], "continuation_ids": [b], "top_k": 0, "return": ["logprob"]}),
        )
        .map_err(|e| format!("{task}: /v1/talus/score: {e}"))?;
    if let Some(why) = unsupported_reason(&r) {
        return Err(format!(
            "{NOT_SUPPORTED} {task} needs teacher-forced scoring (POST /v1/talus/score) with engine support: {why}"
        ));
    }
    if r.status != 200 {
        return Err(format!(
            "{task}: /v1/talus/score probe answered {}: {}",
            r.status,
            error_message(&r.body)
        ));
    }
    if r.body["positions"].as_array().map(Vec::len) != Some(1) {
        return Err(format!(
            "{task}: /v1/talus/score probe: reply without one position"
        ));
    }
    Ok(())
}

fn build_cases(task: &str, ctx: &TaskContext) -> Result<Vec<Case>, String> {
    let client = ctx
        .client
        .ok_or_else(|| format!("{task}: needs the server (tokenizer and /v1/talus/score)"))?;
    let p = plan(task, ctx)?;
    let (text, dataset, rev, source) = text(ctx)?;
    let r = client
        .post("/tokenize", &json!({"content": text}))
        .map_err(|e| format!("{task}: /tokenize: {e}"))?;
    if r.status != 200 {
        return Err(format!(
            "{task}: /tokenize answered {}: {}",
            r.status,
            error_message(&r.body)
        ));
    }
    let toks: Vec<u64> = r.body["tokens"]
        .as_array()
        .ok_or_else(|| format!("{task}: /tokenize reply without tokens"))?
        .iter()
        .map(|t| t.as_u64().or_else(|| t["id"].as_u64()))
        .collect::<Option<Vec<u64>>>()
        .ok_or_else(|| format!("{task}: /tokenize returned non-integer tokens"))?;
    if toks.len() < 2 {
        return Err(format!("{task}: the text has fewer than 2 tokens"));
    }
    probe(task, client, toks[0], toks[1])?;
    let ctx_max = client
        .get("/v1/talus/status")
        .ok()
        .filter(|r| r.status == 200)
        .and_then(|r| r.body["context"]["max"].as_u64())
        .filter(|m| *m > 0);
    if let Some(m) = ctx_max {
        if (p.prompt + p.cont) as u64 > m {
            return Err(format!(
                "{task}: prompt_tokens + continuation_tokens ({}) exceed the served context {m}",
                p.prompt + p.cont
            ));
        }
    }
    let fit = toks.len().saturating_sub(p.prompt) / p.cont;
    let n = p.windows.min(fit);
    if n == 0 {
        return Err(format!(
            "{task}: the text has {} tokens, fewer than one window ({} + {})",
            toks.len(),
            p.prompt,
            p.cont
        ));
    }
    let ret: Vec<&str> = if p.top_k > 0 {
        vec!["logprob", "top_k"]
    } else {
        vec!["logprob"]
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let s = i * p.cont;
        out.push(Case {
            task: task.into(),
            case_id: format!("{task}/w{i:03}"),
            dataset: dataset.clone(),
            dataset_revision: rev.clone(),
            messages: Vec::new(),
            expected: json!({"window": i, "offset_tokens": s, "prompt_tokens": p.prompt,
                             "continuation_tokens": p.cont, "text_tokens": toks.len(),
                             "windows_requested": p.windows, "source": source}),
            meta: Value::Null,
            critical: false,
            max_tokens: None,
            tools: Value::Null,
            request: json!({"prompt_ids": &toks[s..s + p.prompt],
                            "continuation_ids": &toks[s + p.prompt..s + p.prompt + p.cont],
                            "top_k": p.top_k, "return": ret}),
        });
    }
    Ok(take_limit(out, ctx.limit))
}

/// One scored position: the forced id, its logprob and the top-k list `(id, logprob)`.
pub type Position = (u64, f64, Vec<(u64, f64)>);

/// A score reply checked against the case: per position (forced id, logprob, top-k list).
pub fn parse_positions(body: &Value, case: &Case) -> Result<Vec<Position>, String> {
    let want: Vec<u64> = case.request["continuation_ids"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default();
    let pos = body["positions"]
        .as_array()
        .ok_or("score reply without positions")?;
    if pos.len() != want.len() {
        return Err(format!(
            "the server scored {} positions for {} continuation tokens",
            pos.len(),
            want.len()
        ));
    }
    let mut out = Vec::with_capacity(pos.len());
    for (j, (p, w)) in pos.iter().zip(&want).enumerate() {
        let id = p["id"].as_u64().ok_or("position without id")?;
        if id != *w {
            return Err(format!(
                "position {j}: the server scored id {id}, the case asked for {w}"
            ));
        }
        let lp = p["logprob"]
            .as_f64()
            .ok_or_else(|| format!("position {j} without logprob"))?;
        let top = p["top_k"]
            .as_array()
            .map(|t| {
                t.iter()
                    .map(|e| Some((e["id"].as_u64()?, e["logprob"].as_f64()?)))
                    .collect::<Option<Vec<_>>>()
            })
            .unwrap_or(Some(Vec::new()))
            .ok_or_else(|| format!("position {j}: malformed top_k entry"))?;
        out.push((id, lp, top));
    }
    Ok(out)
}

/// NLL summary of the forced tokens (a logprob of -9999 is the server's minus infinity; it counts as 9999 nats).
pub fn nll_summary(lps: &[f64]) -> Value {
    let n = lps.len();
    let nll: f64 = lps.iter().map(|lp| -lp.max(-9999.0)).sum();
    let mean = if n > 0 { nll / n as f64 } else { 0.0 };
    json!({"nll_sum": nll, "tokens": n, "mean_nll": mean, "ppl": mean.exp(),
           "neg_inf_tokens": lps.iter().filter(|lp| **lp <= -9999.0).count()})
}

fn score_rows(case: &Case, body: &Value, kld: bool) -> Result<Score, String> {
    let pos = parse_positions(body, case)?;
    let lps: Vec<f64> = pos.iter().map(|p| p.1).collect();
    let mut parsed = nll_summary(&lps);
    parsed["lp"] = json!(lps);
    let mut details = json!({"stage": "scored", "server_timings": body["timings"],
                             "prompt_tokens": body["prompt_tokens"]});
    if kld {
        let want = case.request["top_k"].as_u64().unwrap_or(0) as usize;
        if pos.iter().any(|p| p.2.len() != want) {
            return Err(format!(
                "the server returned top-k lists of another size than {want}"
            ));
        }
        parsed["ids"] = json!(pos.iter().map(|p| p.0).collect::<Vec<_>>());
        // flat [id, logprob, id, logprob, ...] per position keeps cases.jsonl compact
        parsed["top"] = json!(pos
            .iter()
            .map(|p| p
                .2
                .iter()
                .flat_map(|&(i, l)| [json!(i), json!(l)])
                .collect::<Vec<_>>())
            .collect::<Vec<_>>());
        let cover: Vec<f64> = pos
            .iter()
            .map(|p| p.2.iter().map(|x| x.1.exp()).sum::<f64>().min(1.0))
            .collect();
        details["top_k"] = json!(want);
        details["covered_mean"] = json!(cover.iter().sum::<f64>() / cover.len().max(1) as f64);
        details["covered_min"] = json!(cover.iter().copied().fold(1.0, f64::min));
    }
    Ok(Score {
        pass: true,
        value: parsed["mean_nll"].as_f64().unwrap_or(0.0),
        parsed,
        details,
    })
}

/// The top-k rows of a scored kld_topk case (`score.parsed`), for the pairing.
pub fn topk_rows(parsed: &Value) -> Option<Vec<crate::kl::TopkRow>> {
    let ids = parsed["ids"].as_array()?;
    let lps = parsed["lp"].as_array()?;
    let top = parsed["top"].as_array()?;
    if ids.len() != lps.len() || ids.len() != top.len() {
        return None;
    }
    ids.iter()
        .zip(lps)
        .zip(top)
        .map(|((i, l), t)| {
            let flat = t.as_array()?;
            let list = flat
                .chunks(2)
                .map(|c| Some((c.first()?.as_u64()? as u32, c.get(1)?.as_f64()?)))
                .collect::<Option<Vec<_>>>()?;
            Some(crate::kl::TopkRow {
                id: i.as_u64()? as u32,
                lp: l.as_f64()?,
                top: list,
            })
        })
        .collect()
}

fn execute_case(case: &Case, env: &ExecEnv, kld: bool) -> ExecOutcome {
    let timeout = env.params["timeout_s"].as_u64().unwrap_or(600);
    let t0 = std::time::Instant::now();
    let fail = |status: CaseStatus, e: String| ExecOutcome {
        status,
        raw: None,
        score: Score::binary(false, Value::Null, json!({"not_scored": status.as_str()})),
        error: Some(e),
        wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
    };
    let client = match Client::new(&env.base_url, env.api_key.clone(), timeout) {
        Ok(c) => c,
        Err(e) => return fail(CaseStatus::ConnectError, e),
    };
    let r = match client.score(&case.request) {
        Ok(r) => r,
        Err(HttpError::Timeout(m)) => return fail(CaseStatus::Timeout, m),
        Err(HttpError::Connect(m)) => return fail(CaseStatus::ConnectError, m),
        Err(HttpError::Protocol(m)) => return fail(CaseStatus::ProtocolError, m),
    };
    if r.status != 200 {
        let msg = match unsupported_reason(&r) {
            Some(why) => format!("{NOT_SUPPORTED} {why}"),
            None => format!("HTTP {}: {}", r.status, error_message(&r.body)),
        };
        let st = if r.status >= 500 {
            CaseStatus::ServerError
        } else {
            CaseStatus::HttpError
        };
        return fail(st, msg);
    }
    match score_rows(case, &r.body, kld) {
        Ok(score) => ExecOutcome {
            status: CaseStatus::Ok,
            raw: None,
            score,
            error: None,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
        },
        Err(e) => fail(CaseStatus::ProtocolError, e),
    }
}

fn not_scored_here() -> Score {
    Score::binary(
        false,
        Value::Null,
        json!({"stage": "not_scored", "note": "teacher-forced tasks are scored when they execute"}),
    )
}

impl Task for PerplexityTask {
    fn id(&self) -> &str {
        "perplexity"
    }
    fn scorer_version(&self) -> &'static str {
        "perplexity/1"
    }
    fn datasets(&self, _tier: Tier) -> Vec<&'static DatasetPin> {
        datasets::pin(TEXT_PIN).into_iter().collect()
    }
    fn needs_server(&self) -> bool {
        true
    }
    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String> {
        build_cases(self.id(), ctx)
    }
    fn score(&self, _case: &Case, _out: &RawOutput) -> Score {
        not_scored_here()
    }
    fn execute(&self, case: &Case, env: &ExecEnv) -> Option<ExecOutcome> {
        Some(execute_case(case, env, false))
    }
    fn caveats(&self) -> Vec<String> {
        vec!["perplexity: teacher-forced through POST /v1/talus/score on pinned public-domain prose (no sampling); the ratio is computed only over cases whose token ids are identical on both endpoints (same tokenizer), and absolute PPL depends on the tokenizer, so it is not comparable across model families".into()]
    }
}

impl Task for KldTopkTask {
    fn id(&self) -> &str {
        "kld_topk"
    }
    fn scorer_version(&self) -> &'static str {
        "kld_topk/1"
    }
    fn datasets(&self, _tier: Tier) -> Vec<&'static DatasetPin> {
        datasets::pin(TEXT_PIN).into_iter().collect()
    }
    fn needs_server(&self) -> bool {
        true
    }
    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String> {
        build_cases(self.id(), ctx)
    }
    fn score(&self, _case: &Case, _out: &RawOutput) -> Score {
        not_scored_here()
    }
    fn execute(&self, case: &Case, env: &ExecEnv) -> Option<ExecOutcome> {
        Some(execute_case(case, env, true))
    }
    fn caveats(&self) -> Vec<String> {
        vec!["kld_topk is a screening metric: KL(reference || candidate) from the two top-k lists (default k = 20) with the unseen tail mass in one bucket, a coarse-grained approximation of the full-vocabulary KL; its thresholds are provisional and derived separately from the full-vocabulary release gate (kld_teacher_forced, GPU host), so by default it can only WARN (quality_gates.kld_topk.enforce: false)".into()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_and_bounds() {
        let ctx = TaskContext::for_tests(Tier::Smoke);
        assert_eq!(
            plan("perplexity", &ctx).unwrap(),
            Plan {
                prompt: 256,
                cont: 256,
                windows: 32,
                top_k: 0
            }
        );
        assert_eq!(plan("kld_topk", &ctx).unwrap().top_k, 20);
        assert_eq!(plan("kld_topk", &ctx).unwrap().windows, 16);
        let bad = json!({"top_k": 65});
        let c = TaskContext {
            params: &bad,
            ..TaskContext::for_tests(Tier::Full)
        };
        assert!(plan("kld_topk", &c).unwrap_err().contains("top_k"));
        let bad = json!({"continuation_tokens": 5000});
        let c = TaskContext {
            params: &bad,
            ..TaskContext::for_tests(Tier::Full)
        };
        assert!(plan("perplexity", &c).is_err());
        // without a server the cases cannot be built
        assert!(PerplexityTask.cases(&ctx).is_err());
    }

    #[test]
    fn embedded_text_has_enough_windows() {
        let ctx = TaskContext::for_tests(Tier::Smoke);
        let (t, d, rev, _) = text(&ctx).unwrap();
        assert_eq!(d, TEXT_PIN);
        assert!(rev.starts_with("pd-prose-v1@"));
        // ~4 characters per token for English prose with a BPE tokenizer: enough for the qualification tier's
        // 96 windows of 256 + 256 tokens
        assert!(t.len() / 4 > 256 + 96 * 256, "{}", t.len());
        assert!(!t.contains("Gutenberg"));
    }

    #[test]
    fn parse_and_score() {
        let case = Case {
            task: "kld_topk".into(),
            case_id: "kld_topk/w000".into(),
            dataset: "d".into(),
            dataset_revision: "r".into(),
            messages: vec![],
            expected: Value::Null,
            meta: Value::Null,
            critical: false,
            max_tokens: None,
            tools: Value::Null,
            request: json!({"prompt_ids": [1], "continuation_ids": [5, 6], "top_k": 2}),
        };
        let body = json!({"positions": [
            {"id": 5, "logprob": -0.5, "top_k": [{"id": 5, "logprob": -0.5}, {"id": 9, "logprob": -1.5}]},
            {"id": 6, "logprob": -9999.0, "top_k": [{"id": 7, "logprob": -0.1}, {"id": 8, "logprob": -3.0}]}]});
        let s = score_rows(&case, &body, true).unwrap();
        assert!(s.pass);
        assert_eq!(s.parsed["tokens"], 2);
        assert_eq!(s.parsed["neg_inf_tokens"], 1);
        assert!((s.parsed["nll_sum"].as_f64().unwrap() - 9999.5).abs() < 1e-9);
        let rows = topk_rows(&s.parsed).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].top, vec![(7, -0.1), (8, -3.0)]);
        // the server must score the asked ids, with the asked k
        let mut wrong = body.clone();
        wrong["positions"][1]["id"] = json!(4);
        assert!(score_rows(&case, &wrong, true)
            .unwrap_err()
            .contains("asked for 6"));
        let mut short = body.clone();
        short["positions"][0]["top_k"] = json!([]);
        assert!(score_rows(&case, &short, true).is_err());
        let ppl = score_rows(&case, &body, false).unwrap();
        assert!(ppl.parsed["top"].is_null());
        let s = nll_summary(&[-1.0, -3.0]);
        assert!((s["ppl"].as_f64().unwrap() - 2f64.exp()).abs() < 1e-12);
    }
}
