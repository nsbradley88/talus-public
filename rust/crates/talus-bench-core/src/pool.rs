//! Pool several runs of one task into one estimate. Built for fresh-draw tasks (swebench_verified), whose runs
//! are not comparable instance for instance: per model, the pooled rate over every (run, instance) with a Wilson
//! interval; per candidate, the pooled paired difference to the reference of the same run (paired within a run)
//! with a percentile bootstrap over (run, instance) units. Models are matched by id across runs.

use crate::case::{read_jsonl, CaseResult};
use crate::stats;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

fn rows(dir: &Path, sub: &str, task: &str) -> Vec<CaseResult> {
    read_jsonl(&dir.join(sub).join("cases.jsonl"))
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r.task == task)
        .collect()
}

pub fn pool(
    run_dirs: &[&Path],
    task: &str,
    confidence: f64,
    resamples: u32,
    seed: u64,
) -> Result<Value, String> {
    let mut per_model: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut diffs: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut runs = Vec::new();
    for dir in run_dirs {
        let m: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("manifest.json"))
                .map_err(|e| format!("{}/manifest.json: {e}", dir.display()))?,
        )
        .map_err(|e| e.to_string())?;
        let models = m["models"].as_array().cloned().unwrap_or_default();
        let draw = m["datasets"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|d| d["task"] == task)
            .map(|d| d["provenance"].clone())
            .unwrap_or(Value::Null);
        let ref_id = models
            .first()
            .and_then(|x| x["id"].as_str())
            .unwrap_or("reference")
            .to_string();
        let base = rows(dir, "baseline", task);
        let mut run_models = BTreeMap::new();
        let bmap: BTreeMap<_, _> = base.iter().map(|r| (r.key(), r.pass())).collect();
        let e = per_model.entry(ref_id.clone()).or_default();
        e.0 += base.iter().filter(|r| r.pass()).count();
        e.1 += base.len();
        run_models.insert(
            ref_id.clone(),
            json!([base.iter().filter(|r| r.pass()).count(), base.len()]),
        );
        for c in models.iter().skip(1).filter_map(|x| x["id"].as_str()) {
            let cr = rows(dir, &format!("candidates/{c}"), task);
            let e = per_model.entry(c.to_string()).or_default();
            e.0 += cr.iter().filter(|r| r.pass()).count();
            e.1 += cr.len();
            run_models.insert(
                c.to_string(),
                json!([cr.iter().filter(|r| r.pass()).count(), cr.len()]),
            );
            let d = diffs.entry(c.to_string()).or_default();
            for r in &cr {
                if let Some(bp) = bmap.get(&r.key()) {
                    d.push(*bp as u8 as f64 - r.pass() as u8 as f64);
                }
            }
        }
        if base.is_empty() {
            return Err(format!("{}: no {task} rows", dir.display()));
        }
        runs.push(json!({"dir": dir.display().to_string(), "experiment": m["experiment"]["id"], "seed": draw["seed"],
                         "seed_source": draw["seed_source"], "dataset_source": draw["source"], "models": run_models}));
    }
    let models: serde_json::Map<String, Value> = per_model
        .iter()
        .map(|(k, (s, n))| (k.clone(), json!({"resolved": s, "n": n, "rate": stats::accuracy(*s, *n), "wilson": stats::wilson(*s, *n, confidence)})))
        .collect();
    let paired: serde_json::Map<String, Value> = diffs
        .iter()
        .map(|(k, d)| {
            let mean = if d.is_empty() { None } else { Some(100.0 * d.iter().sum::<f64>() / d.len() as f64) };
            (k.clone(), json!({"pairs": d.len(), "loss_pp": mean, "ci_loss_pp": stats::paired_bootstrap_mean(d, resamples, confidence, seed)}))
        })
        .collect();
    Ok(
        json!({"task": task, "runs": runs, "models": models, "paired_vs_reference": paired, "confidence": confidence}),
    )
}

pub fn pool_md(v: &Value) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "# Pooled {}: {} runs\n",
        v["task"].as_str().unwrap_or("?"),
        v["runs"].as_array().map(Vec::len).unwrap_or(0)
    );
    let _ = writeln!(
        s,
        "| Run | Seed | Source | Results (resolved/n per model) |\n|---|---|---|---|"
    );
    for r in v["runs"].as_array().into_iter().flatten() {
        let _ = writeln!(
            s,
            "| {} | {} ({}) | {} | {} |",
            r["experiment"].as_str().unwrap_or("?"),
            r["seed"],
            r["seed_source"].as_str().unwrap_or("?"),
            r["dataset_source"].as_str().unwrap_or("?"),
            r["models"]
        );
    }
    let conf = v["confidence"].as_f64().unwrap_or(0.95) * 100.0;
    let _ = writeln!(
        s,
        "\n| Model | Resolved | Rate | {conf:.0}% Wilson |\n|---|---|---|---|"
    );
    for (k, m) in v["models"].as_object().into_iter().flatten() {
        let _ = writeln!(
            s,
            "| {k} | {}/{} | {:.1}% | [{:.1}, {:.1}]% |",
            m["resolved"],
            m["n"],
            m["rate"].as_f64().unwrap_or(0.0) * 100.0,
            m["wilson"]["lo"].as_f64().unwrap_or(0.0) * 100.0,
            m["wilson"]["hi"].as_f64().unwrap_or(0.0) * 100.0
        );
    }
    let _ = writeln!(s, "\n| Candidate vs reference (same run) | Pairs | Loss (pp) | {conf:.0}% bootstrap CI |\n|---|---|---|---|");
    for (k, p) in v["paired_vs_reference"].as_object().into_iter().flatten() {
        let _ = writeln!(
            s,
            "| {k} | {} | {} | {} |",
            p["pairs"],
            p["loss_pp"]
                .as_f64()
                .map(|x| format!("{x:.2}"))
                .unwrap_or("n/a".into()),
            p["ci_loss_pp"]
                .as_object()
                .map(|c| format!(
                    "[{:.2}, {:.2}]",
                    c["lo"].as_f64().unwrap_or(0.0),
                    c["hi"].as_f64().unwrap_or(0.0)
                ))
                .unwrap_or("insufficient".into())
        );
    }
    let _ = writeln!(s, "\nEach run is a fresh random draw: instances differ between runs, so only the pooled rates and the within-run paired differences are meaningful.");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{to_jsonl, CaseStatus};
    use crate::compare::tests_support::row;

    fn fake_run(dir: &Path, seed: u64, ref_pass: &[bool], cand_pass: &[bool]) {
        std::fs::create_dir_all(dir.join("baseline")).unwrap();
        std::fs::create_dir_all(dir.join("candidates/fast")).unwrap();
        let mk = |m: &str, p: &[bool]| {
            to_jsonl(
                &p.iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        row(
                            m,
                            "swebench_verified",
                            &format!("{seed}-{i}"),
                            x,
                            CaseStatus::Ok,
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        };
        std::fs::write(dir.join("baseline/cases.jsonl"), mk("balanced", ref_pass)).unwrap();
        std::fs::write(
            dir.join("candidates/fast/cases.jsonl"),
            mk("fast", cand_pass),
        )
        .unwrap();
        std::fs::write(dir.join("manifest.json"), json!({"experiment": {"id": format!("run{seed}")}, "models": [{"id": "balanced"}, {"id": "fast"}],
            "datasets": [{"task": "swebench_verified", "provenance": {"seed": seed, "seed_source": "csprng", "source": "live"}}]}).to_string()).unwrap();
    }

    #[test]
    fn pools_runs() {
        let root = std::env::temp_dir().join(format!("tb-pool-{}", std::process::id()));
        let a = root.join("a");
        let b = root.join("b");
        fake_run(
            &a,
            1,
            &[true, true, false, true],
            &[true, false, false, true],
        );
        fake_run(
            &b,
            2,
            &[true, false, false, false],
            &[true, false, false, false],
        );
        let v = pool(&[&a, &b], "swebench_verified", 0.95, 500, 1).unwrap();
        assert_eq!(v["models"]["balanced"]["resolved"], 4);
        assert_eq!(v["models"]["balanced"]["n"], 8);
        assert_eq!(v["models"]["fast"]["resolved"], 3);
        assert_eq!(v["paired_vs_reference"]["fast"]["pairs"], 8);
        assert!(
            (v["paired_vs_reference"]["fast"]["loss_pp"]
                .as_f64()
                .unwrap()
                - 12.5)
                .abs()
                < 1e-9
        );
        assert_eq!(v["runs"][1]["seed"], 2);
        let md = pool_md(&v);
        assert!(md.contains("| balanced | 4/8 | 50.0% |"), "{md}");
        std::fs::remove_dir_all(&root).ok();
    }
}
