//! SWE-bench Verified instances: a LIVE fetch at run start, a cache only as offline fallback, and a fresh
//! stratified random draw per run.
//!
//! Source: the benchmark's own dataset `SWE-bench/SWE-bench_Verified` (500 instances, the same ids as
//! `princeton-nlp/SWE-bench_Verified`, plus the official per-instance `image`, `eval_script`, `log_parser` and
//! `eval_type` that SWE-bench harness 5.x uses). Rows come from the Hugging Face datasets-server JSON API
//! (`/rows`, 100 per page) with `curl` (talus-bench has no TLS stack); the dataset revision comes from the Hub API.
//! The fetched rows are written to the local cache (never committed) under their content sha256; when the live
//! fetch fails, the newest cached snapshot is used and the report says so.
//!
//! The draw: a seed from the OS CSPRNG (`/dev/urandom`) unless given; instances sorted by id and grouped by repo;
//! `n` split across repos by largest-remainder proportional allocation; within a repo a seeded Fisher-Yates
//! shuffle picks the instances. Same seed + same dataset content = same draw, so `--seed` reproduces any run.

use crate::util::{hash_json, rfc3339, unix_now, Rng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const HF_DATASET: &str = "SWE-bench/SWE-bench_Verified";
pub const ROWS_URL: &str = "https://datasets-server.huggingface.co/rows?dataset=SWE-bench%2FSWE-bench_Verified&config=default&split=test";
pub const API_URL: &str = "https://huggingface.co/api/datasets/SWE-bench/SWE-bench_Verified";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Instance {
    pub instance_id: String,
    pub repo: String,
    pub base_commit: String,
    pub problem_statement: String,
    pub created_at: String,
    pub version: String,
    /// Gold patch: evaluation and memorization checks only; never shown to the agent.
    pub patch: String,
    /// Hidden tests: applied only in the fresh evaluation container.
    pub test_patch: String,
    #[serde(rename = "FAIL_TO_PASS")]
    pub fail_to_pass: Vec<String>,
    #[serde(rename = "PASS_TO_PASS")]
    pub pass_to_pass: Vec<String>,
    pub image: String,
    pub eval_script: String,
    pub log_parser: String,
    pub eval_type: String,
    #[serde(default)]
    pub environment_setup_commit: String,
}

fn str_list(v: &Value) -> Vec<String> {
    match v {
        Value::Array(a) => a
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect(),
        Value::String(s) => serde_json::from_str::<Vec<String>>(s).unwrap_or_default(),
        _ => Vec::new(),
    }
}

impl Instance {
    pub fn from_row(r: &Value) -> Result<Self, String> {
        let s = |k: &str| -> Result<String, String> {
            r[k].as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("swebench row without {k}"))
        };
        Ok(Instance {
            instance_id: s("instance_id")?,
            repo: s("repo")?,
            base_commit: s("base_commit")?,
            problem_statement: s("problem_statement")?,
            created_at: s("created_at")?,
            version: s("version")?,
            patch: s("patch")?,
            test_patch: s("test_patch")?,
            fail_to_pass: str_list(&r["FAIL_TO_PASS"]),
            pass_to_pass: str_list(&r["PASS_TO_PASS"]),
            image: s("image")?,
            eval_script: s("eval_script")?,
            log_parser: s("log_parser")?,
            eval_type: r["eval_type"]
                .as_str()
                .unwrap_or("pass_and_fail")
                .to_string(),
            environment_setup_commit: r["environment_setup_commit"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub instances: Vec<Instance>,
    /// Hub revision (commit sha) when the live fetch got it.
    pub revision: Option<String>,
    /// sha256 of the canonical rows: the pin pairing relies on.
    pub content_sha256: String,
    /// "live" or "cache (offline fallback: <reason>)"
    pub source: String,
    pub fetched_at: String,
}

fn curl_json(url: &str) -> Result<Value, String> {
    let out = std::process::Command::new("curl")
        .args(["-sSfL", "--proto", "=https", "--max-time", "60", url])
        .output()
        .map_err(|e| format!("cannot run curl: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "curl {url}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("{url}: {e}"))
}

fn content_hash(rows: &[Value]) -> String {
    hash_json(&Value::Array(rows.to_vec()))
}

/// Fetch every row live (datasets-server pages) and the Hub revision.
pub fn fetch_rows(
    fetch: &dyn Fn(&str) -> Result<Value, String>,
) -> Result<(Vec<Value>, Option<String>), String> {
    let revision = fetch(API_URL)
        .ok()
        .and_then(|v| v["sha"].as_str().map(str::to_string));
    let mut rows = Vec::new();
    let mut total: Option<u64> = None;
    let mut offset = 0u64;
    loop {
        let page = fetch(&format!("{ROWS_URL}&offset={offset}&length=100"))?;
        if total.is_none() {
            total = page["num_rows_total"].as_u64();
        }
        let got = page["rows"].as_array().cloned().unwrap_or_default();
        if got.is_empty() {
            break;
        }
        offset += got.len() as u64;
        rows.extend(got.into_iter().map(|r| r["row"].clone()));
        if total.is_some_and(|t| offset >= t) {
            break;
        }
    }
    if let Some(t) = total {
        if rows.len() as u64 != t {
            return Err(format!(
                "datasets-server returned {} of {t} rows",
                rows.len()
            ));
        }
    }
    if rows.is_empty() {
        return Err("datasets-server returned no rows".into());
    }
    Ok((rows, revision))
}

fn snapshot_from_rows(
    rows: Vec<Value>,
    revision: Option<String>,
    source: String,
) -> Result<Snapshot, String> {
    let content_sha256 = content_hash(&rows);
    let instances = rows
        .iter()
        .map(Instance::from_row)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Snapshot {
        instances,
        revision,
        content_sha256,
        source,
        fetched_at: rfc3339(unix_now()),
    })
}

fn cache_dir(cache: &Path) -> PathBuf {
    cache.join("swebench_verified")
}

/// Write a fetched snapshot to the cache (rows + meta) and return its path.
pub fn write_cache(
    cache: &Path,
    rows: &[Value],
    revision: &Option<String>,
) -> Result<PathBuf, String> {
    let d = cache_dir(cache);
    std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
    let sha = content_hash(rows);
    let p = d.join(format!("rows-{}.json", &sha[..16]));
    let doc = json!({"dataset": HF_DATASET, "revision": revision, "content_sha256": sha,
                     "fetched_at": rfc3339(unix_now()), "rows": rows});
    std::fs::write(&p, doc.to_string()).map_err(|e| format!("{}: {e}", p.display()))?;
    std::fs::write(
        d.join("latest"),
        p.file_name().unwrap().to_string_lossy().as_bytes(),
    )
    .map_err(|e| e.to_string())?;
    Ok(p)
}

/// A cached snapshot: the one with this content sha (a resumed run), else the latest.
pub fn read_cache(cache: &Path, content_sha256: Option<&str>) -> Result<Snapshot, String> {
    let d = cache_dir(cache);
    let name = match content_sha256 {
        Some(s) => format!("rows-{}.json", &s[..16.min(s.len())]),
        None => std::fs::read_to_string(d.join("latest"))
            .map_err(|_| format!("no cached SWE-bench Verified snapshot in {}", d.display()))?,
    };
    let doc: Value = serde_json::from_str(
        &std::fs::read_to_string(d.join(name.trim()))
            .map_err(|e| format!("{}: {e}", d.join(name.trim()).display()))?,
    )
    .map_err(|e| e.to_string())?;
    let rows = doc["rows"].as_array().cloned().unwrap_or_default();
    let snap = snapshot_from_rows(
        rows,
        doc["revision"].as_str().map(str::to_string),
        String::new(),
    )?;
    if doc["content_sha256"].as_str() != Some(snap.content_sha256.as_str()) {
        return Err("cached SWE-bench snapshot does not match its recorded sha256".into());
    }
    Ok(Snapshot {
        fetched_at: doc["fetched_at"].as_str().unwrap_or_default().to_string(),
        ..snap
    })
}

/// Live fetch, else the cache (with the reason in `source`).
pub fn load(
    cache: &Path,
    fetch: &dyn Fn(&str) -> Result<Value, String>,
) -> Result<Snapshot, String> {
    match fetch_rows(fetch) {
        Ok((rows, revision)) => {
            let _ = write_cache(cache, &rows, &revision);
            snapshot_from_rows(rows, revision, "live".into())
        }
        Err(e) => {
            let mut s =
                read_cache(cache, None).map_err(|c| format!("live fetch failed ({e}) and {c}"))?;
            s.source = format!("cache (offline fallback: {e})");
            Ok(s)
        }
    }
}

pub fn live_fetch(url: &str) -> Result<Value, String> {
    curl_json(url)
}

/// 64 bits from the OS CSPRNG.
pub fn csprng_seed() -> Result<u64, String> {
    use std::io::Read;
    let mut b = [0u8; 8];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .map_err(|e| format!("/dev/urandom: {e}"))?;
    Ok(u64::from_le_bytes(b))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Draw {
    pub seed: u64,
    /// csprng | param | resumed
    pub seed_source: String,
    pub n: usize,
    pub dataset: String,
    pub revision: Option<String>,
    pub content_sha256: String,
    pub source: String,
    pub drawn_at: String,
    pub per_repo: BTreeMap<String, usize>,
    pub instance_ids: Vec<String>,
}

/// Largest-remainder allocation of `n` over the repo counts (ties: larger repo, then name).
pub fn allocate(counts: &BTreeMap<String, usize>, n: usize) -> BTreeMap<String, usize> {
    let total: usize = counts.values().sum();
    let n = n.min(total);
    let mut out: BTreeMap<String, usize> = BTreeMap::new();
    let mut rem: Vec<(f64, usize, &String)> = Vec::new();
    let mut used = 0;
    for (repo, &c) in counts {
        let exact = n as f64 * c as f64 / total as f64;
        let k = exact.floor() as usize;
        out.insert(repo.clone(), k);
        used += k;
        rem.push((exact - k as f64, c, repo));
    }
    rem.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap()
            .then(b.1.cmp(&a.1))
            .then(a.2.cmp(b.2))
    });
    for (_, _, repo) in rem.into_iter().take(n - used) {
        *out.get_mut(repo).unwrap() += 1;
    }
    out
}

/// The stratified draw of `n` instance ids for `seed`.
pub fn draw(instances: &[Instance], n: usize, seed: u64) -> (Vec<String>, BTreeMap<String, usize>) {
    let mut by_repo: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for i in instances {
        by_repo
            .entry(i.repo.clone())
            .or_default()
            .push(i.instance_id.clone());
    }
    for ids in by_repo.values_mut() {
        ids.sort();
    }
    let counts = by_repo.iter().map(|(k, v)| (k.clone(), v.len())).collect();
    let alloc = allocate(&counts, n);
    let mut rng = Rng::new(seed);
    let mut picked = Vec::new();
    for (repo, ids) in &by_repo {
        let mut ids = ids.clone();
        for i in (1..ids.len()).rev() {
            let j = rng.below(i + 1);
            ids.swap(i, j);
        }
        picked.extend(ids.into_iter().take(alloc[repo]));
    }
    picked.sort();
    (picked, alloc.into_iter().filter(|(_, k)| *k > 0).collect())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn synthetic(n_per_repo: &[(&str, usize)]) -> Vec<Value> {
        let mut rows = Vec::new();
        for (repo, n) in n_per_repo {
            for k in 0..*n {
                let id = format!(
                    "{}__{}-{k}",
                    repo.split('/').next().unwrap(),
                    repo.split('/').nth(1).unwrap()
                );
                rows.push(json!({"instance_id": id, "repo": repo, "base_commit": format!("{:040x}", k + 1),
                    "problem_statement": format!("Fix bug {k} in {repo}"), "created_at": "2022-03-03T15:14:54Z",
                    "version": "1.0", "patch": "diff --git a/src/m.py b/src/m.py\n--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
                    "test_patch": "diff --git a/tests/test_m.py b/tests/test_m.py\n--- a/tests/test_m.py\n+++ b/tests/test_m.py\n@@ -0,0 +1 @@\n+HIDDEN_TEST_MARKER\n",
                    "FAIL_TO_PASS": ["tests/test_m.py::test_fix"], "PASS_TO_PASS": "[\"tests/test_m.py::test_old\"]",
                    "image": format!("swebench/sweb.eval.x86_64.{id}:latest"), "eval_script": "#!/bin/bash\nset -uxo pipefail\necho run\n",
                    "log_parser": "parse_log_pytest", "eval_type": "pass_and_fail"}));
            }
        }
        rows
    }

    #[test]
    fn live_fetch_paging_and_cache_fallback() {
        let rows = synthetic(&[("a/x", 150), ("b/y", 80)]);
        let r2 = rows.clone();
        let fetch = move |url: &str| -> Result<Value, String> {
            if url == API_URL {
                return Ok(json!({"sha": "rev123"}));
            }
            let off: usize = url
                .split("offset=")
                .nth(1)
                .unwrap()
                .split('&')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            let page: Vec<Value> = r2
                .iter()
                .skip(off)
                .take(100)
                .map(|r| json!({"row": r}))
                .collect();
            Ok(json!({"num_rows_total": 230, "rows": page}))
        };
        let d = std::env::temp_dir().join(format!("tb-swe-ds-{}", std::process::id()));
        let s = load(&d, &fetch).unwrap();
        assert_eq!(
            (s.instances.len(), s.source.as_str(), s.revision.as_deref()),
            (230, "live", Some("rev123"))
        );
        assert_eq!(
            s.instances[0].pass_to_pass,
            vec!["tests/test_m.py::test_old"],
            "JSON-string lists parse"
        );
        // offline: the cache answers, and says so
        let off = |_: &str| -> Result<Value, String> { Err("no network".into()) };
        let c = load(&d, &off).unwrap();
        assert!(c.source.starts_with("cache (offline fallback"));
        assert_eq!(c.content_sha256, s.content_sha256);
        assert_eq!(
            read_cache(&d, Some(&s.content_sha256))
                .unwrap()
                .instances
                .len(),
            230
        );
        std::fs::remove_dir_all(&d).ok();
        let e = load(&d, &off).unwrap_err();
        assert!(e.contains("live fetch failed"), "{e}");
    }

    #[test]
    fn stratified_draw_is_seeded_and_proportional() {
        // the Verified repo mix
        let mix = [
            ("django/django", 231),
            ("sympy/sympy", 75),
            ("sphinx-doc/sphinx", 44),
            ("matplotlib/matplotlib", 34),
            ("scikit-learn/scikit-learn", 32),
            ("astropy/astropy", 22),
            ("pydata/xarray", 22),
            ("pytest-dev/pytest", 19),
            ("pylint-dev/pylint", 10),
            ("psf/requests", 8),
            ("mwaskom/seaborn", 2),
            ("pallets/flask", 1),
        ];
        let inst: Vec<Instance> = synthetic(&mix)
            .iter()
            .map(|r| Instance::from_row(r).unwrap())
            .collect();
        let (a, alloc) = draw(&inst, 50, 7);
        assert_eq!(a.len(), 50);
        assert_eq!(alloc.values().sum::<usize>(), 50);
        assert_eq!(alloc["django/django"], 23); // 23.1
        assert_eq!(alloc["sympy/sympy"], 8); // 7.5 -> rounds up by remainder
        let (b, _) = draw(&inst, 50, 7);
        assert_eq!(a, b, "same seed, same draw");
        let (c, _) = draw(&inst, 50, 8);
        assert_ne!(a, c, "a new seed draws differently");
        let uniq: std::collections::BTreeSet<_> = a.iter().collect();
        assert_eq!(uniq.len(), 50);
        // every repo's share is within 1 of proportional
        for (repo, n) in mix {
            let k = alloc.get(repo).copied().unwrap_or(0) as f64;
            assert!((k - 50.0 * n as f64 / 500.0).abs() < 1.0, "{repo}");
        }
        assert_ne!(csprng_seed().unwrap(), csprng_seed().unwrap());
    }
}
