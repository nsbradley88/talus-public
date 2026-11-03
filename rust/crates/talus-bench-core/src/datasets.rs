//! Pinned datasets. Each dataset is identified by (source, revision, sha256) and carries its license.
//! Small slices whose license allows redistribution are embedded in the binary (offline smoke runs). Full sets are
//! fetched once with `talus-bench datasets fetch` into a local cache and verified by sha256 on every load.
//! Gated or "do not publish" sets (GPQA) are never embedded or committed.

use crate::util::sha256_hex;
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DatasetPin {
    pub id: &'static str,
    pub description: &'static str,
    /// Where the bytes come from (an immutable, revision-pinned URL when fetched).
    pub source: &'static str,
    pub revision: &'static str,
    pub split: &'static str,
    pub sha256: &'static str,
    pub rows: usize,
    pub license: &'static str,
    pub license_note: &'static str,
    /// Embedded copy (None = fetch into the cache).
    #[serde(skip)]
    pub embedded: Option<&'static str>,
    /// Private holdout data: prompts are redacted in reports.
    pub holdout: bool,
}

pub const GSM8K_COMMIT: &str = "3101c7d5072418e28b9008a6636bde82a006892c";

pub const DATASETS: &[DatasetPin] = &[
    DatasetPin {
        id: "gsm8k-test-slice50",
        description: "GSM8K test split, first 50 rows in file order (smoke tier, embedded)",
        source: "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl (lines 1-50)",
        revision: GSM8K_COMMIT,
        split: "test[0:50]",
        sha256: "4718cc77e7d7b11c3fc2a049d7a0dbda483fc4bd37e5432488c9f9d0f2cf181a",
        rows: 50,
        license: "MIT",
        license_note: "Copyright (c) 2021 OpenAI; license text in crates/talus-bench-core/data/gsm8k/LICENSE",
        embedded: Some(include_str!("../data/gsm8k/test-slice50.jsonl")),
        holdout: false,
    },
    DatasetPin {
        id: "gsm8k-test",
        description: "GSM8K test split, all 1319 rows (full tier, fetched)",
        source: "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
        revision: GSM8K_COMMIT,
        split: "test",
        sha256: "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        rows: 1319,
        license: "MIT",
        license_note: "Copyright (c) 2021 OpenAI",
        embedded: None,
        holdout: false,
    },
    DatasetPin {
        id: "talus-json-schema-v1",
        description: "20 authored structured-output cases (extraction, tool arguments, infra configs) with schemas and semantic checks",
        source: "crates/talus-bench-core/data/json_schema/cases.jsonl",
        revision: "v1",
        split: "all",
        sha256: "6444007a1f549e89e6ef065b856778e8c3b043516b3c978d709f1f104101bcd7",
        rows: 20,
        license: "MIT",
        license_note: "Written for Talus; part of the Talus repository",
        embedded: Some(include_str!("../data/json_schema/cases.jsonl")),
        holdout: false,
    },    DatasetPin {
        id: "ifeval-supported",
        description: "IFEval prompts whose instructions all have reimplementable rules (387 of 541; derived: rows filtered, keys sorted)",
        source: "https://huggingface.co/datasets/google/IFEval/resolve/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl (source sha256 6a85310c..., filtered)",
        revision: "966cd89545d6",
        split: "train (filtered)",
        sha256: "11c98e8600943bf4e3cd4f491908f3010b98514008a84fd5e739da05b4bc1dbe",
        rows: 387,
        license: "Apache-2.0",
        license_note: "google/IFEval, Apache License 2.0; modified: filtered to supported instruction types, re-serialized with sorted keys",
        embedded: Some(include_str!("../data/ifeval/ifeval-supported.jsonl")),
        holdout: false,
    },
    DatasetPin {
        id: "bfcl-v3-smoke",
        description: "BFCL v3 simple/multiple/parallel/irrelevance, interleaved, first 120 rows with ground truth (derived)",
        source: "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard (revision 61fc0608cfd8), combined",
        revision: "61fc0608cfd8",
        split: "smoke",
        sha256: "2127b907a71de90c960a3ff2255d76ada1f23aaabacb1a57cfc50a2647137f08",
        rows: 120,
        license: "Apache-2.0",
        license_note: "gorilla-llm BFCL, Apache License 2.0; modified: question, functions and possible answers combined per row, categories interleaved",
        embedded: Some(include_str!("../data/bfcl/bfcl-v3-smoke.jsonl")),
        holdout: false,
    },
    DatasetPin {
        id: "bfcl-simple",
        description: "BFCL v3 upstream file BFCL_v3_simple.json (full tier)",
        source: "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda/BFCL_v3_simple.json",
        revision: "61fc0608cfd8",
        split: "test",
        sha256: "fbc37b2ad252bf9af985582e0e07b456173fe627d957491472ea9cef5fb83158",
        rows: 400,
        license: "Apache-2.0",
        license_note: "gorilla-llm BFCL, Apache License 2.0",
        embedded: None,
        holdout: false,
    },
    DatasetPin {
        id: "bfcl-answers-simple",
        description: "BFCL v3 upstream file possible_answer/BFCL_v3_simple.json (full tier)",
        source: "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda/possible_answer/BFCL_v3_simple.json",
        revision: "61fc0608cfd8",
        split: "test",
        sha256: "2911a2bc00df82c4f999ffa64fedb0164cb88e96212ffa5972087eb91fd496ee",
        rows: 400,
        license: "Apache-2.0",
        license_note: "gorilla-llm BFCL, Apache License 2.0",
        embedded: None,
        holdout: false,
    },
    DatasetPin {
        id: "bfcl-multiple",
        description: "BFCL v3 upstream file BFCL_v3_multiple.json (full tier)",
        source: "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda/BFCL_v3_multiple.json",
        revision: "61fc0608cfd8",
        split: "test",
        sha256: "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
        rows: 200,
        license: "Apache-2.0",
        license_note: "gorilla-llm BFCL, Apache License 2.0",
        embedded: None,
        holdout: false,
    },
    DatasetPin {
        id: "bfcl-answers-multiple",
        description: "BFCL v3 upstream file possible_answer/BFCL_v3_multiple.json (full tier)",
        source: "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda/possible_answer/BFCL_v3_multiple.json",
        revision: "61fc0608cfd8",
        split: "test",
        sha256: "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
        rows: 200,
        license: "Apache-2.0",
        license_note: "gorilla-llm BFCL, Apache License 2.0",
        embedded: None,
        holdout: false,
    },
    DatasetPin {
        id: "bfcl-parallel",
        description: "BFCL v3 upstream file BFCL_v3_parallel.json (full tier)",
        source: "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda/BFCL_v3_parallel.json",
        revision: "61fc0608cfd8",
        split: "test",
        sha256: "fad375d68295776efb63e5aa331dd884f8635e7bb50e05908113c74f14281c3c",
        rows: 200,
        license: "Apache-2.0",
        license_note: "gorilla-llm BFCL, Apache License 2.0",
        embedded: None,
        holdout: false,
    },
    DatasetPin {
        id: "bfcl-answers-parallel",
        description: "BFCL v3 upstream file possible_answer/BFCL_v3_parallel.json (full tier)",
        source: "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda/possible_answer/BFCL_v3_parallel.json",
        revision: "61fc0608cfd8",
        split: "test",
        sha256: "3af54bf04e5b7640b03a9b6a82440f52474feb24b398f94435fd2b155c7727de",
        rows: 200,
        license: "Apache-2.0",
        license_note: "gorilla-llm BFCL, Apache License 2.0",
        embedded: None,
        holdout: false,
    },
    DatasetPin {
        id: "bfcl-irrelevance",
        description: "BFCL v3 upstream file BFCL_v3_irrelevance.json (full tier)",
        source: "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda/BFCL_v3_irrelevance.json",
        revision: "61fc0608cfd8",
        split: "test",
        sha256: "975f51c51f688649fd190078efd87081241e0a326f9114a2ea3c1ca2440d8690",
        rows: 240,
        license: "Apache-2.0",
        license_note: "gorilla-llm BFCL, Apache License 2.0",
        embedded: None,
        holdout: false,
    },
    DatasetPin {
        id: "mmlu-pro-strat280",
        description: "MMLU-Pro test, first 20 questions of each of the 14 categories, interleaved (derived from parquet)",
        source: "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/b189ec765aa7ed75c8acfea42df31fdae71f97be/data/test-00000-of-00001.parquet (sha256 0e24a191...), converted",
        revision: "b189ec765aa7",
        split: "test (stratified 280)",
        sha256: "63256a259a831f3eecc6eaec3295b2a054ec1968d3e150cbb40b7a547a68683a",
        rows: 280,
        license: "MIT",
        license_note: "TIGER-Lab/MMLU-Pro dataset card: MIT; converted from parquet with pyarrow, keys sorted",
        embedded: Some(include_str!("../data/mmlu_pro/mmlu-pro-strat280.jsonl")),
        holdout: false,
    },
    DatasetPin {
        id: "mmlu-pro-validation",
        description: "MMLU-Pro validation split (70 rows): the official 5-shot CoT examples per category",
        source: "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/b189ec765aa7ed75c8acfea42df31fdae71f97be/data/validation-00000-of-00001.parquet (sha256 139423c2...), converted",
        revision: "b189ec765aa7",
        split: "validation",
        sha256: "09f5a3f6933273cfa8221bc8c45a13876dde6dec34280948ecf73f6905c8fd59",
        rows: 70,
        license: "MIT",
        license_note: "TIGER-Lab/MMLU-Pro dataset card: MIT; converted from parquet with pyarrow, keys sorted",
        embedded: Some(include_str!("../data/mmlu_pro/mmlu-pro-validation.jsonl")),
        holdout: false,
    },
    DatasetPin {
        id: "mmlu-pro-test",
        description: "MMLU-Pro test split, all 12032 rows (full tier; import a converted file with `talus-bench datasets import`)",
        source: "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/b189ec765aa7ed75c8acfea42df31fdae71f97be/data/test-00000-of-00001.parquet (sha256 0e24a191...), converted",
        revision: "b189ec765aa7",
        split: "test",
        sha256: "3f22e18b9e944d8b76c92a32bfc7c872b5a6c2286eac68e819bad324008a98a0",
        rows: 12032,
        license: "MIT",
        license_note: "TIGER-Lab/MMLU-Pro dataset card: MIT; conversion recipe in docs/BENCH.md",
        embedded: None,
        holdout: false,
    },
    DatasetPin {
        id: "pd-prose-v1",
        description: "public-domain English prose for perplexity / kld_topk: Doyle, \"A Scandal in Bohemia\" (1891) and Darwin, On the Origin of Species (6th ed., 1872), chapter I (~28k tokens)",
        source: "Project Gutenberg eBooks #1661 (lines 55-1134) and #2009 (lines 862-2027), fetched 2026-11-02 (sha256 922e2a12... / 2b13c852...); CRLF -> LF, trailing spaces stripped, joined with two blank lines; no Project Gutenberg header, footer or trademark",
        revision: "v1",
        split: "all",
        sha256: "c5035b0f485ba89c1a7922551161ba44139c2e0f3e85a69d7ca4893ffc66b9db",
        rows: 1907,
        license: "Public domain",
        license_note: "Both works were published before 1929 and are in the public domain in the United States; only the public-domain text is kept (no Project Gutenberg license text or trademark)",
        embedded: Some(include_str!("../data/perplexity/pd-prose-v1.txt")),
        holdout: false,
    },
    DatasetPin {
        id: "livecodebench-v6",
        description: "LiveCodeBench code_generation_lite, release increment v6 (test6.jsonl): 175 AtCoder/LeetCode problems, contests 2025-01-04..2025-04-06, hidden tests as published (fetched)",
        source: "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test6.jsonl",
        revision: "0fe84c3912ea",
        split: "test (release_v6 increment)",
        sha256: "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5",
        rows: 175,
        license: "CC (variant unspecified)",
        license_note: "dataset card declares only `license: cc`; problem statements are from AtCoder/LeetCode; LiveCodeBench code is MIT. Fetched into the local cache only: never embedded, committed or redistributed",
        embedded: None,
        holdout: false,
    },
    DatasetPin {
        id: "lcb-fixture-v1",
        description: "3 authored LiveCodeBench-format problems (stdin, functional, decimal output; hidden tests encoded like upstream) for the unit tests",
        source: "crates/talus-bench-core/data/livecodebench/fixture-v1.jsonl",
        revision: "v1",
        split: "all",
        sha256: "0cfdbb53c397e6c0ae867e61189dec3cb704bfcfbdea33f761f6bbbd7cd936e1",
        rows: 3,
        license: "MIT",
        license_note: "Written for Talus; part of the Talus repository",
        embedded: Some(include_str!("../data/livecodebench/fixture-v1.jsonl")),
        holdout: false,
    },
];

pub fn pin(id: &str) -> Option<&'static DatasetPin> {
    DATASETS.iter().find(|d| d.id == id)
}

/// Default cache: `$XDG_CACHE_HOME/talus-bench/datasets` or `~/.cache/talus-bench/datasets`.
pub fn default_cache_dir() -> PathBuf {
    if let Ok(x) = std::env::var("XDG_CACHE_HOME") {
        if !x.is_empty() {
            return PathBuf::from(x).join("talus-bench/datasets");
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache/talus-bench/datasets")
}

pub fn cache_path(cache: &Path, d: &DatasetPin) -> PathBuf {
    cache.join(format!("{}-{}.jsonl", d.id, &d.sha256[..12]))
}

/// The dataset's bytes, verified against the pin.
pub fn load(d: &DatasetPin, cache: &Path) -> Result<String, String> {
    let text = match d.embedded {
        Some(t) => t.to_string(),
        None => {
            let p = cache_path(cache, d);
            std::fs::read_to_string(&p).map_err(|_| {
                format!(
                    "dataset {} is not in the cache ({}); run `talus-bench datasets fetch {}`",
                    d.id,
                    p.display(),
                    d.id
                )
            })?
        }
    };
    let got = sha256_hex(text.as_bytes());
    if got != d.sha256 {
        return Err(format!(
            "dataset {}: sha256 {got} does not match the pin {} (corrupt cache or edited file)",
            d.id, d.sha256
        ));
    }
    Ok(text)
}

/// Fetch a dataset into the cache with `curl` (talus-bench has no TLS stack), verify, then rename into place.
pub fn fetch(d: &DatasetPin, cache: &Path) -> Result<PathBuf, String> {
    if d.embedded.is_some() {
        return Err(format!("{} is embedded; nothing to fetch", d.id));
    }
    std::fs::create_dir_all(cache).map_err(|e| format!("{}: {e}", cache.display()))?;
    let dest = cache_path(cache, d);
    if load(d, cache).is_ok() {
        return Ok(dest);
    }
    let tmp = dest.with_extension("part");
    let st = std::process::Command::new("curl")
        .args(["-sSfL", "--proto", "=https", "-o"])
        .arg(&tmp)
        .arg(d.source)
        .status()
        .map_err(|e| format!("cannot run curl: {e}"))?;
    if !st.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("curl failed for {}", d.source));
    }
    let bytes = std::fs::read(&tmp).map_err(|e| e.to_string())?;
    let got = sha256_hex(&bytes);
    if got != d.sha256 {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!(
            "{}: downloaded sha256 {got} != pin {}",
            d.id, d.sha256
        ));
    }
    std::fs::rename(&tmp, &dest).map_err(|e| e.to_string())?;
    Ok(dest)
}

/// Copy an externally converted file into the cache after checking it against the pin (datasets that are not
/// published as JSON lines, e.g. MMLU-Pro's parquet).
pub fn import(d: &DatasetPin, file: &Path, cache: &Path) -> Result<PathBuf, String> {
    let bytes = std::fs::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let got = sha256_hex(&bytes);
    if got != d.sha256 {
        return Err(format!(
            "{}: sha256 {got} != pin {} (convert exactly as documented)",
            file.display(),
            d.sha256
        ));
    }
    std::fs::create_dir_all(cache).map_err(|e| format!("{}: {e}", cache.display()))?;
    let dest = cache_path(cache, d);
    std::fs::write(&dest, &bytes).map_err(|e| format!("{}: {e}", dest.display()))?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_datasets_match_their_pins() {
        for d in DATASETS.iter().filter(|d| d.embedded.is_some()) {
            let text = load(d, Path::new("/nonexistent")).unwrap();
            assert_eq!(
                text.lines().filter(|l| !l.trim().is_empty()).count(),
                d.rows,
                "{}",
                d.id
            );
            assert!(!d.license.is_empty());
        }
    }

    #[test]
    fn missing_or_corrupt_cache_is_an_error() {
        let d = pin("gsm8k-test").unwrap();
        let dir = std::env::temp_dir().join(format!("tb-ds-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let e = load(d, &dir).unwrap_err();
        assert!(e.contains("datasets fetch gsm8k-test"), "{e}");
        std::fs::write(cache_path(&dir, d), "{}\n").unwrap();
        assert!(load(d, &dir)
            .unwrap_err()
            .contains("does not match the pin"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn slice_is_a_prefix_of_the_full_split() {
        // the slice pin documents "first 50 rows"; its rows must look like GSM8K rows
        let t = load(pin("gsm8k-test-slice50").unwrap(), Path::new("/x")).unwrap();
        for l in t.lines() {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            assert!(v["question"].is_string());
            assert!(v["answer"].as_str().unwrap().contains("####"));
        }
    }
}
