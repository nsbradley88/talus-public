//! talus-bench against real talus-servers driving a mock engine child over the stdio protocol: the `build`, RSS and
//! energy fields of `/v1/talus/status` reach the manifest and the report's fingerprint table, and the
//! reasoning/answer fields of `timings` reach `cases.jsonl` and `perf.json`, end to end.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use talus_bench_core::config::Config;
use talus_bench_core::runner::{self, RunOptions};
use talus_server::config::ServerConfig;
use talus_server::engine::StdioSubprocessEngine;
use talus_server::fingerprint::sha256_hex;
use talus_server::json::JsonValue;
use talus_server::server::{ServerHandle, ServiceState, TalusServer};
use talus_server::tokenizer::ByteTokenizer;

const REASONING: &str = "thinking it through";
const ANSWER: &str = "The answer is 42.";

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("talus-bench-e2e-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A release-shaped directory: bin/ with the engine (this crate's binary as `--mock-stdio-worker`), `COMMIT` and the
/// kernels library, MANIFEST.json above it, the weights beside it, and a run config serving them.
fn serve(root: &Path, commit: &str, weights: &[u8], glm_pch: &str) -> ServerHandle {
    let bin = root.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let exe = bin.join("talus-cli");
    // a copy, not a hard link: each new link changes the shared inode's ctime, and the status ) then
    // drops the hash of every server's engine as a changed file
    std::fs::copy(env!("CARGO_BIN_EXE_talus-server"), &exe).unwrap();
    std::fs::write(bin.join("COMMIT"), format!("{commit}\n")).unwrap();
    std::fs::write(bin.join("libtalus_kernels.so"), b"kernels").unwrap();
    std::fs::write(
        root.join("MANIFEST.json"),
        r#"{"tag":"v0.0.0-test","targets":{"gpu":["gfx1151"]}}"#,
    )
    .unwrap();
    let model = root.join("model.bin");
    std::fs::write(&model, weights).unwrap();
    let cfg = serde_json::json!({
        "exe": exe,
        "model": model,
        "model_name": "qwen3.8-27b",
        "host": "127.0.0.1",
        "args": ["--mock-stdio-worker", "--mock-script",
                 format!("{REASONING}</think>\n\n{ANSWER}")],
        "env": {"GLM_PCH": glm_pch},
    });
    let mut cfg = ServerConfig::from_json(&JsonValue::parse(&cfg.to_string()).unwrap()).unwrap();
    cfg.port = 0;
    // a real child over the stdio protocol; ServiceState::from_config would want the model's tokenizer )
    let engine =
        StdioSubprocessEngine::spawn(&exe, &cfg.engine_args, &cfg.engine_env, cfg.max_context)
            .unwrap();
    let state = Arc::new(ServiceState::new(
        cfg,
        Arc::new(engine),
        ByteTokenizer::new(),
    ));
    let srv = TalusServer::bind(state).unwrap().start_background();
    // the bench reads hashes that are done; how long a debug-built sha256 takes is not what this test is about
    let _ = srv.state.fingerprint.start_hashing(true);
    for _ in 0..3000 {
        if srv.state.fingerprint.complete() {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(srv.state.fingerprint.complete());
    srv
}

fn read(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn status_fields_reach_the_manifest_report_and_perf_json() {
    if std::env::var("STRATA_API_KEY").is_ok() || std::env::var("TALUS_API_KEY").is_ok() {
        return; // the server would demand a key from this test client
    }
    let d = scratch("run");
    let r = serve(&d.join("ref"), "aaaa1111bbbb2222", b"weights-ref", "2048");
    let c = serve(&d.join("cand"), "cccc3333dddd4444", b"weights-cand", "4096");
    let yaml = format!(
        r#"
experiment: {{id: srv-e2e, seed: 11}}
reference: {{id: ref, endpoint: "http://127.0.0.1:{}"}}
candidates:
  - id: cand
    endpoint: "http://127.0.0.1:{}/v1"
inference: {{request_timeout_s: 30, thinking: "on"}}
benchmarks:
  tier: smoke
  tasks: [gsm8k]
quality_gates: {{min_cases: 20, bootstrap_resamples: 200}}
"#,
        r.port(),
        c.port()
    );
    let out = d.join("out");
    let o = runner::run(
        &Config::from_yaml(&yaml).unwrap(),
        &RunOptions {
            output_dir: Some(out.clone()),
            poll_ms: Some(20),
            api_key: None,
            ..Default::default()
        },
        Arc::new(AtomicBool::new(false)),
        &mut |_| {},
    )
    .unwrap();

    // manifest: what the server reported, for each model
    let m = read(&out.join("manifest.json"));
    for (i, (commit, weights)) in [
        ("aaaa1111bbbb2222", &b"weights-ref"[..]),
        ("cccc3333dddd4444", &b"weights-cand"[..]),
    ]
    .iter()
    .enumerate()
    {
        let sv = &m["models"][i]["server"];
        assert_eq!(sv["unavailable"], serde_json::json!([]), "{sv:#}");
        let st = &sv["talus_status"];
        let b = &st["build"];
        assert_eq!(b["mock"], false);
        assert_eq!(b["engine_exe"]["commit"], *commit);
        assert_eq!(b["engine_exe"]["commit_source"], "COMMIT");
        assert_eq!(b["engine_exe"]["tag"], "v0.0.0-test");
        assert_eq!(
            b["kernels_lib"]["sha256"].as_str(),
            Some(sha256_hex(b"kernels").as_str())
        );
        assert_eq!(b["gfx_target"], "gfx1151");
        assert_eq!(
            b["weights"][0]["sha256"].as_str(),
            Some(sha256_hex(weights).as_str())
        );
        assert!(st["engine_rss_bytes"].as_u64().unwrap() > 0);
        assert!(st["server_rss_bytes"].as_u64().unwrap() > 0);
        assert!(st["engine_rss_bytes_end"].as_u64().unwrap() > 0);
        // energy: null on a machine without an amdgpu power file, else the counters
        assert!(st["energy"].is_null() || st["energy"]["joules_total"].is_number());
    }

    // report: the fingerprint table, with what differs from the reference marked
    let md = std::fs::read_to_string(out.join("report.md")).unwrap();
    assert_eq!(md, o.report_md);
    assert!(md.contains("| Field | ref | cand |"), "{md}");
    assert!(
        md.contains("| Engine commit | aaaa1111bbbb | cccc3333dddd **changed** |"),
        "{md}"
    );
    assert!(md.contains("| GPU target | gfx1151 | gfx1151 |"), "{md}");
    let short = |w: &[u8]| sha256_hex(w)[..12].to_string();
    assert!(
        md.contains(&format!(
            "| Weights sha256 | model.bin `{}` | model.bin `{}` **changed** |",
            short(b"weights-ref"),
            short(b"weights-cand")
        )),
        "{md}"
    );
    // the row also has whatever TALUS_* the test process runs under (the engine sees those too)
    let env_row = md
        .lines()
        .find(|l| l.starts_with("| Numerics environment |"))
        .unwrap();
    assert!(env_row.contains("GLM_PCH=2048"), "{env_row}");
    assert!(env_row.contains("GLM_PCH=4096"), "{env_row}");
    assert!(env_row.ends_with("**changed** |"), "{env_row}");
    assert!(
        md.contains(&format!(
            "| Kernels library sha256 | `{0}` | `{0}` |",
            short(b"kernels")
        )),
        "{md}"
    );
    assert!(!md.contains("Not fingerprinted"), "{md}");

    // cases.jsonl: the split of each request
    let rows =
        talus_bench_core::case::read_jsonl(&out.join("candidates/cand/cases.jsonl")).unwrap();
    assert!(rows.len() >= 20);
    for row in &rows {
        assert_eq!(row.usage.reasoning_tokens, Some(REASONING.len() as u64));
        assert_eq!(row.timings.reasoning_n, Some(REASONING.len() as u64));
        assert_eq!(row.timings.answer_n, Some(ANSWER.len() as u64));
    }

    // perf.json: the channel totals per task
    let p = read(&out.join("candidates/cand/perf.json"));
    let ch = &p["channels"]["tasks"]["gsm8k"];
    let n = rows.len() as u64;
    assert_eq!(ch["requests"], n);
    assert_eq!(ch["reasoning_tokens"], n * REASONING.len() as u64);
    assert_eq!(ch["answer_tokens"], n * ANSWER.len() as u64);
    let decode = p["tasks"]["gsm8k"]["decode_ms"].as_f64().unwrap();
    let parts = ch["reasoning_ms"].as_f64().unwrap() + ch["answer_ms"].as_f64().unwrap();
    assert!(
        (parts - decode).abs() <= 0.11 * n as f64,
        "{parts} vs {decode}"
    );

    drop((r, c));
    let _ = std::fs::remove_dir_all(&d);
}
