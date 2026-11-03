//! talus-bench core: paired quality validation of Talus settings and quantizations over talus-server's
//! OpenAI-compatible HTTP API. Native Rust orchestration, case capture, pairing by case id, paired bootstrap and
//! McNemar statistics, release gates and Markdown/JSON reports.
//!
//! Layout: `config` (YAML schema + validation) -> `tasks` (adapter interface, native scorers, registry of
//! NOT_RUN / NOT_SUPPORTED tasks) over pinned `datasets` -> `runner` (fingerprint, Talus apply, execution,
//! frozen references) -> `compare` (pairing, transitions) + `stats` -> `gates` -> `report` / `artifacts`.
//! `mock` is a synthetic server for tests and demos.

pub mod artifacts;
pub mod case;
pub mod compare;
pub mod config;
pub mod datasets;
pub mod gates;
pub mod http;
pub mod jsonschema;
pub mod kl;
pub mod mock;
pub mod pool;
pub mod report;
pub mod runner;
pub mod stats;
pub mod swe;
pub mod tasks;
pub mod util;

/// The smoke config the TUI's Bench tab and `talus-bench smoke` use: the served model at its current selection
/// as the reference vs `candidate_level` of the same model (Talus-vs-Talus), json_schema + gsm8k (20 cases).
pub fn smoke_config(
    endpoint: &str,
    model: Option<&str>,
    candidate_level: Option<&str>,
    api_key_env: Option<&str>,
    out_dir: &str,
) -> config::Config {
    let talus = |level: &str, apply: bool| {
        model.map(|m| config::TalusSelection {
            model: m.to_string(),
            level: Some(level.to_string()),
            choices: Default::default(),
            apply,
            apply_timeout_s: 900,
        })
    };
    let ep = |id: &str, t: Option<config::TalusSelection>| config::Endpoint {
        id: id.to_string(),
        kind: config::EndpointKind::Http,
        endpoint: Some(endpoint.to_string()),
        model: None,
        api_key_env: api_key_env.map(str::to_string),
        frozen: None,
        talus: t,
        provenance: Default::default(),
    };
    let cand_level = candidate_level.unwrap_or("fast");
    let mut limits = std::collections::BTreeMap::new();
    limits.insert("gsm8k".to_string(), 20);
    let mut gates = config::GateConfig {
        min_cases: 20,
        ..Default::default()
    };
    gates.bootstrap_resamples = 2000;
    config::Config {
        experiment: config::Experiment {
            id: format!(
                "smoke-{}",
                util::rfc3339(util::unix_now())
                    .replace([':', '-'], "")
                    .trim_end_matches('Z')
                    .replace('T', "-")
            ),
            seed: 42,
            output_dir: Some(out_dir.to_string()),
            description: Some(format!(
                "TUI smoke: balanced vs {cand_level} on the served model"
            )),
        },
        reference: ep("balanced", talus("balanced", true)),
        candidates: vec![ep(cand_level, talus(cand_level, true))],
        inference: Default::default(),
        benchmarks: config::Benchmarks {
            tier: config::Tier::Smoke,
            tasks: vec!["json_schema".into(), "gsm8k".into()],
            limits,
            required: None,
            dataset_cache: None,
            params: Default::default(),
        },
        quality_gates: gates,
        report: Default::default(),
    }
}
