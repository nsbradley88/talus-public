//! HTTP API server (OpenAI, Anthropic, llama.cpp) and stdio engine protocol.

pub use talus_core_api as api;

pub mod config;
pub mod control;
pub mod engine;
pub mod frontend;
pub mod json;
pub mod prometheus;
pub mod responses;
pub mod server;
pub mod tokenizer;

pub use config::{ServerConfig, DEFERRED_FEATURES, DROPPED_FEATURES};
pub use engine::{
    Engine, EngineError, EngineTimings, GenerateRequest, SamplingParams, ScriptedMockEngine,
    StdioSubprocessEngine,
};
pub use json::JsonValue;
pub use server::{slot_filename_problem, ServerHandle, ServiceState, TalusServer};
pub use tokenizer::{ByteTokenizer, Detokenizer, StopMatcher};

pub const CRATE_NAME: &str = "talus-server";

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn crate_identity() {
        assert_eq!(api::CONTRACT_VERSION, 0);
        assert_eq!(CRATE_NAME, "talus-server");
    }

    #[test]
    fn real_runconfig_and_chat_template_loading() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let jinja_path = repo_root.join("serve/chat_template.jinja");
        assert!(jinja_path.exists());

        let raw = format!(
            r#"{{
                "arch": "glm4_moe_lite",
                "host": "127.0.0.1",
                "port": 0,
                "exe": "/tmp/nonexistent-talus",
                "mock": true,
                "cwd": "/tmp",
                "log": "/tmp/talus.log",
                "chat_template": "{}",
                "stop_tokens": ["<|user|>", "<|observation|>"],
                "fit_max_tokens": true,
                "tool_call_recovery": true,
                "anthropic_thinking": "auto",
                "lib_dirs": ["/opt/rocm/lib", "/usr/local/lib"],
                "env": {{"TALUS_GRAPH": "1"}},
                "args": [
                    "--native", "/models/glm.talus",
                    "--max-context", "65536",
                    "--spec", "4",
                    "--reasoning-effort", "low"
                ]
            }}"#,
            jinja_path.display()
        );
        let val = JsonValue::parse(&raw).expect("parse runconfig JSON");
        let cfg = ServerConfig::from_json(&val).expect("ServerConfig::from_json");
        assert_eq!(cfg.model_name, "glm-4.8-flash");
        assert_eq!(cfg.model_path, "/models/glm.talus");
        assert_eq!(cfg.max_context, 65536);
        assert_eq!(cfg.draft_max, 4);
        assert_eq!(cfg.reasoning_effort, "low");
        assert!(cfg.fit_max_tokens);
        assert!(cfg.tool_call_recovery);
        assert!(cfg.anthropic_think_unasked);
        assert_eq!(cfg.stop_tokens, vec!["<|user|>", "<|observation|>"]);
        assert!(cfg
            .engine_env
            .iter()
            .any(|(k, v)| k == "TALUS_GRAPH" && v == "1"));
        assert!(cfg
            .engine_env
            .iter()
            .any(|(k, v)| k == "LD_LIBRARY_PATH" && v.starts_with("/opt/rocm/lib:/usr/local/lib")));
        assert!(!cfg.engine_args.iter().any(|a| a == "--reasoning-effort"));

        let state = ServiceState::from_config(cfg).expect("ServiceState::from_config");
        assert!(state.chat_template.is_some());
        assert_eq!(state.tokenizer.stop_ids().len(), 4);
    }
}
