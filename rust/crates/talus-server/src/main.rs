//! `talus-server` CLI binary target (`--config`, `--host`, `--port`, `--model-name`, `--api-key`, `--mock`).

use std::io::{self, Write};
use std::sync::Arc;
use talus_server::{ByteTokenizer, ScriptedMockEngine, ServerConfig, ServiceState, TalusServer};

fn main() {
    // TALUS_X works as STRATA_X (TALUS_API_KEY for STRATA_API_KEY, ...); the engine it starts inherits both
    talus_core_api::apply_env_aliases();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "Usage: talus-server [--config <path.json>] [--host <addr>] [--port <port>] [--model-name <name>] [--model-alias <alias>] [--api-key <key>] [--mock] [--mock-stdio-worker]"
        );
        return;
    }

    let cfg = match ServerConfig::parse_cli_args(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[talus-server] error: {e}");
            std::process::exit(2);
        }
    };

    if cfg.mock_stdio_worker {
        let tok = ByteTokenizer::new();
        let engine = ScriptedMockEngine::new(tok);
        engine.set_max_context(cfg.max_context);
        if !cfg.mock_scripts.is_empty() {
            engine.set_scripts(cfg.mock_scripts);
        }
        let stdin = io::stdin();
        let stdout = io::stdout();
        if let Err(e) = engine.run_stdio_loop(stdin.lock(), stdout.lock()) {
            eprintln!("[talus-server] stdio worker error: {e}");
            std::process::exit(1);
        }
        return;
    }

    let state = match ServiceState::from_config(cfg.clone()) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("[talus-server] initialization error: {e}");
            // a talus-tui apply that does not start: go back to the last config that did
            let Some(prev) = talus_server::control::recover_after_failed_start(&cfg, &e) else {
                std::process::exit(1);
            };
            match ServiceState::from_config(prev) {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    eprintln!("[talus-server] initialization error (last good config): {e}");
                    std::process::exit(1);
                }
            }
        }
    };

    let server = match TalusServer::bind(state) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[talus-server] bind error: {e}");
            std::process::exit(1);
        }
    };

    talus_server::control::mark_good(&server.state.config);
    let addr = server.local_addr();
    println!("[talus] listening on http://{addr}");
    let _ = io::stdout().flush();

    server.run_blocking();
}
