//! Run-config JSON loading and CLI flags (`serve/runconfig.py` + `serve/server.py` parity),
//! plus registry of deliberate feature drops / deferrals per Spec §5 (WP-1.4).

use crate::engine::SamplingParams;
use crate::json::JsonValue;
use std::path::{Path, PathBuf};

/// Deliberately dropped features (WP-1.4):
/// - Windows job objects (`serve/winjob.py`) & keepawake (`serve/keepawake.py`)
/// - Multi-instance replica pool (`serve/replicas.py`)
/// - Diffusers / Stable Diffusion image generation (`serve/image.py`)
pub const DROPPED_FEATURES: &[(&str, &str)] = &[
    (
        "winjob",
        "Windows job-object lifecycle management (Linux/Strix Halo only)",
    ),
    ("keepawake", "Windows/macOS display/sleep inhibition"),
    ("replicas", "Multi-process replica poolrouter"),
    ("image", "Image generation /v1/images/generations endpoint"),
];

/// Explicitly deferred features (WP-1.4):
/// - MCP hub (`serve/mcphub.py`)
/// - Vision / multimodal (`serve/vision.py`)
pub const DEFERRED_FEATURES: &[(&str, &str)] = &[
    ("mcp_hub", "Model Context Protocol (MCP) tool server hub"),
    ("vision", "Multimodal image input and vision projector"),
];

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_name: String,
    pub model_aliases: Vec<String>,
    pub api_key: Option<String>,
    pub config_path: Option<PathBuf>,
    pub model_path: String,
    pub exe_path: Option<PathBuf>,
    pub cwd: Option<PathBuf>,
    pub log_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub chat_template_path: Option<PathBuf>,
    pub stop_tokens: Vec<String>,
    pub max_context: usize,
    pub n_predict: i64,
    pub reasoning_effort: String,
    pub reasoning_budget: usize,
    pub reasoning_recovery: bool,
    pub tool_call_recovery: bool,
    pub fit_max_tokens: bool,
    pub anthropic_think_unasked: bool,
    pub sampling: SamplingParams,
    pub draft_max: usize,
    pub draft_p_min: f64,
    pub n_parallel: usize,
    pub kv_unified: bool,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub flash_attn: String,
    pub fit: String,
    pub ctx_shift: bool,
    pub slot_save_path: String,
    pub idle_unload_seconds: u64,
    pub mock_mode: bool,
    pub mock_stdio_worker: bool,
    pub mock_scripts: Vec<String>,
    pub engine_args: Vec<String>,
    pub engine_env: Vec<(String, String)>,
    pub lib_dirs: Vec<String>,
    /// talus-tui control directory (`--control-dir`): per-model run configs `<model-id>.json` and the generated
    /// `active.json`. `None` turns the control endpoint (`POST /v1/talus/apply`) off.
    pub control_dir: Option<PathBuf>,
    pub raw_config: JsonValue,
}

impl Default for ServerConfig {
    fn default() -> Self {
        let env_key = std::env::var("STRATA_API_KEY")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        Self {
            host: "127.0.0.1".to_string(),
            port: 8000,
            model_name: "qwen3.8-27b".to_string(),
            model_aliases: Vec::new(),
            api_key: env_key,
            config_path: None,
            model_path: String::new(),
            exe_path: None,
            cwd: None,
            log_path: None,
            tokenizer_path: None,
            chat_template_path: None,
            stop_tokens: Vec::new(),
            max_context: 32768,
            n_predict: 32768,
            reasoning_effort: "high".to_string(),
            reasoning_budget: 0,
            reasoning_recovery: true,
            tool_call_recovery: true,
            fit_max_tokens: false,
            anthropic_think_unasked: false,
            sampling: SamplingParams::default(),
            draft_max: 0,
            draft_p_min: 0.75,
            n_parallel: 1,
            kv_unified: true,
            cache_type_k: "q8_0".to_string(),
            cache_type_v: "q8_0".to_string(),
            flash_attn: "on".to_string(),
            fit: "off".to_string(),
            ctx_shift: false,
            slot_save_path: String::new(),
            idle_unload_seconds: 0,
            mock_mode: true,
            mock_stdio_worker: false,
            mock_scripts: Vec::new(),
            engine_args: Vec::new(),
            engine_env: Vec::new(),
            lib_dirs: Vec::new(),
            control_dir: None,
            raw_config: JsonValue::empty_obj(),
        }
    }
}

pub fn default_model_name(arch: Option<&str>) -> &'static str {
    match arch.unwrap_or("") {
        "glm4_moe_lite" => "glm-4.8-flash",
        "laguna1" => "mimo-v2-pro",
        "qwen35moe" => "qwen3.8-35b-a3b",
        "qwen35" => "qwen3.8-27b",
        _ => "qwen3.8-27b",
    }
}

impl ServerConfig {
    pub fn from_json_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read config {}: {e}", path.display()))?;
        let val = JsonValue::parse(&text)
            .map_err(|e| format!("invalid JSON in {}: {e}", path.display()))?;
        let mut cfg = Self::from_json(&val)?;
        cfg.config_path = Some(path.to_path_buf());
        Ok(cfg)
    }

    pub fn from_json(val: &JsonValue) -> Result<Self, String> {
        if !val.is_object() {
            return Err("run config must be a JSON object".to_string());
        }
        let mut cfg = Self {
            raw_config: val.clone(),
            ..Self::default()
        };

        if let Some(h) = val.get_str("host").filter(|s| !s.trim().is_empty()) {
            cfg.host = h.trim().to_string();
        }
        if let Some(p) = val.get_u64("port") {
            if (1..=65535).contains(&p) {
                cfg.port = p as u16;
            }
        }
        if let Some(arch) = val.get_str("arch").filter(|s| !s.trim().is_empty()) {
            cfg.model_name = default_model_name(Some(arch.trim())).to_string();
        }
        if let Some(m) = val.get_str("model_name").filter(|s| !s.trim().is_empty()) {
            cfg.model_name = m.trim().to_string();
        }
        if let Some(aliases) = val.get("model_aliases").or_else(|| val.get("aliases")) {
            if let Some(arr) = aliases.as_array() {
                for item in arr {
                    if let Some(s) = item.as_str().map(str::trim).filter(|s| !s.is_empty()) {
                        if s != cfg.model_name && !cfg.model_aliases.iter().any(|a| a == s) {
                            cfg.model_aliases.push(s.to_string());
                        }
                    }
                }
            } else if let Some(s) = aliases.as_str() {
                for part in s.split(',') {
                    let trimmed = part.trim();
                    if !trimmed.is_empty()
                        && trimmed != cfg.model_name
                        && !cfg.model_aliases.iter().any(|a| a == trimmed)
                    {
                        cfg.model_aliases.push(trimmed.to_string());
                    }
                }
            }
        }
        if let Some(mp) = val.get_str("model") {
            cfg.model_path = mp.to_string();
            if val.get_str("model_name").unwrap_or("").trim().is_empty()
                && val.get_str("arch").unwrap_or("").trim().is_empty()
                && !mp.is_empty()
            {
                if let Some(stem) = Path::new(mp).file_stem().and_then(|s| s.to_str()) {
                    cfg.model_name = stem.to_string();
                }
            }
        }
        if let Some(exe) = val.get_str("exe").filter(|s| !s.trim().is_empty()) {
            cfg.exe_path = Some(PathBuf::from(exe));
            cfg.mock_mode = false;
        }
        if let Some(cwd) = val.get_str("cwd").filter(|s| !s.trim().is_empty()) {
            cfg.cwd = Some(PathBuf::from(cwd));
        }
        if let Some(log) = val.get_str("log").filter(|s| !s.trim().is_empty()) {
            cfg.log_path = Some(PathBuf::from(log));
        }
        if let Some(tok) = val.get_str("tokenizer").filter(|s| !s.trim().is_empty()) {
            cfg.tokenizer_path = Some(PathBuf::from(tok));
        }
        if let Some(ct) = val
            .get_str("chat_template")
            .filter(|s| !s.trim().is_empty())
        {
            cfg.chat_template_path = Some(PathBuf::from(ct));
        }
        if let Some(stops) = val.get("stop_tokens").and_then(|v| v.as_array()) {
            for item in stops {
                if let Some(s) = item.as_str().filter(|s| !s.is_empty()) {
                    cfg.stop_tokens.push(s.to_string());
                }
            }
        }
        if let Some(key) = val
            .get_str("api_key")
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            cfg.api_key = Some(key.to_string());
        }
        if let Some(c) = val
            .get_usize("context")
            .or_else(|| val.get_usize("ctx_size"))
            .or_else(|| val.get_usize("max_context"))
        {
            if c > 0 {
                cfg.max_context = c;
            }
        }
        if let Some(np) = val.get_i64("n_predict") {
            cfg.n_predict = np;
        }
        if let Some(re) = val
            .get_str("reasoning_effort")
            .filter(|s| !s.trim().is_empty())
        {
            cfg.reasoning_effort = re.trim().to_string();
        }
        if let Some(rb) = val
            .get_usize("reasoning_budget")
            .or_else(|| val.get_usize("reasoning_budget_tokens"))
        {
            cfg.reasoning_budget = rb;
        }
        if let Some(rr) = val.get_bool("reasoning_recovery") {
            cfg.reasoning_recovery = rr;
        }
        if let Some(tcr) = val.get_bool("tool_call_recovery") {
            cfg.tool_call_recovery = tcr;
        }
        if let Some(fmt) = val.get_bool("fit_max_tokens") {
            cfg.fit_max_tokens = fmt;
        }
        if let Some(at) = val.get_str("anthropic_thinking") {
            cfg.anthropic_think_unasked = at.trim().eq_ignore_ascii_case("auto");
        } else if let Some(at_b) = val.get_bool("anthropic_thinking") {
            cfg.anthropic_think_unasked = at_b;
        }
        if let Some(dm) = val.get_usize("draft_max") {
            cfg.draft_max = dm;
        }
        if let Some(dp) = val.get_f64("draft_p_min") {
            cfg.draft_p_min = dp;
        }
        if let Some(np) = val.get_usize("n_parallel") {
            if np >= 1 {
                cfg.n_parallel = np;
            }
        }
        if let Some(ku) = val.get_bool("kv_unified") {
            cfg.kv_unified = ku;
        }
        if let Some(ck) = val.get_str("cache_type_k") {
            cfg.cache_type_k = ck.to_string();
        }
        if let Some(cv) = val.get_str("cache_type_v") {
            cfg.cache_type_v = cv.to_string();
        }
        if let Some(fa) = val.get_str("flash_attn") {
            cfg.flash_attn = fa.to_string();
        }
        if let Some(fit) = val.get_str("fit") {
            cfg.fit = fit.to_string();
        }
        if let Some(cs) = val.get_bool("ctx_shift") {
            cfg.ctx_shift = cs;
        }
        if let Some(sp) = val
            .get_str("slot_save_path")
            .or_else(|| val.get_str("session_dir"))
        {
            cfg.slot_save_path = sp.to_string();
        }
        if let Some(iu) = val
            .get_u64("idle_unload_seconds")
            .or_else(|| val.get_u64("idle_unload_s"))
        {
            cfg.idle_unload_seconds = iu;
        }

        // Sampling defaults from top-level or nested `sampling` object.
        let samp = val.get("sampling").unwrap_or(val);
        if let Some(t) = samp.get_f64("temperature") {
            cfg.sampling.temperature = t;
        }
        if let Some(tp) = samp.get_f64("top_p") {
            cfg.sampling.top_p = tp;
        }
        if let Some(tk) = samp.get_i64("top_k") {
            cfg.sampling.top_k = tk;
        }
        if let Some(mp) = samp.get_f64("min_p") {
            cfg.sampling.min_p = mp;
        }
        if let Some(rp) = samp.get_f64("repeat_penalty") {
            cfg.sampling.repeat_penalty = rp;
        }
        if let Some(pp) = samp.get_f64("presence_penalty") {
            cfg.sampling.presence_penalty = pp;
        }
        if let Some(fp) = samp.get_f64("frequency_penalty") {
            cfg.sampling.frequency_penalty = fp;
        }

        // Parse extra_args if present (matching serve/server.py _extra_arg)
        if let Some(extra) = val.get("extra_args").and_then(|v| v.as_array()) {
            let args: Vec<&str> = extra.iter().filter_map(|v| v.as_str()).collect();
            let mut i = 0;
            while i < args.len() {
                match args[i] {
                    "--draft-max" if i + 1 < args.len() => {
                        if let Ok(v) = args[i + 1].parse() {
                            cfg.draft_max = v;
                        }
                        i += 2;
                    }
                    "--draft-p-min" if i + 1 < args.len() => {
                        if let Ok(v) = args[i + 1].parse() {
                            cfg.draft_p_min = v;
                        }
                        i += 2;
                    }
                    "--parallel" | "-np" if i + 1 < args.len() => {
                        if let Ok(v) = args[i + 1].parse() {
                            cfg.n_parallel = v;
                        }
                        i += 2;
                    }
                    "--cache-type-k" | "-ctk" if i + 1 < args.len() => {
                        cfg.cache_type_k = args[i + 1].to_string();
                        i += 2;
                    }
                    "--cache-type-v" | "-ctv" if i + 1 < args.len() => {
                        cfg.cache_type_v = args[i + 1].to_string();
                        i += 2;
                    }
                    "--flash-attn" | "-fa" if i + 1 < args.len() => {
                        cfg.flash_attn = args[i + 1].to_string();
                        i += 2;
                    }
                    "--fit" if i + 1 < args.len() => {
                        cfg.fit = args[i + 1].to_string();
                        i += 2;
                    }
                    "--slot-save-path" if i + 1 < args.len() => {
                        cfg.slot_save_path = args[i + 1].to_string();
                        i += 2;
                    }
                    "--ctx-shift" => {
                        cfg.ctx_shift = true;
                        i += 1;
                    }
                    "--no-ctx-shift" => {
                        cfg.ctx_shift = false;
                        i += 1;
                    }
                    "--kv-unified" | "-kvu" => {
                        cfg.kv_unified = true;
                        i += 1;
                    }
                    _ => {
                        i += 1;
                    }
                }
            }
        }

        if let Some(cd) = val.get_str("control_dir").filter(|s| !s.trim().is_empty()) {
            cfg.control_dir = Some(PathBuf::from(cd.trim()));
        }
        if val.get_bool("mock") == Some(true) {
            cfg.mock_mode = true;
        }
        if let Some(scripts) = val.get("mock_scripts").and_then(|v| v.as_array()) {
            for item in scripts {
                if let Some(s) = item.as_str() {
                    cfg.mock_scripts.push(s.to_string());
                }
            }
        }

        // Parse "args" or "engine_args" (matching serve/runconfig.py)
        if let Some(ea) = val
            .get("engine_args")
            .or_else(|| val.get("args"))
            .and_then(|v| v.as_array())
        {
            let raw_args: Vec<String> = ea
                .iter()
                .filter_map(|v| v.as_str().map(ToString::to_string))
                .collect();
            let mut i = 0;
            while i < raw_args.len() {
                match raw_args[i].as_str() {
                    "--reasoning-effort" if i + 1 < raw_args.len() => {
                        if val.get_str("reasoning_effort").is_none() {
                            cfg.reasoning_effort = raw_args[i + 1].clone();
                        }
                        i += 2;
                        continue;
                    }
                    "--max-context" | "--ctx" | "-c" if i + 1 < raw_args.len() => {
                        if let Ok(c) = raw_args[i + 1].parse::<usize>() {
                            if c > 0 {
                                cfg.max_context = c;
                            }
                        }
                        cfg.engine_args.push(raw_args[i].clone());
                        cfg.engine_args.push(raw_args[i + 1].clone());
                        i += 2;
                        continue;
                    }
                    "--model" | "-m" if i + 1 < raw_args.len() => {
                        if cfg.model_path.is_empty() {
                            cfg.model_path = raw_args[i + 1].clone();
                        }
                        cfg.engine_args.push(raw_args[i].clone());
                        cfg.engine_args.push(raw_args[i + 1].clone());
                        i += 2;
                        continue;
                    }
                    "--native" | "--pack" if i + 1 < raw_args.len() => {
                        if cfg.model_path.is_empty() {
                            cfg.model_path = raw_args[i + 1].clone();
                        }
                        cfg.engine_args.push(raw_args[i].clone());
                        cfg.engine_args.push(raw_args[i + 1].clone());
                        i += 2;
                        continue;
                    }
                    "--spec" if i + 1 < raw_args.len() => {
                        if let Ok(d) = raw_args[i + 1].parse::<usize>() {
                            if cfg.draft_max == 0 {
                                cfg.draft_max = d;
                            }
                        }
                        cfg.engine_args.push(raw_args[i].clone());
                        cfg.engine_args.push(raw_args[i + 1].clone());
                        i += 2;
                        continue;
                    }
                    _ => {
                        cfg.engine_args.push(raw_args[i].clone());
                        i += 1;
                    }
                }
            }
        }

        // Parse "env" object and "lib_dirs" array (matching serve/runconfig.py)
        if let Some(env_obj) = val.get("env").and_then(|v| v.as_object()) {
            for (k, v) in env_obj {
                let s = match v {
                    JsonValue::Str(s) => s.clone(),
                    other => other.to_json_string(),
                };
                cfg.engine_env.push((k.clone(), s));
            }
        }
        // Disk tier of the prompt cache: run-config keys for the engine's TALUS_PCACHE_DISK* environment
        // ("prompt_cache_disk": "auto" | folder | false, "prompt_cache_disk_gib": N). An explicit "env" entry wins.
        let disk = match val.get("prompt_cache_disk") {
            Some(JsonValue::Str(s)) => Some(s.clone()),
            Some(JsonValue::Bool(true)) => Some("auto".to_string()),
            Some(JsonValue::Bool(false)) => Some("off".to_string()),
            _ => None,
        };
        let disk_gib = val
            .get("prompt_cache_disk_gib")
            .and_then(|v| v.as_f64())
            .map(|g| format!("{g}"));
        for (k, v) in [
            ("TALUS_PCACHE_DISK", disk),
            ("TALUS_PCACHE_DISK_GIB", disk_gib),
        ] {
            if let Some(v) = v {
                if !cfg.engine_env.iter().any(|(ek, _)| ek == k) {
                    cfg.engine_env.push((k.to_string(), v));
                }
            }
        }
        if let Some(libs) = val.get("lib_dirs").and_then(|v| v.as_array()) {
            for item in libs {
                if let Some(s) = item.as_str().filter(|s| !s.is_empty()) {
                    cfg.lib_dirs.push(s.to_string());
                }
            }
            if !cfg.lib_dirs.is_empty() {
                let joined = cfg.lib_dirs.join(":");
                let existing = cfg
                    .engine_env
                    .iter()
                    .find(|(k, _)| k == "LD_LIBRARY_PATH")
                    .map(|(_, v)| v.clone())
                    .or_else(|| std::env::var("LD_LIBRARY_PATH").ok());
                let full_ld = match existing.filter(|s| !s.is_empty()) {
                    Some(prev) => format!("{joined}:{prev}"),
                    None => joined,
                };
                if let Some(slot) = cfg
                    .engine_env
                    .iter_mut()
                    .find(|(k, _)| k == "LD_LIBRARY_PATH")
                {
                    slot.1 = full_ld;
                } else {
                    cfg.engine_env
                        .push(("LD_LIBRARY_PATH".to_string(), full_ld));
                }
            }
        }

        Ok(cfg)
    }

    pub fn parse_cli_args<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let argv: Vec<String> = args.into_iter().map(|s| s.as_ref().to_string()).collect();
        let mut config_path: Option<PathBuf> = None;
        let mut host_override: Option<String> = None;
        let mut port_override: Option<u16> = None;
        let mut model_name_override: Option<String> = None;
        let mut model_aliases_override: Vec<String> = Vec::new();
        let mut api_key_override: Option<String> = None;
        let mut max_context_override: Option<usize> = None;
        let mut slot_save_override: Option<String> = None;
        let mut model_path_override: Option<String> = None;
        let mut tokenizer_override: Option<PathBuf> = None;
        let mut chat_template_override: Option<PathBuf> = None;
        let mut idle_unload_override: Option<u64> = None;
        let mut fit_max_tokens_override: Option<bool> = None;
        let mut mock_override: Option<bool> = None;
        let mut mock_stdio_worker = false;
        let mut mock_scripts_override: Vec<String> = Vec::new();
        let mut extra_engine_args: Vec<String> = Vec::new();
        let mut control_dir_override: Option<PathBuf> = None;

        let mut i = 0;
        while i < argv.len() {
            match argv[i].as_str() {
                "--config" | "-c" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--config requires a path")?;
                    config_path = Some(PathBuf::from(val));
                }
                "--host" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--host requires a value")?;
                    host_override = Some(val.clone());
                }
                "--port" | "-p" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--port requires a value")?;
                    let p: u16 = val
                        .parse()
                        .map_err(|_| format!("invalid --port value: {val}"))?;
                    port_override = Some(p);
                }
                "--model" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--model requires a path")?;
                    model_path_override = Some(val.clone());
                }
                "--tokenizer" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--tokenizer requires a path")?;
                    tokenizer_override = Some(PathBuf::from(val));
                }
                "--chat-template" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--chat-template requires a path")?;
                    chat_template_override = Some(PathBuf::from(val));
                }
                "--model-name" | "--alias" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--model-name requires a value")?;
                    if model_name_override.is_none() {
                        model_name_override = Some(val.clone());
                    } else {
                        model_aliases_override.push(val.clone());
                    }
                }
                "--model-alias" | "--model-aliases" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--model-alias requires a value")?;
                    for part in val.split(',') {
                        let trimmed = part.trim();
                        if !trimmed.is_empty() {
                            model_aliases_override.push(trimmed.to_string());
                        }
                    }
                }
                "--api-key" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--api-key requires a value")?;
                    let trimmed = val.trim().to_string();
                    if trimmed.is_empty() {
                        return Err("an API key was given but it is empty".to_string());
                    }
                    api_key_override = Some(trimmed);
                }
                "--context" | "--ctx-size" | "--max-context" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--context requires a value")?;
                    let c: usize = val
                        .parse()
                        .map_err(|_| format!("invalid --context value: {val}"))?;
                    max_context_override = Some(c);
                }
                "--idle-unload" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--idle-unload requires a value")?;
                    let secs: u64 = val
                        .parse()
                        .map_err(|_| format!("invalid --idle-unload value: {val}"))?;
                    idle_unload_override = Some(secs);
                }
                "--slot-save-path" | "--session-dir" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--slot-save-path requires a path")?;
                    slot_save_override = Some(val.clone());
                }
                "--engine" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--engine requires a value")?;
                    match val.as_str() {
                        "mock" => mock_override = Some(true),
                        "strata" => mock_override = Some(false),
                        other => {
                            // Treat a path to an engine executable as exe_path
                            if Path::new(other).exists() || other.contains('/') {
                                mock_override = Some(false);
                                extra_engine_args.insert(0, format!("__EXE__:{other}"));
                            } else {
                                return Err(format!("invalid --engine value: {other}"));
                            }
                        }
                    }
                }
                "--control-dir" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--control-dir requires a path")?;
                    control_dir_override = Some(PathBuf::from(val));
                }
                "--fit-max-tokens" => {
                    fit_max_tokens_override = Some(true);
                }
                "--no-fit-max-tokens" => {
                    fit_max_tokens_override = Some(false);
                }
                "--mock" => {
                    mock_override = Some(true);
                }
                "--mock-stdio-worker" | "--serve" => {
                    mock_stdio_worker = true;
                }
                "--mock-script" | "--script" => {
                    i += 1;
                    let val = argv.get(i).ok_or("--script requires a value")?;
                    mock_scripts_override.push(val.clone());
                }
                flag @ ("--mtp-gguf"
                | "--mtp"
                | "--spec-draft-vocab"
                | "--draft-vocab"
                | "--ngram"
                | "--dflash-gguf"
                | "--spec"
                | "--native"
                | "--pack"
                | "--prefill"
                | "--kv-type"
                | "--kv-Auto"
                | "--expert-cache"
                | "--expert-slots"
                | "--resident-budget-gib") => {
                    i += 1;
                    let val = argv
                        .get(i)
                        .ok_or_else(|| format!("{flag} requires a value"))?;
                    extra_engine_args.push(flag.to_string());
                    extra_engine_args.push(val.clone());
                }
                flag @ ("--resident-experts" | "--mmap-experts" | "--no-mtp" | "--no-ngram"
                | "--pin-arena") => {
                    extra_engine_args.push(flag.to_string());
                }
                other if !other.starts_with('-') && config_path.is_none() => {
                    config_path = Some(PathBuf::from(other));
                }
                other => {
                    return Err(format!("unrecognized argument: {other}"));
                }
            }
            i += 1;
        }

        // With a control directory and no run config, serve what talus-tui applied last (`active.json`).
        if config_path.is_none() {
            if let Some(active) = control_dir_override
                .as_ref()
                .map(|d| d.join(crate::control::ACTIVE_CONFIG))
                .filter(|p| p.exists())
            {
                config_path = Some(active);
            }
        }

        let mut cfg = if let Some(ref p) = config_path {
            Self::from_json_file(p)?
        } else {
            Self::default()
        };

        if let Some(h) = host_override {
            cfg.host = h;
        }
        if let Some(p) = port_override {
            cfg.port = p;
        }
        if let Some(mp) = model_path_override {
            cfg.model_path = mp;
        }
        if let Some(tok) = tokenizer_override {
            cfg.tokenizer_path = Some(tok);
        }
        if let Some(ct) = chat_template_override {
            cfg.chat_template_path = Some(ct);
        }
        if let Some(iu) = idle_unload_override {
            cfg.idle_unload_seconds = iu;
        }
        if let Some(fmt) = fit_max_tokens_override {
            cfg.fit_max_tokens = fmt;
        }
        if let Some(m) = model_name_override {
            cfg.model_name = m;
        }
        for alias in model_aliases_override {
            if alias != cfg.model_name && !cfg.model_aliases.contains(&alias) {
                cfg.model_aliases.push(alias);
            }
        }
        if api_key_override.is_some() {
            cfg.api_key = api_key_override;
        } else if let Ok(env_key) = std::env::var("STRATA_API_KEY") {
            let trimmed = env_key.trim().to_string();
            if trimmed.is_empty() {
                return Err("an API key was given but it is empty".to_string());
            }
            cfg.api_key = Some(trimmed);
        }
        if let Some(c) = max_context_override {
            cfg.max_context = c;
            // Also update or append --max-context in engine_args if engine_args is non-empty
            if !cfg.engine_args.is_empty() {
                let mut found = false;
                let mut idx = 0;
                while idx + 1 < cfg.engine_args.len() {
                    if matches!(
                        cfg.engine_args[idx].as_str(),
                        "--max-context" | "--ctx" | "-c"
                    ) {
                        cfg.engine_args[idx + 1] = c.to_string();
                        found = true;
                        break;
                    }
                    idx += 1;
                }
                if !found {
                    cfg.engine_args.push("--max-context".to_string());
                    cfg.engine_args.push(c.to_string());
                }
            }
        }
        if let Some(sp) = slot_save_override {
            cfg.slot_save_path = sp;
        }
        if let Some(m) = mock_override {
            cfg.mock_mode = m;
        }
        for ea in extra_engine_args {
            if let Some(exe) = ea.strip_prefix("__EXE__:") {
                cfg.exe_path = Some(PathBuf::from(exe));
            } else {
                cfg.engine_args.push(ea);
            }
        }
        if !mock_scripts_override.is_empty() {
            cfg.mock_scripts = mock_scripts_override;
        }
        cfg.mock_stdio_worker = mock_stdio_worker;
        if control_dir_override.is_some() {
            cfg.control_dir = control_dir_override;
        }
        Ok(cfg)
    }

    pub fn matches_model(&self, requested: Option<&str>) -> bool {
        let Some(req) = requested.map(str::trim).filter(|s| !s.is_empty()) else {
            return true;
        };
        if req == self.model_name || req == "default" {
            return true;
        }
        self.model_aliases.iter().any(|a| a == req)
    }

    pub fn all_model_names(&self) -> Vec<String> {
        let mut names = vec![self.model_name.clone()];
        for a in &self.model_aliases {
            if !names.contains(a) {
                names.push(a.clone());
            }
        }
        names
    }
}
