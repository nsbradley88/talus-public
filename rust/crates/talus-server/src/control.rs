//! talus-tui support: a compact status snapshot and a small control endpoint.
//!
//! - `GET /v1/talus/status`: what the TUI's status screen shows, built from counters the server already keeps
//!   (no engine call, nothing per token). The TUI polls it every one to two seconds.
//! - `GET /v1/talus/presets`: which models have a run config in the control directory, and the current selection.
//! - `POST /v1/talus/apply`: switch the model and/or its tuning. Off unless the server runs with `--control-dir DIR`.
//!   The body names a model id and option choices from `talus-presets`; the server builds the new run config from
//!   `DIR/<model-id>.json` (installed by the operator) plus the catalog's edits, writes `DIR/active.json`, stops the
//!   engine and re-executes itself with `--config DIR/active.json`. A client can never send raw arguments or
//!   environment variables. Only from this computer unless the server has an API key.

use crate::json::{unix_timestamp_secs, JsonValue};
use crate::server::ServiceState;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use talus_presets::{apply_edits, describe_edit, Bind, Selection, MIN_API_KEY_LEN};

/// The generated run config in the control directory.
pub const ACTIVE_CONFIG: &str = "active.json";
/// The last run config that started (the fallback when a new one fails to start).
pub const LAST_GOOD_CONFIG: &str = "last-good.json";
/// Why the last applied config failed to start (cleared by the next apply).
pub const LAST_ERROR: &str = "last-error.json";
/// Key of the TUI's selection record inside `active.json` (ignored by the run-config parser).
pub const SELECTION_KEY: &str = "_talus_tui";
/// How long a restart waits for running requests to finish.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(120);
/// Recent requests in the status snapshot.
const RECENT: usize = 20;

type RestartHook = Box<dyn Fn(&Path) + Send + Sync>;

pub struct ControlState {
    pub dir: Option<PathBuf>,
    pub restarting: AtomicBool,
    hook: Mutex<Option<RestartHook>>,
}

impl ControlState {
    pub fn new(dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            restarting: AtomicBool::new(false),
            hook: Mutex::new(None),
        }
    }

    /// Replace the restart (engine shutdown + re-exec) with `f`, called with the new run config's path. For tests.
    pub fn set_restart_hook(&self, f: impl Fn(&Path) + Send + Sync + 'static) {
        *self.hook.lock().unwrap() = Some(Box::new(f));
    }
}

pub struct PostReply {
    pub code: u16,
    pub body: JsonValue,
    /// Restart with this run config once the reply is sent.
    pub restart: Option<PathBuf>,
}

fn error(code: u16, msg: impl Into<String>) -> PostReply {
    PostReply {
        code,
        body: JsonValue::obj(vec![(
            "error",
            JsonValue::obj(vec![
                ("type", JsonValue::from("invalid_request_error")),
                ("message", JsonValue::from(msg.into())),
            ]),
        )]),
        restart: None,
    }
}

/// `MemAvailable` and `MemTotal` in KiB from `/proc/meminfo` text.
pub fn parse_meminfo(text: &str) -> Option<(u64, u64)> {
    let field = |name: &str| -> Option<u64> {
        text.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l[name.len()..].split_whitespace().next())
            .and_then(|v| v.parse().ok())
    };
    Some((field("MemAvailable:")?, field("MemTotal:")?))
}

fn memory_json() -> JsonValue {
    match std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| parse_meminfo(&t))
    {
        Some((avail, total)) => {
            let gib = |kib: u64| ((kib as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0;
            JsonValue::obj(vec![
                ("available_gib", JsonValue::from(gib(avail))),
                ("total_gib", JsonValue::from(gib(total))),
            ])
        }
        None => JsonValue::Null,
    }
}

fn selection_json(state: &ServiceState) -> JsonValue {
    state
        .config
        .raw_config
        .get(SELECTION_KEY)
        .cloned()
        .unwrap_or(JsonValue::Null)
}

/// The catalog id of the model being served: the TUI's record, else a catalog id equal to the model name.
fn current_model_id(state: &ServiceState) -> Option<String> {
    if let Some(id) = state
        .config
        .raw_config
        .get(SELECTION_KEY)
        .and_then(|s| s.get_str("model"))
    {
        return Some(id.to_string());
    }
    talus_presets::model(&state.config.model_name).map(|m| m.id.to_string())
}

fn key_from_env() -> bool {
    std::env::var("STRATA_API_KEY")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
}

fn last_error_json(state: &ServiceState) -> JsonValue {
    state
        .control
        .dir
        .as_ref()
        .and_then(|d| std::fs::read_to_string(d.join(LAST_ERROR)).ok())
        .and_then(|t| JsonValue::parse(&t).ok())
        .unwrap_or(JsonValue::Null)
}

/// After the server started serving: remember its run config as the fallback for a failed apply.
pub fn mark_good(cfg: &crate::config::ServerConfig) {
    let (Some(dir), Some(path)) = (cfg.control_dir.as_ref(), cfg.config_path.as_ref()) else {
        return;
    };
    if let Ok(text) = std::fs::read_to_string(path) {
        let _ = write_private(&dir.join(LAST_GOOD_CONFIG), &text);
    }
}

/// The server could not start with `cfg`. When that config is the control directory's `active.json` (written by an
/// apply) and a last good config exists, record the error, put the last good config back as `active.json` and
/// return it, so the server comes back with the previous settings instead of staying down.
pub fn recover_after_failed_start(
    cfg: &crate::config::ServerConfig,
    err: &str,
) -> Option<crate::config::ServerConfig> {
    let dir = cfg.control_dir.as_ref()?;
    let active = dir.join(ACTIVE_CONFIG);
    if cfg.config_path.as_deref() != Some(active.as_path()) {
        return None;
    }
    let good = std::fs::read_to_string(dir.join(LAST_GOOD_CONFIG)).ok()?;
    let failed_sel = cfg
        .raw_config
        .get(SELECTION_KEY)
        .cloned()
        .unwrap_or(JsonValue::Null);
    let record = JsonValue::obj(vec![
        ("error", JsonValue::from(err)),
        ("selection", failed_sel),
        ("at", JsonValue::from(unix_timestamp_secs())),
    ]);
    let _ = write_private(&dir.join(LAST_ERROR), &record.to_json_pretty());
    let _ = std::fs::rename(&active, dir.join("active.failed.json"));
    write_private(&active, &good).ok()?;
    eprintln!("[talus-server] control: the new settings did not start ({err}); back to the last good config");
    let argv: Vec<String> = std::env::args().skip(1).collect();
    crate::config::ServerConfig::parse_cli_args(exec_args(&argv, &active)).ok()
}

pub fn status_json(state: &ServiceState) -> JsonValue {
    let running = state.running_requests.load(Ordering::Relaxed);
    let waiting = state.waiting_requests.load(Ordering::Relaxed);
    let loaded = state.engine.is_loaded();
    let m = state.metrics.lock().unwrap();
    let in_use = if running > 0 {
        state.active_prompt_tokens.load(Ordering::Relaxed)
            + state.active_generated_tokens.load(Ordering::Relaxed)
    } else {
        m.last_prompt_tokens + m.last_generated_tokens
    };
    let recent: Vec<JsonValue> = m.history.iter().rev().take(RECENT).cloned().collect();
    let totals = JsonValue::obj(vec![
        ("requests", JsonValue::from(m.requests_total)),
        ("prompt_tokens", JsonValue::from(m.prompt_tokens_total)),
        ("reused", JsonValue::from(m.reused_tokens_total)),
        ("output_tokens", JsonValue::from(m.output_tokens_total)),
        ("prompt_ms", JsonValue::from(m.prompt_ms_total)),
        ("decode_ms", JsonValue::from(m.decode_ms_total)),
        ("drafts_offered", JsonValue::from(m.drafts_offered_total)),
        ("drafts_accepted", JsonValue::from(m.drafts_accepted_total)),
    ]);
    drop(m);
    let st = if state.control.restarting.load(Ordering::Relaxed) {
        "restarting"
    } else if !loaded {
        "unloaded"
    } else if running > 0 {
        "generating"
    } else {
        "idle"
    };
    JsonValue::obj(vec![
        ("service", JsonValue::from("talus")),
        ("model", JsonValue::from(state.config.model_name.as_str())),
        (
            "preset",
            current_model_id(state)
                .map(JsonValue::from)
                .unwrap_or(JsonValue::Null),
        ),
        ("state", JsonValue::from(st)),
        ("loaded", JsonValue::from(loaded)),
        ("running", JsonValue::from(running)),
        ("queued", JsonValue::from(waiting)),
        (
            "uptime_s",
            JsonValue::from(state.started_instant.elapsed().as_secs()),
        ),
        (
            "context",
            JsonValue::obj(vec![
                ("max", JsonValue::from(state.reported_ctx())),
                ("in_use", JsonValue::from(in_use)),
            ]),
        ),
        ("memory", memory_json()),
        ("totals", totals),
        ("recent", JsonValue::Array(recent)),
        (
            "server",
            JsonValue::obj(vec![
                ("host", JsonValue::from(state.config.host.as_str())),
                ("port", JsonValue::from(state.config.port as u64)),
                ("api_key", JsonValue::from(state.config.api_key.is_some())),
                ("api_key_from_env", JsonValue::from(key_from_env())),
            ]),
        ),
        (
            "control",
            JsonValue::obj(vec![
                ("enabled", JsonValue::from(state.control.dir.is_some())),
                ("selection", selection_json(state)),
                ("last_error", last_error_json(state)),
            ]),
        ),
    ])
}

/// The `--model` GGUF of the run config `DIR/<id>.json`, if it names one.
fn config_gguf(dir: &Path, id: &str) -> Option<PathBuf> {
    let text = std::fs::read_to_string(dir.join(format!("{id}.json"))).ok()?;
    let cfg = JsonValue::parse(&text).ok()?;
    let key = if cfg.contains_key("engine_args") && !cfg.contains_key("args") {
        "engine_args"
    } else {
        "args"
    };
    let args = string_array(cfg.get(key)).ok()?;
    let i = args.iter().position(|a| a == "--model")?;
    args.get(i + 1).map(PathBuf::from)
}

/// Choices of model `id` whose weights file (talus_presets::FILE_NEEDS) is not next to its GGUF:
/// (option, value, file).
fn missing_files(dir: &Path, id: &str) -> Vec<(&'static str, &'static str, PathBuf)> {
    let Some(g) = config_gguf(dir, id) else {
        return Vec::new();
    };
    talus_presets::FILE_NEEDS
        .iter()
        .filter(|(m, ..)| *m == id)
        .filter_map(|&(_, o, v, _)| {
            let f = talus_presets::needed_file(id, o, v, &g)?;
            (!f.is_file()).then_some((o, v, f))
        })
        .collect()
}

pub fn presets_json(state: &ServiceState) -> JsonValue {
    let dir = state.control.dir.as_deref();
    let models: Vec<JsonValue> = talus_presets::MODELS
        .iter()
        .map(|m| {
            let available = dir
                .map(|d| d.join(format!("{}.json", m.id)).is_file())
                .unwrap_or(false);
            // choices that need a weights file the model directory lacks ("weights not installed" in the TUI)
            let missing: Vec<JsonValue> = match dir {
                Some(d) if available => missing_files(d, m.id)
                    .into_iter()
                    .map(|(o, v, f)| {
                        JsonValue::obj(vec![
                            ("option", JsonValue::from(o)),
                            ("value", JsonValue::from(v)),
                            ("file", JsonValue::from(f.to_string_lossy().into_owned())),
                        ])
                    })
                    .collect(),
                _ => Vec::new(),
            };
            JsonValue::obj(vec![
                ("id", JsonValue::from(m.id)),
                ("name", JsonValue::from(m.name)),
                ("available", JsonValue::from(available)),
                ("missing", JsonValue::Array(missing)),
            ])
        })
        .collect();
    JsonValue::obj(vec![
        ("enabled", JsonValue::from(dir.is_some())),
        ("models", JsonValue::Array(models)),
        (
            "active",
            current_model_id(state)
                .map(JsonValue::from)
                .unwrap_or(JsonValue::Null),
        ),
        ("selection", selection_json(state)),
    ])
}

pub fn handle_get(path: &str, state: &ServiceState) -> (u16, JsonValue) {
    match path {
        "/v1/talus/status" => (200, status_json(state)),
        "/v1/talus/presets" => (200, presets_json(state)),
        _ => (
            404,
            JsonValue::obj(vec![(
                "error",
                JsonValue::obj(vec![("message", JsonValue::from("not found"))]),
            )]),
        ),
    }
}

fn parse_selection(body: &JsonValue) -> Result<Selection, String> {
    let mut sel = Selection::new();
    match body.get("choices") {
        None | Some(JsonValue::Null) => {}
        Some(JsonValue::Object(entries)) => {
            for (k, v) in entries {
                let v = v
                    .as_str()
                    .ok_or_else(|| format!("choice {k:?} must be a string"))?;
                sel.insert(k.clone(), v.to_string());
            }
        }
        Some(_) => return Err("choices must be an object".to_string()),
    }
    Ok(sel)
}

fn string_array(v: Option<&JsonValue>) -> Result<Vec<String>, String> {
    match v {
        None => Ok(Vec::new()),
        Some(JsonValue::Array(items)) => items
            .iter()
            .map(|i| {
                i.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| "args must be strings".to_string())
            })
            .collect(),
        Some(_) => Err("args must be an array".to_string()),
    }
}

/// Build the new run config. Returns (config, human-readable changes).
pub fn build_config(
    state: &ServiceState,
    dir: &Path,
    body: &JsonValue,
) -> Result<(JsonValue, Vec<String>), PostReply> {
    let id = body
        .get_str("model")
        .ok_or_else(|| error(400, "the body needs \"model\": a model id"))?;
    let spec =
        talus_presets::model(id).ok_or_else(|| error(400, format!("unknown model {id:?}")))?;
    let sel = parse_selection(body).map_err(|e| error(400, e))?;
    let edits = spec.edits(&sel).map_err(|e| error(400, e))?;
    // a non-default choice whose weights file is missing would only fall back inside the engine: refuse it here
    for (o, v, f) in missing_files(dir, id) {
        let chosen = sel.get(o).map(String::as_str);
        let default = spec.option(o).map(|x| x.default);
        if chosen == Some(v) && default != Some(v) {
            return Err(error(
                409,
                format!("weights not installed: {} (needed by {o}={v})", f.display()),
            ));
        }
    }

    let base_path = dir.join(format!("{id}.json"));
    let text = std::fs::read_to_string(&base_path).map_err(|_| {
        error(
            404,
            format!(
                "no run config for {id}: put one at {} (see docs/TUI.md)",
                base_path.display()
            ),
        )
    })?;
    let mut cfg = JsonValue::parse(&text)
        .ok()
        .filter(JsonValue::is_object)
        .ok_or_else(|| error(500, format!("{} is not a JSON object", base_path.display())))?;

    let args_key = if cfg.contains_key("engine_args") && !cfg.contains_key("args") {
        "engine_args"
    } else {
        "args"
    };
    let mut args = string_array(cfg.get(args_key)).map_err(|e| error(500, e))?;
    let mut env: Vec<(String, String)> = cfg
        .get("env")
        .and_then(JsonValue::as_object)
        .map(|o| {
            o.iter()
                .map(|(k, v)| {
                    let s = match v {
                        JsonValue::Str(s) => s.clone(),
                        other => other.to_json_string(),
                    };
                    (k.clone(), s)
                })
                .collect()
        })
        .unwrap_or_default();
    apply_edits(&mut args, &mut env, &edits);
    cfg.insert(
        args_key,
        JsonValue::Array(args.into_iter().map(JsonValue::from).collect()),
    );
    if !env.is_empty() {
        cfg.insert(
            "env",
            JsonValue::Object(
                env.into_iter()
                    .map(|(k, v)| (k, JsonValue::from(v)))
                    .collect(),
            ),
        );
    }

    // Server settings: bind, port, API key.
    let bind = match body.get_str("bind") {
        None => Bind::from_host(&state.config.host),
        Some(b) => Bind::parse(b).ok_or_else(|| error(400, "bind must be \"local\" or \"lan\""))?,
    };
    let host = match (body.get_str("bind"), bind) {
        // keep a custom host (e.g. a tailnet address) unless the client chose a bind
        (None, _) => state.config.host.clone(),
        (Some(_), b) => b.host().to_string(),
    };
    cfg.insert("host", host.as_str());
    cfg.insert("port", state.config.port as u64);
    let env_key = key_from_env();
    let mut has_key = env_key;
    match body.get_str("api_key") {
        Some(k) => {
            if env_key {
                return Err(error(
                    409,
                    "the API key comes from TALUS_API_KEY (or STRATA_API_KEY) in the service environment; change it there",
                ));
            }
            if k.trim().len() < MIN_API_KEY_LEN || k.chars().any(char::is_whitespace) {
                return Err(error(
                    400,
                    format!("an API key needs at least {MIN_API_KEY_LEN} characters and no spaces"),
                ));
            }
            cfg.insert("api_key", k.trim());
            has_key = true;
        }
        None => {
            if let Some(k) = state.config.raw_config.get_str("api_key") {
                cfg.insert("api_key", k);
                has_key = true;
            } else if state.config.api_key.is_some() && !env_key {
                // a key given with --api-key on the command line: re-exec drops that flag, so keep the key
                if let Some(k) = state.config.api_key.as_deref() {
                    cfg.insert("api_key", k);
                    has_key = true;
                }
            }
        }
    }
    if Bind::from_host(&host) == Bind::Lan && !has_key {
        return Err(error(
            400,
            "local network access needs an API key: set one, or keep \"This computer only\"",
        ));
    }
    cfg.insert("control_dir", dir.to_string_lossy().as_ref());

    let level = spec
        .level_of(&sel)
        .map(|l| l.id())
        .unwrap_or("custom")
        .to_string();
    let norm = spec.normalize(&sel);
    cfg.insert(
        SELECTION_KEY,
        JsonValue::obj(vec![
            ("model", JsonValue::from(id)),
            ("level", JsonValue::from(level)),
            (
                "choices",
                JsonValue::Object(
                    norm.iter()
                        .map(|(k, v)| (k.clone(), JsonValue::from(v.as_str())))
                        .collect(),
                ),
            ),
            ("bind", JsonValue::from(Bind::from_host(&host).id())),
            ("applied_at", JsonValue::from(unix_timestamp_secs())),
        ]),
    );

    let mut changes = Vec::new();
    if current_model_id(state).as_deref() != Some(id) {
        changes.push(format!(
            "model: {} -> {}",
            state.config.model_name, spec.name
        ));
    }
    changes.extend(edits.iter().map(describe_edit));
    if host != state.config.host {
        changes.push(format!("listen on {host}"));
    }
    if body.get_str("api_key").is_some() {
        changes.push("new API key".to_string());
    }
    Ok((cfg, changes))
}

fn write_private(path: &Path, text: &str) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    {
        use std::io::Write;
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp)?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

pub fn handle_post(
    path: &str,
    body: &JsonValue,
    state: &ServiceState,
    peer_is_local: bool,
) -> PostReply {
    if path != "/v1/talus/apply" {
        return error(404, "not found");
    }
    let Some(dir) = state.control.dir.clone() else {
        return error(
            403,
            "control is off: start talus-server with --control-dir DIR (see docs/TUI.md)",
        );
    };
    if !peer_is_local && state.config.api_key.is_none() {
        return error(
            403,
            "changes are accepted only from this computer unless the server has an API key",
        );
    }
    if state.control.restarting.load(Ordering::SeqCst) {
        return error(409, "a restart is already under way");
    }
    let (cfg, changes) = match build_config(state, &dir, body) {
        Ok(v) => v,
        Err(reply) => return reply,
    };
    let model = body.get_str("model").unwrap_or_default().to_string();
    let changes_json = JsonValue::Array(changes.into_iter().map(JsonValue::from).collect());
    if body.get_bool("dry_run") == Some(true) {
        return PostReply {
            code: 200,
            body: JsonValue::obj(vec![
                ("status", JsonValue::from("dry_run")),
                ("model", JsonValue::from(model)),
                ("changes", changes_json),
            ]),
            restart: None,
        };
    }
    let active = dir.join(ACTIVE_CONFIG);
    let _ = std::fs::remove_file(dir.join(LAST_ERROR));
    if let Err(e) = write_private(&active, &cfg.to_json_pretty()) {
        return error(500, format!("cannot write {}: {e}", active.display()));
    }
    state.control.restarting.store(true, Ordering::SeqCst);
    PostReply {
        code: 202,
        body: JsonValue::obj(vec![
            ("status", JsonValue::from("restarting")),
            ("model", JsonValue::from(model)),
            ("changes", changes_json),
        ]),
        restart: Some(active),
    }
}

/// The server's arguments for the re-exec: the run config replaced by `config`, and only the flags that are not
/// part of a run config kept (port, control directory, mock mode, idle unload, slot directory). Everything else
/// (host, key, model, engine flags, a positional config) is dropped: the new run config decides it.
pub fn exec_args(orig: &[String], config: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < orig.len() {
        let a = orig[i].as_str();
        match a {
            "--port" | "-p" | "--control-dir" | "--idle-unload" | "--slot-save-path"
            | "--session-dir" => {
                if let Some(v) = orig.get(i + 1) {
                    out.push(a.to_string());
                    out.push(v.clone());
                }
                i += 2;
                continue;
            }
            "--mock" | "--fit-max-tokens" | "--no-fit-max-tokens" => out.push(a.to_string()),
            // flags with a value that the run config now decides (bare flags and positionals fall through)
            "--config"
            | "-c"
            | "--host"
            | "--model"
            | "--tokenizer"
            | "--chat-template"
            | "--model-name"
            | "--alias"
            | "--model-alias"
            | "--model-aliases"
            | "--api-key"
            | "--context"
            | "--ctx-size"
            | "--max-context"
            | "--engine"
            | "--mock-script"
            | "--script"
            | "--mtp-gguf"
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
            | "--resident-budget-gib" => {
                i += 2;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    out.push("--config".to_string());
    out.push(config.to_string_lossy().into_owned());
    out
}

/// Wait for running requests to finish, stop the engine, and replace this process with one serving `config`.
pub fn spawn_restart(state: Arc<ServiceState>, config: PathBuf) {
    std::thread::spawn(move || {
        // let the reply reach the client first
        std::thread::sleep(Duration::from_millis(300));
        let deadline = Instant::now() + DRAIN_TIMEOUT;
        while state.running_requests.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(200));
        }
        if let Some(hook) = state.control.hook.lock().unwrap().as_ref() {
            hook(&config);
            return;
        }
        eprintln!(
            "[talus-server] control: restarting with {}",
            config.display()
        );
        let t0 = Instant::now();
        state.engine.shutdown();
        eprintln!(
            "[talus-server] control: engine stopped in {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        exec_self(&config);
    });
}

/// This server's executable. When the binary was replaced while running (an upgrade), `/proc/self/exe` names the
/// deleted file ("<path> (deleted)"): use the new file at the same path, else argv[0].
pub fn exe_candidate(current: Option<PathBuf>, argv0: Option<String>) -> Option<PathBuf> {
    if let Some(p) = current {
        if p.exists() {
            return Some(p);
        }
        let s = p.to_string_lossy();
        if let Some(stripped) = s.strip_suffix(" (deleted)") {
            let q = PathBuf::from(stripped);
            if q.exists() {
                return Some(q);
            }
        }
    }
    argv0.map(PathBuf::from).filter(|p| p.exists())
}

fn own_exe() -> Option<PathBuf> {
    exe_candidate(std::env::current_exe().ok(), std::env::args().next())
}

#[cfg(unix)]
fn exec_self(config: &Path) {
    use std::os::unix::process::CommandExt;
    let orig: Vec<String> = std::env::args().skip(1).collect();
    let args = exec_args(&orig, config);
    match own_exe() {
        Some(exe) => {
            let err = std::process::Command::new(exe).args(&args).exec();
            eprintln!("[talus-server] control: re-exec failed: {err}");
        }
        None => eprintln!("[talus-server] control: cannot find own executable"),
    }
    // the engine is already stopped: exit so a supervisor (systemd) can start the server again
    std::process::exit(1);
}

#[cfg(not(unix))]
fn exec_self(_config: &Path) {
    eprintln!("[talus-server] control: restart needs a Unix system; exiting");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    #[test]
    fn missing_weight_files() {
        let d = std::env::temp_dir().join(format!("talus-w4i-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let g = d.join("Q.gguf");
        std::fs::write(&g, b"x").unwrap();
        std::fs::write(
            d.join("qwen3.8-27b.json"),
            format!("{{\"args\": [\"--model\", \"{}\"]}}", g.display()),
        )
        .unwrap();
        let m: Vec<&str> = super::missing_files(&d, "qwen3.8-27b")
            .iter()
            .map(|x| x.0)
            .collect();
        assert_eq!(m, vec!["w4_prompt", "w4_fast"]);
        std::fs::write(d.join("Q.w4i"), b"x").unwrap();
        let m: Vec<&str> = super::missing_files(&d, "qwen3.8-27b")
            .iter()
            .map(|x| x.0)
            .collect();
        assert_eq!(m, vec!["w4_fast"]);
        let _ = std::fs::remove_dir_all(&d);
    }

    use super::*;

    #[test]
    fn exe_after_upgrade() {
        let d = std::env::temp_dir().join(format!("talus-exe-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let bin = d.join("talus-server");
        std::fs::write(&bin, b"x").unwrap();
        let deleted = PathBuf::from(format!("{} (deleted)", bin.display()));
        assert_eq!(exe_candidate(Some(bin.clone()), None), Some(bin.clone()));
        assert_eq!(exe_candidate(Some(deleted), None), Some(bin.clone()));
        let gone = d.join("nope");
        assert_eq!(
            exe_candidate(Some(gone), Some(bin.to_string_lossy().into_owned())),
            Some(bin.clone())
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn meminfo_parse() {
        let t = "MemTotal:       131072000 kB\nMemFree:  1 kB\nMemAvailable:   65536000 kB\n";
        assert_eq!(parse_meminfo(t), Some((65536000, 131072000)));
        assert_eq!(parse_meminfo("MemTotal: 5 kB\n"), None);
    }

    #[test]
    fn exec_args_keep_only_server_flags() {
        let orig: Vec<String> = [
            "--config",
            "old.json",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--api-key",
            "secret",
            "--control-dir",
            "/etc/talus",
            "--mock",
            "--max-context",
            "4096",
            "--no-mtp",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let got = exec_args(&orig, Path::new("/etc/talus/active.json"));
        assert_eq!(
            got,
            [
                "--port",
                "8000",
                "--control-dir",
                "/etc/talus",
                "--mock",
                "--config",
                "/etc/talus/active.json"
            ]
        );
        assert!(!got.iter().any(|a| a == "secret"));
    }
}
