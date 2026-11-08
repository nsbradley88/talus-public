//! HTTP/1.1 & SSE server implementing OpenAI, Anthropic, and llama.cpp endpoints
//! with full parity against `serve/server.py`.

use crate::config::ServerConfig;
use crate::engine::{
    Engine, EngineError, EngineTimings, GenerateRequest, SamplingParams, ScriptedMockEngine,
    StdioSubprocessEngine,
};
use crate::frontend::{
    anthropic_to_messages, forced_call, openai_to_messages, render_chat_prompt, tool_choice_of,
    unmark_think_literals, Event, EventKind, OutputParser,
};
use crate::json::{next_hex_id, unix_timestamp_secs, JsonValue};
use crate::prometheus::{
    render_llamacpp_prometheus, render_prometheus, wants_prometheus, LatencyHistograms,
    PrometheusSnapshot, CONTENT_TYPE as PROM_CONTENT_TYPE,
};
use crate::responses::{
    check_request as check_responses_request, collect_responses,
    input_messages as responses_input_messages, request_tools as responses_request_tools,
    responses_error_body, template_kwargs as responses_template_kwargs, Assembler,
};
use crate::tokenizer::{ByteTokenizer, Detokenizer, StopMatcher};

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;
/// Default of [`HttpLimits::max_body_in_flight`].
pub const MAX_BODY_IN_FLIGHT: usize = 128 * 1024 * 1024;
/// Bodies up to this size are never held back by [`HttpLimits::max_body_in_flight`].
pub const BODY_CHARGED_FROM: usize = 1024 * 1024;
const BODY_BUSY_RETRY_SECS: &str = "5";

#[derive(Clone, Debug, Default)]
pub struct MetricsState {
    pub requests_total: u64,
    pub prompt_tokens_total: u64,
    pub reused_tokens_total: u64,
    pub output_tokens_total: u64,
    pub prompt_ms_total: f64,
    pub decode_ms_total: f64,
    pub drafts_offered_total: u64,
    pub drafts_accepted_total: u64,
    pub reasoning_recoveries_total: u64,
    pub tool_calls_from_reasoning_total: u64,
    pub last_prompt_tokens: usize,
    pub last_generated_tokens: usize,
    pub last_timings: Option<EngineTimings>,
    pub histograms: LatencyHistograms,
    pub history: Vec<JsonValue>,
}

pub struct ServiceState {
    pub config: ServerConfig,
    pub engine: Arc<dyn Engine>,
    pub tokenizer: ByteTokenizer,
    pub chat_template: Option<Arc<talus_text::ChatTemplate>>,
    pub literal_tags: Vec<(String, String)>,
    pub metrics: Mutex<MetricsState>,
    pub shared_defaults: Mutex<JsonValue>,
    pub running_requests: AtomicUsize,
    /// Charged against [`HttpLimits::max_body_in_flight`].
    pub body_bytes_in_flight: AtomicUsize,
    pub waiting_requests: AtomicUsize,
    pub active_prompt_tokens: AtomicUsize,
    pub active_generated_tokens: AtomicUsize,
    pub started_instant: Instant,
    pub started_unix: i64,
    pub last_activity_ms: AtomicU64,
    /// talus-tui control endpoint state (`/v1/talus/*`).
    pub control: crate::control::ControlState,
    /// `build` in `/v1/talus/status` (bench extension E3).
    pub fingerprint: Arc<crate::fingerprint::Fingerprint>,
    /// `energy` in `/v1/talus/status` (bench extension E5).
    pub energy: Arc<crate::energy::EnergyMeter>,
    /// HTTP request limits and deadlines (set before the server starts).
    pub http_limits: HttpLimits,
    /// Connections being served now (capped at `http_limits.max_connections`).
    pub open_connections: AtomicUsize,
}

impl ServiceState {
    pub fn new(config: ServerConfig, engine: Arc<dyn Engine>, tokenizer: ByteTokenizer) -> Self {
        let literal_tags = talus_text::literal_tags(tokenizer.control_tokens());
        let control = crate::control::ControlState::new(config.control_dir.clone());
        let fingerprint = Arc::new(crate::fingerprint::Fingerprint::new(&config));
        let http_limits = HttpLimits {
            max_connections: config.max_connections.max(1),
            ..HttpLimits::default()
        };
        Self {
            config,
            engine,
            tokenizer,
            chat_template: None,
            literal_tags,
            metrics: Mutex::new(MetricsState::default()),
            shared_defaults: Mutex::new(JsonValue::empty_obj()),
            running_requests: AtomicUsize::new(0),
            body_bytes_in_flight: AtomicUsize::new(0),
            waiting_requests: AtomicUsize::new(0),
            active_prompt_tokens: AtomicUsize::new(0),
            active_generated_tokens: AtomicUsize::new(0),
            started_instant: Instant::now(),
            started_unix: unix_timestamp_secs(),
            last_activity_ms: AtomicU64::new(0),
            control,
            fingerprint,
            energy: Arc::new(crate::energy::EnergyMeter::new()),
            http_limits,
            open_connections: AtomicUsize::new(0),
        }
    }

    pub fn from_config(config: ServerConfig) -> Result<Self, String> {
        let real_engine = config.exe_path.is_some() && !config.mock_mode;
        let mut tokenizer = load_tokenizer(&config, real_engine)?;
        let chat_template = load_chat_template(&config, &tokenizer)?;

        let engine: Arc<dyn Engine> = match config.exe_path.as_ref().filter(|_| !config.mock_mode) {
            None => {
                let mock = ScriptedMockEngine::new(tokenizer.clone());
                mock.set_max_context(config.max_context);
                if !config.mock_scripts.is_empty() {
                    mock.set_scripts(config.mock_scripts.clone());
                }
                Arc::new(mock)
            }
            Some(exe) => {
                let mut args = config.engine_args.clone();
                let has_model_arg = args
                    .iter()
                    .any(|a| a == "--model" || a == "-m" || a == "--native" || a == "--pack");
                if !has_model_arg && !config.model_path.is_empty() {
                    args.insert(0, config.model_path.clone());
                    args.insert(0, "--model".to_string());
                }
                let sub = StdioSubprocessEngine::spawn_with_log(
                    exe,
                    &args,
                    &config.engine_env,
                    config.cwd.as_deref(),
                    config.log_path.as_deref(),
                    config.max_context,
                )
                .map_err(|e| format!("failed to start engine {}: {e}", exe.display()))?;
                if let Some(st) = sub.stop_token() {
                    if !tokenizer.stop_tokens.contains(&st) {
                        tokenizer.stop_tokens.push(st);
                    }
                }
                Arc::new(sub)
            }
        };
        if config.exe_path.is_some() && !config.mock_mode {
            warm_up(&config, engine.as_ref(), &tokenizer);
        }
        let mut st = Self::new(config, engine, tokenizer);
        st.chat_template = chat_template;
        Ok(st)
    }

    /// (running, queued) for reports: requests inside the engine vs. waiting for it. The engine runs one sequence at
    /// a time; the HTTP threads of the others wait on its lock. Engines that don't track it report every active request as running.
    pub fn queue_counts(&self) -> (usize, usize) {
        let active = self.running_requests.load(Ordering::Relaxed);
        match self.engine.in_flight() {
            Some(inside) => {
                let running = inside.min(active);
                (running, active - running)
            }
            None => (active, self.waiting_requests.load(Ordering::Relaxed)),
        }
    }

    /// The engine's crash/restart state as JSON (`/health`, `/metrics`, `/v1/talus/status`).
    pub fn engine_health_json(&self, h: &crate::engine::EngineHealth) -> JsonValue {
        JsonValue::obj(vec![
            ("crashes", JsonValue::from(h.crashes)),
            ("restarts", JsonValue::from(h.restarts)),
            ("restart_pending", JsonValue::from(h.restart_pending)),
            ("restarting", JsonValue::from(h.restarting)),
            ("down", JsonValue::from(h.down.is_some())),
            ("down_reason", JsonValue::from(h.down.clone())),
            ("last_crash", JsonValue::from(h.last_crash.clone())),
        ])
    }

    pub fn touch_activity(&self) {
        let elapsed_ms = self.started_instant.elapsed().as_millis() as u64;
        self.last_activity_ms.store(elapsed_ms, Ordering::Relaxed);
    }

    pub fn idle_seconds(&self) -> f64 {
        let now_ms = self.started_instant.elapsed().as_millis() as u64;
        let last_ms = self.last_activity_ms.load(Ordering::Relaxed);
        (now_ms.saturating_sub(last_ms) as f64) / 1000.0
    }

    pub fn reported_ctx(&self) -> usize {
        let eng_ctx = self.engine.max_context();
        if eng_ctx > 0 {
            eng_ctx
        } else {
            self.config.max_context
        }
    }

    pub fn model_for(&self, req: &JsonValue) -> String {
        if let Some(m) = req
            .get_str("model")
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            if self.config.matches_model(Some(m)) {
                return m.to_string();
            }
        }
        self.config.model_name.clone()
    }

    pub fn prometheus_snapshot(&self) -> PrometheusSnapshot {
        let (running, waiting) = self.queue_counts();
        let health = self.engine.health();
        let m = self.metrics.lock().unwrap();
        PrometheusSnapshot {
            model_name: self.config.model_name.clone(),
            loaded: self.engine.is_loaded(),
            max_context: self.reported_ctx(),
            running,
            waiting,
            engine_crashes: health.crashes,
            engine_restarts: health.restarts,
            engine_down: health.down.is_some(),
            active_prompt_tokens: self.active_prompt_tokens.load(Ordering::Relaxed),
            active_generated_tokens: self.active_generated_tokens.load(Ordering::Relaxed),
            requests_total: m.requests_total,
            prompt_tokens_total: m.prompt_tokens_total,
            reused_tokens_total: m.reused_tokens_total,
            output_tokens_total: m.output_tokens_total,
            prompt_ms_total: m.prompt_ms_total,
            decode_ms_total: m.decode_ms_total,
            drafts_offered_total: m.drafts_offered_total,
            drafts_accepted_total: m.drafts_accepted_total,
            reasoning_recoveries_total: m.reasoning_recoveries_total,
            tool_calls_from_reasoning_total: m.tool_calls_from_reasoning_total,
            histograms: m.histograms.clone(),
        }
    }

    pub fn metrics_json(&self, all_requests: bool) -> JsonValue {
        let (running, waiting) = self.queue_counts();
        let health = self.engine.health();
        let m = self.metrics.lock().unwrap();
        let state = if health.down.is_some() {
            "down"
        } else if health.restarting {
            "engine_restarting"
        } else if health.restart_pending {
            "crashed"
        } else if !self.engine.is_loaded() {
            "unloaded"
        } else if running > 0 {
            "generating"
        } else {
            "idle"
        };
        let reqs: Vec<JsonValue> = if all_requests {
            m.history.iter().rev().cloned().collect()
        } else {
            m.history.iter().rev().take(12).cloned().collect()
        };
        JsonValue::obj(vec![
            (
                "engine",
                JsonValue::obj(vec![
                    ("model", JsonValue::from(self.config.model_name.as_str())),
                    ("max_context", JsonValue::from(self.reported_ctx())),
                    ("images", JsonValue::from(false)),
                    ("health", self.engine_health_json(&health)),
                ]),
            ),
            (
                "live",
                JsonValue::obj(vec![
                    ("state", JsonValue::from(state)),
                    ("running", JsonValue::from(running)),
                    ("queued", JsonValue::from(waiting)),
                    ("waiting", JsonValue::from(waiting)),
                    (
                        "prompt_tokens",
                        if running > 0 {
                            JsonValue::from(self.active_prompt_tokens.load(Ordering::Relaxed))
                        } else {
                            JsonValue::Null
                        },
                    ),
                    (
                        "generated",
                        if running > 0 {
                            JsonValue::from(self.active_generated_tokens.load(Ordering::Relaxed))
                        } else {
                            JsonValue::Null
                        },
                    ),
                ]),
            ),
            (
                "totals",
                JsonValue::obj(vec![
                    ("requests", JsonValue::from(m.requests_total)),
                    ("prompt_tokens", JsonValue::from(m.prompt_tokens_total)),
                    ("reused", JsonValue::from(m.reused_tokens_total)),
                    ("output_tokens", JsonValue::from(m.output_tokens_total)),
                    ("prompt_ms", JsonValue::from(m.prompt_ms_total)),
                    ("decode_ms", JsonValue::from(m.decode_ms_total)),
                    ("drafts_offered", JsonValue::from(m.drafts_offered_total)),
                    ("drafts_accepted", JsonValue::from(m.drafts_accepted_total)),
                ]),
            ),
            (
                "prompt_tokens_total",
                JsonValue::from(m.prompt_tokens_total),
            ),
            (
                "tokens_predicted_total",
                JsonValue::from(m.output_tokens_total),
            ),
            ("requests", JsonValue::Array(reqs)),
            ("requests_kept", JsonValue::from(m.history.len())),
        ])
    }
}

/// The tokenizer the server encodes prompts with (a missing or unreadable tokenizer used to fall
/// back to byte tokens silently, so a real model got meaningless input while /health stayed green).
///
/// - an explicit `tokenizer` must exist and load, in every mode;
/// - otherwise the model path (a GGUF or a tokenizer directory) must load when a real engine runs;
/// - byte tokens only in mock mode, or with a real engine when the run config sets `"byte_tokenizer": true`
///   (protocol tests with a stub engine).
fn load_tokenizer(config: &ServerConfig, real_engine: bool) -> Result<ByteTokenizer, String> {
    let stops = &config.stop_tokens;
    if let Some(ref p) = config.tokenizer_path {
        if !p.exists() {
            return Err(format!("tokenizer {} does not exist", p.display()));
        }
        return ByteTokenizer::load(p, stops);
    }
    let mp = std::path::Path::new(&config.model_path);
    let model_has_tokenizer = !config.model_path.is_empty()
        && mp.exists()
        && (mp.is_dir() || mp.extension().and_then(|s| s.to_str()) == Some("gguf"));
    if model_has_tokenizer {
        match ByteTokenizer::load(mp, stops) {
            Ok(t) => return Ok(t),
            Err(e) if real_engine => {
                return Err(format!(
                "cannot load the tokenizer of model {}: {e} (set \"tokenizer\" in the run config)",
                mp.display()
            ))
            }
            Err(e) => eprintln!("[talus-server] mock mode: {e}; using byte tokens"),
        }
    }
    if real_engine && !config.byte_tokenizer {
        return Err(if config.model_path.is_empty() {
            "the engine needs the model's tokenizer: set \"tokenizer\" (a GGUF or tokenizer directory) in the run config"
                .to_string()
        } else {
            format!(
                "no tokenizer for model {}: set \"tokenizer\" (a GGUF or tokenizer directory) in the run config",
                config.model_path
            )
        });
    }
    Ok(ByteTokenizer::new().with_stop_tokens(stops))
}

/// An explicit `chat_template` must exist and compile; so must the template embedded in the tokenizer (both used to
/// be dropped silently, leaving the built-in ChatML renderer to format another model's prompts).
fn load_chat_template(
    config: &ServerConfig,
    tokenizer: &ByteTokenizer,
) -> Result<Option<Arc<talus_text::ChatTemplate>>, String> {
    if let Some(ref ct_path) = config.chat_template_path {
        if !ct_path.exists() {
            return Err(format!(
                "chat template {} does not exist",
                ct_path.display()
            ));
        }
        return talus_text::ChatTemplate::from_file(ct_path)
            .map(|t| Some(Arc::new(t)))
            .map_err(|e| format!("failed to load chat template {}: {e}", ct_path.display()));
    }
    let Some(src) = tokenizer
        .real_tokenizer()
        .and_then(|t| t.chat_template.as_deref())
    else {
        return Ok(None);
    };
    talus_text::ChatTemplate::from_str(src)
        .map(|t| Some(Arc::new(t)))
        .map_err(|e| {
            let from = config
                .tokenizer_path
                .as_ref()
                .map_or(config.model_path.clone(), |p| p.display().to_string());
            format!(
                "the chat template embedded in {from} does not compile: {e} (set \"chat_template\" to a template file)"
            )
        })
}

pub struct ServerHandle {
    addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    pub state: Arc<ServiceState>,
}

impl ServerHandle {
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.addr.port())
    }

    pub fn stop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_millis(100));
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// A paragraph long enough that the repeated prompt isn't a run of one token; its content doesn't matter.
const WARMUP_TEXT: &str = "fn merge_runs(rows: &mut Vec<(u32, u32)>) -> usize {\n    // Sort by start, then fold every overlapping pair into one run.\n    rows.sort_unstable();\n    let mut out = 0;\n    for i in 1..rows.len() {\n        if rows[i].0 <= rows[out].1 {\n            rows[out].1 = rows[out].1.max(rows[i].1);\n        } else {\n            out += 1;\n            rows[out] = rows[i];\n        }\n    }\n    out + 1\n}\n\nThe harbour office kept its ledgers in a cold back room, where the tide tables hung beside a map of the northern channel. Each morning the clerk copied the night's arrivals, checked the weights against the manifests, and sent a boy up the hill with whatever the customs house had asked for.\n\n";

/// One throwaway request before the port opens: the first request's lazy costs (kernel code, first-touch pages, first
/// verify window) land here instead of on a user. No prompt cache and no checkpoint, and it fits the K/V the engine
/// already has. A failure only costs the warm start: the first real request reports it.
fn warm_up(config: &ServerConfig, engine: &dyn Engine, tokenizer: &ByteTokenizer) {
    let n = std::env::var("TALUS_WARMUP_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .or(config.warmup_tokens)
        .unwrap_or_else(|| crate::config::default_warmup_tokens(&config.model_name));
    // A small max_context can't take the prompt plus the 16 generated tokens.
    let n = n.min(engine.max_context().saturating_sub(32));
    if n == 0 {
        return;
    }
    let mut ids = Vec::with_capacity(n + 64);
    while ids.len() < n {
        let piece = tokenizer.encode(WARMUP_TEXT, false);
        if piece.is_empty() {
            return;
        }
        ids.extend(piece);
    }
    ids.truncate(n);
    let req = GenerateRequest {
        prompt_ids: ids,
        max_new_tokens: 16,
        sampling: SamplingParams {
            temperature: 0.0,
            ignore_eos: true,
            cache_prompt: false,
            checkpoint: false,
            ..SamplingParams::default()
        },
        stop_tokens: Vec::new(),
        stop_strings: Vec::new(),
    };
    let t0 = Instant::now();
    let r = engine.generate(
        &req,
        Arc::new(AtomicBool::new(false)),
        &mut |_| true,
        &mut |_, _| {},
    );
    // MiMo resumes from its token history even after cache_prompt=false (seen: 61 tokens of the warm-up's text reused,
    // and the reply changed). The other two warm-up models keep nothing here (kv-prefix A/B: 0 cached, same reply).
    if config.model_name == "mimo-v2.6-flash" {
        if let Err(e) = engine.erase_slot(0) {
            eprintln!("[talus-server] warm-up: clear failed: {e}");
        }
    }
    match r {
        Ok(_) => eprintln!(
            "[talus-server] warm-up: {n} tokens in {:.0} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        ),
        Err(e) => eprintln!("[talus-server] warm-up failed: {e}"),
    }
}

pub struct TalusServer {
    listener: TcpListener,
    addr: SocketAddr,
    pub state: Arc<ServiceState>,
}

impl TalusServer {
    pub fn bind(state: Arc<ServiceState>) -> Result<Self, String> {
        if state.config.idle_unload_seconds > 0 && !state.engine.supports_lifecycle() {
            return Err(format!(
                "idle_unload_seconds = {}: this engine cannot unload (set it to 0)",
                state.config.idle_unload_seconds
            ));
        }
        let bind_addr = format!("{}:{}", state.config.host, state.config.port);
        let listener = TcpListener::bind(&bind_addr)
            .map_err(|e| format!("failed to bind {bind_addr}: {e}"))?;
        let addr = listener
            .local_addr()
            .map_err(|e| format!("failed to read local_addr: {e}"))?;
        Ok(Self {
            listener,
            addr,
            state,
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    pub fn start_background(self) -> ServerHandle {
        let addr = self.addr;
        let state = Arc::clone(&self.state);
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_bg = Arc::clone(&shutdown);
        let listener = self.listener;
        spawn_idle_unloader(Arc::clone(&state), Arc::clone(&shutdown));
        spawn_engine_supervisor(Arc::clone(&state), Arc::clone(&shutdown));

        let thread = thread::spawn(move || {
            run_accept_loop(listener, state, shutdown_bg);
        });

        ServerHandle {
            addr,
            shutdown,
            thread: Some(thread),
            state: self.state,
        }
    }

    pub fn run_blocking(self) {
        let shutdown = Arc::new(AtomicBool::new(false));
        spawn_idle_unloader(Arc::clone(&self.state), Arc::clone(&shutdown));
        spawn_engine_supervisor(Arc::clone(&self.state), Arc::clone(&shutdown));
        run_accept_loop(self.listener, self.state, shutdown);
    }
}

/// HTTP request limits and deadlines. Request metadata used to be read with unbounded
/// `read_line` (a 1 MiB header was accepted), a slow client could hold a connection forever, and the API key was
/// checked only after the body had been buffered.
#[derive(Clone, Debug)]
pub struct HttpLimits {
    /// Request line, terminator included (longer: 414).
    pub max_request_line: usize,
    /// One header or trailer line (longer: 431).
    pub max_header_line: usize,
    /// All header and trailer lines together (more: 431).
    pub max_header_bytes: usize,
    /// Header and trailer lines, repeated names counted (more: 431).
    pub max_header_count: usize,
    /// Request body, after de-chunking (more: 413).
    pub max_body: usize,
    /// Bytes of bodies over [`BODY_CHARGED_FROM`] held at once, until their requests end (more: 503). What a request
    /// costs in memory follows its body, so this bounds large requests together.
    pub max_body_in_flight: usize,
    /// Simultaneous connections; one more gets a 503 at accept, without a thread.
    pub max_connections: usize,
    /// Longest wait for any one read.
    pub read_idle: Duration,
    /// From accept to the end of the headers (408 after).
    pub header_deadline: Duration,
    /// From accept to the end of the body (408 after).
    pub request_deadline: Duration,
    /// Longest stall of any one write (streamed replies included: a client that stops reading is dropped and its
    /// generation cancelled instead of holding the engine).
    pub write_stall: Duration,
    /// A whole non-streamed reply.
    pub response_deadline: Duration,
}

impl Default for HttpLimits {
    fn default() -> Self {
        Self {
            max_request_line: 16 * 1024,
            max_header_line: 16 * 1024,
            max_header_bytes: 64 * 1024,
            max_header_count: 100,
            max_body: MAX_REQUEST_BYTES,
            max_body_in_flight: MAX_BODY_IN_FLIGHT,
            max_connections: crate::config::DEFAULT_MAX_CONNECTIONS,
            read_idle: Duration::from_secs(30),
            header_deadline: Duration::from_secs(30),
            request_deadline: Duration::from_secs(300),
            write_stall: Duration::from_secs(60),
            response_deadline: Duration::from_secs(120),
        }
    }
}

/// Decrements the open-connection count when a connection thread ends (or never starts).
struct ConnSlot(Arc<ServiceState>);

impl Drop for ConnSlot {
    fn drop(&mut self) {
        self.0.open_connections.fetch_sub(1, Ordering::SeqCst);
    }
}

fn run_accept_loop(listener: TcpListener, state: Arc<ServiceState>, shutdown: Arc<AtomicBool>) {
    for incoming in listener.incoming() {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        match incoming {
            Ok(stream) => {
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                let max = state.http_limits.max_connections.max(1);
                if state.open_connections.fetch_add(1, Ordering::SeqCst) >= max {
                    state.open_connections.fetch_sub(1, Ordering::SeqCst);
                    refuse_connection(stream, max, &state.http_limits);
                    continue;
                }
                let slot = ConnSlot(Arc::clone(&state));
                let st = Arc::clone(&state);
                // a failed spawn (thread limit) drops the closure, its stream and its slot; it used to panic the
                // accept loop and stop the server
                let _ = thread::Builder::new()
                    .name("talus-http".to_string())
                    .spawn(move || {
                        let _slot = slot;
                        let _ = handle_connection(stream, st);
                    });
            }
            Err(_) => {
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                // EMFILE and friends: don't spin
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

/// Over the connection cap: a short 503 from the accept thread (bounded write, no read), then close, or, for a few at
/// a time, discard what the client is sending first so it can read the reply.
fn refuse_connection(mut stream: TcpStream, max: usize, lim: &HttpLimits) {
    let _ = stream.set_write_timeout(Some(Duration::from_millis(200)));
    let body = format!(
        "{{\"error\":{{\"type\":\"server_busy\",\"message\":\"too many open connections (max {max})\"}}}}"
    );
    let head = format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\nRetry-After: 1\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.shutdown(std::net::Shutdown::Write);

    static DISCARDING: AtomicUsize = AtomicUsize::new(0);
    struct Discarding;
    impl Drop for Discarding {
        fn drop(&mut self) {
            DISCARDING.fetch_sub(1, Ordering::SeqCst);
        }
    }
    if DISCARDING.fetch_add(1, Ordering::SeqCst) >= MAX_REFUSED_DISCARDING {
        DISCARDING.fetch_sub(1, Ordering::SeqCst);
        return;
    }
    let held = Discarding;
    let limit = discard_limit(None, lim);
    let _ = thread::Builder::new()
        .name("talus-refused".to_string())
        .spawn(move || {
            let _held = held;
            let mut reader = DeadlineReader {
                stream,
                deadline: Instant::now() + DISCARD_TIME,
                idle: DISCARD_IDLE,
            };
            discard_incoming(&mut reader, limit);
        });
}

struct HttpRequest {
    method: String,
    path: String,
    query: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl HttpRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(|s| s.as_str())
    }

    fn query_param(&self, key: &str) -> Option<String> {
        for pair in self.query.split('&') {
            if pair.is_empty() {
                continue;
            }
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            if k == key {
                return Some(url_decode(v));
            }
        }
        None
    }

    /// How the body is framed: None = no body, Some(None) = chunked, Some(Some(n)) = Content-Length n.
    fn body_framing(&self) -> Option<Option<usize>> {
        if self.headers.contains_key("transfer-encoding") {
            Some(None)
        } else {
            self.headers
                .get("content-length")
                .map(|n| Some(n.parse().unwrap_or(usize::MAX)))
        }
    }
}

fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
        } else if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (
                (bytes[i + 1] as char).to_digit(16),
                (bytes[i + 2] as char).to_digit(16),
            ) {
                out.push(((h << 4) | l) as u8);
                i += 3;
                continue;
            }
            out.push(bytes[i]);
            i += 1;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// One request's share of [`HttpLimits::max_body_in_flight`]; returned when the request ends, however it ends.
struct BodyCharge<'a> {
    in_flight: &'a AtomicUsize,
    limit: usize,
    held: usize,
}

impl<'a> BodyCharge<'a> {
    fn new(in_flight: &'a AtomicUsize, limit: usize) -> Self {
        Self {
            in_flight,
            limit,
            held: 0,
        }
    }

    /// Make room for a body of `total` bytes. False: no room now, and nothing more is held.
    fn reserve(&mut self, total: usize) -> bool {
        if total <= BODY_CHARGED_FROM || total <= self.held {
            return true;
        }
        let more = total - self.held;
        let got = self
            .in_flight
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |now| {
                now.checked_add(more).filter(|n| *n <= self.limit)
            })
            .is_ok();
        if got {
            self.held = total;
        }
        got
    }
}

impl Drop for BodyCharge<'_> {
    fn drop(&mut self) {
        self.in_flight.fetch_sub(self.held, Ordering::SeqCst);
    }
}

enum ReadRequestError {
    /// No room for this body right now ([`HttpLimits::max_body_in_flight`]).
    Busy,
    /// Closed (or silent until a deadline) before the first byte: close without a reply.
    Eof,
    BadRequest(u16, String),
}

fn bad(code: u16, msg: impl Into<String>) -> ReadRequestError {
    ReadRequestError::BadRequest(code, msg.into())
}

/// Reads from the client with a per-read idle timeout and an absolute deadline (the socket's read timeout is set to
/// whichever is sooner before every read).
struct DeadlineReader {
    stream: TcpStream,
    deadline: Instant,
    idle: Duration,
}

impl Read for DeadlineReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self.deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "request deadline passed",
            ));
        }
        self.stream.set_read_timeout(Some(left.min(self.idle)))?;
        self.stream.read(buf)
    }
}

type ClientReader = BufReader<DeadlineReader>;

fn is_timeout(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
    )
}

enum LineError {
    TooLong,
    Io(std::io::Error),
}

/// One line through `\n`, at most `max` bytes with the terminator; never buffers more than `max`. An empty result
/// is EOF before the first byte; a result without `\n` is EOF mid-line.
fn read_line_bounded<R: BufRead>(r: &mut R, max: usize) -> Result<Vec<u8>, LineError> {
    let mut line = Vec::new();
    loop {
        let buf = match r.fill_buf() {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(LineError::Io(e)),
        };
        if buf.is_empty() {
            return Ok(line);
        }
        let (take, done) = match buf.iter().position(|&b| b == b'\n') {
            Some(i) => (i + 1, true),
            None => (buf.len(), false),
        };
        if line.len() + take > max {
            return Err(LineError::TooLong);
        }
        line.extend_from_slice(&buf[..take]);
        r.consume(take);
        if done {
            return Ok(line);
        }
    }
}

fn strip_eol(line: &[u8]) -> &[u8] {
    let line = line.strip_suffix(b"\n").unwrap_or(line);
    line.strip_suffix(b"\r").unwrap_or(line)
}

/// Header and trailer lines share one byte and line budget.
struct HeaderBudget {
    bytes: usize,
    lines: usize,
}

impl HeaderBudget {
    fn take(&mut self, n: usize) -> Result<(), ReadRequestError> {
        self.lines = self
            .lines
            .checked_sub(1)
            .ok_or_else(|| bad(431, "too many HTTP header lines"))?;
        self.bytes = self
            .bytes
            .checked_sub(n)
            .ok_or_else(|| bad(431, "HTTP headers too large"))?;
        Ok(())
    }
}

/// One header (or trailer) line, `None` at the blank line that ends the block.
fn read_header_line(
    reader: &mut ClientReader,
    lim: &HttpLimits,
    budget: &mut HeaderBudget,
    what: &str,
) -> Result<Option<String>, ReadRequestError> {
    let line = match read_line_bounded(reader, lim.max_header_line) {
        Ok(l) => l,
        Err(LineError::TooLong) => return Err(bad(431, format!("HTTP {what} line too long"))),
        Err(LineError::Io(e)) if is_timeout(&e) => {
            return Err(bad(408, format!("timed out reading the {what}s")))
        }
        Err(LineError::Io(e)) => return Err(bad(400, format!("{what} read error: {e}"))),
    };
    if !line.ends_with(b"\n") {
        return Err(bad(400, format!("unexpected EOF in {what}s")));
    }
    let l = strip_eol(&line);
    if l.is_empty() {
        return Ok(None);
    }
    budget.take(line.len())?;
    let l = std::str::from_utf8(l).map_err(|_| bad(400, format!("{what} is not valid UTF-8")))?;
    Ok(Some(l.to_string()))
}

/// Request line and headers, bounded by `lim` and the header deadline. The body is read later
/// ([`read_http_body`]), after the API key has been checked.
fn read_http_head(
    reader: &mut ClientReader,
    lim: &HttpLimits,
) -> Result<(HttpRequest, HeaderBudget), ReadRequestError> {
    let line = match read_line_bounded(reader, lim.max_request_line) {
        Ok(l) if l.is_empty() => return Err(ReadRequestError::Eof),
        Ok(l) => l,
        Err(LineError::TooLong) => return Err(bad(414, "request line too long")),
        Err(LineError::Io(_)) => return Err(ReadRequestError::Eof),
    };
    if !line.ends_with(b"\n") {
        return Err(ReadRequestError::Eof);
    }
    let trimmed = std::str::from_utf8(strip_eol(&line))
        .map_err(|_| bad(400, "request line is not valid UTF-8"))?;
    if trimmed.is_empty() {
        return Err(ReadRequestError::Eof);
    }
    let mut parts = trimmed.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| bad(400, "malformed request line"))?
        .to_uppercase();
    let raw_target = parts
        .next()
        .ok_or_else(|| bad(400, "missing request target"))?
        .to_string();

    let mut budget = HeaderBudget {
        bytes: lim.max_header_bytes,
        lines: lim.max_header_count,
    };
    let mut headers: HashMap<String, String> = HashMap::new();
    while let Some(l) = read_header_line(reader, lim, &mut budget, "header")? {
        let Some((k, v)) = l.split_once(':') else {
            return Err(bad(400, "malformed header line"));
        };
        let name = k.trim().to_ascii_lowercase();
        let value = v.trim().to_string();
        // conflicting framing is a request-smuggling shape: refuse it
        if name == "content-length" && headers.get(&name).is_some_and(|old| *old != value) {
            return Err(bad(400, "conflicting Content-Length headers"));
        }
        headers.insert(name, value);
    }

    if let Some(te) = headers.get("transfer-encoding") {
        if !te.trim().eq_ignore_ascii_case("chunked") {
            return Err(bad(
                400,
                format!("unsupported Transfer-Encoding {te:?} (only chunked)"),
            ));
        }
        if headers.contains_key("content-length") {
            return Err(bad(400, "both Transfer-Encoding and Content-Length"));
        }
    } else if let Some(cl) = headers.get("content-length") {
        if cl.is_empty() || !cl.bytes().all(|b| b.is_ascii_digit()) {
            return Err(bad(400, "invalid Content-Length"));
        }
    }

    let (raw_path, query) = match raw_target.split_once('?') {
        Some((p, q)) => (p, q.to_string()),
        None => (raw_target.as_str(), String::new()),
    };
    let clean_path = if raw_path.len() > 1 {
        raw_path.trim_end_matches('/').to_string()
    } else {
        raw_path.to_string()
    };

    Ok((
        HttpRequest {
            method,
            path: clean_path,
            query,
            headers,
            body: Vec::new(),
        },
        budget,
    ))
}

/// Bytes of a body read and charged at a time.
const BODY_READ_STEP: usize = 256 * 1024;

fn body_read_error(e: std::io::Error, what: &str) -> ReadRequestError {
    if is_timeout(&e) {
        bad(408, "timed out reading the request body")
    } else {
        bad(400, what.to_string())
    }
}

/// Append `n` bytes of body to `out`. A body is charged as it arrives, not as announced: a client holds a share of
/// [`HttpLimits::max_body_in_flight`], and this memory, only for what it has really sent, plus one step.
fn read_charged<R: Read>(
    reader: &mut R,
    out: &mut Vec<u8>,
    n: usize,
    charge: &mut BodyCharge,
    what: &str,
) -> Result<(), ReadRequestError> {
    let end = out.len() + n;
    while out.len() < end {
        let next = end.min(out.len() + BODY_READ_STEP);
        if !charge.reserve(next) {
            return Err(ReadRequestError::Busy);
        }
        let start = out.len();
        out.resize(next, 0);
        reader
            .read_exact(&mut out[start..])
            .map_err(|e| body_read_error(e, what))?;
    }
    Ok(())
}

/// The body announced by `req`'s headers (Content-Length or chunked), at most `lim.max_body` bytes.
fn read_http_body(
    reader: &mut ClientReader,
    req: &HttpRequest,
    lim: &HttpLimits,
    budget: &mut HeaderBudget,
    charge: &mut BodyCharge,
) -> Result<Vec<u8>, ReadRequestError> {
    match req.body_framing() {
        None => Ok(Vec::new()),
        Some(None) => read_chunked_body(reader, lim, budget, charge),
        Some(Some(cl)) => {
            if cl > lim.max_body {
                return Err(bad(
                    413,
                    format!("request body exceeds maximum of {} bytes", lim.max_body),
                ));
            }
            let mut buf = Vec::with_capacity(cl);
            read_charged(reader, &mut buf, cl, charge, "incomplete request body")?;
            Ok(buf)
        }
    }
}

/// Longest chunk-size line (hex size plus extensions).
const MAX_CHUNK_LINE: usize = 1024;

fn read_chunked_body(
    reader: &mut ClientReader,
    lim: &HttpLimits,
    budget: &mut HeaderBudget,
    charge: &mut BodyCharge,
) -> Result<Vec<u8>, ReadRequestError> {
    let mut out = Vec::new();
    loop {
        let size_line = match read_line_bounded(reader, MAX_CHUNK_LINE) {
            Ok(l) => l,
            Err(LineError::TooLong) => return Err(bad(400, "chunk header too long")),
            Err(LineError::Io(e)) => return Err(body_read_error(e, "invalid chunked body")),
        };
        if !size_line.ends_with(b"\r\n") {
            return Err(bad(400, "malformed chunk header"));
        }
        let size_line = std::str::from_utf8(strip_eol(&size_line))
            .map_err(|_| bad(400, "malformed chunk header"))?;
        let hex_part = size_line.split(';').next().unwrap_or("").trim();
        if hex_part.is_empty() || !hex_part.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(bad(400, "invalid chunk size"));
        }
        let chunk_size =
            usize::from_str_radix(hex_part, 16).map_err(|_| bad(413, "chunk size overflow"))?;
        if chunk_size == 0 {
            // trailers until the blank line; they share the header budget
            while read_header_line(reader, lim, budget, "trailer")?.is_some() {}
            break;
        }
        if out.len().saturating_add(chunk_size) > lim.max_body {
            return Err(bad(413, "chunked request body too large"));
        }
        read_charged(reader, &mut out, chunk_size, charge, "truncated chunk data")?;
        let mut crlf = [0u8; 2];
        reader
            .read_exact(&mut crlf)
            .map_err(|e| body_read_error(e, "missing CRLF after chunk"))?;
        if &crlf != b"\r\n" {
            return Err(bad(400, "expected CRLF after chunk"));
        }
    }
    Ok(out)
}

fn status_reason(code: u16) -> &'static str {
    match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        413 => "Payload Too Large",
        408 => "Request Timeout",
        414 => "URI Too Long",
        415 => "Unsupported Media Type",
        431 => "Request Header Fields Too Large",
        501 => "Not Implemented",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

fn write_response(
    stream: &mut TcpStream,
    code: u16,
    content_type: &str,
    body: &[u8],
    extra_headers: &[(&str, &str)],
) -> std::io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {code} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n",
        status_reason(code),
        body.len()
    );
    for (k, v) in extra_headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    let deadline = Instant::now() + RESPONSE_WRITE_DEADLINE;
    write_all_deadline(stream, head.as_bytes(), deadline)?;
    write_all_deadline(stream, body, deadline)?;
    stream.flush()
}

/// A whole non-streamed reply must be written within this (each write also within the socket's stall timeout).
const RESPONSE_WRITE_DEADLINE: Duration = Duration::from_secs(120);

/// `write_all` that gives up at `deadline`: a client reading a large reply a byte at a time cannot hold the thread.
fn write_all_deadline(
    stream: &mut TcpStream,
    mut buf: &[u8],
    deadline: Instant,
) -> std::io::Result<()> {
    let stall = stream
        .write_timeout()
        .ok()
        .flatten()
        .unwrap_or(Duration::from_secs(60));
    let res = loop {
        if buf.is_empty() {
            break Ok(());
        }
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "response write deadline passed",
            ));
        }
        stream.set_write_timeout(Some(left.min(stall)))?;
        match stream.write(&buf[..buf.len().min(64 * 1024)]) {
            Ok(0) => break Err(std::io::ErrorKind::WriteZero.into()),
            Ok(n) => buf = &buf[n..],
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => break Err(e),
        }
    };
    let _ = stream.set_write_timeout(Some(stall));
    res
}

fn write_json(
    stream: &mut TcpStream,
    code: u16,
    val: &JsonValue,
    extra_headers: &[(&str, &str)],
) -> std::io::Result<()> {
    let s = val.to_json_string();
    write_response(
        stream,
        code,
        "application/json",
        s.as_bytes(),
        extra_headers,
    )
}

fn write_sse_headers(stream: &mut TcpStream) -> std::io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\nConnection: close\r\n\r\n";
    stream.write_all(head.as_bytes())?;
    stream.flush()
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn check_auth(req: &HttpRequest, expected_key: Option<&str>) -> bool {
    let Some(expected) = expected_key.filter(|k| !k.is_empty()) else {
        return true;
    };
    let given = if let Some(auth) = req.header("authorization") {
        // compare the scheme as bytes: after an ASCII "bearer " byte 7 is a char boundary
        match auth.as_bytes().get(..7) {
            Some(scheme) if scheme.eq_ignore_ascii_case(b"bearer ") => {
                auth.get(7..).unwrap_or("").trim()
            }
            _ => "",
        }
    } else if let Some(xkey) = req.header("x-api-key") {
        xkey.trim()
    } else {
        ""
    };
    if given.is_empty() {
        return false;
    }
    constant_time_eq(given.as_bytes(), expected.as_bytes())
}

fn is_local_origin(origin: &str, host_header: Option<&str>, configured_host: &str) -> bool {
    let lower = origin.to_ascii_lowercase();
    let Some(rest) = lower
        .strip_prefix("http://")
        .or_else(|| lower.strip_prefix("https://"))
    else {
        return false;
    };
    let host_port = rest.split('/').next().unwrap_or("");
    let host = if host_port.starts_with('[') {
        host_port
            .split(']')
            .next()
            .map(|s| format!("{s}]"))
            .unwrap_or_default()
    } else {
        host_port.split(':').next().unwrap_or("").to_string()
    };
    if matches!(host.as_str(), "localhost" | "127.0.0.1" | "[::1]") {
        return true;
    }
    if host.eq_ignore_ascii_case(configured_host) {
        return true;
    }
    if let Some(h) = host_header {
        let h_only = h.split(':').next().unwrap_or("");
        if host.eq_ignore_ascii_case(h_only) {
            return true;
        }
    }
    false
}

fn is_allowed_host(host_header: Option<&str>, configured_host: &str) -> bool {
    let Some(raw) = host_header.map(str::trim).filter(|s| !s.is_empty()) else {
        return true;
    };
    let lower = raw.to_ascii_lowercase();
    let name = if let Some(stripped) = lower.strip_prefix('[') {
        let Some((inside, rest)) = stripped.split_once(']') else {
            return false;
        };
        if !rest.is_empty()
            && !rest
                .strip_prefix(':')
                .is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
        {
            return false;
        }
        inside
    } else if let Some((h, port)) = lower.split_once(':') {
        if port.contains(':') {
            lower.as_str()
        } else if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) {
            h
        } else {
            return false;
        }
    } else {
        lower.as_str()
    };
    let name = name.trim_end_matches('.');
    if name.is_empty() {
        return false;
    }
    if name == "localhost"
        || name.ends_with(".localhost")
        || name == "127.0.0.1"
        || name == "::1"
        || name.eq_ignore_ascii_case(configured_host)
        || name.parse::<std::net::IpAddr>().is_ok()
    {
        return true;
    }
    false
}

fn handle_connection(mut stream: TcpStream, state: Arc<ServiceState>) -> std::io::Result<()> {
    let lim = state.http_limits.clone();
    let accepted = Instant::now();
    let _ = stream.set_nodelay(true);
    let _ = stream.set_write_timeout(Some(lim.write_stall));
    let mut reader = BufReader::new(DeadlineReader {
        stream: stream.try_clone()?,
        deadline: accepted + lim.header_deadline,
        idle: lim.read_idle,
    });
    let (mut req, mut budget) = match read_http_head(&mut reader, &lim) {
        Ok(r) => r,
        Err(ReadRequestError::Eof) => return Ok(()),
        Err(e) => {
            // the client may still be sending (an oversized header): reply, then close without a reset
            let res = reply_read_error(&mut stream, e);
            close_unread(&stream, &mut reader, discard_limit(None, &lim));
            return res;
        }
    };

    // Host, CORS preflight, health and the API key are settled on the headers alone: an unauthenticated client
    // never gets its body buffered.
    if let Some(res) = early_reply(&mut stream, &req, &state) {
        if let Some(framing) = req.body_framing() {
            close_unread(&stream, &mut reader, discard_limit(framing, &lim));
        }
        return res;
    }

    let mut charge = BodyCharge::new(&state.body_bytes_in_flight, lim.max_body_in_flight);

    reader.get_mut().deadline = accepted + lim.request_deadline;
    // a client waiting for 100-continue (curl does above 1 MiB) sends the body now instead of after its own timeout;
    // an oversized Content-Length is refused below without it
    let wants_body = match req.body_framing() {
        Some(None) => true,
        Some(Some(n)) => n > 0 && n <= lim.max_body,
        None => false,
    };
    if wants_body
        && req
            .header("expect")
            .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
    {
        write_all_deadline(
            &mut stream,
            b"HTTP/1.1 100 Continue\r\n\r\n",
            Instant::now() + lim.write_stall,
        )?;
    }
    match read_http_body(&mut reader, &req, &lim, &mut budget, &mut charge) {
        Ok(body) => req.body = body,
        Err(e) => {
            let res = reply_read_error(&mut stream, e);
            // what was read is gone: it no longer counts while the rest is discarded
            drop(charge);
            close_unread(&stream, &mut reader, discard_limit(None, &lim));
            return res;
        }
    }
    drop(reader);

    match req.method.as_str() {
        "GET" => handle_get(&mut stream, &req, &state),
        "POST" => handle_post(&mut stream, &req, &state),
        "DELETE" => handle_delete(&mut stream, &req),
        _ => {
            let err = JsonValue::obj(vec![(
                "error",
                JsonValue::obj(vec![
                    ("type", JsonValue::from("invalid_request_error")),
                    ("message", JsonValue::from("method not allowed")),
                ]),
            )]);
            write_json(&mut stream, 405, &err, &[])
        }
    }
}

fn reply_read_error(stream: &mut TcpStream, e: ReadRequestError) -> std::io::Result<()> {
    if matches!(e, ReadRequestError::Busy) {
        let err = JsonValue::obj(vec![(
            "error",
            JsonValue::obj(vec![
                ("type", JsonValue::from("server_busy")),
                (
                    "message",
                    JsonValue::from(
                        "too many large requests in progress: retry shortly, or send a smaller request",
                    ),
                ),
            ]),
        )]);
        return write_json(stream, 503, &err, &[("Retry-After", BODY_BUSY_RETRY_SECS)]);
    }
    let ReadRequestError::BadRequest(code, msg) = e else {
        return Ok(());
    };
    let err = JsonValue::obj(vec![(
        "error",
        JsonValue::obj(vec![
            ("type", JsonValue::from("invalid_request_error")),
            ("message", JsonValue::from(msg)),
        ]),
    )]);
    write_json(stream, code, &err, &[])
}

/// How long, in all, a client that got its reply early may keep sending before the connection closes.
const DISCARD_TIME: Duration = Duration::from_secs(5);
/// How long it may stay silent meanwhile.
const DISCARD_IDLE: Duration = Duration::from_secs(1);
/// Connections refused at accept that may be discarding at once; the ones past this close right away.
const MAX_REFUSED_DISCARDING: usize = 8;

/// Bytes a client may still send after a reply: what it announced (`body`, when known), at most twice the largest body.
fn discard_limit(body: Option<usize>, lim: &HttpLimits) -> usize {
    let cap = lim.max_body.saturating_mul(2);
    body.map_or(cap, |n| n.min(cap))
}

/// Read and drop what the client is still sending, within [`DISCARD_TIME`] and `limit` bytes. A client blocked writing
/// its body only reads the reply once the body is gone; closing on unread data would reset the connection instead.
/// Nothing is kept: a small buffer, overwritten.
fn discard_incoming<R: Read>(reader: &mut R, limit: usize) {
    let mut sink = [0u8; 16 * 1024];
    let mut left = limit;
    while left > 0 {
        let want = left.min(sink.len());
        match reader.read(&mut sink[..want]) {
            Ok(0) | Err(_) => break,
            Ok(n) => left -= n,
        }
    }
}

/// After a reply sent without reading the request body: finish our side, then discard what the client is still
/// sending, so it reads the reply instead of a connection reset.
fn close_unread(stream: &TcpStream, reader: &mut ClientReader, limit: usize) {
    let _ = stream.shutdown(std::net::Shutdown::Write);
    let r = reader.get_mut();
    r.deadline = Instant::now() + DISCARD_TIME;
    r.idle = DISCARD_IDLE;
    discard_incoming(reader, limit);
}

/// Replies decided by the request head: Host check, CORS preflight, health, API key. None = go on to the body.
fn early_reply(
    stream: &mut TcpStream,
    req: &HttpRequest,
    state: &Arc<ServiceState>,
) -> Option<std::io::Result<()>> {
    if state.config.api_key.is_none() && !is_allowed_host(req.header("host"), &state.config.host) {
        let err = JsonValue::obj(vec![(
            "error",
            JsonValue::obj(vec![
                ("type", JsonValue::from("forbidden")),
                (
                    "message",
                    JsonValue::from("disallowed Host header (DNS rebinding protection)"),
                ),
            ]),
        )]);
        return Some(write_json(stream, 403, &err, &[]));
    }

    if req.method == "OPTIONS" {
        return Some(write_response(
            stream,
            204,
            "text/plain",
            b"",
            &[
                ("Access-Control-Allow-Origin", "*"),
                (
                    "Access-Control-Allow-Methods",
                    "GET, POST, PUT, DELETE, OPTIONS",
                ),
                (
                    "Access-Control-Allow-Headers",
                    "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta",
                ),
            ],
        ));
    }

    // /health and /api/health are exempt from API key authentication
    if req.method == "GET" && (req.path == "/health" || req.path == "/api/health") {
        let health = state.engine.health();
        let loaded = state.engine.is_loaded();
        let busy = state.running_requests.load(Ordering::Relaxed) > 0;
        // down (crash-loop cap) > restarting > crashed (about to restart) > unloaded
        let status = if health.down.is_some() {
            "down"
        } else if health.restarting {
            "restarting"
        } else if health.restart_pending {
            "crashed"
        } else if loaded {
            "ok"
        } else {
            "unloaded"
        };
        let body = JsonValue::obj(vec![
            ("status", JsonValue::from(status)),
            ("reason", JsonValue::from(health.down.clone())),
            ("engine", state.engine_health_json(&health)),
            ("max_context", JsonValue::from(state.reported_ctx())),
            ("context", JsonValue::from(state.reported_ctx())),
            ("model", JsonValue::from(state.config.model_name.as_str())),
            ("images", JsonValue::from(false)),
            ("api_key", JsonValue::from(state.config.api_key.is_some())),
            ("image_generation", JsonValue::from(false)),
            ("loaded", JsonValue::from(loaded)),
            ("busy", JsonValue::from(busy)),
            ("idle_seconds", JsonValue::from(state.idle_seconds())),
            (
                "idle_unload_seconds",
                JsonValue::from(state.config.idle_unload_seconds),
            ),
            ("service", JsonValue::from("strata")),
        ]);
        return Some(write_json(
            stream,
            if status == "ok" { 200 } else { 503 },
            &body,
            &[],
        ));
    }

    // Enforce API key on all other endpoints
    if !check_auth(req, state.config.api_key.as_deref()) {
        let is_anthropic = req.path.starts_with("/v1/messages");
        let err = if is_anthropic {
            JsonValue::obj(vec![
                ("type", JsonValue::from("error")),
                (
                    "error",
                    JsonValue::obj(vec![
                        ("type", JsonValue::from("authentication_error")),
                        ("message", JsonValue::from("missing or wrong API key")),
                    ]),
                ),
            ])
        } else {
            JsonValue::obj(vec![(
                "error",
                JsonValue::obj(vec![
                    ("message", JsonValue::from("missing or wrong API key")),
                    ("type", JsonValue::from("authentication_error")),
                    ("param", JsonValue::Null),
                    ("code", JsonValue::from("invalid_api_key")),
                ]),
            )])
        };
        return Some(write_json(
            stream,
            401,
            &err,
            &[("WWW-Authenticate", "Bearer")],
        ));
    }

    None
}

fn handle_delete(stream: &mut TcpStream, req: &HttpRequest) -> std::io::Result<()> {
    if req.path.starts_with("/v1/responses/") {
        let body = responses_error_body(
            "this server keeps no responses (stateless); send the whole conversation to POST /v1/responses",
            "invalid_request_error",
            Some("response_id"),
            Some("not_found"),
        );
        return write_json(stream, 404, &body, &[]);
    }
    let err = JsonValue::obj(vec![(
        "error",
        JsonValue::obj(vec![("message", JsonValue::from("not found"))]),
    )]);
    write_json(stream, 404, &err, &[])
}

fn handle_get(
    stream: &mut TcpStream,
    req: &HttpRequest,
    state: &Arc<ServiceState>,
) -> std::io::Result<()> {
    match req.path.as_str() {
        "/v1/models" | "/models" => {
            let loaded = state.engine.is_loaded();
            let ctx = state.reported_ctx();
            let mut data = Vec::new();
            let mut primary = JsonValue::obj(vec![
                ("id", JsonValue::from(state.config.model_name.as_str())),
                ("object", JsonValue::from("model")),
                ("created", JsonValue::from(state.started_unix)),
                ("owned_by", JsonValue::from("local")),
                (
                    "status",
                    JsonValue::obj(vec![(
                        "value",
                        JsonValue::from(if loaded { "loaded" } else { "unloaded" }),
                    )]),
                ),
                (
                    "meta",
                    JsonValue::obj(vec![
                        ("n_ctx", JsonValue::from(ctx)),
                        ("n_ctx_train", JsonValue::from(ctx)),
                    ]),
                ),
                (
                    "architecture",
                    JsonValue::obj(vec![
                        (
                            "input_modalities",
                            JsonValue::Array(vec![JsonValue::from("text")]),
                        ),
                        (
                            "output_modalities",
                            JsonValue::Array(vec![JsonValue::from("text")]),
                        ),
                    ]),
                ),
            ]);
            if !state.config.model_aliases.is_empty() {
                primary.insert(
                    "aliases",
                    JsonValue::Array(
                        state
                            .config
                            .model_aliases
                            .iter()
                            .map(|a| JsonValue::from(a.as_str()))
                            .collect(),
                    ),
                );
            }
            data.push(primary.clone());
            for alias in &state.config.model_aliases {
                let mut entry = primary.clone();
                entry.insert("id", JsonValue::from(alias.as_str()));
                entry.insert(
                    "alias_of",
                    JsonValue::from(state.config.model_name.as_str()),
                );
                data.push(entry);
            }
            let body = JsonValue::obj(vec![
                ("object", JsonValue::from("list")),
                ("data", JsonValue::Array(data)),
            ]);
            write_json(stream, 200, &body, &[])
        }
        "/props" => {
            if let Some(model_q) = req.query_param("model") {
                if !state.config.matches_model(Some(&model_q)) {
                    let err = JsonValue::obj(vec![(
                        "error",
                        JsonValue::obj(vec![("message", JsonValue::from("model not found"))]),
                    )]);
                    return write_json(stream, 404, &err, &[]);
                }
            }
            let ctx = state.reported_ctx();
            let s = &state.config.sampling;
            let params = JsonValue::obj(vec![
                ("n_predict", JsonValue::from(state.config.n_predict)),
                ("seed", JsonValue::from(4294967295i64)),
                ("temperature", JsonValue::from(s.temperature)),
                ("dynatemp_range", JsonValue::from(0.0)),
                ("dynatemp_exponent", JsonValue::from(1.0)),
                ("top_k", JsonValue::from(s.top_k)),
                ("top_p", JsonValue::from(s.top_p)),
                ("min_p", JsonValue::from(s.min_p)),
                ("repeat_last_n", JsonValue::from(64)),
                ("repeat_penalty", JsonValue::from(s.repeat_penalty)),
                ("presence_penalty", JsonValue::from(s.presence_penalty)),
                ("frequency_penalty", JsonValue::from(s.frequency_penalty)),
                ("stop", JsonValue::empty_arr()),
                ("max_tokens", JsonValue::from(state.config.n_predict)),
                ("ignore_eos", JsonValue::from(false)),
                ("stream", JsonValue::from(true)),
                ("speculative.n_max", JsonValue::from(state.config.draft_max)),
                ("speculative.n_min", JsonValue::from(0)),
                (
                    "speculative.p_min",
                    JsonValue::from(state.config.draft_p_min),
                ),
            ]);
            let props = JsonValue::obj(vec![
                (
                    "default_generation_settings",
                    JsonValue::obj(vec![
                        ("id", JsonValue::from(0)),
                        ("id_task", JsonValue::from(0)),
                        ("n_ctx", JsonValue::from(ctx)),
                        ("speculative", JsonValue::from(state.config.draft_max > 0)),
                        (
                            "is_processing",
                            JsonValue::from(state.running_requests.load(Ordering::Relaxed) > 0),
                        ),
                        ("params", params),
                    ]),
                ),
                (
                    "total_slots",
                    JsonValue::from(state.config.n_parallel.max(1)),
                ),
                (
                    "model_path",
                    JsonValue::from(state.config.model_path.as_str()),
                ),
                (
                    "model_alias",
                    JsonValue::from(state.config.model_name.as_str()),
                ),
                ("chat_template", JsonValue::from("chatml")),
                (
                    "chat_template_caps",
                    JsonValue::obj(vec![
                        ("supports_tools", JsonValue::from(true)),
                        ("supports_tool_calls", JsonValue::from(true)),
                        ("supports_system_role", JsonValue::from(true)),
                        ("supports_parallel_tool_calls", JsonValue::from(true)),
                    ]),
                ),
                (
                    "modalities",
                    JsonValue::obj(vec![
                        ("vision", JsonValue::from(false)),
                        ("audio", JsonValue::from(false)),
                    ]),
                ),
                ("bos_token", JsonValue::from("<|im_start|>")),
                ("eos_token", JsonValue::from("<|im_end|>")),
                ("build_info", JsonValue::from("talus-server")),
                ("is_sleeping", JsonValue::from(!state.engine.is_loaded())),
                (
                    "hip_Runtime_tuning",
                    JsonValue::obj(vec![
                        ("n_ctx", JsonValue::from(ctx)),
                        ("n_ctx_per_slot", JsonValue::from(ctx)),
                        (
                            "n_parallel",
                            JsonValue::from(state.config.n_parallel.max(1)),
                        ),
                        ("kv_unified", JsonValue::from(state.config.kv_unified)),
                        (
                            "cache_type_k",
                            JsonValue::from(state.config.cache_type_k.as_str()),
                        ),
                        (
                            "cache_type_v",
                            JsonValue::from(state.config.cache_type_v.as_str()),
                        ),
                        (
                            "flash_attn",
                            JsonValue::from(state.config.flash_attn.as_str()),
                        ),
                        ("fit", JsonValue::from(state.config.fit.as_str())),
                        ("ctx_shift", JsonValue::from(state.config.ctx_shift)),
                        ("draft_max", JsonValue::from(state.config.draft_max)),
                        ("draft_p_min", JsonValue::from(state.config.draft_p_min)),
                        (
                            "slot_save_path",
                            JsonValue::from(state.config.slot_save_path.as_str()),
                        ),
                    ]),
                ),
            ]);
            write_json(stream, 200, &props, &[])
        }
        "/slots" => {
            if !state.engine.is_loaded() {
                return write_json(stream, 200, &JsonValue::empty_arr(), &[]);
            }
            let ctx = state.reported_ctx();
            let busy = state.running_requests.load(Ordering::Relaxed) > 0;
            let m = state.metrics.lock().unwrap();
            let in_use = if busy {
                state.active_prompt_tokens.load(Ordering::Relaxed)
            } else {
                m.last_prompt_tokens
            };
            let cache_n = m.last_timings.as_ref().map(|t| t.cache_n).unwrap_or(0);
            drop(m);
            let n_slots = state.config.n_parallel.max(1);
            let mut slots = Vec::with_capacity(n_slots);
            for slot_id in 0..n_slots {
                let slot_busy = slot_id == 0 && busy;
                slots.push(JsonValue::obj(vec![
                    ("id", JsonValue::from(slot_id)),
                    ("id_task", JsonValue::from(0)),
                    ("n_ctx", JsonValue::from(ctx)),
                    ("speculative", JsonValue::from(state.config.draft_max > 0)),
                    ("is_processing", JsonValue::from(slot_busy)),
                    ("state", JsonValue::from(if slot_busy { 1 } else { 0 })),
                    ("n_prompt_tokens", JsonValue::from(in_use)),
                    ("cache_n", JsonValue::from(cache_n)),
                ]));
            }
            write_json(stream, 200, &JsonValue::Array(slots), &[])
        }
        "/metrics/prometheus" => {
            let snap = state.prometheus_snapshot();
            let text = render_llamacpp_prometheus(&snap);
            write_response(stream, 200, PROM_CONTENT_TYPE, text.as_bytes(), &[])
        }
        "/metrics" => {
            if wants_prometheus(req.header("accept"), Some(&req.query)) {
                let snap = state.prometheus_snapshot();
                let mut text = render_prometheus(&snap);
                text.push_str(&render_llamacpp_prometheus(&snap));
                write_response(stream, 200, PROM_CONTENT_TYPE, text.as_bytes(), &[])
            } else {
                let all_reqs = req.query.contains("requests=all");
                let body = state.metrics_json(all_reqs);
                write_json(stream, 200, &body, &[])
            }
        }
        "/status" => {
            let busy = state.running_requests.load(Ordering::Relaxed) > 0;
            let (_, waiting) = state.queue_counts();
            let body = JsonValue::obj(vec![
                ("busy", JsonValue::from(busy)),
                ("queued", JsonValue::from(waiting)),
                (
                    "prompt_tokens",
                    JsonValue::from(state.active_prompt_tokens.load(Ordering::Relaxed)),
                ),
                (
                    "generated",
                    JsonValue::from(state.active_generated_tokens.load(Ordering::Relaxed)),
                ),
            ]);
            write_json(stream, 200, &body, &[])
        }
        "/v1/status" => {
            let ctx = state.reported_ctx();
            let body = JsonValue::obj(vec![
                ("service", JsonValue::from("strata")),
                ("model", JsonValue::from(state.config.model_name.as_str())),
                ("loaded", JsonValue::from(state.engine.is_loaded())),
                ("cache_max_tokens", JsonValue::from(ctx)),
                (
                    "context",
                    JsonValue::obj(vec![
                        ("native", JsonValue::from(ctx)),
                        ("max_positions", JsonValue::from(ctx)),
                    ]),
                ),
                (
                    "dialects",
                    JsonValue::Array(vec![
                        JsonValue::from("/v1/chat/completions"),
                        JsonValue::from("/v1/messages"),
                        JsonValue::from("/v1/responses"),
                    ]),
                ),
            ]);
            write_json(stream, 200, &body, &[])
        }
        "/settings" => {
            let shared = state.shared_defaults.lock().unwrap().clone();
            let has_shared = shared.as_object().map(|o| !o.is_empty()).unwrap_or(false);
            let body = JsonValue::obj(vec![
                ("shared", JsonValue::from(has_shared)),
                ("defaults", shared),
            ]);
            write_json(stream, 200, &body, &[])
        }
        "/config" => write_json(stream, 200, &state.config.raw_config, &[]),
        p if p.starts_with("/v1/talus/") => {
            let peer_is_local = stream
                .peer_addr()
                .map(|a| a.ip().is_loopback())
                .unwrap_or(false);
            let (code, body) =
                crate::control::handle_get_query(p, &req.query, state, peer_is_local);
            write_json(stream, code, &body, &[])
        }
        p if p.starts_with("/v1/responses/") => {
            let body = responses_error_body(
                "this server keeps no responses (stateless); send the whole conversation to POST /v1/responses",
                "invalid_request_error",
                Some("response_id"),
                Some("not_found"),
            );
            write_json(stream, 404, &body, &[])
        }
        _ => {
            let err = JsonValue::obj(vec![(
                "error",
                JsonValue::obj(vec![("message", JsonValue::from("not found"))]),
            )]);
            write_json(stream, 404, &err, &[])
        }
    }
}

fn handle_post(
    stream: &mut TcpStream,
    req: &HttpRequest,
    state: &Arc<ServiceState>,
) -> std::io::Result<()> {
    // Foreign browser page protection when no API key is set (matching serve/server.py _foreign_page)
    if state.config.api_key.is_none() {
        if let Some(origin) = req.header("origin") {
            if !is_local_origin(origin, req.header("host"), &state.config.host) {
                let err = JsonValue::obj(vec![(
                    "error",
                    JsonValue::obj(vec![
                        ("type", JsonValue::from("forbidden")),
                        (
                            "message",
                            JsonValue::from(format!(
                                "web pages of {origin} may not use this server without an API key"
                            )),
                        ),
                    ]),
                )]);
                return write_json(stream, 403, &err, &[]);
            }
            if !req
                .header("content-type")
                .unwrap_or("")
                .to_ascii_lowercase()
                .starts_with("application/json")
            {
                let err = JsonValue::obj(vec![(
                    "error",
                    JsonValue::obj(vec![("message", JsonValue::from("send application/json"))]),
                )]);
                return write_json(stream, 415, &err, &[]);
            }
        }
    }

    if req.path == "/load" || req.path == "/v1/load" {
        if let Err(e) = state.engine.load() {
            let err = JsonValue::obj(vec![(
                "error",
                JsonValue::obj(vec![
                    ("type", JsonValue::from("server_error")),
                    ("message", JsonValue::from(e.to_string())),
                ]),
            )]);
            return write_json(stream, 503, &err, &[]);
        }
        // a load counts as activity: the idle unloader waits a full period after it
        state.touch_activity();
        let body = JsonValue::obj(vec![("status", JsonValue::from("loaded"))]);
        return write_json(stream, 200, &body, &[]);
    }

    if req.path == "/unload" || req.path == "/v1/unload" {
        return handle_unload(stream, req, state);
    }

    let _values = crate::json::RequestValues::start();
    let body_str = match std::str::from_utf8(&req.body) {
        Ok(s) => s,
        Err(_) => {
            return send_post_error(
                stream,
                &req.path,
                400,
                "request body is not valid UTF-8",
                None,
                None,
            );
        }
    };

    let json_body = if body_str.trim().is_empty() {
        JsonValue::empty_obj()
    } else {
        match JsonValue::parse(body_str) {
            Ok(v) if v.is_object() => v,
            Ok(_) => {
                return send_post_error(stream, &req.path, 400, "send a JSON object", None, None);
            }
            Err(e) => {
                return send_post_error(
                    stream,
                    &req.path,
                    400,
                    &format!("invalid JSON: {e}"),
                    None,
                    None,
                );
            }
        }
    };

    if json_body.contains_key("reasoning_budget_tokens")
        && !BUDGET_WARNED.swap(true, Ordering::Relaxed)
    {
        eprintln!(
            "[talus-server] warning: a request set reasoning_budget_tokens: this server does not implement it and ignores it"
        );
    }

    if req.path.starts_with("/v1/responses/") {
        let body = responses_error_body(
            "this server keeps no responses (stateless); send the whole conversation to POST /v1/responses",
            "invalid_request_error",
            Some("response_id"),
            Some("not_found"),
        );
        return write_json(stream, 404, &body, &[]);
    }

    if req.path.starts_with("/slots/") {
        return handle_slot_action(stream, req, &json_body, state);
    }

    if req.path == "/v1/talus/score" {
        // E2 (talus-bench): teacher-forced scoring, see logprobs.rs
        let peer_is_local = stream
            .peer_addr()
            .map(|a| a.ip().is_loopback())
            .unwrap_or(false);
        let (code, body) = crate::logprobs::handle_score(&json_body, state, peer_is_local);
        return write_json(stream, code, &body, &[]);
    }

    if req.path.starts_with("/v1/talus/") {
        let peer_is_local = stream
            .peer_addr()
            .map(|a| a.ip().is_loopback())
            .unwrap_or(false);
        let reply = crate::control::handle_post(&req.path, &json_body, state, peer_is_local);
        write_json(stream, reply.code, &reply.body, &[])?;
        let _ = stream.flush();
        if let Some(cfg) = reply.restart {
            crate::control::spawn_restart(Arc::clone(state), cfg);
        }
        return Ok(());
    }

    match req.path.as_str() {
        "/settings" => {
            let Some(defaults) = json_body.get("defaults") else {
                return send_post_error(
                    stream,
                    &req.path,
                    400,
                    "the body must be {\"defaults\": {...}} or {\"defaults\": null}",
                    None,
                    None,
                );
            };
            let mut guard = state.shared_defaults.lock().unwrap();
            if defaults.is_null() {
                *guard = JsonValue::empty_obj();
            } else if defaults.is_object() {
                *guard = defaults.clone();
            } else {
                return send_post_error(
                    stream,
                    &req.path,
                    400,
                    "defaults must be an object or null",
                    None,
                    None,
                );
            }
            let has_shared = guard.as_object().map(|o| !o.is_empty()).unwrap_or(false);
            let out = JsonValue::obj(vec![
                ("shared", JsonValue::from(has_shared)),
                ("defaults", guard.clone()),
            ]);
            write_json(stream, 200, &out, &[])
        }
        "/tokenize" => {
            let content = json_body.get_str("content").unwrap_or("");
            let parse_special = json_body
                .get_bool("parse_special")
                .or_else(|| json_body.get_bool("add_special"))
                .unwrap_or(false);
            let with_pieces = json_body.get_bool("with_pieces").unwrap_or(false);
            let raw_ids = state.tokenizer.encode(content, parse_special);
            let tokens: Vec<JsonValue> = if with_pieces {
                raw_ids
                    .into_iter()
                    .map(|id| {
                        let piece = state.tokenizer.decode(&[id]);
                        JsonValue::obj(vec![
                            ("id", JsonValue::from(id)),
                            ("piece", JsonValue::from(piece)),
                        ])
                    })
                    .collect()
            } else {
                raw_ids.into_iter().map(JsonValue::from).collect()
            };
            let out = JsonValue::obj(vec![("tokens", JsonValue::Array(tokens))]);
            write_json(stream, 200, &out, &[])
        }
        "/detokenize" => {
            let mut ids = Vec::new();
            if let Some(arr) = json_body.get("tokens").and_then(|v| v.as_array()) {
                for item in arr {
                    let id = match item.as_i64() {
                        Some(n) => state.tokenizer.checked_id(n),
                        None => Err("tokens must be integers".to_string()),
                    };
                    match id {
                        Ok(id) => ids.push(id),
                        Err(e) => {
                            return send_post_error(
                                stream,
                                &req.path,
                                400,
                                &e,
                                Some("tokens"),
                                None,
                            )
                        }
                    }
                }
            }
            let text = state.tokenizer.decode(&ids);
            let out = JsonValue::obj(vec![("content", JsonValue::from(text))]);
            write_json(stream, 200, &out, &[])
        }
        "/completion" | "/completions" => handle_llama_completion(stream, &json_body, state),
        "/v1/completions" => handle_openai_completions(stream, &json_body, state),
        "/v1/chat/completions" => handle_openai_chat(stream, &json_body, state),
        "/v1/responses" => handle_responses(stream, &json_body, state),
        "/v1/messages/count_tokens" => handle_anthropic_count_tokens(stream, &json_body, state),
        "/v1/messages" => handle_anthropic_messages(stream, &json_body, state),
        _ => {
            let err = JsonValue::obj(vec![(
                "error",
                JsonValue::obj(vec![("message", JsonValue::from("not found"))]),
            )]);
            write_json(stream, 404, &err, &[])
        }
    }
}

/// `POST /unload`: stop the engine process so its memory is free (this used to answer
/// `unloaded` and do nothing). `?wait=1` waits up to 300 s for a running request first, `?wait=<seconds>` that long;
/// otherwise a running request is a 409. An engine that cannot unload is a 501.
fn handle_unload(
    stream: &mut TcpStream,
    req: &HttpRequest,
    state: &Arc<ServiceState>,
) -> std::io::Result<()> {
    let error = |kind: &str, msg: String| {
        JsonValue::obj(vec![(
            "error",
            JsonValue::obj(vec![
                ("type", JsonValue::from(kind)),
                ("message", JsonValue::from(msg)),
            ]),
        )])
    };
    if !state.engine.supports_lifecycle() {
        let err = error(
            "not_implemented",
            "this engine cannot unload; stop talus-server to free its memory".to_string(),
        );
        return write_json(stream, 501, &err, &[]);
    }
    let wait_s = match req.query_param("wait").as_deref() {
        None | Some("" | "0" | "false") => 0,
        Some("1" | "true") => 300,
        Some(n) => n.parse::<u64>().unwrap_or(0).min(3600),
    };
    let until = Instant::now() + Duration::from_secs(wait_s);
    while state.running_requests.load(Ordering::SeqCst) > 0 && Instant::now() < until {
        thread::sleep(Duration::from_millis(100));
    }
    if state.running_requests.load(Ordering::SeqCst) > 0 {
        let body = JsonValue::obj(vec![("status", JsonValue::from("busy"))]);
        return write_json(stream, 409, &body, &[]);
    }
    match state.engine.unload() {
        Ok(()) => {
            let body = JsonValue::obj(vec![("status", JsonValue::from("unloaded"))]);
            write_json(stream, 200, &body, &[])
        }
        Err(EngineError::Unsupported(m)) => {
            write_json(stream, 501, &error("not_implemented", m), &[])
        }
        Err(e) => write_json(stream, 500, &error("server_error", e.to_string()), &[]),
    }
}

/// Restart a crashed engine right away, within its cap: clients that only poll `/health` recover on their own.
fn spawn_engine_supervisor(state: Arc<ServiceState>, shutdown: Arc<AtomicBool>) {
    let spawned = thread::Builder::new()
        .name("talus-supervisor".to_string())
        .spawn(move || {
            while !shutdown.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(250));
                state.engine.supervise();
            }
        });
    if let Err(e) = spawned {
        eprintln!("[talus-server] warning: cannot start the engine supervisor: {e}");
    }
}

/// `idle_unload_seconds` > 0: unload the engine after that long without a request; the next request loads it again.
fn spawn_idle_unloader(state: Arc<ServiceState>, shutdown: Arc<AtomicBool>) {
    let secs = state.config.idle_unload_seconds;
    if secs == 0 {
        return;
    }
    let tick = Duration::from_millis(250);
    let spawned = thread::Builder::new()
        .name("talus-idle-unload".to_string())
        .spawn(move || {
            while !shutdown.load(Ordering::SeqCst) {
                thread::sleep(tick);
                if state.engine.is_loaded()
                    && state.running_requests.load(Ordering::SeqCst) == 0
                    && state.idle_seconds() >= secs as f64
                {
                    eprintln!("[talus-server] no request for {secs} s: unloading the engine");
                    if let Err(e) = state.engine.unload() {
                        eprintln!("[talus-server] idle unload failed: {e}");
                    }
                }
            }
        });
    if let Err(e) = spawned {
        eprintln!("[talus-server] warning: cannot start the idle unloader: {e}");
    }
}

/// #40: an engine's memory refusal (`ERR insufficient_memory: ...`: the host stayed under its MemAvailable floor
/// after the engine shed its caches; the engine keeps serving) -> the OpenAI error code; None for other errors.
pub fn engine_err_code(msg: &str) -> Option<&'static str> {
    if msg.starts_with(crate::engine::ENGINE_DOWN) {
        // the crash-loop cap stopped automatic restarts: 503 with the reason
        return Some("engine_down");
    }
    msg.contains("insufficient_memory")
        .then_some("insufficient_memory")
}

/// `reasoning_budget_tokens` in a request is warned about once per server run.
static BUDGET_WARNED: AtomicBool = AtomicBool::new(false);

/// What a streaming client is told when generation fails after the 200 went out.
fn stream_error_message(e: &EngineError) -> String {
    let msg = e.to_string();
    if matches!(e, EngineError::EngineDied(_)) && engine_err_code(&msg).is_none() {
        format!("{msg}; the engine restarts automatically")
    } else {
        msg
    }
}

/// The OpenAI/llama.cpp form of that: an `error` data event, then `[DONE]` on the routes that end with it. Best effort:
/// the client may be gone. (A stream that just closed looked like a short answer to clients that don't require `[DONE]`.)
fn write_stream_error(stream: &mut TcpStream, e: &EngineError, done: bool) {
    let msg = stream_error_message(e);
    let code = engine_err_code(&msg).unwrap_or("server_error");
    let err = JsonValue::obj(vec![(
        "error",
        JsonValue::obj(vec![
            ("type", JsonValue::from("server_error")),
            ("message", JsonValue::from(msg)),
            ("param", JsonValue::Null),
            ("code", JsonValue::from(code)),
        ]),
    )]);
    let mut text = format!("data: {}\n\n", err.to_json_string());
    if done {
        text.push_str("data: [DONE]\n\n");
    }
    let _ = stream.write_all(text.as_bytes());
    let _ = stream.flush();
}

fn send_post_error(
    stream: &mut TcpStream,
    path: &str,
    code: u16,
    message: &str,
    param: Option<&str>,
    err_code: Option<&str>,
) -> std::io::Result<()> {
    if path == "/v1/responses" {
        let body = responses_error_body(message, "invalid_request_error", param, err_code);
        write_json(stream, code, &body, &[])
    } else if path.starts_with("/v1/messages") {
        let body = JsonValue::obj(vec![
            ("type", JsonValue::from("error")),
            (
                "error",
                JsonValue::obj(vec![
                    ("type", JsonValue::from("invalid_request_error")),
                    ("message", JsonValue::from(message)),
                ]),
            ),
        ]);
        write_json(stream, code, &body, &[])
    } else {
        let kind = if matches!(err_code, Some("insufficient_memory" | "engine_down")) {
            "server_error"
        } else {
            "invalid_request_error"
        };
        let body = JsonValue::obj(vec![(
            "error",
            JsonValue::obj(vec![
                ("type", JsonValue::from(kind)),
                ("message", JsonValue::from(message)),
                ("param", JsonValue::from(param)),
                ("code", JsonValue::from(err_code)),
            ]),
        )]);
        write_json(stream, code, &body, &[])
    }
}

pub fn slot_filename_problem(name: &str) -> Option<&'static str> {
    if name.is_empty() || name.len() > 200 {
        return Some("filename must be 1..200 characters");
    }
    if name.starts_with('.') || name.ends_with('.') || name.ends_with(' ') {
        return Some("filename cannot start/end with '.' or end with space");
    }
    for ch in name.chars() {
        if ch.is_control() || matches!(ch, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') {
            return Some("filename contains forbidden path or device characters");
        }
    }
    let stem = name.split('.').next().unwrap_or(name).to_ascii_uppercase();
    if matches!(
        stem.as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
    ) {
        return Some("filename is a reserved device name");
    }
    None
}

fn handle_slot_action(
    stream: &mut TcpStream,
    req: &HttpRequest,
    json_body: &JsonValue,
    state: &Arc<ServiceState>,
) -> std::io::Result<()> {
    if !req
        .header("content-type")
        .unwrap_or("")
        .to_ascii_lowercase()
        .starts_with("application/json")
    {
        let err = JsonValue::obj(vec![(
            "error",
            JsonValue::obj(vec![("message", JsonValue::from("send application/json"))]),
        )]);
        return write_json(stream, 415, &err, &[]);
    }
    let slot_str = req.path.trim_start_matches("/slots/");
    let Ok(slot_id) = slot_str.parse::<usize>() else {
        return send_post_error(stream, &req.path, 400, "invalid slot id", None, None);
    };
    let action = req.query_param("action").unwrap_or_default();
    match action.as_str() {
        "erase" => match state.engine.erase_slot(slot_id) {
            Ok(n_erased) => {
                let out = JsonValue::obj(vec![
                    ("id_slot", JsonValue::from(slot_id)),
                    ("n_erased", JsonValue::from(n_erased)),
                ]);
                write_json(stream, 200, &out, &[])
            }
            Err(e) => send_post_error(stream, &req.path, 400, &e.to_string(), None, None),
        },
        "save" | "restore" => {
            let Some(filename) = json_body.get_str("filename") else {
                return send_post_error(
                    stream,
                    &req.path,
                    400,
                    "filename must be a non-empty string",
                    Some("filename"),
                    None,
                );
            };
            if let Some(problem) = slot_filename_problem(filename) {
                return send_post_error(stream, &req.path, 400, problem, Some("filename"), None);
            }
            if state.config.slot_save_path.trim().is_empty() {
                return send_post_error(
                    stream,
                    &req.path,
                    400,
                    "slot_save_path is not configured",
                    None,
                    None,
                );
            }
            let dir = PathBuf::from(&state.config.slot_save_path);
            let full_path = dir.join(filename);
            if action == "save" {
                let _ = std::fs::create_dir_all(&dir);
                match state.engine.save_slot(slot_id, &full_path) {
                    Ok((n_saved, n_written, save_ms)) => {
                        let out = JsonValue::obj(vec![
                            ("id_slot", JsonValue::from(slot_id)),
                            ("filename", JsonValue::from(filename)),
                            ("n_saved", JsonValue::from(n_saved)),
                            ("n_written", JsonValue::from(n_written)),
                            (
                                "timings",
                                JsonValue::obj(vec![("save_ms", JsonValue::from(save_ms))]),
                            ),
                        ]);
                        write_json(stream, 200, &out, &[])
                    }
                    Err(e) => send_post_error(stream, &req.path, 400, &e.to_string(), None, None),
                }
            } else {
                if !full_path.exists() {
                    let err = JsonValue::obj(vec![(
                        "error",
                        JsonValue::obj(vec![(
                            "message",
                            JsonValue::from("session file not found"),
                        )]),
                    )]);
                    return write_json(stream, 404, &err, &[]);
                }
                match state.engine.restore_slot(slot_id, &full_path) {
                    Ok((n_restored, n_read, restore_ms)) => {
                        let out = JsonValue::obj(vec![
                            ("id_slot", JsonValue::from(slot_id)),
                            ("filename", JsonValue::from(filename)),
                            ("n_restored", JsonValue::from(n_restored)),
                            ("n_read", JsonValue::from(n_read)),
                            (
                                "timings",
                                JsonValue::obj(vec![("restore_ms", JsonValue::from(restore_ms))]),
                            ),
                        ]);
                        write_json(stream, 200, &out, &[])
                    }
                    Err(e) => send_post_error(stream, &req.path, 400, &e.to_string(), None, None),
                }
            }
        }
        _ => send_post_error(
            stream,
            &req.path,
            400,
            "unknown slot action (expected save, restore, or erase)",
            None,
            None,
        ),
    }
}

/// Longest stop string accepted; matching costs up to its length per generated token.
const MAX_STOP_BYTES: usize = 64 * 1024;

fn check_stop_len(key: &str, st: &str) -> Result<(), String> {
    if st.len() > MAX_STOP_BYTES {
        return Err(format!(
            "{key}: a stop string may be at most {MAX_STOP_BYTES} bytes"
        ));
    }
    Ok(())
}

fn parse_sampling_from_req(
    req: &JsonValue,
    defaults: &SamplingParams,
) -> Result<(SamplingParams, Vec<String>), String> {
    let mut s = defaults.clone();
    if let Some(v) = req.get("temperature") {
        if !v.is_null() {
            let f = v
                .as_f64()
                .ok_or_else(|| "temperature: must be a number".to_string())?;
            if !(0.0..=2.0).contains(&f) {
                return Err("temperature: must be between 0 and 2".to_string());
            }
            s.temperature = f;
        }
    }
    if let Some(v) = req.get("top_p") {
        if !v.is_null() {
            let f = v
                .as_f64()
                .ok_or_else(|| "top_p: must be a number".to_string())?;
            if !(0.0..=1.0).contains(&f) {
                return Err("top_p: must be between 0 and 1".to_string());
            }
            s.top_p = f;
        }
    }
    if let Some(v) = req.get("top_k") {
        if !v.is_null() {
            let i = v
                .as_i64()
                .ok_or_else(|| "top_k: must be an integer".to_string())?;
            if i < -1 {
                return Err("top_k: must be >= -1".to_string());
            }
            s.top_k = i;
        }
    }
    if let Some(v) = req.get("min_p") {
        if !v.is_null() {
            let f = v
                .as_f64()
                .ok_or_else(|| "min_p: must be a number".to_string())?;
            if !(0.0..=1.0).contains(&f) {
                return Err("min_p: must be between 0 and 1".to_string());
            }
            s.min_p = f;
        }
    }
    if let Some(v) = req
        .get("repeat_penalty")
        .or_else(|| req.get("repetition_penalty"))
    {
        if !v.is_null() {
            let f = v
                .as_f64()
                .ok_or_else(|| "repeat_penalty: must be a number".to_string())?;
            // the engine divides logits by it, as f32: 0, and what rounds to 0 or infinity, gives non-finite logits.
            // 1 is off, not 0 (v0.1.3 with 0 sampled "the the the ..."). Same rule as llama.cpp's samplers.
            let g = f as f32;
            if !(g.is_finite() && g > 0.0) {
                return Err(
                    "repeat_penalty: must be finite and greater than 0 (1 turns it off)"
                        .to_string(),
                );
            }
            s.repeat_penalty = f;
        }
    }
    if let Some(v) = req.get("presence_penalty") {
        if !v.is_null() {
            let f = v
                .as_f64()
                .ok_or_else(|| "presence_penalty: must be a number".to_string())?;
            if !(f as f32).is_finite() {
                return Err("presence_penalty: must be finite".to_string());
            }
            s.presence_penalty = f;
        }
    }
    if let Some(v) = req.get("frequency_penalty") {
        if !v.is_null() {
            let f = v
                .as_f64()
                .ok_or_else(|| "frequency_penalty: must be a number".to_string())?;
            if !(f as f32).is_finite() {
                return Err("frequency_penalty: must be finite".to_string());
            }
            s.frequency_penalty = f;
        }
    }
    if let Some(v) = req.get("seed") {
        if !v.is_null() {
            let i = v
                .as_i64()
                .ok_or_else(|| "seed: must be an integer".to_string())?;
            s.seed = Some(i);
        }
    }
    if req.get_bool("ignore_eos") == Some(true) || req.get_bool("strata_ignore_eos") == Some(true) {
        s.ignore_eos = true;
    }
    if req.get_bool("cache_prompt") == Some(false) || req.get_bool("strata_no_cache") == Some(true)
    {
        s.cache_prompt = false;
    }

    let mut stops = Vec::new();
    for key in ["stop", "stop_sequences"] {
        if let Some(v) = req.get(key) {
            if v.is_null() {
                continue;
            }
            if let Some(st) = v.as_str() {
                check_stop_len(key, st)?;
                if !st.is_empty() {
                    stops.push(st.to_string());
                }
            } else if let Some(arr) = v.as_array() {
                if arr.len() > 16 {
                    return Err(format!("{key}: at most 16 stop strings are supported"));
                }
                for item in arr {
                    let Some(st) = item.as_str() else {
                        return Err(format!("{key}: must be strings"));
                    };
                    check_stop_len(key, st)?;
                    if !st.is_empty() {
                        stops.push(st.to_string());
                    }
                }
            } else {
                return Err(format!("{key}: must be a string or list of strings"));
            }
        }
    }

    Ok((s, stops))
}

fn parse_prompt_ids(
    prompt_val: Option<&JsonValue>,
    tokenizer: &ByteTokenizer,
) -> Result<Vec<u32>, String> {
    let Some(v) = prompt_val else {
        return Err("prompt is required".to_string());
    };
    if let Some(s) = v.as_str() {
        let ids = tokenizer.encode(s, true);
        if ids.is_empty() {
            return Err("an empty prompt".to_string());
        }
        return Ok(ids);
    }
    if let Some(arr) = v.as_array() {
        if arr.is_empty() {
            return Err("an empty prompt".to_string());
        }
        // Check if array of ints
        let mut ids = Vec::with_capacity(arr.len());
        for item in arr {
            if let Some(n) = item.as_i64() {
                ids.push(tokenizer.checked_id(n)?);
            } else if let Some(s) = item.as_str() {
                ids.extend(tokenizer.encode(s, true));
            } else {
                return Err("prompt array must contain token IDs or strings".to_string());
            }
        }
        if ids.is_empty() {
            return Err("an empty prompt".to_string());
        }
        return Ok(ids);
    }
    Err("prompt must be a string or array of token IDs".to_string())
}

struct RequestTracker<'a> {
    state: &'a ServiceState,
    start: Instant,
    first_token_at: Option<Instant>,
    prompt_tokens: usize,
}

impl<'a> RequestTracker<'a> {
    fn begin(state: &'a ServiceState, prompt_tokens: usize) -> Self {
        state.touch_activity();
        state.running_requests.fetch_add(1, Ordering::SeqCst);
        state
            .active_prompt_tokens
            .store(prompt_tokens, Ordering::Relaxed);
        state.active_generated_tokens.store(0, Ordering::Relaxed);
        Self {
            state,
            start: Instant::now(),
            first_token_at: None,
            prompt_tokens,
        }
    }

    fn on_token(&mut self, count: usize) {
        if self.first_token_at.is_none() {
            self.first_token_at = Some(Instant::now());
        }
        self.state
            .active_generated_tokens
            .store(count, Ordering::Relaxed);
    }

    fn finish(self, timings: &EngineTimings, rescued_tool_calls: u64) {
        let e2e_s = self.start.elapsed().as_secs_f64();
        let ttft_s = self
            .first_token_at
            .map(|t| t.duration_since(self.start).as_secs_f64())
            .or(Some((timings.prompt_ms / 1000.0).max(0.0001)));
        let decode_s = (timings.predicted_ms / 1000.0).max(0.0);

        let mut m = self.state.metrics.lock().unwrap();
        m.requests_total += 1;
        m.prompt_tokens_total += self.prompt_tokens as u64;
        m.reused_tokens_total += (timings.cache_n.min(self.prompt_tokens)) as u64;
        m.output_tokens_total += timings.predicted_n as u64;
        m.prompt_ms_total += timings.prompt_ms;
        m.decode_ms_total += timings.predicted_ms;
        if let Some(d) = timings.draft_n {
            m.drafts_offered_total += d as u64;
        }
        if let Some(a) = timings.draft_accepted {
            m.drafts_accepted_total += a as u64;
        }
        m.tool_calls_from_reasoning_total += rescued_tool_calls;
        m.last_prompt_tokens = self.prompt_tokens;
        m.last_generated_tokens = timings.predicted_n;
        m.last_timings = Some(timings.clone());
        m.histograms.observe(
            ttft_s,
            timings.predicted_n,
            decode_s,
            e2e_s,
            &timings.finish_reason,
        );
        m.history.push(JsonValue::obj(vec![
            ("prompt_tokens", JsonValue::from(self.prompt_tokens)),
            ("reused", JsonValue::from(timings.cache_n)),
            ("output_tokens", JsonValue::from(timings.predicted_n)),
            ("prompt_ms", JsonValue::from(timings.prompt_ms)),
            ("decode_ms", JsonValue::from(timings.predicted_ms)),
            ("finish", JsonValue::from(timings.finish_reason.as_str())),
            (
                "ttft_ms",
                JsonValue::from(
                    ttft_s
                        .map(|s| (s * 1000.0 * 10.0).round() / 10.0)
                        .unwrap_or(0.0),
                ),
            ),
            ("drafts", JsonValue::from(timings.draft_n.unwrap_or(0))),
            (
                "accepted",
                JsonValue::from(timings.draft_accepted.unwrap_or(0)),
            ),
        ]));
        if m.history.len() > 100 {
            m.history.remove(0);
        }
    }
}

impl ServiceState {
    /// E2 scoring counts as one running request (no decode metrics are recorded for it).
    pub(crate) fn begin_score_request(&self, prompt_tokens: usize) -> impl Drop + '_ {
        RequestTracker::begin(self, prompt_tokens)
    }
}

impl Drop for RequestTracker<'_> {
    fn drop(&mut self) {
        self.state.running_requests.fetch_sub(1, Ordering::SeqCst);
        self.state.touch_activity();
    }
}

fn handle_llama_completion(
    stream: &mut TcpStream,
    req: &JsonValue,
    state: &Arc<ServiceState>,
) -> std::io::Result<()> {
    let ids = match parse_prompt_ids(req.get("prompt"), &state.tokenizer) {
        Ok(v) => v,
        Err(e) => return send_post_error(stream, "/completion", 400, &e, Some("prompt"), None),
    };
    let (mut sampling, stops) = match parse_sampling_from_req(req, &state.config.sampling) {
        Ok(v) => v,
        Err(e) => return send_post_error(stream, "/completion", 400, &e, None, None),
    };
    sampling.logprobs = match crate::logprobs::parse_request(req, false, state) {
        Ok(k) => k,
        Err(e) => return send_post_error(stream, "/completion", 400, &e, Some("logprobs"), None),
    };
    let want_lp = sampling.logprobs.is_some();
    let mut lp_entries: Vec<JsonValue> = Vec::new();
    let ctx = state.reported_ctx();
    if ids.len() >= ctx {
        return send_post_error(
            stream,
            "/completion",
            400,
            "prompt exceeds context length",
            Some("prompt"),
            Some("context_length_exceeded"),
        );
    }
    let raw_n = req
        .get_i64("n_predict")
        .or_else(|| req.get_i64("max_tokens"))
        .unwrap_or(-1);
    let max_new = if raw_n < 0 {
        (ctx - ids.len()).max(1)
    } else if raw_n == 0 {
        0
    } else {
        (raw_n as usize).min(ctx - ids.len())
    };

    let want_stream = req.get_bool("stream").unwrap_or(false);
    let cancel = Arc::new(AtomicBool::new(false));
    let mut tracker = RequestTracker::begin(state, ids.len());

    let gen_req = GenerateRequest {
        prompt_ids: ids.clone(),
        max_new_tokens: max_new,
        sampling,
        stop_tokens: state.tokenizer.stop_tokens.clone(),
        stop_strings: stops.clone(),
    };

    if !want_stream {
        let mut out_tokens = Vec::new();
        let mut detok = Detokenizer::new(&state.tokenizer);
        let mut stop_matcher = if stops.is_empty() {
            None
        } else {
            Some(StopMatcher::new(stops))
        };
        let mut content = String::new();
        let mut stopped_by_word = false;

        let res = state.engine.generate_with_logprobs(
            &gen_req,
            cancel.clone(),
            &mut |tok_id, lp| {
                if let Some(lp) = lp {
                    lp_entries.push(crate::logprobs::entry_json(
                        &state.tokenizer,
                        tok_id,
                        &lp,
                        "content",
                    ));
                }
                out_tokens.push(tok_id);
                tracker.on_token(out_tokens.len());
                let piece = detok.push(tok_id);
                if let Some(ref mut sm) = stop_matcher {
                    let (emit, hit) = sm.push(&piece);
                    content.push_str(&emit);
                    if hit.is_some() {
                        stopped_by_word = true;
                        return false;
                    }
                } else {
                    content.push_str(&piece);
                }
                true
            },
            &mut |_read, _total| {},
        );

        let mut timings = match res {
            Ok(t) => t,
            Err(e) => {
                let msg = e.to_string();
                let code = engine_err_code(&msg);
                let status = if code.is_some() { 503 } else { 500 };
                return send_post_error(stream, "/completion", status, &msg, None, code);
            }
        };
        let tail = detok.flush();
        if !stopped_by_word {
            if let Some(ref mut sm) = stop_matcher {
                let (emit, hit) = sm.push(&tail);
                content.push_str(&emit);
                if hit.is_none() {
                    content.push_str(&sm.flush());
                } else {
                    stopped_by_word = true;
                }
            } else {
                content.push_str(&tail);
            }
        }
        if stopped_by_word {
            timings.finish_reason = "stop".to_string();
        }
        let timings_json = timings.to_json(ids.len(), out_tokens.len());
        let cache_n = timings.cache_n;
        let stop_type = timings.finish_reason.clone();
        tracker.finish(&timings, 0);

        let tok_arr: Vec<JsonValue> = out_tokens.into_iter().map(JsonValue::from).collect();
        let body = JsonValue::obj(vec![
            ("content", JsonValue::from(content)),
            ("tokens", JsonValue::Array(tok_arr.clone())),
            ("tokens_predicted", JsonValue::from(tok_arr.len())),
            ("tokens_evaluated", JsonValue::from(ids.len())),
            ("tokens_cached", JsonValue::from(cache_n)),
            ("stop", JsonValue::from(true)),
            ("stop_type", JsonValue::from(stop_type)),
            ("timings", timings_json),
        ]);
        let mut body = body;
        if want_lp {
            body.insert(
                "logprobs",
                crate::logprobs::logprobs_obj(std::mem::take(&mut lp_entries)),
            );
        }
        write_json(stream, 200, &body, &[])
    } else {
        write_sse_headers(stream)?;
        let mut out_tokens = Vec::new();
        let mut detok = Detokenizer::new(&state.tokenizer);
        let mut stop_matcher = if stops.is_empty() {
            None
        } else {
            Some(StopMatcher::new(stops))
        };
        let mut stopped_by_word = false;
        let mut stream_err = false;

        let res = state.engine.generate_with_logprobs(
            &gen_req,
            cancel.clone(),
            &mut |tok_id, lp| {
                if let Some(lp) = lp {
                    lp_entries.push(crate::logprobs::entry_json(
                        &state.tokenizer,
                        tok_id,
                        &lp,
                        "content",
                    ));
                }
                out_tokens.push(tok_id);
                tracker.on_token(out_tokens.len());
                let piece = detok.push(tok_id);
                let emit = if let Some(ref mut sm) = stop_matcher {
                    let (em, hit) = sm.push(&piece);
                    if hit.is_some() {
                        stopped_by_word = true;
                    }
                    em
                } else {
                    piece
                };
                if !emit.is_empty() {
                    let mut chunk = JsonValue::obj(vec![
                        ("content", JsonValue::from(emit)),
                        ("tokens", JsonValue::Array(vec![JsonValue::from(tok_id)])),
                        ("stop", JsonValue::from(false)),
                    ]);
                    if want_lp {
                        chunk.insert(
                            "logprobs",
                            crate::logprobs::logprobs_obj(std::mem::take(&mut lp_entries)),
                        );
                    }
                    let line = format!("data: {}\n\n", chunk.to_json_string());
                    if stream.write_all(line.as_bytes()).is_err() || stream.flush().is_err() {
                        stream_err = true;
                        cancel.store(true, Ordering::SeqCst);
                        return false;
                    }
                }
                !stopped_by_word
            },
            &mut |_read, _total| {},
        );

        if stream_err {
            return Ok(());
        }
        let mut timings = match res {
            Ok(t) => t,
            Err(e) => {
                write_stream_error(stream, &e, false);
                return Ok(());
            }
        };
        if stopped_by_word {
            timings.finish_reason = "stop".to_string();
        }
        let timings_json = timings.to_json(ids.len(), out_tokens.len());
        let cache_n = timings.cache_n;
        let stop_type = timings.finish_reason.clone();
        tracker.finish(&timings, 0);

        let final_chunk = JsonValue::obj(vec![
            ("content", JsonValue::from("")),
            ("tokens", JsonValue::empty_arr()),
            ("tokens_predicted", JsonValue::from(out_tokens.len())),
            ("tokens_evaluated", JsonValue::from(ids.len())),
            ("tokens_cached", JsonValue::from(cache_n)),
            ("stop", JsonValue::from(true)),
            ("stop_type", JsonValue::from(stop_type)),
            ("timings", timings_json),
        ]);
        let mut final_chunk = final_chunk;
        if want_lp {
            final_chunk.insert(
                "logprobs",
                crate::logprobs::logprobs_obj(std::mem::take(&mut lp_entries)),
            );
        }
        let line = format!("data: {}\n\n", final_chunk.to_json_string());
        let _ = stream.write_all(line.as_bytes());
        let _ = stream.flush();
        Ok(())
    }
}

fn handle_openai_completions(
    stream: &mut TcpStream,
    req: &JsonValue,
    state: &Arc<ServiceState>,
) -> std::io::Result<()> {
    let ids = match parse_prompt_ids(req.get("prompt"), &state.tokenizer) {
        Ok(v) => v,
        Err(e) => return send_post_error(stream, "/v1/completions", 400, &e, Some("prompt"), None),
    };
    let (mut sampling, stops) = match parse_sampling_from_req(req, &state.config.sampling) {
        Ok(v) => v,
        Err(e) => return send_post_error(stream, "/v1/completions", 400, &e, None, None),
    };
    sampling.logprobs = match crate::logprobs::parse_request(req, false, state) {
        Ok(k) => k,
        Err(e) => {
            return send_post_error(stream, "/v1/completions", 400, &e, Some("logprobs"), None)
        }
    };
    let want_lp = sampling.logprobs.is_some();
    let mut lp_entries: Vec<JsonValue> = Vec::new();
    let ctx = state.reported_ctx();
    if ids.len() >= ctx {
        return send_post_error(
            stream,
            "/v1/completions",
            400,
            "prompt exceeds context length",
            Some("prompt"),
            Some("context_length_exceeded"),
        );
    }
    let raw_n = req
        .get_i64("max_tokens")
        .or_else(|| req.get_i64("n_predict"))
        .unwrap_or(256);
    let max_new = if raw_n < 0 {
        (ctx - ids.len()).max(1)
    } else if raw_n == 0 {
        0
    } else {
        (raw_n as usize).min(ctx - ids.len())
    };

    let want_stream = req.get_bool("stream").unwrap_or(false);
    let cid = format!("cmpl-{}", next_hex_id(24));
    let created = unix_timestamp_secs();
    let model = state.model_for(req);
    let cancel = Arc::new(AtomicBool::new(false));
    let mut tracker = RequestTracker::begin(state, ids.len());

    let gen_req = GenerateRequest {
        prompt_ids: ids.clone(),
        max_new_tokens: max_new,
        sampling,
        stop_tokens: state.tokenizer.stop_tokens.clone(),
        stop_strings: stops.clone(),
    };

    if !want_stream {
        let mut out_tokens = Vec::new();
        let mut detok = Detokenizer::new(&state.tokenizer);
        let mut stop_matcher = if stops.is_empty() {
            None
        } else {
            Some(StopMatcher::new(stops))
        };
        let mut text = String::new();
        let mut stopped_by_word = false;

        let res = state.engine.generate_with_logprobs(
            &gen_req,
            cancel.clone(),
            &mut |tok_id, lp| {
                if let Some(lp) = lp {
                    lp_entries.push(crate::logprobs::entry_json(
                        &state.tokenizer,
                        tok_id,
                        &lp,
                        "content",
                    ));
                }
                out_tokens.push(tok_id);
                tracker.on_token(out_tokens.len());
                let piece = detok.push(tok_id);
                if let Some(ref mut sm) = stop_matcher {
                    let (emit, hit) = sm.push(&piece);
                    text.push_str(&emit);
                    if hit.is_some() {
                        stopped_by_word = true;
                        return false;
                    }
                } else {
                    text.push_str(&piece);
                }
                true
            },
            &mut |_read, _total| {},
        );

        let mut timings = match res {
            Ok(t) => t,
            Err(e) => {
                let msg = e.to_string();
                let code = engine_err_code(&msg);
                let status = if code.is_some() { 503 } else { 500 };
                return send_post_error(stream, "/v1/completions", status, &msg, None, code);
            }
        };
        let tail = detok.flush();
        if !stopped_by_word {
            if let Some(ref mut sm) = stop_matcher {
                let (emit, hit) = sm.push(&tail);
                text.push_str(&emit);
                if hit.is_none() {
                    text.push_str(&sm.flush());
                } else {
                    stopped_by_word = true;
                }
            } else {
                text.push_str(&tail);
            }
        }
        if stopped_by_word {
            timings.finish_reason = "stop".to_string();
        }
        let finish_reason = if timings.finish_reason == "length" {
            "length"
        } else {
            "stop"
        };
        let timings_json = timings.to_json(ids.len(), out_tokens.len());
        let cache_n = timings.cache_n;
        tracker.finish(&timings, 0);

        let body = JsonValue::obj(vec![
            ("id", JsonValue::from(cid)),
            ("object", JsonValue::from("text_completion")),
            ("created", JsonValue::from(created)),
            ("model", JsonValue::from(model)),
            (
                "choices",
                JsonValue::Array(vec![JsonValue::obj(vec![
                    ("text", JsonValue::from(text)),
                    ("index", JsonValue::from(0)),
                    (
                        "logprobs",
                        if want_lp {
                            crate::logprobs::logprobs_obj(std::mem::take(&mut lp_entries))
                        } else {
                            JsonValue::Null
                        },
                    ),
                    ("finish_reason", JsonValue::from(finish_reason)),
                ])]),
            ),
            (
                "usage",
                JsonValue::obj(vec![
                    ("prompt_tokens", JsonValue::from(ids.len())),
                    ("completion_tokens", JsonValue::from(out_tokens.len())),
                    (
                        "total_tokens",
                        JsonValue::from(ids.len() + out_tokens.len()),
                    ),
                    (
                        "prompt_tokens_details",
                        JsonValue::obj(vec![("cached_tokens", JsonValue::from(cache_n))]),
                    ),
                ]),
            ),
            ("timings", timings_json),
        ]);
        write_json(stream, 200, &body, &[])
    } else {
        write_sse_headers(stream)?;
        let mut out_tokens = Vec::new();
        let mut detok = Detokenizer::new(&state.tokenizer);
        let mut stop_matcher = if stops.is_empty() {
            None
        } else {
            Some(StopMatcher::new(stops))
        };
        let mut stopped_by_word = false;
        let mut stream_err = false;

        let res = state.engine.generate_with_logprobs(
            &gen_req,
            cancel.clone(),
            &mut |tok_id, lp| {
                if let Some(lp) = lp {
                    lp_entries.push(crate::logprobs::entry_json(
                        &state.tokenizer,
                        tok_id,
                        &lp,
                        "content",
                    ));
                }
                out_tokens.push(tok_id);
                tracker.on_token(out_tokens.len());
                let piece = detok.push(tok_id);
                let emit = if let Some(ref mut sm) = stop_matcher {
                    let (em, hit) = sm.push(&piece);
                    if hit.is_some() {
                        stopped_by_word = true;
                    }
                    em
                } else {
                    piece
                };
                if !emit.is_empty() {
                    let chunk = JsonValue::obj(vec![
                        ("id", JsonValue::from(cid.as_str())),
                        ("object", JsonValue::from("text_completion")),
                        ("created", JsonValue::from(created)),
                        ("model", JsonValue::from(model.as_str())),
                        (
                            "choices",
                            JsonValue::Array(vec![JsonValue::obj(vec![
                                ("text", JsonValue::from(emit)),
                                ("index", JsonValue::from(0)),
                                (
                                    "logprobs",
                                    if want_lp {
                                        crate::logprobs::logprobs_obj(std::mem::take(
                                            &mut lp_entries,
                                        ))
                                    } else {
                                        JsonValue::Null
                                    },
                                ),
                                ("finish_reason", JsonValue::Null),
                            ])]),
                        ),
                    ]);
                    let line = format!("data: {}\n\n", chunk.to_json_string());
                    if stream.write_all(line.as_bytes()).is_err() || stream.flush().is_err() {
                        stream_err = true;
                        cancel.store(true, Ordering::SeqCst);
                        return false;
                    }
                }
                !stopped_by_word
            },
            &mut |_read, _total| {},
        );

        if stream_err {
            return Ok(());
        }
        let mut timings = match res {
            Ok(t) => t,
            Err(e) => {
                write_stream_error(stream, &e, true);
                return Ok(());
            }
        };
        if stopped_by_word {
            timings.finish_reason = "stop".to_string();
        }
        let finish_reason = if timings.finish_reason == "length" {
            "length"
        } else {
            "stop"
        };
        let timings_json = timings.to_json(ids.len(), out_tokens.len());
        let cache_n = timings.cache_n;
        tracker.finish(&timings, 0);

        let final_chunk = JsonValue::obj(vec![
            ("id", JsonValue::from(cid)),
            ("object", JsonValue::from("text_completion")),
            ("created", JsonValue::from(created)),
            ("model", JsonValue::from(model)),
            (
                "choices",
                JsonValue::Array(vec![JsonValue::obj(vec![
                    ("text", JsonValue::from("")),
                    ("index", JsonValue::from(0)),
                    (
                        "logprobs",
                        if want_lp {
                            crate::logprobs::logprobs_obj(std::mem::take(&mut lp_entries))
                        } else {
                            JsonValue::Null
                        },
                    ),
                    ("finish_reason", JsonValue::from(finish_reason)),
                ])]),
            ),
            (
                "usage",
                JsonValue::obj(vec![
                    ("prompt_tokens", JsonValue::from(ids.len())),
                    ("completion_tokens", JsonValue::from(out_tokens.len())),
                    (
                        "total_tokens",
                        JsonValue::from(ids.len() + out_tokens.len()),
                    ),
                    (
                        "prompt_tokens_details",
                        JsonValue::obj(vec![("cached_tokens", JsonValue::from(cache_n))]),
                    ),
                ]),
            ),
            ("timings", timings_json),
        ]);
        let _ = stream.write_all(format!("data: {}\n\n", final_chunk.to_json_string()).as_bytes());
        let _ = stream.write_all(b"data: [DONE]\n\n");
        let _ = stream.flush();
        Ok(())
    }
}

struct PreparedChat {
    ids: Vec<u32>,
    thinking: bool,
    tools: Option<Vec<JsonValue>>,
    max_new: usize,
    sampling: SamplingParams,
    stops: Vec<String>,
    force: Option<String>,
    /// E1 (opt-in): entries of generated tokens not yet handed to the client (chat streams drain it per chunk).
    lp_entries: std::cell::RefCell<Vec<JsonValue>>,
}

fn render_and_encode_chat(
    messages: &[JsonValue],
    tools: Option<&[JsonValue]>,
    kw: &crate::frontend::TemplateKwargs,
    state: &ServiceState,
    force: Option<&str>,
) -> Result<(Vec<u32>, bool), String> {
    if let Some(ref tpl) = state.chat_template {
        let s_msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(crate::frontend::to_serde_value)
            .collect();
        let s_tools: Option<Vec<serde_json::Value>> =
            tools.map(|ts| ts.iter().map(crate::frontend::to_serde_value).collect());
        let (marked_msgs, marked_tools, changed) =
            talus_text::mark_think_literals(&s_msgs, s_tools.as_deref(), Some(&state.literal_tags));
        let mut kw_map = serde_json::Map::new();
        if let Some(et) = kw.enable_thinking {
            kw_map.insert("enable_thinking".to_string(), serde_json::Value::Bool(et));
        }
        if let Some(ref eff) = kw.reasoning_effort {
            kw_map.insert(
                "reasoning_effort".to_string(),
                serde_json::Value::String(eff.clone()),
            );
        }
        if let Some(pt) = kw.preserve_thinking {
            kw_map.insert("preserve_thinking".to_string(), serde_json::Value::Bool(pt));
        }
        let mut rendered = tpl
            .render(
                &marked_msgs,
                marked_tools.as_deref(),
                kw.add_generation_prompt,
                Some(&kw_map),
            )
            .map_err(|e| e.to_string())?;
        let thinking = kw.thinking_enabled();
        if let Some(f) = force {
            if !thinking {
                rendered.push_str(f);
            }
        }
        let ids = if changed {
            let (clean, plain_spans) =
                talus_text::unmark_think_literals(&rendered, Some(&state.literal_tags));
            state
                .tokenizer
                .encode_with_plain_spans(&clean, true, &plain_spans)
        } else {
            state.tokenizer.encode(&rendered, true)
        };
        Ok((ids, thinking))
    } else {
        let mut rendered = render_chat_prompt(messages, tools, kw);
        let thinking = kw.thinking_enabled();
        if let Some(f) = force {
            if !thinking {
                rendered.push_str(f);
            }
        }
        let segments = unmark_think_literals(&rendered);
        let ids = state.tokenizer.encode_with_plain(&segments);
        Ok((ids, thinking))
    }
}

fn prepare_chat(
    messages: &[JsonValue],
    tools: Option<Vec<JsonValue>>,
    kw: &crate::frontend::TemplateKwargs,
    req: &JsonValue,
    state: &ServiceState,
    raw_max_tokens: i64,
    force: Option<String>,
) -> Result<PreparedChat, String> {
    let (sampling, stops) = parse_sampling_from_req(req, &state.config.sampling)?;
    let (ids, thinking) =
        render_and_encode_chat(messages, tools.as_deref(), kw, state, force.as_deref())?;
    let ctx = state.reported_ctx();
    if ids.len() >= ctx {
        return Err(format!(
            "prompt ({} tokens) exceeds context length ({ctx} tokens)",
            ids.len()
        ));
    }
    let max_new = if raw_max_tokens <= 0 {
        (ctx - ids.len()).max(1)
    } else {
        (raw_max_tokens as usize).min(ctx - ids.len())
    };
    Ok(PreparedChat {
        ids,
        thinking,
        tools,
        max_new,
        sampling,
        stops,
        force,
        lp_entries: std::cell::RefCell::new(Vec::new()),
    })
}

struct ChatRunOutput {
    events: Vec<Event>,
    completion_tokens: usize,
    reasoning_tokens: usize,
    /// Share of the decode time before the first answer (non-reasoning) token, from the server's token arrival
    /// times (0 without reasoning, 1 without an answer).
    reasoning_share: f64,
    timings: EngineTimings,
    stop_sequence: Option<String>,
}

impl ChatRunOutput {
    /// Bench extension E4: `usage.completion_tokens_details.reasoning_tokens` and the reasoning/answer split of
    /// `timings` (`reasoning_n` + `answer_n` = `predicted_n`; `reasoning_ms` + `answer_ms` = `predicted_ms`, split at
    /// the arrival of the first answer token). Only adds keys.
    fn add_channel_detail(&self, usage: &mut JsonValue, timings: &mut JsonValue) {
        usage.insert(
            "completion_tokens_details",
            JsonValue::obj(vec![(
                "reasoning_tokens",
                JsonValue::from(self.reasoning_tokens),
            )]),
        );
        let total_n = timings
            .get_usize("predicted_n")
            .unwrap_or(self.completion_tokens);
        let r_n = self.reasoning_tokens.min(total_n);
        let total_ms = timings.get_f64("predicted_ms").unwrap_or(0.0);
        let r_ms = EngineTimings::round1(total_ms * self.reasoning_share.clamp(0.0, 1.0));
        timings.insert("reasoning_n", JsonValue::from(r_n));
        timings.insert("answer_n", JsonValue::from(total_n - r_n));
        timings.insert("reasoning_ms", JsonValue::from(r_ms));
        timings.insert(
            "answer_ms",
            JsonValue::from(EngineTimings::round1((total_ms - r_ms).max(0.0))),
        );
    }
}

fn run_chat_generation<F>(
    prep: &PreparedChat,
    state: &ServiceState,
    cancel: Arc<AtomicBool>,
    mut on_event: F,
) -> Result<ChatRunOutput, EngineError>
where
    F: FnMut(&Event) -> bool,
{
    let mut parser = OutputParser::new(
        prep.thinking,
        prep.tools.clone(),
        true,
        state.config.tool_call_recovery,
        true,
    )
    .with_reasoning_rescue(state.config.reasoning_recovery);
    let mut detok = Detokenizer::new(&state.tokenizer);
    let mut stop_matcher = if prep.stops.is_empty() {
        None
    } else {
        Some(StopMatcher::new(prep.stops.clone()))
    };
    let mut all_events = Vec::new();
    let mut out_tokens = 0usize;
    let mut reasoning_tokens = 0usize;
    // arrival of the first token, the first answer token and the last token (reasoning/answer time split)
    let mut t_first: Option<Instant> = None;
    let mut t_answer: Option<Instant> = None;
    let mut t_last: Option<Instant> = None;
    let mut stopped_by_word: Option<String> = None;
    let mut client_aborted = false;

    if let Some(ref forced_prefix) = prep.force {
        if !prep.thinking {
            for ev in parser.feed(forced_prefix) {
                if !on_event(&ev) {
                    client_aborted = true;
                    break;
                }
                all_events.push(ev);
            }
        }
    }

    let mut tracker = RequestTracker::begin(state, prep.ids.len());
    let gen_req = GenerateRequest {
        prompt_ids: prep.ids.clone(),
        max_new_tokens: prep.max_new,
        sampling: prep.sampling.clone(),
        stop_tokens: state.tokenizer.stop_tokens.clone(),
        stop_strings: Vec::new(),
    };

    let emit_filtered = |evs: Vec<Event>,
                         stop_matcher: &mut Option<StopMatcher>,
                         stopped_by_word: &mut Option<String>,
                         all_events: &mut Vec<Event>,
                         on_event: &mut F|
     -> bool {
        for ev in evs {
            if stopped_by_word.is_some() {
                break;
            }
            if ev.kind == EventKind::Content {
                if let Some(ref mut sm) = stop_matcher {
                    let (text, hit) = sm.push(&ev.text);
                    if !text.is_empty() {
                        let cev = Event::content(text);
                        if !on_event(&cev) {
                            return false;
                        }
                        all_events.push(cev);
                    }
                    if let Some(word) = hit {
                        *stopped_by_word = Some(word);
                        break;
                    }
                    continue;
                }
            } else if let Some(ref mut sm) = stop_matcher {
                let held = sm.flush();
                if !held.is_empty() {
                    let cev = Event::content(held);
                    if !on_event(&cev) {
                        return false;
                    }
                    all_events.push(cev);
                }
            }
            if !on_event(&ev) {
                return false;
            }
            all_events.push(ev);
        }
        true
    };

    let mut channels = crate::logprobs::ChannelTracker::default();
    let mut timings = if client_aborted {
        EngineTimings::default()
    } else {
        state.engine.generate_with_logprobs(
            &gen_req,
            cancel.clone(),
            &mut |tok_id, lp| {
                out_tokens += 1;
                tracker.on_token(out_tokens);
                let piece = detok.push(tok_id);
                let evs = parser.feed(&piece);
                let now = Instant::now();
                t_first.get_or_insert(now);
                t_last = Some(now);
                if evs.iter().any(|e| e.kind == EventKind::Reasoning) {
                    reasoning_tokens += 1;
                } else if t_answer.is_none() && !evs.is_empty() {
                    t_answer = Some(now);
                }
                if let Some(lp) = lp {
                    let r = evs.iter().any(|e| e.kind == EventKind::Reasoning);
                    let a = evs.iter().any(|e| e.kind != EventKind::Reasoning);
                    let ch = channels.channel(r, a, prep.thinking);
                    prep.lp_entries
                        .borrow_mut()
                        .push(crate::logprobs::entry_json(
                            &state.tokenizer,
                            tok_id,
                            &lp,
                            ch,
                        ));
                }
                if !emit_filtered(
                    evs,
                    &mut stop_matcher,
                    &mut stopped_by_word,
                    &mut all_events,
                    &mut on_event,
                ) {
                    client_aborted = true;
                    cancel.store(true, Ordering::SeqCst);
                    return false;
                }
                stopped_by_word.is_none()
            },
            &mut |_read, _total| {},
        )?
    };

    if !client_aborted && stopped_by_word.is_none() {
        let tail = detok.flush();
        if !tail.is_empty() {
            let evs = parser.feed(&tail);
            let _ = emit_filtered(
                evs,
                &mut stop_matcher,
                &mut stopped_by_word,
                &mut all_events,
                &mut on_event,
            );
        }
    }
    if !client_aborted {
        let end_evs = parser.finish();
        let _ = emit_filtered(
            end_evs,
            &mut stop_matcher,
            &mut stopped_by_word,
            &mut all_events,
            &mut on_event,
        );
        if stopped_by_word.is_none() {
            if let Some(ref mut sm) = stop_matcher {
                let held = sm.flush();
                if !held.is_empty() {
                    let cev = Event::content(held);
                    let _ = on_event(&cev);
                    all_events.push(cev);
                }
            }
        }
    }

    if stopped_by_word.is_some() {
        timings.finish_reason = "stop".to_string();
    }
    timings.predicted_n = out_tokens;
    let reasoning_share = match (t_first, t_answer, t_last) {
        _ if reasoning_tokens == 0 => 0.0,
        (_, None, _) => 1.0,
        (Some(a), Some(b), Some(c)) if c > a => {
            b.saturating_duration_since(a).as_secs_f64() / (c - a).as_secs_f64()
        }
        _ => reasoning_tokens as f64 / out_tokens.max(1) as f64,
    };
    let rescued = parser.recovered_from_reasoning as u64;
    tracker.finish(&timings, rescued);

    Ok(ChatRunOutput {
        events: all_events,
        completion_tokens: out_tokens,
        reasoning_tokens,
        reasoning_share,
        timings,
        stop_sequence: stopped_by_word,
    })
}

const MAX_RESPONSE_FORMAT_VALUES: usize = 1 << 20;

/// Does `v` hold more than `max` JSON values? Stops counting there.
fn json_values_over(v: &JsonValue, max: usize) -> bool {
    fn count(v: &JsonValue, left: &mut usize) -> bool {
        if *left == 0 {
            return false;
        }
        *left -= 1;
        match v {
            JsonValue::Array(a) => a.iter().all(|x| count(x, left)),
            JsonValue::Object(o) => o.iter().all(|(_, x)| count(x, left)),
            _ => true,
        }
    }
    let mut left = max;
    !count(v, &mut left)
}

/// One schema compile at a time. A request that panicked while holding it leaves nothing to protect, so poison is ignored.
fn schema_compile_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// E6: the schema of `response_format` (`json_schema`, or `json_object` as `{"type":"object"}`), checked and
/// compacted by `talus_text::json_grammar::prepare`. `Ok(None)` = no constraint requested (absent, null, `text`).
pub fn response_format_schema(req: &JsonValue) -> Result<Option<String>, String> {
    let Some(rf) = req.get("response_format").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    // before the copies below
    if json_values_over(rf, MAX_RESPONSE_FORMAT_VALUES) {
        return Err(format!(
            "response_format: too large (more than {MAX_RESPONSE_FORMAT_VALUES} JSON values)"
        ));
    }
    // held for the copy and the compile only: this function returns before any engine work
    let _one = schema_compile_lock();
    let rf = crate::frontend::to_serde_value(rf);
    let kind = rf.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let schema = match kind {
        "text" => return Ok(None),
        "json_object" => serde_json::json!({"type": "object"}),
        "json_schema" => {
            let js = rf.get("json_schema");
            match js
                .and_then(|j| j.get("schema"))
                .or_else(|| rf.get("schema"))
            {
                Some(s) => s.clone(),
                None => {
                    return Err(
                        "response_format: json_schema.schema is required for type json_schema"
                            .to_string(),
                    )
                }
            }
        }
        other => {
            return Err(format!(
                "response_format: type must be text, json_object or json_schema (got {other:?})"
            ))
        }
    };
    talus_text::json_grammar::prepare(&schema)
        .map(Some)
        .map_err(|e| format!("response_format: {e}"))
}

fn handle_openai_chat(
    stream: &mut TcpStream,
    req: &JsonValue,
    state: &Arc<ServiceState>,
) -> std::io::Result<()> {
    let (messages, mut tools, mut kw) = match openai_to_messages(req) {
        Ok(v) => v,
        Err(e) => {
            let code = crate::frontend::content_err_code(&e);
            return send_post_error(stream, "/v1/chat/completions", 400, &e, None, code);
        }
    };
    // E6: grammar-constrained output. Thinking is turned off for these requests (the whole reply is the JSON value);
    // json_object stays ignored on engines without support (as before), json_schema gets an honest 400 there.
    let json_schema = match response_format_schema(req) {
        Ok(None) => None,
        Ok(Some(_)) if !state.engine.supports_json_schema() => {
            let is_object = req
                .get("response_format")
                .and_then(|r| r.get_str("type"))
                .is_some_and(|t| t == "json_object");
            if !is_object {
                return send_post_error(
                    stream,
                    "/v1/chat/completions",
                    400,
                    "json_schema not supported by this engine",
                    Some("response_format"),
                    None,
                );
            }
            None
        }
        Ok(Some(s)) => {
            kw.enable_thinking = Some(false);
            Some(s)
        }
        Err(e) => {
            return send_post_error(
                stream,
                "/v1/chat/completions",
                400,
                &e,
                Some("response_format"),
                None,
            )
        }
    };
    let (choice_mode, _) = match tool_choice_of(req.get("tool_choice")) {
        Ok(v) => v,
        Err(e) => {
            return send_post_error(
                stream,
                "/v1/chat/completions",
                400,
                &e,
                Some("tool_choice"),
                None,
            );
        }
    };
    if choice_mode == "none" {
        tools = None;
    }
    let force = match forced_call(req.get("tool_choice"), tools.as_deref()) {
        Ok(v) => v,
        Err(e) => {
            return send_post_error(
                stream,
                "/v1/chat/completions",
                400,
                &e,
                Some("tool_choice"),
                None,
            );
        }
    };
    let raw_max = req
        .get_i64("max_completion_tokens")
        .or_else(|| req.get_i64("max_tokens"))
        .unwrap_or(0);
    if json_schema.is_some() && (tools.as_ref().is_some_and(|t| !t.is_empty()) || force.is_some()) {
        return send_post_error(
            stream,
            "/v1/chat/completions",
            400,
            "response_format json_schema cannot be combined with tools (use tool_choice none)",
            Some("response_format"),
            None,
        );
    }
    let mut prep = match prepare_chat(&messages, tools, &kw, req, state, raw_max, force) {
        Ok(p) => p,
        Err(e) => {
            let code = crate::frontend::content_err_code(&e);
            return send_post_error(stream, "/v1/chat/completions", 400, &e, None, code);
        }
    };
    prep.sampling.logprobs = match crate::logprobs::parse_request(req, true, state) {
        Ok(k) => k,
        Err(e) => {
            return send_post_error(
                stream,
                "/v1/chat/completions",
                400,
                &e,
                Some("logprobs"),
                None,
            )
        }
    };
    let want_lp = prep.sampling.logprobs.is_some();
    let lp_take = |p: &PreparedChat| crate::logprobs::logprobs_obj(p.lp_entries.take());
    prep.sampling.json_schema = json_schema;

    let want_stream = req.get_bool("stream").unwrap_or(false);
    let cid = format!("chatcmpl-{}", next_hex_id(24));
    let created = unix_timestamp_secs();
    let model = state.model_for(req);
    let cancel = Arc::new(AtomicBool::new(false));

    let make_chunk = |delta: JsonValue, finish: Option<&str>| -> JsonValue {
        JsonValue::obj(vec![
            ("id", JsonValue::from(cid.as_str())),
            ("object", JsonValue::from("chat.completion.chunk")),
            ("created", JsonValue::from(created)),
            ("model", JsonValue::from(model.as_str())),
            (
                "choices",
                JsonValue::Array(vec![JsonValue::obj(vec![
                    ("index", JsonValue::from(0)),
                    ("delta", delta),
                    ("finish_reason", JsonValue::from(finish)),
                ])]),
            ),
        ])
    };

    if !want_stream {
        let out = match run_chat_generation(&prep, state, cancel, |_ev| true) {
            Ok(o) => o,
            Err(e) => {
                let msg = e.to_string();
                return send_post_error(
                    stream,
                    "/v1/chat/completions",
                    503,
                    &msg,
                    None,
                    engine_err_code(&msg),
                );
            }
        };

        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = Vec::new();
        for ev in &out.events {
            match ev.kind {
                EventKind::Reasoning => reasoning.push_str(&ev.text),
                EventKind::Content => content.push_str(&ev.text),
                EventKind::ToolCall => {
                    if let Some(ref call) = ev.call {
                        tool_calls.push(JsonValue::obj(vec![
                            ("id", JsonValue::from(call.id.as_str())),
                            ("type", JsonValue::from("function")),
                            (
                                "function",
                                JsonValue::obj(vec![
                                    ("name", JsonValue::from(call.name.as_str())),
                                    (
                                        "arguments",
                                        JsonValue::from(call.arguments.to_json_string()),
                                    ),
                                ]),
                            ),
                        ]));
                    }
                }
                _ => {}
            }
        }

        let finish_reason = if !tool_calls.is_empty() && out.timings.finish_reason == "stop" {
            "tool_calls"
        } else if out.timings.finish_reason == "length" {
            "length"
        } else {
            "stop"
        };

        let mut msg = JsonValue::obj(vec![
            ("role", JsonValue::from("assistant")),
            (
                "content",
                if content.is_empty() {
                    JsonValue::Null
                } else {
                    JsonValue::from(content)
                },
            ),
        ]);
        if !reasoning.is_empty() {
            msg.insert("reasoning_content", JsonValue::from(reasoning));
        }
        if !tool_calls.is_empty() {
            msg.insert("tool_calls", JsonValue::Array(tool_calls));
        }

        let pt = prep.ids.len();
        let ct = out.completion_tokens;
        let reused = out.timings.cache_n.min(pt);
        let mut timings_json = out.timings.to_json(pt, ct);
        let mut usage = JsonValue::obj(vec![
            ("prompt_tokens", JsonValue::from(pt)),
            ("completion_tokens", JsonValue::from(ct)),
            ("total_tokens", JsonValue::from(pt + ct)),
            (
                "prompt_tokens_details",
                JsonValue::obj(vec![("cached_tokens", JsonValue::from(reused))]),
            ),
        ]);
        out.add_channel_detail(&mut usage, &mut timings_json);

        let body = JsonValue::obj(vec![
            ("id", JsonValue::from(cid)),
            ("object", JsonValue::from("chat.completion")),
            ("created", JsonValue::from(created)),
            ("model", JsonValue::from(model)),
            (
                "choices",
                JsonValue::Array(vec![{
                    let mut c = JsonValue::obj(vec![
                        ("index", JsonValue::from(0)),
                        ("message", msg),
                        ("finish_reason", JsonValue::from(finish_reason)),
                    ]);
                    if want_lp {
                        c.insert("logprobs", lp_take(&prep));
                    }
                    c
                }]),
            ),
            ("usage", usage),
            ("timings", timings_json),
        ]);
        write_json(stream, 200, &body, &[])
    } else {
        write_sse_headers(stream)?;
        let first = make_chunk(
            JsonValue::obj(vec![
                ("role", JsonValue::from("assistant")),
                ("content", JsonValue::from("")),
            ]),
            None,
        );
        if stream
            .write_all(format!("data: {}\n\n", first.to_json_string()).as_bytes())
            .is_err()
        {
            return Ok(());
        }
        let _ = stream.flush();

        let mut calls_count = 0usize;
        let mut streamed_ids: HashMap<String, usize> = HashMap::new();
        let mut finished_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut stream_err = false;

        let out = match run_chat_generation(&prep, state, cancel, |ev| {
            let chunk_opt = match ev.kind {
                EventKind::Reasoning if !ev.text.is_empty() => Some(make_chunk(
                    JsonValue::obj(vec![(
                        "reasoning_content",
                        JsonValue::from(ev.text.as_str()),
                    )]),
                    None,
                )),
                EventKind::Content if !ev.text.is_empty() => Some(make_chunk(
                    JsonValue::obj(vec![("content", JsonValue::from(ev.text.as_str()))]),
                    None,
                )),
                EventKind::ToolStart => {
                    if let Some(ref call) = ev.call {
                        let idx = calls_count;
                        calls_count += 1;
                        streamed_ids.insert(call.id.clone(), idx);
                        Some(make_chunk(
                            JsonValue::obj(vec![(
                                "tool_calls",
                                JsonValue::Array(vec![JsonValue::obj(vec![
                                    ("index", JsonValue::from(idx)),
                                    ("id", JsonValue::from(call.id.as_str())),
                                    ("type", JsonValue::from("function")),
                                    (
                                        "function",
                                        JsonValue::obj(vec![
                                            ("name", JsonValue::from(call.name.as_str())),
                                            ("arguments", JsonValue::from("")),
                                        ]),
                                    ),
                                ])]),
                            )]),
                            None,
                        ))
                    } else {
                        None
                    }
                }
                EventKind::ToolArgs => {
                    if let Some(ref call) = ev.call {
                        if let Some(&idx) = streamed_ids.get(&call.id) {
                            Some(make_chunk(
                                JsonValue::obj(vec![(
                                    "tool_calls",
                                    JsonValue::Array(vec![JsonValue::obj(vec![
                                        ("index", JsonValue::from(idx)),
                                        (
                                            "function",
                                            JsonValue::obj(vec![(
                                                "arguments",
                                                JsonValue::from(ev.text.as_str()),
                                            )]),
                                        ),
                                    ])]),
                                )]),
                                None,
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                EventKind::ToolCall => {
                    if let Some(ref call) = ev.call {
                        if streamed_ids.contains_key(&call.id) {
                            finished_ids.insert(call.id.clone());
                            None
                        } else {
                            let idx = calls_count;
                            calls_count += 1;
                            Some(make_chunk(
                                JsonValue::obj(vec![(
                                    "tool_calls",
                                    JsonValue::Array(vec![JsonValue::obj(vec![
                                        ("index", JsonValue::from(idx)),
                                        ("id", JsonValue::from(call.id.as_str())),
                                        ("type", JsonValue::from("function")),
                                        (
                                            "function",
                                            JsonValue::obj(vec![
                                                ("name", JsonValue::from(call.name.as_str())),
                                                (
                                                    "arguments",
                                                    JsonValue::from(
                                                        call.arguments.to_json_string(),
                                                    ),
                                                ),
                                            ]),
                                        ),
                                    ])]),
                                )]),
                                None,
                            ))
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(mut c) = chunk_opt {
                if want_lp {
                    if let Some(ch) = c
                        .get_mut("choices")
                        .and_then(|v| v.as_array_mut())
                        .and_then(|a| a.first_mut())
                    {
                        ch.insert("logprobs", lp_take(&prep));
                    }
                }
                if stream
                    .write_all(format!("data: {}\n\n", c.to_json_string()).as_bytes())
                    .is_err()
                    || stream.flush().is_err()
                {
                    stream_err = true;
                    return false;
                }
            }
            true
        }) {
            Ok(o) => o,
            Err(e) => {
                if !stream_err {
                    write_stream_error(stream, &e, true);
                }
                return Ok(());
            }
        };

        if stream_err {
            return Ok(());
        }

        let whole_calls = calls_count > 0 && streamed_ids.keys().all(|k| finished_ids.contains(k));
        let finish_reason = if whole_calls && out.timings.finish_reason == "stop" {
            "tool_calls"
        } else if out.timings.finish_reason == "length" {
            "length"
        } else {
            "stop"
        };

        let pt = prep.ids.len();
        let ct = out.completion_tokens;
        let reused = out.timings.cache_n.min(pt);
        let mut timings_json = out.timings.to_json(pt, ct);
        let mut usage = JsonValue::obj(vec![
            ("prompt_tokens", JsonValue::from(pt)),
            ("completion_tokens", JsonValue::from(ct)),
            ("total_tokens", JsonValue::from(pt + ct)),
            (
                "prompt_tokens_details",
                JsonValue::obj(vec![("cached_tokens", JsonValue::from(reused))]),
            ),
        ]);
        out.add_channel_detail(&mut usage, &mut timings_json);

        let mut last_chunk = make_chunk(JsonValue::empty_obj(), Some(finish_reason));
        if want_lp {
            if let Some(ch) = last_chunk
                .get_mut("choices")
                .and_then(|v| v.as_array_mut())
                .and_then(|a| a.first_mut())
            {
                ch.insert("logprobs", lp_take(&prep));
            }
        }
        last_chunk.insert("usage", usage);
        last_chunk.insert("timings", timings_json);
        let _ = stream.write_all(format!("data: {}\n\n", last_chunk.to_json_string()).as_bytes());
        let _ = stream.write_all(b"data: [DONE]\n\n");
        let _ = stream.flush();
        Ok(())
    }
}

fn handle_responses(
    stream: &mut TcpStream,
    req: &JsonValue,
    state: &Arc<ServiceState>,
) -> std::io::Result<()> {
    if let Err(e) = check_responses_request(req) {
        return write_json(stream, e.status, &e.to_body(), &[]);
    }
    let messages = match responses_input_messages(req) {
        Ok(m) => m,
        Err(e) => return write_json(stream, e.status, &e.to_body(), &[]),
    };
    let (tools, tools_by_name) = match responses_request_tools(req) {
        Ok(t) => t,
        Err(e) => return write_json(stream, e.status, &e.to_body(), &[]),
    };
    let kw = match responses_template_kwargs(req, Some(&state.config.reasoning_effort)) {
        Ok(k) => k,
        Err(e) => return write_json(stream, e.status, &e.to_body(), &[]),
    };
    let raw_max = req.get_i64("max_output_tokens").unwrap_or(0);
    let prep = match prepare_chat(&messages, tools, &kw, req, state, raw_max, None) {
        Ok(p) => p,
        Err(e) => {
            let code = if e.contains("context") {
                Some("context_length_exceeded")
            } else {
                crate::frontend::content_err_code(&e)
            };
            let body = responses_error_body(&e, "invalid_request_error", Some("input"), code);
            return write_json(stream, 400, &body, &[]);
        }
    };

    let want_encrypted = req
        .get("include")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .any(|x| x.as_str() == Some("reasoning.encrypted_content"))
        })
        .unwrap_or(false);

    let model = state.model_for(req);
    let mut asm = Assembler::new(req, model, prep.ids.len(), tools_by_name, want_encrypted);
    let want_stream = req.get_bool("stream").unwrap_or(false);
    let cancel = Arc::new(AtomicBool::new(false));

    if !want_stream {
        let mut all_sse = asm.start();
        let out = match run_chat_generation(&prep, state, cancel, |ev| {
            all_sse.extend(asm.feed(ev));
            true
        }) {
            Ok(o) => o,
            Err(e) => {
                let msg = e.to_string();
                let body = responses_error_body(
                    &msg,
                    "server_error",
                    None,
                    Some(engine_err_code(&msg).unwrap_or("server_error")),
                );
                return write_json(stream, 503, &body, &[]);
            }
        };
        all_sse.extend(asm.finish(
            &out.timings.finish_reason,
            out.completion_tokens,
            out.timings.cache_n,
            out.reasoning_tokens,
        ));
        let Some(resp_obj) = collect_responses(&all_sse) else {
            let body = responses_error_body("empty response", "server_error", None, None);
            return write_json(stream, 500, &body, &[]);
        };
        write_json(stream, 200, &resp_obj, &[])
    } else {
        write_sse_headers(stream)?;
        let mut stream_err = false;
        let write_ev = |s: &mut TcpStream, ev: &JsonValue| -> bool {
            let t = ev.get_str("type").unwrap_or("message");
            let line = format!("event: {t}\ndata: {}\n\n", ev.to_json_string());
            s.write_all(line.as_bytes()).is_ok() && s.flush().is_ok()
        };

        for ev in asm.start() {
            if !write_ev(stream, &ev) {
                return Ok(());
            }
        }

        let out = match run_chat_generation(&prep, state, cancel, |ev| {
            let items = asm.feed(ev);
            for item in items {
                if !write_ev(stream, &item) {
                    stream_err = true;
                    return false;
                }
            }
            true
        }) {
            Ok(o) => o,
            Err(e) => {
                if !stream_err {
                    let msg = stream_error_message(&e);
                    let code = engine_err_code(&msg).unwrap_or("server_error");
                    let ev = asm.failed(&msg, code);
                    let _ = write_ev(stream, &ev);
                }
                return Ok(());
            }
        };

        if stream_err {
            return Ok(());
        }

        for ev in asm.finish(
            &out.timings.finish_reason,
            out.completion_tokens,
            out.timings.cache_n,
            out.reasoning_tokens,
        ) {
            if !write_ev(stream, &ev) {
                break;
            }
        }
        Ok(())
    }
}

fn handle_anthropic_count_tokens(
    stream: &mut TcpStream,
    req: &JsonValue,
    state: &Arc<ServiceState>,
) -> std::io::Result<()> {
    let (messages, tools, kw) =
        match anthropic_to_messages(req, state.config.anthropic_think_unasked) {
            Ok(v) => v,
            Err(e) => {
                return send_post_error(stream, "/v1/messages/count_tokens", 400, &e, None, None)
            }
        };
    if let Err(e) = parse_sampling_from_req(req, &state.config.sampling) {
        return send_post_error(stream, "/v1/messages/count_tokens", 400, &e, None, None);
    }
    let (ids, _) = match render_and_encode_chat(&messages, tools.as_deref(), &kw, state, None) {
        Ok(v) => v,
        Err(e) => return send_post_error(stream, "/v1/messages/count_tokens", 400, &e, None, None),
    };
    let body = JsonValue::obj(vec![("input_tokens", JsonValue::from(ids.len()))]);
    write_json(stream, 200, &body, &[])
}

fn handle_anthropic_messages(
    stream: &mut TcpStream,
    req: &JsonValue,
    state: &Arc<ServiceState>,
) -> std::io::Result<()> {
    let (messages, mut tools, kw) =
        match anthropic_to_messages(req, state.config.anthropic_think_unasked) {
            Ok(v) => v,
            Err(e) => return send_post_error(stream, "/v1/messages", 400, &e, None, None),
        };
    let (choice_mode, _) = match tool_choice_of(req.get("tool_choice")) {
        Ok(v) => v,
        Err(e) => {
            return send_post_error(stream, "/v1/messages", 400, &e, Some("tool_choice"), None);
        }
    };
    if choice_mode == "none" {
        tools = None;
    }
    let force = match forced_call(req.get("tool_choice"), tools.as_deref()) {
        Ok(v) => v,
        Err(e) => {
            return send_post_error(stream, "/v1/messages", 400, &e, Some("tool_choice"), None);
        }
    };
    let raw_max = req.get_i64("max_tokens").unwrap_or(0);
    let prep = match prepare_chat(&messages, tools, &kw, req, state, raw_max, force) {
        Ok(p) => p,
        Err(e) => return send_post_error(stream, "/v1/messages", 400, &e, None, None),
    };

    let want_stream = req.get_bool("stream").unwrap_or(false);
    let mid = format!("msg_{}", next_hex_id(24));
    let model = state.model_for(req);
    let cancel = Arc::new(AtomicBool::new(false));
    let prompt_len = prep.ids.len();

    // Build SSE events list using the exact state machine from serve/server.py anthropic_events
    let mut sse_events: Vec<(String, JsonValue)> = Vec::new();
    sse_events.push((
        "message_start".to_string(),
        JsonValue::obj(vec![
            ("type", JsonValue::from("message_start")),
            (
                "message",
                JsonValue::obj(vec![
                    ("id", JsonValue::from(mid.as_str())),
                    ("type", JsonValue::from("message")),
                    ("role", JsonValue::from("assistant")),
                    ("model", JsonValue::from(model.as_str())),
                    ("content", JsonValue::empty_arr()),
                    ("stop_reason", JsonValue::Null),
                    ("stop_sequence", JsonValue::Null),
                    (
                        "usage",
                        JsonValue::obj(vec![
                            ("input_tokens", JsonValue::from(prompt_len)),
                            ("output_tokens", JsonValue::from(0)),
                        ]),
                    ),
                ]),
            ),
        ]),
    ));

    let mut block_index: i64 = -1;
    let mut open_kind: Option<&'static str> = None;
    let mut used_tool = false;
    let mut streamed_tools = std::collections::HashSet::new();
    let mut finished_tools = std::collections::HashSet::new();
    let mut stream_err = false;

    if want_stream {
        write_sse_headers(stream)?;
        for (name, ev) in &sse_events {
            let line = format!("event: {name}\ndata: {}\n\n", ev.to_json_string());
            if stream.write_all(line.as_bytes()).is_err() || stream.flush().is_err() {
                return Ok(());
            }
        }
    }

    let out = match run_chat_generation(&prep, state, cancel, |ev| {
        let mut batch: Vec<(String, JsonValue)> = Vec::new();
        match ev.kind {
            EventKind::ToolArgs => {
                batch.push((
                    "content_block_delta".to_string(),
                    JsonValue::obj(vec![
                        ("type", JsonValue::from("content_block_delta")),
                        ("index", JsonValue::from(block_index)),
                        (
                            "delta",
                            JsonValue::obj(vec![
                                ("type", JsonValue::from("input_json_delta")),
                                ("partial_json", JsonValue::from(ev.text.as_str())),
                            ]),
                        ),
                    ]),
                ));
            }
            EventKind::ToolCall
                if ev
                    .call
                    .as_ref()
                    .map(|c| streamed_tools.contains(&c.id))
                    .unwrap_or(false) =>
            {
                if let Some(ref c) = ev.call {
                    finished_tools.insert(c.id.clone());
                }
            }
            _ => {
                let want = match ev.kind {
                    EventKind::Reasoning => "thinking",
                    EventKind::Content => "text",
                    EventKind::ToolStart | EventKind::ToolCall => "tool_use",
                    EventKind::ToolArgs => unreachable!(),
                };
                if !matches!(ev.kind, EventKind::ToolStart | EventKind::ToolCall)
                    && ev.text.is_empty()
                {
                    return true;
                }
                if ev.kind == EventKind::ToolStart {
                    if let Some(ref c) = ev.call {
                        streamed_tools.insert(c.id.clone());
                    }
                    used_tool = true;
                }
                if open_kind != Some(want) || want == "tool_use" {
                    if open_kind.is_some() {
                        batch.push((
                            "content_block_stop".to_string(),
                            JsonValue::obj(vec![
                                ("type", JsonValue::from("content_block_stop")),
                                ("index", JsonValue::from(block_index)),
                            ]),
                        ));
                    }
                    block_index += 1;
                    open_kind = Some(want);
                    let block = match want {
                        "thinking" => JsonValue::obj(vec![
                            ("type", JsonValue::from("thinking")),
                            ("thinking", JsonValue::from("")),
                            ("signature", JsonValue::from("")),
                        ]),
                        "text" => JsonValue::obj(vec![
                            ("type", JsonValue::from("text")),
                            ("text", JsonValue::from("")),
                        ]),
                        _ => {
                            let (cid, cname) = ev
                                .call
                                .as_ref()
                                .map(|c| (c.id.as_str(), c.name.as_str()))
                                .unwrap_or(("", ""));
                            JsonValue::obj(vec![
                                ("type", JsonValue::from("tool_use")),
                                ("id", JsonValue::from(cid)),
                                ("name", JsonValue::from(cname)),
                                ("input", JsonValue::empty_obj()),
                            ])
                        }
                    };
                    batch.push((
                        "content_block_start".to_string(),
                        JsonValue::obj(vec![
                            ("type", JsonValue::from("content_block_start")),
                            ("index", JsonValue::from(block_index)),
                            ("content_block", block),
                        ]),
                    ));
                }
                if want == "thinking" {
                    batch.push((
                        "content_block_delta".to_string(),
                        JsonValue::obj(vec![
                            ("type", JsonValue::from("content_block_delta")),
                            ("index", JsonValue::from(block_index)),
                            (
                                "delta",
                                JsonValue::obj(vec![
                                    ("type", JsonValue::from("thinking_delta")),
                                    ("thinking", JsonValue::from(ev.text.as_str())),
                                ]),
                            ),
                        ]),
                    ));
                } else if want == "text" {
                    batch.push((
                        "content_block_delta".to_string(),
                        JsonValue::obj(vec![
                            ("type", JsonValue::from("content_block_delta")),
                            ("index", JsonValue::from(block_index)),
                            (
                                "delta",
                                JsonValue::obj(vec![
                                    ("type", JsonValue::from("text_delta")),
                                    ("text", JsonValue::from(ev.text.as_str())),
                                ]),
                            ),
                        ]),
                    ));
                } else if ev.kind == EventKind::ToolCall {
                    used_tool = true;
                    if let Some(ref c) = ev.call {
                        batch.push((
                            "content_block_delta".to_string(),
                            JsonValue::obj(vec![
                                ("type", JsonValue::from("content_block_delta")),
                                ("index", JsonValue::from(block_index)),
                                (
                                    "delta",
                                    JsonValue::obj(vec![
                                        ("type", JsonValue::from("input_json_delta")),
                                        (
                                            "partial_json",
                                            JsonValue::from(c.arguments.to_json_string()),
                                        ),
                                    ]),
                                ),
                            ]),
                        ));
                    }
                }
            }
        }
        for (name, item) in batch {
            if want_stream {
                let line = format!("event: {name}\ndata: {}\n\n", item.to_json_string());
                if stream.write_all(line.as_bytes()).is_err() || stream.flush().is_err() {
                    stream_err = true;
                    return false;
                }
            }
            sse_events.push((name, item));
        }
        true
    }) {
        Ok(o) => o,
        Err(e) => {
            if want_stream {
                if !stream_err {
                    let err = JsonValue::obj(vec![
                        ("type", JsonValue::from("error")),
                        (
                            "error",
                            JsonValue::obj(vec![
                                ("type", JsonValue::from("api_error")),
                                ("message", JsonValue::from(stream_error_message(&e))),
                            ]),
                        ),
                    ]);
                    let line = format!("event: error\ndata: {}\n\n", err.to_json_string());
                    let _ = stream.write_all(line.as_bytes());
                    let _ = stream.flush();
                }
                return Ok(());
            }
            return send_post_error(stream, "/v1/messages", 503, &e.to_string(), None, None);
        }
    };

    if stream_err {
        return Ok(());
    }

    let mut tail_events: Vec<(String, JsonValue)> = Vec::new();
    if open_kind.is_some() {
        tail_events.push((
            "content_block_stop".to_string(),
            JsonValue::obj(vec![
                ("type", JsonValue::from("content_block_stop")),
                ("index", JsonValue::from(block_index)),
            ]),
        ));
    }

    let all_streamed_done = streamed_tools.iter().all(|id| finished_tools.contains(id));
    let stop_reason = if out.stop_sequence.is_some() {
        "stop_sequence"
    } else if used_tool && all_streamed_done && out.timings.finish_reason == "stop" {
        "tool_use"
    } else if out.timings.finish_reason == "length" {
        "max_tokens"
    } else {
        "end_turn"
    };

    let reused = out.timings.cache_n.min(prompt_len.saturating_sub(1));
    tail_events.push((
        "message_delta".to_string(),
        JsonValue::obj(vec![
            ("type", JsonValue::from("message_delta")),
            (
                "delta",
                JsonValue::obj(vec![
                    ("stop_reason", JsonValue::from(stop_reason)),
                    (
                        "stop_sequence",
                        JsonValue::from(out.stop_sequence.as_deref()),
                    ),
                ]),
            ),
            (
                "usage",
                JsonValue::obj(vec![
                    ("input_tokens", JsonValue::from(prompt_len - reused)),
                    ("cache_read_input_tokens", JsonValue::from(reused)),
                    ("output_tokens", JsonValue::from(out.completion_tokens)),
                ]),
            ),
        ]),
    ));
    tail_events.push((
        "message_stop".to_string(),
        JsonValue::obj(vec![("type", JsonValue::from("message_stop"))]),
    ));

    if want_stream {
        for (name, item) in &tail_events {
            let line = format!("event: {name}\ndata: {}\n\n", item.to_json_string());
            if stream.write_all(line.as_bytes()).is_err() || stream.flush().is_err() {
                break;
            }
        }
        Ok(())
    } else {
        sse_events.extend(tail_events);
        let msg = anthropic_collect(&sse_events);
        write_json(stream, 200, &msg, &[])
    }
}

fn anthropic_collect(events: &[(String, JsonValue)]) -> JsonValue {
    let mut msg = JsonValue::empty_obj();
    let mut blocks: Vec<JsonValue> = Vec::new();
    let mut partial_jsons: HashMap<usize, String> = HashMap::new();

    for (name, e) in events {
        match name.as_str() {
            "message_start" => {
                if let Some(m) = e.get("message") {
                    msg = m.clone();
                }
            }
            "content_block_start" => {
                if let Some(cb) = e.get("content_block") {
                    blocks.push(cb.clone());
                }
            }
            "content_block_delta" => {
                if let (Some(d), Some(last_block)) = (e.get("delta"), blocks.last_mut()) {
                    let dtype = d.get_str("type").unwrap_or("");
                    if dtype == "text_delta" {
                        let prev = last_block.get_str("text").unwrap_or("").to_string();
                        let add = d.get_str("text").unwrap_or("");
                        last_block.insert("text", JsonValue::from(format!("{prev}{add}")));
                    } else if dtype == "thinking_delta" {
                        let prev = last_block.get_str("thinking").unwrap_or("").to_string();
                        let add = d.get_str("thinking").unwrap_or("");
                        last_block.insert("thinking", JsonValue::from(format!("{prev}{add}")));
                    } else if dtype == "input_json_delta" {
                        let idx = blocks.len() - 1;
                        let entry = partial_jsons.entry(idx).or_default();
                        entry.push_str(d.get_str("partial_json").unwrap_or(""));
                    }
                }
            }
            "content_block_stop" => {
                if !blocks.is_empty() {
                    let idx = blocks.len() - 1;
                    if let Some(raw_json) = partial_jsons.remove(&idx) {
                        let trimmed = if raw_json.trim().is_empty() {
                            "{}"
                        } else {
                            raw_json.as_str()
                        };
                        if let Ok(parsed) = JsonValue::parse(trimmed) {
                            blocks[idx].insert("input", parsed);
                        } else {
                            blocks.pop();
                        }
                    }
                }
            }
            "message_delta" => {
                if let Some(d) = e.get("delta") {
                    if let Some(sr) = d.get("stop_reason") {
                        msg.insert("stop_reason", sr.clone());
                    }
                    if let Some(ss) = d.get("stop_sequence") {
                        msg.insert("stop_sequence", ss.clone());
                    }
                }
                if let Some(u) = e.get("usage").and_then(|v| v.as_object()) {
                    let mut cur_usage = msg
                        .get("usage")
                        .cloned()
                        .unwrap_or_else(JsonValue::empty_obj);
                    for (k, v) in u {
                        cur_usage.insert(k.clone(), v.clone());
                    }
                    msg.insert("usage", cur_usage);
                }
            }
            _ => {}
        }
    }
    msg.insert("content", JsonValue::Array(blocks));
    msg
}

#[cfg(test)]
mod body_charge_tests {
    use super::*;

    #[test]
    fn small_bodies_are_free_and_large_ones_share_the_limit() {
        let n = AtomicUsize::new(0);
        let limit = 3 * BODY_CHARGED_FROM;
        let mut small = BodyCharge::new(&n, limit);
        assert!(small.reserve(BODY_CHARGED_FROM));
        assert_eq!(n.load(Ordering::SeqCst), 0);
        let mut a = BodyCharge::new(&n, limit);
        assert!(a.reserve(2 * BODY_CHARGED_FROM));
        let mut b = BodyCharge::new(&n, limit);
        assert!(!b.reserve(2 * BODY_CHARGED_FROM));
        assert_eq!(n.load(Ordering::SeqCst), 2 * BODY_CHARGED_FROM);
        // growing only charges the difference
        assert!(a.reserve(3 * BODY_CHARGED_FROM));
        assert_eq!(n.load(Ordering::SeqCst), 3 * BODY_CHARGED_FROM);
        drop(a);
        assert_eq!(n.load(Ordering::SeqCst), 0);
        assert!(b.reserve(2 * BODY_CHARGED_FROM));
        drop(b);
        assert_eq!(n.load(Ordering::SeqCst), 0);
    }
}

#[cfg(test)]
mod mempr_tests {
    use super::engine_err_code;

    #[test]
    fn schema_compile_lock_survives_a_panic() {
        use super::{response_format_schema, schema_compile_lock, JsonValue};
        let t = std::thread::spawn(|| {
            let _held = schema_compile_lock();
            panic!("while holding the lock");
        });
        assert!(t.join().is_err());
        drop(schema_compile_lock());
        let req = JsonValue::parse(
            r#"{"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}}"#,
        )
        .unwrap();
        assert!(response_format_schema(&req).unwrap().is_some());
    }

    #[test]
    fn engine_memory_refusal_has_its_code() {
        assert_eq!(
            engine_err_code(
                "insufficient_memory: host MemAvailable 7900 MiB is below the 8192 MiB floor"
            ),
            Some("insufficient_memory")
        );
        assert_eq!(engine_err_code("bad prompt ids"), None);
    }
}
