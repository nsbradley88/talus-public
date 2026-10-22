//! Engine trait, `EngineTimings`, `ScriptedMockEngine`, and `StdioSubprocessEngine`
//! implementing the Strata/Talus stdio engine protocol (`READY`, `GEN`, `PP`, `RESUME`, `T`, `DONE`, `STOP`, `QUIT`).

use crate::json::JsonValue;
use crate::tokenizer::{ByteTokenizer, ID_IM_END};
use std::collections::HashMap;
use std::fmt;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq)]
pub enum EngineError {
    InvalidRequest(String),
    EngineDied(String),
    EngineStarting(String),
    NonFinite(String),
    Cancelled,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InvalidRequest(s) => f.write_str(s),
            EngineError::EngineDied(s) => f.write_str(s),
            EngineError::EngineStarting(s) => f.write_str(s),
            EngineError::NonFinite(s) => f.write_str(s),
            EngineError::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for EngineError {}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
    pub repeat_penalty: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub penalty_last_n: i64,
    pub seed: Option<i64>,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
    pub cache_prompt: bool,
    pub checkpoint: bool,
    pub pin_prefix_tokens: Option<usize>,
    pub reasoning_budget_tokens: Option<i64>,
    pub reasoning_effort: Option<String>,
    pub enable_thinking: Option<bool>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            penalty_last_n: 64,
            seed: None,
            stop: Vec::new(),
            ignore_eos: false,
            cache_prompt: true,
            checkpoint: true,
            pin_prefix_tokens: None,
            reasoning_budget_tokens: None,
            reasoning_effort: None,
            enable_thinking: None,
        }
    }
}

impl SamplingParams {
    /// Format GEN sampling key=value tokens matching `StrataEngine.sampling_keys` in `serve/server.py`.
    pub fn to_gen_keys(&self) -> String {
        let mut keys = String::new();
        if self.temperature > 0.0 {
            keys.push_str(&format!(" temperature={}", self.temperature));
        }
        if self.top_p < 1.0 {
            keys.push_str(&format!(" top_p={}", self.top_p));
        }
        if self.top_k >= 0 {
            let clamped = if (1..=64).contains(&self.top_k) {
                self.top_k
            } else {
                64
            };
            keys.push_str(&format!(" top_k={clamped}"));
        }
        if self.min_p > 0.0 && self.min_p <= 1.0 {
            keys.push_str(&format!(" min_p={}", self.min_p));
        }
        let rp_on = (self.repeat_penalty - 1.0).abs() > f64::EPSILON;
        let pf_on = self.frequency_penalty.abs() > f64::EPSILON;
        let pp_on = self.presence_penalty.abs() > f64::EPSILON;
        if rp_on {
            keys.push_str(&format!(" penalty_repeat={}", self.repeat_penalty));
        }
        if pf_on {
            keys.push_str(&format!(" penalty_freq={}", self.frequency_penalty));
        }
        if pp_on {
            keys.push_str(&format!(" penalty_present={}", self.presence_penalty));
        }
        if rp_on || pf_on || pp_on {
            let pln = if self.penalty_last_n > 0 {
                self.penalty_last_n
            } else {
                64
            };
            keys.push_str(&format!(" penalty_last_n={pln}"));
        }
        if let Some(seed) = self.seed {
            if seed > 0 {
                keys.push_str(&format!(" seed={seed}"));
            }
        }
        if !self.checkpoint {
            keys.push_str(" ckpt=0");
        }
        if self.ignore_eos {
            keys.push_str(" ignore_eos=1");
        }
        if !self.cache_prompt {
            keys.push_str(" cache=0");
        }
        if let Some(pin) = self.pin_prefix_tokens {
            if pin > 0 {
                keys.push_str(&format!(" pin={pin}"));
            }
        }
        keys
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerateRequest {
    pub prompt_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
    pub stop_tokens: Vec<u32>,
    pub stop_strings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EngineTimings {
    pub cache_n: usize,
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub draft_n: Option<usize>,
    pub draft_accepted: Option<usize>,
    pub finish_reason: String,
}

impl Default for EngineTimings {
    fn default() -> Self {
        Self {
            cache_n: 0,
            prompt_n: 0,
            prompt_ms: 1.0,
            predicted_n: 0,
            predicted_ms: 1.0,
            draft_n: None,
            draft_accepted: None,
            finish_reason: "stop".to_string(),
        }
    }
}

impl EngineTimings {
    pub fn round1(v: f64) -> f64 {
        (v * 10.0).round() / 10.0
    }

    pub fn round3(v: f64) -> f64 {
        (v * 1000.0).round() / 1000.0
    }

    /// Format into the `timings` JSON object matching `request_timings` in `serve/server.py`.
    pub fn to_json(&self, prompt_tokens: usize, completion_tokens: usize) -> JsonValue {
        let reused = self.cache_n.min(prompt_tokens);
        let p_n = if self.prompt_n > 0 {
            self.prompt_n
        } else {
            prompt_tokens.saturating_sub(reused)
        };
        let p_ms = Self::round1(self.prompt_ms);
        let d_n = if completion_tokens > 0 {
            completion_tokens
        } else {
            self.predicted_n
        };
        let d_ms = Self::round1(self.predicted_ms);
        let p_per_tok = if p_n > 0 {
            Self::round3(p_ms / p_n as f64)
        } else {
            0.0
        };
        let p_per_sec = if p_n > 0 && p_ms > 0.0 {
            Self::round1(p_n as f64 * 1000.0 / p_ms)
        } else {
            0.0
        };
        let d_per_tok = if d_n > 0 {
            Self::round3(d_ms / d_n as f64)
        } else {
            0.0
        };
        let d_per_sec = if d_n > 0 && d_ms > 0.0 {
            Self::round1(d_n as f64 * 1000.0 / d_ms)
        } else {
            0.0
        };
        let mut obj = JsonValue::obj(vec![
            ("cache_n", JsonValue::from(reused)),
            ("prompt_n", JsonValue::from(p_n)),
            ("prompt_ms", JsonValue::from(p_ms)),
            ("prompt_per_token_ms", JsonValue::from(p_per_tok)),
            ("prompt_per_second", JsonValue::from(p_per_sec)),
            ("predicted_n", JsonValue::from(d_n)),
            ("predicted_ms", JsonValue::from(d_ms)),
            ("predicted_per_token_ms", JsonValue::from(d_per_tok)),
            ("predicted_per_second", JsonValue::from(d_per_sec)),
        ]);
        if let Some(draft_n) = self.draft_n {
            obj.insert("draft_n", JsonValue::from(draft_n));
            obj.insert(
                "draft_n_accepted",
                JsonValue::from(self.draft_accepted.unwrap_or(0)),
            );
        }
        obj
    }
}

/// Core `Engine` trait for `talus-server`.
pub trait Engine: Send + Sync {
    fn max_context(&self) -> usize;

    fn is_loaded(&self) -> bool {
        true
    }

    fn total_slots(&self) -> usize {
        1
    }

    fn model_path(&self) -> Option<&str> {
        None
    }

    fn build_version(&self) -> Option<&str> {
        Some("0.1.44")
    }

    fn load(&self) -> Result<(), EngineError> {
        Ok(())
    }

    fn unload(&self) -> Result<(), EngineError> {
        Ok(())
    }

    /// Stop the engine for good and wait until it has exited (its memory is free), before the server replaces
    /// itself with a new run config (talus-tui's apply). The default does nothing.
    fn shutdown(&self) {}

    /// Generate tokens for `req`, invoking `on_progress(read_tokens, total_tokens)` during prompt read
    /// and `on_token(token_id)` for each generated token (returning `false` stops early).
    fn generate(
        &self,
        req: &GenerateRequest,
        cancel: Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32) -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<EngineTimings, EngineError>;

    fn tokenize(&self, text: &str, parse_special: bool) -> Vec<u32>;

    fn detokenize(&self, ids: &[u32]) -> String;

    fn cancel(&self);

    fn save_slot(&self, _slot_id: usize, _path: &Path) -> Result<(usize, usize, f64), EngineError> {
        Err(EngineError::InvalidRequest(
            "slot save/restore is not supported by this engine".to_string(),
        ))
    }

    fn restore_slot(
        &self,
        _slot_id: usize,
        _path: &Path,
    ) -> Result<(usize, usize, f64), EngineError> {
        Err(EngineError::InvalidRequest(
            "slot save/restore is not supported by this engine".to_string(),
        ))
    }

    fn erase_slot(&self, _slot_id: usize) -> Result<usize, EngineError> {
        Ok(0)
    }
}

#[derive(Clone, Debug)]
struct MockState {
    turn: usize,
    last_prompt: Vec<u32>,
    /// Multi-conversation prefix cache history (snapshots of prompt and prompt+reply).
    cache_snapshots: Vec<Vec<u32>>,
    saved_slots: HashMap<PathBuf, Vec<u32>>,
}

/// Configurable in-process mock engine (`ScriptedMockEngine`) supporting:
/// - Multi-turn scripted token streams (each request gets the next script and the last repeats, as in `serve/server.py`)
/// - Tool-call responses (Qwen XML, JSON, and GLM `<arg_key>` formats)
/// - `<think>...</think>` reasoning responses
/// - Prompt cache hits (`cache_n` via prefix matching across cached conversation snapshots or explicit override)
/// - Echoed prompts (`echo_prompt`)
/// - `ignore_eos` (runs through `<|im_end|>` up to `max_tokens`)
/// - `cache_prompt: false` (forces `cache_n = 0`)
/// - Speculative draft counts (`draft_n`, `draft_accepted`)
/// - Full stdio engine protocol simulation (`run_stdio_loop`: `INFO`, `READY`, `GEN`, `PP`, `RESUME`, `T`, `DONE`, `STOP`, `QUIT`).
pub struct ScriptedMockEngine {
    tok: ByteTokenizer,
    max_context: AtomicUsize,
    loaded: AtomicBool,
    scripts: Mutex<Vec<Vec<u32>>>,
    raw_scripts: Mutex<Vec<String>>,
    echo_prompt: AtomicBool,
    fixed_cache_n: Mutex<Option<usize>>,
    fixed_drafts: Mutex<Option<(usize, usize)>>,
    delay: Duration,
    cancelled: AtomicBool,
    state: Mutex<MockState>,
}

impl Default for ScriptedMockEngine {
    fn default() -> Self {
        Self::new(ByteTokenizer::new())
    }
}

impl ScriptedMockEngine {
    pub fn new(tok: ByteTokenizer) -> Self {
        Self::with_tokenizer(
            tok,
            vec!["Thinking about it.</think>\n\nHello from the mock engine.".to_string()],
            32768,
        )
    }

    pub fn with_scripts(scripts: Vec<String>) -> Self {
        Self::with_tokenizer(ByteTokenizer::new(), scripts, 32768)
    }

    pub fn with_tokenizer(tok: ByteTokenizer, scripts: Vec<String>, max_context: usize) -> Self {
        let scripts = if scripts.is_empty() {
            vec!["Thinking about it.</think>\n\nHello from the mock engine.".to_string()]
        } else {
            scripts
        };
        let end = tok.encode("<|im_end|>", true);
        let encoded = scripts
            .iter()
            .map(|s| {
                let mut ids = tok.encode(s, true);
                ids.extend_from_slice(&end);
                ids
            })
            .collect();
        Self {
            tok,
            max_context: AtomicUsize::new(max_context),
            loaded: AtomicBool::new(true),
            scripts: Mutex::new(encoded),
            raw_scripts: Mutex::new(scripts),
            echo_prompt: AtomicBool::new(false),
            fixed_cache_n: Mutex::new(None),
            fixed_drafts: Mutex::new(None),
            delay: Duration::ZERO,
            cancelled: AtomicBool::new(false),
            state: Mutex::new(MockState {
                turn: 0,
                last_prompt: Vec::new(),
                cache_snapshots: Vec::new(),
                saved_slots: HashMap::new(),
            }),
        }
    }

    pub fn set_max_context(&self, ctx: usize) {
        self.max_context.store(ctx.max(1), Ordering::Relaxed);
    }

    pub fn set_scripts(&self, scripts: Vec<String>) {
        let end = self.tok.encode("<|im_end|>", true);
        let encoded = scripts
            .iter()
            .map(|s| {
                let mut ids = self.tok.encode(s, true);
                ids.extend_from_slice(&end);
                ids
            })
            .collect();
        *self.scripts.lock().unwrap() = encoded;
        *self.raw_scripts.lock().unwrap() = scripts;
        self.state.lock().unwrap().turn = 0;
    }

    pub fn set_echo_prompt(&self, echo: bool) {
        self.echo_prompt.store(echo, Ordering::Relaxed);
    }

    pub fn set_fixed_cache_n(&self, cache_n: Option<usize>) {
        *self.fixed_cache_n.lock().unwrap() = cache_n;
    }

    pub fn set_fixed_drafts(&self, drafts: Option<(usize, usize)>) {
        *self.fixed_drafts.lock().unwrap() = drafts;
    }

    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    pub fn last_prompt(&self) -> Vec<u32> {
        self.state.lock().unwrap().last_prompt.clone()
    }

    fn compute_reused(prompt_ids: &[u32], snapshots: &[Vec<u32>]) -> usize {
        if prompt_ids.len() <= 1 {
            return 0;
        }
        let max_reused = prompt_ids.len() - 1;
        let mut best = 0usize;
        for snap in snapshots {
            let common = prompt_ids
                .iter()
                .zip(snap.iter())
                .take_while(|(a, b)| a == b)
                .count();
            best = best.max(common.min(max_reused));
        }
        best
    }

    fn record_snapshot(state: &mut MockState, seq: Vec<u32>) {
        if seq.is_empty() {
            return;
        }
        if !state.cache_snapshots.iter().any(|s| s == &seq) {
            if state.cache_snapshots.len() >= 32 {
                state.cache_snapshots.remove(0);
            }
            state.cache_snapshots.push(seq);
        }
    }

    /// Run a Strata/Talus stdio engine protocol loop over `reader` and `writer`.
    /// Useful both for testing `StdioSubprocessEngine` and for running `talus-server --mock-stdio-worker`
    /// as the backend engine executable for `serve/server.py --engine strata`!
    pub fn run_stdio_loop<R: BufRead, W: Write>(
        &self,
        mut reader: R,
        mut writer: W,
    ) -> std::io::Result<()> {
        let ctx = self.max_context();
        writeln!(
            writer,
            "INFO version=0.1.44 arena_mib=1024 vram_free_mib=8192 slot_cache=1"
        )?;
        writeln!(writer, "READY {ctx} stop")?;
        writer.flush()?;

        let mut line = String::new();
        while reader.read_line(&mut line)? > 0 {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                line.clear();
                continue;
            }
            if trimmed == "QUIT" {
                break;
            }
            if trimmed == "STOP" {
                self.cancel();
                line.clear();
                continue;
            }
            if trimmed == "CLEAR" {
                let n = self.erase_slot(0).unwrap_or(0);
                writeln!(writer, "CLEARED {n}")?;
                writer.flush()?;
                line.clear();
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("SAVE ") {
                let path = Path::new(rest.trim());
                match self.save_slot(0, path) {
                    Ok((n, _, _)) => writeln!(writer, "SAVED {n}")?,
                    Err(e) => writeln!(writer, "ERR {e}")?,
                }
                writer.flush()?;
                line.clear();
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("RESTORE ") {
                let path = Path::new(rest.trim());
                match self.restore_slot(0, path) {
                    Ok((n, _, _)) => writeln!(writer, "RESTORED {n}")?,
                    Err(e) => writeln!(writer, "ERR {e}")?,
                }
                writer.flush()?;
                line.clear();
                continue;
            }
            if trimmed.starts_with("VRAM") {
                writeln!(
                    writer,
                    "VRAM free_mib=8192 target_mib=700 released_mib=0 evicted=0"
                )?;
                writer.flush()?;
                line.clear();
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("GEN ") {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if parts.len() < 2 {
                    writeln!(writer, "ERR malformed GEN command")?;
                    writer.flush()?;
                    line.clear();
                    continue;
                }
                let max_new_tokens = parts[0].parse::<usize>().unwrap_or(64);
                let mut sampling = SamplingParams::default();
                for &kv in &parts[1..parts.len() - 1] {
                    if let Some((k, v)) = kv.split_once('=') {
                        match k {
                            "temperature" => {
                                if let Ok(val) = v.parse() {
                                    sampling.temperature = val;
                                }
                            }
                            "top_p" => {
                                if let Ok(val) = v.parse() {
                                    sampling.top_p = val;
                                }
                            }
                            "top_k" => {
                                if let Ok(val) = v.parse() {
                                    sampling.top_k = val;
                                }
                            }
                            "min_p" => {
                                if let Ok(val) = v.parse() {
                                    sampling.min_p = val;
                                }
                            }
                            "penalty_repeat" => {
                                if let Ok(val) = v.parse() {
                                    sampling.repeat_penalty = val;
                                }
                            }
                            "penalty_freq" => {
                                if let Ok(val) = v.parse() {
                                    sampling.frequency_penalty = val;
                                }
                            }
                            "penalty_present" => {
                                if let Ok(val) = v.parse() {
                                    sampling.presence_penalty = val;
                                }
                            }
                            "penalty_last_n" => {
                                if let Ok(val) = v.parse() {
                                    sampling.penalty_last_n = val;
                                }
                            }
                            "seed" => sampling.seed = v.parse().ok(),
                            "ignore_eos" => sampling.ignore_eos = v == "1" || v == "true",
                            "cache" => sampling.cache_prompt = !(v == "0" || v == "false"),
                            "ckpt" => sampling.checkpoint = !(v == "0" || v == "false"),
                            "pin" => sampling.pin_prefix_tokens = v.parse().ok(),
                            _ => {}
                        }
                    }
                }
                let ids_str = parts[parts.len() - 1];
                let prompt_ids: Vec<u32> = ids_str
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse::<u32>().ok())
                    .collect();
                let req = GenerateRequest {
                    prompt_ids: prompt_ids.clone(),
                    max_new_tokens,
                    sampling,
                    stop_tokens: self.tok.stop_tokens.clone(),
                    stop_strings: Vec::new(),
                };
                let cancel = Arc::new(AtomicBool::new(false));
                let mut emitted_tokens = Vec::new();
                let res = self.generate(
                    &req,
                    cancel,
                    &mut |tok_id| {
                        emitted_tokens.push(tok_id);
                        true
                    },
                    &mut |_read, _total| {},
                );
                match res {
                    Ok(timings) => {
                        if timings.cache_n > 0 {
                            writeln!(writer, "RESUME {}", timings.cache_n)?;
                        }
                        writeln!(
                            writer,
                            "PP {} {} {:.1} 1000.0",
                            timings.prompt_n,
                            prompt_ids.len(),
                            timings.prompt_ms
                        )?;
                        for tok_id in emitted_tokens {
                            writeln!(writer, "T {tok_id}")?;
                        }
                        let draft_acc = timings.draft_accepted.unwrap_or(0);
                        let draft_off = timings.draft_n.unwrap_or(0);
                        writeln!(
                            writer,
                            "DONE {} {} {:.1} {:.1} {} {} {} {} 0 0 0 0 0.0 {}",
                            timings.predicted_n,
                            prompt_ids.len(),
                            timings.prompt_ms,
                            timings.predicted_ms,
                            timings.finish_reason,
                            draft_acc,
                            draft_off,
                            timings.cache_n,
                            timings.prompt_n
                        )?;
                        writer.flush()?;
                    }
                    Err(e) => {
                        writeln!(writer, "ERR {e}")?;
                        writer.flush()?;
                    }
                }
            }
            line.clear();
        }
        Ok(())
    }
}

impl Engine for ScriptedMockEngine {
    fn max_context(&self) -> usize {
        self.max_context.load(Ordering::Relaxed)
    }

    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::Relaxed)
    }

    fn load(&self) -> Result<(), EngineError> {
        self.loaded.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn unload(&self) -> Result<(), EngineError> {
        self.loaded.store(false, Ordering::Relaxed);
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerateRequest,
        cancel: Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32) -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<EngineTimings, EngineError> {
        self.load()?;
        self.cancelled.store(false, Ordering::Relaxed);
        let prompt_len = req.prompt_ids.len();

        let (script_tokens, reused) = {
            let scripts = self.scripts.lock().unwrap();
            let mut st = self.state.lock().unwrap();
            st.last_prompt = req.prompt_ids.clone();

            let fixed = *self.fixed_cache_n.lock().unwrap();
            let reused = if !req.sampling.cache_prompt {
                0
            } else if let Some(c) = fixed {
                c.min(prompt_len.saturating_sub(1))
            } else {
                Self::compute_reused(&req.prompt_ids, &st.cache_snapshots)
            };

            let tokens = if self.echo_prompt.load(Ordering::Relaxed) {
                let mut t = req.prompt_ids.clone();
                t.push(ID_IM_END);
                t
            } else {
                let idx = st.turn.min(scripts.len().saturating_sub(1));
                if scripts.len() > 1 {
                    st.turn += 1;
                }
                scripts.get(idx).cloned().unwrap_or_else(|| vec![ID_IM_END])
            };
            (tokens, reused)
        };

        let prompt_n = prompt_len.saturating_sub(reused);
        let prompt_ms = ((prompt_n as f64) * 0.05).max(1.0);
        on_progress(prompt_len, prompt_len);

        let mut predicted_n = 0usize;
        let mut finish_reason = "length".to_string();
        let mut generated_ids: Vec<u32> = Vec::new();
        let t0 = Instant::now();

        if req.sampling.ignore_eos {
            // Strip stop tokens and repeat filler tokens until `max_new_tokens` is reached.
            let non_eos: Vec<u32> = script_tokens
                .iter()
                .copied()
                .filter(|&id| !self.tok.is_stop_token(id) && !req.stop_tokens.contains(&id))
                .collect();
            let filler = if non_eos.is_empty() {
                vec![u32::from(b'x')]
            } else {
                non_eos
            };
            for i in 0..req.max_new_tokens {
                if cancel.load(Ordering::Relaxed) || self.cancelled.load(Ordering::Relaxed) {
                    finish_reason = "cancel".to_string();
                    break;
                }
                if !self.delay.is_zero() {
                    thread::sleep(self.delay);
                }
                let tok_id = filler[i % filler.len()];
                predicted_n += 1;
                generated_ids.push(tok_id);
                if !on_token(tok_id) {
                    finish_reason = "stop".to_string();
                    break;
                }
            }
        } else {
            for &tok_id in script_tokens.iter().take(req.max_new_tokens) {
                if cancel.load(Ordering::Relaxed) || self.cancelled.load(Ordering::Relaxed) {
                    finish_reason = "cancel".to_string();
                    break;
                }
                if !self.delay.is_zero() {
                    thread::sleep(self.delay);
                }
                if self.tok.is_stop_token(tok_id) || req.stop_tokens.contains(&tok_id) {
                    finish_reason = "stop".to_string();
                    break;
                }
                predicted_n += 1;
                generated_ids.push(tok_id);
                if !on_token(tok_id) {
                    finish_reason = "stop".to_string();
                    break;
                }
            }
        }

        if req.sampling.checkpoint {
            let mut st = self.state.lock().unwrap();
            // Save system-prompt boundary snapshot if `<|im_end|>\n` appears in prompt_ids
            if let Some(pos) = req.prompt_ids.iter().position(|&id| id == ID_IM_END) {
                let end_idx = (pos + 2).min(req.prompt_ids.len());
                Self::record_snapshot(&mut st, req.prompt_ids[..end_idx].to_vec());
            }
            if let Some(pin) = req.sampling.pin_prefix_tokens {
                if pin > 0 && pin <= req.prompt_ids.len() {
                    Self::record_snapshot(&mut st, req.prompt_ids[..pin].to_vec());
                }
            }
            Self::record_snapshot(&mut st, req.prompt_ids.clone());
            let mut full = req.prompt_ids.clone();
            full.extend_from_slice(&generated_ids);
            Self::record_snapshot(&mut st, full);
        }

        let elapsed_ms = (t0.elapsed().as_secs_f64() * 1000.0).max(1.0);
        let fixed_drafts = *self.fixed_drafts.lock().unwrap();
        let (draft_n, draft_accepted) = if let Some((dn, da)) = fixed_drafts {
            (Some(dn), Some(da))
        } else if predicted_n > 0 {
            let dn = (predicted_n / 2).max(1);
            let da = (dn * 2 / 3).max(1).min(dn);
            (Some(dn), Some(da))
        } else {
            (Some(0), Some(0))
        };

        Ok(EngineTimings {
            cache_n: reused,
            prompt_n,
            prompt_ms,
            predicted_n,
            predicted_ms: elapsed_ms,
            draft_n,
            draft_accepted,
            finish_reason,
        })
    }

    fn tokenize(&self, text: &str, parse_special: bool) -> Vec<u32> {
        self.tok.encode(text, parse_special)
    }

    fn detokenize(&self, ids: &[u32]) -> String {
        self.tok.decode(ids)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    fn save_slot(&self, _slot_id: usize, path: &Path) -> Result<(usize, usize, f64), EngineError> {
        let mut st = self.state.lock().unwrap();
        let tokens = st.last_prompt.clone();
        let count = tokens.len();
        let content: String = tokens
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path, content.as_bytes())
            .map_err(|e| EngineError::InvalidRequest(format!("cannot save slot: {e}")))?;
        let n_written = content.len();
        st.saved_slots.insert(path.to_path_buf(), tokens);
        Ok((count, n_written, 1.0))
    }

    fn restore_slot(
        &self,
        _slot_id: usize,
        path: &Path,
    ) -> Result<(usize, usize, f64), EngineError> {
        let raw = std::fs::read_to_string(path).map_err(|_| {
            EngineError::InvalidRequest(format!(
                "no saved session named {:?}",
                path.file_name().unwrap_or_default()
            ))
        })?;
        let n_read = raw.len();
        let tokens: Vec<u32> = raw
            .trim()
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse::<u32>().ok())
            .collect();
        let count = tokens.len();
        let mut st = self.state.lock().unwrap();
        st.last_prompt = tokens.clone();
        Self::record_snapshot(&mut st, tokens);
        Ok((count, n_read, 1.0))
    }

    fn erase_slot(&self, _slot_id: usize) -> Result<usize, EngineError> {
        let mut st = self.state.lock().unwrap();
        let n = st.cache_snapshots.len();
        st.cache_snapshots.clear();
        st.last_prompt.clear();
        Ok(n)
    }
}

struct SubprocessHandle {
    child: Child,
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: BufReader<ChildStdout>,
}

/// Subprocess engine that spawns an external executable speaking the Strata/Talus stdio protocol:
/// `INFO ...`, `READY <ctx> [stop]`, `GEN <max_new> [keys] <ids>`, `PP`, `RESUME`, `T`, `DONE`, `STOP`, `QUIT`.
pub struct StdioSubprocessEngine {
    exe: PathBuf,
    args: Vec<String>,
    env: Vec<(String, String)>,
    cwd: Option<PathBuf>,
    tok: ByteTokenizer,
    max_context: usize,
    stop_token: Option<u32>,
    info: HashMap<String, String>,
    proc: Mutex<SubprocessHandle>,
    stdin_cancel: Arc<Mutex<ChildStdin>>,
}

impl StdioSubprocessEngine {
    pub fn spawn(
        exe: impl AsRef<Path>,
        args: &[String],
        env: &[(String, String)],
        default_ctx: usize,
    ) -> Result<Self, EngineError> {
        Self::spawn_with_cwd(exe, args, env, None, default_ctx)
    }

    pub fn spawn_with_cwd(
        exe: impl AsRef<Path>,
        args: &[String],
        env: &[(String, String)],
        cwd: Option<&Path>,
        default_ctx: usize,
    ) -> Result<Self, EngineError> {
        let exe_path = exe.as_ref().to_path_buf();
        let mut cmd = Command::new(&exe_path);
        cmd.args(args);
        if !args
            .iter()
            .any(|a| a == "--serve" || a == "--mock-stdio-worker")
        {
            cmd.arg("--serve");
        }
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt;
            extern "C" {
                fn prctl(option: i32, arg2: usize, arg3: usize, arg4: usize, arg5: usize) -> i32;
            }
            const PR_SET_PDEATHSIG: i32 = 1;
            const SIGTERM: usize = 15;
            unsafe {
                cmd.pre_exec(|| {
                    let _ = prctl(PR_SET_PDEATHSIG, SIGTERM, 0, 0, 0);
                    Ok(())
                });
            }
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| EngineError::EngineDied(format!("failed to spawn {:?}: {e}", exe_path)))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| EngineError::EngineDied("missing child stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| EngineError::EngineDied("missing child stdout".to_string()))?;
        let mut reader = BufReader::new(stdout);

        let mut max_context = default_ctx.max(1);
        let mut stop_token = None;
        let mut info = HashMap::new();
        let mut line = String::new();
        loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| EngineError::EngineDied(format!("reading engine stdout: {e}")))?;
            if n == 0 {
                return Err(EngineError::EngineDied(
                    "engine exited before READY".to_string(),
                ));
            }
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("INFO ") {
                for kv in rest.split_whitespace() {
                    if let Some((k, v)) = kv.split_once('=') {
                        info.insert(k.to_string(), v.to_string());
                    }
                }
            } else if let Some(rest) = trimmed.strip_prefix("READY") {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if let Some(first) = parts.first() {
                    if let Ok(ctx) = first.parse::<usize>() {
                        max_context = ctx;
                    }
                }
                if let Some(second) = parts.get(1) {
                    if let Ok(st) = second.parse::<u32>() {
                        stop_token = Some(st);
                    }
                }
                break;
            } else if let Some(rest) = trimmed.strip_prefix("ERR") {
                return Err(EngineError::EngineDied(rest.trim().to_string()));
            }
        }

        let shared_stdin = Arc::new(Mutex::new(stdin));
        let mut tok = ByteTokenizer::new();
        tok.stop_tokens = stop_token.into_iter().collect();

        Ok(Self {
            exe: exe_path,
            args: args.to_vec(),
            env: env.to_vec(),
            cwd: cwd.map(|p| p.to_path_buf()),
            tok,
            max_context,
            stop_token,
            info,
            proc: Mutex::new(SubprocessHandle {
                child,
                stdin: shared_stdin.clone(),
                stdout: reader,
            }),
            stdin_cancel: shared_stdin,
        })
    }

    pub fn stop_token(&self) -> Option<u32> {
        self.stop_token
    }

    pub fn cwd(&self) -> Option<&Path> {
        self.cwd.as_deref()
    }

    pub fn info(&self) -> &HashMap<String, String> {
        &self.info
    }

    pub fn exe(&self) -> &Path {
        &self.exe
    }

    pub fn args(&self) -> &[String] {
        &self.args
    }

    pub fn env(&self) -> &[(String, String)] {
        &self.env
    }
}

impl Engine for StdioSubprocessEngine {
    fn max_context(&self) -> usize {
        self.max_context
    }

    fn shutdown(&self) {
        // engines that ignore QUIT while idle (GLM was seen to) are killed after a short grace; the GPU driver frees
        // their memory either way
        self.quit_and_wait(Duration::from_secs(15));
    }

    fn build_version(&self) -> Option<&str> {
        self.info.get("version").map(|s| s.as_str())
    }

    fn generate(
        &self,
        req: &GenerateRequest,
        cancel: Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32) -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<EngineTimings, EngineError> {
        let mut handle = self.proc.lock().unwrap();
        let keys = req.sampling.to_gen_keys();
        let csv_ids: String = req
            .prompt_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let cmd = format!("GEN {}{} {}\n", req.max_new_tokens, keys, csv_ids);
        {
            let mut stdin = handle.stdin.lock().unwrap();
            stdin
                .write_all(cmd.as_bytes())
                .and_then(|_| stdin.flush())
                .map_err(|e| EngineError::EngineDied(format!("engine stdin write failed: {e}")))?;
        }

        let mut reused = 0usize;
        let mut stop_sent = false;
        let mut line = String::new();

        loop {
            if cancel.load(Ordering::Relaxed) && !stop_sent {
                if let Ok(mut stdin) = handle.stdin.lock() {
                    let _ = stdin.write_all(b"STOP\n");
                    let _ = stdin.flush();
                }
                stop_sent = true;
            }
            line.clear();
            let n = handle
                .stdout
                .read_line(&mut line)
                .map_err(|e| EngineError::EngineDied(format!("engine stdout read failed: {e}")))?;
            if n == 0 {
                return Err(EngineError::EngineDied(
                    "the engine process ended unexpectedly".to_string(),
                ));
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("RESUME ") {
                reused = rest.trim().parse::<usize>().unwrap_or(0);
            } else if let Some(rest) = trimmed.strip_prefix("PP ") {
                let f: Vec<&str> = rest.split_whitespace().collect();
                if f.len() >= 2 {
                    let read = f[0].parse::<usize>().unwrap_or(0);
                    let total = f[1].parse::<usize>().unwrap_or(0);
                    on_progress(reused + read, total);
                }
            } else if let Some(rest) = trimmed.strip_prefix("T ") {
                if let Ok(tok_id) = rest.trim().parse::<u32>() {
                    if (self.tok.is_stop_token(tok_id) || req.stop_tokens.contains(&tok_id))
                        && !req.sampling.ignore_eos
                    {
                        if !stop_sent {
                            if let Ok(mut stdin) = handle.stdin.lock() {
                                let _ = stdin.write_all(b"STOP\n");
                                let _ = stdin.flush();
                            }
                            stop_sent = true;
                        }
                    } else if !stop_sent && !on_token(tok_id) {
                        if let Ok(mut stdin) = handle.stdin.lock() {
                            let _ = stdin.write_all(b"STOP\n");
                            let _ = stdin.flush();
                        }
                        stop_sent = true;
                    }
                }
            } else if let Some(rest) = trimmed.strip_prefix("DONE ") {
                let f: Vec<&str> = rest.split_whitespace().collect();
                if f.len() < 5 {
                    return Err(EngineError::EngineDied(format!(
                        "malformed DONE line: {trimmed}"
                    )));
                }
                let predicted_n = f[0].parse::<usize>().unwrap_or(0);
                let prompt_tokens = f[1].parse::<usize>().unwrap_or(req.prompt_ids.len());
                let prompt_ms = f[2].parse::<f64>().unwrap_or(0.0);
                let predicted_ms = f[3].parse::<f64>().unwrap_or(0.0);
                let finish = f[4].to_string();
                if finish == "nonfinite" {
                    return Err(EngineError::NonFinite(
                        "the engine encountered non-finite logits".to_string(),
                    ));
                }
                let (draft_accepted, draft_n, done_reused) = if f.len() >= 8 {
                    (
                        Some(f[5].parse::<usize>().unwrap_or(0)),
                        Some(f[6].parse::<usize>().unwrap_or(0)),
                        f[7].parse::<usize>().unwrap_or(reused),
                    )
                } else {
                    (None, None, reused)
                };
                let cache_n = done_reused.min(prompt_tokens);
                let prompt_n = prompt_tokens.saturating_sub(cache_n);
                return Ok(EngineTimings {
                    cache_n,
                    prompt_n,
                    prompt_ms,
                    predicted_n,
                    predicted_ms,
                    draft_n,
                    draft_accepted,
                    finish_reason: if stop_sent && finish == "cancel" {
                        if cancel.load(Ordering::Relaxed) {
                            "cancel".to_string()
                        } else {
                            "stop".to_string()
                        }
                    } else {
                        finish
                    },
                });
            } else if let Some(rest) = trimmed.strip_prefix("ERR") {
                return Err(EngineError::InvalidRequest(rest.trim().to_string()));
            }
        }
    }

    fn tokenize(&self, text: &str, parse_special: bool) -> Vec<u32> {
        self.tok.encode(text, parse_special)
    }

    fn detokenize(&self, ids: &[u32]) -> String {
        self.tok.decode(ids)
    }

    fn cancel(&self) {
        if let Ok(mut stdin) = self.stdin_cancel.lock() {
            let _ = stdin.write_all(b"STOP\n");
            let _ = stdin.flush();
        }
    }

    fn save_slot(&self, _slot_id: usize, path: &Path) -> Result<(usize, usize, f64), EngineError> {
        let mut handle = self.proc.lock().unwrap();
        let cmd = format!("SAVE {}\n", path.display());
        {
            let mut stdin = handle.stdin.lock().unwrap();
            stdin
                .write_all(cmd.as_bytes())
                .and_then(|_| stdin.flush())
                .map_err(|e| EngineError::EngineDied(format!("SAVE write failed: {e}")))?;
        }
        let mut line = String::new();
        handle
            .stdout
            .read_line(&mut line)
            .map_err(|e| EngineError::EngineDied(format!("SAVE read failed: {e}")))?;
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("SAVED") {
            let n = rest.trim().parse::<usize>().unwrap_or(0);
            let written = std::fs::metadata(path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);
            Ok((n, written, 1.0))
        } else if let Some(rest) = trimmed.strip_prefix("ERR") {
            Err(EngineError::InvalidRequest(rest.trim().to_string()))
        } else {
            Err(EngineError::EngineDied(format!(
                "unexpected SAVE reply: {trimmed}"
            )))
        }
    }

    fn restore_slot(
        &self,
        _slot_id: usize,
        path: &Path,
    ) -> Result<(usize, usize, f64), EngineError> {
        let mut handle = self.proc.lock().unwrap();
        let cmd = format!("RESTORE {}\n", path.display());
        {
            let mut stdin = handle.stdin.lock().unwrap();
            stdin
                .write_all(cmd.as_bytes())
                .and_then(|_| stdin.flush())
                .map_err(|e| EngineError::EngineDied(format!("RESTORE write failed: {e}")))?;
        }
        let mut line = String::new();
        handle
            .stdout
            .read_line(&mut line)
            .map_err(|e| EngineError::EngineDied(format!("RESTORE read failed: {e}")))?;
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("RESTORED") {
            let n = rest.trim().parse::<usize>().unwrap_or(0);
            let read_bytes = std::fs::metadata(path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);
            Ok((n, read_bytes, 1.0))
        } else if let Some(rest) = trimmed.strip_prefix("ERR") {
            Err(EngineError::InvalidRequest(rest.trim().to_string()))
        } else {
            Err(EngineError::EngineDied(format!(
                "unexpected RESTORE reply: {trimmed}"
            )))
        }
    }

    fn erase_slot(&self, _slot_id: usize) -> Result<usize, EngineError> {
        let mut handle = self.proc.lock().unwrap();
        {
            let mut stdin = handle.stdin.lock().unwrap();
            stdin
                .write_all(b"CLEAR\n")
                .and_then(|_| stdin.flush())
                .map_err(|e| EngineError::EngineDied(format!("CLEAR write failed: {e}")))?;
        }
        let mut line = String::new();
        handle
            .stdout
            .read_line(&mut line)
            .map_err(|e| EngineError::EngineDied(format!("CLEAR read failed: {e}")))?;
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("CLEARED") {
            Ok(rest.trim().parse::<usize>().unwrap_or(0))
        } else {
            Ok(0)
        }
    }
}

impl StdioSubprocessEngine {
    /// Send QUIT, wait up to `grace` for the engine to exit, then kill it. Idempotent.
    pub fn quit_and_wait(&self, grace: Duration) {
        if let Ok(mut handle) = self.proc.lock() {
            if let Ok(mut stdin) = handle.stdin.lock() {
                let _ = stdin.write_all(b"QUIT\n");
                let _ = stdin.flush();
            }
            let deadline = Instant::now() + grace;
            loop {
                match handle.child.try_wait() {
                    Ok(Some(_)) | Err(_) => return,
                    Ok(None) if Instant::now() >= deadline => break,
                    Ok(None) => std::thread::sleep(Duration::from_millis(100)),
                }
            }
            let _ = handle.child.kill();
            let _ = handle.child.wait();
        }
    }
}

impl Drop for StdioSubprocessEngine {
    fn drop(&mut self) {
        if let Ok(mut handle) = self.proc.lock() {
            if let Ok(mut stdin) = handle.stdin.lock() {
                let _ = stdin.write_all(b"QUIT\n");
                let _ = stdin.flush();
            }
            let _ = handle.child.kill();
            let _ = handle.child.wait();
        }
    }
}
