//! Request queue, single-sequence executor, cancellation, and Slot abstraction.
//!
//! Implements the Talus single-sequence scheduler (`Slot`, `SlotState`, `RequestQueue`,
//! `CancelToken`) and the line-oriented stdio engine protocol (`GEN` / `STOP` ->
//! `READY` / `RESUME` / `PP` / `T` / `DONE` / `ERR`), matching `src/qwen35/q36_engine.cu`
//! (`serve`, lines 2617–2910).

use std::collections::VecDeque;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

pub const CRATE_NAME: &str = "talus-sched";

/// Qwen3.5 EOS token IDs (`<|endoftext|> = 248044`, `<|im_end|> = 248046`).
#[inline]
pub fn is_eos(t: i32) -> bool {
    t == 248044 || t == 248046
}

// ============================================================================
// Slot & SlotState
// ============================================================================

/// Execution phase of a sequence slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlotState {
    #[default]
    Idle,
    Prefilling,
    Decoding,
}

/// Single-sequence execution slot tracking live KV/recurrent token history and position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slot {
    pub id: usize,
    pub state: SlotState,
    pub live_tokens: Vec<i32>,
    pub n_past: usize,
    pub context_limit: usize,
}

impl Slot {
    /// Creates a new idle slot with the given `id` and `context_limit`.
    pub fn new(id: usize, context_limit: usize) -> Self {
        Self {
            id,
            state: SlotState::Idle,
            live_tokens: Vec::new(),
            n_past: 0,
            context_limit,
        }
    }

    /// Returns `true` if the slot is currently `Idle`.
    #[inline]
    pub fn is_idle(&self) -> bool {
        self.state == SlotState::Idle
    }

    /// Clears the slot's held token sequence and resets `n_past` to `0` and `state` to `Idle`.
    pub fn reset(&mut self) {
        self.state = SlotState::Idle;
        self.live_tokens.clear();
        self.n_past = 0;
    }

    /// Transitions the slot into the `Prefilling` state.
    #[inline]
    pub fn begin_prefill(&mut self) {
        self.state = SlotState::Prefilling;
    }

    /// Transitions the slot into the `Decoding` state.
    #[inline]
    pub fn begin_decode(&mut self) {
        self.state = SlotState::Decoding;
    }

    /// Marks the current generation complete and transitions back to `Idle` while keeping `live_tokens`.
    #[inline]
    pub fn finish(&mut self) {
        self.state = SlotState::Idle;
    }

    /// Appends a slice of tokens to `live_tokens` and advances `n_past`.
    pub fn append_tokens(&mut self, tokens: &[i32]) {
        self.live_tokens.extend_from_slice(tokens);
        self.n_past = self.live_tokens.len();
    }

    /// Appends a single token to `live_tokens` and advances `n_past`.
    pub fn push_token(&mut self, token: i32) {
        self.live_tokens.push(token);
        self.n_past = self.live_tokens.len();
    }

    /// Replaces the held prefix with `tokens` and sets `n_past = tokens.len()`.
    pub fn set_live_tokens(&mut self, tokens: Vec<i32>) {
        self.n_past = tokens.len();
        self.live_tokens = tokens;
    }

    /// Computes the longest common prefix length between `live_tokens` and `prompt`.
    pub fn lcp(&self, prompt: &[i32]) -> usize {
        self.live_tokens
            .iter()
            .zip(prompt.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Returns `Some(live_tokens.len())` when `prompt` strictly extends `live_tokens`
    /// (`live.size() < ids.size() && std::equal(live.begin(), live.end(), ids.begin())`).
    pub fn strict_prefix_reuse(&self, prompt: &[i32]) -> usize {
        if !self.live_tokens.is_empty()
            && self.live_tokens.len() < prompt.len()
            && prompt.starts_with(&self.live_tokens)
        {
            self.live_tokens.len()
        } else {
            0
        }
    }

    /// Remaining context capacity (`context_limit.saturating_sub(n_past)`).
    #[inline]
    pub fn remaining_context(&self) -> usize {
        self.context_limit.saturating_sub(self.n_past)
    }
}

// ============================================================================
// CancelToken & RequestQueue
// ============================================================================

/// Thread-safe cancellation token wrapping `Arc<AtomicBool>`.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    inner: Arc<AtomicBool>,
}

impl CancelToken {
    /// Creates a new uncancelled `CancelToken`.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Wraps an existing `Arc<AtomicBool>`.
    pub fn from_arc(inner: Arc<AtomicBool>) -> Self {
        Self { inner }
    }

    /// Returns a clone of the underlying `Arc<AtomicBool>`.
    pub fn as_arc(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.inner)
    }

    /// Signals cancellation (`store(true)`).
    #[inline]
    pub fn cancel(&self) {
        self.inner.store(true, Ordering::SeqCst);
    }

    /// Clears cancellation (`store(false)`).
    #[inline]
    pub fn reset(&self) {
        self.inner.store(false, Ordering::SeqCst);
    }

    /// Returns `true` if cancellation has been requested.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.inner.load(Ordering::SeqCst)
    }
}

/// Non-empty protocol lines queued and not yet taken, over every [`RequestQueue`] of the process.
static LINES_WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// True when a request line has arrived and waits for the engine thread (e.g. the next GEN arrived while the
/// engine still does after-reply work such as staging a prompt-cache disk write, which it can then skip).
pub fn requests_waiting() -> bool {
    LINES_WAITING.load(Ordering::Acquire) > 0
}

fn took_line(line: &Option<String>) {
    if line.as_deref().is_some_and(|l| !l.is_empty()) {
        // every non-empty line was counted when it was pushed
        LINES_WAITING.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Debug, Default)]
struct RequestQueueState {
    lines: VecDeque<String>,
    eof: bool,
    active_or_queued: usize,
}

/// Thread-safe request line queue with immediate `"STOP"` interception via `CancelToken`.
#[derive(Debug, Clone)]
pub struct RequestQueue {
    inner: Arc<(Mutex<RequestQueueState>, Condvar)>,
    cancel: CancelToken,
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new(CancelToken::new())
    }
}

impl RequestQueue {
    /// Creates a new `RequestQueue` bound to `cancel`.
    pub fn new(cancel: CancelToken) -> Self {
        Self {
            inner: Arc::new((Mutex::new(RequestQueueState::default()), Condvar::new())),
            cancel,
        }
    }

    /// Returns a reference to the associated `CancelToken`.
    #[inline]
    pub fn cancel_token(&self) -> &CancelToken {
        &self.cancel
    }

    /// Pushes an incoming protocol line.
    ///
    /// Strips trailing `\r`/`\n`. If the line is `"STOP"`, sets `CancelToken` immediately
    /// without queuing a command line (matching `q36_engine.cu:2629`).
    pub fn push_line(&self, raw_line: &str) {
        let line = raw_line.trim_end_matches(['\r', '\n']);
        if line == "STOP" {
            self.cancel.cancel();
            return;
        }
        let (mu, cv) = &*self.inner;
        let mut st = mu.lock().unwrap();
        if !line.is_empty() {
            if st.active_or_queued == 0 {
                self.cancel.reset();
            }
            st.active_or_queued += 1;
            LINES_WAITING.fetch_add(1, Ordering::AcqRel);
        }
        st.lines.push_back(line.to_string());
        cv.notify_one();
    }

    /// Marks the input stream as closed (EOF) and wakes any waiting consumer.
    pub fn mark_eof(&self) {
        let (mu, cv) = &*self.inner;
        let mut st = mu.lock().unwrap();
        st.eof = true;
        cv.notify_all();
    }

    /// Blocks until a line is available or EOF is reached with an empty queue.
    pub fn pop_line(&self) -> Option<String> {
        let (mu, cv) = &*self.inner;
        let mut st = mu.lock().unwrap();
        while !st.eof && st.lines.is_empty() {
            st = cv.wait(st).unwrap();
        }
        let line = st.lines.pop_front();
        took_line(&line);
        line
    }

    /// Attempts to pop a queued line without blocking.
    pub fn try_pop_line(&self) -> Option<String> {
        let (mu, _) = &*self.inner;
        let mut st = mu.lock().unwrap();
        let line = st.lines.pop_front();
        took_line(&line);
        line
    }

    /// Marks one non-empty dequeued request as completed and resets the cancel token for the next request.
    pub fn complete_request(&self) {
        let (mu, _) = &*self.inner;
        let mut st = mu.lock().unwrap();
        st.active_or_queued = st.active_or_queued.saturating_sub(1);
        self.cancel.reset();
    }

    /// Returns the number of queued lines.
    pub fn len(&self) -> usize {
        let (mu, _) = &*self.inner;
        mu.lock().unwrap().lines.len()
    }

    /// Returns `true` if no lines are currently queued.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================================
// GenCommand & parse_gen_line
// ============================================================================

/// Parsed `GEN <max_new> [key=value ...] <id,id,...>` command from the stdio engine protocol.
#[derive(Debug, Clone, PartialEq)]
pub struct GenCommand {
    pub max_new: i64,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub seed: u64,
    pub ignore_eos: bool,
    pub use_cache: bool,
    pub prompt_ids: Vec<i32>,
}

impl Default for GenCommand {
    fn default() -> Self {
        Self {
            max_new: 0,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            penalty_last_n: 64,
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            seed: 1,
            ignore_eos: false,
            use_cache: true,
            prompt_ids: Vec::new(),
        }
    }
}

impl GenCommand {
    /// Returns `true` if full host logits are needed for sampling or repetition penalties.
    #[inline]
    pub fn needs_logits(&self) -> bool {
        self.temperature > 0.0
            || self.penalty_repeat != 1.0
            || self.penalty_freq != 0.0
            || self.penalty_present != 0.0
    }

    /// Converts the command's sampling fields into `talus_spec::SampCfg`.
    pub fn to_samp_cfg(&self) -> talus_spec::SampCfg {
        talus_spec::SampCfg {
            temp: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: self.penalty_repeat,
            frequency_penalty: self.penalty_freq,
            presence_penalty: self.penalty_present,
            repeat_last_n: self.penalty_last_n,
            seed: self.seed,
            ..talus_spec::SampCfg::default()
        }
    }
}

/// Parses comma/space-separated base-10 token IDs, matching `parse_ids` in `q36_engine.cu:2599–2610`.
pub fn parse_ids_str(s: &str) -> Vec<i32> {
    let bytes = s.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    let mut out = Vec::new();

    while i < n {
        while i < n && (bytes[i] == b',' || bytes[i] == b' ') {
            i += 1;
        }
        if i >= n {
            break;
        }
        let start = i;
        if bytes[i] == b'+' || bytes[i] == b'-' {
            i += 1;
        }
        let digit_start = i;
        while i < n && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == digit_start {
            break;
        }
        if let Ok(val) = s[start..i].parse::<i64>() {
            out.push(val as i32);
        } else {
            break;
        }
    }

    out
}

fn parse_c_int64_prefix(s: &str) -> (i64, usize) {
    let bytes = s.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n && bytes[i] == b' ' {
        i += 1;
    }
    let start = i;
    if i < n && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    let digit_start = i;
    while i < n && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == digit_start {
        return (0, 0);
    }
    let val = s[start..i].parse::<i64>().unwrap_or(0);
    (val, i)
}

fn parse_c_i32(v: &str) -> i32 {
    let (val, _) = parse_c_int64_prefix(v.trim_start());
    val as i32
}

fn parse_c_u64(v: &str) -> u64 {
    let trimmed = v.trim_start();
    let end = trimmed
        .bytes()
        .take_while(|b| b.is_ascii_digit() || *b == b'+')
        .count();
    if end == 0 {
        0
    } else {
        trimmed[..end].parse::<u64>().unwrap_or(0)
    }
}

fn parse_c_f32(v: &str) -> f32 {
    let trimmed = v.trim();
    if let Ok(x) = trimmed.parse::<f32>() {
        return x;
    }
    let end = trimmed
        .bytes()
        .take_while(|b| b.is_ascii_digit() || matches!(*b, b'+' | b'-' | b'.' | b'e' | b'E'))
        .count();
    if end == 0 {
        0.0
    } else {
        trimmed[..end].parse::<f32>().unwrap_or(0.0)
    }
}

/// Parses a `GEN <max_new> [key=value ...] <id,id,...>` line, matching `q36_engine.cu:2656–2692`.
pub fn parse_gen_line(line: &str) -> Result<GenCommand, String> {
    let line = line.trim_end_matches(['\r', '\n']);
    let Some(after_gen) = line.strip_prefix("GEN ") else {
        return Err("ERR expected: GEN <max_new> [key=value ...] <id,id,...>".into());
    };

    let (max_new, consumed) = parse_c_int64_prefix(after_gen);
    let mut cmd = GenCommand {
        max_new,
        ..GenCommand::default()
    };

    let rest = &after_gen[consumed..];
    let bytes = rest.as_bytes();
    let n = bytes.len();
    let mut pos = 0;

    loop {
        while pos < n && bytes[pos] == b' ' {
            pos += 1;
        }
        let st0 = pos;
        while pos < n && bytes[pos] != b' ' {
            pos += 1;
        }
        let tk = &rest[st0..pos];
        let Some(eq) = tk.find('=') else {
            pos = st0;
            break;
        };
        if tk.is_empty() {
            pos = st0;
            break;
        }
        let key = &tk[..eq];
        let v = &tk[eq + 1..];
        match key {
            "temperature" => cmd.temperature = parse_c_f32(v),
            "top_p" => cmd.top_p = parse_c_f32(v),
            "top_k" => cmd.top_k = parse_c_i32(v),
            "min_p" => cmd.min_p = parse_c_f32(v),
            "penalty_last_n" => cmd.penalty_last_n = parse_c_i32(v),
            "penalty_repeat" => cmd.penalty_repeat = parse_c_f32(v),
            "penalty_freq" => cmd.penalty_freq = parse_c_f32(v),
            "penalty_present" => cmd.penalty_present = parse_c_f32(v),
            "seed" => cmd.seed = parse_c_u64(v),
            "ignore_eos" => cmd.ignore_eos = v.eq_ignore_ascii_case("true") || parse_c_i32(v) != 0,
            "cache" => cmd.use_cache = v.eq_ignore_ascii_case("true") || parse_c_i32(v) != 0,
            _ => {}
        }
    }

    cmd.prompt_ids = parse_ids_str(&rest[pos..]);
    if cmd.max_new < 1 || cmd.prompt_ids.is_empty() {
        return Err("ERR bad request: max_new or the ids".into());
    }

    Ok(cmd)
}

// ============================================================================
// Protocol output formatting helpers
// ============================================================================

/// Summary metrics emitted on the `DONE` protocol line (`q36_engine.cu:2922–2924`).
#[derive(Debug, Clone, PartialEq)]
pub struct DoneStats {
    pub produced: i64,
    pub prompt_tokens: usize,
    pub prompt_ms: f64,
    pub decode_ms: f64,
    pub finish: &'static str,
    pub accepted: i64,
    pub offered: i64,
    pub reuse: usize,
    pub fresh: i64,
}

/// Emits `READY <ctx> stop\n` and flushes `out`.
pub fn write_ready_line(out: &mut dyn Write, ctx: i32) -> std::io::Result<()> {
    writeln!(out, "READY {ctx} stop")?;
    out.flush()
}

/// Emits `RESUME <reuse>\n` and flushes `out`.
pub fn write_resume_line(out: &mut dyn Write, reuse: usize) -> std::io::Result<()> {
    writeln!(out, "RESUME {reuse}")?;
    out.flush()
}

/// Emits `PP <done> <fresh> <ms:.0> <tok_per_s:.1>\n` and flushes `out`.
pub fn write_pp_line(
    out: &mut dyn Write,
    done: i64,
    fresh: i64,
    elapsed_ms: f64,
) -> std::io::Result<()> {
    let tok_per_s = (done as f64) / (elapsed_ms / 1000.0).max(1e-9);
    writeln!(out, "PP {done} {fresh} {:.0} {:.1}", elapsed_ms, tok_per_s)?;
    out.flush()
}

/// Emits `T <tok>\n` and flushes `out`.
pub fn write_token_line(out: &mut dyn Write, token: i32) -> std::io::Result<()> {
    writeln!(out, "T {token}")?;
    out.flush()
}

/// Emits `DONE <produced> <prompt_tokens> <prompt_ms:.1> <decode_ms:.1> <finish> <accepted> <offered> <reuse> 0 0 0 0 0.0 <fresh> 0\n` and flushes `out`.
pub fn write_done_line(out: &mut dyn Write, stats: &DoneStats) -> std::io::Result<()> {
    writeln!(
        out,
        "DONE {} {} {:.1} {:.1} {} {} {} {} 0 0 0 0 0.0 {} 0",
        stats.produced,
        stats.prompt_tokens,
        stats.prompt_ms,
        stats.decode_ms,
        stats.finish,
        stats.accepted,
        stats.offered,
        stats.reuse,
        stats.fresh,
    )?;
    out.flush()
}

// ============================================================================
// EngineProtocolBackend & serve_stdio_with_io / serve_stdio
// ============================================================================

/// Backend interface for the Talus stdio engine protocol (`READY` / `GEN` / `STOP`).
pub trait EngineProtocolBackend {
    /// Maximum context length in tokens (`E.ctx`).
    fn max_context(&self) -> i32;

    /// Vocabulary size (`E.m->g.n_vocab`).
    fn n_vocab(&self) -> i32;

    /// Executes a validated `GEN` command, streaming `RESUME`, `PP`, `T`, and `DONE` lines to `out`.
    fn handle_gen(
        &mut self,
        cmd: &GenCommand,
        stop: &CancelToken,
        out: &mut dyn std::io::Write,
    ) -> std::io::Result<()>;
}

/// Runs the stdio engine protocol loop over arbitrary `reader` and `out` streams.
///
/// Matches `serve()` in `q36_engine.cu:2617–2910`:
/// 1. Emits `READY <ctx> stop\n` and flushes.
/// 2. Spawns a background reader thread that intercepts `"STOP"` and sets `CancelToken`
///    immediately while `handle_gen` is running, queuing all other lines on a
///    `Mutex<VecDeque<String>>` + `Condvar` (`RequestQueue`).
/// 3. Validates `GEN` syntax, `max_new >= 1`, non-empty `prompt_ids`, context budget
///    (`prompt_ids.len() + max_new <= ctx`), and token range (`0 <= t < n_vocab`).
pub fn serve_stdio_with_io(
    backend: &mut dyn EngineProtocolBackend,
    reader: impl BufRead + Send,
    out: &mut dyn Write,
) -> std::io::Result<i32> {
    let ctx = backend.max_context();
    let n_vocab = backend.n_vocab();
    write_ready_line(out, ctx)?;

    let stop = CancelToken::new();
    let queue = RequestQueue::new(stop.clone());
    let queue_for_reader = queue.clone();

    std::thread::scope(|s| -> std::io::Result<i32> {
        s.spawn(move || {
            let mut rdr = reader;
            let mut line = String::new();
            loop {
                line.clear();
                match rdr.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        queue_for_reader.push_line(&line);
                    }
                    Err(_) => break,
                }
            }
            queue_for_reader.mark_eof();
        });

        while let Some(line) = queue.pop_line() {
            if line.is_empty() {
                continue;
            }
            let cmd = match parse_gen_line(&line) {
                Ok(cmd) => cmd,
                Err(err_line) => {
                    writeln!(out, "{err_line}")?;
                    out.flush()?;
                    queue.complete_request();
                    continue;
                }
            };

            if (cmd.prompt_ids.len() as i64) + cmd.max_new > (ctx as i64) {
                writeln!(
                    out,
                    "ERR prompt ({} tokens) + max_new ({}) exceeds the context ({})",
                    cmd.prompt_ids.len(),
                    cmd.max_new,
                    ctx
                )?;
                out.flush()?;
                queue.complete_request();
                continue;
            }

            let bad_vocab = cmd.prompt_ids.iter().any(|&t| t < 0 || t >= n_vocab);
            if bad_vocab {
                writeln!(out, "ERR a token id is outside the vocabulary")?;
                out.flush()?;
                queue.complete_request();
                continue;
            }

            let res = backend.handle_gen(&cmd, &stop, out);
            queue.complete_request();
            res?;
        }

        Ok(0)
    })
}

/// Runs the stdio engine protocol loop over standard input and standard output.
///
/// Note: Like `q36_engine.cu:2637`, the stdin reader thread is detached so that when the
/// queued command stream finishes on EOF or error, the process can exit cleanly.
pub fn serve_stdio(backend: &mut dyn EngineProtocolBackend) -> i32 {
    let ctx = backend.max_context();
    let n_vocab = backend.n_vocab();
    let mut stdout = std::io::stdout();
    if write_ready_line(&mut stdout, ctx).is_err() {
        return 1;
    }

    let stop = CancelToken::new();
    let queue = RequestQueue::new(stop.clone());
    let queue_for_reader = queue.clone();

    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut rdr = stdin.lock();
        let mut line = String::new();
        loop {
            line.clear();
            match rdr.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    queue_for_reader.push_line(&line);
                }
                Err(_) => break,
            }
        }
        queue_for_reader.mark_eof();
    });

    while let Some(line) = queue.pop_line() {
        if line.is_empty() {
            continue;
        }
        let cmd = match parse_gen_line(&line) {
            Ok(cmd) => cmd,
            Err(err_line) => {
                let _ = writeln!(stdout, "{err_line}");
                let _ = stdout.flush();
                queue.complete_request();
                continue;
            }
        };

        if (cmd.prompt_ids.len() as i64) + cmd.max_new > (ctx as i64) {
            let _ = writeln!(
                stdout,
                "ERR prompt ({} tokens) + max_new ({}) exceeds the context ({})",
                cmd.prompt_ids.len(),
                cmd.max_new,
                ctx
            );
            let _ = stdout.flush();
            queue.complete_request();
            continue;
        }

        if cmd.prompt_ids.iter().any(|&t| t < 0 || t >= n_vocab) {
            let _ = writeln!(stdout, "ERR a token id is outside the vocabulary");
            let _ = stdout.flush();
            queue.complete_request();
            continue;
        }

        let res = backend.handle_gen(&cmd, &stop, &mut stdout);
        queue.complete_request();
        if res.is_err() {
            return 1;
        }
    }

    0
}

// ============================================================================
// ScriptedProtocolBackend (mock backend for CPU testing & protocol verification)
// ============================================================================

/// Mock implementation of [`EngineProtocolBackend`] for testing the `GEN`/`STOP` protocol,
/// `Slot` prefix reuse (`RESUME`), `PP` prefill reporting, `T` token streaming, EOS stopping,
/// and mid-generation `STOP` cancellation (`finish = "cancel"`).
pub struct ScriptedProtocolBackend {
    pub max_ctx: i32,
    pub vocab_size: i32,
    pub slot: Slot,
    pub tokens_to_emit: Vec<i32>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub accepted_drafts: i64,
    pub offered_drafts: i64,
    pub cancel_after_tokens: Option<usize>,
    pub step_delay: Option<std::time::Duration>,
    pub recorded_commands: Vec<GenCommand>,
}

impl std::fmt::Debug for ScriptedProtocolBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedProtocolBackend")
            .field("max_ctx", &self.max_ctx)
            .field("vocab_size", &self.vocab_size)
            .field("slot", &self.slot)
            .field("tokens_to_emit", &self.tokens_to_emit)
            .field("recorded_commands", &self.recorded_commands.len())
            .finish()
    }
}

impl Default for ScriptedProtocolBackend {
    fn default() -> Self {
        Self::new(32768, 248320)
    }
}

impl ScriptedProtocolBackend {
    /// Creates a new `ScriptedProtocolBackend` with the given `max_ctx` and `vocab_size`.
    pub fn new(max_ctx: i32, vocab_size: i32) -> Self {
        Self {
            max_ctx,
            vocab_size,
            slot: Slot::new(0, max_ctx.max(0) as usize),
            tokens_to_emit: vec![101, 102, 103, 104],
            prefill_ms: 12.0,
            decode_ms: 25.0,
            accepted_drafts: 0,
            offered_drafts: 0,
            cancel_after_tokens: None,
            step_delay: None,
            recorded_commands: Vec::new(),
        }
    }

    /// Builder helper to configure scripted output token IDs.
    pub fn with_tokens(mut self, tokens: Vec<i32>) -> Self {
        self.tokens_to_emit = tokens;
        self
    }

    /// Builder helper to simulate mid-generation `STOP` after `n` emitted tokens.
    pub fn with_cancel_after(mut self, n: usize) -> Self {
        self.cancel_after_tokens = Some(n);
        self
    }
}

impl EngineProtocolBackend for ScriptedProtocolBackend {
    fn max_context(&self) -> i32 {
        self.max_ctx
    }

    fn n_vocab(&self) -> i32 {
        self.vocab_size
    }

    fn handle_gen(
        &mut self,
        cmd: &GenCommand,
        stop: &CancelToken,
        out: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        self.recorded_commands.push(cmd.clone());

        let reuse = if !cmd.use_cache {
            self.slot.reset();
            0
        } else {
            let r = self.slot.strict_prefix_reuse(&cmd.prompt_ids);
            if r == 0 {
                self.slot.reset();
            }
            r
        };

        if reuse > 0 {
            write_resume_line(out, reuse)?;
        }

        let fresh = (cmd.prompt_ids.len() - reuse) as i64;
        self.slot.begin_prefill();
        if fresh > 0 {
            write_pp_line(out, fresh, fresh, self.prefill_ms)?;
            self.slot.append_tokens(&cmd.prompt_ids[reuse..]);
        }

        self.slot.begin_decode();
        let mut produced: i64 = 0;
        let mut finish: &'static str = "length";

        while produced < cmd.max_new {
            let idx = (produced as usize) % self.tokens_to_emit.len().max(1);
            let tok = self
                .tokens_to_emit
                .get(idx)
                .copied()
                .unwrap_or(100 + produced as i32);

            write_token_line(out, tok)?;
            produced += 1;
            self.slot.push_token(tok);

            if let Some(delay) = self.step_delay {
                std::thread::sleep(delay);
            }

            if self.cancel_after_tokens == Some(produced as usize) {
                stop.cancel();
            }

            if is_eos(tok) && !cmd.ignore_eos {
                finish = "stop";
                break;
            }
            if stop.is_cancelled() {
                finish = "cancel";
                break;
            }
            if produced >= cmd.max_new {
                finish = "length";
                break;
            }
        }

        self.slot.finish();

        write_done_line(
            out,
            &DoneStats {
                produced,
                prompt_tokens: cmd.prompt_ids.len(),
                prompt_ms: self.prefill_ms,
                decode_ms: self.decode_ms,
                finish,
                accepted: self.accepted_drafts,
                offered: self.offered_drafts,
                reuse,
                fresh,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn crate_identity() {
        assert_eq!(CRATE_NAME, "talus-sched");
    }

    #[test]
    fn slot_lifecycle_and_prefix_reuse() {
        let mut slot = Slot::new(0, 4096);
        assert!(slot.is_idle());
        assert_eq!(slot.remaining_context(), 4096);

        slot.begin_prefill();
        assert_eq!(slot.state, SlotState::Prefilling);
        slot.append_tokens(&[10, 20, 30]);
        assert_eq!(slot.n_past, 3);
        assert_eq!(slot.live_tokens, vec![10, 20, 30]);

        slot.begin_decode();
        assert_eq!(slot.state, SlotState::Decoding);
        slot.push_token(40);
        slot.finish();
        assert!(slot.is_idle());
        assert_eq!(slot.n_past, 4);

        // Strict prefix extension reuses all 4 live tokens
        assert_eq!(slot.strict_prefix_reuse(&[10, 20, 30, 40, 50, 60]), 4);
        // Identical length is not a strict extension (fresh must be >= 1)
        assert_eq!(slot.strict_prefix_reuse(&[10, 20, 30, 40]), 0);
        // Divergent prefix has LCP 2 but strict_prefix_reuse 0
        assert_eq!(slot.lcp(&[10, 20, 99, 100]), 2);
        assert_eq!(slot.strict_prefix_reuse(&[10, 20, 99, 100]), 0);

        slot.reset();
        assert!(slot.live_tokens.is_empty());
        assert_eq!(slot.n_past, 0);
        assert!(slot.is_idle());
    }

    #[test]
    fn cancel_token_and_request_queue_stop_interception() {
        let cancel = CancelToken::new();
        assert!(!cancel.is_cancelled());
        let q = RequestQueue::new(cancel.clone());

        q.push_line("GEN 8 1,2,3\r\n");
        assert_eq!(q.len(), 1);
        assert!(!cancel.is_cancelled());

        // STOP must not be queued as a line, and must immediately trip CancelToken
        q.push_line("STOP\r\n");
        assert_eq!(q.len(), 1);
        assert!(cancel.is_cancelled());

        let popped = q.pop_line().unwrap();
        assert_eq!(popped, "GEN 8 1,2,3");
        q.complete_request();
        assert!(!cancel.is_cancelled());

        q.mark_eof();
        assert_eq!(q.pop_line(), None);
    }

    #[test]
    fn parse_gen_line_defaults_and_all_sampling_params() {
        let cmd = parse_gen_line("GEN 16 151644,872,198").unwrap();
        assert_eq!(cmd.max_new, 16);
        assert_eq!(cmd.temperature, 0.0);
        assert_eq!(cmd.top_p, 1.0);
        assert_eq!(cmd.top_k, 0);
        assert_eq!(cmd.min_p, 0.0);
        assert_eq!(cmd.penalty_last_n, 64);
        assert_eq!(cmd.penalty_repeat, 1.0);
        assert_eq!(cmd.penalty_freq, 0.0);
        assert_eq!(cmd.penalty_present, 0.0);
        assert_eq!(cmd.seed, 1);
        assert!(!cmd.ignore_eos);
        assert!(cmd.use_cache);
        assert_eq!(cmd.prompt_ids, vec![151644, 872, 198]);
        assert!(!cmd.needs_logits());

        let full = parse_gen_line(
            "GEN 128 temperature=0.7 top_p=0.9 top_k=40 min_p=0.05 penalty_last_n=128 penalty_repeat=1.15 penalty_freq=0.2 penalty_present=0.1 seed=42 ignore_eos=1 cache=0 10,20,30,40",
        )
        .unwrap();
        assert_eq!(full.max_new, 128);
        assert!((full.temperature - 0.7).abs() < 1e-6);
        assert!((full.top_p - 0.9).abs() < 1e-6);
        assert_eq!(full.top_k, 40);
        assert!((full.min_p - 0.05).abs() < 1e-6);
        assert_eq!(full.penalty_last_n, 128);
        assert!((full.penalty_repeat - 1.15).abs() < 1e-6);
        assert!((full.penalty_freq - 0.2).abs() < 1e-6);
        assert!((full.penalty_present - 0.1).abs() < 1e-6);
        assert_eq!(full.seed, 42);
        assert!(full.ignore_eos);
        assert!(!full.use_cache);
        assert_eq!(full.prompt_ids, vec![10, 20, 30, 40]);
        assert!(full.needs_logits());
    }

    #[test]
    fn parse_gen_line_errors() {
        assert_eq!(
            parse_gen_line("HELLO 10 1,2,3"),
            Err("ERR expected: GEN <max_new> [key=value ...] <id,id,...>".into())
        );
        assert_eq!(
            parse_gen_line("GEN 0 1,2,3"),
            Err("ERR bad request: max_new or the ids".into())
        );
        assert_eq!(
            parse_gen_line("GEN -5 1,2,3"),
            Err("ERR bad request: max_new or the ids".into())
        );
        assert_eq!(
            parse_gen_line("GEN 10 temperature=0.5"),
            Err("ERR bad request: max_new or the ids".into())
        );
    }

    #[test]
    fn serve_stdio_handshake_resume_pp_tokens_and_done() {
        let mut backend = ScriptedProtocolBackend::new(1024, 248320).with_tokens(vec![50, 51, 52]);
        let input = "\
GEN 2 temperature=0.0 cache=1 1,2,3\n\
GEN 2 cache=1 1,2,3,50,51,4,5\n";
        let mut out = Vec::new();
        let rc =
            serve_stdio_with_io(&mut backend, Cursor::new(input.as_bytes()), &mut out).unwrap();
        assert_eq!(rc, 0);

        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "READY 1024 stop");
        // Turn 1: fresh 3 tokens, no RESUME, 2 generated tokens (50, 51), DONE length
        assert_eq!(lines[1], "PP 3 3 12 250.0");
        assert_eq!(lines[2], "T 50");
        assert_eq!(lines[3], "T 51");
        assert_eq!(lines[4], "DONE 2 3 12.0 25.0 length 0 0 0 0 0 0 0 0.0 3 0");

        // Turn 2: extends [1, 2, 3, 50, 51] with [4, 5] -> RESUME 5, fresh 2
        assert_eq!(lines[5], "RESUME 5");
        assert_eq!(lines[6], "PP 2 2 12 166.7");
        assert_eq!(lines[7], "T 50");
        assert_eq!(lines[8], "T 51");
        assert_eq!(lines[9], "DONE 2 7 12.0 25.0 length 0 0 5 0 0 0 0 0.0 2 0");
    }

    #[test]
    fn serve_stdio_eos_and_ignore_eos() {
        // 248046 is <|im_end|>
        let mut backend =
            ScriptedProtocolBackend::new(1024, 248320).with_tokens(vec![100, 248046, 101]);
        let input = "\
GEN 5 ignore_eos=0 1,2,3\n\
GEN 3 ignore_eos=1 cache=0 1,2,3\n";
        let mut out = Vec::new();
        serve_stdio_with_io(&mut backend, Cursor::new(input.as_bytes()), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();

        // Request 1 stops at token 2 (248046) with finish = "stop"
        assert!(lines.contains(&"DONE 2 3 12.0 25.0 stop 0 0 0 0 0 0 0 0.0 3 0"));
        // Request 2 ignores EOS and produces all 3 tokens with finish = "length"
        assert!(lines.contains(&"DONE 3 3 12.0 25.0 length 0 0 0 0 0 0 0 0.0 3 0"));
    }

    #[test]
    fn serve_stdio_mid_generation_stop_cancellation() {
        let mut backend =
            ScriptedProtocolBackend::new(1024, 248320).with_tokens(vec![11, 12, 13, 14, 15]);
        // Sending STOP after GEN cancels the generation with finish = "cancel"
        let input = "\
GEN 10 1,2,3\n\
STOP\n\
GEN 2 cache=0 4,5\n";
        let mut out = Vec::new();
        serve_stdio_with_io(&mut backend, Cursor::new(input.as_bytes()), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();

        assert_eq!(lines[0], "READY 1024 stop");
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("DONE ") && l.contains(" cancel ")),
            "expected a cancelled DONE line, got: {lines:?}"
        );
        // Subsequent GEN after cancellation resets cancel state and finishes with "length"
        assert!(
            lines.last().unwrap().contains(" length "),
            "expected second GEN to complete with length, got: {lines:?}"
        );
    }

    #[test]
    fn serve_stdio_cancel_after_n_tokens_mid_decode() {
        let mut backend = ScriptedProtocolBackend::new(1024, 248320)
            .with_tokens(vec![21, 22, 23, 24, 25])
            .with_cancel_after(2);
        let input = "GEN 5 1,2,3\n";
        let mut out = Vec::new();
        serve_stdio_with_io(&mut backend, Cursor::new(input.as_bytes()), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();

        assert_eq!(
            lines,
            vec![
                "READY 1024 stop",
                "PP 3 3 12 250.0",
                "T 21",
                "T 22",
                "DONE 2 3 12.0 25.0 cancel 0 0 0 0 0 0 0 0.0 3 0",
            ]
        );
    }

    #[test]
    fn serve_stdio_error_handling() {
        let mut backend = ScriptedProtocolBackend::new(10, 1000);
        let input = "\
INVALID command\n\
GEN 0 1,2,3\n\
GEN 8 1,2,3,4\n\
GEN 2 1,2,1000\n\
GEN 2 1,2,-1\n";
        let mut out = Vec::new();
        serve_stdio_with_io(&mut backend, Cursor::new(input.as_bytes()), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();

        assert_eq!(
            lines,
            vec![
                "READY 10 stop",
                "ERR expected: GEN <max_new> [key=value ...] <id,id,...>",
                "ERR bad request: max_new or the ids",
                "ERR prompt (4 tokens) + max_new (8) exceeds the context (10)",
                "ERR a token id is outside the vocabulary",
                "ERR a token id is outside the vocabulary",
            ]
        );
    }
}
