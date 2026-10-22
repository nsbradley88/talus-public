//! ByteTokenizer, incremental Detokenizer, and streaming StopMatcher matching `serve/server.py`.

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const ENDOFTEXT: &str = "<|endoftext|>";
pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
pub const VISION_START: &str = "<|vision_start|>";
pub const VISION_END: &str = "<|vision_end|>";

pub const ID_ENDOFTEXT: u32 = 248044;
pub const ID_IM_START: u32 = 248045;
pub const ID_IM_END: u32 = 248046;
pub const ID_VISION_START: u32 = 248053;
pub const ID_VISION_END: u32 = 248054;
pub const ID_IMAGE_PAD: u32 = 248056;
pub const ID_THINK_OPEN: u32 = 248068;
pub const ID_THINK_CLOSE: u32 = 248069;

const DEFAULT_SPECIALS: &[(&str, u32)] = &[
    (IM_START, ID_IM_START),
    (IM_END, ID_IM_END),
    (ENDOFTEXT, ID_ENDOFTEXT),
    (THINK_OPEN, ID_THINK_OPEN),
    (THINK_CLOSE, ID_THINK_CLOSE),
    (IMAGE_PAD, ID_IMAGE_PAD),
    (VISION_START, ID_VISION_START),
    (VISION_END, ID_VISION_END),
];

use std::path::Path;
use std::sync::Arc;

/// Fallback byte-level tokenizer OR real BPE tokenizer (`talus_text::Tokenizer`) with Qwen/Strata/GLM/MiMo special tokens.
#[derive(Clone, Debug)]
pub struct ByteTokenizer {
    real: Option<Arc<talus_text::Tokenizer>>,
    specials: Vec<(String, u32)>,
    pub stop_tokens: Vec<u32>,
}

impl Default for ByteTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteTokenizer {
    pub fn new() -> Self {
        let specials = DEFAULT_SPECIALS
            .iter()
            .map(|(s, id)| ((*s).to_string(), *id))
            .collect();
        Self {
            real: None,
            specials,
            stop_tokens: vec![ID_IM_END, ID_ENDOFTEXT],
        }
    }

    pub fn from_real(tok: talus_text::Tokenizer, extra_stop_strings: &[String]) -> Self {
        let mut specials = Vec::new();
        for (i, s) in tok.tokens.iter().enumerate() {
            if matches!(tok.token_type.get(i).copied(), Some(3 | 4)) {
                specials.push((s.clone(), i as u32));
            }
        }
        let mut stop_tokens = Vec::new();
        if let Some(eos) = tok.eos_token_id() {
            stop_tokens.push(eos);
        }
        let default_stops: &[&str] = if extra_stop_strings.is_empty() {
            &[IM_END, ENDOFTEXT, "<|user|>", "<|observation|>"]
        } else {
            &[]
        };
        for &s in default_stops {
            if let Some(id) = tok.token_to_id(s) {
                if !stop_tokens.contains(&id) {
                    stop_tokens.push(id);
                }
            }
        }
        for s in extra_stop_strings {
            if let Some(id) = tok.token_to_id(s) {
                if !stop_tokens.contains(&id) {
                    stop_tokens.push(id);
                }
            } else {
                let ids = tok.encode(s, true);
                if ids.len() == 1 && !stop_tokens.contains(&ids[0]) {
                    stop_tokens.push(ids[0]);
                }
            }
        }
        Self {
            real: Some(Arc::new(tok)),
            specials,
            stop_tokens,
        }
    }

    pub fn load(path: &Path, extra_stop_strings: &[String]) -> Result<Self, String> {
        let tok = if path.is_dir() {
            talus_text::Tokenizer::from_dir(path)
                .map_err(|e| format!("failed to load tokenizer from dir {}: {e}", path.display()))?
        } else {
            talus_text::Tokenizer::from_gguf(path).map_err(|e| {
                format!("failed to load tokenizer from GGUF {}: {e}", path.display())
            })?
        };
        Ok(Self::from_real(tok, extra_stop_strings))
    }

    pub fn real_tokenizer(&self) -> Option<&talus_text::Tokenizer> {
        self.real.as_deref()
    }

    pub fn control_tokens(&self) -> Vec<String> {
        if let Some(ref real) = self.real {
            real.control_tokens()
                .into_iter()
                .map(ToString::to_string)
                .collect()
        } else {
            DEFAULT_SPECIALS
                .iter()
                .map(|(s, _)| (*s).to_string())
                .collect()
        }
    }

    pub fn with_stop_tokens(mut self, extra_stops: &[String]) -> Self {
        if let Some(ref real) = self.real {
            for s in extra_stops {
                if s.is_empty() {
                    continue;
                }
                if let Some(id) = real.token_to_id(s) {
                    if !self.stop_tokens.contains(&id) {
                        self.stop_tokens.push(id);
                    }
                } else {
                    let ids = real.encode(s, true);
                    if ids.len() == 1 && !self.stop_tokens.contains(&ids[0]) {
                        self.stop_tokens.push(ids[0]);
                    }
                }
            }
            return self;
        }
        let mut next_id = 248070u32;
        for s in extra_stops {
            if s.is_empty() {
                continue;
            }
            if let Some((_, id)) = self.specials.iter().find(|(tok, _)| tok == s) {
                if !self.stop_tokens.contains(id) {
                    self.stop_tokens.push(*id);
                }
            } else {
                let id = next_id;
                next_id += 1;
                self.specials.push((s.clone(), id));
                self.stop_tokens.push(id);
            }
        }
        self
    }

    pub fn stop_ids(&self) -> &[u32] {
        &self.stop_tokens
    }

    pub fn is_stop_token(&self, id: u32) -> bool {
        self.stop_tokens.contains(&id)
    }

    pub fn specials(&self) -> &[(String, u32)] {
        &self.specials
    }

    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        if let Some(ref real) = self.real {
            return real.encode(text, parse_special);
        }
        if !parse_special {
            return text.as_bytes().iter().map(|&b| u32::from(b)).collect();
        }
        self.encode_segment_special(text)
    }

    /// Encode `text` with `(start, end)` byte spans of user literal special tokens that must be
    /// encoded as plain BPE text rather than control/special token IDs.
    pub fn encode_with_plain_spans(
        &self,
        text: &str,
        parse_special: bool,
        plain: &[(usize, usize)],
    ) -> Vec<u32> {
        if let Some(ref real) = self.real {
            return real.encode_with_plain(text, parse_special, plain);
        }
        if plain.is_empty() {
            return self.encode(text, parse_special);
        }
        let mut out = Vec::new();
        let mut pos = 0usize;
        for &(a, b) in plain {
            if a > pos && a <= text.len() {
                out.extend(self.encode(&text[pos..a], parse_special));
            }
            if b > a && b <= text.len() {
                out.extend(text.as_bytes()[a..b].iter().map(|&byte| u32::from(byte)));
                pos = b;
            }
        }
        if pos < text.len() {
            out.extend(self.encode(&text[pos..], parse_special));
        }
        out
    }

    /// Encode `segments` produced by `unmark_think_literals` where `(segment, is_plain)` pairs
    /// mark user literal `<think>`/`<|im_end|>` spans as `is_plain = true`.
    pub fn encode_with_plain(&self, segments: &[(String, bool)]) -> Vec<u32> {
        if let Some(ref real) = self.real {
            let mut full = String::new();
            let mut spans = Vec::new();
            for (seg, is_plain) in segments {
                let start = full.len();
                full.push_str(seg);
                if *is_plain && !seg.is_empty() {
                    spans.push((start, full.len()));
                }
            }
            return real.encode_with_plain(&full, true, &spans);
        }
        let mut out = Vec::new();
        for (seg, is_plain) in segments {
            if *is_plain {
                out.extend(seg.as_bytes().iter().map(|&b| u32::from(b)));
            } else {
                out.extend(self.encode_segment_special(seg));
            }
        }
        out
    }

    fn encode_segment_special(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len());
        let bytes = text.as_bytes();
        let mut i = 0usize;
        'outer: while i < bytes.len() {
            if bytes[i] == b'<' {
                let rem = &text[i..];
                for (tok, id) in &self.specials {
                    if rem.starts_with(tok.as_str()) {
                        out.push(*id);
                        i += tok.len();
                        continue 'outer;
                    }
                }
            }
            out.push(u32::from(bytes[i]));
            i += 1;
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        if let Some(ref real) = self.real {
            return real.decode(ids, false);
        }
        let mut out = String::new();
        let mut buf: Vec<u8> = Vec::new();
        for &id in ids {
            if id < 256 {
                buf.push(id as u8);
            } else if let Some((tok, _)) = self.specials.iter().find(|(_, sid)| *sid == id) {
                if !buf.is_empty() {
                    out.push_str(&String::from_utf8_lossy(&buf));
                    buf.clear();
                }
                out.push_str(tok);
            }
        }
        if !buf.is_empty() {
            out.push_str(&String::from_utf8_lossy(&buf));
        }
        out
    }
}

/// Incremental UTF-8 detokenizer that skips control/stop tokens
/// and buffers incomplete multi-byte UTF-8 sequences across tokens, matching `Detokenizer` in `serve/server.py`.
#[derive(Clone, Debug, Default)]
pub struct Detokenizer {
    tok: ByteTokenizer,
    buf: Vec<u8>,
}

impl Detokenizer {
    pub fn new(tok: &ByteTokenizer) -> Self {
        Self {
            tok: tok.clone(),
            buf: Vec::new(),
        }
    }

    pub fn push(&mut self, token_id: u32) -> String {
        if self.tok.is_stop_token(token_id) {
            return String::new();
        }
        if let Some(real) = self.tok.real_tokenizer() {
            if real.token_type.get(token_id as usize).copied() == Some(3) {
                return String::new();
            }
            self.buf.extend_from_slice(real.token_bytes(token_id));
            return self.drain_valid_utf8();
        }
        let tok = &self.tok;
        if token_id == ID_IM_END || token_id == ID_ENDOFTEXT || token_id == ID_IM_START {
            return String::new();
        }
        if token_id >= 256 {
            let mut out = String::new();
            if !self.buf.is_empty() {
                out.push_str(&String::from_utf8_lossy(&self.buf));
                self.buf.clear();
            }
            if let Some((special, _)) = tok.specials().iter().find(|(_, id)| *id == token_id) {
                out.push_str(special);
            }
            return out;
        }
        self.buf.push(token_id as u8);
        self.drain_valid_utf8()
    }

    pub fn flush(&mut self) -> String {
        if self.buf.is_empty() {
            return String::new();
        }
        let s = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        s
    }

    fn drain_valid_utf8(&mut self) -> String {
        match std::str::from_utf8(&self.buf) {
            Ok(s) => {
                let res = s.to_string();
                self.buf.clear();
                res
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if e.error_len().is_none() {
                    // Incomplete multi-byte sequence at the end of `self.buf`: keep the trailing bytes.
                    let valid = std::str::from_utf8(&self.buf[..valid_up_to])
                        .unwrap_or("")
                        .to_string();
                    self.buf.drain(..valid_up_to);
                    valid
                } else {
                    let s = String::from_utf8_lossy(&self.buf).into_owned();
                    self.buf.clear();
                    s
                }
            }
        }
    }
}

/// Streaming stop-string matcher matching `StopMatcher` in `serve/server.py`.
/// Holds back partial suffixes that could start a stop string and cuts content at the first match.
#[derive(Clone, Debug)]
pub struct StopMatcher {
    stops: Vec<String>,
    pub held: String,
    pub hit: Option<String>,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
        Self {
            stops,
            held: String::new(),
            hit: None,
        }
    }

    pub fn push(&mut self, piece: &str) -> (String, Option<String>) {
        if self.hit.is_some() || self.stops.is_empty() {
            return if self.hit.is_some() {
                (String::new(), self.hit.clone())
            } else {
                (piece.to_string(), None)
            };
        }
        self.held.push_str(piece);
        // Find earliest stop match
        let mut best: Option<(usize, &str)> = None;
        for stop in &self.stops {
            if let Some(pos) = self.held.find(stop.as_str()) {
                match best {
                    None => best = Some((pos, stop.as_str())),
                    Some((best_pos, _)) if pos < best_pos => best = Some((pos, stop.as_str())),
                    _ => {}
                }
            }
        }
        if let Some((pos, matched)) = best {
            let before = self.held[..pos].to_string();
            let hit_word = matched.to_string();
            self.hit = Some(hit_word.clone());
            self.held.clear();
            return (before, Some(hit_word));
        }
        // Check longest suffix of `self.held` that is a proper prefix of any stop string.
        let mut keep_len = 0usize;
        for stop in &self.stops {
            let max_k = stop.len().saturating_sub(1).min(self.held.len());
            for k in (1..=max_k).rev() {
                let start = self.held.len() - k;
                if self.held.is_char_boundary(start) && stop.starts_with(&self.held[start..]) {
                    if k > keep_len {
                        keep_len = k;
                    }
                    break;
                }
            }
        }
        if keep_len == 0 {
            (std::mem::take(&mut self.held), None)
        } else {
            let split = self.held.len() - keep_len;
            let emit = self.held[..split].to_string();
            self.held.drain(..split);
            (emit, None)
        }
    }

    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}
