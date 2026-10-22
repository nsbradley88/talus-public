//! Byte-level BPE tokenizer and streaming UTF-8 detokenizer.
//!
//! Direct port of `tools/strata_tokenizer.py`, supporting:
//! - GPT-2 byte-to-unicode mapping (`bytes_to_unicode`)
//! - Multi-stage regex pre-tokenizers (`qwen35`, `laguna`, `qwen2`, `glm4`, `default`)
//! - Exact BPE merge ordering (lowest merge rank first, leftmost tie-break) with heap fast-path
//! - Special (type 4 `USER_DEFINED`) and control (type 3 `CONTROL`) token splitting and `plain` spans
//! - Loading from extracted tokenizer directories or directly from GGUF headers/metadata
//! - Incremental streaming UTF-8 `Detokenizer` (`U+FFFD` replacement on invalid sequences, buffering incomplete multi-byte sequences)

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::path::Path;
use std::sync::RwLock;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use fancy_regex::Regex;
use thiserror::Error;

/// Qwen3.5 pre-tokenizer regex pattern (`tokenizer.ggml.pre == "qwen35"`).
pub const QWEN35_PATTERN: &str = concat!(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// Qwen2 pre-tokenizer regex pattern (`tokenizer.ggml.pre == "qwen2"` and Laguna stage 2).
pub const QWEN2_PATTERN: &str = concat!(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// Laguna two-stage pre-tokenizer regex patterns (`tokenizer.ggml.pre == "laguna"`).
pub const LAGUNA_PATTERNS: [&str; 2] = [r"[^\n]+|[\n]+", QWEN2_PATTERN];

/// GLM-4 / Llama-3 pre-tokenizer regex pattern (`tokenizer.ggml.pre == "glm4"`).
pub const GLM4_PATTERN: &str = concat!(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// GPT-2 pre-tokenizer regex pattern (`tokenizer.ggml.pre == "gpt2"`).
pub const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Return the pre-tokenizer regex stages for a `tokenizer.ggml.pre` identifier.
pub fn pre_patterns(pre: &str) -> Result<&'static [&'static str], TokenizerError> {
    match pre {
        "qwen35" | "default" | "" => Ok(&[QWEN35_PATTERN]),
        "laguna" => Ok(&LAGUNA_PATTERNS),
        "qwen2" => Ok(&[QWEN2_PATTERN]),
        "glm4" | "chatglm4" | "llama3" | "llama-bpe" => Ok(&[GLM4_PATTERN]),
        "gpt2" => Ok(&[GPT2_PATTERN]),
        other => Err(TokenizerError::UnknownPreTokenizer(other.to_string())),
    }
}

/// Errors produced when loading or using a [`Tokenizer`].
#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("unknown pre-tokenizer for tokenizer.ggml.pre = {0:?}")]
    UnknownPreTokenizer(String),
    #[error("regex compilation failed: {0}")]
    Regex(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid GGUF file: {0}")]
    InvalidGguf(String),
    #[error("missing required tokenizer metadata: {0}")]
    MissingMetadata(String),
}

/// GGUF tokenizer metadata fields used to construct a [`Tokenizer`].
#[derive(Debug, Clone, Default)]
pub struct GgufTokenizerMetadata {
    pub tokens: Vec<String>,
    pub merges: Vec<String>,
    pub token_type: Option<Vec<i32>>,
    pub pre: Option<String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub add_bos_token: Option<bool>,
    pub chat_template: Option<String>,
}

/// Build the GPT-2 reversible byte-to-unicode mapping tables.
pub fn bytes_to_unicode() -> ([char; 256], [Option<u8>; 324]) {
    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = [None; 324];
    let mut n = 0u32;

    for b in 0u16..=255 {
        let byte = b as u8;
        let is_direct = (b'!'..=b'~').contains(&byte)
            || (0xA1..=0xAC).contains(&byte)
            || (0xAE..=0xFF).contains(&byte);
        let code = if is_direct {
            b as u32
        } else {
            let c = 256 + n;
            n += 1;
            c
        };
        let ch = char::from_u32(code).expect("valid unicode scalar");
        byte_to_char[b as usize] = ch;
        if (code as usize) < char_to_byte.len() {
            char_to_byte[code as usize] = Some(byte);
        }
    }
    (byte_to_char, char_to_byte)
}

#[derive(Debug)]
struct SpecialMatcher {
    ac: AhoCorasick,
    ids: Vec<u32>,
}

impl SpecialMatcher {
    fn build(mut pairs: Vec<(String, u32)>) -> Option<Self> {
        pairs.retain(|(s, _)| !s.is_empty());
        if pairs.is_empty() {
            return None;
        }
        // Match longer literals first in case one is a prefix of another, matching strata_tokenizer.py.
        pairs.sort_by_key(|a| Reverse(a.0.len()));
        let patterns: Vec<&str> = pairs.iter().map(|(s, _)| s.as_str()).collect();
        let ids: Vec<u32> = pairs.iter().map(|(_, id)| *id).collect();
        let ac = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostFirst)
            .build(&patterns)
            .ok()?;
        Some(Self { ac, ids })
    }
}

/// Byte-level BPE tokenizer matching `tools/strata_tokenizer.py`.
pub struct Tokenizer {
    pub tokens: Vec<String>,
    pub vocab: HashMap<String, u32>,
    pub token_type: Vec<i32>,
    pub pre: String,
    pub special_ids: HashMap<String, Option<u32>>,
    pub add_bos_token: bool,
    pub merges_list: Vec<String>,
    pub merges: HashMap<(String, String), usize>,
    pub chat_template: Option<String>,
    stages: Vec<Regex>,
    special_matcher: Option<SpecialMatcher>,
    all_special_matcher: Option<SpecialMatcher>,
    byte_to_char: [char; 256],
    token_raw_bytes: Vec<Vec<u8>>,
    bpe_cache: RwLock<HashMap<String, Vec<u32>>>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.tokens.len())
            .field("merges_count", &self.merges.len())
            .field("pre", &self.pre)
            .field("special_ids", &self.special_ids)
            .finish()
    }
}

impl Tokenizer {
    /// Construct a [`Tokenizer`] from raw vocabulary, merges, token types, pre-tokenizer name, and special IDs.
    pub fn new(
        tokens: Vec<String>,
        merges: Vec<String>,
        token_type: Option<Vec<i32>>,
        pre: &str,
        special_ids: HashMap<String, Option<u32>>,
    ) -> Result<Self, TokenizerError> {
        let mut vocab = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            vocab.insert(t.clone(), i as u32);
        }

        let mut token_type = token_type.unwrap_or_else(|| vec![1; tokens.len()]);
        if token_type.len() < tokens.len() {
            token_type.resize(tokens.len(), 1);
        }

        let add_bos_token = special_ids
            .get("add_bos")
            .and_then(|v| *v)
            .map(|v| v != 0)
            .unwrap_or(false);

        let mut merges_map = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            if let Some((a, b)) = m.split_once(' ') {
                merges_map.insert((a.to_string(), b.to_string()), rank);
            }
        }

        let patterns = pre_patterns(pre)?;
        let mut stages = Vec::with_capacity(patterns.len());
        for pat in patterns {
            let rx = Regex::new(pat).map_err(|e| TokenizerError::Regex(e.to_string()))?;
            stages.push(rx);
        }

        // Token type 4 = USER_DEFINED (matched always); type 3 = CONTROL (matched when add_special=true).
        let mut special = Vec::new();
        let mut control = Vec::new();
        for (i, tok) in tokens.iter().enumerate() {
            match token_type[i] {
                4 => special.push((tok.clone(), i as u32)),
                3 => control.push((tok.clone(), i as u32)),
                _ => {}
            }
        }
        let mut all_special = special.clone();
        all_special.extend(control);

        let special_matcher = SpecialMatcher::build(special);
        let all_special_matcher = SpecialMatcher::build(all_special);

        let (byte_to_char, char_to_byte) = bytes_to_unicode();
        let mut token_raw_bytes = Vec::with_capacity(tokens.len());
        for (i, s) in tokens.iter().enumerate() {
            if matches!(token_type[i], 3 | 4) {
                token_raw_bytes.push(s.as_bytes().to_vec());
            } else {
                let mut bytes = Vec::with_capacity(s.len());
                for c in s.chars() {
                    let code = c as usize;
                    if let Some(Some(b)) = char_to_byte.get(code) {
                        bytes.push(*b);
                    } else {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
                token_raw_bytes.push(bytes);
            }
        }

        Ok(Self {
            tokens,
            vocab,
            token_type,
            pre: if pre.is_empty() || pre == "default" {
                "qwen35".to_string()
            } else {
                pre.to_string()
            },
            special_ids,
            add_bos_token,
            merges_list: merges,
            merges: merges_map,
            chat_template: None,
            stages,
            special_matcher,
            all_special_matcher,
            byte_to_char,
            token_raw_bytes,
            bpe_cache: RwLock::new(HashMap::new()),
        })
    }

    /// Construct a [`Tokenizer`] from [`GgufTokenizerMetadata`].
    pub fn from_gguf_metadata(meta: GgufTokenizerMetadata) -> Result<Self, TokenizerError> {
        if meta.tokens.is_empty() {
            return Err(TokenizerError::MissingMetadata(
                "tokenizer.ggml.tokens".to_string(),
            ));
        }
        let mut special = HashMap::new();
        special.insert("bos".to_string(), meta.bos_token_id);
        special.insert("eos".to_string(), meta.eos_token_id);
        special.insert("pad".to_string(), meta.pad_token_id);
        if let Some(add_bos) = meta.add_bos_token {
            special.insert("add_bos".to_string(), Some(u32::from(add_bos)));
        }
        let pre = meta.pre.as_deref().unwrap_or("qwen35");
        let mut tok = Self::new(meta.tokens, meta.merges, meta.token_type, pre, special)?;
        tok.chat_template = meta.chat_template;
        Ok(tok)
    }

    /// Load a [`Tokenizer`] directly from a GGUF file (or `.gguf.hdr` header file).
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let meta = read_gguf_tokenizer_metadata(&mut reader)?;
        Self::from_gguf_metadata(meta)
    }

    /// Load a [`Tokenizer`] directly from in-memory GGUF header bytes.
    pub fn from_gguf_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let mut cursor = Cursor::new(bytes);
        let meta = read_gguf_tokenizer_metadata(&mut cursor)?;
        Self::from_gguf_metadata(meta)
    }

    /// Load a [`Tokenizer`] from an extracted tokenizer directory (`vocab.json`, `merges.txt`,
    /// `token_type.json` / `special_tokens.json` / `tokenizer.json` / `tokenizer_config.json`).
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let dir = dir.as_ref();
        let vocab_path = dir.join("vocab.json");
        if !vocab_path.exists() {
            return Err(TokenizerError::MissingMetadata(format!(
                "vocab.json not found in {}",
                dir.display()
            )));
        }
        let vocab_val: serde_json::Value = serde_json::from_reader(File::open(&vocab_path)?)?;
        let tokens: Vec<String> = match vocab_val {
            serde_json::Value::Array(arr) => arr
                .into_iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect(),
            serde_json::Value::Object(map) => {
                let mut pairs: Vec<(usize, String)> = Vec::with_capacity(map.len());
                for (tok, id_val) in map {
                    let id = id_val.as_u64().ok_or_else(|| {
                        TokenizerError::MissingMetadata("vocab.json token id must be u64".into())
                    })? as usize;
                    pairs.push((id, tok));
                }
                pairs.sort_by_key(|(id, _)| *id);
                let max_id = pairs.last().map(|(id, _)| *id + 1).unwrap_or(0);
                let mut vec = vec![String::new(); max_id];
                for (id, tok) in pairs {
                    vec[id] = tok;
                }
                vec
            }
            _ => {
                return Err(TokenizerError::MissingMetadata(
                    "vocab.json must be object or array".into(),
                ));
            }
        };

        let merges_path = dir.join("merges.txt");
        let merges: Vec<String> = if merges_path.exists() {
            let content = std::fs::read_to_string(&merges_path)?;
            content
                .lines()
                .filter(|line| !line.is_empty() && !line.starts_with("#version:"))
                .map(ToString::to_string)
                .collect()
        } else {
            Vec::new()
        };

        let mut token_type: Option<Vec<i32>> = None;
        let tt_path = dir.join("token_type.json");
        if tt_path.exists() {
            token_type = Some(serde_json::from_reader(File::open(&tt_path)?)?);
        }

        let mut pre = "qwen35".to_string();
        let mut special_ids: HashMap<String, Option<u32>> = HashMap::new();
        let mut chat_template: Option<String> = None;

        for cfg_name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens.json",
        ] {
            let p = dir.join(cfg_name);
            if p.exists() {
                if let Ok(serde_json::Value::Object(cfg)) = serde_json::from_reader(File::open(&p)?)
                {
                    if let Some(p_str) = cfg.get("pre").and_then(|v| v.as_str()) {
                        pre = p_str.to_string();
                    }
                    if let Some(serde_json::Value::Object(sp)) = cfg.get("special_ids") {
                        for (k, v) in sp {
                            special_ids.insert(k.clone(), v.as_u64().map(|x| x as u32));
                        }
                    }
                    if let Some(bos) = cfg.get("bos_token_id").and_then(|v| v.as_u64()) {
                        special_ids
                            .entry("bos".to_string())
                            .or_insert(Some(bos as u32));
                    }
                    if let Some(eos) = cfg.get("eos_token_id").and_then(|v| v.as_u64()) {
                        special_ids
                            .entry("eos".to_string())
                            .or_insert(Some(eos as u32));
                    }
                    if let Some(pad) = cfg.get("pad_token_id").and_then(|v| v.as_u64()) {
                        special_ids
                            .entry("pad".to_string())
                            .or_insert(Some(pad as u32));
                    }
                    if let Some(add_bos) = cfg.get("add_bos_token").and_then(|v| v.as_bool()) {
                        special_ids.insert("add_bos".to_string(), Some(u32::from(add_bos)));
                    }
                    if let Some(tpl) = cfg.get("chat_template").and_then(|v| v.as_str()) {
                        chat_template = Some(tpl.to_string());
                    }
                    if token_type.is_none() {
                        if let Some(serde_json::Value::Object(added)) =
                            cfg.get("added_tokens_decoder")
                        {
                            let mut tt = vec![1i32; tokens.len()];
                            for (id_str, info) in added {
                                if let Ok(idx) = id_str.parse::<usize>() {
                                    if idx < tt.len() {
                                        let is_special = info
                                            .get("special")
                                            .and_then(|v| v.as_bool())
                                            .unwrap_or(false);
                                        tt[idx] = if is_special { 3 } else { 4 };
                                    }
                                }
                            }
                            token_type = Some(tt);
                        }
                    }
                }
            }
        }

        let jinja_path = dir.join("chat_template.jinja");
        if jinja_path.exists() {
            chat_template = Some(std::fs::read_to_string(&jinja_path)?);
        }

        let mut tok = Self::new(tokens, merges, token_type, &pre, special_ids)?;
        tok.chat_template = chat_template;
        Ok(tok)
    }

    /// Write the tokenizer files (`vocab.json`, `merges.txt`, `token_type.json`, `tokenizer.json`,
    /// and `chat_template.jinja` if present) to `out_dir`, matching `strata_tokenizer.py --extract`.
    pub fn extract_to_dir(&self, out_dir: impl AsRef<Path>) -> Result<(), TokenizerError> {
        let out_dir = out_dir.as_ref();
        std::fs::create_dir_all(out_dir)?;

        let vocab_map: serde_json::Map<String, serde_json::Value> = self
            .tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), serde_json::Value::from(i)))
            .collect();
        std::fs::write(
            out_dir.join("vocab.json"),
            serde_json::to_string(&vocab_map)?,
        )?;

        let mut merges_txt = self.merges_list.join("\n");
        merges_txt.push('\n');
        std::fs::write(out_dir.join("merges.txt"), merges_txt)?;

        std::fs::write(
            out_dir.join("token_type.json"),
            serde_json::to_string(&self.token_type)?,
        )?;

        let pats = pre_patterns(&self.pre)?;
        let meta = serde_json::json!({
            "model": "gpt2",
            "pre": self.pre,
            "vocab_size": self.tokens.len(),
            "num_merges": self.merges_list.len(),
            "special_ids": self.special_ids,
            "add_bos_token": self.add_bos_token,
            "pre_pattern": pats.last().copied().unwrap_or(QWEN35_PATTERN),
            "pre_patterns": pats,
        });
        std::fs::write(
            out_dir.join("tokenizer.json"),
            serde_json::to_string_pretty(&meta)? + "\n",
        )?;

        if let Some(tpl) = &self.chat_template {
            std::fs::write(out_dir.join("chat_template.jinja"), tpl)?;
        }
        Ok(())
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Look up the token ID for a token string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Look up the token string for a token ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// Return all CONTROL (type 3) token strings.
    pub fn control_tokens(&self) -> Vec<&str> {
        self.tokens
            .iter()
            .enumerate()
            .filter_map(|(i, s)| (self.token_type[i] == 3).then_some(s.as_str()))
            .collect()
    }

    /// Return the BOS token ID if configured.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.special_ids.get("bos").copied().flatten()
    }

    /// Return the EOS token ID if configured.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.special_ids.get("eos").copied().flatten()
    }

    /// Create a streaming [`Detokenizer`] for this tokenizer.
    pub fn detokenizer(&self, skip_special: bool) -> Detokenizer<'_> {
        Detokenizer::new(self, skip_special)
    }

    /// Split `text` through the configured pre-tokenizer regex stages.
    pub fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut pieces = vec![text.to_string()];
        for rx in &self.stages {
            let mut nxt = Vec::new();
            for p in &pieces {
                let mut pos = 0;
                let mut iter = rx.find_iter(p);
                while let Some(Ok(m)) = iter.next() {
                    if m.start() > pos {
                        nxt.push(p[pos..m.start()].to_string());
                    }
                    if !m.as_str().is_empty() {
                        nxt.push(m.as_str().to_string());
                    }
                    pos = m.end();
                }
                if pos < p.len() {
                    nxt.push(p[pos..].to_string());
                }
            }
            pieces = nxt;
        }
        pieces
    }

    fn bpe(&self, piece: &str) -> Vec<u32> {
        if piece.is_empty() {
            return Vec::new();
        }
        let mut mapped = String::with_capacity(piece.len() * 2);
        for &b in piece.as_bytes() {
            mapped.push(self.byte_to_char[b as usize]);
        }
        if piece.len() == 1 {
            return self.vocab.get(&mapped).copied().into_iter().collect();
        }
        if let Ok(cache) = self.bpe_cache.read() {
            if let Some(hit) = cache.get(&mapped) {
                return hit.clone();
            }
        }
        if piece.len() > 64 {
            return self.bpe_heap(piece, mapped);
        }

        let mut word: Vec<String> = piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_to_char[b as usize].to_string())
            .collect();

        while word.len() >= 2 {
            let mut best_rank = usize::MAX;
            let mut best_idx: Option<usize> = None;
            for i in 0..(word.len() - 1) {
                if let Some(&r) = self.merges.get(&(word[i].clone(), word[i + 1].clone())) {
                    if r < best_rank {
                        best_rank = r;
                        best_idx = Some(i);
                    }
                }
            }
            let Some(best) = best_idx else {
                break;
            };
            let right = word.remove(best + 1);
            word[best].push_str(&right);
        }

        let ids: Vec<u32> = word
            .iter()
            .filter_map(|w| self.vocab.get(w).copied())
            .collect();
        if let Ok(mut cache) = self.bpe_cache.write() {
            cache.insert(mapped, ids.clone());
        }
        ids
    }

    fn bpe_heap(&self, piece: &str, mapped: String) -> Vec<u32> {
        let mut sym: Vec<Option<String>> = piece
            .as_bytes()
            .iter()
            .map(|&b| Some(self.byte_to_char[b as usize].to_string()))
            .collect();
        let n = sym.len();
        let mut prev: Vec<isize> = (0..n).map(|i| i as isize - 1).collect();
        let mut nxt: Vec<usize> = (0..n).map(|i| i + 1).collect();
        if n > 0 {
            nxt[n - 1] = usize::MAX;
        }

        let mut heap: BinaryHeap<Reverse<(usize, usize, String, String)>> = BinaryHeap::new();
        for i in 0..n.saturating_sub(1) {
            let a = sym[i].as_ref().unwrap();
            let b = sym[i + 1].as_ref().unwrap();
            if let Some(&r) = self.merges.get(&(a.clone(), b.clone())) {
                heap.push(Reverse((r, i, a.clone(), b.clone())));
            }
        }

        while let Some(Reverse((_, i, a, b))) = heap.pop() {
            let j = nxt[i];
            if j == usize::MAX {
                continue;
            }
            if sym[i].as_deref() != Some(a.as_str()) || sym[j].as_deref() != Some(b.as_str()) {
                continue;
            }
            let ab = format!("{a}{b}");
            sym[i] = Some(ab.clone());
            sym[j] = None;
            let nj = nxt[j];
            nxt[i] = nj;
            if nj != usize::MAX {
                prev[nj] = i as isize;
                if let Some(right) = &sym[nj] {
                    if let Some(&r) = self.merges.get(&(ab.clone(), right.clone())) {
                        heap.push(Reverse((r, i, ab.clone(), right.clone())));
                    }
                }
            }
            let p = prev[i];
            if p >= 0 {
                let pu = p as usize;
                if let Some(left) = &sym[pu] {
                    if let Some(&r) = self.merges.get(&(left.clone(), ab.clone())) {
                        heap.push(Reverse((r, pu, left.clone(), ab)));
                    }
                }
            }
        }

        let mut ids = Vec::new();
        let mut k = 0;
        while k != usize::MAX && k < n {
            if let Some(s) = &sym[k] {
                if let Some(&id) = self.vocab.get(s) {
                    ids.push(id);
                }
            }
            k = nxt[k];
        }
        if let Ok(mut cache) = self.bpe_cache.write() {
            cache.insert(mapped, ids.clone());
        }
        ids
    }

    fn encode_chunk(&self, chunk: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in self.pre_tokenize(chunk) {
            out.extend(self.bpe(&piece));
        }
        out
    }

    /// Encode `text` into token IDs.
    ///
    /// - `add_special = false`: splits on type-4 (`USER_DEFINED`) tokens (`<think>`, `<tool_call>`, etc.)
    ///   while encoding type-3 (`CONTROL`) tokens (`<|im_start|>`, `<|im_end|>`) as literal text.
    /// - `add_special = true`: splits on both type-4 and type-3 special tokens.
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        self.encode_with_plain(text, add_special, &[])
    }

    /// Encode `text` into token IDs, leaving any special token matches whose start byte offset falls
    /// within one of the `plain` `(start, end)` byte spans to be encoded as ordinary BPE text.
    pub fn encode_with_plain(
        &self,
        text: &str,
        add_special: bool,
        plain: &[(usize, usize)],
    ) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let matcher = if add_special {
            self.all_special_matcher.as_ref()
        } else {
            self.special_matcher.as_ref()
        };
        let Some(matcher) = matcher else {
            return self.encode_chunk(text);
        };

        let mut out = Vec::new();
        let mut pos = 0;
        for m in matcher.ac.find_iter(text) {
            if !plain.is_empty() && plain.iter().any(|&(a, b)| a <= m.start() && m.start() < b) {
                continue;
            }
            if m.start() > pos {
                out.extend(self.encode_chunk(&text[pos..m.start()]));
            }
            out.push(matcher.ids[m.pattern().as_usize()]);
            pos = m.end();
        }
        if pos < text.len() {
            out.extend(self.encode_chunk(&text[pos..]));
        }
        out
    }

    /// Return the raw UTF-8 byte fragment represented by a single token ID.
    pub fn token_bytes(&self, tid: u32) -> &[u8] {
        self.token_raw_bytes
            .get(tid as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Decode a sequence of token IDs into a UTF-8 string (replacing invalid UTF-8 sequences with U+FFFD).
    ///
    /// When `skip_special` is `true`, control tokens (`token_type == 3`) are omitted.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut raw = Vec::new();
        for &id in ids {
            if skip_special && self.token_type.get(id as usize).copied() == Some(3) {
                continue;
            }
            raw.extend_from_slice(self.token_bytes(id));
        }
        String::from_utf8_lossy(&raw).into_owned()
    }
}

/// Incremental UTF-8 detokenizer that buffers incomplete multi-byte UTF-8 sequences across tokens.
///
/// Matches Python's `codecs.getincrementaldecoder("utf-8")(errors="replace")` and `serve/test_detok.py`.
pub struct Detokenizer<'a> {
    tokenizer: &'a Tokenizer,
    skip_special: bool,
    pending: Vec<u8>,
}

impl<'a> Detokenizer<'a> {
    pub fn new(tokenizer: &'a Tokenizer, skip_special: bool) -> Self {
        Self {
            tokenizer,
            skip_special,
            pending: Vec::with_capacity(8),
        }
    }

    /// Push a single token ID and return any newly completed UTF-8 text.
    pub fn push(&mut self, token_id: u32) -> String {
        if self.skip_special && self.tokenizer.token_type.get(token_id as usize).copied() == Some(3)
        {
            return String::new();
        }
        self.push_bytes(self.tokenizer.token_bytes(token_id))
    }

    /// Push a slice of token IDs and return any newly completed UTF-8 text.
    pub fn push_tokens(&mut self, token_ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in token_ids {
            out.push_str(&self.push(id));
        }
        out
    }

    /// Push raw token bytes and decode complete UTF-8 characters, holding only trailing incomplete
    /// multi-byte UTF-8 sequences (1–3 bytes) for the next call.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> String {
        if bytes.is_empty() {
            return String::new();
        }
        self.pending.extend_from_slice(bytes);
        self.drain_ready(false)
    }

    /// Flush any remaining incomplete UTF-8 bytes at end of stream (emitting `U+FFFD` if incomplete).
    pub fn finish(&mut self) -> String {
        self.drain_ready(true)
    }

    fn drain_ready(&mut self, final_flush: bool) -> String {
        let mut out = String::new();
        while !self.pending.is_empty() {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    out.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to > 0 {
                        let valid_str = std::str::from_utf8(&self.pending[..valid_up_to]).unwrap();
                        out.push_str(valid_str);
                        self.pending.drain(..valid_up_to);
                    }
                    match err.error_len() {
                        Some(invalid_len) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..invalid_len);
                        }
                        None => {
                            if final_flush {
                                out.push(char::REPLACEMENT_CHARACTER);
                                self.pending.clear();
                            }
                            break;
                        }
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Minimal GGUF v2/v3 metadata reader for tokenizer fields
// ---------------------------------------------------------------------------

fn read_u8<R: Read>(r: &mut R) -> std::io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32<R: Read>(r: &mut R) -> std::io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_gguf_string<R: Read>(r: &mut R) -> Result<String, TokenizerError> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip_gguf_value<R: Read>(r: &mut R, vtype: u32) -> Result<(), TokenizerError> {
    let fixed_size = match vtype {
        0 | 1 | 7 => Some(1usize),
        2 | 3 => Some(2usize),
        4..=6 => Some(4usize),
        10..=12 => Some(8usize),
        _ => None,
    };
    if let Some(sz) = fixed_size {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf[..sz])?;
        return Ok(());
    }
    match vtype {
        8 => {
            let len = read_u64(r)?;
            std::io::copy(&mut r.take(len), &mut std::io::sink())?;
            Ok(())
        }
        9 => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            let elem_fixed = match elem_type {
                0 | 1 | 7 => Some(1u64),
                2 | 3 => Some(2u64),
                4..=6 => Some(4u64),
                10..=12 => Some(8u64),
                _ => None,
            };
            if let Some(esz) = elem_fixed {
                std::io::copy(&mut r.take(count * esz), &mut std::io::sink())?;
            } else {
                for _ in 0..count {
                    skip_gguf_value(r, elem_type)?;
                }
            }
            Ok(())
        }
        other => Err(TokenizerError::InvalidGguf(format!(
            "unknown GGUF value type {other}"
        ))),
    }
}

fn read_gguf_string_array<R: Read>(r: &mut R) -> Result<Vec<String>, TokenizerError> {
    let elem_type = read_u32(r)?;
    let count = read_u64(r)? as usize;
    if elem_type != 8 {
        return Err(TokenizerError::InvalidGguf(format!(
            "expected string array (type 8), got {elem_type}"
        )));
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(read_gguf_string(r)?);
    }
    Ok(out)
}

fn read_gguf_i32_array<R: Read>(r: &mut R) -> Result<Vec<i32>, TokenizerError> {
    let elem_type = read_u32(r)?;
    let count = read_u64(r)? as usize;
    let mut out = Vec::with_capacity(count);
    match elem_type {
        4 => {
            for _ in 0..count {
                out.push(read_u32(r)? as i32);
            }
        }
        5 => {
            for _ in 0..count {
                out.push(read_i32(r)?);
            }
        }
        other => {
            return Err(TokenizerError::InvalidGguf(format!(
                "expected i32/u32 array (4 or 5), got {other}"
            )));
        }
    }
    Ok(out)
}

fn read_gguf_u32_scalar<R: Read>(r: &mut R, vtype: u32) -> Result<u32, TokenizerError> {
    match vtype {
        0 => Ok(read_u8(r)? as u32),
        4 => Ok(read_u32(r)?),
        5 => Ok(read_i32(r)? as u32),
        10 => Ok(read_u64(r)? as u32),
        other => Err(TokenizerError::InvalidGguf(format!(
            "expected integer scalar, got type {other}"
        ))),
    }
}

/// Read [`GgufTokenizerMetadata`] from a GGUF v2/v3 stream.
pub fn read_gguf_tokenizer_metadata<R: Read>(
    r: &mut R,
) -> Result<GgufTokenizerMetadata, TokenizerError> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(TokenizerError::InvalidGguf(format!(
            "bad GGUF magic: {magic:?}"
        )));
    }
    let version = read_u32(r)?;
    if !(2..=3).contains(&version) {
        return Err(TokenizerError::InvalidGguf(format!(
            "unsupported GGUF version: {version}"
        )));
    }
    let _n_tensors = read_u64(r)?;
    let n_kv = read_u64(r)?;

    let mut meta = GgufTokenizerMetadata::default();
    for _ in 0..n_kv {
        let key = read_gguf_string(r)?;
        let vtype = read_u32(r)?;
        match key.as_str() {
            "tokenizer.ggml.tokens" if vtype == 9 => {
                meta.tokens = read_gguf_string_array(r)?;
            }
            "tokenizer.ggml.merges" if vtype == 9 => {
                meta.merges = read_gguf_string_array(r)?;
            }
            "tokenizer.ggml.token_type" if vtype == 9 => {
                meta.token_type = Some(read_gguf_i32_array(r)?);
            }
            "tokenizer.ggml.pre" if vtype == 8 => {
                meta.pre = Some(read_gguf_string(r)?);
            }
            "tokenizer.ggml.bos_token_id" => {
                meta.bos_token_id = Some(read_gguf_u32_scalar(r, vtype)?);
            }
            "tokenizer.ggml.eos_token_id" => {
                meta.eos_token_id = Some(read_gguf_u32_scalar(r, vtype)?);
            }
            "tokenizer.ggml.padding_token_id" => {
                meta.pad_token_id = Some(read_gguf_u32_scalar(r, vtype)?);
            }
            "tokenizer.ggml.add_bos_token" if vtype == 7 => {
                meta.add_bos_token = Some(read_u8(r)? != 0);
            }
            "tokenizer.chat_template" if vtype == 8 => {
                meta.chat_template = Some(read_gguf_string(r)?);
            }
            _ => {
                skip_gguf_value(r, vtype)?;
            }
        }
    }
    Ok(meta)
}
