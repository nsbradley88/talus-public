//! E6: grammar-constrained decoding (`response_format: json_schema`) for the Q36 stdio backend.
//!
//! Off unless a GEN line carries `json_schema=`; the vocabulary table is built on the first such request, so
//! unconstrained serving pays nothing. A constrained request decodes without drafting (MTP / DFlash / n-gram), one
//! token per step: the grammar mask is applied to the target logits before greedy pick or sampling.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use talus_text::json_grammar::{self, mask_has, Grammar, Matcher, Vocab};

/// Compiled schemas kept per process (each also memoises its masks).
const CACHE: usize = 8;

pub struct GrammarCtx {
    vocab: Arc<Vocab>,
    cache: Vec<(String, Arc<Grammar>)>,
}

impl GrammarCtx {
    /// Builds the vocabulary table from the model's tokenizer: ordinary tokens only (control and user-defined
    /// tokens such as `<think>` never appear inside the JSON); `eos` are allowed once the value is complete.
    pub fn load(model: &Path, n_vocab: usize, eos: &[i32]) -> Result<Self, String> {
        let t0 = Instant::now();
        let tok = talus_text::Tokenizer::from_gguf(model)
            .map_err(|e| format!("json_schema: cannot load the tokenizer: {e}"))?;
        let eos: Vec<u32> = eos.iter().filter(|&&e| e >= 0).map(|&e| e as u32).collect();
        let n_tok = tok.tokens.len();
        let vocab = Vocab::new(
            n_vocab,
            |id| {
                let i = id as usize;
                (i < n_tok && tok.token_type.get(i).copied() == Some(1))
                    .then(|| tok.token_bytes(id).to_vec())
            },
            &eos,
        );
        eprintln!(
            "talus-serve: json_schema vocabulary ready ({n_tok} tokens) in {:.0} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
        Ok(Self {
            vocab: Arc::new(vocab),
            cache: Vec::new(),
        })
    }

    /// A fresh matcher for `schema` (compiled once per distinct schema text).
    pub fn matcher(&mut self, schema: &str) -> Result<Matcher, String> {
        let g = match self.cache.iter().position(|(s, _)| s == schema) {
            Some(i) => self.cache[i].1.clone(),
            None => {
                let g = Arc::new(json_grammar::compile(schema)?);
                if self.cache.len() >= CACHE {
                    self.cache.remove(0);
                }
                self.cache.push((schema.to_string(), g.clone()));
                g
            }
        };
        Ok(Matcher::new(g, self.vocab.clone()))
    }
}

/// The next token among those the grammar allows: the masked argmax when `greedy`, else `sample` over the logits
/// with the disallowed entries at -inf. `None` = no token is allowed (the schema cannot be continued).
pub fn pick(
    m: &Matcher,
    logits: &[f32],
    greedy: bool,
    scratch: &mut Vec<f32>,
    sample: impl FnOnce(&[f32]) -> i32,
) -> Option<i32> {
    let mask = m.mask();
    let masked_argmax = || {
        let mut best: Option<(usize, f32)> = None;
        for (i, &v) in logits.iter().enumerate() {
            if !mask_has(&mask, i) {
                continue;
            }
            match best {
                None => best = Some((i, v)),
                Some((_, bv)) if v > bv => best = Some((i, v)),
                _ => {}
            }
        }
        best.map(|(i, _)| i as i32)
    };
    if greedy {
        return masked_argmax();
    }
    scratch.clear();
    scratch.extend(logits.iter().enumerate().map(|(i, &v)| {
        if mask_has(&mask, i) {
            v
        } else {
            f32::NEG_INFINITY
        }
    }));
    let t = sample(scratch);
    if t >= 0 && mask_has(&mask, t as usize) {
        Some(t)
    } else {
        masked_argmax()
    }
}
