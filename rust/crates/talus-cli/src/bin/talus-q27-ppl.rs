//! teacher-forced logits of a qwen35 model over llama-perplexity's chunks, for in-engine KL measurements.
//!
//! talus-q27-ppl --model GGUF --tokens TOKENS.i32 [--n-ctx 2048] [--chunks N] [--rows 256] [--max-context N] [--out F]
//!
//! TOKENS.i32 holds n_chunk * n_ctx little-endian int32 token ids (the token array of a llama-perplexity
//! `--kl-divergence-base` file; `tools/kldstream.py tokens` extracts it). For every chunk the engine is reset, the first
//! n_ctx / 2 tokens are prefilled and the logits of positions n_ctx / 2 .. n_ctx - 2 are written (llama-perplexity's
//! scored rows), each as n_vocab f32, after a 20-byte header: b"TLG1", n_vocab, n_ctx, n_chunk, first (u32 LE).
//! `--out -` (default) streams to stdout, for `tools/kldstream.py score|compact --raw -`.
//! No drafter is loaded (`--no-mtp` semantics); the engine options otherwise come from the environment.

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

use talus_model::qwen35::{Q36Engine, Q36Model, Q36Options};

fn arg_val(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().collect();
    let (Some(model_path), Some(tok_path)) =
        (arg_val(&args, "--model"), arg_val(&args, "--tokens"))
    else {
        eprintln!("usage: talus-q27-ppl --model GGUF --tokens TOKENS.i32 [--n-ctx 2048] [--chunks N] [--rows 256] [--max-context N] [--out F|-]");
        return 2;
    };
    let num = |n: &str, d: usize| {
        arg_val(&args, n)
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(d)
    };
    let n_ctx = num("--n-ctx", 2048);
    let rows = num("--rows", 256);
    let max_ctx = num("--max-context", n_ctx + 256);
    let out = arg_val(&args, "--out").unwrap_or_else(|| "-".to_string());

    let mut raw = Vec::new();
    if let Err(e) = File::open(&tok_path).and_then(|mut f| f.read_to_end(&mut raw)) {
        eprintln!("talus-q27-ppl: cannot read {tok_path}: {e}");
        return 1;
    }
    let toks: Vec<i32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    if n_ctx < 4 || toks.len() < n_ctx {
        eprintln!(
            "talus-q27-ppl: {} tokens is less than one {n_ctx}-token chunk",
            toks.len()
        );
        return 2;
    }
    let mut n_chunk = toks.len() / n_ctx;
    let want = num("--chunks", 0);
    if want > 0 {
        n_chunk = n_chunk.min(want);
    }
    let first = n_ctx / 2;

    let mut opts = Q36Options::from_env();
    opts.max_seq = max_ctx as i32;
    opts.no_mtp = true;
    opts.no_draft_vocab = true;
    let t_load = Instant::now();
    let model = match Q36Model::load(Path::new(&model_path), None, None, &opts) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("talus-q27-ppl: failed to load model {model_path}: {e}");
            return 1;
        }
    };
    let mut eng = match Q36Engine::new(&model, max_ctx as i32) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("talus-q27-ppl: failed to initialize engine: {e}");
            return 1;
        }
    };
    let n_vocab = eng.n_vocab() as usize;
    eprintln!(
        "talus-q27-ppl: loaded in {:.1} s; {n_chunk} chunks of {n_ctx}, scoring positions {first}..{} (vocab {n_vocab})",
        t_load.elapsed().as_secs_f64(),
        n_ctx - 2
    );

    let sink: Box<dyn Write> = if out == "-" {
        Box::new(io::stdout().lock())
    } else {
        match File::create(&out) {
            Ok(f) => Box::new(f),
            Err(e) => {
                eprintln!("talus-q27-ppl: cannot create {out}: {e}");
                return 1;
            }
        }
    };
    let mut w = BufWriter::with_capacity(8 << 20, sink);
    let mut hdr = Vec::with_capacity(20);
    hdr.extend_from_slice(b"TLG1");
    for v in [n_vocab, n_ctx, n_chunk, first] {
        hdr.extend_from_slice(&(v as u32).to_le_bytes());
    }
    if w.write_all(&hdr).is_err() {
        return 1;
    }
    let t0 = Instant::now();
    for c in 0..n_chunk {
        let tc = Instant::now();
        let seq = &toks[c * n_ctx..(c + 1) * n_ctx];
        let mut werr = false;
        let r = eng.score_rows(seq, first, rows, |row| {
            let bytes =
                unsafe { std::slice::from_raw_parts(row.as_ptr().cast::<u8>(), row.len() * 4) };
            if !werr && w.write_all(bytes).is_err() {
                werr = true;
            }
        });
        if let Err(e) = r {
            eprintln!("talus-q27-ppl: chunk {c}: {e}");
            return 1;
        }
        if werr {
            eprintln!("talus-q27-ppl: output closed at chunk {c}");
            return 1;
        }
        eprintln!(
            "talus-q27-ppl: chunk {}/{n_chunk} {:.2} s ({:.0} tok/s)",
            c + 1,
            tc.elapsed().as_secs_f64(),
            (n_ctx - 1) as f64 / tc.elapsed().as_secs_f64().max(1e-9)
        );
    }
    if w.flush().is_err() {
        return 1;
    }
    eprintln!(
        "talus-q27-ppl: done, {n_chunk} chunks in {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    0
}
