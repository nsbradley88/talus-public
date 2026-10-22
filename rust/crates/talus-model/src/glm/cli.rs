//! `talus-cli` entry for GLM-5.3-Flash models (dispatched on `general.architecture == "glm5-next"`).
//!
//! Modes:
//!   --glm-eval TOKFILE [--all-logits] [--out DIR] [--dump]   teacher-forced eval; logits (+ per-layer dumps) to DIR
//!   serve | --serve | --engine-protocol                      stdio engine protocol (talus-sched)
//! Common: --model PATH  --ctx N (default 65536)  --no-mtp  --kernels PATH  --minstart GIB (default 112)  -v

use std::io::Write;
use std::path::{Path, PathBuf};

use talus_core_api::{CoreError, Result};
use talus_gguf::GgufModel;
use talus_mem::{Budget, BudgetCategory, HostMemProbe, GIB};

use super::engine::{self, Dump, GlmConfig, GlmEngine};
use super::ffi::GlmLib;
use super::weights::{self, GlmHparams, GlmWeights};

#[derive(Debug, Clone)]
pub struct GlmCli {
    pub model: PathBuf,
    pub ctx: usize,
    pub mtp: bool,
    pub kernels: Option<PathBuf>,
    pub minstart_gib: f64,
    pub verbose: bool,
    pub eval: Option<PathBuf>,
    /// llama-perplexity KL-divergence base file (--kld BASE), chunks to score (--kld-chunks)
    pub kld: Option<PathBuf>,
    pub kld_chunks: usize,
    pub all_logits: bool,
    pub last_logits: usize,
    pub out: Option<PathBuf>,
    pub dump: bool,
    pub serve: bool,
    pub load_only: bool,
    pub inject: Option<PathBuf>,
    pub gen: Option<PathBuf>,
    pub n_gen: usize,
    pub temp: f32,
    pub seed: u64,
    pub draft_vocab: Option<PathBuf>,
    pub minavail_gib: f64,
    pub extra: Vec<String>,
}

pub fn parse(args: &[String]) -> std::result::Result<GlmCli, String> {
    let mut c = GlmCli {
        model: PathBuf::new(),
        ctx: 65536,
        mtp: true,
        kernels: None,
        minstart_gib: 112.0,
        verbose: false,
        eval: None,
        kld: None,
        kld_chunks: 5,
        all_logits: false,
        last_logits: 1,
        out: None,
        dump: false,
        serve: false,
        load_only: false,
        inject: None,
        gen: None,
        n_gen: 64,
        temp: 0.0,
        seed: 0,
        draft_vocab: None,
        minavail_gib: 8.0,
        extra: vec![],
    };
    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        let mut val = || -> std::result::Result<String, String> {
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("missing value for {a}"))
        };
        match a {
            "--model" | "-m" => c.model = PathBuf::from(val()?),
            "--ctx" | "--max-context" | "-c" => {
                c.ctx = val()?.parse().map_err(|e| format!("{a}: {e}"))?
            }
            "--no-mtp" => c.mtp = false,
            "--kernels" => c.kernels = Some(PathBuf::from(val()?)),
            "--minstart" => c.minstart_gib = val()?.parse().map_err(|e| format!("{a}: {e}"))?,
            "-v" | "--verbose" => c.verbose = true,
            "--glm-eval" => c.eval = Some(PathBuf::from(val()?)),
            "--kld" => c.kld = Some(PathBuf::from(val()?)),
            "--kld-chunks" => c.kld_chunks = val()?.parse().map_err(|e| format!("{a}: {e}"))?,
            "--all-logits" => c.all_logits = true,
            "--last-logits" => c.last_logits = val()?.parse().map_err(|e| format!("{a}: {e}"))?,
            "--out" | "-o" => c.out = Some(PathBuf::from(val()?)),
            "--dump" => c.dump = true,
            "--glm-load-only" => c.load_only = true,
            "--inject" => c.inject = Some(PathBuf::from(val()?)),
            "--glm-gen" => c.gen = Some(PathBuf::from(val()?)),
            "-n" | "--n-gen" => c.n_gen = val()?.parse().map_err(|e| format!("{a}: {e}"))?,
            "--temp" => c.temp = val()?.parse().map_err(|e| format!("{a}: {e}"))?,
            "--seed" => c.seed = val()?.parse().map_err(|e| format!("{a}: {e}"))?,
            "--draft-vocab" | "--spec-draft-vocab" => c.draft_vocab = Some(PathBuf::from(val()?)),
            "--minavail" => c.minavail_gib = val()?.parse().map_err(|e| format!("{a}: {e}"))?,
            "serve" | "--serve" | "--engine-protocol" => c.serve = true,
            _ => c.extra.push(a.to_string()),
        }
        i += 1;
    }
    if c.model.as_os_str().is_empty() {
        return Err("--model is required".into());
    }
    Ok(c)
}

fn mem_available_gib() -> f64 {
    talus_mem::LinuxProcMemProbe::default()
        .mem_available_bytes()
        .map(|b| b as f64 / GIB as f64)
        .unwrap_or(0.0)
}

/// Load the model once, refusing configurations that would leave MemAvailable below the 8 GiB floor.
pub fn load_engine(c: &GlmCli) -> Result<GlmEngine> {
    let avail = mem_available_gib();
    if avail < c.minstart_gib {
        return Err(CoreError::InvalidArgument {
            reason: format!(
                "MemAvailable {avail:.1} GiB < minstart {:.1} GiB: refusing to load GLM-5.3-Flash",
                c.minstart_gib
            ),
        });
    }
    let g = GgufModel::open(&c.model).map_err(|e| CoreError::InvalidArgument {
        reason: format!("open {}: {e}", c.model.display()),
    })?;
    let hp = GlmHparams::from_gguf(&g)?;
    let lib = GlmLib::load(c.kernels.as_deref())?;
    if c.verbose {
        eprintln!(
            "glm: kernel library + HIP context up; MemAvailable {:.2} GiB",
            mem_available_gib()
        );
    }
    let wbytes = weights::total_weight_bytes(&g)
        - if weights::embed_on_host() {
            weights::tok_embd_bytes(&g)
        } else {
            0
        };
    let (kv, rec) = engine::state_bytes(&hp, c.ctx, c.mtp);
    // activations (phase-aliased, measured 406 MiB at GLM_PCH=512) and 2 prompt-cache snapshots
    let acts: u64 = 640 << 20; // 512-token chunks with phase-aliased buffers: 406 MiB + MMQ scratch
    let snaps: u64 = (34 * (4 << 20) + 34 * 3 * 8192 * 3 * 4) as u64; // 1 by default (GLM_PROMPT_SNAPS)
    let budget = Budget::strix_default();
    let ids = weights::reserve(
        &budget,
        &[
            (BudgetCategory::Weights, "glm.weights", wbytes),
            (BudgetCategory::KvCache, "glm.kv+indexer", kv as u64),
            (BudgetCategory::RecurrentState, "glm.kda", rec as u64),
            (BudgetCategory::Scratch, "glm.activations", acts),
            (
                BudgetCategory::PromptSnapshots,
                "glm.prompt_snapshots",
                snaps,
            ),
        ],
    )?;
    if c.verbose {
        eprintln!(
            "glm: MemAvailable {avail:.1} GiB; reserved weights {:.2} + kv {:.2} + kda {:.2} + acts {:.2} GiB (floor 8 GiB)",
            wbytes as f64 / GIB as f64,
            kv as f64 / GIB as f64,
            rec as f64 / GIB as f64,
            acts as f64 / GIB as f64
        );
    }
    let w = GlmWeights::load(&g, &hp, c.mtp, c.verbose)?;
    drop(g);
    if c.verbose {
        eprintln!(
            "glm: weights resident; MemAvailable {:.2} GiB",
            mem_available_gib()
        );
    }
    let eng = GlmEngine::new(
        lib,
        w,
        GlmConfig {
            ctx: c.ctx,
            mtp: c.mtp,
            verbose: c.verbose,
        },
    )?;
    for id in ids {
        let _ = budget.mark_materialized(id);
    }
    if c.verbose {
        eprintln!(
            "glm: loaded; MemAvailable now {:.2} GiB",
            mem_available_gib()
        );
    }
    Ok(eng)
}

fn read_tokens(p: &Path) -> Result<Vec<u32>> {
    let s = std::fs::read_to_string(p).map_err(|e| CoreError::IoError {
        message: format!("{}: {e}", p.display()),
    })?;
    Ok(s.split(|ch: char| ch.is_whitespace() || ch == ',')
        .filter(|x| !x.is_empty())
        .filter_map(|x| x.parse().ok())
        .collect())
}

fn write_f32(p: &Path, v: &[f32]) -> Result<()> {
    let mut f = std::fs::File::create(p).map_err(|e| CoreError::IoError {
        message: format!("{}: {e}", p.display()),
    })?;
    // SAFETY: f32 slice viewed as bytes.
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    f.write_all(b).map_err(|e| CoreError::IoError {
        message: e.to_string(),
    })
}

/// KL divergence vs a llama-perplexity `--kl-divergence-base` file (same statistics as `llama-perplexity
/// --kl-divergence`): every chunk of n_ctx tokens is read from scratch, the logits of positions n_ctx/2 .. n_ctx-2 are
/// scored against the base log-probs (uint16-quantized). GLM's vocabulary adds no BOS.
fn run_kld(c: &GlmCli, eng: &mut GlmEngine) -> Result<()> {
    use std::io::Read;
    let path = c.kld.as_ref().unwrap();
    let io = |e: std::io::Error| CoreError::IoError {
        message: format!("{}: {e}", path.display()),
    };
    let mut f = std::fs::File::open(path).map_err(io)?;
    let mut hdr = [0u8; 20];
    f.read_exact(&mut hdr).map_err(io)?;
    if &hdr[..8] != b"_logits_" {
        return Err(CoreError::InvalidArgument {
            reason: "not a llama-perplexity logits file".into(),
        });
    }
    let n_ctx = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let n_vocab = i32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let n_chunk_file = i32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
    if n_vocab != eng.n_vocab() {
        return Err(CoreError::InvalidArgument {
            reason: format!("vocab {n_vocab} != {}", eng.n_vocab()),
        });
    }
    let mut tb = vec![0u8; n_ctx * n_chunk_file * 4];
    f.read_exact(&mut tb).map_err(io)?;
    let tokens: Vec<u32> = tb
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect();
    let first = n_ctx / 2;
    let n_tok = n_ctx - 1 - first;
    let nv = 2 * n_vocab.div_ceil(2) + 4;
    let n_chunk = c.kld_chunks.min(n_chunk_file);
    let (mut s_kld, mut s_kld2, mut same, mut count) = (0f64, 0f64, 0usize, 0usize);
    let (mut s_nll, mut s_nll_base) = (0f64, 0f64);
    let mut kvals: Vec<f32> = vec![];
    let mut base = vec![0u8; n_tok * nv * 2];
    for ch in 0..n_chunk {
        f.read_exact(&mut base).map_err(io)?;
        let toks = &tokens[ch * n_ctx..(ch + 1) * n_ctx];
        eng.reset()?;
        let t0 = std::time::Instant::now();
        // rows for positions first .. n_ctx-1 (decode-size windows: the verify mat-vec path)
        let lg = eng.eval_last(toks, 0, n_ctx - first)?;
        let dt = t0.elapsed().as_secs_f64();
        let (mut ck, mut cn) = (0f64, 0usize);
        for i in 0..n_tok {
            let l = &lg[i * n_vocab..(i + 1) * n_vocab];
            let b = &base[i * nv * 2..(i + 1) * nv * 2];
            let scale = f32::from_le_bytes(b[0..4].try_into().unwrap());
            let min_lp = f32::from_le_bytes(b[4..8].try_into().unwrap());
            let q = |j: usize| u16::from_le_bytes([b[8 + 2 * j], b[9 + 2 * j]]) as f32;
            let (mut mx, mut imax) = (l[0], 0usize);
            for (j, &v) in l.iter().enumerate() {
                if v > mx {
                    mx = v;
                    imax = j;
                }
            }
            let se: f64 = l.iter().map(|&v| ((v - mx) as f64).exp()).sum();
            let lse = mx + se.ln() as f32;
            let tok = toks[first + i + 1] as usize;
            s_nll += (lse - l[tok]) as f64;
            s_nll_base += -(scale * q(tok) + min_lp) as f64;
            let (mut sum, mut pbm, mut ib) = (0f64, 0f32, 0usize);
            for j in 0..n_vocab {
                let plb = scale * q(j) + min_lp;
                if j == 0 || plb > pbm {
                    pbm = plb;
                    ib = j;
                }
                if plb > -16.0 {
                    sum += (plb.exp() * (plb - l[j] + lse)) as f64;
                }
            }
            s_kld += sum;
            s_kld2 += sum * sum;
            kvals.push(sum as f32);
            count += 1;
            same += (imax == ib) as usize;
            ck += sum;
            cn += 1;
        }
        println!(
            "kld chunk {} : {:.6} (eval {:.1} s) | running mean {:.6} same top {:.2}%",
            ch + 1,
            ck / cn as f64,
            dt,
            s_kld / count as f64,
            100.0 * same as f64 / count as f64
        );
    }
    let mean = s_kld / count as f64;
    let unc = ((s_kld2 / count as f64 - mean * mean) / (count as f64 - 1.0))
        .max(0.0)
        .sqrt();
    kvals.sort_by(|a, b| a.total_cmp(b));
    let p = same as f64 / count as f64;
    println!(
        "Mean    KLD: {mean:10.6} ± {unc:10.6}\nMedian  KLD: {:10.6}\nSame top p: {:6.3} ± {:5.3} %\nMean ln(PPL(Q)/PPL(base)): {:.6}\ncount {count}",
        kvals[kvals.len() / 2],
        100.0 * p,
        100.0 * (p * (1.0 - p) / (count as f64 - 1.0)).sqrt(),
        (s_nll - s_nll_base) / count as f64
    );
    Ok(())
}

fn run_eval(c: &GlmCli, eng: &mut GlmEngine) -> Result<()> {
    let toks = read_tokens(c.eval.as_ref().unwrap())?;
    let out = c
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from("glm-eval-out"));
    std::fs::create_dir_all(&out).map_err(|e| CoreError::IoError {
        message: e.to_string(),
    })?;
    if c.dump {
        let mut d = Dump::default();
        if let Ok(l) = std::env::var("GLM_DUMP_LAYERS") {
            d.layers = Some(l.split(',').filter_map(|x| x.trim().parse().ok()).collect());
        }
        if let Some(dir) = &c.inject {
            // glm-ref-dump index: name \t ne \t file \t op ; take the first p.l_out-N of each layer
            let idx =
                std::fs::read_to_string(dir.join("index.tsv")).map_err(|e| CoreError::IoError {
                    message: e.to_string(),
                })?;
            for line in idx.lines() {
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() < 3 {
                    continue;
                }
                if let Some(n) = f[0].strip_prefix("p.l_out-") {
                    if let Ok(il) = n.parse::<usize>() {
                        if d.inject.contains_key(&il) {
                            continue;
                        }
                        let b = std::fs::read(dir.join(f[2])).map_err(|e| CoreError::IoError {
                            message: e.to_string(),
                        })?;
                        let v: Vec<f32> = b
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        d.inject.insert(il, v);
                    }
                }
            }
            eprintln!(
                "glm: injecting reference l_out for {} layers",
                d.inject.len()
            );
        }
        eng.dump = Some(d);
    }
    eng.reset()?;
    let t0 = std::time::Instant::now();
    let n_last = if c.all_logits {
        toks.len()
    } else {
        c.last_logits
    };
    let logits = eng.eval_last(&toks, 0, n_last)?;
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "glm: eval {} tokens in {:.3} s ({:.1} tok/s)",
        toks.len(),
        dt,
        toks.len() as f64 / dt
    );
    write_f32(&out.join("logits.f32"), &logits)?;
    // greedy argmax of the last row
    let nv = eng.n_vocab();
    let last = &logits[logits.len() - nv..];
    let (am, _) = last
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
            if v > b.1 {
                (i, v)
            } else {
                b
            }
        });
    println!("argmax_last {am}");
    if let Some(d) = eng.dump.take() {
        let mut idx = String::new();
        for (n, name) in d.order.iter().enumerate() {
            let v = &d.tensors[name];
            let fname = format!("{n:05}_{name}.bin");
            write_f32(&out.join(&fname), v)?;
            idx.push_str(&format!("{name}\t{}\t{fname}\n", v.len()));
        }
        std::fs::write(out.join("index.tsv"), idx).map_err(|e| CoreError::IoError {
            message: e.to_string(),
        })?;
    }
    Ok(())
}

/// Draft vocabulary file: int32 little-endian token ids (llama.cpp --spec-draft-vocab format).
pub fn read_draft_vocab(p: &Path) -> Result<Vec<u32>> {
    let b = std::fs::read(p).map_err(|e| CoreError::IoError {
        message: format!("{}: {e}", p.display()),
    })?;
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .filter(|&x| x >= 0)
        .map(|x| x as u32)
        .collect())
}

fn run_gen(c: &GlmCli, eng: &mut GlmEngine) -> Result<()> {
    use super::spec::{self, GenParams, GenStats, SeqPos};
    let toks = read_tokens(c.gen.as_ref().unwrap())?;
    if let Some(dv) = &c.draft_vocab {
        let ids = read_draft_vocab(dv)?;
        eprintln!("glm: draft vocabulary {} tokens", ids.len());
        eng.set_draft_vocab(&ids)?;
    }
    let p = GenParams {
        max_new: c.n_gen,
        samp: super::sample::SamplerParams {
            temp: c.temp,
            seed: c.seed,
            ..Default::default()
        },
        mtp: c.mtp,
        eos: eng.w.hp.eos.clone(),
        ignore_eos: true,
        ctx: toks.iter().map(|&t| t as u32).collect(),
    };
    for rep in 0..2 {
        eng.reset()?;
        let mut seq = SeqPos::default();
        let mut st = GenStats::default();
        let t0 = std::time::Instant::now();
        let last = spec::prefill(eng, &mut seq, &toks, c.mtp)?;
        st.prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
        let out = spec::generate(eng, &mut seq, &last, &p, &mut st, |_| true)?;
        let ids: Vec<String> = out.iter().map(|x| x.to_string()).collect();
        println!("gen[{rep}] {}", ids.join(" "));
        println!(
            "stats[{rep}] prompt {} tok {:.0} ms ({:.1} tok/s) | gen {} tok {:.0} ms ({:.2} tok/s) | drafted {} accepted {} ({:.3}) rounds {}",
            toks.len(), st.prompt_ms, toks.len() as f64 / st.prompt_ms * 1e3, out.len(), st.decode_ms,
            out.len() as f64 / st.decode_ms * 1e3, st.drafted, st.accepted,
            st.accepted as f64 / st.drafted.max(1) as f64, st.rounds
        );
    }
    Ok(())
}

/// Entry point from talus-cli (argv including the program name).
pub fn run(args: &[String]) -> i32 {
    let c = match parse(args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("talus-glm: {e}");
            return 2;
        }
    };
    weights::spawn_watchdog((c.minavail_gib * GIB as f64) as u64);
    let mut eng = match load_engine(&c) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("talus-glm: {e}");
            return 1;
        }
    };
    let r = if c.load_only {
        eprintln!("glm: load-only done");
        Ok(())
    } else if c.kld.is_some() {
        run_kld(&c, &mut eng)
    } else if c.gen.is_some() {
        run_gen(&c, &mut eng)
    } else if c.eval.is_some() {
        run_eval(&c, &mut eng)
    } else if c.serve {
        Err(CoreError::InvalidArgument {
            reason: "serve: not implemented yet".into(),
        })
    } else {
        Err(CoreError::InvalidArgument {
            reason: "nothing to do (use --glm-eval TOKFILE or serve)".into(),
        })
    };
    match r {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("talus-glm: {e}");
            1
        }
    }
}
