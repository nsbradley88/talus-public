//! Talus CLI library (`strata-q36` / `talus-qwen35` / `talus-cli` drop-in replacement).
//!
//! Provides [`CliConfig`], [`parse_cli_args`], [`parse_ids_arg`], and CLI execution dispatch
//! matching `strata::qwen35::run` in `src/qwen35/q36_engine.cu` (lines 3300–3671).

#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::collapsible_if,
    clippy::field_reassign_with_default,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

mod disk_tier;
pub mod glm_serve;
pub mod grammar;
use core::ffi::c_void;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use talus_model::qwen35::{
    sample, DrafterPick, NgramMod, Q36Engine, Q36Model, Q36Options, Sampling, SplitMix64,
};
use talus_sched::{
    parse_ids_str, serve_stdio, write_done_line, write_pp_line, write_resume_line,
    write_token_line, CancelToken, DoneStats, EngineProtocolBackend, GenCommand,
    ScriptedProtocolBackend,
};
use talus_state::{Q36LayerStateView, Q36PromptCache};

pub const CRATE_NAME: &str = "talus-cli";

/// Parsed command-line configuration for `talus-cli` / `strata-q36` (`talus-qwen35`).
#[derive(Debug, Clone, PartialEq)]
pub struct CliConfig {
    /// Path to the main GGUF model (`--model` / `--native <PATH>`).
    pub model: Option<String>,
    /// Maximum context length (`--max-context` / `--ctx <N>`, default `32768`).
    pub max_context: i32,
    /// Prefill chunk size (`--chunk` / `--prefill <N|auto>`, default `0` = auto).
    pub chunk: i32,
    /// Whether single-token / verify HIP graphs are enabled (`true` unless `--no-graph` / `--no-token-graph`).
    pub use_graph: bool,
    /// Explicit flag indicating `--no-graph` / `--no-token-graph` was passed.
    pub no_graph: bool,
    /// Whether startup warmup is enabled (`true` unless `--no-warmup`).
    pub warmup: bool,
    /// Explicit flag indicating `--no-warmup` was passed.
    pub no_warmup: bool,
    /// Number of MTP speculative drafts (`--mtp <N>`, default `0`).
    pub mtp: i32,
    /// Path to an external MTP GGUF (`--mtp <FILE.gguf>` or `--mtp-gguf <PATH>`).
    pub mtp_gguf: Option<String>,
    /// Whether MTP uses the main model's output head (`--mtp-main-head`).
    pub mtp_main_head: bool,
    /// Path to a draft vocabulary subset GGUF (`--spec-draft-vocab` / `--draft-vocab <PATH>`).
    pub draft_vocab: Option<String>,
    /// Path to a DFlash block-diffusion drafter GGUF (`--dflash-gguf <PATH>`).
    pub dflash_gguf: Option<String>,
    /// Context-depth drafter selector (`--drafter <mtp|dflash|auto>`).
    pub drafter: Option<String>,
    /// N-gram draft lookup order (`--ngram <N>`, default `0` = off).
    pub ngram: i32,
    /// Number of prompt cache state snapshots (`--prompt-cache <N>`, default `8`).
    pub prompt_cache: i32,
    /// Device memory budget in MiB for KV stashes (`--kv-stash-mib <M>`, default `8192`).
    pub kv_stash_mib: i32,
    /// Minimum tokens before message-boundary snapshot (`--snap-msg <N>`, default `1024`).
    pub snap_msg: i32,
    /// Disk tier folder (`--prompt-cache-disk DIR|auto|off`; else `TALUS_PCACHE_DISK`; default off).
    pub prompt_cache_disk: Option<String>,
    /// Disk tier budget in GiB (`--prompt-cache-disk-gib N`; else `TALUS_PCACHE_DISK_GIB`; default 32).
    pub prompt_cache_disk_gib: Option<String>,
    /// Shortest prompt written to the disk tier (`--prompt-cache-disk-min N`; else `TALUS_PCACHE_DISK_MIN`).
    pub prompt_cache_disk_min: Option<String>,
    /// Prompt cache live continuation policy (`--cache-live <N>`, default `0` = grid-aligned cold-exact policy).
    pub cache_live: i32,
    /// Run warm vs cold prompt cache exactness self-test (`--livecold-test <N>`, default `0`).
    pub livecold_test: i32,
    /// Whether stdio serve mode is active (`serve` subcommand, `--serve`, or `--engine-protocol`).
    pub serve: bool,
    /// Whether `--engine-protocol` (or `--serve` / `serve`) was requested.
    pub engine_protocol: bool,
    /// Prompt token specification (`--tokens <IDS|@FILE>`).
    pub tokens: Option<String>,
    /// Maximum new tokens to generate (`--n` / `--max-new <N>`, default `64`).
    pub max_new: i32,
    /// Forced continuation token specification (`--force <IDS|@FILE>`).
    pub force: Option<String>,
    /// Path to dump full float32 logits (`--dump-full-logits <PATH>`).
    pub dump_full_logits: Option<String>,
    /// Prefill benchmark length (`--bench-pp <N>`, default `0`).
    pub bench_pp: i32,
    /// Token generation benchmark length (`--bench-tg <N>`, default `0`).
    pub bench_tg: i32,
    /// Prior context depth for benchmarks (`--bench-depth <N>`, default `0`).
    pub bench_depth: i32,
    /// Speculative verify window sizes to benchmark (`--bench-window <W1,W2,...>`).
    pub bench_window: Vec<i32>,
    /// Number of timed benchmark steps (`--bench-steps <N>`, default `32`).
    pub bench_steps: i32,
    /// MMVQ microbenchmark filter (`--mmvq-bench <FILTER|->`).
    pub mmvq_bench: Option<String>,
    /// Run CPU sampling & speculative distribution self-test (`--sample-test`).
    pub sample_test: bool,
    /// Run T=4 vs T=1 exactness self-test at position N (`--exact-test <N>`, default `0`).
    pub exact_test: i32,
    /// Run WMMA attention self-test (`--attn-test`).
    pub attn_test: bool,
    /// Run WMMA attention microbenchmark (`--attn-bench`).
    pub attn_bench: bool,
    /// Hybrid MoE expert VRAM budget in GiB (`--expert-vram <GIB>`, default `-1.0`).
    pub expert_vram: f32,
    /// CPU worker thread count (`--cpu-threads` / `--pool-workers <N>`, default `0`).
    pub cpu_threads: i32,
    /// Hybrid MoE expert slot count (`--expert-slots <N>`, default `0`).
    pub expert_slots: i32,
    /// Hybrid MoE rows threshold (`--hybrid-rows <N>`, default `128`).
    pub hybrid_rows: i32,
    /// HIP GPU device ID (`--gpu <ID>`).
    pub gpu: Option<i32>,
    /// Explicit greedy flag (`--greedy`).
    pub greedy: bool,
    /// Disable MTP (`--no-mtp`).
    pub no_mtp: bool,
    /// Convert Q8_0 MTP weights to FP16 on load (`--mtp-f16`).
    pub mtp_f16: bool,
    /// Disable draft vocabulary subset head (`--no-draft-vocab`).
    pub no_draft_vocab: bool,
    /// Enable 8-wave WMMA attention (`--attn-w8`).
    pub attn_w8: bool,
    /// Disable 16-row verify threshold (`--no-v16`).
    pub no_v16: bool,
    /// Text prompt for one-shot CLI generation (`--prompt <TEXT>`).
    pub prompt: Option<String>,
    /// Optional HTTP port when `--serve <PORT>` is passed (`None` = stdio protocol).
    pub serve_port: Option<u16>,
    /// Display help and exit (`-h` / `--help`).
    pub help: bool,
    /// Unknown `--flag [val]` tokens ignored when running in `--serve` / `--engine-protocol` mode.
    pub ignored_flags: Vec<String>,
    /// Laguna: number of layers to load (`--n-layers <N>`, default `48`).
    pub n_layers: i32,
    /// Laguna: print per-layer output sum/rms/values (`--dump-layers`).
    pub dump_layers: bool,
    /// Laguna: path to write raw f32 logits (`--dump-logits <OUT.f32>`).
    pub dump_logits: Option<String>,
    /// Laguna: top-K logits to print (`--top <K>`, default `5`).
    pub top: i32,
    /// Laguna: eager reference batch size (`--batch <B>`, default `8`).
    pub batch: i32,
    /// Laguna: resident MMQ prefill micro-batch size (`--ubatch <U>`, default `0`).
    pub ubatch: i32,
    /// Laguna: greedy tokens to generate (`--generate <N>`, default `0`).
    pub generate: i32,
    /// Laguna: KV cache format (`--kv <f16|q8>`, default `true` = `q8`).
    pub kv_q8: bool,
    /// `--kv` was passed (the engines that have no int8 KV say so instead of ignoring it).
    pub kv_explicit: bool,
    /// Laguna: teacher-forced decode tokens at end of prompt (`--force-decode <F>`, default `0`).
    pub force_decode: i32,
    /// Laguna: dump only forced-decode rows to `--dump-logits` (`--dump-decode-only`).
    pub dump_decode_only: bool,
    /// Laguna: verify(n) vs sequential decode self-test (`--verify-test <N>`, default `0`).
    pub verify_test: i32,
    /// Laguna: prompt-lookup speculation draft count (`--spec <K>`, default `0`).
    pub spec: i32,
    /// Laguna: minimum suffix match for prompt-lookup speculation (`--spec-min-match <N>`, default `8`).
    pub spec_min_match: i32,
    /// Laguna: block-sparse decode attention enabled (`--sparse <K[,MIN[,SINK[,RECENT[,BS]]]]|0>`).
    pub sparse: bool,
    /// Laguna: sparse top-K blocks (`default 64`).
    pub sparse_k: i32,
    /// Laguna: sparse minimum context threshold (`default 16384`).
    pub sparse_min: i32,
    /// Laguna: sparse sink tokens (`default 128`).
    pub sparse_sink: i32,
    /// Laguna: sparse recent window tokens (`default 1024`).
    pub sparse_recent: i32,
    /// Laguna: sparse block size (`default 64`).
    pub sparse_bs: i32,
    /// Laguna: EOS token IDs for `--serve` (`--eos-ids <2,24>`, default `[2, 24]`).
    pub eos_ids: Vec<i32>,
    /// Whether `--ctx` / `--max-context` was explicitly passed on the command line.
    pub ctx_explicit: bool,
    /// Whether `--n-layers` was explicitly passed on the command line.
    pub n_layers_explicit: bool,
    /// Whether `--n` / `--max-new` was explicitly passed on the command line.
    pub n_explicit: bool,
    /// talus#47 clustering (27B): `--cluster-role head|worker` (or `TALUS_CLUSTER_ROLE`); None = one process.
    pub cluster_role: Option<String>,
    /// `--cluster-addr HOST:PORT`: where the worker listens / the head connects (default 127.0.0.1:7701).
    pub cluster_addr: Option<String>,
    /// `--cluster-split N`: the worker owns layers N.. (default: the middle, on a 4-layer boundary).
    pub cluster_split: Option<u32>,
    /// `--cluster PATH` (or `TALUS_CLUSTER`): a cluster.toml; with `--node NAME` (or `TALUS_CLUSTER_NODE`) it gives
    /// the role, addresses (LAN ones from its hosts file) and split instead of the three flags above.
    pub cluster_file: Option<String>,
    pub cluster_node: Option<String>,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            model: None,
            max_context: 32768,
            chunk: 0,
            use_graph: true,
            no_graph: false,
            warmup: true,
            no_warmup: false,
            mtp: 0,
            mtp_gguf: None,
            mtp_main_head: false,
            draft_vocab: None,
            dflash_gguf: None,
            drafter: None,
            ngram: 0,
            prompt_cache: 8,
            kv_stash_mib: 8192,
            snap_msg: 1024,
            prompt_cache_disk: None,
            prompt_cache_disk_gib: None,
            prompt_cache_disk_min: None,
            cache_live: 0,
            livecold_test: 0,
            serve: false,
            engine_protocol: false,
            tokens: None,
            max_new: 64,
            force: None,
            dump_full_logits: None,
            bench_pp: 0,
            bench_tg: 0,
            bench_depth: 0,
            bench_window: Vec::new(),
            bench_steps: 32,
            mmvq_bench: None,
            sample_test: false,
            exact_test: 0,
            attn_test: false,
            attn_bench: false,
            expert_vram: -1.0,
            cpu_threads: 0,
            expert_slots: 0,
            hybrid_rows: 128,
            gpu: None,
            greedy: false,
            no_mtp: false,
            mtp_f16: false,
            no_draft_vocab: false,
            attn_w8: false,
            no_v16: false,
            prompt: None,
            serve_port: None,
            help: false,
            ignored_flags: Vec::new(),
            n_layers: talus_model::laguna::K_LAYERS as i32,
            dump_layers: false,
            dump_logits: None,
            top: 5,
            batch: 8,
            ubatch: 0,
            generate: 0,
            kv_q8: true,
            kv_explicit: false,
            force_decode: 0,
            dump_decode_only: false,
            verify_test: 0,
            spec: 0,
            spec_min_match: 8,
            sparse: false,
            sparse_k: 64,
            sparse_min: 16384,
            sparse_sink: 128,
            sparse_recent: 1024,
            sparse_bs: 64,
            eos_ids: talus_model::laguna::DEFAULT_EOS_IDS.to_vec(),
            ctx_explicit: false,
            n_layers_explicit: false,
            n_explicit: false,
            cluster_role: None,
            cluster_addr: None,
            cluster_split: None,
            cluster_file: None,
            cluster_node: None,
        }
    }
}

impl CliConfig {
    /// Parses CLI arguments into a [`CliConfig`].
    pub fn parse_cli_args<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        parse_cli_args(args)
    }
}

/// Parses comma/space-separated integer token IDs or loads them from `@FILE` (or `FILE`),
/// matching `ids_arg` in `q36_engine.cu:3566–3575` and `read_tokens` in `laguna_main.cpp:25–34`.
pub fn parse_ids_arg(spec: &str) -> Result<Vec<i32>, String> {
    if spec.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(path) = spec.strip_prefix('@') {
        let raw = fs::read_to_string(path).map_err(|_| format!("talus-cli: cannot read {path}"))?;
        let normalized: String = raw
            .chars()
            .map(|c| {
                if matches!(c, '\n' | '\r' | '\t') {
                    ' '
                } else {
                    c
                }
            })
            .collect();
        Ok(parse_ids_str(&normalized))
    } else {
        let normalized: String = spec
            .chars()
            .map(|c| {
                if matches!(c, '\n' | '\r' | '\t') {
                    ' '
                } else {
                    c
                }
            })
            .collect();
        let parsed = parse_ids_str(&normalized);
        if parsed.is_empty() && !spec.trim().is_empty() {
            let raw =
                fs::read_to_string(spec).map_err(|_| format!("talus-cli: cannot read {spec}"))?;
            let file_norm: String = raw
                .chars()
                .map(|c| {
                    if matches!(c, '\n' | '\r' | '\t') {
                        ' '
                    } else {
                        c
                    }
                })
                .collect();
            return Ok(parse_ids_str(&file_norm));
        }
        Ok(parsed)
    }
}

fn is_program_name(arg: &str) -> bool {
    let base = arg.rsplit('/').next().unwrap_or(arg);
    matches!(
        base,
        "talus-cli" | "talus-qwen35" | "strata-q36" | "talus-laguna" | "talus-mimo"
    )
}

fn parse_i32_arg(flag: &str, val: &str) -> Result<i32, String> {
    val.trim()
        .parse::<i32>()
        .map_err(|_| format!("talus-cli: invalid integer for {flag}: {val}"))
}

fn parse_f32_arg(flag: &str, val: &str) -> Result<f32, String> {
    val.trim()
        .parse::<f32>()
        .map_err(|_| format!("talus-cli: invalid float for {flag}: {val}"))
}

/// Parses command-line arguments for `talus-cli` / `strata-q36`, supporting both
/// `talus-cli serve ...` and `talus-cli --model ... --serve` / `--engine-protocol`.
pub fn parse_cli_args<I, S>(args: I) -> Result<CliConfig, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let argv: Vec<String> = args.into_iter().map(|s| s.as_ref().to_string()).collect();
    let mut cfg = CliConfig::default();

    let mut i = 0;
    if i < argv.len() && is_program_name(&argv[i]) {
        i += 1;
    }
    if i < argv.len() && argv[i] == "serve" {
        cfg.serve = true;
        cfg.engine_protocol = true;
        i += 1;
    }

    // Pre-scan for `--serve` or `--engine-protocol` so unknown model-specific flags
    // passed before `--serve` are also skipped with a log message (`q36_engine.cu:3400-3406`).
    if argv[i..]
        .iter()
        .any(|a| a == "--serve" || a == "--engine-protocol")
    {
        cfg.serve = true;
        cfg.engine_protocol = true;
    }

    let next_val = |idx: &mut usize, flag: &str| -> Result<String, String> {
        if *idx + 1 >= argv.len() {
            return Err(format!("talus-cli: {flag} requires a value"));
        }
        *idx += 1;
        Ok(argv[*idx].clone())
    };

    while i < argv.len() {
        let a = argv[i].as_str();
        match a {
            "-h" | "--help" => {
                cfg.help = true;
            }
            "--serve" | "--engine-protocol" => {
                cfg.serve = true;
                cfg.engine_protocol = true;
                if a == "--serve" && i + 1 < argv.len() && !argv[i + 1].starts_with('-') {
                    if let Ok(port) = argv[i + 1].trim().parse::<u16>() {
                        if port > 0 {
                            cfg.serve_port = Some(port);
                            i += 1;
                        }
                    }
                }
            }
            "--model" | "--native" | "--gguf" => {
                let v = next_val(&mut i, a)?;
                cfg.model = Some(v);
            }
            "--max-context" | "--ctx" => {
                let v = next_val(&mut i, a)?;
                // `auto` = as if not given (the 27B then sizes its context to the memory)
                if v != "auto" {
                    cfg.max_context = parse_i32_arg(a, &v)?;
                    cfg.ctx_explicit = true;
                }
            }
            "--chunk" | "--prefill" => {
                let v = next_val(&mut i, a)?;
                cfg.chunk = if v == "auto" {
                    0
                } else {
                    parse_i32_arg(a, &v)?
                };
            }
            "--no-graph" | "--no-token-graph" => {
                cfg.use_graph = false;
                cfg.no_graph = true;
            }
            "--no-warmup" => {
                cfg.warmup = false;
                cfg.no_warmup = true;
            }
            "--mtp" => {
                if i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                    let v = next_val(&mut i, a)?;
                    if v.as_bytes().first().is_some_and(|b| b.is_ascii_digit()) {
                        cfg.mtp = parse_i32_arg(a, &v)?;
                    } else if v.len() > 5 && v.ends_with(".gguf") {
                        cfg.mtp_gguf = Some(v);
                    } else {
                        eprintln!(
                            "talus-cli: ignoring --mtp {v} (this family's MTP is the GGUF's own block: --mtp N drafts)"
                        );
                    }
                } else if cfg.mtp == 0 {
                    cfg.mtp = 1;
                }
            }
            "--mtp-n" => {
                let v = next_val(&mut i, a)?;
                cfg.mtp = parse_i32_arg(a, &v)?;
            }
            "--mtp-gguf" => {
                let v = next_val(&mut i, a)?;
                cfg.mtp_gguf = Some(v);
            }
            "--no-mtp" => {
                cfg.no_mtp = true;
            }
            "--mtp-f16" => {
                cfg.mtp_f16 = true;
            }
            "--mtp-main-head" => {
                cfg.mtp_main_head = true;
            }
            "--spec-draft-vocab" | "--draft-vocab" => {
                if i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                    let v = next_val(&mut i, a)?;
                    cfg.draft_vocab = Some(v);
                }
            }
            "--no-draft-vocab" => {
                cfg.no_draft_vocab = true;
            }
            "--dflash-gguf" | "--dflash" => {
                let v = next_val(&mut i, a)?;
                cfg.dflash_gguf = Some(v);
            }
            "--drafter" => {
                let v = next_val(&mut i, a)?;
                cfg.drafter = Some(v);
            }
            "--attn-w8" => {
                cfg.attn_w8 = true;
            }
            "--no-v16" => {
                cfg.no_v16 = true;
            }
            "--prompt" | "-p" => {
                let v = next_val(&mut i, a)?;
                cfg.prompt = Some(v);
            }
            "--ngram" => {
                let v = next_val(&mut i, a)?;
                cfg.ngram = parse_i32_arg(a, &v)?;
            }
            "--prompt-cache" => {
                let v = next_val(&mut i, a)?;
                cfg.prompt_cache = parse_i32_arg(a, &v)?;
            }
            "--kv-stash-mib" => {
                let v = next_val(&mut i, a)?;
                cfg.kv_stash_mib = parse_i32_arg(a, &v)?;
            }
            "--snap-msg" => {
                let v = next_val(&mut i, a)?;
                cfg.snap_msg = parse_i32_arg(a, &v)?;
            }
            "--prompt-cache-disk" => {
                cfg.prompt_cache_disk = Some(next_val(&mut i, a)?);
            }
            "--prompt-cache-disk-gib" => {
                cfg.prompt_cache_disk_gib = Some(next_val(&mut i, a)?);
            }
            "--prompt-cache-disk-min" => {
                cfg.prompt_cache_disk_min = Some(next_val(&mut i, a)?);
            }
            "--cache-live" => {
                let v = next_val(&mut i, a)?;
                cfg.cache_live = parse_i32_arg(a, &v)?;
            }
            "--cluster" => {
                cfg.cluster_file = Some(next_val(&mut i, a)?);
            }
            "--node" => {
                cfg.cluster_node = Some(next_val(&mut i, a)?);
            }
            "--cluster-role" => {
                cfg.cluster_role = Some(next_val(&mut i, a)?);
            }
            "--cluster-addr" => {
                cfg.cluster_addr = Some(next_val(&mut i, a)?);
            }
            "--cluster-split" => {
                let v = next_val(&mut i, a)?;
                cfg.cluster_split = Some(parse_i32_arg(a, &v)?.max(0) as u32);
            }
            "--livecold-test" => {
                let v = next_val(&mut i, a)?;
                cfg.livecold_test = parse_i32_arg(a, &v)?;
            }
            "--tokens" => {
                let v = next_val(&mut i, a)?;
                cfg.tokens = Some(v);
            }
            "--n" | "--max-new" => {
                let v = next_val(&mut i, a)?;
                cfg.max_new = parse_i32_arg(a, &v)?;
                cfg.n_explicit = true;
            }
            "--force" => {
                let v = next_val(&mut i, a)?;
                cfg.force = Some(v);
            }
            "--dump-full-logits" => {
                let v = next_val(&mut i, a)?;
                cfg.dump_full_logits = Some(v);
            }
            "--bench-pp" => {
                let v = next_val(&mut i, a)?;
                cfg.bench_pp = parse_i32_arg(a, &v)?;
            }
            "--bench-tg" => {
                let v = next_val(&mut i, a)?;
                cfg.bench_tg = parse_i32_arg(a, &v)?;
            }
            "--bench-depth" => {
                let v = next_val(&mut i, a)?;
                cfg.bench_depth = parse_i32_arg(a, &v)?;
            }
            "--bench-window" => {
                let v = next_val(&mut i, a)?;
                cfg.bench_window = parse_ids_str(&v);
            }
            "--bench-steps" => {
                let v = next_val(&mut i, a)?;
                cfg.bench_steps = parse_i32_arg(a, &v)?;
            }
            "--mmvq-bench" => {
                let v = next_val(&mut i, a)?;
                cfg.mmvq_bench = Some(v);
            }
            "--sample-test" => {
                cfg.sample_test = true;
            }
            "--exact-test" => {
                let v = next_val(&mut i, a)?;
                cfg.exact_test = parse_i32_arg(a, &v)?;
            }
            "--attn-test" => {
                cfg.attn_test = true;
            }
            "--attn-bench" => {
                cfg.attn_bench = true;
            }
            "--expert-vram" => {
                let v = next_val(&mut i, a)?;
                cfg.expert_vram = parse_f32_arg(a, &v)?;
            }
            "--cpu-threads" | "--pool-workers" => {
                let v = next_val(&mut i, a)?;
                cfg.cpu_threads = parse_i32_arg(a, &v)?;
            }
            "--expert-slots" => {
                let v = next_val(&mut i, a)?;
                cfg.expert_slots = parse_i32_arg(a, &v)?;
            }
            "--hybrid-rows" => {
                let v = next_val(&mut i, a)?;
                cfg.hybrid_rows = parse_i32_arg(a, &v)?;
            }
            "--gpu" => {
                let v = next_val(&mut i, a)?;
                cfg.gpu = Some(parse_i32_arg(a, &v)?);
            }
            "--greedy" => {
                cfg.greedy = true;
            }
            "--n-layers" => {
                let v = next_val(&mut i, a)?;
                cfg.n_layers = parse_i32_arg(a, &v)?;
                cfg.n_layers_explicit = true;
            }
            "--dump-layers" => {
                cfg.dump_layers = true;
            }
            "--dump-logits" => {
                let v = next_val(&mut i, a)?;
                cfg.dump_logits = Some(v);
            }
            "--top" => {
                let v = next_val(&mut i, a)?;
                cfg.top = parse_i32_arg(a, &v)?;
            }
            "--batch" => {
                let v = next_val(&mut i, a)?;
                cfg.batch = parse_i32_arg(a, &v)?;
            }
            "--ubatch" => {
                let v = next_val(&mut i, a)?;
                cfg.ubatch = parse_i32_arg(a, &v)?;
            }
            "--generate" => {
                let v = next_val(&mut i, a)?;
                cfg.generate = parse_i32_arg(a, &v)?;
            }
            "--kv" => {
                let v = next_val(&mut i, a)?;
                match v.as_str() {
                    "q8" | "q8_0" | "int8" => cfg.kv_q8 = true,
                    "f16" => cfg.kv_q8 = false,
                    _ => return Err("talus-laguna: --kv takes f16 or q8".to_string()),
                }
                cfg.kv_explicit = true;
            }
            "--force-decode" => {
                let v = next_val(&mut i, a)?;
                cfg.force_decode = parse_i32_arg(a, &v)?;
            }
            "--dump-decode-only" => {
                cfg.dump_decode_only = true;
            }
            "--verify-test" => {
                let v = next_val(&mut i, a)?;
                cfg.verify_test = parse_i32_arg(a, &v)?;
            }
            "--spec" | "--spec-draft-n-max" => {
                let v = next_val(&mut i, a)?;
                cfg.spec = parse_i32_arg(a, &v)?;
            }
            "--spec-min-match" => {
                let v = next_val(&mut i, a)?;
                cfg.spec_min_match = parse_i32_arg(a, &v)?;
            }
            "--sparse" => {
                let v = next_val(&mut i, a)?;
                let mut f = [64i32, 16384, 128, 1024, 64];
                for (idx, part) in v.split(',').take(5).enumerate() {
                    if !part.trim().is_empty() {
                        f[idx] = parse_i32_arg(a, part)?;
                    }
                }
                cfg.sparse = f[0] > 0;
                cfg.sparse_k = f[0];
                cfg.sparse_min = f[1];
                cfg.sparse_sink = f[2];
                cfg.sparse_recent = f[3];
                cfg.sparse_bs = f[4];
            }
            "--eos-ids" => {
                let v = next_val(&mut i, a)?;
                cfg.eos_ids.clear();
                for part in v.split(',') {
                    if !part.trim().is_empty() {
                        cfg.eos_ids.push(parse_i32_arg(a, part)?);
                    }
                }
            }
            _ if cfg.serve && a.starts_with("--") => {
                let has_val = i + 1 < argv.len() && !argv[i + 1].starts_with("--");
                if has_val {
                    let val = &argv[i + 1];
                    eprintln!("talus-cli: ignoring {a} {val} (not an option of this engine)");
                    cfg.ignored_flags.push(format!("{a} {val}"));
                    i += 1;
                } else {
                    eprintln!("talus-cli: ignoring {a} (not an option of this engine)");
                    cfg.ignored_flags.push(a.to_string());
                }
            }
            _ => {
                return Err(format!("talus-cli: unknown argument: {a} (try --help)"));
            }
        }
        i += 1;
    }

    Ok(cfg)
}

/// Prints the `talus-cli` usage message (`q36_engine.cu:3310–3347`).
pub fn print_usage() {
    println!(
        "\
Usage: talus-cli [serve] --model PATH.gguf [options]
The engine follows the model: Qwen3.8-27B, MiMo-V2.6-Flash and Laguna-S-2.1 take the options below;
GLM-5.3-Flash (picked from its GGUF) and Qwen3.8-Flash-Next (--pack DIR --native GGUF) have their own.
Every model's shipped arguments: docs/MODELS.md. Environment switches: TALUS_X also sets an unset STRATA_X.
  --model, --native PATH     the model's GGUF (the first shard of a split GGUF)
  --max-context, --ctx N     context length (default: 32768)
  --chunk, --prefill N|auto  prefill chunk (default: auto)
  --no-graph                 do not capture the single-token HIP graph
  --no-warmup                skip the warmup token at startup
  --mtp N|FILE.gguf          speculative decoding with N drafts or separate MTP GGUF
  --mtp-gguf PATH            separate GGUF whose blk.mtp.* is used on top of --model
  --mtp-main-head            project MTP hidden states with the main model's output.weight
  --spec-draft-vocab PATH    subset draft vocabulary GGUF for the MTP head
  --dflash-gguf PATH         DFlash block-diffusion drafter GGUF
  --drafter mtp|dflash|auto  drafter selector when --dflash-gguf and --mtp N are both set (env kill switch: STRATA_DRAFTER=mtp)
  --ngram N                  n-gram draft lookup order before MTP (default: 0 = off)
  --prompt-cache N           state snapshots kept by --serve across turns (default: 8)
  --kv-stash-mib M           device memory budget in MiB for KV stashes (default: 8192)
  --snap-msg N               snapshot before last user/tool message after >= N tokens (default: 1024)
  --prompt-cache-disk DIR    keep prompt-cache entries on disk across restarts (27B, Laguna serve); DIR, auto, or
                             off/0 (env TALUS_PCACHE_DISK; default: on, $XDG_CACHE_HOME/talus/pcache or
                             ~/.cache/talus/pcache; never fills the disk past max(20 GiB, 10 %) free)
  --prompt-cache-disk-gib N  disk budget in GiB, least recently used entries go first (default: 32)
  --prompt-cache-disk-min N  shortest prompt worth a disk entry (default: 1024)
  --cache-live N             1 = continue from live decoded cells; 0 = grid-aligned cold-exact policy (default: 0)
  --livecold-test N          verify warm turns match cold greedy output for N tokens
  --serve, --engine-protocol run the line-oriented GEN/STOP/T/DONE protocol on stdin/stdout
  --tokens IDS|@FILE         comma/space-separated token IDs (or @path) for one-shot generation
  --n, --max-new N           tokens to generate with --tokens (default: 64)
  --force IDS|@FILE          append forced token IDs one by one after --tokens
  --dump-full-logits PATH    write float32[1 + |force|, V] logits to PATH
  --bench-pp N               prefill benchmark of N tokens
  --bench-tg N               decode benchmark of N greedy tokens
  --bench-depth N            prior context depth before --bench-pp / --bench-tg
  --bench-window W1,W2,...   verify-window benchmark for each W
  --bench-steps N            timed iterations per --bench-window size (default: 32)
  --mmvq-bench FILTER|-      time every unique weight shape's MMVQ kernel
  --sample-test              CPU self-test of make_dist vs sample() and speculative draw
  --exact-test N             verify T=4 batch step equals four T=1 steps at pos N
  --attn-test                WMMA decode/verify attention vs scalar reference
  --attn-bench               WMMA attention microbenchmark
  --cluster-role head|worker 27B pipeline split across two processes (talus#47; off by default)
  --cluster-addr HOST:PORT   worker listen / head connect address (default 127.0.0.1:7701)
  --cluster-split N          the worker owns layers N.. (default: the middle, on a 4-layer boundary)
  --cluster PATH --node NAME cluster.toml (role, addresses from its hosts file, split) for this node"
    );
}

fn make_layer_state_view(eng: &mut Q36Engine) -> Q36LayerStateView {
    eng.flush_gdn();
    let g = &eng.model().g;
    let na = eng.n_attn_state as usize;
    let (dfl_k_ring, dfl_v_ring, dfl_ring, dfl_keep, dfl_kv_dim, dfl_block) =
        if eng.use_dflash && eng.df.ready {
            (
                eng.df.k_ring.clone(),
                eng.df.v_ring.clone(),
                eng.df.ring(),
                eng.df.keep(),
                eng.df.mdl.n_kv * eng.df.mdl.head_dim,
                eng.df.block(),
            )
        } else {
            (Vec::new(), Vec::new(), 0, 0, 0, 0)
        };
    Q36LayerStateView {
        ssm_d_conv: 4,
        ssm_conv_channels: g.ssm_qkv_dim() as i32,
        ssm_v_heads: g.ssm_time_step_rank as i32,
        ssm_state_size: g.ssm_state_size as i32,
        n_head_kv: g.n_head_kv as i32,
        head_dim: g.head_dim as i32,
        n_embd: g.n_embd as i32,
        ctx: eng.ctx_len,
        vt_ctx: eng.vt_ctx,
        has_mtp: eng.model().has_mtp,
        conv: eng.conv.clone(),
        s: eng.S.clone(),
        kc: eng.kc[..na].iter().map(|&p| p.cast::<c_void>()).collect(),
        vc: eng.vc[..na].iter().map(|&p| p.cast::<c_void>()).collect(),
        vtc: eng.vtc[..na].iter().map(|&p| p.cast::<c_void>()).collect(),
        hlast: eng.h_mtp,
        gctl: eng.gctl,
        gdn_pending: 0,
        pos: eng.n_past as i64,
        st: eng.s_main,
        dfl_k_ring,
        dfl_v_ring,
        dfl_ring,
        dfl_keep,
        dfl_kv_dim,
        dfl_block,
        dctx_lo: eng.dctx.lo(),
        dctx_hi: eng.dctx.hi(),
    }
}

fn sync_layer_state_view_back(eng: &mut Q36Engine, view: &Q36LayerStateView) {
    eng.n_past = view.pos as i32;
    eng.gdn_pending = view.gdn_pending;
    eng.gdn_valid = true;
    eng.dctx.set_bounds(view.dctx_lo, view.dctx_hi);
    // a restore copies f16 cells only; rebuild their int8 copy (no-op unless TALUS_Q27_KV8=1)
    let n = eng.n_past;
    eng.kv8_rebuild(0, n);
}

/// Real `Q36Engine` stdio protocol backend matching `serve(Engine& E)` in `q36_engine.cu:2617-3120`.
pub struct Q36StdioBackend {
    pub eng: Q36Engine,
    pub pc: Q36PromptCache,
    pub live: Vec<i32>,
    pub live_pf: usize,
    pub cache_live: bool,
    pub ng: NgramMod,
    pub ngram_k: i32,
    pub n_spec: i32,
    pub snap_msg: i32,
    /// : spacing of the prefill snapshot ladder in tokens (`TALUS_PCACHE_LADDER`, 0 = off), see [`ladder_marks`].
    pub ladder: usize,
    /// Disk tier of the prompt cache (`--prompt-cache-disk`), `None` when off.
    pub disk: Option<talus_state::disk::DiskCache>,
    /// E6: the model file (its tokenizer backs json_schema masks) and the lazily built grammar context.
    pub model_path: Option<std::path::PathBuf>,
    pub grammar: Option<grammar::GrammarCtx>,
    /// K31: whether the after-reply catch-up has been paying off (`TALUS_Q27_CATCHUP_SKIP`).
    catch_up_trigger: CatchUpTrigger,
    /// K31: when the last reply's DONE went out (the log line with the client's gap).
    done_at: Option<Instant>,
}

/// K31: tracks whether the requests that follow a reply continue it. After each reply that crossed a grid point the
/// first chunk the catch-up would read is remembered, whether or not the catch-up ran, so the check costs no GPU work
/// and a paused catch-up comes back at the first request that would have used it.
#[derive(Debug, Default)]
struct CatchUpTrigger {
    /// Prompt + reply up to the end of the catch-up's first chunk; empty = nothing to check.
    expect: Vec<i32>,
    /// Requests in a row that did not continue the reply far enough to use that chunk.
    misses: u32,
}

impl CatchUpTrigger {
    /// A reply crossed a grid point: `prefix` ends where the first catch-up chunk would end.
    fn arm(&mut self, prefix: &[i32]) {
        self.expect.clear();
        self.expect.extend_from_slice(prefix);
    }

    /// The next request arrived. `Some(true)` = it would use the first chunk, `None` = nothing was armed.
    fn observe(&mut self, ids: &[i32]) -> Option<bool> {
        if self.expect.is_empty() {
            return None;
        }
        let hit = ids.starts_with(&self.expect);
        self.expect.clear();
        self.misses = if hit {
            0
        } else {
            self.misses.saturating_add(1)
        };
        Some(hit)
    }

    /// `skip` misses in a row pause the catch-up (0 = never).
    fn paused(&self, skip: u32) -> bool {
        skip > 0 && self.misses >= skip
    }
}

/// default ladder spacing (tokens). A long prompt also gets recurrent-state snapshots at these grid points, so a
/// later request that shares only part of it (another tail, an edited or dropped message past the last boundary,
/// a sub-agent on the same long prefix) resumes from the deepest point at or before where it diverges instead of
/// from scratch: the live KV still holds that prefix, so such a restore copies one snapshot and no KV.
pub const DEFAULT_LADDER_TOKENS: usize = 16384;
/// Ladder points per request at most; the spacing widens on longer prompts. They share the turn-end snapshot pool
/// (`--prompt-cache`, LRU), so this keeps a long prompt from evicting every other conversation's snapshot.
pub const LADDER_MAX_POINTS: usize = 3;

/// `TALUS_PCACHE_LADDER` (tokens, `0` = off), default [`DEFAULT_LADDER_TOKENS`]; off without a snapshot pool.
pub fn ladder_spacing(max_snaps: usize) -> usize {
    if max_snaps < 2 {
        return 0;
    }
    std::env::var("TALUS_PCACHE_LADDER")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_LADDER_TOKENS)
}

/// The prefill stops of one request under the grid-aligned policy: the message-boundary snapshot `cg`, the grid
/// end `g_end` and ) up to `max_pts` ladder points strictly between `at` (the restored length) and `g_end`, on
/// multiples of `grid`, at least `ladder` apart and at least half a step from `cg` and `g_end`. Sorted, unique, all `> at`; `cg` is kept even when it is 0
/// (the caller skips it then, as before).
pub fn ladder_marks(
    at: usize,
    cg: usize,
    g_end: usize,
    grid: usize,
    ladder: usize,
    max_pts: usize,
) -> Vec<usize> {
    let mut marks = vec![cg, g_end];
    let grid = grid.max(1);
    if ladder > 0 && max_pts > 0 && g_end > at {
        let span = g_end - at;
        let step = ladder.max(span.div_ceil(max_pts + 1)).div_ceil(grid) * grid;
        let mut m = (at / step + 1) * step;
        let mut n = 0;
        while m < g_end && n < max_pts {
            // not within half a step of another stop (a snapshot there would add little)
            if m > at && m.abs_diff(cg) >= step / 2 && g_end - m >= step / 2 {
                marks.push(m);
                n += 1;
            }
            m += step;
        }
    }
    marks.sort_unstable();
    marks.dedup();
    marks
}

/// End tokens of the Q36 decode loop (also the tokens a json_schema request may end with).
const Q36_EOS: [i32; 4] = [248044, 248046, 151643, 151645];

/// E6: the next token of a constrained request from the engine's host logits (-1 = the schema cannot continue).
fn grammar_next(
    eng: &Q36Engine,
    m: &talus_text::json_grammar::Matcher,
    cmd: &GenCommand,
    samp: &Sampling,
    hist: &[i32],
    rng: &mut SplitMix64,
    scratch: &mut Vec<f32>,
) -> i32 {
    let sl = unsafe { std::slice::from_raw_parts(eng.logits_h, eng.n_vocab() as usize) };
    grammar::pick(m, sl, !cmd.needs_logits(), scratch, |x| {
        sample(x, samp, hist, rng)
    })
    .unwrap_or(-1)
}

impl Q36StdioBackend {
    pub fn new(mut eng: Q36Engine, cfg: &CliConfig) -> Self {
        let max_snaps = cfg.prompt_cache.max(0) as usize;
        let stash_budget = (cfg.kv_stash_mib.max(0) as usize) << 20;
        let snap_msg = cfg.snap_msg.max(0);
        let mut pc = Q36PromptCache::new();
        pc.max_snaps = max_snaps;
        pc.max_msg = if max_snaps > 0 { 4 } else { 0 };
        pc.stash_budget = stash_budget;
        // : no new KV stash below the pressure guard's shedding threshold (8 GiB floor + 512 MiB)
        pc.mem_floor = talus_mem::pressure::global()
            .map(|p| p.config().threshold())
            .unwrap_or(talus_state::disk::DEFAULT_MEM_FLOOR_BYTES + (512 << 20));
        eng.cl_pc_init(&mut pc); // talus#47: shard the cache over the cluster stages (no-op on one process)
        let n_spec = if eng.use_dflash && eng.df.ready {
            if eng.dp.uses_mtp() && cfg.mtp > 0 {
                cfg.mtp
            } else {
                eng.df.n_draft_file()
            }
        } else if eng.model().has_mtp {
            cfg.mtp
        } else {
            0
        };
        if eng.mtp_d == 0 && n_spec > 0 {
            eng.mtp_d = n_spec;
        }
        let ng_n = if cfg.ngram > 1 {
            cfg.ngram as usize
        } else {
            24
        };
        let mut me = Self {
            eng,
            pc,
            live: Vec::new(),
            live_pf: 0,
            catch_up_trigger: CatchUpTrigger::default(),
            done_at: None,
            cache_live: cfg.cache_live != 0,
            ng: NgramMod::with_n(ng_n),
            ngram_k: cfg.ngram,
            n_spec,
            snap_msg,
            ladder: ladder_spacing(max_snaps),
            disk: None,
            model_path: cfg.model.as_ref().map(std::path::PathBuf::from),
            grammar: None,
        };
        me.disk = disk_tier::open(&mut me, cfg);
        me
    }
}

/// #40 memory pressure: the 27B's rebuildable memory is its prompt cache (recurrent snapshots, then the KV stashes
/// of the RAM tier). Restores are grid-aligned and cold-exact, so dropping them only costs prompt time.
impl talus_mem::pressure::Shed for Q36StdioBackend {
    fn shed(&mut self, class: talus_mem::pressure::ShedClass) -> u64 {
        use talus_mem::pressure::ShedClass;
        match class {
            ShedClass::PromptSnapshots => self.pc.shed_snaps(),
            ShedClass::PromptCacheRam => self.pc.shed_stashes(),
            ShedClass::TransientArenas => talus_mem::pressure::trim_host_heap(),
            ShedClass::DecodeGraphs => 0,
        }
    }
}

impl EngineProtocolBackend for Q36StdioBackend {
    fn max_context(&self) -> i32 {
        self.eng.max_seq()
    }

    fn n_vocab(&self) -> i32 {
        self.eng.n_vocab()
    }

    // E1/E2 (talus-bench): host log-softmax over D2H copies of the target logits rows; not on a cluster stage
    fn supports_logprobs(&self) -> bool {
        self.eng.cl.is_none()
    }

    fn supports_score(&self) -> bool {
        self.eng.cl.is_none()
    }

    fn handle_score(
        &mut self,
        top_k: usize,
        prompt: &[i32],
        cont: &[i32],
        cancel: &CancelToken,
        out: &mut dyn Write,
    ) -> io::Result<()> {
        q36_score(self, top_k, prompt, cont, cancel, out)
    }

    fn handle_gen(
        &mut self,
        cmd: &GenCommand,
        cancel: &CancelToken,
        out: &mut dyn Write,
    ) -> io::Result<()> {
        if !self.eng.is_cluster_head() {
            return self.handle_gen_one(cmd, cancel, out);
        }
        // talus#47 Phase 2: a lost worker fails this request with `ERR stage_unavailable` (HTTP 503) instead of
        // ending the engine; the link is re-established before the next request (the worker keeps its weights).
        if !self.eng.cluster_link_up() {
            if let Err(e) = self.cluster_recover(2) {
                return writeln!(out, "ERR stage_unavailable: {e}").and_then(|_| out.flush());
            }
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.handle_gen_one(cmd, cancel, out)
        }));
        match r {
            Ok(v) => v,
            Err(p) => {
                let why = p
                    .downcast_ref::<talus_model::qwen35::cluster::ClusterLost>()
                    .map(|c| c.0.clone())
                    .or_else(|| p.downcast_ref::<String>().cloned())
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "panic".into());
                eprintln!("talus-serve: request failed on the cluster link: {why}");
                writeln!(out, "ERR stage_unavailable: {why}")?;
                out.flush()?;
                if let Err(e) = self.cluster_recover(5) {
                    eprintln!(
                        "talus-serve: worker not back yet ({e}); retrying on the next request"
                    );
                }
                Ok(())
            }
        }
    }
}

impl Q36StdioBackend {
    /// After a lost link: reconnect (waiting up to `wait_s`), and drop the prompt-cache state (the worker's shard
    /// went with its session) and the live sequence.
    fn cluster_recover(&mut self, wait_s: u64) -> Result<(), String> {
        self.eng.cluster_reconnect(wait_s)?;
        let mut pc = Q36PromptCache::new();
        pc.max_snaps = self.pc.max_snaps;
        pc.max_msg = self.pc.max_msg;
        pc.stash_budget = self.pc.stash_budget;
        self.eng.cl_pc_init(&mut pc);
        self.pc = pc;
        self.live.clear();
        self.live_pf = 0;
        Ok(())
    }

    fn handle_gen_one(
        &mut self,
        cmd: &GenCommand,
        cancel: &CancelToken,
        out: &mut dyn Write,
    ) -> io::Result<()> {
        let ids = &cmd.prompt_ids;
        let cu_hit = self.catch_up_trigger.observe(ids);
        if let (Some(done_at), Some(arrived)) =
            (self.done_at.take(), talus_sched::last_line_arrival())
        {
            eprintln!(
                "talus-serve: request arrived {:.0} ms after the last reply, waited {:.0} ms for after-reply work{}",
                arrived.saturating_duration_since(done_at).as_secs_f64() * 1000.0,
                arrived.elapsed().as_secs_f64() * 1000.0,
                match cu_hit {
                    Some(true) => ", continues the reply",
                    Some(false) => ", does not continue the reply",
                    None => "",
                }
            );
        }
        let max_ctx = self.eng.max_seq() as usize;
        if ids.is_empty() || ids.len() >= max_ctx || cmd.max_new <= 0 {
            let done = DoneStats {
                produced: 0,
                prompt_tokens: ids.len(),
                prompt_ms: 0.0,
                decode_ms: 0.0,
                finish: if ids.len() >= max_ctx {
                    "length"
                } else {
                    "stop"
                },
                accepted: 0,
                offered: 0,
                reuse: 0,
                fresh: 0,
            };
            return write_done_line(out, &done);
        }

        // #40: admission near the MemAvailable floor (sheds the prompt cache first; still short -> this request fails)
        if let Err(e) = talus_mem::pressure::admit(self) {
            writeln!(out, "{}", talus_mem::pressure::err_line(&e.to_string()))?;
            return out.flush();
        }

        // E6: a json_schema request is compiled before the prompt is read, so a bad schema fails fast
        let mut gm: Option<talus_text::json_grammar::Matcher> = None;
        if let Some(ref js) = cmd.json_schema {
            if self.grammar.is_none() {
                let n_vocab = self.eng.n_vocab().max(0) as usize;
                let loaded = match self.model_path.as_deref() {
                    Some(p) => grammar::GrammarCtx::load(p, n_vocab, &Q36_EOS),
                    None => Err("json_schema: no model path for the tokenizer".to_string()),
                };
                match loaded {
                    Ok(g) => self.grammar = Some(g),
                    Err(e) => {
                        writeln!(out, "ERR {e}")?;
                        return out.flush();
                    }
                }
            }
            match self.grammar.as_mut().map(|g| g.matcher(js)) {
                Some(Ok(m)) => gm = Some(m),
                Some(Err(e)) => {
                    writeln!(out, "ERR {e}")?;
                    return out.flush();
                }
                None => {}
            }
        }

        let cache_enabled = self.pc.max_snaps > 0 || self.pc.max_msg > 0;
        let gridc = !self.cache_live;
        let grid = (self.eng.max_batch.max(1)) as usize;
        let t0 = Instant::now();
        let mut from = "live";
        let reuse = if !cmd.use_cache {
            if cache_enabled && !self.live.is_empty() {
                let view = make_layer_state_view(&mut self.eng);
                self.pc.stash_kv_with_pf(&view, &self.live, 0, self.live_pf);
                self.eng.cl_pc_stash(&self.live, 0, self.live_pf);
            }
            self.eng.reset();
            self.live.clear();
            self.live_pf = 0;
            from = "off (cache=0)";
            0
        } else if !gridc
            && !self.live.is_empty()
            && self.live.len() < ids.len()
            && ids.starts_with(&self.live)
        {
            write_resume_line(out, self.live.len())?;
            self.live.len()
        } else if cache_enabled {
            let lcp = self
                .live
                .iter()
                .zip(ids.iter())
                .take_while(|(a, b)| a == b)
                .count();
            let mut view = make_layer_state_view(&mut self.eng);
            if !self.live.is_empty() {
                self.pc
                    .stash_kv_with_pf(&view, &self.live, lcp, self.live_pf);
                self.eng.cl_pc_stash(&self.live, lcp, self.live_pf);
            }
            if self.disk.is_some() {
                disk_tier::restore_into_ram(self, &mut view, ids, gridc);
            }
            // (restore split): a live cell is only cold-exact when its whole 512-grid chunk matches: a chunk's cells can
            // depend, in the last bits, on later tokens of that chunk. Keep live cells up to the last grid point inside
            // the common prefix; the rest comes from the stash (bench27 edit,edit replay: 17 common tokens kept in place
            // changed the restored prompt's first logits; with this, warm is bitwise cold).
            let (live_pf, pf_only) = if gridc {
                (self.live_pf.min(lcp) / grid * grid, true)
            } else {
                (usize::MAX, false)
            };
            let (r, r_from) = self
                .pc
                .restore_with_pf(&mut view, ids, &self.live, live_pf, pf_only);
            self.eng.cl_pc_restore(ids, &self.live, live_pf, pf_only, r);
            if r > 0 {
                sync_layer_state_view_back(&mut self.eng, &view);
                self.live.clear();
                self.live.extend_from_slice(&ids[..r]);
                from = r_from;
                write_resume_line(out, r)?;
                r
            } else {
                self.eng.reset();
                self.live.clear();
                self.live_pf = 0;
                from = "none";
                0
            }
        } else {
            self.eng.reset();
            self.live.clear();
            self.live_pf = 0;
            from = "none";
            0
        };
        if reuse == 0 && from == "live" {
            from = "none";
        }
        eprintln!(
            "talus-serve: prompt {} tokens, cached {} ({}), {:.1} ms to restore",
            ids.len(),
            reuse,
            from,
            t0.elapsed().as_secs_f64() * 1000.0
        );

        let fresh = ids.len().saturating_sub(reuse);

        let mut cut = 0usize;
        if cache_enabled && self.snap_msg > 0 && (fresh as i32) >= self.snap_msg {
            for idx in ((reuse + 1)..ids.len()).rev() {
                if ids[idx] == 248045 && (idx + 1 >= ids.len() || ids[idx + 1] != 74455) {
                    cut = idx;
                    break;
                }
            }
            if cut > 0 && ((cut - reuse) as i32) < self.snap_msg {
                cut = 0;
            }
        }

        if cancel.is_cancelled() {
            self.live.clear();
            self.live_pf = 0;
            let done = DoneStats {
                produced: 0,
                prompt_tokens: ids.len(),
                prompt_ms: t0.elapsed().as_secs_f64() * 1000.0,
                decode_ms: 0.0,
                finish: "cancel",
                accepted: 0,
                offered: 0,
                reuse,
                fresh: 0,
            };
            return write_done_line(out, &done);
        }

        let mut pp_err: Option<io::Error> = None;
        if gridc {
            let mut at = reuse;
            let g_end = (ids.len() / grid) * grid;
            let cg = (cut / grid) * grid;
            let ladder = if cache_enabled { self.ladder } else { 0 };
            for mk in ladder_marks(at, cg, g_end, grid, ladder, LADDER_MAX_POINTS) {
                if mk <= at || (mk == cg && mk == g_end && cut == 0) {
                    continue;
                }
                let base = (at - reuse) as i64;
                self.eng.process_with_callback(&ids[at..mk], true, |done| {
                    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    if let Err(e) =
                        write_pp_line(out, base + (done as i64), fresh as i64, elapsed_ms)
                    {
                        pp_err = Some(e);
                    }
                });
                if let Some(e) = pp_err {
                    return Err(e);
                }
                self.live.extend_from_slice(&ids[at..mk]);
                at = mk;
                if cache_enabled && (self.eng.n_past as usize) == self.live.len() {
                    let mut view = make_layer_state_view(&mut self.eng);
                    self.pc.save(&mut view, &self.live, mk == cg && cut > 0);
                    self.eng.cl_pc_save(&self.live, mk == cg && cut > 0);
                }
            }
            if at < ids.len() {
                let base = (at - reuse) as i64;
                self.eng.process_with_callback(&ids[at..], true, |done| {
                    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    if let Err(e) =
                        write_pp_line(out, base + (done as i64), fresh as i64, elapsed_ms)
                    {
                        pp_err = Some(e);
                    }
                });
                if let Some(e) = pp_err {
                    return Err(e);
                }
                self.live.extend_from_slice(&ids[at..]);
            }
        } else if cut > 0 {
            self.eng
                .process_with_callback(&ids[reuse..cut], true, |done| {
                    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    if let Err(e) = write_pp_line(out, done as i64, fresh as i64, elapsed_ms) {
                        pp_err = Some(e);
                    }
                });
            if let Some(e) = pp_err {
                return Err(e);
            }
            self.live.extend_from_slice(&ids[reuse..cut]);
            if cache_enabled && (self.eng.n_past as usize) == self.live.len() {
                let mut view = make_layer_state_view(&mut self.eng);
                self.pc.save(&mut view, &self.live, true);
                self.eng.cl_pc_save(&self.live, true);
            }
            let base = (cut - reuse) as i64;
            self.eng.process_with_callback(&ids[cut..], true, |done| {
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                if let Err(e) = write_pp_line(out, base + (done as i64), fresh as i64, elapsed_ms) {
                    pp_err = Some(e);
                }
            });
            if let Some(e) = pp_err {
                return Err(e);
            }
            self.live.extend_from_slice(&ids[cut..]);
        } else if fresh > 0 {
            self.eng.process_with_callback(&ids[reuse..], true, |done| {
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                if let Err(e) = write_pp_line(out, done as i64, fresh as i64, elapsed_ms) {
                    pp_err = Some(e);
                }
            });
            if let Some(e) = pp_err {
                return Err(e);
            }
            self.live.extend_from_slice(&ids[reuse..]);
        }

        if !gridc && fresh > 0 && cache_enabled && (self.eng.n_past as usize) == self.live.len() {
            let mut view = make_layer_state_view(&mut self.eng);
            self.pc.save(&mut view, &self.live, false);
            self.eng.cl_pc_save(&self.live, false);
        }
        self.live_pf = self.live.len();

        let prompt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let samp = Sampling {
            temp: cmd.temperature,
            top_p: cmd.top_p,
            min_p: cmd.min_p,
            top_k: cmd.top_k,
            rep_pen: cmd.penalty_repeat,
            pres_pen: cmd.penalty_present,
            freq_pen: cmd.penalty_freq,
            rep_last_n: cmd.penalty_last_n,
            seed: cmd.seed,
        };
        let mut rng = SplitMix64::new(samp.seed);
        let mut hist = ids.clone();
        if self.ngram_k > 0 {
            self.ng.reset();
        }

        let eos_list: Vec<i32> = if cmd.ignore_eos {
            Vec::new()
        } else {
            vec![248044, 248046, 151643, 151645]
        };
        let is_eos = |t: i32| !cmd.ignore_eos && eos_list.contains(&t);

        self.eng.st_rounds = 0;
        self.eng.st_decide_ms = 0.0;
        let d0 = Instant::now();
        let mut produced = 0usize;
        let mut offered = 0i64;
        let mut accepted = 0i64;
        let finish: &'static str;

        let mut gscratch: Vec<f32> = Vec::new();
        let mut nx = if let Some(ref m) = gm {
            grammar_next(&self.eng, m, cmd, &samp, &hist, &mut rng, &mut gscratch)
        } else if cmd.needs_logits() {
            let sl = unsafe {
                std::slice::from_raw_parts(self.eng.logits_h, self.eng.n_vocab() as usize)
            };
            sample(sl, &samp, &hist, &mut rng)
        } else {
            self.eng.greedy_from_host_logits()
        };

        // E6: constrained requests decode without drafts (the masked target picks every token)
        let spec_active =
            gm.is_none() && self.n_spec > 0 && (self.eng.model().has_mtp || self.eng.use_dflash);
        let max_new = cmd.max_new.max(0) as usize;

        if spec_active && max_new > 1 && (self.eng.n_past + self.n_spec + 2) < self.eng.max_seq() {
            let mut spec_out = Vec::new();
            let mut io_err: Option<io::Error> = None;
            let mut stop_hit = false;
            let mut cancel_hit = false;
            let ng_opt = if self.ngram_k > 0 {
                Some(&mut self.ng)
            } else {
                None
            };
            self.eng.lp_k = cmd.logprobs;
            self.eng.spec_run_with_callback(
                self.n_spec,
                max_new as i32,
                &eos_list,
                nx,
                &samp,
                &mut hist,
                &mut spec_out,
                &mut offered,
                &mut accepted,
                ng_opt,
                |tok, lp| {
                    if let Some(lp) = lp {
                        if let Err(e) = writeln!(out, "{}", talus_spec::logprobs::format_l_line(lp))
                        {
                            io_err = Some(e);
                            return false;
                        }
                    }
                    if let Err(e) = write_token_line(out, tok) {
                        io_err = Some(e);
                        return false;
                    }
                    if is_eos(tok) {
                        stop_hit = true;
                        return false;
                    }
                    if cancel.is_cancelled() {
                        cancel_hit = true;
                        return false;
                    }
                    true
                },
            );
            self.eng.lp_k = None;
            if let Some(e) = io_err {
                return Err(e);
            }
            produced += spec_out.len();
            let live_target = (self.eng.n_past.max(0) as usize).min(hist.len());
            if live_target > self.live.len() {
                self.live
                    .extend_from_slice(&hist[self.live.len()..live_target]);
            }
            if stop_hit {
                finish = "stop";
            } else if cancel_hit {
                finish = "cancel";
            } else {
                finish = "length";
            }
        } else if let Some(m) = gm.as_mut() {
            loop {
                if nx < 0 {
                    eprintln!(
                        "talus-serve: json_schema: no token is allowed here; ending the reply"
                    );
                    finish = "stop";
                    break;
                }
                write_token_line(out, nx)?;
                produced += 1;
                hist.push(nx);
                if is_eos(nx) || !m.advance(nx as u32) {
                    finish = "stop";
                    break;
                }
                if m.finished() {
                    // the value is complete: only an end token could follow
                    finish = "stop";
                    break;
                }
                if produced >= max_new {
                    finish = "length";
                    break;
                }
                if cancel.is_cancelled() {
                    finish = "cancel";
                    break;
                }
                if (self.eng.n_past + 1) >= self.eng.max_seq() {
                    finish = "length";
                    break;
                }
                let one = [nx];
                self.eng.process(&one, true);
                self.live.push(nx);
                nx = grammar_next(&self.eng, m, cmd, &samp, &hist, &mut rng, &mut gscratch);
            }
            let (computed, reused, mask_ms) = m.stats();
            eprintln!(
                "talus-serve: json_schema: {computed} masks computed ({mask_ms:.1} ms), {reused} reused, no drafts"
            );
        } else {
            loop {
                if let Some(k) = cmd.logprobs {
                    // E1: nx came from the host logits row (the prompt's last row or the last decode step)
                    let lp = self.eng.host_row_logprobs(nx, k);
                    writeln!(out, "{}", talus_spec::logprobs::format_l_line(&lp))?;
                }
                write_token_line(out, nx)?;
                produced += 1;
                hist.push(nx);
                if self.ngram_k > 0 {
                    self.ng.feed(nx);
                }
                if is_eos(nx) {
                    finish = "stop";
                    break;
                }
                if produced >= max_new {
                    finish = "length";
                    break;
                }
                if cancel.is_cancelled() {
                    finish = "cancel";
                    break;
                }
                if (self.eng.n_past + 1) >= self.eng.max_seq() {
                    finish = "length";
                    break;
                }
                let one = [nx];
                self.eng.process(&one, true);
                self.live.push(nx);
                nx = if cmd.needs_logits() {
                    let sl = unsafe {
                        std::slice::from_raw_parts(self.eng.logits_h, self.eng.n_vocab() as usize)
                    };
                    sample(sl, &samp, &hist, &mut rng)
                } else {
                    self.eng.greedy_from_host_logits()
                };
            }
        }

        let decode_ms = d0.elapsed().as_secs_f64() * 1000.0;
        if let Some(c) = self.eng.copy.as_mut() {
            let s = c.take_stats();
            if s.windows > 0 {
                eprintln!(
                    "talus copy: {} windows, {} of {} copied drafts accepted",
                    s.windows, s.accepted, s.proposed
                );
            }
        }
        eprintln!(
            "talus-serve: {} tokens in {:.1} ms ({:.2} tok/s), {} rounds ({:.2} ms each), host decisions {:.1} ms, drafts {}/{}, prompt {} ({} read, {:.1} ms)",
            produced,
            decode_ms,
            (produced as f64) / (decode_ms / 1000.0).max(1e-9),
            self.eng.st_rounds,
            if self.eng.st_rounds > 0 {
                decode_ms / (self.eng.st_rounds as f64)
            } else {
                0.0
            },
            self.eng.st_decide_ms,
            accepted,
            offered,
            ids.len(),
            fresh,
            prompt_ms
        );
        if let Some((hops, hop_s)) = self.eng.cluster_stats() {
            eprintln!(
                "talus-cluster: {hops} hops so far, mean {:.3} ms host time per hop",
                if hops > 0 {
                    1000.0 * hop_s / hops as f64
                } else {
                    0.0
                }
            );
        }
        if self.eng.use_dflash && self.eng.dp.mode() == DrafterPick::K_AUTO {
            eprintln!(
                "talus-serve: drafter windows DFlash2 {}, MTP {} (fallback {})",
                self.eng.dp.n_dfl, self.eng.dp.n_mtp, self.eng.dp.n_fallback
            );
            self.eng.dp.n_dfl = 0;
            self.eng.dp.n_mtp = 0;
            self.eng.dp.n_fallback = 0;
        }
        self.eng.st_rounds = 0;
        self.eng.st_decide_ms = 0.0;
        if !gridc
            && cache_enabled
            && (self.eng.n_past as usize) == self.live.len()
            && !self.live.is_empty()
        {
            let mut view = make_layer_state_view(&mut self.eng);
            self.pc.save(&mut view, &self.live, false);
            self.eng.cl_pc_save(&self.live, false);
        } else if !gridc && cache_enabled {
            eprintln!(
                "talus-serve: no snapshot (engine at {}, {} tokens held)",
                self.eng.n_past,
                self.live.len()
            );
        }

        let done = DoneStats {
            produced: produced as i64,
            prompt_tokens: ids.len(),
            prompt_ms,
            decode_ms,
            finish,
            accepted,
            offered,
            reuse,
            fresh: fresh as i64,
        };
        // Taken before DONE: a stop signal that follows the reply waits for this turn's disk write.
        let _save_guard =
            (self.disk.is_some() && cache_enabled).then(talus_state::disk::SaveGuard::new);
        write_done_line(out, &done)?;
        self.done_at = Some(Instant::now());
        // w2: read the reply up to its last grid point with the prompt kernels, so the next turn restores past it
        if gridc && cache_enabled && cmd.use_cache && finish != "cancel" && catch_up_enabled() {
            out.flush()?;
            self.catch_up_after_turn(&hist);
        }
        if self.disk.is_some() && cache_enabled {
            // The reply is complete: let the client have it before the state is copied off the GPU.
            out.flush()?;
            disk_tier::save_after_turn(self);
        }
        Ok(())
    }

    /// w2 (27B agentic follow-ups): after the reply, read `full` (prompt + reply) from the last grid point at or before
    /// the prompt end up to the reply's last grid point with the prompt kernels, in 512-grid chunks, and snapshot there.
    /// The next turn of the conversation then restores past the reply instead of re-reading it (rc1 agentic gate: 56
    /// of 57 follow-ups restored before the previous prompt end). Exact: the cells are written by prompt chunks on the
    /// grid and the snapshot is the state a cold prompt reaches at that grid point (the livecold rule: no decoded cell
    /// is kept). Runs after DONE; stops at a chunk boundary when the next request is waiting (the state stays at a grid
    /// point, snapshotted). `TALUS_Q27_CATCHUP=0` turns it off; `TALUS_Q27_CATCHUP_MAX` caps the tokens read (4096).
    fn catch_up_after_turn(&mut self, full: &[i32]) {
        let grid = self.eng.max_batch.max(1) as usize;
        let target = (full.len() / grid) * grid;
        if target <= (self.live_pf / grid) * grid
            || self.live_pf > full.len()
            || full[..self.live_pf] != self.live[..self.live_pf.min(self.live.len())]
        {
            return; // the reply did not cross a grid point (or the live sequence is not this turn's)
        }
        let r = self
            .pc
            .peek_with_pf(&full[..target], &self.live, self.live_pf, true);
        if r == 0 || r % grid != 0 || target - r > catch_up_max() {
            return;
        }
        if r < target {
            self.catch_up_trigger.arm(&full[..r + grid]);
        }
        let skip = catch_up_skip();
        if self.catch_up_trigger.paused(skip) {
            eprintln!(
                "talus-serve: catch-up skipped ({} requests in a row did not continue the reply)",
                self.catch_up_trigger.misses
            );
            return;
        }
        // K31: a client that answers at once gets the engine idle; the delay costs a slower client at most its length
        let delay = catch_up_delay();
        let d0 = Instant::now();
        while !talus_sched::requests_waiting() && d0.elapsed() < delay {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        if talus_sched::requests_waiting() {
            eprintln!("talus-serve: catch-up skipped (next request waiting)");
            return;
        }
        let t0 = Instant::now();
        let mut view = make_layer_state_view(&mut self.eng);
        let in_cache = self.live_pf.min(
            self.live
                .iter()
                .zip(full.iter())
                .take_while(|(a, b)| a == b)
                .count(),
        );
        let (r2, from) =
            self.pc
                .restore_with_pf(&mut view, &full[..target], &self.live, self.live_pf, true);
        self.eng
            .cl_pc_restore(&full[..target], &self.live, self.live_pf, true, r2);
        if r2 != r {
            // cannot happen (peek and restore pick the same point); leave a clean state
            self.eng.reset();
            self.live.clear();
            self.live_pf = 0;
            return;
        }
        if r2 > in_cache {
            sync_layer_state_view_back(&mut self.eng, &view); // KV came from a stash: rebuild the int8 copy too
        } else {
            // the live cells below r2 are the prompt's own: only the recurrent state moved back
            self.eng.n_past = view.pos as i32;
            self.eng.gdn_pending = view.gdn_pending;
            self.eng.gdn_valid = true;
            self.eng.dctx.set_bounds(view.dctx_lo, view.dctx_hi);
        }
        self.live.clear();
        self.live.extend_from_slice(&full[..r2]);
        self.live_pf = r2;
        let mut at = r2;
        let mut stopped = false;
        while at < target {
            if at > r2 && talus_sched::requests_waiting() {
                stopped = true;
                break;
            }
            let mk = (at + grid).min(target);
            self.eng.process_with_callback(&full[at..mk], true, |_| {});
            self.live.extend_from_slice(&full[at..mk]);
            at = mk;
        }
        self.live_pf = self.live.len();
        if at > r2 && (self.eng.n_past as usize) == self.live.len() {
            let mut view = make_layer_state_view(&mut self.eng);
            self.pc.save(&mut view, &self.live, false);
            self.eng.cl_pc_save(&self.live, false);
        }
        eprintln!(
            "talus-serve: catch-up {} tokens ({} -> {}, from {}{}) in {:.1} ms",
            at - r2,
            r2,
            at,
            from,
            if stopped {
                ", stopped: next request waiting"
            } else {
                ""
            },
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// w2: `TALUS_Q27_CATCHUP` (default on; `0` / `off` turns the after-reply catch-up off).
fn catch_up_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("TALUS_Q27_CATCHUP").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off"))
    })
}

/// w2: `TALUS_Q27_CATCHUP_MAX` tokens read at most per catch-up (default 4096).
fn catch_up_max() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("TALUS_Q27_CATCHUP_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096)
    })
}

/// K31: `TALUS_Q27_CATCHUP_SKIP`: this many requests in a row that do not continue the reply pause the catch-up until
/// one does (default 0 = never pause). Which grid point the next turn restores from doesn't change its output.
fn catch_up_skip() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("TALUS_Q27_CATCHUP_SKIP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// K31: `TALUS_Q27_CATCHUP_DELAY_MS`: idle time after DONE before the catch-up starts (default 0).
fn catch_up_delay() -> std::time::Duration {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    std::time::Duration::from_millis(*N.get_or_init(|| {
        std::env::var("TALUS_Q27_CATCHUP_DELAY_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }))
}

/// E2 `SCORE` for the Qwen3.8-27B engine: one cold prefill of the prompt (prompt cache off: the live state is
/// stashed as for `cache=0`, nothing restored or saved), the first continuation position from the prompt's last row,
/// then the continuation teacher-forced in chunks of <= 16 rows with a head on every row (`all_heads`); with
/// `TALUS_SCORE_STEP=1` one token per step instead (the plain decode path). The engine is reset afterwards.
fn q36_score(
    b: &mut Q36StdioBackend,
    top_k: usize,
    prompt: &[i32],
    cont: &[i32],
    cancel: &CancelToken,
    out: &mut dyn Write,
) -> io::Result<()> {
    use talus_spec::logprobs::format_s_line;
    let cache_enabled = b.pc.max_snaps > 0 || b.pc.max_msg > 0;
    if cache_enabled && !b.live.is_empty() {
        let view = make_layer_state_view(&mut b.eng);
        b.pc.stash_kv_with_pf(&view, &b.live, 0, b.live_pf);
    }
    b.eng.reset();
    b.live.clear();
    b.live_pf = 0;
    let t0 = Instant::now();
    b.eng.process(prompt, true);
    let prompt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    writeln!(
        out,
        "{}",
        format_s_line(&b.eng.host_row_logprobs(cont[0], top_k))
    )?;
    let step = std::env::var("TALUS_SCORE_STEP").is_ok_and(|v| v == "1");
    let chunk = if step {
        1
    } else {
        16usize.min(b.eng.max_batch.max(1) as usize)
    };
    let mut finish = "stop";
    let mut done = 1usize;
    let forced = &cont[..cont.len() - 1];
    for (ci, part) in forced.chunks(chunk).enumerate() {
        if cancel.is_cancelled() {
            finish = "cancel";
            break;
        }
        let next = &cont[1 + ci * chunk..1 + ci * chunk + part.len()];
        if step {
            b.eng.process(part, true);
            writeln!(
                out,
                "{}",
                format_s_line(&b.eng.host_row_logprobs(next[0], top_k))
            )?;
        } else {
            b.eng.all_heads = true;
            b.eng.process(part, false);
            b.eng.all_heads = false;
            for r in b.eng.device_rows_logprobs(next, top_k, false) {
                writeln!(out, "{}", format_s_line(&r))?;
            }
        }
        done += part.len();
        out.flush()?;
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    b.eng.reset();
    eprintln!(
        "talus-serve: scored {done} of {} continuation tokens after a {}-token prompt in {total_ms:.1} ms (prompt {prompt_ms:.1} ms, {})",
        cont.len(),
        prompt.len(),
        if step { "per-token steps" } else { "16-row chunks" }
    );
    write_done_line(
        out,
        &DoneStats {
            produced: 0,
            prompt_tokens: prompt.len() + cont.len(),
            prompt_ms,
            decode_ms: total_ms,
            finish,
            accepted: 0,
            offered: 0,
            reuse: 0,
            fresh: (prompt.len() + cont.len()) as i64,
        },
    )
}

/// Adapter implementing `talus_server::Engine` directly over `Q36StdioBackend` when `--serve <PORT>` is used.
struct Q36HttpServerEngine {
    inner: Mutex<Q36StdioBackend>,
    max_ctx: usize,
    tokenizer: Option<talus_text::Tokenizer>,
    byte_tok: talus_server::ByteTokenizer,
    cancel_flag: Arc<AtomicBool>,
}

impl talus_server::Engine for Q36HttpServerEngine {
    fn max_context(&self) -> usize {
        self.max_ctx
    }

    fn tokenize(&self, text: &str, parse_special: bool) -> Vec<u32> {
        if let Some(ref tok) = self.tokenizer {
            tok.encode(text, parse_special)
        } else {
            self.byte_tok.encode(text, parse_special)
        }
    }

    fn detokenize(&self, ids: &[u32]) -> String {
        if let Some(ref tok) = self.tokenizer {
            tok.decode(ids, false)
        } else {
            self.byte_tok.decode(ids)
        }
    }

    fn cancel(&self) {
        self.cancel_flag
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn supports_json_schema(&self) -> bool {
        true
    }

    fn generate(
        &self,
        req: &talus_server::GenerateRequest,
        cancel: Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32) -> bool,
        on_pp: &mut dyn FnMut(usize, usize),
    ) -> Result<talus_server::EngineTimings, talus_server::EngineError> {
        self.cancel_flag
            .store(false, std::sync::atomic::Ordering::SeqCst);
        struct CallbackWriter<'a> {
            buf: String,
            prompt_n: usize,
            reused: usize,
            cancel: &'a Arc<AtomicBool>,
            on_token: &'a mut dyn FnMut(u32) -> bool,
            on_pp: &'a mut dyn FnMut(usize, usize),
            done: Option<DoneStats>,
        }
        impl Write for CallbackWriter<'_> {
            fn write(&mut self, data: &[u8]) -> io::Result<usize> {
                self.buf.push_str(&String::from_utf8_lossy(data));
                while let Some(pos) = self.buf.find('\n') {
                    let line = self.buf[..pos].trim().to_string();
                    self.buf.drain(..=pos);
                    if let Some(rest) = line.strip_prefix("RESUME ") {
                        self.reused = rest.trim().parse::<usize>().unwrap_or(0);
                    } else if let Some(rest) = line.strip_prefix("PP ") {
                        let mut parts = rest.split_whitespace();
                        let done = parts
                            .next()
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(0);
                        (self.on_pp)(self.reused + done, self.prompt_n);
                    } else if let Some(rest) = line.strip_prefix("T ") {
                        if let Ok(tok) = rest.trim().parse::<i32>() {
                            if tok >= 0 && !(self.on_token)(tok as u32) {
                                self.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                            }
                        }
                    } else if let Some(rest) = line.strip_prefix("DONE ") {
                        let p: Vec<&str> = rest.split_whitespace().collect();
                        if p.len() >= 5 {
                            let finish_str: &'static str = match p[4] {
                                "stop" => "stop",
                                "cancel" => "cancel",
                                _ => "length",
                            };
                            let ds = DoneStats {
                                produced: p[0].parse().unwrap_or(0),
                                prompt_tokens: p[1].parse().unwrap_or(self.prompt_n),
                                prompt_ms: p[2].parse().unwrap_or(0.0),
                                decode_ms: p[3].parse().unwrap_or(0.0),
                                finish: finish_str,
                                accepted: p.get(5).and_then(|s| s.parse().ok()).unwrap_or(0),
                                offered: p.get(6).and_then(|s| s.parse().ok()).unwrap_or(0),
                                reuse: p.get(7).and_then(|s| s.parse().ok()).unwrap_or(self.reused),
                                fresh: p.get(13).and_then(|s| s.parse().ok()).unwrap_or(0),
                            };
                            self.done = Some(ds);
                        }
                    }
                }
                Ok(data.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let cmd = GenCommand {
            max_new: req.max_new_tokens as i64,
            temperature: req.sampling.temperature as f32,
            top_p: req.sampling.top_p as f32,
            top_k: req.sampling.top_k as i32,
            min_p: req.sampling.min_p as f32,
            penalty_last_n: req.sampling.penalty_last_n as i32,
            penalty_repeat: req.sampling.repeat_penalty as f32,
            penalty_freq: req.sampling.frequency_penalty as f32,
            penalty_present: req.sampling.presence_penalty as f32,
            seed: req.sampling.seed.unwrap_or(1).max(1) as u64,
            ignore_eos: req.sampling.ignore_eos,
            use_cache: req.sampling.cache_prompt,
            prompt_ids: req.prompt_ids.iter().map(|&u| u as i32).collect(),
            // the in-process adapter does not advertise E1 (talus-server answers 400)
            logprobs: None,
            json_schema: req.sampling.json_schema.clone(),
        };

        let cancel_tok = CancelToken::from_arc(Arc::clone(&cancel));
        let mut writer = CallbackWriter {
            buf: String::new(),
            prompt_n: req.prompt_ids.len(),
            reused: 0,
            cancel: &cancel,
            on_token,
            on_pp,
            done: None,
        };
        let mut guard = self.inner.lock().map_err(|_| {
            talus_server::EngineError::EngineDied("engine lock poisoned".to_string())
        })?;
        guard
            .handle_gen(&cmd, &cancel_tok, &mut writer)
            .map_err(|e| talus_server::EngineError::EngineDied(e.to_string()))?;

        let ds = writer.done.unwrap_or(DoneStats {
            produced: 0,
            prompt_tokens: req.prompt_ids.len(),
            prompt_ms: 0.0,
            decode_ms: 0.0,
            finish: "stop",
            accepted: 0,
            offered: 0,
            reuse: 0,
            fresh: req.prompt_ids.len() as i64,
        });
        let prompt_fresh = ds.prompt_tokens.saturating_sub(ds.reuse);
        Ok(talus_server::EngineTimings {
            cache_n: ds.reuse,
            prompt_n: prompt_fresh,
            prompt_ms: ds.prompt_ms,
            predicted_n: ds.produced.max(0) as usize,
            predicted_ms: ds.decode_ms,
            draft_n: (ds.offered > 0).then_some(ds.offered as usize),
            draft_accepted: (ds.offered > 0).then_some(ds.accepted.max(0) as usize),
            finish_reason: ds.finish.to_string(),
        })
    }
}

/// Checks whether a GGUF file at `path` has `general.architecture == "laguna"`.
pub fn is_laguna_model(path: &Path) -> bool {
    if let Ok(gguf) = talus_gguf::GgufFile::open(path) {
        if gguf.get_str("general.architecture") == Some("laguna") {
            return true;
        }
    }
    if let Ok(model) = talus_gguf::GgufModel::open(path) {
        if model.meta_shard().get_str("general.architecture") == Some("laguna") {
            return true;
        }
    }
    false
}

/// Checks whether a GGUF file at `path` has `general.architecture == "mimo2"`.
pub fn is_mimo_model(path: &Path) -> bool {
    if let Ok(gguf) = talus_gguf::GgufFile::open(path) {
        if gguf.get_str("general.architecture") == Some(talus_model::mimo::ARCH_NAME) {
            return true;
        }
    }
    if let Ok(model) = talus_gguf::GgufModel::open(path) {
        if model.meta_shard().get_str("general.architecture") == Some(talus_model::mimo::ARCH_NAME)
        {
            return true;
        }
    }
    false
}

/// Runs the MiMo-V2.6-Flash (`mimo2`) CLI / server path (WP-3.3, Phase H).
/// talus-public#5: `--kv q8` / `--kv int8` given to an engine whose KV cache has no int8 format. The flag
/// was accepted and ignored without a word (and MiMo's INFO line even said `kv=q8`); say what runs instead.
/// `engine`: `"mimo"` or `"qwen35"` (27B). Laguna honours the flag (q8 is its default) and Flash-Next / GLM parse
/// their own flags.
pub fn kv_flag_warning(engine: &str, cfg: &CliConfig) -> Option<&'static str> {
    if !cfg.kv_explicit || !cfg.kv_q8 {
        return None;
    }
    match engine {
        "mimo" => Some(
            "talus-mimo: WARN: --kv int8/q8 is not supported by the MiMo engine and is ignored: the KV cache is F16 \
             (the 8 full-attention layers grow with --max-context; the 40 sliding-window layers keep a fixed ring)",
        ),
        "qwen35" => Some(
            "talus-cli: WARN: --kv int8/q8 does not change this engine's KV cache and is ignored: the 27B keeps F16 \
             KV cells. Its int8 decode/verify copy is TALUS_Q27_KV8 (on by default for the quantized dense 27B since \
             v0.1.3); it speeds up deep decode and adds memory rather than halving the KV cache",
        ),
        _ => None,
    }
}

pub fn run_mimo_cli(cfg: &CliConfig, model_path: &Path) -> i32 {
    if let Some(w) = kv_flag_warning("mimo", cfg) {
        eprintln!("{w}");
    }
    let ctx = if cfg.ctx_explicit {
        cfg.max_context.max(256) as usize
    } else {
        4096
    };
    let draft_max = if cfg.no_mtp {
        0
    } else if cfg.mtp > 0 {
        cfg.mtp as usize
    } else if cfg.spec > 0 {
        cfg.spec as usize
    } else {
        talus_model::mimo::DFLASH_DEFAULT_DRAFT_MAX
    };

    let catalog = match talus_model::mimo::MimoCatalog::open_shards(&[model_path]) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("talus-mimo: {e}");
            return 1;
        }
    };

    let resolved_dflash_path: Option<std::path::PathBuf> = if let Some(ref p) = cfg.dflash_gguf {
        Some(std::path::PathBuf::from(p))
    } else if draft_max > 0 {
        [
            talus_model::mimo::weights::DEFAULT_MIMO_DFLASH_PATH,
            talus_model::mimo::weights::DEFAULT_MIMO_MOPD_DFLASH_PATH,
        ]
        .into_iter()
        .map(talus_model::mimo::weights::default_model_path)
        .find(|p| p.exists())
    } else {
        None
    };

    let dflash_catalog = match resolved_dflash_path.as_deref() {
        Some(df_path) => match talus_model::mimo::MimoDFlashCatalog::open(df_path) {
            Ok(dc) => Some(dc),
            Err(e) => {
                eprintln!("talus-mimo: {e}");
                return 1;
            }
        },
        None => None,
    };

    let ubatch = if cfg.ubatch > 0 {
        cfg.ubatch as usize
    } else {
        std::env::var("TALUS_MIMO_UBATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(2048)
    };
    let eng_cfg = talus_model::mimo::MimoEngineConfig {
        max_seq_len: ctx,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        seed: 42,
        dflash_gguf: resolved_dflash_path.clone(),
        draft_vocab: if cfg.no_draft_vocab {
            None
        } else {
            cfg.draft_vocab.as_ref().map(std::path::PathBuf::from)
        },
        draft_max,
        ubatch,
        // the engine has F16 KV only: the config must not claim q8 (the serve INFO line prints it)
        kv_q8: false,
    };

    let is_hdr_only = model_path.to_string_lossy().ends_with(".hdr");
    let has_gpu = !is_hdr_only && talus_hip::Device::count().unwrap_or(0) > 0;

    let mut eng = if has_gpu {
        let hip = match talus_model::mimo::ffi::HipRuntime::load() {
            Ok(h) => Arc::new(h),
            Err(e) => {
                eprintln!("talus-mimo: {e}");
                return 1;
            }
        };
        let tk = match talus_model::mimo::ffi::load_talus_kernels(None) {
            Ok((k, _)) => Arc::new(k),
            Err(e) => {
                eprintln!("talus-mimo: {e}");
                return 1;
            }
        };
        let budget =
            Arc::new(<talus_mem::Budget as talus_model::mimo::MimoBudgetExt>::for_mimo_11gib(None));
        let dflash_path = resolved_dflash_path.as_deref();
        let max_layers = if cfg.n_layers_explicit {
            Some(cfg.n_layers.max(1) as usize)
        } else {
            None
        };
        let t_load = Instant::now();
        let model = match talus_model::mimo::MimoModel::load_with_layers(
            model_path,
            dflash_path,
            max_layers,
            Some(Arc::clone(&tk)),
            Arc::clone(&hip),
            Arc::clone(&budget),
        ) {
            Ok(m) => Arc::new(m),
            Err(e) => {
                eprintln!("talus-mimo: {e}");
                return 1;
            }
        };
        let n_loaded_layers = model.layers.len();
        let eng = match talus_model::mimo::MimoEngine::new_gpu(model, tk, hip, budget, eng_cfg) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("talus-mimo: {e}");
                return 1;
            }
        };
        let load_s = t_load.elapsed().as_secs_f64();
        let mem_avail_gib = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines().find_map(|l| {
                    l.strip_prefix("MemAvailable:")
                        .and_then(|rest| rest.split_whitespace().next())
                        .and_then(|kb| kb.parse::<f64>().ok())
                        .map(|kb| kb / (1024.0 * 1024.0))
                })
            })
            .unwrap_or(0.0);
        eprintln!(
            "talus-mimo: loaded {} layers in {:.2} s (MemAvailable: {:.2} GiB)",
            n_loaded_layers, load_s, mem_avail_gib
        );
        eng
    } else {
        match talus_model::mimo::MimoEngine::from_catalog(
            &catalog,
            dflash_catalog.as_ref(),
            eng_cfg,
            None,
        ) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("talus-mimo: {e}");
                return 1;
            }
        }
    };

    if cfg.serve {
        // Run-time guard at the common 8 GiB floor. MIMO_FLOOR_BYTES (11 GiB) is the load-time budget: it leaves room
        // for run-time growth (served MiMo reaches ~11 GiB MemAvailable at 81K), so refusing requests below 11 GiB
        // would fail requests that complete today.
        talus_mem::pressure::install(
            talus_mem::pressure::PressureConfig::strix_default(),
            "talus-mimo",
        );
        return match talus_model::mimo::serve_main(&mut eng) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("talus-mimo: {e}");
                1
            }
        };
    }

    let Some(ref tok_arg) = cfg.tokens else {
        print_usage();
        return 2;
    };
    let prompt = match parse_ids_arg(tok_arg) {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => {
            eprintln!("talus-mimo: no tokens in {tok_arg}");
            return 1;
        }
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let dump_path = cfg
        .dump_logits
        .as_deref()
        .or(cfg.dump_full_logits.as_deref());
    let n_gen = if cfg.generate > 0 {
        cfg.generate as usize
    } else if cfg.n_explicit {
        cfg.max_new.max(0) as usize
    } else if dump_path.is_some() || cfg.force_decode > 0 {
        0
    } else {
        cfg.max_new.max(1) as usize
    };

    if n_gen == 0 && (dump_path.is_some() || cfg.force_decode > 0) {
        let force_dec = (cfg.force_decode.max(0) as usize).min(prompt.len().saturating_sub(1));
        let n_prefill = prompt.len() - force_dec;
        let t0 = Instant::now();
        let prefill_logits = match eng.forward_batch(&prompt[..n_prefill]) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("talus-mimo: {e}");
                return 1;
            }
        };
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mut decode_rows = Vec::with_capacity(force_dec + 1);
        if let Some(last) = prefill_logits.last() {
            decode_rows.push(last.clone());
        }
        // TALUS_MIMO_FD_WINDOW=W (release row-parity gate): forced rows through W-row windows, the way serving verifies
        // drafts, so the dump can be compared bit for bit with the one-row dump. Default 1, the plain path.
        let fd_window = std::env::var("TALUS_MIMO_FD_WINDOW")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 8);
        let t1 = Instant::now();
        for window in prompt[n_prefill..].chunks(fd_window) {
            match eng.forward_batch(window) {
                Ok(rows) => decode_rows.extend(rows),
                Err(e) => {
                    eprintln!("talus-mimo: {e}");
                    return 1;
                }
            }
        }
        let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;
        if let Some(dp) = dump_path {
            use std::io::Write;
            let mut f = match std::fs::File::create(dp) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("talus-mimo: cannot write {dp}: {e}");
                    return 1;
                }
            };
            let rows_to_write = if cfg.dump_decode_only {
                &decode_rows[..]
            } else {
                &decode_rows[..1]
            };
            for row in rows_to_write {
                let bytes =
                    unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, row.len() * 4) };
                if let Err(e) = f.write_all(bytes) {
                    eprintln!("talus-mimo: write {dp}: {e}");
                    return 1;
                }
            }
        }
        eprintln!(
            "talus-mimo: prefill {} tok in {:.1} ms ({:.1} tok/s), force_decode {} tok in {:.1} ms ({:.2} tok/s)",
            n_prefill,
            prefill_ms,
            if prefill_ms > 0.0 {
                (n_prefill as f64) * 1000.0 / prefill_ms
            } else {
                0.0
            },
            force_dec,
            decode_ms,
            if decode_ms > 0.0 {
                (force_dec as f64) * 1000.0 / decode_ms
            } else {
                0.0
            }
        );
        return 0;
    }

    match eng.generate(&prompt, n_gen) {
        Ok(res) => {
            if let Some(dp) = dump_path {
                if !res.first_logits.is_empty() {
                    use std::io::Write;
                    let mut f = match std::fs::File::create(dp) {
                        Ok(f) => f,
                        Err(e) => {
                            eprintln!("talus-mimo: cannot write {dp}: {e}");
                            return 1;
                        }
                    };
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            res.first_logits.as_ptr() as *const u8,
                            res.first_logits.len() * 4,
                        )
                    };
                    if let Err(e) = f.write_all(bytes) {
                        eprintln!("talus-mimo: write {dp}: {e}");
                        return 1;
                    }
                }
            }
            let pp_tok_s = if res.prefill_ms > 0.0 {
                (prompt.len() as f64) * 1000.0 / res.prefill_ms
            } else {
                0.0
            };
            let tg_tok_s = if res.decode_ms > 0.0 && res.tokens.len() > 1 {
                ((res.tokens.len() - 1) as f64) * 1000.0 / res.decode_ms
            } else {
                0.0
            };
            let spec_str = if let Some(ref s) = res.spec_stats {
                format!(
                    ", dflash_n4(steps={}, proposed={}, accepted={}, bonus={}, accept_rate={:.3})",
                    s.verify_steps,
                    s.draft_proposed,
                    s.draft_accepted,
                    s.bonus_emitted,
                    s.acceptance_rate()
                )
            } else {
                String::new()
            };
            eprintln!(
                "talus-mimo: prefill {} tok in {:.1} ms ({:.1} tok/s), decode {} tok in {:.1} ms ({:.2} tok/s){}",
                prompt.len(),
                res.prefill_ms,
                pp_tok_s,
                res.tokens.len(),
                res.decode_ms,
                tg_tok_s,
                spec_str
            );
            print!("gen:");
            for &t in &res.tokens {
                print!(" {t}");
            }
            println!();
            0
        }
        Err(e) => {
            eprintln!("talus-mimo: {e}");
            1
        }
    }
}

/// Locate the Laguna DFlash drafter for `model`: a `*dflash*q8_0*.gguf` in the model's directory, else in a sibling
/// directory of the same model family (the name up to its third `-`, e.g. `Laguna-S-2.1-UD-Q5_K_XL` for
/// `Laguna-S-2.1-Q4-lean`).
pub fn find_laguna_dflash(model: &Path) -> Option<String> {
    let dir = model.parent()?;
    let pick = |d: &Path| -> Option<String> {
        let mut v: Vec<_> = std::fs::read_dir(d)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                let n = p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                n.ends_with(".gguf") && n.contains("dflash") && n.contains("q8_0")
            })
            .collect();
        v.sort();
        v.first().map(|p| p.to_string_lossy().into_owned())
    };
    if let Some(p) = pick(dir) {
        return Some(p);
    }
    let name = dir.file_name()?.to_str()?;
    let family: String = name.splitn(4, '-').take(3).collect::<Vec<_>>().join("-");
    let mut sibs: Vec<_> = std::fs::read_dir(dir.parent()?)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(&family))
        })
        .collect();
    sibs.sort();
    sibs.iter().find_map(|d| pick(d))
}

/// Runs the Laguna-S-2.1 (`laguna`) CLI / server path, matching `src/laguna/laguna_main.cpp`.
pub fn run_laguna_cli(cfg: &CliConfig, model_path: &Path) -> i32 {
    let ctx = if cfg.ctx_explicit {
        cfg.max_context.max(1) as usize
    } else {
        4096
    };
    let batch = cfg.batch.clamp(1, 8) as usize;
    let gen = if cfg.generate > 0 {
        cfg.generate as usize
    } else if !cfg.serve && cfg.tokens.is_some() && (cfg.n_explicit || cfg.max_new != 64) {
        cfg.max_new.max(0) as usize
    } else {
        0
    };
    let force_dec = cfg.force_decode.max(0) as usize;
    let verify_test = cfg.verify_test.max(0) as usize;
    let spec_k = cfg.spec.max(0) as usize;
    let spec_min = cfg.spec_min_match.max(1) as usize;

    let mut ubatch = if cfg.ubatch > 0 {
        cfg.ubatch as usize
    } else if cfg.serve {
        2048
    } else {
        0
    };
    if (gen > 0 || force_dec > 0 || verify_test > 0 || spec_k > 0) && ubatch == 0 {
        ubatch = batch;
    }

    let toks = if !cfg.serve {
        let Some(ref tok_arg) = cfg.tokens else {
            print_usage();
            return 2;
        };
        let v = match parse_ids_arg(tok_arg) {
            Ok(v) if !v.is_empty() => v,
            Ok(_) => {
                eprintln!("talus-laguna: no tokens in {tok_arg}");
                return 1;
            }
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        };
        if v.len() + gen + 8 > ctx {
            eprintln!(
                "talus-laguna: tokens ({}) + gen ({}) > ctx ({})",
                v.len(),
                gen,
                ctx
            );
            return 1;
        }
        v
    } else {
        Vec::new()
    };

    if ubatch == 0 {
        ubatch = batch;
    }

    let hip = match talus_model::laguna::HipRuntime::load() {
        Ok(h) => Arc::new(h),
        Err(e) => {
            eprintln!("talus-laguna: {e}");
            return 1;
        }
    };
    let tk = match talus_model::laguna::load_talus_kernels(None) {
        Ok((k, _)) => Arc::new(k),
        Err(e) => {
            eprintln!("talus-laguna: {e}");
            return 1;
        }
    };

    let t0 = Instant::now();
    let model = match talus_model::laguna::LagunaModel::load(
        model_path,
        cfg.n_layers.max(1),
        None,
        Arc::clone(&hip),
    ) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            eprintln!("talus-laguna: {e}");
            return 1;
        }
    };
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "talus-laguna: loaded {} layers in {:.2} s",
        model.layers.len(),
        load_s
    );

    let mut opt = talus_model::laguna::Options::default();
    opt.ubatch = ubatch;
    // D4  S6c): the WMMA prompt attention exists only for gfx11 (it traps elsewhere)
    opt.use_wmma = talus_hip::Device::new(talus_core_api::DeviceId(0))
        .map(|d| d.arch().has_gfx11_wmma())
        .unwrap_or(true);
    opt.kv_q8 = cfg.kv_q8;
    opt.graph = cfg.use_graph && !cfg.no_graph;
    opt.sparse = cfg.sparse;
    opt.sparse_bs = cfg.sparse_bs;
    opt.sparse_k = cfg.sparse_k;
    opt.sparse_min = cfg.sparse_min;
    opt.sparse_sink = cfg.sparse_sink;
    opt.sparse_recent = cfg.sparse_recent;
    // --dflash-gguf turns on the Laguna DFlash drafter (serve mode); the engine then records its taps
    // (or TALUS_LAGUNA_DFLASH=PATH).
    // Default Laguna serve mode (TALUS_LAGUNA_LSRV=0: legacy behaviour): the DFlash drafter
    // (--dflash-gguf, TALUS_LAGUNA_DFLASH, else a *dflash*Q8_0*.gguf next to the model), prompt-lookup drafts up to 63,
    // decode through verify(1), wave32 dense mat-vecs (STRATA_MMVQ_W32=1 unless set; 0 disables).
    // (TALUS_LAGUNA_LSRV=1 outside serve enables the same kernels for CLI probes, e.g. --verify-test)
    let fast_serve_env = std::env::var("TALUS_LAGUNA_LSRV").ok();
    let fast_serve_mode = if cfg.serve {
        fast_serve_env.as_deref() != Some("0")
    } else {
        fast_serve_env.as_deref() == Some("1")
    };
    let df_path = if fast_serve_mode {
        cfg.dflash_gguf
            .clone()
            .or_else(|| std::env::var("TALUS_LAGUNA_DFLASH").ok())
            .or_else(|| find_laguna_dflash(model_path))
            .filter(|v| !v.is_empty() && v != "0")
    } else {
        None
    };
    opt.taps = df_path.is_some();
    if fast_serve_mode {
        // the wave32 kernels, and wide dense mat-vecs (verify windows) through the exact tile kernels from 9
        // columns, every column bitwise the single-column wave32 result (STRATA_MMVQ_XT=0: off, the 16-column wave32
        // chunks); unset only (talus_kernels::knobs)
        talus_kernels::knobs::apply("laguna-serve", talus_kernels::knobs::LAGUNA_SERVE_DEFAULTS);
        opt.dense_mmvq = true;
    }

    let mut eng = match talus_model::laguna::Engine::new(model, ctx, opt, None, hip, tk) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("talus-laguna: {e}");
            return 1;
        }
    };
    let mut df = match df_path.clone() {
        Some(p) => match talus_model::laguna::LagunaDFlash::load(Path::new(&p), &eng, ctx) {
            Ok(d) => {
                eprintln!("talus-laguna: dflash drafter {p}");
                Some(d)
            }
            Err(e) => {
                eprintln!("talus-laguna: dflash: {e}");
                return 1;
            }
        },
        None => None,
    };

    if cfg.serve {
        talus_mem::pressure::install(
            talus_mem::pressure::PressureConfig::strix_default(),
            "talus-laguna",
        );
        let disk =
            disk_tier::open_laguna(cfg, model_path, df_path.as_deref(), eng.ring(), eng.ctx16());
        return match talus_model::laguna::serve_main(
            &mut eng,
            &cfg.eos_ids,
            spec_k,
            spec_min,
            load_s,
            df.as_mut(),
            fast_serve_mode,
            disk.as_ref(),
        ) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("talus-laguna: {e}");
                1
            }
        };
    }

    // TALUS_LAGUNA_LSRV=1 puts the CLI probes on the served kernels: decode too, as serve_main does (a decode step is
    // verify(1)). Without this, --verify-test and --force-decode compared the verify kernels with a release check's decode
    // kernels, which serving never runs (TALUS_LAGUNA_DEC_VERIFY=0: those kernels).
    eng.set_dec_verify(fast_serve_mode);
    let lcfg = talus_model::laguna::LagunaCliConfig {
        tokens: toks,
        top: cfg.top.max(0) as usize,
        generate: gen,
        force_decode: force_dec,
        verify_test,
        dump_layers: cfg.dump_layers,
        dump_logits: cfg.dump_logits.as_ref().map(std::path::PathBuf::from),
        dump_full_logits: cfg.dump_full_logits.as_ref().map(std::path::PathBuf::from),
        dump_decode_only: cfg.dump_decode_only,
        use_graph: opt.graph,
    };
    match talus_model::laguna::run_cli_ubatch(&mut eng, &lcfg) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("talus-laguna: {e}");
            1
        }
    }
}

/// Executes a parsed [`CliConfig`] and returns the process exit status code.
pub fn run_cli(cfg: &CliConfig) -> i32 {
    if cfg.help {
        print_usage();
        return 0;
    }

    if cfg.sample_test {
        return if talus_spec::sample_selftest() { 0 } else { 1 };
    }

    if cfg.serve && std::env::var("TALUS_CLI_MOCK_SERVE").is_ok_and(|v| v == "1") {
        let mut backend = ScriptedProtocolBackend::new(cfg.max_context, 248320);
        return serve_stdio(&mut backend);
    }

    let Some(ref model_path) = cfg.model else {
        eprintln!("talus-cli: --model PATH.gguf is required (try --help)");
        return 2;
    };

    if is_laguna_model(Path::new(model_path)) {
        return run_laguna_cli(cfg, Path::new(model_path));
    }

    if is_mimo_model(Path::new(model_path)) {
        return run_mimo_cli(cfg, Path::new(model_path));
    }

    if let Some(w) = kv_flag_warning("qwen35", cfg) {
        eprintln!("{w}");
    }

    let prompt_tokens = match cfg.tokens.as_deref() {
        Some(spec) => match parse_ids_arg(spec) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        },
        None => Vec::new(),
    };
    let force_tokens = match cfg.force.as_deref() {
        Some(spec) => match parse_ids_arg(spec) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        },
        None => Vec::new(),
    };

    if cfg.mtp < 0 || cfg.mtp > 15 {
        eprintln!("talus-cli: --mtp N must be in 0 .. 15");
        return 2;
    }
    if cfg.chunk < 0 || cfg.chunk > 4096 || cfg.max_context < 256 {
        eprintln!("talus-cli: bad --chunk or --max-context");
        return 2;
    }

    let draft_vocab_vec = match cfg.draft_vocab.as_deref() {
        Some(dv_path) if !cfg.no_draft_vocab => match std::fs::read(dv_path) {
            Ok(bytes) if !bytes.is_empty() && bytes.len() % 4 == 0 => {
                let is_ascii = bytes.iter().take(64).all(|&b| {
                    b.is_ascii_digit()
                        || b.is_ascii_whitespace()
                        || b == b','
                        || b == b'-'
                        || b == b'+'
                });
                if is_ascii {
                    parse_ids_arg(&format!("@{dv_path}")).ok()
                } else {
                    Some(
                        bytes
                            .chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .filter(|&id| id >= 0)
                            .collect::<Vec<i32>>(),
                    )
                }
            }
            Ok(_) => parse_ids_arg(&format!("@{dv_path}")).ok(),
            Err(e) => {
                eprintln!("talus-cli: failed to open draft vocab {dv_path}: {e}");
                return 2;
            }
        },
        _ => None,
    };

    let mut opts = Q36Options::from_env();
    opts.max_seq = cfg.max_context;
    opts.use_graph = cfg.use_graph;
    opts.use_mtp_graph = cfg.use_graph;
    opts.mtp_gguf = cfg.mtp_gguf.as_ref().map(std::path::PathBuf::from);
    opts.no_mtp = cfg.no_mtp;
    opts.mtp_main_head = cfg.mtp_main_head;
    opts.mtp_f16 = cfg.mtp_f16;
    opts.no_draft_vocab = cfg.no_draft_vocab;
    opts.attn_w8 = cfg.attn_w8;
    opts.no_v16 = cfg.no_v16;
    opts.dflash_path = cfg.dflash_gguf.clone().unwrap_or_default();

    // talus#47: a cluster stage loads only its layers. Off unless --cluster-role / TALUS_CLUSTER_ROLE.
    let cl_spec = match talus_cluster::Spec::from_file(
        cfg.cluster_file.as_deref(),
        cfg.cluster_node.as_deref(),
    )
    .and_then(|f| match f {
        Some(s) => Ok(Some(s)),
        None => talus_cluster::Spec::from_parts(
            cfg.cluster_role.as_deref(),
            cfg.cluster_addr.as_deref(),
            cfg.cluster_split,
        ),
    }) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("talus-cli: {e}");
            return 2;
        }
    };
    let mut draft_vocab_vec = draft_vocab_vec;
    if let Some(sp) = &cl_spec {
        // the handshake's environment identity: snapshot it before the kernel library adds its own defaults
        let _ = talus_cluster::ident::env_pairs();
        let n_layer = match talus_gguf::GgufFile::open(Path::new(model_path))
            .map_err(|e| e.to_string())
            .and_then(|f| talus_model::qwen35::describe_gguf(&f).map_err(|e| e.to_string()))
        {
            Ok(g) => g.n_layer as usize,
            Err(e) => {
                eprintln!("talus-cli: cluster: cannot read {model_path}: {e}");
                return 1;
            }
        };
        let split = if sp.split > 0 {
            sp.split as usize
        } else {
            talus_model::qwen35::cluster::default_split(n_layer)
        };
        if split == 0 || split >= n_layer {
            eprintln!("talus-cli: cluster split {split} outside 1..{n_layer}");
            return 2;
        }
        if cfg.mmvq_bench.is_some() || cfg.attn_test || cfg.attn_bench {
            eprintln!(
                "talus-cli: --mmvq-bench / --attn-test / --attn-bench run on one process only"
            );
            return 2;
        }
        match sp.role {
            talus_cluster::Role::Head => {
                opts.stage_lo = 0;
                opts.stage_hi = split;
            }
            talus_cluster::Role::Worker => {
                opts.stage_lo = split;
                opts.stage_hi = 0;
                opts.mtp_gguf = None;
                opts.no_mtp = true;
                opts.dflash_path.clear();
                draft_vocab_vec = None;
            }
        }
    }

    let mut model = match Q36Model::load(
        Path::new(model_path),
        opts.mtp_gguf.as_deref(),
        draft_vocab_vec.as_deref(),
        &opts,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("talus-cli: failed to load model {model_path}: {e}");
            return 1;
        }
    };
    if let Some(sp) = cl_spec
        .as_ref()
        .filter(|s| s.role == talus_cluster::Role::Worker)
    {
        return talus_model::qwen35::cluster::worker_main(&mut model, &sp.addr);
    }
    // talus#47 head: connect to the worker, agree on the context (per-stage auto context when none is given), and
    // have the worker build its engine first, so each stage's memory plan sees the other's allocations.
    let mut cl_link = None;
    let mut cl_cfg;
    let cfg = if let Some(sp) = cl_spec.as_ref() {
        let max_batch = if cfg.chunk > 0 { cfg.chunk } else { 512 };
        let explicit = cfg.ctx_explicit.then_some(cfg.max_context);
        let wait_s = std::env::var("TALUS_CLUSTER_WAIT_S")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(900);
        let hl = match talus_model::qwen35::cluster::cluster_connect_head(
            &model,
            &sp.addr,
            sp.alt.as_deref(),
            explicit,
            max_batch,
            cfg.use_graph,
            wait_s,
        ) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("talus-cli: cluster: {e}");
                return 1;
            }
        };
        let ctx = hl.ctx;
        cl_link = Some((
            hl,
            talus_model::qwen35::cluster::Recon {
                addr: sp.addr.clone(),
                alt: sp.alt.clone(),
                ctx,
                max_batch,
                use_graph: cfg.use_graph,
            },
        ));
        model.cfg.max_seq = ctx;
        model.opts.max_seq = ctx;
        cl_cfg = cfg.clone();
        cl_cfg.max_context = ctx;
        // the RAM prompt cache is sharded over the stages (cl_pc_*); the disk tier is not yet (one box's files)
        if cl_cfg.serve {
            eprintln!(
                "talus-cli: cluster: prompt cache sharded over the stages (RAM); disk tier off"
            );
        }
        cl_cfg.prompt_cache_disk = Some("off".to_string());
        &cl_cfg
    } else {
        cfg
    };
    // without --max-context the 27B takes the largest of 256K / 128K / 64K that fits in memory beside
    // the weights and the int8 prompt expansion. An explicit --max-context is always honoured.
    let mut auto_cfg;
    let cfg = if model.dense && !cfg.ctx_explicit && cl_link.is_none() {
        let (ctx, why) = talus_model::qwen35::engine::q27_auto_context(&model);
        eprintln!("talus-cli: {why}");
        model.cfg.max_seq = ctx;
        model.opts.max_seq = ctx;
        auto_cfg = cfg.clone();
        auto_cfg.max_context = ctx;
        &auto_cfg
    } else {
        cfg
    };

    // --chunk above 512: the engine has to size its row buffers for it (it used to be set after they were allocated)
    let mut eng = match Q36Engine::new_with_batch(&model, cfg.max_context, cfg.chunk) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("talus-cli: failed to initialize engine: {e}");
            return 1;
        }
    };
    if cfg.chunk > 0 {
        eng.max_batch = cfg.chunk;
    }
    let mtp_cli = if eng.model().has_mtp { cfg.mtp } else { 0 };
    if eng.use_dflash && eng.df.ready {
        eng.dctx
            .init(eng.df.window(), eng.df.ring(), eng.df.block());
        eng.dp
            .init("gfx1151", mtp_cli > 0, cfg.drafter.as_deref().unwrap_or(""));
        eng.mtp_d = if eng.dp.uses_mtp() {
            mtp_cli
        } else {
            eng.df.n_draft_file()
        };
        eprintln!("talus-cli: {}", eng.dp.describe());
    } else {
        eng.mtp_d = mtp_cli;
    }
    let dev_rounds = eng.model().dense
        && eng.opts.use_graph
        && (eng.model().has_mtp || eng.use_dflash)
        && std::env::var("STRATA_Q36_OLD_ROUNDS").is_err();
    eng.dl.init_arch(
        "gfx1151",
        if dev_rounds && (!eng.use_dflash || eng.dp.uses_mtp()) {
            eng.mtp_d
        } else {
            0
        },
    );
    eng.dfl_sample = eng.use_dflash && eng.dl.dfl_sample();
    if let Some(cd) = eng.dl.cd.as_ref() {
        eprintln!("talus-cli: MTP greedy windows: {}", cd.describe());
    }
    if let Some(cd) = eng.dl.cdd.as_ref() {
        eprintln!("talus-cli: DFlash2 greedy windows: {}", cd.describe());
    }
    if let Some((hl, recon)) = cl_link.take() {
        if let Err(e) = eng.cluster_attach_head(hl, recon) {
            eprintln!("talus-cli: cluster: {e}");
            return 1;
        }
    }

    if cfg.mmvq_bench.is_some() {
        eng.mmvq_bench();
        return 0;
    }
    if cfg.attn_test {
        return if eng.attn_selftest() { 0 } else { 1 };
    }
    if cfg.attn_bench {
        eng.attn_bench();
        return 0;
    }

    if cfg.warmup {
        eng.warmup();
    }

    if cfg.exact_test > 0 {
        let mut pr = prompt_tokens.clone();
        let mut fz = force_tokens.clone();
        // talus#78: N <= vt_max: a verify window of N rows at the prompt's end; N > vt_max: position N with
        // TALUS_EXACT_T rows (default 4)
        let vmax = eng.vt_max.max(16);
        if pr.is_empty() && cfg.exact_test > vmax {
            if cfg.exact_test + 4 > eng.max_seq() {
                eprintln!("talus-cli: --exact-test position exceeds --max-context");
                return 2;
            }
            pr.resize(cfg.exact_test as usize, 0);
            for (i, slot) in pr.iter_mut().enumerate() {
                *slot = 100 + ((i * 131) % 30000) as i32;
            }
            if fz.is_empty() {
                let nt = std::env::var("TALUS_EXACT_T")
                    .ok()
                    .and_then(|v| v.parse::<i32>().ok())
                    .unwrap_or(4)
                    .clamp(1, vmax);
                for i in 0..nt {
                    fz.push(198 + i);
                }
            }
        } else {
            let n = cfg.exact_test.min(vmax);
            if (pr.len() as i32) + n > eng.max_seq() {
                eprintln!("talus-cli: --exact-test position exceeds --max-context");
                return 2;
            }
            if fz.is_empty() {
                for i in 0..n {
                    fz.push(198 + i);
                }
            }
            fz.truncate(n as usize);
        }
        let ok = eng.exact_test(&pr, &fz);
        talus_model::qwen35::cluster::cluster_bye(&mut eng);
        return if ok { 0 } else { 1 };
    }

    if cfg.livecold_test > 0 {
        return livecold_test(&mut eng, &prompt_tokens, cfg.livecold_test);
    }

    if cfg.serve {
        let mut backend = Q36StdioBackend::new(eng, cfg);
        if let Some(port) = cfg.serve_port {
            let max_ctx = backend.max_context().max(1) as usize;
            let tokenizer = talus_text::Tokenizer::from_gguf(Path::new(model_path)).ok();
            let byte_tok = talus_server::ByteTokenizer::new();
            let srv_eng = Arc::new(Q36HttpServerEngine {
                inner: Mutex::new(backend),
                max_ctx,
                tokenizer,
                byte_tok: byte_tok.clone(),
                cancel_flag: Arc::new(AtomicBool::new(false)),
            });
            let mut srv_cfg = talus_server::ServerConfig::default();
            srv_cfg.port = port;
            srv_cfg.max_context = max_ctx;
            srv_cfg.model_path = model_path.clone();
            let state = Arc::new(talus_server::ServiceState::new(srv_cfg, srv_eng, byte_tok));
            let server = match talus_server::TalusServer::bind(state) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("talus-cli: bind error on port {port}: {e}");
                    return 1;
                }
            };
            println!("[talus] listening on http://{}", server.local_addr());
            let _ = io::stdout().flush();
            server.run_blocking();
            return 0;
        }
        talus_mem::pressure::install(
            talus_mem::pressure::PressureConfig::strix_default(),
            "talus-serve",
        );
        // E6: talus-server sends json_schema= GEN keys only to engines that advertise this
        println!("INFO grammar=json_schema");
        let _ = io::stdout().flush();
        return serve_stdio(&mut backend);
    }

    if cfg.bench_depth > 0 {
        let d_depth = cfg.bench_depth.max(32);
        let n_steps = cfg.bench_steps.max(1);
        // one plain token per step, then up to one window of tokens per speculative round (8, or the DFlash2
        // block when TALUS_Q27_DFL_BLOCK runs a longer one), plus 16 rows of slack
        let round_max = if eng.use_dflash && eng.df.ready {
            eng.df.block().max(8)
        } else {
            8
        };
        let need = d_depth + n_steps * (1 + round_max) + 16;
        if need > eng.max_seq() {
            eprintln!("talus-cli: --bench-depth needs --max-context >= {need}");
            return 2;
        }
        let mut rng = SplitMix64::new(1234);
        let mut toks = [0i32; 16];
        for slot in &mut toks {
            *slot = (rng.next_u64() % 200_000) as i32;
        }
        eng.reset();
        eng.n_past = d_depth - 16;
        eng.process(&toks, true);
        let mut nx = eng.greedy_from_host_logits();
        let t1 = Instant::now();
        for _ in 0..n_steps {
            let one = [nx];
            eng.process(&one, true);
            nx = eng.greedy_from_host_logits();
        }
        let tg_ms = t1.elapsed().as_secs_f64() * 1000.0;
        println!(
            "bench-depth {}: decode {:.2} ms/token ({:.2} tok/s)",
            d_depth,
            tg_ms / (n_steps as f64),
            1000.0 * (n_steps as f64) / tg_ms.max(1e-9)
        );

        for &w in &cfg.bench_window {
            if w < 1 || w > eng.gemm_t || eng.n_past + n_steps * w + 16 > eng.max_seq() {
                println!("bench-window {}: skipped", w);
                continue;
            }
            let mut wt = vec![0i32; w as usize];
            for slot in &mut wt {
                *slot = (rng.next_u64() % 200_000) as i32;
            }
            eng.process(&wt, true);
            let _ = eng.greedy_from_host_logits();
            let tw = Instant::now();
            for _ in 0..n_steps {
                eng.process(&wt, true);
            }
            let _ = eng.greedy_from_host_logits();
            let w_ms = tw.elapsed().as_secs_f64() * 1000.0;
            println!(
                "bench-depth {}: window {}: {:.2} ms/window ({:.2}x a decode step)",
                d_depth,
                w,
                w_ms / (n_steps as f64),
                w_ms / tg_ms.max(1e-9)
            );
        }

        if eng.mtp_d > 0 {
            let mut acc = 0i64;
            let mut toks_out = 0i64;
            let mut offered = 0i64;
            let t2 = Instant::now();
            if eng.dev_rounds {
                eng.spec_start_window(nx, eng.mtp_d);
                let mut picks = [0i32; talus_model::qwen35::engine::Q27_VMAX];
                let mut win = [0i32; talus_model::qwen35::engine::Q27_VMAX];
                for i in 0..n_steps {
                    let mut dw = 0i32;
                    eng.spec_verify_window(&mut dw, &mut win, &mut picks);
                    let mut a = 0usize;
                    while (a as i32) < dw && picks[a] == win[a + 1] {
                        a += 1;
                    }
                    acc += a as i64;
                    offered += dw as i64;
                    toks_out += (a as i64) + 1;
                    if i + 1 < n_steps {
                        eng.spec_advance_window(a as i32, picks[a], eng.mtp_d, None);
                    } else {
                        eng.spec_finish_window(a as i32, picks[a]);
                    }
                }
            } else {
                let mut drafts = Vec::new();
                let mut bonus = 0i32;
                for _ in 0..n_steps {
                    drafts.clear();
                    drafts.push(nx);
                    let (a, _, _) = eng.spec_step(&mut drafts, &mut bonus);
                    acc += a as i64;
                    offered += eng.mtp_d as i64;
                    toks_out += (a as i64) + 1;
                    nx = bonus;
                }
            }
            let sp_ms = t2.elapsed().as_secs_f64() * 1000.0;
            println!(
                "bench-depth {}: MTP {}: {:.2} ms/round, {} of {} drafts accepted, {:.2} tok/s",
                d_depth,
                eng.mtp_d,
                sp_ms / (n_steps as f64),
                acc,
                offered,
                1000.0 * (toks_out as f64) / sp_ms.max(1e-9)
            );
        }
        return 0;
    }

    if cfg.bench_pp > 0 || cfg.bench_tg > 0 {
        let pp = cfg.bench_pp.max(1);
        if pp + cfg.bench_tg + 2 > eng.max_seq() {
            eprintln!("talus-cli: bench exceeds --max-context");
            return 2;
        }
        let mut rng = SplitMix64::new(1234);
        let mut ptoks = vec![0i32; pp as usize];
        for slot in &mut ptoks {
            *slot = (rng.next_u64() % 200_000) as i32;
        }
        for rep in 0..3 {
            eng.reset();
            let t0 = Instant::now();
            eng.process(&ptoks, true);
            let mut nx = eng.greedy_from_host_logits();
            let pp_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let t1 = Instant::now();
            for _ in 0..cfg.bench_tg {
                let one = [nx];
                eng.process(&one, true);
                nx = eng.greedy_from_host_logits();
            }
            let tg_ms = t1.elapsed().as_secs_f64() * 1000.0;
            println!(
                "bench {}: pp{} {:.1} tok/s, tg{} {:.1} tok/s",
                rep,
                ptoks.len(),
                (ptoks.len() as f64) / (pp_ms / 1000.0).max(1e-9),
                cfg.bench_tg,
                if cfg.bench_tg > 0 {
                    (cfg.bench_tg as f64) / (tg_ms / 1000.0).max(1e-9)
                } else {
                    0.0
                }
            );
        }
        return 0;
    }

    let mut prompt = prompt_tokens;
    if prompt.is_empty() {
        if let Some(ref text) = cfg.prompt {
            if let Ok(tok) = talus_text::Tokenizer::from_gguf(Path::new(model_path)) {
                prompt = tok
                    .encode(text, true)
                    .into_iter()
                    .map(|u| u as i32)
                    .collect();
            }
        }
    }
    if prompt.is_empty() {
        eprintln!("talus-cli: pass --serve, --tokens ID,ID,..., or --bench-pp/--bench-tg");
        return 2;
    }
    let n_gen = if !force_tokens.is_empty() {
        force_tokens.len() + 1
    } else {
        cfg.max_new.max(0) as usize
    };
    if prompt.len() + n_gen + 2 > eng.max_seq() as usize {
        eprintln!("talus-cli: prompt + max_new exceeds --max-context");
        return 2;
    }

    let mut dump_file = match cfg.dump_full_logits.as_deref() {
        Some(path) => match File::create(path) {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!("talus-cli: cannot open {path}: {e}");
                return 1;
            }
        },
        None => None,
    };

    let write_logits_row = |f_opt: &mut Option<File>, eng: &Q36Engine| {
        if let Some(ref mut f) = f_opt {
            let v = eng.n_vocab().max(0) as usize;
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    eng.logits_h.cast::<u8>(),
                    v * std::mem::size_of::<f32>(),
                )
            };
            let _ = f.write_all(bytes);
        }
    };

    let use_spec = eng.mtp_d > 0
        && (eng.model().has_mtp || eng.use_dflash)
        && force_tokens.is_empty()
        && dump_file.is_none()
        && n_gen > 0;

    let t0 = Instant::now();
    eng.process(&prompt, true);
    let pp_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut out_toks: Vec<i32> = Vec::new();
    let mut offered = 0i64;
    let mut accepted = 0i64;
    let t1 = Instant::now();

    if n_gen > 0 {
        if use_spec {
            let nx = eng.greedy_from_host_logits();
            let eos: [i32; 0] = [];
            if cfg.ngram > 0 {
                let ng_n = if cfg.ngram > 1 {
                    cfg.ngram as usize
                } else {
                    24
                };
                let mut ng = NgramMod::with_n(ng_n);
                for &t in &prompt {
                    ng.feed(t);
                }
                let greedy = Sampling {
                    temp: 0.0,
                    top_p: 1.0,
                    min_p: 0.0,
                    top_k: 1,
                    rep_pen: 1.0,
                    pres_pen: 0.0,
                    freq_pen: 0.0,
                    rep_last_n: 0,
                    seed: 1,
                };
                let mut recent = prompt.clone();
                eng.spec_run(
                    eng.mtp_d,
                    n_gen as i32,
                    &eos,
                    nx,
                    &greedy,
                    &mut recent,
                    &mut out_toks,
                    &mut offered,
                    &mut accepted,
                    Some(&mut ng),
                );
            } else {
                eng.spec_greedy(
                    eng.mtp_d,
                    n_gen as i32,
                    &eos,
                    nx,
                    &mut out_toks,
                    &mut offered,
                    &mut accepted,
                );
            }
            if out_toks.len() > n_gen {
                out_toks.truncate(n_gen);
            }
        } else {
            for s in 0..n_gen {
                let nx = eng.greedy_from_host_logits();
                write_logits_row(&mut dump_file, &eng);
                out_toks.push(nx);
                if s + 1 < n_gen {
                    let next_in = if s < force_tokens.len() {
                        force_tokens[s]
                    } else {
                        nx
                    };
                    let one = [next_in];
                    eng.process(&one, true);
                }
            }
        }
    }
    let tg_ms = t1.elapsed().as_secs_f64() * 1000.0;

    print!("gen:");
    for &t in &out_toks {
        print!(" {t}");
    }
    println!();

    let gen_denom = if use_spec {
        out_toks.len()
    } else {
        out_toks.len().saturating_sub(1)
    };
    eprintln!(
        "talus-cli: prompt {} tok in {:.1} ms ({:.1} tok/s), gen {} tok in {:.1} ms ({:.2} tok/s)",
        prompt.len(),
        pp_ms,
        (prompt.len() as f64) / (pp_ms / 1000.0).max(1e-9),
        out_toks.len(),
        tg_ms,
        (gen_denom as f64) / (tg_ms / 1000.0).max(1e-9)
    );
    if use_spec && offered > 0 {
        eprintln!(
            "talus-cli: MTP {} drafts: {} of {} accepted ({:.0}%)",
            eng.mtp_d,
            accepted,
            offered,
            100.0 * (accepted as f64) / (offered as f64)
        );
    }
    talus_model::qwen35::cluster::cluster_bye(&mut eng);

    0
}

/// Runs `--livecold-test N`, verifying that warm prompt-cache turns produce bitwise identical
/// first-token logits and `N` greedy tokens compared to cold execution (`q36_engine.cu:3386-3455`).
pub fn livecold_test(eng: &mut Q36Engine, ids: &[i32], n_tok: i32) -> i32 {
    let grid = eng.max_batch.max(1) as usize;
    let n = n_tok.max(0) as usize;
    if ids.len() < grid * 2 + 64
        || n_tok < 1
        || (ids.len() as i32) + 2 * n_tok + 400 > eng.max_seq()
    {
        println!(
            "livecold-test: needs --tokens >= {} ids and --max-context >= their count + 2 N + 400",
            grid * 2 + 64
        );
        return 2;
    }
    let mut na = ids.len() * 2 / 3;
    if na % grid == 0 {
        na -= 7;
    }
    let a_part = ids[..na].to_vec();
    let b_part = ids[na..].to_vec();
    let vocab = eng.n_vocab().max(0) as usize;

    let answer =
        |eng: &mut Q36Engine, l0: &mut Vec<f32>, live: Option<&mut Vec<i32>>| -> Vec<i32> {
            let mut out = Vec::with_capacity(n);
            let sl = unsafe { std::slice::from_raw_parts(eng.logits_h, vocab) };
            l0.clear();
            l0.extend_from_slice(sl);
            let mut live_opt = live;
            for i in 0..n {
                let t = eng.greedy_from_host_logits();
                out.push(t);
                if i + 1 == n {
                    break;
                }
                let one = [t];
                eng.process(&one, true);
                if let Some(ref mut lv) = live_opt {
                    lv.push(t);
                }
            }
            out
        };

    let cold = |eng: &mut Q36Engine, p: &[i32], l0: &mut Vec<f32>| -> Vec<i32> {
        eng.reset();
        eng.process(p, true);
        answer(eng, l0, None)
    };

    let first_diff = |a: &[i32], b: &[i32]| -> i32 {
        let mut i = 0usize;
        while i < a.len() && i < b.len() && a[i] == b[i] {
            i += 1;
        }
        if i == a.len() && i == b.len() {
            -1
        } else {
            i as i32
        }
    };

    let mut pc = Q36PromptCache::new();
    eng.cl_pc_init(&mut pc); // talus#47: shard over the cluster stages (no-op on one process)
    let mut live: Vec<i32> = Vec::new();
    let mut live_pf: usize = 0;

    let mut turn =
        |eng: &mut Q36Engine, p: &[i32], l0: &mut Vec<f32>| -> (Vec<i32>, usize, &'static str) {
            let lcp = live
                .iter()
                .zip(p.iter())
                .take_while(|(x, y)| x == y)
                .count();
            let mut view = make_layer_state_view(eng);
            pc.stash_kv_with_pf(&view, &live, lcp, live_pf);
            eng.cl_pc_stash(&live, lcp, live_pf);
            let (reuse, mut from) = pc.restore_with_pf(&mut view, p, &live, live_pf, true);
            eng.cl_pc_restore(p, &live, live_pf, true, reuse);
            if reuse > 0 {
                sync_layer_state_view_back(eng, &view);
                live.clear();
                live.extend_from_slice(&p[..reuse]);
            } else {
                eng.reset();
                live.clear();
                from = "none";
            }
            let mut at = reuse;
            let g_end = (p.len() / grid) * grid;
            if g_end > at {
                eng.process(&p[at..g_end], true);
                live.extend_from_slice(&p[at..g_end]);
                at = g_end;
                if (eng.n_past as usize) == live.len() {
                    let mut v2 = make_layer_state_view(eng);
                    pc.save(&mut v2, &live, false);
                    eng.cl_pc_save(&live, false);
                }
            }
            if at < p.len() {
                eng.process(&p[at..], true);
                live.extend_from_slice(&p[at..]);
            }
            live_pf = live.len();
            let out = answer(eng, l0, Some(&mut live));
            (out, reuse, from)
        };

    struct WarmCase {
        name: &'static str,
        p: Vec<i32>,
        w: Vec<i32>,
        l0: Vec<f32>,
        reuse: usize,
        from: &'static str,
    }
    let mut warm: Vec<WarmCase> = Vec::new();
    let mut lw = Vec::new();
    let mut lc = Vec::new();
    let mut lo = Vec::new();

    eng.reset();
    let (a1, _, _) = turn(eng, &a_part, &mut lw);
    let mut p2 = a_part.clone();
    p2.extend_from_slice(&a1);
    p2.extend_from_slice(&b_part);
    let (t2_out, t2_reuse, t2_from) = turn(eng, &p2, &mut lw);
    warm.push(WarmCase {
        name: "turn 2 (cache)",
        p: p2.clone(),
        w: t2_out.clone(),
        l0: lw.clone(),
        reuse: t2_reuse,
        from: t2_from,
    });

    let _ = turn(eng, &b_part, &mut lo);
    let mut p3 = p2.clone();
    p3.extend_from_slice(&t2_out);
    p3.extend_from_slice(&a_part[..300.min(a_part.len())]);
    let (t3_out, t3_reuse, t3_from) = turn(eng, &p3, &mut lw);
    warm.push(WarmCase {
        name: "turn 3 (snapshot + KV stash)",
        p: p3,
        w: t3_out,
        l0: lw.clone(),
        reuse: t3_reuse,
        from: t3_from,
    });

    let mut fails = 0;
    for x in &warm {
        let c = cold(eng, &x.p, &mut lc);
        let same_l = x.l0.len() == lc.len()
            && x.l0
                .iter()
                .zip(lc.iter())
                .all(|(u, v)| u.to_bits() == v.to_bits());
        let d = first_diff(&x.w, &c);
        let ok = same_l && d < 0 && x.reuse > 0;
        if !ok {
            fails += 1;
        }
        let d_str = if d < 0 {
            "= cold".to_string()
        } else {
            format!("differ from cold at {d}")
        };
        println!(
            "livecold-test: {}: {} tokens, restored {} ({}): first logits {}, {} tokens {}  {}",
            x.name,
            x.p.len(),
            x.reuse,
            x.from,
            if same_l {
                "bitwise cold's"
            } else {
                "DIFFER from cold"
            },
            n_tok,
            d_str,
            if ok { "ok" } else { "FAIL" }
        );
    }

    eng.reset();
    eng.process(&a_part, true);
    let mut l1 = Vec::new();
    let _ = answer(eng, &mut lw, Some(&mut l1));
    eng.process(&p2[(a_part.len() + l1.len())..], true);
    let wl = answer(eng, &mut lw, None);
    let c2 = cold(eng, &p2, &mut lc);
    let dl = first_diff(&wl, &c2);
    let same_l2 = lw.len() == lc.len()
        && lw
            .iter()
            .zip(lc.iter())
            .all(|(u, v)| u.to_bits() == v.to_bits());
    println!(
        "livecold-test: (info) --cache-live continuation of turn 2: first logits {}, tokens {}",
        if same_l2 {
            "bitwise cold's"
        } else {
            "differ from cold's"
        },
        if dl < 0 {
            "= cold".to_string()
        } else {
            format!("differ from cold at {dl}")
        }
    );
    println!(
        "livecold-test: grid {}, A {} tokens: {}",
        grid,
        a_part.len(),
        if fails > 0 { "FAIL" } else { "ok" }
    );
    if fails > 0 {
        1
    } else {
        0
    }
}

/// Parses command-line arguments and executes the CLI, returning the exit status code.
pub fn run_cli_with_args<I, S>(args: I) -> i32
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let argv: Vec<String> = args.into_iter().map(|a| a.as_ref().to_string()).collect();
    for arg in argv.iter().skip(1).take(2) {
        if arg == "--version" || arg == "-V" {
            println!(
                "talus {} (based on strata 0.1.44.1)",
                env!("CARGO_PKG_VERSION")
            );
            return 0;
        }
        if arg == "--families" {
            println!("qwen4exp qwen35 qwen35moe laguna mimo2 glm4moe glm5-next");
            return 0;
        }
    }
    // Flash-Next (qwen4exp / --pack): dispatched to the Rust qwen4exp orchestrator before the Qwen35/Laguna/MiMo parser
    if talus_model::qwen4exp::is_qwen4exp_invocation(&argv) {
        return talus_model::qwen4exp::run_qwen4exp_cli(&argv);
    }
    // GLM-5.3-Flash : dispatched on the GGUF architecture before the Qwen/Laguna/MiMo parser
    if let Some(m) = argv
        .iter()
        .position(|a| a == "--model" || a == "--native" || a == "-m")
        .and_then(|i| argv.get(i + 1))
    {
        if talus_model::glm::is_glm_gguf(std::path::Path::new(m)) {
            return glm_serve::run(&argv);
        }
    }
    match parse_cli_args(argv) {
        Ok(cfg) => run_cli(&cfg),
        Err(e) => {
            eprintln!("{e}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn catch_up_trigger_pauses_after_misses_and_comes_back_on_a_continuation() {
        let mut t = super::CatchUpTrigger::default();
        assert_eq!(t.observe(&[1, 2, 3]), None); // first request: nothing armed
        assert!(!t.paused(2));
        t.arm(&[1, 2, 3]);
        assert_eq!(t.observe(&[1, 2, 3, 4]), Some(true));
        assert_eq!(t.misses, 0);
        t.arm(&[1, 2, 3]);
        assert_eq!(t.observe(&[1, 2]), Some(false)); // too short to use the chunk
        assert!(!t.paused(2));
        assert_eq!(t.observe(&[7]), None); // armed only once per reply: not a second miss
        assert_eq!(t.misses, 1);
        t.arm(&[1, 2, 3]);
        assert_eq!(t.observe(&[1, 9, 3, 4]), Some(false));
        assert!(t.paused(2));
        assert!(!t.paused(0)); // 0 = the v0.1.5 behaviour
        assert!(!t.paused(3));
        t.arm(&[5, 6]); // armed while paused too, which is how it comes back
        assert_eq!(t.observe(&[5, 6, 7]), Some(true));
        assert!(!t.paused(2));
    }

    use super::*;
    use std::path::PathBuf;

    #[test]
    fn kv_flag_warning_k15() {
        let cfg = parse_cli_args(["--model", "m.gguf", "--kv", "int8"]).unwrap();
        assert!(cfg.kv_explicit && cfg.kv_q8);
        assert!(kv_flag_warning("mimo", &cfg).unwrap().contains("F16"));
        assert!(kv_flag_warning("qwen35", &cfg)
            .unwrap()
            .contains("TALUS_Q27_KV8"));
        assert!(kv_flag_warning("laguna", &cfg).is_none());
        // the default (q8, not passed) and an explicit f16 say nothing
        let cfg = parse_cli_args(["--model", "m.gguf"]).unwrap();
        assert!(!cfg.kv_explicit && cfg.kv_q8);
        assert!(kv_flag_warning("mimo", &cfg).is_none());
        let cfg = parse_cli_args(["--model", "m.gguf", "--kv", "f16"]).unwrap();
        assert!(kv_flag_warning("qwen35", &cfg).is_none());
    }

    #[test]
    fn ladder_marks_k17() {
        // off: the two stops of before
        assert_eq!(ladder_marks(0, 4096, 98304, 2048, 0, 3), vec![4096, 98304]);
        // 96K prompt from scratch: spacing widens to span/4 (24K), 3 points
        assert_eq!(
            ladder_marks(0, 0, 98304, 2048, 16384, 3),
            vec![0, 24576, 49152, 73728, 98304]
        );
        // short prompt: spacing 16K; 32768 is too close to the grid end
        assert_eq!(
            ladder_marks(0, 0, 36864, 2048, 16384, 3),
            vec![0, 16384, 36864]
        );
        // resumed at 40K: points only past the restored length; one too close to cg is dropped
        assert_eq!(
            ladder_marks(40960, 51200, 90112, 2048, 16384, 3),
            vec![51200, 65536, 81920, 90112]
        );
        // every mark is on the grid
        for m in ladder_marks(2048, 6144, 1 << 20, 2048, 10000, 3) {
            assert_eq!(m % 2048, 0);
        }
    }

    #[test]
    fn crate_identity() {
        assert_eq!(CRATE_NAME, "talus-cli");
    }

    #[test]
    fn parse_cli_args_defaults_and_serve_variants() {
        let cfg = parse_cli_args(["--model", "qwen35.gguf", "--serve"]).unwrap();
        assert_eq!(cfg.model.as_deref(), Some("qwen35.gguf"));
        assert!(cfg.serve);
        assert!(cfg.engine_protocol);
        assert_eq!(cfg.max_context, 32768);
        assert_eq!(cfg.chunk, 0);
        assert!(cfg.use_graph);
        assert!(cfg.warmup);
        assert_eq!(cfg.prompt_cache, 8);
        assert_eq!(cfg.kv_stash_mib, 8192);
        assert_eq!(cfg.snap_msg, 1024);
        assert_eq!(cfg.max_new, 64);
        assert_eq!(cfg.bench_steps, 32);
        assert_eq!(cfg.hybrid_rows, 128);
        assert_eq!(cfg.prompt_cache_disk, None);

        let d = parse_cli_args([
            "--model",
            "m.gguf",
            "--serve",
            "--prompt-cache-disk",
            "auto",
            "--prompt-cache-disk-gib",
            "48",
            "--prompt-cache-disk-min",
            "2048",
        ])
        .unwrap();
        assert_eq!(d.prompt_cache_disk.as_deref(), Some("auto"));
        let dc = disk_tier::disk_config(&d).unwrap();
        assert_eq!(dc.budget_bytes, 48 << 30);
        assert_eq!(dc.min_tokens, 2048);
        let off = parse_cli_args(["--model", "m.gguf", "--prompt-cache-disk", "off"]).unwrap();
        assert!(disk_tier::disk_config(&off).is_none());

        // Leading `serve` subcommand with `--engine-protocol`
        let cfg2 = parse_cli_args([
            "talus-cli",
            "serve",
            "--native",
            "model.gguf",
            "--engine-protocol",
            "--ctx",
            "65536",
            "--prefill",
            "auto",
        ])
        .unwrap();
        assert!(cfg2.serve);
        assert!(cfg2.engine_protocol);
        assert_eq!(cfg2.model.as_deref(), Some("model.gguf"));
        assert_eq!(cfg2.max_context, 65536);
        assert_eq!(cfg2.chunk, 0);
    }

    #[test]
    fn parse_cli_args_mtp_dflash_and_spec_flags() {
        // `--mtp 5` sets draft count
        let cfg_num = parse_cli_args([
            "--model",
            "base.gguf",
            "--mtp",
            "5",
            "--mtp-main-head",
            "--draft-vocab",
            "vocab.gguf",
            "--dflash-gguf",
            "dflash.gguf",
            "--drafter",
            "auto",
            "--cache-live",
            "1",
            "--livecold-test",
            "16",
            "--ngram",
            "4",
        ])
        .unwrap();
        assert_eq!(cfg_num.mtp, 5);
        assert_eq!(cfg_num.mtp_gguf, None);
        assert!(cfg_num.mtp_main_head);
        assert_eq!(cfg_num.draft_vocab.as_deref(), Some("vocab.gguf"));
        assert_eq!(cfg_num.dflash_gguf.as_deref(), Some("dflash.gguf"));
        assert_eq!(cfg_num.drafter.as_deref(), Some("auto"));
        assert_eq!(cfg_num.cache_live, 1);
        assert_eq!(cfg_num.livecold_test, 16);
        assert_eq!(cfg_num.ngram, 4);

        // `--mtp FILE.gguf` sets mtp_gguf path
        let cfg_file = parse_cli_args(["--model", "base.gguf", "--mtp", "extra_mtp.gguf"]).unwrap();
        assert_eq!(cfg_file.mtp, 0);
        assert_eq!(cfg_file.mtp_gguf.as_deref(), Some("extra_mtp.gguf"));

        // `--mtp-gguf PATH` and `--spec-draft-vocab PATH`
        let cfg_explicit = parse_cli_args([
            "--model",
            "base.gguf",
            "--mtp",
            "3",
            "--mtp-gguf",
            "sep.gguf",
            "--spec-draft-vocab",
            "dv.gguf",
        ])
        .unwrap();
        assert_eq!(cfg_explicit.mtp, 3);
        assert_eq!(cfg_explicit.mtp_gguf.as_deref(), Some("sep.gguf"));
        assert_eq!(cfg_explicit.draft_vocab.as_deref(), Some("dv.gguf"));
    }

    #[test]
    fn parse_cli_args_benchmarks_and_hardware_flags() {
        let cfg = parse_cli_args([
            "--model",
            "m.gguf",
            "--no-graph",
            "--no-warmup",
            "--chunk",
            "512",
            "--prompt-cache",
            "16",
            "--kv-stash-mib",
            "4096",
            "--snap-msg",
            "512",
            "--tokens",
            "1,2,3",
            "--max-new",
            "128",
            "--force",
            "4,5",
            "--dump-full-logits",
            "logits.bin",
            "--bench-pp",
            "512",
            "--bench-tg",
            "128",
            "--bench-depth",
            "2048",
            "--bench-window",
            "2,6,16",
            "--bench-steps",
            "50",
            "--mmvq-bench",
            "attn_q",
            "--exact-test",
            "64",
            "--attn-test",
            "--attn-bench",
            "--expert-vram",
            "12.5",
            "--pool-workers",
            "8",
            "--expert-slots",
            "32",
            "--hybrid-rows",
            "256",
            "--gpu",
            "1",
            "--greedy",
        ])
        .unwrap();

        assert!(!cfg.use_graph);
        assert!(cfg.no_graph);
        assert!(!cfg.warmup);
        assert!(cfg.no_warmup);
        assert_eq!(cfg.chunk, 512);
        assert_eq!(cfg.prompt_cache, 16);
        assert_eq!(cfg.kv_stash_mib, 4096);
        assert_eq!(cfg.snap_msg, 512);
        assert_eq!(cfg.tokens.as_deref(), Some("1,2,3"));
        assert_eq!(cfg.max_new, 128);
        assert_eq!(cfg.force.as_deref(), Some("4,5"));
        assert_eq!(cfg.dump_full_logits.as_deref(), Some("logits.bin"));
        assert_eq!(cfg.bench_pp, 512);
        assert_eq!(cfg.bench_tg, 128);
        assert_eq!(cfg.bench_depth, 2048);
        assert_eq!(cfg.bench_window, vec![2, 6, 16]);
        assert_eq!(cfg.bench_steps, 50);
        assert_eq!(cfg.mmvq_bench.as_deref(), Some("attn_q"));
        assert_eq!(cfg.exact_test, 64);
        assert!(cfg.attn_test);
        assert!(cfg.attn_bench);
        assert!((cfg.expert_vram - 12.5).abs() < 1e-6);
        assert_eq!(cfg.cpu_threads, 8);
        assert_eq!(cfg.expert_slots, 32);
        assert_eq!(cfg.hybrid_rows, 256);
        assert_eq!(cfg.gpu, Some(1));
        assert!(cfg.greedy);
    }

    #[test]
    fn parse_cli_args_unknown_flags_in_serve_mode_vs_non_serve() {
        // In `--serve` mode, unknown `--flag [val]` options are logged and skipped without error
        let cfg = parse_cli_args([
            "serve",
            "--model",
            "m.gguf",
            "--unknown-opt",
            "foo",
            "--flag-without-val",
            "--ctx",
            "8192",
        ])
        .unwrap();
        assert!(cfg.serve);
        assert_eq!(cfg.max_context, 8192);
        assert_eq!(
            cfg.ignored_flags,
            vec![
                "--unknown-opt foo".to_string(),
                "--flag-without-val".to_string()
            ]
        );

        // Outside `--serve` mode, unknown flags fail with error
        let err = parse_cli_args(["--model", "m.gguf", "--unknown-opt", "foo"]).unwrap_err();
        assert!(err.contains("unknown argument: --unknown-opt"));
    }

    #[test]
    fn parse_ids_arg_inline_and_at_file() {
        assert_eq!(
            parse_ids_arg("151644, 872 198\n248046").unwrap(),
            vec![151644, 872, 198, 248046]
        );

        let scratch_dir = std::env::var("TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."));
        fs::create_dir_all(&scratch_dir).unwrap();
        let file_path = scratch_dir.join("talus_cli_test_tokens.txt");
        fs::write(&file_path, "101, 102\n103\t104\r\n105").unwrap();

        let spec = format!("@{}", file_path.display());
        let ids = parse_ids_arg(&spec).unwrap();
        let _ = fs::remove_file(&file_path);
        assert_eq!(ids, vec![101, 102, 103, 104, 105]);
    }

    #[test]
    fn sample_test_flag_dispatches_selftest() {
        let cfg = parse_cli_args(["--sample-test"]).unwrap();
        assert!(cfg.sample_test);
        assert_eq!(run_cli(&cfg), 0);
    }

    #[test]
    fn parse_cli_args_laguna_flags_and_model_detection() {
        let cfg = parse_cli_args([
            "talus-laguna",
            "--model",
            "laguna.gguf",
            "--n-layers",
            "24",
            "--dump-layers",
            "--dump-logits",
            "out.f32",
            "--top",
            "10",
            "--batch",
            "4",
            "--ubatch",
            "512",
            "--generate",
            "32",
            "--kv",
            "f16",
            "--force-decode",
            "8",
            "--dump-decode-only",
            "--verify-test",
            "16",
            "--spec",
            "5",
            "--spec-min-match",
            "6",
            "--sparse",
            "32,8192,64,512,32",
            "--eos-ids",
            "2,24,100",
        ])
        .unwrap();

        assert_eq!(cfg.model.as_deref(), Some("laguna.gguf"));
        assert_eq!(cfg.n_layers, 24);
        assert!(cfg.dump_layers);
        assert_eq!(cfg.dump_logits.as_deref(), Some("out.f32"));
        assert_eq!(cfg.top, 10);
        assert_eq!(cfg.batch, 4);
        assert_eq!(cfg.ubatch, 512);
        assert_eq!(cfg.generate, 32);
        assert!(!cfg.kv_q8);
        assert_eq!(cfg.force_decode, 8);
        assert!(cfg.dump_decode_only);
        assert_eq!(cfg.verify_test, 16);
        assert_eq!(cfg.spec, 5);
        assert_eq!(cfg.spec_min_match, 6);
        assert!(cfg.sparse);
        assert_eq!(cfg.sparse_k, 32);
        assert_eq!(cfg.sparse_min, 8192);
        assert_eq!(cfg.sparse_sink, 64);
        assert_eq!(cfg.sparse_recent, 512);
        assert_eq!(cfg.sparse_bs, 32);
        assert_eq!(cfg.eos_ids, vec![2, 24, 100]);

        if let Some(shapes_dir) = std::env::var_os("TALUS_SHAPES_DIR").map(PathBuf::from) {
            let hdr = shapes_dir.join("hdr/Laguna-S-2.1-Q4-lean-00001-of-00003.gguf.hdr");
            if hdr.exists() {
                assert!(is_laguna_model(&hdr));
            }
            let q35_hdr = shapes_dir.join("hdr/Qwen3.5-27B-Q4_K_M.gguf.hdr");
            if q35_hdr.exists() {
                assert!(!is_laguna_model(&q35_hdr));
            }
        }
    }

    #[test]
    fn parse_cli_args_mimo_flags_and_model_detection() {
        let cfg = parse_cli_args([
            "talus-mimo",
            "--model",
            "models/gguf/MiMo-V2.6-Flash/MiMo-V2.6-Flash-UD-IQ2_XXS-00001-of-00002.gguf",
            "--dflash-gguf",
            "models/gguf/MiMo-V2.6-Flash/DFlash/MiMo-V2-Flash-DFlash-Q8_0.gguf",
            "--spec-draft-vocab",
            "models/gguf/MiMo-V2.6-Flash/DFlash/mimo-dv-65536.bin",
            "--mtp",
            "4",
            "--ctx",
            "196608",
            "--serve",
        ])
        .unwrap();

        assert_eq!(
            cfg.model.as_deref(),
            Some("models/gguf/MiMo-V2.6-Flash/MiMo-V2.6-Flash-UD-IQ2_XXS-00001-of-00002.gguf")
        );
        assert_eq!(
            cfg.dflash_gguf.as_deref(),
            Some("models/gguf/MiMo-V2.6-Flash/DFlash/MiMo-V2-Flash-DFlash-Q8_0.gguf")
        );
        assert_eq!(
            cfg.draft_vocab.as_deref(),
            Some("models/gguf/MiMo-V2.6-Flash/DFlash/mimo-dv-65536.bin")
        );
        assert_eq!(cfg.mtp, 4);
        assert_eq!(cfg.max_context, 196608);
        assert!(cfg.ctx_explicit);
        assert!(cfg.serve);
        assert!(cfg.engine_protocol);

        if let Some(shapes_dir) = std::env::var_os("TALUS_SHAPES_DIR").map(PathBuf::from) {
            let mimo_hdr = shapes_dir.join("hdr/MiMo-V2.6-Flash-MOPD-a4-00001-of-00003.gguf.hdr");
            if mimo_hdr.exists() {
                assert!(is_mimo_model(&mimo_hdr));
                assert!(!is_laguna_model(&mimo_hdr));
            }

            let laguna_hdr = shapes_dir.join("hdr/Laguna-S-2.1-Q4-lean-00001-of-00003.gguf.hdr");
            if laguna_hdr.exists() {
                assert!(!is_mimo_model(&laguna_hdr));
            }
        }

        // Verify `talus_spec::dflash::DFlashModelProfile::mimo_v2_flash_dflash()` and
        // `talus_model::mimo::MimoDFlashProfile::mimo_v2_flash()` match on 5-layer geometry,
        // 152,064 Q8_0 head, 64K draft vocab (`mimo-dv-65536.bin`), and `n4`.
        let spec_prof = talus_spec::dflash::DFlashModelProfile::mimo_v2_flash_dflash();
        let mimo_prof = talus_model::mimo::MimoDFlashProfile::mimo_v2_flash();
        assert_eq!(spec_prof.n_layers, mimo_prof.block_count);
        assert_eq!(spec_prof.d_model, mimo_prof.embedding_length);
        assert_eq!(spec_prof.n_q, mimo_prof.n_head);
        assert_eq!(spec_prof.n_kv, mimo_prof.n_head_kv);
        assert_eq!(spec_prof.target_vocab, mimo_prof.target_vocab_size);
        assert_eq!(spec_prof.head_vocab, mimo_prof.head_vocab_size);
        assert_eq!(spec_prof.draft_vocab_size, mimo_prof.draft_vocab_size);
        assert_eq!(spec_prof.max_draft, mimo_prof.default_draft_max);
        assert_eq!(spec_prof.tap_layers, mimo_prof.target_layer_taps);

        if let Some(dv_path) = talus_model::mimo::resolve_mimo_draft_vocab_path() {
            let dv_vec = talus_model::mimo::weights::load_draft_vocab_bin(&dv_path)
                .expect("load mimo-dv-65536.bin");
            assert_eq!(dv_vec.len(), 65_536);
            let target_vocab = talus_model::mimo::N_VOCAB_UD as i32;
            let in_range: Vec<i32> = dv_vec
                .into_iter()
                .filter(|&id| id >= 0 && id < target_vocab)
                .collect();
            let dv = talus_spec::DraftVocab::from_ids(&in_range, target_vocab)
                .expect("build DraftVocab::from_ids");
            let drafter = talus_spec::dflash::DFlashDrafterSpec::mimo_v2_flash()
                .with_draft_vocab(dv)
                .expect("attach draft vocab to DFlashDrafterSpec::mimo_v2_flash");
            assert_eq!(drafter.profile.max_draft, 4);
        }
    }
}
