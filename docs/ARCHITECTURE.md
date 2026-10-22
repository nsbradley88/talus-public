# Architecture

Talus has three layers:

1. **A HIP kernel library**, `libtalus_kernels.so`, built with CMake from `src/` and `include/`. It holds every GPU
   kernel: quantized matrix-vector products for decode, prompt GEMMs, attention, MoE routing and expert GEMMs, and the
   model-specific fused kernels. Its interface is a C ABI.
2. **A Rust host**, the Cargo workspace in `rust/`. It loads the weights, plans memory, runs each model's forward
   pass by calling the kernel library, runs speculative decoding, keeps the prompt cache, and speaks the engine
   protocol.
3. **An HTTP server**, `talus-server` (Rust, ported from Strata's Python server), that turns OpenAI and Anthropic API
   requests into engine requests. It starts `talus-cli` as a subprocess (see [Serving](#serving)).

```
 client --HTTP--> talus-server --stdin/stdout--> talus-cli --C ABI--> libtalus_kernels.so --> GPU
                 (tokenizer, chat template,          (model engine,        (HIP kernels, per-arch
                  tool calls, thinking)               spec decoding,        tuning tables)
                                                      prompt cache)
```

The kernel library is C++/HIP derived from Strata and ggml. The host was ported from Strata's C++ host to Rust.
Keeping the two apart lets the host change without rebuilding kernels, lets the kernels be tested from C++ and Rust
alike, and keeps one place (the C ABI) where a new GPU architecture has to plug in.

## The kernel library and its C ABI

The main header is `include/talus_kernels.h` ("Contract v0", `TALUS_KERNELS_ABI_VERSION 0`). The Rust build checks
this version when it compiles (`rust/crates/talus-kernels-sys/build.rs`). The contract's rules:

- **No exceptions cross the boundary.** Every `tk_*` function is `noexcept`, returns a `tk_status` (`TK_OK` is 0) and
  stores a message that `tk_last_error(ctx)` returns.
- **Configuration through `tk_config`.** The ABI has a `tk_config` handle passed to `tk_ctx_create`:
  `tk_config_create_for_arch` fills in an architecture's defaults and `tk_config_from_env` overlays the environment.
  This is the target design. Today the kernels still read their switches from the environment (`getenv`), and each
  engine sets its own defaults in the environment when they are unset, so a variable you export always wins. The
  switches are listed per model in [MODELS.md](MODELS.md).
- **Explicit weight handles.** Weights repacked at load time for faster decode (Q8_0 and Q6_K plane splits) are owned
  by `tk_packed_weights` handles instead of global registries.
- **The host owns the streams.** A `tk_stream` is a `hipStream_t` the host created. Both sides load the same
  `libamdhip64.so`, so a HIP graph that the host captures records the kernel launches made through the ABI. Entry
  points marked `[CAPTURABLE]` only enqueue device work, and take values that change per step (position, token id, KV
  length) as device pointers, so a captured graph replays correctly without being updated.

Exported names:

| Prefix | What |
|---|---|
| `tk_ctx_*`, `tk_config_*` | context, configuration, errors |
| `tk_mmvq*`, `tk_attn_*`, `tk_moe_*`, `tk_gdn_*`, `tk_prefill_*` | shared decode and prompt kernels |
| `tk_q36_*` | Qwen3.8-27B (dense, gated delta net + attention) kernels |
| `tk_laguna_*` | Laguna-S-2.1 kernels |
| `talus_pf_*` | prompt kernels: int8 GEMM on expanded weights, grouped MoE GEMM, SwiGLU into q8_1, prompt flash attention, W4A4 |

Two models have their own headers. `include/talus_glm_kernels.h` (`tkg_*`, versioned with `tkg_abi_version()`) holds
the GLM-5.3-Flash kernels. It takes plain device pointers and a stream, and every entry point is graph-capturable.
`include/talus_qwen4exp_kernels.h` (`tk_qwen4exp_*`) connects the Rust host to the Qwen3.8-Flash-Next data plane,
which still runs in C++ with Rust callbacks for the request loop, stop checks and logit bias.

The Rust host loads the library with `dlopen` at run time by default, so building and testing the host needs neither
ROCm nor the library. It looks for `TALUS_KERNELS_LIB` (a file), then `TALUS_KERNELS_PREBUILT` (a directory), then
the directory of the running executable. The simplest setup is to put `libtalus_kernels.so` next to `talus-cli`.
With the `link-kernels` Cargo feature, the host links the library at build time instead.

## Per-architecture tables (gfx1151 and the gfx1201 seam)

Kernel choices that depend on the GPU (tile sizes, rows per wave, variants, which exact fast paths are on) are kept in
tables keyed by the GPU's architecture name (`gcnArchName`, for example `gfx1151`). Only gfx1151 (Strix Halo,
Radeon 8060S) is measured and tuned. The others are seams: the code paths and enum values exist so that another GPU
can be added by adding a table row, not by changing the kernels or the host.

| Table | Where | gfx1151 | gfx1201 (Radeon AI PRO R9700, RDNA 4) and others |
|---|---|---|---|
| Architecture enum | `tk_arch` in `talus_kernels.h`; `GpuArch` in `rust/crates/talus-core-api` | `TK_ARCH_GFX1151` | `TK_ARCH_GFX1201`; anything else is `TK_ARCH_OTHER` (portable path) |
| Exact speed switches | `src/core/arch_defaults.cpp` (applied by the Flash-Next path; the other engines set their own defaults) | prompt and decode switches on (all bitwise) | gfx1200/gfx1201 get the prompt half only (the decode half was slower there) |
| Prompt kernels (`TpTune`) | `include/strata/prefill/tp_tune.hpp` | tuned row: GEMM variant per weight layout, MoE kernel version, fused gate/up, fp16 expert path | conservative default row |
| Wave32 decode matvec (`W32Rule`) | `src/kernels/cuda/native_mmvq.cu` (`kW32Gfx1151`) | rows per wave, waves-per-EU cap and column groups per (type, columns) | generic table |
| hipBLASLt solutions | `tools/hip/<arch>-hipblaslt-<version>.txt`, loaded by `rust/crates/talus-kernels/src/tuning.rs` | per ROCm version; when the runtime's version differs, the loader picks the file for the runtime | gfx1201 tables exist; RDNA 4 siblings share tables, gfx1151 needs an exact match |
| Rust `KernelSet` | `rust/crates/talus-kernels/src/kernel_set.rs` | populated | returns `UnsupportedArch` for now |

The kill switches for the architecture defaults are `STRATA_GFX1151_DEFAULTS=0` and `STRATA_GFX12_DEFAULTS=0`.
To build the kernels for more than one target, pass the list to CMake (for example
`-DCMAKE_HIP_ARCHITECTURES="gfx1151;gfx1201"`). The prompt WMMA kernels are compiled for gfx11 only.

## The Rust workspace

| Crate | Role |
|---|---|
| `talus-core-api` | The host's contract: traits and types shared by every crate (`ModelAdapter`, drafters, `SpecController`, `PrefixCache`, `MemoryModel`, `GpuArch`) |
| `hip-sys` | Raw HIP bindings, loaded with `dlopen`, plus a simulated runtime for machines without a GPU |
| `talus-kernels-sys` | Raw bindings to `talus_kernels.h` |
| `talus-hip` | Safe wrappers: device, stream, event, graph capture, device and host-mapped buffers |
| `talus-kernels` | Typed kernel calls, `KernelSet`, tuning tables, the FFI backend and a CPU reference backend used in tests |
| `talus-gguf` | GGUF v2/v3 reading with `mmap`, split shards, Strata packs, MTP packs, draft vocabularies |
| `talus-mem` | Memory placement on unified memory, the `MemAvailable` / cgroup probe, budgets and floors, arenas |
| `talus-text` | Tokenizer (BPE with per-model pre-tokenizers), Jinja chat templates, tool-call and thinking parsers |
| `talus-spec` | Speculative decoding: MTP, DFlash, n-gram, prompt lookup, suffix drafter, draft vocabularies, adaptive draft length, exact sampling |
| `talus-state` | Prompt cache: state snapshots, KV stash, prefix matching, disk sessions |
| `talus-model` | The per-model engines (below) |
| `talus-sched` | Request slot, queue, cancellation, and the stdin/stdout engine protocol |
| `talus-server` | HTTP server (OpenAI, Anthropic and llama.cpp-style endpoints) |
| `talus-cli` | The `talus-cli` binary: picks the engine for a GGUF and runs it, one-shot or as an engine process |
| `tools/oracle` | Compares golden dumps (tokens, logits, layer outputs) from a reference engine against Talus within tolerances |

`talus-cli` reads `general.architecture` from the GGUF and picks the engine:

| GGUF architecture | Engine (`rust/crates/talus-model/src/...`) | Model |
|---|---|---|
| `qwen4exp` (or `--pack`) | `qwen4exp` | Qwen3.8-Flash-Next |
| `glm5-next` | `glm` | GLM-5.3-Flash |
| `laguna` | `laguna` | Laguna-S-2.1 |
| `mimo2` | `mimo` | MiMo-V2.6-Flash |
| anything else | `qwen35` | Qwen3.8-27B (and other Qwen3.5/3.6 family models) |

## The per-model engines

Each engine implements the forward pass for one model family and shares the rest of the host: the memory planner,
the speculative-decoding service, the prompt cache and the protocol. What each engine adds:

- **`qwen35` (Qwen3.8-27B, dense).** Gated delta net and attention layers. Decode and verify run on wave32 matvecs
  over 1 to 8 columns, where every column is bitwise the single-column result, so verifying drafts gives the same
  numbers as decoding them one at a time. WMMA attention for decode and verify. Prompts run on an exact int8 GEMM over
  weights expanded at load time. Drafting: the model's MTP head, the DFlash2 block-diffusion drafter, or both with a
  switch by context depth, plus n-gram drafts and a 64K draft vocabulary. The prompt cache keeps snapshots of the
  recurrent state on a fixed grid.
- **`qwen4exp` (Qwen3.8-Flash-Next, MoE).** The Rust host drives Strata's C++ data plane through
  `talus_qwen4exp_kernels.h`; a pure-Rust port is in progress. Experts are streamed from a Strata pack with an expert
  cache, and prompts use fp16 expert kernels for large batches. Drafting: MTP with a window, plus a suffix drafter.
- **`glm` (GLM-5.3-Flash, MoE with MLA latent attention).** Kernels in `talus_glm_kernels.h`. Built-in NextN MTP
  drafting and a 64K draft vocabulary. Prompt chunks on fp16 experts. Fused decode kernels and a side stream, with
  greedy argmax on the device. The engine refuses to start without enough free memory and stops if `MemAvailable`
  falls below the floor.
- **`mimo` (MiMo-V2.6-Flash, MoE).** A DFlash drafter with a 64K draft vocabulary. Grouped prompt GEMMs and WMMA
  flash attention. Decode steps and drafter rounds are captured as HIP graphs.
- **`laguna` (Laguna-S-2.1, MoE with sliding-window layers).** The sliding-window layers keep a ring buffer, so only the
  full-attention layers grow with the context. Block-sparse attention for long contexts, prompt-lookup speculation,
  graph-captured decode with routing on the device.

Every model's defaults, opt-ins and kill switches are in [MODELS.md](MODELS.md).

## Memory

The laptop has 128 GB of unified memory that the CPU and GPU share. `talus-mem` places weights to avoid copies:
embedding tables stay in `mmap`ed host memory, and GPU buffers are allocated with the budget in mind. The budget reads
`/proc/meminfo` and the cgroup limits and keeps a floor of free memory (8 GiB by default, more for some models). One
model runs per machine, so engines may trade memory for speed, for example by expanding prompt weights to int8 at
load time.

## Serving

`talus-server --config <config.json>` starts the engine (`talus-cli` with the model's arguments and environment from
the config) and talks to it over stdin/stdout:

- the engine prints `READY <context> stop` when it has loaded;
- a request is `GEN <max_new> [key=value ...] <token ids>` (sampling parameters, seed, `cache`, `ignore_eos`);
- the engine answers with prompt progress (`PP`), the cached prefix it reused (`RESUME n`), one `T <id>` per token,
  then `DONE` (or `ERR`); `STOP` cancels a running request.

The server owns tokenization, chat templates, tool-call parsing and thinking control. It serves
`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`, the Anthropic `/v1/messages`, and
llama.cpp's `/tokenize`, `/detokenize` and `/completion`. It also serves `/health` and `/metrics`.

Streaming uses server-sent events. Auth is `Authorization: Bearer` or `x-api-key`, with the key from
`STRATA_API_KEY` or `--api-key`.

Strata's Python server (`serve/server.py --engine strata`) speaks the same protocol and reads the same config, so it
can drive `talus-cli` too.

## Tests

- `cargo test --workspace` (in `rust/`) runs without a GPU. It covers argument parsing, GGUF reading on synthetic
  files, memory budgets, the CPU reference backend, sampling, the drafters, the prompt cache, graph capture on the
  simulated HIP runtime, and the ABI against a C stub library. It needs `cc`, `gcc`, `g++` and a `python3` with the
  `regex` package (the tokenizer tests compare against `tools/strata_tokenizer.py`).
- `rust/ci.sh` runs `cargo fmt --check`, `cargo clippy -D warnings`, the tests and `cargo deny check licenses`.
- Tests that need real weights or a GPU skip when their inputs are missing. The engines' own self-tests
  (`--exact-test`, `--attn-test`, `--sample-test`, `--livecold-test`, `--verify-test`) run on the GPU with real weights.
- Model-quality gates (needle, agentic, KLD) run outside Cargo; see [METHODOLOGY.md](METHODOLOGY.md).
