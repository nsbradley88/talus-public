# How Talus is measured

This page explains how we measure Talus: what runs, on which machine, what counts as a pass, and how to run the same
protocol yourself. The numbers are in [RESULTS.md](RESULTS.md), generated from each model's recorded freeze run. No
other figures appear in this repository. A model whose release run has not passed shows "pending".

## Hardware and power

Every Talus figure comes from one machine:

| Item | Value |
|---|---|
| Machine | HP ZBook Ultra G1a (a laptop) |
| APU | AMD Ryzen AI Max+ 395 with a Radeon 8060S iGPU (gfx1151, RDNA 3.5, 40 compute units) |
| Memory | 128 GB LPDDR5X, unified: CPU and GPU share it (the GPU uses it as GTT memory) |
| Power | 70 W firmware ceiling. We change no clocks or power limits. |
| Kernel / boot | Linux 7.0, headless (`multi-user.target`, no desktop session). Kernel parameters: `ttm.pages_limit=31457280 ttm.page_pool_size=31457280` (GPU memory pool 120 GiB instead of the default), `kho=off` (Ubuntu's default kexec-handover reserve otherwise keeps about 9 GiB away from the GPU), `amd_iommu=off`, `amdgpu.lockup_timeout=10000,60000,10000,10000` (compute jobs may run 60 s). GLM-5.3-Flash and MiMo-V2.6-Flash do not fit without the first two. See [MODELS.md](MODELS.md#system-setup). |
| OS / runtime | Linux; ROCm SDK 10.2.0a20261014 |
| Talus build | release commit 43c94474c43d; each model's figures come from its own frozen release run (see [RESULTS.md](RESULTS.md)) |

Notes on the context:

- **One model at a time.** The box serves one model and nothing else, so each engine can spend memory for speed.
  We keep at least 8 GiB of `MemAvailable` free while serving (GLM-5.3-Flash runs closest to it, at about 9 GiB). The engine's memory
  accountant refuses any allocation that would take it below that floor.
- **70 W is low for this chip.** Most published Strix Halo results come from desktops and mini-PCs that run the same
  APU at 85-140 W. Expect higher numbers on those machines, and lower ones under thermal load on a laptop.
- **Served, end to end.** Every figure is measured through the HTTP server, with the client on another machine on
  the local network, one request at a time, and a fresh server per model and run. So it includes tokenization,
  templating, sampling, speculative decoding and the HTTP stack, as a user would see it.

## The matrix

A change is adopted only if the cross-model matrix shows no regression on any of the five models. The matrix is a
registry of (engine, model) pairs. Each entry records the exact serving configuration (arguments, environment, weight
files, context length) and a baseline. The release run measures the configurations that Talus ships by default.

Tier 1 (speed, every candidate build):

| Cell | What it measures |
|---|---|
| `chat` | Decode speed over a mix of six chat prompts, sampled at temperature 1.0 / top-p 0.95, up to 700 tokens each |
| `code` | Decode speed over six coding prompts, greedy, thinking off, up to 700 tokens each |
| `edit` | Decode speed over three whole-file edits of a C++ source file (about 9 KB each), greedy, up to 1,500 tokens |
| `pp2k`, `pp32k` | Prompt (prefill) speed for a cold prompt of about 2K and 32K tokens, with the prompt cache off |
| `tg2k`, `tg32k` | Decode speed of a 128-token greedy continuation after that 2K / 32K prompt (multi-prompt, see below) |
| TTFT a / b / c | Time to first streamed token: a = cold 2K prompt, b = cold 32K prompt, c = a warm follow-up that adds about 300 new tokens to the cached 32K conversation |
| MemAvailable | The lowest `MemAvailable` seen during the run |

The decode cells report generated tokens divided by decode time, summed over the mix. Speculative decoding is on as
each model ships it, so the cells include draft acceptance. TTFT is recorded passively from the streamed requests;
we run no separate latency job.

The `edit` prompts are three source files from llama.cpp (MIT), each with a mechanical instruction ("add a doc comment
above every function that lacks one", "rename these locals", "prefix log messages with the function name"). Edits
repeat most of the input, so they reward drafters that copy from the prompt.

The 2K / 32K prompts are cut from a fixed corpus: the source files of the Python standard library (`[a-m]*.py`,
sorted by name, concatenated), followed by "Continue the code above:".

### Multi-prompt decode cells

With speculative decoding, the decode speed after a long prompt depends on the text that the model generates: some
continuations are easy to draft and some are not. Two builds whose numerics differ slightly can choose different
continuations of the same prompt, and the single-prompt cell then measures the text more than the engine. On
Qwen3.8-27B we saw one 32K prompt move by about 21% (one standard deviation) between builds
whose per-step speed was the same.

So `tg2k` and `tg32k` use four prompts, cut at corpus offsets 0, 100K, 200K and 300K tokens. The cell is all
generated tokens over all decode time across the four. The four-prompt aggregate still carries about
10.5% (one standard error) of text-driven noise between numerically different builds or
engines. We do not claim a `tg` win smaller than that against another engine.

### Noise bands

The same build measured in two server sessions does not give identical numbers. For each cell we measured that
session-to-session spread on repeat pairs (the same configuration in two sessions). The band is twice the RMS of the
relative differences, rounded up to the next half point, with a floor of one point:

| Cell | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a | TTFT b | TTFT c |
|---|---|---|---|---|---|---|---|---|---|---|
| Band | 3% | 3.5% | 2.5% | 6% | 5% | 4% | 4% | 6% | 5% | 8.5% |

A cell that is slower than its baseline by more than the band is a **regression**. A difference inside the band is a
tie. These bands describe one build's session noise (the same greedy text). Between builds whose text differs, judge
`tg2k` / `tg32k` with the text noise above instead.

## Quality gates

Speed counts only if the output is right. Each model has to pass all of these at the release build:

| Gate | What it checks | Pass |
|---|---|---|
| Needle 60K | Long-context recall: five codenames hidden at five evenly spaced depths (from a tenth to nine tenths of the way in) of a 60K-token prompt from the corpus; the model must list all five. Three seeds. | every codename in every seed |
| Needle 81K | The same at 81K tokens, for the models served with more than 64K of context (Flash-Next, MiMo, Laguna) | every codename in every seed |
| Agentic | Five SWE-bench-style tasks on the Go library `charmbracelet/bubbles` v0.21.1, driven by the opencode agent against the served model; graded offline by hidden Go tests that the agent never sees | all five tasks |
| KLD | Kullback-Leibler divergence of the model's next-token distribution against a reference, per model (below) | within the model's threshold |
| Greedy identity | The greedy outputs of the `code` and `edit` mixes are byte-identical to the previous gated build, where the change claims to keep numerics unchanged | every output identical |
| Soak | Continuous mixed traffic on the served configuration (below) | see below |

The needle prompts use temperature 1.0 / top-p 0.95 with a fixed seed. The agentic suite runs the model's own
sampling defaults. 64K-context configurations (Qwen3.8-27B, GLM-5.3-Flash) let opencode compact the conversation.

### KLD references per model

| Model | Reference | Threshold |
|---|---|---|
| Qwen3.8-27B | Teacher-forced KL against llama.cpp's CPU backend on the same GGUF | KL vs llama.cpp CPU reference (UD-IQ4_XS) <= 0.00291 (short prompts) and 4K-prompt KL ratio vs exact <= 1.35 |
| Qwen3.8-Flash-Next | First-token KL over a fixed prompt set (4.5K to 32K tokens) against a reference engine run | first-token KL vs reference (43 prompts 4.5K-32K) <= 0.0275 (ref 0.02623) |
| GLM-5.3-Flash | llama.cpp's `llama-perplexity` KL method against two references: the logits of the release quant lineage, and the logits of the original FP8 model on a gate text (gated) and on code (recorded) | KLD vs release logits <= 0.0440 (ref 0.04194); FP8 gate-text line <= 0.20 (ref 0.1934); FP8 code recorded |
| MiMo-V2.6-Flash | Teacher-forced KL from the previous gated build to this build at 2K and 32K | vs original weights: kld-text <= 0.182 (same-top >= 88.5%), wiki2 <= 0.240; vs old release ref: KLD <= 0.10, PPL ratio <= 1.01; teacher-forced KL(previous gated build -> this build) @2K and @32K: bitwise, or mean <= 1e-3 and top-1 >= 0.99 |
| Laguna-S-2.1 | Teacher-forced KL from the previous gated build to this build at 2K and 32K (with a relaxed threshold for the served-speed decode path's long-context rounding drift) | teacher-forced KL(previous gated build -> this build) @2K and @32K: bitwise, or mean <= 3e-3 and top-1 >= 0.98 (served-speed numerics drift at 32K) |

The thresholds are provisional: each is the current reference value plus a small margin, and the maintainer confirms
them before a release.

**Qwen3.8-27B, changes to the prompt path only.** The short-prompt teacher-forced check above runs one row at a time,
so it exercises the decode matvec (`mmvq`) even when a change only touches the chunked prompt path (`ubatch >= 64`).
Prompt-only changes on 27B (such as the 4-bit prompt GEMM in [MODELS.md](MODELS.md#qwen38-27b)) are therefore also
checked on four 4,096-token prompts cut from the corpus at offsets 0, 100K, 200K and 300K, processed in 2,048-token
chunks as in serving and scored over the last 256 positions of each prompt (1,024 positions total) against
llama.cpp's CPU backend on the same GGUF. In the same run we measure both the exact int8 prompt path
(`TALUS_Q27_W4I=0`, the default) and the opt-in `.w4i` prompt path (`TALUS_Q27_W4I=auto`) with the same flash attention and dense MMQ settings; the gate requires the `.w4i` path's mean KL to stay within the ratio threshold in the table above (`KL vs llama.cpp CPU reference (UD-IQ4_XS) <= 0.00291 (short prompts) and 4K-prompt KL ratio vs exact <= 1.35`), matching the headroom of the short-prompt gate. We also score the two paths against each other to record their direct divergence (reported in [RESULTS.md](RESULTS.md)).

### Soak

The soak runs rounds of mixed traffic against the served model for 45 minutes or more: two `code` requests (greedy),
two `chat` requests (sampled, seeded), one `edit` (greedy), a cold 2K prompt, a cold 32K prompt and a warm 32K
follow-up. It passes when there are no request errors or hangs, every greedy answer is byte-identical across rounds
(against round 0 for cold-prompt requests, and against round 1 across rounds 1..N for requests served from the prefix
cache such as warm 32K once the cached prefix reaches steady state; sampled kinds such as `chat` are never scored for
greedy identity, even when `cache_n > 0` from a shared template prefix), `MemAvailable` stays above the floor, the engine's resident memory grows by no more than
10% from the end of warm-up (the later of the first full prompt rotation and 20 minutes of elapsed
traffic, once every `code`, `chat` and `edit` prompt and lazy KV allocation has been touched) to the last round and does not grow monotonically over the second half (warm-up RSS
growth is reported separately as informational), and each request kind's median speed in the second half is
within 3% of the first half when computed over whole prompt-rotation periods (each half trimmed to a
multiple of the rotation period so per-rotation speed patterns do not read as drift; sampled `chat` is reported but not judged).

## "Exact" and "opt-in numerics"

Every kernel change in Talus falls into one of two classes.

- **Exact.** The change keeps every logit bit for bit the same as before. We check this with greedy byte-identity
  over the matrix mixes and with the engines' self-tests: decode and verify rows equal the single-row result, and
  speculative greedy output equals plain greedy. Speculative decoding is exact in this sense: greedy requests give the
  same text as plain greedy decoding, and sampled requests are rejection-sampled against the draft distribution, so
  they keep the model's output distribution. Exact changes can become the default as soon as the matrix shows no
  regression.
- **Opt-in numerics.** The change rounds differently (for example a faster attention kernel that accumulates in a
  different order, or a lower-precision prompt path). It ships behind an environment variable, off by default. It
  can become the default only if it passes the KLD gate within the noise of the current default, needle at 60K and
  the full agentic suite, and the maintainer then decides. The defaults that change numerics are listed per model,
  with their kill switches, in [MODELS.md](MODELS.md).

What "exact" covers for the prompt cache: a request that restores cached state gives the same answer as the same
request read cold. Prompt (prefill) kernels and decode kernels round differently, so the cache restores only state
that a prefill computed and re-reads the rest; it never restores rows that a decode step wrote. A warm answer
therefore equals a cold one, bit for bit.

## Comparisons with other engines

- **llama.cpp.** Our llama.cpp rows come from our own patched build of llama.cpp (we call it "halo4": upstream
  llama.cpp with our Vulkan and speculative-decoding patches), with the best configuration we found for each model.
  They are not stock llama.cpp results.
- **Talus C++.** The previous Talus engine (C++ host over the same kernel family), where it supports the model.
- **gufo.** [gufo-org/gufo](https://github.com/gufo-org/gufo) (MIT, commit `82c62b4`), built from its Nix flake and
  measured on the same Strix Halo machine in a single-session `Talus control -> gufo -> Talus control` sandwich for
  Qwen3.8-27B and Qwen3.8-Flash-Next, using `gufo`'s documented `UD-Q4_K_XL` weight files (`gufo` does not load our
  `UD-IQ4_XS` files), plus a `Talus (UD-Q4_K_XL)` row on 27B to separate the weight-size difference from the engines.
- **Third-party published numbers.** Other Strix Halo numbers published on the web mostly come from desktops and
  mini-PCs at 85-140 W with different quantizations, prompts and "prompt at depth" definitions, so they are not
  directly comparable to cold prompts on a 70 W laptop.

## Reproducing

You need a Strix Halo machine with 128 GB, ROCm, the release build and the weights listed in [MODELS.md](MODELS.md).
The clients are in `bench/`. They talk to the server at `TALUS_URL` (default `http://127.0.0.1:8000`) with the key
from `TALUS_API_KEY` (or `TALUS_API_KEY_FILE`); run them from another machine, one at a time.

1. Build and serve the model as in the [README](../README.md), with the shipped defaults (no tuning variables set).
2. Tier 1:
   - `bench/bench27.py LABEL chat,code,edit` for the three mixes (`edit` uses the files in `bench/editsrc/`;
     `DUMP=FILE` writes the greedy outputs for the byte-identity check);
   - `DEPTH_OFFSETS=0,100000,200000,300000 bench/bench.py LABEL 2000,32000` for `pp2k`, `pp32k`, the four-prompt
     `tg2k` / `tg32k` and the TTFT samples (`bench/chatdepth.py` does the same through `/v1/chat/completions`).
     The corpus is the Python standard library of the client machine.
3. Gates:
   - `bench/needle.py LABEL 60000` (and `81000`; seeds 1, 2 and 3 by default);
   - `bench/agentic/run_agentic.py LABEL <provider/model>` with `BUBBLES_SRC` (a checkout of charmbracelet/bubbles
     v0.21.1) and `opencode` on `PATH`; `AGENTIC_AUTOCOMPACT=1` for 64K-context models;
   - `bench/soak.py LABEL MINUTES` for the soak;
   - the KLD check with the CLI's KLD mode against the model's reference (above): teacher-forced logits from
     llama.cpp's CPU backend on the same GGUF (Qwen3.8-27B), first-token logits over the fixed prompt set from the
     reference engine with numerics switches off (Qwen3.8-Flash-Next), `llama-perplexity --kl-divergence` against
     logits dumped on the original FP8 weights and the quant lineage (GLM-5.3-Flash), or teacher-forced
     full logits (`--dump-full-logits`) at 2K and 32K, scored as KL(reference || build) against the reference build with
     numerics switches off (MiMo-V2.6-Flash and Laguna-S-2.1).
4. Compare cell by cell with the noise bands above. Compare `tg` cells across engines only on the four-prompt
   aggregate.

Results inside a noise band are a tie. A release run records every cell, the gate results and a manifest (commit
and binary checksums) of the build it measured; [RESULTS.md](RESULTS.md) is generated from those records.
