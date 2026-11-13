# How Talus is measured

This page explains how we measure Talus: what runs, on which machine, what counts as a pass, and how to run the same
protocol yourself. The numbers are in [RESULTS.md](RESULTS.md), generated from the recorded release runs. No
other figures appear in this repository. A model whose release run has not passed shows "pending".

## Hardware and power

Talus figures come from two machines with the same chip and memory. Every table in [RESULTS.md](RESULTS.md) names its
machine; a figure from one is not comparable with a figure from the other.

| Item | Laptop | Desktop |
|---|---|---|
| Machine | HP ZBook Ultra G1a | GMKtec EVO-X2 |
| APU | AMD Ryzen AI Max+ 395 with a Radeon 8060S iGPU (gfx1151, RDNA 3.5, 40 compute units) | the same |
| Memory | 128 GB LPDDR5X, unified: CPU and GPU share it (the GPU uses it as GTT memory) | the same |
| Power | 70 W firmware ceiling (HP's sustained package power limit). ACPI platform profile `performance`, set at every boot (the firmware default is `balanced`, under which GLM-5.3-Flash decoded at 11.2 tok/s instead of about 24 tok/s); `talus setup` offers the same setting. | About 120 W sustained. No ACPI platform profile: the BIOS power mode is set to its highest sustained mode. |
| Kernel / boot | Linux 7.0, headless (`multi-user.target`, no desktop session). Kernel parameters: `ttm.pages_limit=31457280 ttm.page_pool_size=31457280` (GPU memory pool 120 GiB instead of the default), `kho=off` (Ubuntu's default kexec-handover reserve otherwise keeps about 9 GiB away from the GPU), `amd_iommu=off`, `amdgpu.lockup_timeout=10000,60000,10000,10000` (compute jobs may run 60 s). GLM-5.3-Flash and MiMo-V2.6-Flash do not fit without the first two. `amd_iommu=off` turns the IOMMU off: we have not measured its effect on speed (Talus runs with it on), it makes the NPU unavailable, and `talus setup` does not set it. See [MODELS.md](MODELS.md#system-setup). | The same kernel parameters, `amd_iommu=off` included |
| OS / runtime | Linux; ROCm SDK 10.2.0a20261014 | the same |

Which figures come from which machine:

| Figures | Machine | Build |
|---|---|---|
| Speed of the current release (RESULTS "Speed" tables, the README's first table) | desktop | the v0.1.9 release candidate tarball (rc1; the same engine source as the published tarball, see RESULTS) |
| The previous release on the same machine (RESULTS "v0.1.2" rows and the change table) | desktop | the published v0.1.2 tarball, interleaved with the current release in the same session |
| Qwen3.8-27B 4-bit weights against BF16 (RESULTS) | desktop | v0.1.2 (release candidate 2); not re-measured for v0.1.3 |
| Quality gates of the current release | laptop (some soaks on the desktop; RESULTS names the machine) | the same release (RESULTS says which build each gate ran on) |
| Comparisons with other engines (llama.cpp "halo4", gufo, Talus C++) | laptop | the v0.1.0 release runs; not re-measured since |

Notes on the context:

- **One model at a time.** The box serves one model and nothing else, so each engine can spend memory for speed.
  We keep at least 8 GiB of `MemAvailable` free while serving (GLM-5.3-Flash runs closest to it, at about 9 GiB). The engine's memory
  accountant refuses any allocation that would take it below that floor.
- **Power.** 70 W is low for this chip; the desktop holds about 120 W. Other machines with the same APU run it at
  different limits (most published Strix Halo results come from desktops and mini-PCs at 85-140 W). Expect different
  numbers on them, and lower ones under thermal load on a laptop. Apart from the platform profile and the BIOS power
  mode we change no clocks or power limits through Talus.
- **Laptop host changes after the v0.1.0 release runs.** The v0.1.0 runs (2026-10-28) used the settings in the table.
  Since 2026-10-29 the laptop loads `amd_pmf` with `smart_pc_support=0` (HP's Smart PC power policy off), and since
  2026-10-31 it raises the firmware's skin-temperature target from 43 C to 45 C, so the 70 W limit holds through long
  jobs instead of sagging to about 55-58 W. The v0.1.2 and v0.1.3 quality gates ran with both.
- **Served, end to end, as shipped.** Every figure is measured through the HTTP server, with the client on another
  machine on the local network, one request at a time, and a fresh server per model and run. So it includes
  tokenization, templating, sampling, speculative decoding and the HTTP stack, as a user would see it. From v0.1.2 the
  release runs unpack the release tarball for each run and serve its own `talus-server` and engine with the run
  configs `talus setup` writes (arguments, environment, sampling block, stop tokens, the tokenizer from the GGUF), on
  the model files listed in [MODELS.md](MODELS.md). The v0.1.0 runs went through an earlier Python server with the
  same engine invocations.

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

So `tg2k` and `tg32k` use several prompts. In the current release runs: 12 prompts at about 2K tokens (corpus
offsets 0, 30K, 60K, 100K, 130K, 160K, 200K, 230K, 260K, 300K, 330K and 360K tokens) and 8 at about 32K (0, 50K,
100K, 150K, 200K, 250K, 300K and 340K). The cell is all generated tokens over all decode time across the prompts, and
`pp2k` / `pp32k` are likewise all prompt tokens over all prompt time across the same prompts; each prompt's own
figure is recorded too. The v0.1.0 runs (the comparison rows) used four prompts at offsets 0, 100K, 200K and 300K for
`tg` and the first prompt alone for `pp`. The four-prompt aggregate carries about 10.5% (one
standard error) of text-driven noise between numerically different builds or engines; we have not derived the figure
for the 12 / 8-prompt cells yet, so we do not claim a `tg` win smaller than the four-prompt figure against another
engine.

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

Speed counts only if the output is right. Each model has to pass all of these at the release build (for v0.1.3 on
the laptop, with the release candidate tarball and its `talus-server`):

| Gate | What it checks | Pass |
|---|---|---|
| Needle 60K | Long-context recall: five codenames hidden at five evenly spaced depths (from a tenth to nine tenths of the way in) of a 60K-token prompt from the corpus; the model must list all five. Three seeds. | every codename in every seed |
| Needle 81K | The same at 81K tokens, for the models served with more than 64K of context (Flash-Next, MiMo, Laguna) | every codename in every seed |
| Agentic smoke test | Five SWE-bench-style tasks on the Go library `charmbracelet/bubbles` v0.21.1, driven by the opencode agent against the served model; graded offline by hidden Go tests that the agent never sees. Each task runs once, so this is a smoke test: it catches broken tool use or editing, but cannot distinguish small quality differences (five pass/fail outcomes). Model quality is gated by KL divergence, greedy output identity and long-context recall | all five tasks |
| KLD | The logit rows a build produces, per model (below): for most models compared bit for bit with a frozen set; KL against an outside reference is recorded beside them | see the table below |
| Row parity | The rows the KLD gate scores come from one-row decoding. Serving verifies drafts in windows of several rows. On Qwen3.8-27B, GLM-5.3-Flash, MiMo-V2.6-Flash and Laguna-S-2.1 we compare the two bit for bit at each window width serving uses, at 2K and deeper | every row identical; a difference fails, whatever its size |
| Drafts on vs off | Greedy output with drafts on against drafts off, on prompts that place the reply across each model's kernel and window boundaries, all five models: three fresh engines per run, with a second drafts-off engine as the control. Flash-Next also compares a warm request with and without an earlier resend | every token identical; a difference fails, whatever its size |
| Greedy identity | The greedy outputs of the `code` and `edit` mixes are byte-identical to the previous gated build, where the change claims to keep numerics unchanged | every output identical |
| Soak | Continuous mixed traffic on the served configuration (below) | see below |

The needle prompts use temperature 1.0 / top-p 0.95 with a fixed seed. The agentic smoke test runs the model's own
sampling defaults. 64K-context configurations (GLM-5.3-Flash, and Qwen3.8-27B in the release runs) let opencode compact the conversation.

### KLD references per model

| Model | Reference | Threshold |
|---|---|---|
| Qwen3.8-27B | Teacher-forced rows on short prompts, UD-IQ4_XS and UD-Q4_K_XL, against a set frozen from this release's first candidate. KL against llama.cpp's CPU backend on the same GGUF is recorded, not gated | Teacher-forced logit rows on short prompts (UD-IQ4_XS, UD-Q4_K_XL) bitwise equal to a frozen set; any change goes to human review and does not pass by itself. KL vs the llama.cpp CPU reference and the 4K-prompt ratio are recorded, not gated |
| Qwen3.8-Flash-Next | First-token rows over a fixed prompt set (4.5K to 32K tokens) against a frozen set. KL against llama.cpp's CPU backend on the original BF16 weights (below) is the published value and the bar for a changed build | First-token rows (43 prompts 4.5K-32K) bitwise equal to a frozen set. A changed build also needs KL vs llama.cpp CPU on BF16 <= 0.1064 and top-1 >= 79%, then human review |
| GLM-5.3-Flash | Rows from the generation path (forced decode, one row at a time) at 2K and 16K, against a frozen set. KL with llama.cpp's `llama-perplexity` method against the release quant lineage and the original FP8 model (gate text and code) is recorded, not gated | KLD vs release logits <= 0.0440 (ref 0.04194); FP8 gate-text line <= 0.20 (ref 0.1934); FP8 code recorded |
| MiMo-V2.6-Flash | Engine: teacher-forced KL from a fixed earlier Talus build (not the previous release) to this build at 2K and 32K. Weights checks, not engine gates: three lines run llama.cpp on the GGUF against the original weights and an old release reference; the build under test does not run in them, so they only show the GGUF hasn't changed | Engine: teacher-forced KL(a fixed earlier Talus build -> this build) @2K and @32K: bitwise, or mean <= 1e-3 and top-1 >= 0.99. Weights checks, not engine gates (llama.cpp on the GGUF, the build does not run): vs original weights kld-text <= 0.182 (same-top >= 88.5%), wiki2 <= 0.240; vs old release ref: KLD <= 0.10, PPL ratio <= 1.01 |
| Laguna-S-2.1 | Served rows (decode through the verify kernels, as serving runs) teacher-forced at 2K and 32K, against a frozen served set. KL against a fixed earlier Talus build is recorded, not gated | Served rows @2K and @32K bitwise equal to a frozen set; a change needs KL vs frozen <= the rounding floor (8.2e-3 @2K, 1.4e-3 @32K), top-1 >= 0.98 and human review. KL(a fixed earlier Talus build -> this build) recorded, not gated |

**Frozen rows, and what a change means.** Rounding alone moves KL against one fixed reference by about as much as the
old KL limits allowed, so a limit that close could pass a real change or fail a harmless one. From v0.1.6, Qwen3.8-27B,
Qwen3.8-Flash-Next and Laguna-S-2.1 compare their rows bit for bit with a set frozen from this release's first candidate,
and GLM-5.3-Flash with a set frozen after its window fix (its replies changed in this release), each checked by hash.
Identical rows pass. Any change stops the gate for a person to review, with its KL against the frozen rows in the record;
it never passes by itself. These limits are checked by people before a release, and the frozen rows say a build hasn't
moved, not that it is closer to the true model. The KL against an outside reference says that, roughly, and stays in the record.

**Qwen3.8-27B, changes to the prompt path only.** The short-prompt teacher-forced check above runs one row at a time,
so it exercises the decode matvec (`mmvq`) even when a change only touches the chunked prompt path (`ubatch >= 64`).
Prompt-only changes on 27B (such as the 4-bit prompt GEMM in [MODELS.md](MODELS.md#qwen38-27b)) are therefore also
checked on four 4,096-token prompts cut from the corpus at offsets 0, 100K, 200K and 300K, processed in 2,048-token
chunks as in serving and scored over the last 256 positions of each prompt (1,024 positions total) against
llama.cpp's CPU backend on the same GGUF. In the same run we measure both the exact int8 prompt path
(`TALUS_Q27_W4I=0`) and the `.w4i` prompt path (the default when the v3 `.w4i` file is installed) with the same flash attention and dense MMQ settings; the ratio of the `.w4i` path's mean KL to the exact path's is recorded (it was gated at 1.35 until v0.1.5; history 1.236, 1.160, 1.145). We also score the two paths against each other to record their direct divergence (reported in [RESULTS.md](RESULTS.md)).

**Qwen3.8-Flash-Next reference.** The reference is llama.cpp's CPU backend (b11972, cf24317, the Zen 4 build) running
the original BF16 weights on the 43 fixed prompts, with one scored row per prompt: the next token after the prompt. The
previous reference was a slower run of our own engine on the same 4-bit file, which measured only how far a build moved
from another build of ours.

**Why the published number went from 0.0242 to 0.0967.** The logits didn't change: they are bit for bit the same as in
earlier releases. The reference changed. 0.0242 was the distance to another build of ours on the same 4-bit file. 0.0967
is the distance to the original weights on another engine, which also counts what the 4-bit file itself loses. For scale,
llama.cpp itself measures 0.128 on the same 4-bit file and 0.087 on Q8_0. The new number isn't worse quality, and the old
one wasn't better.

A build whose rows are identical to the frozen rows passes on that alone. A changed build also has to stay within ten
percent of the current default's mean KL, and keep top-1 agreement on at most two prompts fewer (the figures are in the table
above), and then goes to review.

Read this number with its limits:

- It is noisy. The bootstrap interval over prompts for the default's mean is 0.061 to 0.139, so ten percent is inside the sampling noise
  and a small change can pass or fail by chance. Three prompts carry about a third of the total.
- The reference is exact only for its own build. Thread count, batch size and repeat runs gave identical first-row logits,
  but the same BF16 weights on llama.cpp's Haswell CPU build moved the first-token score on five prompts, by 0.004 up to 0.127 (four
  of them among the gate's hardest). BF16 does not fit on these machines' GPUs, so there is no GPU cross-check as there is
  for Qwen3.8-27B. So the gate says a build is no further from this reference, not that it is closer to the true model.
- Smaller sets of prompts are recorded but not gated, because they are too noisy. A prompt-length band has 7 or 8 prompts,
  and one prompt can double a band's value.

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

v0.1.3: the Qwen3.8-27B soak had 0 errors and flat memory, but 5 of 96 greedy checks differed, all on one `edit`
prompt whose cold and cache-restored answers differ (the prompt-cache split effect under "Exact" below). The
maintainer accepted it for this release; RESULTS has the details.

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
  the agentic smoke test, and the maintainer then decides. The defaults that change numerics are listed per model,
  with their kill switches, in [MODELS.md](MODELS.md).

What "exact" covers for the prompt cache. A prompt is read in chunks, and an engine may round a row slightly differently
depending on where the chunks are cut. The cache only restores state at a cut point a cold read of the same text would
also use, and re-reads the rest; it never restores rows that a decode step wrote. So:

- An exact repeat or resend of a request gives the same answer as the same request read cold, checked on
  all five models (from v0.1.6 on Qwen3.8-27B, Qwen3.8-Flash-Next, MiMo-V2.6-Flash and Laguna-S-2.1; before it, a
  MiMo-V2.6-Flash resend could differ from cold, and on Qwen3.8-Flash-Next the turn after a resend could differ from the
  same conversation without the resend. From v0.1.7 on GLM-5.3-Flash: a resend used to resume from an earlier cut point
  and re-read the rest of the prompt in other chunks; it now resumes where the first pass ended).
- A conversation restored at the same cuts gives the same bits as reading it cold with those cuts.
- A later turn of a conversation, or a raw prompt that extends an earlier one, is cut at other points than the same
  text read cold in one go, so its answer can differ from that cold read. Both answers are valid, and the same history
  gives the same answer every time. The soak gate scores cached requests against their own first cached round for this
  reason.

Qwen3.8-27B, v0.1.2 to v0.1.5: a restored prefix that ended inside a prompt chunk was not cut where a cold read cuts,
so even a request whose start came from the cache could differ from the same request read cold (in the v0.1.3 soak,
one prompt whose first 2,048 tokens came from the cache). From v0.1.6 the cache keeps only the part up to the last cut
point both reads share, and that case matches cold.

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
   - `DEPTH_OFFSETS="0,30000,60000,100000,130000,160000,200000,230000,260000,300000,330000,360000;16000:0,50000,100000,150000,200000,250000,300000,340000" bench/bench.py LABEL 2000,32000`
     for `pp2k`, `pp32k`, the 12 / 8-prompt `tg2k` / `tg32k` and the TTFT samples (`bench/chatdepth.py` does the same through `/v1/chat/completions`).
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
4. Compare cell by cell with the noise bands above. Compare `tg` cells across engines only on multi-prompt
   aggregates with the same prompts, and only on the same machine.

Results inside a noise band are a tie. A release run records every cell, the gate results and a manifest (commit
and binary checksums) of the build it measured; [RESULTS.md](RESULTS.md) is generated from those records.
