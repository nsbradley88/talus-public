<p align="center"><img src="docs/assets/banner.jpg" alt="Talus: the LLM inference engine for AMD Strix Halo, written in Rust"></p>

# Talus

Talus runs large open-weight language models on one AMD Strix Halo machine (Ryzen AI Max+ 395 with the Radeon 8060S,
gfx1151) with 128 GB of unified memory. It is a Rust host over a library of HIP kernels tuned for that GPU, and it
serves each model through an OpenAI-compatible HTTP API.

## How fast is it?

The current release is **v0.1.9**: two-box mode with the prompt cache gives the right replies again, `--chunk` above
512 no longer faults at start-up, and Laguna-S-2.1 processes long prompts faster
([release notes](https://github.com/nsbradley88/talus-public/releases)). Greedy output is unchanged on all five models.

Tokens per second for Talus v0.1.9 (its release candidate 1) on a Ryzen AI Max+ 395 desktop (GMKtec
EVO-X2, 128 GB, about 120 W sustained, all 16 CPU cores online), end to end through the HTTP server, each model in its shipped configuration.
`chat` is the sampled chat mix, `code` the coding-prompt mix, `edit` the code-editing mix, `pp2k` / `pp32k` prompt
processing and `tg32k` generation at about 2K / 32K tokens of context. In brackets: the change from v0.1.8, measured
in the same session on the same desktop. Before each timed run, every weight file was read in
full and checked against its pinned SHA-256.

| Model | chat | code | edit | pp2k | pp32k | tg32k |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | 51.2 (0.0 %) | 66.2 (0.0 %) | 133.2 (0.0 %) | 598 | 554 (+1.3 %) | 38.6 (0.0 %) |
| Qwen3.8-Flash-Next | 62.4 (−1.0 %) | 73.6 (0.0 %) | 162.2 (−0.4 %) | 1,323 | 1,738 (+0.1 %) | 65.5 (0.0 %) |
| GLM-5.3-Flash | 33.4 (+1.5 %) | 34.8 (−0.3 %) | 35.5 (−0.3 %) | 577 | 584 (+0.3 %) | 35.0 (0.0 %) |
| MiMo-V2.6-Flash | 40.8 (+0.2 %) | 49.8 (0.0 %) | 65.1 (0.0 %) | 736 | 662 (0.0 %) | 42.8 (0.0 %) |
| Laguna-S-2.1 | 43.6 (−0.9 %) | 54.6 (0.0 %) | 123.8 (−0.1 %) | 1,160 | 1,034 (+2.8 %) | 36.8 (−0.3 %) |

The quality gates (long-context recall, an agentic smoke test on five coding tasks, logit rows against a frozen set,
greedy output with drafts on and off, a 45-minute soak test) ran on both machines: the soaks on the desktop, the others
mostly on our 70 W laptop (HP ZBook Ultra G1a); [docs/RESULTS.md](docs/RESULTS.md) says where each ran. The two
machines have the same chip but different power limits, so their figures are not comparable with each other.

How much the 27B's shipped 4-bit weights give up against the full BF16 weights. Speeds: v0.1.2 on the same desktop
(not re-measured since). KL and top-1 against Talus's BF16; the `UD-IQ4_XS` row is the shipped default, with the 4-bit
prompt weights on (measured on the laptop with a v0.1.6 build, same text and BF16 reference):

| Weights | Size | KL code | top-1 code | code | pp32k | tg32k, no speculation |
|---|---|---|---|---|---|---|
| BF16 (reference, opt-in) | 54.7 GB | reference | reference | 19.4 | 506 | 4.0 |
| `UD-Q4_K_XL` (option) | 17.6 GB | 0.0357 | 95.8 % | 57.4 | 487 | 11.3 |
| `UD-IQ4_XS` (shipped default) | 14.3 GB (+2.2 GB W4A4 prompt sidecar) | 0.1187 | 92.2 % | 67.8 | 546 | 13.2 |

Lower KL and higher top-1 = closer to BF16. With the exact int8 prompt path (`TALUS_Q27_W4I=0`, slower prompts)
`UD-IQ4_XS` measures KL 0.0815 on code. `code` is served with speculative decoding as above; `tg32k` here is plain
decode without it. Details, wiki text and all speed cells:
[docs/RESULTS.md](docs/RESULTS.md#qwen38-27b-our-4-bit-weights-against-bf16).

Against other engines on the same desktop, with the published v0.1.3 tarball (2026-11-03/04, one engine at a time,
the same HTTP client and prompts; package power about 117-120 W for every engine):

| Model | Engine | code | edit | pp32k | tg32k |
|---|---|---|---|---|---|
| Qwen3.8-27B | **Talus v0.1.3** | 66.7 | 123 | 542 | 37.2 |
| | **Talus v0.1.3** on `UD-Q4_K_XL` (the file gufo and tubo use) | 57.5 | 101 | 481 | 36.3 |
| | llama.cpp (our patched build) | 45.5 | 69.2 ¹ | 329 | 32.8 |
| | gufo (`82c62b4`, `UD-Q4_K_XL`) | 48.9 | 72.5 | 572 | 30.2 |
| | tubo (`2a258b9`, `UD-Q4_K_XL`) | 53.0 | 58.6 | 576 | 32.7 |
| Qwen3.8-Flash-Next | **Talus v0.1.3** | 74.2 | 144 | 1,475 | 66.8 |
| | gufo (`82c62b4`, `UD-Q4_K_XL`) | 65.6 | 105 | 1,710 | 55.1 |
| GLM-5.3-Flash | **Talus v0.1.3** | 34.5 | 36.5 | 564 | 36.5 |
| | llama.cpp (our patched build) | 31.6 | 31.6 | 265 | 28.8 |
| MiMo-V2.6-Flash | **Talus v0.1.3** | 49.2 | 64.4 | 661 | 42.6 |
| | llama.cpp (our patched build) | 43.0 | 53.0 | 341 | 36.0 |
| Laguna-S-2.1 | **Talus v0.1.3** | 54.6 | 122 | 1,003 | 35.0 |
| | llama.cpp (our patched build) | 47.9 | 89.9 | 621 | 33.6 |

¹ Host-specific, cause unknown: the same llama.cpp build and settings give 82.4 on our laptop.

Prompt processing is where Talus v0.1.3 trailed: gufo and tubo prefill the 27B's `UD-Q4_K_XL` file faster, and gufo
prefilled Flash-Next 16 % faster at 32K (47 % at 2K). Those gaps are
v0.1.3's: Talus's Flash-Next prompt processing was faster in each of v0.1.4 to v0.1.6, and in v0.1.7 to v0.1.9 its change is
within noise (v0.1.9 against v0.1.8 is in the first table above), which narrows them. The other engines have not been re-measured, so this table still shows v0.1.3.

Full tables (edit, 2K cells, time to first token, memory, the v0.1.2 figures from the same session, quality gates):
[docs/RESULTS.md](docs/RESULTS.md). Every figure there names its machine and build. A model whose release run has not
passed shows "pending". The llama.cpp rows are our own patched build ("halo4"), not stock llama.cpp; gufo and tubo
cannot load our `UD-IQ4_XS` files, so they ran their documented `UD-Q4_K_XL` files. Conditions for every engine:
[docs/RESULTS.md](docs/RESULTS.md#reference-engine-conditions-desktop-comparison). How everything is measured:
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

Talus is a fork of [Strata](https://github.com/Niko1221/Strata) (MIT). Strata runs Qwen3.8-Flash-Next on consumer
GPUs. Talus keeps Strata's kernels, ports the host and the server to Rust, and adds four more models and kernels tuned
for Strix Halo.

In geology, talus is the slope of broken rock that collects at the foot of strata.

## Supported hardware

| | |
|---|---|
| GPU | AMD Radeon 8060S (gfx1151) in the Ryzen AI Max+ 395 (Strix Halo) |
| Memory | 128 GB unified memory. Each model needs most of it, and GLM-5.3-Flash and MiMo-V2.6-Flash need the GPU memory pool raised with a kernel parameter: [System setup](docs/MODELS.md#system-setup). |
| OS | Linux with a ROCm SDK (see [Build](#build)) |

We develop and measure on two machines with the same chip and 128 GB: an HP ZBook Ultra G1a laptop, whose firmware
limits the APU to 70 W, and a GMKtec EVO-X2 desktop, which holds about 120 W. The speed figures come from the
desktop and most quality gates from the laptop; every figure in [docs/RESULTS.md](docs/RESULTS.md) names its machine.
Other Strix Halo machines may allow other power limits; we have not measured them.

The kernel library has per-architecture tables keyed by GPU target. Only gfx1151 is tuned. The code also knows
gfx1201 (Radeon AI PRO R9700), so that card can be added by filling in its tables; it is not supported yet.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how.

## Supported models

One model runs at a time; the machine serves nothing else.

| Model | Type | Weights we run | Speculative decoding |
|---|---|---|---|
| Qwen3.8-27B | dense | Unsloth UD-IQ4_XS | MTP head + DFlash2 drafter, switched by context depth; n-gram drafts; 64K draft vocabulary |
| Qwen3.8-Flash-Next | MoE | Unsloth UD-IQ4_XS | MTP head with a window, plus a suffix drafter |
| GLM-5.3-Flash | MoE, MLA attention | our mixed 4-bit quant ("a4e3s") | built-in NextN MTP layer, 64K draft vocabulary |
| MiMo-V2.6-Flash | MoE | 4-bit mixed quant ("MOPD-a4-x") | DFlash drafter (Q8_0), 64K draft vocabulary |
| Laguna-S-2.1 | MoE, sliding-window attention | 4-bit quant ("Q4-lean") | DFlash drafter (Q8_0) + prompt lookup |

Files, defaults, switches and memory per model: [docs/MODELS.md](docs/MODELS.md).

## Quick start

### Easy install

On a Strix Halo machine, one command installs everything into your home directory and walks you through choosing
and downloading a model:

```bash
curl -fsSL https://raw.githubusercontent.com/nsbradley88/talus-public/main/install.sh | bash
```

It checks the machine, installs the ROCm SDK and the latest release's prebuilt programs (it checks their SHA-256 and needs no
compiler, only git, curl and Python 3), and starts `talus setup`: pick models (Qwen3.8-27B is suggested for a first install), download them
(resumable, checksum-verified), and, for GLM-5.3-Flash and MiMo-V2.6-Flash, set the kernel parameters they need
(it shows the change and asks; it never reboots). Afterwards `talus doctor` checks and repairs the install,
`talus update` updates it, and `talus-tui` is the dashboard. Details: [docs/INSTALL-GUIDED.md](docs/INSTALL-GUIDED.md).

Each [release](https://github.com/nsbradley88/talus-public/releases) has `talus-<tag>-linux-x86_64-gfx1151.tar.gz` with its
`.sha256`, a build manifest and `SHA256SUMS`; `TALUS_RELEASE=<tag>` installs a particular one. To compile instead (about 40
minutes; the installer then also installs a compiler, CMake, Ninja and Rust, asking before it uses `sudo`):

```bash
curl -fsSL https://raw.githubusercontent.com/nsbradley88/talus-public/main/install.sh | bash -s -- --from-source
```

If a machine has no prebuilt programs, the installer says so and compiles.

With `--yes` (no questions) the user service is installed only when you add `--service`; interactive installs still
offer it. The release notes show how to unpack the tarball by hand; [scripts/release/README.md](scripts/release/README.md)
describes its layout and how it is built.

The manual steps follow.

### Build

You need a ROCm SDK with a clang that targets gfx1151, CMake 3.24 or newer, Ninja, and Rust (the toolchain version is
pinned in `rust/rust-toolchain.toml`; `rustup` picks it up).

```bash
# 1. The kernel library (libtalus_kernels.so) into build/
ROCM_SDK=/path/to/rocm ./build.sh

# 2. The Rust host (talus-cli, talus-server, talus-tui)
cd rust && cargo build --release && cd ..

# 3. Put the kernel library next to the binary (the host loads it from there)
cp build/libtalus_kernels.so rust/target/release/
```

`ROCM_SDK` is the root of the ROCm install (it has `bin/`, `lib/` and `lib/llvm/`). A TheRock wheel's
`_rocm_sdk_devel` directory works too. At run time, put `$ROCM_SDK/lib` on the library path (the server config's
`lib_dirs` does this for you). Instead of copying the library you can set `TALUS_KERNELS_LIB=/path/to/libtalus_kernels.so`.

The Rust workspace builds and tests without a GPU or ROCm: `cd rust && cargo test --workspace`.

The release was built and measured with ROCm SDK 10.2.0a20261014.

### Get a model

Download the GGUF files listed for the model in [docs/MODELS.md](docs/MODELS.md), with its drafter and draft
vocabulary. The server reads the tokenizer and the chat template straight from the model's GGUF (the run config's
`tokenizer` is the GGUF's path; no separate tokenizer files are needed). Qwen3.8-Flash-Next is the exception: its
`tokenizer` is the `tokenizer` folder inside its pack. `talus setup` writes these paths for you.

### Serve

`talus-server` reads a run config: the engine binary and its arguments and environment, the tokenizer (the model's
GGUF), stop tokens and default sampling. An example for Qwen3.8-27B:

```json
{
  "exe": "rust/target/release/talus-cli",
  "args": ["--model", "/models/Qwen3.8-27B-UD-IQ4_XS.gguf",
           "--mtp-gguf", "/models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "--mtp", "5",
           "--spec-draft-vocab", "/models/MTP/qwen-dv-65536.bin", "--ngram", "24",
           "--dflash-gguf", "/models/DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"],
  "tokenizer": "/models/Qwen3.8-27B-UD-IQ4_XS.gguf",
  "model_name": "qwen3.8-27b",
  "stop_tokens": ["<|im_end|>"],
  "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 40, "min_p": 0.05},
  "env": {},
  "lib_dirs": ["/path/to/rocm/lib", "/path/to/rocm/lib/llvm/lib", "/path/to/rocm/lib/rocm_sysdeps/lib"]
}
```

```bash
TALUS_API_KEY=<choose-a-key> rust/target/release/talus-server --config qwen27b.json --host 0.0.0.0 --port 8000
```

Set `TALUS_API_KEY` (or pass `--api-key`) whenever the server listens beyond this machine; clients send it as
`Authorization: Bearer <key>` or `x-api-key`.

With the prebuilt programs of a release tarball instead of a source build, use the tarball's `bin/talus-cli` as
`exe` and start `bin/talus-server`; the kernel library and the hipBLASLt tables are already next to them in `bin/`.
ROCm is not bundled: make `rocm-sdk` in the tarball's top folder a link to your ROCm SDK root, install ROCm in `/opt/rocm`, or list its
library folders in `lib_dirs` ([scripts/release/README.md](scripts/release/README.md)). The installer sets all of this up.

The other models' arguments are in [docs/MODELS.md](docs/MODELS.md), and each model's default sampling (the
`sampling` block) in [docs/MODELS.md, Default sampling](docs/MODELS.md#default-sampling). The engine loads the model
(this takes a while the first time) and the server answers on `http://<host>:8000/v1`.

### Use it

Any OpenAI-compatible client works:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TALUS_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Write a haiku about rocks."}]}'
```

## The server

The server speaks the OpenAI API (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`), the
Anthropic API (`/v1/messages`), and llama.cpp's `/tokenize`, `/detokenize` and `/completion`, so most coding agents
and chat UIs connect to it directly.

- **Streaming** with server-sent events.
- **Tool calls** in the model's own format (Qwen XML, GLM `<arg_key>`/`<arg_value>`, JSON), returned as typed
  OpenAI or Anthropic tool calls, for requests that declare tools. A request with no `tools` gets text. With
  `"tool_call_recovery": true` (the default) a declared tool's `<function=NAME>` without `<tool_call>` around it, or a
  call missing its `</tool_call>`, still counts as the call.
- **`max_tokens`** past the context is shortened to the room left. `reasoning_budget_tokens` is ignored, with a warning
  in the server log.
- **Thinking control:** `enable_thinking` and `reasoning_effort` through `chat_template_kwargs` or top-level fields,
  or Anthropic's `thinking` object.
- **Prompt cache:** a new request reuses the longest cached prefix of an earlier conversation and reads only the rest.
  The usage reports the reused tokens (`usage.prompt_tokens_details.cached_tokens`). An exact repeat or resend of a
  request gives the same answer as reading it cold (checked on all five models: Qwen3.8-27B, Qwen3.8-Flash-Next,
  GLM-5.3-Flash, MiMo-V2.6-Flash and Laguna-S-2.1; GLM-5.3-Flash from v0.1.7). A later turn of a conversation, or a raw
  prompt that extends an earlier one, can differ from the same text read cold in one go, because the engine splits the
  prompt at other points; both answers are valid, and the same history gives the same answer every time (see "exact" in
  [docs/METHODOLOGY.md](docs/METHODOLOGY.md#exact-and-opt-in-numerics)).
- **Auth:** `Authorization: Bearer <key>` or `x-api-key`, with the key from `TALUS_API_KEY` (or `STRATA_API_KEY`) or `--api-key`.
- **Request limits:** a request body may be at most 32 MiB (HTTP 413 past that). A request with more than about
  1M (2^20) JSON values gets HTTP 400, and so does a `response_format` schema that is too large or too complex to
  compile; the error says what to reduce. Requests over 1 MiB share a 128 MiB budget while they are being received,
  room for four full-size requests at once. A request that doesn't fit gets HTTP 503 with `Retry-After`, possibly
  before its upload has finished; send it again later. Requests of 1 MiB or less never wait for this budget. A stop
  string may be at most 64 KiB (HTTP 400). At most 64 connections are open at once; more get HTTP 503.
- **Engine restart:** if the engine crashes, the server restarts it, up to 3 times in 10 minutes
  ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#serving)). A streamed reply cut short by a crash ends with an error event.
- **Timings:** each response carries prompt and decode timings, as llama.cpp's server does.
- **Text only:** the models are served as text models. A request with an image, audio or file content part
  (`image_url`, `input_image`, `image`, `input_audio`, `file`, ...) gets HTTP 400 with the error code
  `unsupported_content` (on `/v1/messages`, an Anthropic `invalid_request_error`), instead of an answer that ignores
  the attachment.

`talus-server` starts the engine (`talus-cli`) as a subprocess and talks to it over stdin/stdout; see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#serving).

## How speculative decoding works here

A large model reads its weights from memory once per forward pass, and on Strix Halo that memory traffic sets the
decode speed. Speculative decoding drafts several tokens cheaply, then checks them all in one forward pass of the
large model. Every accepted draft is a token that did not need its own pass.

Talus keeps speculative decoding exact. For greedy requests, the output is byte-identical to plain greedy decoding:
the verify pass computes each row bit for bit as a single-token decode would. Sampled requests are rejection-sampled
against the drafter's distribution, so the output follows the model's own distribution. Each model uses the drafter
that works best for it:

- **Qwen3.8-27B:** two drafters, and a switch by context depth. The model's MTP head (a small extra layer trained to
  predict the next tokens) drafts a few tokens at a time, and its length adapts to the draft's confidence. DFlash2,
  a separate block-diffusion drafter, proposes a block of tokens per round from the model's hidden states; it sees a
  limited window, so it drafts at short context and the MTP head takes over deeper in a conversation. N-gram drafts
  catch repeated text, and a 64K draft vocabulary keeps the drafters' output layers small.
- **Qwen3.8-Flash-Next:** the MTP head, plus a suffix drafter that proposes continuations of text seen earlier in the
  conversation.
- **GLM-5.3-Flash:** the model's own NextN (multi-token prediction) layer, with a 64K draft vocabulary.
- **MiMo-V2.6-Flash:** a DFlash drafter model (Q8_0) with a 64K draft vocabulary.
- **Laguna-S-2.1:** a DFlash block-diffusion drafter (Q8_0) that proposes drafts from the model's hidden states, plus
  prompt lookup when the text repeats earlier text (code edits, quoting), with the draft length sized per step from
  the measured acceptance.

## Documentation

- [docs/MODELS.md](docs/MODELS.md): per model, the files, defaults, switches and memory.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the Rust crates, the kernel library's C ABI, the per-architecture
  tables, the engines.
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md): how we measure, the quality gates, how to reproduce.
- [docs/RESULTS.md](docs/RESULTS.md): the numbers of the current release.
- [docs/TUI.md](docs/TUI.md): `talus-tui`, a terminal dashboard to watch the server, switch models and tune them.
- `tools/check_claims.py`: the check that keeps unmeasured performance claims out of these docs.

## License and credits

Talus is MIT-licensed; see [LICENSE](LICENSE).

Talus is built on [Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors (MIT). The
kernel library, the server and much of the host's design come from Strata. Kernels and formats also derive from
[ggml and llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT). The full list, with licenses, is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Thanks to the model authors (Qwen, Z.ai, Xiaomi MiMo, poolside),
to Unsloth for the GGUF quants, and to z-lab for the DFlash drafters.
