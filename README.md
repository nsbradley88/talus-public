# Talus

Talus runs large open-weight language models on one AMD Strix Halo machine (Ryzen AI Max+ 395 with the Radeon 8060S,
gfx1151) with 128 GB of unified memory. It is a Rust host over a library of HIP kernels tuned for that GPU, and it
serves each model through an OpenAI-compatible HTTP API.

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

We develop and measure on one laptop, an HP ZBook Ultra G1a, whose firmware limits the APU to 70 W. Desktop and
mini-PC Strix Halo machines usually allow more power; we have not measured them.

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

## How fast is it?

See [docs/RESULTS.md](docs/RESULTS.md). Every number there comes from each model's recorded freeze run, measured end to end
through the HTTP server, with quality gates (long-context recall, an agentic coding suite, KL divergence against a
reference, greedy byte-identity, a soak test) that each model must pass. A model whose release run has not passed
shows "pending". The comparison rows are our own patched build of llama.cpp ("halo4"), not stock llama.cpp, and other
engines measured on the same machine under stated conditions. How everything is measured:
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Quick start

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
vocabulary. Then extract the tokenizer and chat template from the GGUF into a directory the server reads:

```bash
python3 -c "import sys; sys.path.insert(0, 'tools'); import strata_tokenizer as ST; \
            ST.extract('/models/Qwen3.8-27B-UD-IQ4_XS.gguf', '/models/qwen3.8-27b/tokenizer')"
```

### Serve

`talus-server` reads a run config: the engine binary and its arguments and environment, the tokenizer directory, stop
tokens and default sampling. An example for Qwen3.8-27B:

```json
{
  "exe": "rust/target/release/talus-cli",
  "args": ["--model", "/models/Qwen3.8-27B-UD-IQ4_XS.gguf",
           "--max-context", "65536",
           "--mtp-gguf", "/models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "--mtp", "5",
           "--spec-draft-vocab", "/models/MTP/qwen-dv-65536.bin", "--ngram", "24",
           "--dflash-gguf", "/models/DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"],
  "tokenizer": "/models/qwen3.8-27b/tokenizer",
  "model_name": "qwen3.8-27b",
  "stop_tokens": ["<|im_end|>"],
  "env": {},
  "lib_dirs": ["/path/to/rocm/lib", "/path/to/rocm/lib/llvm/lib"]
}
```

```bash
STRATA_API_KEY=<choose-a-key> rust/target/release/talus-server --config qwen27b.json --host 0.0.0.0 --port 8000
```

The other models' arguments are in [docs/MODELS.md](docs/MODELS.md). The engine loads the model (this takes a while
the first time) and the server answers on `http://<host>:8000/v1`.

### Use it

Any OpenAI-compatible client works:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $STRATA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Write a haiku about rocks."}]}'
```

## The server

The server speaks the OpenAI API (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`), the
Anthropic API (`/v1/messages`), and llama.cpp's `/tokenize`, `/detokenize` and `/completion`, so most coding agents
and chat UIs connect to it directly.

- **Streaming** with server-sent events.
- **Tool calls** in the model's own format (Qwen XML, GLM `<arg_key>`/`<arg_value>`, JSON), returned as typed
  OpenAI or Anthropic tool calls.
- **Thinking control:** `enable_thinking` and `reasoning_effort` through `chat_template_kwargs` or top-level fields,
  or Anthropic's `thinking` object.
- **Prompt cache:** a new request reuses the longest cached prefix of an earlier conversation and reads only the rest.
  The usage reports the reused tokens (`usage.prompt_tokens_details.cached_tokens`). A warm answer is identical to
  the same request read cold (see "exact" in [docs/METHODOLOGY.md](docs/METHODOLOGY.md#exact-and-opt-in-numerics)).
- **Auth:** `Authorization: Bearer <key>` or `x-api-key`, with the key from `STRATA_API_KEY` or `--api-key`.
- **Timings:** each response carries prompt and decode timings, as llama.cpp's server does.

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
