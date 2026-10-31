# Talus TUI

`talus-tui` is a terminal dashboard for `talus-server`. The server keeps running in the background; the TUI shows
how fast it is serving, lets you choose which model it serves, and lets you tune that model a little, starting
from our presets. Close the TUI at any time: serving goes on.

It is light on purpose. It asks the server for one small status reply every 2 seconds (1 to 10 s with
`--interval`). The server answers from counters it already keeps, without asking the engine anything, so the GPU
stays on your requests.

## Quick start

Build it with the rest of the Rust workspace (`cargo build --release` in `rust/`), then:

```
talus-tui                                   # talks to http://127.0.0.1:8000
talus-tui --server http://box:8000          # a server on another machine
talus-tui --once                            # print the status once (for scripts)
```

If the server needs an API key, the TUI reads it from `TALUS_API_KEY`, else `STRATA_API_KEY`, else
`--api-key-file PATH`, else `~/.config/talus/api-key`.

To try the TUI without a GPU, start the fake server, which makes up plausible numbers:

```
talus-tui fake-daemon --port 8000 &
talus-tui
```

## Let the TUI switch models and apply tuning

Watching works with any `talus-server`. Changing what it serves needs a control directory, so the server knows
the run config of each model you have installed:

1. Make a directory, for example `/etc/talus`, and put one run config per model in it, named by the model id:

   | Model | File |
   |---|---|
   | Qwen3.8-27B | `qwen3.8-27b.json` |
   | Qwen3.8-Flash-Next | `qwen3.8-flash-next.json` |
   | GLM-5.3-Flash | `glm-5.3-flash.json` |
   | MiMo-V2.6-Flash | `mimo-v2.6-flash.json` |
   | Laguna-S-2.1 | `laguna-s-2.1.json` |

   Start from `configs/*.example.json` (27B, MiMo, Laguna) or the shipped arguments in [MODELS.md](MODELS.md), with your file
   paths filled in. These files are "our tuned defaults": choosing the defaults in the TUI serves them unchanged.
   Models without a file show as "not installed".

2. Start the server with the directory, once with the model you want first:

   ```
   talus-server --control-dir /etc/talus --config /etc/talus/qwen3.8-27b.json
   ```

   After the first change from the TUI, the server writes `/etc/talus/active.json`. From then on
   `talus-server --control-dir /etc/talus` (no `--config`) starts with whatever you applied last, which is what a
   service unit should run.

When you apply a change, the server checks it against the list of options below, writes `active.json`, waits for
running requests to finish, stops the engine and starts again with the new settings. Serving pauses while that
happens: a few seconds for a tuning change, a few minutes when it loads a different model. The TUI shows
"restarting" and picks the server up again when it answers. If the new settings fail to start (a missing file, not
enough memory), the server goes back to the last settings that worked and the TUI tells you what went wrong.

Only the options listed here can be changed; the TUI never sends raw arguments or environment variables.
Changes are accepted only from the same computer, unless the server has an API key.

## Screens

Switch screens with `1`, `2`, `3` or Tab. Press `?` for the keys.

### 1 Status

The model being served and its state, the accuracy level, free memory (with the minimum to keep for this model),
how the server can be reached, and how much of the context the last request used. "Speed" averages the last 10
requests:

- **Answer**: tokens per second while writing the reply.
- **Prompt**: tokens per second while reading new prompt text (text reused from the prompt cache is not counted).
- **1st token**: time from the request to the first token of the reply.
- **Accepted**: the share of drafted tokens the model kept (speculative decoding).

```
 Talus   1 Status  2 Models  3 Tuning                                       ● http://127.0.0.1:8000
┌ Serving ─────────────────────────────────────────────────────────────────────────────────────────┐
│ Model      MiMo-V2.6-Flash   ● generating   up 0s                                                │
│ Accuracy   Balanced (our tuned defaults)                                                         │
│ Memory     16.1 GiB free of 125 GiB   (keep at least 11 GiB free)                                │
│ Server     127.0.0.1:8000 · This computer only · no API key                                      │
│ Context    ██                       5,128 of 192K tokens (last request)                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
┌ Speed, last 10 requests ───────────────────────────────┐┌ Answer tok/s per request ──────────────┐
│ Answer     xx.x tok/s  generating the reply            ││▄▄▆██▄▃▂▃▃▅▅▄▇▅▆                        │
│ Prompt     xxx tok/s  reading new prompt text          ││████████████████                        │
│ 1st token  x.xx s  wait before the reply starts        ││████████████████                        │
│ Accepted   xx %  drafted tokens kept                   ││████████████████                        │
└────────────────────────────────────────────────────────┘└────────────────────────────────────────┘
┌ Recent requests, newest first ───────────────────────────────────────────────────────────────────┐
│prompt   reused   answer  prompt tok/s  answer tok/s  1st token  accepted  end                    │
│4,623    2,754    505     xxx           xx.x          x.xx s     xx%       stop                   │
│10,087   0        538     xxx           xx.x          xx.xx s    xx%       stop                   │
│18,855   11,065   389     xxx           xx.x          x.xx s     xx%       stop                   │
│2,823    1,998    149     xxx           xx.x          x.xx s     xx%       stop                   │
│1,148    644      93      xxx           xx.x          x.xx s     xx%       stop                   │
│23,712   16,880   202     xxx           xx.x          x.xx s     xx%       stop                   │
│22,821   14,577   622     xxx           xx.x          x.xx s     xx%       stop                   │
│17,278   12,970   568     xxx           xx.x          x.xx s     xx%       length                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
 1-3 screens · g refresh · ? keys · q quit (the server keeps running)
```

These pictures come from the fake server, with its made-up speeds shown as x.

### 2 Models

The five models with their size, shipped context and whether they are installed on this server. Move with the
arrow keys and press Enter to open a model's tuning panel. Applying there switches the server to that model.

```
 Talus   1 Status  2 Models  3 Tuning                                       ● http://127.0.0.1:8000
┌ Choose a model ──────────────────────────────────────────────────────────────────────────────────┐
│  Model                Engine    Weights                            Context  On this server       │
│  Qwen3.8-27B          qwen35    about 17 GB with drafters          256K     ● serving            │
│  Qwen3.8-Flash-Next   qwen4exp  about 88 GB                        256K     installed            │
│  GLM-5.3-Flash        glm       about 110 GB                       64K      not installed        │
│▸ MiMo-V2.6-Flash      mimo      about 92 GB with the drafter       192K     installed            │
│  Laguna-S-2.1         laguna    about 68 GB                        256K     installed            │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
┌ MiMo-V2.6-Flash ─────────────────────────────────────────────────────────────────────────────────┐
│Mixture-of-experts model with 192K context and a DFlash drafter.                                  │
│                                                                                                  │
│Drafter    DFlash (Q8_0)                                                                          │
│Weights    about 92 GB with the drafter                                                           │
│Memory     keep at least 11 GiB free while serving                                                │
│License    MIT                                                                                    │
│Fast       Turns on the low-bit expert prompt kernel: faster prompts, output quality may drop     │
│slightly.                                                                                         │
│Preset     our tuned defaults                                                                     │
│                                                                                                  │
│Enter: open its tuning panel. Applying there switches the server to it.                           │
│                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
 ↑↓ move · Enter tune this model · ? keys · q quit
```

### 3 Tuning

The options of the selected model, in three groups. Every option shows its value and whether it is "our tuned
default" or "changed". The panel below explains the highlighted option: what it changes, its default, what it
trades off, whether the output stays the same, and the real switch behind it. Options that only matter with
speculative decoding are greyed out while it is off.

Keys: Left/Right change the value, `d` puts the highlighted option back to its default, `r` puts every option
back to our tuned defaults, `a` applies.

```
 Talus   1 Status  2 Models  3 Tuning                                       ● http://127.0.0.1:8000
 MiMo-V2.6-Flash  ● serving now
┌ Tuning: MiMo-V2.6-Flash ─────────────────────────────────────────────────────────────────────────┐
│ Accuracy                                                                                         │
│   Accuracy ↔ Speed              ‹ Reference · Balanced · [Fast] ›  changed                       │
│   Rounding-level speed-ups      ‹ On ›  our tuned default                                        │
│   Low-bit expert prompt kernel  ‹ On ›  changed  ! may lower quality                             │
│                                                                                                  │
│ Common                                                                                           │
│ ▸ Context length                ‹ 128K ›  changed                                                │
│   Speculative decoding          ‹ On ›  our tuned default                                        │
│   Speculative sampling          ‹ On ›  our tuned default                                        │
│                                                                                                  │
│ Server                                                                                           │
│   Network access                ‹ This computer only ›  current                                  │
│   API key                       ‹ None ›  current                                                │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
┌ Context length ──────────────────────────────────────────────────────────────────────────────────┐
│The longest conversation (prompt plus answer) the model can hold.                                 │
│                                                                                                  │
│Default    192K (our tuned default)                                                               │
│Trade-off  Shorter frees memory for other programs; longer chats are cut off or refused sooner.   │
│Output     same output                                                                            │
│Switch     --max-context N                                                                        │
│                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
 ↑↓ move · ←→ change · d default · r reset all · a apply · ? keys
```

Applying asks first and names every change. Options that may lower quality carry a warning:

```
 Talus   1 Status  2 Models  3 Tuning                                       ● http://127.0.0.1:8000
 MiMo-V2.6-Flash  ● serving now
┌ Tuning: MiMo-V2.6-Flash ─────────────────────────────────────────────────────────────────────────┐
│ Accuracy                                                                                         │
│ ▸ Accuracy ↔ Speed              ‹ Reference · Balanced · [Fast] ›  changed                       │
│   Rounding-level speed-ups      ‹ On ›  our tuned default                                        │
│   Low-bit expert prompt kernel  ‹ On ›  changed  ! may lower quality                             │
│                                                                                                  │
│ Common                                                                                           │
│   Context length                ‹ 192K ›  our tuned default                                      │
│   Speculative decoding          ‹ On ›  our tuned default                                        │
│   Specula┌ Apply to MiMo-V2.6-Flash? ─────────────────────────────────────────────────┐          │
│          │Low-bit expert prompt kernel: Off -> On                                     │          │
│ Server   │                                                                            │          │
│   Network│Serving pauses briefly while the engine restarts.                           │          │
│   API key│                                                                            │          │
└──────────│! Output quality may drop slightly: Low-bit expert prompt kernel did not    │──────────┘
┌ Accuracy │pass our quality gate.                                                      │──────────┐
│How exact │                                                                            │          │
│          │[y] apply   [n] cancel                                                      │          │
│Reference └────────────────────────────────────────────────────────────────────────────┘utput.    │
│Balanced   Our release defaults: the recommended level.                                           │
│Fast       Turns on the low-bit expert prompt kernel: faster prompts, output quality may drop     │
│slightly.                                                                                         │
│                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
 ↑↓ move · ←→ change · d default · r reset all · a apply · ? keys
```

Your choices per model are saved in `~/.config/talus/tui.json`, so a model you pick again comes back as you
left it.

## Accuracy ↔ Speed

One control with three levels. It sets the options in the "Accuracy" group and leaves the others alone.

- **Reference**: the exact reference path. Defaults that change numerics at rounding level are turned off, and so
  are all opt-ins. Slower.
- **Balanced**: our release defaults, the recommended level. Everything on in this level passed our quality gates
  (see [METHODOLOGY.md](METHODOLOGY.md)).
- **Fast**: Balanced plus the opt-ins that are faster but did not pass the quality gates. Output quality may drop
  slightly. Not every model has one; where none exists, Fast is the same as Balanced and the TUI says so.

| Model | Reference turns off | Fast turns on |
|---|---|---|
| Qwen3.8-27B | flash prompt attention (`TALUS_PF_FA=0`); 4-bit prompt math is already off by default | 4-bit prompt math in all 64 layers (`TALUS_Q27_W4I=fast`, needs the 8.0 GiB `.fast.w4i` file): faster prompts, but it failed the quality gate |
| Qwen3.8-Flash-Next | fp16 prompt experts, merged long-prompt projections, tuned decode layouts (`STRATA_PF_W16=0 STRATA_PF_HCDOWN=0 STRATA_MMVQ_KQ8=0 STRATA_MMVQ_W32=0`) | 16-bit HC streams (`TALUS_FN_HC_BF16=2`): faster prompts, but it failed the quality gate (the top token changes far more often) |
| GLM-5.3-Flash | fp16 prompt experts, 4096-row chunks, two-pass sparse prompt attention and fp16 prompt projections (`STRATA_PF_W16_MOE=0 GLM_PCH=2048 GLM_SPARSE_2P=0 GLM_N2=0 GLM_N3=0`) | nothing recommended |
| MiMo-V2.6-Flash | wave32 expert mat-vecs and WMMA prompt attention (`STRATA_IDS_W32=0 TALUS_MIMO_FA=0`) | low-bit expert prompt kernel (`MIMO_IQ1M_TPMOE=1`): faster prompts, but it failed the quality gate |
| Laguna-S-2.1 | sparse attention past 32K (no `--sparse`) and served-speed mode (`TALUS_LAGUNA_LSRV=0`) | batched verify (`TALUS_LAGUNA_VERIFY_GEMM=9`): much faster edits, but it failed the quality gate (outputs change; edits lost doc comments in tests) |

On Qwen3.8-27B, the 4-bit prompt math opt-in (`TALUS_Q27_W4I=auto`) uses `<model>.w4i` and Fast (`TALUS_Q27_W4I=fast`)
uses `<model>.fast.w4i` if those files are placed next to the GGUF ([MODELS.md](MODELS.md#qwen38-27b)); when a file is
absent, the TUI shows "weights not installed" on that option, the server refuses that choice, and Fast stays Balanced.
Without `<model>.w4i` (or by default, when `TALUS_Q27_W4I` is unset or `0`) the engine reads prompts on the exact int8
path.

Exact speed-ups, the ones that leave the output bit-identical, stay on at every level and are not shown.

## Options per model

"Same output" means the option changes speed or memory only.

| Option | What it changes | Models and switch |
|---|---|---|
| Context length | The longest conversation the model can hold. Shorter frees memory. Choices go up to the shipped length. | `--max-context N` (GLM: `--ctx N`) |
| Speculative decoding | Drafters guess a few tokens ahead and the model checks them in one pass. Off is much slower; same output. | 27B: the drafter flags; Flash-Next: without `--mtp`, `--suffix-draft 0`; GLM, MiMo: `--no-mtp`; Laguna: `--spec 0 TALUS_LAGUNA_DFLASH=0` |
| Drafter | Which drafter guesses: Auto (DFlash2 early in a chat, the MTP head deeper in), MTP only or DFlash2 only. Same output. | 27B: `--drafter auto\|mtp\|dflash` |
| Draft vocabulary | Which tokens the MTP drafter may propose: English and code (the default, faster for code) or the full vocabulary (best for non-English text such as Chinese, which drafts poorly from the subset). Same output. | Flash-Next: `--mtp-draft-vocab data/draft_vocab_en.bin` (removed for full) |
| Repeat-text drafts | Also drafts text that repeats earlier parts of the chat. Same output. | 27B: `--ngram 24`; Flash-Next: `--suffix-draft 3` |
| Speculative sampling | For sampled requests, drafts are accepted by probability, so more are kept. Same output distribution; a given seed gives different text. | Flash-Next: `STRATA_SPEC_PROB=0`; MiMo: `TALUS_MIMO_SSAMP=0`; Laguna: `TALUS_LAGUNA_SSAMP=0` (off) |
| Prompt cache reuse | Keeps recent conversations so a follow-up reads only the new part. Same output. | 27B, Flash-Next: `--prompt-cache 0`; GLM: `GLM_PROMPT_SNAPS=0` (off) |
| Int8 prompt weights | An int8 copy of the weights for faster prompt reading, about 26 GiB more memory. Same output. | 27B: `TALUS_PF_I8=0` (off) |
| Network access | This computer only (127.0.0.1) or the local network (0.0.0.0, needs an API key). | server `host` |
| API key | Keep the current key or generate a new one. A new key is shown once and saved to `~/.config/talus/api-key` (readable by you only). A key set with `TALUS_API_KEY` or `STRATA_API_KEY` in the service environment wins and can't be changed here. | server `api_key` |

The Accuracy options are in the table above.

## HTTP endpoints

The TUI uses three endpoints, which scripts can use too (with the API key when the server has one):

- `GET /v1/talus/status`: model, state, context, free memory, totals and the last 20 requests (prompt and answer
  tokens, prompt and answer time, time to first token, drafts offered and accepted).
- `GET /v1/talus/presets`: the five models, which ones are installed, and the current selection.
- `POST /v1/talus/apply`: `{"model": "<id>", "choices": {"<option>": "<value>"}, "bind": "local"|"lan",
  "api_key": "<new key>", "dry_run": true}`. Only `model` is required; options left out take our tuned default.
  With `dry_run` the server only lists the changes. Otherwise it answers 202 and restarts. Option ids and values
  are the ones in `rust/crates/talus-presets/src/catalog.rs`.
