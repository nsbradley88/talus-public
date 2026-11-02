# Models

Talus serves five models. This page lists, for each one, the files we run, the drafter, the arguments and
environment we ship, the switches that turn defaults off or opt into other numerics, and the memory it needs.

How to read the switch tables:

- **Kill switch:** the feature is on by default; setting the variable as shown turns it off.
- **Opt-in:** the feature is off by default. "Changes numerics" means its logits differ from the default path, so
  it has to pass the quality gates before it can become a default ([METHODOLOGY.md](METHODOLOGY.md#exact-and-opt-in-numerics)).
- "Exact" (bitwise) switches change speed only: the output is identical either way.
- A variable you export always wins over an engine's built-in default.
- Every `STRATA_*` variable (the names Talus inherited from Strata) also works under its `TALUS_` name:
  `TALUS_PF_FUSED=1` means `STRATA_PF_FUSED=1`. Existing configs keep working; when both names are set, the
  `STRATA_*` value wins.

`talus-cli` picks the engine from the GGUF's `general.architecture`. In the server's run config, put the arguments in
`"args"` and the variables in its `"env"` (or export them before starting the server). Speed and quality
figures for every model are on [RESULTS.md](RESULTS.md).

All five models run with a minimum of free memory (`MemAvailable`) in mind: the engine's memory accountant refuses any allocation
that would leave less than 8 GiB free (GLM-5.3-Flash runs closest to it, at about 9 GiB). While serving, a shared memory
guard frees caches it can rebuild when `MemAvailable` comes within 512 MiB of that floor, and if memory stays below the
floor only the current request fails (`insufficient_memory`, HTTP 503) and the server keeps serving (GLM-5.3-Flash and
Qwen3.8-27B free caches first; MiMo-V2.6-Flash and Laguna-S-2.1 refuse the request; Flash-Next keeps its own admission
control). One model runs per machine.

## System setup

On a default install, the GPU can use only part of the 128 GB, and GLM-5.3-Flash and MiMo-V2.6-Flash do not fit.
Our measurements use these kernel parameters (Ubuntu: a file in `/etc/default/grub.d/`, then `sudo update-grub` and a
reboot; `talus setup` writes it for you after asking):

```
ttm.pages_limit=31457280 ttm.page_pool_size=31457280   # GPU memory pool 120 GiB
kho=off                                                # free the ~9 GiB kexec-handover reserve (Ubuntu 7.0 kernels)
amdgpu.lockup_timeout=10000,60000,10000,10000          # let long prompt passes run (60 s compute timeout)
```

We also run headless (`sudo systemctl set-default multi-user.target`). This is approximate (release-run free memory,
not a separate measurement): with a desktop session running, Qwen3.8-27B, Qwen3.8-Flash-Next and Laguna-S-2.1 still
have tens of GiB to spare, MiMo-V2.6-Flash is tight, and GLM-5.3-Flash needs headless. These are boot settings: check
them against your distribution's documentation before you apply them.

Two more host settings were on for every measurement:

- **ACPI platform profile `performance`** (laptops and some mini-PCs). Most Strix Halo machines boot in `balanced`,
  which holds the chip's sustained power back: GLM-5.3-Flash decoded at 11.2 tok/s with balanced
  instead of about 24 tok/s on our laptop. Check it with
  `cat /sys/firmware/acpi/platform_profile` (the choices are in `platform_profile_choices` next to it).
  `talus setup` offers to set it (default yes; `--no-power-profile` skips) and keeps it across reboots: with
  `powerprofilesctl set performance` when power-profiles-daemon runs, otherwise with one systemd-tmpfiles line in
  `/etc/tmpfiles.d/90-talus-platform-profile.conf`. `talus doctor` warns when it is not `performance`. It is not a
  boot setting and needs no reboot. Performance draws more power and runs the fans harder (on a laptop: warmer,
  louder, shorter on battery). Machines without the file (our desktop, and many others) set the power mode in the
  BIOS: pick the highest sustained mode.
- **`amd_iommu=off`** on the kernel command line, next to the parameters above. We have not measured what it is
  worth, Talus runs with the IOMMU on, and `talus setup` does not set it. With it, the NPU is unavailable.

## Default sampling

Each run config carries a `sampling` block: the values a request gets for the sampling fields it leaves out. A
request's own `temperature`, `top_p`, `top_k` and `min_p` always win. The blocks are the server settings every
published figure of the model was measured under (the `chat` cell itself sends temperature 1.0 / top-p 0.95; the
other fields come from the block):

| Model | temperature | top_p | top_k | min_p |
|---|---|---|---|---|
| Qwen3.8-27B | 0.8 | 0.95 | 40 | 0.05 |
| Qwen3.8-Flash-Next | 1.0 | 0.95 | 20 | 0 |
| GLM-5.3-Flash | 0.8 | 0.95 | 40 | 0.05 |
| MiMo-V2.6-Flash | 1.0 | 1.0 | 20 | 0 |
| Laguna-S-2.1 | 1.0 | 1.0 | 20 | 0 |

Qwen3.8-27B's block changed in v0.1.2: v0.1.1 shipped top_k 20 / min_p 0, which differed from what its figures were
measured with. A run config without a `sampling` block gets talus-server's built-in defaults for the missing fields:
temperature 0.7, top_p 0.95, top_k 20, min_p 0. The engines keep at most 64 candidates, so a `top_k` of 0 (off) or above
64 means 64.

Our own files (the GLM, MiMo and Laguna quants and the 64K draft vocabularies) are published on Hugging Face with
model cards that give the recipe, the tensor types and the licenses.

## Qwen3.8-27B

Dense model with gated delta net and attention layers. Engine `qwen35`.

| | |
|---|---|
| Weights | `Qwen3.8-27B-UD-IQ4_XS.gguf` from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (about 14 GB, the default). `UD-Q4_K_XL` (about 17 GB) from the same repository is a supported quality option: see **Choosing the quant** below. |
| MTP head | `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` from the same repository (about 1.3 GB) |
| DFlash2 drafter | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` from [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF), z-lab's block-diffusion drafter (about 1.1 GB) |
| Draft vocabulary | `qwen-dv-65536.bin` ([j2ipper/draft-vocab](https://huggingface.co/j2ipper/draft-vocab)): 64K token ids in llama.cpp's int32 format |
| 4-bit prompt weights | `Qwen3.8-27B-UD-IQ4_XS.w4i` format v3 ([j2ipper/Qwen3.8-27B-W4A4-prompt-v3](https://huggingface.co/j2ipper/Qwen3.8-27B-W4A4-prompt-v3), about 2.06 GiB, sha256 `2a4f69c3918218d3f1b6fbdf3a3278c65077ee042c4cee51e5b99e455862935d`), placed next to the GGUF; `talus setup` downloads it by default (`--advanced` can skip it). Used by default when present (see **4-bit prompt math** below); `TALUS_Q27_W4I=0` (or the TUI's Reference level) turns it off, and without the file prompts run on the exact int8 path. The v2 file of [j2ipper/Qwen3.8-27B-W4A4-prompt](https://huggingface.co/j2ipper/Qwen3.8-27B-W4A4-prompt) is for v0.1.1 and older (opt-in there); v0.1.2 ignores a v2 file unless `TALUS_Q27_W4I=auto`. `Qwen3.8-27B-UD-IQ4_XS.fast.w4i` (about 8.0 GiB, v2 repository) is only for the opt-in fast profile. Built for `UD-IQ4_XS` only. |
| Context / KV | Sized to the machine when `--max-context` is not given: the largest of 256K (the model's native context, no RoPE scaling), 128K and 64K that fits in memory beside the weights and the int8 prompt weights (see **Context by machine** below). f16 KV: 68 KiB per token (K and a transposed V copy for the 16 attention layers plus the MTP layer; 17 GiB at 256K) |
| Stop token | `<|im_end|>` |
| License | Apache-2.0 (model, MTP head, DFlash2 drafter) |

Shipped arguments:

```
--model Qwen3.8-27B-UD-IQ4_XS.gguf
--mtp-gguf mtp-Qwen3.8-27B-Q4_0.gguf --mtp 5
--spec-draft-vocab qwen-dv-65536.bin --ngram 24
--dflash-gguf Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

No environment variables are needed. 4-bit prompt math is on when the v3 prompt-weight file is installed
(`TALUS_Q27_W4I=0` turns it off, see **4-bit prompt math** below). Without `--max-context` (or with `--max-context auto`) the engine picks the context by memory and logs
the choice and the reason in one line; an explicit `--max-context` is always honoured. Without the drafter flags, no
speculative decoding runs.

**Drafting.** With both the MTP head and DFlash2 loaded, the drafter is chosen by context depth (`--drafter auto`,
the default when both are given). DFlash2 drafts while the window starts below 16K tokens (32K for sampled
requests), and the MTP head drafts above that. DFlash2 sees a window of about 2K tokens, which is why it is not used
deep in a conversation. The MTP draft length adapts to the draft's confidence (up to `--mtp N` drafts). `--ngram 24`
adds n-gram drafts from repeated text. Long copies (see **Long copies** under Laguna) are opt-in here
(`TALUS_Q27_COPY=1`): the n-gram lookup already copies long repeats, and they cost code speed. Greedy output equals
plain greedy, and sampled requests are rejection-sampled at the request's temperature. `TALUS_Q27_QTEMP=0.5` (opt-in)
sharpens the drafters' proposals. The target distribution stays exact and chat can be faster, but with a fixed seed
the sampled text differs; we keep it off by default.

**Defaults that change numerics**

| Default | Kill switch | Notes |
|---|---|---|
| Prompt flash attention | `TALUS_PF_FA=0` | Rounds differently from the legacy prompt attention (checked with the KLD gate before it became the default). `STRATA_FATTN=0` turns off every flash-attention prompt path. |

**Exact defaults (kill switches)**

| Feature | Kill switch |
|---|---|
| Int8 prompt GEMM on weights expanded at load (needs extra GPU memory; see below) | `TALUS_PF_I8=0` (back to MMQ) |
| Int8 expansion for K-quant weights | `TALUS_PF_I8_KQ=0` |
| SwiGLU written straight into the q8_1 activation blocks | `TALUS_PF_FUSE=0` |
| Dense MMQ prompt path | `STRATA_DENSE_MMQ=0` |
| K-quant MMQ | `STRATA_MMQ_KQ=0` |
| Repacked Q8_0 decode weights | `STRATA_Q8_PACKED=0` |
| Wave32 decode matvec | `STRATA_MMVQ_W32=0` |
| 16-row verify windows | `STRATA_Q36_VERIFY16=0` or `--no-v16` (8 rows) |
| Wide verify tiles: the 12-16-row windows' IQ4_XS / IQ3_S / IQ3_XXS / IQ2_S / IQ2_XS mat-vecs and the 8-16-row Q4_K / Q5_K ones on the matrix cores (16 rows x 16 columns a tile), every row bitwise the 1-row result; gfx1151 (served edit mix at 256K: 109.4 tok/s with the tiles, 87.9 tok/s without) | `TALUS_MMVQ_WIDE=0` |
| Narrow verify cells (row-per-lane and split tile decode for 5-16-row windows, per weight type and width, measured on gfx1151) and the wave32 mat-vecs' tree-sum epilogue at 2-8 rows, every row bitwise the 1-row result (chat 4.7 %, code 5.4 %, edit 3.2 % faster than the wide tiles alone) | `TALUS_MMVQ_W32_TS=0` (epilogue); `TALUS_MMVQ_WIDE_TAB="13:5-7:0;13:8-10:2;12:6-7:0;12:8-10:2;22:6-11:0;22:12-15:2;21:7:0;21:9:0;21:10-16:2;23:9-11:0;18:9-11:0;18:12-16:2"` (the earlier cell table) |
| Tile-plane copies of those weights (same bytes, regrouped so a wave reads whole cache lines; 13 GB, placed only when they fit, see **Memory** below) | `TALUS_MMVQ_WIDE_PACK=0` (`=1` forces them) |
| DPP softmax reductions in decode/verify attention (same pairing order as the shuffle version) | `TALUS_ATTN_RED=0` |
| GDN commit fused into the next forward pass, and draft/verify rounds on the device | `STRATA_FUSED=0`; `STRATA_NO_DEV_ROUNDS=1` keeps the fusion but runs rounds on the host |
| Depth switch between drafters | `STRATA_DRAFTER=mtp` or `--drafter mtp` (MTP only); `--drafter dflash` (DFlash2 only) |
| Adaptive draft length | `STRATA_ADRAFT=0` (fixed length) |
| One-tile row groups in decode attention | `TALUS_ATTN_RT1G=0` |
| Split decode attention | `STRATA_ATTN_SPLIT=0` |
| HIP graphs | `--no-graph` |

**4-bit prompt math (default with the v3 file).** It is used when the installed `.w4i` file is format v3
(header `W4I3`). Measured against the exact int8 path on the laptop (tier 1, two control runs): prompts
6.5-7.2 % faster at 2K and 6.5-6.7 % at 32K; decode after a 2K prompt
+1.6 to +1.7 %, after 32K -0.5 %; draft acceptance unchanged. Quality against the
BF16 weights (teacher-forced, 12 wiki and 6 code chunks): the added KL over the exact path is
+0.00018 on wiki and +0.00099 on code, below the `Q8_0` floor of
0.00084 / 0.00148; needle 60K 15/15, agentic smoke
test 5/5. The v3 file comes from a new recipe for the same layers (SmoothQuant, best-of-4 integer
group scales, GPTQ with 256-column groups, per-row bias correction) and replaces v0.1.1's v2 file, which lowered draft
acceptance after the prompt and therefore stayed opt-in. When on, the prompt (prefill) GEMMs of the feed-forward in layers 48-63 run on the `iu4`
matrix instruction, twice the int8 rate. Activations are rotated per 256 values (seeded Hadamard) and quantized to int4
per 32 values; the 32-value scales are small integers times a per-token scale, so the kernel rescales with one integer
multiply-add. The int4 weights are a separate GPTQ quantization of the GGUF's weights in the same rotated basis (`.w4i`
file; build it with `tools/w4i/build_sidecar.sh`, recipe in `tools/w4i/README.md`). Decode, verification and drafting
never use it. Only the last 16 layers: earlier layers cost several times more accuracy per layer (layer sweep in
[METHODOLOGY.md](METHODOLOGY.md)). The file records the GGUF's name and size; a different GGUF, a missing file, a v2 file (unless
`TALUS_Q27_W4I=auto`) or `TALUS_Q27_W4I=0` keeps the exact int8 prompt path (a missing file logs one line). `TALUS_Q27_W4I=<path>` loads a given
file, `TALUS_Q27_W4I_PLANES` picks the layers (`ffn:48-63` = the default).

**Opt-ins:** `TALUS_Q27_W4I=fast` (needs `Qwen3.8-27B-UD-IQ4_XS.fast.w4i`): 4-bit prompt math in the feed-forward of all
64 layers, about a third faster on prompts at 2K and 32K. **It failed our quality gate**: divergence to the CPU
reference on 4K prompts exceeds the gate limit (KL vs llama.cpp CPU reference (UD-IQ4_XS) <= 0.00291 (short prompts) and 4K-prompt KL ratio vs exact <= 1.35). The TUI's Fast level selects it, with a
warning. `STRATA_ATTN_W8=1` / `--attn-w8` (8-row attention
windows). Draft-only variants that cannot change the output (verification is exact): `--mtp-f16`,
`STRATA_DRAFT_NUCLEUS=1`.

Do not set `STRATA_BF16_KV`: it only changes the KV layout that the prompt cache expects.

**Prompt cache.** `--prompt-cache N` state snapshots (default 8) and `--kv-stash-mib` of stashed KV (default 8192).
Warm answers equal cold answers; `--cache-live 1` keeps live decode state instead, with fewer tokens re-read per turn
but warm answers that may differ from cold. The cache is also kept on disk so it survives restarts and model switches
(on by default; see [Prompt cache on disk](#prompt-cache-on-disk)).

**Memory.** Weights and drafters are about 17 GB. The int8 prompt GEMM expands the trunk weights at load time and
uses about 26 GiB more GPU memory (`TALUS_PF_I8=0` frees it). The KV cache is sized by the context: 68 KiB per token,
about 17 GiB at 256K, 8.5 GiB at 128K and 4.3 GiB at 64K; output is identical at any length the shorter context holds.
The legacy prompt attention (`TALUS_PF_FA=0` or `STRATA_FATTN=0`) keeps a second V copy, 34 KiB more per token
(8.5 GiB at 256K). The 4-bit prompt weights add about 2.0 GiB (8.0 GiB with the fast profile). The wide verify
tiles' tile-plane copies add 13 GB (12.0 GiB) when they fit: they come last (context > int8 prompt weights > copies),
take only what is left above the 8 GiB floor and a 3 GiB reserve, and never shrink the automatic context. Without them
the tiles read the GGUF rows, somewhat slower. One log line says which and why (`wide verify tiles on with tile-plane
copies of 387 matrices, 12.0 GiB (auto: they fit with the reserve; ...)`). Lowest `MemAvailable`
in the v0.1.2 release run on our desktop: 42 GiB at the shipped 256K context,
56 GiB at 64K (`--max-context 65536`, measured on the first release candidate).

**Context by machine.** Without `--max-context` the engine takes the largest of 256K, 128K and 64K whose KV cache fits
beside the weights, the int8 prompt weights and a 3 GiB reserve, keeping 8 GiB of `MemAvailable` free, and logs one
line such as `context 65536 (auto, no --max-context): 37.1 GiB free above the 8 GiB floor; needs 33.4 GiB = ...`.

| Machine | Automatic context | Notes |
|---|---|---|
| 128 GB Strix Halo | 256K | int8 prompt weights and the wide tiles' copies on. Lowest `MemAvailable` while serving (laptop, v0.1.0): about 46 GiB at 256K (59 GiB without the copies), 66 GiB at 64K; v0.1.2 on the desktop: see **Memory** above. |
| 64 GB Strix Halo | 64K (checked with a 62 GiB budget, `TALUS_MEM_LIMIT_MIB=63488`, on a 128 GB machine) | int8 prompt weights on; the wide tiles run without their copies. `--max-context 131072` still keeps the int8 prompt weights, by about 0.3 GiB when nothing else runs; a desktop session or other programs push them out. `--max-context 262144` does not leave room for them. |

An explicit `--max-context` is always honoured. When the int8 prompt weights do not fit beside its KV cache, the engine
prints a `WARN` and runs prompts on the slower MMQ path; to get the int8 path back, pick a smaller `--max-context`.
`TALUS_PF_I8=0` skips the expansion on purpose. `TALUS_MEM_LIMIT_MIB=N` caps the engine's memory budget at N MiB of
total use, to check a smaller machine.

Long context costs time, not accuracy: the needle test (five codenames spread through the prompt) passed
15/15 at 120K, 15/15 at 170K and 15/15 at 250K.
Decode slows with depth because every token reads the whole KV cache: about 23 tok/s at 128K,
16.5 tok/s at 170K and 18-20 tok/s at 250K (greedy, with the drafters, on 128 GB; different
prompts; the 250K figure is a follow-up turn on a cached prompt). A cold 250K prompt takes about
19 minutes; later turns reuse the cached prefix and start in seconds.

**Choosing the quant.** `UD-IQ4_XS` is the default: it is the smaller file and the faster one in Talus (compare the
`Talus` and `Talus (UD-Q4_K_XL)` rows in [RESULTS.md](RESULTS.md#qwen38-27b)). `UD-Q4_K_XL` stays closer to the
original weights, most of all on code. Divergence from the BF16 weights, measured with llama.cpp's KL-divergence tool
on the GPU (2,048-token windows; text = the wikitext-2 test set, first 60 windows; code = Python standard-library
source, 26 windows; the `Q8_0` row is the measurement floor):

| Quant | text: mean KL | text: same top-1 token | text: PPL ratio | code: mean KL | code: same top-1 token | code: PPL ratio |
|---|---|---|---|---|---|---|
| `UD-IQ4_XS` (default, 13.3 GiB) | 0.0181 | 94.4 % | 1.003 | 0.0819 | 93.5 % | 1.038 |
| `UD-Q4_K_XL` (16.3 GiB) | 0.0113 | 95.0 % | 1.003 | 0.0373 | 95.5 % | 1.017 |
| `Q8_0` (floor) | 0.0012 | 98.2 % | 1.000 | 0.0018 | 98.9 % | 1.001 |

On prose the two 4-bit files are close. On code `UD-Q4_K_XL` has about half the divergence of `UD-IQ4_XS`; the median
token is nearly unchanged in both, and the difference sits in a tail of uncertain tokens. To run `UD-Q4_K_XL`, download
`Qwen3.8-27B-UD-Q4_K_XL.gguf` and pass it to `--model` (and as the server's `tokenizer`); everything else stays the same,
except the opt-in 4-bit prompt weights (`.w4i`), which exist for `UD-IQ4_XS` only. **Opt-in reference path:** from
v0.1.2 Talus also runs the BF16 (or F16) weights: pass the BF16 GGUF (the first file of a split GGUF) to `--model`. It is
a reference for quality comparisons, not for daily use: its output matches llama.cpp's BF16 (KL 0.0003), at
about 490 tok/s prompt at 2K and about 4 tok/s decode on the laptop.

## Qwen3.8-Flash-Next

Mixture-of-experts model, Strata's original model. Engine `qwen4exp`: the Rust host drives Strata's C++ data plane in
the kernel library (a pure-Rust port is in progress). All of Strata's flags and `STRATA_*` variables apply.

| | |
|---|---|
| Weights | Unsloth `Qwen3.8-Flash-Next-UD-IQ4_XS` GGUF, three shards (about 88 GB), from `unsloth/Qwen3.8-Flash-Next-GGUF` |
| Pack | A Strata pack built from the GGUF with `tools/iq_pack.py --gguf <shard 1> --out <pack> --compat-bf16`; it also holds the tokenizer: the server's run config points `tokenizer` at `<pack>/tokenizer` (`talus setup` writes it) |
| MTP | The MTP runtime directory built with `tools/mtp_fetch.py`, `tools/mtp_pack.py` and `tools/mtp_rt.py` (each script's header shows its usage) |
| Expert profile | `data/expert-profile.bin` (shipped: `<prefix>/data/` of the release tarball / the installer, also in this repository) |
| Context / KV | 256K shipped (`--max-context 262144`), int8 KV |
| License | Qwen Community License 1.0. Note its condition 2: a business that offers model inference as a service or an AI coding / office assistant needs a separate license from Qwen for commercial use. |

Shipped arguments (`<prefix>` is the install prefix: the release tarball's top folder, `~/.local/share/talus` for
the installer; `bin/`, `data/` and `tools/` are under it, and a source checkout has the same `data/` and `tools/`):

```
--pack <pack> --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
--expert-profile <prefix>/data/expert-profile.bin --expert-cache 24576 --resident-budget-gib 55
--prefill auto:16384 --spec 5 --spec-min-p 0.5 --mtp <mtp runtime dir> --mtp-window 8192
--mtp-draft-vocab <prefix>/data/draft_vocab_en.bin --max-context 262144 --kv int8
```

Shipped environment (these turn on prompt paths that are off in Strata's code defaults):

```
STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_HC_UPMIX=1 STRATA_PA_FAST=1 STRATA_HIP_WMMA=1
STRATA_SELECT_WMMA=1 STRATA_HC_Q8=1 STRATA_PF_SWITCH_MIN_T=0 STRATA_PREFILL_STREAM_MIN=128
```

**The hipBLASLt prompt table** ships in the release tarball and the installer's `bin/hip/`
(`gfx1151-hipblaslt-100500.txt` on the ROCm 10.2 pin) and is found by the kernel library itself, with no variable: it
loads `<arch>-hipblaslt-<running hipBLASLt version>.txt` from the first of `<dir of libtalus_kernels.so>/hip/`,
`../share/talus/hip/` and `../tools/hip/` (a build in the source tree), and logs `prefill gemm: hipBLASLt tuning
enabled (... found next to libtalus_kernels.so)`. `STRATA_HIPBLASLT_TUNING` overrides it: a table file (when that file
is for another hipBLASLt version, the file for the running version in the same directory is loaded), a directory of
tables, or empty / `off` for no table. Without a table the engine log says `talus: WARN: no hipBLASLt tuning table ...`
and prompts run untuned (hipBLASEx: about 22 % slower at 2K tokens, and different numerics);
`talus doctor` shows which table applies. v0.1.1's tarball did not contain the table (see its known issues).

**Drafting.** The MTP head with `--spec 5` and a window of 8192 tokens, plus a suffix drafter (on by default,
`--suffix-draft 3`; `--suffix-draft 0` for MTP only) that proposes continuations of earlier text. A long copy (see
**Long copies** under Laguna) fills a whole 8-token verify window without the MTP head's drafts (the head then only
catches up), or extends the head's drafts; the verify window is 8 tokens (`--spec 8 --mtp-max-t 5`). A copy whose estimated per-token acceptance is at least
`TALUS_COPY_WIDE_Q` (default 0.95) verifies up to `TALUS_COPY_VERIFY` (default 16) tokens in one window instead; every
row equals one-row decoding bit for bit (edit mix 22.7 % faster on our desktop). `TALUS_COPY_VERIFY=8`
keeps every window at 8 tokens. The MTP head drafts
from the English-and-code token subset `data/draft_vocab_en.bin` (40,525 tokens, an 81 MB head instead of the runtime
directory's 106,299-token, 213 MB one): each draft step reads fewer bytes. Verification is unchanged, so greedy output
stays exact and sampled requests keep their exact distribution; only the drafts differ. For mostly non-English (CJK)
use, where tokens outside the subset cannot be drafted, remove the flag (the full head).

**Defaults that change numerics** (on gfx1151, or with the shipped environment)

| Default | Kill switch | Notes |
|---|---|---|
| HC down and inject projections as one WMMA GEMM, for prompts of 16K tokens or more | `STRATA_PF_HCDOWN=0` (`=1`: on for every prompt) | Rounding-level; shorter prompts stay exact. Threshold `STRATA_PF_HCDOWN_MIN` (16384). Disabled by default on `hipBLASLt 100500` unless `STRATA_HCD_EXACT=1` is set. |
| fp16 prompt experts | `STRATA_PF_W16=0` | Runs inside the fused prompt path, so it is on only with the shipped `STRATA_PF_FUSED=1`. |
| Q4_K / Q5_K decode lane layout | `STRATA_MMVQ_KQ8=0` | |
| Wave32 decode matvec | `STRATA_MMVQ_W32=0` | |

**Opt-in (changes numerics):** `STRATA_PREFILL_EQUAL=1` reads a prompt longer than one prompt chunk (16K by
default) in equal chunks instead of full chunks and a short last one (from upstream Strata PR #693; idea raised by
Avalon). It makes prompts 2.2-4.2 % faster at 19K-33K tokens (laptop), but it failed the v0.1.2
release KLD gate on prompts over 16K (first-token KL 0.0311 against the
0.0275 limit; 0.0242 with it off), so it is opt-in. Prompts within one chunk
are unchanged either way.

**Opt-in (changes numerics):** `TALUS_FN_HC_BF16=2` stores the prompt-path residual (hyper-connection) streams in fp16
(`=1`: bf16). It speeds up prompts and passes needle and the agentic smoke test, but its first-token KL is over the release gate, so
it is off by default.

**Exact defaults.** `STRATA_GFX1151_DEFAULTS=0` turns off the whole gfx1151 table of bitwise speed switches (and the
HC-down default above). The table is in `src/core/arch_defaults.cpp`; each entry can also be set on its own.
`TALUS_COPY=0` turns off long prompt copies (in every engine). The turn-checkpoint GDN buffer pool omits unused `conv_state` staging to save host/device memory.

**Memory.** Weights about 88 GB (`UD-IQ4_XS`), read through the expert cache (`--expert-cache` in MiB) and a resident budget
(`--resident-budget-gib`). Unsloth's `UD-Q4_K_XL` GGUF also runs when packed with `tools/iq_pack.py`: its larger `K`-quant experts use more memory and increase expert-cache traffic on long prompts and long-context decode, while reading slightly faster on short cold prompts. Lowest `MemAvailable` in the v0.1.2 release run on our desktop: 51 GiB.

## GLM-5.3-Flash

Mixture-of-experts model with MLA (latent) attention and KDA layers. Engine `glm`.

| | |
|---|---|
| Weights | `GLM-5.3-Flash-a4e3s` ([j2ipper/GLM-5.3-Flash-a4e3s](https://huggingface.co/j2ipper/GLM-5.3-Flash-a4e3s)), four shards (about 110 GB). Our requant of Unsloth's UD-IQ3_XXS: attention, KDA, shared experts and dense FFN at Q4_K; routed experts at IQ2_S / IQ3_S; output Q5_K. |
| Drafter | Built in: the model's NextN (MTP) layer. No extra file. |
| Draft vocabulary | `glm-dv-65536.bin` ([j2ipper/draft-vocab](https://huggingface.co/j2ipper/draft-vocab), 64K token ids) |
| Context / KV | 64K (`--ctx 65536`, the default), f16 MLA latent KV |
| Stop tokens | `<|user|>`, `<|observation|>`, `<|endoftext|>` |
| License | MIT |

Shipped arguments:

```
--model GLM-5.3-Flash-a4e3s-00001-of-00004.gguf --ctx 65536 --draft-vocab glm-dv-65536.bin
```

No environment variables are needed. The GLM engine ignores flags it does not know.

**Drafting.** The NextN layer drafts two tokens per round and the model verifies three positions. `--no-mtp` turns
drafting off. The draft vocabulary only affects drafting.

**Memory guard.** The engine refuses to start unless `MemAvailable` is at least `--minstart` GiB (default 112). While
running, when `MemAvailable` comes within 512 MiB of the `--minavail` floor (default 8 GiB) it frees memory it can
rebuild (decode graphs, prompt work buffers, then the prompt-cache snapshot; a turn that loses its snapshot reads its
prompt cold, and its greedy text can differ from a warm resume, as on any cache miss; `TALUS_MEM_SHED_INEXACT=0`
fails the request instead). If memory stays below the floor, only that request fails (`insufficient_memory`, HTTP
503) and the server keeps running; it exits only 2 GiB below the floor. `TALUS_MEM_FLOOR_GIB` (2-64) moves the 8 GiB
floor for the load check and the run-time guard together, for a machine with a little less free memory than ours;
the engine logs the floor it uses. Start it on an otherwise idle machine.

**Defaults that change numerics**

| Default | Kill switch | Notes |
|---|---|---|
| fp16 routed experts in prompt chunks | `STRATA_PF_W16_MOE=0` | The engine sets `STRATA_PF_W16_MOE=1` when it is unset. |
| 4096-row prompt chunks | `GLM_PCH=2048` (the earlier size) | `GLM_PCH` takes 16 to 4096 in multiples of 8. |
| two-pass WMMA sparse attention in prompt windows | `GLM_SPARSE_2P=0` | Exact row max, softmax weights in fp16; windows of 9+ tokens (`GLM_SPARSE_2P_MIN`); 42 MiB scratch. Decode and verify windows keep the previous kernel. |
| fp16 WMMA for the MLA `wk_b` / `wv_b` and KDA `f_b` / `g_b` projections in prompt windows | `GLM_N2=0` | `Q8_0` codes exact in fp16, activations split into two fp16 parts, fp32 accumulation: only the summation order changes. Windows of 9+ tokens. |
| fp16 WMMA for the sparse-attention indexer scores in prompt windows | `GLM_N3=0` | Queries rounded to fp16 (the key cache is already fp16), fp32 accumulation. Windows of 9+ tokens. |

**Exact defaults (kill switches, `=0`):** `GLM_SIDE_STREAM` (concurrent side stream), `GLM_PAIR`, `GLM_KQ_IDS2`
(paired MTP matvecs), `GLM_DEV_GREEDY` (greedy argmax on the device), `GLM_KDA_FUSED` (fused KDA step),
`GLM_FUSE_MOE_POST`, `TALUS_W16_CMP` (compact active-expert grid for the fp16 prompt experts), `GLM_GRAPHS` (HIP graphs),
`GLM_SPARSE_SKIP`, `STRATA_MMQ_KQ`, and `STRATA_PREFILL_NO_BLAS` (memory: no hipBLAS workspace).

**Opt-ins:** `GLM_N4=1` (fp16 WMMA for the `Q4_K` attention and shared-expert projections in prompt windows; changes
numerics; speeds up prompts and passes the KLD and needle gates and the agentic smoke test, kept off by default because it is the
largest numeric step of the three), `GLM_COPY=1` (long prompt copies; see **Long copies** under Laguna),
`GLM_SPARSE_WMMA=1` or `2` (changes numerics; only with `GLM_SPARSE_2P=0`), `GLM_HC_BF16=1` (fp16 mHC streams in
prompt windows; changes numerics, measured slower). Exact but measured slower inside the decode graph:
`GLM_FUSE_ROUTER=1`, `GLM_FUSE_HC_PRE=1`.

**Where time goes.** In long-prompt chunks, routed expert GEMMs take the largest share of kernel time, followed by the
dense MLA/KDA projections and two-pass sparse attention. In decode and NextN verification, memory reads for the
46 routed-expert layers dominate each step, followed by MLA/KDA projections, sparse attention and the NextN draft head.

**Prompt cache.** Exact: recurrent state is snapshotted only on the 512-token prompt grid (`GLM_PROMPT_SNAPS`
snapshots, default 1), never from decode.

**Memory.** Weights about 110 GB, the largest of the five. It serves at about 9 GiB free, closest of the five to the 8 GiB floor. Lowest
`MemAvailable` in the v0.1.2 release run on our desktop: 9 GiB.

## MiMo-V2.6-Flash

Mixture-of-experts model (the MOPD variant). Engine `mimo`.

| | |
|---|---|
| Weights | `MiMo-V2.6-Flash-a4-x` ([j2ipper/MiMo-V2.6-Flash-MOPD-a4-x](https://huggingface.co/j2ipper/MiMo-V2.6-Flash-MOPD-a4-x)), three shards (`88.81 GiB` / `95,359,839,096 B` total: `MiMo-V2.6-Flash-a4-x-00001-of-00003.gguf` `5,949,240 B` sha256 `b31e4530e9d0735bcd98b1f6f647fbec406594e0e4ab77134589d8fb9afd0f49`, `MiMo-V2.6-Flash-a4-x-00002-of-00003.gguf` `48,728,061,760 B` sha256 `8c02693229bde8c7266ac2d06727ca55387024f50c0bb1eb002d044d886d31f1`, `MiMo-V2.6-Flash-a4-x-00003-of-00003.gguf` `46,625,828,096 B` sha256 `9b0243a7901eff27006a8dcb49487d4d5984034e7332048bcbb5a48e0ad4fbe9`): our low-bit quant of `XiaomiMiMo/MiMo-V2.6-Flash-MOPD`, made from Xiaomi's original weights with an importance matrix (`mimo-a4x.imatrix.gguf`, `992,394,624 B`, sha256 `958147270e4e3037bb291cdab64ee2949de95e2c31cdd53e7734e41ccf4f4809`) from public datasets |
| Drafter | `dflash-MiMo-V2.6-Flash-MOPD-Q8_0.gguf` from [ggml-org/MiMo-V2.6-Flash-MOPD-GGUF](https://huggingface.co/ggml-org/MiMo-V2.6-Flash-MOPD-GGUF), the model's DFlash drafter at Q8_0 (about 2.7 GB) |
| Draft vocabulary | `mimo-dv-65536.bin` ([j2ipper/draft-vocab](https://huggingface.co/j2ipper/draft-vocab), 64K token ids) |
| Context / KV | 192K shipped (`--max-context 196608`); KV is f16 (the engine ignores `--kv q8`) |
| License | MIT |

Shipped arguments:

```
--model MiMo-V2.6-Flash-a4-x-00001-of-00003.gguf --max-context 196608 --ubatch 2048
--dflash-gguf dflash-MiMo-V2.6-Flash-MOPD-Q8_0.gguf --spec 4 --spec-draft-vocab mimo-dv-65536.bin
```

No environment variables are needed. Without `--max-context` the engine starts at 4K.

**Drafting.** The DFlash drafter proposes up to `--spec N` tokens per round (`--mtp N` is accepted as the same
thing; it does not select an MTP head). Long prompt copies (see **Long copies** under Laguna) are on by default
(`TALUS_MIMO_COPY=0` turns them off). The draft length and pauses adapt to the measured acceptance
(`TALUS_MIMO_ADAPTIVE=0` turns that off). `--no-mtp` turns drafting off. If `--dflash-gguf` is not given and no
drafter is found, the engine runs without drafting.

**Defaults that change numerics**

| Default | Kill switch | Notes |
|---|---|---|
| Wave32 matvec for routed experts | `STRATA_IDS_W32=0` | The engine sets `STRATA_IDS_W32=1` unless `STRATA_IDS_W32` or `TALUS_MIMO_W32_DENSE` is already set (setting the latter at all, even to 0, keeps the default off). Logit differences are at floating-point rounding level. |
| WMMA f16 flash attention in prompts | `TALUS_MIMO_FA=0` | `TALUS_MIMO_FA_VER=1` selects the earlier v1 tile. |
| Confident-prefix verify for greedy requests | `TALUS_MIMO_DF_CONF=0` | After the DFlash drafter runs, the round verifies only the longest prefix of its drafts whose running product of the drafter's own confidence in each draft stays at or above 0.6; the rest would mostly be rejected. Greedy output is unchanged (every token is still the model's own argmax); code about 6 % faster. Sampled requests keep the full block (`TALUS_MIMO_DF_CONF_S=<threshold>` opts in) |
| Last-layer tail-row pruning in prompts | `TALUS_MIMO_TAIL_PRUNE=0` | The last layer computes only the rows the output head and the drafter read (about 3 % faster prompts at 32K). Prompt results are bitwise equal; a short request that resumes inside another prompt's pruned rows re-reads up to 1152 rows (1-2 s), so its text can differ from a run without pruning |

**Exact defaults (kill switches, `=0`):** `TALUS_MIMO_ATT2`, `TALUS_MIMO_ATT2_DRAFT`, `TALUS_MIMO_ATT3` (decode and
drafter attention kernels), `TALUS_MIMO_ATT_Q8`, `TALUS_MIMO_QKV_MULTI`, `TALUS_MIMO_ROPE_STORE`,
`TALUS_MIMO_AXPY_NORM`, `TALUS_MIMO_MOE_ADDNORM` (fusions), `TALUS_MIMO_I8` (int8 prompt cache for dense `attn_qkv`
and layer-0 `Q6_K` FFN), `TP_MOE_IQ1M_ONE` (shared-input `IQ1_M` prompt GEMM), `TALUS_MIMO_COPY` (long prompt
copies), `TALUS_MIMO_GRAPH` (HIP graphs), and the shared prompt switches `TALUS_PF_MOE`, `TALUS_PF_MOE_GU`,
`TALUS_PF_DENSE_MOE`.

**Memory.** Weights and drafter about 92 GB. The engine keeps a run-time floor of 8 GiB free for this model. Lowest
`MemAvailable` in the v0.1.2 release run on our desktop: 15 GiB.

## Laguna-S-2.1

Mixture-of-experts model from poolside with sliding-window and full-attention layers. Engine `laguna`.

| | |
|---|---|
| Weights | `Laguna-S-2.1-Q4-lean` ([j2ipper/Laguna-S-2.1-Q4-lean](https://huggingface.co/j2ipper/Laguna-S-2.1-Q4-lean)), three shards (about 68 GB): Unsloth's UD-Q4_K_XL of `poolside/Laguna-S-2.1` with leaner attention and shared experts |
| Drafter | The DFlash block-diffusion drafter `laguna-s-2.1-dflash-Q8_0.gguf` (Q8_0, about 1.2 GB, sha256 `b8dd55dddf48a32442c100493b4bdd7e2b2ec25bafcef46d1c977405abdcebf7`) from [j2ipper/Laguna-S-2.1-DFlash-GGUF](https://huggingface.co/j2ipper/Laguna-S-2.1-DFlash-GGUF): our Q8_0 of poolside's BF16 drafter GGUF ([`poolside/Laguna-S-2.1-GGUF`](https://huggingface.co/poolside/Laguna-S-2.1-GGUF), OpenMDW-1.1, from [`poolside/Laguna-S-2.1-DFlash`](https://huggingface.co/poolside/Laguna-S-2.1-DFlash)), with tensor data identical to the community conversion our earlier figures used; plus prompt lookup, which needs no extra weights. On by default from v0.1.2: the guided install downloads it with the other drafters and passes `--dflash-gguf <file>`; greedy output is byte-identical with and without it. `talus setup --advanced` (drafters off) leaves it out, and Laguna then drafts with prompt lookup only. An install with the earlier community file fails the new checksum check; `talus doctor --fix` downloads ours. |
| Context / KV | 256K, the model's own (`--max-context 262144`), q8_0 KV |
| Stop tokens | `〈|EOS|〉`, `</assistant>` (`--eos-ids 2,24`) |
| License | OpenMDW-1.1 (and poolside's Acceptable Use Policy) |

Shipped arguments:

```
--model Laguna-S-2.1-Q4-lean-00001-of-00003.gguf --max-context 262144 --ubatch 2048
--eos-ids 2,24 --spec 7 --sparse 1024,16384,128,1024,16
--dflash-gguf laguna-s-2.1-dflash-Q8_0.gguf
```

No environment variables are needed. Without `--max-context` the engine starts at 4K.

The 36 sliding-window layers keep a ring of recent positions, so only the 12 full-attention layers grow with the
context. That is why 256K fits.

**Drafting.** The DFlash drafter (`--dflash-gguf`) proposes a block of draft tokens from the model's hidden states,
and prompt lookup (`--spec 7`) proposes up to 7 tokens when the recent text matches earlier text (at least
`--spec-min-match` tokens, default 8), sized per step from the measured acceptance. When both are enabled, the host
compares DFlash's expected accepted tokens (`TALUS_LAGUNA_DF_P0`, default 0.85, scaled by `TALUS_LAGUNA_DF_P0_CMP`,
default 0.6) against the available prompt-lookup or long-copy match before running the DFlash forward pass. Omit
`--dflash-gguf` and set `TALUS_LAGUNA_DFLASH=0` for prompt lookup only (without the flag the engine still loads a
`*dflash*Q8_0*.gguf` it finds next to the model); `--spec 0` turns prompt lookup off. An
n-gram/suffix fallback exists as an opt-in (`TALUS_LAGUNA_SUFFIX=1`).

**Long copies.** When the last few tokens match an earlier span in the prompt or answer (common in file edits), the
engine can propose the continuation straight from that span, and if a neural drafter (MTP or DFlash) is active it can
either skip the drafter for that step or extend the drafter's proposals up to the verify window. Verification is the
same single model pass, so greedy output is bit-for-bit identical and sampled output keeps the target distribution.
Long copies are on by default for Flash-Next (`TALUS_COPY=0` turns them off in every engine), MiMo (`TALUS_MIMO_COPY=0`) and
Laguna (`TALUS_LAGUNA_COPY=0`), and opt-in for Qwen3.8-27B (`TALUS_Q27_COPY=1`) and GLM (`GLM_COPY=1`). The design was
prompted by gufo's speculative-decoding notes (`THIRD_PARTY_NOTICES.md` §6).

**Block-sparse decode attention** (`--sparse K,MIN,SINK,RECENT,BS`, shipped on). Past a context length, decode
attends to the sink, the recent window and the top key blocks instead of every position. It changes numerics, so it
starts only at long context: the start is at least 32,000 tokens (`LAGUNA_SPARSE_32K`, default on), even when `MIN`
is lower. It needs q8 KV.

| Default | Kill switch | Notes |
|---|---|---|
| Confident-prefix verify for greedy requests | `TALUS_LAGUNA_DF_CONF=0` | After the DFlash drafter runs, the round verifies only the longest prefix of its drafts whose running product of the drafter's own confidence in each draft stays at or above 0.45; the rest would mostly be rejected. Greedy output is unchanged (every token is still the model's own argmax); code about 12 % faster. Sampled requests keep the full block (`TALUS_LAGUNA_DF_CONF_S=<threshold>` opts in) |
| Sparse decode from 32K | omit `--sparse` (dense everywhere); `LAGUNA_SPARSE_32K=0` (start from `MIN`, floored at four times `BS`×`K`); `LAGUNA_SPARSE_MIN=65536` (start later) | Changes numerics past the start. |
| Tail-row pruning in prompts | `TALUS_LAGUNA_TAIL_PRUNE=0` | Layers 44-47 compute only the rows that the output head, the drafter and later queries read (about 7.5 % faster prompts at 32K). Prompt results are bitwise equal. After four or more requests in a row that each resume about 1,700 tokens further back, the engine prefills the prompt again from the start, so that request's text can differ from a run without pruning |

**Exact defaults (kill switches):** `LAGUNA_EXPERT_MAJOR=0` (expert-major routed matvec), `STRATA_MMVQ_XT=0` (wide
multi-token dense verify matvec, default 9 rows), `TALUS_LAGUNA_I8=0` (int8 prompt cache for dense `Q5_K`/`Q6_K`
attention and layer-0 `Q8_0` FFN), `TALUS_LAGUNA_COPY=0` (long prompt copies), `LAGUNA_NO_WMMA=1` (no WMMA prompt
attention/GEMM), `STRATA_MMQ_KQ=0`, and the shared `TALUS_PF_MOE=0`, `TALUS_PF_DENSE_MOE=0`. The engine runs the
decode matvec without wave32 to match its reference.

**Opt-ins (change numerics):** `LAGUNA_SPARSE_UNION=1`, `LAGUNA_SPARSE_F32=1`.

**Prompt cache.** The engine reuses the KV of the live conversation and also saves each conversation to disk, so it
survives restarts and model switches (on by default; see [Prompt cache on disk](#prompt-cache-on-disk)).

**Memory.** Weights about 68 GB; 256K of q8 KV for the full-attention layers fits in the rest. Lowest
`MemAvailable` in the v0.1.2 release run on our desktop: 38 GiB.

## Prompt cache on disk

Qwen3.8-27B and Laguna keep their prompt cache on disk as well as in memory. After each reply the engine writes the
conversation's state to disk in the background; when a request continues a conversation that is on disk but not in
memory (after a server restart, a model switch in talus-tui, or because memory evicted it), the engine loads it instead
of reading the whole prompt again.

| Setting | Default | Meaning |
|---|---|---|
| `TALUS_PCACHE_DISK` (27B also `--prompt-cache-disk`) | on: `$XDG_CACHE_HOME/talus/pcache`, else `~/.cache/talus/pcache` | A folder, or `0` / `off` to turn the disk cache off |
| `TALUS_PCACHE_DISK_GIB` (`--prompt-cache-disk-gib`) | 32 | Disk budget; the least recently used entries are deleted first (all models share it). `0` turns it off |
| `TALUS_PCACHE_DISK_MIN` (`--prompt-cache-disk-min`) | 1024 | Shortest prompt, in tokens, worth an entry |

In a talus-server run config, `"prompt_cache_disk": false` (or `"off"`, or a folder) and `"prompt_cache_disk_gib": N`
set the same variables for the engine. talus-tui shows the setting as "Prompt cache on disk".

- **Exact.** For Qwen3.8-27B a restore from disk puts the same bytes back through the same code as a hit in memory,
  so answers are bit-identical to an engine that never restarted. Laguna reloads the KV its live cache held; greedy
  answers are identical, and the drafter starts without context after a restore.
- **Never stale.** Entries are keyed by the model files, the engine build, the kernel and BLAS libraries, the GPU and
  every numerics switch, so a cache written by another model, quant, build or setting is never read. A file that
  fails any check is deleted and the prompt is read normally.
- **Safe writes.** Files are written under a temporary name, flushed and renamed. At most one write runs at a time,
  and a write is skipped when copying the state would leave less than 8 GiB of `MemAvailable`.
- **Never fills the disk.** Besides its budget, the cache does not grow while its filesystem has less than
  max(20 GiB, a tenth of its size) free; it logs this once and keeps using the entries it has.
- **Back-to-back requests.** If the next request is already waiting when a reply ends (or arrives while the turn is
  copied off the GPU), that turn is not written, so the next request is not delayed. A continuing conversation loses
  nothing (its next turn writes a deeper entry); separate long conversations sent back to back may not all reach the
  disk. A stop (SIGTERM / Ctrl-C) waits up to 60 s for the turn being written.
- **Not yet:** MiMo-V2.6-Flash, Qwen3.8-Flash-Next and GLM-5.3-Flash keep their prompt cache in memory only.

## Shared switches

These apply to every engine that uses the shared kernel library paths (all but Flash-Next's prompt path):

| Switch | Default | Effect when changed |
|---|---|---|
| `TALUS_PF_MOE=0` | on | Grouped MoE prompt GEMM off (exact) |
| `TALUS_PF_MOE_GU=0` (`=2` forces on) | on, by the tuning table | Fused gate/up + SwiGLU off (exact) |
| `TALUS_PF_DENSE_MOE=0` | on | Dense-layer prompt path off |
| `TP_MOE_VER`, `TP_MOE_KTAIL=0` | tuned version, on | Kernel version / tail handling (exact) |
| `STRATA_MMVQ_W32=0` | on (gfx11) | Wave32 decode matvec off |
| `STRATA_GFX1151_DEFAULTS=0`, `STRATA_GFX12_DEFAULTS=0` | on | Architecture default tables off (applied by the Flash-Next path) |

Loader and runtime: `TALUS_KERNELS_LIB` (path of `libtalus_kernels.so`), `TALUS_KERNELS_PREBUILT` (its directory),
`TALUS_HIP_LIB` (path of `libamdhip64.so`), `ROCM_SDK` / `ROCM_PATH`, and `TALUS_API_KEY` (or `STRATA_API_KEY`) for the server.

Variables not listed here are tuning or debug knobs (`*_TRACE`, `*_DUMP*`, `*_PROF`, table overrides such as
`STRATA_MMVQ_W32_TAB`). They are for development; leave them unset.
