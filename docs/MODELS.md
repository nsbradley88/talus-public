# Models

Talus serves five models. This page lists, for each one, the files we run, the drafter, the arguments and
environment we ship, the switches that turn defaults off or opt into other numerics, and the memory it needs.

How to read the switch tables:

- **Kill switch:** the feature is on by default; setting the variable as shown turns it off.
- **Opt-in:** the feature is off by default. "Changes numerics" means its logits differ from the default path, so
  it has to pass the quality gates before it can become a default ([METHODOLOGY.md](METHODOLOGY.md#exact-and-opt-in-numerics)).
- "Exact" (bitwise) switches change speed only: the output is identical either way.
- A variable you export always wins over an engine's built-in default.

`talus-cli` picks the engine from the GGUF's `general.architecture`. In the server's run config, put the arguments in
`"args"` and the variables in its `"env"` (or export them before starting the server). Speed and quality
figures for every model are on [RESULTS.md](RESULTS.md).

All five models run with a minimum of free memory (`MemAvailable`) in mind: the serve scripts keep at least 8 GiB
free and stop the server below that (GLM-5.3-Flash runs closest to it, at about 9 GiB). One model runs per machine.

## System setup

On a default install, the GPU can use only part of the 128 GB, and GLM-5.3-Flash and MiMo-V2.6-Flash do not fit.
Our measurements use these kernel parameters (Ubuntu: a file in `/etc/default/grub.d/`, then `sudo update-grub` and a
reboot):

```
ttm.pages_limit=31457280 ttm.page_pool_size=31457280   # GPU memory pool 120 GiB
kho=off                                                # free the ~9 GiB kexec-handover reserve (Ubuntu 7.0 kernels)
amdgpu.lockup_timeout=10000,60000,10000,10000          # let long prompt passes run (60 s compute timeout)
```

We also run headless (`sudo systemctl set-default multi-user.target`). This is approximate (release-run free memory,
not a separate measurement): with a desktop session running, Qwen3.8-27B, Qwen3.8-Flash-Next and Laguna-S-2.1 still
have tens of GiB to spare, MiMo-V2.6-Flash is tight, and GLM-5.3-Flash needs headless. These are boot settings: check
them against your distribution's documentation before you apply them.

Our own files (the GLM, MiMo and Laguna quants and the 64K draft vocabularies) are published on Hugging Face with
model cards that give the recipe, the tensor types and the licenses.

## Qwen3.8-27B

Dense model with gated delta net and attention layers. Engine `qwen35`.

| | |
|---|---|
| Weights | `Qwen3.8-27B-UD-IQ4_XS.gguf` from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (about 14 GB). `UD-Q4_K_XL` also runs. |
| MTP head | `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` from the same repository (about 1.3 GB) |
| DFlash2 drafter | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` from [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF), z-lab's block-diffusion drafter (about 1.1 GB) |
| Draft vocabulary | `qwen-dv-65536.bin` ([j2ipper/draft-vocab](https://huggingface.co/j2ipper/draft-vocab)): 64K token ids in llama.cpp's int32 format |
| 4-bit prompt weights | `Qwen3.8-27B-UD-IQ4_XS.w4i` ([j2ipper/Qwen3.8-27B-W4A4-prompt](https://huggingface.co/j2ipper/Qwen3.8-27B-W4A4-prompt), about 2.0 GiB), placed next to the GGUF. Optional (opt-in `TALUS_Q27_W4I=auto`): by default prompts run on the exact int8 path. `Qwen3.8-27B-UD-IQ4_XS.fast.w4i` (about 8.0 GiB) only for the opt-in fast profile. Built for `UD-IQ4_XS` only. |
| Context / KV | 64K shipped (`--max-context 65536`), f16 KV |
| Stop token | `<|im_end|>` |
| License | Apache-2.0 (model, MTP head, DFlash2 drafter) |

Shipped arguments:

```
--model Qwen3.8-27B-UD-IQ4_XS.gguf --max-context 65536
--mtp-gguf mtp-Qwen3.8-27B-Q4_0.gguf --mtp 5
--spec-draft-vocab qwen-dv-65536.bin --ngram 24
--dflash-gguf Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

No environment variables are needed. 4-bit prompt math is an opt-in (`TALUS_Q27_W4I=auto`, see **4-bit prompt
math** below). Without `--max-context` the CLI starts at 32K. Without the drafter flags, no speculative decoding runs.

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
| GDN commit fused into the next forward pass, and draft/verify rounds on the device | `STRATA_FUSED=0`; `STRATA_NO_DEV_ROUNDS=1` keeps the fusion but runs rounds on the host |
| Depth switch between drafters | `STRATA_DRAFTER=mtp` or `--drafter mtp` (MTP only); `--drafter dflash` (DFlash2 only) |
| Adaptive draft length | `STRATA_ADRAFT=0` (fixed length) |
| One-tile row groups in decode attention | `TALUS_ATTN_RT1G=0` |
| Split decode attention | `STRATA_ATTN_SPLIT=0` |
| HIP graphs | `--no-graph` |

**4-bit prompt math (opt-in, `TALUS_Q27_W4I=auto`).** It passed our quality gate (see [METHODOLOGY.md](METHODOLOGY.md),
needle and agentic suites) and makes prompts about seven percent faster, but it is off by default: the changed prompt
numerics lower draft acceptance afterwards, so generation after a 2K prompt is about eleven percent slower (about three
percent after 32K). It pays off only when each request brings a very long new prompt and a short answer; for multi-turn
chat and coding, leave it off. When on, the prompt (prefill) GEMMs of the feed-forward in layers 48-63 run on the `iu4`
matrix instruction, twice the int8 rate. Activations are rotated per 256 values (seeded Hadamard) and quantized to int4
per 32 values; the 32-value scales are small integers times a per-token scale, so the kernel rescales with one integer
multiply-add. The int4 weights are a separate GPTQ quantization of the GGUF's weights in the same rotated basis (`.w4i`
file; build it with `tools/w4i/build_sidecar.sh`, recipe in `tools/w4i/README.md`). Decode, verification and drafting
never use it. Only the last 16 layers: earlier layers cost several times more accuracy per layer (layer sweep in
[METHODOLOGY.md](METHODOLOGY.md)). The file records the GGUF's name and size; a different GGUF, a missing file or an
unset or `0` value keeps the exact int8 prompt path (a missing file logs one line). `TALUS_Q27_W4I=<path>` loads a given
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
but warm answers that may differ from cold.

**Memory.** Weights and drafters are about 17 GB. The int8 prompt GEMM expands the trunk weights at load time and
uses about 26 GiB more GPU memory (`TALUS_PF_I8=0` frees it). The 4-bit prompt weights add about 2.0 GiB (8.0 GiB
with the fast profile). Lowest `MemAvailable` in the release run: 66 GiB.

## Qwen3.8-Flash-Next

Mixture-of-experts model, Strata's original model. Engine `qwen4exp`: the Rust host drives Strata's C++ data plane in
the kernel library (a pure-Rust port is in progress). All of Strata's flags and `STRATA_*` variables apply.

| | |
|---|---|
| Weights | Unsloth `Qwen3.8-Flash-Next-UD-IQ4_XS` GGUF, three shards (about 88 GB), from `unsloth/Qwen3.8-Flash-Next-GGUF` |
| Pack | A Strata pack built from the GGUF with `tools/iq_pack.py --gguf <shard 1> --out <pack> --compat-bf16`; it also holds the tokenizer |
| MTP | The MTP runtime directory built with `tools/mtp_fetch.py`, `tools/mtp_pack.py` and `tools/mtp_rt.py` (see `docs/UNSLOTH_Q4.md`) |
| Expert profile | `data/expert-profile.bin` (in this repository) |
| Context / KV | 256K shipped (`--max-context 262144`), int8 KV |
| License | Qwen Community License 1.0. Note its condition 2: a business that offers model inference as a service or an AI coding / office assistant needs a separate license from Qwen for commercial use. |

Shipped arguments:

```
--pack <pack> --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
--expert-profile data/expert-profile.bin --expert-cache 24576 --resident-budget-gib 55
--prefill auto:16384 --spec 5 --spec-min-p 0.5 --mtp <mtp runtime dir> --mtp-window 8192
--mtp-draft-vocab data/draft_vocab_en.bin --max-context 262144 --kv int8
```

Shipped environment (these turn on prompt paths that are off in Strata's code defaults):

```
STRATA_HIPBLASLT_TUNING=tools/hip/gfx1151-hipblaslt-100401.txt
STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_HC_UPMIX=1 STRATA_PA_FAST=1 STRATA_HIP_WMMA=1
STRATA_SELECT_WMMA=1 STRATA_HC_Q8=1 STRATA_PF_SWITCH_MIN_T=0 STRATA_PREFILL_STREAM_MIN=128
```

The hipBLASLt table is chosen per ROCm version: if the named table is for another hipBLASLt version, the engine loads
the file for the running version from the same directory.

**Drafting.** The MTP head with `--spec 5` and a window of 8192 tokens, plus a suffix drafter (on by default,
`--suffix-draft 3`; `--suffix-draft 0` for MTP only) that proposes continuations of earlier text. A long copy (see
**Long copies** under Laguna) fills a whole 8-token verify window without the MTP head's drafts (the head then only
catches up), or extends the head's drafts; the verify window is 8 tokens (`--spec 8 --mtp-max-t 5`). The MTP head drafts
from the English-and-code token subset `data/draft_vocab_en.bin` (40,525 tokens, an 81 MB head instead of the runtime
directory's 106,299-token, 213 MB one): each draft step reads fewer bytes. Verification is unchanged, so greedy output
stays exact and sampled requests keep their exact distribution; only the drafts differ. For mostly non-English (CJK)
use, where tokens outside the subset cannot be drafted, remove the flag (the full head) or pass
`data/draft_vocab_cyrillic.bin` (58,963 tokens: Latin and Cyrillic scripts).

**Defaults that change numerics** (on gfx1151, or with the shipped environment)

| Default | Kill switch | Notes |
|---|---|---|
| HC down and inject projections as one WMMA GEMM, for prompts of 16K tokens or more | `STRATA_PF_HCDOWN=0` (`=1`: on for every prompt) | Rounding-level; shorter prompts stay exact. Threshold `STRATA_PF_HCDOWN_MIN` (16384). Disabled by default on `hipBLASLt 100500` unless `STRATA_HCD_EXACT=1` is set. |
| fp16 prompt experts | `STRATA_PF_W16=0` | Runs inside the fused prompt path, so it is on only with the shipped `STRATA_PF_FUSED=1`. |
| Q4_K / Q5_K decode lane layout | `STRATA_MMVQ_KQ8=0` | |
| Wave32 decode matvec | `STRATA_MMVQ_W32=0` | |

**Opt-in (changes numerics):** `TALUS_FN_HC_BF16=2` stores the prompt-path residual (hyper-connection) streams in fp16
(`=1`: bf16). It speeds up prompts and passes needle and agentic, but its first-token KL is over the release gate, so
it is off by default.

**Exact defaults.** `STRATA_GFX1151_DEFAULTS=0` turns off the whole gfx1151 table of bitwise speed switches (and the
HC-down default above). The table is in `src/core/arch_defaults.cpp`; each entry can also be set on its own.
`TALUS_FN_COPY=0` turns off long prompt copies. The turn-checkpoint GDN buffer pool omits unused `conv_state` staging to save host/device memory.

**Memory.** Weights about 88 GB (`UD-IQ4_XS`), read through the expert cache (`--expert-cache` in MiB) and a resident budget
(`--resident-budget-gib`). Unsloth's `UD-Q4_K_XL` GGUF also runs when packed with `tools/iq_pack.py`: its larger `K`-quant experts use more memory and increase expert-cache traffic on long prompts and long-context decode, while reading slightly faster on short cold prompts. Lowest `MemAvailable` in the release run: 51 GiB.

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

**Memory guard.** The engine refuses to start unless `MemAvailable` is at least `--minstart` GiB (default 112), and
exits if it drops below `--minavail` GiB (default 8) while running. Start it on an otherwise idle machine.

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
`GLM_FUSE_MOE_POST`, `GLM_FUSE_POST_NORM` (fused post-attention and post-FFN RMSNorm into the residual),
`GLM_FUSE_GU_W16` (single-pass fp16 MoE gate+up dequant), `GLM_ACT_ONCE` (shared activation quant across routed and
shared experts), `GLM_SWIGLU_Q8` (fused MoE SwiGLU into `q8_1`), `GLM_W16_CMP` (compact active-expert offset remap),
`GLM_RSORT_PAR` (parallel router sort), `GLM_MV_B4` (4-row decode/verify matvec), `GLM_GRAPHS` (HIP graphs),
`GLM_SPARSE_SKIP`, `STRATA_MMQ_KQ`, and `STRATA_PREFILL_NO_BLAS` (memory: no hipBLAS workspace).

**Opt-ins:** `GLM_N4=1` (fp16 WMMA for the `Q4_K` attention and shared-expert projections in prompt windows; changes
numerics; speeds up prompts and passes the KLD, needle and agentic gates, kept off by default because it is the
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
`MemAvailable` in the release run: 9 GiB.

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
| WMMA f16 flash attention in prompts | `TALUS_MIMO_FA=0` | `TALUS_MIMO_FA_V1=1` selects the earlier v1 tile. |

**Exact defaults (kill switches, `=0`):** `TALUS_MIMO_ATT2`, `TALUS_MIMO_ATT2_DRAFT`, `TALUS_MIMO_ATT3` (decode and
drafter attention kernels), `TALUS_MIMO_ATT_Q8`, `TALUS_MIMO_QKV_MULTI`, `TALUS_MIMO_ROPE_STORE`,
`TALUS_MIMO_AXPY_NORM`, `TALUS_MIMO_MOE_ADDNORM` (fusions), `TALUS_MIMO_I8` (int8 prompt cache for dense `attn_qkv`
and layer-0 `Q6_K` FFN), `TALUS_PF_MOE_IQ1M_ONE` (shared-input `IQ1_M` prompt GEMM), `TALUS_MIMO_COPY` (long prompt
copies), `TALUS_MIMO_GRAPH` (HIP graphs), and the shared prompt switches `TALUS_PF_MOE`, `TALUS_PF_MOE_GU`,
`TALUS_PF_DENSE_MOE`.

**Memory.** Weights and drafter about 92 GB. The host keeps a floor of 11 GiB free for this model. Lowest
`MemAvailable` in the release run: 14 GiB.

## Laguna-S-2.1

Mixture-of-experts model from poolside with sliding-window and full-attention layers. Engine `laguna`.

| | |
|---|---|
| Weights | `Laguna-S-2.1-Q4-lean` ([j2ipper/Laguna-S-2.1-Q4-lean](https://huggingface.co/j2ipper/Laguna-S-2.1-Q4-lean)), three shards (about 68 GB): Unsloth's UD-Q4_K_XL of `poolside/Laguna-S-2.1` with leaner attention and shared experts |
| Drafter | `laguna-s-2.1-dflash-Q8_0.gguf` (DFlash block-diffusion drafter at Q8_0, about 1.6 GB), plus prompt lookup |
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
`--dflash-gguf` (or set `TALUS_LAGUNA_DFLASH=""`) for prompt lookup only; `--spec 0` turns prompt lookup off. An
n-gram/suffix fallback exists as an opt-in (`TALUS_LAGUNA_SUFFIX=1`).

**Long copies.** When the last few tokens match an earlier span in the prompt or answer (common in file edits), the
engine can propose the continuation straight from that span, and if a neural drafter (MTP or DFlash) is active it can
either skip the drafter for that step or extend the drafter's proposals up to the verify window. Verification is the
same single model pass, so greedy output is bit-for-bit identical and sampled output keeps the target distribution.
Long copies are on by default for Flash-Next (`TALUS_FN_COPY=0` turns them off), MiMo (`TALUS_MIMO_COPY=0`) and
Laguna (`TALUS_LAGUNA_COPY=0`), and opt-in for Qwen3.8-27B (`TALUS_Q27_COPY=1`) and GLM (`GLM_COPY=1`). The design was
prompted by gufo's speculative-decoding notes (`THIRD_PARTY_NOTICES.md` §6).

**Block-sparse decode attention** (`--sparse K,MIN,SINK,RECENT,BS`, shipped on). Past a context length, decode
attends to the sink, the recent window and the top key blocks instead of every position. It changes numerics, so it
starts only at long context: the start is at least 32,000 tokens (`LAGUNA_SPARSE_32K`, default on), even when `MIN`
is lower. It needs q8 KV.

| Default | Kill switch | Notes |
|---|---|---|
| Sparse decode from 32K | omit `--sparse` (dense everywhere); `LAGUNA_SPARSE_32K=0` (start from `MIN`, floored at four times `BS`×`K`); `LAGUNA_SPARSE_MIN=65536` (start later) | Changes numerics past the start. |

**Exact defaults (kill switches):** `LAGUNA_EXPERT_MAJOR=0` (expert-major routed matvec), `STRATA_MMVQ_XT=0` (wide
multi-token dense verify matvec, default 9 rows), `TALUS_LAGUNA_I8=0` (int8 prompt cache for dense `Q5_K`/`Q6_K`
attention and layer-0 `Q8_0` FFN), `TALUS_LAGUNA_COPY=0` (long prompt copies), `LAGUNA_NO_WMMA=1` (no WMMA prompt
attention/GEMM), `STRATA_MMQ_KQ=0`, and the shared `TALUS_PF_MOE=0`, `TALUS_PF_DENSE_MOE=0`. The engine runs the
decode matvec without wave32 to match its reference.

**Opt-ins (change numerics):** `LAGUNA_SPARSE_UNION=1`, `LAGUNA_SPARSE_F32=1`.

**Memory.** Weights about 68 GB; 256K of q8 KV for the full-attention layers fits in the rest. Lowest
`MemAvailable` in the release run: 38 GiB.

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
`TALUS_HIP_LIB` (path of `libamdhip64.so`), `ROCM_SDK` / `ROCM_PATH`, and `STRATA_API_KEY` for the server.

Variables not listed here are tuning or debug knobs (`*_TRACE`, `*_DUMP*`, `*_PROF`, table overrides such as
`STRATA_MMVQ_W32_TAB`). They are for development; leave them unset.
