# Results

This page is generated from the recorded release runs of Talus v0.1.2. The speed tables come from one
release run of the release tarball on the desktop machine below; the quality gates come mostly from the laptop. The comparisons with other engines further down are older runs on the laptop, with the v0.1.0 builds, and say
so. A model whose release run has not passed every gate shows "pending", and so does a figure that was not measured.
How each cell and gate is measured: [METHODOLOGY.md](METHODOLOGY.md).

| Release | |
|---|---|
| Talus | the v0.1.2 release candidate tarball (rc2), served by its own `talus-server` with the run configs `talus setup` writes. The published v0.1.2 tarball is the same source built from this repository; it gives identical greedy output on all five models |
| Speed runs | 2026-11-02, on the desktop (GMKtec EVO-X2: Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB, about 120 W sustained, BIOS set to its highest sustained power mode) |
| Quality gates | 2026-11-01 to 2026-11-02, mostly on the laptop (HP ZBook Ultra G1a: Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB, 70 W firmware ceiling); "Quality gates" below says which build and machine each gate ran on |
| ROCm SDK | 10.2.0a20261014 |

The two machines have the same chip and memory but different power limits, so a figure from one is not comparable with
a figure from the other. Every table below names its machine.

| Model | Release run |
|---|---|
| Qwen3.8-27B | v012ab2-qwen27b-t (2026-11-02) |
| Qwen3.8-Flash-Next | v012ab2-flashnext-t (2026-11-02) |
| GLM-5.3-Flash | v012rc2-rel3 (2026-11-02) |
| MiMo-V2.6-Flash | v012rc2-rel3 (2026-11-02) |
| Laguna-S-2.1 | v012rc2-rel3 (2026-11-02) |

Read the tables with these points in mind:

- Results inside a noise band are a tie (bands in [METHODOLOGY.md](METHODOLOGY.md#noise-bands)).
- `pp2k` / `pp32k` and `tg2k` / `tg32k` are tokens over time across 12 prompts at about 2K tokens and 8 prompts at
  about 32K tokens. The noise band we have for decode between engines whose generated text differs is the four-prompt
  one (10.5%); we have not derived one for the 12 / 8-prompt cells yet.
- Each model runs its shipped default configuration (see [MODELS.md](MODELS.md)). Qwen3.8-27B's shipped context is
  chosen by memory (256K on 128 GB); its table also shows a run at 64K, measured on the first release candidate of
  the same code.
- The "v0.1.1" rows are the v0.1.1 release binaries on the same desktop, through the same `talus-server`, with the
  engine arguments of our test harness and the sampling that v0.1.2 now ships (the frontend-parity runs of
  2026-11-01). GLM-5.3-Flash v0.1.1 stops at its memory floor on this desktop, so only its `chat` and `code` cells
  exist.

## Speed (desktop)

### Qwen3.8-27B

Speeds in tokens per second.

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.2 (shipped, 256K context) | 51.3 | 67.8 | 122.9 | 597 | 545 | 49.6 | 36.2 | 3.32 s / 58.83 s / 1.60 s | 42 GiB |
| Talus v0.1.2 release candidate 1, 64K context (`--max-context 65536`) | 51.4 | 68.1 | 123.6 | 598 | 547 | 49.7 | 36.2 | 3.28 s / 58.45 s / 1.57 s | 56 GiB |
| Talus v0.1.1 (256K context) | 51.1 | 66.6 | 122.7 | 555 | 512 | 48.2 | 36.4 | 3.51 s / 62.40 s / 1.85 s | 43 GiB |

### Qwen3.8-Flash-Next

Speeds in tokens per second. The depth prompts are about 1.8K and 30K tokens (Flash-Next's tokenizer).

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.2 | 60.9 | 73.8 | 144.8 | 1,100 | 1,486 | 64.5 | 65.6 | 1.69 s / 19.14 s / 936 ms | 51 GiB |
| Talus v0.1.1 | 62.5 | 74.0 | 117.9 | 1,092 | 1,480 | 65.4 | 63.9 | 1.71 s / 19.36 s / 1.10 s | 50 GiB |

### GLM-5.3-Flash

Speeds in tokens per second.

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.2 | 34.0 | 34.7 | 36.5 | 560 | 567 | 34.8 | 35.4 | 3.57 s / 56.54 s / 1.88 s | 9 GiB |
| Talus v0.1.1 | 33.9 | 34.6 | not measured | not measured | not measured | not measured | not measured | not measured / not measured / not measured | 8 GiB |

### MiMo-V2.6-Flash

Speeds in tokens per second.

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.2 | 38.3 | 49.1 | 64.2 | 732 | 661 | 45.3 | 42.3 | 2.74 s / 48.52 s / 986 ms | 15 GiB |
| Talus v0.1.1 | 38.3 | 49.2 | 64.4 | 739 | 665 | 45.4 | 42.4 | 2.67 s / 48.18 s / 1.19 s | 13 GiB |

### Laguna-S-2.1

Speeds in tokens per second. Both rows use the DFlash drafter (on by default from v0.1.2).

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.2 | 37.6 | 52.3 | 101.7 | 1,151 | 1,008 | 64.7 | 36.2 | 1.74 s / 31.79 s / 652 ms | 38 GiB |
| Talus v0.1.1 | 38.0 | 52.3 | 101.8 | 1,156 | 940 | 64.8 | 36.2 | 1.73 s / 34.05 s / 822 ms | 36 GiB |

## Quality gates (laptop)

Needle and the agentic smoke test ran on the laptop through the shipped `talus-server`, KLD on the laptop with the release binaries,
and each 45-minute soak on the machine named in its cell. For MiMo-V2.6-Flash, Laguna-S-2.1 and GLM-5.3-Flash the
needle, agentic smoke test and soak results come from the previous release candidate: their model code is unchanged and their
KLD logits are bitwise identical in both builds. Needle: five codenames spread through one prompt, three seeds.
Agentic smoke test: five fixed coding tasks run once through an agent harness. It catches broken tool use or editing, but cannot distinguish small quality differences (five pass/fail outcomes). Model quality is gated by KL divergence, greedy output identity and long-context recall. A task passes when the repository's tests pass.

| Model | Needle 60K | Needle 81K | Agentic smoke test | KLD | Soak |
|---|---|---|---|---|---|
| Qwen3.8-27B | 15/15 | not run | 5/5 | KL 0.00215, top-1 97.9% (gate <= 0.00291) | PASS: 152 requests over 45.0 min, 0 errors, greedy 102/102 (laptop) |
| Qwen3.8-Flash-Next | 15/15 | not run | 5/5 | KL 0.02422, top-1 90.7% (gate <= 0.0275) | PASS: 264 requests over 45.0 min, 0 errors, greedy 186/186 (desktop) |
| GLM-5.3-Flash | 15/15 | not run | 5/5 | KLD 0.04182, top-1 95.2% (gate <= 0.0440) | PASS: 104 requests over 45.0 min, 0 errors, greedy 66/66 (laptop) |
| MiMo-V2.6-Flash | not run | 15/15 | 5/5 | KLD 0.1721, top-1 89.1% (gate <= 0.182) | PASS: 144 requests over 45.0 min, 0 errors, greedy 98/98 (desktop) |
| Laguna-S-2.1 | not run | 15/15 | 5/5 | KL 0.00185 (@32K; @2K 0.000), top-1 98.5% (gate <= 0.003) | PASS: 200 requests over 45.0 min, 0 errors, greedy 140/140 (desktop) |

KLD thresholds the run is judged by:

| Model | KLD gate |
|---|---|
| Qwen3.8-27B | KL vs llama.cpp CPU reference (UD-IQ4_XS) <= 0.00291 (short prompts) and 4K-prompt KL ratio vs exact <= 1.35 |
| Qwen3.8-Flash-Next | first-token KL vs reference (43 prompts 4.5K-32K) <= 0.0275 (ref 0.02623) |
| GLM-5.3-Flash | KLD vs release logits <= 0.0440 (ref 0.04194); FP8 gate-text line <= 0.20 (ref 0.1934); FP8 code recorded |
| MiMo-V2.6-Flash | vs original weights: kld-text <= 0.182 (same-top >= 88.5%), wiki2 <= 0.240; vs old release ref: KLD <= 0.10, PPL ratio <= 1.01; teacher-forced KL(previous gated build -> this build) @2K and @32K: bitwise, or mean <= 1e-3 and top-1 >= 0.99 |
| Laguna-S-2.1 | teacher-forced KL(previous gated build -> this build) @2K and @32K: bitwise, or mean <= 3e-3 and top-1 >= 0.98 (served-speed numerics drift at 32K) |

<details>
<summary>Full KLD gate records</summary>

- **Qwen3.8-27B**: qwen27b kld_iq4xs talus      vs CPU: mean KL 0.00215 (decode rows 0.00215)  median 0.00011  p99 0.0184  top-1 0.9792  | vs talus: mean KL 0.00000  (576 rows)  [PASS]; qwen27b kld_q4kxl talus      vs CPU: mean KL 0.00358 (decode rows 0.00360)  median 0.00017  p99 0.0549  top-1 0.9809  | vs talus: mean KL 0.00000  (576 rows)  [PASS]; qwen27b kld_4k 4K-mid (6x4K+64) vs CPU: mean KL 0.005062 (exact 0.004364, ratio 1.160 <= 1.35, dKL 0.000902), top-1 vs CPU 96.61% (vs exact 98.70%) [PASS]
- **Qwen3.8-Flash-Next**: flashnext kld: 43 prompts, KL mean 0.02422 p99 0.08175 max 0.16080, top-1 agree 90.7%  [PASS]
- **GLM-5.3-Flash**: glm kld Mean KLD: 0.041819 ± 0.001553 Same top p: 95.181 ± 0.212 % count 10230  [PASS]; glm kld_fp8 Mean KLD: 0.196015 ± 0.005799 Same top p: 90.235 ± 0.294 % count 10230  [PASS]; glm kld_fp8_code Mean KLD: 0.258365 ± 0.007842 Same top p: 90.127 ± 0.295 % count 10230  [PASS]
- **MiMo-V2.6-Flash**: mimo kld_orig_kld_text Mean KLD: 0.172126 ± 0.004728 (<= 0.182) Same top p: 89.130 ± 0.308 % (>= 88.5%) Mean PPL(Q)/PPL(base): 1.126300 ± 0.008012; mimo kld_orig_wiki2 Mean KLD: 0.225455 ± 0.004879 (<= 0.240) Same top p: 82.502 ± 0.420 % Mean PPL(Q)/PPL(base): 1.211233 ± 0.010687 [PASS]; mimo kld_release Mean KLD: 0.082200 ± 0.002238 (<= 0.10) Same top p: 92.587 ± 0.236 % Mean PPL(Q)/PPL(base): 1.003625 ± 0.004404 (<= 1.01) [PASS]; mimo kl_d2000 vs the previous gated build: rows 129 V 152576 KL mean 0.000e+00 p50 0.000e+00 p99 0.000e+00 max 0.000e+00 top1 1.0000 bitwise-identical [PASS]; mimo kl_d32000 vs the previous gated build: rows 129 V 152576 KL mean 0.000e+00 p50 0.000e+00 p99 0.000e+00 max 0.000e+00 top1 1.0000 bitwise-identical [PASS]
- **Laguna-S-2.1**: laguna kl_d2000 vs the previous gated build: rows 129 V 100352 KL mean 0.000e+00 p50 0.000e+00 p99 0.000e+00 max 0.000e+00 top1 1.0000 bitwise-identical [PASS]; laguna kl_d32000 vs the previous gated build: rows 129 V 100352 KL mean 1.850e-03 p50 4.272e-05 p99 3.536e-02 max 4.405e-02 top1 0.9845 [PASS]

</details>

## Qwen3.8-27B: our 4-bit weights against BF16

The 27B ships with Unsloth's `UD-IQ4_XS` weights. This table compares it with `UD-Q4_K_XL` and with the full BF16
weights, all run in Talus v0.1.2 (the release candidate build of the same source) on the desktop
(GMKtec EVO-X2 (Ryzen AI Max+ 395, 128 GB), about 120 W). Quality: perplexity, and teacher-forced KL divergence and top-1 agreement against
BF16 (wiki: wikitext-2, first 30 x 2048-token chunks (scored rows 1024-2046), Talus BF16 as reference; code: code corpus, 26 x 2048-token chunks, Talus BF16 as reference). BF16 is an opt-in reference path (pass the
BF16 GGUF to `--model`): it shows what the quantization costs, it is not a way to serve the model.

| Weights | Size | PPL wiki | PPL code | KL wiki | KL code | top-1 wiki | top-1 code | chat | code | edit | prompt 2K | prompt 32K | decode 2K | decode 32K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 (reference, opt-in) | 54.7 GB | 6.1272 | 1.7200 | reference | reference | reference | reference | 14.6 | 19.4 | 17.8 | 574 | 506 | 4.1 | 4.0 |
| `UD-Q4_K_XL` (option) | 17.6 GB | 6.1194 | 1.7476 | 0.0061 | 0.0357 | 96.8 % | 95.8 % | 43.2 | 57.4 | 101.3 | 553 | 487 | 12.6 | 11.3 |
| `UD-IQ4_XS` (shipped default) | 14.3 GB (+2.2 GB W4A4 prompt sidecar) | 6.1541 | 1.7867 | 0.0159 | 0.0815 | 94.4 % | 93.5 % | 51.3 | 67.8 | 122.7 | 629 | 546 | 15.2 | 13.2 |

- PPL: lower is better. Sizes are the GGUF files.
- chat / code / edit: tok/s through the shipped `talus-server` with each format's run config, including speculative
  decoding. Prompt and decode: plain `talus-cli` benchmarks without speculation, median of three runs, so the decode
  columns are far below the served figures.
- BF16 chat / code / edit were measured with the MTP drafter (`STRATA_DRAFTER=mtp`). In v0.1.2 the automatic drafter
  choice picks DFlash2 on BF16 weights, which gives about 3.9 tok/s, close to plain decode; a fix is planned.
- Talus's BF16 against llama.cpp's BF16 on the same chunks: KL 0.00033 (wiki) / 0.00024 (code), PPL
  ratio 0.9997: the size of llama.cpp's own CPU-vs-GPU BF16 difference, so the BF16 reference is correct.
- `UD-Q4_K_XL`'s wiki perplexity is slightly below BF16's (PPL ratio 0.9987 ± 0.0008). That is
  within measurement noise (about 1.7 standard errors), not a gain over BF16; KL and top-1 measure how far each quant
  is from BF16.

## Comparison with other engines (laptop, v0.1.0)

These rows were not re-measured for v0.1.2. They come from the v0.1.0 release runs on the 70 W laptop: each model's
own freeze run (Talus rows) and reference runs of the other engines recorded on the same laptop on
2026-10-24. Compare them only with each other, not with the desktop tables above.

- The llama.cpp rows are **our patched build of llama.cpp ("halo4")** with the best configuration we found per
  model, not stock llama.cpp.
- `tg2k` / `tg32k` in these rows are four-prompt aggregates; `pp2k` / `pp32k` are the first prompt of each depth.

| Model | v0.1.0 freeze run |
|---|---|
| Qwen3.8-27B | 2026-10-28 |
| Qwen3.8-Flash-Next | 2026-10-28 |
| GLM-5.3-Flash | 2026-10-28 |
| MiMo-V2.6-Flash | 2026-10-28 |
| Laguna-S-2.1 | 2026-10-28 |

### Qwen3.8-27B

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.0 | 46.7 | 59.9 | 89.1 | 443 | 416 | 42.1 | 34.3 | 4.53 s / 77.04 s / 2.09 s | 66 GiB |
| Talus v0.1.0 (`UD-Q4_K_XL`, same quant as gufo) | 38.0 | 49.4 | 70.8 | 365 | 338 | 31.1 | 30.3 | 5.49 s / 94.82 s / 2.48 s | 63 GiB |
| Talus C++ (previous engine) | 36.7 | 44.6 | 62.3 | 352 | 313 | 35.1 | 30.5 | 5.70 s / 102.26 s / 1.18 s | 96 GiB |
| llama.cpp, our patched build (halo4) | 38.3 | 46.8 | 82.5 | 328 | 270 | 30.5 | 30.6 | 6.13 s / 118.61 s / 1.71 s | 100 GiB |
| gufo (`82c62b4`, `UD-Q4_K_XL` + DFlash2 `Q4_K_M`) | 31.6 | 41.9 | 60.9 | 491 | 452 | 31.2 | 24.0 | 4.11 s / 70.81 s / 1.43 s | 93 GiB |

### Qwen3.8-Flash-Next

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.0 | 58.9 | 69.9 | 108 | 970 | 1,262 | 61.6 | 66.1 | 1.95 s / 22.54 s / 1.09 s | 51 GiB |
| Talus C++ (previous engine) | 56.3 | 66.9 | 81.3 | 822 | 1,088 | 60.0 | 63.2 | 2.29 s / 26.14 s / 1.18 s | 51 GiB |
| gufo (`82c62b4`, `UD-Q4_K_XL` + shared-`Q8_0` MTP) | 53.1 | 63.1 | 99.3 | 1,333 | 1,387 | 54.8 | 55.6 | 1.40 s / 20.46 s / 482 ms | 35 GiB |

### GLM-5.3-Flash

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.0 | 32.4 | 33.1 | 34.6 | 452 | 465 | 33.9 | 33.2 | 4.44 s / 68.80 s / 2.24 s | 9 GiB |
| llama.cpp, our patched build (halo4) | 28.7 | 30.1 | 30.1 | 240 | 217 | 28.5 | 27.0 | 8.36 s / 147.57 s / 2.54 s | 8 GiB |

### MiMo-V2.6-Flash

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.0 | 35.8 | 43.0 | 58.9 | 597 | 526 | 35.7 | 32.8 | 3.36 s / 60.81 s / 1.15 s | 14 GiB |
| llama.cpp, our patched build (halo4) | 31.1 | 39.9 | 50.7 | 315 | 287 | 31.7 | 30.1 | 6.37 s / 111.42 s / 2.62 s | 20 GiB |

### Laguna-S-2.1

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.0 | 34.6 | 43.7 | 90.4 | 965 | 783 | 63.1 | 33.4 | 2.08 s / 40.88 s / 714 ms | 38 GiB |
| Talus C++ (previous engine) | 32.6 | 33.6 | 42.0 | 211 | 194 | 47.6 | 30.5 | 9.51 s / 155.00 s / 2.71 s | 41 GiB |
| llama.cpp, our patched build (halo4) | 36.6 | 44.6 | 79.7 | 604 | 514 | 34.5 | 30.4 | 3.33 s / 62.28 s / 1.43 s | 34 GiB |

## Reference engine conditions

Every reference row in the comparison tables was measured on the same 70 W HP ZBook Ultra G1a with the same HTTP
client and corpus:

- **llama.cpp (`halo4`)**: our patched build of llama.cpp (Vulkan + speculative-decoding patches) with the best
  configuration we found for each model, on the same GGUF files as Talus. On Laguna-S-2.1, `chat` runs with reasoning
  enabled and has similar DFlash draft acceptance on both engines, so per-token speed is bound by the 48-layer
  routed-expert memory reads at the 70 W package ceiling, whereas `code` and `edit` benefit from Talus's wider 9-token
  verify windows, long prompt copies and DFlash/lookup arbitration ([MODELS.md](MODELS.md#laguna-s-21)).
- **Talus C++**: the previous C++ host over the same kernel family, on the same weight files and arguments.
- **gufo** ([gufo-org/gufo](https://github.com/gufo-org/gufo), MIT, commit `82c62b4`, built from its Nix flake with
  ROCm 7.2.3 and run in a network-isolated container on the same machine in a `Talus control -> gufo -> Talus control`
  session): `gufo` does not load our `UD-IQ4_XS` files (`IQ3_XXS`/`IQ2_S`/`IQ2_XS`/`Q2_K` on Qwen3.8-27B and
  `IQ3_S`/`IQ4_NL`/`IQ4_XS` experts on Qwen3.8-Flash-Next are unsupported), so it ran on its documented `UD-Q4_K_XL`
  quants (`serve llm --context 65536`: 27B with `DFlash2 Q4_K_M` and 4 GiB RAM cache; Flash-Next with the shared-`Q8_0`
  MTP and 2 GiB RAM cache). On 27B we also ran Talus on the same `UD-Q4_K_XL` file (`Talus v0.1.0 (UD-Q4_K_XL)`) to
  separate the weight-size difference from the engines; on Flash-Next, running Talus on a pack built from `UD-Q4_K_XL`
  uses more memory and increases expert-cache traffic on 32K prompts and long-context decode while reading slightly
  faster on short cold prompts ([MODELS.md](MODELS.md#qwen38-flash-next)). Chat sampling defaults differ (`gufo` uses
  `top_k 20, min_p 0`; Talus uses `top_k 40, min_p 0.05`); `code`, `edit` and the depth cells are greedy. Talus's
  long-copy verify-window drafting ([MODELS.md](MODELS.md#laguna-s-21), `THIRD_PARTY_NOTICES.md` §6) was prompted by
  gufo's speculative-decoding notes.

Other Strix Halo figures published on the web mostly come from desktops and mini-PCs at 85-140 W with different
quantizations, prompts and warm-prefix "prompt at depth" definitions, so they are not directly comparable to cold
`pp32k` on a 70 W laptop.

## The records

The release tooling writes, per run: the release tarball's checksum and the binaries' checksums, the tier-1 records of
every cell (with every prompt's own figure), the needle, agentic, KLD and soak logs, and the values file this page is
filled from.
