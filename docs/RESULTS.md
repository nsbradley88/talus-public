# Results

This page is generated from the recorded release runs of Talus. Each model's figures come from its own freeze run
(table below), and the comparison rows from reference runs recorded on the same machine on 2026-10-24. A model whose
release run has not passed every gate shows "pending", and so does a reference engine that was not re-measured. How each cell and gate is measured: [METHODOLOGY.md](METHODOLOGY.md).

| Release | |
|---|---|
| Runs | one freeze run per model (below); reference runs 2026-10-24 |
| Talus commit | 43c94474c43d |
| ROCm SDK | 10.2.0a20261014 |
| Machine | HP ZBook Ultra G1a, Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB, 70 W firmware ceiling |

Each model was frozen on its own release run as its changes landed; the release commit `43c94474c43d`
incorporates every freeze commit plus later changes that were verified greedy byte-identical on that model
(`code` + `edit` greedy prompts; the cross-check runs are in the last column):

| Model | Freeze run | Commit | Date | Cross-check on `43c94474c43d` |
|---|---|---|---|---|
| Qwen3.8-27B | `rel-final-43c9` | `43c94474c43d` | 2026-10-28 | Frozen directly on `43c94474c43d` (`9/9` greedy match vs `96221159`) |
| Qwen3.8-Flash-Next | `rel-final-e6cf` | `e6cf959e1cad` | 2026-10-28 | Frozen on `e6cf959e1cad` (model code unchanged in `43c94474c43d`; `9/9` greedy match vs `0e0a5ffe`) |
| GLM-5.3-Flash | `rel-final-e6cf` | `e6cf959e1cad` | 2026-10-28 | Frozen on `e6cf959e1cad` (model code unchanged in `43c94474c43d`) |
| MiMo-V2.6-Flash | `rel-final-e6cf` | `e6cf959e1cad` | 2026-10-28 | Frozen on `e6cf959e1cad` (model code unchanged in `43c94474c43d`; `9/9` greedy match vs `0e0a5ffe`) |
| Laguna-S-2.1 | `rel-final-e6cf` | `e6cf959e1cad` | 2026-10-28 | Frozen on `e6cf959e1cad` (model code unchanged in `43c94474c43d`; `9/9` greedy match vs `0e0a5ffe`) |

Read the tables with these points in mind:

- Results inside a noise band are a tie (bands in [METHODOLOGY.md](METHODOLOGY.md#noise-bands)).
- `tg2k` / `tg32k` are four-prompt aggregates. Between engines whose generated text differs, a difference smaller
  than the text noise (10.5%) is not significant.
- The llama.cpp rows are **our patched build of llama.cpp ("halo4")** with the best configuration we found per
  model, not stock llama.cpp.
- Each model runs its shipped default configuration (see [MODELS.md](MODELS.md)).

## Speed

### Qwen3.8-27B

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus | 46.7 | 59.9 | 89.1 | 443 | 416 | 42.1 | 34.3 | 4.53 s / 77.04 s / 2.09 s | 66 GiB |
| Talus (`UD-Q4_K_XL`, same quant as gufo) | 38.0 | 49.4 | 70.8 | 365 | 338 | 31.1 | 30.3 | 5.49 s / 94.82 s / 2.48 s | 63 GiB |
| Talus C++ (previous engine) | 36.7 | 44.6 | 62.3 | 352 | 313 | 35.1 | 30.5 | 5.70 s / 102.26 s / 1.18 s | 96 GiB |
| llama.cpp, our patched build (halo4) | 38.3 | 46.8 | 82.5 | 328 | 270 | 30.5 | 30.6 | 6.13 s / 118.61 s / 1.71 s | 100 GiB |
| gufo (`82c62b4`, `UD-Q4_K_XL` + DFlash2 `Q4_K_M`) | 31.6 | 41.9 | 60.9 | 491 | 452 | 31.2 | 24.0 | 4.11 s / 70.81 s / 1.43 s | 93 GiB |

### Qwen3.8-Flash-Next

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus | 58.9 | 69.9 | 108 | 970 | 1,262 | 61.6 | 66.1 | 1.95 s / 22.54 s / 1.09 s | 51 GiB |
| Talus C++ (previous engine) | 56.3 | 66.9 | 81.3 | 822 | 1,088 | 60.0 | 63.2 | 2.29 s / 26.14 s / 1.18 s | 51 GiB |
| gufo (`82c62b4`, `UD-Q4_K_XL` + shared-`Q8_0` MTP) | 53.1 | 63.1 | 99.3 | 1,333 | 1,387 | 54.8 | 55.6 | 1.40 s / 20.46 s / 482 ms | 35 GiB |

### GLM-5.3-Flash

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus | 32.4 | 33.1 | 34.6 | 452 | 465 | 33.9 | 33.2 | 4.44 s / 68.80 s / 2.24 s | 9 GiB |
| llama.cpp, our patched build (halo4) | 28.7 | 30.1 | 30.1 | 240 | 217 | 28.5 | 27.0 | 8.36 s / 147.57 s / 2.54 s | 8 GiB |

### MiMo-V2.6-Flash

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus | 35.8 | 43.0 | 58.9 | 597 | 526 | 35.7 | 32.8 | 3.36 s / 60.81 s / 1.15 s | 14 GiB |
| llama.cpp, our patched build (halo4) | 31.1 | 39.9 | 50.7 | 315 | 287 | 31.7 | 30.1 | 6.37 s / 111.42 s / 2.62 s | 20 GiB |

### Laguna-S-2.1

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus | 34.6 | 43.7 | 90.4 | 965 | 783 | 63.1 | 33.4 | 2.08 s / 40.88 s / 714 ms | 38 GiB |
| Talus C++ (previous engine) | 32.6 | 33.6 | 42.0 | 211 | 194 | 47.6 | 30.5 | 9.51 s / 155.00 s / 2.71 s | 41 GiB |
| llama.cpp, our patched build (halo4) | 36.6 | 44.6 | 79.7 | 604 | 514 | 34.5 | 30.4 | 3.33 s / 62.28 s / 1.43 s | 34 GiB |
## Quality gates

| Model | Needle 60K | Needle 81K | Agentic | KLD | Soak |
|---|---|---|---|---|---|
| Qwen3.8-27B | 15/15 | n/a (64K-context config) | 5/5 | KL 0.00215, top-1 97.9% (gate <= 0.00291) | PASS: 120 requests over 45.0 min, 0 errors |
| Qwen3.8-Flash-Next | 15/15 | 15/15 | 5/5 | KL 0.02422, top-1 90.7% (gate <= 0.0275) | PASS: 200 requests over 45.0 min, 0 errors |
| GLM-5.3-Flash | 15/15 | n/a (64K-context config) | 5/5 | KLD 0.04182, top-1 95.2% (gate <= 0.0440) | PASS: 96 requests over 45.0 min, 0 errors |
| MiMo-V2.6-Flash | 15/15 | 15/15 | 5/5 | KLD 0.1721, top-1 89.1% (gate <= 0.182) | PASS: 120 requests over 45.0 min, 0 errors |
| Laguna-S-2.1 | 15/15 | 15/15 | 5/5 | KL 0.00185 (@32K; @2K 0.000), top-1 98.5% (gate <= 0.003) | PASS: 168 requests over 45.0 min, 0 errors |

KLD thresholds the run was judged by:

| Model | KLD gate |
|---|---|
| Qwen3.8-27B | KL vs llama.cpp CPU reference (UD-IQ4_XS) <= 0.00291 (short prompts) and 4K-prompt KL ratio vs exact <= 1.35 |
| Qwen3.8-Flash-Next | first-token KL vs reference (43 prompts 4.5K-32K) <= 0.0275 (ref 0.02623) |
| GLM-5.3-Flash | KLD vs release logits <= 0.0440 (ref 0.04194); FP8 gate-text line <= 0.20 (ref 0.1934); FP8 code recorded |
| MiMo-V2.6-Flash | vs original weights: kld-text <= 0.182 (same-top >= 88.5%), wiki2 <= 0.240; vs old release ref: KLD <= 0.10, PPL ratio <= 1.01; teacher-forced KL(previous gated build -> this build) @2K and @32K: bitwise, or mean <= 1e-3 and top-1 >= 0.99 |
| Laguna-S-2.1 | teacher-forced KL(previous gated build -> this build) @2K and @32K: bitwise, or mean <= 3e-3 and top-1 >= 0.98 (served-speed numerics drift at 32K) |

<details>
<summary>Full KLD gate records</summary>

- **Qwen3.8-27B**: qwen27b kld_iq4xs talus      vs CPU: mean KL 0.00215 (decode rows 0.00215)  median 0.00011  p99 0.0184  top-1 0.9792, vs talus: mean KL 0.00000  (576 rows)  [PASS]; qwen27b kld_q4kxl talus      vs CPU: mean KL 0.00358 (decode rows 0.00360)  median 0.00017  p99 0.0549  top-1 0.9809, vs talus: mean KL 0.00000  (576 rows)  [PASS]; qwen27b kld_4k 4K-mid (6x4K+64) vs CPU: mean KL 0.004364 (exact 0.004364, ratio 1.000 <= 1.35, dKL 0.000000), top-1 vs CPU 96.88% (vs exact 100.00%) [PASS]
- **Qwen3.8-Flash-Next**: flashnext 43 prompts, KL mean 0.02422 p99 0.08175 max 0.16080, top-1 agree 90.7%  [PASS]
- **GLM-5.3-Flash**: glm kld Mean KLD: 0.041819 ± 0.001553 Same top p: 95.181 ± 0.212 % count 10230  [PASS]; glm kld_fp8 Mean KLD: 0.196015 ± 0.005799 Same top p: 90.235 ± 0.294 % count 10230  [PASS]; glm kld_fp8_code Mean KLD: 0.258365 ± 0.007842 Same top p: 90.127 ± 0.295 % count 10230  [PASS]
- **MiMo-V2.6-Flash**: mimo kld_orig_kld_text Mean KLD: 0.172126 ± 0.004728 (<= 0.182) Same top p: 89.130 ± 0.308 % (>= 88.5%) Mean PPL(Q)/PPL(base): 1.126300 ± 0.008012; mimo kld_orig_wiki2 Mean KLD: 0.225455 ± 0.004879 (<= 0.240) Same top p: 82.502 ± 0.420 % Mean PPL(Q)/PPL(base): 1.211233 ± 0.010687 [PASS]; mimo kld_release Mean KLD: 0.082200 ± 0.002238 (<= 0.10) Same top p: 92.587 ± 0.236 % Mean PPL(Q)/PPL(base): 1.003625 ± 0.004404 (<= 1.01) [PASS]; mimo kl_d2000 vs the previous gated build: rows 129 V 152576 KL mean 0.000e+00 p50 0.000e+00 p99 0.000e+00 max 0.000e+00 top1 1.0000 bitwise-identical [PASS]; mimo kl_d32000 vs the previous gated build: rows 129 V 152576 KL mean 0.000e+00 p50 0.000e+00 p99 0.000e+00 max 0.000e+00 top1 1.0000 bitwise-identical [PASS]
- **Laguna-S-2.1**: laguna kl_d2000 vs the previous gated build: rows 129 V 100352 KL mean 0.000e+00 p50 0.000e+00 p99 0.000e+00 max 0.000e+00 top1 1.0000 bitwise-identical [PASS]; laguna kl_d32000 vs the previous gated build: rows 129 V 100352 KL mean 1.850e-03 p50 4.272e-05 p99 3.536e-02 max 4.405e-02 top1 0.9845 [PASS]

</details>

## Reference engine conditions

Every reference row in the tables above was measured on the same 70 W HP ZBook Ultra G1a with the same HTTP client
and corpus:

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
  MTP and 2 GiB RAM cache). On 27B we also ran Talus on the same `UD-Q4_K_XL` file (`Talus (UD-Q4_K_XL)`) to separate
  the weight-size difference from the engines; on Flash-Next, running Talus on a pack built from `UD-Q4_K_XL` uses
  more memory and increases expert-cache traffic on 32K prompts and long-context decode while reading slightly faster
  on short cold prompts ([MODELS.md](MODELS.md#qwen38-flash-next)). Chat sampling defaults differ (`gufo` uses
  `top_k 20, min_p 0`; Talus uses `top_k 40, min_p 0.05`); `code`, `edit` and the depth cells are greedy. Talus's
  long-copy verify-window drafting ([MODELS.md](MODELS.md#laguna-s-21), `THIRD_PARTY_NOTICES.md` §6) was prompted by
  gufo's speculative-decoding notes.

Other Strix Halo figures published on the web mostly come from desktops and mini-PCs at 85-140 W with different
quantizations, prompts and warm-prefix "prompt at depth" definitions, so they are not directly comparable to cold
`pp32k` on a 70 W laptop.

## The records

The release tooling writes, per run: the build manifest (commit and the checksum of every binary), the tier-1
records of every cell, the needle, agentic, KLD and soak logs, and the values file this page is filled from.
