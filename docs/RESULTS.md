# Results

This page is generated from the recorded release runs of Talus v0.1.3. The speed tables come from one
session on the desktop machine below, in which the v0.1.3 release candidate and the published v0.1.2 tarball ran
interleaved; the quality gates come mostly from the laptop. The quant-vs-BF16 table was measured with v0.1.2, and the
comparisons with other engines further down are older runs on the laptop with the v0.1.0 builds; each says so. A model
whose release run has not passed every gate shows "pending", and so does a figure that was not measured.
How each cell and gate is measured: [METHODOLOGY.md](METHODOLOGY.md).

| Release | |
|---|---|
| Talus | the v0.1.3 release candidate tarball (rc1), served by its own `talus-server` with the run configs `talus setup` writes. The published v0.1.3 tarball is the same engine source built from this repository (it differs in the speed table `talus setup` shows, in documentation, and in source comments, test fixtures and example paths cleaned of internal names); it gives identical greedy output on all five models |
| Speed runs | 2026-11-03, on the desktop (GMKtec EVO-X2: Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB, about 120 W sustained, BIOS set to its highest sustained power mode) |
| Comparison build | the published v0.1.2 tarball, in the same session on the same desktop: v0.1.2, then the v0.1.3 candidate, then v0.1.2 again (v0.1.2 = the mean of its two runs) |
| Quality gates | 2026-11-03, mostly on the laptop (HP ZBook Ultra G1a: Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB, 70 W firmware ceiling), on the v0.1.3 release candidate; "Quality gates" below says which machine each gate ran on |
| ROCm SDK | 10.2.0a20261014 |

The two machines have the same chip and memory but different power limits, so a figure from one is not comparable with
a figure from the other. Every table below names its machine and build.

| Model | Release run |
|---|---|
| Qwen3.8-27B | v013ab-qwen27b-t (2026-11-03) |
| Qwen3.8-Flash-Next | v013ab-flashnext-t (2026-11-03) |
| GLM-5.3-Flash | v013ab-glm-t (2026-11-03) |
| MiMo-V2.6-Flash | v013ab-mimo-t (2026-11-03) |
| Laguna-S-2.1 | v013ab-laguna-t (2026-11-03) |

Read the tables with these points in mind:

- Results inside a noise band are a tie (bands in [METHODOLOGY.md](METHODOLOGY.md#noise-bands)).
- `pp2k` / `pp32k` and `tg2k` / `tg32k` are tokens over time across 12 prompts at about 2K tokens and 8 prompts at
  about 32K tokens. The noise band we have for decode between engines whose generated text differs is the four-prompt
  one (10.5%); we have not derived one for the 12 / 8-prompt cells yet.
- Each model runs its shipped default configuration (see [MODELS.md](MODELS.md)), in both builds.
- The v0.1.1 figures and the 27B at 64K context are in the v0.1.2 release's RESULTS; they were not re-measured.

## What changed from v0.1.2 (desktop)

Change of each cell from the published v0.1.2 tarball to the v0.1.3 release candidate, same desktop, same session,
computed from the tables below. Speeds: positive is faster. `TTFT c` (the warm 32K follow-up's time to first token):
positive is slower.

| Model | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT c |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | −2.9 % | −1.8 % | −0.1 % | +0.5 % | −0.2 % | +1.8 % | +7.7 % | +1.9 % |
| Qwen3.8-Flash-Next | +4.4 % | +0.1 % | −0.3 % | +1.2 % | +0.2 % | −0.2 % | +0.8 % | −2.1 % |
| GLM-5.3-Flash | +0.3 % | +0.3 % | 0.0 % | +0.4 % | 0.0 % | 0.0 % | 0.0 % | +1.1 % |
| MiMo-V2.6-Flash | +4.2 % | 0.0 % | 0.0 % | +0.1 % | +0.3 % | 0.0 % | 0.0 % | −0.4 % |
| Laguna-S-2.1 | +16.3 % | +4.4 % | +20.9 % | +0.4 % | +0.4 % | +4.5 % | +1.4 % | −0.5 % |

Changes beyond the noise bands: Qwen3.8-27B decode at 32K (the int8 KV copy; faster on 7 of the 8 prompts); Laguna-S-2.1
chat, code, edit and decode at 2K (the faster expert pass for verify batches; chat also the sampled confident-prefix
verify); Qwen3.8-Flash-Next chat (draft depth from a measured cost table); MiMo-V2.6-Flash chat (sampled confident-prefix
verify). Everything else is inside its band. Qwen3.8-27B chat (−2.9 %) is inside its 3% band:
the int8 KV copy changes the 27B's text slightly, and on the sampled chat mix its drafts are then accepted a little
less often.

The 27B's decode at 2K is not a speed change: the prompts whose text did not change decode at the same speed in
both builds; the cell moves only with the prompts whose generated text changed.

## Speed (desktop)

### Qwen3.8-27B

Speeds in tokens per second. Shipped configuration: context chosen by memory (256K on 128 GB). From v0.1.3 the
decode and verify attention read an int8 copy of the KV cache, which costs memory (8.77 GiB at 256K; MemAvailable
column).

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.3 (release candidate 1), shipped, 256K context | 49.8 | 66.7 | 122.9 | 595 | 544 | 50.5 | 39.0 | 3.34 s / 58.89 s / 1.63 s | 27 GiB |
| Talus v0.1.2 (published tarball, same session), 256K context | 51.3 | 67.9 | 123.0 | 592 | 545 | 49.6 | 36.2 | 3.33 s / 58.77 s / 1.60 s | 36 GiB |

### Qwen3.8-Flash-Next

Speeds in tokens per second. The depth prompts are about 1.8K and 30K tokens (Flash-Next's tokenizer).

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.3 (release candidate 1) | 63.9 | 73.9 | 144.3 | 1,102 | 1,484 | 64.5 | 66.3 | 1.69 s / 19.17 s / 932 ms | 51 GiB |
| Talus v0.1.2 (published tarball, same session) | 61.2 | 73.8 | 144.7 | 1,089 | 1,481 | 64.6 | 65.8 | 1.71 s / 19.22 s / 952 ms | 51 GiB |

### GLM-5.3-Flash

Speeds in tokens per second.

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.3 (release candidate 1) | 34.1 | 34.7 | 36.6 | 561 | 567 | 34.8 | 35.4 | 3.55 s / 56.55 s / 1.89 s | 9 GiB |
| Talus v0.1.2 (published tarball, same session) | 34.0 | 34.6 | 36.6 | 559 | 567 | 34.8 | 35.4 | 3.56 s / 56.64 s / 1.87 s | 9 GiB |

### MiMo-V2.6-Flash

Speeds in tokens per second.

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.3 (release candidate 1) | 40.0 | 49.3 | 64.6 | 734 | 664 | 45.5 | 42.5 | 2.72 s / 48.20 s / 978 ms | 15 GiB |
| Talus v0.1.2 (published tarball, same session) | 38.4 | 49.3 | 64.6 | 733 | 662 | 45.5 | 42.5 | 2.72 s / 48.34 s / 982 ms | 14 GiB |

### Laguna-S-2.1

Speeds in tokens per second. Both rows use the DFlash drafter (on by default from v0.1.2).

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.3 (release candidate 1) | 44.2 | 54.6 | 122.4 | 1,154 | 1,009 | 67.7 | 36.7 | 1.73 s / 31.74 s / 648 ms | 38 GiB |
| Talus v0.1.2 (published tarball, same session) | 38.0 | 52.3 | 101.2 | 1,149 | 1,005 | 64.8 | 36.2 | 1.74 s / 31.86 s / 651 ms | 38 GiB |

## Quality gates (laptop, soaks on both machines)

All gates ran on the v0.1.3 release candidate. Needle and the agentic smoke test ran on the laptop through the shipped
`talus-server`, KLD on the laptop with the release binaries, and each 45-minute soak on the machine named in its cell.
Needle: five codenames spread through one prompt, three seeds.
Agentic smoke test: five fixed coding tasks run once through an agent harness. It catches broken tool use or editing, but cannot distinguish small quality differences (five pass/fail outcomes). Model quality is gated by KL divergence, greedy output identity and long-context recall. A task passes when the repository's tests pass.

| Model | Needle 60K | Needle 81K | Agentic smoke test | KLD | Soak |
|---|---|---|---|---|---|
| Qwen3.8-27B | 15/15 | not run | 5/5 | KL 0.00219, top-1 98.4% (gate <= 0.00291) | 144 requests over 45.0 min, 0 errors, greedy 91/96 (laptop): the misses are one prompt's cold vs prompt-cache-restored text (a split effect also in v0.1.2; accepted for v0.1.3, see below) |
| Qwen3.8-Flash-Next | 15/15 | not run | 5/5 | KL 0.02422, top-1 90.7% (gate <= 0.0275) | PASS: 272 requests over 45.0 min, 0 errors, greedy 192/192 (desktop) |
| GLM-5.3-Flash | 15/15 | not run | 5/5 | KLD 0.04182, top-1 95.2% (gate <= 0.0440) | PASS: 112 requests over 45.0 min, 0 errors, greedy 72/72 (desktop) |
| MiMo-V2.6-Flash | not run | 15/15 | 5/5 | KLD 0.1721, top-1 89.1% (gate <= 0.182) | PASS: 128 requests over 45.0 min, 0 errors, greedy 86/86 (laptop) |
| Laguna-S-2.1 | not run | 15/15 | 5/5 | KL 0.00185 (@32K; @2K 0.000), top-1 98.5% (gate <= 0.003) | PASS: 208 requests over 45.0 min, 0 errors, greedy 146/146 (desktop) |

KLD thresholds the run is judged by:

| Model | KLD gate |
|---|---|
| Qwen3.8-27B | KL vs llama.cpp CPU reference (UD-IQ4_XS) <= 0.00291 (short prompts) and 4K-prompt KL ratio vs exact <= 1.35 |
| Qwen3.8-Flash-Next | first-token KL vs reference (43 prompts 4.5K-32K) <= 0.0275 (ref 0.02623) |
| GLM-5.3-Flash | KLD vs release logits <= 0.0440 (ref 0.04194); FP8 gate-text line <= 0.20 (ref 0.1934); FP8 code recorded |
| MiMo-V2.6-Flash | vs original weights: kld-text <= 0.182 (same-top >= 88.5%), wiki2 <= 0.240; vs old release ref: KLD <= 0.10, PPL ratio <= 1.01; teacher-forced KL(previous gated build -> this build) @2K and @32K: bitwise, or mean <= 1e-3 and top-1 >= 0.99 |
| Laguna-S-2.1 | teacher-forced KL(previous gated build -> this build) @2K and @32K: bitwise, or mean <= 3e-3 and top-1 >= 0.98 (served-speed numerics drift at 32K) |

Qwen3.8-27B soak: 0 errors and flat memory, but 5 of its 96 greedy checks differ. All five are one edit prompt, whose
greedy text differs between a cold read and a read whose first 2,048 tokens come from the prompt cache. The same
split effect is in v0.1.2 (same prompt, same request order, also with the int8 KV copy off and with drafting off);
v0.1.2's soak happened to pass because its cold and cache-restored texts of that prompt matched. The maintainer
accepted the result for v0.1.3; see Known issues in the release notes. The other four greedy kinds of the soak were
all identical.

<details>
<summary>Full KLD gate records</summary>

- **Qwen3.8-27B**: qwen27b kld_iq4xs talus      vs CPU: mean KL 0.00219 (decode rows 0.00219)  median 0.00012  p99 0.0179  top-1 0.9844  | vs talus: mean KL 0.00000  (576 rows)  [PASS]; qwen27b kld_q4kxl talus      vs CPU: mean KL 0.00449 (decode rows 0.00452)  median 0.00018  p99 0.0778  top-1 0.9809  | vs talus: mean KL 0.00000  (576 rows)  [PASS]; qwen27b kld_4k 4K-mid (6x4K+64) vs CPU: mean KL 0.005172 (exact 0.004515, ratio 1.145 <= 1.35, dKL 0.000942), top-1 vs CPU 96.61% (vs exact 98.96%) [PASS]
- **Qwen3.8-Flash-Next**: flashnext kld: 43 prompts, KL mean 0.02422 p99 0.08175 max 0.16080, top-1 agree 90.7%  [PASS]
- **GLM-5.3-Flash**: glm kld Mean KLD: 0.041819 ± 0.001553 Same top p: 95.181 ± 0.212 % count 10230  [PASS]; glm kld_fp8 Mean KLD: 0.196015 ± 0.005799 Same top p: 90.235 ± 0.294 % count 10230  [PASS]; glm kld_fp8_code Mean KLD: 0.258365 ± 0.007842 Same top p: 90.127 ± 0.295 % count 10230  [PASS]
- **MiMo-V2.6-Flash**: mimo kld_orig_kld_text Mean KLD: 0.172126 ± 0.004728 (<= 0.182) Same top p: 89.130 ± 0.308 % (>= 88.5%) Mean PPL(Q)/PPL(base): 1.126300 ± 0.008012; mimo kld_orig_wiki2 Mean KLD: 0.225455 ± 0.004879 (<= 0.240) Same top p: 82.502 ± 0.420 % Mean PPL(Q)/PPL(base): 1.211233 ± 0.010687 [PASS]; mimo kld_release Mean KLD: 0.082200 ± 0.002238 (<= 0.10) Same top p: 92.587 ± 0.236 % Mean PPL(Q)/PPL(base): 1.003625 ± 0.004404 (<= 1.01) [PASS]; mimo kl_d2000 vs the previous gated build: rows 129 V 152576 KL mean 0.000e+00 p50 0.000e+00 p99 0.000e+00 max 0.000e+00 top1 1.0000 bitwise-identical [PASS]; mimo kl_d32000 vs the previous gated build: rows 129 V 152576 KL mean 0.000e+00 p50 0.000e+00 p99 0.000e+00 max 0.000e+00 top1 1.0000 bitwise-identical [PASS]
- **Laguna-S-2.1**: laguna kl_d2000 vs the previous gated build: rows 129 V 100352 KL mean 0.000e+00 p50 0.000e+00 p99 0.000e+00 max 0.000e+00 top1 1.0000 bitwise-identical [PASS]; laguna kl_d32000 vs the previous gated build: rows 129 V 100352 KL mean 1.850e-03 p50 4.272e-05 p99 3.536e-02 max 4.405e-02 top1 0.9845 [PASS]

</details>

## Qwen3.8-27B: our 4-bit weights against BF16 (v0.1.2)

Measured with **v0.1.2** and not re-measured for v0.1.3. In v0.1.3 the quantized 27B's decode and verify attention read
an int8 copy of the KV cache, so its served speeds and decode at 32K differ from these (see the tables above); the
weights and the prompt path are unchanged. The 27B ships with Unsloth's `UD-IQ4_XS` weights. This table compares it with
`UD-Q4_K_XL` and with the full BF16 weights, all run in Talus v0.1.2 (its release candidate 2) on the desktop
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
- BF16 chat / code / edit were measured with the MTP drafter (`STRATA_DRAFTER=mtp`). In v0.1.2 (unchanged in v0.1.3) the automatic drafter
  choice picks DFlash2 on BF16 weights, which gave about 3.9 tok/s in v0.1.2, close to plain decode; a fix is planned.
- Talus's BF16 against llama.cpp's BF16 on the same chunks: KL 0.00033 (wiki) / 0.00024 (code), PPL
  ratio 0.9997: the size of llama.cpp's own CPU-vs-GPU BF16 difference, so the BF16 reference is correct.
- `UD-Q4_K_XL`'s wiki perplexity is slightly below BF16's (PPL ratio 0.9987 ± 0.0008). That is
  within measurement noise (about 1.7 standard errors), not a gain over BF16; KL and top-1 measure how far each quant
  is from BF16.

## Comparison with other engines (laptop, v0.1.0)

These rows were not re-measured for v0.1.2 or v0.1.3. They come from the v0.1.0 release runs on the 70 W laptop: each model's
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
