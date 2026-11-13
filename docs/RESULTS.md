# Results

This page is generated from the recorded release runs of Talus v0.1.9. The speed tables come from one
session on the desktop machine below, in which the v0.1.9 release candidate and v0.1.8 ran
interleaved; the quality gates come mostly from the laptop. The quant-vs-BF16 table was measured with v0.1.2, the
desktop comparison with other engines with v0.1.3, and the laptop comparison with the v0.1.0 builds; each says so. A model
whose release run has not passed every gate shows "pending", and so does a figure that was not measured.
How each cell and gate is measured: [METHODOLOGY.md](METHODOLOGY.md).

| Release | |
|---|---|
| Talus | the v0.1.9 release candidate 1 tarball (rc1), served by its own `talus-server` with the run configs `talus setup` writes. The published v0.1.9 tarball is the same engine source built from this repository (it differs in the speed table `talus setup` shows, in documentation, and in source comments, test fixtures and example paths cleaned of internal names); it gives identical greedy output on all five models. |
| Speed runs | 2026-11-12 to 2026-11-13, on the desktop (GMKtec EVO-X2: Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB, about 120 W sustained, BIOS set to its highest sustained power mode; all 16 CPU cores online) |
| Comparison build | v0.1.8: the tarball of its last release candidate, from which the published v0.1.8 was cut (same engine source, GPU kernels byte-identical), in the same session on the same desktop, alternating: the v0.1.9 candidate, v0.1.8, the candidate, v0.1.8 (each = the mean of its two runs). Before each run every weight file was read in full and checked against its pinned SHA-256 |
| Quality gates | 2026-11-12 to 2026-11-13, mostly on the laptop (HP ZBook Ultra G1a: Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB, 70 W firmware ceiling), on the v0.1.9 release candidate 1; "Quality gates" below says which machine each gate ran on |
| ROCm SDK | 10.2.0a20261014 |

The two machines have the same chip and memory but different power limits, so a figure from one is not comparable with
a figure from the other. Every table below names its machine and build.

| Model | Release run |
|---|---|
| Qwen3.8-27B | v019sp-qwen27b-t1 (2026-11-12) |
| Qwen3.8-Flash-Next | v019sp-flashnext-t1 (2026-11-12) |
| GLM-5.3-Flash | v019sp-glm-t1 (2026-11-13) |
| MiMo-V2.6-Flash | v019sp-mimo-t1 (2026-11-13) |
| Laguna-S-2.1 | v019sp-laguna-t1 (2026-11-13) |

Read the tables with these points in mind:

- Results inside a noise band are a tie (bands in [METHODOLOGY.md](METHODOLOGY.md#noise-bands)).
- `pp2k` / `pp32k` and `tg2k` / `tg32k` are tokens over time across 12 prompts at about 2K tokens and 8 prompts at
  about 32K tokens. The noise band we have for decode between engines whose generated text differs is the four-prompt
  one (10.5%); we have not derived one for the 12 / 8-prompt cells yet.
- Each model runs its shipped default configuration (see [MODELS.md](MODELS.md)), in both builds.
- Older release-to-release tables (v0.1.2 to v0.1.3 and earlier) are in those releases' RESULTS; they were not re-measured.

## What changed from v0.1.8 (desktop)

Change of each cell from v0.1.8 to the v0.1.9 release candidate, same desktop, same session,
computed from the tables below. Speeds: positive is faster. `TTFT c` (the warm 32K follow-up's time to first token):
positive is slower.

| Model | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT c |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | 0.0 % | 0.0 % | 0.0 % | −0.5 % | +1.3 % | 0.0 % | 0.0 % | −1.3 % |
| Qwen3.8-Flash-Next | −1.0 % | 0.0 % | −0.4 % | +0.1 % | +0.1 % | −0.2 % | 0.0 % | +1.5 % |
| GLM-5.3-Flash | +1.5 % | −0.3 % | −0.3 % | +0.3 % | +0.3 % | −0.3 % | 0.0 % | 0.0 % |
| MiMo-V2.6-Flash | +0.2 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | −0.4 % |
| Laguna-S-2.1 | −0.9 % | 0.0 % | −0.1 % | +0.4 % | +2.8 % | −0.1 % | −0.3 % | −2.6 % |

Every change is inside its noise band.

## Speed (desktop)

### Qwen3.8-27B

Speeds in tokens per second. Shipped configuration: context chosen by memory (256K on 128 GB). From v0.1.3 the
decode and verify attention read an int8 copy of the KV cache, which costs memory (8.77 GiB at 256K; MemAvailable
column).

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.9 (release candidate 1), shipped, 256K context | 51.2 | 66.2 | 133.2 | 598 | 554 | 50.2 | 38.6 | 3.58 s / 57.75 s / 1.56 s | 31 GiB |
| Talus v0.1.8 (same session), 256K context | 51.2 | 66.2 | 133.2 | 601 | 547 | 50.2 | 38.6 | 3.59 s / 58.54 s / 1.58 s | 31 GiB |

### Qwen3.8-Flash-Next

Speeds in tokens per second. The depth prompts are about 1.8K and 30K tokens (Flash-Next's tokenizer).

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.9 (release candidate 1) | 62.4 | 73.6 | 162.2 | 1,323 | 1,738 | 66.5 | 65.5 | 1.43 s / 16.40 s / 658 ms | 42 GiB |
| Talus v0.1.8 (same session) | 63.0 | 73.6 | 162.9 | 1,322 | 1,736 | 66.6 | 65.5 | 1.43 s / 16.44 s / 648 ms | 42 GiB |

### GLM-5.3-Flash

Speeds in tokens per second.

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.9 (release candidate 1) | 33.4 | 34.8 | 35.5 | 577 | 584 | 33.9 | 35.0 | 3.45 s / 54.97 s / 1.85 s | 8 GiB |
| Talus v0.1.8 (same session) | 32.9 | 34.9 | 35.6 | 575 | 582 | 34.0 | 35.0 | 3.44 s / 55.17 s / 1.85 s | 8 GiB |

### MiMo-V2.6-Flash

Speeds in tokens per second.

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.9 (release candidate 1) | 40.8 | 49.8 | 65.1 | 736 | 662 | 46.0 | 42.8 | 2.69 s / 48.36 s / 986 ms | 15 GiB |
| Talus v0.1.8 (same session) | 40.7 | 49.8 | 65.1 | 736 | 662 | 46.0 | 42.8 | 2.69 s / 48.35 s / 990 ms | 15 GiB |

### Laguna-S-2.1

Speeds in tokens per second. Both rows use the DFlash drafter (on by default from v0.1.2).

| Build | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.9 (release candidate 1) | 43.6 | 54.6 | 123.8 | 1,160 | 1,034 | 68.0 | 36.8 | 1.72 s / 30.97 s / 633 ms | 36 GiB |
| Talus v0.1.8 (same session) | 44.0 | 54.6 | 123.9 | 1,155 | 1,006 | 68.1 | 36.9 | 1.73 s / 31.83 s / 650 ms | 36 GiB |

## Quality gates (laptop, soaks on both machines)

Every gate ran on release candidate 1, the build this release was cut from. MiMo-V2.6-Flash: logit-row, row-parity and drafts gates ran on candidate 1; its needle, agentic and soak results are v0.1.6's. Laguna-S-2.1: logit-row, row-parity, drafts and soak ran on candidate 1 (its prompt attention is new in this release); its needle and agentic results are v0.1.6's. Needle and the agentic smoke test ran on the laptop through the shipped `talus-server`,
the logit-row, row-parity and drafts gates on the laptop with the release binaries, and each 45-minute soak on the
machine named in its cell.
Needle: five codenames spread through one prompt, three seeds.
Agentic smoke test: five fixed coding tasks run once through an agent harness. It catches broken tool use or editing, but cannot distinguish small quality differences (five pass/fail outcomes). Model quality is gated by KL divergence, greedy output identity and long-context recall. A task passes when the repository's tests pass.

| Model | Needle 60K | Needle 81K | Agentic smoke test | Logit rows / KLD | Row parity | Drafts on vs off | Soak |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | 15/15 | not run | 5/5 | rows bitwise vs frozen; KL vs CPU reference 0.00219, top-1 98.4% (not gated) | 14/14 identical | 6/6 identical | PASS: 168 requests over 45.0 min, 0 errors, greedy 113/113 (desktop) |
| Qwen3.8-Flash-Next | 15/15 | not run | 5/5 | rows bitwise vs frozen (43/43); KL vs BF16 0.09108, top-1 86.0% (bar for a changed build: <= 0.1064, top-1 >= 79%) | 16/16 identical (widths 8 to 32) | 5/5 identical | PASS: 288 requests over 45.0 min, 0 errors, greedy 203/203 (desktop) |
| GLM-5.3-Flash | 15/15 | not run | 5/5 | rows bitwise vs frozen (2/2); KLD vs release logits 0.04174, top-1 95.0% (not gated) | 6/6 identical | 9/9 identical | PASS: 112 requests over 45.0 min, 0 errors, greedy 71/71 (desktop) |
| MiMo-V2.6-Flash | not run | 15/15 | 5/5 | engine rows bitwise vs a fixed earlier Talus build at 2K and 32K; weights check vs original weights 0.1721 | 11/11 identical | 5/5 identical | PASS: 152 requests over 45.0 min, 0 errors, greedy 101/101 (desktop) |
| Laguna-S-2.1 | not run | 15/15 | 5/5 | served rows bitwise vs frozen at 2K and 32K | 2/2 identical | 5/5 identical | PASS: 216 requests over 45.0 min, 0 errors, greedy 149/149 (desktop) |

KLD thresholds the run is judged by:

| Model | KLD gate |
|---|---|
| Qwen3.8-27B | Teacher-forced logit rows on short prompts (UD-IQ4_XS, UD-Q4_K_XL) bitwise equal to a frozen set; any change goes to human review and does not pass by itself. KL vs the llama.cpp CPU reference and the 4K-prompt ratio are recorded, not gated |
| Qwen3.8-Flash-Next | First-token rows (43 prompts 4.5K-32K) bitwise equal to a frozen set. A changed build also needs KL vs llama.cpp CPU on BF16 <= 0.1064 and top-1 >= 79%, then human review |
| GLM-5.3-Flash | KLD vs release logits <= 0.0440 (ref 0.04194); FP8 gate-text line <= 0.20 (ref 0.1934); FP8 code recorded |
| MiMo-V2.6-Flash | Engine: teacher-forced KL(a fixed earlier Talus build -> this build) @2K and @32K: bitwise, or mean <= 1e-3 and top-1 >= 0.99. Weights checks, not engine gates (llama.cpp on the GGUF, the build does not run): vs original weights kld-text <= 0.182 (same-top >= 88.5%), wiki2 <= 0.240; vs old release ref: KLD <= 0.10, PPL ratio <= 1.01 |
| Laguna-S-2.1 | Served rows @2K and @32K bitwise equal to a frozen set; a change needs KL vs frozen <= the rounding floor (8.2e-3 @2K, 1.4e-3 @32K), top-1 >= 0.98 and human review. KL(a fixed earlier Talus build -> this build) recorded, not gated |

Flash-Next's logit rows are bit for bit the ones frozen from v0.1.7, with its short-chunk prompt path (from v0.1.8) on and off, so its KL to the BF16 weights is v0.1.7's: 0.0911 (top-1 86.0 %) on the gate path (prompt cache off), 0.0943 (top-1 81.4 %) on the chat path (prompt cache on, the end of the prompt read through a verify window, as a served chat request is). MiMo's three llama.cpp lines are weights checks, not engine gates. See "KLD references per model" in [METHODOLOGY.md](METHODOLOGY.md).

Every soak ran without errors and every greedy check matched. The soak memory rule now ignores memory steps of 2 MiB or less (page-cache noise); under it Qwen3.8-Flash-Next's 45-minute soak, which failed the old rule in v0.1.7 and v0.1.8, passes.

<details>
<summary>Full KLD gate records</summary>

- **Qwen3.8-27B**: qwen27b kld_iq4xs (not gated): talus      vs CPU: mean KL 0.00219 (decode rows 0.00219)  median 0.00012  p99 0.0179  top-1 0.9844, vs talus: mean KL 0.00000  (576 rows); qwen27b rows_iq4xs vs frozen: bitwise-identical 6/6 [PASS]; qwen27b kld_q4kxl (not gated): talus      vs CPU: mean KL 0.00449 (decode rows 0.00452)  median 0.00018  p99 0.0778  top-1 0.9809, vs talus: mean KL 0.00000  (576 rows); qwen27b rows_q4kxl vs frozen: bitwise-identical 6/6 [PASS]; qwen27b kld_4k (not gated): 4K-mid (6x4K+64) vs CPU: mean KL 0.005172 (exact 0.004515, ratio 1.145, dKL 0.000942), top-1 vs CPU 96.61% (vs exact 98.96%)
- **Qwen3.8-Flash-Next**: flashnext 43 prompts, KL mean 0.09108 p99 0.39449 max 0.68699, top-1 agree 86.0%  rows vs frozen: gate settings bitwise-identical 43/43; shipped settings bitwise-identical 43/43 [PASS]; flashnext _ship vs BF16 (not gated): -ship: 43 prompts, KL mean 0.09108 p99 0.39449 max 0.68699, top-1 agree 86.0%; flashnext _old vs fnp/res/kld/ref (not gated, gate until v0.1.5): 43 prompts, KL mean 0.05285 p99 0.19578 max 0.92915, top-1 agree 81.4%
- **GLM-5.3-Flash**: glm rows vs frozen: bitwise-identical 2/2 [PASS]; glm kld (not gated): Mean KLD: 0.041744 ± 0.001575 Same top p: 95.044 ± 0.215 % count 10230; glm kld_fp8 (not gated): Mean KLD: 0.194853 ± 0.005785 Same top p: 90.078 ± 0.296 % count 10230; glm kld_fp8_code Mean KLD: 0.258527 ± 0.007847 Same top p: 90.264 ± 0.293 % count 10230  [RECORDED]
- **MiMo-V2.6-Flash**: mimo kld_orig_kld_text weights check (llama.cpp on the GGUF, the build under test does not run): Mean KLD: 0.172126 ± 0.004728 (<= 0.182) Same top p: 89.130 ± 0.308 % (>= 88.5%) Mean PPL(Q)/PPL(base): 1.126300 ± 0.008012 [PASS]; mimo kld_orig_wiki2 weights check (llama.cpp on the GGUF, the build under test does not run): Mean KLD: 0.225455 ± 0.004879 (<= 0.240) Same top p: 82.502 ± 0.420 % Mean PPL(Q)/PPL(base): 1.211233 ± 0.010687 [PASS]; mimo kld_release weights check (llama.cpp on the GGUF, the build under test does not run): Mean KLD: 0.082200 ± 0.002238 (<= 0.10) Same top p: 92.587 ± 0.236 % Mean PPL(Q)/PPL(base): 1.003625 ± 0.004404 (<= 1.01) [PASS]; mimo kl_d2000 vs a fixed earlier Talus build: rows 129 V 152576 KL mean 0.000e+00 p50 0.000e+00 p99 0.000e+00 max 0.000e+00 top1 1.0000 bitwise-identical [PASS]; mimo kl_d32000 vs a fixed earlier Talus build: rows 129 V 152576 KL mean 0.000e+00 p50 0.000e+00 p99 0.000e+00 max 0.000e+00 top1 1.0000 bitwise-identical [PASS]
- **Laguna-S-2.1**: laguna kl_d2000 served-vs-a fixed earlier Talus build (not gated): rows 129 V 100352 KL mean 9.149e-03 p50 1.146e-03 p99 8.448e-02 max 2.576e-01 top1 0.9845; laguna kl_d2000 decode-vs-a fixed earlier Talus build (not gated): rows 129 V 100352 KL mean 0.000e+00 p50 0.000e+00 p99 0.000e+00 max 0.000e+00 top1 1.0000; laguna kl_d2000 served-rows vs frozen: bitwise-identical [PASS]; laguna kl_d32000 served-vs-a fixed earlier Talus build (not gated): rows 129 V 100352 KL mean 1.923e-03 p50 4.231e-05 p99 2.940e-02 max 3.897e-02 top1 1.0000; laguna kl_d32000 decode-vs-a fixed earlier Talus build (not gated): rows 129 V 100352 KL mean 1.850e-03 p50 4.272e-05 p99 3.536e-02 max 4.405e-02 top1 0.9845; laguna kl_d32000 served-rows vs frozen: bitwise-identical [PASS]

</details>

## Qwen3.8-27B: our 4-bit weights against BF16 (v0.1.2)

Measured with **v0.1.2** and not re-measured since, except the `UD-IQ4_XS` quality cells (below). In v0.1.3 the quantized 27B's decode and verify attention read
an int8 copy of the KV cache, so its served speeds and decode at 32K differ from these (see the tables above); the
weights are unchanged. The 27B ships with Unsloth's `UD-IQ4_XS` weights. This table compares it with
`UD-Q4_K_XL` and with the full BF16 weights, all run in Talus v0.1.2 (its release candidate 2) on the desktop
(GMKtec EVO-X2 (Ryzen AI Max+ 395, 128 GB), about 120 W). Quality: perplexity, and teacher-forced KL divergence and top-1 agreement against
BF16 (wiki: wikitext-2, first 30 x 2048-token chunks (scored rows 1024-2046), Talus BF16 as reference; code: code corpus, 26 x 2048-token chunks, Talus BF16 as reference). BF16 is an opt-in reference path (pass the
BF16 GGUF to `--model`): it shows what the quantization costs, it is not a way to serve the model.

| Weights | Size | PPL wiki | PPL code | KL wiki | KL code | top-1 wiki | top-1 code | chat | code | edit | prompt 2K | prompt 32K | decode 2K | decode 32K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 (reference, opt-in) | 54.7 GB | 6.1272 | 1.7200 | reference | reference | reference | reference | 14.6 | 19.4 | 17.8 | 574 | 506 | 4.1 | 4.0 |
| `UD-Q4_K_XL` (option) | 17.6 GB | 6.1194 | 1.7476 | 0.0061 | 0.0357 | 96.8 % | 95.8 % | 43.2 | 57.4 | 101.3 | 553 | 487 | 12.6 | 11.3 |
| `UD-IQ4_XS` (shipped default) | 14.3 GB (+2.2 GB W4A4 prompt sidecar) | 6.2817 | 1.8361 | 0.0370 | 0.1187 | 91.7 % | 92.2 % | 51.3 | 67.8 | 122.7 | 629 | 546 | 15.2 | 13.2 |

- PPL: lower is better. Sizes are the GGUF files.
- `UD-IQ4_XS` PPL, KL and top-1 are the shipped default prompt path, with the 4-bit prompt weights (`.w4i`) on,
  measured on the laptop with a v0.1.6 build on the same chunks against the same BF16 reference. The exact int8 prompt
  path (`TALUS_Q27_W4I=0`, slower prompts) measures KL 0.0159 (wiki) / 0.0815 (code). The 4-bit prompt
  weights change the prompt's logits, so the first token of each reply (and prompt log-probabilities) see this
  difference; later tokens come from decode, where the effect is much smaller.
- chat / code / edit: tok/s through the shipped `talus-server` with each format's run config, including speculative
  decoding. Prompt and decode: plain `talus-cli` benchmarks without speculation, median of three runs, so the decode
  columns are far below the served figures.
- BF16 chat / code / edit were measured with the MTP drafter (`STRATA_DRAFTER=mtp`). In v0.1.2 (unchanged since) the automatic drafter
  choice picks DFlash2 on BF16 weights, which gave about 3.9 tok/s in v0.1.2, close to plain decode; a fix is planned.
- Talus's BF16 against llama.cpp's BF16 on the same chunks: KL 0.00033 (wiki) / 0.00024 (code), PPL
  ratio 0.9997: the size of llama.cpp's own CPU-vs-GPU BF16 difference, so the BF16 reference is correct.
- `UD-Q4_K_XL`'s wiki perplexity is slightly below BF16's (PPL ratio 0.9987 ± 0.0008). That is
  within measurement noise (about 1.7 standard errors), not a gain over BF16; KL and top-1 measure how far each quant
  is from BF16.

## Comparison with other engines (desktop, Talus v0.1.3)

These rows were measured on the desktop (GMKtec EVO-X2, about 120 W) with the **published v0.1.3 tarball**, in one
session per model on 2026-11-03 to 2026-11-04. In each session the engines ran one after another with Talus before
and after them (`Talus -> other engine(s) -> Talus`); the Talus row is the mean of those Talus runs, which agreed within
1.3% of each other. Every engine ran the same HTTP client, prompts and cells as the speed
tables above (12 prompts at about 2K and 8 at about 32K for `tg2k` / `tg32k`, median per prompt; `pp2k` / `pp32k` the
first prompt of each depth), so these rows can be compared with the desktop speed tables, but not with the laptop rows
of earlier releases.

- The llama.cpp rows are **our patched build of llama.cpp ("halo4")** with the best configuration we found per model,
  not stock llama.cpp.
- gufo and tubo cannot load our `UD-IQ4_XS` files, so they ran on their documented `UD-Q4_K_XL` files. On Qwen3.8-27B
  the second Talus row runs that same file, which separates the engines from the weight-size difference.
- Package power was the same for every engine (about 117-120 W socket while busy); see "Reference engine conditions".

### Qwen3.8-27B

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.3 | 49.7 | 66.7 | 123 | 601 | 542 | 54.1 | 37.2 | 3.34 s / 59.05 s / 1.63 s | 32 GiB |
| Talus v0.1.3 (`UD-Q4_K_XL`, the same file as gufo and tubo) | 42.8 | 57.5 | 101 | 487 | 481 | 37.0 | 36.3 | 4.12 s / 66.61 s / 1.82 s | 28 GiB |
| llama.cpp, our patched build (halo4) | 38.7 | 45.5 | 69.2 ¹ | 402 | 329 | 36.2 | 32.8 | 4.99 s / 97.24 s / 1.41 s | 100 GiB |
| gufo (`82c62b4`, `UD-Q4_K_XL` + DFlash2 `Q4_K_M`) | 35.9 | 48.9 | 72.5 | 605 | 572 | 33.6 | 30.2 | 3.34 s / 56.02 s / 1.11 s | 93 GiB |
| tubo (`2a258b9`, `UD-Q4_K_XL` + DFlash2 `Q4_K_M`) | 40.2 | 53.0 | 58.6 | 625 | 576 | 33.9 | 32.7 | 3.23 s / 55.56 s / 1.06 s | 93 GiB |

¹ Host-specific, cause unknown: with the same binaries, Vulkan driver, configuration and generated text, this cell measures 82.4 on the laptop (70 W) and 69.2 on the desktop; every other llama.cpp cell is faster on the desktop.

### Qwen3.8-Flash-Next

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.3 | 62.9 | 74.2 | 144 | 1,110 | 1,475 | 64.7 | 66.8 | 1.70 s / 19.30 s / 942 ms | 51 GiB |
| gufo (`82c62b4`, `UD-Q4_K_XL` + shared-`Q8_0` MTP) | 54.9 | 65.6 | 105 | 1,627 | 1,710 | 55.3 | 55.1 | 1.15 s / 16.64 s / 431 ms | 35 GiB |

### GLM-5.3-Flash

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.3 | 33.9 | 34.5 | 36.5 | 563 | 564 | 35.1 | 36.5 | 3.57 s / 56.82 s / 1.88 s | 9 GiB |
| llama.cpp, our patched build (halo4) | 30.2 | 31.6 | 31.6 | 291 | 265 | 30.8 | 28.8 | 6.90 s / 120.93 s / 2.06 s | 8 GiB |

### MiMo-V2.6-Flash

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.3 | 39.9 | 49.2 | 64.4 | 735 | 661 | 43.0 | 42.6 | 2.73 s / 48.45 s / 988 ms | 14 GiB |
| llama.cpp, our patched build (halo4) | 33.9 | 43.0 | 53.0 | 376 | 341 | 36.4 | 36.0 | 5.34 s / 93.99 s / 2.19 s | 21 GiB |

### Laguna-S-2.1

Speeds in tokens per second.

| Engine | chat | code | edit | pp2k | pp32k | tg2k | tg32k | TTFT a / b / c | MemAvailable min |
|---|---|---|---|---|---|---|---|---|---|
| Talus v0.1.3 | 43.8 | 54.6 | 122 | 1,149 | 1,003 | 65.3 | 35.0 | 1.75 s / 31.96 s / 652 ms | 38 GiB |
| llama.cpp, our patched build (halo4) | 39.5 | 47.9 | 89.9 | 732 | 621 | 37.2 | 33.6 | 2.75 s / 51.58 s / 1.14 s | 34 GiB |

## Reference engine conditions (desktop comparison)

Every row in the desktop comparison tables was measured on the same GMKtec EVO-X2 (Ryzen AI Max+ 395 / Radeon 8060S,
128 GB, BIOS in its highest sustained power mode, about 120 W) with the same HTTP client and corpus, one engine at a
time:

- **Talus v0.1.3**: the published tarball, served by its own `talus-server` with the run configs `talus setup` writes.
- **llama.cpp (`halo4`)**: our patched build of llama.cpp (Vulkan + speculative-decoding patches), the same binaries
  as in the v0.1.0 comparison, with the best configuration we found for each model, on the same GGUF files as Talus
  (MiMo-V2.6-Flash: the `a4-x` file Talus ships). The Vulkan driver is the pinned RADV build from the Strix Halo
  llama.cpp toolbox, as on the laptop.
- **gufo** ([gufo-org/gufo](https://github.com/gufo-org/gufo), MIT, commit `82c62b4`, the same image as in the v0.1.0
  comparison, built from its Nix flake with ROCm 7.2.3) and **tubo** ([BonkusCheemus/tubo](https://github.com/BonkusCheemus/tubo),
  MIT, a gufo fork, commit `2a258b9`, built the same way, with its documented `tubo.env` minus the TurboQuant KV, NPU and
  SSD-embedding settings, and `--draft-tokens 7 --min-draft-tokens 1 --draft-policy fixed`): run in network-isolated
  containers (`--network none`, read-only, all capabilities dropped) on their documented `UD-Q4_K_XL` quants
  (`serve llm --context 65536`: 27B with `DFlash2 Q4_K_M` and a 4 GiB RAM cache; Flash-Next with the shared-`Q8_0` MTP
  and a 2 GiB RAM cache). Chat sampling defaults differ (gufo and tubo use `top_k 20, min_p 0`; Talus uses
  `top_k 40, min_p 0.05`); `code`, `edit` and the depth cells are greedy. Talus's long-copy verify-window drafting
  (`THIRD_PARTY_NOTICES.md` §6) was prompted by gufo's speculative-decoding notes.

## Comparison with other engines (laptop, v0.1.0, earlier)

These rows were not re-measured after v0.1.0. They come from the v0.1.0 release runs on the 70 W laptop: each model's
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

## Reference engine conditions (laptop comparison)

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
