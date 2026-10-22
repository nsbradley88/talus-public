# tools/w4i: the Qwen3.8-27B 4-bit prompt sidecars

Talus's 27B engine reads prompts with 4-bit math (W4A4 on `v_wmma_i32_16x16x16_iu4`, `src/prefill/tp_w4i.cu`) in
the feed-forward of layers 48-63 when a `.w4i` file sits next to the GGUF (docs/MODELS.md, "4-bit prompt math").
The file holds int4 copies of those weights, quantized with GPTQ in the rotated basis the kernel uses. These tools
rebuild the files from scratch on a CPU.

| File | Layers | Size | Use |
|---|---|---|---|
| `Qwen3.8-27B-UD-IQ4_XS.w4i` | FFN of 48-63 | about 2.0 GiB | release default (passed the quality gate) |
| `Qwen3.8-27B-UD-IQ4_XS.fast.w4i` | FFN of 0-63 | about 8.0 GiB | opt-in fast profile `TALUS_Q27_W4I=fast` (failed the gate) |

Our files (2026-10-28, built with this recipe on an AMD EPYC, numpy 2.4.4):

```
b4f8d947e0a2d48382d331c9a4eb3e72032801e80c48cb2c1172bfe15bb4a105  Qwen3.8-27B-UD-IQ4_XS.w4i       (2,142,248,960 bytes)
d293452b9031fa58d3fe6e70fd8f6534575bfe548d0abfc2208346d6f9887cc3  Qwen3.8-27B-UD-IQ4_XS.fast.w4i  (8,568,983,552 bytes)
```

## Recipe

```bash
# 1. llama.cpp at tag b11062 (the version Talus pins), CPU build
git clone https://github.com/ggml-org/llama.cpp && git -C llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
cmake -S llama.cpp -B llama.cpp/build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=ON \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build llama.cpp/build-cpu -j --target llama llama-tokenize
# 2. the sidecars (hours of CPU; ~30 GiB RAM, ~30 GiB free disk)
LLAMA_CPP_SRC=$PWD/llama.cpp LLAMA_CPP_BUILD=$PWD/llama.cpp/build-cpu \
  tools/w4i/build_sidecar.sh /path/to/Qwen3.8-27B-UD-IQ4_XS.gguf /path/to/outdir
```

`QUARTERS=3` builds only what the default file needs (layers 48-63, about a quarter of the time).
`CALIB_SHA256SUMS=tools/w4i/calib-SHA256SUMS` makes the script check that your calibration ids equal ours.

## What the steps do

1. **Calibration set** (`calib.py`): 151,552 tokens from public sources only, tokenized with the target GGUF:
   20 x 2048 tokens of source code (llama.cpp's C++, CUDA and Python files), 12 x 4096 tokens of agentic edit
   transcripts in Qwen's tool-call template (built from llama.cpp's git history: the commit message as the task,
   the touched files as tool results, the diff hunks as edits), 10 x 2048 tokens of chat answers (llama.cpp's
   documentation), 2 x 16384 tokens of long-context prompts (concatenated sources) and 4 x 2048 tokens of prose
   (the GPL-3, GFDL-1.3, Apache-2.0 and MPL-2.0 texts from `/usr/share/common-licenses`). Seeded and sorted, so
   the ids are byte-identical: `calib-SHA256SUMS`.
2. **Hessians** (`actdump.cpp`, `hess_acc.py`): llama.cpp's CPU forward over the set, a quarter of the layers at a
   time; the inputs of `ffn_gate` / `ffn_up` and of `ffn_down` are summed into X^T X per layer.
3. **GPTQ** (`gptq_sidecar.py`): the GGUF's FFN weights are dequantized, rotated per 256 columns (Hadamard with
   seeded signs, seed 20261017) and quantized to int4 (-7..7) with GPTQ against the rotated Hessian (damp 0.1,
   blocks of 128 columns), one scale per output row chosen by an MSE clip search.
4. **Packing** (`w4i_convert.py`): the kernel's tile layout, the scales and the signs, plus the GGUF's file name and
   size (the engine uses the file only with that GGUF).

The sidecar bytes depend on floating point (the CPU forward and numpy), so a different CPU or library build can give
slightly different bytes; the calibration ids do not.

## Quality

The default (layers 48-63) passed the 27B gates for prompt-path changes (docs/METHODOLOGY.md): KL to llama.cpp's
CPU backend on 4K prompts 0.00539 against the exact path's 0.00436 = x1.24 (limit x1.35), and passed the needle, agentic, and tier-1 A/B gates (see docs/RESULTS.md). The fast profile (all layers) measured 0.01269 = x2.91: it failed the gate and is opt-in (see docs/MODELS.md).
