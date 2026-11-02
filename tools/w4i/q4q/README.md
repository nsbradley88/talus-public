# tools/w4i/q4q: the v3 27B 4-bit prompt sidecar

Same kernels as tools/w4i (`src/prefill/tp_w4i.cu`). The quantizer is better and still feeds the iu4 WMMA GEMM:

* **SmoothQuant scale**: s_j = rms(x_j)^a / rms(w_.j)^(1-a), with a = 0.5 for gate/up (shared, since they share the
  packed input) and 0.25 for down, normalised to geomean 1, then folded into the weights (W s). The pack kernel
  multiplies the activations by 1/s (`inv_s` in the file) before the 256-point Hadamard.
* **Activation group scales**: per 32-value group, the best of 4 clips (k from amax x {1, 0.9, 0.8, 0.7}, least
  squared error), still integer and still in Abel form (`kmode` 1 in the file flags).
* **Weights**: GPTQ in the smoothed, rotated basis, 256-column groups (wg 256, tile v5 = `TALUS_PF_W4I_V256`), damp
  0.1, full public calibration (tools/w4i/calib.py, 151,552 tokens).
* **Bias correction** (`correct_parts.py`): the MSE-optimal clip searches shrink the FFN output (gain about 0.985). Each
  output row's scale is multiplied by c_w[r] x c_a. The weight part is exact from the calibration Hessian; c_a is the
  activation quantizer's shrink, measured on Gaussian rows. Without this the sidecar is no better than v2 on the GPU,
  even though its per-matrix MSE is lower.

Layout: FFN of layers 48-63 (`c48`). Wider layouts pass the 4K gate but fail the 32K one: every matrix of layers
40-47 costs x1.4-1.7 at 32K (rd/rd-q4quant).

Recipe (CPU, ~31 GiB RAM, ~80 GiB of Hessians plus ~70 GiB of temporary dumps):
```
WORK=/path/work; mkdir -p $WORK; python3 ../calib.py $LLAMA_CPP_SRC $TOK $GGUF $WORK/calib
ACTDUMP=/path/to/actdump ./hess_all.sh $GGUF $WORK                       # X^T X of ffn_in / ffn_down, all layers
for L in $(seq 48 63); do printf "$L ffn_gate 256 0.5 1\n$L ffn_up 256 0.5 1\n$L ffn_down 256 0.25 1\n"; done > plan
GGUFPY=$LLAMA_CPP_SRC/gguf-py python3 q4q_sidecar.py $GGUF $WORK/hess plan $WORK/parts --jobs 2
GGUFPY=$LLAMA_CPP_SRC/gguf-py python3 correct_parts.py $GGUF $WORK/hess $WORK/parts $WORK/parts_c $(seq 48 63)
python3 q4q_pack.py Qwen3.8-27B-UD-IQ4_XS.w4i $GGUF $WORK/parts_c/*.npz        # format v3
```
Our file (2026-11-01): `2a4f69c3918218d3f1b6fbdf3a3278c65077ee042c4cee51e5b99e455862935d` (2,208,309,248 bytes).

Gates (UD-IQ4_XS, this file, layers 48-63), all against the exact int8 prompt path:
* 4K KL vs llama.cpp CPU: x1.160. The v2 file is x1.236; the bar is x1.35.
* 32K KL (3 prompts): x0.94; the v2 file is x1.31.
* needle 60K and agentic: pass.
* speed vs the exact path: faster prompts at 2K and 32K, decode and draft acceptance unchanged (figures: the
  v0.1.2 release notes).
