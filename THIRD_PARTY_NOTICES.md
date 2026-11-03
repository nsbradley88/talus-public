# Third-Party Notices

Talus incorporates code and designs from the following open-source projects, and enforces a permissive license policy for its Rust dependencies.

---

## 1. Strata

The Talus C++/HIP kernel library (`libtalus_kernels`) and the Talus Rust host (`rust/`, which translates Strata's host architecture, model description registry, verify/commit semantics, chunk planner, and conversation cache) are derived from [Strata](https://github.com/Niko1221/Strata) (upstream tag `v0.1.44.1`, commit `61b3fb5d`).

```text
MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. ggml and llama.cpp

Kernel and host components (including `third_party/ggml/ggml-common.h` at commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` recorded in `third_party/ggml/VERSION.txt`, `src/prefill/fattn-mma-f16-f32acc.cuh`, `src/prefill/fattn-common.cuh`, `src/prefill/fattn-mma.cuh`, `src/prefill/fattn-cp-async.cuh`, `src/kernels/cuda/talus_iq_split.cuh`, `src/kernels/cuda/ggml_cuda_host.cu`, `src/kernels/cuda/mimo_kernels.cu`, `bench/editsrc/{sampling.cpp,speculative.cpp,llama-batch.cpp}`, MMQ kernels, FlashAttention, IQ quantization tables, `kv_q4`/router code, and ported ngram-mod, draft-vocab `.bin` format, DFlash, and block-sparse designs) are derived from [ggml](https://github.com/ggml-org/ggml) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 3. Rust Dependencies

All crates in the `rust/` workspace are licensed under `MIT`. Third-party Rust dependencies are checked in CI via `cargo deny check licenses` (`rust/deny.toml`) against the permissive SPDX allowlist:

- `MIT`
- `Apache-2.0`
- `Apache-2.0 WITH LLVM-exception`
- `BSD-2-Clause`
- `BSD-3-Clause`
- `ISC`
- `Unicode-3.0`
- `Zlib`

### Third-Party Rust Crates in the Workspace Lockfile

- **`minijinja`** (`Apache-2.0`) — Copyright (c) 2021-2026 Armin Ronacher (`https://github.com/mitsuhiko/minijinja`)
- **`memo-map`** (`Apache-2.0`) — Copyright (c) Armin Ronacher (`https://github.com/mitsuhiko/memo-map`)
- **`fancy-regex`** (`MIT`) — Copyright (c) 2016 The Fancy Regex Authors (`https://github.com/fancy-regex/fancy-regex`)
- **`aho-corasick`**, **`memchr`**, **`regex-automata`**, **`regex-syntax`** (`Unlicense OR MIT` / `MIT OR Apache-2.0`) — Copyright (c) Andrew Gallant and the Rust Project Developers
- **`serde`**, **`serde_core`**, **`serde_derive`**, **`serde_json`**, **`thiserror`**, **`thiserror-impl`**, **`ryu`**, **`itoa`**, **`zmij`**, **`proc-macro2`**, **`quote`**, **`syn`** (`MIT OR Apache-2.0`) — Copyright (c) Erick Tryzelaar, David Tolnay, and contributors
- **`bitflags`**, **`bit-set`**, **`bit-vec`**, **`cfg-if`**, **`equivalent`**, **`hashbrown`**, **`indexmap`**, **`libc`** (`MIT OR Apache-2.0`) — Copyright (c) The Rust Project Developers and contributors
- **`foldhash`** (`Zlib`) — Copyright (c) Orson Peters (`https://github.com/orlp/foldhash`)
- **`unicode-ident`** (`(MIT OR Apache-2.0) AND Unicode-3.0`) — Copyright (c) David Tolnay and Unicode, Inc.

Full third-party Rust dependency attributions for binary distributions can be generated from the workspace lockfile with `cargo about`.

---

## 4. Vendored Data and ROCm Runtime Libraries

- **Vendored data assets:** `data/*.bin` (expert profile tables and draft vocabularies) and `tools/hip/gfx1151-hipblaslt-*.txt` are inherited from [Strata](https://github.com/Niko1221/Strata) under the MIT License (§1). They contain no model weights.
- **ROCm runtime libraries:** ROCm libraries (`libamdhip64`, `hipBLAS`, `hipBLASLt`) are linked dynamically at runtime from the installed ROCm SDK and carry their respective licenses distributed with the ROCm SDK.

---

## 5. Benchmark and Agentic Evaluation Suite

The edit-benchmark prompts in `bench/editsrc/` embed three source files (`sampling.cpp`, `speculative.cpp`, `llama-batch.cpp`) from [llama.cpp](https://github.com/ggml-org/llama.cpp) under the MIT License (§2).

The agentic coding evaluation suite in `bench/agentic/` defines task patches and hidden grading tests against [charmbracelet/bubbles](https://github.com/charmbracelet/bubbles) (`v0.21.1`) and invokes `opencode` ([anomalyco/opencode](https://github.com/anomalyco/opencode), MIT; external tool, not redistributed in this repository) as an external CLI harness:

```text
MIT License

Copyright (c) 2020-2026 Charmbracelet, Inc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The talus-bench quality validator embeds the first 50 rows of the GSM8K test split
(`rust/crates/talus-bench-core/data/gsm8k/test-slice50.jsonl`, from
[openai/grade-school-math](https://github.com/openai/grade-school-math) commit `3101c7d5`, sha256 pinned in
`datasets.rs`) under the MIT License; the full split is fetched on demand and not redistributed.

talus-bench also embeds, each with its sha256 pinned in `datasets.rs`:

- `data/ifeval/ifeval-supported.jsonl`: 387 of the 541 prompts of [google/IFEval](https://huggingface.co/datasets/google/IFEval)
  (revision `966cd895`), licensed under the Apache License, Version 2.0
  (http://www.apache.org/licenses/LICENSE-2.0). Modified: filtered to the instruction types whose rules are
  reimplemented, re-serialized with sorted keys.
- `data/bfcl/bfcl-v3-smoke.jsonl`: 120 rows of the
  [Berkeley Function Calling Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
  v3 (revision `61fc0608`; simple, multiple, parallel, irrelevance), licensed under the Apache License, Version 2.0.
  Modified: question, functions and possible answers combined per row, categories interleaved. The full files
  are fetched on demand and not redistributed.
- `data/mmlu_pro/`: 280 test rows (20 per category) and the 70 validation rows of
  [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (revision `b189ec76`), MIT License per
  the dataset card. Modified: converted from parquet to JSON lines.

The scoring rules of IFEval (google-research `instruction_following_eval`, Apache-2.0), the BFCL AST checker
(ShishirPatil/gorilla, Apache-2.0) and MMLU-Pro's answer extraction (TIGER-AI-Lab/MMLU-Pro, Apache-2.0) are
reimplemented in Rust from those projects' published rules; no third-party evaluation code is included.

```text
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The talus-bench task `swebench_verified` (`rust/crates/talus-bench-core/src/swe/harness.rs`) translates to Rust the
evaluation rules of the SWE-bench harness ([SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench), package
`swebench` 5.0.2): the Python log parsers, log extraction and grading, the eval-script exit-code hook and the
patch-apply sequence. No SWE-bench data is included in this repository: the instance list is fetched at run time
from the Hugging Face dataset `SWE-bench/SWE-bench_Verified` into a local cache, and the evaluation images and the
target repositories (each under its own license) are pulled into the local docker store, never committed.

```text
MIT License

Copyright (c) 2023 Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik R Narasimhan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

The `swebench_verified` task's optional `mini` agent harness runs the upstream
[mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) (package `mini-swe-agent` 2.4.6, by Kilian A. Lieret
and Carlos E. Jimenez): its `DefaultAgent`, `LocalEnvironment`, `LitellmModel` and `config/benchmarks/swebench.yaml`,
unmodified. It is not vendored: the operator builds a pinned, self-contained bundle outside the repository (Python
3.12 + `mini-swe-agent==2.4.6` and its dependencies, e.g. litellm, resolved with `--exclude-newer 2026-10-14`), which
is bind-mounted read-only into the agent container. talus-bench adds only a small runner script
(`rust/crates/talus-bench-core/src/swe/runtime.rs`, `MINI_RUN_PY`) that points the model at the local endpoint.

```text
MIT License

Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 6. gufo (design idea only)

No code from [gufo](https://github.com/gufo-org/gufo) is in this repository. Two drafting choices in Talus (looking
for a long prompt match before running a draft head, and letting a prompt match fill a wider verify window on edits;
see "Long copies" in `docs/MODELS.md`) were prompted by reading gufo's speculative-decoding notes during our benchmark
comparison. gufo is licensed under the MIT License (`Copyright (c) 2026 Mostafa`).

