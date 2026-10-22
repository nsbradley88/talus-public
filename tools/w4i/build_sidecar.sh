#!/bin/bash
# Copyright (c) 2026 nsbradley88
# SPDX-License-Identifier: MIT
#
# build_sidecar.sh GGUF OUTDIR : the Qwen3.8-27B W4A4 prompt sidecars (tp_w4i, docs/MODELS.md) from scratch, on CPU.
#   <OUTDIR>/<stem>.w4i       FFN of layers 48-63 (release default, quality-gated; ~2.0 GiB)
#   <OUTDIR>/<stem>.fast.w4i  FFN of all 64 layers (opt-in "fast" profile, failed the gate; ~8.2 GiB)
# Steps: 1. calibration token ids from public sources (calib.py: llama.cpp b11062 + license texts, 149.5K tokens);
#        2. per quarter of the layers: a llama.cpp CPU forward (actdump) dumps the FFN inputs, hess_acc.py sums
#           X^T X, gptq_sidecar.py quantizes ffn_gate / ffn_up / ffn_down to int4 (GPTQ, per-output-row scales,
#           damp 0.1, 256-point Hadamard basis with seed 20261017), the Hessians are deleted;
#        3. w4i_convert.py packs the tp_w4i layout (format v2, records the GGUF's name and size); sha256sum.
# Env: LLAMA_CPP_SRC   llama.cpp checkout at tag b11062 (3cf03257f219afbe7334045ff7c6a06ac68c627d; also its gguf-py)
#      LLAMA_CPP_BUILD CPU build of it (libllama, bin/llama-tokenize), e.g.
#                      cmake -S $LLAMA_CPP_SRC -B b -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_AVX2=ON \
#                        -DGGML_AVX512=ON -DLLAMA_CURL=OFF && cmake --build b --target llama llama-tokenize
#      WORK (default OUTDIR/w4i-work), THREADS (16), JOBS (2: GPTQ processes, ~8 GiB RAM each for ffn_down),
#      QUARTERS ("3 2 1 0"), CALIB_SHA256SUMS (optional: a published SHA256SUMS the calibration ids must match).
# Needs: g++, python3 + numpy, ~25 GiB free disk per quarter (Hessians + one dump batch), ~30 GiB RAM, hours of CPU.
# Determinism: the calibration ids are byte-identical given the pins; the sidecar bytes depend on the floating point
# of the CPU forward and numpy (same build + CPU features -> same bytes; ours are listed in docs/MODELS.md).
set -euo pipefail
GGUF=$(readlink -f "${1:?usage: build_sidecar.sh GGUF OUTDIR}"); OUT=$(readlink -f "${2:?OUTDIR}")
T=$(cd "$(dirname "$0")" && pwd)
: "${LLAMA_CPP_SRC:?set LLAMA_CPP_SRC (llama.cpp b11062)}" "${LLAMA_CPP_BUILD:?set LLAMA_CPP_BUILD}"
WORK=${WORK:-$OUT/w4i-work}; THREADS=${THREADS:-16}; JOBS=${JOBS:-2}; QUARTERS=${QUARTERS:-3 2 1 0}
export GGUFPY=$LLAMA_CPP_SRC/gguf-py
stem=$(basename "$GGUF" .gguf)
mkdir -p "$OUT" "$WORK"
cd "$WORK"
log() { echo "$(date +%T) $*" | tee -a build.log; }

# 0. the activation dumper
if [ ! -x actdump ]; then
  g++ -O2 -std=c++17 "$T/actdump.cpp" -I"$LLAMA_CPP_SRC/include" -I"$LLAMA_CPP_SRC/ggml/include" \
    -L"$LLAMA_CPP_BUILD/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LLAMA_CPP_BUILD/bin" -o actdump
fi

# 1. calibration ids
if [ ! -e calib/SHA256SUMS ]; then
  python3 "$T/calib.py" "$LLAMA_CPP_SRC" "$LLAMA_CPP_BUILD/bin/llama-tokenize" "$GGUF" calib | tee -a build.log
fi
if [ -n "${CALIB_SHA256SUMS:-}" ]; then
  diff <(sort -k2 calib/SHA256SUMS) <(sort -k2 "$CALIB_SHA256SUMS") > /dev/null || { log "calibration ids differ from $CALIB_SHA256SUMS"; exit 1; }
  log "calibration ids match $CALIB_SHA256SUMS"
fi

# 2. Hessians + GPTQ per layer quarter
for q in $QUARTERS; do
  lo=$((16 * q)); hi=$((16 * q + 15)); HD=hq$q
  [ -e q$q.tw4 ] && { log "q$q done"; continue; }
  if [ ! -e $HD/DONE ]; then
    mkdir -p $HD; B=(); n=0
    run() {
      rm -rf tmp; mkdir tmp
      W4C_LAYERS=$(seq -s, $lo $hi) W4C_KINDS=ffn_in,ffn_down DUMP_THREADS=$THREADS nice ./actdump --model "$GGUF" --out tmp "$@" >> dump-q$q.log 2>&1
      OPENBLAS_NUM_THREADS=$THREADS nice python3 "$T/hess_acc.py" tmp $HD >> build.log
      printf '%s\n' "$@" >> $HD/fed
      log "q$q hessians += $*"
    }
    for f in calib/train-*.ids; do
      grep -qx "$f" $HD/fed 2>/dev/null && continue
      B+=("$f"); n=$((n + $(tr ',' '\n' < "$f" | wc -l)))
      if [ $n -ge 12000 ]; then run "${B[@]}"; B=(); n=0; fi
    done
    [ ${#B[@]} -gt 0 ] && run "${B[@]}"
    rm -rf tmp; touch $HD/DONE
  fi
  nice python3 "$T/gptq_sidecar.py" --gguf "$GGUF" --out q$q.tw4.tmp --classes ffn_in,ffn_down --layers $lo-$hi \
    --wgroup 0 --method gptq --hess $HD --damp 0.1 --jobs $JOBS > gptq-q$q.log 2>&1
  mv q$q.tw4.tmp q$q.tw4
  rm -rf $HD
  log "q$q sidecar part done: $(tail -1 gptq-q$q.log)"
done

# 3. pack + checksums
parts=(); for q in 0 1 2 3; do [ -e q$q.tw4 ] && parts+=(q$q.tw4); done
python3 "$T/w4i_convert.py" --gguf "$GGUF" --layers 48-63 "$OUT/$stem.w4i" "${parts[@]}" | tee -a build.log
if [ ${#parts[@]} = 4 ]; then
  python3 "$T/w4i_convert.py" --gguf "$GGUF" "$OUT/$stem.fast.w4i" "${parts[@]}" | tee -a build.log
fi
(cd "$OUT" && sha256sum "$stem".*w4i) | tee -a build.log
