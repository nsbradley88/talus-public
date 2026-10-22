#!/bin/bash
# Build Talus for gfx1151 with a ROCm SDK into build/ (MMQ K-quants on: the Laguna and qwen35 prefill need them).
#   ROCM_SDK=<sdk root>   default ~/rocm/install (pinned release SDK: ROCm 10.2.0a20261014 / TheRock gfx1151 _rocm_sdk_devel)
#   BUILD=<dir>           default build
#   TARGETS="..."  JOBS=N
set -eo pipefail
cd "$(dirname "$0")"
SDK=${ROCM_SDK:-$HOME/rocm/install}
BUILD=${BUILD:-build}
# the device libraries: $SDK/amdgcn/bitcode in the wheels' layout, $SDK/lib/llvm/amdgcn/bitcode in an install tree
BC=$SDK/amdgcn/bitcode
[ -d "$BC" ] || BC=$SDK/lib/llvm/amdgcn/bitcode
CXX_HIP=$SDK/lib/llvm/bin/clang++
[ -x "$CXX_HIP" ] || CXX_HIP=$SDK/llvm/bin/clang++
export ROCM_PATH=$SDK HIP_PATH=$SDK HIP_PLATFORM=amd
export PATH=$SDK/bin:$SDK/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib:${LD_LIBRARY_PATH:-}
cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DGPU_TARGETS=gfx1151 -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_HIP_COMPILER=$CXX_HIP -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$SDK --rocm-device-lib-path=$BC" ${CMAKE_EXTRA:-}
cmake --build "$BUILD" --target ${TARGETS:-talus_kernels talus-pf-bench talus-attn-bench hip_laguna_mmvq_check} \
  -j ${JOBS:-12} 2>&1 | { grep -v "^\[" || [ $? -eq 1 ]; }

# Stale-build guard: every requested target's output must be newer than each source file it depends on (the direct
# inputs ninja knows plus the headers in its objects' depfiles). Catches a build that "succeeded" while leaving an old
# libtalus_kernels.so behind (e.g. the target was not built, or a failure got swallowed). STALE_CHECK=0 skips it.
if [ "${STALE_CHECK:-1}" != 0 ]; then
  SRC=$(pwd -P)
  for t in ${TARGETS:-talus_kernels talus-pf-bench talus-attn-bench hip_laguna_mmvq_check}; do
    q=$(ninja -C "$BUILD" -t query "$t")
    if printf '%s\n' "$q" | sed -n 2p | grep -q 'input: phony'; then out=$(printf '%s\n' "$q" | sed -n 3p | tr -d ' '); else out=$t; fi
    [ -f "$BUILD/$out" ] || { echo "build.sh: STALE CHECK FAILED: $t -> $BUILD/$out does not exist" >&2; exit 1; }
    ins=$(ninja -C "$BUILD" -t inputs "$t")
    newer=$( { printf '%s\n' "$ins"; ninja -C "$BUILD" -t deps $(printf '%s\n' "$ins" | grep '\.o$') 2>/dev/null \
               | grep '^    ' ; } | sed 's/^ *//' | grep "^$SRC/" | sort -u \
             | while read -r f; do if [ "$f" -nt "$BUILD/$out" ]; then echo "$f"; fi; done | head -5 || true)
    [ -z "$newer" ] || { echo "build.sh: STALE CHECK FAILED: $BUILD/$out is older than:" >&2; echo "$newer" >&2; exit 1; }
  done
  echo "build.sh: stale check OK (${TARGETS:-default targets})"
fi
