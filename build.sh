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
[ -r "$SDK/include/hip/hip_runtime.h" ] || { echo "build.sh: $SDK/include/hip/hip_runtime.h not found (ROCM_SDK must be the SDK root)" >&2; exit 1; }
export ROCM_PATH=$SDK HIP_PATH=$SDK HIP_PLATFORM=amd
export PATH=$SDK/bin:$SDK/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib:${LD_LIBRARY_PATH:-}
# A system ROCm's headers (Fedora's rocm-hip-devel: older ones in /usr/include/hip) shadow the SDK's: clang adds
# --rocm-path's include dir with -idirafter, i.e. AFTER /usr/include (talus-public#2). Then -isystem $SDK/include puts
# the SDK's first. Only when such headers exist: the flag changes every TU's CUID (a hash of the command line), so the
# library's symbol names; without a system ROCm the command line and the library stay byte-identical.
HIPFLAGS="--rocm-path=$SDK --rocm-device-lib-path=$BC"
for d in /usr/include/hip /usr/local/include/hip; do
  if [ -e "$d" ]; then
    echo "build.sh: $d (a system ROCm) found: the SDK's headers go first (-isystem $SDK/include)"
    HIPFLAGS="$HIPFLAGS -isystem $SDK/include"; break
  fi
done
# An existing cache that spells the compiler's path differently (a venv's lib64 -> lib link, `talus update`) makes CMake
# delete the cache and configure again WITHOUT the options below: no HIP, "unknown target talus_kernels". Start fresh.
if [ -f "$BUILD/CMakeCache.txt" ]; then
  OLD_HIP=$(sed -n 's/^CMAKE_HIP_COMPILER:[A-Z]*=//p' "$BUILD/CMakeCache.txt" | head -n 1)
  if [ -n "$OLD_HIP" ] && [ "$OLD_HIP" != "$CXX_HIP" ]; then
    echo "build.sh: the HIP compiler path changed ($OLD_HIP -> $CXX_HIP): configuring from scratch"
    rm -rf "$BUILD/CMakeCache.txt" "$BUILD/CMakeFiles"
  fi
fi
cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DGPU_TARGETS=gfx1151 -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_HIP_COMPILER=$CXX_HIP -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=$HIPFLAGS" ${CMAKE_EXTRA:-}
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
