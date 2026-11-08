#!/usr/bin/env bash
# tests/install/build-error.sh : what install.sh shows when the kernel build fails, how many jobs it runs, and when it
# clears a build directory an earlier attempt left (talus-public#2: a parallel build kept compiling after the first
# error, so the last 30 lines of build-kernels.log were warnings from other files; -j came from MemTotal; a failed
# first attempt may leave a bad build directory).
#
# Runs install.sh --yes --force --no-setup in a temporary HOME on a copy of this tree whose build.sh writes a canned
# log and fails. The PATH holds the basic tools plus shims for cc, c++, make, cmake, ninja and rustup (the real
# ones are never called); ROCM_SDK is an empty stand-in. Checks:
#   1. a compile error early in the log, warnings after it: the first error is shown;
#   2. a compiler killed by the OOM killer: the hint to set TALUS_JOBS lower;
#   3. a log with no error line: the tail;
#   4. TALUS_JOBS=3 reaches build.sh as JOBS=3; without it, JOBS <= CPU threads and <= MemAvailable / 2 GiB;
#   5. TALUS_JOBS=0 or a non-number stops with a message;
#   6. src/build from another source path or ROCm SDK, without a CMakeCache.txt, or after an unfinished build is
#      removed; a matching one, or TALUS_KEEP_BUILD=1, is kept.
# Nothing outside the temporary directory changes.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fail=0
no() { echo "build-error: FAIL: $*" >&2; fail=1; }

# the PATH
mkdir -p "$T/path"
for t in sh bash env uname awk sed grep cut tr head tail sort df id getent date mkdir rm ln install tar \
         curl git python3 mktemp dirname basename cat ls nproc sleep cp chmod readlink wc touch find xargs \
         expr test true false printf timeout; do
  p=$(command -v "$t") && [ "${p#/}" != "$p" ] && ln -s "$p" "$T/path/$t"
done
for t in cc c++ make; do printf '#!/bin/sh\nexit 0\n' > "$T/path/$t"; done
# python3 with venv/ensurepip as far as install.sh's check goes (Debian/Ubuntu split them into python3-venv)
rm -f "$T/path/python3"
printf '#!/bin/sh\ncase "$*" in *ensurepip*) exit 0 ;; esac\nexec %s "$@"\n' "$(command -v python3)" > "$T/path/python3"
printf '#!/bin/sh\necho "cmake version 3.31.6"\n' > "$T/path/cmake"
printf '#!/bin/sh\necho 1.12.1\n' > "$T/path/ninja"
printf '#!/bin/sh\ncase "$1" in toolchain) echo "${TC}-x86_64-unknown-linux-gnu" ;; *) echo "rustup 1.28.2 (shim)" ;; esac\n' \
  > "$T/path/rustup"
chmod +x "$T/path/python3" "$T/path/cc" "$T/path/c++" "$T/path/make" "$T/path/cmake" "$T/path/ninja" "$T/path/rustup"

# the source: this tree with a build.sh that records JOBS and prints $FIXTURE
mkdir -p "$T/src"
if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  (cd "$ROOT" && git ls-files -co --exclude-standard -z | tar --null -T - -cf - | tar -xf - -C "$T/src")
else   # a tree without .git (a git archive, a build VM's copy)
  tar -C "$ROOT" --exclude=./rust/target --exclude=./build --exclude=./build-release -cf - . | tar -xf - -C "$T/src"
fi
printf '#!/bin/bash\necho "$JOBS" > "$HOME/jobs"\ncat "$FIXTURE"\nexit 1\n' > "$T/src/build.sh"
TC=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$T/src/rust/rust-toolchain.toml")
mkdir -p "$T/rocm/lib" && touch "$T/rocm/lib/libamdhip64.so"

# the fixtures
warn() {  # N warning blocks like the ones that filled the screen in #2
  for i in $(seq 1 "$1"); do
    printf '%s\n' "/src/kernels/cuda/native_mmvq.cu:831:25: warning: comparison of integers of different signs: 'unsigned int' and 'int' [-Wsign-compare]" \
      "      831 |         if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];" "          |             ~~~~~~~~~~~ ^  ~"
  done
}
{ echo "[12/164] Building HIP object src/CMakeFiles/talus_kernels.dir/kernels/cuda/rmsnorm.cu.o"
  echo "FAILED: src/CMakeFiles/talus_kernels.dir/laguna/talus_kernels_laguna.cpp.o"
  echo "/rocm/llvm/bin/clang++ -DTALUS -O3 --offload-arch=gfx1151 -c /src/laguna/talus_kernels_laguna.cpp"
  echo "/src/laguna/talus_kernels_laguna.cpp:1:10: fatal error: 'issue2_missing.h' file not found"
  echo "    1 | #include <issue2_missing.h>"
  echo "1 error generated when compiling for gfx1151."
  warn 40
  echo "247 warnings generated when compiling for host."
  echo "ninja: build stopped: subcommand failed."; } > "$T/compile.log"
{ warn 3
  echo "FAILED: src/CMakeFiles/talus_kernels.dir/kernels/cuda/tp_moe.cu.o"
  echo "clang++: error: unable to execute command: Killed"
  echo "clang++: error: clang frontend command failed due to signal (use -v to see invocation)"
  warn 20
  echo "ninja: build stopped: subcommand failed."; } > "$T/oom.log"
{ warn 5; echo "something odd happened at the end"; echo "ninja: build stopped: subcommand failed."; } > "$T/plain.log"

# inst FIXTURE [ENV...] : output of install.sh (stdout+stderr)
inst() {
  local fx=$1; shift
  [ -n "${KEEP_HOME:-}" ] || { rm -rf "$T/home"; mkdir -p "$T/home"; }
  (cd "$T/home" && env -i HOME="$T/home" PATH="$T/path" TC="$TC" FIXTURE="$T/$fx" ROCM_SDK="$T/rocm" NO_COLOR=1 \
     "$@" timeout 120 bash "$T/src/install.sh" --yes --force --no-setup --src "$T/src" 2>&1 </dev/null)
}
has() { grep -qF -- "$2" <<<"$1" || { no "$3: missing '$2'"; tail -n 4 <<<"$1" | sed 's/^/    | /' >&2; }; }
hasnt() { if grep -qF -- "$2" <<<"$1"; then no "$3: should not show '$2'"; fi; }

out=$(inst compile.log)
[ -z "${SHOW:-}" ] || sed -n '/\[4\/5\]/,$p' <<<"$out"   # SHOW=1: print what the user sees
has "$out" "first error (line 2 of the log)" "compile error"
has "$out" "fatal error: 'issue2_missing.h' file not found" "compile error"
has "$out" "FAILED: src/CMakeFiles/talus_kernels.dir/laguna/talus_kernels_laguna.cpp.o" "compile error"
has "$out" "build-kernels failed" "compile error"
hasnt "$out" "Run again with fewer" "compile error (no OOM hint)"
[ $fail = 0 ] && echo "build-error: compile error ok"

f0=$fail
out=$(inst oom.log TALUS_JOBS=8)
[ -z "${SHOW:-}" ] || sed -n '/\[4\/5\]/,$p' <<<"$out"
has "$out" "clang++: error: unable to execute command: Killed" "oom"
has "$out" "Run again with fewer: TALUS_JOBS=4" "oom"
[ $fail = "$f0" ] && echo "build-error: oom hint ok"

f0=$fail
out=$(inst plain.log)
has "$out" "something odd happened at the end" "no error line (tail)"
has "$out" "ninja: build stopped: subcommand failed." "no error line (tail)"
[ $fail = "$f0" ] && echo "build-error: tail fallback ok"

f0=$fail
inst plain.log TALUS_JOBS=3 >/dev/null
[ "$(cat "$T/home/jobs" 2>/dev/null)" = 3 ] || no "TALUS_JOBS=3 gave JOBS=$(cat "$T/home/jobs" 2>/dev/null)"
inst plain.log >/dev/null
j=$(cat "$T/home/jobs" 2>/dev/null || echo 0)
avail=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)
lim=$(( (avail + 4) / 2 )); [ "$lim" -ge 1 ] || lim=1   # +4 GiB: memory moves between the two readings
{ [ "$j" -ge 1 ] && [ "$j" -le "$(nproc)" ] && [ "$j" -le "$lim" ]; } || no "default JOBS=$j (nproc $(nproc), MemAvailable $avail GiB)"
[ $fail = "$f0" ] && echo "build-error: jobs ok (default $j)"

f0=$fail
for bad in 0 x 2x; do
  out=$(inst plain.log TALUS_JOBS=$bad)
  has "$out" "TALUS_JOBS must be a whole number" "TALUS_JOBS=$bad"
  hasnt "$out" "Compiling the HIP kernel library" "TALUS_JOBS=$bad"
done
[ $fail = "$f0" ] && echo "build-error: bad TALUS_JOBS ok"
# 6. a build directory from an earlier attempt
f0=$fail
cache() {  # HOME_DIR ROCM_ROOT : a src/build with this CMakeCache.txt
  rm -rf "$T/src/build"; mkdir -p "$T/src/build"; touch "$T/src/build/keep"
  printf 'CMAKE_HOME_DIRECTORY:INTERNAL=%s\nCMAKE_HIP_COMPILER_ROCM_ROOT:UNINITIALIZED=%s\n' "$1" "$2" > "$T/src/build/CMakeCache.txt"
}
cache /elsewhere/talus "$T/rocm"; out=$(inst plain.log)
has "$out" "configured for /elsewhere/talus" "stale build (other source)"; [ ! -e "$T/src/build" ] || no "stale build (other source) kept"
cache "$T/src" /opt/rocm-other; out=$(inst plain.log)
has "$out" "configured with the ROCm SDK /opt/rocm-other" "stale build (other SDK)"; [ ! -e "$T/src/build" ] || no "stale build (other SDK) kept"
rm -rf "$T/src/build"; mkdir -p "$T/src/build"; out=$(inst plain.log)
has "$out" "it has no CMakeCache.txt" "stale build (no cache)"
cache "$T/src" "$T/rocm"; out=$(inst plain.log)
hasnt "$out" "Removing" "matching build"; [ -e "$T/src/build/keep" ] || no "matching build removed"
# that run failed: the next one starts over
out=$(KEEP_HOME=1 inst plain.log)
has "$out" "the last build did not finish" "unfinished build"; [ ! -e "$T/src/build" ] || no "unfinished build kept"
cache "$T/src" /opt/rocm-other; out=$(inst plain.log TALUS_KEEP_BUILD=1)
hasnt "$out" "Removing" "TALUS_KEEP_BUILD=1"; [ -e "$T/src/build/keep" ] || no "TALUS_KEEP_BUILD=1 removed it"
rm -rf "$T/src/build"
[ $fail = "$f0" ] && echo "build-error: stale build directory ok"
exit $fail
