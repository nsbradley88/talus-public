#!/bin/sh
# Talus installer for AMD Strix Halo (Ryzen AI Max, Radeon 8060S / gfx1151) on Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/nsbradley88/talus-public/main/install.sh | bash
#
# What it does, each step skipped when already done (run it again any time):
#   1. checks this computer: Linux, x86-64, the gfx1151 GPU, 128 GB of memory, free disk;
#   2. installs what the build needs: a C/C++ compiler, git, curl, Python 3 with venv (asks before sudo),
#      CMake and Ninja (into your home when the system's are missing or old), Rust (rustup, the pinned toolchain);
#   3. the ROCm SDK into your home (AMD TheRock wheels in ~/.local/share/talus/rocm), or one you already have
#      (ROCM_SDK=/path);
#   4. builds Talus (the HIP kernel library and the Rust programs) and installs it to ~/.local/share/talus/bin, with
#      `talus` and `talus-tui` in ~/.local/bin;
#   5. starts `talus setup`: choose models, download them, the kernel parameters (shown and asked, never silent),
#      the run configs and an optional user service.
# Nothing is installed system-wide except the distribution packages in step 2, and only after you say yes.
#
# Options: --yes (take every default) --dry-run (print the actions, change nothing) --advanced (all choices)
#          --allow-system-changes (with --yes: apply the kernel parameters too) --force (continue on unsupported
#          hardware) --build-only (steps 1-4, no setup; what `talus update` runs, so the GPU check only warns) --src DIR (a Talus checkout to build) --no-setup
# Environment: TALUS_HOME, ROCM_SDK (use an SDK you have), TALUS_ROCM_VERSION / TALUS_ROCM_INDEX (override the pin
#              in rocm.version / AMD's nightly index), TALUS_ROCM_STABLE, TALUS_REPO, TALUS_BRANCH,
#              TALUS_RELEASE=<tag> or TALUS_PREBUILT_URL=<tar.gz> (a release tarball instead of compiling),
#              HF_TOKEN (optional, never stored).

# Everything is inside main(), so a download cut short runs nothing.
main() {
set -eu

REPO=${TALUS_REPO:-https://github.com/nsbradley88/talus-public}
BRANCH=${TALUS_BRANCH:-main}
DATA=${TALUS_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/talus}
ROCM_PIN=10.2.0a20261014   # used only when the source has no rocm.version (the pin; read once the source is here)
ROCM_INDEX=${TALUS_ROCM_INDEX:-https://nightly.repo.amd.com/rocm/whl-next/}
ROCM_LEGACY_INDEX=https://rocm.nightlies.amd.com/whl-multi-arch/
ROCM_STABLE=${TALUS_ROCM_STABLE:-10.1.0}
ROCM_STABLE_INDEX=https://stable.repo.amd.com/rocm/whl-next/
RUST_FALLBACK=1.99.1

YES=0 DRY=0 ADVANCED=0 ALLOW_SYS=0 FORCE=0 BUILD_ONLY=0 NO_SETUP=0 SRC=""
while [ $# -gt 0 ]; do
  case $1 in
    --yes|-y) YES=1 ;;
    --dry-run|-n) DRY=1 ;;
    --advanced) ADVANCED=1 ;;
    --allow-system-changes) ALLOW_SYS=1 ;;
    --force) FORCE=1 ;;
    --build-only) BUILD_ONLY=1 ;;
    --no-setup) NO_SETUP=1 ;;
    --src) SRC=$2; shift ;;
    -h|--help) sed -n '2,27p' "$0" 2>/dev/null || echo "see https://github.com/nsbradley88/talus-public"; return 0 ;;
    *) echo "install.sh: unknown option $1" >&2; return 2 ;;
  esac
  shift
done

# ---- output and questions (answers come from the terminal, not from the pipe that carries this script)
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C=$(printf '\033[36m') G=$(printf '\033[32m') Y=$(printf '\033[33m') R=$(printf '\033[31m') B=$(printf '\033[1m') D=$(printf '\033[90m') N=$(printf '\033[0m')
else C="" G="" Y="" R="" B="" D="" N=""; fi
T_START=$(date +%s) T_STEP=""
took() { [ -z "$T_STEP" ] || note "($(( $(date +%s) - T_STEP )) s)"; T_STEP=$(date +%s); }
step() { took; printf '\n%s[%s/5]%s %s%s%s\n' "$C" "$1" "$N" "$B" "$2" "$N"; }
ok()   { printf '  %s[ ok ]%s %-22s %s\n' "$G" "$N" "$1" "$2"; }
wrn()  { printf '  %s[warn]%s %-22s %s\n' "$Y" "$N" "$1" "$2"; }
bad()  { printf '  %s[FAIL]%s %-22s %s\n' "$R" "$N" "$1" "$2"; }
say()  { printf '  %s\n' "$*"; }
note() { printf '  %s%s%s\n' "$D" "$*" "$N"; }
die()  { printf '\n  %serror:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }
TTY=""
if [ "$YES" = 0 ] && (: </dev/tty) 2>/dev/null; then TTY=/dev/tty; fi
# ask "question" default(y|n) -> 0 for yes
ask() {
  if [ -z "$TTY" ]; then
    printf '  %s [%s] (default)\n' "$1" "$2"; [ "$2" = y ]; return
  fi
  hint="y/N"; [ "$2" = y ] && hint="Y/n"
  printf '  %s [%s%s%s] ' "$1" "$C" "$hint" "$N"
  read -r a <"$TTY" || a=""
  case $a in [yY]*) return 0 ;; [nN]*) return 1 ;; *) [ "$2" = y ] ;; esac
}
# run a command, or print it in a dry run
run() {
  if [ "$DRY" = 1 ]; then printf '  %s[dry-run] would run:%s %s\n' "$C" "$N" "$*"; return 0; fi
  "$@"
}
have() { command -v "$1" >/dev/null 2>&1; }
# run a long command with its output in a log file; on failure show the end of the log
logged() {
  name=$1; shift
  if [ "$DRY" = 1 ]; then run "$@"; return 0; fi
  mkdir -p "$DATA/logs"
  if "$@" >"$DATA/logs/$name.log" 2>&1; then return 0; fi
  tail -n 30 "$DATA/logs/$name.log" | sed 's/^/    /'
  die "$name failed; the whole log is $DATA/logs/$name.log"
}
SUDO=""
if [ "$(id -u)" != 0 ]; then SUDO=sudo; fi

printf '\n%s Talus %s %sinstaller%s%s\n' "$(printf '\033[30;46;1m')" "$N" "$B" "$N" "$( [ "$DRY" = 1 ] && echo " (dry run: nothing is changed)" )"

# ---- 1. this computer
step 1 "This computer"
[ "$(uname -s)" = Linux ] || die "Talus runs on Linux only."
ARCH=$(uname -m)
OS_ID="" OS_LIKE="" OS_NAME="Linux"
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  OS_ID=$(. /etc/os-release; echo "${ID:-}") OS_LIKE=$(. /etc/os-release; echo "${ID_LIKE:-}")
  # shellcheck disable=SC1091
  OS_NAME=$(. /etc/os-release; echo "${PRETTY_NAME:-Linux}")
fi
if [ "$ARCH" = x86_64 ]; then ok "Operating system" "$OS_NAME, kernel $(uname -r)"; else bad "Operating system" "$OS_NAME on $ARCH (x86-64 needed)"; [ "$FORCE" = 1 ] || die "unsupported CPU architecture"; fi
GFX=""
for p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
  [ -r "$p" ] || continue
  v=$(awk '$1=="gfx_target_version"{print $2}' "$p")
  if [ -n "$v" ] && [ "$v" != 0 ]; then GFX="$GFX $(awk -v v="$v" 'BEGIN{printf "gfx%d%d%x", int(v/10000), int(v/100)%100, v%100}')"; fi
done
if [ -z "$GFX" ] && have rocminfo; then GFX=$(rocminfo 2>/dev/null | awk '/Name: *gfx/{print $2}' | sort -u | tr '\n' ' '); fi
CPU=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo 2>/dev/null || echo "?")
case " $GFX " in
  *" gfx1151 "*) ok "GPU" "gfx1151 (Radeon 8060S) in $CPU" ;;
  "  ") bad "GPU" "no AMD GPU compute device (/dev/kfd) found ($CPU)"; [ "$FORCE" = 1 ] || [ "$BUILD_ONLY" = 1 ] || die "Talus needs a Strix Halo GPU (gfx1151) with the amdgpu driver. --force continues anyway." ;;
  *) bad "GPU" "found$GFX, Talus is tuned for gfx1151 only"; [ "$FORCE" = 1 ] || [ "$BUILD_ONLY" = 1 ] || die "unsupported GPU. --force continues anyway." ;;
esac
MEM_GIB=$(awk '/^MemTotal:/{printf "%d", $2/1048576}' /proc/meminfo)
if [ "$MEM_GIB" -ge 120 ]; then ok "Memory" "$MEM_GIB GiB"; else wrn "Memory" "$MEM_GIB GiB (Talus is built for 128 GB; only Qwen3.8-27B may fit)"; fi
mkdir_p_parent=$DATA; while [ ! -d "$mkdir_p_parent" ]; do mkdir_p_parent=$(dirname "$mkdir_p_parent"); done
FREE_GB=$(df -Pk "$mkdir_p_parent" | awk 'NR==2{printf "%d", $4/1000000}')
if [ "$FREE_GB" -ge 40 ]; then ok "Disk" "$FREE_GB GB free in $mkdir_p_parent"; else wrn "Disk" "$FREE_GB GB free in $mkdir_p_parent (the build needs ~15 GB, models 17-120 GB each)"; fi
for g in render video; do
  if getent group "$g" >/dev/null 2>&1 && ! id -nG | tr ' ' '\n' | grep -qx "$g"; then
    wrn "GPU access" "you are not in the $g group: sudo usermod -aG render,video \$USER, then log in again"
    break
  fi
done

# ---- 2. build tools
step 2 "Build tools"
PM="" PKGS=""
case " $OS_ID $OS_LIKE " in
  *" ubuntu "*|*" debian "*) PM="apt-get"; PKGS="build-essential git curl python3 python3-venv cmake ninja-build pkg-config" ;;
  *" fedora "*|*" rhel "*) PM="dnf"; PKGS="gcc gcc-c++ make git curl python3 cmake ninja-build" ;;
  *" arch "*|*" cachyos "*|*" endeavouros "*|*" manjaro "*) PM="pacman"; PKGS="base-devel git curl python cmake ninja" ;;
  *" opensuse-tumbleweed "*|*" suse "*|*" opensuse "*) PM="zypper"; PKGS="gcc gcc-c++ make git curl python3 cmake ninja" ;;
esac
MISSING=""
for t in cc c++ make git curl python3 bash; do have "$t" || MISSING="$MISSING $t"; done
if have python3 && ! python3 -c 'import venv, ensurepip' >/dev/null 2>&1; then MISSING="$MISSING python3-venv"; fi
if [ -z "$MISSING" ]; then
  ok "Compiler, git, Python" "present"
elif [ -z "$PM" ]; then
  bad "Build tools" "missing:$MISSING"
  die "Install them with your package manager (a C/C++ compiler, make, git, curl, Python 3 with venv), then run this again."
else
  wrn "Build tools" "missing:$MISSING"
  case $PM in
    apt-get) CMD="$SUDO apt-get update && $SUDO apt-get install -y $PKGS" ;;
    dnf) CMD="$SUDO dnf install -y $PKGS" ;;
    pacman) CMD="$SUDO pacman -S --needed --noconfirm $PKGS" ;;
    zypper) CMD="$SUDO zypper install -y $PKGS" ;;
  esac
  say "To install them: $CMD"
  if [ "$DRY" = 1 ]; then run sh -c "$CMD"
  elif ask "Install them now (sudo asks for your password)?" y; then sh -c "$CMD" || die "the package install failed (see above)"
  else die "install the packages above, then run this again"; fi
fi
TOOLS="$DATA/tools"
cmake_ok() { have cmake && cmake --version | awk 'NR==1{split($3,v,"."); exit !(v[1]>3 || (v[1]==3 && v[2]>=24))}'; }
if cmake_ok && have ninja; then ok "CMake, Ninja" "$(cmake --version | awk 'NR==1{print $3}'), $(ninja --version)"
elif [ -x "$TOOLS/bin/cmake" ] && [ -x "$TOOLS/bin/ninja" ]; then ok "CMake, Ninja" "in $TOOLS"
else
  say "CMake 3.24+ and Ninja go into $TOOLS (no sudo) ..."
  run python3 -m venv "$TOOLS"
  run "$TOOLS/bin/pip" install -q cmake ninja
fi
[ -x "$TOOLS/bin/cmake" ] && PATH="$TOOLS/bin:$PATH"
CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}
[ -x "$CARGO_HOME/bin/cargo" ] && PATH="$CARGO_HOME/bin:$PATH"
export PATH
if have rustup; then ok "Rust" "$(rustup --version 2>/dev/null | head -1)"
else
  say "Installing Rust with rustup into $CARGO_HOME (your shell files are not changed) ..."
  if [ "$DRY" = 1 ]; then run sh rustup-init.sh -y --profile minimal --default-toolchain none --no-modify-path
  else
    t=$(mktemp -d)
    curl -fsSL --proto '=https' --tlsv1.2 https://sh.rustup.rs -o "$t/rustup-init.sh" || die "could not download rustup"
    sh "$t/rustup-init.sh" -y --profile minimal --default-toolchain none --no-modify-path >/dev/null || die "rustup failed"
    rm -rf "$t"
    PATH="$CARGO_HOME/bin:$PATH"; export PATH
  fi
fi

# ---- 3. the source and the ROCm SDK
step 3 "Talus source and ROCm SDK"
if [ -z "$SRC" ]; then
  here=$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)
  if [ -n "$here" ] && [ -f "$here/build.sh" ] && [ -f "$here/rust/Cargo.toml" ]; then SRC=$here; fi
fi
if [ -n "$SRC" ]; then
  ok "Source" "$SRC"
elif [ -d "$DATA/src/.git" ]; then
  SRC="$DATA/src"
  ok "Source" "$SRC ($(git -C "$SRC" log -1 --format=%h 2>/dev/null))"
else
  SRC="$DATA/src"
  say "Getting Talus from $REPO ..."
  run mkdir -p "$DATA"
  run git clone --quiet --depth 1 -b "$BRANCH" "$REPO" "$SRC"
fi
TOOLCHAIN=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$SRC/rust/rust-toolchain.toml" 2>/dev/null || true)
TOOLCHAIN=${TOOLCHAIN:-$RUST_FALLBACK}
if rustup toolchain list 2>/dev/null | grep -q "^$TOOLCHAIN"; then ok "Rust toolchain" "$TOOLCHAIN"
else say "Rust $TOOLCHAIN (pinned in rust/rust-toolchain.toml) ..."; run rustup toolchain install "$TOOLCHAIN" --profile minimal -c rustfmt,clippy; fi

# the ROCm version this release was built and measured with: rocm.version in the source (one place to re-pin)
[ -r "$SRC/rocm.version" ] && ROCM_PIN=$(grep -v '^[[:space:]]*#' "$SRC/rocm.version" | grep -m1 . | tr -d ' \t\r')
ROCM_VERSION=${TALUS_ROCM_VERSION:-$ROCM_PIN}
ROCM_ROOT=""
if [ -n "${ROCM_SDK:-}" ]; then
  [ -e "$ROCM_SDK/lib/libamdhip64.so" ] || die "ROCM_SDK=$ROCM_SDK has no lib/libamdhip64.so"
  ROCM_ROOT=$ROCM_SDK
elif [ -x "$DATA/rocm/bin/rocm-sdk" ]; then
  ROCM_ROOT=$("$DATA/rocm/bin/rocm-sdk" path --root 2>/dev/null || true)
fi
if [ -n "$ROCM_ROOT" ]; then
  ok "ROCm SDK" "$ROCM_ROOT"
else
  say "The ROCm SDK $ROCM_VERSION (AMD TheRock, several GB) goes into $DATA/rocm ..."
  run python3 -m venv "$DATA/rocm"
  # pip_rocm VERSION INDEX: the SDK wheels (runtime, libraries, compiler, gfx1151 device code)
  pip_rocm() { "$DATA/rocm/bin/pip" install -q --index-url "$2" "rocm[libraries,devel,device-gfx1151]==$1" 2>"$DATA/rocm/pip-error.log"; }
  if [ "$DRY" = 1 ]; then
    run "$DATA/rocm/bin/pip" install --index-url "$ROCM_INDEX" "rocm[libraries,devel,device-gfx1151]==$ROCM_VERSION"
  elif pip_rocm "$ROCM_VERSION" "$ROCM_INDEX"; then :
  elif [ -z "${TALUS_ROCM_INDEX:-}" ] && pip_rocm "$ROCM_VERSION" "$ROCM_LEGACY_INDEX"; then
    note "(from AMD's older index $ROCM_LEGACY_INDEX)"
  else
    note "$(grep -m1 'No matching distribution' "$DATA/rocm/pip-error.log" || tail -n 1 "$DATA/rocm/pip-error.log")"
    wrn "ROCm SDK" "$ROCM_VERSION (the release's) is unreachable on $ROCM_INDEX"
    say "AMD's stable release $ROCM_STABLE ($ROCM_STABLE_INDEX) can stand in. It is not the version this"
    say "release was measured with; it should work. Or stop and point ROCM_SDK at an SDK you have."
    if [ "$YES" = 1 ] || ask "Install ROCm $ROCM_STABLE instead?" y; then
      pip_rocm "$ROCM_STABLE" "$ROCM_STABLE_INDEX" || die "the ROCm install failed; see $DATA/rocm/pip-error.log"
      wrn "ROCm SDK" "using $ROCM_STABLE instead of $ROCM_VERSION"
      ROCM_VERSION=$ROCM_STABLE
    else
      die "no ROCm SDK installed. Point ROCM_SDK at one you have, then run this again."
    fi
  fi
  run "$DATA/rocm/bin/rocm-sdk" init
  if [ "$DRY" = 1 ]; then ROCM_ROOT="$DATA/rocm/<site-packages>/_rocm_sdk_devel"
  else ROCM_ROOT=$("$DATA/rocm/bin/rocm-sdk" path --root) || die "rocm-sdk did not report its root"; fi
  ok "ROCm SDK" "$ROCM_VERSION in $ROCM_ROOT"
fi

# ---- 4. build and install
step 4 "Build"
BIN="$DATA/bin"
build_kernels() {
  JOBS=$(nproc 2>/dev/null || echo 8)
  # the kernel compile needs ~2 GB of memory per job
  MAXJ=$(( MEM_GIB / 2 )); [ "$MAXJ" -ge 1 ] || MAXJ=1; [ "$JOBS" -le "$MAXJ" ] || JOBS=$MAXJ
  say "Compiling the HIP kernel library ($JOBS jobs; the first build takes a while) ..."
  note "log: $DATA/logs/build-kernels.log"
  logged build-kernels env ROCM_SDK="$ROCM_ROOT" BUILD="$SRC/build" TARGETS=talus_kernels JOBS="$JOBS" bash "$SRC/build.sh"
  took
}
# build_rust CRATE... : release build of these crates, installed into $BIN
build_rust() {
  say "Compiling the Rust programs ($*) ..."
  note "log: $DATA/logs/build-rust.log"
  pk=""; for c in "$@"; do pk="$pk -p $c"; done
  logged build-rust sh -c "cd '$SRC/rust' && cargo build --release --locked$pk"
  run mkdir -p "$BIN"
  for f in "$@"; do run install -m 0755 "$SRC/rust/target/release/$f" "$BIN/$f"; done
}
# a release tarball instead of compiling: TALUS_PREBUILT_URL, or the GitHub release asset of TALUS_RELEASE=<tag>
PREBUILT=${TALUS_PREBUILT_URL:-}
if [ -z "$PREBUILT" ] && [ -n "${TALUS_RELEASE:-}" ]; then
  PREBUILT="https://github.com/nsbradley88/talus-public/releases/download/$TALUS_RELEASE/talus-$TALUS_RELEASE-linux-x86_64-gfx1151.tar.gz"
fi
if [ -n "$PREBUILT" ]; then
  say "Prebuilt release: $PREBUILT"
  if [ "$DRY" = 1 ]; then
    run curl -fL "$PREBUILT" -o talus.tar.gz
    run curl -fsSL "$PREBUILT.sha256" -o talus.tar.gz.sha256
    run tar -xzf talus.tar.gz -C "$DATA" --strip-components=1
  else
    t=$(mktemp -d)
    curl -fL --retry 5 -sS "$PREBUILT" -o "$t/b.tgz" || die "could not download $PREBUILT"
    curl -fsSL --retry 3 "$PREBUILT.sha256" -o "$t/b.sha" || die "the release has no .sha256 next to the archive"
    [ "$(sha256sum "$t/b.tgz" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$t/b.sha")" ] || die "the prebuilt archive does not match its sha256"
    ok "Download" "sha256 verified"
    # what the binaries need: MANIFEST.json in the archive's top folder
    top=$(tar -tzf "$t/b.tgz" | head -n 1 | cut -d/ -f1)
    tar -xzf "$t/b.tgz" -C "$t" "$top/MANIFEST.json" 2>/dev/null || true
    need_rocm="" need_glibc=""
    if [ -r "$t/$top/MANIFEST.json" ]; then
      need_rocm=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("runtime",{}).get("rocm_required",""))' "$t/$top/MANIFEST.json")
      need_glibc=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("runtime",{}).get("glibc_min",""))' "$t/$top/MANIFEST.json")
    fi
    have_glibc=$(ldd --version 2>/dev/null | awk 'NR==1{print $NF}')
    if [ -n "$need_glibc" ] && [ -n "$have_glibc" ] && \
       [ "$(printf '%s\n%s\n' "$need_glibc" "$have_glibc" | sort -V | head -n 1)" != "$need_glibc" ]; then
      wrn "Prebuilt" "needs glibc $need_glibc, this system has $have_glibc: compiling instead"
      PREBUILT=""
    else
      if [ -n "$need_rocm" ] && [ "$need_rocm" != "$ROCM_VERSION" ]; then
        wrn "Prebuilt" "built for ROCm $need_rocm, this install has $ROCM_VERSION (usually fine; report problems with talus doctor --report)"
      fi
      mkdir -p "$DATA"
      tar -xzf "$t/b.tgz" -C "$DATA" --strip-components=1 || die "could not unpack the release"
      ok "Prebuilt" "$(basename "$PREBUILT" .tar.gz) unpacked into $DATA"
    fi
    rm -rf "$t"
  fi
  # older release archives have no talus-setup: build just that (small) program
  if [ -n "$PREBUILT" ] && [ ! -x "$BIN/talus-setup" ] && [ "$DRY" = 0 ]; then
    build_rust talus-setup
  fi
fi
if [ -z "$PREBUILT" ]; then
  build_kernels
  build_rust talus-cli talus-server talus-tui talus-setup
  run install -m 0755 "$SRC/build/libtalus_kernels.so" "$BIN/libtalus_kernels.so"
fi
# the programs find ROCm through this link (their RUNPATH is $ORIGIN/../rocm-sdk/lib): no LD_LIBRARY_PATH needed
run ln -sfn "$ROCM_ROOT" "$DATA/rocm-sdk"
run mkdir -p "$HOME/.local/bin"
run ln -sf "$BIN/talus-setup" "$HOME/.local/bin/talus"
run ln -sf "$BIN/talus-tui" "$HOME/.local/bin/talus-tui"
ok "Installed" "$BIN (commands: talus, talus-tui in ~/.local/bin)"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) note "Add ~/.local/bin to your PATH to type \`talus\` directly (Ubuntu does this at your next login)." ;; esac

took
note "Installed in $(( ($(date +%s) - T_START + 30) / 60 )) min."
[ "$BUILD_ONLY" = 1 ] && return 0
[ "$NO_SETUP" = 1 ] && { say "Next: talus setup"; return 0; }

# ---- 5. models and settings: the Rust wizard
step 5 "Models and settings (talus setup)"
SETUP="$BIN/talus-setup"
[ -x "$SETUP" ] || SETUP="$SRC/rust/target/release/talus-setup"
set -- setup --src "$SRC" --rocm-sdk "$ROCM_ROOT"
[ "$YES" = 1 ] && set -- "$@" --yes
[ "$DRY" = 1 ] && set -- "$@" --dry-run
[ "$ADVANCED" = 1 ] && set -- "$@" --advanced
[ "$ALLOW_SYS" = 1 ] && set -- "$@" --allow-system-changes
[ "$FORCE" = 1 ] && set -- "$@" --force
if [ -x "$SETUP" ]; then
  exec "$SETUP" "$@"
else
  printf '  %s[dry-run] would run:%s talus %s\n' "$C" "$N" "$*"
fi
}

main "$@"
