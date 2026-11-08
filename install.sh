#!/bin/sh
# Talus installer for AMD Strix Halo (Ryzen AI Max, Radeon 8060S / gfx1151) on Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/nsbradley88/talus-public/main/install.sh | bash
#
# What it does, each step skipped when already done (run it again any time):
#   1. checks this computer: Linux, x86-64, the gfx1151 GPU, 128 GB of memory, free disk;
#   2. installs what it needs: git, curl, Python 3 with venv (asks before sudo); to compile (no prebuilt release),
#      also a C/C++ compiler, CMake and Ninja (into your home when the system's are missing or old) and Rust
#      (rustup, the pinned toolchain). A prebuilt release (TALUS_RELEASE / TALUS_PREBUILT_URL) needs none of those;
#   3. the ROCm SDK into your home (AMD TheRock wheels in ~/.local/share/talus/rocm), or one you already have
#      (ROCM_SDK=/path);
#   4. builds Talus (the HIP kernel library and the Rust programs), or unpacks the prebuilt release, and installs it
#      to ~/.local/share/talus/bin, with `talus` and `talus-tui` in ~/.local/bin;
#   5. starts `talus setup`: choose models, download them, the kernel parameters (shown and asked, never silent),
#      the run configs and an optional user service (with --yes only when --service is given).
# Nothing is installed system-wide except the distribution packages in step 2, and only after you say yes.
#
# Options: --yes (take every default) --dry-run (print the actions, change nothing) --advanced (all choices)
#          --allow-system-changes (with --yes: apply the kernel parameters too) --force (continue on unsupported
#          hardware) --from-source (compile instead of installing the latest release's prebuilt programs)
#          --build-only (steps 1-4, no setup; what `talus update` runs, so the GPU check only warns) --src DIR (a Talus checkout to build) --no-setup
#          passed on to `talus setup`: --models-dir DIR (where the models go) --no-power-profile (leave the ACPI
#          platform profile alone) --service / --no-service (install the user service that starts the server at login;
#          with --yes it is installed only with --service)
# Environment: TALUS_HOME, ROCM_SDK (use an SDK you have), TALUS_ROCM_VERSION / TALUS_ROCM_INDEX (override the pin
#              in rocm.version / AMD's nightly index), TALUS_ROCM_STABLE, TALUS_REPO, TALUS_BRANCH (default: the newest release tag),
#              TALUS_RELEASE=<tag> or TALUS_PREBUILT_URL=<tar.gz> (a release tarball instead of compiling),
#              TALUS_RELEASES_URL (another copy of the release files), TALUS_FROM_SOURCE=1 (same as --from-source; also
#              makes `talus update` compile on an install from a release tarball; TALUS_NO_PREBUILT=1 is the old name),
#              TALUS_JOBS=N (parallel compile jobs; default: CPU threads, at most one per 2 GiB of available memory),
#              TALUS_KEEP_BUILD=1 (keep a build directory an earlier attempt left),
#              HF_TOKEN (optional, never stored).

# Everything is inside main(), so a download cut short runs nothing.
main() {
set -eu

REPO=${TALUS_REPO:-https://github.com/nsbradley88/talus-public}
BRANCH=${TALUS_BRANCH:-}   # empty: the release tag of the binaries (release_ref below), not main
DATA=${TALUS_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/talus}
ROCM_PIN=10.2.0a20261014   # used only when the source has no rocm.version (the pin; read once the source is here)
ROCM_INDEX=${TALUS_ROCM_INDEX:-https://nightly.repo.amd.com/rocm/whl-next/}
ROCM_LEGACY_INDEX=https://rocm.nightlies.amd.com/whl-multi-arch/
ROCM_STABLE=${TALUS_ROCM_STABLE:-10.1.0}
ROCM_STABLE_INDEX=https://stable.repo.amd.com/rocm/whl-next/
RUST_FALLBACK=1.99.1

YES=0 DRY=0 ADVANCED=0 ALLOW_SYS=0 FORCE=0 BUILD_ONLY=0 NO_SETUP=0 SRC=""
FROM_SOURCE=0; [ -z "${TALUS_FROM_SOURCE:-}${TALUS_NO_PREBUILT:-}" ] || FROM_SOURCE=1
MODELS_DIR="" NO_POWER=0 SERVICE=""   # passed on to talus setup; SERVICE: "", --service or --no-service
while [ $# -gt 0 ]; do
  case $1 in
    --yes|-y) YES=1 ;;
    --dry-run|-n) DRY=1 ;;
    --advanced) ADVANCED=1 ;;
    --allow-system-changes) ALLOW_SYS=1 ;;
    --force) FORCE=1 ;;
    --build-only) BUILD_ONLY=1 ;;
    --from-source) FROM_SOURCE=1 ;;
    --no-setup) NO_SETUP=1 ;;
    --src) [ $# -ge 2 ] || { echo "install.sh: --src needs a directory" >&2; return 2; }; SRC=$2; shift ;;
    --models-dir) [ $# -ge 2 ] || { echo "install.sh: --models-dir needs a directory" >&2; return 2; }; MODELS_DIR=$2; shift ;;
    --no-power-profile) NO_POWER=1 ;;
    --service|--no-service) SERVICE=$1 ;;
    -h|--help) sed -n '2,31p' "$0" 2>/dev/null || echo "see https://github.com/nsbradley88/talus-public"; return 0 ;;
    *) echo "install.sh: unknown option $1" >&2; return 2 ;;
  esac
  shift
done

# ---- output and questions (answers come from the terminal, not from the pipe that carries this script)
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C=$(printf '\033[36m') G=$(printf '\033[32m') Y=$(printf '\033[33m') R=$(printf '\033[31m') B=$(printf '\033[1m') D=$(printf '\033[90m') N=$(printf '\033[0m')
  BN=$(printf '\033[30;46;1m')   # the banner
else C="" G="" Y="" R="" B="" D="" N="" BN=""; fi   # NO_COLOR or not a terminal: no escape codes at all
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
# The source to clone: the release the binaries come from, so data files, tools and the build match them (talus#50
# A3). TALUS_BRANCH (any branch or tag) overrides; then TALUS_RELEASE, the tag in TALUS_PREBUILT_URL, the newest
# vX.Y.Z tag of the repository; main only when the repository has no release tag.
release_ref() {
  if [ -n "$BRANCH" ]; then echo "$BRANCH"; return; fi
  if [ -n "${TALUS_RELEASE:-}" ]; then echo "$TALUS_RELEASE"; return; fi
  case ${TALUS_PREBUILT_URL:-} in
    */releases/download/*/*) echo "$TALUS_PREBUILT_URL" | sed 's#.*/releases/download/\([^/]*\)/.*#\1#'; return ;;
  esac
  t=$(git ls-remote --tags --refs "$REPO" 'v*' 2>/dev/null | sed 's#.*refs/tags/##' \
      | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n 1)
  echo "${t:-main}"
}
# run a long command with its output in a log file; on failure show the end of the log
logged() {
  name=$1; shift
  if [ "$DRY" = 1 ]; then run "$@"; return 0; fi
  mkdir -p "$DATA/logs"
  if "$@" >"$DATA/logs/$name.log" 2>&1; then return 0; fi
  # a parallel build keeps compiling after the first failure, so the tail is often warnings from other files
  # (talus-public#2): show the first error first
  first=$(grep -n -m1 -E '^FAILED:|(^|[[:space:]])(fatal )?error(\[E[0-9]+\])?:|Killed|terminated by signal' \
          "$DATA/logs/$name.log" | cut -d: -f1)
  if [ -n "$first" ]; then
    note "first error (line $first of the log):"
    sed -n "$(( first > 3 ? first - 3 : 1 )),$(( first + 30 ))p" "$DATA/logs/$name.log" | sed 's/^/    /'
    note "last lines:"
  fi
  tail -n 30 "$DATA/logs/$name.log" | sed 's/^/    /'
  # a compiler killed for lack of memory: fewer jobs
  if grep -q -E 'Killed|terminated by signal|[Oo]ut of memory|Cannot allocate memory' "$DATA/logs/$name.log"; then
    j=$(( ${JOBS:-2} / 2 )); [ "$j" -ge 1 ] || j=1
    note "A compiler was killed, most likely out of memory (${JOBS:-?} jobs). Run again with fewer: TALUS_JOBS=$j"
  fi
  die "$name failed; the whole log is $DATA/logs/$name.log"
}
SUDO=""
if [ "$(id -u)" != 0 ]; then SUDO=sudo; fi

printf '\n%s Talus %s %sinstaller%s%s\n' "$BN" "$N" "$B" "$N" "$( [ "$DRY" = 1 ] && echo " (dry run: nothing is changed)" )"

# a release tarball instead of compiling: TALUS_PREBUILT_URL, or the GitHub release asset of TALUS_RELEASE=<tag>.
# Known before step 2: a prebuilt install needs no compiler, CMake, Ninja or Rust (talus#18 itest18).
PREBUILT=${TALUS_PREBUILT_URL:-}
RELEASES=${TALUS_RELEASES_URL:-https://github.com/nsbradley88/talus-public/releases/download}
if [ -z "$PREBUILT" ] && [ -n "${TALUS_RELEASE:-}" ]; then
  PREBUILT="$RELEASES/$TALUS_RELEASE/talus-$TALUS_RELEASE-linux-x86_64-gfx1151.tar.gz"
fi
# `talus update` (--build-only) on an install that came from a release tarball (it left MANIFEST.json): the new release's
# tarball again, not a compile (that machine has no compiler, and the tarball is the build we tested). A source
# install has no MANIFEST.json and compiles. TALUS_FROM_SOURCE=1 compiles anyway.
if [ -z "$PREBUILT" ] && [ "$BUILD_ONLY" = 1 ] && [ "$FROM_SOURCE" = 0 ] && [ -f "$DATA/MANIFEST.json" ]; then
  tag=$(git -C "${SRC:-$DATA/src}" describe --tags --exact-match HEAD 2>/dev/null || true)
  case $tag in
    v[0-9]*.[0-9]*.[0-9]*-*|"") ;;
    v[0-9]*.[0-9]*.[0-9]*)
      PREBUILT="$RELEASES/$tag/talus-$tag-linux-x86_64-gfx1151.tar.gz"
      if [ "$DRY" = 0 ] && ! curl -fsIL --retry 2 "$PREBUILT.sha256" >/dev/null 2>&1; then
        die "the prebuilt programs of $tag are not available yet ($PREBUILT). Your programs were not changed: run \`talus update --rebuild\` again later (TALUS_FROM_SOURCE=1 compiles from source instead)."
      fi ;;
  esac
fi

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
# build_jobs: parallel compile jobs. TALUS_JOBS=N wins; otherwise one per CPU thread, at most one per 2 GiB of
# *available* memory (MemAvailable, not MemTotal: a desktop session, a browser or a running model already hold some;
# talus-public#2). The kernel files peak at 1.2-2.5 GB each, 38 GiB summed over all 164.
build_jobs() {
  if [ -n "${TALUS_JOBS:-}" ]; then
    case $TALUS_JOBS in *[!0-9]*|0*) die "TALUS_JOBS must be a whole number of 1 or more, not '$TALUS_JOBS'" ;; esac
    echo "$TALUS_JOBS"; return
  fi
  j=$(nproc 2>/dev/null || echo 8)
  avail=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
  [ -n "$avail" ] || avail=$MEM_GIB
  m=$(( avail / 2 )); [ "$m" -ge 1 ] || m=1
  [ "$j" -le "$m" ] || j=$m
  echo "$j"
}
JOBS=$(build_jobs) || exit 1
mkdir_p_parent=$DATA; while [ ! -d "$mkdir_p_parent" ]; do mkdir_p_parent=$(dirname "$mkdir_p_parent"); done
FREE_GB=$(df -Pk "$mkdir_p_parent" | awk 'NR==2{printf "%d", $4/1000000}')
if [ "$FREE_GB" -ge 40 ]; then ok "Disk" "$FREE_GB GB free in $mkdir_p_parent"; else wrn "Disk" "$FREE_GB GB free in $mkdir_p_parent (the build needs ~15 GB, models 17-120 GB each)"; fi
for g in render video; do
  if getent group "$g" >/dev/null 2>&1 && ! id -nG | tr ' ' '\n' | grep -qx "$g"; then
    wrn "GPU access" "you are not in the $g group: sudo usermod -aG render,video \$USER, then log in again"
    break
  fi
done

# The default install: the latest release's prebuilt programs. It compiles only when asked (--from-source), when pointed
# at source (TALUS_BRANCH, --src, running from a checkout), on an install already built from source, or, with a message,
# when this machine has no tarball. Decided before step 2: a prebuilt install needs no build tools.
default_install() {
  [ -z "$PREBUILT" ] && [ "$BUILD_ONLY" = 0 ] || return 0
  have curl || die "this installer needs curl to find the latest release: install curl and run it again."
  here=$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)
  chosen=""
  if [ "$FROM_SOURCE" = 1 ]; then chosen="you asked for it"
  elif [ -n "$BRANCH" ]; then chosen="TALUS_BRANCH=$BRANCH"
  elif [ -n "$SRC" ]; then chosen="--src $SRC"
  elif [ -n "$here" ] && [ -f "$here/build.sh" ] && [ -f "$here/rust/Cargo.toml" ]; then chosen="running from the checkout $here"
  elif [ -d "$DATA/src/.git" ] && [ ! -f "$DATA/MANIFEST.json" ]; then chosen="this install was built from source"
  fi
  if [ -n "$chosen" ]; then ok "Install" "from source ($chosen)"; return 0; fi
  if [ "$ARCH" != x86_64 ]; then compile_instead "there are no prebuilt programs for $ARCH"; return 0; fi
  tag=""
  if [ -n "${TALUS_REPO:-}" ]; then
    tag=$(git ls-remote --tags --refs "$REPO" 'v*' 2>/dev/null | sed 's#.*refs/tags/##' \
        | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n 1 || true)
  else
    url=$(curl -fsSIL --connect-timeout 15 --retry 2 -o /dev/null -w '%{url_effective}' "$REPO/releases/latest") \
      || die "could not reach $REPO to find the latest release. Nothing was changed: check the network and run this again."
    case $url in */releases/tag/v*) tag=${url##*/} ;; esac
  fi
  if [ -z "$tag" ]; then compile_instead "no release is published yet"; return 0; fi
  try="$RELEASES/$tag/talus-$tag-linux-x86_64-gfx1151.tar.gz"
  rc=0; curl -fsIL --connect-timeout 15 --retry 2 "$try.sha256" >/dev/null 2>&1 || rc=$?
  case $rc in
    0) TALUS_RELEASE=$tag PREBUILT=$try; ok "Install" "prebuilt programs of $tag (--from-source compiles instead)" ;;
    22|37) compile_instead "$tag has no prebuilt programs for this machine" ;;
    *) die "could not reach $try. Nothing was changed: check the network and run this again." ;;
  esac
}
compile_instead() {
  wrn "Install" "$1: compiling from source instead (about 40 minutes, and it needs a compiler, CMake, Ninja and Rust)"
}
default_install

# ---- 2. tools
step 2 "Tools"
PM="" PKGS="" BASE_PKGS=""
case " $OS_ID $OS_LIKE " in
  *" ubuntu "*|*" debian "*) PM="apt-get"; BASE_PKGS="git curl ca-certificates python3 python3-venv"; PKGS="build-essential $BASE_PKGS cmake ninja-build pkg-config" ;;
  *" fedora "*|*" rhel "*) PM="dnf"; BASE_PKGS="git curl python3"; PKGS="gcc gcc-c++ make $BASE_PKGS cmake ninja-build" ;;
  *" arch "*|*" cachyos "*|*" endeavouros "*|*" manjaro "*) PM="pacman"; BASE_PKGS="git curl python"; PKGS="base-devel $BASE_PKGS cmake ninja" ;;
  *" opensuse-tumbleweed "*|*" suse "*|*" opensuse "*) PM="zypper"; BASE_PKGS="git curl python3"; PKGS="gcc gcc-c++ make $BASE_PKGS cmake ninja" ;;
esac
# need_pkgs LABEL "TOOLS" "PACKAGES" WHAT: install the distribution packages when one of the tools is missing
need_pkgs() {
  MISSING=""
  for t in $2; do have "$t" || MISSING="$MISSING $t"; done
  # Python's venv: the ROCm SDK wheels (and CMake/Ninja for a build) go into virtual environments
  if have python3 && [ "$NEED_VENV" = 1 ] && ! python3 -c 'import venv, ensurepip' >/dev/null 2>&1; then MISSING="$MISSING python3-venv"; fi
  if [ -z "$MISSING" ]; then ok "$1" "present"; return 0; fi
  if [ -z "$PM" ]; then
    bad "$1" "missing:$MISSING"
    die "Install them with your package manager ($4), then run this again."
  fi
  wrn "$1" "missing:$MISSING"
  case $PM in
    apt-get) CMD="$SUDO apt-get update && $SUDO apt-get install -y $3" ;;
    dnf) CMD="$SUDO dnf install -y $3" ;;
    pacman) CMD="$SUDO pacman -S --needed --noconfirm $3" ;;
    zypper) CMD="$SUDO zypper install -y $3" ;;
  esac
  say "To install them: $CMD"
  if [ "$DRY" = 1 ]; then run sh -c "$CMD"
  elif ask "Install them now (sudo asks for your password)?" y; then sh -c "$CMD" || die "the package install failed (see above)"
  else die "install the packages above, then run this again"; fi
}
TOOLS="$DATA/tools"
CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}
[ -x "$TOOLS/bin/cmake" ] && PATH="$TOOLS/bin:$PATH"
[ -x "$CARGO_HOME/bin/cargo" ] && PATH="$CARGO_HOME/bin:$PATH"
export PATH
NEED_VENV=1
if [ -n "$PREBUILT" ] && { [ -n "${ROCM_SDK:-}" ] || [ -x "$DATA/rocm/bin/rocm-sdk" ]; }; then NEED_VENV=0; fi
# build_tools: what compiling needs (a C/C++ compiler, make, CMake 3.24+, Ninja, rustup). Run in this step for a
# source build; for a prebuilt release only if it has to compile after all (step 4).
BUILD_TOOLS=0
build_tools() {
  [ "$BUILD_TOOLS" = 0 ] || return 0
  BUILD_TOOLS=1 NEED_VENV=1
  need_pkgs "Compiler, git, Python" "cc c++ make git curl python3 bash" "$PKGS" "a C/C++ compiler, make, git, curl, Python 3 with venv"
  cmake_ok() { have cmake && cmake --version | awk 'NR==1{split($3,v,"."); exit !(v[1]>3 || (v[1]==3 && v[2]>=24))}'; }
  if cmake_ok && have ninja; then ok "CMake, Ninja" "$(cmake --version | awk 'NR==1{print $3}'), $(ninja --version)"
  elif [ -x "$TOOLS/bin/cmake" ] && [ -x "$TOOLS/bin/ninja" ]; then ok "CMake, Ninja" "in $TOOLS"
  else
    say "CMake 3.24+ and Ninja go into $TOOLS (no sudo) ..."
    run python3 -m venv "$TOOLS"
    run "$TOOLS/bin/pip" install -q cmake ninja
  fi
  [ -x "$TOOLS/bin/cmake" ] && PATH="$TOOLS/bin:$PATH"
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
  [ "$SRC_READY" = 1 ] && rust_toolchain
  return 0
}
# rust_toolchain: the toolchain pinned in the source (rust/rust-toolchain.toml); needs the source and rustup
SRC_READY=0
rust_toolchain() {
  TOOLCHAIN=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$SRC/rust/rust-toolchain.toml" 2>/dev/null || true)
  TOOLCHAIN=${TOOLCHAIN:-$RUST_FALLBACK}
  if rustup toolchain list 2>/dev/null | grep -q "^$TOOLCHAIN"; then ok "Rust toolchain" "$TOOLCHAIN"
  else say "Rust $TOOLCHAIN (pinned in rust/rust-toolchain.toml) ..."; run rustup toolchain install "$TOOLCHAIN" --profile minimal -c rustfmt,clippy; fi
}
if [ -n "$PREBUILT" ]; then
  need_pkgs "git, curl, Python" "git curl python3" "$BASE_PKGS" "git, curl, Python 3 with venv"
  ok "Build tools" "not needed: prebuilt release (no compiler, CMake or Rust)"
else
  build_tools
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
  # installing a release over an older one: its source goes with it (data files, tools, update)
  if [ -n "$PREBUILT" ] && [ -n "${TALUS_RELEASE:-}" ] \
     && [ "$(git -C "$SRC" describe --tags --exact-match HEAD 2>/dev/null || true)" != "$TALUS_RELEASE" ]; then
    run git -C "$SRC" fetch --quiet --depth 1 origin tag "$TALUS_RELEASE" \
      && run git -C "$SRC" -c advice.detachedHead=false checkout --quiet "$TALUS_RELEASE" \
      || wrn "Source" "$SRC stays where it is; it could not move to $TALUS_RELEASE"
  fi
  ok "Source" "$SRC ($(git -C "$SRC" log -1 --format=%h 2>/dev/null))"
else
  SRC="$DATA/src"
  REF=$(release_ref)
  say "Getting Talus $REF from $REPO ..."
  run mkdir -p "$DATA"
  # a release tag is a detached HEAD: no advice about it (talus#18)
  run git -c advice.detachedHead=false clone --quiet --depth 1 -b "$REF" "$REPO" "$SRC"
fi
SRC_READY=1
[ "$BUILD_TOOLS" = 1 ] && rust_toolchain

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

# sysrocm_env VAR...: warn about directories in these path variables that hold ROCm files from outside the SDK (a
# system ROCm). CPATH & co. come before the SDK's headers in a build, LD_LIBRARY_PATH before its libraries at run time.
sysrocm_env() {
  [ "$DRY" = 0 ] || return 0
  sdk_real=$(cd "$ROCM_ROOT" 2>/dev/null && pwd -P) || sdk_real=$ROCM_ROOT
  for v in "$@"; do
    eval "val=\${$v:-}"
    old_ifs=$IFS; IFS=:; set -f
    for d in $val; do
      [ -n "$d" ] && { [ -r "$d/hip/hip_runtime.h" ] || [ -e "$d/libamdhip64.so" ] || ls "$d"/libamdhip64.so.* >/dev/null 2>&1; } || continue
      dr=$(cd "$d" 2>/dev/null && pwd -P) || dr=$d
      case $dr/ in "$sdk_real"/*) continue ;; esac
      wrn "System ROCm" "$v has $d (ROCm files outside the SDK; they can shadow it): unset $v for Talus"
    done
    set +f; IFS=$old_ifs
  done
}
sysrocm_env LD_LIBRARY_PATH

# ---- 4. build and install
step 4 "Build"
BIN="$DATA/bin"
# clear_stale_build: a build directory left by an earlier attempt starts over. CMake keeps the compiler and SDK of its
# first configure, so a directory configured for another source path or ROCm SDK, without a CMakeCache.txt, or left by
# a failed or interrupted build (the marker below) is removed (talus-public#2: a failed first attempt may leave a bad
# build directory). TALUS_KEEP_BUILD=1 keeps it.
clear_stale_build() {
  b="$SRC/build"; c="$b/CMakeCache.txt"; why=""
  [ -d "$b" ] && [ "${TALUS_KEEP_BUILD:-0}" != 1 ] || return 0
  if [ -e "$DATA/logs/build-kernels.unfinished" ]; then why="the last build did not finish"
  elif [ ! -f "$c" ]; then why="it has no CMakeCache.txt"
  else
    home=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$c")
    root=$(sed -n 's/^CMAKE_HIP_COMPILER_ROCM_ROOT:[A-Z]*=//p' "$c")
    same() { [ "$1" = "$2" ] || [ "$(cd "$1" 2>/dev/null && pwd -P)" = "$(cd "$2" 2>/dev/null && pwd -P)" ]; }
    if ! same "$home" "$SRC"; then why="it was configured for $home"
    elif [ -n "$root" ] && ! same "$root" "$ROCM_ROOT"; then why="it was configured with the ROCm SDK $root"; fi
  fi
  [ -n "$why" ] || return 0
  note "Removing $b from an earlier attempt ($why); the kernel library builds from scratch"
  run rm -rf "$b"
}
build_kernels() {
  clear_stale_build
  # a system ROCm's headers (Fedora's rocm-hip-devel: /usr/include/hip) came before the SDK's in builds up to v0.1.3
  # and broke them (talus-public#2); build.sh now puts the SDK's first (-isystem)
  for d in /usr/include/hip /usr/local/include/hip; do
    [ -r "$d/hip_runtime.h" ] || continue
    if grep -q -- '-isystem $SDK/include' "$SRC/build.sh" 2>/dev/null; then
      note "System ROCm headers in $d (a distribution package): the build uses the SDK's ahead of them."
    else
      wrn "System ROCm" "headers in $d can shadow the SDK's and fail this build; TALUS_RELEASE=<tag> installs prebuilt"
    fi
  done
  sysrocm_env CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
  say "Compiling the HIP kernel library ($JOBS jobs; the first build takes a while; TALUS_JOBS=N sets the number) ..."
  note "log: $DATA/logs/build-kernels.log"
  [ "$DRY" = 1 ] || { mkdir -p "$DATA/logs"; : > "$DATA/logs/build-kernels.unfinished"; }
  logged build-kernels env ROCM_SDK="$ROCM_ROOT" BUILD="$SRC/build" TARGETS=talus_kernels JOBS="$JOBS" bash "$SRC/build.sh"
  rm -f "$DATA/logs/build-kernels.unfinished"
  took
}
# build_rust CRATE... : release build of these crates, installed into $BIN
build_rust() {
  say "Compiling the Rust programs ($*) ..."
  note "log: $DATA/logs/build-rust.log"
  pk=""; for c in "$@"; do pk="$pk -p $c"; done
  logged build-rust sh -c "cd '$SRC/rust' && cargo build --release --locked -j $JOBS$pk"
  run mkdir -p "$BIN"
  for f in "$@"; do run install -m 0755 "$SRC/rust/target/release/$f" "$BIN/$f"; done
}
# a prebuilt release (PREBUILT, set before step 2)
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
      # the source next to it should be the commit the binaries were built from (data files, tools, update)
      bin_commit=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("commit",""))' "$DATA/MANIFEST.json" 2>/dev/null || true)
      src_commit=$(git -C "$SRC" rev-parse HEAD 2>/dev/null || true)
      if [ -n "$bin_commit" ] && [ -n "$src_commit" ] && [ "$bin_commit" != "$src_commit" ]; then
        wrn "Source" "$SRC is at ${src_commit%"${src_commit#????????????}"}, the binaries were built from ${bin_commit%"${bin_commit#????????????}"} (TALUS_BRANCH=<their tag> to match)"
      fi
    fi
    rm -rf "$t"
  fi
  # older release archives have no talus-setup: build just that (small) program
  if [ -n "$PREBUILT" ] && [ ! -x "$BIN/talus-setup" ] && [ "$DRY" = 0 ]; then
    build_tools
    build_rust talus-setup
  fi
fi
if [ -z "$PREBUILT" ]; then
  build_tools   # no-op unless a prebuilt release fell back to compiling (glibc too old)
  build_kernels
  build_rust talus-cli talus-server talus-tui talus-setup
  run install -m 0755 "$SRC/build/libtalus_kernels.so" "$BIN/libtalus_kernels.so"
  run rm -f "$DATA/MANIFEST.json"   # these programs did not come from a release tarball any more
fi
# the hipBLASLt prefill tuning tables next to the kernel library ($BIN/hip, where it finds them without
# STRATA_HIPBLASLT_TUNING; talus#50): a release tarball brings them, a source build (or an older tarball) copies them
if [ -z "$PREBUILT" ] || ! ls "$BIN"/hip/*-hipblaslt-*.txt >/dev/null 2>&1; then
  run mkdir -p "$BIN/hip"
  run sh -c "install -m 0644 '$SRC'/tools/hip/*-hipblaslt-*.txt '$BIN/hip/'"
fi
# Flash-Next's data files and preparation tools at their source-tree paths under $DATA (talus setup takes them from
# there; scripts/release/side-files.txt): a release tarball brings them, a source build copies them
if [ -r "$SRC/scripts/release/side-files.txt" ]; then
  for f in $(grep -v '^[[:space:]]*#' "$SRC/scripts/release/side-files.txt" | grep .); do
    if [ -z "$PREBUILT" ] || [ ! -f "$DATA/$f" ]; then run install -D -m 0644 "$SRC/$f" "$DATA/$f"; fi
  done
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
[ -n "$MODELS_DIR" ] && set -- "$@" --models-dir "$MODELS_DIR"
[ "$NO_POWER" = 1 ] && set -- "$@" --no-power-profile
[ -n "$SERVICE" ] && set -- "$@" "$SERVICE"
if [ -x "$SETUP" ]; then
  exec "$SETUP" "$@"
else
  printf '  %s[dry-run] would run:%s talus %s\n' "$C" "$N" "$*"
fi
}

main "$@"
