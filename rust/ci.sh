#!/usr/bin/env bash
# rust/ci.sh : the repo's CI gate.
#   CPU stages (always): pinned-toolchain check, cargo fmt --check, clippy -D warnings, cargo test, cargo deny,
#   install.sh's NO_COLOR / no-terminal output (tests/install/no-color.sh), its build-failure output, job count and
#   stale build directory handling (tests/install/build-error.sh).
#   Kernel stage (opt-in): TALUS_CI_KERNELS=1 builds libtalus_kernels with ../build.sh (needs a ROCm SDK:
#   ROCM_SDK=<sdk root>, BUILD=<dir> default ../build-ci, TARGETS default talus_kernels, CMAKE_EXTRA passed through).
#   build.sh itself fails if any built target is older than its sources (the stale-build guard).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Tests make Unix sockets under TMPDIR (SWE bridge: $TMPDIR/tb-bridge-<pid>/model.sock), and a socket path must fit
# sun_path (107 bytes). target/tmp under a deep checkout did not, so a long TMPDIR is replaced by a
# private directory under /tmp, removed on exit.
S="${TMPDIR:-$SCRIPT_DIR/target/tmp}"
if [ "${#S}" -gt 60 ]; then
  echo "ci.sh: TMPDIR $S is too long for Unix socket paths; using a private dir under /tmp"
  S=$(mktemp -d /tmp/talus-ci.XXXXXX)
  trap 'rm -rf "$S"' EXIT
fi
mkdir -p "$S"
export TMPDIR="$S" TMP="$S" TEMP="$S"
export PATH="$HOME/.cargo/bin:$PATH"

# Needs: rustup (the pinned toolchain installs itself), cargo-deny (below), util-linux `script` (installer colour test).
# cargo-deny (license gate) is a separate install; say how instead of failing on "no such command"
DENY_VERSION=0.20.2
if ! cargo deny --version >/dev/null 2>&1; then
  echo "ci.sh: cargo-deny is not installed; install it with: cargo install cargo-deny --version $DENY_VERSION --locked" >&2
  exit 1
fi

# Pinned toolchain: every cargo call below runs from rust/, so rustup selects rust-toolchain.toml's channel.
# Fail loudly if something (RUSTUP_TOOLCHAIN, a rustup override, a missing rustup) picked another compiler.
PIN=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml)
[ -n "$PIN" ] || { echo "ci.sh: no channel in rust-toolchain.toml" >&2; exit 1; }
HAVE=$(rustc -V | awk '{print $2}')
[ "$HAVE" = "$PIN" ] || { echo "ci.sh: rustc $HAVE, but rust-toolchain.toml pins $PIN" >&2; exit 1; }
cargo clippy -V >/dev/null
echo "ci.sh: toolchain $HAVE (pinned $PIN)"

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo deny check licenses
"$SCRIPT_DIR/../tests/install/no-color.sh"
"$SCRIPT_DIR/../tests/install/build-error.sh"

if [ "${TALUS_CI_KERNELS:-0}" = 1 ]; then
  echo "ci.sh: kernel build check (build.sh, stale-build guard)"
  BUILD="${BUILD:-$SCRIPT_DIR/../build-ci}" TARGETS="${TARGETS:-talus_kernels}" "$SCRIPT_DIR/../build.sh"
  # GDN prompt-scan kernels: no spills, the production grid resident in one pass on gfx1151 (exit 77 = cannot check here)
  rc=0; ROCM_SDK="${ROCM_SDK:-}" python3 "$SCRIPT_DIR/../tools/check_gdn_occupancy.py" \
    "${BUILD:-$SCRIPT_DIR/../build-ci}/libtalus_kernels.so" || rc=$?
  [ "$rc" = 0 ] || [ "$rc" = 77 ] || exit "$rc"
fi
echo "ci.sh: OK"
