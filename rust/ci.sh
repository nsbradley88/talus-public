#!/usr/bin/env bash
# rust/ci.sh : the repo's CI gate.
#   CPU stages (always): pinned-toolchain check, cargo fmt --check, clippy -D warnings, cargo test, cargo deny.
#   Kernel stage (opt-in): TALUS_CI_KERNELS=1 builds libtalus_kernels with ../build.sh (needs a ROCm SDK:
#   ROCM_SDK=<sdk root>, BUILD=<dir> default ../build-ci, TARGETS default talus_kernels, CMAKE_EXTRA passed through).
#   build.sh itself fails if any built target is older than its sources (the stale-build guard).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

S="${TMPDIR:-$SCRIPT_DIR/target/tmp}"
mkdir -p "$S"
export TMPDIR="$S" TMP="$S" TEMP="$S"
export PATH="$HOME/.cargo/bin:$PATH"

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

if [ "${TALUS_CI_KERNELS:-0}" = 1 ]; then
  echo "ci.sh: kernel build check (build.sh, stale-build guard)"
  BUILD="${BUILD:-$SCRIPT_DIR/../build-ci}" TARGETS="${TARGETS:-talus_kernels}" "$SCRIPT_DIR/../build.sh"
fi
echo "ci.sh: OK"
