#!/bin/sh
# End-to-end test of the installer's non-GPU steps in a throwaway container (docker or podman).
#
#   tests/install/container-e2e.sh IMAGE MODE [ROCM_SDK]
#
# IMAGE  e.g. ubuntu:24.04, debian:13, fedora:42, archlinux:latest
# MODE   deps  - install.sh steps 1-3 (packages through sudo, CMake/Ninja, rustup and the pinned toolchain, source),
#                then the Rust build of talus-setup only, then `talus setup --dry-run` and `talus doctor`
#        full  - all of install.sh with the given ROCm SDK (mounted read-only): the kernel library and every Rust
#                program; then `talus setup --yes --no-download` (configs, state), `talus fetch` of one small real
#                file, `talus doctor`, and talus-server in mock mode on the generated control directory
# The source is this working tree (tracked and untracked-but-not-ignored files). Runs as a normal user with sudo.
# The container is removed afterwards; nothing on the host changes.
set -eu
IMAGE=${1:?image}
MODE=${2:?deps|full}
SDK=${3:-}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RT=$(command -v docker || command -v podman) || { echo "needs docker or podman" >&2; exit 2; }
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
(cd "$ROOT" && git ls-files -co --exclude-standard -z | tar --null -T - -cf "$T/src.tar")
case $IMAGE in
  *ubuntu*|*debian*) PRE="apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl sudo ca-certificates >/dev/null" ;;
  *fedora*) PRE="dnf install -y -q curl sudo util-linux shadow-utils >/dev/null" ;;
  *arch*) PRE="pacman -Sy --noconfirm --needed curl sudo >/dev/null" ;;
  *suse*) PRE="zypper -q install -y curl sudo >/dev/null" ;;
  *) PRE="true" ;;
esac
MOUNT=""
ENVSDK=""
if [ "$MODE" = full ]; then
  [ -n "$SDK" ] || { echo "full mode needs a ROCm SDK root" >&2; exit 2; }
  # mount the SDK's parent (site-packages for TheRock wheels) so the version dist-info next to it is visible
  MOUNT="-v $(dirname "$SDK"):/rocm-parent:ro"
  ENVSDK="ROCM_SDK=/rocm-parent/$(basename "$SDK")"
fi
cat > "$T/run.sh" <<EOF
set -eu
$PRE
useradd -m -s /bin/bash u
echo 'u ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/u
chmod 440 /etc/sudoers.d/u
su - u -c 'mkdir -p talus && tar -xf /in/src.tar -C talus'
if [ "$MODE" = deps ]; then
  su - u -c 'cd talus && ./install.sh --yes --force --build-only --dry-run >/dev/null'
  # the dry run above must not have changed anything; now the real steps 1-3 and the Rust installer only
  su - u -c 'test ! -e ~/.local/share/talus/bin'
  su - u -c 'cd talus && ROCM_SDK=/nonexistent sh -c "./install.sh --yes --force --no-setup" 2>&1 | tail -n 12' || true
  su - u -c '. ~/.cargo/env 2>/dev/null; export PATH=\$HOME/.local/share/talus/tools/bin:\$PATH; cd talus/rust && cargo build --release --locked -q -p talus-setup && ./target/release/talus-setup setup --dry-run --yes --force --models 27b >/tmp/setup-dry.txt; tail -n 25 /tmp/setup-dry.txt; ./target/release/talus-setup doctor --offline | tail -n 8; cargo test --release --locked -q -p talus-setup 2>&1 | tail -n 3'
else
  su - u -c 'cd talus && $ENVSDK ./install.sh --yes --force --no-setup 2>&1 | tail -n 25'
  su - u -c 'ls -l ~/.local/share/talus/bin ~/.local/bin'
  su - u -c '~/.local/bin/talus setup --yes --force --models 27b,laguna --no-download --no-service --no-start 2>&1 | tail -n 20'
  su - u -c 'cat ~/.config/talus/setup.json; ls -l ~/.config/talus/models; head -c 700 ~/.config/talus/models/qwen3.8-27b.json'
  su - u -c '~/.local/bin/talus fetch 27b qwen-dv-65536.bin'
  su - u -c '~/.local/bin/talus doctor --offline' || true
  su - u -c '~/.local/bin/talus doctor --fix --dry-run --offline 2>&1 | sed -n "/Fixing/,\\\$p"' || true
  # talus-server in mock mode on the generated control directory: it must parse our configs and list the models
  su - u -c 'B=~/.local/share/talus/bin; \$B/talus-server --control-dir ~/.config/talus/models --config ~/.config/talus/models/qwen3.8-27b.json --mock --port 8123 >/tmp/srv.log 2>&1 & sleep 3; curl -s localhost:8123/health; echo; curl -s localhost:8123/v1/talus/presets | head -c 600; echo; ~/.local/bin/talus doctor --offline | grep -i server; kill %1' || true
fi
EOF
chmod 755 "$T"; chmod 644 "$T"/*
echo "== $IMAGE ($MODE)"
# shellcheck disable=SC2086
"$RT" run --rm -v "$T:/in:ro" $MOUNT "$IMAGE" sh /in/run.sh
