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
#        prebuilt - `curl | bash` with a (fake) prebuilt release on an image without a compiler: install.sh may install
#                only git/curl/Python, never a compiler, make, CMake, Ninja or Rust (talus#18); the release tag is
#                cloned without the detached-HEAD advice. No ROCm SDK or GPU needed (a stand-in SDK directory).
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
if [ "$MODE" = prebuilt ]; then
  # the fixture, built here: a bare repository with this tree tagged v9.9.9, a release tarball whose talus-setup
  # records its arguments, a stand-in ROCm SDK
  mkdir -p "$T/w" && tar -xf "$T/src.tar" -C "$T/w"
  (cd "$T/w" && git init -q && git add -A && git -c user.name=t -c user.email=t@t commit -qm t && git tag v9.9.9 \
    && git clone -q --bare . "$T/repo.git")
  R="$T/rel/talus-v9.9.9-linux-x86_64-gfx1151"
  mkdir -p "$R/bin/hip" "$T/rocm/lib"
  printf '#!/bin/sh\necho "$@" > "$HOME/setup-args"\n' > "$R/bin/talus-setup"
  printf '#!/bin/sh\nexit 0\n' > "$R/bin/talus-tui"
  chmod 755 "$R/bin/talus-setup" "$R/bin/talus-tui"
  echo x > "$R/bin/hip/gfx1151-hipblaslt-test.txt"
  printf '{"tag":"v9.9.9","commit":"%s","runtime":{"glibc_min":"2.17"}}\n' "$(git -C "$T/w" rev-parse HEAD)" > "$R/MANIFEST.json"
  tar -czf "$T/rel.tgz" -C "$T/rel" "$(basename "$R")"
  sha256sum "$T/rel.tgz" | cut -d' ' -f1 > "$T/rel.tgz.sha256"
  touch "$T/rocm/lib/libamdhip64.so"
  rm -rf "$T/w" "$T/rel"
  chmod -R a+rX "$T"
fi
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
elif [ "$MODE" = prebuilt ]; then
  for t in cc c++ gcc make cmake ninja cargo rustup; do if command -v \$t >/dev/null; then echo "image already has \$t"; fi; done
  su - u -c 'cp /in/install.sh-copy /tmp/install.sh; cp -r /in/repo.git /tmp/repo.git; cd /tmp && NO_COLOR=1 TALUS_REPO=file:///tmp/repo.git TALUS_PREBUILT_URL=file:///in/rel.tgz ROCM_SDK=/in/rocm bash /tmp/install.sh --yes --force --models-dir /tmp/m --no-power-profile --service 2>&1 | tee /tmp/install.log | sed -n "/2\/5/,/3\/5/p"'
  bad=0
  for t in cc c++ gcc make cmake ninja cargo rustup; do if command -v \$t >/dev/null; then echo "FAIL: \$t was installed"; bad=1; fi; done
  for d in /home/u/.cargo /home/u/.rustup /home/u/.local/share/talus/tools/bin; do if [ -e \$d ]; then echo "FAIL: \$d exists"; bad=1; fi; done
  if grep -q "detached HEAD" /tmp/install.log; then echo "FAIL: detached-HEAD advice"; bad=1; fi
  echo "setup args: \$(cat /home/u/setup-args)"
  grep -q -- "--models-dir /tmp/m --no-power-profile --service" /home/u/setup-args || { echo "FAIL: setup args"; bad=1; }
  for t in git curl python3; do command -v \$t >/dev/null || { echo "FAIL: \$t missing"; bad=1; }; done
  [ \$bad = 0 ] && echo "PREBUILT OK" || tail -n 20 /tmp/install.log
  exit \$bad
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
cp "$ROOT/install.sh" "$T/install.sh-copy"
chmod 755 "$T"; find "$T" -maxdepth 1 -type f -exec chmod 644 {} +
echo "== $IMAGE ($MODE)"
# shellcheck disable=SC2086
"$RT" run --rm -v "$T:/in:ro" $MOUNT "$IMAGE" sh /in/run.sh
