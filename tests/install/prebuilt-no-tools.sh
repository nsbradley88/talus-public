#!/usr/bin/env bash
# tests/install/prebuilt-no-tools.sh : an install from a prebuilt release needs no compiler, make, CMake, Ninja or
# Rust (talus#18 itest18: it installed CMake/Ninja for 5.6 min and a 617 MB Rust toolchain it never used), passes
# --models-dir / --no-power-profile / --service on to `talus setup`, and clones the release tag without git's
# detached-HEAD advice.
#
# Runs install.sh (--yes --force) in a temporary HOME with a PATH that holds only the tools a prebuilt install may
# use; cc, c++, gcc, g++, make, cmake, ninja, rustup, cargo, rustc and sudo are shims that log and fail. The release
# is a fake tarball (talus-setup records its arguments), the source a local bare repository with a release tag,
# the ROCm SDK an empty directory with lib/libamdhip64.so. Nothing outside the temporary directory changes.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fail=0
no() { echo "prebuilt-no-tools: FAIL: $*" >&2; fail=1; }

# the PATH: allowed tools as links, forbidden ones as logging shims
mkdir -p "$T/path"
for t in sh bash env uname awk sed grep cut tr head tail sort df id getent date mkdir rm ln install tar gzip \
         sha256sum curl git python3 mktemp dirname basename cat ls nproc ldd sleep cp chmod readlink wc touch \
         find xargs expr test true false printf; do
  p=$(command -v "$t") && [ "${p#/}" != "$p" ] && ln -s "$p" "$T/path/$t"
done
for t in cc c++ gcc g++ make cmake ninja rustup cargo rustc sudo; do
  printf '#!/bin/sh\necho "%s $*" >> "%s/forbidden.log"\nexit 1\n' "$t" "$T" > "$T/path/$t"
  chmod +x "$T/path/$t"
done

# the source: a bare repository with this tree tagged as a release
git -C "$ROOT" ls-files -co --exclude-standard -z | (cd "$ROOT" && tar --null -T - -cf "$T/src.tar")
mkdir -p "$T/srcwork" && tar -xf "$T/src.tar" -C "$T/srcwork"
(cd "$T/srcwork" && git init -q && git add -A && git -c user.name=t -c user.email=t@t commit -qm t && git tag v9.9.9 \
  && git clone -q --bare . "$T/repo.git")
COMMIT=$(git -C "$T/srcwork" rev-parse HEAD)

# the release: bin/talus-setup records its arguments
R="$T/rel/talus-v9.9.9-linux-x86_64-gfx1151"
mkdir -p "$R/bin/hip"
printf '#!/bin/sh\necho "$@" > "$HOME/setup-args"\n' > "$R/bin/talus-setup"
printf '#!/bin/sh\nexit 0\n' > "$R/bin/talus-tui"
chmod +x "$R/bin/talus-setup" "$R/bin/talus-tui"
echo x > "$R/bin/hip/gfx1151-hipblaslt-test.txt"
printf '{"tag":"v9.9.9","commit":"%s","runtime":{"glibc_min":"2.17","rocm_required":""}}\n' "$COMMIT" > "$R/MANIFEST.json"
tar -czf "$T/rel.tgz" -C "$T/rel" "$(basename "$R")"
sha256sum "$T/rel.tgz" | cut -d' ' -f1 > "$T/rel.tgz.sha256"
mkdir -p "$T/rocm/lib" && touch "$T/rocm/lib/libamdhip64.so"

mkdir -p "$T/home"
cp "$ROOT/install.sh" "$T/install.sh"   # run outside the checkout, as `curl | bash` does: it clones
out=$(cd "$T/home" && env -i HOME="$T/home" PATH="$T/path" NO_COLOR=1 TALUS_REPO="file://$T/repo.git" \
      TALUS_PREBUILT_URL="file://$T/rel.tgz" ROCM_SDK="$T/rocm" \
      bash "$T/install.sh" --yes --force --models-dir /data/m --no-power-profile --service 2>&1 </dev/null)
rc=$?
[ $rc = 0 ] || no "install.sh exit $rc"
[ ! -s "$T/forbidden.log" ] || no "called: $(tr '\n' ';' < "$T/forbidden.log")"
for d in "$T/home/.cargo" "$T/home/.rustup" "$T/home/.local/share/talus/tools/bin"; do
  [ ! -e "$d" ] || no "created $d"
done
case $out in *"detached HEAD"*|*"detachedHead"*) no "git printed its detached-HEAD advice" ;; esac
case $out in *"not needed: prebuilt release"*) ;; *) no "no 'build tools not needed' line" ;; esac
args=$(cat "$T/home/setup-args" 2>/dev/null)
for want in "--yes" "--models-dir /data/m" "--no-power-profile" "--service" "--force"; do
  case " $args " in *" $want "*) ;; *) no "talus setup got '$args', missing $want" ;; esac
done
[ "$(git -C "$T/home/.local/share/talus/src" rev-parse HEAD 2>/dev/null)" = "$COMMIT" ] || no "source not at the tag"
[ -x "$T/home/.local/share/talus/bin/talus-setup" ] || no "talus-setup not installed"
if [ $fail = 0 ]; then echo "prebuilt-no-tools: ok (setup args: $args)"; else printf '%s\n' "$out" | tail -n 40 >&2; fi
exit $fail
