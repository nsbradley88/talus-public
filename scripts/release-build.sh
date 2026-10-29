#!/bin/bash
# scripts/release-build.sh : build the prebuilt Talus release tarball in a pinned container (docker).
#
#   scripts/release-build.sh [--ref REF] [--tag TAG] [--out DIR] [--jobs N] [--rocm-sdk DIR] [--ggml-git DIR]
#                            [--keep-work] [--allow-rocm-mismatch]
#
#   --ref       git commit / tag / branch to build (default HEAD). The source is `git archive REF`: committed files
#               only, so a dirty worktree cannot leak into a release.
#   --tag       release tag, used in the file names (default v<workspace version>-g<short sha>)
#   --out       output directory (default build-release/out-<TAG>)
#   --jobs      parallel compile jobs (default 16; the HIP compile needs ~2 GB per job)
#   --rocm-sdk  the ROCm SDK root (the `_rocm_sdk_devel` dir of the TheRock wheels; default $ROCM_SDK). Its version
#               must equal ROCM_VERSION below unless --allow-rocm-mismatch.
#   --ggml-git  a llama.cpp git checkout that has the commit CMakeLists.txt pins (default: fetched from GitHub)
#
# Outputs (in --out): talus-<TAG>-linux-x86_64-gfx1151.tar.gz, .tar.gz.sha256 (the installer's check),
# .manifest.json (the build manifest, also inside the tarball) and SHA256SUMS (covers both).
# Wrap it in a build lock on shared machines; it runs one container with --network none for the compile.
# See scripts/release/README.md for the artifact layout, reproducibility and the publish steps.
set -euo pipefail
ROCM_VERSION=10.2.0a20261014                 # the SDK every published number was measured with
ROCM_INDEX=https://nightly.repo.amd.com/rocm/whl-next/
REPO=$(cd "$(dirname "$0")/.." && pwd)
REF=HEAD TAG="" OUT="" JOBS=16 SDK=${ROCM_SDK:-} GGML_GIT="" KEEP=0 ALLOW=0
while [ $# -gt 0 ]; do
  case $1 in
    --ref) REF=$2; shift 2;; --tag) TAG=$2; shift 2;; --out) OUT=$2; shift 2;; --jobs) JOBS=$2; shift 2;;
    --rocm-sdk) SDK=$2; shift 2;; --ggml-git) GGML_GIT=$2; shift 2;; --keep-work) KEEP=1; shift;;
    --allow-rocm-mismatch) ALLOW=1; shift;;
    -h|--help) sed -n '2,21p' "$0"; exit 0;;
    *) echo "release-build: unknown option $1" >&2; exit 2;;
  esac
done
die() { echo "release-build: $*" >&2; exit 1; }
command -v docker >/dev/null || die "docker is required"
COMMIT=$(git -C "$REPO" rev-parse --verify "$REF^{commit}") || die "no commit $REF"
SDE=$(git -C "$REPO" log -1 --format=%ct "$COMMIT")
VER=$(git -C "$REPO" show "$COMMIT:rust/Cargo.toml" | sed -n '/^\[workspace.package\]/,/^\[/s/^version *= *"\(.*\)"/\1/p')
TAG=${TAG:-v$VER-g${COMMIT:0:8}}
PKG=talus-$TAG-linux-x86_64-gfx1151
OUT=${OUT:-$REPO/build-release/out-$TAG}

# ---- ROCm SDK: right version, the wheels' layout (mounted read-only at a fixed path)
[ -n "$SDK" ] || die "no ROCm SDK: pass --rocm-sdk (pip install --index-url $ROCM_INDEX \"rocm[libraries,devel,device-gfx1151]==$ROCM_VERSION\"; rocm-sdk init; rocm-sdk path --root)"
SDK=$(cd "$SDK" && pwd -P)
[ "$(basename "$SDK")" = _rocm_sdk_devel ] || die "--rocm-sdk must be the wheels' _rocm_sdk_devel directory (got $SDK)"
SITE=$(dirname "$SDK")
HAVE=$(ls -d "$SITE"/rocm_sdk_devel-*.dist-info 2>/dev/null | sed 's/.*rocm_sdk_devel-\(.*\)\.dist-info/\1/' | head -1)
[ -d "$SITE/rocm_sdk_device_gfx1151-$HAVE.dist-info" ] || die "no gfx1151 device wheel for ROCm $HAVE in $SITE"
if [ "$HAVE" != "$ROCM_VERSION" ]; then
  [ $ALLOW = 1 ] || die "ROCm SDK is $HAVE, the release pins $ROCM_VERSION (--allow-rocm-mismatch to build anyway)"
  echo "release-build: WARNING building with ROCm $HAVE instead of $ROCM_VERSION" >&2
fi

# ---- work tree: source at COMMIT, ggml at the pinned llama.cpp commit, cargo cache
mkdir -p "$REPO/build-release"
WORK=$(mktemp -d "$REPO/build-release/work.XXXXXX")
cleanup() { [ $KEEP = 1 ] || rm -rf "$WORK"; }
mkdir -p "$WORK/src" "$WORK/ggml" "$OUT"
git -C "$REPO" archive "$COMMIT" | tar -x -C "$WORK/src"
GGML_SHA=$(sed -n 's/^ *GIT_TAG \([0-9a-f]\{40\}\).*/\1/p' "$WORK/src/CMakeLists.txt" | head -1)
[ -n "$GGML_SHA" ] || die "no pinned llama.cpp commit (GIT_TAG) in CMakeLists.txt"
if [ -z "$GGML_GIT" ]; then
  GGML_GIT="$REPO/build-release/llama.cpp.git"
  [ -d "$GGML_GIT" ] || git init -q --bare "$GGML_GIT"
  git -C "$GGML_GIT" cat-file -e "$GGML_SHA^{commit}" 2>/dev/null \
    || git -C "$GGML_GIT" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp.git "$GGML_SHA"
fi
git -C "$GGML_GIT" archive "$GGML_SHA" | tar -x -C "$WORK/ggml"
CARGO_CACHE="$REPO/build-release/cargo-home"; mkdir -p "$CARGO_CACHE"   # registry crates: verified by Cargo.lock

# the packaging tools come from THIS checkout (mounted at /tools), so an older --ref can be packaged too
# ---- builder image (content-addressed tag: the Dockerfile is the recipe)
H=$(sha256sum "$REPO/scripts/release/Dockerfile" | cut -c1-12)
IMG=talus-release-builder:$H
docker image inspect "$IMG" >/dev/null 2>&1 || docker build -q -t "$IMG" "$REPO/scripts/release" >/dev/null
BASE=$(sed -n 's/^ARG BASE=//p' "$REPO/scripts/release/Dockerfile")
trap cleanup EXIT

U="$(id -u):$(id -g)"
echo "release-build: $PKG  commit $COMMIT  ROCm $HAVE  image $IMG"
# 1) fetch the crates (network), 2) build with no network
docker run --rm -u "$U" -e HOME=/tmp -e CARGO_HOME=/cargo -v "$WORK/src:/src" -v "$CARGO_CACHE:/cargo" -w /src/rust \
  "$IMG" cargo fetch --locked >/dev/null
docker run --rm -u "$U" --network none --tmpfs /tmp:exec,size=8g \
  -v "$WORK/src:/src" -v "$WORK/ggml:/ggml:ro" -v "$SITE:/opt/rocm-wheels:ro" -v "$CARGO_CACHE:/cargo" -v "$OUT:/out" \
  -v "$REPO/scripts/release:/tools:ro" \
  -e TAG="$TAG" -e COMMIT="$COMMIT" -e SOURCE_DATE_EPOCH="$SDE" -e JOBS="$JOBS" -e ROCM_VERSION="$HAVE" \
  -e PKG="$PKG" -e BASE_IMAGE="$BASE" \
  "$IMG" bash /tools/build-in-container.sh
echo "release-build: done -> $OUT"
ls -l "$OUT"
