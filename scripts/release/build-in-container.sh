#!/bin/bash
# Runs INSIDE the pinned builder container (scripts/release/Dockerfile); scripts/release-build.sh starts it.
# Fixed paths (they end up in the binaries, so they are part of the reproducible recipe):
#   /src   the source tree at COMMIT (git archive; read-write: build/ and rust/target/ go here)
#   /ggml  llama.cpp at the commit CMakeLists.txt pins (read-only; replaces the configure-time FetchContent download)
#   /opt/rocm-wheels/_rocm_sdk_devel   the ROCm SDK (read-only)
#   /cargo CARGO_HOME with the crates of Cargo.lock already fetched (the build runs with --offline, no network)
#   /out   the release assets
#   /tools scripts/release of the checkout that runs release-build.sh (read-only)
# Env: TAG COMMIT SOURCE_DATE_EPOCH JOBS ROCM_VERSION PKG BASE_IMAGE
set -euo pipefail
: "${TAG:?}" "${COMMIT:?}" "${SOURCE_DATE_EPOCH:?}" "${ROCM_VERSION:?}" "${PKG:?}"
JOBS=${JOBS:-16}
SDK=/opt/rocm-wheels/_rocm_sdk_devel
export HOME=/tmp CARGO_HOME=/cargo TZ=UTC LC_ALL=C.UTF-8 SOURCE_DATE_EPOCH
umask 022

# RUNPATH of everything we ship: the ROCm SDK the installer links next to the install (PREFIX/rocm-sdk -> SDK root),
# then a distribution ROCm in /opt/rocm. LD_LIBRARY_PATH (searched first) still overrides both.
RPATH_REL='$ORIGIN/../rocm-sdk/lib:/opt/rocm/lib'

# ---- 1. libtalus_kernels.so (gfx1151)
# ggml-cpu ISA: the release must not depend on the build host (-march=native is the dev default). STRATA_GGML_MARCH
# gives ggml-cpu exactly what -march=native expands to (gcc 13.3) on the Zen 4 host every measured build was made on
# (`gcc -### -march=native`), so ggml-cpu is the same code as those builds; Strix Halo's Zen 5 runs all of it.
GGML_MARCH="-march=znver4;--param=l1-cache-size=32;--param=l1-cache-line-size=64;--param=l2-cache-size=1024;-mtune=znver4"
echo "== kernels: build.sh (gfx1151, ROCm $ROCM_VERSION, -j $JOBS)"
( cd /src && ROCM_SDK=$SDK BUILD=/src/build TARGETS=talus_kernels JOBS=$JOBS \
    CMAKE_EXTRA="-DSTRATA_GGML_DIR=/ggml -DSTRATA_GGML_MARCH=$GGML_MARCH -DCMAKE_SKIP_RPATH=ON" ./build.sh )
SO=/src/build/libtalus_kernels.so
patchelf --set-rpath "$RPATH_REL" "$SO"

# ---- 2. Rust programs (dlopen libamdhip64 / libtalus_kernels at run time: no ROCm at link time)
BINS="talus-cli talus-server talus-tui"
PKGS="-p talus-cli -p talus-server -p talus-tui"
if [ -d /src/rust/crates/talus-setup ]; then BINS="$BINS talus-setup"; PKGS="$PKGS -p talus-setup"; fi
echo "== rust: cargo build --release --locked --offline $PKGS"
export CARGO_ENCODED_RUSTFLAGS="$(printf '%s\x1f' -C "link-arg=-Wl,-rpath,$RPATH_REL" \
  --remap-path-prefix=/src=/talus --remap-path-prefix=/cargo/registry/src=/crates | sed 's/\x1f$//')"
export CARGO_PROFILE_RELEASE_STRIP=debuginfo CARGO_INCREMENTAL=0
( cd /src/rust && cargo build --release --locked --offline $PKGS )

# ---- 3. stage
ST=/tmp/stage/$PKG; rm -rf /tmp/stage; mkdir -p $ST/bin $ST/licenses
for b in $BINS; do install -m 0755 /src/rust/target/release/$b $ST/bin/$b; done
install -m 0755 "$SO" $ST/bin/libtalus_kernels.so
cp -r /src/configs $ST/configs
mkdir -p $ST/docs; cp /src/docs/*.md $ST/docs/
for f in README.md LICENSE THIRD_PARTY_NOTICES.md SECURITY.md; do [ -f /src/$f ] && cp /src/$f $ST/$f; done
# licenses: every Rust crate linked into the programs, ggml (statically in the .so), and the ROCm parts compiled in
# (HIP headers, device libraries in the GPU code objects) or loaded at run time
python3 /tools/collect_licenses.py /src/rust $ST/licenses/rust-crates $BINS
# Rust's standard library is linked in statically (MIT OR Apache-2.0)
RD=$(rustc --print sysroot)/share/doc/rust; mkdir -p $ST/licenses/rust-std
for f in COPYRIGHT-library.html COPYRIGHT.html licenses; do [ -e $RD/$f ] && cp -r $RD/$f $ST/licenses/rust-std/; done
mkdir -p $ST/licenses/ggml $ST/licenses/rocm
cp /ggml/LICENSE $ST/licenses/ggml/LICENSE
for d in hip hipblas hipblaslt rocblas rocr amd_comgr rocm-core; do
  f=$(ls $SDK/share/doc/$d/LICENSE* $SDK/share/doc/$d/License* 2>/dev/null | head -1 || true)
  [ -n "$f" ] && cp "$f" $ST/licenses/rocm/$d-$(basename "$f")
done
# the AMDGPU device libraries linked into the GPU code objects live in ROCm's llvm-project (Apache-2.0 WITH
# LLVM-exception); the SDK ships no separate file for them, so the LLVM license text stands for both
cp $SDK/lib/llvm/include/llvm/Support/LICENSE.TXT $ST/licenses/rocm/llvm-and-device-libs-LICENSE.TXT

# ---- 4. manifest + checksums
python3 /tools/make_manifest.py $ST "$TAG" "$COMMIT" "$SOURCE_DATE_EPOCH" "$ROCM_VERSION" $SDK \
  "${BASE_IMAGE:-}" > $ST/MANIFEST.json
( cd $ST && find . -type f ! -name SHA256SUMS -printf '%P\n' | LC_ALL=C sort | xargs -d '\n' sha256sum > SHA256SUMS )

# ---- 5. deterministic tarball (sorted names, fixed mtime/owner/mode, no gzip timestamp)
find $ST -exec touch -h -d "@$SOURCE_DATE_EPOCH" {} +
chmod -R u+rwX,go+rX,go-w $ST
tar --sort=name --format=gnu --mtime="@$SOURCE_DATE_EPOCH" --owner=0 --group=0 --numeric-owner \
    -C /tmp/stage -cf - $PKG | gzip -9 -n > /out/$PKG.tar.gz
( cd /out && sha256sum $PKG.tar.gz > $PKG.tar.gz.sha256 && cp $ST/MANIFEST.json $PKG.manifest.json \
  && sha256sum $PKG.tar.gz $PKG.manifest.json > SHA256SUMS )
echo "== done: /out/$PKG.tar.gz ($(du -h /out/$PKG.tar.gz | cut -f1))"
cat /out/SHA256SUMS
