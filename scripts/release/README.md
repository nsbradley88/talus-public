# Prebuilt release binaries

`scripts/release-build.sh` builds the release tarball in a pinned container, so users (and `install.sh` /
`talus update`) skip the long HIP kernel compile. Nothing here publishes anything: `gh-release.sh` only prints the
commands.

## The artifact

One tarball per release, for Linux x86_64 + gfx1151 (Strix Halo):

```
talus-<TAG>-linux-x86_64-gfx1151.tar.gz
└── talus-<TAG>-linux-x86_64-gfx1151/
    ├── bin/  talus-cli  talus-server  talus-tui  [talus-setup]  libtalus_kernels.so
    ├── configs/   docs/*.md   README.md  LICENSE  THIRD_PARTY_NOTICES.md  SECURITY.md
    ├── licenses/  rust-crates/<crate>-<ver>/ + INDEX.tsv, rust-std/, ggml/, rocm/
    ├── MANIFEST.json   (commit, tag, ROCm version + SDK fingerprint, rustc/gcc/cmake, base image, apt snapshot,
    │                    targets, kernel ABI version, required ROCm sonames, glibc floor, per-file sha256/NEEDED/RUNPATH)
    └── SHA256SUMS      (every file in the tree)
```

Release assets next to it: `<pkg>.tar.gz.sha256` (one line, what `install.sh` checks), `<pkg>.manifest.json` (the
same MANIFEST.json, readable before downloading), `SHA256SUMS` (tarball + manifest).

URL pattern: `https://github.com/nsbradley88/talus-public/releases/download/<TAG>/<pkg>.tar.gz` (+ `.sha256`,
`.manifest.json`). The latest tag: follow the redirect of `https://github.com/nsbradley88/talus-public/releases/latest`
(`Location: .../releases/tag/<TAG>`), no API token needed.

### How the programs find their libraries

- `libtalus_kernels.so` is found next to the executable (`bin/`); `TALUS_KERNELS_LIB` overrides.
- ROCm is **not bundled**. Every ELF in `bin/` carries `RUNPATH=$ORIGIN/../rocm-sdk/lib:/opt/rocm/lib`: link
  `PREFIX/rocm-sdk` to the ROCm SDK root (the installer does this), or install ROCm in `/opt/rocm`, or set
  `LD_LIBRARY_PATH` (searched first). `ROCM_SDK` / `ROCM_PATH` also work for `libamdhip64` (hip-sys).
- Needed from ROCm: `libamdhip64.so.7`, `libhipblas.so.3`, `libhipblaslt.so.1` (and what they pull in from their own
  SDK via `$ORIGIN`). Release pin: ROCm `10.2.0a20261014`, from
  `pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel,device-gfx1151]==10.2.0a20261014"`.
- System libraries only: libc, libm, libgcc_s, libstdc++, ld-linux. glibc floor is in MANIFEST.json (built on
  Ubuntu 24.04, glibc 2.39).

### CPU code in the kernel library

ggml's CPU part is built with `-DSTRATA_GGML_MARCH=-march=znver4;--param=l1-cache-size=32;...;-mtune=znver4`, which
is exactly what `-march=native` expands to on the Zen 4 host every measured build was made on, instead of the build
host's `-march=native`. So the release does not depend on where it is built, and its ggml-cpu code is the same as the
measured builds'. It needs a Zen 4 / Zen 5 CPU (AVX-512), which every Strix Halo is. Talus' own AVX2/AVX-512 kernels
choose at run time either way.

## Building

```sh
scripts/release-build.sh --ref <commit|tag> --tag vX.Y.Z --rocm-sdk <...>/site-packages/_rocm_sdk_devel [--jobs 16]
scripts/release/verify-tarball.sh build-release/out-vX.Y.Z/talus-vX.Y.Z-linux-x86_64-gfx1151.tar.gz --rocm-sdk <sdk>
scripts/release/gh-release.sh build-release/out-vX.Y.Z          # prints the gh commands + renders RELEASE-NOTES.md
```

What is pinned: the base image by digest (`ubuntu:24.04@sha256:...`), Ubuntu packages from snapshot.ubuntu.com at a
fixed timestamp (gcc 13.3.0, cmake 3.28.3, glibc 2.39), rustup-init by version + sha256, Rust 1.99.1 (the channel
manifest is signed and every component hash-checked), the crates by Cargo.lock (fetched first, then the compile runs
with `--network none --offline`), ggml by the llama.cpp commit CMakeLists.txt pins (exported from git, no
configure-time download), the ROCm SDK by version (the script refuses another one) and recorded by fingerprint
(sha256 of clang, libamdhip64, hip_version.h, the device bitcode). The source is `git archive` of the commit, so
uncommitted files cannot leak in. Paths inside the container are fixed (`/src`, `/opt/rocm-wheels`, `/cargo`), so
no build-machine path ends up in the binaries.

## Reproducibility

Tested 2026-10-28 on the build machine: two clean builds of the same commit (fresh work trees, separate runs) gave the
**same tarball, byte for byte** (the same sha256 both times; every file inside identical). That holds as long as
the inputs are the same: the source commit, the ROCm SDK bytes (MANIFEST.json `rocm_sdk.fingerprint_sha256`), the
base image digest + apt snapshot, the Rust toolchain and Cargo.lock.

What is NOT covered / not bit-identical:
- **A build outside the container** (plain `build.sh` in a dev tree) differs: HIP's per-file `__hip_cuid_*` symbols
  are hashes of the source path and options, and `__FILE__` strings carry the tree path. The container's fixed paths
  are what make the release repeatable. Against the measured release build (same code, dev tree,
  ROCm 10.2): `.text` and `.hip_fatbin` have the same size, the exported symbols are the same apart from the cuids,
  and all 139 ggml-cpu functions are instruction-identical; the GPU fat binary bytes differ because its compressed
  bundles carry the cuids. Greedy output identity with the measured build is checked on the GPU.
- **Another build host** was not tried yet (e.g. the CI runner). Nothing host-specific should leak in any more
  (ggml-cpu uses explicit `-march=znver4` flags, not `-march=native`), but it is untested.
- **The builder image** is not bit-reproducible (layer timestamps), only what it produces. The `ca-certificates`
  bootstrap comes from the live archive (TLS only; it does not reach the binaries).
- **A different ROCm SDK** (even the same version string re-downloaded from another index) is only the same build
  if the fingerprint matches; release-build.sh refuses other versions without `--allow-rocm-mismatch`.

## ROCm runtime: depend, don't bundle (recommendation)

Licenses would allow bundling the runtime: the closure of `libamdhip64` + `libhipblas` + `libhipblaslt` is HIP/CLR,
hipBLAS, hipBLASLt, rocBLAS, rocRoller, rocprofiler-register, roctx (MIT), ROCr/HSA runtime (NCSA), comgr and libomp
(Apache-2.0 WITH LLVM-exception), rocSOLVER (BSD-2-Clause), plus TheRock's `rocm_sysdeps` copies of libdrm (MIT),
zlib, zstd (BSD), bzip2, liblzma (0BSD), and two LGPL libraries: libnuma (LGPL-2.1) and libelf (elfutils,
LGPL-3.0-or-later / GPL-2.0-or-later). Nothing in that closure is under AMD's proprietary EULA. Redistributing it
would require shipping the SDK's NOTICES.txt and every license text (the sysdeps' texts are NOT in the SDK, we would
have to add them) and, for the LGPL libraries, their corresponding source (or a written offer) with each release.

We still recommend depending on the user's ROCm and recording the version:
- Size: the minimal runtime is ~840 MB unpacked (librocroller 590 MB, comgr 162 MB) plus the gfx1151 device archives,
  vs ~37 MB (gzip) for Talus itself.
- The installer installs the exact pinned SDK from AMD's index anyway (it needs it to run, and to build from source).
- Bundling makes us the distributor of a nightly runtime: its security fixes, its LGPL source obligations.
- The pinned nightly is public and byte-identical to what every published number was measured with; the stable
  10.1.0 release has the same sonames (libamdhip64.so.7, libhipblas.so.3, libhipblaslt.so.1).
Fallback if AMD's nightly ever disappears: mirror the four pinned wheels (same license work as bundling), or move the
pin to a stable release after a GPU A/B.

## CI (optional)

A manual (`workflow_dispatch`) GitHub Actions job can run the same script on a GitHub-hosted `ubuntu-24.04` runner:
free disk space, pip-install the pinned ROCm SDK, `scripts/release-build.sh --jobs 4`, `verify-tarball.sh`, and upload
the tarball as a workflow artifact (publishing stays a maintainer step).
- Cost: standard runners are free and unmetered for public repositories (larger runners are billed per minute).
- Hardware: 4 vCPU / 16 GB RAM -> `-j 4` for the HIP compile; disk is tight (~14 GB free on `/` by default), so the
  job deletes preinstalled toolchains first.
- ROCm SDK: ~1.7 GB of wheels per run from AMD's CDN, ~7 GB installed. Not cached: the repository cache limit is
  10 GB and the SDK alone would take most of it; downloading takes a few minutes.
- Time: about 11 min on a 32-core build machine at -j 16; a 4-vCPU runner at -j 4 should take roughly 45-90 min
  (estimate, not run).
