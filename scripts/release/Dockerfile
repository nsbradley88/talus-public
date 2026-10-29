# Pinned build environment for the Talus release tarball (scripts/release-build.sh builds and runs it).
# Everything that ends up in the binaries is pinned here or in the source tree:
#   base image by digest, Ubuntu packages from a dated snapshot of the archive, rustup-init by version + sha256,
#   the Rust toolchain by version (rustup verifies every component against the signed channel manifest).
# The ROCm SDK is NOT in the image (several GB, and the pinned nightly is not an apt package): release-build.sh mounts
# it read-only at /opt/rocm-wheels and checks its version and fingerprint first.
ARG BASE=ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
FROM ${BASE}
ARG APT_SNAPSHOT=20261027T000000Z
ARG RUSTUP_VERSION=1.29.1
ARG RUSTUP_SHA256=dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71
ARG RUST_TOOLCHAIN=1.99.1
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC \
    RUSTUP_HOME=/opt/rustup PATH=/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# snapshot.ubuntu.com is https-only and the base image has no CA bundle: bootstrap ca-certificates from the live
# archive first (it only verifies TLS; the snapshot's InRelease is GPG-checked either way), then pin everything else.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && apt-get update --snapshot ${APT_SNAPSHOT} \
 && apt-get install -y --no-install-recommends --snapshot ${APT_SNAPSHOT} \
      ca-certificates curl gcc g++ make cmake ninja-build python3 patchelf binutils file xz-utils \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg-query -W -f '${Package}=${Version}\n' > /opt/apt-packages.txt \
 && echo "${APT_SNAPSHOT}" > /opt/apt-snapshot.txt
RUN curl -fsSL --proto '=https' --tlsv1.2 -o /tmp/rustup-init \
      https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init \
 && echo "${RUSTUP_SHA256}  /tmp/rustup-init" | sha256sum -c - \
 && chmod +x /tmp/rustup-init \
 && CARGO_HOME=/opt/cargo /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain ${RUST_TOOLCHAIN} \
 && CARGO_HOME=/opt/cargo rustup component add --toolchain ${RUST_TOOLCHAIN} rustfmt clippy \
 && rm /tmp/rustup-init && chmod -R a+rX /opt/rustup /opt/cargo
