#!/bin/bash
# verify-tarball.sh TARBALL [--rocm-sdk DIR] [--keep]
#   Unpacks a release tarball into a fresh temp prefix and checks it WITHOUT touching a GPU:
#   - the .sha256 next to the tarball (if present) and the SHA256SUMS inside it
#   - every program runs --help (talus-tui: --once against a closed port, must fail cleanly, not crash)
#   - ELF dependencies: only the expected system libraries + the ROCm sonames of the manifest (readelf NEEDED)
#   - no build-machine paths (/home/, /root/) in the binaries
#   - with --rocm-sdk: links PREFIX/rocm-sdk -> DIR and requires `ldd -r` to resolve every library and symbol of
#     libtalus_kernels.so from that SDK (ldd only maps the libraries; no constructors run, no GPU access)
#   Exit 0 = all checks passed.
set -uo pipefail
TB=${1:?usage: verify-tarball.sh TARBALL [--rocm-sdk DIR] [--keep]}; shift
SDK="" KEEP=0
while [ $# -gt 0 ]; do case $1 in --rocm-sdk) SDK=$2; shift 2;; --keep) KEEP=1; shift;; *) echo "unknown $1"; exit 2;; esac; done
FAIL=0; ok() { echo "  ok    $*"; }; bad() { echo "  FAIL  $*"; FAIL=1; }
T=$(mktemp -d "${TMPDIR:-/tmp}/talus-verify.XXXXXX"); [ $KEEP = 1 ] || trap 'rm -rf "$T"' EXIT
echo "verify: $TB -> $T"
if [ -f "$TB.sha256" ]; then
  [ "$(sha256sum "$TB" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$TB.sha256")" ] && ok "tarball sha256" || bad "tarball sha256 mismatch"
fi
tar -xzf "$TB" -C "$T" || { bad "untar"; exit 1; }
P=$(ls -d "$T"/talus-*/ | head -1); P=${P%/}; echo "  prefix $P"
(cd "$P" && sha256sum --quiet -c SHA256SUMS) && ok "SHA256SUMS ($(wc -l < "$P/SHA256SUMS") files)" || bad "SHA256SUMS"
for f in MANIFEST.json LICENSE THIRD_PARTY_NOTICES.md bin/libtalus_kernels.so bin/talus-cli bin/talus-server bin/talus-tui; do
  [ -e "$P/$f" ] || bad "missing $f"; done

# programs run (no model, no GPU: --help paths do not load HIP)
for b in talus-cli talus-server talus-setup; do
  [ -x "$P/bin/$b" ] || continue
  out=$(env -u LD_LIBRARY_PATH "$P/bin/$b" --help 2>&1); rc=$?
  [ $rc = 0 ] && [ -n "$out" ] && ok "$b --help ($(echo "$out" | head -1 | cut -c1-60))" || bad "$b --help rc=$rc: $(echo "$out" | head -2)"
done
out=$(env -u LD_LIBRARY_PATH "$P/bin/talus-tui" --help 2>&1); rc=$?
[ $rc = 0 ] && ok "talus-tui --help" || bad "talus-tui --help rc=$rc"
out=$(timeout 20 env -u LD_LIBRARY_PATH "$P/bin/talus-tui" --once --server http://127.0.0.1:9 2>&1); rc=$?
if [ $rc -ge 1 ] && [ $rc -le 100 ] && ! echo "$out" | grep -q "panicked"; then ok "talus-tui --once (no server): rc=$rc, '$(echo "$out" | head -1 | cut -c1-70)'"
else bad "talus-tui --once rc=$rc: $out"; fi

# dependencies
SYS="libc.so.6 libm.so.6 libgcc_s.so.1 ld-linux-x86-64.so.2 libstdc++.so.6 libdl.so.2 libpthread.so.0 librt.so.1"
ROCM=$(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['runtime']['rocm_sonames']))" "$P/MANIFEST.json")
for f in "$P"/bin/*; do
  n=$(readelf -d "$f" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | tr '\n' ' ')
  for l in $n; do
    case " $SYS " in *" $l "*) continue;; esac
    if [ "$(basename "$f")" = libtalus_kernels.so ] && [[ " $ROCM " == *" $l "* ]]; then continue; fi
    bad "$(basename "$f") needs unexpected $l"
  done
  ok "$(basename "$f") NEEDED: $n| $(readelf -d "$f" | sed -n 's/.*(\(RUNPATH\|RPATH\)).*\[\(.*\)\]/\1 \2/p')"
  if strings -a "$f" | grep -qE '/home/[a-z]|/root/'; then bad "$(basename "$f") contains build-machine paths: $(strings -a "$f" | grep -E '/home/[a-z]|/root/' | head -2 | tr '\n' ' ')"; fi
done
g=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['runtime']['glibc_min'])" "$P/MANIFEST.json")
have=$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$')
python3 -c "import sys; a,b=[tuple(map(int,x.split('.'))) for x in sys.argv[1:]]; sys.exit(0 if b>=a else 1)" "$g" "$have" \
  && ok "glibc $have >= required $g" || bad "glibc $have < required $g"

# ldd: system-only without ROCm; everything resolved (and from the SDK) with it
nf=$(env -u LD_LIBRARY_PATH ldd "$P/bin/libtalus_kernels.so" | grep "not found" | awk '{print $1}' | tr '\n' ' ')
for l in $nf; do [[ " $ROCM " == *" $l "* ]] || bad "without ROCm: $l not found (not a ROCm library)"; done
ok "without ROCm, unresolved = ROCm sonames only: ${nf:-none}"
for b in talus-cli talus-server talus-tui; do
  nf=$(env -u LD_LIBRARY_PATH ldd "$P/bin/$b" | grep -c "not found"); [ "$nf" = 0 ] && ok "ldd $b: all resolved" || bad "ldd $b: $nf not found"
done
if [ -n "$SDK" ]; then
  ln -sfn "$(cd "$SDK" && pwd -P)" "$P/rocm-sdk"
  out=$(env -u LD_LIBRARY_PATH ldd -r "$P/bin/libtalus_kernels.so" 2>&1)
  echo "$out" | grep -q "not found\|undefined symbol" && bad "ldd -r with rocm-sdk: $(echo "$out" | grep 'not found\|undefined' | head -3)" \
    || ok "ldd -r with PREFIX/rocm-sdk -> $SDK: all libraries and symbols resolved"
  for l in $ROCM; do
    w=$(echo "$out" | awk -v l=$l '$1==l {print $3}')
    [[ "$w" == "$P/rocm-sdk/"* || "$w" == "$P/bin/../rocm-sdk/"* ]] && ok "$l -> $w" || bad "$l resolved to '$w', not via PREFIX/rocm-sdk"
  done
fi
[ $FAIL = 0 ] && echo "verify: PASS" || echo "verify: FAIL"
exit $FAIL
