#!/bin/bash
# tools/src-manifest.sh snap|refresh|verify|record BUILD : content check of the sources for build.sh (talus#78, PR #87).
#
# ninja and build.sh's stale-build guard compare mtimes only. That misses an object built from OLDER content whose
# mtime is still newer than the source: remote-build.sh uploads the tree, the source is edited on the dev host while the VM
# compiles, and the object comes back with the VM's (later) mtime. Every later build then keeps the old object
# (talus#78: q36_kernels.cu.o without the gdn_ab 32-row kernels, 2026-11-07).
#
# BUILD/src-manifest records "sha1  path" for every source the last VERIFIED build compiled from:
#   snap     hash the sources now into BUILD/src-manifest.new (the content this build compiles from)
#   refresh  (after configure) delete every compiled object (.o / .obj) in BUILD whose inputs (ninja's direct inputs
#            and its depfile headers) include a source whose content differs from BUILD/src-manifest, so ninja remakes
#            it. Generated build files (build.ninja and the rest of CMake's) are never deleted: a CMakeLists.txt /
#            cmake/*.cmake edit is configure's to apply, and ninja rebuilds an object whose command line changed.
#            Sources are NEVER touched: a touched source on a build VM is newer than remote-build's stamp and would be
#            copied back over an edit made on the dev host meanwhile (PR #87 review). A build dir with a ninja log but no
#            manifest (built before this check, or by STALE_CHECK=0) is unknown: every object is removed once, so an
#            already stale object cannot survive. A fresh build dir (no ninja log) needs nothing.
#   verify   the sources still match BUILD/src-manifest.new (exit 1 and list them otherwise: edited during the build)
#   record   BUILD/src-manifest.new becomes BUILD/src-manifest; build.sh runs it only after verify passed
# Sources: every file under src include cmake tests bench third_party plus CMakeLists.txt, relative to the repo root
# (the cwd). Needs sha1sum, find, xargs (GNU); refresh also ninja and python3.
set -euo pipefail
cmd=${1:?snap|refresh|verify|record}; B=${2:?build dir}
M=$B/src-manifest
hashes() {  # (paths that do not exist are skipped)
  local d=() p
  for p in src include cmake tests bench third_party CMakeLists.txt; do [ -e "$p" ] && d+=("$p"); done
  [ ${#d[@]} -gt 0 ] || return 0
  find "${d[@]}" -type f | LC_ALL=C sort | xargs -r -d '\n' sha1sum
}
changed() {  # OLD NEW : paths whose hash differs or that are new in NEW
  awk 'NR == FNR { h[substr($0, 43)] = $1; next } { p = substr($0, 43); if (!(p in h) || h[p] != $1) print p }' "$1" "$2"
}
case $cmd in
  snap) mkdir -p "$B"; hashes > "$M.new" ;;
  refresh)
    [ -f "$B/.ninja_log" ] || exit 0                       # a fresh build dir: everything gets built anyway
    root=$(pwd -P)
    if [ ! -s "$M" ]; then                                 # unknown provenance: rebuild every object once
      n=$(find "$B" -type f \( -name '*.o' -o -name '*.obj' \) -print -delete | wc -l)
      echo "src-manifest: no manifest in $B (built before the content check): removed $n objects, rebuilding once"
      exit 0
    fi
    cur=$(mktemp); chg=$(mktemp); trap 'rm -f "$cur" "$chg"' EXIT
    hashes > "$cur"
    changed "$M" "$cur" | sed "s#^#$root/#" > "$chg"
    [ -s "$chg" ] || exit 0
    head -10 "$chg" | sed "s#^$root/#src-manifest: changed since the last build: #"
    # outputs to remake: ninja's consumers of a changed file (direct inputs: -t query) and every object whose depfile
    # deps name one (-t deps); removing them makes ninja rebuild them and what links them
    n=0
    while IFS= read -r o; do
      [ -n "$o" ] || continue
      rm -f -- "$B/$o"; n=$((n + 1))
      [ $n -le 10 ] && echo "src-manifest: removed $o"
    done < <(python3 - "$B" "$chg" <<'PY'
import os, subprocess, sys
b, chg = sys.argv[1], sys.argv[2]
want = {os.path.realpath(l.rstrip("\n")) for l in open(chg) if l.strip()}
out = set()
deps = subprocess.run(["ninja", "-C", b, "-t", "deps"], capture_output=True, text=True).stdout
o = None
for l in deps.splitlines():
    if l and not l.startswith(" "):
        o = l.split(": #deps")[0] if ": #deps" in l else None
    elif o and l.startswith("    "):
        f = l[4:]
        if os.path.realpath(f if os.path.isabs(f) else os.path.join(b, f)) in want:
            out.add(o)
for f in want:   # CMake's ninja files name sources by absolute path; a relative spelling is tried as well
    for name in (f, os.path.relpath(f, b)):
        q = subprocess.run(["ninja", "-C", b, "-t", "query", name], capture_output=True, text=True)
        if q.returncode == 0:
            break
    sec = False
    for l in q.stdout.splitlines():
        if l.startswith("  ") and not l.startswith("    "):
            sec = l.strip() == "outputs:"
        elif sec and l.startswith("    "):
            out.add(l.strip())
# only compiled objects: removing one makes ninja remake it and relink what uses it. Never the build system's own files
# (build.ninja consumes CMakeLists.txt and cmake/*.cmake; deleting it after configure broke every later build, PR #87
# re-review): regenerating those is CMake's job (configure runs before refresh, and ninja's RERUN_CMAKE rule).
print("\n".join(sorted(o for o in out if o.endswith(('.o', '.obj')))))
PY
)
    [ $n -le 10 ] || echo "src-manifest: ... $n outputs removed in all"
    ;;
  verify)
    [ -s "$M.new" ] || { echo "src-manifest: no snapshot ($M.new)" >&2; exit 1; }
    cur=$(mktemp); trap 'rm -f "$cur"' EXIT
    hashes > "$cur"
    d=$(changed "$M.new" "$cur" | head -5)
    [ -z "$d" ] || { echo "src-manifest: sources changed while building (the outputs hold the old content):" >&2; echo "$d" >&2; exit 1; }
    ;;
  record) mv -f "$M.new" "$M" ;;
  *) echo "src-manifest: unknown command $cmd" >&2; exit 2 ;;
esac
