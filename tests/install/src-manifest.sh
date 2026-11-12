#!/usr/bin/env bash
# tests/install/src-manifest.sh : tools/src-manifest.sh catches what mtimes miss (talus#78), without ever touching a
# source (PR #87 review). A toy ninja project in a temp dir: a "compile" copies the .cu and the header it includes
# (depfile, deps = gcc, as CMake's HIP rules), a "link" concatenates the objects.
#   1. talus#78: a source edited after its object was compiled, but with an mtime OLDER than the object (as when the
#      object comes back from a build VM): plain ninja keeps the old object; after `refresh` the object holds the new content;
#   2. a header edit (depfile dep) rebuilds the object that includes it; refresh never changes a source's mtime;
#   3. nothing changed: refresh removes nothing;
#   4. a source edited between snap and verify (during a build): verify fails and names it;
#   5. upgrade: a build dir with a ninja log and objects but no manifest: refresh removes every object once;
#      a fresh build dir (no ninja log) is left alone;
#   6. coverage: a file under bench/ and under third_party/ is in the manifest;
#   7. the review's blocker, end to end through remote-build's download rule (find -newer STAMP | rsync --update): a
#      "VM" copy refreshes and builds while the "dev" copy is edited to v2; the download must not bring v1 back.
#   8. the real build.sh with a stub cmake: STALE_CHECK=0 keeps no manifest; build-repo.result OK / FAILED rc=N.
#   9. a real CMake project: a cmake/*.cmake and a CMakeLists.txt edit, two builds each in build.sh's order; build.ninja
#      survives refresh and the edit reaches the object's command line (cmake and cc; skipped without them).
# Needs ninja and python3. Nothing outside the temporary directory changes.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SM=$ROOT/tools/src-manifest.sh
command -v ninja >/dev/null || { echo "src-manifest test: ninja not found, skipped"; exit 0; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fail=0
no() { echo "src-manifest test: FAIL: $*" >&2; fail=1; }
mkproj() {  # DIR : the toy project (k.cu includes k.h; lib = k.o + j.o)
  mkdir -p "$1/src" "$1/include" "$1/b"
  echo v1 > "$1/src/k.cu"; echo h1 > "$1/include/k.h"; echo j1 > "$1/src/j.cu"
  cat > "$1/b/build.ninja" <<'EOF'
rule cc
  command = cat $in ../include/k.h > $out && printf '%s: %s %s\n' $out $in ../include/k.h > $out.d
  depfile = $out.d
  deps = gcc
rule cc0
  command = cp $in $out
rule ld
  command = cat $in > $out
build k.o: cc ../src/k.cu
build j.o: cc0 ../src/j.cu
build lib.so: ld k.o j.o
EOF
}
mt() { stat -c %Y "$1"; }
cd "$T"; P=$T/p; mkproj "$P"; cd "$P"; B=$P/b
# 5b. fresh build dir: refresh is a no-op
"$SM" refresh "$B" || no "refresh on a fresh build dir failed"
"$SM" snap "$B"; ninja -C b >/dev/null || no "first build"; "$SM" verify "$B" && "$SM" record "$B" || no "verify after a clean build"
[ "$(head -1 b/k.o)" = v1 ] || no "first build content"
# 1. edit, then make the objects look newer than the edit (the VM's mtime)
echo v2 > src/k.cu; touch -d '-1 hour' src/k.cu   # older than the object: what came back from the VM
ninja -C b >/dev/null
[ "$(head -1 b/k.o)" = v1 ] || no "precondition: plain ninja should have kept the stale object"
m0=$(mt src/k.cu)
out=$("$SM" refresh "$B")
echo "$out" | grep -q 'changed since the last build: src/k.cu' || no "refresh did not report src/k.cu: $out"
[ "$(mt src/k.cu)" = "$m0" ] || no "refresh changed a source's mtime"
"$SM" snap "$B"; ninja -C b >/dev/null
[ "$(head -1 b/k.o)" = v2 ] && grep -q '^v2$' b/lib.so || no "after refresh the object / library still hold the old content"
"$SM" verify "$B" && "$SM" record "$B" || no "verify after the rebuild"
# 2. header (depfile dep), object newer than the edit
echo h2 > include/k.h; touch -d '-1 hour' include/k.h
mh=$(mt include/k.h); jo=$(mt b/j.o)
"$SM" refresh "$B" >/dev/null; "$SM" snap "$B"; ninja -C b >/dev/null
grep -q '^h2$' b/k.o || no "a header edit did not rebuild the object that includes it"
[ "$(mt include/k.h)" = "$mh" ] || no "refresh changed a header's mtime"
[ "$(mt b/j.o)" = "$jo" ] || no "a header edit rebuilt an object that does not include it"
"$SM" verify "$B" && "$SM" record "$B" || no "verify after the header rebuild"
# 3. nothing changed
out=$("$SM" refresh "$B"); [ -z "$out" ] && [ -e b/k.o ] && [ -e b/j.o ] || no "refresh removed something with nothing changed: $out"
# 4. edited during the build
"$SM" snap "$B"; echo v3 > src/k.cu
err=$("$SM" verify "$B" 2>&1) && no "verify passed with a source edited during the build"
echo "$err" | grep -q 'src/k.cu' || no "verify did not name src/k.cu: $err"
echo v2 > src/k.cu
# 5. upgrade: no manifest, objects present
rm -f b/src-manifest b/src-manifest.new
out=$("$SM" refresh "$B")
echo "$out" | grep -q 'no manifest' && [ ! -e b/k.o ] && [ ! -e b/j.o ] || no "missing manifest did not remove the objects: $out"
"$SM" snap "$B"; ninja -C b >/dev/null; "$SM" verify "$B" && "$SM" record "$B" || no "rebuild after the upgrade"
# 6. coverage
mkdir -p bench third_party/ggml; echo x > bench/b.cu; echo y > third_party/ggml/g.h
"$SM" snap "$B"
grep -q ' bench/b.cu$' b/src-manifest.new && grep -q ' third_party/ggml/g.h$' b/src-manifest.new || no "bench/ or third_party/ not in the manifest"
rm -rf bench third_party; "$SM" snap "$B"; "$SM" record "$B"
# 7. the blocker: remote-build's download rule must not revert an dev edit. The rule is replayed with cp (the build
#    image has no rsync): every file under the VM copy newer than STAMP replaces the dev file unless that is newer
#    still (rsync --update's rule).
E=$T/dev; V=$T/vm; mkproj "$E"; (cd "$E" && "$SM" snap b && ninja -C b >/dev/null && "$SM" verify b && "$SM" record b)
cp -a "$E" "$V"                                          # upload
STAMP=$T/stamp; touch "$STAMP"; sleep 1
echo v2 > "$E/src/k.cu"                                  # the dev edit while the VM waits / builds
sleep 1
# on the VM: make the uploaded src/k.cu read as changed against the manifest, so refresh does its work
(cd "$V" && sed -i 's/^[0-9a-f]\{40\}  src\/k.cu$/0000000000000000000000000000000000000000  src\/k.cu/' b/src-manifest \
   && "$SM" refresh b >/dev/null && "$SM" snap b && ninja -C b >/dev/null && "$SM" verify b && "$SM" record b)
(cd "$V" && find . -newer "$STAMP" \( -type f -o -type l \) -print) | while IFS= read -r f; do
  [ -e "$E/$f" ] && [ ! "$V/$f" -nt "$E/$f" ] && continue   # --update: keep the receiver's newer file
  mkdir -p "$(dirname "$E/$f")"; cp -a "$V/$f" "$E/$f"
done
[ "$(cat "$E/src/k.cu")" = v2 ] || no "remote-build download reverted the dev edit: dev src/k.cu is now '$(cat "$E/src/k.cu")'"
(cd "$V" && find . -newer "$STAMP" -type f -print) | grep -q '^\./src/' && no "the VM build changed a source (it would be downloaded)"
# 8. through the real build.sh (stub cmake on PATH runs the toy ninja project; empty stand-in SDK):
#    STALE_CHECK=0 writes no manifest and removes an old one; a checked build records it and writes build-repo.result OK;
#    a failing build writes "FAILED rc=N" even when the caller hides the status (`./build.sh; true`)
R=$T/repo; mkdir -p "$R/tools" "$R/sdk/include/hip" "$R/path"; cp "$ROOT/build.sh" "$R/"; cp "$SM" "$R/tools/"
touch "$R/sdk/include/hip/hip_runtime.h"; mkproj "$R"; mv "$R/b" "$R/toy"
{
  echo '#!/bin/bash'
  echo '# stub: "cmake -S . -B DIR ..." puts the toy build.ninja in DIR; "cmake --build DIR ..." runs ninja there'
  echo 'if [ "$1" = --build ]; then [ -e "$PWD/FAIL" ] && { echo "error: stub compile failed" >&2; exit 1; }; exec ninja -C "$2"; fi'
  echo 'while [ $# -gt 0 ]; do [ "$1" = -B ] && { mkdir -p "$2"; cp toy/build.ninja "$2/"; }; shift; done'
} > "$R/path/cmake"
chmod +x "$R/path/cmake"
bs() { (cd "$R" && env PATH="$R/path:$PATH" ROCM_SDK="$R/sdk" BUILD=bd TARGETS=lib.so "$@" bash build.sh) > "$T/bs.out" 2>&1; }
bs STALE_CHECK=0 || no "STALE_CHECK=0 build failed: $(tail -3 "$T/bs.out")"
[ ! -e "$R/bd/src-manifest" ] && [ ! -e "$R/bd/src-manifest.new" ] || no "STALE_CHECK=0 wrote a manifest"
bs || no "checked build failed: $(tail -3 "$T/bs.out")"
[ -s "$R/bd/src-manifest" ] || no "a checked build did not record the manifest"
[ "$(cat "$R/bd/build-repo.result" 2>/dev/null)" = "build.sh: OK" ] || no "build-repo.result after a good build: '$(cat "$R/bd/build-repo.result" 2>/dev/null)'"
bs STALE_CHECK=0 || no "second STALE_CHECK=0 build failed"
[ ! -e "$R/bd/src-manifest" ] || no "STALE_CHECK=0 left the old manifest in place (an unchecked build would be vouched for)"
touch "$R/FAIL"; (cd "$R" && env PATH="$R/path:$PATH" ROCM_SDK="$R/sdk" BUILD=bd TARGETS=lib.so bash build.sh > /dev/null 2>&1; true)
grep -q '^build.sh: FAILED rc=[1-9]' "$R/bd/build-repo.result" 2>/dev/null || no "a failed build hidden by '; true' left '$(cat "$R/bd/build-repo.result" 2>/dev/null)'"
rm -f "$R/FAIL"
# 9. a real CMake + ninja project (PR #87 re-review): an edit to an included cmake/*.cmake file and to CMakeLists.txt,
#    each followed by two builds in build.sh's order (configure, refresh, snap, build, verify, record). Every build must
#    succeed, build.ninja must survive refresh, and the edit must reach the object (a -D define read from the .cmake).
if command -v cmake >/dev/null && command -v cc >/dev/null; then
  C=$T/cm; mkdir -p "$C/src" "$C/cmake"
  echo 'set(K_VAL 1)' > "$C/cmake/k.cmake"
  printf 'int k(void) { return K_VAL; }\n' > "$C/src/k.c"
  printf 'int j(void) { return 7; }\n' > "$C/src/j.c"
  cat > "$C/CMakeLists.txt" <<'CML'
cmake_minimum_required(VERSION 3.16)
project(t C)
include(cmake/k.cmake)
add_library(t STATIC src/k.c src/j.c)
target_compile_definitions(t PRIVATE K_VAL=${K_VAL})
CML
  cm_build() {  # build.sh's order; prints the step that failed
    ( cd "$C" && cmake -S . -B b -G Ninja > "$T/cm.log" 2>&1 || { echo configure; exit 1; }
      "$SM" refresh b >> "$T/cm.log" 2>&1 || { echo refresh; exit 1; }
      [ -f b/build.ninja ] || { echo "refresh deleted build.ninja"; exit 1; }
      "$SM" snap b && cmake --build b >> "$T/cm.log" 2>&1 || { echo build; exit 1; }
      "$SM" verify b && "$SM" record b || { echo verify; exit 1; } )
  }
  kval() { (cd "$C/b" && ninja -t commands CMakeFiles/t.dir/src/k.c.o | grep -o 'K_VAL=[0-9]*' | head -1); }
  r=$(cm_build) || no "cmake project: first build failed at: $r ($(tail -2 "$T/cm.log"))"
  echo 'set(K_VAL 2)' > "$C/cmake/k.cmake"                    # a .cmake edit
  for i in 1 2; do r=$(cm_build) || no "cmake project: build $i after a cmake/*.cmake edit failed at: $r ($(tail -2 "$T/cm.log"))"; done
  [ "$(kval)" = K_VAL=2 ] || no "the cmake/k.cmake edit did not reach k.c.o's command line: $(kval)"
  grep -q 'K_VAL=2' "$C/b/build.ninja" || no "build.ninja does not hold the edited define"
  printf '\n# a comment\n' >> "$C/CMakeLists.txt"              # a CMakeLists.txt edit
  for i in 1 2; do r=$(cm_build) || no "cmake project: build $i after a CMakeLists.txt edit failed at: $r ($(tail -2 "$T/cm.log"))"; done
  printf 'int k(void) { return K_VAL + 1; }\n' > "$C/src/k.c"; touch -d '-1 hour' "$C/src/k.c"   # and a source, back-dated
  r=$(cm_build) || no "cmake project: build after a back-dated source edit failed at: $r"
  grep -q 'removed CMakeFiles/t.dir/src/k.c.o' "$T/cm.log" || no "refresh did not remove k.c.o after a back-dated k.c edit"
else
  echo "src-manifest test: cmake or cc not found, case 9 skipped"
fi
[ $fail = 0 ] && echo "src-manifest test: PASS (9 cases)"
exit $fail
