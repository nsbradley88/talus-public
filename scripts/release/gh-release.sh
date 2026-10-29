#!/bin/bash
# gh-release.sh OUT_DIR [--notes-changes FILE] : PRINT the GitHub release commands for a built tarball. Runs nothing
# remote: it renders OUT_DIR/RELEASE-NOTES.md from release-notes.md.tmpl and prints the `gh release create` line.
# Publishing is a maintainer decision; copy the printed commands to publish.
set -euo pipefail
OUT=${1:?usage: gh-release.sh OUT_DIR [--notes-changes FILE]}; CH=""
[ "${2:-}" = --notes-changes ] && CH=$(cat "$3")
REPO_GH=${REPO_GH:-nsbradley88/talus-public}
M=$(ls "$OUT"/*.manifest.json | head -1); PKG=$(basename "$M" .manifest.json)
j() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$M" "$1"; }
TAG=$(j "d['tag']"); COMMIT=$(j "d['commit']"); ROCM=$(j "d['build']['rocm_sdk']['version']"); GLIBC=$(j "d['runtime']['glibc_min']")
SETUP=$(j "' and talus-setup' if 'bin/talus-setup' in d['files'] else ''")
python3 - "$(dirname "$0")/release-notes.md.tmpl" "$OUT/RELEASE-NOTES.md" <<PY
import sys
s = open(sys.argv[1]).read()
for k, v in {"TAG": "$TAG", "COMMIT": "$COMMIT", "PKG": "$PKG", "ROCM_VERSION": "$ROCM", "GLIBC_MIN": "$GLIBC",
             "SETUP_BIN": "$SETUP", "CHANGES": """${CH:-(fill in)}""", "SHA256SUMS": open("$OUT/SHA256SUMS").read().strip()}.items():
    s = s.replace("{{" + k + "}}", v)
open(sys.argv[2], "w").write(s)
PY
echo "# notes: $OUT/RELEASE-NOTES.md   (review it; the tag $TAG must exist on $REPO_GH and point at $COMMIT)"
echo "gh release create '$TAG' --repo '$REPO_GH' --verify-tag --draft --title 'Talus $TAG' \\"
echo "  --notes-file '$OUT/RELEASE-NOTES.md' \\"
echo "  '$OUT/$PKG.tar.gz' '$OUT/$PKG.tar.gz.sha256' '$OUT/$PKG.manifest.json' '$OUT/SHA256SUMS'"
echo "# then, after checking the draft on GitHub:  gh release edit '$TAG' --repo '$REPO_GH' --draft=false --latest"
