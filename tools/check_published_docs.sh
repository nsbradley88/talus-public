#!/usr/bin/env bash
# Claim check on the dev copies of the PUBLISHED Markdown only: README.md + docs/*.md (published from the public doc
# templates) and the other *.md files the public tree carries (synced from this repository by the release refresh).
# Internal records (bench/results, TALUS.md, translations, other docs) are not published and are not checked.
# Any doc published later MUST be added to FILES below.
# Usage: tools/check_published_docs.sh VALUES_JSON   (the release values file)
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
[ $# -eq 1 ] || { echo "usage: $0 VALUES_JSON" >&2; exit 2; }
VALUES=$1
FILES=(README.md docs/ARCHITECTURE.md docs/INSTALL-GUIDED.md docs/METHODOLOGY.md docs/MODELS.md docs/RESULTS.md docs/TUI.md
       AGENTS.md SECURITY.md THIRD_PARTY_NOTICES.md scripts/release/README.md tools/w4i/README.md tools/w4i/q4q/README.md)
tmp=$(mktemp -d "${TMPDIR:-/tmp}/talus-claims.XXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
mkdir -p "$tmp/tools"
for f in "${FILES[@]}"; do mkdir -p "$tmp/$(dirname "$f")"; cp -- "$here/$f" "$tmp/$f"; done
cp -- "$here/tools/check_claims.py" "$tmp/tools/check_claims.py"
python3 -B "$tmp/tools/check_claims.py" check --values "$VALUES" "$tmp"
