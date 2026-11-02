#!/usr/bin/env bash
# tests/install/no-color.sh : install.sh prints no escape codes with NO_COLOR set or when stdout is not a terminal
# (itest18: the banner printed ESC[30;46;1m in both cases). Reads only the banner line: install.sh runs with --dry-run
# in a temporary HOME and is cut off by the pipe right after it. Needs `script` (util-linux) for the terminal cases.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H=$(mktemp -d)
trap 'rm -rf "$H"' EXIT
ESC=$(printf '\033')
banner() {  # $1: "pipe" or "tty"; the rest: env assignments
  local how=$1; shift
  if [ "$how" = pipe ]; then
    (cd "$H" && env HOME="$H" "$@" timeout 30 bash "$ROOT/install.sh" --dry-run --yes 2>&1 </dev/null | head -n 2)
  else
    (cd "$H" && env HOME="$H" "$@" timeout 30 script -qec "bash '$ROOT/install.sh' --dry-run --yes" /dev/null 2>&1 </dev/null | head -n 2)
  fi
}
fail=0
check() {  # $1: label, $2: want escapes (1/0), rest: banner args
  local label=$1 want=$2; shift 2
  local out; out=$(banner "$@")
  case "$out" in *Talus*installer*) ;; *) echo "no-color: $label: no banner in: $out" >&2; fail=1; return ;; esac
  local has=0; case "$out" in *"$ESC"*) has=1 ;; esac
  if [ "$has" != "$want" ]; then echo "no-color: $label: escape codes $has, want $want" >&2; fail=1; else echo "no-color: $label ok"; fi
}
check "pipe" 0 pipe
check "pipe + NO_COLOR" 0 pipe NO_COLOR=1
if command -v script >/dev/null; then
  check "tty + NO_COLOR" 0 tty NO_COLOR=1
  check "tty" 1 tty
else
  echo "no-color: no script(1), terminal cases skipped"
fi
exit $fail
