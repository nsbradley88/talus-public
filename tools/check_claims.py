#!/usr/bin/env python3
"""check_claims.py: keep performance and quality numbers out of the docs unless a release run produced them.

Every performance or quality figure in this repository's Markdown is a placeholder such as {{rel:qwen27b:chat}}. A
release run writes a values file (JSON, format below); the `fill` step substitutes the placeholders from it. A model
whose release run did not pass renders every one of its placeholders as "pending".

Usage
  check_claims.py check [ROOT] [--values FILE] [--cite-ok PATH ...]
      Scan every *.md under ROOT (default: the repository root, the parent of tools/) and fail (exit 1) on any number
      that looks like a performance or quality claim: tok/s, %, a multiplier such as 2x, a KL / KLD / perplexity /
      top-1 value, a score such as 14/15 next to needle / agentic / tasks / requests, a TTFT or latency in s / ms.
      Numbers inside {{rel:...}} placeholders are fine. With --values, a number equal to a value of a passing model
      (or a build value) in FILE is fine too: that is how a filled tree passes.
      Table rows that start with "| Community" cite other people's published results; they are allowed only in the
      files given with --cite-ok (default: docs/RESULTS.md), and only if that file links each source.
  check_claims.py check --release [ROOT]
      The same, and also fail on leftover "TODO(release)" / "TBD(" markers: run it on the tree you publish.
  check_claims.py fill --values FILE [--out DIR | --in-place] [ROOT]
      Substitute every {{rel:...}} placeholder under ROOT and write the filled Markdown to DIR (a mirror of ROOT's
      *.md files) or in place. Then run `check` on the result with the same values file.
  check_claims.py keys [ROOT]
      List every placeholder used under ROOT, with the files that use it.
  check_claims.py trace --template TDIR --values FILE [ROOT]
      Prove that every figure in a filled tree comes from FILE: each *.md under TDIR (the placeholder template) must
      pass `check` on its own (no figures outside placeholders) and, filled from FILE, be byte-identical to the same
      file under ROOT; every other *.md under ROOT must pass `check` without any values (no figures at all). No model
      may render "pending" unless --allow-pending is given.

Placeholders
  {{rel:<model>:<key>}}           a figure of the Talus release run for <model>
  {{rel:<model>:<key>:<engine>}}  the same figure for a reference engine re-measured in that run (e.g. halo4, cpp)
  {{rel:build:<key>}}             facts of the frozen build (commit, tag, date, rocm)
  {{rel:rocm_version}}            short form of {{rel:build:rocm}} (also {{rel:commit}}, {{rel:tag}})
  {{rel:gate:<key>}}              the pass criteria the release run was judged by (noise bands band_<metric>,
                                  text_noise_1p / text_noise_4p, soak_rss, soak_drift, kld_<model>)
  <model> is one of qwen27b, flashnext, glm, mimo, laguna. Keys used by the docs: chat code edit pp2k pp32k tg2k
  tg32k ttft_a ttft_b ttft_c memavail_min needle60 needle81 agentic kld kld_raw kld_gate soak run_tag run_commit
  run_date crosscheck kld_orig. Build keys: commit tag date rocm freeze_note.

Values file (written by the release tooling from the run records, e.g. runs/<TAG>/release-values.json)
  {"tag": "...", "build": {"commit": "...", ...}, "gates": {"band_chat": "...", ...},
   "models": {"qwen27b": {"status": "pass", "values": {"chat": "...", "chat:halo4": "...", ...}}, ...}}
  Only status "pass" fills a model's placeholders; any other status, a missing model or a missing key renders
  "pending". Values are strings, inserted verbatim.

Exit codes: 0 clean, 1 claims found (or unknown placeholders in fill), 2 usage error.
Standard library only."""
import argparse
import json
import os
import re
import sys

MODELS = ("qwen27b", "flashnext", "glm", "mimo", "laguna")
KNOWN_KEYS = ("chat", "code", "edit", "pp2k", "pp32k", "tg2k", "tg32k", "ttft_a", "ttft_b", "ttft_c", "memavail_min",
              "needle60", "needle81", "agentic", "kld", "kld_raw", "kld_gate", "soak",
              "run_tag", "run_commit", "run_date", "crosscheck", "kld_orig")
BUILD_KEYS = ("commit", "tag", "date", "rocm", "freeze_note")
GATE_KEYS = tuple(f"band_{k}" for k in ("chat", "code", "edit", "pp2k", "pp32k", "tg2k", "tg32k", "ttft_a", "ttft_b",
                                        "ttft_c")) + ("text_noise_1p", "text_noise_4p", "soak_rss", "soak_drift") + \
    tuple(f"kld_{m}" for m in MODELS)
PENDING = "pending"

PLACEHOLDER = re.compile(r"\{\{\s*rel:([A-Za-z0-9_.-]+)(?::([A-Za-z0-9_.-]+))?(?::([A-Za-z0-9_.-]+))?\s*\}\}")
ALIASES = {"rocm_version": ("build", "rocm"), "commit": ("build", "commit"), "tag": ("build", "tag")}

NUM = r"(?<![\w.])(?<![A-Za-z_]-)(\d[\d,]*(?:\.\d+)?|\.\d+)"           # 12, 1,138, 0.5, .25 (not part of a word or version)
DEC = r"(?<![\w.])(?<![A-Za-z_]-)(\d+\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\d.]|[KMG]\b)"   # 0.0028, 4.3e-14
# (name, regex, group holding the number). Each finding is one number in a claim-like context.
RULES = [
    ("rate", re.compile(NUM + r"\s*(?:-|to|–)?\s*(?:\d[\d,.]*\s*)?(?:tok/s|tokens?/s|t/s|tps|tokens per second)\b",
                        re.I), 1),
    ("percent", re.compile(r"[-+±]?" + NUM + r"\s*(?:%|percent\b)", re.I), 1),
    ("multiplier", re.compile(NUM + r"\s?[x×](?![\w\d])(?!\s*\d)", re.I), 1),
    ("times", re.compile(NUM + r"\s*(?:times|-fold|fold)\s+(?:faster|slower|higher|lower|more|less|the)\b", re.I), 1),
    ("divergence", re.compile(r"\b(?:KLD|KL|kl|perplexity|PPL|ppl|top-1|top1|same top)\b[^.\n|]{0,40}?" + DEC), 1),
    ("score", re.compile(r"(?<![\w/.])(\d+)\s*/\s*(\d+)(?![\w/]|\.\d)(?=[^\n|]{0,40}\b(?:needle|agentic|tasks?|requests?|"
                         r"probes?|pass(?:ed|es)?|seeds?)\b)", re.I), 0),
    ("score", re.compile(r"\b(?:needle|agentic|tasks?|requests?|probes?|pass(?:ed|es)?)\b[^\n|]{0,40}?(?<![\w/.])"
                         r"(\d+\s*/\s*\d+)(?![\w/]|\.\d)", re.I), 1),
    ("latency", re.compile(r"\b(?:TTFT|latency|first token)\b[^.\n|]{0,40}?" + NUM + r"\s*(?:ms|s|sec|seconds)\b", re.I),
     1),
]
# Context lengths and sizes (2K, 64K, 128 GB, 70 W) are facts, not claims, and are never matched by the rules above.

FENCE = re.compile(r"^\s*(```|~~~)")

# Tables: a bare number in a cell of a table whose header names a metric is a claim even without a unit next to it.
METRIC_HDR = re.compile(r"\b(?:chat|code|edit|pp\s?@?\d+k?|tg\s?@?\d+k?|tok/s|tokens/s|TTFT|KLD|KL|needle|agentic|soak|"
                        r"MemAvail\w*|decode|prefill|prompt|speed|throughput|latency|score|result)\b", re.I)
TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
BARE_CELL = re.compile(r"^[\s*_`~<>≈≥≤=†‡+\-]*(\d[\d,]*(?:\.\d+)?)(?:\s*(?:-|–|to|/)\s*\d[\d,]*(?:\.\d+)?)?"
                       r"(?:\s*(?:s|ms|tok/s|GiB|GB|%|x|×))?[\s*_`†‡]*(?:\((?:1p|\d+p)\))?[\s*_`†‡]*$", re.I)


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md_files(root):
    skip = {".git", "target", "build", "node_modules", "third_party", ".venv"}
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in skip)
        for f in sorted(files):
            if f.lower().endswith(".md"):
                yield os.path.join(d, f)


def load_values(path):
    if not path:
        return None
    with open(path) as fh:
        v = json.load(fh)
    if not isinstance(v, dict) or not isinstance(v.get("models", {}), dict):
        sys.exit(f"check_claims: {path}: not a values file (needs a 'models' object)")
    return v


def norm_num(s):
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return repr(float(s))
    except ValueError:
        return s


# Which value keys may justify a number found by each rule (after fill). Multipliers and "N times faster" are never
# release values: they are derived comparisons, so they always fail.
RULE_KEYS = {
    "rate": ("chat", "code", "edit", "pp2k", "pp32k", "tg2k", "tg32k", "kld", "kld_raw", "needle60", "needle81"),
    "percent": ("band_", "text_noise_", "soak", "soak_", "kld", "kld_raw", "kld_gate", "kld_orig"),
    "divergence": ("kld", "kld_raw", "kld_gate", "kld_"),
    "score": ("needle60", "needle81", "agentic", "soak", "kld", "kld_raw"),
    "latency": ("ttft_a", "ttft_b", "ttft_c"),
    "table-cell": None,   # a filled table may show any release value
}


def allowed_numbers(values):
    """rule -> numbers that appear in the matching values of a passing model, the build or the gates"""
    ok = {r: set() for r in RULE_KEYS}
    if not values:
        return ok
    items = [(k, str(v)) for k, v in (values.get("build") or {}).items()]
    items += [(k, str(v)) for k, v in (values.get("gates") or {}).items()]
    for m in (values.get("models") or {}).values():
        if isinstance(m, dict) and m.get("status") == "pass":
            items += [(k.split(":")[0], str(v)) for k, v in (m.get("values") or {}).items()]
    for k, t in items:
        nums = {norm_num(n) for n in re.findall(r"\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?|\.\d+", t)}
        nums |= {f"{a}/{b}" for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", t)}
        for rule, keys in RULE_KEYS.items():
            if keys is None or k in keys or any(x.endswith("_") and k.startswith(x) for x in keys):
                ok[rule] |= nums
    return ok


def resolve(values, scope, key, engine):
    """placeholder -> text. Returns (text, known) where known is False for a placeholder outside the documented grammar"""
    if key is None:                      # one-part form, e.g. {{rel:rocm_version}}
        if scope not in ALIASES:
            return PENDING, False
        scope, key = ALIASES[scope]
    if scope in ("build", "gate"):
        known = key in (BUILD_KEYS if scope == "build" else GATE_KEYS) and engine is None
        v = (values or {}).get(scope if scope == "build" else "gates", {}).get(key) if values else None
        return (str(v) if v not in (None, "") else PENDING), known
    known = scope in MODELS and key in KNOWN_KEYS
    if not values:
        return PENDING, known
    m = (values.get("models") or {}).get(scope)
    if not isinstance(m, dict) or m.get("status") != "pass":
        return PENDING, known
    v = (m.get("values") or {}).get(key if engine is None else f"{key}:{engine}")
    return (str(v) if v not in (None, "") else PENDING), known


def scan_text(text, path, ok_numbers, cite_ok):
    """findings: (line number, rule, matched text, line)"""
    out = []
    in_fence = False
    cite_file = cite_ok and (os.path.abspath(path) in cite_ok)
    has_links = bool(re.search(r"https?://", text))
    lines = text.split("\n")
    metric_table = False
    for i, raw in enumerate(lines, 1):
        if not raw.lstrip().startswith("|"):
            metric_table = False
        elif i < len(lines) and TABLE_SEP.match(lines[i]):          # raw is a header row (the next line separates)
            metric_table = bool(METRIC_HDR.search(PLACEHOLDER.sub(" ", raw)))
        elif metric_table and not TABLE_SEP.match(raw) and not re.match(r"^\s*\|\s*Community", raw):
            cells = PLACEHOLDER.sub(" ", raw).strip().strip("|").split("|")[1:]
            for cell in [c for cell in cells for c in re.split(r"<br\s*/?>", cell, flags=re.I)]:
                mt = BARE_CELL.match(cell)
                if mt and norm_num(mt.group(1)) not in ok_numbers["table-cell"]:
                    out.append((i, "table-cell", cell.strip(), raw, "a number in a metric table"))
        if FENCE.match(raw):
            in_fence = not in_fence   # code blocks are scanned too: a number in sample output is still a claim
        line = PLACEHOLDER.sub(" ", raw)
        if re.match(r"^\s*\|\s*Community", line):
            if cite_file and has_links:
                continue
            out.append((i, "community-row", line.strip()[:60], raw, "community rows only in --cite-ok files with links"))
            continue
        for name, rx, g in RULES:
            for mt in rx.finditer(line):
                num = mt.group(g) if g else f"{mt.group(1)}/{mt.group(2)}"
                n = re.sub(r"\s+", "", num) if "/" in num else norm_num(num)
                if n in ok_numbers.get(name, ()):
                    continue
                out.append((i, name, mt.group(0).strip(), raw, ""))
    return out


def cmd_check(root, values, cite_ok, quiet=False):
    ok = allowed_numbers(values)
    total = 0
    bad_ph = 0
    for p in md_files(root):
        text = open(p, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(p, root)
        for mt in PLACEHOLDER.finditer(text):
            _, known = resolve(None, mt.group(1), mt.group(2), mt.group(3))
            if not known:
                ln = text.count("\n", 0, mt.start()) + 1
                print(f"{rel}:{ln}: unknown placeholder {mt.group(0)}")
                bad_ph += 1
        for ln, rule, frag, raw, note in scan_text(text, p, ok, cite_ok):
            total += 1
            print(f"{rel}:{ln}: [{rule}] {frag!r}{'  (' + note + ')' if note else ''}\n    {raw.strip()[:160]}")
    if total or bad_ph:
        print(f"\ncheck_claims: FAIL: {total} claim-like number(s), {bad_ph} unknown placeholder(s). Replace each figure "
              "with a {{rel:...}} placeholder (or cite it in the results page's Community rows).")
        return 1
    if not quiet:
        print(f"check_claims: OK ({root})")
    return 0


def cmd_fill(root, values, out_dir, in_place, cite_ok):
    if not values:
        sys.exit("check_claims fill: --values FILE is required")
    if not out_dir and not in_place:
        sys.exit("check_claims fill: give --out DIR or --in-place")
    unknown = 0
    pend = {}
    filled = 0
    for p in md_files(root):
        text = open(p, encoding="utf-8").read()

        def sub(mt):
            nonlocal unknown, filled
            v, known = resolve(values, mt.group(1), mt.group(2), mt.group(3))
            if not known:
                unknown += 1
                print(f"{os.path.relpath(p, root)}: unknown placeholder {mt.group(0)}")
            if v == PENDING:
                pend[mt.group(1)] = pend.get(mt.group(1), 0) + 1
            else:
                filled += 1
            return v

        new = PLACEHOLDER.sub(sub, text)
        dst = p if in_place else os.path.join(out_dir, os.path.relpath(p, root))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(new)
    print(f"check_claims fill: {filled} placeholder(s) filled from {values.get('tag', '?')}; pending: "
          + (", ".join(f"{k} {v}" for k, v in sorted(pend.items())) or "none"))
    for m in MODELS:
        st = ((values.get("models") or {}).get(m) or {}).get("status", "missing")
        if st != "pass":
            print(f"  {m}: {st} -> pending")
    target = root if in_place else out_dir
    if in_place or out_dir:
        cite = {os.path.join(os.path.abspath(target), os.path.relpath(c, root)) for c in cite_ok} if cite_ok else set()
        rc = cmd_check(target, values, cite)
        return 1 if (rc or unknown) else 0
    return 1 if unknown else 0


def cmd_keys(root):
    seen = {}
    for p in md_files(root):
        for mt in PLACEHOLDER.finditer(open(p, encoding="utf-8").read()):
            seen.setdefault(mt.group(0), set()).add(os.path.relpath(p, root))
    for k in sorted(seen):
        print(f"{k}  {', '.join(sorted(seen[k]))}")
    return 0


def fill_text(text, values):
    """(filled text, unknown placeholders, pending placeholders)"""
    unknown, pend = [], []

    def sub(mt):
        v, known = resolve(values, mt.group(1), mt.group(2), mt.group(3))
        if not known:
            unknown.append(mt.group(0))
        if v == PENDING:
            pend.append(mt.group(0))
        return v
    return PLACEHOLDER.sub(sub, text), unknown, pend


def cmd_trace(root, template, values, cite_ok, allow_pending=False):
    if not values:
        sys.exit("check_claims trace: --values FILE is required")
    bad = 0
    print(f"trace: template {template}")
    tcite = {os.path.join(template, os.path.relpath(c, root)) for c in cite_ok}
    if cmd_check(template, None, tcite, quiet=True):
        print("trace: the template itself holds figures outside placeholders (above)")
        bad += 1
    tfiles = {os.path.relpath(p, template) for p in md_files(template)}
    for rel in sorted(tfiles):
        dst = os.path.join(root, rel)
        new, unknown, pend = fill_text(open(os.path.join(template, rel), encoding="utf-8").read(), values)
        for u in unknown:
            print(f"{rel}: unknown placeholder {u}")
            bad += 1
        if pend and not allow_pending:
            print(f"{rel}: {len(pend)} placeholder(s) render pending: {', '.join(sorted(set(pend)))[:300]}")
            bad += 1
        if not os.path.exists(dst):
            print(f"{rel}: in the template but missing under {root}")
            bad += 1
            continue
        cur = open(dst, encoding="utf-8").read()
        if cur != new:
            a, b = cur.split("\n"), new.split("\n")
            i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            print(f"{rel}:{i + 1}: differs from the template filled with the values file\n    tree:     "
                  f"{(a[i] if i < len(a) else '<eof>')[:150]}\n    expected: {(b[i] if i < len(b) else '<eof>')[:150]}")
            bad += 1
    others = [p for p in md_files(root) if os.path.relpath(p, root) not in tfiles]
    ok = allowed_numbers(None)
    for p in others:
        text = open(p, encoding="utf-8", errors="replace").read()
        for ln, rule, frag, raw, note in scan_text(text, p, ok, cite_ok):
            print(f"{os.path.relpath(p, root)}:{ln}: [{rule}] {frag!r} (not a template file: no figures allowed)")
            bad += 1
    if bad:
        print(f"check_claims trace: FAIL ({bad} problem(s))")
        return 1
    print(f"check_claims trace: OK: {len(tfiles)} template file(s) reproduce exactly from "
          f"{values.get('tag', '?')}, {len(others)} other file(s) hold no figures ({root})")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("check", "fill", "keys", "trace"))
    ap.add_argument("root", nargs="?", default=None)
    ap.add_argument("--values", help="release values JSON (written by the release tooling)")
    ap.add_argument("--out", help="fill: write the filled tree here")
    ap.add_argument("--in-place", action="store_true", help="fill: overwrite the Markdown under ROOT")
    ap.add_argument("--release", action="store_true",
                    help="check: also fail on leftover TODO(release) / TBD markers (run before publishing)")
    ap.add_argument("--template", help="trace: the placeholder template the tree was filled from")
    ap.add_argument("--allow-pending", action="store_true", help="trace: allow placeholders that render pending")
    ap.add_argument("--cite-ok", action="append", default=None,
                    help="file (relative to ROOT) allowed to hold '| Community' rows; default docs/RESULTS.md")
    a = ap.parse_intermixed_args(argv)
    root = os.path.abspath(a.root or repo_root())
    if not os.path.isdir(root):
        print(f"check_claims: {root} is not a directory", file=sys.stderr)
        return 2
    cite = {os.path.abspath(os.path.join(root, c)) for c in (a.cite_ok or ["docs/RESULTS.md"])}
    values = load_values(a.values)
    if a.cmd == "check":
        rc = cmd_check(root, values, cite)
        if a.release:
            left = [(os.path.relpath(p, root), i) for p in md_files(root)
                    for i, l in enumerate(open(p, encoding="utf-8", errors="replace"), 1)
                    if re.search(r"TODO\(release\)|TBD\(", l)]
            for f, i in left:
                print(f"{f}:{i}: leftover release TODO")
            if left:
                print(f"check_claims: FAIL: {len(left)} leftover release TODO marker(s)")
                rc = 1
        return rc
    if a.cmd == "fill":
        if a.out:
            out = os.path.abspath(a.out)
            if out == root or out.startswith(root + os.sep):
                print("check_claims fill: --out must be outside ROOT", file=sys.stderr)
                return 2
        return cmd_fill(root, values, a.out and os.path.abspath(a.out), a.in_place, cite)
    if a.cmd == "trace":
        if not a.template:
            print("check_claims trace: --template DIR is required", file=sys.stderr)
            return 2
        return cmd_trace(root, os.path.abspath(a.template), values, cite, a.allow_pending)
    return cmd_keys(root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
