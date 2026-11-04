#!/usr/bin/env python3
"""check-hf-pins.py [--ranged] : release check, needs network. Every file in talus-setup's manifest must have the
manifest's size and SHA-256 (the LFS object id) on Hugging Face at the repository's pinned commit (HF_REVISIONS).
--ranged also downloads the first MiB of each repository's smallest file through /resolve/<commit>/ (and of the
Flash-Next checkpoint tools/mtp_fetch.py reads). HF_ENDPOINT names a mirror. Exit status 0 = all match."""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "rust/crates/talus-setup/src/manifest.rs"
EP = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").strip().rstrip("/")


def opener(url, headers=None):
    h = {"User-Agent": "talus-check-hf-pins"}
    h.update(headers or {})
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        h["Authorization"] = "Bearer " + tok.strip()
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120)


def parse(text):
    text = re.sub(r"(?m)^\s*//.*$", "", text)
    pins = dict(re.findall(r'\(\s*"([^"]+)",\s*"([0-9a-f]{40})",?\s*\)', text))
    vocab = re.search(r'const DRAFT_VOCAB: &str = "([^"]+)"', text).group(1)
    files = []
    for m in re.finditer(r"FileSpec \{(.*?)\}", text, re.S):
        b = m.group(1)
        repo = re.search(r'repo:\s*(?:"([^"]+)"|DRAFT_VOCAB)', b)
        path, size, sha = (re.search(p, b) for p in (r'path:\s*"([^"]+)"', r"size:\s*([\d_]+)", r'sha256:\s*"([0-9a-f]{64})"'))
        if repo and path and size and sha:
            files.append((repo.group(1) or vocab, path.group(1), int(size.group(1).replace("_", "")), sha.group(1)))
    return pins, files


def main():
    pins, files = parse(SRC.read_text())
    bad = 0
    trees = {}
    for repo, path, size, sha in files:
        rev = pins.get(repo)
        if not rev:
            print(f"FAIL {repo}: no pinned commit")
            bad += 1
            continue
        d = path.rsplit("/", 1)[0] if "/" in path else ""
        key = (repo, d)
        if key not in trees:
            trees[key] = json.load(opener(f"{EP}/api/models/{repo}/tree/{rev}" + (f"/{d}" if d else "")))
        e = next((x for x in trees[key] if x.get("path") == path), None)
        oid = (e or {}).get("lfs", {}).get("oid")
        ok = e is not None and e.get("size") == size and oid == sha
        print(("ok  " if ok else "FAIL"), f"{repo}@{rev[:10]} {path} size {(e or {}).get('size')} sha {str(oid)[:12]}")
        bad += not ok
    if "--ranged" in sys.argv[1:]:
        smallest = {}
        for f in files:
            if f[0] not in smallest or f[2] < smallest[f[0]][2]:
                smallest[f[0]] = f
        targets = [(r, pins[r], f[1], min(f[2], 1 << 20)) for r, f in sorted(smallest.items())]
        mt = re.search(r'PINNED_REVISION = "([0-9a-f]{40})"', (SRC.parents[4] / "tools/mtp_fetch.py").read_text())
        targets.append(("Qwen/Qwen3.8-Flash-Next", mt.group(1), "config.json", None))
        for repo, rev, path, want in targets:
            n = len(opener(f"{EP}/{repo}/resolve/{rev}/{path}", {"Range": "bytes=0-1048575"}).read())
            ok = n > 0 if want is None else n == want
            print(("ok  " if ok else "FAIL"), f"ranged {repo}@{rev[:10]} {path} {n} B")
            bad += not ok
    print(f"{len(files)} files, {len(pins)} pinned repositories, {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
