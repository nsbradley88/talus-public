#!/usr/bin/env python3
"""collect_licenses.py RUST_DIR OUT_DIR BIN... : the license files of every crate linked into the given binaries.

Walks `cargo metadata` (offline, locked) from the packages that own BIN, following normal dependencies only (build
and dev dependencies are not shipped), on x86_64-unknown-linux-gnu. For each third-party crate it copies
LICENSE* / COPYING* / NOTICE* / UNLICENSE* from the crate's source directory to OUT_DIR/<name>-<version>/ and writes
OUT_DIR/INDEX.tsv (name, version, SPDX license expression, repository, files copied). Workspace crates are the
project's own (MIT, see LICENSE) and are listed without copies. Exits 1 if a third-party crate has no license file
(MIT / Apache-2.0 / BSD require the text to travel with binaries).
"""
import json, os, shutil, subprocess, sys

def main():
    rust, out, bins = sys.argv[1], sys.argv[2], set(sys.argv[3:])
    meta = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--locked", "--offline",
         "--filter-platform", "x86_64-unknown-linux-gnu"], cwd=rust))
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    ws = set(meta["workspace_members"])
    roots = [p["id"] for p in meta["packages"] if p["id"] in ws
             and any(t["name"] in bins and "bin" in t["kind"] for t in p["targets"])]
    missing = bins - {t["name"] for r in roots for t in pkgs[r]["targets"]}
    if missing:
        sys.exit(f"collect_licenses: no package builds {sorted(missing)}")
    seen, todo = set(), list(roots)
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        for d in nodes[i]["deps"]:
            if any(k["kind"] is None for k in d["dep_kinds"]):  # normal dependency
                todo.append(d["pkg"])
    os.makedirs(out, exist_ok=True)
    rows, bad = [], []
    for i in sorted(seen, key=lambda x: (pkgs[x]["name"], pkgs[x]["version"])):
        p = pkgs[i]
        if i in ws:
            rows.append((p["name"], p["version"], p.get("license") or "", "(this project)", "-"))
            continue
        src = os.path.dirname(p["manifest_path"])
        files = sorted(f for f in os.listdir(src)
                       if f.upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE", "UNLICENSE", "COPYRIGHT")))
        if p.get("license_file"):
            lf = os.path.normpath(os.path.join(src, p["license_file"]))
            if os.path.isfile(lf) and os.path.basename(lf) not in files:
                files.append(os.path.relpath(lf, src))
        dst = os.path.join(out, f'{p["name"]}-{p["version"]}')
        os.makedirs(dst, exist_ok=True)
        for f in files:
            s = os.path.join(src, f)
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(dst, os.path.basename(f)), dirs_exist_ok=True)
            else:
                shutil.copyfile(s, os.path.join(dst, os.path.basename(f)))
        if not files:
            bad.append(f'{p["name"]}-{p["version"]} ({p.get("license")})')
        rows.append((p["name"], p["version"], p.get("license") or p.get("license_file") or "?",
                     p.get("repository") or "", ",".join(os.path.basename(f) for f in files) or "NONE"))
    with open(os.path.join(out, "INDEX.tsv"), "w") as f:
        f.write("crate\tversion\tlicense\trepository\tfiles\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"collect_licenses: {len(rows)} crates ({len(rows) - sum(1 for i in seen if i in ws)} third-party)")
    if bad:
        sys.exit("collect_licenses: no license file in: " + ", ".join(bad))

if __name__ == "__main__":
    main()
