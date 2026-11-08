//! Layer 4, no installed future copies. Before the agent starts, `SCAN_PY` runs with the instance's test
//! environment python (`/opt/miniconda3/envs/testbed/bin/python`, Python 3.6+, standard library only) and reports
//! every installation of the target distribution on `sys.path`, whether it is an editable install of the
//! snapshot, where the module actually imports from, and any archive of it on the filesystem (pip and conda
//! caches, wheels, sdists). Archives are deleted before the agent starts (they are not needed at run time) and
//! the scan repeats; any non-editable install, an import outside the repo, or a remaining archive marks the
//! instance INVALID (environment). The audit layer separately flags reads of these paths.
//!
//! `--all-envs` also looks in every other Python environment of the image: each conda environment (the base env
//! and `envs/*` of every conda root, `--env-roots` to name them) and the system's site/dist-packages. Those are
//! not on the test environment's `sys.path`, but the agent can read them: every requests image ships a later
//! requests in the conda base env (conda depends on it; SWE breakdown #27, requests-2931). With `--quarantine`,
//! such a copy (not byte-identical to the repo) is made readable by its owner only (root), so the agent (uid
//! 4242) cannot open it; a copy left readable marks the instance INVALID (environment), a quarantined one is a
//! note.

use serde_json::Value;

/// Repository -> (distribution name, import name).
pub fn target_of(repo: &str) -> (&'static str, &'static str) {
    match repo {
        "django/django" => ("django", "django"),
        "sympy/sympy" => ("sympy", "sympy"),
        "sphinx-doc/sphinx" => ("sphinx", "sphinx"),
        "matplotlib/matplotlib" => ("matplotlib", "matplotlib"),
        "scikit-learn/scikit-learn" => ("scikit-learn", "sklearn"),
        "astropy/astropy" => ("astropy", "astropy"),
        "pydata/xarray" => ("xarray", "xarray"),
        "pytest-dev/pytest" => ("pytest", "_pytest"),
        "pylint-dev/pylint" => ("pylint", "pylint"),
        "psf/requests" => ("requests", "requests"),
        "mwaskom/seaborn" => ("seaborn", "seaborn"),
        "pallets/flask" => ("flask", "flask"),
        _ => ("", ""),
    }
}

/// `python SCAN.py DIST MODULE REPO [--paths p1:p2] [--roots r1:r2] [--clean] [--all-envs [--env-roots r1:r2]
/// [--quarantine]]` -> JSON on stdout.
pub const SCAN_PY: &str = r#"
import json, os, re, sys
args = sys.argv[1:]
dist, module, repo = args[0], args[1], os.path.realpath(args[2])
# value options (--paths, --roots, --env-roots) and switches (--clean, --all-envs, --quarantine), in any order
switches = ("--clean", "--all-envs", "--quarantine")
opt, rest = {}, args[3:]
while rest:
    a = rest.pop(0)
    if a not in switches and rest:
        opt[a] = rest.pop(0)
clean = "--clean" in args[3:]
all_envs = "--all-envs" in args[3:]
quarantine = "--quarantine" in args[3:]
paths = opt.get("--paths", "").split(":") if opt.get("--paths") else [p for p in sys.path if p]
roots = opt.get("--roots", "/").split(":")
norm = lambda s: re.sub(r"[-_.]+", "-", s).lower()
want = norm(dist)
def under(p, base):
    p = os.path.realpath(p); return p == base or p.startswith(base + os.sep)
installs = []
for d in paths:
    if not os.path.isdir(d) or under(d, repo):
        continue
    for name in sorted(os.listdir(d)):
        full = os.path.join(d, name)
        m = re.match(r"^(.+?)-([0-9][^-]*?)(?:-py[0-9.]+)?\.(dist-info|egg-info)$", name)
        if m and norm(m.group(1)) == want:
            editable = False
            du = os.path.join(full, "direct_url.json")
            if os.path.isfile(du):
                try:
                    j = json.load(open(du))
                    editable = bool(j.get("dir_info", {}).get("editable")) and under(j.get("url", "").replace("file://", ""), repo)
                except Exception:
                    pass
            installs.append({"path": full, "version": m.group(2), "editable": editable})
        elif name.endswith(".egg-link") and norm(name[:-9]) == want:
            target = open(full).read().split("\n")[0].strip()
            installs.append({"path": full, "version": None, "editable": under(target, repo), "points_to": target})
        elif name.startswith("__editable__") and want in norm(name):
            installs.append({"path": full, "version": None, "editable": True})
# a dist-info of an editable install sits next to its __editable__/.egg-link/.pth marker
if any(i["editable"] for i in installs):
    for i in installs:
        if not i["editable"] and i["path"].endswith(".dist-info"):
            for marker in os.listdir(os.path.dirname(i["path"])):
                if marker.endswith(".pth") and want.replace("-", "_") in marker.replace("-", "_").lower():
                    try:
                        if repo in open(os.path.join(os.path.dirname(i["path"]), marker)).read():
                            i["editable"] = True
                    except Exception:
                        pass
import_file = None
try:
    sys.path[:0] = [p for p in paths if p not in sys.path]
    mod = __import__(module)
    import_file = os.path.realpath(getattr(mod, "__file__", "") or "")
except Exception as e:
    import_file = "import failed: %s" % e
# an install outside the repo whose code is byte-identical to the repo (the image's own `pip install .` of the
# base commit, e.g. the requests images) holds nothing from the future
def same_as_repo(pkg):
    cands = [os.path.join(repo, x, module) for x in ("", "src", "lib")]
    src = next((c for c in cands if os.path.isdir(c)), None)
    if not src:
        return None
    n = 0; same = True
    for dp, dns, fns in os.walk(pkg):
        for f in fns:
            if not f.endswith(".py"):
                continue
            a = os.path.join(dp, f); b = os.path.join(src, os.path.relpath(a, pkg))
            n += 1
            try:
                if open(a, "rb").read() != open(b, "rb").read():
                    same = False
            except Exception:
                same = False
    return same and n > 0
matches_base = None
if import_file and os.path.isfile(import_file) and not under(import_file, repo):
    pkg = os.path.dirname(import_file) if os.path.basename(import_file) == "__init__.py" else None
    if pkg:
        matches_base = same_as_repo(pkg)
# every other environment: the target installed where the test env's sys.path does not look, but the agent can
other_envs = []
if all_envs:
    import glob
    seen = set(os.path.realpath(p) for p in paths if os.path.isdir(p))
    if opt.get("--env-roots"):
        croots = opt["--env-roots"].split(":"); system = []
    else:
        croots = ["/opt/miniconda3", "/opt/conda", "/opt/anaconda3", "/root/miniconda3", "/usr/local/miniconda3",
                  "/usr/local/conda"]
        pre = os.path.realpath(sys.prefix)
        if os.sep + "envs" + os.sep in pre:
            croots.append(pre.split(os.sep + "envs" + os.sep)[0])
        system = sorted(set(glob.glob("/usr/lib/python3*/site-packages") + glob.glob("/usr/lib/python3/dist-packages")
                            + glob.glob("/usr/local/lib/python3*/site-packages")
                            + glob.glob("/usr/local/lib/python3*/dist-packages")))
    sites = []
    for r in dict.fromkeys(os.path.realpath(c) for c in croots if os.path.isdir(c)):
        for env in [r] + sorted(glob.glob(os.path.join(r, "envs", "*"))):
            sites += [(env, d) for d in sorted(glob.glob(os.path.join(env, "lib", "python*", "site-packages")))]
    sites += [("system", d) for d in system]
    for env, d in sites:
        rd = os.path.realpath(d)
        if rd in seen or under(rd, repo) or not os.path.isdir(rd):
            continue
        seen.add(rd)
        names = sorted(os.listdir(rd))
        meta = [n for n in names
                if (lambda m: m and norm(m.group(1)) == want)(re.match(r"^(.+?)-([0-9][^-]*?)(?:-py[0-9.]+)?\.(dist-info|egg-info)$", n))]
        code = [n for n in (module, module + ".py") if n in names]
        if not meta and not code:
            continue
        version = re.match(r"^.+?-([0-9][^-]*?)(?:-py[0-9.]+)?\.(?:dist-info|egg-info)$", meta[0]).group(1) if meta else None
        pkg = os.path.join(rd, module)
        base = same_as_repo(pkg) if os.path.isdir(pkg) else None
        held = [os.path.join(rd, n) for n in code + meta]
        if quarantine and base is not True:
            for h in held:
                try:
                    os.chmod(h, 0o700 if os.path.isdir(h) else 0o600)
                except Exception:
                    pass
        # readable by the agent: any of the copy's paths open to "others" (the agent's uid owns nothing here)
        readable = any(os.stat(h).st_mode & 0o004 for h in held if os.path.exists(h))
        other_envs.append({"env": env, "site": rd, "version": version, "paths": held, "matches_base": base,
                           "readable": readable})
pat = re.compile(r"^%s[-_][0-9].*\.(whl|tar\.gz|tar\.bz2|zip|conda|egg)$" % re.escape(dist).replace(r"\-", "[-_]"), re.I)
archives = []
for r in roots:
    for dp, dns, fns in os.walk(r):
        if dp.startswith(("/proc", "/sys", "/dev")) or under(dp, repo):
            dns[:] = []
            continue
        for f in fns:
            if pat.match(f):
                archives.append(os.path.join(dp, f))
        for dn in list(dns):
            if pat.match(dn) and ("pkgs" in dp or "cache" in dp):
                archives.append(os.path.join(dp, dn))
removed = []
if clean:
    import shutil
    for a in archives:
        try:
            (shutil.rmtree if os.path.isdir(a) else os.remove)(a); removed.append(a)
        except Exception:
            pass
    archives = [a for a in archives if os.path.exists(a)]
print(json.dumps({"dist": dist, "module": module, "repo": repo, "installs": installs,
                  "import_file": import_file, "matches_base": matches_base, "archives": archives[:50], "removed": removed[:50],
                  "all_envs": all_envs, "other_envs": other_envs}))
"#;

/// Reasons the environment holds a copy of the target other than the snapshot (empty = clean).
pub fn flags(v: &Value) -> Vec<String> {
    let mut f = Vec::new();
    let repo = v["repo"].as_str().unwrap_or("/testbed");
    // the installed copy the module imports is byte-identical to the repo at the base commit: not a future copy
    // (see `notes`); archives elsewhere still count
    let base_copy = v["matches_base"] == true;
    for i in v["installs"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|_| !base_copy)
    {
        if i["editable"] != true {
            f.push(format!(
                "non-editable install of {} {} at {}",
                v["dist"].as_str().unwrap_or("?"),
                i["version"].as_str().unwrap_or("?"),
                i["path"].as_str().unwrap_or("?")
            ));
        }
    }
    match v["import_file"].as_str() {
        Some(p) if p.starts_with(repo) => {}
        Some(_) if base_copy => {}
        Some(p) => f.push(format!(
            "{} imports from {p}, not the repo",
            v["module"].as_str().unwrap_or("?")
        )),
        None => f.push("import location unknown".into()),
    }
    for a in v["archives"].as_array().into_iter().flatten() {
        f.push(format!(
            "archive of the target on disk: {}",
            a.as_str().unwrap_or("?")
        ));
    }
    for o in other_envs(v) {
        if o["matches_base"] != true && o["readable"] != false {
            f.push(format!(
                "{} {} installed in another environment the agent can read: {}",
                v["dist"].as_str().unwrap_or("?"),
                o["version"].as_str().unwrap_or("?"),
                o["site"].as_str().unwrap_or("?")
            ));
        }
    }
    f
}

fn other_envs(v: &Value) -> impl Iterator<Item = &Value> {
    v["other_envs"].as_array().into_iter().flatten()
}

/// Informational findings of the scan.
pub fn notes(v: &Value) -> Vec<String> {
    let mut n = Vec::new();
    if v["matches_base"] == true {
        n.push(format!(
            "{} imports from {}: an install of the base commit (byte-identical to the repo)",
            v["module"].as_str().unwrap_or("?"),
            v["import_file"].as_str().unwrap_or("?")
        ));
    }
    let dist = v["dist"].as_str().unwrap_or("?");
    for o in other_envs(v) {
        let (ver, site) = (
            o["version"].as_str().unwrap_or("?"),
            o["site"].as_str().unwrap_or("?"),
        );
        if o["matches_base"] == true {
            n.push(format!(
                "{dist} {ver} in another environment ({site}) is the base commit (byte-identical to the repo)"
            ));
        } else if o["readable"] == false {
            n.push(format!(
                "{dist} {ver} in another environment ({site}) made unreadable to the agent"
            ));
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn scan(root: &Path, extra: &[&str]) -> Value {
        let script = root.join("scan.py");
        std::fs::write(&script, SCAN_PY).unwrap();
        let repo = root.join("testbed");
        let site = root.join("site");
        let o = std::process::Command::new("python3")
            .arg("-I")
            .arg(&script)
            .args(["demo-lib", "demolib", repo.to_str().unwrap(), "--paths"])
            .arg(format!("{}:{}", site.display(), repo.display()))
            .args(["--roots", root.to_str().unwrap()])
            .args(extra)
            .output()
            .unwrap();
        serde_json::from_slice(&o.stdout)
            .unwrap_or_else(|_| panic!("{}", String::from_utf8_lossy(&o.stderr)))
    }

    /// a conda root under `root/conda` with the test env and a base env holding demo-lib `ver` (code `src`)
    fn conda_base_copy(root: &Path, ver: &str, src: &str) {
        let base = root.join("conda/lib/python3.11/site-packages");
        std::fs::create_dir_all(base.join(format!("demo_lib-{ver}.dist-info"))).unwrap();
        std::fs::create_dir_all(base.join("demolib")).unwrap();
        std::fs::write(base.join("demolib/__init__.py"), src).unwrap();
        std::fs::create_dir_all(root.join("conda/envs/testbed/lib/python3.9/site-packages"))
            .unwrap();
        std::fs::create_dir_all(root.join("conda/envs/other/lib/python3.12/site-packages"))
            .unwrap();
    }

    fn mode(p: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(p).unwrap().permissions().mode() & 0o777
    }

    #[test]
    fn copy_in_another_conda_env_is_flagged_then_quarantined() {
        // SWE breakdown #27 requests-2931: a later requests in the image's conda BASE env, off the testbed path
        let root = setup("envs");
        std::fs::write(
            root.join("site/demo-lib.egg-link"),
            format!("{}\n.\n", root.join("testbed").display()),
        )
        .unwrap();
        conda_base_copy(&root, "2.34.2", "V = 'future'\n");
        let roots = root.join("conda");
        let er = ["--all-envs", "--env-roots", roots.to_str().unwrap()];
        // the old scan (testbed path only) sees nothing
        assert!(flags(&scan(&root, &[])).is_empty());
        let v = scan(&root, &er);
        assert_eq!(v["other_envs"].as_array().unwrap().len(), 1, "{v}");
        let f = flags(&v).join("; ");
        assert!(
            f.contains("demo-lib 2.34.2 installed in another environment the agent can read")
                && f.contains("conda/lib/python3.11/site-packages"),
            "{f}"
        );
        // --quarantine: owner-only, a note instead of a flag
        let q: Vec<&str> = er.iter().copied().chain(["--quarantine"]).collect();
        let v = scan(&root, &q);
        assert!(flags(&v).is_empty(), "{:?}", flags(&v));
        assert!(notes(&v)
            .join("; ")
            .contains("made unreadable to the agent"));
        let base = root.join("conda/lib/python3.11/site-packages");
        assert_eq!(mode(&base.join("demolib")), 0o700);
        assert_eq!(mode(&base.join("demo_lib-2.34.2.dist-info")), 0o700);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn base_commit_copy_in_another_env_is_a_note_and_left_alone() {
        let root = setup("envbase");
        std::fs::write(
            root.join("site/demo-lib.egg-link"),
            format!("{}\n.\n", root.join("testbed").display()),
        )
        .unwrap();
        conda_base_copy(&root, "2.9.0", "V = 'snapshot'\n");
        let roots = root.join("conda");
        let v = scan(
            &root,
            &[
                "--all-envs",
                "--env-roots",
                roots.to_str().unwrap(),
                "--quarantine",
            ],
        );
        assert!(flags(&v).is_empty(), "{:?}", flags(&v));
        assert!(notes(&v).join("; ").contains("is the base commit"));
        let pkg = root.join("conda/lib/python3.11/site-packages/demolib");
        assert_ne!(mode(&pkg) & 0o005, 0, "not quarantined");
        std::fs::remove_dir_all(&root).ok();
    }

    fn setup(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("tb-site-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("testbed/demolib")).unwrap();
        std::fs::write(root.join("testbed/demolib/__init__.py"), "V = 'snapshot'\n").unwrap();
        std::fs::create_dir_all(root.join("site")).unwrap();
        root
    }

    #[test]
    fn editable_snapshot_install_is_clean() {
        let root = setup("ok");
        std::fs::write(
            root.join("site/demo-lib.egg-link"),
            format!("{}\n.\n", root.join("testbed").display()),
        )
        .unwrap();
        let v = scan(&root, &[]);
        assert!(flags(&v).is_empty(), "{v}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn non_editable_install_of_the_base_is_a_note() {
        // bench-swe-q27 requests-2931: the image installs the repo at the base commit non-editably
        let root = setup("base");
        std::fs::create_dir_all(root.join("site/demo_lib-2.9.0.dist-info")).unwrap();
        std::fs::create_dir_all(root.join("site/demolib")).unwrap();
        std::fs::write(root.join("site/demolib/__init__.py"), "V = 'snapshot'\n").unwrap();
        let v = scan(&root, &[]);
        assert_eq!(v["matches_base"], true, "{v}");
        assert!(flags(&v).is_empty(), "{:?}", flags(&v));
        assert_eq!(notes(&v).len(), 1);
        // one byte different from the repo: a future copy again
        std::fs::write(root.join("site/demolib/__init__.py"), "V = 'snapshoT'\n").unwrap();
        let v = scan(&root, &[]);
        assert_eq!(v["matches_base"], false);
        assert!(flags(&v)
            .join("; ")
            .contains("non-editable install of demo-lib 2.9.0"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn future_release_and_cached_archives_are_flagged() {
        let root = setup("bad");
        // a released copy of a later version installed in site-packages, importable first
        std::fs::create_dir_all(root.join("site/demo_lib-9.9.dist-info")).unwrap();
        std::fs::create_dir_all(root.join("site/demolib")).unwrap();
        std::fs::write(root.join("site/demolib/__init__.py"), "V = 'future'\n").unwrap();
        // and a wheel in a pip cache
        std::fs::create_dir_all(root.join("home/.cache/pip/wheels")).unwrap();
        std::fs::write(
            root.join("home/.cache/pip/wheels/demo_lib-9.9-py3-none-any.whl"),
            "x",
        )
        .unwrap();
        let v = scan(&root, &[]);
        let f = flags(&v).join("; ");
        assert!(f.contains("non-editable install of demo-lib 9.9"), "{f}");
        assert!(
            f.contains("imports from") && f.contains("/site/demolib"),
            "{f}"
        );
        assert!(f.contains("demo_lib-9.9-py3-none-any.whl"), "{f}");
        // --clean removes the cached archive; the installed copy still flags
        let v = scan(&root, &["--clean"]);
        assert_eq!(v["removed"].as_array().unwrap().len(), 1);
        let f = flags(&v).join("; ");
        assert!(!f.contains(".whl") && f.contains("non-editable"), "{f}");
        std::fs::remove_dir_all(&root).ok();
    }
}
