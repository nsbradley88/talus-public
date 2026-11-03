//! Layer 2, no future history. The official images clone the whole upstream repository into /testbed, so the
//! agent container first turns /testbed into a fresh single-commit snapshot of `base_commit` and then verifies,
//! before the agent starts, that nothing else is reachable: one commit, one branch, no remotes, tags, reflogs,
//! packed refs, alternates, shallow file or unreachable objects, and no other git directories or pack files
//! anywhere on the container's filesystem. The scripts run as root inside the container (bash + git only);
//! a failed verification aborts the instance (INVALID, environment).

use serde_json::Value;

/// `snapshot.sh REPO BASE_COMMIT`
pub const SNAPSHOT_SH: &str = r#"#!/bin/bash
set -euo pipefail
REPO="$1"; BASE="$2"
cd "$REPO"
git config --global --add safe.directory "$REPO" 2>/dev/null || true
git -c advice.detachedHead=false checkout -q -f "$BASE"
BASE_TREE=$(git rev-parse "$BASE^{tree}")
# drop every trace of history; keep the working tree (built extensions and egg-info the env needs)
rm -rf .git
git init -q
git symbolic-ref HEAD refs/heads/main
git config core.logAllRefUpdates false
git config gc.reflogExpire now
# no auto-gc: on a big tree (django: ~9000 loose objects > gc.auto) a background gc packs refs/heads/main
# into packed-refs while this script runs
git config gc.auto 0
git config gc.autoDetach false
git config user.name snapshot
git config user.email snapshot@localhost
git add -A
GIT_AUTHOR_DATE="2000-01-01T00:00:00Z" GIT_COMMITTER_DATE="2000-01-01T00:00:00Z" \
  git -c core.hooksPath=/dev/null commit -q --no-verify -m "snapshot of the base commit"
# the repo is fresh (no upstream refs); packed-refs, if any, holds main itself: unpack it instead of deleting it
if [ -e .git/packed-refs ]; then
  H=$(git rev-parse HEAD); rm -f .git/packed-refs; git update-ref refs/heads/main "$H"
fi
rm -rf .git/logs .git/ORIG_HEAD .git/FETCH_HEAD .git/hooks .git/gc.log .git/gc.log.lock .git/gc.pid
mkdir -p .git/hooks
echo "$BASE_TREE" > /run/talus-snapshot-base-tree 2>/dev/null || true
"#;

/// `verify.sh REPO [SEARCH_ROOTS...]` -> one JSON object on stdout.
pub const VERIFY_SH: &str = r#"#!/bin/bash
set -uo pipefail
REPO="$1"; shift
ROOTS=("$@"); [ ${#ROOTS[@]} -eq 0 ] && ROOTS=(/)
cd "$REPO" || { echo '{"ok": false, "error": "no repo"}'; exit 0; }
G=$(cd "$REPO" && git rev-parse --absolute-git-dir 2>/dev/null)
commits=$(git rev-list --all 2>/dev/null | wc -l)
refs=$(git for-each-ref --format='%(refname)' | tr '\n' ' ')
remotes=$(git remote | tr '\n' ' ')
tags=$(git tag | wc -l)
reflog=$(git reflog 2>/dev/null | wc -l)
logs=$( [ -d "$G/logs" ] && find "$G/logs" -type f | wc -l || echo 0)
packed=$( [ -e "$G/packed-refs" ] && echo true || echo false)
alternates=$( [ -s "$G/objects/info/alternates" ] && echo true || echo false)
shallow=$( [ -e "$G/shallow" ] && echo true || echo false)
special=$(ls "$G" | grep -E '^(ORIG_HEAD|FETCH_HEAD|MERGE_HEAD|CHERRY_PICK_HEAD)$' | tr '\n' ' ')
unreachable=$(git fsck --unreachable --no-reflogs --no-progress 2>/dev/null | grep -c . || true)
objects=$(git count-objects -v | awk '/^count:/{c=$2} /^in-pack:/{p=$2} END {print c+p}')
reachable=$(git rev-list --objects --all | wc -l)
others=$(for r in "${ROOTS[@]}"; do find "$r" -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o \
  \( -name .git -o -name '*.pack' -o -name '*.bundle' -o -name ORIG_HEAD -o -name packed-refs \) -print 2>/dev/null; done \
  | grep -v -F "$G" | while IFS= read -r f; do
      # a `.git` FILE is a gitlink ("gitdir: PATH"): history only if PATH exists (conda ships dangling
      # submodule links in package test files, e.g. referencing's suite/.git -> ../.git/modules/suite)
      if [ -f "$f" ] && [ "${f##*/}" = .git ]; then
        t=$(sed -n 's/^gitdir: *//p' "$f" | head -1)
        [ -n "$t" ] || continue
        case "$t" in /*) ;; *) t="${f%/*}/$t" ;; esac
        [ -e "$t" ] || continue
      fi
      printf '%s\n' "$f"
    done | head -50 | sed 's/"/\\"/g' | awk '{printf "%s\"%s\"", (NR>1?",":""), $0}')
ok=true
[ "$commits" -eq 1 ] || ok=false
[ "$(echo $refs)" = "refs/heads/main" ] || ok=false
[ -z "$(echo $remotes)" ] || ok=false
[ "$tags" -eq 0 ] || ok=false
[ "$reflog" -eq 0 ] || ok=false
[ "$logs" -eq 0 ] || ok=false
[ "$packed" = false ] || ok=false
[ "$alternates" = false ] || ok=false
[ "$shallow" = false ] || ok=false
[ -z "$(echo $special)" ] || ok=false
[ "$unreachable" -eq 0 ] || ok=false
[ "$objects" -eq "$reachable" ] || ok=false
[ -z "$others" ] || ok=false
printf '{"ok": %s, "commits": %s, "refs": "%s", "remotes": "%s", "tags": %s, "reflog": %s, "log_files": %s, "packed_refs": %s, "alternates": %s, "shallow": %s, "special_heads": "%s", "unreachable": %s, "objects": %s, "reachable_objects": %s, "other_git_files": [%s]}\n' \
  "$ok" "$commits" "$(echo $refs)" "$(echo $remotes)" "$tags" "$reflog" "$logs" "$packed" "$alternates" "$shallow" \
  "$(echo $special)" "$unreachable" "$objects" "$reachable" "$others"
"#;

/// Reasons a verification report fails (empty = clean).
pub fn problems(v: &Value) -> Vec<String> {
    let mut p = Vec::new();
    if v["ok"] != true {
        let n = |k: &str| v[k].as_i64().unwrap_or(-1);
        if n("commits") != 1 {
            p.push(format!("{} reachable commits (want 1)", n("commits")));
        }
        if v["refs"] != "refs/heads/main" {
            p.push(format!("refs: {}", v["refs"]));
        }
        if v["remotes"].as_str().is_some_and(|s| !s.is_empty()) {
            p.push(format!("remotes: {}", v["remotes"]));
        }
        for k in ["tags", "reflog", "log_files", "unreachable"] {
            if n(k) > 0 {
                p.push(format!("{k}: {}", n(k)));
            }
        }
        for k in ["packed_refs", "alternates", "shallow"] {
            if v[k] == true {
                p.push(k.to_string());
            }
        }
        if v["special_heads"].as_str().is_some_and(|s| !s.is_empty()) {
            p.push(format!("special heads: {}", v["special_heads"]));
        }
        if n("objects") != n("reachable_objects") {
            p.push(format!(
                "{} objects stored, {} reachable",
                n("objects"),
                n("reachable_objects")
            ));
        }
        if let Some(o) = v["other_git_files"].as_array().filter(|a| !a.is_empty()) {
            p.push(format!(
                "other git data on the filesystem: {}",
                o.iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if v["error"].is_string() {
            p.push(v["error"].as_str().unwrap_or_default().to_string());
        }
        if p.is_empty() {
            p.push("verification failed".into());
        }
    }
    p
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn sh(dir: &Path, cmd: &str) -> String {
        let o = Command::new("bash")
            .arg("-c")
            .arg(cmd)
            .current_dir(dir)
            .env("GIT_CONFIG_GLOBAL", dir.join("../gitconfig"))
            .output()
            .unwrap();
        assert!(
            o.status.success(),
            "{cmd}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        String::from_utf8_lossy(&o.stdout).trim().to_string()
    }

    /// upstream with a future fix + a tag, cloned (remote, packed refs, reflog): the image's /testbed situation.
    pub fn fixture(name: &str) -> (PathBuf, String, String) {
        let root = std::env::temp_dir().join(format!("tb-snap-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("up")).unwrap();
        std::fs::write(
            root.join("gitconfig"),
            "[user]\n\tname = t\n\temail = t@x\n[init]\n\tdefaultBranch = main\n",
        )
        .unwrap();
        let up = root.join("up");
        sh(
            &up,
            "git init -q && echo 'x = 1' > m.py && git add . && git commit -qm base",
        );
        let base = sh(&up, "git rev-parse HEAD");
        sh(&up, "echo 'x = 2  # FUTURE_FIX' > m.py && git commit -qam 'the fix' && git tag v9 && echo y > n.py && git add . && git commit -qm later");
        let future = sh(&up, "git rev-parse HEAD~1");
        sh(&root, "git clone -q up testbed && cd testbed && git pack-refs --all && git checkout -q -b work && git checkout -q main");
        (root, base, future)
    }

    fn run_script(script: &str, args: &[&str], cwd: &Path) -> String {
        let p = cwd.join("../script.sh");
        std::fs::write(&p, script).unwrap();
        let o = Command::new("bash")
            .arg(&p)
            .args(args)
            .current_dir(cwd)
            .env("GIT_CONFIG_GLOBAL", cwd.join("../gitconfig"))
            .output()
            .unwrap();
        String::from_utf8_lossy(&o.stdout).to_string() + &String::from_utf8_lossy(&o.stderr)
    }

    #[test]
    fn snapshot_survives_auto_gc_on_a_big_tree() {
        // trial finding: django's ~9000 loose objects tripped auto-gc, which packed refs/heads/main and the
        // old script then deleted packed-refs (no branch left). Force the condition with gc.auto=1, synchronous.
        let (root, base, _) = fixture("gc");
        std::fs::write(
            root.join("gitconfig"),
            "[user]\n\tname = t\n\temail = t@x\n[init]\n\tdefaultBranch = main\n[gc]\n\tauto = 1\n\tautoDetach = false\n",
        )
        .unwrap();
        let repo = root.join("testbed");
        for i in 0..3000 {
            std::fs::write(repo.join(format!("f{i}.txt")), format!("{i}\n")).unwrap();
        }
        // pretend the files are part of the base (untracked files are kept by the snapshot, like built artifacts)
        let out = run_script(SNAPSHOT_SH, &[repo.to_str().unwrap(), &base], &repo);
        assert!(!out.contains("fatal"), "{out}");
        let v: Value = serde_json::from_str(
            run_script(
                VERIFY_SH,
                &[repo.to_str().unwrap(), repo.to_str().unwrap()],
                &repo,
            )
            .trim(),
        )
        .unwrap();
        assert!(problems(&v).is_empty(), "{v}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn snapshot_leaves_no_future_history_and_a_cheater_finds_nothing() {
        let (root, base, future) = fixture("ok");
        let repo = root.join("testbed");
        let out = run_script(SNAPSHOT_SH, &[repo.to_str().unwrap(), &base], &repo);
        assert!(!out.contains("fatal"), "{out}");
        let v: Value = serde_json::from_str(
            run_script(
                VERIFY_SH,
                &[repo.to_str().unwrap(), repo.to_str().unwrap()],
                &repo,
            )
            .trim(),
        )
        .unwrap();
        assert!(problems(&v).is_empty(), "{v}");
        assert_eq!(
            std::fs::read_to_string(repo.join("m.py")).unwrap(),
            "x = 1\n",
            "working tree is the base commit"
        );
        // the cheater's moves
        let tryit = |c: &str| {
            let o = Command::new("bash")
                .arg("-c")
                .arg(c)
                .current_dir(&repo)
                .env("GIT_CONFIG_GLOBAL", root.join("gitconfig"))
                .output()
                .unwrap();
            (
                o.status.success(),
                String::from_utf8_lossy(&o.stdout).to_string(),
            )
        };
        assert_eq!(tryit("git log --all --oneline | wc -l").1.trim(), "1");
        assert!(
            !tryit(&format!("git cat-file -e {future}")).0,
            "future commit object unreachable and absent"
        );
        assert!(
            !tryit(&format!("git show {base}")).0,
            "even the upstream base sha is gone (fresh snapshot)"
        );
        assert_eq!(tryit("git reflog | wc -l").1.trim(), "0");
        assert_eq!(tryit("git tag | wc -l").1.trim(), "0");
        assert_eq!(tryit("git remote | wc -l").1.trim(), "0");
        assert!(!tryit("git fetch origin 2>&1").0);
        assert_eq!(
            tryit("git grep -l FUTURE_FIX $(git rev-list --all) | wc -l")
                .1
                .trim(),
            "0"
        );
        assert_eq!(
            tryit("git fsck --lost-found --no-progress 2>/dev/null | wc -l")
                .1
                .trim(),
            "0"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn dangling_gitlink_is_not_history_but_a_live_one_is() {
        // trial finding (matplotlib-24026): conda's referencing package ships suite/.git = "gitdir:
        // ../.git/modules/suite" with no target
        let (root, base, _) = fixture("gitlink");
        let repo = root.join("testbed");
        run_script(SNAPSHOT_SH, &[repo.to_str().unwrap(), &base], &repo);
        let env = root.join("env/test-files/referencing/1/suite");
        std::fs::create_dir_all(&env).unwrap();
        std::fs::write(env.join(".git"), "gitdir: ../.git/modules/suite\n").unwrap();
        let verify = || -> Value {
            serde_json::from_str(
                run_script(
                    VERIFY_SH,
                    &[repo.to_str().unwrap(), root.join("env").to_str().unwrap()],
                    &repo,
                )
                .trim(),
            )
            .unwrap()
        };
        let v = verify();
        assert!(problems(&v).is_empty(), "{v}");
        // the link's target exists: real git data reachable through it
        std::fs::create_dir_all(root.join("env/test-files/referencing/1/.git/modules/suite"))
            .unwrap();
        let p = problems(&verify()).join("; ");
        assert!(p.contains("other git data"), "{p}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verification_catches_leftovers() {
        let (root, base, _) = fixture("leak");
        let repo = root.join("testbed");
        // verify the untouched clone: history, remote, tags, packed refs all present
        let v: Value = serde_json::from_str(
            run_script(
                VERIFY_SH,
                &[repo.to_str().unwrap(), repo.to_str().unwrap()],
                &repo,
            )
            .trim(),
        )
        .unwrap();
        let p = problems(&v).join("; ");
        for want in ["reachable commits", "remotes", "packed_refs", "tags"] {
            assert!(p.contains(want), "{want} not in {p}");
        }
        // snapshot, then plant: an unreachable object, a stray bundle elsewhere, a remote
        run_script(SNAPSHOT_SH, &[repo.to_str().unwrap(), &base], &repo);
        sh(&repo, "echo secret | git hash-object -w --stdin >/dev/null");
        std::fs::write(root.join("upstream.bundle"), "x").unwrap();
        let v: Value = serde_json::from_str(
            run_script(
                VERIFY_SH,
                &[repo.to_str().unwrap(), root.to_str().unwrap()],
                &repo,
            )
            .trim(),
        )
        .unwrap();
        let p = problems(&v).join("; ");
        assert!(
            p.contains("unreachable") || p.contains("objects stored"),
            "{p}"
        );
        assert!(p.contains("upstream.bundle"), "{p}");
        assert!(p.contains("other git data"), "{p}");
        sh(&repo, "git remote add origin https://example.invalid/x.git");
        let v: Value = serde_json::from_str(
            run_script(
                VERIFY_SH,
                &[repo.to_str().unwrap(), repo.to_str().unwrap()],
                &repo,
            )
            .trim(),
        )
        .unwrap();
        assert!(problems(&v).join(";").contains("remotes"));
        std::fs::remove_dir_all(&root).ok();
    }
}
