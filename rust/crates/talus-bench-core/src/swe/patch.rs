//! Model patches: which files they touch, whether they touch tests (rejected: the hidden tests decide), and how
//! close they are to the gold patch (memorization is measured, not blocked).

use std::collections::BTreeSet;

/// Paths a unified diff touches (both sides of renames; /dev/null ignored).
pub fn touched_files(diff: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for l in diff.lines() {
        if let Some(rest) = l.strip_prefix("diff --git ") {
            for part in rest.split_whitespace() {
                let p = part.trim_start_matches("a/").trim_start_matches("b/");
                if !p.is_empty() {
                    out.insert(p.to_string());
                }
            }
        } else if let Some(p) = l.strip_prefix("+++ ").or_else(|| l.strip_prefix("--- ")) {
            let p = p.split('\t').next().unwrap_or("").trim();
            if p != "/dev/null" && !p.is_empty() {
                out.insert(
                    p.trim_start_matches("a/")
                        .trim_start_matches("b/")
                        .to_string(),
                );
            }
        }
    }
    out
}

/// A test file by path: a `test`/`tests`/`testing`/`e2e` directory component, `test_*.py`, `*_test.py`,
/// `*_tests.py`, `conftest.py`, or `tests.py`.
pub fn is_test_path(p: &str) -> bool {
    let parts: Vec<&str> = p.split('/').collect();
    let name = parts.last().copied().unwrap_or("");
    parts[..parts.len().saturating_sub(1)]
        .iter()
        .any(|d| ["test", "tests", "testing", "e2e"].contains(d))
        || (name.starts_with("test_") && name.ends_with(".py"))
        || name.ends_with("_test.py")
        || name.ends_with("_tests.py")
        || name == "conftest.py"
        || name == "tests.py"
}

/// Files of the model patch that are tests, or that the hidden test patch touches.
pub fn test_edits(model_patch: &str, test_patch: &str) -> Vec<String> {
    let hidden = touched_files(test_patch);
    touched_files(model_patch)
        .into_iter()
        .filter(|f| is_test_path(f) || hidden.contains(f))
        .collect()
}

/// Changed lines of a diff, normalized: `+`/`-` marker, whitespace collapsed, blank and diff-header lines dropped,
/// prefixed by the file so the same edit in another file does not match.
pub fn normalized_changes(diff: &str) -> Vec<String> {
    let mut file = String::new();
    let mut out = Vec::new();
    for l in diff.lines() {
        if let Some(rest) = l.strip_prefix("+++ ") {
            file = rest.trim().trim_start_matches("b/").to_string();
            continue;
        }
        if l.starts_with("--- ")
            || l.starts_with("diff --git")
            || l.starts_with("@@")
            || l.starts_with("index ")
        {
            continue;
        }
        let (sign, body) = match l.chars().next() {
            Some('+') => ('+', &l[1..]),
            Some('-') => ('-', &l[1..]),
            _ => continue,
        };
        let norm = body.split_whitespace().collect::<Vec<_>>().join(" ");
        if !norm.is_empty() {
            out.push(format!("{file}{sign}{norm}"));
        }
    }
    out
}

fn lcs(a: &[String], b: &[String]) -> usize {
    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    let mut prev = vec![0usize; b.len() + 1];
    let mut cur = vec![0usize; b.len() + 1];
    for x in a {
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = if x == y {
                prev[j] + 1
            } else {
                prev[j + 1].max(cur[j])
            };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Similarity {
    /// 2 * LCS / (|model| + |gold|) over normalized changed lines (difflib-style ratio), 0..1.
    pub ratio: f64,
    /// Every normalized changed line identical, in order.
    pub exact: bool,
    pub model_lines: usize,
    pub gold_lines: usize,
}

/// Near-verbatim threshold for the report's memorization flag.
pub const NEAR_VERBATIM: f64 = 0.9;

pub fn similarity(model: &str, gold: &str) -> Similarity {
    let cap = |mut v: Vec<String>| {
        v.truncate(4000);
        v
    };
    let a = cap(normalized_changes(model));
    let b = cap(normalized_changes(gold));
    let ratio = if a.is_empty() && b.is_empty() {
        1.0
    } else {
        2.0 * lcs(&a, &b) as f64 / (a.len() + b.len()) as f64
    };
    Similarity {
        ratio,
        exact: !a.is_empty() && a == b,
        model_lines: a.len(),
        gold_lines: b.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLD: &str = "diff --git a/src/m.py b/src/m.py\n--- a/src/m.py\n+++ b/src/m.py\n@@ -1,3 +1,3 @@\n def f(x):\n-    return x+1\n+    return x + 2\n";

    #[test]
    fn test_file_rules() {
        for p in [
            "tests/test_a.py",
            "pkg/tests/x.py",
            "pkg/test_b.py",
            "pkg/b_test.py",
            "conftest.py",
            "django/tests.py",
            "testing/x.py",
        ] {
            assert!(is_test_path(p), "{p}");
        }
        for p in [
            "src/m.py",
            "pkg/testutils.py",
            "docs/testing.rst",
            "latest/x.py",
        ] {
            assert!(!is_test_path(p), "{p}");
        }
        let model = "diff --git a/src/m.py b/src/m.py\n--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1 @@\n-a\n+b\ndiff --git a/lib/check.py b/lib/check.py\n--- a/lib/check.py\n+++ b/lib/check.py\n@@ -1 +1 @@\n-a\n+b\n";
        let hidden =
            "diff --git a/lib/check.py b/lib/check.py\n--- a/lib/check.py\n+++ b/lib/check.py\n";
        assert_eq!(
            test_edits(model, hidden),
            vec!["lib/check.py"],
            "a file the hidden tests touch counts as a test edit"
        );
        assert!(test_edits(GOLD, hidden).is_empty());
    }

    #[test]
    fn similarity_flags_near_verbatim() {
        let same_spacing_changed = GOLD
            .replace("x + 2", "x  +  2")
            .replace("@@ -1,3 +1,3 @@", "@@ -10,3 +10,3 @@");
        let s = similarity(&same_spacing_changed, GOLD);
        assert!(s.exact && s.ratio == 1.0, "{s:?}");
        let different = "diff --git a/src/m.py b/src/m.py\n--- a/src/m.py\n+++ b/src/m.py\n@@ -1,3 +1,4 @@\n def f(x):\n-    return x+1\n+    y = x * 1\n+    return y + 2\n";
        let d = similarity(different, GOLD);
        assert!(d.ratio < NEAR_VERBATIM && !d.exact, "{d:?}");
        assert!(similarity("", GOLD).ratio == 0.0);
    }
}
