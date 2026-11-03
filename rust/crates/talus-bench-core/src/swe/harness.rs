//! The official SWE-bench evaluation rules, ported from the `swebench` package 5.0.2 (MIT): the Python log
//! parsers (`harness/log_parsers/python.py`), log extraction and grading (`harness/grading.py`:
//! `get_logs_eval`, `get_eval_tests_report`, `get_resolution_status`, truncated-id matching), the eval-script
//! exit-code hook (`harness/utils.py`: `parse_eval_script`, `record_test_exit_code`) and the patch-apply chain
//! (`run_evaluation.py`: `GIT_APPLY_CMDS`). The eval script itself is the instance's official `eval_script`.
//! Only the advisory infra-failure classifier is not ported (it never changes `resolved`).

use regex::Regex;
use std::collections::BTreeMap;

pub const APPLY_PATCH_FAIL: &str = ">>>>> Patch Apply Failed";
pub const APPLY_PATCH_PASS: &str = ">>>>> Applied Patch";
pub const RESET_FAILED: &str = ">>>>> Reset Failed";
pub const TESTS_ERROR: &str = ">>>>> Tests Errored";
pub const TESTS_TIMEOUT: &str = ">>>>> Tests Timed Out";
pub const START_TEST_OUTPUT: &str = ">>>>> Start Test Output";
pub const END_TEST_OUTPUT: &str = ">>>>> End Test Output";
pub const TEST_EXIT_CODE: &str = ">>>>> Test Exit Code";
pub const TEST_EXIT_CODE_VAR: &str = "SWEBENCH_TEST_EXIT_CODE";
pub const CONTAINER_PATCH_FILE: &str = "/tmp/patch.diff";
pub const GIT_APPLY_CMDS: [&str; 4] = [
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
    "patch --batch --forward --fuzz=5 -p1 -i",
];

const STATUSES: [&str; 5] = ["FAILED", "PASSED", "SKIPPED", "ERROR", "XFAIL"];

pub type StatusMap = BTreeMap<String, String>;

/// `parse_eval_script` + `record_test_exit_code`.
pub fn prepare_eval_script(eval_script: &str) -> String {
    let lines: Vec<&str> = eval_script
        .trim()
        .split('\n')
        .filter(|l| *l != "#!/bin/bash" && *l != "set -uxo pipefail")
        .collect();
    let mut out: Vec<String> = Vec::new();
    let mut done = false;
    for l in lines {
        if !done && l.contains(END_TEST_OUTPUT) {
            out.push(format!("{TEST_EXIT_CODE_VAR}=$?"));
            out.push(l.to_string());
            out.push(format!("echo \"{TEST_EXIT_CODE}: ${TEST_EXIT_CODE_VAR}\""));
            done = true;
        } else {
            out.push(l.to_string());
        }
    }
    // the harness writes the list back as a script with the same header
    format!("#!/bin/bash\nset -uxo pipefail\n{}\n", out.join("\n"))
}

fn starts_with_status(line: &str) -> bool {
    STATUSES.iter().any(|s| line.starts_with(s))
}

fn is_skip_summary(status: &str, name: &str) -> bool {
    status == "SKIPPED" && Regex::new(r"^\[\d+\]$").unwrap().is_match(name)
}

pub fn parse_log_pytest(log: &str) -> StatusMap {
    let mut m = StatusMap::new();
    for line in log.split('\n') {
        if starts_with_status(line) {
            let line = if line.starts_with("FAILED") {
                line.replace(" - ", " ")
            } else {
                line.to_string()
            };
            let tc: Vec<&str> = line.split_whitespace().collect();
            if tc.len() <= 1 || is_skip_summary(tc[0], tc[1]) {
                continue;
            }
            m.insert(tc[1].to_string(), tc[0].to_string());
        }
    }
    m
}

pub fn parse_log_pytest_options(log: &str) -> StatusMap {
    let opt = Regex::new(r"(.*?)\[(.*)\]").unwrap();
    let mut m = StatusMap::new();
    for line in log.split('\n') {
        if starts_with_status(line) {
            let line = if line.starts_with("FAILED") {
                line.replace(" - ", " ")
            } else {
                line.to_string()
            };
            let tc: Vec<&str> = line.split_whitespace().collect();
            if tc.len() <= 1 || is_skip_summary(tc[0], tc[1]) {
                continue;
            }
            let name = match opt.captures(tc[1]) {
                Some(c) => {
                    let main = c.get(1).unwrap().as_str();
                    let mut option = c.get(2).unwrap().as_str().to_string();
                    if option.starts_with('/') && !option.starts_with("//") && !option.contains('*')
                    {
                        option = format!("/{}", option.rsplit('/').next().unwrap_or(""));
                    }
                    format!("{main}[{option}]")
                }
                None => tc[1].to_string(),
            };
            m.insert(name, tc[0].to_string());
        }
    }
    m
}

pub fn parse_log_django(log: &str) -> StatusMap {
    let mut m = StatusMap::new();
    let mut prev_test: Option<String> = None;
    for raw in log.split('\n') {
        let mut line = raw.trim().to_string();
        if line.contains("--version is equivalent to version") {
            m.insert("--version is equivalent to version".into(), "PASSED".into());
        }
        if line.contains(" ... ") {
            prev_test = Some(line.split(" ... ").next().unwrap_or("").to_string());
        }
        for suffix in [" ... ok", " ... OK", " ...  OK"] {
            if line.ends_with(suffix) {
                if line
                    .trim()
                    .starts_with("Applying sites.0002_alter_domain_unique...test_no_migrations")
                {
                    line = line
                        .split_once("...")
                        .map(|x| x.1)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                }
                let test = line
                    .rsplit_once(suffix)
                    .map(|x| x.0)
                    .unwrap_or(&line)
                    .to_string();
                m.insert(test, "PASSED".into());
                break;
            }
        }
        if line.contains(" ... skipped") {
            m.insert(
                line.split(" ... skipped").next().unwrap_or("").to_string(),
                "SKIPPED".into(),
            );
        }
        if line.ends_with(" ... FAIL") {
            m.insert(
                line.split(" ... FAIL").next().unwrap_or("").to_string(),
                "FAILED".into(),
            );
        }
        if line.starts_with("FAIL:") {
            if let Some(t) = line.split_whitespace().nth(1) {
                m.insert(t.trim().to_string(), "FAILED".into());
            }
        }
        if line.ends_with(" ... ERROR") {
            m.insert(
                line.split(" ... ERROR").next().unwrap_or("").to_string(),
                "ERROR".into(),
            );
        }
        if line.starts_with("ERROR:") {
            if let Some(t) = line.split_whitespace().nth(1) {
                m.insert(t.trim().to_string(), "ERROR".into());
            }
        }
        if line.trim_start().starts_with("ok") {
            if let Some(p) = &prev_test {
                m.insert(p.clone(), "PASSED".into());
            }
        }
    }
    let pats = [
        r"(?m)^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
        r"(?m)^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ /(.*)/\nok$",
        r"(?m)^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$",
    ];
    for p in pats {
        for c in Regex::new(p).unwrap().captures_iter(log) {
            m.insert(c[1].to_string(), "PASSED".into());
        }
    }
    m
}

pub fn parse_log_pytest_v2(log: &str) -> StatusMap {
    let color = Regex::new(r"\[(\d+)m").unwrap();
    let mut m = StatusMap::new();
    for raw in log.split('\n') {
        let line: String = color
            .replace_all(raw, "")
            .chars()
            .filter(|c| !(1..32).contains(&(*c as u32)))
            .collect();
        if starts_with_status(&line) {
            let line = if line.starts_with("FAILED") {
                line.split(" - ").next().unwrap_or("").to_string()
            } else {
                line.clone()
            };
            let tc: Vec<&str> = line.split_whitespace().collect();
            if tc.len() >= 2 && !is_skip_summary(tc[0], tc[1]) {
                m.insert(tc[1..].join(" "), tc[0].to_string());
            }
        } else if STATUSES.iter().any(|s| line.ends_with(s)) {
            let tc: Vec<&str> = line.split_whitespace().collect();
            if tc.len() >= 2 {
                m.insert(tc[..tc.len() - 1].join(" "), tc[tc.len() - 1].to_string());
            }
        }
    }
    m
}

pub fn parse_log_seaborn(log: &str) -> StatusMap {
    let mut m = StatusMap::new();
    for line in log.split('\n') {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        if line.starts_with("FAILED") {
            m.insert(parts[1].to_string(), "FAILED".into());
        } else if line.contains(" PASSED ") {
            if parts[1] == "PASSED" {
                m.insert(parts[0].to_string(), "PASSED".into());
            }
        } else if line.starts_with("PASSED") {
            m.insert(parts[1].to_string(), "PASSED".into());
        }
    }
    m
}

pub fn parse_log_sympy(log: &str) -> StatusMap {
    let mut m = StatusMap::new();
    for c in Regex::new(r"(_*) (.*)\.py:(.*) (_*)")
        .unwrap()
        .captures_iter(log)
    {
        m.insert(format!("{}.py:{}", &c[2], &c[3]), "FAILED".into());
    }
    for raw in log.split('\n') {
        let line = raw.trim();
        if line.starts_with("test_") {
            let first = line.split_whitespace().next().unwrap_or("").to_string();
            if line.ends_with(" E") {
                m.insert(first.clone(), "ERROR".into());
            }
            if line.ends_with(" F") {
                m.insert(first.clone(), "FAILED".into());
            }
            if line.ends_with(" ok") {
                m.insert(first, "PASSED".into());
            }
        }
    }
    m
}

pub fn parse_log_matplotlib(log: &str) -> StatusMap {
    let mut m = StatusMap::new();
    for raw in log.split('\n') {
        let line = raw
            .replace("MouseButton.LEFT", "1")
            .replace("MouseButton.RIGHT", "3");
        if starts_with_status(&line) {
            let line = if line.starts_with("FAILED") {
                line.replace(" - ", " ")
            } else {
                line
            };
            let tc: Vec<&str> = line.split_whitespace().collect();
            if tc.len() <= 1 || is_skip_summary(tc[0], tc[1]) {
                continue;
            }
            m.insert(tc[1].to_string(), tc[0].to_string());
        }
    }
    m
}

/// The parser registry entries the Verified instances name.
pub fn parse(parser: &str, log: &str) -> Result<StatusMap, String> {
    Ok(match parser {
        "parse_log_pytest"
        | "parse_log_astroid"
        | "parse_log_flask"
        | "parse_log_marshmallow"
        | "parse_log_pvlib"
        | "parse_log_pyvista"
        | "parse_log_sqlfluff"
        | "parse_log_xarray" => parse_log_pytest(log),
        "parse_log_pytest_options"
        | "parse_log_pydicom"
        | "parse_log_requests"
        | "parse_log_pylint" => parse_log_pytest_options(log),
        "parse_log_pytest_v2" | "parse_log_astropy" | "parse_log_scikit" | "parse_log_sphinx" => {
            parse_log_pytest_v2(log)
        }
        "parse_log_django" => parse_log_django(log),
        "parse_log_seaborn" => parse_log_seaborn(log),
        "parse_log_sympy" => parse_log_sympy(log),
        "parse_log_matplotlib" => parse_log_matplotlib(log),
        other => return Err(format!("log parser {other} is not ported")),
    })
}

fn suite_ran(content: &str) -> bool {
    Regex::new(
        r"(?m)Executed [1-9]\d* of \d+|TOTAL: [1-9]\d* (?:SUCCESS|FAILED)|[1-9]\d* passing|Tests:\s+[1-9]\d*|Test Suites:\s+(?:\d+ \w+, )*[1-9]\d* total|^# tests [1-9]\d*|[1-9]\d* specs?, \d+ failures?|': ok$",
    )
    .unwrap()
    .is_match(content)
}

pub fn parse_test_exit_code(content: &str) -> Option<i64> {
    Regex::new(&format!(r"{}:\s*(-?\d+)", regex::escape(TEST_EXIT_CODE)))
        .unwrap()
        .captures(content)
        .and_then(|c| c[1].parse().ok())
}

/// `get_logs_eval`: (status map, found).
pub fn logs_eval(parser: &str, content: &str) -> Result<(StatusMap, bool), String> {
    if [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT]
        .iter()
        .any(|b| content.contains(b))
    {
        return Ok((StatusMap::new(), false));
    }
    if !(content.contains(START_TEST_OUTPUT) && content.contains(END_TEST_OUTPUT)) {
        return Ok((StatusMap::new(), false));
    }
    let sliced = content
        .split(START_TEST_OUTPUT)
        .nth(1)
        .unwrap_or("")
        .split(END_TEST_OUTPUT)
        .next()
        .unwrap_or("");
    let mut sm = parse(parser, sliced)?;
    if sm.is_empty() {
        sm = parse(parser, content)?;
    }
    if sm.is_empty() && !suite_ran(content) {
        return Ok((StatusMap::new(), false));
    }
    if let Some(code) = parse_test_exit_code(content) {
        if code != 0 && !sm.is_empty() && !sm.values().any(|s| s == "FAILED" || s == "ERROR") {
            return Ok((StatusMap::new(), false));
        }
    }
    Ok((sm, true))
}

fn resolve_case<'a>(case: &str, sm: &'a StatusMap) -> Option<&'a str> {
    if let Some((k, _)) = sm.get_key_value(case) {
        return Some(k.as_str());
    }
    if case.matches('[').count() > case.matches(']').count() {
        let matches: Vec<&String> = sm.keys().filter(|k| k.starts_with(case)).collect();
        let kinds: std::collections::BTreeSet<bool> = matches
            .iter()
            .map(|k| sm[*k] == "PASSED" || sm[*k] == "XFAIL")
            .collect();
        if !matches.is_empty() && kinds.len() == 1 {
            return Some(matches[0].as_str());
        }
    }
    None
}

fn passed(case: &str, sm: &StatusMap) -> bool {
    resolve_case(case, sm).is_some_and(|k| sm[k] == "PASSED" || sm[k] == "XFAIL")
}

fn maintained(case: &str, sm: &StatusMap) -> bool {
    passed(case, sm) || resolve_case(case, sm).is_some_and(|k| sm[k] == "SKIPPED")
}

fn failed(case: &str, sm: &StatusMap) -> bool {
    match resolve_case(case, sm) {
        None => true,
        Some(k) => ["FAILED", "ERROR", "SKIPPED"].contains(&sm[k].as_str()),
    }
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct TestsReport {
    pub f2p_success: Vec<String>,
    pub f2p_failure: Vec<String>,
    pub p2p_success: Vec<String>,
    pub p2p_failure: Vec<String>,
}

/// `get_eval_tests_report` for the eval types Verified uses.
pub fn tests_report(
    sm: &StatusMap,
    f2p: &[String],
    p2p: &[String],
    eval_type: &str,
) -> TestsReport {
    let mut r = TestsReport::default();
    let fail_only = eval_type == "fail_only";
    for t in f2p {
        if fail_only {
            if sm.get(t).map(String::as_str) == Some("FAILED") {
                r.f2p_failure.push(t.clone())
            } else {
                r.f2p_success.push(t.clone())
            }
        } else if passed(t, sm) {
            r.f2p_success.push(t.clone());
        } else if failed(t, sm) {
            r.f2p_failure.push(t.clone());
        }
    }
    for t in p2p {
        if fail_only {
            if sm.get(t).map(String::as_str) == Some("FAILED") {
                r.p2p_failure.push(t.clone())
            } else {
                r.p2p_success.push(t.clone())
            }
        } else if maintained(t, sm) {
            r.p2p_success.push(t.clone());
        } else if failed(t, sm) {
            r.p2p_failure.push(t.clone());
        }
    }
    r
}

/// `get_resolution_status` == RESOLVED_FULL.
pub fn resolved(r: &TestsReport) -> bool {
    let ratio = |s: usize, f: usize| {
        if s + f == 0 {
            1.0
        } else {
            s as f64 / (s + f) as f64
        }
    };
    ratio(r.f2p_success.len(), r.f2p_failure.len()) == 1.0
        && ratio(r.p2p_success.len(), r.p2p_failure.len()) == 1.0
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Grade {
    pub patch_applied: bool,
    pub resolved: bool,
    pub report: Option<TestsReport>,
}

/// `get_eval_report` for one instance from its test output log.
pub fn grade(
    parser: &str,
    eval_type: &str,
    f2p: &[String],
    p2p: &[String],
    log: &str,
) -> Result<Grade, String> {
    let (sm, found) = logs_eval(parser, log)?;
    if !found {
        return Ok(Grade {
            patch_applied: false,
            resolved: false,
            report: None,
        });
    }
    let r = tests_report(&sm, f2p, p2p, eval_type);
    Ok(Grade {
        patch_applied: true,
        resolved: resolved(&r),
        report: Some(r),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected maps from the official swebench 5.0.2 parsers on the same inputs (generated with the package).
    #[test]
    fn parsers_match_the_official_ones() {
        let py = "PASSED tests/a.py::test_x\nFAILED tests/a.py::test_y - AssertionError: no\nSKIPPED [1] tests/a.py:3: why\nXFAIL tests/a.py::test_z\nnoise\n";
        let m = parse_log_pytest(py);
        assert_eq!(m.get("tests/a.py::test_x").unwrap(), "PASSED");
        assert_eq!(m.get("tests/a.py::test_y").unwrap(), "FAILED");
        assert!(!m.contains_key("[1]"), "skip summary dropped");
        assert_eq!(m.get("tests/a.py::test_z").unwrap(), "XFAIL");
        let opt = parse_log_pytest_options(
            "PASSED tests/t.py::test_p[/usr/lib/x/name]\nPASSED tests/t.py::test_q[a-b]\n",
        );
        assert!(
            opt.contains_key("tests/t.py::test_p[/name]")
                && opt.contains_key("tests/t.py::test_q[a-b]")
        );
        let dj = "test_a (m.tests.T) ... ok\ntest_b (m.tests.T) ... FAIL\ntest_c (m.tests.T) ... skipped 'x'\ntest_d (m.tests.T) ... ERROR\nFAIL: test_e (m.tests.T)\ntest_f (m.tests.T) ... long\nok\n";
        let d = parse_log_django(dj);
        assert_eq!(d["test_a (m.tests.T)"], "PASSED");
        assert_eq!(d["test_b (m.tests.T)"], "FAILED");
        assert_eq!(d["test_c (m.tests.T)"], "SKIPPED");
        assert_eq!(d["test_d (m.tests.T)"], "ERROR");
        assert_eq!(d["test_e"], "FAILED");
        assert_eq!(d["test_f (m.tests.T)"], "PASSED");
        let v2 = parse_log_pytest_v2("\x1b[32mPASSED\x1b[0m tests/b.py::test_one\nFAILED tests/b.py::test_two - boom\ntests/b.py::test_three PASSED\n");
        assert_eq!(v2["tests/b.py::test_one"], "PASSED");
        assert_eq!(v2["tests/b.py::test_two"], "FAILED");
        assert_eq!(v2["tests/b.py::test_three"], "PASSED");
        let sy = parse_log_sympy("test_alpha ok\ntest_beta F\ntest_gamma E\n________ sympy/core/tests/test_x.py:test_delta ________\n");
        assert_eq!(sy["test_alpha"], "PASSED");
        assert_eq!(sy["test_beta"], "FAILED");
        assert_eq!(sy["test_gamma"], "ERROR");
        assert_eq!(sy["sympy/core/tests/test_x.py:test_delta"], "FAILED");
        let mp = parse_log_matplotlib("PASSED lib/t.py::test_b[MouseButton.LEFT]\n");
        assert!(mp.contains_key("lib/t.py::test_b[1]"));
        let sb = parse_log_seaborn("tests/t.py::test_a PASSED [ 50%]\nFAILED tests/t.py::test_b\n");
        assert_eq!(sb["tests/t.py::test_a"], "PASSED");
        assert_eq!(sb["tests/t.py::test_b"], "FAILED");
        // whole maps, as printed by swebench 5.0.2 for the same inputs (json, sorted keys)
        let j = |m: &StatusMap| serde_json::to_string(m).unwrap();
        assert_eq!(
            j(&m),
            r#"{"tests/a.py::test_x":"PASSED","tests/a.py::test_y":"FAILED","tests/a.py::test_z":"XFAIL"}"#
        );
        assert_eq!(
            j(&d),
            r#"{"test_a (m.tests.T)":"PASSED","test_b (m.tests.T)":"FAILED","test_c (m.tests.T)":"SKIPPED","test_d (m.tests.T)":"ERROR","test_e":"FAILED","test_f (m.tests.T)":"PASSED"}"#
        );
        assert_eq!(
            j(&v2),
            r#"{"tests/b.py::test_one":"PASSED","tests/b.py::test_three":"PASSED","tests/b.py::test_two":"FAILED"}"#
        );
        assert_eq!(
            j(&sy),
            r#"{"sympy/core/tests/test_x.py:test_delta":"FAILED","test_alpha":"PASSED","test_beta":"FAILED","test_gamma":"ERROR"}"#
        );
    }

    fn log(body: &str) -> String {
        format!(
            "+ git apply\n{START_TEST_OUTPUT}\n{body}\n{END_TEST_OUTPUT}\n{TEST_EXIT_CODE}: 1\n"
        )
    }

    #[test]
    fn grading_rules() {
        let f2p = vec!["t.py::test_fix".to_string()];
        let p2p = vec!["t.py::test_old".to_string(), "t.py::test_skip".to_string()];
        let ok = log("PASSED t.py::test_fix\nPASSED t.py::test_old\nSKIPPED t.py::test_skip\nFAILED t.py::test_other");
        let g = grade("parse_log_pytest", "pass_and_fail", &f2p, &p2p, &ok).unwrap();
        assert!(g.patch_applied && g.resolved, "{g:?}");
        let bad = log("FAILED t.py::test_fix\nPASSED t.py::test_old\nPASSED t.py::test_skip");
        assert!(
            !grade("parse_log_pytest", "pass_and_fail", &f2p, &p2p, &bad)
                .unwrap()
                .resolved
        );
        let missing = log("PASSED t.py::test_old");
        let g = grade("parse_log_pytest", "pass_and_fail", &f2p, &p2p, &missing).unwrap();
        assert!(
            !g.resolved && !g.patch_applied,
            "official: non-zero exit and no failure parsed -> not found"
        );
        let apply_fail = format!("{APPLY_PATCH_FAIL}\n{}", ok);
        let g = grade("parse_log_pytest", "pass_and_fail", &f2p, &p2p, &apply_fail).unwrap();
        assert!(!g.patch_applied && !g.resolved);
        // non-zero exit with every parsed test passing is not trusted
        let liar = log("PASSED t.py::test_fix\nPASSED t.py::test_old\nPASSED t.py::test_skip");
        assert!(
            !grade("parse_log_pytest", "pass_and_fail", &f2p, &p2p, &liar)
                .unwrap()
                .patch_applied
        );
        // truncated parametrized id (prefix match when unambiguous)
        let f2p_t = vec!["t.py::test_p[log(photon".to_string()];
        let tr = log("PASSED t.py::test_p[log(photon)]\nFAILED t.py::test_q");
        assert!(
            grade("parse_log_pytest", "pass_and_fail", &f2p_t, &[], &tr)
                .unwrap()
                .resolved
        );
    }

    #[test]
    fn eval_script_exit_code_hook() {
        let s = format!("#!/bin/bash\nset -uxo pipefail\ncd /testbed\n: '{START_TEST_OUTPUT}'\npytest -rA t.py\n: '{END_TEST_OUTPUT}'\ngit checkout x t.py\n");
        let p = prepare_eval_script(&s);
        let lines: Vec<&str> = p.lines().collect();
        let i = lines
            .iter()
            .position(|l| l.contains(END_TEST_OUTPUT))
            .unwrap();
        assert_eq!(lines[i - 1], format!("{TEST_EXIT_CODE_VAR}=$?"));
        assert!(lines[i + 1].contains(TEST_EXIT_CODE));
        assert_eq!(lines[0], "#!/bin/bash");
        assert_eq!(p.matches("set -uxo pipefail").count(), 1);
    }
}
