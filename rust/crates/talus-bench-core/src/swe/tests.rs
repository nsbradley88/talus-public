//! The mock-cheater suite: every anti-cheating layer is exercised by an agent that tries the vector, through the
//! real pipeline (`SwebenchTask::execute`), with a local runtime that runs the real snapshot / verify / scan
//! scripts and the agent under real strace (no containers here; the container-only properties, network
//! namespace and container freshness, are covered by `runtime` tests and the opt-in docker test).

use super::*;
use crate::mock::{spawn_openai, Degrade};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

fn sh(dir: &Path, cmd: &str) -> String {
    let o = Command::new("bash")
        .arg("-c")
        .arg(cmd)
        .current_dir(dir)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@x")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@x")
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "{cmd}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8_lossy(&o.stdout).trim().to_string()
}

pub struct Fixture {
    pub root: PathBuf,
    pub base: String,
    pub future: String,
}

/// An upstream "flask" with the fix in a later commit, cloned with history (the image's /testbed situation).
fn fixture(root: &Path) -> Fixture {
    let up = root.join("up");
    std::fs::create_dir_all(up.join("flask")).unwrap();
    std::fs::create_dir_all(up.join("tests")).unwrap();
    std::fs::write(up.join("flask/__init__.py"), "x = 1\n").unwrap();
    std::fs::write(
        up.join("tests/test_m.py"),
        "def test_old():\n    assert True\n",
    )
    .unwrap();
    sh(
        &up,
        "git init -q -b main && git add -A && git commit -qm base",
    );
    let base = sh(&up, "git rev-parse HEAD");
    sh(
        &up,
        "echo 'x = 2  # FUTURE_FIX' > flask/__init__.py && git commit -qam fix && git tag 9.9",
    );
    let future = sh(&up, "git rev-parse HEAD");
    sh(
        root,
        "git clone -q up orig && cd orig && git checkout -q main",
    );
    for d in ["site", "cache", "data", "bin"] {
        std::fs::create_dir_all(root.join(d)).unwrap();
    }
    std::fs::write(
        root.join("data/SWE-bench_Verified.jsonl"),
        "{\"instance_id\": \"gold answers\"}\n",
    )
    .unwrap();
    std::fs::write(root.join("bin/pip"), "#!/bin/sh\nexit 0\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(root.join("bin/pip"), std::fs::Permissions::from_mode(0o755)).unwrap();
    Fixture {
        root: root.to_path_buf(),
        base,
        future,
    }
}

const GOLD: &str = "diff --git a/flask/__init__.py b/flask/__init__.py\n--- a/flask/__init__.py\n+++ b/flask/__init__.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n";
const TEST_PATCH: &str = "diff --git a/tests/test_m.py b/tests/test_m.py\n--- a/tests/test_m.py\n+++ b/tests/test_m.py\n@@ -1,2 +1,5 @@\n def test_old():\n     assert True\n+\n+def test_fix():\n+    HIDDEN_TEST_MARKER = 1\n";

pub const SCENARIOS: &[&str] = &[
    "honest",
    "memorizer",
    "history",
    "network",
    "swebench",
    "testedit",
    "sitepkg_read",
    "webtool",
    "bridge",
    "future_env",
    "leaky_history",
    "netfetch",
];

struct LocalRuntime {
    fx: Fixture,
    evals: AtomicUsize,
    work: Mutex<BTreeMap<String, (PathBuf, PathBuf)>>,
}

fn scenario(inst: &Instance) -> String {
    inst.instance_id
        .rsplit('-')
        .next()
        .unwrap_or("")
        .to_string()
}

impl LocalRuntime {
    fn agent_script(&self, sc: &str, repo: &Path, sock: &Path) -> String {
        let r = self.fx.root.display();
        let fix = "printf 'x = 1 + 1\\n' > flask/__init__.py";
        let body = match sc {
            // the honest agent also explores the way models do (stash round trip, listing, stat): never flagged
            // and the false-positive shapes of the bench-swe-q27 run: stash (git's reflog helper), rev-list
            // --count, describe --tags, its own script naming SWE-bench, and a snapshot probe (a note only)
            "honest" => format!("{fix}; git -c user.name=a -c user.email=a@x stash -q; git stash list >/dev/null; git -c user.name=a -c user.email=a@x stash pop -q; git log --oneline -3 >/dev/null; git rev-list --count HEAD >/dev/null; git describe --tags >/dev/null 2>&1; python3 -I -c '# Simulate the SWE-bench harness' ; git log --all --oneline >/dev/null; ls / >/dev/null; stat /etc/hostname >/dev/null 2>&1; true"),
            "webtool" => fix.to_string(),
            // the right fix, plus the upstream file fetched at import time (only the eval network can see it)
            "netfetch" => "printf 'import urllib.request\\ntry:\\n    urllib.request.urlopen(\"https://raw.githubusercontent.com/pallets/flask/main/src/flask/__init__.py\", timeout=2)\\nexcept Exception:\\n    pass\\nx = 1 + 1\\n' > flask/__init__.py".to_string(),
            "memorizer" => "printf 'x = 2\\n' > flask/__init__.py".to_string(),
            "history" => format!(
                "git log --all --oneline > /dev/null; if git show {} > /dev/null 2>&1; then echo FOUND_FUTURE; else echo future-unreachable; fi; {fix}",
                self.fx.future
            ),
            "network" => format!("timeout 2 bash -c 'exec 3<>/dev/tcp/192.0.2.1/80' 2>/dev/null; (exec 3<>/dev/tcp/127.0.0.1/9) 2>/dev/null; {r}/bin/pip download flask==9.9 >/dev/null; {fix}"),
            "swebench" => format!("cat {r}/data/SWE-bench_Verified.jsonl > /dev/null; {fix}"),
            "testedit" => format!("{fix}; printf 'def test_fix():\\n    pass\\n' >> tests/test_m.py"),
            "sitepkg_read" => format!("cat {r}/site/flask/__init__.py 2>/dev/null; cat {r}/cache/flask-9.9-py3-none-any.whl 2>/dev/null; {fix}"),
            "bridge" => format!(
                "python3 -I -c \"import socket; s=socket.socket(socket.AF_UNIX); s.connect('{}'); s.sendall(b'GET /v1/talus/status HTTP/1.1\\r\\nHost: x\\r\\n\\r\\n'); print(s.recv(100))\"; {fix}",
                sock.display()
            ),
            _ => fix.to_string(),
        };
        format!("cd {}\n{body}\n", repo.display())
    }
}

impl Runtime for LocalRuntime {
    fn prepare(&self, _inst: &Instance) -> Result<runtime::PrepInfo, String> {
        Ok(runtime::PrepInfo {
            image: "local".into(),
            gold_ok: Some(true),
            ..Default::default()
        })
    }

    fn run_agent(&self, inst: &Instance, spec: &AgentSpec) -> Result<runtime::AgentRun, String> {
        let sc = scenario(inst);
        let w = &spec.work_dir;
        let repo = w.join("testbed");
        self.work.lock().unwrap().insert(
            inst.instance_id.clone(),
            (w.clone(), spec.bridge_socket_dir.clone()),
        );
        sh(
            w,
            &format!(
                "cp -a {} {}",
                self.fx.root.join("orig").display(),
                repo.display()
            ),
        );
        // the image's environment: an editable install of the repo; future_env adds a released future copy
        let site = w.join("site");
        std::fs::create_dir_all(&site).unwrap();
        std::fs::write(
            site.join("flask.egg-link"),
            format!("{}\n.\n", repo.display()),
        )
        .unwrap();
        if sc == "future_env" {
            std::fs::create_dir_all(site.join("flask-9.9.dist-info")).unwrap();
            std::fs::create_dir_all(site.join("flask")).unwrap();
            std::fs::write(site.join("flask/__init__.py"), "x = 2\n").unwrap();
        }
        let write = |name: &str, body: &str| {
            let p = w.join(name);
            std::fs::write(&p, body).unwrap();
            p
        };
        if sc != "leaky_history" {
            let s = write("snapshot.sh", SNAPSHOT_SH_LOCAL);
            sh(
                w,
                &format!(
                    "bash {} {} {}",
                    s.display(),
                    repo.display(),
                    inst.base_commit
                ),
            );
        }
        let head = sh(&repo, "git rev-parse HEAD");
        let v = write("verify.sh", snapshot::VERIFY_SH);
        let verify = sh(
            w,
            &format!("bash {} {} {}", v.display(), repo.display(), w.display()),
        );
        let scan = write("scan.py", sitepkg::SCAN_PY);
        let sitepkg_out = sh(
            w,
            &format!(
                "python3 -I {} flask flask {} --paths {}:{} --roots {}",
                scan.display(),
                repo.display(),
                site.display(),
                repo.display(),
                site.display()
            ),
        );
        // the agent's script sits in the bridge directory (mounted at /run/talus in the container, allowed)
        let agent = spec.bridge_socket_dir.join("agent.sh");
        std::fs::write(
            &agent,
            self.agent_script(&sc, &repo, &spec.bridge_socket_dir.join("model.sock")),
        )
        .unwrap();
        // the artifacts land where the docker runtime leaves them (swe-rescore rebuilds runs from them)
        let audit_dir = w.join("audit");
        std::fs::create_dir_all(&audit_dir).unwrap();
        let trace = audit_dir.join("trace.log");
        // a stand-in for opencode: it probes the filesystem and queries git itself (harness behaviour, never
        // flagged), then runs the agent's commands through a shell, as opencode's bash tool does
        let oc_dir = spec.bridge_socket_dir.join("oc");
        std::fs::create_dir_all(&oc_dir).unwrap();
        let oc = oc_dir.join("opencode");
        std::fs::write(
            &oc,
            "#!/bin/bash\ncat /etc/hostname /package.json /nonexistent/.git >/dev/null 2>&1\ngit rev-list --max-parents=0 HEAD >/dev/null 2>&1\ngit remote >/dev/null 2>&1\n/bin/bash \"$1\"\n",
        )
        .unwrap();
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&oc, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        // a clean environment, as the container's `env -i`
        let o = Command::new("strace")
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .args(audit::STRACE_ARGS)
            .arg("-o")
            .arg(&trace)
            .arg(&oc)
            .arg(&agent)
            .current_dir(&repo)
            .env("HOME", &spec.bridge_socket_dir)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .output()
            .map_err(|e| e.to_string())?;
        let events = if sc == "webtool" {
            r#"{"part":{"type":"tool","tool":"webfetch","state":{"input":{"url":"https://github.com/pallets/flask/pull/1"}}}}"#.to_string()
        } else {
            json!({"part": {"type": "tool", "tool": "edit", "state": {"input": {"filePath": repo.join("flask/__init__.py")}}}}).to_string()
        };
        let patch = sh(&repo, "git add -A && git diff --cached --binary HEAD");
        let patch = if patch.is_empty() {
            patch
        } else {
            patch + "\n"
        };
        for (f, body) in [
            ("head", &head),
            ("patch.diff", &patch),
            ("events.jsonl", &events),
            ("verify.json", &verify),
            ("sitepkg.json", &sitepkg_out),
        ] {
            std::fs::write(audit_dir.join(f), body).unwrap();
        }
        Ok(runtime::AgentRun {
            head,
            patch,
            events,
            // exercise the streaming path the docker runtime uses
            strace: String::new(),
            strace_file: Some(trace.clone()),
            verify: serde_json::from_str(&verify).unwrap_or(Value::Null),
            sitepkg: serde_json::from_str(&sitepkg_out).unwrap_or(Value::Null),
            exit: String::from_utf8_lossy(&o.stdout).to_string(),
            wall_s: 0.0,
        })
    }

    fn run_eval(&self, inst: &Instance, patch: &str) -> Result<runtime::EvalRun, String> {
        self.evals.fetch_add(1, Ordering::Relaxed);
        // a fresh copy of the image's repo (with its history: the agent is gone), base checked out
        let d = self.fx.root.join(format!(
            "eval-{}-{}",
            inst.instance_id,
            self.evals.load(Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&d);
        sh(&self.fx.root, &format!("cp -a orig {}", d.display()));
        sh(&d, &format!("git checkout -q {}", inst.base_commit));
        std::fs::write(d.join("model.patch"), patch).unwrap();
        let applied = Command::new("git")
            .args(["apply", "model.patch"])
            .current_dir(&d)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !applied {
            return Ok(runtime::EvalRun {
                log: format!("{}\n", harness::APPLY_PATCH_FAIL),
                network: Value::Null,
            });
        }
        // a stand-in for the egress gateway (CI stays offline): every lookup or connect the tests make is
        // logged in the gateway's log formats and refused, then judged against the gold run (no network use)
        let hook = d.join("tbnethook");
        std::fs::create_dir_all(&hook).unwrap();
        let dns_log = d.join("dns.log");
        let conn_log = d.join("conn.log");
        std::fs::write(
            hook.join("tbnethook.py"),
            format!(
                r#"import socket
def _q(host):
    open({dns:?}, "a").write("dnsmasq[0]: 1 127.0.0.1/1 query[A] %s from 127.0.0.1\n" % host)
    raise socket.gaierror(-2, "Name or service not known")
def _getaddrinfo(host, *a, **k):
    if host and host not in ("localhost",) and not str(host).startswith("127."):
        try:
            socket.inet_aton(host)
        except OSError:
            _q(host)
    return _orig(host, *a, **k)
def _connect(self, addr):
    if isinstance(addr, tuple) and not str(addr[0]).startswith("127."):
        open({conn:?}, "a").write("0.0 IP 10.0.0.2.1 > %s.%s: Flags [S], seq 1, length 0\n" % (addr[0], addr[1]))
        raise TimeoutError("blackholed")
    return _oconnect(self, addr)
_orig = socket.getaddrinfo
_oconnect = socket.socket.connect
socket.getaddrinfo = _getaddrinfo
socket.socket.connect = _connect
"#,
                dns = dns_log.display().to_string(),
                conn = conn_log.display().to_string()
            ),
        )
        .unwrap();
        std::fs::write(d.join("tp.patch"), &inst.test_patch).unwrap();
        // as the official eval script: the test files are reset to the base before the hidden tests go in
        sh(
            &d,
            &format!("git checkout -q {} -- tests", inst.base_commit),
        );
        sh(&d, "git apply tp.patch");
        let out = sh(&d, &format!("python3 -I -c \"import sys; sys.path.insert(0, '{}'); import tbnethook; sys.path.insert(0, '.'); import flask; ok = flask.x == 2; print(('PASSED' if ok else 'FAILED') + ' tests/test_m.py::test_fix'); print('PASSED tests/test_m.py::test_old'); sys.exit(0 if ok else 1)\" || true", hook.display()));
        let code = if out.contains("FAILED") { 1 } else { 0 };
        let ev = super::egress::parse(
            "allow",
            &std::fs::read_to_string(&dns_log).unwrap_or_default(),
            &std::fs::read_to_string(&conn_log).unwrap_or_default(),
            "",
            "",
        );
        // the gold fix of the fixture uses no network
        Ok(runtime::EvalRun {
            log: format!(
                "{}\n{out}\n{}\n{}: {code}\n",
                harness::START_TEST_OUTPUT,
                harness::END_TEST_OUTPUT,
                harness::TEST_EXIT_CODE
            ),
            network: super::egress::to_value(&ev, &super::egress::NetEvidence::default()),
        })
    }

    fn policy(&self, inst: &Instance, module: &str, dist: &str, port: u16) -> audit::Policy {
        let (w, sockdir) = self
            .work
            .lock()
            .unwrap()
            .get(&inst.instance_id)
            .cloned()
            .unwrap_or_default();
        let mut p = audit::Policy::container(module, dist, port);
        p.repo = w.join("testbed").display().to_string();
        p.allowed = [
            "/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc", "/dev", "/sys",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        p.allowed
            .push(self.fx.root.join("bin").display().to_string());
        p.allowed.push(sockdir.display().to_string()); // the bridge directory (/run/talus in the container)
        p.harness_exe = "/nonexistent".into();
        p
    }
}

/// SNAPSHOT_SH without the global git config write (tests must not touch ~/.gitconfig).
const SNAPSHOT_SH_LOCAL: &str = snapshot::SNAPSHOT_SH;

fn rows(fx: &Fixture) -> Vec<Value> {
    SCENARIOS
        .iter()
        .map(|sc| {
            json!({"instance_id": format!("pallets__flask-{sc}"), "repo": "pallets/flask", "base_commit": fx.base,
                   "problem_statement": format!("x should be 2 ({sc})"), "created_at": "2023-04-01T00:00:00Z", "version": "2.3",
                   "patch": GOLD, "test_patch": TEST_PATCH, "FAIL_TO_PASS": ["tests/test_m.py::test_fix"],
                   "PASS_TO_PASS": ["tests/test_m.py::test_old"], "image": "local", "eval_script": "#!/bin/bash\n",
                   "log_parser": "parse_log_pytest", "eval_type": "pass_and_fail"})
        })
        .collect()
}

#[test]
fn cheater_suite_every_layer_blocks_or_flags() {
    if Command::new("strace").arg("-V").output().is_err()
        || Command::new("git").arg("--version").output().is_err()
    {
        eprintln!("needs strace and git: skipped");
        return;
    }
    let root = std::env::temp_dir().join(format!("tb-swe-suite-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let fx = fixture(&root);
    let cache = root.join("dscache");
    dataset::write_cache(&cache, &rows(&fx), &Some("test-rev".into())).unwrap();
    let rt = Arc::new(LocalRuntime {
        fx,
        evals: AtomicUsize::new(0),
        work: Mutex::new(BTreeMap::new()),
    });
    set_test_runtime(Some(rt.clone()));
    let task = SwebenchTask::default();
    let params = json!({"n": SCENARIOS.len(), "seed": 1, "offline": true, "runtime": "test", "model_cutoff": "2023-01-01"});
    let cases = task
        .cases(&TaskContext {
            params: &params,
            cache: &cache,
            run_dir: Some(&root),
            ..TaskContext::for_tests(Tier::Smoke)
        })
        .unwrap();
    assert_eq!(cases.len(), SCENARIOS.len());
    // layer 3: nothing hidden in what is recorded or shown to the agent
    let case_json = serde_json::to_string(&cases).unwrap();
    for hidden in [
        "HIDDEN_TEST_MARKER",
        "test_fix",
        "x = 2",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    ] {
        assert!(!case_json.contains(hidden), "the cases leak {hidden}");
    }
    let (mock, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let mut results: BTreeMap<String, Value> = BTreeMap::new();
    for c in &cases {
        let env = ExecEnv {
            base_url: format!("http://{mock}"),
            api_key: None,
            model: "m".into(),
            context: None,
            work_dir: root.join("work"),
            label: "t".into(),
            params: params.clone(),
            limit: None,
            temperature: 0.0,
            seed: None,
            max_tokens: 100,
            cancel: Default::default(),
        };
        let o = task.execute(c, &env).unwrap();
        let sc = c.case_id.rsplit('-').next().unwrap().to_string();
        let mut d = o.score.details.clone();
        d["pass"] = json!(o.score.pass);
        d["error"] = json!(o.error);
        results.insert(sc, d);
    }
    let stage = |sc: &str| results[sc]["stage"].as_str().unwrap_or("?").to_string();
    let kinds = |sc: &str| -> Vec<String> {
        results[sc]["flags"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|f| f["kind"].as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default()
    };
    let has = |sc: &str, k: &str| kinds(sc).iter().any(|x| x == k);
    // the honest agent resolves, clean
    assert_eq!(stage("honest"), "resolved", "{:#?}", results["honest"]);
    assert!(
        kinds("honest").is_empty(),
        "{:#?}",
        results["honest"]["flags"]
    );
    assert!(results["honest"]["pass"].as_bool().unwrap());
    assert!(
        results["honest"]["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n["kind"] == "git_probe"),
        "the snapshot probe is recorded as a note: {:#?}",
        results["honest"]
    );
    // layer 6: memorization is measured (near-verbatim), not blocked
    assert_eq!(stage("memorizer"), "resolved");
    assert_eq!(results["memorizer"]["near_verbatim"], true);
    assert_eq!(results["honest"]["near_verbatim"], false);
    assert_eq!(
        results["honest"]["after_cutoff"], true,
        "created 2023-04 > cutoff 2023-01"
    );
    // layer 2: history is gone, and reaching for it is flagged
    assert!(has("history", "git_history"), "{:#?}", results["history"]);
    assert!(results["history"]["agent_exit"]
        .as_str()
        .unwrap()
        .contains("future-unreachable"));
    assert!(!results["history"]["agent_exit"]
        .as_str()
        .unwrap()
        .contains("FOUND_FUTURE"));
    assert!(
        has("leaky_history", "history"),
        "an unverified snapshot is INVALID: {:#?}",
        results["leaky_history"]
    );
    // layer 1: network attempts and network tools; the bridge refuses other APIs; web tools flagged
    assert!(
        has("network", "network") && has("network", "network_tool"),
        "{:#?}",
        results["network"]
    );
    assert!(has("bridge", "bridge_denied"), "{:#?}", results["bridge"]);
    assert!(has("webtool", "web_tool"));
    // layer 5: reads of SWE-bench data and of the target's installed copy
    assert!(
        has("swebench", "swebench_read"),
        "{:#?}",
        results["swebench"]
    );
    assert!(
        has("sitepkg_read", "outside_repo") || has("sitepkg_read", "target_copy_read"),
        "{:#?}",
        results["sitepkg_read"]
    );
    // layer 4: a future release in the environment
    assert!(
        has("future_env", "future_copy"),
        "{:#?}",
        results["future_env"]
    );
    // layer 3: editing tests is rejected
    assert!(has("testedit", "test_edit"), "{:#?}", results["testedit"]);
    // every flagged instance is INVALID and not passed, and was never evaluated
    for sc in [
        "history",
        "leaky_history",
        "network",
        "bridge",
        "webtool",
        "swebench",
        "sitepkg_read",
        "future_env",
        "testedit",
        "netfetch",
    ] {
        assert_eq!(stage(sc), "invalid", "{sc}: {:#?}", results[sc]);
        assert_eq!(results[sc]["pass"], false, "{sc}");
        // pass/fail and flags are reported separately: the patch was still evaluated
        assert!(
            results[sc]["eval"].is_object() || results[sc]["eval_stage"].is_string(),
            "{sc}: {:#?}",
            results[sc]
        );
    }
    // the cheaters wrote the right fix: resolved ignoring flags, but not passed
    assert_eq!(results["history"]["resolved_ignoring_flags"], true);
    // the eval network: an import-time fetch of the upstream fix is a contact the gold run never made
    assert!(
        has("netfetch", "eval_network"),
        "{:#?}",
        results["netfetch"]
    );
    assert_eq!(results["netfetch"]["resolved_ignoring_flags"], true);
    assert!(
        results["netfetch"]["eval_network"]["novel"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n == "name raw.githubusercontent.com"),
        "{:#?}",
        results["netfetch"]["eval_network"]
    );
    assert!(
        results["netfetch"]["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n["kind"] == "patch_network_code"),
        "the patch scan notes the fetch code"
    );
    // the honest agent's tests used no network
    assert!(results["honest"]["eval_network"]["novel"]
        .as_array()
        .unwrap()
        .is_empty());
    assert_eq!(
        rt.evals.load(Ordering::Relaxed),
        SCENARIOS.len(),
        "every instance with a patch is evaluated, flagged or not"
    );

    // the same runtime through the runner: reference and candidate get the same draw, paired, with provenance
    let (mock2, _) = spawn_openai(0, Degrade::NONE).unwrap();
    let cfg = crate::config::Config::from_yaml(&format!(
        r#"
experiment: {{id: swe-run, seed: 3}}
reference: {{id: ref, endpoint: "http://{mock}"}}
candidates: [{{id: cand, endpoint: "http://{mock2}"}}]
benchmarks:
  tier: smoke
  tasks: [swebench_verified]
  dataset_cache: "{}"
  params: {{swebench_verified: {{n: 4, seed: 4, offline: true, runtime: test}}}}
quality_gates: {{min_cases: 2}}
"#,
        cache.display()
    ))
    .unwrap();
    let out = root.join("run");
    let o = crate::runner::run(
        &cfg,
        &crate::runner::RunOptions {
            output_dir: Some(out.clone()),
            ..Default::default()
        },
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        &mut |_| {},
    )
    .unwrap();
    let base = crate::case::read_jsonl(&out.join("baseline/cases.jsonl")).unwrap();
    let cand = crate::case::read_jsonl(&out.join("candidates/cand/cases.jsonl")).unwrap();
    let ids =
        |v: &Vec<crate::case::CaseResult>| v.iter().map(|r| r.case_id.clone()).collect::<Vec<_>>();
    assert_eq!(base.len(), 4);
    assert_eq!(
        ids(&base),
        ids(&cand),
        "the reference and the candidate get the same draw"
    );
    let ds = std::fs::read_to_string(out.join("datasets.json")).unwrap();
    assert!(
        ds.contains("\"seed\": 4") && ds.contains("\"seed_source\": \"param\""),
        "{ds}"
    );
    let md = std::fs::read_to_string(out.join("report.md")).unwrap();
    assert!(
        md.contains("draw seed 4 (param)") && md.contains("memorization cannot be blocked"),
        "caveats in the report"
    );
    assert!(md.contains("Wilson intervals"));
    let metrics = std::fs::read_to_string(out.join("baseline/metrics.json")).unwrap();
    assert!(metrics.contains("invalid_instances") && metrics.contains("flags_by_kind"));
    assert!(!o.verdicts.is_empty());

    // swe-rescore (on the candidate: LocalRuntime's per-instance allowlist holds the last run's work dirs):
    // the same auditor changes nothing
    let r = super::rescore::rescore(&out, "candidates/cand", false).unwrap();
    assert!(r.changes.is_empty(), "{:?}", r.changes);
    assert_eq!(r.kept, 4, "{:?}", r.skipped);
    // an older auditor had wrongly flagged a resolved instance: re-judging restores it, and its patch
    // (never evaluated while it was invalid) is evaluated now
    let path = out.join("candidates/cand/cases.jsonl");
    let mut rows = crate::case::read_jsonl(&path).unwrap();
    let i = rows
        .iter()
        .position(|r| r.score.details["eval"].is_object())
        .expect("an evaluated instance in the draw");
    let want = rows[i].score.details["stage"].as_str().unwrap().to_string();
    let want_pass = rows[i].score.pass;
    let id = rows[i].score.details["instance_id"]
        .as_str()
        .unwrap()
        .to_string();
    let d = &mut rows[i].score.details;
    d["stage"] = json!("invalid");
    d["flags"] = json!([{"kind": "git_history", "detail": "git reflog delete --updateref --rewrite refs/stash@{0}", "source": "strace"}]);
    d.as_object_mut().unwrap().remove("eval");
    rows[i].score.pass = false;
    std::fs::write(&path, crate::case::to_jsonl(&rows)).unwrap();
    let evals_before = rt.evals.load(Ordering::Relaxed);
    let r = super::rescore::rescore(&out, "candidates/cand", false).unwrap();
    assert_eq!(r.changes.len(), 1, "{:?}", r.changes);
    assert_eq!(r.changes[0].0, id);
    assert_eq!(
        (r.changes[0].1.as_str(), r.changes[0].2.as_str()),
        ("invalid", want.as_str())
    );
    assert_eq!(
        rt.evals.load(Ordering::Relaxed),
        evals_before + 1,
        "only the restored patch is evaluated"
    );
    let rows = crate::case::read_jsonl(&path).unwrap();
    let row = rows
        .iter()
        .find(|r| r.score.details["instance_id"] == id.as_str())
        .unwrap();
    assert_eq!(row.score.pass, want_pass);
    assert_eq!(row.score.details["rescored"]["previous_stage"], "invalid");
    assert!(std::fs::read_dir(out.join("candidates/cand"))
        .unwrap()
        .any(|e| e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("pre-rescore")));
    set_test_runtime(None);
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn draw_is_persisted_and_reused_on_resume() {
    let root = std::env::temp_dir().join(format!("tb-swe-draw-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let cache = root.join("c");
    let rows = dataset::tests::synthetic(&[("django/django", 40), ("sympy/sympy", 20)]);
    dataset::write_cache(&cache, &rows, &None).unwrap();
    let p = json!({"n": 10, "offline": true});
    let t1 = SwebenchTask::default();
    let c1 = t1
        .cases(&TaskContext {
            params: &p,
            cache: &cache,
            run_dir: Some(&root),
            ..TaskContext::for_tests(Tier::Smoke)
        })
        .unwrap();
    let d1: Draw =
        serde_json::from_str(&std::fs::read_to_string(root.join(DRAW_FILE)).unwrap()).unwrap();
    assert_eq!(d1.seed_source, "csprng");
    assert_eq!(d1.per_repo.values().sum::<usize>(), 10);
    // a fresh run draws anew (new CSPRNG seed)
    let t2 = SwebenchTask::default();
    let c2 = t2
        .cases(&TaskContext {
            params: &p,
            cache: &cache,
            run_dir: Some(&root.join("other")),
            ..TaskContext::for_tests(Tier::Smoke)
        })
        .unwrap();
    assert_ne!(
        c1.iter().map(|c| &c.case_id).collect::<Vec<_>>(),
        c2.iter().map(|c| &c.case_id).collect::<Vec<_>>()
    );
    // a resumed run reuses the draw exactly
    let t3 = SwebenchTask::default();
    let c3 = t3
        .cases(&TaskContext {
            params: &p,
            cache: &cache,
            run_dir: Some(&root),
            resume: true,
            ..TaskContext::for_tests(Tier::Smoke)
        })
        .unwrap();
    assert_eq!(c1, c3);
    assert_eq!(t3.provenance()["seed"], d1.seed);
    assert_eq!(t3.provenance()["seed_source"], "resumed");
    // --seed reproduces it in a new run
    let ps = json!({"n": 10, "offline": true, "seed": d1.seed});
    let t4 = SwebenchTask::default();
    let c4 = t4
        .cases(&TaskContext {
            params: &ps,
            cache: &cache,
            run_dir: Some(&root.join("x")),
            ..TaskContext::for_tests(Tier::Smoke)
        })
        .unwrap();
    assert_eq!(c1, c4);
    assert!(t4
        .caveats()
        .iter()
        .any(|c| c.contains("offline run requested")));
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn strip_mode_removes_test_hunks() {
    let p = format!("{GOLD}{TEST_PATCH}");
    let s = strip_files(&p, &["tests/test_m.py".to_string()]);
    assert!(s.contains("flask/__init__.py") && !s.contains("tests/test_m.py"));
}

/// Opt-in (TALUS_BENCH_DOCKER_TESTS=1, needs docker and a local `ubuntu:26.04` image; pulls nothing):
/// a container started like the agent container cannot reach anything.
#[test]
fn docker_network_none_blocks_everything() {
    if std::env::var("TALUS_BENCH_DOCKER_TESTS").as_deref() != Ok("1") {
        return;
    }
    let o = Command::new("docker")
        .args(["run", "--rm", "--pull", "never", "--network", "none", "ubuntu:26.04", "bash", "-c"])
        .arg("for t in /dev/tcp/1.1.1.1/443 /dev/tcp/140.82.112.3/443 /dev/tcp/8.8.8.8/53; do (exec 3<>$t) 2>/dev/null && echo OPEN $t || echo BLOCKED $t; done; ip -o link 2>/dev/null | wc -l; cat /proc/net/dev | tail -n +3 | cut -d: -f1 | tr -d ' '")
        .output()
        .unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(!out.contains("OPEN"), "{out}");
    assert_eq!(out.matches("BLOCKED").count(), 3, "{out}");
    let ifaces: Vec<&str> = out
        .lines()
        .filter(|l| !l.starts_with("BLOCKED") && !l.chars().all(|c| c.is_ascii_digit()))
        .collect();
    assert_eq!(ifaces, vec!["lo"], "only loopback exists: {out}");
}

#[test]
fn a_fixed_subset_replaces_the_draw() {
    let root = std::env::temp_dir().join(format!("tb-swe-fixed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let cache = root.join("c");
    let rows = dataset::tests::synthetic(&[("django/django", 10), ("sympy/sympy", 5)]);
    dataset::write_cache(&cache, &rows, &None).unwrap();
    let ids: Vec<String> = rows
        .iter()
        .map(|r| r["instance_id"].as_str().unwrap().to_string())
        .collect();
    let pick = vec![ids[3].clone(), ids[12].clone()];
    std::fs::create_dir_all(&root).unwrap();
    let f = root.join("subset.json");
    std::fs::write(&f, json!({"instance_ids": pick, "seed": 27}).to_string()).unwrap();
    let p = json!({"offline": true, "instances_file": f.to_str().unwrap()});
    let t = SwebenchTask::default();
    let cases = t
        .cases(&TaskContext {
            params: &p,
            cache: &cache,
            run_dir: Some(&root.join("run")),
            ..TaskContext::for_tests(Tier::Smoke)
        })
        .unwrap();
    let got: Vec<String> = cases
        .iter()
        .map(|c| c.expected["instance_id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(got, pick);
    let d: Draw =
        serde_json::from_str(&std::fs::read_to_string(root.join("run").join(DRAW_FILE)).unwrap())
            .unwrap();
    assert_eq!(d.seed_source, "fixed:subset.json");
    assert_eq!(d.seed, 27);
    assert_eq!(d.per_repo.values().sum::<usize>(), 2);
    // an id outside the dataset is refused
    let bad = json!({"offline": true, "instance_ids": ["nope__nope-1"]});
    assert!(SwebenchTask::default()
        .cases(&TaskContext {
            params: &bad,
            cache: &cache,
            run_dir: Some(&root.join("run2")),
            ..TaskContext::for_tests(Tier::Smoke)
        })
        .is_err());
    let _ = std::fs::remove_dir_all(&root);
}
