#!/usr/bin/env python3
"""Run the agentic task suite with opencode against one model and grade with hidden tests.

usage: run_agentic.py <label> <opencode model, e.g. glm/glm-5.3-flash> [task ...]
       env AGENTIC_TIMEOUT (s, default 3600), AGENTIC_AUTOCOMPACT=1 lets opencode compact (64K-context models)
Per task: fresh git copy of bubbles + the task's setup -> `opencode run` (build agent, test config: edits/bash
allowed, no web) -> hidden test written -> `go test ./<pkg>/` offline. Results: results/agentic.jsonl, logs and
final repos under results/agentic/<label>/<task>/.
"""
import re, json, os, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import SRC, TASKS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "agentic.jsonl")
CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opencode-test.json")
GOENV = {"GOFLAGS": "-mod=mod", "GOPROXY": "off", "GOTOOLCHAIN": "local"}


def sh(cmd, cwd, timeout=None, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **GOENV, **(env or {})})


HARNESS = os.path.dirname(os.path.abspath(__file__))
PATH_RE = re.compile(r"(?:/Users/|/tmp/|/root/|~/)[^\s\"'\\;|&)<>`]*")
# the upstream bubbles sources in the Go module cache (bubbles@v0.21.1 / @v1.0.0) are the reference solutions of the
# tasks; external_directory=deny stops opencode's read tool but not bash (cat / grep / diff, $(go env GOMODCACHE))
UPSTREAM_RE = re.compile(r"bubbles@v[0-9][^\s\"'/]*/?([a-z]*)")


def audit(events, base):
    """Absolute paths the agent's tool calls named outside its task dir (opencode's external_directory=deny should
    refuse them; this records the attempts), and whether any named the harness (tasks.py holds the hidden tests)."""
    outside, harness, upstream = set(), False, set()
    for line in events.splitlines():
        try:
            p = json.loads(line).get("part", {})
        except (ValueError, AttributeError):
            continue
        if p.get("type") != "tool":
            continue
        st = p.get("state", {})
        s = json.dumps(st.get("input", {}))
        harness |= "tasks.py" in s or HARNESS in s
        if st.get("status") == "completed":   # a refused call read nothing
            upstream.update(m or "(root)" for m in UPSTREAM_RE.findall(s))
        for m in PATH_RE.findall(s):
            if not m.startswith(base) and not base.startswith(m.rstrip("/")):
                outside.add(m[:120])
    return sorted(outside), harness, sorted(upstream)


def masked(cmd, base):
    """#modcache-leak: run the agent under bwrap with the reference solutions hidden (tmpfs over them; nothing on disk
    changes): the upstream bubbles sources and their download zips in the Go module cache, every other run's repos
    under results/agentic (many solved), and this harness dir (tasks.py holds the hidden tests; the opencode config is
    bound back read-only).  AGENTIC_MASK=0 runs unmasked (the control)."""
    if os.environ.get("AGENTIC_MASK", "1") == "0":
        return cmd
    mod = os.path.expanduser("~/go/pkg/mod")
    hide = [os.path.join(mod, "github.com/charmbracelet", v) for v in ("bubbles@v0.21.1", "bubbles@v1.0.0")]
    hide += [os.path.join(mod, "cache/download/github.com/charmbracelet/bubbles"), os.path.join(HERE, "results", "agentic"),
             HARNESS]
    args = ["bwrap", "--bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc"]
    for h in hide:
        args += ["--tmpfs", h]
    return args + ["--bind", base, base, "--ro-bind", CFG, CFG, "--chdir", os.path.join(base, "repo"), "--", *cmd]


def prepare(task, base):
    repo = os.path.join(base, "repo")
    shutil.rmtree(base, ignore_errors=True)
    shutil.copytree(SRC, repo)
    for root, dirs, files in os.walk(repo):
        os.chmod(root, 0o755)
        for f in files:
            os.chmod(os.path.join(root, f), 0o644)
    TASKS[task]["setup"](repo)
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.name=bench", "-c", "user.email=bench@localhost", "commit", "-qm", "task start"]):
        sh(c, repo)
    return repo


def main():
    label, model = sys.argv[1], sys.argv[2]
    tasks = sys.argv[3:] or list(TASKS)
    timeout = int(os.environ.get("AGENTIC_TIMEOUT", "3600"))
    for task in tasks:
        t = TASKS[task]
        base = os.path.join(HERE, "results", "agentic", label, task)
        repo = prepare(task, base)
        tmp = os.path.join(base, "tmp")   # agent scratch on disk (/tmp is tmpfs on this host)
        os.makedirs(tmp, exist_ok=True)
        # pin every helper agent (title/summary/compaction/...) to the model under test: the global opencode config
        # routes them to the production provider, whose key the test server rejects (401 killed a compacting session)
        run_cfg = os.path.join(base, "opencode-run.json")
        cfg = json.load(open(CFG))
        cfg["model"] = cfg["small_model"] = model
        cfg["agent"] = {a: {"model": model} for a in ("build", "plan", "explore", "general", "title", "summary", "compaction")}
        json.dump(cfg, open(run_cfg, "w"), indent=1)
        t0 = time.time()
        try:
            p = sh(masked(["opencode", "run", "-m", model, "--dir", repo, "--format", "json", t["prompt"]], base), repo,
                   timeout=timeout, env={"OPENCODE_CONFIG": run_cfg, "TMPDIR": tmp, "OPENCODE_DISABLE_CLAUDE_CODE": "1",
                        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
                        **({} if os.environ.get("AGENTIC_AUTOCOMPACT") == "1" else {"OPENCODE_DISABLE_AUTOCOMPACT": "1"})})
            agent_rc, events = p.returncode, p.stdout
            open(os.path.join(base, "opencode.err"), "w").write(p.stderr)
        except subprocess.TimeoutExpired as e:
            agent_rc, events = "timeout", (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        wall = round(time.time() - t0)
        open(os.path.join(base, "events.jsonl"), "w").write(events)
        tests_touched = [l for l in sh(["git", "diff", "--name-only", "HEAD"], repo).stdout.split() if l.endswith("_test.go")]
        diff = sh(["git", "diff", "--stat", "HEAD"], repo).stdout.strip().splitlines()
        if t["hidden"]:
            open(os.path.join(repo, t["pkg"], "zz_hidden_test.go"), "w").write(t["hidden"])
        g = sh(["go", "test", "-count=1", f"./{t['pkg']}/"], repo, timeout=600)
        open(os.path.join(base, "gotest.log"), "w").write(g.stdout + g.stderr)
        passed = g.returncode == 0 and not (t["protect_tests"] and tests_touched)
        n_tools = sum(1 for l in events.splitlines() if '"tool' in l)
        outside, harness, upstream = audit(events, base)
        rec = {"label": label, "model": model, "task": task, "pass": passed, "go_test_rc": g.returncode,
               "tests_touched": tests_touched, "agent_rc": agent_rc, "wall_s": wall, "tool_events": n_tools,
               "diff": diff[-1] if diff else "no changes",
               "clean": not harness and t["pkg"] not in upstream, "upstream_src": upstream, "outside_paths": outside[:20], "n_outside": len(outside), "fail_tail": (g.stdout + g.stderr)[-300:] if g.returncode else ""}
        print(json.dumps(rec), flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
