# talus-bench livecodebench sandbox runner (written for Talus; MIT). Runs inside the sandbox container as its
# entrypoint (`python3 -I -B -c <this>`), or on the host in the test-only "local" executor.
#
# Input: one JSON object on stdin
#   {"code": str, "mode": "stdin"|"functional", "fn_name": str|null, "public": [{"input","output"}...],
#    "private": <list, a JSON string, or LiveCodeBench's base64(zlib(pickle(json string))) blob>,
#    "timeout_s": float, "max_output_bytes": int, "demote": bool}
# Output: one JSON object on stdout (see `result` below).
#
# The parent (uid 0 in the container) keeps the tests, including the expected outputs, in memory only; they are
# never written to the filesystem. Each test runs the solution in a fresh child process demoted to uid/gid 65534
# (no capabilities left; the runner keeps SETUID/SETGID/KILL), in its own session, with CPU / file-size / process rlimits and a wall-clock timeout; the
# child's output goes to files in /tmp (tmpfs) so a flood cannot exhaust the parent; it runs at nice 10 so a
# fork bomb or busy threads cannot starve the parent's timeout. The child installs a Python
# audit hook that reports network, process-spawn, file-write and ctypes attempts on its result pipe: these are
# evidence (the container is the enforcement: no network, read-only root, memory/pids/cpu limits).
#
# Grading rules follow LiveCodeBench's published semantics (reimplemented, no code copied):
# - stdin tests: output and expected are stripped, split into lines and each line stripped; the line counts must
#   match; a line matches if equal, or if both split on whitespace into the same number of tokens that are equal
#   as decimal numbers.
# - functional tests: each input line is a JSON value (one per argument); the expected output is JSON; the
#   method `Solution().<fn_name>(*args)` must return an equal value (tuples compare as lists).
# - the first failing test ends the evaluation (as upstream); a problem passes only if every test passes.
import base64, json, os, resource, signal, subprocess, sys, tempfile, threading, time, zlib
from decimal import Decimal, InvalidOperation

PRELUDE = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string, re, datetime, collections, heapq, bisect, copy, math, random, statistics, itertools, functools
import operator, io, sys, json
sys.setrecursionlimit(600000)
"""

# The child: argv = mode, fn_name, solution path, result fd. Reads a functional test's input from stdin before
# the solution runs; a stdin-mode solution reads stdin itself.
CHILD = r"""
import json, os, sys
mode, fn, path, rfd = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
SPAWN = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "pty.spawn"}
FS = {"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.truncate", "os.link",
      "os.symlink", "shutil.rmtree", "shutil.copyfile", "shutil.move"}
WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
seen = set()
def hook(ev, a):
    try:
        if ev.startswith("socket."):
            kind = "network"
        elif ev in SPAWN:
            kind = "spawn"
        elif ev.startswith("ctypes."):
            kind = "ctypes"
        elif ev in FS:
            kind = "file_write"
        elif ev == "open":
            p, m, fl = (tuple(a) + (None, None, None))[:3]
            if not ((isinstance(m, str) and any(c in m for c in "wax+")) or (isinstance(fl, int) and fl & WFLAGS)):
                return
            kind = "file_write"
        else:
            return
        key = (kind, ev)
        if key in seen:
            return
        seen.add(key)
        os.write(rfd, ("F " + json.dumps({"kind": kind, "event": ev, "detail": repr(a)[:200]}) + "\n").encode())
    except Exception:
        pass
args = None
if mode == "functional":
    raw = sys.stdin.read()
    args = [json.loads(l) for l in raw.rstrip("\n").split("\n")]
src = open(path).read()
sys.addaudithook(hook)
if mode == "stdin":
    sys.argv = [path]
    code = compile(src, path, "exec")
    g = {"__name__": "__main__", "__file__": path, "__builtins__": __builtins__}
    exec(code, g)
else:
    g = {"__name__": "tb_solution", "__builtins__": __builtins__}
    exec(compile(src, path, "exec"), g)
    if "Solution" in g:
        f = getattr(g["Solution"](), fn)
    else:
        f = g[fn]
    r = f(*args)
    try:
        payload = json.dumps(r)
    except Exception:
        payload = json.dumps({"__unserializable__": repr(r)[:200]})
    os.write(rfd, ("R " + payload + "\n").encode())
"""


def unpickle_str(b):
    """The str inside a pickle that holds exactly one str (what LiveCodeBench's private tests are). A tiny parser
    for those opcodes only; anything else is refused, so no pickle code ever runs."""
    i = 2 if b[:1] == b"\x80" else 0
    out = None
    while i < len(b):
        op = b[i]
        i += 1
        if op == 0x95:  # FRAME
            i += 8
        elif op in (0x58, 0x8C, 0x8D):  # BINUNICODE, SHORT_BINUNICODE, BINUNICODE8
            w = {0x58: 4, 0x8C: 1, 0x8D: 8}[op]
            n = int.from_bytes(b[i:i + w], "little")
            i += w
            out = b[i:i + n].decode("utf-8")
            i += n
        elif op == 0x94:  # MEMOIZE
            pass
        elif op == 0x2E:  # STOP
            if out is None:
                raise ValueError("pickle holds no string")
            return out
        else:
            raise ValueError("unsupported pickle opcode 0x%02x" % op)
    raise ValueError("pickle without STOP")


def decode_tests(x):
    if isinstance(x, list):
        return x
    if not x:
        return []
    try:
        return json.loads(x)
    except ValueError:
        pass
    return json.loads(unpickle_str(zlib.decompress(base64.b64decode(x))))


def stripped_lines(s):
    return [l.strip() for l in s.strip().split("\n")]


def stdio_equal(got, want):
    a, b = stripped_lines(got), stripped_lines(want)
    if len(a) != len(b):
        return False, "line count %d != %d" % (len(a), len(b))
    for k, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        try:
            dx = [Decimal(t) for t in x.split()]
            dy = [Decimal(t) for t in y.split()]
        except (InvalidOperation, ValueError):
            return False, "line %d differs" % (k + 1)
        if len(dx) != len(dy) or dx != dy:
            return False, "line %d differs" % (k + 1)
    return True, ""


def as_lists(v):
    if isinstance(v, (list, tuple)):
        return [as_lists(x) for x in v]
    if isinstance(v, dict):
        return {k: as_lists(x) for k, x in v.items()}
    return v


def main():
    job = json.loads(sys.stdin.read())
    t_start = time.time()
    result = {"tests_total": 0, "tests_run": 0, "passed": 0, "flags": [], "first_fail": None, "max_test_ms": 0.0,
              "decode_error": None}
    try:
        public = decode_tests(job.get("public") or [])
        tests = public + decode_tests(job.get("private"))
        n_public = len(public)
    except Exception as e:
        result["decode_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        print(json.dumps(result))
        return
    result["tests_total"] = len(tests)
    mode = job["mode"]
    fn = job.get("fn_name") or ""
    timeout = float(job.get("timeout_s") or 6)
    max_out = int(job.get("max_output_bytes") or (64 << 20))
    demote = bool(job.get("demote")) and os.geteuid() == 0
    work = tempfile.mkdtemp(prefix="tb-lcb-")
    os.chmod(work, 0o755)
    sol = os.path.join(work, "sol.py")
    with open(sol, "w") as f:
        f.write(PRELUDE + "\n" + job["code"] + "\n")
    os.chmod(sol, 0o644)
    flags = {}
    for idx, t in enumerate(tests):
        tin, tout = t["input"], t["output"]
        out_p = os.path.join(work, "out%d" % idx)
        err_p = os.path.join(work, "err%d" % idx)
        rr, rw = os.pipe()
        cwd = os.path.join(work, "cwd%d" % idx)
        os.mkdir(cwd)

        def pre(cwd=cwd):
            os.setsid()
            # below the runner's priority: a fork bomb or busy threads must not starve the runner's timeout
            os.nice(10)
            for r, v in ((resource.RLIMIT_CPU, int(timeout) + 2), (resource.RLIMIT_FSIZE, max_out),
                         (resource.RLIMIT_CORE, 0)):
                resource.setrlimit(r, (v, v))
            if demote:
                os.setgroups([])
                os.setgid(65534)
                os.setuid(65534)
                resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))  # threads count too
                d = tempfile.mkdtemp(prefix="run-", dir="/tmp")
                os.chdir(d)
            else:
                os.chdir(cwd)

        t0 = time.time()
        with open(out_p, "wb") as fo, open(err_p, "wb") as fe:
            p = subprocess.Popen([sys.executable, "-I", "-B", "-c", CHILD, mode, fn, sol, str(rw)],
                                 stdin=subprocess.PIPE, stdout=fo, stderr=fe, pass_fds=(rw,), preexec_fn=pre,
                                 close_fds=True)
            os.close(rw)
            kind = None
            chunks = []

            def feed(pipe=p.stdin, data=tin.encode()):
                try:
                    pipe.write(data)
                    pipe.close()
                except (BrokenPipeError, OSError, ValueError):
                    pass

            def rd():
                n = 0
                while True:
                    c = os.read(rr, 65536)
                    if not c:
                        break
                    if n < max_out:
                        chunks.append(c)
                        n += len(c)

            tw = threading.Thread(target=feed, daemon=True)
            th = threading.Thread(target=rd, daemon=True)
            tw.start()
            th.start()
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                kind = "timeout"
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
                p.wait()
            th.join(timeout=5)
            os.close(rr)
        ms = (time.time() - t0) * 1000.0
        result["max_test_ms"] = max(result["max_test_ms"], round(ms, 1))
        result["tests_run"] = idx + 1
        res_line = None
        for line in b"".join(chunks).decode("utf-8", "replace").split("\n"):
            if line.startswith("F "):
                try:
                    fl = json.loads(line[2:])
                    flags.setdefault((fl.get("kind"), fl.get("event")), fl)
                except ValueError:
                    pass
            elif line.startswith("R "):
                res_line = line[2:]
        detail = ""
        if kind is None:
            rc = p.returncode
            if rc != 0:
                kind = "runtime_error"
                if rc < 0:
                    detail = "killed by signal %d" % -rc
                    if -rc == signal.SIGKILL:
                        kind = "killed"  # out of memory (cgroup) or the CPU rlimit
                    elif -rc == signal.SIGXCPU:
                        kind = "timeout"
                else:
                    with open(err_p, "rb") as fe:
                        e = fe.read()[-400:].decode("utf-8", "replace").strip().split("\n")
                    detail = e[-1] if e else "exit %d" % rc
            elif mode == "stdin":
                with open(out_p, "rb") as fo:
                    got = fo.read(max_out).decode("utf-8", "replace")
                ok, why = stdio_equal(got, tout)
                if not ok:
                    kind, detail = "wrong_answer", why
            else:
                if res_line is None:
                    kind, detail = "runtime_error", "no return value"
                else:
                    try:
                        ok = as_lists(json.loads(res_line)) == as_lists(json.loads(tout))
                    except ValueError:
                        ok = False
                    if not ok:
                        kind, detail = "wrong_answer", "returned value differs"
        for pth in (out_p, err_p):
            try:
                os.remove(pth)
            except OSError:
                pass
        if kind is not None:
            result["first_fail"] = {"index": idx, "kind": kind, "detail": detail[:300], "public": idx < n_public}
            break
        result["passed"] += 1
    result["flags"] = list(flags.values())
    result["wall_s"] = round(time.time() - t_start, 3)
    print(json.dumps(result))


main()
