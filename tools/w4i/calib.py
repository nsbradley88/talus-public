#!/usr/bin/env python3
# Copyright (c) 2026 nsbradley88
# SPDX-License-Identifier: MIT
"""calib.py LLAMA_CPP_SRC TOKENIZE_BIN GGUF OUTDIR : the W4A4 sidecar calibration set, from PUBLIC sources only.

Sources (all pinned, so the token ids are reproducible):
  * llama.cpp at tag b11062 (commit 3cf03257f219afbe7334045ff7c6a06ac68c627d, MIT): C/C++/CUDA/Python sources
    (code), docs/**/*.md (chat answers), the git history's diffs (synthetic agentic edit transcripts in the
    Qwen tool-call template), concatenated sources (long-context prompts);
  * the Debian/Ubuntu /usr/share/common-licenses texts GPL-3, GFDL-1.3, Apache-2.0, MPL-2.0 (prose).
Plan (149,516 tokens): 20 x 2048 code, 12 x 4096 agentic, 10 x 2048 chat, 2 x 16384 long, 4 x 2048 prose
(= an earlier calibration plan whose sources were partly internal). Files are sorted before the seeded shuffle; the
tokenizer is llama.cpp's llama-tokenize on the target GGUF. Writes OUTDIR/train-<cat><i>.ids (comma-separated ids)
and OUTDIR/SHA256SUMS.
"""
import ast
import glob
import hashlib
import json
import os
import random
import subprocess
import sys

LL, TOK, GGUF, O = sys.argv[1:5]
PIN = '3cf03257f219afbe7334045ff7c6a06ac68c627d'
LIC = '/usr/share/common-licenses'
os.makedirs(O, exist_ok=True)
head = subprocess.run(['git', '-C', LL, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
if head != PIN:
    sys.exit(f'{LL} is at {head}, expected llama.cpp b11062 = {PIN}')
random.seed(20261027)


def rd(p, n=None):
    try:
        s = open(p, errors='replace').read()
    except OSError:
        return ''
    return s if n is None else s[:n]


def tok(text):
    p = os.path.join(O, '_t.txt')
    open(p, 'w').write(text)
    out = subprocess.run([TOK, '-m', GGUF, '-f', p, '--ids', '--log-disable'], capture_output=True, text=True)
    os.remove(p)
    return ast.literal_eval(out.stdout.strip().splitlines()[-1])


def files(pats):
    r = []
    for p in pats:
        r += [f for f in glob.glob(os.path.join(LL, p), recursive=True) if os.path.isfile(f) and os.path.getsize(f) > 2000]
    r = sorted(set(r))
    random.shuffle(r)
    return r


def git_hunks(n):
    log = subprocess.run(['git', '-C', LL, 'log', '-p', '--no-color', '-n', str(n), PIN, '--', '*.cpp', '*.h', '*.py', '*.cu'],
                         capture_output=True, text=True, errors='replace').stdout
    return [c for c in log.split('\ncommit ') if 400 < len(c) < 40000]


TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p, "required": list(p)}}}
         for n, d, p in [
             ("read_file", "Read a file from the workspace.", {"path": {"type": "string"}}),
             ("edit_file", "Replace old_string with new_string in a file.",
              {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}),
             ("bash", "Run a shell command and return its output.", {"command": {"type": "string"}}),
             ("grep", "Search the workspace for a regex.", {"pattern": {"type": "string"}, "path": {"type": "string"}})]]


def sysmsg():
    s = '<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>'
    for t in TOOLS:
        s += '\n' + json.dumps(t)
    s += ('\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n'
          '<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>\n\n'
          'You are a coding agent working in the user\'s repository.<|im_end|>\n')
    return s


def call(name, **args):
    s = f'<tool_call>\n<function={name}>\n'
    for k, v in args.items():
        s += f'<parameter={k}>\n{v}\n</parameter>\n'
    return s + '</function>\n</tool_call>'


def agentic(commit):
    head_, _, diff = commit.partition('\ndiff --git ')
    msg = '\n'.join(l.strip() for l in head_.splitlines()[4:] if l.strip())[:600] or 'fix'
    fl = [l.split(' b/')[-1] for l in ('diff --git ' + diff).splitlines() if l.startswith('diff --git ')]
    s = sysmsg() + f'<|im_start|>user\nPlease implement this change in the repo: {msg}<|im_end|>\n'
    s += (f'<|im_start|>assistant\n<think>\nI should look at the relevant files first: {", ".join(fl[:4])}.\n</think>\n\n'
          + call('grep', pattern=msg.split()[0][:30], path='.') + '<|im_end|>\n')
    s += '<|im_start|>user\n<tool_response>\n' + '\n'.join(f'{f}:{random.randint(1, 900)}: ...' for f in fl[:6]) + '\n</tool_response><|im_end|>\n'
    for f in fl[:2]:
        s += '<|im_start|>assistant\n<think>\n\n</think>\n\n' + call('read_file', path=f) + '<|im_end|>\n'
        s += '<|im_start|>user\n<tool_response>\n' + rd(os.path.join(LL, f), 9000) + '\n</tool_response><|im_end|>\n'
    for h in diff.split('\n@@')[1:5]:
        old = '\n'.join(l[1:] for l in h.splitlines()[1:] if l.startswith((' ', '-')))[:1500]
        new = '\n'.join(l[1:] for l in h.splitlines()[1:] if l.startswith((' ', '+')))[:1500]
        s += ('<|im_start|>assistant\n<think>\nNow apply the edit.\n</think>\n\n'
              + call('edit_file', path=fl[0] if fl else 'x', old_string=old, new_string=new) + '<|im_end|>\n')
        s += '<|im_start|>user\n<tool_response>\nEdit applied.\n</tool_response><|im_end|>\n'
    s += '<|im_start|>assistant\n<think>\n\n</think>\n\n' + call('bash', command='cmake --build build -j 16 && ctest --output-on-failure') + '<|im_end|>\n'
    s += ('<|im_start|>user\n<tool_response>\n[1/4] Building CXX object\n[4/4] Linking\n100% tests passed, 0 tests failed out of 37\n'
          f'</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nDone. {msg}\n\nChanged files: {", ".join(fl)}.<|im_end|>\n')
    return s


def chat(doc):
    title = doc.strip().splitlines()[0][:120] if doc.strip() else 'this'
    return (f'<|im_start|>user\nCan you explain "{title}" in detail, with the key numbers?<|im_end|>\n<|im_start|>assistant\n<think>\n'
            f'The user wants an explanation of {title}. Let me summarize carefully.\n</think>\n\n' + doc + '<|im_end|>\n')


def emit(name, text, n):
    ids = tok(text)
    if len(ids) < n:
        print('short', name, len(ids), file=sys.stderr)
        n = len(ids)
    open(os.path.join(O, f'{name}.ids'), 'w').write(','.join(map(str, ids[:n])))
    return n


code = files(['src/**/*.cpp', 'tools/**/*.cpp', 'ggml/src/ggml-cuda/**/*.cu', 'common/**/*.cpp', 'gguf-py/**/*.py'])
docs = files(['docs/**/*.md', 'tools/**/README.md'])
hunks = git_hunks(400)
random.shuffle(hunks)
lic = [os.path.join(LIC, n) for n in ('GPL-3', 'GFDL-1.3', 'Apache-2.0', 'MPL-2.0')]
plan = [('code', 20, 2048), ('agent', 12, 4096), ('chat', 10, 2048), ('long', 2, 16384), ('prose', 4, 2048)]
ci = di = hi = 0
tot = 0
for cat, cnt, n in plan:
    for i in range(cnt):
        if cat == 'code':
            t = ''
            while len(t) < n * 5:
                t += f'// file: {os.path.relpath(code[ci], LL)}\n' + rd(code[ci], 30000) + '\n'
                ci += 1
        elif cat == 'agent':
            t = ''
            while len(t) < n * 4:
                t += agentic(hunks[hi])
                hi += 1
        elif cat == 'chat':
            t = ''
            while len(t) < n * 5:
                t += chat(rd(docs[di], 12000))
                di += 1
        elif cat == 'long':
            t = sysmsg() + '<|im_start|>user\nHere is the repository content:\n\n'
            while len(t) < n * 4.5:
                t += f'=== {os.path.relpath(code[ci], LL)} ===\n' + rd(code[ci], 40000) + '\n'
                ci += 1
            t += ('\nWhere is the bug that makes the second request hang? Explain and propose a fix.<|im_end|>\n'
                  '<|im_start|>assistant\n<think>\n')
        else:
            t = rd(lic[i])
        tot += emit(f'train-{cat}{i}', t, n)
with open(os.path.join(O, 'SHA256SUMS'), 'w') as f:
    for p in sorted(glob.glob(os.path.join(O, 'train-*.ids'))):
        f.write(f'{hashlib.sha256(open(p, "rb").read()).hexdigest()}  {os.path.basename(p)}\n')
print('train', tot, 'tokens ->', O)
