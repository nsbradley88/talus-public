#!/usr/bin/env python3
"""Qwen3.8-27B speed bench against the Talus server (:8000). Two mixes, aggregated decode tok/s + draft acceptance:
  chat: 6 mixed prompts, temperature 1.0 / top_p 0.95, ~600 tokens each (realistic chat)
  code: 6 coding prompts, greedy (temperature 0), thinking off, ~700 tokens (comparable to community MTP numbers)
usage: bench27.py LABEL [chat,code]
env DUMP=FILE appends each response (label, mix, index, text) as JSON lines, to compare greedy outputs.
Requests are streamed (bench.post_ttft): each mix record also carries ttft_med_ms / ttft_p95_ms / ttft_srv_med_ms
(request -> first token, thinking counts; TTFT_STREAM=0 turns streaming off)."""
import json, os, sys, time
mixes = (sys.argv[2] if len(sys.argv) > 2 else "chat,code").split(",")
sys.argv = sys.argv[:2]
from bench import post_ttft, memavail, ttft_stats
CHAT = [
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H12M' into a timedelta, with tests.",
    "Explain the difference between optimistic and pessimistic locking in databases, with a concrete example of each.",
    "Write a bash script that finds the 10 largest files under a directory, excluding .git, and prints human-readable sizes.",
    "Refactor this JavaScript into async/await and explain the changes:\nfunction load(u,cb){fetch(u).then(r=>r.json()).then(d=>cb(null,d)).catch(e=>cb(e))}",
    "Write a Go HTTP handler that accepts a JSON body {\"items\":[...]} and returns the count and the sum of numeric items, with error handling.",
    "Summarize the CAP theorem and then argue which two properties a shopping-cart service should prioritize.",
]
CODE = [
    "Write a Python class LRUCache with get/put in O(1) using a dict and a doubly linked list, with docstrings and pytest tests.",
    "Implement a Go function that merges overlapping intervals ([][2]int), sorts by start, and include table-driven tests.",
    "Write a TypeScript function debounce<T extends (...args:any[])=>void>(fn:T, ms:number) with cancel() and flush(), plus Jest tests.",
    "Write a Rust function that parses a CSV line with quoted fields (commas and escaped quotes inside quotes) into Vec<String>, with unit tests.",
    "Refactor this Python into a dataclass-based design with type hints and explain each change:\nclass P:\n def __init__(s,n,a): s.n=n; s.a=a\n def older(s,o): return s.a>o.a\n def __repr__(s): return 'P('+s.n+','+str(s.a)+')'",
    "Write a SQL schema for orders, order_items and products, then a query returning the top 5 products by revenue in the last 30 days, with indexes.",
]

def _edit_tasks():
    import glob
    ed = os.environ.get("TALUS_EDITSRC_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "editsrc"))
    files = [os.path.join(ed, "sampling.cpp"), os.path.join(ed, "speculative.cpp"), os.path.join(ed, "llama-batch.cpp")]
    asks = ["Add a short doc comment above every function that lacks one. Output the complete updated file, nothing else.",
            "Rename every local variable named `i` to `idx` and every `n` to `count`. Output the complete updated file, nothing else.",
            "Replace every use of `fprintf(stderr, ...)` or LOG_* macro message text so it starts with the function name. Output the complete updated file, nothing else."]
    out = []
    for f, a in zip(files, asks):
        src = open(f).read()[:9000]
        out.append(f"Here is a C++ source file:\n```cpp\n{src}\n```\n{a}")
    return out
EDIT = _edit_tasks()
label = sys.argv[1]
for mix in mixes:
    tasks = CHAT if mix == "chat" else (EDIT if mix == "edit" else CODE)
    tn = tt = dn = da = 0; tf = []; tfs = []
    for i, t in enumerate(tasks):
        body = {"messages": [{"role": "user", "content": t}], "max_tokens": 700 if mix != "edit" else 1500, "seed": i}
        if mix == "chat": body.update({"temperature": 1.0, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": False}})
        else: body.update({"temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}})
        r = post_ttft("/v1/chat/completions", body); tm = r["timings"]; tf.append(r["_ttft"]["ttft_ms"]); tfs.append(r["_ttft"]["ttft_srv_ms"])
        if os.environ.get("DUMP"):
            open(os.environ["DUMP"], "a").write(json.dumps({"label": label, "mix": mix, "i": i,
                "text": r["choices"][0]["message"].get("content"),
                "reasoning": r["choices"][0]["message"].get("reasoning_content")}) + "\n")
        tn += tm["predicted_n"]; tt += tm["predicted_ms"]; dn += tm.get("draft_n") or 0; da += tm.get("draft_n_accepted") or 0
    rec = {"label": label, "test": mix, "tg_tps": round(tn / tt * 1000, 2), "tokens": tn,
           "accept": round(da / dn, 3) if dn else None, "mem": memavail(), "time": time.strftime("%F %T")}
    rec["ttft_med_ms"], rec["ttft_p95_ms"] = ttft_stats(tf); rec["ttft_srv_med_ms"] = ttft_stats(tfs)[0]
    rec["ttft_n"] = sum(x is not None for x in tf)
    print(json.dumps(rec), flush=True)
    open(os.environ.get("TALUS_RESULTS_JSONL", "results.jsonl"), "a").write(json.dumps(rec) + "\n")
