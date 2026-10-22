#!/usr/bin/env python3
"""Long-context recall (port of strata's bench.needle_test): 5 codenames hidden at 10-90% depth of an N-token prompt.
usage: needle.py LABEL N_TOKENS [seeds]"""
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import post, post_ttft, corpus, memavail  # noqa: E402  (bench.py runs nothing at import when guarded)

WORDS = ["amber", "basalt", "cobalt", "delta", "ember", "falcon", "garnet", "harbor", "indigo", "juniper", "kestrel",
         "lumen", "marble", "nimbus", "onyx", "pebble", "quartz", "raven", "sable", "tundra"]
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
label, n = sys.argv[1], int(sys.argv[2])
seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3").split(",")]
toks = post("/tokenize", {"content": corpus()})["tokens"]
text = post("/detokenize", {"tokens": toks[:n]})["content"]
for seed in seeds:
    rng = random.Random(seed)
    modules = rng.sample(["billing", "ingest", "scheduler", "auth", "search", "mailer", "cache", "export"], 5)
    codes = {m: f"{rng.choice(WORDS)}-{rng.randint(1000, 9999)}" for m in modules}
    doc = text
    for m, frac in sorted(zip(modules, DEPTHS), key=lambda x: -x[1]):
        pos = doc.rfind("\n", 0, int(len(doc) * frac)) + 1
        doc = doc[:pos] + f"# NOTE: the deployment codename for the {m} module is {codes[m]}.\n" + doc[pos:]
    q = ("Somewhere in the code above are comments giving the deployment codename for these modules: "
         + ", ".join(modules) + ". List each module with its codename, one per line as `module: codename`.")
    t0 = time.time()
    r = post_ttft("/v1/chat/completions", {"messages": [{"role": "user", "content": f"{doc}\n\n{q}"}], "max_tokens": 4000,
                                      "seed": seed, "temperature": 1.0, "top_p": 0.95,
                                      "chat_template_kwargs": {"reasoning_effort": "low"}}, timeout=7200)
    ans = r["choices"][0]["message"].get("content") or ""
    hits = {m: codes[m] in ans for m in modules}
    rec = {"label": label, "test": "needle", "n": n, "seed": seed, "score": sum(hits.values()), "of": 5,
           "missed": [f"{m}@{d}" for m, d in zip(modules, DEPTHS) if not hits[m]], "pp_n": r["timings"]["prompt_n"],
           "pp_tps": round(r["timings"]["prompt_per_second"], 1), "wall_s": round(time.time() - t0), "mem": memavail(),
           "ttft_ms": r["_ttft"]["ttft_ms"], "ttft_srv_ms": r["_ttft"]["ttft_srv_ms"], "cache_n": r["_ttft"]["cache_n"]}
    print(json.dumps(rec), flush=True)
    open(os.environ.get("TALUS_NEEDLE_JSONL", "needle.jsonl"), "a").write(json.dumps(rec) + "\n")
