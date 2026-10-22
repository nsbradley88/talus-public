#!/usr/bin/env python3
"""Prompt/decode at depth through /v1/chat/completions only (for servers without /tokenize or /completion).
The same corpus as bench.py, cut to ~N tokens by characters (ratio calibrated with one short request), then
"Continue the code above", greedy, 128 tokens. usage: chatdepth.py LABEL 2000,32000
Streamed (bench.post_ttft): each run carries ttft_ms / ttft_srv_ms / cache_n; then one warm follow-up (the last depth's
conversation + its answer + ~300 new tokens) goes into the record's "warm" field (TTFT_WARM=0: none).
Env DEPTH_OFFSETS (as in bench.py): after the legacy sequence and the warm follow-up, one more sample per depth at each
non-zero corpus offset (token offset x the calibrated chars/token ratio); "mp_runs" / "mp" as in bench.py."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv, args = sys.argv[:1], sys.argv[1:]
from bench import post, post_ttft, corpus, memavail, depth_offsets, mp_aggregate
label, depths = args[0], [int(x) for x in args[1].split(",")]
text = corpus()
def run(nchars, ntok=128, msgs=None, cold=True, start=0):
    msgs = msgs or [{"role": "user", "content": text[start:start + nchars] + "\n\nContinue the code above:\n"}]
    body = {"messages": msgs, "max_tokens": ntok, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    if cold: body["cache_prompt"] = False
    r = post_ttft("/v1/chat/completions", body)
    global last; last = (msgs, r["choices"][0]["message"].get("content") or "", r["_ttft"])
    return r["timings"]
cal = run(20000, 1); ratio = 20000 / max(1, cal["prompt_n"])
res = {"label": label, "time": time.strftime("%F %T"), "via": "chat", "runs": []}
def rec(t):
    return {"depth": t["prompt_n"], "pp_tps": t.get("prompt_per_second"), "tg_tps": t.get("predicted_per_second"),
         "tg_n": t.get("predicted_n"), "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"), "mem_avail_gib": memavail(),
         "ttft_ms": last[2]["ttft_ms"], "ttft_srv_ms": last[2]["ttft_srv_ms"], "cache_n": last[2]["cache_n"]}
for d in depths:   # the legacy sequence (historical single-prompt cells)
    r = rec(run(int(d * ratio)))
    print(json.dumps(r), flush=True); res["runs"].append(r)
if os.environ.get("TTFT_WARM", "1") != "0" and depths:
    try:   # warm follow-up: the last depth's conversation + its answer + ~300 new tokens, cache allowed
        msgs, ans, _ = last
        new = text[-int(300 * ratio) - 50000:-50000]
        t = run(0, 8, msgs + [{"role": "assistant", "content": ans},
                              {"role": "user", "content": f"Another file:\n{new}\n\nIn one sentence, what does it do?"}], cold=False)
        res["warm"] = {"depth": (t.get("prompt_n") or 0) + (t.get("cache_n") or 0) or None, "new_tokens": 300,
                       **{k: v for k, v in last[2].items() if k != "streamed"}}
        print(json.dumps({"warm": res["warm"]}), flush=True)
    except Exception as e:
        res["warm"] = {"error": str(e)[:200]}
offs = depth_offsets()
if offs and depths:   # multi-prompt cells after the legacy sequence and the warm follow-up
    mp = [dict(x, offset=0, depth_nominal=d) for d, x in zip(depths, res["runs"])]
    try:
        for d in depths:
            for o in offs:
                st = int(o * ratio)
                if st + int(d * ratio) > len(text):
                    print(json.dumps({"mp_skip": {"depth": d, "offset": o, "corpus_chars": len(text)}}), flush=True); continue
                r = dict(rec(run(int(d * ratio), start=st)), offset=o, depth_nominal=d)
                print(json.dumps(r), flush=True); mp.append(r)
    except Exception as e:
        res["mp_error"] = str(e)[:200]
    res["mp_runs"] = mp; res["mp"] = {"offsets": [0] + offs, "agg": mp_aggregate(mp)}
    print(json.dumps({"mp": res["mp"]}), flush=True)
open(os.environ.get("TALUS_RESULTS_JSONL", "results.jsonl"), "a").write(json.dumps(res) + "\n")
