#!/usr/bin/env python3
"""GLM-5.3-Flash speed bench against Talus HTTP server: prompt + decode tok/s at several depths.
Each depth request is streamed (post_ttft) so its run also carries ttft_ms (client: request sent -> first token) and
ttft_srv_ms / cache_n (the server's timings); after the depths, one warm follow-up (the last depth's prompt + its
output + ~300 new tokens, cache allowed) goes into the record's "warm" field (not into "runs": no matrix metric moves).
Env TTFT_STREAM=0: the old non-streamed requests (no client TTFT); TTFT_WARM=0: no follow-up.
Multi-prompt depth cells (multi-prompt aggregate): env DEPTH_OFFSETS="0,100000,200000,300000" (corpus token offsets) adds,
AFTER the unchanged legacy sequence (offset 0 at every depth, then the warm follow-up), one more greedy NPRED-token sample
per depth at each non-zero offset (prompt = corpus tokens [off, off + depth) + the same "Continue" suffix). "runs" keeps
only the offset-0 samples (the historical single-prompt cells, TTFT a/b); "mp_runs" holds every sample (with "offset")
and "mp" the per-depth aggregate (tg_tps = tokens over time across the prompts). Unset or "0" = the old behaviour."""
import glob, json, os, subprocess, sys, time, urllib.request

URL = os.environ.get("TALUS_URL", os.environ.get("GLM_URL", "http://127.0.0.1:8000"))
def _load_api_key():
    k = os.environ.get("STRATA_API_KEY") or os.environ.get("TALUS_API_KEY") or ""
    if k:
        return k.strip()
    kf = os.environ.get("TALUS_API_KEY_FILE", "")
    if kf and os.path.exists(os.path.expanduser(kf)):
        return open(os.path.expanduser(kf)).read().strip()
    return "none"
KEY = _load_api_key()
DEPTHS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2000,8000,32000").split(",")]
LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
NPRED = int(os.environ.get("NPRED", "128"))


def depth_offsets():
    """the non-zero corpus offsets of env DEPTH_OFFSETS (offset 0 is always the legacy sample)"""
    v = [int(x) for x in os.environ.get("DEPTH_OFFSETS", "0").replace(" ", "").split(",") if x]
    return [o for o in dict.fromkeys(v) if o > 0]


def mp_aggregate(samples):
    """per nominal depth: tokens-over-time decode tok/s across the prompts (sum tg_n / sum decode seconds), the same for
    prompt tok/s, plus the per-prompt values. samples: run dicts with depth_nominal, offset, tg_tps, tg_n, pp_tps, depth"""
    by = {}
    for x in samples:
        by.setdefault(x["depth_nominal"], []).append(x)
    out = []
    for d, xs in by.items():
        ok = [x for x in xs if x.get("tg_tps") and x.get("tg_n")]
        if not ok: continue
        tn = sum(x["tg_n"] for x in ok); ts = sum(x["tg_n"] / x["tg_tps"] for x in ok)
        pp = [x for x in ok if x.get("pp_tps") and x.get("depth")]
        pn = sum(x["depth"] for x in pp); psec = sum(x["depth"] / x["pp_tps"] for x in pp)
        dn = [x.get("draft_n") for x in ok]; da = [x.get("draft_accepted") for x in ok]
        out.append({"depth_nominal": d, "n_prompts": len(ok), "offsets": [x["offset"] for x in ok],
                    "depth": round(sum(x["depth"] for x in ok) / len(ok)), "tg_tps": round(tn / ts, 2), "tg_n": tn,
                    "pp_tps": round(pn / psec, 1) if psec else None,
                    "draft_n": sum(dn) if None not in dn else None, "draft_accepted": sum(da) if None not in da else None,
                    "tg_tps_each": [x["tg_tps"] for x in ok]})
    return out

def corpus():
    # deterministic mix of real code and prose, same every run
    fs = sorted(glob.glob("/usr/lib/python3.*/[a-m]*.py"))
    return "\n\n".join(open(f, errors="ignore").read() for f in fs)

def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

STREAM = os.environ.get("TTFT_STREAM", "1") != "0"
_NO_STREAM_TIMINGS = set()      # paths whose streamed answer came without `timings`: they go back to plain post()


def post_ttft(path, body, timeout=1800):
    """post() with "stream": true, for a true time to first token. Returns the same shape as the non-streamed answer
    (choices[0].message content / reasoning_content / tool_calls + finish_reason for chat; content / stop for
    /completion; timings; usage) plus "_ttft": {ttft_ms, ttft_srv_ms (timings.prompt_ms), prompt_n, cache_n, streamed}.
    The first token is the first delta with text: content, reasoning (thinking counts) or a tool call.
    A server that answers a stream request with plain JSON (Talus's /completion) is read as JSON (no client ttft_ms).
    A server whose stream has no `timings` gets this request again without streaming (once per path per process;
    the drivers read the speed from `timings`)."""
    if not STREAM or path in _NO_STREAM_TIMINGS:
        r = post(path, body, timeout)
        r["_ttft"] = _ttft_fields(r.get("timings"), None, False)
        return r
    sb = dict(body, stream=True)
    chat = "chat/completions" in path
    if chat:
        sb.setdefault("stream_options", {"include_usage": True})
    req = urllib.request.Request(URL + path, data=json.dumps(sb).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json", "Accept": "text/event-stream"})
    t0 = time.perf_counter(); first = None
    content, reasoning, tools, final = [], [], {}, {}
    finish = timings = usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if "text/event-stream" not in (resp.headers.get("Content-Type") or ""):
            r = json.load(resp)                                       # the server ignored "stream"
            r["_ttft"] = _ttft_fields(r.get("timings"), None, False)
            return r
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                ch = json.loads(data)
            except Exception:
                continue
            if ch.get("error"):
                raise RuntimeError(f"stream error: {ch['error']}")
            timings = ch.get("timings") or timings
            usage = ch.get("usage") or usage
            if chat:
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    t, rs = d.get("content") or "", d.get("reasoning_content") or ""
                    if first is None and (t or rs or d.get("tool_calls")):
                        first = time.perf_counter() - t0
                    content.append(t); reasoning.append(rs)
                    for tc in d.get("tool_calls") or []:
                        cur = tools.setdefault(tc.get("index", 0), {"type": "function", "function": {"name": "", "arguments": ""}})
                        if tc.get("id"): cur["id"] = tc["id"]
                        f = tc.get("function") or {}
                        cur["function"]["name"] += f.get("name") or ""; cur["function"]["arguments"] += f.get("arguments") or ""
                    finish = c.get("finish_reason") or finish
            else:
                t = ch.get("content") or ""
                if first is None and t:
                    first = time.perf_counter() - t0
                content.append(t)
                final = ch
    if timings is None:
        _NO_STREAM_TIMINGS.add(path)
        return post_ttft(path, body, timeout)
    if chat:
        msg = {"role": "assistant", "content": "".join(content)}
        if any(reasoning): msg["reasoning_content"] = "".join(reasoning)
        if tools: msg["tool_calls"] = [tools[k] for k in sorted(tools)]
        r = {"choices": [{"index": 0, "message": msg, "finish_reason": finish}], "timings": timings}
        if usage: r["usage"] = usage
    else:
        r = dict(final, content="".join(content), timings=timings)
    r["_ttft"] = _ttft_fields(timings, None if first is None else first * 1000, True)
    return r


def _ttft_fields(t, ttft_ms, streamed):
    t = t or {}
    return {"ttft_ms": None if ttft_ms is None else round(ttft_ms, 1),
            "ttft_srv_ms": None if t.get("prompt_ms") is None else round(t["prompt_ms"], 1),
            "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "streamed": streamed}


def ttft_stats(v):
    """median / p95 (nearest rank) of a list of ms values, None-safe"""
    v = sorted(x for x in v if x is not None)
    if not v: return None, None
    n = len(v); med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    return round(med, 1), round(v[max(0, -(-95 * n // 100) - 1)], 1)


def memavail():
    try:
        ssh_host = os.environ.get("TALUS_SSH_HOST", "")
        if ssh_host:
            out = subprocess.run(["ssh", "-o", "BatchMode=yes", ssh_host,
                "awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo"], capture_output=True, text=True, timeout=20)
            return int(out.stdout.strip())
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1048576
        return None
    except Exception:
        return None

if __name__ == "__main__":
    text = corpus()
    toks = post("/tokenize", {"content": text})["tokens"]
    res = {"label": LABEL, "time": time.strftime("%F %T"), "mem_avail_gib_idle": memavail(), "runs": []}
    SUFFIX = "\n\nContinue the code above:\n"

    def sample(d, off=0):
        prompt = post("/detokenize", {"tokens": toks[off:off + d]})["content"] + SUFFIX
        r = post_ttft("/completion", {"prompt": prompt, "n_predict": NPRED, "temperature": 0, "cache_prompt": False,
                                      "ignore_eos": True})
        t = r["timings"]
        run = {"depth": t["prompt_n"], "pp_tps": round(t["prompt_per_second"], 1), "tg_tps": round(t["predicted_per_second"], 2),
               "tg_n": t["predicted_n"], "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"),
               "mem_avail_gib": memavail(), "ttft_ms": r["_ttft"]["ttft_ms"], "ttft_srv_ms": r["_ttft"]["ttft_srv_ms"],
               "cache_n": r["_ttft"]["cache_n"]}
        return prompt, r, run

    for d in DEPTHS:   # the legacy sequence: offset 0 at every depth (historical single-prompt cells)
        prompt, r, run = sample(d)
        last = (prompt, r.get("content") or "")
        print(json.dumps(run), flush=True)
        res["runs"].append(run)
    if os.environ.get("TTFT_WARM", "1") != "0" and DEPTHS:
        try:   # warm follow-up: the last depth's prompt + its output + ~300 new tokens, prompt cache allowed
            new = post("/detokenize", {"tokens": toks[-(300 + 50000):-50000]})["content"]
            r = post_ttft("/completion", {"prompt": last[0] + last[1] + "\n\n# Another file:\n" + new +
                                          "\n\nIn one sentence, what does the second file do?\n", "n_predict": 8,
                                          "temperature": 0})
            t = r["timings"]; res["warm"] = {"depth": (t.get("prompt_n") or 0) + (t.get("cache_n") or 0) or None,
                                             "new_tokens": 300, **{k: v for k, v in r["_ttft"].items() if k != "streamed"}}
            print(json.dumps({"warm": res["warm"]}), flush=True)
        except Exception as e:
            res["warm"] = {"error": str(e)[:200]}
    offs = depth_offsets()
    if offs and DEPTHS:   # multi-prompt cells: the extra offsets after the legacy sequence (history stays comparable)
        mp = [dict(x, offset=0, depth_nominal=d) for d, x in zip(DEPTHS, res["runs"])]
        try:
            for d in DEPTHS:
                for o in offs:
                    if o + d > len(toks):
                        print(json.dumps({"mp_skip": {"depth": d, "offset": o, "corpus_tokens": len(toks)}}), flush=True)
                        continue
                    _, _, run = sample(d, o)
                    run.update(offset=o, depth_nominal=d)
                    print(json.dumps(run), flush=True)
                    mp.append(run)
        except Exception as e:   # keep what was measured; compare.py only uses complete aggregates
            res["mp_error"] = str(e)[:200]
        res["mp_runs"] = mp
        res["mp"] = {"offsets": [0] + offs, "agg": mp_aggregate(mp)}
        print(json.dumps({"mp": res["mp"]}), flush=True)
    with open(os.environ.get("TALUS_RESULTS_JSONL", "results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
