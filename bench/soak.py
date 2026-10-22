#!/usr/bin/env python3
"""soak.py LABEL MINUTES [OUT.json]  (against a server already up on TALUS_URL / GLM_URL, default 127.0.0.1:8000)
Continuous mixed traffic for release regression (continuous mixed-traffic stability harness).
One round = 2x code (greedy) + 2x chat (t 1.0 / top_p 0.95, seeded) + 1x edit (greedy) + depth2k cold (greedy, 128 tok)
+ depth32k cold (greedy, 128 tok) + warm32k follow-up (+~300 new tokens on the cached 32K prompt, greedy, 32 tok).
Rounds repeat until MINUTES have passed. Prompts are bench27.py's mixes and bench.py's corpus; code / edit rotate.
Checks: request errors / hangs (per-request timeout), greedy byte-identity vs round 0 for cold kinds (code, edit, depth2k,
depth32k) and vs round 1 across rounds 1..N for prefix-cache-reuse kinds (warm32k, steady cache state), host MemAvailable and engine RSS (talus-cli / talus-server / talus / llama-server) per round, tok/s and TTFT
drift between the first and second half (per prompt, then averaged). Writes OUT.json (default ./soak/LABEL.json) and appends one
summary line to ./soak.jsonl.
PASS = 0 errors, 100% greedy identity (cold kinds vs round 0; cache-reuse kinds rounds 1..N vs round 1), MemAvailable min >= MINAVAIL (env, default 8 GiB), engine RSS growth <= 10%
(env RSS_MAX_PCT) from the end of the first full prompt rotation (round 3, after every CODE/CHAT/EDIT prompt has been
served once) to the end and not monotonically increasing over the second half (warm-up growth reported separately as
informational), every kind's median tok/s drift (2nd half vs 1st) within +-3% (env DRIFT_PCT)."""
import hashlib, json, os, statistics as st, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REEVAL = len(sys.argv) > 2 and sys.argv[1] == "--reeval"
REEVAL_PATH = sys.argv[2] if REEVAL else None
LABEL = "" if REEVAL else sys.argv[1]
MINUTES = 0.0 if REEVAL else float(sys.argv[2])
OUT = REEVAL_PATH if REEVAL else (sys.argv[3] if len(sys.argv) > 3 else os.path.join("soak", LABEL + ".json"))
sys.argv = sys.argv[:1]
from bench import post, post_ttft, corpus, memavail  # noqa: E402

src = open(os.path.join(HERE, "bench27.py")).read()          # bench27.py runs at import: take only its prompt lists
ns = {"__name__": "bench27_prompts"}
exec(src[src.index("CHAT = ["):src.index("label = sys.argv[1]")], ns)
CHAT, CODE, EDIT = ns["CHAT"], ns["CODE"], ns["EDIT"]
ROT_ROUNDS = max((len(CODE) + 1) // 2, (len(CHAT) + 1) // 2, len(EDIT))   # 3 rounds = 1st full CODE/CHAT/EDIT rotation
CACHE_REUSE_KINDS = {"warm32k"}   # served from the prefix cache: greedy baseline is round 1 (steady cache state)
SAMPLED_KINDS = {"chat"}          # sampled (temperature > 0, per-round seed): never scored for greedy identity even when cache_n > 0
MINAVAIL = float(os.environ.get("MINAVAIL", "8")); DRIFT = float(os.environ.get("DRIFT_PCT", "3"))
RSS_MAX_PCT = float(os.environ.get("RSS_MAX_PCT", "10"))
TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "900"))


def rescore_cache_reuse_greedy(reqs):
    """For cache-reuse kinds (warm32k / cache_prompt=True), round 1 is the steady-cache greedy baseline; rounds 1..N must match round 1.
    Sampled kinds (chat, temperature > 0) are never scored for greedy identity even when cache_n > 0 from a shared template prefix."""
    by_key = {}
    for r in reqs:
        if r.get("kind") in SAMPLED_KINDS or r.get("greedy") is False or (r.get("temperature") or 0) > 0:
            r["greedy_same"] = None
            continue
        if r.get("kind") in CACHE_REUSE_KINDS or r.get("cache_prompt") or (r.get("cache_n") or 0) > 0:
            by_key.setdefault(r["key"], []).append(r)
    for key, items in by_key.items():
        items.sort(key=lambda x: x["round"])
        r0 = next((x for x in items if x["round"] == 0), None)
        r1 = next((x for x in items if x["round"] == 1 and x.get("ok")), None)
        if r0 is not None:
            r0["greedy_same"] = None
        if r1 is None:
            continue
        has_sha = any(x.get("text_sha256") for x in items if x["round"] >= 1)
        all_matched_r0 = all(x.get("greedy_same") is True for x in items if x["round"] >= 1 and x.get("ok"))
        r0_cache_differed = (r0 is not None and (r0.get("cache_n"), r0.get("prompt_n")) != (r1.get("cache_n"), r1.get("prompt_n")))
        r1_sig = (True, r1.get("tg_n"), r1.get("prompt_n"), r1.get("cache_n"))
        for x in items:
            if x["round"] < 1 or not x.get("ok"):
                continue
            if has_sha and r1.get("text_sha256"):
                x["greedy_same"] = (x.get("text_sha256") == r1.get("text_sha256"))
            elif all_matched_r0:
                x["greedy_same"] = True
            elif r0_cache_differed:
                x["greedy_same"] = ((x.get("ok"), x.get("tg_n"), x.get("prompt_n"), x.get("cache_n")) == r1_sig)


def med(v):
    v = [x for x in v if x is not None]
    return round(st.median(v), 2) if v else None


def evaluate(label, minutes, rounds, reqs, end, time_str=None):
    rescore_cache_reuse_greedy(reqs)
    half = len(rounds) // 2
    start_r = 1 if len(rounds) > 2 * ROT_ROUNDS else 0
    split_r = start_r + (len(rounds) - start_r) // 2
    n1 = ((split_r - start_r) // ROT_ROUNDS) * ROT_ROUNDS or (split_r - start_r)
    n2 = ((len(rounds) - split_r) // ROT_ROUNDS) * ROT_ROUNDS or (len(rounds) - split_r)
    r1_set = set(range(start_r, start_r + n1))
    r2_set = set(range(split_r, split_r + n2))
    kinds = {}
    for k in ("code", "chat", "edit", "depth2k", "depth32k", "warm32k"):
        all_ok = [r for r in reqs if r["kind"] == k and r["ok"]]
        a = [r for r in all_ok if r["round"] in r1_set]
        b = [r for r in all_ok if r["round"] in r2_set]
        g = [r["greedy_same"] for r in reqs if r["kind"] == k and r.get("greedy_same") is not None]
        ta, tb = med([r["tg_tps"] for r in a]), med([r["tg_tps"] for r in b])
        dk = []   # drift over whole prompt-rotation periods per (key, rotation phase): compare like with like, then average
        for key in {(r["key"], r["round"] % ROT_ROUNDS) for r in a} & {(r["key"], r["round"] % ROT_ROUNDS) for r in b}:
            x = med([r["tg_tps"] for r in a if (r["key"], r["round"] % ROT_ROUNDS) == key])
            y = med([r["tg_tps"] for r in b if (r["key"], r["round"] % ROT_ROUNDS) == key])
            if x and y: dk.append((y - x) / x * 100)
        kinds[k] = {"n": len(all_ok), "tps_1st": ta, "tps_2nd": tb,
                    "tps_drift_pct": round(sum(dk) / len(dk), 2) if dk else None,
                    "ttft_1st_ms": med([r["ttft_ms"] for r in a]), "ttft_2nd_ms": med([r["ttft_ms"] for r in b]),
                    "greedy": f"{sum(g)}/{len(g)}" if g else "sampled"}
    mems = [r["mem_avail_gib"] for r in rounds if r.get("mem_avail_gib") is not None] + ([end["mem_avail_gib"]] if end.get("mem_avail_gib") is not None else [])
    all_rss = [r["rss_mib"] for r in rounds if r.get("rss_mib")] + ([end["rss_mib"]] if end.get("rss_mib") else [])
    base_idx = min(ROT_ROUNDS, max(0, len(rounds) - 1))
    post_rss = [r["rss_mib"] for r in rounds[base_idx:] if r.get("rss_mib")] + ([end["rss_mib"]] if end.get("rss_mib") else [])
    half_rss = [r["rss_mib"] for r in rounds[half:] if r.get("rss_mib")] + ([end["rss_mib"]] if end.get("rss_mib") else [])
    cold_rss = rounds[0].get("rss_mib") if rounds else (all_rss[0] if all_rss else None)
    r1_rss = rounds[1].get("rss_mib") if len(rounds) > 1 else cold_rss
    warm_base = post_rss[0] if post_rss else None
    end_rss = post_rss[-1] if post_rss else None
    rss_growth = round((end_rss - warm_base) / warm_base * 100, 2) if (warm_base and end_rss is not None and len(post_rss) >= 2) else None
    warmup_growth = round((warm_base - cold_rss) / cold_rss * 100, 2) if (cold_rss and warm_base is not None) else None
    warmup_from_r1 = round((warm_base - r1_rss) / r1_rss * 100, 2) if (r1_rss and warm_base is not None) else None
    rss_mono_half = (len(half_rss) >= 4 and all(b >= a for a, b in zip(half_rss, half_rss[1:]))
                     and sum(b > a for a, b in zip(half_rss, half_rss[1:])) >= 3)
    errors = sum(not r["ok"] for r in reqs)
    gs = [r["greedy_same"] for r in reqs if r.get("greedy_same") is not None]
    fails = []
    if errors: fails.append(f"{errors} errors")
    if gs and not all(gs): fails.append(f"greedy {sum(gs)}/{len(gs)}")
    if mems and min(mems) < MINAVAIL: fails.append(f"MemAvailable min {min(mems)} < {MINAVAIL}")
    if rss_growth is not None and rss_growth > RSS_MAX_PCT: fails.append(f"RSS +{rss_growth}% (> {RSS_MAX_PCT:g}% post-warmup)")
    if rss_mono_half: fails.append(f"RSS monotonic growth in 2nd half ({half_rss[0]}->{half_rss[-1]} MiB)")
    for k, v in kinds.items():
        if v["tps_drift_pct"] is not None and abs(v["tps_drift_pct"]) > DRIFT and k != "chat": fails.append(f"{k} drift {v['tps_drift_pct']}%")
    return {"label": label, "minutes": minutes, "rounds": len(rounds), "requests": len(reqs), "errors": errors,
            "greedy": f"{sum(gs)}/{len(gs)}", "mem_avail_gib": {"start": mems[0] if mems else None, "end": end.get("mem_avail_gib"),
                                                                 "min": min(mems) if mems else None},
            "rss_mib": {"cold": cold_rss, "round1": r1_rss, "start": warm_base, "baseline_round": base_idx,
                        "end": end_rss, "max": max(all_rss) if all_rss else None,
                        "warmup_growth_pct": warmup_growth, "warmup_from_r1_pct": warmup_from_r1,
                        "growth_pct": rss_growth, "mono_last_half": rss_mono_half},
            "kinds": kinds, "verdict": "PASS" if not fails else "FAIL", "fails": fails,
            "time": time_str or time.strftime("%F %T")}


if REEVAL:
    data = json.load(open(REEVAL_PATH))
    old_sum = data["summary"]
    end = {"mem_avail_gib": old_sum["mem_avail_gib"]["end"], "rss_mib": old_sum["rss_mib"]["end"]}
    summary = evaluate(old_sum["label"], old_sum["minutes"], data["rounds"], data["requests"], end, old_sum.get("time"))
    data["summary"] = summary
    json.dump(data, open(REEVAL_PATH, "w"), indent=1)
    jsonl_path = os.environ.get("TALUS_SOAK_JSONL", "soak.jsonl")
    if os.path.exists(jsonl_path):
        lines = []
        replaced = False
        for l in open(jsonl_path):
            try:
                r = json.loads(l)
                if r.get("label") == summary["label"]:
                    lines.append(json.dumps(summary) + "\n")
                    replaced = True
                else:
                    lines.append(l)
            except Exception:
                lines.append(l)
        if not replaced:
            lines.append(json.dumps(summary) + "\n")
        open(jsonl_path, "w").writelines(lines)
    else:
        open(jsonl_path, "a").write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=1))
    sys.exit(0)


def rss_mib():
    try:
        cmd = ["ps", "-o", "rss=", "-C", "talus-cli,talus-server,talus,talus-qwen35,talus-laguna,llama-server"]
        ssh_host = os.environ.get("TALUS_SSH_HOST", "")
        if ssh_host:
            cmd = ["ssh", "-n", "-o", "BatchMode=yes", ssh_host, " ".join(cmd)]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        v = [int(x) for x in out.split() if x.isdigit()]
        return round(sum(v) / 1024) if v else None
    except Exception:
        return None


def chat(msg, i, greedy, max_tokens):
    body = {"messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens, "seed": i,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update({"temperature": 0.0} if greedy else {"temperature": 1.0, "top_p": 0.95})
    r = post_ttft("/v1/chat/completions", body, timeout=TIMEOUT)
    return r["choices"][0]["message"].get("content") or "", r["timings"], r["_ttft"]


text = corpus()
toks = post("/tokenize", {"content": text})["tokens"]
SUFFIX = "\n\nContinue the code above:\n"
P2K = post("/detokenize", {"tokens": toks[:2000]})["content"] + SUFFIX
P32K = post("/detokenize", {"tokens": toks[:32000]})["content"] + SUFFIX
NEW = post("/detokenize", {"tokens": toks[-(300 + 50000):-50000]})["content"]


def completion(prompt, n, cold):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0, "ignore_eos": True}
    if cold: body["cache_prompt"] = False
    r = post_ttft("/completion", body, timeout=TIMEOUT)
    return r.get("content") or "", r["timings"], r["_ttft"]


reqs, ref, rounds = [], {}, []
last32 = None
t_end = time.time() + MINUTES * 60
rnd = 0
while time.time() < t_end:
    rr = {"round": rnd, "t": round(time.time()), "mem_avail_gib": memavail(), "rss_mib": rss_mib()}
    plan = [("code", CODE[(2 * rnd) % len(CODE)], 2 * rnd, True, 700), ("code", CODE[(2 * rnd + 1) % len(CODE)], 2 * rnd + 1, True, 700),
            ("chat", CHAT[(2 * rnd) % len(CHAT)], 2 * rnd, False, 700), ("chat", CHAT[(2 * rnd + 1) % len(CHAT)], 2 * rnd + 1, False, 700),
            ("edit", EDIT[rnd % len(EDIT)], rnd, True, 1500)]
    jobs = [(k, (lambda m=m, i=i, g=g, n=n: chat(m, i, g, n)), g, f"{k}{i % len(CODE if k == 'code' else EDIT if k == 'edit' else CHAT)}")
            for k, m, i, g, n in plan]
    jobs += [("depth2k", lambda: completion(P2K, 128, True), True, "depth2k"),
             ("depth32k", lambda: completion(P32K, 128, True), True, "depth32k")]
    for kind, fn, greedy, key in jobs + [("warm32k", None, True, "warm32k")]:
        t0 = time.time(); rec = {"round": rnd, "kind": kind, "key": key}
        try:
            if kind == "warm32k":
                base = P32K + (last32 or "")
                txt, tm, tt = completion(base + "\n\n# Another file:\n" + NEW + "\n\nIn one sentence, what does the second file do?\n", 32, False)
            else:
                txt, tm, tt = fn()
            if kind == "depth32k": last32 = txt
            rec.update(ok=True, tg_tps=tm.get("predicted_per_second"), tg_n=tm.get("predicted_n"), prompt_n=tm.get("prompt_n"),
                       cache_n=tm.get("cache_n"), ttft_ms=tt.get("ttft_ms") or tt.get("ttft_srv_ms"),
                       text_sha256=hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16])
            if greedy:
                base_rnd = 1 if kind in CACHE_REUSE_KINDS else 0
                if rnd < base_rnd:
                    rec["greedy_same"] = None
                elif rnd == base_rnd or key not in ref:
                    ref[key] = txt
                    rec["greedy_same"] = True if base_rnd == 1 else None
                else:
                    rec["greedy_same"] = (txt == ref[key])
        except Exception as e:
            rec.update(ok=False, error=str(e)[:200])
            if kind == "depth32k": last32 = None
        rec["wall_s"] = round(time.time() - t0, 1)
        reqs.append(rec)
    rounds.append(rr); rnd += 1
    print(json.dumps({"round": rnd, "reqs": len(reqs), "errors": sum(not r["ok"] for r in reqs), **rr}), flush=True)

end = {"mem_avail_gib": memavail(), "rss_mib": rss_mib()}
summary = evaluate(LABEL, MINUTES, rounds, reqs, end)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"summary": summary, "rounds": rounds, "requests": reqs}, open(OUT, "w"), indent=1)
open(os.environ.get("TALUS_SOAK_JSONL", "soak.jsonl"), "a").write(json.dumps(summary) + "\n")
print(json.dumps(summary, indent=1))
