#!/usr/bin/env python3
"""q4q_sidecar.py (rd-q4quant phase 1 D): per-matrix W4A4 sidecar parts for tp_w4i, written straight in the .w4i
layout (format v3 pieces), from the GGUF's FFN weights and full-calibration Hessians (tools/hess_all.sh).

Per matrix options (a PLAN file, one line per matrix: `L name wg smooth kmode`, e.g. `30 ffn_down 128 0.5 1`):
  wg     0 (per-row scale) | 128 | 256   weight groups
  smooth 0 | alpha                        SmoothQuant s_j = rms(x_j)^a / rms(w_.j)^(1-a) (geomean 1), from diag(H);
                                          ffn_gate and ffn_up share one s (they share the packed activation): their
                                          rms(w) is over both matrices' columns.  W' = W s, H' = H / (s s^T).
  kmode  0 | 1                            activation group scale rule recorded in the entry flags (pack side)
GPTQ as tools/w4i/gptq_sidecar.py: 256-pt Hadamard with seeded signs (seed 20261017), damp 0.1, blocks of 128,
MSE clip search per group, int4 -7..7.

Writes OUTDIR/<L>.<name>.part (npz: q packed tile bytes, ws, signs, inv_s, meta); tools/q4q_pack.py assembles the
.w4i v3.  Usage: q4q_sidecar.py GGUF HESSDIR PLAN OUTDIR [--jobs 2]
"""
import os, sys, time, argparse
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault('GGUFPY', os.path.join(os.environ.get('LLAMA_CPP_SRC', ''), 'gguf-py'))
import gptq_sidecar as G  # noqa: E402  (hadamard, rot_rows, rot_hess, gptq_prep, gptq_q, rtn_q, signs_for)

SEED, KMAX = 20261017, 17408
CLIPS = [1, 0.95, 0.9, 0.85, 0.8]
INPUT = {'ffn_gate': 'ffn_in', 'ffn_up': 'ffn_in', 'ffn_down': 'ffn_down'}


def load_w(r, name):
    t = r['m'][name]
    n, k = int(t.shape[1]), int(t.shape[0])
    return G.quants.dequantize(t.data, t.tensor_type).reshape(n, k).astype(np.float64)


def smooth_vec(H, cnt, wcols, a):
    rx = np.sqrt(np.maximum(np.diag(H) / cnt, 1e-20))
    rw = np.sqrt(np.maximum((wcols ** 2).mean(0), 1e-30))
    s = rx ** a / rw ** (1 - a)
    return s / np.exp(np.log(s).mean())


def pack_tiles(q):
    """int q [n][k] (-7..7) -> the tp_w4i layout wq [n/16][k/32][16 rows][16 B] (low nibble = even k)"""
    n, k = q.shape
    u = (q.astype(np.int16) & 0xF).astype(np.uint8)
    b = (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)          # [n][k/2]
    b = b.reshape(n // 16, 16, k // 32, 16).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(b)


R = {}


def work(job):
    gguf_path, hd, L, name, wg, a, kmode, out = job
    of = f'{out}/{L}.{name}.npz'
    if os.path.exists(of):
        return of, 'cached', 0.0
    t0 = time.time()
    if 'm' not in R:
        R['r'] = G.gguf.GGUFReader(gguf_path)
        R['m'] = {t.name: t for t in R['r'].tensors}
    w = load_w(R, f'blk.{L}.{name}.weight')
    n, k = w.shape
    inp = INPUT[name]
    H = np.load(f'{hd}/{L}.{inp}.npy').astype(np.float64)
    cnt = int(open(f'{hd}/{L}.{inp}.count').read())
    inv_s = None
    if a > 0:
        if inp == 'ffn_in':
            other = 'ffn_up' if name == 'ffn_gate' else 'ffn_gate'
            wc = np.concatenate([w, load_w(R, f'blk.{L}.{other}.weight')], 0)
        else:
            wc = w
        s = smooth_vec(H, cnt, wc, a)
        del wc
        w = w * s[None, :]
        H = H / s[:, None] / s[None, :]
        inv_s = (1.0 / s).astype(np.float32)
    signs = G.signs_for(SEED, KMAX)
    x = G.rot_rows(w, signs)
    hr = G.rot_hess(H, signs)
    del H
    U = G.gptq_prep(hr, 0.1)
    q, sc = G.gptq_q(x, hr, wg or k, CLIPS, damp=0.1, U=U)
    del U
    g = k // (wg or k)
    deq = (q.reshape(n, g, -1) * sc[:, :, None]).reshape(n, k)
    d = deq - x
    hrel = float(np.sqrt(((d @ hr) * d).sum() / ((x @ hr) * x).sum()))
    rel = float(np.sqrt((d ** 2).sum() / (x ** 2).sum()))
    np.savez(of + '.tmp.npz', wq=pack_tiles(q), ws=sc.astype(np.float32).reshape(n, g),
             signs=signs[:k].astype(np.int8), inv_s=inv_s if inv_s is not None else np.zeros(0, np.float32),
             meta=np.array([n, k, wg, kmode], np.int64), rel=np.array([rel, hrel]))
    os.replace(of + '.tmp.npz', of)
    return of, f'rel {rel:.4f} hrel {hrel:.4f}', time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gguf'); ap.add_argument('hess'); ap.add_argument('plan'); ap.add_argument('out')
    ap.add_argument('--jobs', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for line in open(a.plan):
        p = line.split('#')[0].split()
        if not p:
            continue
        L, name, wg, al, km = int(p[0]), p[1], int(p[2]), float(p[3]), int(p[4])
        jobs.append((a.gguf, a.hess, L, name, wg, al, km, a.out))
    # ffn_down (k 17408: ~10 GiB in GPTQ) first, at most `jobs` at once
    jobs.sort(key=lambda j: (j[3] != 'ffn_down', j[2]))
    t0 = time.time()
    with Pool(a.jobs) as pool:
        for i, (of, msg, dt) in enumerate(pool.imap_unordered(work, jobs)):
            print(f'[{i + 1}/{len(jobs)}] {os.path.basename(of)} {msg} ({dt:.0f} s, total {time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
