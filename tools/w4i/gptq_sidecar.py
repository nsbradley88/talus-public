#!/usr/bin/env python3
"""gptq_sidecar.py (tools/w4i): build a .tw4 int4 weight sidecar (format v1, read by
w4i_convert.py) with selectable rounding method, from the weights the exact path multiplies.

  --src deployed  : quantize the weights the exact path multiplies (the GGUF's own tensors, any ggml type, dequantized
                    with gguf-py) instead of a Q8_0 copy: the int4 copy then only adds its own rounding error to the
                    deployed model, not the deployed model's quantization error as well.
  --method rtn    : round to nearest after the rotation, MSE clip search (the original tool's method)
  --method gptq   : GPTQ error-compensating rounding in the rotated basis against calibration Hessians
                    (--hess DIR from hess_acc.py: <layer>.<input>.npy, X^T X of the layer input over the calib set)
--wgroup 0: one scale per output row (per-channel; the tp_w4i default sidecars).
Usage: gptq_sidecar.py --gguf UD.gguf --out F.tw4 [--wgroup 256|0] [--method rtn|gptq] [--hess DIR] [--jobs 6]
"""
from __future__ import annotations
import argparse, os, struct, sys, time, pathlib
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
GGUFPY = os.environ.get('GGUFPY') or os.path.join(os.environ.get('LLAMA_CPP_SRC', ''), 'gguf-py')   # llama.cpp's gguf-py
sys.path.insert(0, GGUFPY)
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

HB = 256
CLASSES = {"gdn_in": ["attn_qkv", "attn_gate"], "gdn_out": ["ssm_out"], "attn_in": ["attn_q", "attn_k", "attn_v"],
           "attn_out": ["attn_output"], "ffn_in": ["ffn_gate", "ffn_up"], "ffn_down": ["ffn_down"]}
INPUT_OF = {"attn_qkv": "gdn_in", "attn_gate": "gdn_in", "ssm_out": "gdn_out", "attn_q": "attn_in", "attn_k": "attn_in",
            "attn_v": "attn_in", "attn_output": "attn_out", "ffn_gate": "ffn_in", "ffn_up": "ffn_in", "ffn_down": "ffn_down"}


def hadamard(n):
    h = np.array([[1.0]], np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


H = hadamard(HB)


def signs_for(seed, kmax):
    return (np.random.default_rng(seed).integers(0, 2, size=kmax, dtype=np.int8) * 2 - 1).astype(np.int8)


def rot_rows(w, signs):        # W R^T: each 256-column block of each row times R = H D / 16 (R w per row)
    n, k = w.shape
    return ((w * signs[None, :k]).reshape(-1, HB) @ (H.T / 16.0)).reshape(n, k)


def rot_hess(h, signs):        # R H R^T for the block-diagonal R
    k = h.shape[0]
    hs = h * signs[None, :k] * signs[:k, None]
    nb = k // HB
    hb = hs.reshape(nb, HB, nb, HB)
    hb = np.einsum('ij,ajbk,lk->aibl', H / 16.0, hb, H / 16.0, optimize=True)
    return hb.reshape(k, k)


def rtn_q(x, wg, clips, qmax=7):
    n, k = x.shape
    g = k // wg
    xr = x.reshape(n, g, wg)
    amax = np.abs(xr).max(2)
    best = bs = None
    for c in clips:
        s = np.maximum(amax * c / qmax, 1e-12)
        e = ((np.clip(np.rint(xr / s[:, :, None]), -qmax, qmax) * s[:, :, None] - xr) ** 2).sum(2)
        if best is None:
            best, bs = e, s.copy()
        else:
            m = e < best
            best = np.where(m, e, best)
            bs = np.where(m, s, bs)
    q = np.clip(np.rint(xr / bs[:, :, None]), -qmax, qmax)
    return q.reshape(n, k).astype(np.int8), bs


def gptq_prep(hr, damp=0.01):
    """upper Cholesky factor of (H + damp mean(diag) I)^-1 (GPTQ's form), shared by all matrices of one input.
    Memory-lean: at most ~3 k x k float64 arrays alive besides hr."""
    k = hr.shape[0]
    Hm = hr.astype(np.float64)                            # copy
    dead = np.diag(Hm) <= 0
    Hm[dead, dead] = 1.0
    Hm[np.diag_indices(k)] += damp * np.mean(np.diag(Hm))
    Lc = np.linalg.cholesky(Hm)                           # H = Lc Lc^T
    del Hm
    Li = np.linalg.inv(Lc)
    del Lc
    Hinv = Li.T @ Li
    del Li
    U = np.linalg.cholesky(Hinv).T.astype(np.float32)
    return np.ascontiguousarray(U)


def gptq_q(x, hr, wg, clips, damp=0.01, blk=128, qmax=7, U=None):
    """GPTQ (Frantar et al.) on rows of x (rotated weights) with rotated Hessian hr; scales per (row, wg-group) chosen
    by the RTN clip search on the error-updated weights when the group starts. Returns int8 q [n,k], scales [n,g].
    Works on W^T (columns contiguous), float32 updates."""
    n, k = x.shape
    if U is None:
        U = gptq_prep(hr, damp)
    U = U.astype(np.float32)
    Wt = np.ascontiguousarray(x.T.astype(np.float32))     # [k, n]
    g = k // wg
    Qt = np.zeros((k, n), np.int8)
    S = np.zeros((n, g), np.float32)
    for i1 in range(0, k, blk):
        i2 = min(i1 + blk, k)
        W1 = Wt[i1:i2]                                     # view: updated in place
        E1 = np.zeros((i2 - i1, n), np.float32)
        for i in range(i2 - i1):
            c = i1 + i
            if c % wg == 0:
                _, sg = rtn_q(np.ascontiguousarray(Wt[c:c + wg].T), wg, clips, qmax)
                S[:, c // wg] = sg[:, 0]
            sc = S[:, c // wg]
            w = W1[i]
            q = np.clip(np.rint(w / sc), -qmax, qmax)
            Qt[c] = q
            err = (w - q * sc) / U[c, c]
            if i + 1 < i2 - i1:
                W1[i + 1:] -= np.outer(U[c, c + 1:i2], err)
            E1[i] = err
        if i2 < k:
            Wt[i2:] -= U[i1:i2, i2:].T @ E1
    return np.ascontiguousarray(Qt.T), S.astype(np.float64)


def pack(q, s, wg):
    n, k = q.shape
    g = k // wg
    gs = k // HB
    q = q.reshape(n, gs, HB)
    u = (q.astype(np.int16) & 0xF).astype(np.uint8)
    b = (u[:, :, 0::2] | (u[:, :, 1::2] << 4)).astype(np.uint8)
    b = b.reshape(n // 16, 16, gs, 128).transpose(0, 2, 1, 3)
    order = np.array([0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15])
    sc = s.reshape(n // 16, 16, g).transpose(0, 2, 1)[:, :, order]
    return np.ascontiguousarray(b), np.ascontiguousarray(sc.astype(np.float32))


G = {}
SEM = None


def init_worker(sem):
    global SEM
    SEM = sem


def work_group(jobs):
    out = []
    G['U'] = None
    big = jobs[0][2] > 8192          # ffn_down (k = 17408): one at a time (memory)
    if big and SEM is not None:
        SEM.acquire()
    try:
        for j in jobs:
            out.append(work(j))
    finally:
        G['U'] = None
        G['hr'] = None
        if big and SEM is not None:
            SEM.release()
    return out


def work(job):
    name, n, k, q_off, s_off, a = job
    r = G.get('r')
    if r is None:
        r = G['r'] = gguf.GGUFReader(a['gguf'])
        G['m'] = {t.name: t for t in r.tensors}
    t = G['m'][name]
    w = quants.dequantize(t.data, t.tensor_type).reshape(n, k).astype(np.float64)
    signs = signs_for(a['seed'], a['kmax'])
    x = rot_rows(w, signs)
    clips = a['clips']
    wg = a['wgroup'] or k   # (0: one scale per output row)
    L = int(name.split('.')[1])
    hp = os.path.join(a['hess'], f"{L}.{INPUT_OF[name.split('.')[2]]}.npy") if a['hess'] else ''
    if a['method'] == 'gptq' and os.path.exists(hp):   # (no Hessian, e.g. the MTP layer: RTN)
        if G.get('U') is None or G.get('Ukey') != hp:
            h = np.load(hp).astype(np.float64)
            G['hr'] = rot_hess(h, signs)
            G['U'] = gptq_prep(G['hr'], a['damp'])
            G['Ukey'] = hp
        hr = G['hr']
        q, s = gptq_q(x, hr, wg, clips, damp=a['damp'], U=G['U'])
    else:
        q, s = rtn_q(x, wg, clips)
    deq = (q.reshape(n, k // wg, wg) * s[:, :, None]).reshape(n, k)
    rel = float(np.sqrt(((deq - x) ** 2).sum() / (x ** 2).sum()))
    hrel = float('nan')
    if a['method'] == 'gptq' and os.path.exists(hp):
        d = deq - x
        hrel = float(np.sqrt(((d @ hr) * d).sum() / ((x @ hr) * x).sum()))
    if a.get('hess_te') and k <= 8192:   # held-out output-energy error, this method and RTN (not ffn_down: memory)
        pt = os.path.join(a['hess_te'], f"{L}.{INPUT_OF[name.split('.')[2]]}.npy")
        if os.path.exists(pt):
            ht = rot_hess(np.load(pt).astype(np.float64), signs)
            den = ((x @ ht) * x).sum()
            d = deq - x
            qr, sr = rtn_q(x, wg, clips)
            dr = (qr.reshape(n, k // wg, wg) * sr[:, :, None]).reshape(n, k) - x
            print(f"HELDOUT {name} {((d @ ht) * d).sum() / den:.6f} rtn {((dr @ ht) * dr).sum() / den:.6f}", flush=True)
            del ht
    qb, sc = pack(q, s, wg)
    with open(a['tmp'], 'r+b') as f:
        f.seek(q_off); f.write(qb.tobytes())
        f.seek(s_off); f.write(sc.tobytes())
    return name, t.tensor_type.name, rel, hrel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--classes', default='all')
    ap.add_argument('--layers', default='')
    ap.add_argument('--seed', type=int, default=20261017)
    ap.add_argument('--clips', default='1,0.95,0.9,0.85,0.8')
    ap.add_argument('--wgroup', type=int, default=HB)
    ap.add_argument('--method', default='rtn', choices=['rtn', 'gptq'])
    ap.add_argument('--hess', default='')
    ap.add_argument('--damp', type=float, default=0.01)
    ap.add_argument('--hess-te', default='', help='held-out Hessians: print held-out output error (and RTN) per matrix')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--resume', default='', help='log of an interrupted run: keep OUT.part, skip the tensors it lists')
    a = ap.parse_args()
    r = gguf.GGUFReader(a.gguf)
    want = set(sum(CLASSES.values(), [])) if a.classes == 'all' else set(sum((CLASSES[c] for c in a.classes.split(',')), []))
    layers = None
    if a.layers:
        lo, hi = (int(v) for v in a.layers.split('-'))
        layers = set(range(lo, hi + 1))
    sel = []
    for t in r.tensors:
        p = t.name.split('.')
        if len(p) != 4 or p[0] != 'blk' or p[3] != 'weight' or p[2] not in want or len(t.shape) != 2:
            continue
        if layers is not None and int(p[1]) not in layers:
            continue
        k, n = int(t.shape[0]), int(t.shape[1])
        if k % HB or n % 16:
            continue
        sel.append((t.name, n, k))
    kmax = max(k for _, _, k in sel)
    signs = signs_for(a.seed, kmax)
    hdr_size, ent_size, align = 128, 128, 4096
    signs_off = hdr_size
    table_off = (signs_off + kmax + 255) // 256 * 256
    data_off = (table_off + ent_size * len(sel) + align - 1) // align * align
    entries, pos = [], data_off
    for name, n, k in sel:
        qb, sb = n * k // 2, (n // 16) * (k // (a.wgroup or k)) * 16 * 4
        q_off = pos
        s_off = (q_off + qb + align - 1) // align * align
        pos = (s_off + sb + align - 1) // align * align
        entries.append((name, n, k, q_off, s_off))
    tmp = a.out + '.part'
    done = set()
    if a.resume:
        done = {l.split()[1] for l in open(a.resume) if l.startswith('[')}
    with open(tmp, 'r+b' if a.resume else 'wb') as f:
        f.truncate(pos)
        f.write(struct.pack('<4sIIIIIQQQQI', b'TW4A', 1, len(sel), a.wgroup, HB, kmax, a.seed, signs_off, table_off,
                            data_off, 0).ljust(hdr_size, b'\0'))
        f.seek(signs_off); f.write(signs.tobytes())
        f.seek(table_off)
        for name, n, k, q_off, s_off in entries:
            f.write(struct.pack('<96sIIQQ', name.encode()[:95], n, k, q_off, s_off).ljust(ent_size, b'\0'))
    cfg = dict(gguf=a.gguf, seed=a.seed, kmax=kmax, clips=[float(c) for c in a.clips.split(',')], wgroup=a.wgroup,
               method=a.method, hess=a.hess, damp=a.damp, tmp=tmp, hess_te=a.hess_te)
    t0 = time.time()
    groups = {}
    for name, n, k, q_off, s_off in entries:
        if name in done:
            continue
        p = name.split('.')
        groups.setdefault((p[1], INPUT_OF[p[2]]), []).append((name, n, k, q_off, s_off, cfg))
    glist = sorted(groups.values(), key=lambda g: -sum(j[1] * j[2] for j in g) * (g[0][2] / 5120) ** 0.5)
    rels = []
    res = []
    from multiprocessing import Semaphore
    with Pool(a.jobs, initializer=init_worker, initargs=(Semaphore(2),)) as p:
        for out in p.imap_unordered(work_group, glist):
          for (name, ty, rel, hrel) in out:
            res.append(name)
            i = len(res) - 1
            rels.append(rel)
            print(f'[{i + 1}/{len(entries)}] {name} ({ty}): rel {rel:.4f} hrel {hrel:.4f} ({time.time() - t0:.0f} s)', flush=True)
    os.replace(tmp, a.out)
    print(f'wrote {a.out}: {len(entries)} tensors, {pos / 2**30:.2f} GiB, mean rel {np.mean(rels):.4f}, {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
