#!/usr/bin/env python3
"""correct_parts.py GGUF HESSDIR SRC DST [L ...]: unbias q4q parts (the MSE-optimal clip searches shrink the output ~1.5%). Per output row r, c_r = c_w[r] * c_a with
  c_w[r] = (x_r H x_r^T) / (wq_r H x_r^T)   (rotated, smoothed weights x, quantized wq, calibration Hessian H)
  c_a    = 1 / (1 + <e, v> / <v, v>)        (e = kernel quantizer error on synthetic Gaussian rows with a per-row spread;
                                             post-Hadamard activations are Gaussian, so this depends only on the quantizer)
Scales ws[r, :] *= c_r.  Writes DST/<L>.<name>.npz (same format)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4q_sidecar as Q
import q4q_common as em
G = Q.G
# usage: correct_parts.py GGUF HESSDIR SRC DST [L ...]
gguf_path, HD, src, dst = sys.argv[1:5]
Ls = [int(x) for x in sys.argv[5:]] or list(range(64))
os.makedirs(dst, exist_ok=True)
R = {'r': G.gguf.GGUFReader(gguf_path)}
R['m'] = {t.name: t for t in R['r'].tensors}
# activation shrink per kmode from Gaussian rows with a heavy per-token spread (as real rows)
rng = np.random.default_rng(1)
g = (rng.standard_normal((2048, 5120)) * np.exp(0.3 * rng.standard_normal((2048, 1)))).astype(np.float32)
CA = {}
for km, f in ((0, em.qa_kernel), (1, em.qa_kernel_mse)):
    e = f(g) - g
    CA[km] = float(1.0 / (1.0 + (e * g).sum() / (g * g).sum()))
print('activation gain correction', CA, flush=True)
for L in Ls:
    for name in ('ffn_gate', 'ffn_up', 'ffn_down'):
        p = f'{src}/{L}.{name}.npz'
        if not os.path.exists(p) or os.path.exists(f'{dst}/{L}.{name}.npz'):
            continue
        z = dict(np.load(p))
        n, k, wg, km = (int(v) for v in z['meta'])
        b = z['wq'].transpose(0, 2, 1, 3).reshape(n, k // 2)
        lo = (b & 0xF).astype(np.int8); hi = (b >> 4).astype(np.int8)
        q = np.empty((n, k), np.float32); q[:, 0::2] = np.where(lo > 7, lo - 16, lo); q[:, 1::2] = np.where(hi > 7, hi - 16, hi)
        gs = wg or k
        wq = (q.reshape(n, k // gs, gs) * z['ws'][:, :, None]).reshape(n, k)
        w = Q.load_w(R, f'blk.{L}.{name}.weight').astype(np.float32)
        s = 1.0 / z['inv_s'] if z['inv_s'].size == k else np.ones(k, np.float32)
        x = em.rot(w * s[None, :], z['signs'].astype(np.float32))
        H = np.load(f'{HD}/{L}.{Q.INPUT[name]}.npy')
        Hs = (H / s[:, None] / s[None, :]).astype(np.float32)
        hr = G.rot_hess(Hs.astype(np.float64), z['signs'].astype(np.float64)).astype(np.float32)
        xh = x @ hr
        cw = (xh * x).sum(1) / (xh * wq).sum(1)
        c = (cw * CA[km]).astype(np.float32)
        z['ws'] = (z['ws'] * c[:, None]).astype(np.float32)
        np.savez(f'{dst}/{L}.{name}.npz.tmp.npz', **z); os.replace(f'{dst}/{L}.{name}.npz.tmp.npz', f'{dst}/{L}.{name}.npz')
        print(f'L{L} {name}: c_w median {np.median(cw):.5f} [{cw.min():.4f}, {cw.max():.4f}], c_a {CA[km]:.5f}', flush=True)
