"""q4q_common.py (tools/w4i/q4q): rotation and the tp_w4i activation quantizers (host models), shared by the q4q tools."""


import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
POOL = ThreadPoolExecutor(int(os.environ.get("PAR", "16")))


def rowpar(fn):
    def w(x, *a, **k):
        n = x.shape[0]
        ch = max(16, -(-n // (2 * POOL._max_workers)) // 16 * 16)
        parts = list(POOL.map(lambda i: fn(x[i:i + ch], *a, **k), range(0, n, ch)))
        return np.concatenate(parts, 0)
    return w


HCACHE = {}


HCACHE = {}


def hadamard(n):
    h = np.array([[1.0]], np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


from concurrent.futures import ThreadPoolExecutor
POOL = ThreadPoolExecutor(int(os.environ.get('PAR', '24')))


def rot(x, signs, hb=256):          # rows of x: v <- H (D v) / sqrt(hb) per hb-block (hb=256: the kernel's R)
    H = HCACHE.setdefault(hb, hadamard(hb))
    t, k = x.shape
    return ((x * signs[None, :k]).reshape(-1, hb) @ (H / np.sqrt(hb))).reshape(t, k)


def irot(u, signs, hb=256):
    H = HCACHE.setdefault(hb, hadamard(hb))
    t, k = u.shape
    return (u.reshape(-1, hb) @ (H / np.sqrt(hb))).reshape(t, k) * signs[None, :k]


@rowpar
def qa_kernel(v, ag=32, kmax=127, qmax=7):
    """tp_w4i exactly: S_t = max|v|/(7*kmax), k_g = clamp(ceil(amax_g/(7 S_t)),1,kmax), q = clamp(rint(v/(k_g S_t)))"""
    t, k = v.shape
    g = v.reshape(t, k // ag, ag)
    amax = np.abs(g).max(2)
    S = np.maximum(amax.max(1), 1e-30) / (qmax * kmax)
    kg = np.clip(np.ceil(amax / (qmax * S[:, None])), 1, kmax)
    sc = kg * S[:, None]
    q = np.clip(np.rint(g / sc[:, :, None]), -qmax, qmax)
    return (q * sc[:, :, None]).reshape(t, k)


@rowpar
def qa_kernel_mse(v, ag=32, kmax=127, qmax=7, clips=(1.0, 0.9, 0.8, 0.7)):
    """integer-k kernel format, but each group picks the k_g (from amax*clip) with the lowest error"""
    t, k = v.shape
    g = v.reshape(t, k // ag, ag)
    amax = np.abs(g).max(2)
    S = np.maximum(amax.max(1), 1e-30) / (qmax * kmax)
    best = None
    for c in clips:
        kg = np.clip(np.ceil(amax * c / (qmax * S[:, None])), 1, kmax)
        sc = kg * S[:, None]
        d = np.clip(np.rint(g / sc[:, :, None]), -qmax, qmax) * sc[:, :, None]
        e = ((d - g) ** 2).sum(2)
        if best is None:
            best, out = e, d
        else:
            m = e < best
            best = np.where(m, e, best)
            out = np.where(m[:, :, None], d, out)
    return out.reshape(t, k)
