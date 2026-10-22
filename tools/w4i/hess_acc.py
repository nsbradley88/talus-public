"""hess_acc.py TMPDIR HESSDIR (tools/w4i) : H += X^T X for every <L>.<kind>.f16 dump in TMPDIR (running sums, f32 .npy), then delete dumps."""
import os, sys, glob, numpy as np
K = {'gdn_in': 5120, 'gdn_out': 6144, 'attn_in': 5120, 'attn_out': 6144, 'ffn_in': 5120, 'ffn_down': 17408}
t, hd = sys.argv[1], sys.argv[2]; os.makedirs(hd, exist_ok=True)
for p in sorted(glob.glob(f'{t}/*.f16')):
    L, kind = os.path.basename(p).split('.')[:2]; k = K[kind]
    x = np.memmap(p, np.float16, 'r').reshape(-1, k)
    g = np.zeros((k, k), np.float64)
    for i in range(0, len(x), 4096):
        c = np.asarray(x[i:i + 4096], np.float32); g += c.T @ c
    o = f'{hd}/{L}.{kind}.npy'
    if os.path.exists(o):
        h = np.load(o, mmap_mode='r+'); h += g.astype(np.float32); h.flush(); del h
    else:
        np.save(o, g.astype(np.float32))
    cnt = f'{hd}/{L}.{kind}.count'
    n = int(open(cnt).read()) if os.path.exists(cnt) else 0
    open(cnt, 'w').write(str(n + len(x)))
    os.remove(p)
print('acc', t, '->', hd, flush=True)
