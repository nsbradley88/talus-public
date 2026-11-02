"""hess_acc32.py TMPDIR HESSDIR: H += X^T X (f32 GEMM per 4096-row chunk, f64 running sum f64 in memory, f32 .npy on disk),
plus per-channel sum of squares; deletes the dumps."""
import os, sys, glob, numpy as np
K = {'ffn_in': 5120, 'ffn_down': 17408}
t, hd = sys.argv[1], sys.argv[2]
for p in sorted(glob.glob(f'{t}/*.f16')):
    L, kind = os.path.basename(p).split('.')[:2]; k = K[kind]
    x = np.memmap(p, np.float16, 'r').reshape(-1, k)
    o = f'{hd}/{L}.{kind}.npy'
    g = np.load(o).astype(np.float64) if os.path.exists(o) else np.zeros((k, k), np.float64)
    for i in range(0, len(x), 4096):
        c = np.asarray(x[i:i + 4096], np.float32); g += c.T @ c
    np.save(o + '.tmp.npy', g.astype(np.float32)); os.replace(o + '.tmp.npy', o)
    del g
    cnt = f'{hd}/{L}.{kind}.count'
    n = int(open(cnt).read()) if os.path.exists(cnt) else 0
    open(cnt, 'w').write(str(n + len(x)))
    os.remove(p)
print('acc', t, flush=True)
