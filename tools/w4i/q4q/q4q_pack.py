#!/usr/bin/env python3
"""q4q_pack.py OUT.w4i GGUF PART.npz... : assemble a tp_w4i sidecar, format v3, magic "W4I3" (rust/.../qwen35/w4i.rs), from
q4q_sidecar.py parts.  v3 entry = v2 entry with flags (bit 0: kmode 1) in the spare u32 and a trailing u64
off_ismooth (f32 [k] 1/s; 0 = no smoothing)."""
import os, sys, struct
import numpy as np

AL = 4096


def main():
    out, src, parts = sys.argv[1], sys.argv[2], sys.argv[3:]
    mats = []
    for p in parts:
        z = np.load(p)
        L, name = os.path.basename(p).split('.')[:2]
        n, k, wg, km = (int(v) for v in z['meta'])
        mats.append((int(L), f'blk.{L}.{name}.weight', n, k, wg, km, p))
    mats.sort(key=lambda m: (m[0], m[1]))
    hdr = 16 + 128 + 112 * len(mats)
    pos = (hdr + AL - 1) // AL * AL
    plan = []
    for L, nm, n, k, wg, km, p in mats:
        g = k // (wg or k)
        z = np.load(p)
        has_s = z['inv_s'].size == k
        o_wq = pos
        o_ws = (o_wq + n * k // 2 + AL - 1) // AL * AL
        o_sg = (o_ws + n * g * 4 + AL - 1) // AL * AL
        o_sm = (o_sg + k + AL - 1) // AL * AL if has_s else 0
        pos = ((o_sm + 4 * k) if has_s else (o_sg + k))
        pos = (pos + AL - 1) // AL * AL
        plan.append((nm, n, k, wg, km, o_wq, o_ws, o_sg, o_sm, p))
    with open(out, 'wb') as o:
        # magic W4I3: engines older than format v3 refuse the file (they would misread v3 entries as v2)
        o.write(struct.pack('<4sIII', b'W4I3', 3, len(plan), 0))
        o.write(struct.pack('<96sQQQQ', os.path.basename(src).encode()[:95], os.path.getsize(src), 0, 0, 0))
        for nm, n, k, wg, km, o_wq, o_ws, o_sg, o_sm, _ in plan:
            o.write(struct.pack('<64sIIIIQQQQ', nm.encode()[:63], n, k, wg, km & 1, o_wq, o_ws, o_sg, o_sm))
        for nm, n, k, wg, km, o_wq, o_ws, o_sg, o_sm, p in plan:
            z = np.load(p)
            o.seek(o_wq); o.write(z['wq'].tobytes())
            o.seek(o_ws); o.write(np.ascontiguousarray(z['ws'], np.float32).tobytes())
            o.seek(o_sg); o.write(z['signs'].astype(np.int8).tobytes())
            if o_sm:
                o.seek(o_sm); o.write(z['inv_s'].astype(np.float32).tobytes())
        o.truncate(pos)
    print(f'wrote {out}: {len(plan)} matrices, {pos / 2**30:.2f} GiB')


if __name__ == '__main__':
    main()
