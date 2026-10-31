#!/usr/bin/env python3
"""w4i_convert.py [--gguf SRC.gguf] [--layers A-B] OUT.w4i IN.tw4 [IN.tw4 ...] : convert .tw4 W4A4 sidecars (format v1, int4 weights in
the 256-block Hadamard basis, one scale per 16-row tile x weight group) into the tp_w4i layout the Rust 27B engine
loads with TALUS_Q27_W4I (see rust/crates/talus-model/src/qwen35/w4i.rs for the format).  A .tw4 whose weight group
equals the matrix width becomes per-row scales (wg 0); 256-value groups stay wg 256.  Later inputs win on duplicate
names.  Only the FFN matrices (ffn_gate / ffn_up / ffn_down) are kept (of layers A..B with --layers).
Format v2: --gguf records the source GGUF's file name and size; the engine refuses a sidecar whose recorded size
differs from the GGUF it loads (falls back to the exact path with a log line)."""
import os, struct, sys
import numpy as np

ORDER = np.array([0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15])
AL = 4096


def read_tw4(path):
    f = open(path, 'rb')
    h = f.read(128)
    magic, ver, n_mat, wgroup, hb, kmax, seed, signs_off, table_off, data_off, _ = struct.unpack('<4sIIIIIQQQQI', h[:60])
    assert magic == b'TW4A' and ver == 1 and hb == 256, path
    f.seek(signs_off)
    signs = np.frombuffer(f.read(kmax), np.int8)
    f.seek(table_off)
    ents = []
    for _ in range(n_mat):
        e = f.read(128)
        name, n, k, q_off, s_off = struct.unpack('<96sIIQQ', e[:120])
        ents.append((name.split(b'\0')[0].decode(), n, k, q_off, s_off))
    return f, wgroup, signs, ents


def main():
    args = sys.argv[1:]
    src, lay = None, None
    while args and args[0].startswith('--'):
        if args[0] == '--gguf':
            src = args[1]
        elif args[0] == '--layers':
            a_, b_ = args[1].split('-')
            lay = (int(a_), int(b_))
        args = args[2:]
    out, ins = args[0], args[1:]
    mats = {}
    for p in ins:
        f, wg, signs, ents = read_tw4(p)
        for name, n, k, q_off, s_off in ents:
            if not any(t in name for t in ('ffn_gate', 'ffn_up', 'ffn_down')):
                continue
            if lay and not lay[0] <= int(name.split('.')[1]) <= lay[1]:
                continue
            mats[name] = (p, f, wg, signs, n, k, q_off, s_off)
    names = sorted(mats, key=lambda s: (int(s.split('.')[1]), s))
    hdr = 16 + 128 + 104 * len(names)
    pos = (hdr + AL - 1) // AL * AL
    plan = []
    for name in names:
        p, f, wg, signs, n, k, q_off, s_off = mats[name]
        g = k // (wg or k)
        owg = 0 if g == 1 else wg
        assert owg in (0, 256), (name, wg)
        o_wq = pos
        o_ws = (o_wq + n * k // 2 + AL - 1) // AL * AL
        o_sg = (o_ws + n * g * 4 + AL - 1) // AL * AL
        pos = (o_sg + k + AL - 1) // AL * AL
        plan.append((name, n, k, owg, o_wq, o_ws, o_sg))
    with open(out, 'wb') as o:
        o.write(struct.pack('<4sIII', b'W4I1', 2, len(names), 0))
        sname = os.path.basename(src).encode()[:95] if src else b''
        o.write(struct.pack('<96sQQQQ', sname, os.path.getsize(src) if src else 0, 0, 0, 0))
        for name, n, k, owg, o_wq, o_ws, o_sg in plan:
            o.write(struct.pack('<64sIIIIQQQ', name.encode()[:63], n, k, owg, 0, o_wq, o_ws, o_sg))
        for (name, n, k, owg, o_wq, o_ws, o_sg) in plan:
            p, f, wg, signs, _, _, q_off, s_off = mats[name]
            g = k // (wg or k)
            f.seek(q_off)
            q = np.frombuffer(f.read(n * k // 2), np.uint8).reshape(n // 16, k // 256, 16, 8, 16)
            o.seek(o_wq)
            o.write(np.ascontiguousarray(q.transpose(0, 1, 3, 2, 4)).tobytes())
            f.seek(s_off)
            sc = np.frombuffer(f.read((n // 16) * g * 16 * 4), np.float32).reshape(n // 16, g, 16)
            s = np.zeros((n // 16, 16, g), np.float32)
            s[:, ORDER, :] = sc.transpose(0, 2, 1)
            o.seek(o_ws)
            o.write(np.ascontiguousarray(s.reshape(n, g)).tobytes())
            o.seek(o_sg)
            o.write(np.ascontiguousarray(signs[:k]).tobytes())
        o.truncate(pos)
    print(f'wrote {out}: {len(names)} matrices, {pos / 2**30:.2f} GiB')


if __name__ == '__main__':
    main()
