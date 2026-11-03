#!/usr/bin/env python3
"""Guard the GDN (gated delta net) prompt-scan kernels against the occupancy cliff hipfire hit (wave 1 item 1.10).

hipfire's chunked GDN kernel grew to ~250 VGPRs, only 40 of its 48 head blocks fit on the GPU at once and the rest ran
as a second serial pass (+64% kernel time). Ours were checked on 2026-10-29: the 27B prompt scan
k_gdn_seq_warp<8> uses 123 VGPRs -> 10 waves per SIMD -> 800 wave slots for its 768-wave grid (48 value heads x 4
blocks x 4 waves); at more than 144 VGPRs only 9 waves fit (720 slots) and the grid needs a second pass. Flash-Next's
gdn_rec_dpp2c kernel (198 VGPRs, 7 waves) has a 384-wave grid in 560 slots.

This reads the gfx1151 code objects inside libtalus_kernels.so (the .hip_fatbin clang offload bundles), finds those
kernels and fails when one spills or when its production grid no longer fits in one pass. gfx1151: 20 WGPs x 4 SIMDs,
1536 VGPRs per lane per SIMD in granules of 24, at most 16 waves per SIMD (wave32).

  check_gdn_occupancy.py libtalus_kernels.so      (needs llvm-readelf: $ROCM_SDK/lib/llvm/bin or on PATH)

Exit 0 = OK, 1 = a kernel over its budget / spilling / missing, 77 = cannot run here (no tool or no gfx1151 objects).
"""
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

SIMDS = 80            # gfx1151: 20 WGPs x 4 SIMD32
VGPR_FILE = 1536      # VGPRs per lane per SIMD (wave32)
GRANULE = 24
MAX_WAVES = 16

# (kernel name regex on the mangled symbol, description, production grid in waves)
KERNELS = [
    (r"k_gdn_seq_warp.*Li8E", "27B prompt GDN scan k_gdn_seq_warp<8> (48 heads x 4 blocks x 4 waves)", 768),
    (r"gdn_rec_dpp2c", "Flash-Next prompt GDN scan gdn_rec_dpp2c", 384),
]
MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"


def readelf():
    for d in (os.environ.get("ROCM_SDK", ""), os.environ.get("ROCM_PATH", "")):
        p = os.path.join(d, "lib", "llvm", "bin", "llvm-readelf")
        if d and os.path.exists(p):
            return p
    return shutil.which("llvm-readelf")


def fatbin(lib):
    """The .hip_fatbin section bytes of an ELF64 shared object."""
    data = open(lib, "rb").read()
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return b""
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    sec = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    stro = sec[shstrndx][4]
    for s in sec:
        name = data[stro + s[0]:data.index(b"\0", stro + s[0])]
        if name == b".hip_fatbin":
            return data[s[4]:s[4] + s[5]]
    return b""


def gfx1151_objects(fb):
    """Every gfx1151 device code object in a run of (uncompressed) clang offload bundles."""
    out, pos = [], 0
    while True:
        pos = fb.find(MAGIC, pos)
        if pos < 0:
            return out
        n, = struct.unpack_from("<Q", fb, pos + 24)
        q = pos + 32
        for _ in range(n):
            off, size, idlen = struct.unpack_from("<QQQ", fb, q)
            ident = fb[q + 24:q + 24 + idlen].decode(errors="replace")
            q += 24 + idlen
            if "gfx1151" in ident and size:
                out.append(fb[pos + off:pos + off + size])
        pos += len(MAGIC)


def kernels(tool, co):
    with tempfile.NamedTemporaryFile(suffix=".co") as f:
        f.write(co)
        f.flush()
        notes = subprocess.run([tool, "--notes", f.name], capture_output=True, text=True).stdout
    ks, blk = [], {}
    for line in notes.splitlines():
        if re.match(r"\s+- \.args:", line) or re.match(r"\s+amdhsa\.target", line):
            if "name" in blk:
                ks.append(blk)
            blk = {}
        m = re.match(r"\s+\.(name|vgpr_count|vgpr_spill_count|sgpr_spill_count):\s+(\S+)", line)
        if m:
            blk.setdefault(m.group(1), m.group(2))
    if "name" in blk:
        ks.append(blk)
    return ks


def waves(vgpr):
    g = -(-max(vgpr, 1) // GRANULE) * GRANULE
    return min(MAX_WAVES, VGPR_FILE // g)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    tool = readelf()
    if not tool:
        print("check_gdn_occupancy: llvm-readelf not found (set ROCM_SDK)")
        return 77
    objs = gfx1151_objects(fatbin(sys.argv[1]))
    if not objs:
        print("check_gdn_occupancy: no gfx1151 code objects in %s" % sys.argv[1])
        return 77
    found = {}
    for co in objs:
        if b"gdn" not in co:
            continue
        for k in kernels(tool, co):
            for pat, _, _ in KERNELS:
                if re.search(pat, k["name"]):
                    found.setdefault(pat, []).append(k)
    bad = 0
    for pat, desc, grid in KERNELS:
        ks = found.get(pat)
        if not ks:
            print("FAIL %s: not found (renamed? update tools/check_gdn_occupancy.py)" % desc)
            bad += 1
            continue
        for k in ks:
            v = int(k.get("vgpr_count", 0))
            sp = int(k.get("vgpr_spill_count", 0)) + int(k.get("sgpr_spill_count", 0))
            w = waves(v)
            slots = w * SIMDS
            ok = slots >= grid and sp == 0
            # the largest VGPR count that still fits the grid in one pass
            budget = max((g for g in range(GRANULE, 257, GRANULE) if waves(g) * SIMDS >= grid), default=0)
            print("%s %s: %s VGPRs (budget %d), %d spills, %d waves/SIMD -> %d slots for a %d-wave grid"
                  % ("OK  " if ok else "FAIL", desc, v, budget, sp, w, slots, grid))
            bad += not ok
    print("check_gdn_occupancy: %s" % ("OK" if not bad else "%d problem(s)" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
