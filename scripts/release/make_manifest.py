#!/usr/bin/env python3
"""make_manifest.py STAGE TAG COMMIT SOURCE_DATE_EPOCH ROCM_VERSION SDK [BASE_IMAGE] > MANIFEST.json

The build manifest of a release tarball: what was built (commit, tag), with what (ROCm SDK version + fingerprint,
Rust toolchain, C/C++ compiler, cmake, base image, apt snapshot), for what (GPU targets, CPU ISA of the ggml part,
the kernel C ABI version), and what the user needs at run time (ROCm sonames, glibc floor, how libraries resolve).
"""
import hashlib, json, os, re, subprocess, sys

def sh(*a):
    try:
        return subprocess.check_output(a, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:  # noqa: BLE001 - record, don't fail the build over a version string
        return f"unavailable ({e.__class__.__name__})"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def elf_info(p):
    dyn = sh("readelf", "-d", p)
    needed = re.findall(r"\(NEEDED\).*\[(.+?)\]", dyn)
    rpath = re.findall(r"\((?:RUNPATH|RPATH)\).*\[(.+?)\]", dyn)
    vers = re.findall(r"GLIBC_(\d+)\.(\d+)", sh("objdump", "-T", p))
    glibc = max(((int(a), int(b)) for a, b in vers), default=None)
    return needed, rpath, glibc

def main():
    st, tag, commit, sde, rocm, sdk = sys.argv[1:7]
    base = sys.argv[7] if len(sys.argv) > 7 else ""
    # fingerprint of the SDK parts that shape the output: the compiler, the device libraries, the HIP headers + runtime
    fp = hashlib.sha256()
    for rel in ["lib/llvm/bin/clang-24" if os.path.exists(f"{sdk}/lib/llvm/bin/clang-24") else "lib/llvm/bin/clang",
                "lib/libamdhip64.so.7", "include/hip/hip_version.h"]:
        fp.update(f"{rel} {sha256(os.path.join(sdk, rel))}\n".encode())
    bc = os.path.join(sdk, "lib/llvm/amdgcn/bitcode")
    for f in sorted(os.listdir(bc)) if os.path.isdir(bc) else []:
        fp.update(f"bitcode/{f} {sha256(os.path.join(bc, f))}\n".encode())
    hdr = open("/src/include/talus_kernels.h").read()
    abi = re.search(r"#define\s+TALUS_KERNELS_ABI_VERSION\s+(\d+)", hdr)
    files, glibc_floor, rocm_needed = {}, (0, 0), set()
    for name in sorted(os.listdir(os.path.join(st, "bin"))):
        p = os.path.join(st, "bin", name)
        needed, rpath, glibc = elf_info(p)
        files[f"bin/{name}"] = {"sha256": sha256(p), "size": os.path.getsize(p), "needed": needed, "runpath": rpath}
        if glibc and glibc > glibc_floor:
            glibc_floor = glibc
        rocm_needed |= {n for n in needed if re.match(r"lib(amdhip64|hipblas|hipblaslt|rocblas|hsa-runtime64)", n)}
    apt = open("/opt/apt-packages.txt").read().split() if os.path.exists("/opt/apt-packages.txt") else []
    keep = ("gcc-13=", "g++-13=", "libstdc++-13-dev=", "libc6-dev=", "binutils=", "cmake=", "ninja-build=", "patchelf=")
    m = {
        "name": "talus", "tag": tag, "commit": commit, "source_date_epoch": int(sde),
        "targets": {"host": "x86_64-unknown-linux-gnu", "gpu": ["gfx1151"],
                    "ggml_cpu_isa": "-march=znver4 -mtune=znver4 (= -march=native of the Zen 4 build host of the "
                                    "measured builds; runs on Zen 4 / Zen 5, Strix Halo is Zen 5)"},
        "kernels_abi_version": int(abi.group(1)) if abi else None,
        "build": {
            "rocm_sdk": {"version": rocm, "fingerprint_sha256": fp.hexdigest(),
                         "clang": sh(f"{sdk}/lib/llvm/bin/clang", "--version").splitlines()[0]},
            "rustc": sh("rustc", "-vV"), "cargo": sh("cargo", "-V"),
            "cc": sh("gcc", "--version").splitlines()[0], "cmake": sh("cmake", "--version").splitlines()[0],
            "ninja": sh("ninja", "--version"), "base_image": base,
            "apt_snapshot": open("/opt/apt-snapshot.txt").read().strip() if os.path.exists("/opt/apt-snapshot.txt") else "",
            "apt_packages": [p for p in apt if p.startswith(keep)],
        },
        "runtime": {
            "rocm_required": rocm, "rocm_sonames": sorted(rocm_needed | {"libamdhip64.so.7"}),
            "rocm_lookup": ["LD_LIBRARY_PATH", "PREFIX/rocm-sdk/lib (symlink PREFIX/rocm-sdk -> the ROCm SDK root)",
                            "/opt/rocm/lib", "ROCM_SDK / ROCM_PATH (+ /lib) for libamdhip64 (hip-sys)"],
            "glibc_min": "%d.%d" % glibc_floor,
            "kernels_lookup": "libtalus_kernels.so next to the executables (bin/); TALUS_KERNELS_LIB overrides",
        },
        "files": files,
    }
    json.dump(m, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")

if __name__ == "__main__":
    main()
