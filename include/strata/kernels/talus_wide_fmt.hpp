// talus_wide_fmt.hpp - Talus the weight formats of the wide (9..16-column) exact mat-vec tiles, host and device.
//
// Shared by the HIP kernels (src/kernels/cuda/talus_wide.cuh) and the CPU tests (tests/cpu/talus_wide_fmt_test.cpp):
//   * per type, the block's fields as a register record (Rec) and the field list (offset, bytes, piece size) of a block;
//   * the load-time repack ("tile planes"): for a group of 16 rows and one 256-value superblock, every field is stored as
//     planes of 16 rows x P bytes (piece p of the field: [p][row][P]), so a wave's load of one piece for its 16 rows reads
//     16 * P contiguous bytes.  The bytes are the GGUF bytes, permuted (lossless; tile rows past n_out are zero).
//     Idea from tubo (BonkusCheemus/tubo, MIT): byte-exact load-time repacks so neighbouring lanes read one cache line.
//   * per type, the decode of sub-block j (32 values) to signed int8 words in the activation's K order, and the integer
//     step that turns the two 16-value dots (dA: values 0..15, dB: 16..31) into the dot's `sumi`.  These are the
//     operations of the wave32 kernels' traits (native_mmvq.cu IQ4XSTraits, KQ8Traits; talus_iq_split.cuh Split<TY>),
//     regrouped: integer sums are exact, so any grouping of a dot gives the same sumi.
//   * dec_half(j, h, v) = words 4 h .. 4 h + 3 of dec(j) (h = the lane's half-wave), so lanes 0-15 and 16-31
//     decode different halves of a sub-block and swap them (permlanex16) instead of both decoding all of it (the WMMA
//     operands must be the same in both half-waves).  kSplit: the type has dec_half (the K-quants' decode is too cheap).
// Grid tables (iq3s_grid, iq3xxs_grid, iq2xs_grid, iq2s_grid) come from ggml-common.h, which the includer must include
// first (GGML_COMMON_IMPL_CUDA on the device, GGML_COMMON_IMPL_CPP on the host).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__clang__)
#define TW_UNROLL _Pragma("unroll")
#else
#define TW_UNROLL
#endif
#if defined(__HIPCC__) || defined(__CUDACC__)
#define TW_HD __host__ __device__ __forceinline__
#else
#define TW_HD inline
#endif

namespace talus_wide {

#if defined(__HIP_DEVICE_COMPILE__)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
#endif

constexpr int kTileRows = 16;   // rows per tile (the WMMA N side)

// v_perm_b32 (selectors 0..7: a byte of {hi, lo}; 12: 0x00; 13: 0xff), as __builtin_amdgcn_perm(hi, lo, sel)
TW_HD uint32_t perm(uint32_t hi, uint32_t lo, uint32_t sel) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_perm(hi, lo, sel);
#else
    const uint64_t v = (uint64_t(hi) << 32) | lo;
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t s = (sel >> (8 * i)) & 0xffu;
        uint32_t b = 0;
        if (s < 8) b = uint32_t(v >> (8 * s)) & 0xffu;
        else if (s == 12) b = 0x00;
        else if (s == 13) b = 0xff;
        else b = 0;   // (8..11 and 14..15 are not used here)
        r |= b << (8 * i);
    }
    return r;
#endif
}

TW_HD int popc32(uint32_t v) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __popc(v);
#else
    return __builtin_popcount(v);
#endif
}

// talus_iq_split.cuh's talus_sign4: bit i of nib negates byte i of g (g's bytes are 1..127)
TW_HD int sign4(uint32_t g, uint32_t nib) {
    const uint32_t one = ((nib & 0x0fu) * 0x00204081u) & 0x01010101u;
    const uint32_t m = perm(0u, 0u, one | 0x0c0c0c0cu);
    return int((g ^ m) + one);
}
// llama.cpp's unpack_ksigns (7 sign bits + their parity as bit 7), on the low byte of v
TW_HD uint32_t ksigns(uint32_t v) {
    v &= 0xffu;
    const uint32_t p = uint32_t(popc32(v)) & 1u;
    return (v ^ (p << 7)) * 0x01010101u;
}

// the integer block-scale step as the kernels do it (v_mul_i32_i24: both factors well inside +-2^23)
TW_HD int imul24(int a, int b) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __mul24(a, b);
#else
    return int(int64_t(int32_t(uint32_t(a) << 8) >> 8) * int64_t(int32_t(uint32_t(b) << 8) >> 8));
#endif
}

// ---------------------------------------------------------------- the block fields
struct Field { int off, bytes, piece; };   // GGUF byte offset in the block, size, piece size P (bytes per row per plane)

// Per type: ggml id, block bytes, fields in plane order (piece sizes descending: every plane 16-byte aligned).
// IQ4_XS: d f16 @0 | scales_h u16 @2 | scales_l[4] @4 | qs[128] @8
// IQ3_S: d @0 | qs[64] @2 | qh[8] @66 | signs[32] @74 | scales[4] @106
// IQ3_XXS: d @0 | qs[64] (grid indices) @2 | aux32[8] (signs + scale) @66
// IQ2_XS: d @0 | qs u16[32] @2 | scales[8] @66
// IQ2_S: d @0 | qs[32] @2 | signs[32] @34 | qh[8] @66 | scales[8] @74
// Q4_K: dm half2 @0 | scales[12] @4 | qs[128] @16;  Q5_K: dm @0 | scales[12] @4 | qh[32] @16 | qs[128] @48
struct TypeDesc { int type, block_bytes, nfields; Field f[5]; };
constexpr TypeDesc kTypes[] = {
    {23, 136, 2, {{8, 128, 16}, {0, 8, 8}}},
    {21, 110, 5, {{2, 64, 16}, {74, 32, 16}, {66, 8, 8}, {106, 4, 4}, {0, 2, 2}}},
    {18, 98, 3, {{2, 64, 16}, {66, 32, 16}, {0, 2, 2}}},
    {17, 74, 3, {{2, 64, 16}, {66, 8, 8}, {0, 2, 2}}},
    {22, 82, 5, {{2, 32, 16}, {34, 32, 16}, {66, 8, 8}, {74, 8, 8}, {0, 2, 2}}},
    {12, 144, 3, {{16, 128, 16}, {4, 12, 4}, {0, 4, 4}}},
    {13, 176, 4, {{48, 128, 16}, {16, 32, 16}, {4, 12, 4}, {0, 4, 4}}},
};
TW_HD const TypeDesc* type_desc(int type) {
    for (const TypeDesc& d : kTypes)
        if (d.type == type) return &d;
    return nullptr;
}
// bytes of a packed matrix: ceil(n_out / 16) tiles of 16 full rows
inline std::size_t packed_bytes(int type, int n_in, int n_out) {
    const TypeDesc* d = type_desc(type);
    if (!d || n_in % 256 != 0 || n_out <= 0) return 0;
    return std::size_t((n_out + kTileRows - 1) / kTileRows) * kTileRows * std::size_t(n_in / 256) * d->block_bytes;
}
// byte offset, inside a tile superblock (16 blocks), of row r's piece p of field k
TW_HD int plane_off(const TypeDesc& d, int k, int p, int r) {
    int o = 0;
    for (int i = 0; i < k; ++i) o += kTileRows * d.f[i].bytes;
    return o + (p * kTileRows + r) * d.f[k].piece;
}

// CPU repack: GGUF rows (n_out x n_in / 256 blocks) -> tile planes.  Tile g, superblock sb starts at
// ((g * bpr) + sb) * 16 * block_bytes.  Returns false when the type is not covered.
inline bool pack_host(int type, const void* gguf, void* out, int n_in, int n_out) {
    const TypeDesc* d = type_desc(type);
    if (!d || n_in % 256 != 0) return false;
    const int bpr = n_in / 256, bs = d->block_bytes, nt = (n_out + kTileRows - 1) / kTileRows;
    const auto* src = static_cast<const uint8_t*>(gguf);
    auto* dst = static_cast<uint8_t*>(out);
    for (int g = 0; g < nt; ++g)
        for (int sb = 0; sb < bpr; ++sb) {
            uint8_t* t = dst + (std::size_t(g) * bpr + sb) * kTileRows * bs;
            for (int r = 0; r < kTileRows; ++r) {
                const int row = g * kTileRows + r;
                for (int k = 0; k < d->nfields; ++k) {
                    const Field& f = d->f[k];
                    for (int p = 0; p < f.bytes / f.piece; ++p) {
                        uint8_t* o = t + plane_off(*d, k, p, r);
                        if (row < n_out) std::memcpy(o, src + (std::size_t(row) * bpr + sb) * bs + f.off + p * f.piece, f.piece);
                        else std::memset(o, 0, f.piece);
                    }
                }
            }
        }
    return true;
}
// CPU inverse (tests): tile planes -> GGUF rows
inline bool unpack_host(int type, const void* packed, void* gguf, int n_in, int n_out) {
    const TypeDesc* d = type_desc(type);
    if (!d || n_in % 256 != 0) return false;
    const int bpr = n_in / 256, bs = d->block_bytes;
    const auto* src = static_cast<const uint8_t*>(packed);
    auto* dst = static_cast<uint8_t*>(gguf);
    for (int row = 0; row < n_out; ++row)
        for (int sb = 0; sb < bpr; ++sb) {
            const uint8_t* t = src + (std::size_t(row / kTileRows) * bpr + sb) * kTileRows * bs;
            for (int k = 0; k < d->nfields; ++k) {
                const Field& f = d->f[k];
                for (int p = 0; p < f.bytes / f.piece; ++p)
                    std::memcpy(dst + (std::size_t(row) * bpr + sb) * bs + f.off + p * f.piece,
                                t + plane_off(*d, k, p, row % kTileRows), f.piece);
            }
        }
    return true;
}

// ---------------------------------------------------------------- field loads
// A field's 32-bit words of one row's block.  GGUF: from the block at `b` (alignment A: 2 = u16 pairs, as get_int_b2;
// 4+ = u32, 16 = u32x4).  Packed: the tile superblock `t`, row r (pieces are P-aligned).
template<int A>
TW_HD uint32_t ld32(const uint8_t* p) {
    if constexpr (A >= 4) return *reinterpret_cast<const uint32_t*>(p);
    else {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
        return uint32_t(h[0]) | (uint32_t(h[1]) << 16);
    }
}
TW_HD uint32_t ld16(const uint8_t* p) { return *reinterpret_cast<const uint16_t*>(p); }

// Loader: GGUF block (`Packed` false: base = block pointer) or tile planes (true: base = tile superblock, row r).
// field_words<K, NW>(dst): NW 32-bit words of field K (NW * 4 == field bytes; 2-byte fields: one zero-extended word).
template<bool Packed, int BS>
struct Loader {
    const uint8_t* base;
    int r;
    static constexpr int kBlockAlign = (BS & -BS) >= 16 ? 16 : (BS & -BS);
    template<int OFF, int BYTES, int P, int FIELD_PLANE_OFF>
    TW_HD void words(uint32_t* dst) const {
        if constexpr (BYTES == 2) {
            if constexpr (Packed) dst[0] = ld16(base + FIELD_PLANE_OFF + r * 2);
            else dst[0] = ld16(base + OFF);
        } else if constexpr (Packed) {
            constexpr int NP = BYTES / P, WP = P / 4;
TW_UNROLL
            for (int p = 0; p < NP; ++p) {
                const uint8_t* s = base + FIELD_PLANE_OFF + (p * kTileRows + r) * P;
#if defined(__HIP_DEVICE_COMPILE__)
                if constexpr (P == 16) {
                    const u32x4 v = *reinterpret_cast<const u32x4*>(s);
                    dst[4 * p + 0] = v.x; dst[4 * p + 1] = v.y; dst[4 * p + 2] = v.z; dst[4 * p + 3] = v.w;
                } else if constexpr (P == 8) {
                    const u32x2 v = *reinterpret_cast<const u32x2*>(s);
                    dst[2 * p + 0] = v.x; dst[2 * p + 1] = v.y;
                } else
#endif
                {
TW_UNROLL
                    for (int w = 0; w < WP; ++w) dst[WP * p + w] = ld32<4>(s + 4 * w);
                }
            }
        } else {
            constexpr int A0 = (OFF & -OFF) == 0 ? 16 : (OFF & -OFF);
            constexpr int A = A0 < kBlockAlign ? A0 : kBlockAlign;
TW_UNROLL
            for (int w = 0; w < BYTES / 4; ++w) dst[w] = ld32<(A >= 4 ? 4 : 2)>(base + OFF + 4 * w);
        }
    }
};

// ---------------------------------------------------------------- per-type records and decodes
// kind 0: value = (dw * d8) * float(sumi)  (i-quants, IQ4_XS);  kind 1: KQ8 (Q4_K / Q5_K) min/scale epilogue
// dec(j, v): the 32 signed values of sub-block j as 8 words, word k = activation values 4k..4k+3.
// sumi(j, dA, dB): the dot's integer result as the traits compute it from the 16-value halves.

struct IQ4XS {
    static constexpr int type = 23, BS = 136, kind = 0;
    static constexpr bool kHalves = false, kSplit = true;
    uint32_t qs[32], hdr[2];
    template<bool P> TW_HD void load(const Loader<P, BS>& L) {
        L.template words<8, 128, 16, 0>(qs);
        L.template words<0, 8, 8, 16 * 128>(hdr);
    }
    // native_mmvq.cu iq4_table_lookup (the literal-word path): byte i of q -> iq4nl_values[nibble]
    TW_HD static void lut(uint32_t q4, int& lo, int& hi) {
        constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
        const uint32_t qe = q4, qo = q4 >> 4;
        const uint32_t se = qe & 0x07070707u, so = qo & 0x07070707u;
        const uint32_t el = perm(t1, t0, se), ol = perm(t1, t0, so);
        const uint32_t eh = perm(t3, t2, se), oh = perm(t3, t2, so);
        lo = int(perm(eh, el, 0x03020100u | ((qe & 0x08080808u) >> 1)));
        hi = int(perm(oh, ol, 0x03020100u | ((qo & 0x08080808u) >> 1)));
    }
    TW_HD void dec(int j, int v[8]) const {
TW_UNROLL
        for (int k = 0; k < 4; ++k) lut(qs[4 * j + k], v[k], v[k + 4]);
    }
    // lut's first output on q (lut's second output on q is this on q >> 4)
    TW_HD static int lut1(uint32_t q) {
        constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
        const uint32_t se = q & 0x07070707u;
        return int(perm(perm(t3, t2, se), perm(t1, t0, se), 0x03020100u | ((q & 0x08080808u) >> 1)));
    }
    TW_HD void dec_half(int j, int h, int v[4]) const {   // h 0: the low nibbles (values 0..15), 1: the high ones
TW_UNROLL
        for (int k = 0; k < 4; ++k) v[k] = lut1(qs[4 * j + k] >> (4 * h));
    }
    TW_HD int ls(int j) const {   // ((scales_l[j/2] >> 4 (j & 1)) & 15) | ((scales_h >> 2 j) & 3) << 4
        const uint32_t sl = (hdr[1] >> (8 * (j >> 1))) & 0xffu, sh = hdr[0] >> 16;
        return int(((sl >> (4 * (j & 1))) & 0x0fu) | (((sh >> (2 * j)) & 0x03u) << 4));
    }
    TW_HD int sumi(int j, int dA, int dB) const { return imul24(dA + dB, ls(j) - 32); }
    TW_HD uint32_t d16() const { return hdr[0] & 0xffffu; }
};

struct IQ3S {
    static constexpr int type = 21, BS = 110, kind = 0;
    static constexpr bool kHalves = false, kSplit = true;
    uint32_t qs[16], signs[8], qh[2], sc, d;
    template<bool P> TW_HD void load(const Loader<P, BS>& L) {
        L.template words<2, 64, 16, 0>(qs);
        L.template words<74, 32, 16, 16 * 64>(signs);
        L.template words<66, 8, 8, 16 * 96>(qh);
        L.template words<106, 4, 4, 16 * 104>(&sc);
        L.template words<0, 2, 2, 16 * 108>(&d);
    }
    TW_HD void dec(int j, int v[8]) const {
        const uint32_t q0 = qs[2 * j], q1 = qs[2 * j + 1];
        const uint32_t h = (qh[j >> 2] >> (8 * (j & 3))) & 0xffu;
        const uint32_t s = signs[j];
TW_UNROLL
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t qw = l0 < 4 ? q0 : q1;
            const uint32_t b0 = (qw >> (8 * (l0 & 3))) & 0xffu, b1 = (qw >> (8 * ((l0 & 3) + 1))) & 0xffu;
            const uint32_t gx = iq3s_grid[b0 | ((h << (8 - l0)) & 0x100u)];
            const uint32_t gy = iq3s_grid[b1 | ((h << (7 - l0)) & 0x100u)];
            v[l0 + 0] = sign4(gx, s >> (4 * l0));
            v[l0 + 1] = sign4(gy, s >> (4 * l0 + 4));
        }
    }
    TW_HD void dec_half(int j, int hf, int v[4]) const {   // dec's pairs l0 = 4 hf + m, m = 0, 2
        const uint32_t qw = hf ? qs[2 * j + 1] : qs[2 * j];
        const uint32_t h = (qh[j >> 2] >> (8 * (j & 3))) & 0xffu;
        const uint32_t s = signs[j];
TW_UNROLL
        for (int m = 0; m < 4; m += 2) {
            const int l0 = 4 * hf + m;
            const uint32_t b0 = (qw >> (8 * m)) & 0xffu, b1 = (qw >> (8 * (m + 1))) & 0xffu;
            const uint32_t gx = iq3s_grid[b0 | ((h << (8 - l0)) & 0x100u)];
            const uint32_t gy = iq3s_grid[b1 | ((h << (7 - l0)) & 0x100u)];
            v[m + 0] = sign4(gx, s >> (4 * l0));
            v[m + 1] = sign4(gy, s >> (4 * l0 + 4));
        }
    }
    TW_HD int sumi(int j, int dA, int dB) const {
        const int ls = 1 + 2 * int((sc >> (8 * (j >> 1) + 4 * (j & 1))) & 0x0fu);
        return imul24(dA + dB, ls);
    }
    TW_HD uint32_t d16() const { return d; }
};

struct IQ3XXS {
    static constexpr int type = 18, BS = 98, kind = 0;
    static constexpr bool kHalves = false, kSplit = true;
    uint32_t qs[16], aux[8], d;
    template<bool P> TW_HD void load(const Loader<P, BS>& L) {
        L.template words<2, 64, 16, 0>(qs);
        L.template words<66, 32, 16, 16 * 64>(aux);
        L.template words<0, 2, 2, 16 * 96>(&d);
    }
    TW_HD void dec(int j, int v[8]) const {
        const uint32_t q0 = qs[2 * j], q1 = qs[2 * j + 1], a = aux[j];
TW_UNROLL
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t qw = l0 < 4 ? q0 : q1;
            const uint32_t gx = iq3xxs_grid[(qw >> (8 * (l0 & 3))) & 0xffu];
            const uint32_t gy = iq3xxs_grid[(qw >> (8 * ((l0 & 3) + 1))) & 0xffu];
            const uint32_t s = ksigns(a >> (7 * l0 / 2));
            v[l0 + 0] = sign4(gx, s);
            v[l0 + 1] = sign4(gy, s >> 4);
        }
    }
    TW_HD void dec_half(int j, int hf, int v[4]) const {
        const uint32_t qw = hf ? qs[2 * j + 1] : qs[2 * j], a = aux[j];
TW_UNROLL
        for (int m = 0; m < 4; m += 2) {
            const uint32_t gx = iq3xxs_grid[(qw >> (8 * m)) & 0xffu];
            const uint32_t gy = iq3xxs_grid[(qw >> (8 * (m + 1))) & 0xffu];
            const uint32_t s = ksigns(a >> (14 * hf + 7 * m / 2));   // 7 l0 / 2, l0 = 4 hf + m
            v[m + 0] = sign4(gx, s);
            v[m + 1] = sign4(gy, s >> 4);
        }
    }
    TW_HD int sumi(int j, int dA, int dB) const {
        const int s = dA + dB, ls = int(aux[j] >> 28);
        return (imul24(ls, s) + s / 2) / 2;
    }
    TW_HD uint32_t d16() const { return d; }
};

struct IQ2XS {
    static constexpr int type = 17, BS = 74, kind = 0;
    static constexpr bool kHalves = true, kSplit = true;
    uint32_t qs[16], sc[2], d;
    template<bool P> TW_HD void load(const Loader<P, BS>& L) {
        L.template words<2, 64, 16, 0>(qs);
        L.template words<66, 8, 8, 16 * 64>(sc);
        L.template words<0, 2, 2, 16 * 72>(&d);
    }
    TW_HD void dec(int j, int v[8]) const {
TW_UNROLL
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t q = (qs[2 * j + (l0 >> 2)] >> (16 * ((l0 >> 1) & 1))) & 0xffffu;
            const uint64_t g = iq2xs_grid[q & 0x1ffu];
            const uint32_t s = ksigns(q >> 9);
            v[l0 + 0] = sign4(uint32_t(g), s);
            v[l0 + 1] = sign4(uint32_t(g >> 32), s >> 4);
        }
    }
    TW_HD void dec_half(int j, int hf, int v[4]) const {
        const uint32_t qw = hf ? qs[2 * j + 1] : qs[2 * j];
TW_UNROLL
        for (int m = 0; m < 4; m += 2) {
            const uint32_t q = (qw >> (8 * m)) & 0xffffu;   // 16 ((l0 >> 1) & 1) = 8 m
            const uint64_t g = iq2xs_grid[q & 0x1ffu];
            const uint32_t s = ksigns(q >> 9);
            v[m + 0] = sign4(uint32_t(g), s);
            v[m + 1] = sign4(uint32_t(g >> 32), s >> 4);
        }
    }
    TW_HD int sumi(int j, int dA, int dB) const {
        const uint32_t s8 = (sc[j >> 2] >> (8 * (j & 3))) & 0xffu;
        const int ls0 = int(s8 & 0x0fu), ls1 = int(s8 >> 4);
        return (imul24(dA, ls0) + imul24(dB, ls1) + (dA + dB) / 2) / 4;
    }
    TW_HD uint32_t d16() const { return d; }
};

struct IQ2S {
    static constexpr int type = 22, BS = 82, kind = 0;
    static constexpr bool kHalves = true, kSplit = true;
    uint32_t qs[8], signs[8], qh[2], sc[2], d;
    template<bool P> TW_HD void load(const Loader<P, BS>& L) {
        L.template words<2, 32, 16, 0>(qs);
        L.template words<34, 32, 16, 16 * 32>(signs);
        L.template words<66, 8, 8, 16 * 64>(qh);
        L.template words<74, 8, 8, 16 * 72>(sc);
        L.template words<0, 2, 2, 16 * 80>(&d);
    }
    TW_HD void dec(int j, int v[8]) const {
        const uint32_t q = qs[j], s = signs[j];
        const uint32_t h = (qh[j >> 2] >> (8 * (j & 3))) & 0xffu;
TW_UNROLL
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint64_t g = iq2s_grid[((q >> (4 * l0)) & 0xffu) | ((h << (8 - l0)) & 0x300u)];
            v[l0 + 0] = sign4(uint32_t(g), s >> (4 * l0));
            v[l0 + 1] = sign4(uint32_t(g >> 32), s >> (4 * l0 + 4));
        }
    }
    TW_HD void dec_half(int j, int hf, int v[4]) const {
        const uint32_t q = qs[j] >> (16 * hf), s = signs[j] >> (16 * hf);   // pairs l0 = 4 hf + m
        const uint32_t h = (qh[j >> 2] >> (8 * (j & 3))) & 0xffu;
TW_UNROLL
        for (int m = 0; m < 4; m += 2) {
            const int l0 = 4 * hf + m;
            const uint64_t g = iq2s_grid[((q >> (4 * m)) & 0xffu) | ((h << (8 - l0)) & 0x300u)];
            v[m + 0] = sign4(uint32_t(g), s >> (4 * m));
            v[m + 1] = sign4(uint32_t(g >> 32), s >> (4 * m + 4));
        }
    }
    TW_HD int sumi(int j, int dA, int dB) const {
        const uint32_t s8 = (sc[j >> 2] >> (8 * (j & 3))) & 0xffu;
        const int ls0 = int(s8 & 0x0fu), ls1 = int(s8 >> 4);
        return (imul24(dA, ls0) + imul24(dB, ls1) + (dA + dB) / 2) / 4;
    }
    TW_HD uint32_t d16() const { return d; }
};

// Q4_K / Q5_K as native_mmvq.cu's KQ8Traits (one lane per 32-value sub-block): values (q >> 4 (j & 1)) & 15 of qs
// bytes 32 (j / 2) .., Q5_K's fifth bit (qh >> j) & 1; the 6-bit scale / min of sub-block j (kq_scale_min).
template<bool Q5>
struct KQ8 {
    static constexpr int type = Q5 ? 13 : 12, BS = Q5 ? 176 : 144, kind = 1;
    static constexpr bool kHalves = false, kSplit = false;
    uint32_t qs[32], qh[Q5 ? 8 : 1], sc[3], dm;
    template<bool P> TW_HD void load(const Loader<P, BS>& L) {
        if constexpr (Q5) {
            L.template words<48, 128, 16, 0>(qs);
            L.template words<16, 32, 16, 16 * 128>(qh);
            L.template words<4, 12, 4, 16 * 160>(sc);
            L.template words<0, 4, 4, 16 * 172>(&dm);
        } else {
            L.template words<16, 128, 16, 0>(qs);
            L.template words<4, 12, 4, 16 * 128>(sc);
            L.template words<0, 4, 4, 16 * 140>(&dm);
        }
    }
    TW_HD void dec(int j, int v[8]) const {
        const int k = j >> 1, sh = 4 * (j & 1);
TW_UNROLL
        for (int i = 0; i < 8; ++i) {
            uint32_t x = (qs[8 * k + i] >> sh) & 0x0f0f0f0fu;
            if constexpr (Q5) x |= ((qh[i] >> j) & 0x01010101u) << 4;
            v[i] = int(x);
        }
    }
    TW_HD uint32_t scb(int i) const { return (sc[i >> 2] >> (8 * (i & 3))) & 0xffu; }
    TW_HD void scale_min(int j, int& s, int& m) const {
        if (j < 4) { s = int(scb(j) & 63u); m = int(scb(j + 4) & 63u); }
        else { s = int((scb(j + 4) & 0xFu) | ((scb(j - 4) >> 6) << 4)); m = int((scb(j + 4) >> 4) | ((scb(j) >> 6) << 4)); }
    }
};
using Q4K = KQ8<false>;
using Q5K = KQ8<true>;

// The w32 kernels' final combine of a (row, column): lane l = 8 c + j holds p(c, j), the sequential sum of chain c
// (superblocks c, c + 4, ...) of sub-block j; the XOR butterfly 16, 8, 4, 2, 1 gives
//   q_j = (p(0,j) + p(2,j)) + (p(1,j) + p(3,j)),   y = ((q0 + q4) + (q2 + q6)) + ((q1 + q5) + (q3 + q7)).
TW_HD float tree(const float p[4][8]) {
    float q[8];
TW_UNROLL
    for (int j = 0; j < 8; ++j) q[j] = (p[0][j] + p[2][j]) + (p[1][j] + p[3][j]);
    return ((q[0] + q[4]) + (q[2] + q[6])) + ((q[1] + q[5]) + (q[3] + q[7]));
}

}  // namespace talus_wide
