// src/prefill/tp_w4a4.cu - see include/strata/prefill/tp_w4a4.hpp.
//
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#include "strata/prefill/tp_w4a4.hpp"
#include "strata/prefill/tp_tune.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define TP_W4_WMMA 1
#endif

namespace talus::prefill {
namespace {

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v8i wmma_i4(v2i a, v2i b, v8i c) {
#if defined(TP_W4_WMMA)
    return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false);
#else
    (void) a; (void) b;
    return c;
#endif
}

/// One wave per (token, 256-value group): lane L holds values 8 L .. 8 L + 7.  R v = H (D v) / 16 by the in-register
/// butterflies (strides 1, 2, 4) then cross-lane ones (strides 8 .. 128), the Sylvester order the sidecar uses.
template <int AG>
__device__ __forceinline__ void rotate_pack(float v[8], const int8_t* __restrict__ sg, int lane, uint8_t* __restrict__ xq,
                                            float* __restrict__ xs) {
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] *= (float) sg[e];
#pragma unroll
    for (int h = 1; h < 8; h <<= 1)
#pragma unroll
        for (int e = 0; e < 8; ++e)
            if (!(e & h)) {
                const float a = v[e], b = v[e + h];
                v[e] = a + b;
                v[e + h] = a - b;
            }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float p = __shfl_xor(v[e], m, 32);
            v[e] = (lane & m) ? p - v[e] : v[e] + p;
        }
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        v[e] *= 0.0625f;
        amax = fmaxf(amax, fabsf(v[e]));
    }
#pragma unroll
    for (int m = 1; m < AG / 8; m <<= 1) amax = fmaxf(amax, __shfl_xor(amax, m, 32));   // over the AG-value group
    const float inv = amax > 0.f ? 7.0f / amax : 0.f;
    unsigned word = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int q = (int) fminf(fmaxf(rintf(v[e] * inv), -7.f), 7.f);
        word |= (unsigned) (q & 0xF) << (4 * e);
    }
    *(unsigned*) (xq + lane * 4) = word;
    if (lane % (AG / 8) == 0) xs[lane / (AG / 8)] = amax / 7.0f;
}

/// grid (ceil(k / 256 / 8), T), block 256: wave w packs group blockIdx.x * 8 + w of token blockIdx.y.
template <int AG>
__global__ void __launch_bounds__(256) k_w4_pack(const float* __restrict__ x, int k, int64_t ldx,
                                                 const int8_t* __restrict__ signs, uint8_t* __restrict__ xq,
                                                 float* __restrict__ xs) {
    const int lane = threadIdx.x & 31, G = k / kW4Group, g = blockIdx.x * 8 + (threadIdx.x >> 5), t = blockIdx.y;
    if (g >= G) return;
    const int c0 = g * kW4Group + lane * 8;
    const float4 a = *(const float4*) (x + (size_t) t * ldx + c0), b = *(const float4*) (x + (size_t) t * ldx + c0 + 4);
    float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    rotate_pack<AG>(v, signs + c0, lane, xq + (size_t) t * (k / 2) + g * 128, xs + (size_t) t * (k / AG) + g * (256 / AG));
}

template <int AG>
__global__ void __launch_bounds__(256) k_w4_pack_swiglu(const float* __restrict__ gt, const float* __restrict__ up, int k,
                                                        const int8_t* __restrict__ signs, uint8_t* __restrict__ xq,
                                                        float* __restrict__ xs) {
    const int lane = threadIdx.x & 31, G = k / kW4Group, g = blockIdx.x * 8 + (threadIdx.x >> 5), t = blockIdx.y;
    if (g >= G) return;
    const int c0 = g * kW4Group + lane * 8;
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float gv = gt[(size_t) t * k + c0 + e];
        v[e] = gv / (1.0f + expf(-gv)) * up[(size_t) t * k + c0 + e];   // = q36_engine's silu(g) * u
    }
    rotate_pack<AG>(v, signs + c0, lane, xq + (size_t) t * (k / 2) + g * 128, xs + (size_t) t * (k / AG) + g * (256 / AG));
}

/// grid (ceil(T / 128), n / 128), block 256 = 4 (weight rows) x 2 (tokens) waves, a wave WM x WT 16 x 16 tiles.
/// A = weights (rows n), B = activations (tokens): the accumulator lane holds token l16, rows 2 i + hi.
template <int WM, int WT, int AG>
__global__ void __launch_bounds__(256) k_w4_gemm(const uint8_t* __restrict__ wq, const float* __restrict__ ws,
                                                 const uint8_t* __restrict__ xq, const float* __restrict__ xs,
                                                 float* __restrict__ dst, int K, int T, int64_t ldd) {
    const int G = K / kW4Group;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int wn = w & 3, wt = w >> 2;
    const int m0 = blockIdx.y * 8 + wn * WM, t0 = blockIdx.x * 128 + wt * (16 * WT);
    float sum[WM][WT][8];
#pragma unroll
    for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int b = 0; b < WT; ++b)
#pragma unroll
            for (int i = 0; i < 8; ++i) sum[a][b][i] = 0.f;
    const uint8_t* xr[WT];
    const float* xsr[WT];
#pragma unroll
    for (int b = 0; b < WT; ++b) {
        const int t = min(t0 + b * 16 + l16, T - 1);
        xr[b] = xq + (size_t) t * (K / 2);
        xsr[b] = xs + (size_t) t * (K / AG);
    }
    for (int g = 0; g < G; ++g) {
        float dwg[WM][8];
#pragma unroll
        for (int a = 0; a < WM; ++a) {
            const float* sp = ws + ((size_t) (m0 + a) * G + g) * 16 + hi * 8;
            const float4 s0 = *(const float4*) sp, s1 = *(const float4*) (sp + 4);
            dwg[a][0] = s0.x; dwg[a][1] = s0.y; dwg[a][2] = s0.z; dwg[a][3] = s0.w;
            dwg[a][4] = s1.x; dwg[a][5] = s1.y; dwg[a][6] = s1.z; dwg[a][7] = s1.w;
        }
        v8i acc[WM][WT];
#pragma unroll
        for (int a = 0; a < WM; ++a)
#pragma unroll
            for (int b = 0; b < WT; ++b) acc[a][b] = (v8i) {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int c = 0; c < 4; ++c) {   // 64 columns per step: 4 WMMA k-steps of 16
            v4i A[WM][2], B[WT][2];
#pragma unroll
            for (int a = 0; a < WM; ++a) {
                const uint8_t* p = wq + ((size_t) (m0 + a) * G + g) * 2048 + l16 * 128 + c * 32;
                A[a][0] = *(const v4i*) p;
                A[a][1] = *(const v4i*) (p + 16);
            }
#pragma unroll
            for (int b = 0; b < WT; ++b) {
                const uint8_t* p = xr[b] + g * 128 + c * 32;
                B[b][0] = *(const v4i*) p;
                B[b][1] = *(const v4i*) (p + 16);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int a = 0; a < WM; ++a)
#pragma unroll
                    for (int b = 0; b < WT; ++b) {
                        const v2i av = {A[a][j >> 1][(j & 1) * 2], A[a][j >> 1][(j & 1) * 2 + 1]};
                        const v2i bv = {B[b][j >> 1][(j & 1) * 2], B[b][j >> 1][(j & 1) * 2 + 1]};
                        acc[a][b] = wmma_i4(av, bv, acc[a][b]);
                    }
            if (AG == 64 || c == 3) {   // the epilogue per activation group
                float dx[WT];
#pragma unroll
                for (int b = 0; b < WT; ++b) dx[b] = xsr[b][AG == 64 ? g * 4 + c : g];
#pragma unroll
                for (int a = 0; a < WM; ++a)
#pragma unroll
                    for (int b = 0; b < WT; ++b)
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            sum[a][b][i] += (float) acc[a][b][i] * (dwg[a][i] * dx[b]);
                            if (AG == 64) acc[a][b][i] = 0;
                        }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < WT; ++b) {
        const int t = t0 + b * 16 + l16;
        if (t >= T) continue;
#pragma unroll
        for (int a = 0; a < WM; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) dst[(size_t) t * ldd + (m0 + a) * 16 + 2 * i + hi] = sum[a][b][i];
    }
}


// ---- diagnostics (TALUS_PF_W4DIAG): which side of W4A4 carries the error
typedef _Float16 h16v __attribute__((ext_vector_type(16)));
typedef float f8v __attribute__((ext_vector_type(8)));

/// R x per 256-value block, unquantized, as FP16 (diagnostic a's activations): grid (ceil(k/256/8), T), block 256.
__global__ void __launch_bounds__(256) k_w4_rot16(const float* __restrict__ x, int k, int64_t ldx,
                                                  const int8_t* __restrict__ signs, _Float16* __restrict__ xh) {
    const int lane = threadIdx.x & 31, G = k / kW4Group, g = blockIdx.x * 8 + (threadIdx.x >> 5), t = blockIdx.y;
    if (g >= G) return;
    const int c0 = g * kW4Group + lane * 8;
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = x[(size_t) t * ldx + c0 + e] * (float) signs[c0 + e];
#pragma unroll
    for (int h = 1; h < 8; h <<= 1)
#pragma unroll
        for (int e = 0; e < 8; ++e)
            if (!(e & h)) { const float a = v[e], b = v[e + h]; v[e] = a + b; v[e + h] = a - b; }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int e = 0; e < 8; ++e) { const float p = __shfl_xor(v[e], m, 32); v[e] = (lane & m) ? p - v[e] : v[e] + p; }
#pragma unroll
    for (int e = 0; e < 8; ++e) xh[(size_t) t * k + c0 + e] = (_Float16) (v[e] * 0.0625f);
}

/// x <- R^T deq(Q4(R x)) per 256-value block, in place (diagnostic b: int4 activations, the original weights).
template <int AG>
__global__ void __launch_bounds__(256) k_w4_fq(float* __restrict__ x, int k, int64_t ldx, const int8_t* __restrict__ signs) {
    const int lane = threadIdx.x & 31, G = k / kW4Group, g = blockIdx.x * 8 + (threadIdx.x >> 5), t = blockIdx.y;
    if (g >= G) return;
    const int c0 = g * kW4Group + lane * 8;
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = x[(size_t) t * ldx + c0 + e] * (float) signs[c0 + e];
    auto fwht = [&]() {
#pragma unroll
        for (int h = 1; h < 8; h <<= 1)
#pragma unroll
            for (int e = 0; e < 8; ++e)
                if (!(e & h)) { const float a = v[e], b = v[e + h]; v[e] = a + b; v[e + h] = a - b; }
#pragma unroll
        for (int m = 1; m < 32; m <<= 1)
#pragma unroll
            for (int e = 0; e < 8; ++e) { const float p = __shfl_xor(v[e], m, 32); v[e] = (lane & m) ? p - v[e] : v[e] + p; }
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] *= 0.0625f;
    };
    fwht();
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(v[e]));
#pragma unroll
    for (int m = 1; m < AG / 8; m <<= 1) amax = fmaxf(amax, __shfl_xor(amax, m, 32));
    const float inv = amax > 0.f ? 7.0f / amax : 0.f, sc = amax / 7.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = fminf(fmaxf(rintf(v[e] * inv), -7.f), 7.f) * sc;
    fwht();   // H / 16 is its own inverse (H H = 256 I); then the signs
#pragma unroll
    for (int e = 0; e < 8; ++e) x[(size_t) t * ldx + c0 + e] = v[e] * (float) signs[c0 + e];
}

/// The int4 sidecar weights against FP16 rotated activations (diagnostic a): f16 WMMA per 16 columns, the group's FP32
/// sum times the weight scale.  Same block / wave layout as k_w4_gemm (2 x 4 tiles per wave).
__global__ void __launch_bounds__(256) k_w4f16_gemm(const uint8_t* __restrict__ wq, const float* __restrict__ ws,
                                                    const _Float16* __restrict__ xh, float* __restrict__ dst, int K, int T,
                                                    int64_t ldd, int wg) {
    constexpr int WM = 2, WT = 4;
    const int G = K / kW4Group;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int wn = w & 3, wt = w >> 2;
    const int m0 = blockIdx.y * 8 + wn * WM, t0 = blockIdx.x * 128 + wt * (16 * WT);
    float sum[WM][WT][8] = {};
    for (int g = 0; g < G; ++g) {
        f8v acc[WM][WT];
#pragma unroll
        for (int a = 0; a < WM; ++a)
#pragma unroll
            for (int b = 0; b < WT; ++b)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[a][b][i] = 0.f;
        const int GW = K / wg, spg = wg / 16;   // weight groups per row, WMMA steps per weight group
        for (int j = 0; j < 16; ++j) {   // 16 columns per WMMA
            h16v A[WM], B[WT];
#pragma unroll
            for (int a = 0; a < WM; ++a) {
                const uint8_t* p = wq + ((size_t) (m0 + a) * G + g) * 2048 + l16 * 128 + j * 8;
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const int nib = (p[e >> 1] >> (4 * (e & 1))) & 0xF;
                    A[a][e] = (_Float16) (float) ((nib ^ 8) - 8);
                }
            }
#pragma unroll
            for (int b = 0; b < WT; ++b) {
                const int t = min(t0 + b * 16 + l16, T - 1);
#pragma unroll
                for (int e = 0; e < 16; ++e) B[b][e] = xh[(size_t) t * K + g * kW4Group + j * 16 + e];
            }
#pragma unroll
            for (int a = 0; a < WM; ++a)
#pragma unroll
                for (int b = 0; b < WT; ++b) {
#if defined(TP_W4_WMMA)
                    acc[a][b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A[a], B[b], acc[a][b]);
#endif
                }
            if ((j + 1) % spg == 0) {   // the weight group ends: its sum times its scale
                const int gw = (g * kW4Group + j * 16) / wg;
#pragma unroll
                for (int a = 0; a < WM; ++a) {
                    const float* sp = ws + ((size_t) (m0 + a) * GW + gw) * 16 + hi * 8;
#pragma unroll
                    for (int b = 0; b < WT; ++b)
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            sum[a][b][i] += acc[a][b][i] * sp[i];
                            acc[a][b][i] = 0.f;
                        }
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < WT; ++b) {
        const int t = t0 + b * 16 + l16;
        if (t >= T) continue;
#pragma unroll
        for (int a = 0; a < WM; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) dst[(size_t) t * ldd + (m0 + a) * 16 + 2 * i + hi] = sum[a][b][i];
    }
}

}  // namespace

void w4_diag_rot16(const float* x, int T, int k, int64_t ldx, const int8_t* signs, void* xh, void* stream) {
    k_w4_rot16<<<dim3((unsigned) ((k / kW4Group + 7) / 8), (unsigned) T), 256, 0, (hipStream_t) stream>>>(x, k, ldx, signs, (_Float16*) xh);
}
void w4_diag_gemm16(const W4Mat& w, const void* xh, int T, float* dst, int64_t ld_dst, void* stream) {
    const dim3 grid((unsigned) ((T + 127) / 128), (unsigned) (w.n / 128));
    k_w4f16_gemm<<<grid, 256, 0, (hipStream_t) stream>>>(w.q, w.s, (const _Float16*) xh, dst, w.k, T, ld_dst, w.wg);
}
void w4_diag_fq(float* x, int T, int k, int64_t ldx, const int8_t* signs, void* stream) {
    const dim3 grid((unsigned) ((k / kW4Group + 7) / 8), (unsigned) T);
    if (w4_act_group() == 64) k_w4_fq<64><<<grid, 256, 0, (hipStream_t) stream>>>(x, k, ldx, signs);
    else k_w4_fq<256><<<grid, 256, 0, (hipStream_t) stream>>>(x, k, ldx, signs);
}

size_t w4_act_bytes(int T, int k) { return (size_t) T * (k / 2) + (size_t) T * (k / 64) * sizeof(float); }

int w4_act_group() {   // TALUS_PF_W4A4_AG=64: one activation scale per 64 values (default 256)
    static const int ag = [] { const char* e = std::getenv("TALUS_PF_W4A4_AG"); return e && std::atoi(e) == 64 ? 64 : 256; }();
    return ag;
}

void w4_pack(const float* x, int T, int k, int64_t ldx, const int8_t* signs, uint8_t* xq, float* xs, void* stream) {
    const int G = k / kW4Group;
    if (w4_act_group() == 64) k_w4_pack<64><<<dim3((unsigned) ((G + 7) / 8), (unsigned) T), 256, 0, (hipStream_t) stream>>>(x, k, ldx, signs, xq, xs);
    else k_w4_pack<256><<<dim3((unsigned) ((G + 7) / 8), (unsigned) T), 256, 0, (hipStream_t) stream>>>(x, k, ldx, signs, xq, xs);
}

void w4_pack_swiglu(const float* g, const float* u, int T, int k, const int8_t* signs, uint8_t* xq, float* xs, void* stream) {
    const int G = k / kW4Group;
    if (w4_act_group() == 64) k_w4_pack_swiglu<64><<<dim3((unsigned) ((G + 7) / 8), (unsigned) T), 256, 0, (hipStream_t) stream>>>(g, u, k, signs, xq, xs);
    else k_w4_pack_swiglu<256><<<dim3((unsigned) ((G + 7) / 8), (unsigned) T), 256, 0, (hipStream_t) stream>>>(g, u, k, signs, xq, xs);
}

void w4_gemm(const W4Mat& w, const uint8_t* xq, const float* xs, int T, float* dst, int64_t ld_dst, void* stream) {
    if (T <= 0) return;
    const dim3 grid((unsigned) ((T + 127) / 128), (unsigned) (w.n / 128));
    if (w4_act_group() == 64) k_w4_gemm<2, 4, 64><<<grid, 256, 0, (hipStream_t) stream>>>(w.q, w.s, xq, xs, dst, w.k, T, ld_dst);
    else k_w4_gemm<2, 4, 256><<<grid, 256, 0, (hipStream_t) stream>>>(w.q, w.s, xq, xs, dst, w.k, T, ld_dst);
}

W4Sidecar::~W4Sidecar() {
    for (void* p : allocs_) hipFree(p);
}

bool W4Sidecar::open(const std::string& path, void* stream) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "tp_w4a4: cannot open %s\n", path.c_str());
        return false;
    }
    auto fail = [&](const char* why) {
        std::fprintf(stderr, "tp_w4a4: %s: %s\n", path.c_str(), why);
        std::fclose(f);
        return false;
    };
    unsigned char hdr[128];
    if (std::fread(hdr, 1, sizeof hdr, f) != sizeof hdr || std::memcmp(hdr, "TW4A", 4) != 0) return fail("not a .tw4 file");
    uint32_t ver, n_t, group, hblock, kmax;
    uint64_t seed, signs_off, table_off, data_off;
    std::memcpy(&ver, hdr + 4, 4);
    std::memcpy(&n_t, hdr + 8, 4);
    std::memcpy(&group, hdr + 12, 4);
    std::memcpy(&hblock, hdr + 16, 4);
    std::memcpy(&kmax, hdr + 20, 4);
    std::memcpy(&seed, hdr + 24, 8);
    std::memcpy(&signs_off, hdr + 32, 8);
    std::memcpy(&table_off, hdr + 40, 8);
    std::memcpy(&data_off, hdr + 48, 8);
    (void) data_off;
    if (ver != 1 || (group != 256 && group != 64 && group != 32) || hblock != (uint32_t) kW4Group)
        return fail("unsupported version / group");
    wg_ = (int) group;
    seed_ = seed;
    kmax_ = (int) kmax;
    std::vector<int8_t> sg(kmax);
    if (std::fseek(f, (long) signs_off, SEEK_SET) || std::fread(sg.data(), 1, kmax, f) != kmax) return fail("short read (signs)");
    if (hipMalloc(&signs_, kmax) != hipSuccess) return fail("hipMalloc");
    allocs_.push_back(signs_);
    hipMemcpy(signs_, sg.data(), kmax, hipMemcpyHostToDevice);
    struct Ent { char name[96]; uint32_t n, k; uint64_t q_off, s_off; };
    static_assert(sizeof(Ent) == 120, "entry layout");
    std::vector<Ent> ents(n_t);
    for (uint32_t i = 0; i < n_t; ++i) {
        unsigned char e[128];
        if (std::fseek(f, (long) (table_off + 128ull * i), SEEK_SET) || std::fread(e, 1, 128, f) != 128) return fail("short read (table)");
        std::memcpy(&ents[i], e, sizeof(Ent));
        ents[i].name[95] = 0;
    }
    std::vector<char> buf;
    for (const Ent& e : ents) {
        const size_t qb = (size_t) e.n * e.k / 2, sb = (size_t) (e.n / 16) * (e.k / wg_) * 16 * sizeof(float);
        void* d = nullptr;
        if (hipMalloc(&d, qb + sb) != hipSuccess) return fail("hipMalloc (tensor)");
        allocs_.push_back(d);
        buf.resize(qb > sb ? qb : sb);
        if (std::fseek(f, (long) e.q_off, SEEK_SET) || std::fread(buf.data(), 1, qb, f) != qb) return fail("short read (q)");
        hipMemcpyAsync(d, buf.data(), qb, hipMemcpyHostToDevice, (hipStream_t) stream);
        hipStreamSynchronize((hipStream_t) stream);
        if (std::fseek(f, (long) e.s_off, SEEK_SET) || std::fread(buf.data(), 1, sb, f) != sb) return fail("short read (s)");
        hipMemcpyAsync((char*) d + qb, buf.data(), sb, hipMemcpyHostToDevice, (hipStream_t) stream);
        hipStreamSynchronize((hipStream_t) stream);
        W4Mat m;
        m.q = (const uint8_t*) d;
        m.s = (const float*) ((char*) d + qb);
        m.n = (int) e.n;
        m.k = (int) e.k;
        m.wg = wg_;
        m_[e.name] = m;
        bytes_ += qb + sb;
    }
    std::fclose(f);
    return true;
}

}  // namespace talus::prefill

extern "C" int talus_pf_w4_pack(const float* x, int T, int k, int64_t ldx, const int8_t* signs, uint8_t* xq, float* xs,
                                void* stream) {
    if (k % talus::prefill::kW4Group != 0) return -1;
    talus::prefill::w4_pack(x, T, k, ldx, signs, xq, xs, stream);
    return 0;
}

extern "C" int talus_pf_w4_gemm(const uint8_t* wq, const float* ws, int n, int k, const uint8_t* xq, const float* xs, int T,
                                float* dst, int64_t ld_dst, void* stream) {
    if (n % 128 != 0 || k % talus::prefill::kW4Group != 0 || !talus::prefill::tp_gfx11_wmma()) return -1;
    talus::prefill::W4Mat m;
    m.q = wq;
    m.s = ws;
    m.n = n;
    m.k = k;
    talus::prefill::w4_gemm(m, xq, xs, T, dst, ld_dst, stream);
    return 0;
}
