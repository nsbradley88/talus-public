// src/prefill/tp_i8x.cu - see include/strata/prefill/tp_i8gemm.hpp: the weight expansion.  Built into strata_mmq with
// the pinned ggml-cuda sources: each 128-row x 256-column tile goes through MMQ's own tile loader
// (ggml_cuda_mmq_get_load_tiles, the J = 128 configuration: 256 threads, I = 128) into LDS exactly as mul_mat_q fills it,
// and is copied out to the GEMM's layout - the int8 values and per-32 scales MMQ multiplies, bit for bit.
//
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#include "common.cuh"
#include "mmq.cuh"

#include "strata/prefill/tp_i8gemm.hpp"
#include "strata/prefill/tp_tune.hpp"
#include "tp_tiles.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace talus::prefill {
namespace {

constexpr int kJ = 128;

template <ggml_type type, int MODE>
__global__ void __launch_bounds__(256) k_i8x(const char* __restrict__ x, int stride_blocks, int k, int8_t* __restrict__ qs,
                                             float* __restrict__ ws) {
#if defined(AMD_WMMA_AVAILABLE)
    constexpr int I = TpTiles<type>::I;
    constexpr int sram_stride = TpTiles<type>::ss;
    constexpr int blocks_per_iter = TpTiles<type>::blocks_per_iter;
    static_assert(I == 128, "the expansion assumes MMQ's 128-row tile");
    static_assert(type != GGML_TYPE_Q6_K || sram_stride == ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K),
                  "Q6_K: MMQ's J = 128 tile must be the Q6_K layout");
    extern __shared__ int tile[];
    const int row0 = blockIdx.y * I, kiter = blockIdx.x;
    TpTiles<type>::load(x, tile, row0 * stride_blocks + kiter * blocks_per_iter, I - 1, stride_blocks);
    __syncthreads();
    const int KB = k / 32, tid = threadIdx.y * 32 + threadIdx.x;
    for (int e = tid; e < I * 64; e += 256) {   // 64 ints (256 values) per row, in K order
        const int i = e >> 6, kk = e & 63, r = row0 + i;
        const size_t blk = (size_t) (r >> 4) * KB + kiter * 8 + (kk >> 3);
        ((int*) qs)[blk * 128 + (r & 15) * 8 + (kk & 7)] = tile[i * sram_stride + kk];
    }
    constexpr int SC = MODE == 0 ? 1 : 2;   // scales per row and 32 values
    for (int e = tid; e < I * 8 * SC; e += 256) {
        const int i = e / (8 * SC), jj = e % (8 * SC), r = row0 + i;
        const int j = jj / SC, c = jj % SC;   // 32-value block, scale within it
        const size_t blk = (size_t) (r >> 4) * KB + kiter * 8 + j;
        float v;
        if constexpr (MODE == 0) v = ((const float*) (tile + 2 * MMQ_TILE_NE_K))[i * sram_stride + j];
        else if constexpr (type == GGML_TYPE_Q6_K) {
            // MMQ's Q6_K tile holds d once per row (x_df[i * ss]) and 16 int8 scales (x_sc, one per 16
            // values); its WMMA dot is sum += C * sc * d * dB per 16 values, with C * sc an exact int (|.| < 2^24),
            // so round((C * sc) * d) == round(C * (d * sc)) and d * sc (fp16 d x int8) is exact in fp32: the
            // q8_0_16 (mode 1) product with the per-16 scale d * sc is bit for bit MMQ's.
            const float* xdf = (const float*) (tile + 2 * MMQ_TILE_NE_K);
            const int8_t* sc = (const int8_t*) ((const int*) (xdf + MMQ_TILE_NE_K / QI6_K) + i * sram_stride);
            v = xdf[i * sram_stride] * (float) sc[2 * j + c];
        } else if constexpr (MODE == 1) v = ((const float*) (tile + 2 * MMQ_TILE_NE_K))[i * sram_stride + 2 * j + c];   // per 16
        else {
            const float2 dm = __half22float2(((const half2*) (tile + 2 * MMQ_TILE_NE_K))[i * sram_stride + j]);
            v = c == 0 ? dm.x : dm.y;
        }
        ws[(blk * SC + c) * 16 + (r & 1) * 8 + ((r & 15) >> 1)] = v;
    }
#else
    (void) x; (void) stride_blocks; (void) k; (void) qs; (void) ws;
#endif
}

// the largest tile stride of the covered layouts (Q8_0 / Q8_1 76 ints, Q3_K 84; Q2_K's 100 as the bound)
constexpr size_t kTileBytes = (size_t) 128 * ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K) * sizeof(int);

template <ggml_type type, int MODE> void launch(const void* w, int n, int k, I8Mat& out, cudaStream_t s) {
    const int stride_blocks = k / ggml_blck_size(type);
    k_i8x<type, MODE><<<dim3((unsigned) (k / MMQ_ITER_K), (unsigned) (n / 128)), dim3(32, 8), kTileBytes, s>>>(
        (const char*) w, stride_blocks, k, out.qs, out.ws);
}

bool gfx11() {   // (tp_fa.cu's tp_gfx11_wmma lives in strata_prefill, linked after this library)
    static const bool ok = [] {
        int dev = 0;
        cudaDeviceProp p{};
        if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
        for (const char* a : {"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151"}) {
            const size_t n = strlen(a);
            if (strncmp(p.gcnArchName, a, n) == 0 && (p.gcnArchName[n] == '\0' || p.gcnArchName[n] == ':')) return true;
        }
        return false;
    }();
    return ok;
}

}  // namespace

int i8x_mode(int t) {
    switch (t) {
        case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_Q8_0: case GGML_TYPE_Q5_0:
            return 0;
        case GGML_TYPE_Q3_K: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ1_M:
            return 1;
        case GGML_TYPE_Q6_K: {   // mode 1 (see k_i8x); TALUS_PF_I8_Q6K=0 keeps Q6_K on MMQ
            static const bool on = [] { const char* e = std::getenv("TALUS_PF_I8_Q6K"); return e == nullptr || e[0] != '0'; }();
            return on ? 1 : -1;
        }
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
            return 2;
        default:
            return -1;
    }
}

bool i8x_supported(int t) {
    static const bool kq = [] { const char* e = std::getenv("TALUS_PF_I8_KQ"); return e == nullptr || e[0] != '0'; }();
    const int m = i8x_mode(t);
    return m >= 0 && (m == 0 || kq) && gfx11();
}

size_t i8x_bytes(int n, int k, int mode) {
    return (size_t) n * k + (size_t) n * (k / 32) * (mode == 0 ? 1 : 2) * sizeof(float);
}

bool i8x_expand(const void* w, int t, int n, int k, I8Mat& out, void* stream) {
    if (!i8x_supported(t) || n % 128 != 0 || k % MMQ_ITER_K != 0 || out.qs == nullptr || out.ws == nullptr) return false;
    out.n = n;
    out.k = k;
    out.mode = i8x_mode(t);
    const cudaStream_t s = (cudaStream_t) stream;
    switch (t) {
        case GGML_TYPE_IQ4_XS: launch<GGML_TYPE_IQ4_XS, 0>(w, n, k, out, s); break;
        case GGML_TYPE_IQ4_NL: launch<GGML_TYPE_IQ4_NL, 0>(w, n, k, out, s); break;
        case GGML_TYPE_IQ3_S: launch<GGML_TYPE_IQ3_S, 0>(w, n, k, out, s); break;
        case GGML_TYPE_IQ3_XXS: launch<GGML_TYPE_IQ3_XXS, 0>(w, n, k, out, s); break;
        case GGML_TYPE_IQ2_XXS: launch<GGML_TYPE_IQ2_XXS, 0>(w, n, k, out, s); break;
        case GGML_TYPE_Q8_0: launch<GGML_TYPE_Q8_0, 0>(w, n, k, out, s); break;
        case GGML_TYPE_Q5_0: launch<GGML_TYPE_Q5_0, 0>(w, n, k, out, s); break;
#if defined(STRATA_MMQ_KQUANTS)
        case GGML_TYPE_Q3_K: launch<GGML_TYPE_Q3_K, 1>(w, n, k, out, s); break;
        case GGML_TYPE_Q6_K: launch<GGML_TYPE_Q6_K, 1>(w, n, k, out, s); break;
        case GGML_TYPE_Q4_K: launch<GGML_TYPE_Q4_K, 2>(w, n, k, out, s); break;
        case GGML_TYPE_Q5_K: launch<GGML_TYPE_Q5_K, 2>(w, n, k, out, s); break;
#endif
        case GGML_TYPE_IQ2_XS: launch<GGML_TYPE_IQ2_XS, 1>(w, n, k, out, s); break;
        case GGML_TYPE_IQ2_S: launch<GGML_TYPE_IQ2_S, 1>(w, n, k, out, s); break;
        case GGML_TYPE_IQ1_M: launch<GGML_TYPE_IQ1_M, 1>(w, n, k, out, s); break;
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace talus::prefill

extern "C" int talus_pf_i8x_expand(const void* w, int ggml_type, int n, int k, int8_t* qs, float* ws, void* stream) {
    talus::prefill::I8Mat m;
    m.qs = qs;
    m.ws = ws;
    return talus::prefill::i8x_expand(w, ggml_type, n, k, m, stream) ? 0 : -1;
}
