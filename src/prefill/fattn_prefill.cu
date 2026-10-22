// src/prefill/fattn_prefill.cu - see include/strata/prefill/fattn_prefill.hpp.  llama.cpp's fattn-mma-f16 kernel
// (ggml-cuda, MIT) from the pinned checkout, launched through its own launch_fattn on ggml_tensor descriptors that
// view the engine's buffers (no copies): Q [D, T, H], K/V [D, n_kv, HKV] at the cache's head stride, the output
// [D, H, T] - the engine's [T][H][D].
#include "strata/prefill/fattn_prefill.hpp"

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

// the head size and column tiles this engine uses (llama.cpp's choice for GQA 6: ncols2 = 2)
DECL_FATTN_MMA_F16_CASE(256, 256, 16, 2);
DECL_FATTN_MMA_F16_CASE(256, 256, 32, 2);

// The sparse path's mask compaction lives in fattn.cu; launch_fattn calls it only for use_sparse, which the HIP
// build never selects.
void ggml_cuda_flash_attn_ext_compact_mask(const ggml_tensor*, int32_t*, int32_t*, int32_t, int32_t, int32_t, cudaStream_t) {
    std::fprintf(stderr, "fattn_prefill: sparse FA is not built\n");
    std::abort();
}

// launch_fattn converts a non-F16 K / V through convert.cu's tables; this engine's cache is F16, so they are never
// taken (null: a call would fault loudly).  ggml-cuda.cu's device switch, for the context's stream / pool.
to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type) { return nullptr; }
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type) { return nullptr; }
void ggml_cuda_set_device(int device) {
    int cur = -1;
    CUDA_CHECK(cudaGetDevice(&cur));
    if (cur != device) CUDA_CHECK(cudaSetDevice(device));
}

void strata_fa_f32acc_launch(ggml_backend_cuda_context& ctx, ggml_tensor* dst, int T);   // fattn_prefill_f32.cu

namespace strata::prefill::fattn {
namespace {
constexpr int64_t kStride = FATTN_KQ_STRIDE;

// mask[t][j] = 0 for j <= p0 + t, -inf after (and for every key past n_kv)
__global__ void causal_mask_kernel(half* __restrict__ m, int64_t n_pad, int64_t p0) {
    const int64_t t = blockIdx.y;
    const half ninf = __float2half(-INFINITY), zero = __float2half(0.0f);
    for (int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; j < n_pad; j += (int64_t) gridDim.x * blockDim.x)
        m[t * n_pad + j] = j <= p0 + t ? zero : ninf;
}

struct Ctx {
    std::unique_ptr<ggml_backend_cuda_context> c;
    Ctx() {
        int dev = 0;
        cudaGetDevice(&dev);
        c = std::make_unique<ggml_backend_cuda_context>(dev);
    }
    ~Ctx() {   // the stream belongs to the engine
        for (auto& row : c->streams)
            for (auto& s : row) s = nullptr;
    }
};
Ctx& ctx() {
    static Ctx x;
    return x;
}

void set_tensor(ggml_tensor& t, ggml_type type, const void* data, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1,
                size_t nb2) {
    std::memset(&t, 0, sizeof t);
    t.type = type;
    t.data = const_cast<void*>(data);
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = 1;
    t.nb[0] = ggml_type_size(type);
    t.nb[1] = nb1;
    t.nb[2] = nb2;
    t.nb[3] = nb2 * (size_t) ne2;
}
}  // namespace

bool supported(int head_dim) { return head_dim == 256; }

int64_t kv_padded(int64_t n) { return (n + kStride - 1) / kStride * kStride; }

size_t mask_bytes(int64_t T, int64_t n_kv) { return (size_t) T * (size_t) kv_padded(n_kv) * sizeof(half); }

void attend(const float* q, const void* k, const void* v, float* out, int T, int H, int HKV, int D, int64_t kv_stride,
            int64_t p0, float scale, void* mask, void* stream) {
    if (D != 256 || H % HKV != 0 || (H / HKV) % 2 != 0) {
        std::fprintf(stderr, "fattn_prefill: unsupported shape D %d H %d HKV %d\n", D, H, HKV);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const int64_t n_kv = kv_padded(p0 + T);
    causal_mask_kernel<<<dim3((unsigned) std::min<int64_t>((n_kv + 255) / 256, 64), (unsigned) T), 256, 0, s>>>(
        (half*) mask, n_kv, p0);
    // Talus: the cells past the chunk up to the kernel's 256-key stride hold whatever an earlier sequence left there;
    // they are masked, but on gfx11 the matrix cores' P V sum moves with the V of cells whose P is 0, so the chunk's
    // output would depend on the cache's history (two runs of one prompt differed).  Zero them (K too: never NaN).
    // Past the last head's cells this is the cache's padding; a lower head's are capped at its own cells.
    if (n_kv > p0 + T) {
        const int64_t c0 = p0 + T, c1 = std::min<int64_t>(n_kv, kv_stride);
        for (const void* kv : {k, v}) {
            char* base = (char*) kv;
            for (int h = 0; c1 > c0 && h < HKV - 1; ++h)
                cudaMemsetAsync(base + ((size_t) h * kv_stride + c0) * D * 2, 0, (size_t) (c1 - c0) * D * 2, s);
            cudaMemsetAsync(base + ((size_t) (HKV - 1) * kv_stride + c0) * D * 2, 0, (size_t) (n_kv - c0) * D * 2, s);
        }
    }

    ggml_tensor tq, tk, tv, tm, dst;
    set_tensor(tq, GGML_TYPE_F32, q, D, T, H, (size_t) H * D * 4, (size_t) D * 4);
    set_tensor(tk, GGML_TYPE_F16, k, D, n_kv, HKV, (size_t) D * 2, (size_t) kv_stride * D * 2);
    set_tensor(tv, GGML_TYPE_F16, v, D, n_kv, HKV, (size_t) D * 2, (size_t) kv_stride * D * 2);
    set_tensor(tm, GGML_TYPE_F16, mask, n_kv, T, 1, (size_t) n_kv * 2, (size_t) n_kv * T * 2);
    set_tensor(dst, GGML_TYPE_F32, out, D, H, T, (size_t) D * 4, (size_t) H * D * 4);
    dst.op = GGML_OP_FLASH_ATTN_EXT;
    dst.src[0] = &tq;
    dst.src[1] = &tk;
    dst.src[2] = &tv;
    dst.src[3] = &tm;
    dst.src[4] = nullptr;
    const float params[4] = {scale, 0.0f, 0.0f, 0.0f};   // scale, max_bias, logit_softcap; prec (default) = 0
    std::memcpy(dst.op_params, params, sizeof params);

    Ctx& c = ctx();
    c.c->streams[c.c->device][0] = s;
    c.c->curr_stream_no = 0;
    static const bool f32acc = [] { const char* e = std::getenv("STRATA_FA_F32ACC"); return e && e[0] == '1'; }();
    if (f32acc) strata_fa_f32acc_launch(*c.c, &dst, T);
    else if (T <= 16) ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 16, 2>(*c.c, &dst);
    else ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 32, 2>(*c.c, &dst);
}

}  // namespace strata::prefill::fattn
