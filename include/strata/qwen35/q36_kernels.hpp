// include/strata/qwen35/q36_kernels.hpp - Qwen3.5 / Qwen3.6 / Qwen3.8 (q36) device kernels and host launchers.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace talus::q36 {

constexpr int kThreads = 256;
constexpr int kChunkCells = 256;   // attention split-K chunk
constexpr int kAttChunk = 512;     // k_attn_wmma's (--attn-bench, gfx1151 at 32K: 512 > 256 > 128 cells)
// Talus: the WMMA attention's split-K chunks, packed in one int `ck` = big chunk | (small units << 16): the
// first `small units` chunks are kAttSmall cells, the rest `big` cells.  The boundaries depend on the cell only (never
// on the depth or T), so a verify row stays bitwise the decode step's; short contexts get four times the blocks.
constexpr int kAttSmall = 128;
constexpr int kAttSmallCells = 4096;   // default: 128-cell chunks below 4K cells (STRATA_ATTN_SMALL=cells, 0 = off)
constexpr int kAttTile = 32;
constexpr int kArgBlocks = 128;
constexpr int kGdnColGroups = 4;

__host__ __device__ __forceinline__ void att_chunk_range(int c, int ck, int& beg, int& len) {
    const int big = ck & 0xffff, sn = ck >> 16;
    if (c < sn) {
        beg = c * kAttSmall;
        len = kAttSmall;
    } else {
        beg = sn * kAttSmall + (c - sn) * big;
        len = big;
    }
}

__host__ __device__ __forceinline__ int att_chunk_count(int npos, int ck) {
    const int big = ck & 0xffff, sn = ck >> 16, s = sn * kAttSmall;
    return npos <= s ? (npos + kAttSmall - 1) / kAttSmall : sn + (npos - s + big - 1) / big;
}

inline int att_vt_cells(int ctx, int ck) {   // the cells the chunks cover (the V copy's size)
    int beg, len;
    att_chunk_range(att_chunk_count(ctx, ck) - 1, ck, beg, len);
    return beg + len;
}

/// The cell-blocked V copy of k_attn_wmma: element (kv head, cell, d) of a cache with vt_ctx cells.
__host__ __device__ __forceinline__ size_t vt_index(int kvh, int cell, int d, int vt_ctx) {
    return (((size_t) kvh * (vt_ctx / 16) + cell / 16) * 256 + d) * 16 + cell % 16;
}

inline unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }
inline unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }

struct TqQ81 {
    __half2 ds;
    int8_t qs[32];
};

struct DraftSamp {
    float tau = 0.f;
    uint32_t seed = 0, ctr = 0;
    float lmin = -INFINITY;   // Talus: ln(min_p): drafts only from tokens with l / tau >= max(l / tau) + lmin (-inf: all)
    float topp = 1.f;         // Talus (talus-next, STRATA_DRAFT_NUCLEUS): the request's top-p / top-k (draft_support)
    int topk = 0;
};

struct GdnPend {
    float* q[2] = {};
    float* h[2] = {};
    float* a[2] = {};
    float* b[2] = {};
    const int32_t* ctl = nullptr;   // nullptr: no replay, nothing stored
    bool store = false;             // store this call's rows (a verify window)
};

extern bool g_attn_w8;

void launch_rms(int rows, cudaStream_t st, const float* x, const float* w, float* y, int n, float eps,
                void* q8 = nullptr);
void launch_rms_ld(int rows, cudaStream_t st, const float* x, int ldx, const float* w, float* out, int ld_out, int n,
                   float eps, void* q8 = nullptr);
void launch_add_rms(int rows, cudaStream_t st, float* x, const float* y, const float* w, float* h, int n, float eps,
                    void* q8 = nullptr);
void launch_add_rms_q8(int rows, cudaStream_t st, float* x, const float* y, const float* w, float* h, void* q8, int n,
                       float eps);
void launch_moe_out(int rows, cudaStream_t st, float* x, const float* parts, const int32_t* prow, const float* wts,
                    int K_, const float* shared, const float* sgate, int sg_stride, const float* wn, float* h, int n,
                    float eps, void* q8 = nullptr);
void launch_gdn_ab(int T, cudaStream_t st, const float* Wa, const float* Wb, const float* x, const float* dt,
                   const float* ssm_a, float* alpha, float* beta, int n_in, int HV);
void launch_gdn_ab2(cudaStream_t st, const float* Wa, const float* Wb, const float* x, const float* dt,
                    const float* ssm_a, float* alpha, float* beta, int n_in, int HV, int T);
void launch_add(cudaStream_t st, float* x, const float* y, int64_t n);
void launch_swiglu(cudaStream_t st, const float* g, const float* u, float* h, int64_t n);
void launch_gemv_f32(int T, cudaStream_t st, const float* W, const float* x, float* y, int n_in, int n_out);
void launch_gdn_gb(cudaStream_t st, float* alpha, const float* dt, const float* ssm_a, float* beta, int HV, int T);
void launch_gdn_gb2(cudaStream_t st, const float* ab, const float* dt, const float* ssm_a, float* alpha,
                    float* beta, int HV, int T);
void launch_gate_rows(cudaStream_t st, float* sh, const float* gate, int n, int T);
void launch_attn_prep(int T, cudaStream_t st, const float* qg, const float* kx, const float* vx, const float* qw,
                      const float* kw, float* q, float* gate, __half* kc, __half* vc, int H, int HKV, int HD,
                      int n_rot, const int32_t* pos0, int ctx, float base, float eps, __half* vt = nullptr,
                      int vt_ctx = 0);
void launch_attn_part(dim3 grid, cudaStream_t st, const float* q, const __half* kc, const __half* vc,
                      const int32_t* pos0, int t0, int H, int HKV, int ctx, float scale, int n_chunk, float* pacc,
                      float* pm, float* pl);
void launch_attn_merge(int rows, int HD, cudaStream_t st, const float* pacc, const float* pm, const float* pl,
                       const float* gate, const int32_t* pos0, int t0, int H, int n_chunk, float* out,
                       int chunk = kChunkCells, void* q8 = nullptr);
void launch_attn_wmma(bool w8, dim3 grid, cudaStream_t st, const float* q, const __half* kc, const __half* vt,
                      const int32_t* pos0, int T, int H, int HKV, int ctx, int vt_ctx, float sc, int chunk, int n_chunk,
                      float* pacc, float* pm, float* pl);
void launch_group(cudaStream_t st, const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs,
                  unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                  int32_t* ent_dst, int32_t* ent_tok);
void launch_group_hy(cudaStream_t st, const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                     const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                     int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok);
void launch_merge_cpu(int n_ent, cudaStream_t st, float* parts, const float* cpu, const int32_t* miss, int n);
void launch_attn_softmax16(int H, cudaStream_t st, const float* S, __half* S16, int ld, int T, int P, int pos0,
                           float scale);
void launch_gate_mul(cudaStream_t st, float* att, const float* gate, int64_t n);
void launch_route_count(cudaStream_t st, const int32_t* ids, int n_ent, int32_t* cnt);
void launch_route_scan(cudaStream_t st, const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur,
                       int32_t* max_rows);
void launch_route_scatter(cudaStream_t st, const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst,
                          int32_t* src, int32_t* inv);
void launch_argmax(dim3 grid, cudaStream_t st, const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2,
                   const int32_t* map = nullptr);
void launch_argmax_samp(cudaStream_t st, const float* x, int n, float* pv, int32_t* pi, const DraftSamp* ds, int step,
                        const float* thr = nullptr);
void launch_lse(cudaStream_t st, const float* x, int n, const DraftSamp* ds, float* out, float* thr_out = nullptr);
void launch_ctl_end(cudaStream_t st, int32_t* ctl, bool flip);
void launch_round_prep(cudaStream_t st, const int32_t* rctl, int32_t* tok, int32_t* tok_m, int32_t* pos0,
                       int32_t* pos0_m, int32_t* gctl, int n);
void launch_row_copy(cudaStream_t st, float* dst, const float* src, const int32_t* idx, int n);
void launch_chain_prep(cudaStream_t st, const int32_t* tok, int32_t* tok_m, const int32_t* pos0, int32_t* pos0_m,
                       int k);
void launch_sgemm_nt(int tm, dim3 grid, cudaStream_t st, const float* W, const float* x, float* out, int K, int n_out,
                     int T, int ldo, int kz);
void launch_sum_splits(cudaStream_t st, const float* part, float* out, int64_t n, int splits);
void launch_to_f16(cudaStream_t st, const float* x, __half* y, int64_t n);
void launch_conv_seq(cudaStream_t st, float* hist, const float* qkv, const float* w, float* h, int C, int qk_heads,
                     float eps, int T, bool write, GdnPend pd = GdnPend());
void launch_conv_ab(cudaStream_t st, float* hist, const float* qkv, const float* w, float* hc, int C, int qk_heads,
                    float eps, int T, bool write, GdnPend pd, const float* Wa, const float* Wb, const float* x,
                    const float* dt, const float* ssm_a, float* alpha, float* beta, int n_in, int HV);
void launch_conv_par(cudaStream_t st, const float* hist, const float* qkv, const float* w, float* h, int C,
                     int qk_heads, float eps, int T);
void launch_conv_hist(cudaStream_t st, float* hist, const float* qkv, int C, int T);
void launch_gdn_seq(cudaStream_t st, float* state, const float* h, int C, const float* gate, const float* beta,
                    const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int T, bool write,
                    GdnPend pd = GdnPend(), void* yq = nullptr);
void launch_gdn_seq_cols(cudaStream_t st, float* state, const float* h, int C, const float* gate, const float* beta,
                         float* o_raw, int h_k, int h_v, int T);
void launch_gdn_seq_warp(int ncol, cudaStream_t st, float* state, const float* h, int C, const float* gate,
                         const float* beta, float* o_raw, int h_k, int h_v, int T);
void launch_gdn_norm(int T, cudaStream_t st, const float* o_raw, const float* z, const float* gamma, float eps,
                     float* y, int h_v);
void launch_flush_read(cudaStream_t st, const float4* p, size_t n, float* out);

}   // namespace talus::q36
