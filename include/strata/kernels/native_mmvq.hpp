#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

// Native GGUF Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL
// and IQ4_XS / CUDA Q8_1 adapters, pinned to llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, sm_120 generic MMVQ.
// All pointers are device pointers, at least 4-byte aligned, with no overlap.
// All calls enqueue on the explicit non-null CUDA stream; no allocation or wait.
// The translation unit must use --use_fast_math, as the pinned CUDA oracle does.
//
// Shapes use GGUF order: n_in is the contiguous reduction dimension, n_out is
// the weight row count, and ncols is the activation column/token count, 1..8
// (plan v0.3 P3). Columns are contiguous: activation column j is x + j * n_in
// (its Q8_1 blocks at j * n_in / 32) and output column j is y + j * n_out. See
// native_mmvq_set_multi_exact for how ncols > 1 relates to ncols == 1.
// n_in must be a positive multiple of 32 for quantization/Q4_0/Q5_0/Q8_0/IQ4_NL,
// 64 for Q2_0, or 256 for the other formats. n_out must be positive. Weights remain
// unmodified row-major GGUF blocks: Q3_K=110, Q4_K=144, Q5_K=176, Q6_K=210 and
// IQ4_XS=136 bytes per 256 elements; Q2_0 is 18 bytes per 64 elements.
// Q4_0/IQ4_NL=18, Q5_0=22, Q8_0=34 bytes per 32 elements.
// Q8_1 scratch has 36 bytes per 32 elements, with no extra row padding here.
// Input floats must be finite, and their block scales/sums representable in FP16.
std::size_t native_q8_1_bytes(int n_in, int ncols = 1);

// Layout for ncols > 1. false: llama.cpp's generic multi-column table (upstream), equal to ncols == 1 to
// float rounding, speed not yet measured. true (default): the ncols == 1 layout, every column bitwise equal to a
// single-column call. Set before
// graph capture; captured graphs keep the kernels they captured.
void native_mmvq_set_multi_exact(bool exact);
bool native_mmvq_multi_exact();

// One quantization may serve multiple weight matrices sharing the same input.
// Q8_1 stores FP16 scale and FP16 warp sum of the ORIGINAL float inputs; it does
// not reconstruct that sum from the quantized integers.
void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols,
                          void* stream);

void native_swiglu_quantize_q8_1(const float* gate, const float* up, void* x_q8_1,
                                 int n_in, int ncols, void* stream);

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

// Convenience composition: caller owns scratch sized by native_q8_1_bytes.
void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

/// STRATA_Q8_PACKED=1 (opt-in): a lossless load-time repack of a Q8_0 matrix into a qs plane (n_out x n_in int8)
/// followed by a d plane (n_out x n_in / 32 fp16), the same bytes (34 per 32 values). A registered matrix's
/// native_q8_0_mmvq calls (keyed by its GGUF-layout device pointer, which stays valid for the prompt path) read the
/// packed copy instead; every output is bitwise equal to the GGUF-layout kernels.
bool native_q8_0_packed_enabled();
bool native_q8_0_packed_eligible(int n_in, int n_out);
void native_q8_0_pack_host(const void* gguf_blocks, void* out, int n_in, int n_out);
void native_q8_0_packed_register(const void* gguf_weights, const void* packed, int n_in, int n_out);
void native_q8_0_packed_unregister(const void* gguf_weights);

/// STRATA_Q6_PACKED=1 (opt-in): a packed copy of a Q6_K matrix (the output heads) - the same bytes as ql / qh /
/// scales / d planes - that native_q6_k_mmvq calls on `weights` read instead (bitwise equal outputs). The copy is
/// owned here: native_q6_k_pack builds it from the device matrix (false: not eligible or failed; nothing changes),
/// native_q6_k_unpack frees it. STRATA_Q6P_SELFTEST=1 checks it bitwise and times it at load.
bool native_q6_k_packed_enabled();
bool native_q6_k_pack(const void* weights, int n_in, int n_out, const char* what);
void native_q6_k_unpack(const void* weights);

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

// Storage/dispatch helpers take stable GGML type IDs, avoiding a ggml runtime
// dependency in the engine: Q4_0=2, Q5_0=6, Q8_0=8, Q3_K=11, Q4_K=12, Q5_K=13,
// Q6_K=14, IQ4_NL=20, IQ4_XS=23, Q2_0=42. Unsupported IDs throw in the byte-count
// and launch helpers; only the
// capability query returns false.
bool native_mmvq_supported(int ggml_type) noexcept;
// Talus, gfx11 (wave32): one kernel for 1..8 columns (each column bitwise the single-column result), R rows per wave.
// native_mmvq uses it on gfx11 unless STRATA_MMVQ_W32=0; false: not taken (not gfx11, off, or type not covered).
bool native_mmvq_w32_supported(int ggml_type) noexcept;
/// `ncols` (>= STRATA_MMVQ_LDS; unset = off) columns in one launch with the activations staged in LDS; every column
/// bitwise the single-column wave32 result. Returns false (nothing launched) where it does not apply.
bool native_mmvq_lds(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                     void* stream);
/// `ncols` (>= STRATA_MMVQ_XT; unset = off, up to 64) Q4_K / Q5_K columns in one launch on the gfx11 matrix cores,
/// every column bitwise the single-column wave32 result. Returns false (nothing launched) where it does not apply.
bool native_mmvq_xt(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                    void* stream);
bool native_mmvq_w32(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                     void* stream);
// mode 0 off, 1 on; rows per wave (0: the table).  For benches; set before graph capture.
void native_mmvq_set_w32(int mode, int rows);
// Talus: force the waves-per-EU cap of the w32 instances (-1: the per-arch table), and read the (R, W)
// the table picks for a (type, column count)
void native_mmvq_set_w32_waves(int w);
void native_mmvq_w32_config(int type, int ncols, int* r, int* w);
/// Talus: n (1..3) matrices of type ggml_type with the same input in one launch (each as native_mmvq_w32 computes it);
/// false when not available (STRATA_MMVQ_MULTI=0, a type without wave32 kernels): the caller launches them one by one.
bool native_mmvq_w32_multi(int ggml_type, int n, const void* const* w, float* const* y, const int* n_out,
                           const void* x_q8_1, int n_in, int ncols, void* stream);
bool native_mmvq_w32_active();   // native_mmvq takes the wave32 kernels (packed Q8_0 copies are then unused)
std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out);
void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream);
/// S26 STRATA_LFUSE: native_mmvq(w1 -> y1) and native_mmvq(w2 -> y2), same type / shape / input, in ONE launch,
/// bitwise the two calls. Returns false (nothing launched) where that is not the case (Q8_0, 2-8 columns only).
bool native_mmvq_pair(int ggml_type, const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2,
                      int n_in, int n_out, int ncols, void* stream);

// Fork F4 (Eddoursul): 2..4 columns also INTERLEAVED, so one 16-byte load reads a position of every column: CP = 2 (2
// columns) or 4 (3-4), zero-padded; nb = n_in / 32 blocks; int32 bm[nb][8][CP] (block-major), int32 pm[8][nb][CP]
// (position-major), float d[nb][CP] (each block's scale as the dots read it, __low2float of ds).
constexpr int native_q8_1_il_cp(int ncols) { return ncols <= 2 ? 2 : ncols <= 4 ? 4 : 8; }
std::size_t native_q8_1_il_bytes(int n_in, int ncols);
// Writes the interleaved copy of 2..4 columns of plain q8_1 blocks (native_quantize_q8_1's, or a fused producer's)
// to x_il (native_q8_1_il_bytes): the same values in the same bytes.
void native_q8_1_interleave(const void* x_q8_1, void* x_il, int n_in, int ncols, void* stream);
// True when native_mmvq_il runs its own kernel for this call (IQ4_XS, Q4_K, Q5_K, Q6_K, 2-4 columns, the exact layout).
bool native_mmvq_il_supported(int ggml_type, int ncols, int n_out);
// native_mmvq for 2..4 columns whose copy native_q8_1_interleave wrote, bitwise native_mmvq's output: kernels in which a
// warp takes 1, 2 or 4 rows and reads the columns from the interleaved copy; native_mmvq (x_q8_1) when not supported.
void native_mmvq_il(int ggml_type, const void* weights, const void* x_q8_1, const void* x_il, float* y, int n_in,
                    int n_out, int ncols, void* stream);
// The table's rows a warp (0, 1, 2, 4; 0 = native_mmvq's kernels) for a card of compute capability `cc` (major * 10 + minor).
// Per-architecture tables plus STRATA_MMVQ_IL_ROWS; see native_mmvq.cu and docs/MMVQ_IL_TABLE.md.
int native_mmvq_il_rows_for(int cc, int ggml_type, int ncols, int n_out);
// Tests and benchmarks: every native_mmvq_il call takes `rows` a warp (0: the table).
void native_mmvq_il_tune(int rows);

// Laguna: the mat-vec of n_ids experts picked on the device (llama.cpp's mul_mat_vec_q with ids), graph
// capturable.  Expert ids[k]'s matrix is the raw GGUF rows at weights + ids[k]*expert_bytes; its input is Q8_1
// column k of x_q8_1 when x_per_id (the down projection), else column 0 (gate/up share one input); its output is
// row k of y (n_ids x n_out).  ggml types Q4_K (12), Q5_K (13), Q6_K (14), Q8_0 (8).  weights2/y2 (PAIR): the same
// experts of a second matrix of the same type and layout in the same launch (gate and up), into y2.
void native_mmvq_ids(int ggml_type, const void* weights, std::size_t expert_bytes, const int32_t* ids, int n_ids,
                     const void* x_q8_1, bool x_per_id, float* y, int n_in, int n_out, void* stream,
                     const void* weights2 = nullptr, float* y2 = nullptr);

void native_mmvq_ids_cols(int ggml_type, const void* weights, int ncols, const void* x_q8_1, float* y, int n_in,
                          int n_out, void* stream, const void* weights2 = nullptr, float* y2 = nullptr);

// Laguna: up to 8 mat-vecs of different types (Q4_K/Q5_K/Q6_K/Q8_0) in one launch.  Each item is a
// native_mmvq_ids call (ids null: one dense matrix, n_ids 1).  Items run concurrently, so none may read another's y.
struct MmvqItem {
    int type = 0;                     // ggml type
    const void* w = nullptr;          // raw GGUF rows (expert 0)
    std::size_t expert_bytes = 0;     // stride between experts
    const int32_t* ids = nullptr;     // device expert ids, or null (dense)
    int n_ids = 1;
    const void* x = nullptr;          // Q8_1 input, column k / x_group per id when x_per_id
    int x_per_id = 0;
    int x_group = 1;                  // ids per input column (10: each token's routed experts share its input)
    float* y = nullptr;               // n_ids x n_out
    // block order over the ids: 0 row-block-major (the blocks of one row block are neighbours: repeated ids - a
    // verify's tokens sharing experts - meet in the cache), 1 expert-major (each id's matrix streamed by neighbouring
    // blocks: distinct ids, one token's experts, streamed with contiguous per-expert locality on gfx1151)
    int expert_major = 0;
    int n_in = 0, n_out = 0;
};
void native_mmvq_batch(const MmvqItem* items, int n_items, void* stream);

} // namespace strata::kernels
