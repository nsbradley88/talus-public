// include/strata/prefill/tp_i8gemm.hpp - Talus: the exact int8 prompt GEMM on pre-expanded weights.
//
// llama.cpp's MMQ decodes the GGUF blocks into LDS on every tile (and syncs around it); here a weight matrix is expanded
// once, at load, into int8 values plus one FP32 scale per 32 values (`i8x`), taken from MMQ's own tile loader - the
// very int8 values and scales MMQ multiplies - and the GEMM reads them straight into the WMMA operands.  Per output
// element it forms the same integer dot per 32-value block and adds (float) dot * w_scale * x_scale in the same block
// order as MMQ, on MMQ's own q8_1 activations: the output is bitwise MMQ's.
//
// Covered: IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_XXS, Q8_0, Q5_0 (MMQ's q8_0 dot), Q3_K, IQ2_XS, IQ2_S (q8_0_16), Q4_K,
// Q5_K (q8_1).  Q2_K / Q6_K stay on MMQ.  Memory: 1.125 (1.25 for modes 1 / 2) bytes per weight.
//
// Kernel-library entries (libtalus_kernels via strata_mmq / strata_prefill); pointer, shape and stream arguments.
#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace talus::prefill {

/// One expanded matrix [n][k] on the device: qs int8 in 16-row x 32-column blocks ([n/16][k/32][16][32]), ws FP32
/// scales [n/16][k/32][2][8] (rows 0, 2, .., 14 then 1, 3, .., 15 of each block).
struct I8Mat {
    int8_t* qs = nullptr;
    float* ws = nullptr;
    int n = 0, k = 0;
    int mode = 0;   // the MMQ dot it reproduces: 0 q8_0 (one FP32 scale per 32), 1 q8_0_16 (two per 32), 2 q8_1 (d, m)
};

/// `ggml_type` can be expanded (and this build has the expansion and the gfx11 kernel for this device).
bool i8x_supported(int ggml_type);
/// The expansion mode of `ggml_type` (I8Mat::mode), -1 if not covered.
int i8x_mode(int ggml_type);
/// Device bytes of an expanded [n][k] matrix (n % 128 == 0, k % 256 == 0): qs, then the scales (16 floats per 16-row x
/// 32-column block in mode 0, 32 in modes 1 and 2).
size_t i8x_bytes(int n, int k, int mode = 0);
/// Expand the GGUF matrix `w` ([n][k] of `ggml_type`, device memory) into `out` (qs / ws allocated by the caller,
/// i8x_bytes(n, k) in all: qs first, then ws).  false: unsupported type or shape.
bool i8x_expand(const void* w, int ggml_type, int n, int k, I8Mat& out, void* stream);

/// dst [T][ld_dst] (FP32, the first w.n columns) = W x for the T activation rows `xq`: MMQ's q8_1 activations of the
/// matching layout (strata::prefill::mmq::quantize for the weights' type: blocks of 128 values, [k_pad / 128][T]).
void i8gemm(const I8Mat& w, const void* xq, int T, float* dst, int64_t ld_dst, void* stream);

/// The engine's expanded matrices, keyed by the GGUF weights' device pointer (owns the device memory).
class I8Cache {
public:
    ~I8Cache();
    /// Expand and keep `w` if its type and shape qualify; returns the bytes it took (0 when it was not added).
    size_t add(const void* w, int ggml_type, int n, int k, void* stream);
    const I8Mat* find(const void* w) const {
        const auto it = m_.find(w);
        return it == m_.end() ? nullptr : &it->second;
    }
    size_t bytes() const { return bytes_; }
    size_t size() const { return m_.size(); }

private:
    std::unordered_map<const void*, I8Mat> m_;
    size_t bytes_ = 0;
};

}  // namespace talus::prefill

extern "C" {
/// C ABI (Rust port): expansion and GEMM as above.  Return 0 on success, -1 when unsupported.
int talus_pf_i8x_expand(const void* w, int ggml_type, int n, int k, int8_t* qs, float* ws, void* stream);
/// (mode: talus::prefill::i8x_mode of the weights' type)
int talus_pf_i8gemm(const int8_t* qs, const float* ws, int n, int k, int mode, const void* xq, int T, float* dst,
                    int64_t ld_dst, void* stream);
}
