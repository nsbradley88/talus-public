// include/strata/prefill/tp_w4a4.hpp - Talus: the opt-in W4A4 prompt GEMM (not exact; gated).
//
// Weights: the offline sidecar (tools/w4a4_sidecar.py, .tw4): per trunk matrix an int4 copy of W R^T, R = H D / 16 per
// 256-column block (H the 256-point Hadamard matrix, D random signs), one FP32 scale per output channel and 256-column
// group.  Activations: the same R applied per token and 256-value group, then symmetric int4 with one FP32 scale per
// token and group (w4_pack).  GEMM: v_wmma_i32_16x16x16_iu4 (16x16x16 int4 WMMA on gfx11), the int32 sum of each group
// scaled by w_scale * x_scale in the epilogue (w4_gemm).
//
// The engine uses it only for prompt chunks of >= 64 rows (tp_tune w4_min_rows) when asked (--prefill-precision fast,
// TALUS_PF_W4A4=1) and the request did not ask for exact; decode, verify windows and draft rounds never see it.
//
// Kernel-library entries (libtalus_kernels via strata_prefill); pointer, shape and stream arguments.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace talus::prefill {

constexpr int kW4Group = 256;

/// One sidecar matrix on the device: q [n/16][k/256][16][128] bytes (int4 pairs), s [n/16][k/256][2][8] FP32.
struct W4Mat {
    const uint8_t* q = nullptr;
    const float* s = nullptr;
    int n = 0, k = 0;
    int wg = kW4Group;   // columns per weight scale (256; a finer sidecar is read by the diagnostic GEMM only)
};

/// Device bytes of a token block's packed activations: xq [T][k/2] + xs [T][k/64] (room for 64-value groups).
size_t w4_act_bytes(int T, int k);
/// Values per activation scale: 256 (default) or 64 (TALUS_PF_W4A4_AG=64).
int w4_act_group();
/// x [T][ldx] (FP32, k columns) -> R x per 256-value block, symmetric int4: xq [T][k/2] (bytes), xs [T][k/AG]
/// (AG = w4_act_group()).
/// `signs`: the sidecar's +-1 per input column (device, >= k of them).
void w4_pack(const float* x, int T, int k, int64_t ldx, const int8_t* signs, uint8_t* xq, float* xs, void* stream);
/// The same, of silu(g) * u (g, u [T][k]): the down projection's input straight from the gate / up products.
void w4_pack_swiglu(const float* g, const float* u, int T, int k, const int8_t* signs, uint8_t* xq, float* xs, void* stream);
/// dst [T][ld_dst] (FP32, the first w.n columns) = W x, from w4_pack's xq / xs.
void w4_gemm(const W4Mat& w, const uint8_t* xq, const float* xs, int T, float* dst, int64_t ld_dst, void* stream);

/// Diagnostics (TALUS_PF_W4DIAG): a) the sidecar's int4 weights against FP16 rotated activations (xh [T][k] FP16 from
/// w4_diag_rot16); b) x <- R^T deq(Q4(R x)) in place, for the original weights' exact path.
void w4_diag_rot16(const float* x, int T, int k, int64_t ldx, const int8_t* signs, void* xh, void* stream);
void w4_diag_gemm16(const W4Mat& w, const void* xh, int T, float* dst, int64_t ld_dst, void* stream);
void w4_diag_fq(float* x, int T, int k, int64_t ldx, const int8_t* signs, void* stream);

/// A .tw4 sidecar loaded to the device.
class W4Sidecar {
public:
    ~W4Sidecar();
    /// Reads `path` (header, signs, every tensor) into device memory; false (with a message) on any error.
    bool open(const std::string& path, void* stream);
    const W4Mat* find(const std::string& tensor_name) const {
        const auto it = m_.find(tensor_name);
        return it == m_.end() ? nullptr : &it->second;
    }
    const int8_t* signs() const { return signs_; }
    int kmax() const { return kmax_; }
    size_t bytes() const { return bytes_; }
    size_t size() const { return m_.size(); }
    uint64_t seed() const { return seed_; }
    int wgroup() const { return wg_; }

private:
    std::unordered_map<std::string, W4Mat> m_;
    std::vector<void*> allocs_;
    int8_t* signs_ = nullptr;
    int kmax_ = 0;
    size_t bytes_ = 0;
    uint64_t seed_ = 0;
    int wg_ = kW4Group;
};

}  // namespace talus::prefill

extern "C" {
/// C ABI (Rust port) of w4_pack / w4_gemm.  Return 0 on success, -1 when unsupported.
int talus_pf_w4_pack(const float* x, int T, int k, int64_t ldx, const int8_t* signs, uint8_t* xq, float* xs, void* stream);
int talus_pf_w4_gemm(const uint8_t* wq, const float* ws, int n, int k, const uint8_t* xq, const float* xs, int T,
                     float* dst, int64_t ld_dst, void* stream);
}
