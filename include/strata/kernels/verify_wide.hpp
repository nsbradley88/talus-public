// include/strata/kernels/verify_wide.hpp - verify windows wider than kVerifyMaxTBase (Flash-Next copy
// windows of up to kVerifyMaxT = 16 rows) through the row-wise multi kernels that take at most kVerifyMaxTBase rows.
//
// A group of n > kVerifyMaxTBase rows runs such a kernel in two calls of ceil(n/2) and floor(n/2) rows (each 5..8 rows,
// so each call takes a multi-row instantiation a window of <= kVerifyMaxTBase rows takes too).  Every row's arithmetic
// is the one-call kernel's, so a wide window's rows are bitwise what the narrow windows compute (and so verify(1)'s).
// A group of at most kVerifyMaxTBase rows makes exactly the one call it made before.
#pragma once

#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// f(c0, c) for the chunks [c0, c0 + c) of n rows (one chunk [0, n) when n <= kVerifyMaxTBase)
template <typename F> inline void verify_row_chunks(int n, F&& f) {
    if (n <= kVerifyMaxTBase) { f(0, n); return; }
    const int a = (n + 1) / 2;
    f(0, a);
    f(a, n - a);
}

/// bf16_gemv_fp32_mmvf_multi over n rows (any n <= 2 * kVerifyMaxTBase)
inline void bf16_gemv_fp32_mmvf_multi_wide(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                                           int64_t n_in, int64_t n_out, int n, void* stream) {
    verify_row_chunks(n, [&](int c0, int c) {
        bf16_gemv_fp32_mmvf_multi(x + (size_t) c0 * (size_t) ldx, ldx, w, y + (size_t) c0 * (size_t) ldy, ldy, n_in,
                                  n_out, c, stream);
    });
}

/// fused_gr_read_multi over fa[0, n) (any n <= 2 * kFusedGrMaxT).  `xn_row`: the scratch's floats per row (hc * n_embd).
/// A wide group's calls write no q8_1 image (S26 STRATA_QFUSE: q8_mixed / q8_cnt are cleared, the call returns false
/// and the caller quantizes - the bytes the image would hold).
inline bool fused_gr_read_multi_wide(FusedGrArgs* fa, int n, float* xn, int64_t xn_row, void* stream,
                                     unsigned long long* stamp_buf = nullptr, int stamp_i0 = 0) {
    static_assert(kVerifyMaxTBase <= kFusedGrMaxT, "a chunk must fit the multi read");
    if (n <= kVerifyMaxTBase) return fused_gr_read_multi(fa, n, xn, stream, stamp_buf, stamp_i0);
    for (int t = 0; t < n; ++t) { fa[t].q8_mixed = nullptr; fa[t].q8_cnt = nullptr; }
    verify_row_chunks(n, [&](int c0, int c) {
        fused_gr_read_multi(fa + c0, c, xn + (size_t) c0 * (size_t) xn_row, stream, c0 == 0 ? stamp_buf : nullptr,
                            stamp_i0);
    });
    return false;
}

}  // namespace strata::kernels
