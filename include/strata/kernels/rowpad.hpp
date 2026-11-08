// include/strata/kernels/rowpad.hpp - padded copies of f32 weight rows against DRAM channel camping.
// Many blocks reading rows a power of two apart in lockstep (router / gate weights: 12-16 KiB rows, one block or wave
// per row) all hit the same DRAM channel at once; with each row's start moved by a pad the reads spread over the
// channels (rd-exact talus-chan-probe: 114-123 -> 175-198 GB/s).  The copy holds the same bytes per row, so a kernel
// that reads row r at `w + r * ld` instead of `w + r * n_in` computes the same bits.  Opt-in: TALUS_ROUTER_PAD=1
// (512 B) or =N (N bytes, a multiple of 16).  Idea from libstrix (asf0/libstrix, MIT: non-power-of-two strides).
#pragma once

#include <cstdint>

namespace strata::kernels {
/// The padded copy of `w` when one is registered for exactly n_out rows of n_in floats (*ld = its row stride in
/// floats); otherwise `w` itself and *ld = n_in.
const float* rowpad_lookup(const float* w, int64_t n_in, int64_t n_out, int64_t* ld) noexcept;
}  // namespace strata::kernels

extern "C" {
/// Make (or reference again) a padded device copy of `rows` rows of `row_bytes` bytes at `w`.  Returns 1 when a copy
/// is in place, 0 when padding is off (TALUS_ROUTER_PAD unset / 0) or the shape does not qualify, < 0 on error.
/// `stream` orders the copy (nullptr = the null stream); the copy is complete before the next use on that stream.
int talus_rowpad_prepare(const void* w, int64_t rows, int64_t row_bytes, void* stream);
/// Drop one reference; the copy is freed at zero.
void talus_rowpad_release(const void* w);
/// Bytes held by padded copies (diagnostics).
int64_t talus_rowpad_bytes(void);
/// The pad in bytes in effect (0 = off).
int64_t talus_rowpad_pad_bytes(void);
}
