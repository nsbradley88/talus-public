// talus_wide.cuh - Talus exact wide (verify-width) mat-vec tiles on the gfx11 matrix cores for the 27B's
// i-quant / K-quant weights (IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XS, IQ2_S, Q4_K, Q5_K).  Included by native_mmvq.cu inside
// namespace strata::kernels (it uses Q81Block, kq_scale_min, xt_kq8_val, w32_on, g_kq8, launch_check, validate_shape).
//
// What a column's value must be (bitwise): the wave32 kernel (w32_body, T = 8 traits: IQ4XSTraits, IqTraits<TY>,
// KQ8Traits) gives lane l = 8 c + j sub-block j of superblocks sb = c, c + 4, ... in that order, accumulates each
// sub-block's value v(row, sb, j, col) into its lane sum p(c, j), then reduces the 32 lane sums with the XOR butterfly
// 16 .. 1, which is talus_wide::tree(p) (see talus_wide_fmt.hpp).  v is the traits' integer dot of the sub-block (exact,
// so any grouping of its 32 products gives the same int), an integer scale step, and a float epilogue.  The R / W / CG
// choice of the wave32 table does not change any of this (--mmvq-bench proved every instance bitwise equal).
//
// Here (one block = 16 rows, 4 waves): wave c runs chain c (superblocks c, c + 4, ... in order) for all 8 sub-blocks,
// so a wave reads its rows' whole superblocks (contiguous bytes); the 32-value integer dots of 16 rows x 16 columns
// come from two v_wmma_i32_16x16x16_iu8 (values 0..15 and 16..31: exact), the integer step and the float epilogue run
// per (row, column) in the wave32 kernel's operations and order (pinned to its gfx1151 ISA), into p(c, j); the four
// chains meet in LDS and talus_wide::tree finishes each output.  Activations are read once per (wave, superblock):
// lane l loads column l % 16's q8_1 blocks (lanes 16..31 the same addresses), instead of once per row group per column.
// Weights: GGUF rows, or (TALUS_MMVQ_WIDE_PACK, load time) the tile-plane copy of talus_wide_fmt.hpp - idea from tubo
// (BonkusCheemus/tubo, MIT): a byte-exact load-time repack so neighbouring lanes read one cache line.
// WMMA operands (gfx11 wave32): A = activations (lane l: column l % 16, 16 k), B = weights (lane l: row l % 16),
// D lane l holds D[col 2 v + l / 16][row l % 16] (v = 0..7).
#pragma once

#include "strata/kernels/talus_wide_fmt.hpp"

#ifndef TALUS_TW_SCHED
#define TALUS_TW_SCHED 1
#endif
namespace {
namespace tw = talus_wide;

// The float epilogue of an i-quant / IQ4_XS sub-block (kind 0), as IQ4XSTraits / Split<TY>::apply and w32_body compile
// on gfx1151: d = dw * d8 (v_mul_f32), then tmp += d * float(sumi) (v_cvt_f32_i32 + v_fmac_f32: the multiply and the
// add contract into one fma).
__device__ __forceinline__ float tw_acc0(float acc, float dw, float d8, int sumi) {
    const float d = __fmul_rn(dw, d8);
    return __builtin_fmaf(d, float(sumi), acc);
}

template<int I, int N, typename Fn>
__device__ __forceinline__ void tw_static_for(Fn&& f) {
    if constexpr (I < N) {
        f(std::integral_constant<int, I>{});
        tw_static_for<I + 1, N>(f);
    }
}
// the B operand of sub-block j with the decode split between the half-waves (X::kSplit): lanes 0-15 decode
// words 0..3 of dec(j), lanes 16-31 words 4..7 (dec_half), one permlanex16 per word swaps the halves, and a select puts
// each word where dec(j) has it - the same 8 words in every lane as rec.dec(j, bv) (the WMMA operands must be the same
// in both half-waves).  Integer only: no effect on any sum.
template<typename X>
__device__ __forceinline__ void tw_dec_split(const X& rec, int j, int hi, int bv[8]) {
    int own[4];
    rec.dec_half(j, hi, own);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int other = __builtin_amdgcn_permlanex16(own[k], own[k], 0x76543210u, 0xfedcba98u, false, false);
        bv[k] = hi ? other : own[k];
        bv[k + 4] = hi ? own[k] : other;
    }
}
template<typename X, bool PACKED, bool SPLIT = false>
__device__ __forceinline__ void tw_body(const uint8_t* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                                        int n_in, int n_out, int ncols, int rt) {
#if STRATA_XT_W11
    typedef int i4 __attribute__((ext_vector_type(4)));
    typedef int i8 __attribute__((ext_vector_type(8)));
    __shared__ float2 cs[4][8][16];      // per wave, per sub-block: (d8, activation sum) of the 16 columns, slot (col & 1) * 8 + col / 2
    __shared__ float part[8][8][32];     // one wave's p(c, j, v) per lane (the chains' staged reduction)
    const int tid = int(threadIdx.x), lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int row = rt * tw::kTileRows + l16;
    const int rowc = min(row, n_out - 1);
    const int bpr = n_in / 256, xs = n_in / 32;
    const Q81Block* __restrict__ xc = x + std::size_t(l16 < ncols ? l16 : 0) * xs;
    float acc[8][8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) acc[j][v] = 0.0f;
    for (int sb = wave; sb < bpr; sb += 4) {
        X rec;
        if constexpr (PACKED) {
            const tw::Loader<true, X::BS> L{w + (std::size_t(rt) * bpr + sb) * (tw::kTileRows * X::BS), l16};
            rec.load(L);
        } else {
            const tw::Loader<false, X::BS> L{w + (std::size_t(rowc) * bpr + sb) * X::BS, 0};
            rec.load(L);
        }
        float dw = 0.0f, dmx = 0.0f, ndmy = 0.0f;
        if constexpr (X::kind == 0) {
            dw = __half2float(__ushort_as_half((unsigned short) rec.d16()));
            // the columns' q8_1 scales of the 8 sub-blocks, to LDS (lanes l and l + 16 write the same value: no branch)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float d8 = __low2float(xc[8 * sb + j].ds);   // (a column >= ncols reads column 0: never stored)
                cs[wave][j][(l16 & 1) * 8 + (l16 >> 1)] = make_float2(d8, 0.0f);
            }
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        } else {
            const float2 dm = __half22float2(*reinterpret_cast<const half2*>(&rec.dm));
            dmx = dm.x;
            ndmy = -dm.y;
        }
        // (a compile-time loop: `#pragma unroll` left the KQ8 instance rolled, acc[j] then went to scratch)
        tw_static_for<0, 8>([&](auto jc) {
            constexpr int j = decltype(jc)::value;
            // A: this lane's column, q8_1 block 8 sb + j (lanes 16..31 load the same addresses)
            // (no branch on the column: a column >= ncols reads column 0's blocks; its D entries are never stored)
            const Q81Block* __restrict__ xb = xc + 8 * sb + j;
            const int4* u4 = reinterpret_cast<const int4*>(xb->qs);
            const int4 p0 = u4[0], p1 = u4[1];
            const i4 a0 = i4{p0.x, p0.y, p0.z, p0.w}, a1 = i4{p1.x, p1.y, p1.z, p1.w};
            if constexpr (X::kind == 1) {   // KQ8: (d8, the block's quant sum) per column, as native_mmvq_xt_kq8_kernel
                int dot2 = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) dot2 = STRATA_DP4A(0x01010101, a0[i], dot2);
#pragma unroll
                for (int i = 0; i < 4; ++i) dot2 = STRATA_DP4A(0x01010101, a1[i], dot2);
                const float d8 = __low2float(xb->ds);
                cs[wave][j][(l16 & 1) * 8 + (l16 >> 1)] = make_float2(d8, float(dot2));
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
            }
            int bv[8];
            if constexpr (SPLIT && X::kSplit) tw_dec_split(rec, j, hi, bv);
            else rec.dec(j, bv);
            const i4 b0 = i4{bv[0], bv[1], bv[2], bv[3]}, b1 = i4{bv[4], bv[5], bv[6], bv[7]};
            const i8 z = {0, 0, 0, 0, 0, 0, 0, 0};
            i8 dA, dB = z;
            if constexpr (X::kHalves) {
                dA = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, z, false);
                dB = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, z, false);
            } else {
                dA = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, z, false);
                dA = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, dA, false);
            }
            const float4* cp = reinterpret_cast<const float4*>(&cs[wave][j][hi * 8]);
            const float4 c0 = cp[0], c1 = cp[1], c2 = cp[2], c3 = cp[3];
            const float cd8[8] = {c0.x, c0.z, c1.x, c1.z, c2.x, c2.z, c3.x, c3.z};
            if constexpr (X::kind == 0) {
#pragma unroll
                for (int v = 0; v < 8; ++v) acc[j][v] = tw_acc0(acc[j][v], dw, cd8[v], rec.sumi(j, dA[v], dB[v]));
            } else {
                const float cs2[8] = {c0.y, c0.w, c1.y, c1.w, c2.y, c2.w, c3.y, c3.w};
                int sc, m;
                rec.scale_min(j, sc, m);
                const float scf = float(sc), mf = float(m);
#pragma unroll
                for (int v = 0; v < 8; ++v)
                    acc[j][v] = __fadd_rn(acc[j][v], xt_kq8_val(float(dA[v]), cs2[v], scf, mf, cd8[v], dmx, ndmy));
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
            }
#if TALUS_TW_SCHED
            // keep the scheduler from hoisting every sub-block's loads and decode to the top (VGPRs: 221-256 with
            // spills for IQ2 without it): one sub-block's operands live at a time, other waves hide the latency
            __builtin_amdgcn_sched_barrier(0);
#endif
        });
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    }
    // the chains meet: t = p0 + p2 (wave 0), u = p1 + p3 (wave 1), q_j = t + u, then the tree over j (wave 0)
    auto put = [&](int wv) {
        if (wave == wv)
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int v = 0; v < 8; ++v) part[j][v][lane] = acc[j][v];
    };
    auto add = [&](int wv) {   // acc = acc + staged (the staged chain is the right operand: p0 + p2, p1 + p3, t + u)
        if (wave == wv)
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int v = 0; v < 8; ++v) acc[j][v] = acc[j][v] + part[j][v][lane];
    };
    put(2); __syncthreads(); add(0); __syncthreads();
    put(3); __syncthreads(); add(1); __syncthreads();
    put(1); __syncthreads(); add(0);
    if (wave == 0 && row < n_out) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const float s = ((acc[0][v] + acc[4][v]) + (acc[2][v] + acc[6][v])) + ((acc[1][v] + acc[5][v]) + (acc[3][v] + acc[7][v]));
            const int col = 2 * v + hi;
            if (col < ncols) y[std::size_t(col) * n_out + row] = s;
        }
    }
#else
    (void) w; (void) x; (void) y; (void) n_in; (void) n_out; (void) ncols; (void) rt;
#endif
}
template<typename X, bool PACKED>
__launch_bounds__(128)
__global__ void native_mmvq_tw_kernel(const uint8_t* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                                      int n_in, int n_out, int ncols) {
    tw_body<X, PACKED>(w, x, y, n_in, n_out, ncols, int(blockIdx.x));
}
// the tiles with the split decode (TALUS_MMVQ_WIDE_TAB mode 4; a separate kernel so mode 1 / 2 stay as built)
template<typename X, bool PACKED>
__launch_bounds__(128)
__global__ void native_mmvq_tws_kernel(const uint8_t* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                                       int n_in, int n_out, int ncols) {
    tw_body<X, PACKED, true>(w, x, y, n_in, n_out, ncols, int(blockIdx.x));
}
// up to 3 matrices of one type and layout reading the same activations (gate + up, qkv + z) in one launch (idea from
// tubo, BonkusCheemus/tubo, MIT: packed gate+up): block b runs matrix i's tile b - start[i], as its own launch would.
struct TwMulti {
    const uint8_t* w[3];
    float* y[3];
    int n_out[3];
    int start[3];
};
template<typename X, bool PACKED>
__launch_bounds__(128)
__global__ void native_mmvq_tw_multi_kernel(TwMulti m, const Q81Block* __restrict__ x, int n_in, int ncols) {
    const int b = int(blockIdx.x);
    const int i = b >= m.start[2] ? 2 : (b >= m.start[1] ? 1 : 0);
    tw_body<X, PACKED>(m.w[i], x, m.y[i], n_in, m.n_out[i], ncols, b - m.start[i]);
}
template<typename X, bool PACKED>
__launch_bounds__(128)
__global__ void native_mmvq_tws_multi_kernel(TwMulti m, const Q81Block* __restrict__ x, int n_in, int ncols) {
    const int b = int(blockIdx.x);
    const int i = b >= m.start[2] ? 2 : (b >= m.start[1] ? 1 : 0);
    tw_body<X, PACKED, true>(m.w[i], x, m.y[i], n_in, m.n_out[i], ncols, b - m.start[i]);
}

// ---- narrow cells (TALUS_MMVQ_WIDE_TAB mode 3): one row per lane, dp4a, wave-uniform activations ----
// On gfx11 v_wmma_i32_16x16x16_iu8 runs at the v_dot4 rate (128 MACs / clk / SIMD), so the 16 x 16 tiles above pay the
// full 16-column WMMA time at <= 8 columns, decode every row twice (both half-waves) and run 16 columns of epilogue.
// Here a block is 32 rows x 4 waves: wave c runs chain c (superblocks c, c + 4, ...), lane l row 32 b + l (both tile
// planes of the packed copy: lanes 0-15 tile 2 b, 16-31 tile 2 b + 1 - one 256-byte run per half-wave per load).  A
// lane decodes its row once and dots it with each column by v_dot4 (dA: values 0..15, dB: 16..31, exact ints, so
// sumi(j, dA, dB) is the tiles' and the wave32 traits' integer); the column's q8_1 blocks are the same for the whole
// wave (sb, j and the column are wave-uniform), so they come through the scalar cache (s_load), not per lane.  Per
// (row, column): the same p(c, j) = sequential chain sums with the tiles' pinned epilogue (tw_acc0 / xt_kq8_val), the
// same staged chain reduction and tree, so every column is bitwise the 1-column wave32 result.
template<int I, int N, typename Fn>
__device__ __forceinline__ void tr_static_for(Fn&& f) {
    if constexpr (I < N) {
        f(std::integral_constant<int, I>{});
        tr_static_for<I + 1, N>(f);
    }
}
template<typename X, int NC, bool PACKED>
__device__ __forceinline__ void tr_body(const uint8_t* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                                        int n_in, int n_out, int rb) {
    __shared__ float part[8][NC][32];   // one wave's p(c, j, col) per lane (the chains' staged reduction)
    const int tid = int(threadIdx.x), lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);   // (wave-uniform: sb and the activation addresses too)
    const int row = rb * 32 + lane;
    const int bpr = n_in / 256, xs = n_in / 32;
    const int ntile = (n_out + tw::kTileRows - 1) / tw::kTileRows;
    const int tile = min(2 * rb + (lane >> 4), ntile - 1);   // (a lane past the last tile re-reads it: never stored)
    const int rowc = min(row, n_out - 1);
    float acc[8][NC];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[j][c] = 0.0f;
    for (int sb = wave; sb < bpr; sb += 4) {
        X rec;
        if constexpr (PACKED) {
            const tw::Loader<true, X::BS> L{w + (std::size_t(tile) * bpr + sb) * (tw::kTileRows * X::BS), lane & 15};
            rec.load(L);
        } else {
            const tw::Loader<false, X::BS> L{w + (std::size_t(rowc) * bpr + sb) * X::BS, 0};
            rec.load(L);
        }
        float dw = 0.0f, dmx = 0.0f, ndmy = 0.0f;
        // KQ8: the blocks' quant sums (wave-uniform) once per superblock, lane-parallel: lane l sums block
        // (j, c) = (l % 8, l / 8) (+ 32 k: c + 4 k), v_readlane hands them out below (exact ints)
        constexpr int ND2 = X::kind == 1 ? (8 * NC + 31) / 32 : 1;
        int d2v[ND2];
        if constexpr (X::kind == 0) {
            dw = __half2float(__ushort_as_half((unsigned short) rec.d16()));
            d2v[0] = 0;
        } else {
            const float2 dm = __half22float2(*reinterpret_cast<const half2*>(&rec.dm));
            dmx = dm.x;
            ndmy = -dm.y;
#pragma unroll
            for (int k = 0; k < ND2; ++k) {
                const int bi = min(lane + 32 * k, 8 * NC - 1);
                const int4* u4 = reinterpret_cast<const int4*>(x[std::size_t(bi >> 3) * xs + 8 * sb + (bi & 7)].qs);
                const int4 p0 = u4[0], p1 = u4[1];
                int t = 0;
                t = STRATA_DP4A(0x01010101, p0.x, t); t = STRATA_DP4A(0x01010101, p0.y, t);
                t = STRATA_DP4A(0x01010101, p0.z, t); t = STRATA_DP4A(0x01010101, p0.w, t);
                t = STRATA_DP4A(0x01010101, p1.x, t); t = STRATA_DP4A(0x01010101, p1.y, t);
                t = STRATA_DP4A(0x01010101, p1.z, t); t = STRATA_DP4A(0x01010101, p1.w, t);
                d2v[k] = t;
            }
        }
        // one (sub-block j, column c) step: q = the q8_1 block's 8 quant words, ds = its (d, s) half2 bits
        auto step = [&](auto jc, auto cc, const int* bv, const auto& q, uint32_t ds, float scf, float mf) {
            constexpr int j = decltype(jc)::value, c = decltype(cc)::value;
            // (an int dot: any grouping is the same value; the 16-value halves only where sumi needs them)
            int dA = 0, dB = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) dA = STRATA_DP4A(bv[i], int(q[i]), dA);
#pragma unroll
            for (int i = 4; i < 8; ++i) {
                if constexpr (X::kHalves) dB = STRATA_DP4A(bv[i], int(q[i]), dB);
                else dA = STRATA_DP4A(bv[i], int(q[i]), dA);
            }
            const float d8 = __half2float(__ushort_as_half((unsigned short) (ds & 0xffffu)));   // __low2float(ds)
            if constexpr (X::kind == 0) {
                acc[j][c] = tw_acc0(acc[j][c], dw, d8, rec.sumi(j, dA, dB));
            } else {   // KQ8: (d8, the block's quant sum) as native_mmvq_xt_kq8_kernel / the tiles
                constexpr int bi = 8 * c + j;
                const int dot2 = __builtin_amdgcn_readlane(d2v[bi / 32], bi % 32);
                acc[j][c] = __fadd_rn(acc[j][c], xt_kq8_val(float(dA), float(dot2), scf, mf, d8, dmx, ndmy));
            }
        };
        tr_static_for<0, 8>([&](auto jc) {
            constexpr int j = decltype(jc)::value;
            int bv[8];
            rec.dec(j, bv);
            float scf = 0.0f, mf = 0.0f;
            if constexpr (X::kind == 1) {
                int sc, m;
                rec.scale_min(j, sc, m);
                scf = float(sc);
                mf = float(m);
            }
            tr_static_for<0, NC>([&](auto cc) {
                constexpr int c = decltype(cc)::value;
                const Q81Block* __restrict__ xb = x + std::size_t(c) * xs + 8 * sb + j;   // wave-uniform
                step(jc, cc, bv, reinterpret_cast<const int*>(xb->qs), *reinterpret_cast<const uint32_t*>(&xb->ds),
                     scf, mf);
            });
#if TALUS_TW_SCHED
            __builtin_amdgcn_sched_barrier(0);   // (one sub-block's operands live at a time, as in the tiles)
#endif
        });
    }
    // the chains meet as in the tiles: t = p0 + p2 (wave 0), u = p1 + p3 (wave 1), q_j = t + u, the tree over j
    auto put = [&](int wv) {
        if (wave == wv)
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int c = 0; c < NC; ++c) part[j][c][lane] = acc[j][c];
    };
    auto add = [&](int wv) {
        if (wave == wv)
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int c = 0; c < NC; ++c) acc[j][c] = acc[j][c] + part[j][c][lane];
    };
    put(2); __syncthreads(); add(0); __syncthreads();
    put(3); __syncthreads(); add(1); __syncthreads();
    put(1); __syncthreads(); add(0);
    if (wave == 0 && row < n_out) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float s = ((acc[0][c] + acc[4][c]) + (acc[2][c] + acc[6][c])) + ((acc[1][c] + acc[5][c]) + (acc[3][c] + acc[7][c]));
            y[std::size_t(c) * n_out + row] = s;
        }
    }
}
template<typename X, int NC, bool PACKED>
__launch_bounds__(128)
__global__ void native_mmvq_tr_kernel(const uint8_t* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                                      int n_in, int n_out) {
    tr_body<X, NC, PACKED>(w, x, y, n_in, n_out, int(blockIdx.x));
}
template<typename X, int NC, bool PACKED>
__launch_bounds__(128)
__global__ void native_mmvq_tr_multi_kernel(TwMulti m, const Q81Block* __restrict__ x, int n_in) {
    const int b = int(blockIdx.x);
    const int i = b >= m.start[2] ? 2 : (b >= m.start[1] ? 1 : 0);
    tr_body<X, NC, PACKED>(m.w[i], x, m.y[i], n_in, m.n_out[i], b - m.start[i]);
}

// ---- repack on the device (load time): one thread per (row, superblock) block copies its fields into the tile planes
template<int BS>
__global__ void native_mmvq_tw_pack_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int bpr, int n_out,
                                           int n_tiles, tw::TypeDesc d) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;   // (tile row, sb)
    const long long n = (long long) n_tiles * tw::kTileRows * bpr;
    if (i >= n) return;
    const int sb = int(i % bpr), rr = int(i / bpr), g = rr / tw::kTileRows, r = rr % tw::kTileRows;
    uint8_t* t = dst + (std::size_t(g) * bpr + sb) * (tw::kTileRows * BS);
    const uint8_t* s = src + (std::size_t(rr) * bpr + sb) * BS;
    for (int k = 0; k < d.nfields; ++k) {
        const tw::Field f = d.f[k];
        for (int p = 0; p < f.bytes / f.piece; ++p) {
            uint8_t* o = t + tw::plane_off(d, k, p, r);
            for (int b = 0; b < f.piece; ++b) o[b] = rr < n_out ? s[f.off + p * f.piece + b] : 0;
        }
    }
}

}  // namespace
