//! Safe [`KernelBackend`] trait and structured kernel argument descriptors.
//!
//! Both [`crate::FfiKernelBackend`] (backed by `libtalus_kernels.so` via [`talus_kernels_sys`])
//! and [`crate::CpuMockBackend`] (deterministic host `f32` reference implementation)
//! implement [`KernelBackend`], enabling Wave 2/3 model adapters and speculation logic
//! to run full end-to-end unit tests on CPU with no GPU.

use talus_core_api::{GpuArch, QuantType, Result};

/// Arguments for indirect MoE expert quantized matrix-vector projection (`mmvq_ids`).
#[derive(Debug)]
pub struct MmvqIdsArgs<'a> {
    /// Weight quantization type (`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q4_0`, `F32`, `F16`, `BF16`, etc.).
    pub qtype: QuantType,
    /// Contiguous expert weight slab (`n_experts * expert_bytes`).
    pub w: &'a [u8],
    /// Byte stride between consecutive experts in `w` (`n_out * row_bytes(qtype, n_in)`).
    pub expert_bytes: usize,
    /// Selected expert indices (`[n_ids]`). Negative indices write zeros to the corresponding output row.
    pub ids: &'a [i32],
    /// Quantized `Q8_1` input activations (`1` column if `!x_per_id`, or `n_ids` columns if `x_per_id`).
    pub x_q8_1: &'a [u8],
    /// Whether `x_q8_1` has one distinct input column per entry in `ids` (`true` for down projection,
    /// `false` for gate/up sharing one token input).
    pub x_per_id: bool,
    /// Output slice (`[n_ids, n_out]`).
    pub y: &'a mut [f32],
    /// Inner dimension `K` (`n_in`).
    pub n_in: usize,
    /// Output rows per expert `N` (`n_out`).
    pub n_out: usize,
    /// Optional paired second expert weight slab (e.g. `up` alongside `gate`).
    pub w2: Option<&'a [u8]>,
    /// Optional paired second output slice (`[n_ids, n_out]`).
    pub y2: Option<&'a mut [f32]>,
}

/// Arguments for single-token / small-window causal GQA decode/verify attention (`attn_decode`).
#[derive(Debug)]
pub struct AttnDecodeArgs<'a> {
    /// Query tensor (`[n_tok, n_head, head_dim]`).
    pub q: &'a [f32],
    /// Key cache pool (`[kv_stride, n_kv, head_dim]`, stored as `f32` or `f16` bytes depending on `kv_f16`).
    pub k_pool: &'a [u8],
    /// Value cache pool (`[kv_stride, n_kv, head_dim_v]`, stored as `f32` or `f16` bytes depending on `kv_f16`).
    pub v_pool: &'a [u8],
    /// Whether `k_pool` and `v_pool` store `f16` (2 bytes/elem) instead of `f32` (4 bytes/elem) on CPU.
    pub kv_f16: bool,
    /// Optional Q8 key scales (used by FFI backend when Q8 KV cache is active).
    pub k_scale: Option<&'a [u16]>,
    /// Optional Q8 value scales (used by FFI backend when Q8 KV cache is active).
    pub v_scale: Option<&'a [u16]>,
    /// Output tensor (`[n_tok, n_head, head_dim_v]`).
    pub out: &'a mut [f32],
    /// Optional split-K scratch buffer (used by GPU FFI kernel).
    pub scratch: Option<&'a mut [f32]>,
    /// Number of query tokens (`n_tok >= 1`).
    pub n_tok: usize,
    /// Number of query attention heads (`n_head`).
    pub n_head: usize,
    /// Number of KV attention heads (`n_kv`; `n_head % n_kv == 0`).
    pub n_kv: usize,
    /// Query/Key head dimension (`head_dim`).
    pub head_dim: usize,
    /// Value/Output head dimension (`head_dim_v`).
    pub head_dim_v: usize,
    /// Maximum KV cache capacity / stride in tokens (`kv_stride`).
    pub kv_stride: usize,
    /// Sliding window size in tokens (`0` or negative = full causal attention; `> 0` = attend to last `window` cells).
    pub window: i32,
    /// Softmax scale factor (typically `1.0 / (head_dim as f32).sqrt()`).
    pub scale: f32,
    /// Starting KV position for token `0` in this decode/verify window (`pos[0]`; token `t` is at `pos[0] + t`).
    pub pos: &'a [i32],
    /// Optional per-head attention sink logits (`[n_head]`).
    pub sinks: Option<&'a [f32]>,
    /// Optional elementwise sigmoid query gate (`[n_tok, n_head, head_dim_v]`), applied as `out * sigmoid(q_gate)`.
    pub q_gate: Option<&'a [f32]>,
    /// Split-K partition count (`0` = auto-select via [`crate::KernelSet`]).
    pub n_split: i32,
}

/// Arguments for multi-token prompt causal GQA flash attention (`attn_prefill`).
#[derive(Debug)]
pub struct AttnPrefillArgs<'a> {
    /// Query tensor (`[n_tok, n_head, head_dim]`).
    pub q: &'a [f32],
    /// Key cache pool (`[kv_stride, n_kv, head_dim]`, stored as `f32` or `f16` bytes depending on `kv_f16`).
    pub k_pool: &'a [u8],
    /// Value cache pool (`[kv_stride, n_kv, head_dim_v]`, stored as `f32` or `f16` bytes depending on `kv_f16`).
    pub v_pool: &'a [u8],
    /// Whether `k_pool` and `v_pool` store `f16` (2 bytes/elem) instead of `f32` (4 bytes/elem) on CPU.
    pub kv_f16: bool,
    /// Output tensor (`[n_tok, n_head, head_dim_v]`).
    pub out: &'a mut [f32],
    /// Number of query tokens in the prompt chunk (`n_tok`).
    pub n_tok: usize,
    /// Number of query attention heads (`n_head`).
    pub n_head: usize,
    /// Number of KV attention heads (`n_kv`; `n_head % n_kv == 0`).
    pub n_kv: usize,
    /// Query/Key head dimension (`head_dim`).
    pub head_dim: usize,
    /// Value/Output head dimension (`head_dim_v`).
    pub head_dim_v: usize,
    /// KV cache capacity / stride in tokens (`kv_stride`).
    pub kv_stride: usize,
    /// Starting KV position `p0` of token `0` in this prompt chunk.
    pub p0: usize,
    /// Sliding window size (`0` = full causal; `> 0` = attend to last `window` cells).
    pub window: i32,
    /// Softmax scale factor (`1.0 / (head_dim as f32).sqrt()`).
    pub scale: f32,
    /// Optional per-head attention sink logits (`[n_head]`).
    pub sinks: Option<&'a [f32]>,
    /// Optional elementwise sigmoid query gate (`[n_tok, n_head, head_dim_v]`).
    pub q_gate: Option<&'a [f32]>,
}

/// Arguments for a single-token Gated DeltaNet recurrent step (`gdn_step`).
#[derive(Debug)]
pub struct GdnStepArgs<'a> {
    /// Recurrent state tensor (`[state_size, h_v, state_size]`, updated in-place).
    pub state: &'a mut [f32],
    /// Query vector (`[h_k, state_size]`).
    pub q: &'a [f32],
    /// Key vector (`[h_k, state_size]`).
    pub k: &'a [f32],
    /// Value vector (`[h_v, state_size]`).
    pub v: &'a [f32],
    /// Log-space decay gate (`[h_v]`; decay factor `g = gate[hv].exp()`).
    pub gate: &'a [f32],
    /// Delta update rate (`[h_v]`, after sigmoid).
    pub beta: &'a [f32],
    /// Raw head outputs (`[h_v, state_size]`).
    pub out: &'a mut [f32],
    /// Per-head state dimension `S` (e.g. `128`).
    pub state_size: usize,
    /// Number of query/key heads `h_k`.
    pub h_k: usize,
    /// Number of value/state heads `h_v` (`h_v % h_k == 0`).
    pub h_v: usize,
}

/// Arguments for multi-token sequential Gated DeltaNet recurrence + output RMSNorm & gate (`gdn_seq`).
#[derive(Debug)]
pub struct GdnSeqArgs<'a> {
    /// Recurrent state tensor (`[state_size, h_v, state_size]`, updated in-place).
    pub state: &'a mut [f32],
    /// Query sequence (`[n_tok, h_k, state_size]`).
    pub q: &'a [f32],
    /// Key sequence (`[n_tok, h_k, state_size]`).
    pub k: &'a [f32],
    /// Value sequence (`[n_tok, h_v, state_size]`).
    pub v: &'a [f32],
    /// Log-space decay gate sequence (`[n_tok, h_v]`).
    pub gate: &'a [f32],
    /// Delta update rate sequence (`[n_tok, h_v]`).
    pub beta: &'a [f32],
    /// Output gate tensor `z` (`[n_tok, h_v, state_size]`).
    pub z: &'a [f32],
    /// Per-head RMSNorm weight `gamma` (`[h_v, state_size]` or `[state_size]` shared across heads).
    pub gamma: &'a [f32],
    /// Normalized and gated output tensor (`[n_tok, h_v, state_size]`).
    pub out: &'a mut [f32],
    /// Number of tokens in the sequence (`n_tok`).
    pub n_tok: usize,
    /// Per-head state dimension `S`.
    pub state_size: usize,
    /// Number of query/key heads `h_k`.
    pub h_k: usize,
    /// Number of value/state heads `h_v` (`h_v % h_k == 0`).
    pub h_v: usize,
    /// RMSNorm epsilon (`norm_eps`).
    pub norm_eps: f32,
    /// If `true`, output gate uses `silu(z)` (`qwen35`); if `false`, uses `sigmoid(z)` (`qwen4exp`).
    pub silu_gate: bool,
}

/// Arguments for batched prompt dense GEMM (`prefill_gemm`).
#[derive(Debug)]
pub struct PrefillGemmArgs<'a> {
    /// Weight element type (`F16`, `BF16`, or `F32`).
    pub w_qtype: QuantType,
    /// Weight matrix `[n_out, n_in]` in `w_qtype`.
    pub w: &'a [u8],
    /// Activation matrix `[n_tok, ldx]` in `x_qtype`.
    pub x: &'a [u8],
    /// Activation element type (`F16`, `BF16`, or `F32`).
    pub x_qtype: QuantType,
    /// Output matrix `[n_tok, ldy]` in `f32`.
    pub y: &'a mut [f32],
    /// Number of tokens (`n_tok`).
    pub n_tok: usize,
    /// Output dimension (`n_out`).
    pub n_out: usize,
    /// Input dimension (`n_in`).
    pub n_in: usize,
    /// Leading dimension of `y` (`ldy >= n_out`).
    pub ldy: usize,
    /// Leading dimension of `x` in elements (`ldx >= n_in`).
    pub ldx: usize,
    /// Output accumulation scale (`y = X * W^T + beta * y`).
    pub beta: f32,
}

/// Arguments for batched prompt quantized MMQ projection (`prefill_mmq`).
#[derive(Debug)]
pub struct PrefillMmqArgs<'a> {
    /// Weight quantization type (`Q4_0`, `Q4_K`, `Q8_0`, etc.).
    pub w_qtype: QuantType,
    /// Quantized weight matrix `[n_out, n_in]` in `w_qtype`.
    pub w_blocks: &'a [u8],
    /// Quantized activation columns (`[n_tok]` columns of `n_in` in `Q8_1`).
    pub x_q8_1: &'a [u8],
    /// Output matrix `[n_tok, ldy]` in `f32`.
    pub y: &'a mut [f32],
    /// Number of tokens (`n_tok`).
    pub n_tok: usize,
    /// Output dimension (`n_out`).
    pub n_out: usize,
    /// Input dimension (`n_in`).
    pub n_in: usize,
    /// Leading dimension of `y` (`ldy >= n_out`).
    pub ldy: usize,
}

/// Safe kernel execution backend trait implemented by [`crate::FfiKernelBackend`] and
/// [`crate::CpuMockBackend`].
#[allow(clippy::too_many_arguments)]
pub trait KernelBackend: Send + Sync {
    /// Return the target [`GpuArch`] associated with this backend.
    fn arch(&self) -> GpuArch;

    /// In-place weighted RMSNorm over `[rows, cols]`:
    /// `x[r, c] = (x[r, c] / sqrt(mean(x[r]^2) + eps)) * w[c]`.
    fn rms_norm(
        &self,
        x: &mut [f32],
        w: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()>;

    /// Per-head RMSNorm (`w[head_dim]`) followed by in-place NEOX RoPE on the first `n_rot`
    /// dimensions of each row (`[rows, head_dim]`, where `head_dim = w.len()`).
    fn head_norm_rope(
        &self,
        x: &mut [f32],
        w: &[f32],
        rows: usize,
        eps: f32,
        n_rot: usize,
        cos_tab: &[f32],
        sin_tab: &[f32],
        pos: &[i32],
        pos_div: usize,
    ) -> Result<()>;

    /// Partial NEOX RoPE rotation over `[rows, head_dim]`, rotating the first `n_rot` dims of each
    /// row using `pos[row]` into `(cos_tab, sin_tab)` (`[max_pos, n_rot / 2]`).
    fn rope_neox(
        &self,
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        head_dim: usize,
        n_rot: usize,
        cos_tab: &[f32],
        sin_tab: &[f32],
        pos: &[i32],
    ) -> Result<()>;

    /// In-place SwiGLU elementwise product: `gate[i] = silu(gate[i]) * up[i]` for `n` elements.
    fn silu_mul(&self, gate: &mut [f32], up: &[f32], n: usize) -> Result<()>;

    /// Scaled vector addition: `y[i] += alpha * x[i]` for `n` elements.
    fn axpy(&self, y: &mut [f32], x: &[f32], alpha: f32, n: usize) -> Result<()>;

    /// Gather and dequantize `tokens.len()` rows from a `Q8_0` embedding table into `out` (`[n_tok, n_embd]`).
    fn embed_q8_0(
        &self,
        table: &[u8],
        tokens: &[i32],
        out: &mut [f32],
        n_embd: usize,
    ) -> Result<()>;

    /// Apply per-head softplus attention output gate in-place: `o[h, d] *= softplus(g[h])`.
    fn softplus_head_gate(
        &self,
        o: &mut [f32],
        g: &[f32],
        n_head: usize,
        head_dim: usize,
    ) -> Result<()>;

    /// Quantize `ncols` columns of `n_in` `f32` activations into `Q8_1` blocks (`36 * (n_in / 32) * ncols` bytes).
    fn quantize_q8_1(
        &self,
        x_f32: &[f32],
        x_q8_1: &mut [u8],
        n_in: usize,
        ncols: usize,
    ) -> Result<()>;

    /// Quantized matrix-vector / small-batch projection:
    /// `y[col, row] = dot(dequant(w[row]), dequant(x_q8_1[col]))` for `ncols` columns.
    fn mmvq(
        &self,
        qtype: QuantType,
        w: &[u8],
        x_q8_1: &[u8],
        y: &mut [f32],
        n_in: usize,
        n_out: usize,
        ncols: usize,
    ) -> Result<()>;

    /// Indirect MoE expert quantized matrix-vector projection (`tk_mmvq_ids`).
    fn mmvq_ids(&self, args: MmvqIdsArgs<'_>) -> Result<()>;

    /// FP32 matrix-vector product (`y[o] = dot(w[o, 0..n_in], x[0..n_in])` with FP64 accumulation).
    fn f32_matvec(
        &self,
        w: &[f32],
        x: &[f32],
        y: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<()>;

    /// Batched FP32 matrix-matrix product (`y[t, o] = dot(w[o, 0..n_in], x[t, 0..n_in])` for `n_tok` rows).
    fn f32_matmul(
        &self,
        w: &[f32],
        x: &[f32],
        y: &mut [f32],
        n_in: usize,
        n_out: usize,
        n_tok: usize,
    ) -> Result<()>;

    /// Sigmoid + bias top-k MoE routing (`p = sigmoid(logits)`, select top-k by `p + bias`,
    /// ties broken to lower expert index, normalized weights `p / max(sum, 6.103515625e-5) * scale`).
    fn route_sigmoid_topk(
        &self,
        logits: &[f32],
        bias: Option<&[f32]>,
        ids_out: &mut [i32],
        weights_out: &mut [f32],
        n_expert: usize,
        k: usize,
        scale: f32,
    ) -> Result<()>;

    /// Accumulate shared expert output and routed expert slots into residual `x`:
    /// `x[t] += shared[t] + sum_{j=0..k-1} weights[t*k+j] * y[map[t*k+j]]`.
    fn moe_add_slots(
        &self,
        x: &mut [f32],
        shared: Option<&[f32]>,
        y: &[f32],
        map: Option<&[i32]>,
        weights: &[f32],
        n_tok: usize,
        n_embd: usize,
        k: usize,
    ) -> Result<()>;

    /// Row-wise greedy argmax over `[n_rows, n_vocab]`, breaking ties toward the lowest index.
    fn argmax(
        &self,
        logits: &[f32],
        n_rows: usize,
        n_vocab: usize,
        out_tokens: &mut [i32],
    ) -> Result<()>;

    /// Single-token or verify-window causal GQA attention with optional sliding window, sinks, and query gate.
    fn attn_decode(&self, args: AttnDecodeArgs<'_>) -> Result<()>;

    /// Multi-token prompt causal GQA flash attention with optional sliding window, sinks, and query gate.
    fn attn_prefill(&self, args: AttnPrefillArgs<'_>) -> Result<()>;

    /// Single-token Gated DeltaNet recurrent step updating `state` (`[S, h_v, S]`) in-place.
    fn gdn_step(&self, args: GdnStepArgs<'_>) -> Result<()>;

    /// Multi-token sequential Gated DeltaNet recurrence + per-head RMSNorm & output gate.
    fn gdn_seq(&self, args: GdnSeqArgs<'_>) -> Result<()>;

    /// Batched prompt dense GEMM (`Y = X * W^T + beta * Y`).
    fn prefill_gemm(&self, args: PrefillGemmArgs<'_>) -> Result<()>;

    /// Batched prompt quantized MMQ projection (`Y = dequant(X_q8_1) * dequant(W)^T`).
    fn prefill_mmq(&self, args: PrefillMmqArgs<'_>) -> Result<()>;
}
