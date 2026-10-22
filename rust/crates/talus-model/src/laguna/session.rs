//! Disk KV session persistence (`TALUSKV2`, `src/laguna/laguna_engine.cpp:693-803`).

use crate::laguna::weights::{is_full, K_HEAD_DIM, K_HEAD_KV};

/// Magic bytes for Laguna v2 KV session files (`"TALUSKV2"`).
pub const SESSION_MAGIC: [u8; 8] = *b"TALUSKV2";

/// Binary session header written at offset 0 of a Laguna KV cache snapshot file.
///
/// Matches C++ `SessionHeader` in `src/laguna/laguna_engine.cpp:693-702` bit-for-bit:
/// ```cpp
/// struct SessionHeader {
///     char magic[8];
///     int32_t n_layers;
///     int32_t kv_q8;
///     int32_t cell;
///     int32_t n_tokens;
///     // 4 bytes implicit C++ padding before 8-byte aligned uint64_t!
///     uint64_t model_id;
///     int32_t ring;
///     int32_t ctx16;
/// };
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionHeader {
    pub magic: [u8; 8],
    pub n_layers: i32,
    pub kv_q8: i32,
    pub cell: i32,
    pub n_tokens: i32,
    pub model_id: u64,
    pub ring: i32,
    pub ctx16: i32,
}

const _: [(); 40] = [(); std::mem::size_of::<SessionHeader>()];

impl SessionHeader {
    pub const SIZE: usize = 40;
    pub const CELL: i32 = (K_HEAD_KV * K_HEAD_DIM) as i32;

    /// Construct a new `SessionHeader` for saving `n_tokens` of KV state.
    pub fn new(
        n_layers: usize,
        kv_q8: bool,
        n_tokens: i32,
        model_id: u64,
        ring: usize,
        ctx16: usize,
    ) -> Self {
        Self {
            magic: SESSION_MAGIC,
            n_layers: n_layers as i32,
            kv_q8: i32::from(kv_q8),
            cell: Self::CELL,
            n_tokens,
            model_id,
            ring: ring as i32,
            ctx16: ctx16 as i32,
        }
    }

    /// Serialize the 40-byte header in native little-endian layout.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        // SAFETY: SessionHeader is #[repr(C)], 40 bytes, with all fields initialized.
        unsafe { std::mem::transmute::<Self, [u8; Self::SIZE]>(*self) }
    }

    /// Deserialize a 40-byte header from native little-endian bytes.
    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
        // SAFETY: Any 40-byte pattern is valid for SessionHeader's integer/array fields.
        unsafe { std::mem::transmute::<[u8; Self::SIZE], Self>(*bytes) }
    }

    /// Validate `SessionHeader` against the running engine configuration, matching
    /// `laguna_engine.cpp:746-751`.
    pub fn validate(
        &self,
        expected_layers: usize,
        expected_kv_q8: bool,
        expected_model_id: u64,
        expected_ring: usize,
        expected_ctx16: usize,
        max_ctx: usize,
    ) -> bool {
        if self.magic != SESSION_MAGIC {
            return false;
        }
        if self.n_layers != expected_layers as i32 {
            return false;
        }
        if self.kv_q8 != i32::from(expected_kv_q8) {
            return false;
        }
        if self.cell != Self::CELL {
            return false;
        }
        if self.model_id != expected_model_id {
            return false;
        }
        if self.ring != expected_ring as i32 {
            return false;
        }
        if self.ctx16 != expected_ctx16 as i32 {
            return false;
        }
        if self.n_tokens <= 0 || (self.n_tokens as usize) > max_ctx {
            return false;
        }
        true
    }
}

/// Compute total payload bytes (excluding the 40-byte header and `n_tokens * 4` token IDs)
/// written by `save_session` and read by `load_session`, matching `laguna_engine.cpp:718-724`.
pub fn session_payload_bytes(
    n_layers: usize,
    kv_q8: bool,
    n_tokens: usize,
    ctx16: usize,
    vtl: &[usize],
) -> usize {
    let cell = K_HEAD_KV * K_HEAD_DIM;
    let esz = if kv_q8 { 1usize } else { 2usize };
    let mut total = 0usize;
    for &vtl_il in vtl.iter().take(n_layers) {
        let np = if vtl_il == ctx16 { n_tokens } else { vtl_il };
        let tiles = (np + 15) / 16;
        total += np * cell * esz;
        if kv_q8 {
            total += np * K_HEAD_KV * 4 * 2;
        }
        total += K_HEAD_KV
            * (tiles * 16 * K_HEAD_DIM * esz + if kv_q8 { tiles * 4 * 16 * 2 } else { 0 });
    }
    let _ = is_full(0);
    total
}

/// Compute `(ctx16, ring)` for a given `(ctx, ubatch)` pair, matching `laguna_engine.cpp:89-90`.
#[inline]
pub fn compute_ctx16_and_ring(ctx: usize, ubatch: usize) -> (usize, usize) {
    let ctx16 = ((ctx + 15) & !15) + 64;
    let ring = ctx16.min(((crate::laguna::weights::K_SWA + ubatch + 64) + 15) & !15);
    (ctx16, ring)
}
