//! Binary KV + RNG session persistence header for MiMo-V2.6-Flash (`mimo2`).
//!
//! Supports both the MiMo-specific `TALUSMM1` magic and the generic v2 `TALUSKV2`
//! magic so session files round-trip cleanly across `talus-cli` and `server.py`.

use talus_core_api::{CoreError, Result};

/// MiMo-V2.6-Flash session magic (`"TALUSMM1"`).
pub const SESSION_MAGIC_MM1: [u8; 8] = *b"TALUSMM1";
/// Generic Talus KV v2 session magic (`"TALUSKV2"`).
pub const SESSION_MAGIC_KV2: [u8; 8] = *b"TALUSKV2";
/// Current session format version (`1`).
pub const SESSION_VERSION: u32 = 1;

/// Fixed 32-byte binary header at the start of a saved MiMo KV session file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionHeader {
    /// 8-byte magic (`TALUSMM1` or `TALUSKV2`).
    pub magic: [u8; 8],
    /// Format version (`1`).
    pub version: u32,
    /// Number of sequence positions currently populated in the KV cache.
    pub n_pos: u32,
    /// Configured maximum context length when the session was written.
    pub max_ctx: u32,
    /// Total number of tokens stored in the token history trail.
    pub n_tokens: u32,
    /// Sampler PRNG state at the time of checkpointing.
    pub rng_state: u64,
}

impl SessionHeader {
    /// Exact serialized byte length of [`SessionHeader`].
    pub const BYTE_LEN: usize = 32;

    /// Construct a new `TALUSMM1` session header.
    pub fn new(n_pos: u32, max_ctx: u32, n_tokens: u32, rng_state: u64) -> Self {
        Self {
            magic: SESSION_MAGIC_MM1,
            version: SESSION_VERSION,
            n_pos,
            max_ctx,
            n_tokens,
            rng_state,
        }
    }

    /// Construct a new `TALUSKV2` session header.
    pub fn new_kv2(n_pos: u32, max_ctx: u32, n_tokens: u32, rng_state: u64) -> Self {
        Self {
            magic: SESSION_MAGIC_KV2,
            version: SESSION_VERSION,
            n_pos,
            max_ctx,
            n_tokens,
            rng_state,
        }
    }

    /// Serialize this header into a 32-byte little-endian array.
    pub fn to_bytes(&self) -> [u8; Self::BYTE_LEN] {
        let mut out = [0u8; Self::BYTE_LEN];
        out[0..8].copy_from_slice(&self.magic);
        out[8..12].copy_from_slice(&self.version.to_le_bytes());
        out[12..16].copy_from_slice(&self.n_pos.to_le_bytes());
        out[16..20].copy_from_slice(&self.max_ctx.to_le_bytes());
        out[20..24].copy_from_slice(&self.n_tokens.to_le_bytes());
        out[24..32].copy_from_slice(&self.rng_state.to_le_bytes());
        out
    }

    /// Parse and validate a 32-byte little-endian header (`TALUSMM1` or `TALUSKV2`).
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::BYTE_LEN {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "session header truncated: got {} bytes, need {}",
                    buf.len(),
                    Self::BYTE_LEN
                ),
            });
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&buf[0..8]);
        if magic != SESSION_MAGIC_MM1 && magic != SESSION_MAGIC_KV2 {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "invalid MiMo session magic {:?} (expected TALUSMM1 or TALUSKV2)",
                    magic
                ),
            });
        }
        let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        if version != SESSION_VERSION {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "unsupported MiMo session version {version} (expected {SESSION_VERSION})"
                ),
            });
        }
        let n_pos = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let max_ctx = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        let n_tokens = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        let rng_state = u64::from_le_bytes(buf[24..32].try_into().unwrap());

        Ok(Self {
            magic,
            version,
            n_pos,
            max_ctx,
            n_tokens,
            rng_state,
        })
    }
}
