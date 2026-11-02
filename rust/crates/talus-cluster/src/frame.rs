//! The 64-byte fixed little-endian frame header (DESIGN.md section 3.3):
//! `magic 'TLSC' | ver u16 | type u16 | flags u32 | epoch u32 | layout_id u32 | seq_id u64 | step u64 |
//!  pos0 u32 | rows u32 | commit_keep u32 | n_tokens u32 | payload_len u64 | hdr_crc32c u32`
//! followed by `payload_len` bytes. With [`FLAG_CRC`] the last 4 payload bytes are the CRC32C of the rest.

pub const MAGIC: [u8; 4] = *b"TLSC";
pub const VERSION: u16 = 1;
pub const HDR_BYTES: usize = 64;
/// The payload ends with a CRC32C of the bytes before it (`TALUS_CLUSTER_CRC=1`).
pub const FLAG_CRC: u32 = 1;

/// Message types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum MsgType {
    Hello = 1,
    HelloAck = 2,
    Init = 3,
    InitAck = 4,
    /// One segment call: a step control block plus the boundary bundle.
    Step = 5,
    /// The return of a step: final hidden rows (+ taps, + debug residuals).
    Ret = 6,
    /// Reset the sequence state of the owned layers.
    Reset = 7,
    /// Flush the pending fused GDN commit of the owned layers (payload: the head's 16-byte gctl).
    Flush = 8,
    /// Replay `commit_keep` accepted rows through the owned GDN layers (unfused path).
    Commit = 9,
    Err = 10,
    Bye = 11,
    Heartbeat = 12,
}

impl MsgType {
    pub fn from_u16(v: u16) -> Option<Self> {
        use MsgType::*;
        Some(match v {
            1 => Hello,
            2 => HelloAck,
            3 => Init,
            4 => InitAck,
            5 => Step,
            6 => Ret,
            7 => Reset,
            8 => Flush,
            9 => Commit,
            10 => Err,
            11 => Bye,
            12 => Heartbeat,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Header {
    pub ty: u16,
    pub flags: u32,
    pub epoch: u32,
    pub layout_id: u32,
    pub seq_id: u64,
    pub step: u64,
    pub pos0: u32,
    pub rows: u32,
    pub commit_keep: u32,
    pub n_tokens: u32,
    pub payload_len: u64,
}

impl Header {
    pub fn new(ty: MsgType) -> Self {
        Self {
            ty: ty as u16,
            ..Default::default()
        }
    }

    pub fn msg(&self) -> Option<MsgType> {
        MsgType::from_u16(self.ty)
    }

    pub fn encode(&self) -> [u8; HDR_BYTES] {
        let mut b = [0u8; HDR_BYTES];
        b[0..4].copy_from_slice(&MAGIC);
        b[4..6].copy_from_slice(&VERSION.to_le_bytes());
        b[6..8].copy_from_slice(&self.ty.to_le_bytes());
        b[8..12].copy_from_slice(&self.flags.to_le_bytes());
        b[12..16].copy_from_slice(&self.epoch.to_le_bytes());
        b[16..20].copy_from_slice(&self.layout_id.to_le_bytes());
        b[20..28].copy_from_slice(&self.seq_id.to_le_bytes());
        b[28..36].copy_from_slice(&self.step.to_le_bytes());
        b[36..40].copy_from_slice(&self.pos0.to_le_bytes());
        b[40..44].copy_from_slice(&self.rows.to_le_bytes());
        b[44..48].copy_from_slice(&self.commit_keep.to_le_bytes());
        b[48..52].copy_from_slice(&self.n_tokens.to_le_bytes());
        b[52..60].copy_from_slice(&self.payload_len.to_le_bytes());
        let c = crc32c(&b[0..60]);
        b[60..64].copy_from_slice(&c.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8; HDR_BYTES]) -> Result<Self, String> {
        if b[0..4] != MAGIC {
            return Err("bad magic (not a talus cluster peer?)".into());
        }
        let ver = u16::from_le_bytes([b[4], b[5]]);
        if ver != VERSION {
            return Err(format!("protocol version {ver}, expected {VERSION}"));
        }
        let c = u32::from_le_bytes([b[60], b[61], b[62], b[63]]);
        if c != crc32c(&b[0..60]) {
            return Err("header crc32c mismatch".into());
        }
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let u64_at = |o: usize| {
            let mut a = [0u8; 8];
            a.copy_from_slice(&b[o..o + 8]);
            u64::from_le_bytes(a)
        };
        Ok(Self {
            ty: u16::from_le_bytes([b[6], b[7]]),
            flags: u32_at(8),
            epoch: u32_at(12),
            layout_id: u32_at(16),
            seq_id: u64_at(20),
            step: u64_at(28),
            pos0: u32_at(36),
            rows: u32_at(40),
            commit_keep: u32_at(44),
            n_tokens: u32_at(48),
            payload_len: u64_at(52),
        })
    }
}

static CRC_TABLE: [u32; 256] = make_table();

const fn make_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0x82F6_3B78
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

/// CRC32C (Castagnoli), continuing from `crc` (pass 0 to start).
pub fn crc32c_update(crc: u32, data: &[u8]) -> u32 {
    let mut c = !crc;
    for &b in data {
        c = CRC_TABLE[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

pub fn crc32c(data: &[u8]) -> u32 {
    crc32c_update(0, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_known_value() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        let a = crc32c_update(crc32c(b"1234"), b"56789");
        assert_eq!(a, 0xE306_9283);
    }

    #[test]
    fn header_roundtrip_and_tamper() {
        let mut h = Header::new(MsgType::Step);
        h.rows = 16;
        h.pos0 = 2047;
        h.step = 99;
        h.payload_len = 1 << 33;
        let b = h.encode();
        assert_eq!(Header::decode(&b).unwrap(), h);
        let mut bad = b;
        bad[40] ^= 1;
        assert!(Header::decode(&bad).is_err());
    }
}
