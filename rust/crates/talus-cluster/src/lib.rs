//! Talus clustering, Phase 1 (talus#47): the wire between pipeline stages.
//!
//! A model is split by layers into stages that run in separate processes (Phase 1: two processes on one box over
//! loopback; Phase 2: two boxes over thunderbolt-net). Stage 0 (the head) owns the API, tokenizer, embedding, head,
//! sampler and drafters; a worker owns a contiguous layer range. One decode step crosses the link twice: the boundary
//! bundle (the live set the fused cross-layer epilogue wrote) goes out, the final hidden state comes back.
//!
//! This crate is engine-agnostic: framing ([`frame`]), the TCP link ([`Link`]), the cluster spec ([`Spec`]) and the
//! build/config identity checked at handshake ([`ident`]). Everything is off unless `--cluster-*` flags or
//! `TALUS_CLUSTER_*` variables are given; nothing here runs on a single-box engine.

pub mod frame;
pub mod ident;
mod link;

pub use frame::{Header, MsgType, HDR_BYTES};
pub use link::Link;

/// The role of this process in a cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Stage 0: API, tokenizer, embedding, layers `0..split`, head, sampler, drafters. Connects to the worker.
    Head,
    /// A worker stage: layers `split..n_layer`. Listens; serves one head at a time; keeps its weights across heads.
    Worker,
}

/// A two-stage pipeline (Phase 1/2): where the worker listens and where the layers are cut.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spec {
    pub role: Role,
    /// `host:port` the worker listens on and the head connects to (TCP 7701-7702 by convention).
    pub addr: String,
    /// First layer of the worker stage (the cut). 0 = engine default (`n_layer / 2`, rounded to the layer pattern).
    pub split: u32,
}

impl Spec {
    /// Parse `head|worker`.
    pub fn parse_role(s: &str) -> Option<Role> {
        match s {
            "head" | "0" => Some(Role::Head),
            "worker" | "stage" | "1" => Some(Role::Worker),
            _ => None,
        }
    }

    /// Fill unset fields from `TALUS_CLUSTER_ROLE`, `TALUS_CLUSTER_ADDR`, `TALUS_CLUSTER_SPLIT`.
    pub fn from_parts(
        role: Option<&str>,
        addr: Option<&str>,
        split: Option<u32>,
    ) -> Result<Option<Spec>, String> {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let role_s = role
            .map(str::to_string)
            .or_else(|| env("TALUS_CLUSTER_ROLE"));
        let Some(role_s) = role_s else {
            return Ok(None);
        };
        let role = Self::parse_role(&role_s)
            .ok_or_else(|| format!("cluster role must be head|worker, got {role_s}"))?;
        let addr = addr
            .map(str::to_string)
            .or_else(|| env("TALUS_CLUSTER_ADDR"))
            .unwrap_or_else(|| "127.0.0.1:7701".into());
        let split = match split {
            Some(s) => s,
            None => match env("TALUS_CLUSTER_SPLIT") {
                Some(v) => v
                    .parse::<u32>()
                    .map_err(|_| format!("TALUS_CLUSTER_SPLIT: bad value {v}"))?,
                None => 0,
            },
        };
        Ok(Some(Spec { role, addr, split }))
    }
}

/// Debug payload checksums (`TALUS_CLUSTER_CRC=1`): CRC32C over every payload, checked on receipt.
pub fn crc_enabled() -> bool {
    std::env::var("TALUS_CLUSTER_CRC")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Simple `k=v` lines, the payload of the setup messages (HELLO, INIT and their ACKs).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Kv(pub Vec<(String, String)>);

impl Kv {
    pub fn new() -> Self {
        Self(Vec::new())
    }
    pub fn put(&mut self, k: &str, v: impl ToString) -> &mut Self {
        self.0
            .push((k.to_string(), v.to_string().replace('\n', " ")));
        self
    }
    pub fn get(&self, k: &str) -> Option<&str> {
        self.0.iter().find(|(a, _)| a == k).map(|(_, b)| b.as_str())
    }
    pub fn get_i64(&self, k: &str) -> Option<i64> {
        self.get(k).and_then(|v| v.parse().ok())
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut s = String::new();
        for (k, v) in &self.0 {
            s.push_str(k);
            s.push('=');
            s.push_str(v);
            s.push('\n');
        }
        s.into_bytes()
    }
    pub fn decode(b: &[u8]) -> Self {
        let s = String::from_utf8_lossy(b);
        Self(
            s.lines()
                .filter_map(|l| {
                    l.split_once('=')
                        .map(|(a, b)| (a.to_string(), b.to_string()))
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_roundtrip() {
        let mut kv = Kv::new();
        kv.put("a", 1).put("b", "x=y").put("c", "two\nlines");
        let d = Kv::decode(&kv.encode());
        assert_eq!(d.get("a"), Some("1"));
        assert_eq!(d.get("b"), Some("x=y"));
        assert_eq!(d.get("c"), Some("two lines"));
        assert_eq!(d.get_i64("a"), Some(1));
    }

    #[test]
    fn spec_parts() {
        let s = Spec::from_parts(Some("worker"), Some("127.0.0.1:7702"), Some(32))
            .unwrap()
            .unwrap();
        assert_eq!(s.role, Role::Worker);
        assert_eq!(s.addr, "127.0.0.1:7702");
        assert_eq!(s.split, 32);
        assert!(Spec::from_parts(Some("bogus"), None, None).is_err());
    }
}
