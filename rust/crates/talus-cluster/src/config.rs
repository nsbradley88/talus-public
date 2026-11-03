//! `cluster.toml` (DESIGN.md 3.6), Phase 2 subset: two nodes, one cut. Host addresses come from the operator's host
//! registry (a `hosts.conf`: `name lan-addr user ...`); the thunderbolt-net address is given per node.
//!
//! ```toml
//! [cluster]
//! name = "halo-pair"
//! hosts = "/etc/talus/hosts.conf"   # LAN addresses by node name
//! port = 7701                                      # data; control (heartbeat) = port + 1
//! link = "tb"                                      # tb | lan: the primary path; the other one is the fallback
//! split = 32                                       # 0 or absent: the engine default (the middle)
//!
//! [[node]]
//! name = "node-a"
//! head = true
//! tb = "198.51.100.2"
//!
//! [[node]]
//! name = "node-b"
//! tb = "198.51.100.1"
//! ```
//! A small parser for exactly this shape (no TOML dependency): sections `[cluster]` and `[[node]]`, `key = value`
//! with quoted strings, integers or booleans, `#` comments.

use crate::{Role, Spec};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NodeCfg {
    pub name: String,
    pub head: bool,
    pub tb: Option<String>,
    pub lan: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterFile {
    pub name: String,
    pub hosts: Option<String>,
    pub port: u16,
    pub link: String,
    pub split: u32,
    pub nodes: Vec<NodeCfg>,
}

fn unquote(v: &str) -> String {
    let v = v.trim();
    if v.len() >= 2
        && ((v.starts_with('"') && v.ends_with('"')) || (v.starts_with('\'') && v.ends_with('\'')))
    {
        v[1..v.len() - 1].to_string()
    } else {
        v.to_string()
    }
}

fn strip_comment(l: &str) -> &str {
    let mut q = false;
    for (i, c) in l.char_indices() {
        match c {
            '"' => q = !q,
            '#' if !q => return &l[..i],
            _ => {}
        }
    }
    l
}

pub fn parse(text: &str) -> Result<ClusterFile, String> {
    let mut f = ClusterFile {
        name: "cluster".into(),
        hosts: None,
        port: 7701,
        link: "tb".into(),
        split: 0,
        nodes: Vec::new(),
    };
    let mut sec = String::new();
    for (ln, raw) in text.lines().enumerate() {
        let l = strip_comment(raw).trim();
        if l.is_empty() {
            continue;
        }
        if l == "[[node]]" {
            sec = "node".into();
            f.nodes.push(NodeCfg::default());
            continue;
        }
        if l.starts_with('[') {
            sec = l.trim_matches(|c| c == '[' || c == ']').trim().to_string();
            continue;
        }
        let (k, v) = l
            .split_once('=')
            .ok_or_else(|| format!("cluster.toml line {}: expected key = value", ln + 1))?;
        let (k, v) = (k.trim(), unquote(v));
        let bad = |what: &str| format!("cluster.toml line {}: bad {what} `{v}`", ln + 1);
        match (sec.as_str(), k) {
            ("cluster", "name") => f.name = v,
            ("cluster", "hosts") => f.hosts = Some(v),
            ("cluster", "port") => f.port = v.parse().map_err(|_| bad("port"))?,
            ("cluster", "link") => {
                if v != "tb" && v != "lan" {
                    return Err(bad("link (tb | lan)"));
                }
                f.link = v
            }
            ("cluster", "split") => f.split = v.parse().map_err(|_| bad("split"))?,
            ("cluster", _) => {} // keys of later phases (exact, heartbeat_ms, ...) are accepted and ignored
            ("node", key) => {
                let n = f.nodes.last_mut().unwrap();
                match key {
                    "name" => n.name = v,
                    "head" => n.head = v.parse().map_err(|_| bad("head (true | false)"))?,
                    "tb" => n.tb = Some(v),
                    "lan" | "addr" => n.lan = Some(v),
                    _ => {}
                }
            }
            _ => {
                return Err(format!(
                    "cluster.toml line {}: `{k}` outside [cluster] / [[node]]",
                    ln + 1
                ))
            }
        }
    }
    if f.nodes.len() != 2 {
        return Err(format!(
            "cluster.toml: Phase 2 needs exactly two [[node]]s, got {}",
            f.nodes.len()
        ));
    }
    if f.nodes.iter().filter(|n| n.head).count() != 1 {
        return Err("cluster.toml: exactly one node must have head = true".into());
    }
    Ok(f)
}

/// The LAN address of host `name` in a hosts.conf (`name addr user ...`, `#` comments).
pub fn hosts_lookup(hosts_conf: &str, name: &str) -> Option<String> {
    hosts_conf.lines().find_map(|l| {
        let mut w = strip_comment(l).split_whitespace();
        (w.next()? == name)
            .then(|| w.next().map(str::to_string))
            .flatten()
    })
}

impl ClusterFile {
    /// Fill each node's LAN address from the hosts file (unless given in the toml).
    pub fn resolve_hosts(&mut self, hosts_conf_text: Option<&str>) {
        if let Some(t) = hosts_conf_text {
            for n in &mut self.nodes {
                if n.lan.is_none() {
                    n.lan = hosts_lookup(t, &n.name);
                }
            }
        }
    }

    /// The spec for node `me`: the head connects to the worker's primary-link address (alt: the other link); the
    /// worker listens on every interface (the cluster firewall admits only the peer).
    pub fn spec_for(&self, me: &str) -> Result<Spec, String> {
        let mine = self.nodes.iter().find(|n| n.name == me).ok_or_else(|| {
            format!(
                "node `{me}` is not in cluster.toml (nodes: {})",
                self.nodes
                    .iter()
                    .map(|n| n.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        let worker = self.nodes.iter().find(|n| !n.head).unwrap();
        let (p, a) = if self.link == "tb" {
            (&worker.tb, &worker.lan)
        } else {
            (&worker.lan, &worker.tb)
        };
        let primary = p
            .clone()
            .or_else(|| a.clone())
            .ok_or_else(|| format!("no address for node `{}`", worker.name))?;
        let alt = if p.is_some() { a.clone() } else { None };
        let fmt = |h: &str| {
            if h.contains(':') {
                format!("[{h}]:{}", self.port)
            } else {
                format!("{h}:{}", self.port)
            }
        };
        Ok(Spec {
            role: if mine.head { Role::Head } else { Role::Worker },
            addr: if mine.head {
                fmt(&primary)
            } else {
                format!("0.0.0.0:{}", self.port)
            },
            alt: if mine.head {
                alt.map(|a| fmt(&a))
            } else {
                None
            },
            split: self.split,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: &str = r#"
[cluster]
name = "halo-pair"   # the pair
hosts = "/x/hosts.conf"
port = 7701
link = "tb"
split = 32
exact = "bitwise"

[[node]]
name = "node-a"
head = true
tb = "198.51.100.2"

[[node]]
name = "node-b"
tb = "198.51.100.1"
"#;
    const HOSTS: &str =
        "# name addr user\nnode-b   192.0.2.78   user  soak\nnode-a   192.0.2.88   user  speed\n";

    #[test]
    fn parse_resolve_and_specs() {
        let mut f = parse(T).unwrap();
        assert_eq!(f.split, 32);
        f.resolve_hosts(Some(HOSTS));
        assert_eq!(f.nodes[1].lan.as_deref(), Some("192.0.2.78"));
        let h = f.spec_for("node-a").unwrap();
        assert_eq!(h.role, Role::Head);
        assert_eq!(h.addr, "198.51.100.1:7701");
        assert_eq!(h.alt.as_deref(), Some("192.0.2.78:7701"));
        let w = f.spec_for("node-b").unwrap();
        assert_eq!(w.role, Role::Worker);
        assert_eq!(w.addr, "0.0.0.0:7701");
        assert!(f.spec_for("node-c").is_err());
        let lan = parse(&T.replace("link = \"tb\"", "link = \"lan\"")).map(|mut f| {
            f.resolve_hosts(Some(HOSTS));
            f.spec_for("node-a").unwrap()
        });
        assert_eq!(lan.unwrap().addr, "192.0.2.78:7701");
    }

    #[test]
    fn rejects_bad_files() {
        assert!(parse("[cluster]\nport = x\n").is_err());
        assert!(parse("[[node]]\nname = \"a\"\n").is_err());
        assert!(parse(&T.replace("head = true", "head = false")).is_err());
    }
}
