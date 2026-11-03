//! Evaluation-time network for the hidden tests (the agent phase stays fully offline).
//!
//! Some instances' tests need network semantics: a connect to an unroutable address must *time out*
//! (requests' connect-timeout tests; `--network none` answers ENETUNREACH at once), or a test fetches a public
//! URL (matplotlib's https imread smoke test). The eval container therefore joins the network namespace of a
//! per-evaluation gateway container (`--network container:<gw>`), which alone holds NET_ADMIN and sets:
//!
//! - an nftables output chain whose default is a blackhole (packets DROPPED, not refused, so timeouts behave as
//!   upstream); loopback stays open (test suites run local servers);
//! - the LAN, tailnet, docker host and every private / link-local / multicast range always dropped
//!   (0/8, 10/8, 100.64/10, 169.254/16, 172.16/12, 192.168/16, 224/3);
//! - DNS only through the gateway's dnsmasq on 127.0.0.1 (the eval container's resolv.conf); docker's embedded
//!   DNS is unreachable from the tests;
//! - every new connection attempt logged (nft `log group 1`, read by tcpdump on `nflog:1`) and every DNS query
//!   logged (dnsmasq `--log-queries`): the per-instance evidence kept with the case row.
//!
//! Two modes. `discover` (the gold patch only, at image prep): public internet open, names resolved live; the
//! names the gold run looks up become the instance's allowlist. `allow` (every model patch): only the
//! allowlisted names resolve (pinned when the gateway starts: no live DNS during the tests), only their
//! addresses are reachable, everything else is blackholed. A model run that contacts (or looks up) anything the
//! gold run did not is an `eval_network` flag: INVALID (e.g. a patch fetching the upstream fix from
//! raw.githubusercontent.com at import time).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const GW_IMAGE: &str = "talus-bench/egress-gw:1";
/// Version of the gateway script / evidence format; cached gold evidence of an older version is re-discovered.
pub const EVIDENCE_VERSION: u32 = 2;
/// A user-defined bridge (NAT) the gateways sit on; created at image prep.
pub const GW_NETWORK: &str = "tb-egress";

pub const GW_DOCKERFILE: &str = "FROM alpine:3.20\n\
RUN apk add --no-cache dnsmasq nftables tcpdump iproute2 bind-tools\n\
LABEL talus-bench.egress=1\n";

/// The gateway's entrypoint: `gw.sh discover|allow` with the shared directory at /tbnet
/// (`allow.txt`: one allowlisted name per line; writes `hosts.pinned`, `conn.log`, `ready`).
pub const GW_SH: &str = r#"#!/bin/sh
MODE=$1; D=/tbnet
DNSUID=$(id -u dnsmasq)
# allow mode: pin the allowlisted names now (docker's DNS), never resolve live during the tests
: > $D/hosts.pinned; ALLOW_IPS=""
if [ "$MODE" = allow ]; then
  for n in $(cat $D/allow.txt 2>/dev/null); do
    for ip in $(dig +short A "$n" | grep -E '^[0-9.]+$'); do echo "$ip $n" >> $D/hosts.pinned; ALLOW_IPS="$ALLOW_IPS $ip,"; done
  done
fi
ALLOWEL=""; [ -n "$ALLOW_IPS" ] && ALLOWEL="elements = { ${ALLOW_IPS%,} }"
if [ "$MODE" = discover ]; then FINAL="accept"; UP="meta skuid $DNSUID ip daddr 127.0.0.11 accept"; else FINAL="drop"; UP=""; fi
nft -f - <<NFT || { echo "nft failed" > $D/error; exit 1; }
table inet tb {
  set blocked { type ipv4_addr; flags interval; elements = { 0.0.0.0/8, 10.0.0.0/8, 100.64.0.0/10, 169.254.0.0/16, 172.16.0.0/12, 192.168.0.0/16, 224.0.0.0/3 } }
  set allow { type ipv4_addr; $ALLOWEL }
  chain out { type filter hook output priority 0; policy drop;
    ip daddr 127.0.0.11 jump dockerdns
    oif lo accept
    ct state established,related accept
    ct state new log group 1 queue-threshold 1
    ip daddr @blocked drop
    ip daddr @allow accept
    $FINAL
  }
  chain dockerdns { $UP
    drop
  }
  chain in { type filter hook input priority 0; policy drop;
    iif lo accept
    ct state established,related accept
  }
}
NFT
if [ "$MODE" = discover ]; then SRV="--server=127.0.0.11"; else SRV="--address=/#/"; fi
dnsmasq --keep-in-foreground --listen-address=127.0.0.1 --bind-interfaces --no-hosts --addn-hosts=$D/hosts.pinned \
  --no-resolv $SRV --log-queries=extra --log-facility=- --user=dnsmasq &
tcpdump -i nflog:1 -n -l -tt > $D/conn.log 2>$D/tcpdump.err &
sleep 1; touch $D/ready
wait
"#;

/// How the eval container reaches the network.
#[derive(Clone, Debug, PartialEq)]
pub enum NetMode {
    /// `--network none` (no gateway).
    Offline,
    /// The gold run: public internet, names recorded.
    Discover,
    /// A model run: only these names (from the gold run) resolve and are reachable.
    Allow(Vec<String>),
}

impl NetMode {
    pub fn arg(&self) -> &'static str {
        match self {
            NetMode::Offline => "offline",
            NetMode::Discover => "discover",
            NetMode::Allow(_) => "allow",
        }
    }
}

/// What one evaluation did on the network (deduplicated), kept with the case row.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct NetEvidence {
    /// [`EVIDENCE_VERSION`] when recorded (0: before versioning).
    #[serde(default)]
    pub version: u32,
    pub mode: String,
    /// Names the tests looked up (normalized; see [`normalize_name`]).
    pub queries: BTreeSet<String>,
    /// Address -> name, from dnsmasq's answers and the pinned hosts.
    pub names: BTreeMap<String, String>,
    /// Connection attempts: `tcp|udp|other ip:port` -> attempts (SYN retransmits included).
    pub attempts: BTreeMap<String, usize>,
    /// The allowlist in force (allow mode).
    pub allowlist: Vec<String>,
}

impl NetEvidence {
    /// The targets compared between runs: a looked-up name, or `ip:port` for an address no name led to.
    pub fn targets(&self) -> BTreeSet<String> {
        let mut t: BTreeSet<String> = self.queries.iter().map(|q| format!("name {q}")).collect();
        for a in self.attempts.keys() {
            let addr = a.split_whitespace().nth(1).unwrap_or("");
            let ip = addr.rsplit_once(':').map(|x| x.0).unwrap_or(addr);
            match self.names.get(ip) {
                Some(n) => {
                    t.insert(format!("name {n}"));
                }
                None => {
                    t.insert(format!("addr {addr}"));
                }
            }
        }
        t
    }

    /// The names an allow-mode run may resolve: every name the gold run looked up.
    pub fn allowlist_from_gold(&self) -> Vec<String> {
        self.queries.iter().cloned().collect()
    }
}

/// Targets of `model` that the gold run never touched and that reach past the blackhole (a name, or a public
/// address): an `eval_network` flag.
pub fn novel(gold: &NetEvidence, model: &NetEvidence) -> Vec<String> {
    novel_split(gold, model).0
}

/// (`novel`, new attempts at always-blocked private addresses: nothing can come back through the blackhole,
/// so these are an informational note).
pub fn novel_split(gold: &NetEvidence, model: &NetEvidence) -> (Vec<String>, Vec<String>) {
    let g = gold.targets();
    let (mut hard, mut soft) = (Vec::new(), Vec::new());
    for t in model.targets().difference(&g) {
        let ip = t
            .strip_prefix("addr ")
            .map(|a| a.rsplit_once(':').map(|x| x.0).unwrap_or(a));
        match ip.and_then(|i| i.parse::<std::net::Ipv4Addr>().ok()) {
            Some(a) if blocked(a) => soft.push(t.clone()),
            _ => hard.push(t.clone()),
        }
    }
    (hard, soft)
}

/// The gateway's always-dropped ranges.
pub fn blocked(a: std::net::Ipv4Addr) -> bool {
    let o = a.octets();
    o[0] == 0
        || o[0] == 10
        || (o[0] == 100 && (64..128).contains(&o[1]))
        || (o[0] == 169 && o[1] == 254)
        || (o[0] == 172 && (16..32).contains(&o[1]))
        || (o[0] == 192 && o[1] == 168)
        || o[0] >= 224
}

/// Lower-case, without the trailing dot; `None` for queries that differ per run by construction (the
/// container's own hostname, reverse lookups of private addresses).
pub fn normalize_name(name: &str, own_hostname: &str) -> Option<String> {
    let n = name.trim_end_matches('.').to_lowercase();
    if n.is_empty() || (!own_hostname.is_empty() && n.starts_with(&own_hostname.to_lowercase())) {
        return None;
    }
    if let Some(rev) = n.strip_suffix(".in-addr.arpa") {
        let mut o: Vec<&str> = rev.split('.').collect();
        o.reverse();
        let first: u32 = o.first().and_then(|x| x.parse().ok()).unwrap_or(0);
        let second: u32 = o.get(1).and_then(|x| x.parse().ok()).unwrap_or(0);
        let private = matches!(first, 0 | 10 | 127)
            || (first == 172 && (16..32).contains(&second))
            || (first == 192 && second == 168)
            || (first == 169 && second == 254)
            || (first == 100 && (64..128).contains(&second));
        if private || o.len() < 4 {
            return None;
        }
    }
    if n == "localhost" || n.ends_with(".localhost") {
        return None;
    }
    Some(n)
}

/// Parse the gateway's logs: dnsmasq (`docker logs` of the gateway), the NFLOG capture (`conn.log`) and the
/// pinned hosts.
pub fn parse(
    mode: &str,
    dns_log: &str,
    conn_log: &str,
    pinned: &str,
    own_hostname: &str,
) -> NetEvidence {
    let mut ev = NetEvidence {
        version: EVIDENCE_VERSION,
        mode: mode.to_string(),
        ..Default::default()
    };
    for l in pinned.lines() {
        let mut it = l.split_whitespace();
        if let (Some(ip), Some(name)) = (it.next(), it.next()) {
            ev.names.insert(ip.to_string(), name.to_lowercase());
        }
    }
    for l in dns_log.lines() {
        // dnsmasq[15]: 1 127.0.0.1/52115 query[A] example.com from 127.0.0.1
        // dnsmasq[15]: 1 127.0.0.1/52115 reply example.com is 104.20.23.154
        let words: Vec<&str> = l.split_whitespace().collect();
        if let Some(i) = words.iter().position(|w| w.starts_with("query[")) {
            if let Some(name) = words.get(i + 1) {
                if let Some(n) = normalize_name(name, own_hostname) {
                    ev.queries.insert(n);
                }
            }
        } else if let Some(i) = words
            .iter()
            .position(|w| *w == "reply" || w.ends_with("hosts.pinned"))
        {
            if let (Some(name), Some("is"), Some(ip)) = (
                words.get(i + 1),
                words.get(i + 2).copied(),
                words.get(i + 3),
            ) {
                if ip.parse::<std::net::IpAddr>().is_ok() {
                    ev.names.insert(ip.to_string(), name.to_lowercase());
                }
            }
        }
    }
    for l in conn_log.lines() {
        // 1793273210.94 IP 172.18.0.2.55904 > 172.66.147.243.443: Flags [S], ...
        // 1793273210.94 IP 172.18.0.2.40000 > 8.8.8.8.53: UDP, length 30
        let words: Vec<&str> = l.split_whitespace().collect();
        let Some(gt) = words.iter().position(|w| *w == ">") else {
            continue;
        };
        let Some(dst) = words.get(gt + 1).map(|d| d.trim_end_matches(':')) else {
            continue;
        };
        let (ip, port) = match dst.rsplit_once('.') {
            Some((ip, port))
                if port.chars().all(|c| c.is_ascii_digit())
                    && ip.parse::<std::net::Ipv4Addr>().is_ok() =>
            {
                (ip.to_string(), port.to_string())
            }
            _ => (dst.to_string(), String::new()),
        };
        let proto = if l.contains("Flags [") {
            "tcp"
        } else if l.contains(" UDP") {
            "udp"
        } else {
            "other"
        };
        let key = if port.is_empty() {
            format!("{proto} {ip}")
        } else {
            format!("{proto} {ip}:{port}")
        };
        *ev.attempts.entry(key).or_default() += 1;
    }
    ev
}

/// Network or fetch code added by a patch (informational: a requests/urllib3 fix legitimately touches such
/// code; what decides is what the evaluation actually contacts).
pub fn patch_network_code(patch: &str) -> Vec<String> {
    let pats = [
        "urlopen",
        "urllib.request",
        "urllib2",
        "requests.get(",
        "requests.post(",
        "requests.request(",
        "http.client",
        "httplib",
        "socket.socket(",
        "socket.create_connection",
        "httpx.",
        "aiohttp",
        "pycurl",
        "subprocess",
        "os.system(",
        "os.popen(",
        "curl ",
        "wget ",
        "git clone",
        "git fetch",
        "pip install",
        "pip download",
        "raw.githubusercontent.com",
        "github.com/",
        "pypi.org",
        "files.pythonhosted.org",
        "gitlab.com",
        "bitbucket.org",
    ];
    let mut out = Vec::new();
    let mut file = String::new();
    for l in patch.lines() {
        if let Some(f) = l.strip_prefix("+++ ") {
            file = f.trim_start_matches("b/").to_string();
            continue;
        }
        let Some(added) = l.strip_prefix('+') else {
            continue;
        };
        let low = added.to_lowercase();
        if let Some(p) = pats.iter().find(|p| low.contains(*p)) {
            if out.len() < 20 {
                out.push(format!(
                    "{file}: {p}: {}",
                    added.trim().chars().take(160).collect::<String>()
                ));
            }
        }
    }
    out
}

/// The evidence as stored in a case row (`novel` = flagged, `novel_blocked` = noted).
pub fn to_value(ev: &NetEvidence, gold: &NetEvidence) -> Value {
    let (hard, soft) = novel_split(gold, ev);
    let mut v = serde_json::to_value(ev).unwrap_or(Value::Null);
    v["novel"] = serde_json::json!(hard);
    v["novel_blocked"] = serde_json::json!(soft);
    v["gold_targets"] = serde_json::json!(gold.targets());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    // shapes recorded from the prototype gateway (dnsmasq 2.90, tcpdump 4.99 on nflog:1)
    const DNS: &str = "dnsmasq[15]: read /tbnet/hosts.pinned - 2 names\n\
dnsmasq[15]: 1 127.0.0.1/52115 query[A] example.com from 127.0.0.1\n\
dnsmasq[15]: 1 127.0.0.1/52115 /tbnet/hosts.pinned example.com is 104.20.23.154\n\
dnsmasq[15]: 3 127.0.0.1/46484 query[A] raw.githubusercontent.com from 127.0.0.1\n\
dnsmasq[15]: 3 127.0.0.1/46484 config raw.githubusercontent.com is NXDOMAIN\n\
dnsmasq[15]: 4 127.0.0.1/46485 query[PTR] 2.0.18.172.in-addr.arpa from 127.0.0.1\n\
dnsmasq[15]: 5 127.0.0.1/46486 query[A] 3fa9c1d2e4b5 from 127.0.0.1\n";
    const CONN: &str = "1793273210.942358 IP 172.18.0.2.55904 > 104.20.23.154.443: Flags [S], seq 1, win 64240, length 0\n\
1793273210.942470 IP 172.18.0.2.36477 > 10.255.255.1.80: Flags [S], seq 3, win 64240, length 0\n\
1793273212.158370 IP 172.18.0.2.36477 > 10.255.255.1.80: Flags [S], seq 3, win 64240, length 0\n\
1793273214.1 IP 172.18.0.2.40000 > 8.8.8.8.53: UDP, length 30\n";

    #[test]
    fn parses_gateway_logs() {
        let ev = parse(
            "allow",
            DNS,
            CONN,
            "104.20.23.154 example.com\n",
            "3fa9c1d2e4b5",
        );
        assert_eq!(
            ev.queries.iter().cloned().collect::<Vec<_>>(),
            vec!["example.com", "raw.githubusercontent.com"],
            "own hostname and private PTR lookups are not evidence"
        );
        assert_eq!(
            ev.attempts["tcp 10.255.255.1:80"], 2,
            "retransmits counted, one target"
        );
        assert_eq!(ev.attempts["udp 8.8.8.8:53"], 1);
        let t = ev.targets();
        assert!(t.contains("name example.com") && t.contains("addr 10.255.255.1:80"));
        assert!(
            !t.iter().any(|x| x.contains("104.20.23.154")),
            "the address maps to its name"
        );
    }

    #[test]
    fn a_fetch_the_gold_run_never_made_is_novel() {
        // gold: the connect-timeout test (blackholed literal) and the allowlisted site
        let gold = parse(
            "discover",
            "dnsmasq[9]: 1 127.0.0.1/1 query[A] example.com from 127.0.0.1\ndnsmasq[9]: 1 127.0.0.1/1 reply example.com is 93.184.216.34\n",
            "1.0 IP 172.18.0.2.1 > 10.255.255.1.80: Flags [S], seq 1, length 0\n1.1 IP 172.18.0.2.2 > 93.184.216.34.443: Flags [S], seq 1, length 0\n",
            "",
            "",
        );
        assert_eq!(gold.allowlist_from_gold(), vec!["example.com"]);
        // the same tests under the allowlist (another address for the same name): nothing novel
        let same = parse(
            "allow",
            "dnsmasq[9]: 1 127.0.0.1/1 query[A] example.com from 127.0.0.1\n",
            "1.0 IP 172.18.0.2.1 > 10.255.255.1.80: Flags [S], seq 1, length 0\n1.1 IP 172.18.0.2.2 > 104.20.23.154.443: Flags [S], seq 1, length 0\n",
            "104.20.23.154 example.com\n",
            "",
        );
        assert!(novel(&gold, &same).is_empty(), "{:?}", novel(&gold, &same));
        // a cheater fetching the upstream fix at import: the lookup alone is evidence (NXDOMAIN in allow mode)
        let cheat = parse(
            "allow",
            "dnsmasq[9]: 1 127.0.0.1/1 query[A] example.com from 127.0.0.1\ndnsmasq[9]: 2 127.0.0.1/2 query[A] raw.githubusercontent.com from 127.0.0.1\n",
            "1.0 IP 172.18.0.2.1 > 10.255.255.1.80: Flags [S], seq 1, length 0\n1.2 IP 172.18.0.2.3 > 185.199.108.133.443: Flags [S], seq 1, length 0\n",
            "",
            "",
        );
        let n = novel(&gold, &cheat);
        assert!(
            n.contains(&"name raw.githubusercontent.com".to_string()),
            "{n:?}"
        );
        assert!(
            n.contains(&"addr 185.199.108.133:443".to_string()),
            "a hard-coded address too: {n:?}"
        );
        // a new attempt at a blackholed private address is a note, not a contact
        let lan = parse(
            "allow",
            "",
            "1.0 IP 172.18.0.2.1 > 10.255.255.1.80: Flags [S], seq 1, length 0\n1.0 IP 172.18.0.2.1 > 192.168.0.1.8080: Flags [S], seq 1, length 0\n",
            "",
            "",
        );
        let (hard, soft) = novel_split(&gold, &lan);
        assert!(
            hard.is_empty() && soft == vec!["addr 192.168.0.1:8080"],
            "{hard:?} {soft:?}"
        );
    }

    #[test]
    fn patch_scan_finds_fetch_code() {
        let p = "--- a/requests/__init__.py\n+++ b/requests/__init__.py\n@@ -1,2 +1,4 @@\n import os\n+import urllib.request\n+exec(urllib.request.urlopen('https://raw.githubusercontent.com/psf/requests/main/requests/models.py').read())\n";
        let s = patch_network_code(p);
        assert!(
            s.len() >= 2 && s[0].starts_with("requests/__init__.py"),
            "{s:?}"
        );
        assert!(
            patch_network_code("+++ b/x.py\n+x = 1\n-import urllib\n").is_empty(),
            "removed lines are not added code"
        );
    }

    #[test]
    fn script_blackholes_and_blocks_private_ranges() {
        // the properties the gateway must keep: drop (never reject) by default, private ranges always dropped,
        // docker's DNS closed to the tests, loopback open, every new attempt logged
        assert!(GW_SH.contains("policy drop") && !GW_SH.contains("reject"));
        for r in [
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "100.64.0.0/10",
            "169.254.0.0/16",
        ] {
            assert!(GW_SH.contains(r), "{r}");
        }
        assert!(GW_SH.contains("ip daddr 127.0.0.11 jump dockerdns"));
        assert!(GW_SH.find("ip daddr @blocked drop") < GW_SH.find("ip daddr @allow accept"));
        assert!(
            GW_SH.contains("ct state new log group 1 queue-threshold 1"),
            "every attempt delivered at once"
        );
        assert!(
            GW_SH.contains("--address=/#/"),
            "allow mode: unknown names are NXDOMAIN"
        );
    }
}
