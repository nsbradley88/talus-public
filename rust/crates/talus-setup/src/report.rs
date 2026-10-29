//! `talus doctor --report`: a Markdown report to paste into a GitHub issue, with secrets and personal details
//! removed: API keys and Hugging Face tokens, the home directory and user name, the host name, IP addresses.

use crate::checks::{Check, Verdict};
use crate::probe::{cmdline_value, SysInfo};

pub struct Redactor {
    /// Literal strings to hide (longest first), with what replaces them.
    literals: Vec<(String, String)>,
}

impl Redactor {
    pub fn new(home: &str, user: &str, host: &str, secrets: &[String]) -> Redactor {
        let mut literals: Vec<(String, String)> = Vec::new();
        for s in secrets {
            if s.len() >= 8 {
                literals.push((s.clone(), "<redacted>".into()));
            }
        }
        if home.len() > 1 {
            literals.push((home.to_string(), "~".into()));
        }
        if user.len() >= 3 {
            literals.push((format!("/home/{user}"), "~".into()));
        }
        if host.len() >= 3 {
            literals.push((host.to_string(), "<host>".into()));
        }
        if user.len() >= 3 {
            literals.push((user.to_string(), "<user>".into()));
        }
        Redactor { literals }
    }

    /// From the environment: $HOME, $USER, the host name, $HF_TOKEN and the API key file.
    pub fn from_env(paths: &crate::paths::Paths) -> Redactor {
        let home = paths.home.display().to_string();
        let user = std::env::var("USER").unwrap_or_default();
        let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map(|h| h.trim().to_string())
            .unwrap_or_default();
        let mut secrets = Vec::new();
        for k in [
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "STRATA_API_KEY",
            "TALUS_API_KEY",
        ] {
            if let Ok(v) = std::env::var(k) {
                secrets.push(v.trim().to_string());
            }
        }
        if let Ok(k) = std::fs::read_to_string(paths.api_key()) {
            secrets.push(k.trim().to_string());
        }
        Redactor::new(&home, &user, &host, &secrets)
    }

    pub fn apply(&self, s: &str) -> String {
        let mut out = s.to_string();
        for (from, to) in &self.literals {
            out = out.replace(from.as_str(), to);
        }
        let out = redact_tokens(&out);
        redact_ips(&out)
    }
}

/// Things that look like credentials: `hf_...`, `sk-...`, and the value after `api_key` / `token` / `Bearer`.
fn redact_tokens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let words: Vec<&str> = s.split_inclusive(|c: char| c.is_whitespace()).collect();
    let mut hide_next = false;
    for w in words {
        let core = w.trim_end();
        let tail = &w[core.len()..];
        let bare = core.trim_matches(|c: char| "\"',;:{}()[]".contains(c));
        let secretish = (bare.starts_with("hf_") || bare.starts_with("sk-")) && bare.len() >= 20;
        if hide_next && bare.len() >= 8 {
            out.push_str("<redacted>");
            out.push_str(tail);
            hide_next = false;
            continue;
        }
        if secretish {
            out.push_str(&core.replace(bare, "<redacted>"));
            out.push_str(tail);
            continue;
        }
        let lower = bare.to_ascii_lowercase();
        hide_next = lower == "bearer"
            || lower.ends_with("api_key")
            || lower.ends_with("api-key")
            || lower.ends_with("token");
        if let Some((k, v)) = core.split_once('=') {
            let kl = k.to_ascii_lowercase();
            if (kl.contains("key") || kl.contains("token")) && v.len() >= 8 {
                out.push_str(&format!("{k}=<redacted>{tail}"));
                hide_next = false;
                continue;
            }
        }
        out.push_str(w);
    }
    out
}

fn is_ipv4(t: &str) -> bool {
    let parts: Vec<&str> = t.split('.').collect();
    parts.len() == 4
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.len() <= 3 && p.parse::<u8>().is_ok())
}

fn is_ipv6(t: &str) -> bool {
    let colons = t.matches(':').count();
    (t.contains("::") || colons >= 4)
        && colons >= 2
        && t.chars().all(|c| c.is_ascii_hexdigit() || c == ':')
        && t.chars().any(|c| c.is_ascii_hexdigit())
}

/// IP addresses become `<ip>`, except the loopback and wildcard addresses (they say how the server listens).
fn redact_ips(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut tok = String::new();
    let flush = |tok: &mut String, out: &mut String| {
        let (body, port) = match tok.rsplit_once(':') {
            Some((b, p)) if is_ipv4(b) && p.chars().all(|c| c.is_ascii_digit()) => {
                (b.to_string(), format!(":{p}"))
            }
            _ => (tok.clone(), String::new()),
        };
        if (is_ipv4(&body) && body != "127.0.0.1" && body != "0.0.0.0")
            || (is_ipv6(&body) && body != "::1" && body != "::")
        {
            out.push_str("<ip>");
            out.push_str(&port);
        } else {
            out.push_str(tok);
        }
        tok.clear();
    };
    for c in s.chars() {
        if c.is_ascii_hexdigit() || c == '.' || c == ':' {
            tok.push(c);
        } else {
            flush(&mut tok, &mut out);
            out.push(c);
        }
    }
    flush(&mut tok, &mut out);
    out
}

pub fn verdict_word(v: Verdict) -> &'static str {
    match v {
        Verdict::Pass => "ok",
        Verdict::Warn => "warn",
        Verdict::Fail => "FAIL",
        Verdict::Info => "info",
        Verdict::Skip => "--",
    }
}

/// The report, before redaction.
pub fn render(checks: &[Check], sys: &SysInfo, talus_version: &str, rocm: &str) -> String {
    let mut s = String::new();
    s.push_str("### talus doctor report\n\n");
    s.push_str(&format!("- Talus: {talus_version}\n"));
    s.push_str(&format!(
        "- OS: {} (kernel {}, {})\n",
        sys.os_pretty, sys.kernel, sys.arch
    ));
    s.push_str(&format!("- CPU: {}\n", sys.cpu_model));
    s.push_str(&format!(
        "- GPU targets: {}\n",
        if sys.gpus.is_empty() {
            "none".to_string()
        } else {
            sys.gpus.join(", ")
        }
    ));
    s.push_str(&format!(
        "- Memory: {:.1} GiB total, {:.1} GiB available\n",
        sys.mem_total_gib(),
        sys.mem_avail_gib()
    ));
    s.push_str(&format!("- ROCm: {rocm}\n"));
    let kp: Vec<String> = [
        "ttm.pages_limit",
        "ttm.page_pool_size",
        "kho",
        "amdgpu.lockup_timeout",
        "amd_iommu",
    ]
    .iter()
    .filter_map(|k| cmdline_value(&sys.cmdline, k).map(|v| format!("{k}={v}")))
    .collect();
    s.push_str(&format!(
        "- Kernel parameters: {}\n",
        if kp.is_empty() {
            "none of ours".to_string()
        } else {
            kp.join(" ")
        }
    ));
    s.push_str(&format!(
        "- Live ttm.pages_limit: {}; GTT: {}\n",
        sys.ttm_pages_limit
            .map(|v| v.to_string())
            .unwrap_or_else(|| "?".into()),
        sys.gtt_total
            .map(|b| format!("{:.0} GiB", b as f64 / (1u64 << 30) as f64))
            .unwrap_or_else(|| "?".into())
    ));
    s.push_str(&format!(
        "- Desktop session: {}; default target: {}\n\n",
        if sys.desktop_active { "running" } else { "no" },
        sys.default_target.as_deref().unwrap_or("?")
    ));
    s.push_str("| Check | Result | Detail |\n|---|---|---|\n");
    for c in checks {
        s.push_str(&format!(
            "| {} | {} | {} |\n",
            c.title,
            verdict_word(c.verdict),
            c.detail.replace('|', "/")
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redaction() {
        let r = Redactor::new(
            "/home/1001",
            "alice",
            "halobox",
            &["s3cr3t-key-0123456789".to_string()],
        );
        let t = r.apply(
            "models in /home/1001/.local/share/talus on halobox, user alice; key s3cr3t-key-0123456789 \
             HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz \"api_key\": \"zzzzzzzzzzzz\" Authorization: Bearer abcdefghijklmnop \
             server 192.0.2.10:8000 and 127.0.0.1:8000 and 0.0.0.0, v6 fe80::1c2b:3cff:fe4d:5e6f, gfx1151 10.2.0a20261014",
        );
        assert!(
            t.contains("models in ~/.local/share/talus on <host>, user <user>"),
            "{t}"
        );
        assert!(!t.contains("s3cr3t"));
        assert!(!t.contains("hf_abc"));
        assert!(!t.contains("zzzzzzzzzzzz"));
        assert!(!t.contains("abcdefghijklmnop"));
        assert!(
            t.contains("server <ip>:8000 and 127.0.0.1:8000 and 0.0.0.0"),
            "{t}"
        );
        assert!(!t.contains("fe80"));
        assert!(
            t.contains("gfx1151 10.2.0a20261014"),
            "versions are not IPs: {t}"
        );
    }

    #[test]
    fn ip_shapes() {
        assert!(is_ipv4("198.51.100.7"));
        assert!(!is_ipv4("10.2.0"));
        assert!(!is_ipv4("1.2.3.400"));
        assert!(is_ipv6("2001:db8::1"));
        assert!(!is_ipv6("17:38:40"));
        assert_eq!(redact_ips("at 17:38:40 ok"), "at 17:38:40 ok");
    }

    #[test]
    fn report_has_a_table() {
        let sys = SysInfo {
            os_pretty: "Ubuntu 26.04.1 LTS".into(),
            cmdline: "ro kho=off root=/dev/mapper/x ttm.pages_limit=31457280".into(),
            ..Default::default()
        };
        let checks = vec![crate::checks::check_ram(&sys)];
        let r = render(&checks, &sys, "abc123", "10.2.0a20261014");
        assert!(r.contains("| Memory | FAIL |"));
        assert!(r.contains("Kernel parameters: ttm.pages_limit=31457280 kho=off"));
        assert!(
            !r.contains("root=/dev/mapper"),
            "only our parameters are reported"
        );
    }
}
