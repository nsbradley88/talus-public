//! Which distributions the installer knows. install.sh carries the package lists; this is the verdict doctor shows.
//!
//! | Level   | Distributions                                                    |
//! |---------|------------------------------------------------------------------|
//! | Tested  | Ubuntu 24.04 and newer (the release machine runs 26.04), and Ubuntu derivatives on those bases |
//! | Handled | Ubuntu 22.04, Debian 12+, Fedora 41+, Arch and derivatives, openSUSE Tumbleweed |
//! | Unknown | anything else: install.sh asks you to install the build tools yourself |

use crate::probe::SysInfo;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    Tested,
    Handled,
    Unknown,
}

fn ver(s: &str) -> (u32, u32) {
    let mut it = s.split('.').map(|x| x.parse::<u32>().unwrap_or(0));
    (it.next().unwrap_or(0), it.next().unwrap_or(0))
}

pub fn distro(sys: &SysInfo) -> Support {
    let like = |w: &str| sys.os_like.split_whitespace().any(|x| x == w);
    let v = ver(&sys.os_version);
    match sys.os_id.as_str() {
        "ubuntu" if v >= (24, 4) => Support::Tested,
        "ubuntu" if v >= (22, 4) => Support::Handled,
        "debian" if v.0 >= 12 => Support::Handled,
        "fedora" if v.0 >= 41 => Support::Handled,
        "arch" | "endeavouros" | "manjaro" | "cachyos" => Support::Handled,
        "opensuse-tumbleweed" => Support::Handled,
        // Mint, Pop!_OS, Zorin...: VERSION_ID is their own, so judge by family only
        _ if like("ubuntu") => Support::Handled,
        _ if like("arch") => Support::Handled,
        _ => Support::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(id: &str, like: &str, v: &str) -> SysInfo {
        SysInfo {
            os_id: id.into(),
            os_like: like.into(),
            os_version: v.into(),
            ..Default::default()
        }
    }

    #[test]
    fn matrix() {
        assert_eq!(distro(&s("ubuntu", "debian", "26.04")), Support::Tested);
        assert_eq!(distro(&s("ubuntu", "debian", "24.04")), Support::Tested);
        assert_eq!(distro(&s("ubuntu", "debian", "22.04")), Support::Handled);
        assert_eq!(distro(&s("ubuntu", "debian", "20.04")), Support::Unknown);
        assert_eq!(distro(&s("debian", "", "13")), Support::Handled);
        assert_eq!(distro(&s("debian", "", "11")), Support::Unknown);
        assert_eq!(distro(&s("fedora", "", "42")), Support::Handled);
        assert_eq!(
            distro(&s("linuxmint", "ubuntu debian", "22")),
            Support::Handled
        );
        assert_eq!(distro(&s("cachyos", "arch", "")), Support::Handled);
        assert_eq!(distro(&s("nixos", "", "25.05")), Support::Unknown);
    }
}
