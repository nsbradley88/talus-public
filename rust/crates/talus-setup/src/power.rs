//! The ACPI platform profile (talus#54).
//!
//! Strix Halo firmware exposes `/sys/firmware/acpi/platform_profile` (choices in `platform_profile_choices`), and most
//! machines boot in `balanced`, which holds the APU's sustained power back. Every Talus figure was measured in
//! `performance`: with `balanced`, GLM-5.3-Flash decoded at about half the speed.
//!
//! This module reads the profile, and builds the plan that sets `performance` now and at every boot:
//!
//! - when power-profiles-daemon (or tuned-ppd) manages the profile, `powerprofilesctl set performance`: the daemon
//!   keeps the choice across reboots, and a file of ours would only fight it;
//! - otherwise one systemd-tmpfiles line, `/etc/tmpfiles.d/90-talus-platform-profile.conf`, which systemd writes to
//!   the sysfs file at boot, applied once now with `systemd-tmpfiles --create`.
//!
//! Like [`crate::boot`], nothing here changes the system: the caller explains, asks (default yes) and runs the steps.
//! It is not a boot setting: no kernel parameter, no reboot, and it can be undone at any time.

use crate::boot::{shell_join, Step};
use crate::probe::SysInfo;
use std::path::PathBuf;

pub const SYSFS: &str = "/sys/firmware/acpi/platform_profile";
pub const SYSFS_CHOICES: &str = "/sys/firmware/acpi/platform_profile_choices";
pub const WANT: &str = "performance";
pub const TMPFILES_PATH: &str = "/etc/tmpfiles.d/90-talus-platform-profile.conf";

/// Why, in plain words (setup and doctor show it).
pub const WHY: &str = "Talus is measured with the ACPI platform profile at \"performance\". Most Strix Halo machines \
    boot in \"balanced\", which holds the chip's sustained power back: GLM-5.3-Flash decoded at 11.2 tok/s with \
    balanced instead of about 24. Performance draws more power and runs the fans harder (on a laptop: warmer, \
    louder, shorter on battery).";

/// The profile on this machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// Not a Strix Halo (no gfx1151): not our business.
    NotHalo,
    /// The firmware has no ACPI platform profile (many desktops: the power mode is a BIOS setting).
    Absent,
    /// The firmware offers no `performance` choice.
    NoPerformance {
        current: String,
        choices: Vec<String>,
    },
    Performance,
    Other {
        current: String,
    },
}

pub fn status(sys: &SysInfo) -> Status {
    if !sys.has_gfx1151() {
        return Status::NotHalo;
    }
    let Some(cur) = sys.platform_profile.clone() else {
        return Status::Absent;
    };
    if cur == WANT {
        return Status::Performance;
    }
    // no choices file: assume performance exists (the kernel validates the write anyway)
    if !sys.platform_profile_choices.is_empty()
        && !sys.platform_profile_choices.iter().any(|c| c == WANT)
    {
        return Status::NoPerformance {
            current: cur,
            choices: sys.platform_profile_choices.clone(),
        };
    }
    Status::Other { current: cur }
}

/// How the profile is made to stick.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mechanism {
    /// power-profiles-daemon / tuned-ppd: it owns the profile and restores its choice at boot.
    Daemon,
    /// A systemd-tmpfiles line written at boot.
    Tmpfiles,
}

pub fn mechanism(sys: &SysInfo) -> Mechanism {
    if sys.ppd_profile.is_some() {
        Mechanism::Daemon
    } else {
        Mechanism::Tmpfiles
    }
}

/// The tmpfiles.d file.
pub fn render_tmpfiles(date: &str) -> String {
    format!(
        "# Talus: ACPI platform profile \"{WANT}\" at every boot (written by `talus setup` on {date}).\n\
         # Talus is measured in performance; with balanced, GLM-5.3-Flash decoded at 11.2 tok/s instead of about 24.\n\
         # See docs/MODELS.md \"System setup\". To undo: sudo rm {TMPFILES_PATH}, then pick another profile\n\
         # (cat {SYSFS_CHOICES}; echo balanced | sudo tee {SYSFS}).\n\
         w {SYSFS} - - - - {WANT}\n"
    )
}

pub struct Plan {
    pub mechanism: Mechanism,
    pub steps: Vec<Step>,
    /// The new file and its text (tmpfiles only).
    pub file: Option<(PathBuf, String)>,
    /// The lines to copy for "I'll do it myself".
    pub manual: String,
    /// How to undo it.
    pub undo: String,
}

pub fn plan(sys: &SysInfo, date: &str) -> Plan {
    match mechanism(sys) {
        Mechanism::Daemon => {
            let cmd: Vec<String> = vec!["powerprofilesctl".into(), "set".into(), WANT.into()];
            Plan {
                mechanism: Mechanism::Daemon,
                manual: format!("sudo {}", shell_join(&cmd)),
                steps: vec![Step::Sudo(cmd)],
                file: None,
                undo: "powerprofilesctl set balanced".into(),
            }
        }
        Mechanism::Tmpfiles => {
            let path = PathBuf::from(TMPFILES_PATH);
            let content = render_tmpfiles(date);
            let apply: Vec<String> = vec![
                "systemd-tmpfiles".into(),
                "--create".into(),
                TMPFILES_PATH.into(),
            ];
            Plan {
                mechanism: Mechanism::Tmpfiles,
                manual: format!(
                    "sudo tee {TMPFILES_PATH} >/dev/null <<'EOF'\n{content}EOF\nsudo {}",
                    shell_join(&apply)
                ),
                steps: vec![
                    Step::WriteRoot {
                        path: path.clone(),
                        content: content.clone(),
                    },
                    Step::Sudo(apply),
                ],
                file: Some((path, content)),
                undo: format!("sudo rm {TMPFILES_PATH}; echo balanced | sudo tee {SYSFS}"),
            }
        }
    }
}

/// One line for reports: the profile and its choices.
pub fn describe(sys: &SysInfo) -> String {
    match &sys.platform_profile {
        None => "none (no ACPI platform profile)".into(),
        Some(p) => {
            let mut s = p.clone();
            if !sys.platform_profile_choices.is_empty() {
                s.push_str(&format!(
                    " (choices: {})",
                    sys.platform_profile_choices.join(" ")
                ));
            }
            if let Some(d) = &sys.ppd_profile {
                s.push_str(&format!("; power-profiles-daemon: {d}"));
            }
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn halo(profile: Option<&str>, choices: &str, ppd: Option<&str>) -> SysInfo {
        SysInfo {
            gpus: vec!["gfx1151".into()],
            platform_profile: profile.map(str::to_string),
            platform_profile_choices: choices.split_whitespace().map(str::to_string).collect(),
            ppd_profile: ppd.map(str::to_string),
            ..Default::default()
        }
    }

    #[test]
    fn status_per_machine() {
        let lp = "low-power balanced performance";
        assert_eq!(
            status(&halo(Some("performance"), lp, None)),
            Status::Performance
        );
        assert_eq!(
            status(&halo(Some("balanced"), lp, None)),
            Status::Other {
                current: "balanced".into()
            }
        );
        assert_eq!(status(&halo(None, "", None)), Status::Absent);
        assert!(matches!(
            status(&halo(Some("quiet"), "quiet balanced", None)),
            Status::NoPerformance { .. }
        ));
        // no choices file: still offered (the kernel rejects a bad write)
        assert!(matches!(
            status(&halo(Some("balanced"), "", None)),
            Status::Other { .. }
        ));
        let mut other = halo(Some("balanced"), lp, None);
        other.gpus = vec!["gfx1100".into()];
        assert_eq!(status(&other), Status::NotHalo);
    }

    #[test]
    fn plan_uses_the_daemon_when_there_is_one() {
        let p = plan(&halo(Some("balanced"), "", Some("balanced")), "d");
        assert_eq!(p.mechanism, Mechanism::Daemon);
        let d: Vec<String> = p.steps.iter().map(Step::describe).collect();
        assert_eq!(d, vec!["sudo powerprofilesctl set performance"]);
        assert!(p.file.is_none());
        assert_eq!(p.manual, "sudo powerprofilesctl set performance");
    }

    #[test]
    fn plan_writes_one_tmpfiles_line_otherwise() {
        let p = plan(&halo(Some("balanced"), "", None), "2026-11-01");
        assert_eq!(p.mechanism, Mechanism::Tmpfiles);
        let d: Vec<String> = p.steps.iter().map(Step::describe).collect();
        assert_eq!(
            d,
            vec![
                "sudo install -m 0644 <the file above> /etc/tmpfiles.d/90-talus-platform-profile.conf",
                "sudo systemd-tmpfiles --create /etc/tmpfiles.d/90-talus-platform-profile.conf",
            ]
        );
        let (path, text) = p.file.unwrap();
        assert_eq!(path, PathBuf::from(TMPFILES_PATH));
        // comments and exactly one directive
        let directives: Vec<&str> = text.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(
            directives,
            vec!["w /sys/firmware/acpi/platform_profile - - - - performance"]
        );
        assert!(text.contains("2026-11-01"));
        assert!(p.manual.contains("systemd-tmpfiles --create"));
        assert!(!p.manual.contains("reboot") && !p.undo.contains("reboot"));
    }

    #[test]
    fn describe_line() {
        let s = halo(Some("balanced"), "balanced performance", Some("balanced"));
        assert_eq!(
            describe(&s),
            "balanced (choices: balanced performance); power-profiles-daemon: balanced"
        );
        assert_eq!(
            describe(&halo(None, "", None)),
            "none (no ACPI platform profile)"
        );
    }
}
