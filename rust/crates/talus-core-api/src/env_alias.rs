//! `TALUS_*` names for the inherited `STRATA_*` environment variables.
//!
//! Talus keeps reading every `STRATA_*` switch it inherited from Strata (configs and service files set them), and
//! accepts the same switch under a `TALUS_` name: `TALUS_PF_FUSED=1` means `STRATA_PF_FUSED=1`. The binaries call
//! [`apply_env_aliases`] first thing in `main`, before any thread starts and before the kernel library is loaded, so
//! the Rust host and the C++ kernel library (which reads its switches with `getenv`) both see the alias.
//!
//! When both names are set, the `STRATA_*` value wins, so an existing setup behaves exactly as before.

use std::ffi::{OsStr, OsString};

const TALUS: &str = "TALUS_";
const STRATA: &str = "STRATA_";

/// The `(STRATA_* name, value)` pairs that `TALUS_*` variables in `vars` add: one per `TALUS_X` whose `STRATA_X`
/// is not in `vars`. Pure; [`apply_env_aliases`] applies it to the process environment.
pub fn env_aliases<I>(vars: I) -> Vec<(OsString, OsString)>
where
    I: IntoIterator<Item = (OsString, OsString)>,
{
    let vars: Vec<(OsString, OsString)> = vars.into_iter().collect();
    let is_set = |name: &OsStr| vars.iter().any(|(k, _)| k == name);
    let mut out: Vec<(OsString, OsString)> = Vec::new();
    for (k, v) in &vars {
        let Some(rest) = k.to_str().and_then(|k| k.strip_prefix(TALUS)) else {
            continue;
        };
        if rest.is_empty() {
            continue;
        }
        let legacy = OsString::from(format!("{STRATA}{rest}"));
        if !is_set(&legacy) && !out.iter().any(|(n, _)| *n == legacy) {
            out.push((legacy, v.clone()));
        }
    }
    out.sort();
    out
}

/// Sets `STRATA_X` from `TALUS_X` for every `TALUS_X` whose `STRATA_X` is unset, and returns the names it set.
///
/// Call it at the top of `main`, before any other thread exists (it changes the process environment).
pub fn apply_env_aliases() -> Vec<String> {
    let set = env_aliases(std::env::vars_os());
    let mut names = Vec::with_capacity(set.len());
    for (k, v) in set {
        std::env::set_var(&k, v);
        names.push(k.to_string_lossy().into_owned());
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(pairs: &[(&str, &str)]) -> Vec<(OsString, OsString)> {
        pairs
            .iter()
            .map(|(k, v)| (OsString::from(k), OsString::from(v)))
            .collect()
    }

    #[test]
    fn talus_name_sets_the_legacy_name() {
        assert_eq!(
            env_aliases(v(&[("TALUS_PF_FUSED", "1"), ("PATH", "/bin")])),
            v(&[("STRATA_PF_FUSED", "1")])
        );
    }

    #[test]
    fn legacy_name_wins_when_both_are_set() {
        assert!(env_aliases(v(&[("TALUS_API_KEY", "new"), ("STRATA_API_KEY", "old")])).is_empty());
        // order does not matter
        assert!(env_aliases(v(&[("STRATA_API_KEY", "old"), ("TALUS_API_KEY", "new")])).is_empty());
    }

    #[test]
    fn legacy_only_and_unrelated_names_are_untouched() {
        assert!(env_aliases(v(&[
            ("STRATA_PF_GEMM", "1"),
            ("GLM_PCH", "2048"),
            ("TALUS_", "x")
        ]))
        .is_empty());
    }

    #[test]
    fn empty_value_is_kept_as_set() {
        // an empty TALUS_X still means "set to empty", exactly as an empty STRATA_X would
        assert_eq!(
            env_aliases(v(&[("TALUS_LAGUNA_DFLASH", "")])),
            v(&[("STRATA_LAGUNA_DFLASH", "")])
        );
    }
}
