//! Per-model presets and tuning options for Talus.
//!
//! One table per model lists the options a user can change: what each one does in plain words, our tuned default,
//! what it trades off, and the real engine switch (argument or environment variable) behind it. The same table
//! drives `talus-tui` (what it shows) and the control endpoint of `talus-server` (what it accepts), so a client
//! can only ask for the switches listed here; it never sends raw arguments or environment variables.
//!
//! Invariant: picking every option's default produces no edit at all, so the run config the operator installed for
//! a model ("our tuned defaults") is served unchanged.

use std::collections::BTreeMap;

mod catalog;

pub use catalog::MODELS;

/// One change to an engine's run config.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edit {
    /// Set an environment variable of the engine process.
    Env(&'static str, &'static str),
    /// Set `flag value`: replace the value when the flag is present, else append both.
    Arg(&'static str, &'static str),
    /// Add a bare flag when it is missing.
    Flag(&'static str),
    /// Remove a flag, and its value when the next item is not a flag.
    Remove(&'static str),
}

/// How a non-default choice affects the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    /// Speed or memory only: the output is the same either way.
    Exact,
    /// Changes numerics at rounding level; our tuned default passed the quality gates (KLD, needle, agentic).
    Gated,
    /// Opt-in that measurably changes the output and did not pass the quality gates.
    Reduced,
}

impl Quality {
    pub fn label(self) -> &'static str {
        match self {
            Quality::Exact => "same output",
            Quality::Gated => "rounding-level, quality-gated",
            Quality::Reduced => "may lower quality",
        }
    }
}

/// Where an option is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    /// The everyday switches (context, drafting, prompt cache).
    Common,
    /// The switches the Accuracy / Speed level sets.
    Accuracy,
    /// Rarely needed (memory savers).
    Advanced,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Choice {
    pub value: &'static str,
    pub label: &'static str,
    pub edits: &'static [Edit],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TuneOption {
    pub id: &'static str,
    pub label: &'static str,
    /// One line: what it changes.
    pub description: &'static str,
    /// One line: what it trades off.
    pub tradeoff: &'static str,
    /// The real switch behind it, as the docs name it.
    pub switch: &'static str,
    pub group: Group,
    pub quality: Quality,
    /// Our tuned default (a `Choice::value`).
    pub default: &'static str,
    pub choices: &'static [Choice],
    /// Only in effect while another option has this value (e.g. the drafter needs speculative decoding on).
    pub requires: Option<(&'static str, &'static str)>,
}

impl TuneOption {
    pub fn choice(&self, value: &str) -> Option<&'static Choice> {
        self.choices.iter().find(|c| c.value == value)
    }

    pub fn default_choice(&self) -> &'static Choice {
        self.choice(self.default)
            .expect("catalog: default is one of the choices")
    }

    pub fn choice_index(&self, value: &str) -> usize {
        self.choices
            .iter()
            .position(|c| c.value == value)
            .unwrap_or(0)
    }
}

/// The Accuracy / Speed level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Exact reference numerics: numerics-changing defaults and opt-ins off.
    Reference,
    /// Our release defaults (recommended).
    Balanced,
    /// Balanced plus the quality-reducing opt-ins that exist for the model.
    Fast,
}

impl Level {
    pub const ALL: [Level; 3] = [Level::Reference, Level::Balanced, Level::Fast];

    pub fn id(self) -> &'static str {
        match self {
            Level::Reference => "reference",
            Level::Balanced => "balanced",
            Level::Fast => "fast",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Level::Reference => "Reference",
            Level::Balanced => "Balanced",
            Level::Fast => "Fast",
        }
    }

    pub fn parse(s: &str) -> Option<Level> {
        Level::ALL.into_iter().find(|l| l.id() == s)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ModelSpec {
    /// Stable id; also the name of the model's run config in the server's control directory (`<id>.json`).
    pub id: &'static str,
    pub name: &'static str,
    /// The talus-cli engine that serves it.
    pub engine: &'static str,
    pub summary: &'static str,
    pub weights: &'static str,
    pub drafter: &'static str,
    /// Keep at least this much `MemAvailable` while serving.
    pub min_free_gib: u32,
    pub license: &'static str,
    pub options: &'static [TuneOption],
    /// Option values the Reference level sets (everything else stays at its default).
    pub reference: &'static [(&'static str, &'static str)],
    pub reference_note: &'static str,
    /// Option values the Fast level sets. Empty when the model has no quality-reducing opt-in.
    pub fast: &'static [(&'static str, &'static str)],
    pub fast_note: &'static str,
}

/// Option id -> chosen value. Options left out are at their default.
pub type Selection = BTreeMap<String, String>;

impl ModelSpec {
    pub fn option(&self, id: &str) -> Option<&'static TuneOption> {
        self.options.iter().find(|o| o.id == id)
    }

    /// The value in effect for `id` under `sel`.
    pub fn value<'a>(&self, sel: &'a Selection, id: &str) -> Option<&'a str> {
        let opt = self.option(id)?;
        Some(sel.get(id).map(String::as_str).unwrap_or(opt.default))
    }

    /// Whether `opt` is in effect under `sel` (its `requires` is met).
    pub fn active(&self, sel: &Selection, opt: &TuneOption) -> bool {
        match opt.requires {
            None => true,
            Some((id, val)) => self.value(sel, id) == Some(val),
        }
    }

    pub fn has_fast(&self) -> bool {
        !self.fast.is_empty()
    }

    /// The selection a level stands for.
    pub fn level_selection(&self, level: Level) -> Selection {
        let pairs: &[(&str, &str)] = match level {
            Level::Reference => self.reference,
            Level::Balanced => &[],
            Level::Fast => self.fast,
        };
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// Drop entries equal to their default, so selections compare by meaning.
    pub fn normalize(&self, sel: &Selection) -> Selection {
        sel.iter()
            .filter(|(k, v)| {
                self.option(k)
                    .map(|o| o.default != v.as_str())
                    .unwrap_or(true)
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// The level whose Accuracy-group values match `sel` (other groups don't count), or `None` (custom).
    pub fn level_of(&self, sel: &Selection) -> Option<Level> {
        let acc = |s: &Selection| -> Selection {
            self.normalize(s)
                .into_iter()
                .filter(|(k, _)| {
                    self.option(k)
                        .map(|o| o.group == Group::Accuracy)
                        .unwrap_or(false)
                })
                .collect()
        };
        let mine = acc(sel);
        // Balanced first: for a model without a fast opt-in, Fast and Balanced are the same selection.
        [Level::Balanced, Level::Reference, Level::Fast]
            .into_iter()
            .find(|&l| acc(&self.level_selection(l)) == mine)
    }

    /// Set the Accuracy-group options to `level`, keeping the other groups as they are.
    pub fn apply_level(&self, sel: &Selection, level: Level) -> Selection {
        let mut out: Selection = sel
            .iter()
            .filter(|(k, _)| {
                self.option(k)
                    .map(|o| o.group != Group::Accuracy)
                    .unwrap_or(false)
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        out.extend(self.level_selection(level));
        self.normalize(&out)
    }

    /// Check that every entry names an option of this model and one of its choices.
    pub fn validate(&self, sel: &Selection) -> Result<(), String> {
        for (k, v) in sel {
            let opt = self
                .option(k)
                .ok_or_else(|| format!("{}: unknown option {k:?}", self.id))?;
            if opt.choice(v).is_none() {
                return Err(format!("{}: option {k:?} has no choice {v:?}", self.id));
            }
        }
        Ok(())
    }

    /// The edits `sel` makes to the model's run config, in catalog order.
    pub fn edits(&self, sel: &Selection) -> Result<Vec<Edit>, String> {
        self.validate(sel)?;
        let mut out = Vec::new();
        for opt in self.options {
            let v = sel.get(opt.id).map(String::as_str).unwrap_or(opt.default);
            if v == opt.default || !self.active(sel, opt) {
                continue;
            }
            out.extend_from_slice(opt.choice(v).expect("validated").edits);
        }
        Ok(out)
    }

    /// Human-readable list of what differs between two selections (for the confirmation dialog).
    pub fn diff(&self, from: &Selection, to: &Selection) -> Vec<String> {
        let mut out = Vec::new();
        for opt in self.options {
            let a = from.get(opt.id).map(String::as_str).unwrap_or(opt.default);
            let b = to.get(opt.id).map(String::as_str).unwrap_or(opt.default);
            if a != b {
                let la = opt.choice(a).map(|c| c.label).unwrap_or(a);
                let lb = opt.choice(b).map(|c| c.label).unwrap_or(b);
                out.push(format!("{}: {la} -> {lb}", opt.label));
            }
        }
        out
    }
}

/// Option choices that need an extra weights file next to the model's `--model` GGUF:
/// (model id, option id, choice value, file suffix replacing ".gguf"). The server reports a missing file, refuses to
/// apply that choice and the TUI shows "weights not installed" (the engine itself also falls back with a log line).
pub static FILE_NEEDS: &[(&str, &str, &str, &str)] = &[
    ("qwen3.8-27b", "w4_prompt", "on", ".w4i"),
    ("qwen3.8-27b", "w4_fast", "on", ".fast.w4i"),
];

/// The file a choice needs next to `gguf` (FILE_NEEDS), if any.
pub fn needed_file(
    model_id: &str,
    option: &str,
    value: &str,
    gguf: &std::path::Path,
) -> Option<std::path::PathBuf> {
    FILE_NEEDS
        .iter()
        .find(|(m, o, v, _)| *m == model_id && *o == option && *v == value)
        .and_then(|(_, _, _, suffix)| {
            let stem = gguf.file_stem()?.to_string_lossy().into_owned();
            Some(gguf.with_file_name(format!("{stem}{suffix}")))
        })
}

pub fn model(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == id)
}

/// Apply `edits` to an engine's arguments and environment.
pub fn apply_edits(args: &mut Vec<String>, env: &mut Vec<(String, String)>, edits: &[Edit]) {
    for e in edits {
        match *e {
            Edit::Env(k, v) => {
                if let Some(slot) = env.iter_mut().find(|(ek, _)| ek == k) {
                    slot.1 = v.to_string();
                } else {
                    env.push((k.to_string(), v.to_string()));
                }
            }
            Edit::Arg(flag, v) => {
                if let Some(i) = args.iter().position(|a| a == flag) {
                    if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                        args[i + 1] = v.to_string();
                    } else {
                        args.insert(i + 1, v.to_string());
                    }
                } else {
                    args.push(flag.to_string());
                    args.push(v.to_string());
                }
            }
            Edit::Flag(flag) => {
                if !args.iter().any(|a| a == flag) {
                    args.push(flag.to_string());
                }
            }
            Edit::Remove(flag) => {
                while let Some(i) = args.iter().position(|a| a == flag) {
                    let with_value = i + 1 < args.len() && !args[i + 1].starts_with("--");
                    args.remove(i);
                    if with_value {
                        args.remove(i);
                    }
                }
            }
        }
    }
}

/// Human-readable form of an edit (for logs and the dry-run reply).
pub fn describe_edit(e: &Edit) -> String {
    match *e {
        Edit::Env(k, v) => format!("{k}={v}"),
        Edit::Arg(f, v) => format!("{f} {v}"),
        Edit::Flag(f) => f.to_string(),
        Edit::Remove(f) => format!("(without {f})"),
    }
}

/// Where the server listens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bind {
    /// 127.0.0.1: only this computer.
    Local,
    /// 0.0.0.0: the local network too (needs an API key).
    Lan,
}

impl Bind {
    pub fn id(self) -> &'static str {
        match self {
            Bind::Local => "local",
            Bind::Lan => "lan",
        }
    }

    pub fn host(self) -> &'static str {
        match self {
            Bind::Local => "127.0.0.1",
            Bind::Lan => "0.0.0.0",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Bind::Local => "This computer only",
            Bind::Lan => "Local network (API key required)",
        }
    }

    pub fn parse(s: &str) -> Option<Bind> {
        match s {
            "local" => Some(Bind::Local),
            "lan" => Some(Bind::Lan),
            _ => None,
        }
    }

    pub fn from_host(host: &str) -> Bind {
        match host {
            "127.0.0.1" | "localhost" | "::1" => Bind::Local,
            _ => Bind::Lan,
        }
    }
}

/// Smallest API key length the control endpoint accepts.
pub const MIN_API_KEY_LEN: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;

    fn s(pairs: &[(&str, &str)]) -> Selection {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn five_models_with_unique_ids() {
        assert_eq!(MODELS.len(), 5);
        let mut ids: Vec<_> = MODELS.iter().map(|m| m.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn catalog_is_well_formed() {
        for m in MODELS {
            let mut ids: Vec<_> = m.options.iter().map(|o| o.id).collect();
            ids.sort();
            ids.dedup();
            assert_eq!(ids.len(), m.options.len(), "{}: duplicate option id", m.id);
            for o in m.options {
                assert!(o.choice(o.default).is_some(), "{}/{}: default", m.id, o.id);
                assert!(
                    o.default_choice().edits.is_empty(),
                    "{}/{}: the tuned default must not edit the run config",
                    m.id,
                    o.id
                );
                assert!(o.choices.len() >= 2, "{}/{}: needs a choice", m.id, o.id);
                for c in o.choices {
                    if c.value != o.default {
                        assert!(
                            !c.edits.is_empty(),
                            "{}/{}/{}: no edit",
                            m.id,
                            o.id,
                            c.value
                        );
                    }
                }
                assert!(!o.description.is_empty() && !o.tradeoff.is_empty());
                assert!(
                    o.description.len() <= 120,
                    "{}/{}: description too long",
                    m.id,
                    o.id
                );
                assert!(
                    o.tradeoff.len() <= 140,
                    "{}/{}: trade-off too long",
                    m.id,
                    o.id
                );
            }
            // every model has the common context and speculative-decoding options
            assert!(m.option("context").is_some(), "{}", m.id);
            assert!(m.option("spec").is_some(), "{}", m.id);
            for (k, v) in m.reference.iter().chain(m.fast.iter()) {
                let o = m
                    .option(k)
                    .unwrap_or_else(|| panic!("{}: level names {k}", m.id));
                assert!(o.choice(v).is_some(), "{}: level value {k}={v}", m.id);
                assert_eq!(
                    o.group,
                    Group::Accuracy,
                    "{}: levels only set accuracy options",
                    m.id
                );
            }
            // Fast only turns on quality-reducing opt-ins; Reference never does
            for (k, _) in m.fast {
                assert_eq!(
                    m.option(k).unwrap().quality,
                    Quality::Reduced,
                    "{}/{k}",
                    m.id
                );
            }
            for (k, _) in m.reference {
                assert_ne!(
                    m.option(k).unwrap().quality,
                    Quality::Reduced,
                    "{}/{k}",
                    m.id
                );
            }
            assert!(!m.fast_note.is_empty() && !m.reference_note.is_empty());
        }
    }

    #[test]
    fn defaults_make_no_edits() {
        for m in MODELS {
            assert!(m.edits(&Selection::new()).unwrap().is_empty());
            let explicit: Selection = m
                .options
                .iter()
                .map(|o| (o.id.to_string(), o.default.to_string()))
                .collect();
            assert!(m.edits(&explicit).unwrap().is_empty());
            assert_eq!(m.level_of(&explicit), Some(Level::Balanced));
        }
    }

    #[test]
    fn levels_round_trip() {
        for m in MODELS {
            for l in Level::ALL {
                let sel = m.apply_level(&Selection::new(), l);
                let got = m.level_of(&sel).unwrap();
                if l == Level::Fast && !m.has_fast() {
                    assert_eq!(got, Level::Balanced, "{}", m.id);
                } else {
                    assert_eq!(got, l, "{}", m.id);
                }
            }
        }
    }

    #[test]
    fn level_keeps_common_choices() {
        let m = model("mimo-v2.6-flash").unwrap();
        let sel = s(&[("context", "65536")]);
        let fast = m.apply_level(&sel, Level::Fast);
        assert_eq!(fast.get("context").map(String::as_str), Some("65536"));
        assert_eq!(fast.get("iq1m").map(String::as_str), Some("on"));
        assert_eq!(m.level_of(&fast), Some(Level::Fast));
        let back = m.apply_level(&fast, Level::Balanced);
        assert_eq!(back, s(&[("context", "65536")]));
    }

    #[test]
    fn validate_rejects_unknown() {
        let m = model("qwen3.8-27b").unwrap();
        assert!(m.validate(&s(&[("bogus", "1")])).is_err());
        assert!(m.validate(&s(&[("context", "123")])).is_err());
        assert!(m.validate(&s(&[("context", "32768")])).is_ok());
        // an option of another model is unknown here
        assert!(m.validate(&s(&[("iq1m", "on")])).is_err());
    }

    #[test]
    fn apply_edits_semantics() {
        let mut args: Vec<String> = [
            "--model",
            "m.gguf",
            "--max-context",
            "65536",
            "--ngram",
            "24",
            "--serve",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let mut env = vec![("A".to_string(), "1".to_string())];
        apply_edits(
            &mut args,
            &mut env,
            &[
                Edit::Arg("--max-context", "32768"),
                Edit::Remove("--ngram"),
                Edit::Flag("--no-mtp"),
                Edit::Flag("--no-mtp"),
                Edit::Arg("--prompt-cache", "0"),
                Edit::Env("A", "0"),
                Edit::Env("B", "1"),
                Edit::Remove("--absent"),
            ],
        );
        assert_eq!(
            args,
            [
                "--model",
                "m.gguf",
                "--max-context",
                "32768",
                "--serve",
                "--no-mtp",
                "--prompt-cache",
                "0"
            ]
        );
        assert_eq!(
            env,
            vec![("A".into(), "0".into()), ("B".into(), "1".into())]
        );
    }

    #[test]
    fn spec_off_removes_27b_drafters() {
        let m = model("qwen3.8-27b").unwrap();
        let mut args: Vec<String> = "--model x --max-context 65536 --mtp-gguf a --mtp 5 --spec-draft-vocab v --ngram 24 --dflash-gguf d --serve"
            .split(' ')
            .map(String::from)
            .collect();
        let mut env = vec![];
        apply_edits(
            &mut args,
            &mut env,
            &m.edits(&s(&[("spec", "off")])).unwrap(),
        );
        assert_eq!(args.join(" "), "--model x --max-context 65536 --serve");
    }

    #[test]
    fn reference_and_fast_edits() {
        let lag = model("laguna-s-2.1").unwrap();
        let r = lag.edits(&lag.level_selection(Level::Reference)).unwrap();
        assert!(r.contains(&Edit::Remove("--sparse")));
        assert!(r.contains(&Edit::Env("TALUS_LAGUNA_LSRV", "0")));
        let f = lag.edits(&lag.level_selection(Level::Fast)).unwrap();
        assert_eq!(f, vec![Edit::Env("TALUS_LAGUNA_VERIFY_GEMM", "9")]);
        let mimo = model("mimo-v2.6-flash").unwrap();
        let f = mimo.edits(&mimo.level_selection(Level::Fast)).unwrap();
        assert_eq!(f, vec![Edit::Env("MIMO_IQ1M_TPMOE", "1")]);
        let fnx = model("qwen3.8-flash-next").unwrap();
        let f = fnx.edits(&fnx.level_selection(Level::Fast)).unwrap();
        assert_eq!(f, vec![Edit::Env("TALUS_FN_HC_BF16", "2")]);
        let mut args: Vec<String> = [
            "--mtp",
            "rt",
            "--mtp-draft-vocab",
            "data/draft_vocab_en.bin",
            "--kv",
            "int8",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let mut env = vec![];
        apply_edits(
            &mut args,
            &mut env,
            &fnx.edits(&s(&[("draft_vocab", "full")])).unwrap(),
        );
        assert_eq!(args.join(" "), "--mtp rt --kv int8");
        assert!(!model("glm-5.3-flash").unwrap().has_fast());
        // 27B (rc-a4): 4-bit prompt math is opt-in (off by default since 2026-10-28); Fast = the all-layer profile
        let q = model("qwen3.8-27b").unwrap();
        let f = q.edits(&q.level_selection(Level::Fast)).unwrap();
        assert_eq!(f, vec![Edit::Env("TALUS_Q27_W4I", "fast")]);
        let b = q.edits(&q.level_selection(Level::Balanced)).unwrap();
        assert!(!b.iter().any(|e| matches!(e, Edit::Env("TALUS_Q27_W4I", _))));
        let r = q.edits(&q.level_selection(Level::Reference)).unwrap();
        assert!(r.contains(&Edit::Env("TALUS_PF_FA", "0")));
        assert!(!r.iter().any(|e| matches!(e, Edit::Env("TALUS_Q27_W4I", _))));
    }

    #[test]
    fn file_needs_name_real_options() {
        for (m, o, v, _) in FILE_NEEDS {
            let spec = model(m).unwrap();
            assert!(
                spec.option(o).and_then(|x| x.choice(v)).is_some(),
                "{m}/{o}={v}"
            );
        }
        let g = std::path::Path::new("/m/Qwen3.8-27B-UD-IQ4_XS.gguf");
        assert_eq!(
            needed_file("qwen3.8-27b", "w4_fast", "on", g).unwrap(),
            std::path::Path::new("/m/Qwen3.8-27B-UD-IQ4_XS.fast.w4i")
        );
        assert!(needed_file("qwen3.8-27b", "w4_fast", "off", g).is_none());
    }

    #[test]
    fn requires_skips_inactive_options() {
        let m = model("qwen3.8-27b").unwrap();
        let e = m.edits(&s(&[("spec", "off"), ("drafter", "mtp")])).unwrap();
        assert!(!e.contains(&Edit::Arg("--drafter", "mtp")));
        let e = m.edits(&s(&[("drafter", "mtp")])).unwrap();
        assert_eq!(e, vec![Edit::Arg("--drafter", "mtp")]);
        for m in MODELS {
            for o in m.options {
                if let Some((id, val)) = o.requires {
                    assert!(
                        m.option(id).and_then(|r| r.choice(val)).is_some(),
                        "{}/{}",
                        m.id,
                        o.id
                    );
                }
            }
        }
    }

    #[test]
    fn diff_names_changes() {
        let m = model("glm-5.3-flash").unwrap();
        let d = m.diff(
            &Selection::new(),
            &s(&[("context", "32768"), ("spec", "off")]),
        );
        assert_eq!(d.len(), 2);
        assert!(d[0].starts_with("Context length: 64K -> 32K"), "{d:?}");
    }

    #[test]
    fn bind_helpers() {
        assert_eq!(Bind::from_host("127.0.0.1"), Bind::Local);
        assert_eq!(Bind::from_host("0.0.0.0"), Bind::Lan);
        assert_eq!(Bind::parse("lan").unwrap().host(), "0.0.0.0");
    }
}
