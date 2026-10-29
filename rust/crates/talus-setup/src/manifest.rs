//! What the installer downloads for each model, and what it tells the user before they choose.
//!
//! Names, licenses, memory floors and the trade-off text of each option come from `talus_presets::MODELS` (the
//! catalog the TUI and the server use). This file adds only what the catalog does not have: the files to download
//! (Hugging Face repository, path, size, SHA-256, exactly as docs/MODELS.md lists them), how well each model fits
//! next to a desktop session (docs/MODELS.md "System setup") and a one-line speed headline from the release run
//! (docs/RESULTS.md, release `RELEASE`).
//!
//! The SHA-256 values are the ones Hugging Face publishes for each file (the LFS object id), and for our own
//! repositories (j2ipper/*) also the repositories' SHA256SUMS files; `download::published_sha256` checks them again
//! online before each download.

use talus_presets::ModelSpec;

/// The release the speed headlines and memory figures come from.
pub const RELEASE: &str = "43c94474";
/// The ROCm SDK the release was built and measured with: the repository's `rocm.version` file, which install.sh
/// reads too (one place to re-pin).
pub const ROCM_VERSION_FILE: &str = include_str!("../../../../rocm.version");

/// The pinned version: the first line of `rocm.version` that is not a `#` comment.
pub fn rocm_version() -> &'static str {
    parse_rocm_version(ROCM_VERSION_FILE).expect("rocm.version names a version")
}

pub fn parse_rocm_version(text: &str) -> Option<&str> {
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
}
/// The model a first install recommends: the smallest, and it fits next to a desktop.
pub const DEFAULT_MODEL: &str = "qwen3.8-27b";

/// What a file is for. Everything but `Weights` can be left out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The model itself.
    Weights,
    /// A drafter or draft head (speculative decoding). On by default.
    Drafter,
    /// A 64K draft vocabulary (makes the drafter cheaper). On with the drafters.
    DraftVocab,
    /// A drafter we do not download by default (its license is unclear). Opt-in, needs the drafters on.
    DrafterOptIn,
    /// Qwen3.8-27B's 4-bit prompt weights (`TALUS_Q27_W4I=auto`). Opt-in.
    W4i,
    /// Qwen3.8-27B's Fast-tier prompt weights (`TALUS_Q27_W4I=fast`). Opt-in; failed the quality gate.
    W4iFast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileSpec {
    pub repo: &'static str,
    /// Path inside the repository.
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub role: Role,
}

impl FileSpec {
    /// The file's name in the model's directory (the repository path without its folders).
    pub fn local_name(&self) -> &'static str {
        self.path.rsplit('/').next().unwrap_or(self.path)
    }

    /// Our own repositories publish a SHA256SUMS file next to the files.
    pub fn has_sha256sums(&self) -> bool {
        self.repo.starts_with("j2ipper/")
    }
}

/// How well a model fits next to a desktop session (docs/MODELS.md "System setup").
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DesktopFit {
    Yes,
    Tight,
    HeadlessOnly,
}

impl DesktopFit {
    pub fn label(self) -> &'static str {
        match self {
            DesktopFit::Yes => "yes",
            DesktopFit::Tight => "tight",
            DesktopFit::HeadlessOnly => "headless only",
        }
    }
}

/// Extra preparation after the download.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prep {
    None,
    /// Qwen3.8-Flash-Next: a Strata pack from the GGUF (`tools/iq_pack.py`) and, with the drafters, the MTP draft
    /// layer built from the original checkpoint's MTP tensors (`tools/mtp_fetch.py`, `mtp_pack.py`, `mtp_rt.py`).
    FlashNextPack,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelInstall {
    /// A `talus_presets` model id.
    pub id: &'static str,
    pub files: &'static [FileSpec],
    pub prep: Prep,
    /// Disk the preparation writes besides the downloads (bytes), without and with the drafters.
    pub prep_bytes: (u64, u64),
    pub desktop_fit: DesktopFit,
    /// Needs the GPU memory pool raised (`ttm.pages_limit`) and the kexec-handover reserve freed (`kho=off`).
    pub needs_big_gtt: bool,
    /// The engine refuses to start below this `MemAvailable` (GiB).
    pub min_start_gib: Option<u32>,
    /// Lowest `MemAvailable` (GiB) in the release run (headless).
    pub release_free_gib: u32,
    /// Release run: answer speed in chat (tok/s) and prompt speed at 2K (tok/s).
    pub chat_tps: f32,
    pub prompt_tps: f32,
    /// Shipped context, for the table.
    pub context: &'static str,
}

impl ModelInstall {
    pub fn spec(&self) -> &'static ModelSpec {
        talus_presets::model(self.id).expect("manifest ids are catalog ids")
    }

    pub fn weights(&self) -> impl Iterator<Item = &'static FileSpec> {
        self.files.iter().filter(|f| f.role == Role::Weights)
    }

    /// The first weights file (shard 1): what `--model` / `--native` and the tokenizer point at.
    pub fn main_file(&self) -> &'static FileSpec {
        self.weights().next().expect("every model has weights")
    }

    pub fn has_role(&self, role: Role) -> bool {
        self.files.iter().any(|f| f.role == role)
    }

    pub fn file(&self, role: Role) -> Option<&'static FileSpec> {
        self.files.iter().find(|f| f.role == role)
    }

    /// The files an install with these choices downloads.
    pub fn selected_files(&self, c: &Choices) -> Vec<&'static FileSpec> {
        self.files.iter().filter(|f| c.wants(f.role)).collect()
    }

    /// Bytes of the files these choices download (before subtracting what is already there).
    pub fn download_bytes(&self, c: &Choices) -> u64 {
        self.selected_files(c).iter().map(|f| f.size).sum()
    }

    pub fn prep_bytes_for(&self, c: &Choices) -> u64 {
        if c.drafters {
            self.prep_bytes.1
        } else {
            self.prep_bytes.0
        }
    }
}

/// The per-model choices of an install.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Choices {
    /// Drafters and draft vocabularies (speculative decoding). On by default.
    pub drafters: bool,
    /// Qwen3.8-27B's 4-bit prompt weights (`.w4i`, 2 GiB). Opt-in.
    #[serde(default)]
    pub w4i: bool,
    /// Qwen3.8-27B's Fast-tier prompt weights (`.fast.w4i`, 8 GiB). Opt-in, failed the quality gate.
    #[serde(default)]
    pub w4i_fast: bool,
    /// Opt-in drafters (Laguna-S-2.1's DFlash drafter: license unclear). Off by default.
    #[serde(default)]
    pub opt_drafter: bool,
}

impl Default for Choices {
    fn default() -> Self {
        Choices {
            drafters: true,
            w4i: false,
            w4i_fast: false,
            opt_drafter: false,
        }
    }
}

impl Choices {
    pub fn wants(&self, role: Role) -> bool {
        match role {
            Role::Weights => true,
            Role::Drafter | Role::DraftVocab => self.drafters,
            Role::W4i => self.w4i,
            Role::W4iFast => self.w4i_fast,
            Role::DrafterOptIn => self.drafters && self.opt_drafter,
        }
    }
}

const GB: u64 = 1_000_000_000;

const DRAFT_VOCAB: &str = "j2ipper/draft-vocab";

pub const MODELS: &[ModelInstall] = &[
    ModelInstall {
        id: "qwen3.8-27b",
        files: &[
            FileSpec {
                repo: "unsloth/Qwen3.8-27B-GGUF",
                path: "Qwen3.8-27B-UD-IQ4_XS.gguf",
                size: 14_252_845_984,
                sha256: "40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199",
                role: Role::Weights,
            },
            FileSpec {
                repo: "unsloth/Qwen3.8-27B-GGUF",
                path: "MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
                size: 1_369_590_656,
                sha256: "50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e",
                role: Role::Drafter,
            },
            FileSpec {
                repo: "z-lab/Qwen3.8-27B-DFlash2-GGUF",
                path: "Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
                size: 1_143_006_816,
                sha256: "1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd",
                role: Role::Drafter,
            },
            FileSpec {
                repo: DRAFT_VOCAB,
                path: "qwen-dv-65536.bin",
                size: 262_144,
                sha256: "a1a7c29a4262fa8658c91f2855c99d2feef443dd01f04a56aca6d8b484167457",
                role: Role::DraftVocab,
            },
            FileSpec {
                repo: "j2ipper/Qwen3.8-27B-W4A4-prompt",
                path: "Qwen3.8-27B-UD-IQ4_XS.w4i",
                size: 2_142_248_960,
                sha256: "b4f8d947e0a2d48382d331c9a4eb3e72032801e80c48cb2c1172bfe15bb4a105",
                role: Role::W4i,
            },
            FileSpec {
                repo: "j2ipper/Qwen3.8-27B-W4A4-prompt",
                path: "Qwen3.8-27B-UD-IQ4_XS.fast.w4i",
                size: 8_568_983_552,
                sha256: "d293452b9031fa58d3fe6e70fd8f6534575bfe548d0abfc2208346d6f9887cc3",
                role: Role::W4iFast,
            },
        ],
        prep: Prep::None,
        prep_bytes: (0, 0),
        desktop_fit: DesktopFit::Yes,
        needs_big_gtt: false,
        min_start_gib: None,
        release_free_gib: 66,
        chat_tps: 46.7,
        prompt_tps: 443.0,
        context: "64K",
    },
    ModelInstall {
        id: "qwen3.8-flash-next",
        files: &[
            FileSpec {
                repo: "unsloth/Qwen3.8-Flash-Next-GGUF",
                path: "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
                size: 10_946_624,
                sha256: "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4",
                role: Role::Weights,
            },
            FileSpec {
                repo: "unsloth/Qwen3.8-Flash-Next-GGUF",
                path: "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf",
                size: 49_835_229_856,
                sha256: "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7",
                role: Role::Weights,
            },
            FileSpec {
                repo: "unsloth/Qwen3.8-Flash-Next-GGUF",
                path: "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf",
                size: 43_836_407_744,
                sha256: "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833",
                role: Role::Weights,
            },
        ],
        prep: Prep::FlashNextPack,
        // the pack (~1.5 GB); with the drafters also the MTP tensors, their Q2_0 GGUF and the runtime files
        prep_bytes: (2 * GB, 9 * GB),
        desktop_fit: DesktopFit::Yes,
        needs_big_gtt: false,
        min_start_gib: None,
        release_free_gib: 51,
        chat_tps: 58.9,
        prompt_tps: 970.0,
        context: "256K",
    },
    ModelInstall {
        id: "glm-5.3-flash",
        files: &[
            FileSpec {
                repo: "j2ipper/GLM-5.3-Flash-a4e3s",
                path: "GLM-5.3-Flash-a4e3s-00001-of-00004.gguf",
                size: 9_430_361,
                sha256: "55ee18988a0f016b6a4d7ee4d30daf137931542eae8d9ecedc782a80800aee2d",
                role: Role::Weights,
            },
            FileSpec {
                repo: "j2ipper/GLM-5.3-Flash-a4e3s",
                path: "GLM-5.3-Flash-a4e3s-00002-of-00004.gguf",
                size: 47_618_687_904,
                sha256: "8501ae69de47a410964bd32a9b1e0d1cad440691868cdbaaae77b84d573fc92b",
                role: Role::Weights,
            },
            FileSpec {
                repo: "j2ipper/GLM-5.3-Flash-a4e3s",
                path: "GLM-5.3-Flash-a4e3s-00003-of-00004.gguf",
                size: 48_374_531_744,
                sha256: "5eb5a47445f3dcca7a9a8ed6e756815e77e4e46330ef4d1bacf596a77198a744",
                role: Role::Weights,
            },
            FileSpec {
                repo: "j2ipper/GLM-5.3-Flash-a4e3s",
                path: "GLM-5.3-Flash-a4e3s-00004-of-00004.gguf",
                size: 21_527_344_480,
                sha256: "dc52cfc611ee12928e054934702df5062ed2f705c91666a164e87b1ed35fb1cc",
                role: Role::Weights,
            },
            FileSpec {
                repo: DRAFT_VOCAB,
                path: "glm-dv-65536.bin",
                size: 262_144,
                sha256: "55d63bbd101508ba4a4ec6ded0736c151dd538b52345420bd33260e45699c4df",
                role: Role::DraftVocab,
            },
        ],
        prep: Prep::None,
        prep_bytes: (0, 0),
        desktop_fit: DesktopFit::HeadlessOnly,
        needs_big_gtt: true,
        min_start_gib: Some(112),
        release_free_gib: 9,
        chat_tps: 32.4,
        prompt_tps: 452.0,
        context: "64K",
    },
    ModelInstall {
        id: "mimo-v2.6-flash",
        files: &[
            FileSpec {
                repo: "j2ipper/MiMo-V2.6-Flash-MOPD-a4-x",
                path: "MiMo-V2.6-Flash-a4-x-00001-of-00003.gguf",
                size: 5_949_240,
                sha256: "b31e4530e9d0735bcd98b1f6f647fbec406594e0e4ab77134589d8fb9afd0f49",
                role: Role::Weights,
            },
            FileSpec {
                repo: "j2ipper/MiMo-V2.6-Flash-MOPD-a4-x",
                path: "MiMo-V2.6-Flash-a4-x-00002-of-00003.gguf",
                size: 48_728_061_760,
                sha256: "8c02693229bde8c7266ac2d06727ca55387024f50c0bb1eb002d044d886d31f1",
                role: Role::Weights,
            },
            FileSpec {
                repo: "j2ipper/MiMo-V2.6-Flash-MOPD-a4-x",
                path: "MiMo-V2.6-Flash-a4-x-00003-of-00003.gguf",
                size: 46_625_828_096,
                sha256: "9b0243a7901eff27006a8dcb49487d4d5984034e7332048bcbb5a48e0ad4fbe9",
                role: Role::Weights,
            },
            FileSpec {
                repo: "ggml-org/MiMo-V2.6-Flash-MOPD-GGUF",
                path: "dflash-MiMo-V2.6-Flash-MOPD-Q8_0.gguf",
                size: 2_893_932_768,
                sha256: "cfd70fef40bc4ef2f6c186516bdfa8c861ebd7b8e5f44ed4703a41dde1492a67",
                role: Role::Drafter,
            },
            FileSpec {
                repo: DRAFT_VOCAB,
                path: "mimo-dv-65536.bin",
                size: 262_144,
                sha256: "9eead88fb62cc1a95be16abcd33b90e61840f188df850be6f777731a018d34a3",
                role: Role::DraftVocab,
            },
        ],
        prep: Prep::None,
        prep_bytes: (0, 0),
        desktop_fit: DesktopFit::Tight,
        needs_big_gtt: true,
        min_start_gib: None,
        release_free_gib: 14,
        chat_tps: 35.8,
        prompt_tps: 597.0,
        context: "192K",
    },
    ModelInstall {
        id: "laguna-s-2.1",
        files: &[
            FileSpec {
                repo: "j2ipper/Laguna-S-2.1-Q4-lean",
                path: "Laguna-S-2.1-Q4-lean-00001-of-00003.gguf",
                size: 3_683_770,
                sha256: "777fe1c031ceb7e424420e86c72c9e2b964054c3bd3ad1285ca96d4fc1dac0e7",
                role: Role::Weights,
            },
            FileSpec {
                repo: "j2ipper/Laguna-S-2.1-Q4-lean",
                path: "Laguna-S-2.1-Q4-lean-00002-of-00003.gguf",
                size: 49_125_350_144,
                sha256: "e2f86845e65cd9c184a4069bcd282513aa4cc5f6dac2a1236f78ceeae15bbde2",
                role: Role::Weights,
            },
            FileSpec {
                repo: "j2ipper/Laguna-S-2.1-Q4-lean",
                path: "Laguna-S-2.1-Q4-lean-00003-of-00003.gguf",
                size: 23_076_635_232,
                sha256: "1856e320077070f3fda08035872aa8d88bbd18c92a98bbc718b078d5d5c243b4",
                role: Role::Weights,
            },
            FileSpec {
                // the DFlash drafter of the release runs (byte-identical); opt-in: its license is unclear
                repo: "wimmmm/poolside-Laguna-S-2.1-DFlash-GGUF",
                path: "laguna-s-2.1-dflash-Q8_0.gguf",
                size: 1_188_531_360,
                sha256: "d4e698ad542a66a961a7194f858193708db32546c5786de2bc5ed6addabc9075",
                role: Role::DrafterOptIn,
            },
        ],
        prep: Prep::None,
        prep_bytes: (0, 0),
        desktop_fit: DesktopFit::Yes,
        needs_big_gtt: false,
        min_start_gib: None,
        release_free_gib: 38,
        chat_tps: 34.6,
        prompt_tps: 965.0,
        context: "256K",
    },
];

/// What setup says about an opt-in drafter before offering it.
pub const OPT_DRAFTER_NOTE: &str = "Laguna-S-2.1's DFlash drafter (laguna-s-2.1-dflash-Q8_0.gguf, 1.2 GB) comes from \
wimmmm/poolside-Laguna-S-2.1-DFlash-GGUF, a community conversion of poolside/Laguna-S-2.1-DFlash. Its license is \
unclear: check the repository before you use it. Without it Laguna drafts with prompt lookup.";

pub fn get(id: &str) -> Option<&'static ModelInstall> {
    MODELS.iter().find(|m| m.id == id)
}

/// Accepts a model id, a 1-based index into `MODELS`, or a short name ("27b", "glm", "mimo", "laguna", "next").
pub fn resolve(token: &str) -> Option<&'static ModelInstall> {
    let t = token.trim().to_ascii_lowercase();
    if t.is_empty() {
        return None;
    }
    if let Ok(i) = t.parse::<usize>() {
        return MODELS.get(i.checked_sub(1)?);
    }
    if let Some(m) = get(&t) {
        return Some(m);
    }
    let short = |m: &ModelInstall| -> &'static [&'static str] {
        match m.id {
            "qwen3.8-27b" => &["27b", "qwen27b", "q27"],
            "qwen3.8-flash-next" => &["next", "flash-next", "flashnext", "fn"],
            "glm-5.3-flash" => &["glm"],
            "mimo-v2.6-flash" => &["mimo"],
            "laguna-s-2.1" => &["laguna"],
            _ => &[],
        }
    };
    MODELS.iter().find(|m| short(m).contains(&t.as_str()))
}

/// Parse "1,3", "glm mimo" or "all".
pub fn parse_list(s: &str) -> Result<Vec<&'static ModelInstall>, String> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("all") {
        return Ok(MODELS.iter().collect());
    }
    let mut out: Vec<&'static ModelInstall> = Vec::new();
    for tok in s.split(|c: char| c == ',' || c.is_whitespace()) {
        if tok.is_empty() {
            continue;
        }
        let m = resolve(tok).ok_or_else(|| format!("unknown model: {tok}"))?;
        if !out.iter().any(|o| o.id == m.id) {
            out.push(m);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_manifest_model_is_in_the_catalog_and_back() {
        for m in MODELS {
            assert_eq!(m.spec().id, m.id);
            assert_eq!(m.main_file().role, Role::Weights);
            for f in m.files {
                assert_eq!(f.sha256.len(), 64, "{}", f.path);
                assert!(f
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
                assert!(f.size > 0);
            }
        }
        for s in talus_presets::MODELS {
            assert!(
                get(s.id).is_some(),
                "catalog model {} has no download manifest",
                s.id
            );
        }
    }

    #[test]
    fn shards_are_complete_and_in_order() {
        for m in MODELS {
            let w: Vec<_> = m.weights().map(|f| f.local_name()).collect();
            if w.len() > 1 {
                for (i, name) in w.iter().enumerate() {
                    let tag = format!("-{:05}-of-{:05}.gguf", i + 1, w.len());
                    assert!(name.ends_with(&tag), "{name} should end with {tag}");
                }
            }
        }
    }

    #[test]
    fn choices_pick_files() {
        let q = get("qwen3.8-27b").unwrap();
        let d = Choices::default();
        let names: Vec<_> = q
            .selected_files(&d)
            .iter()
            .map(|f| f.local_name())
            .collect();
        assert!(names.contains(&"mtp-Qwen3.8-27B-Q4_0.gguf"));
        assert!(!names.iter().any(|n| n.ends_with(".w4i")));
        let no_draft = Choices {
            drafters: false,
            ..d
        };
        assert_eq!(q.selected_files(&no_draft).len(), 1);
        let w4 = Choices { w4i: true, ..d };
        assert!(q
            .selected_files(&w4)
            .iter()
            .any(|f| f.local_name() == "Qwen3.8-27B-UD-IQ4_XS.w4i"));
        // about 17 GB with drafters, as the docs say
        let gb = q.download_bytes(&d) as f64 / 1e9;
        assert!((16.0..18.0).contains(&gb), "{gb}");
    }

    #[test]
    fn laguna_dflash_is_opt_in() {
        let l = get("laguna-s-2.1").unwrap();
        let d = Choices::default();
        assert!(!l
            .selected_files(&d)
            .iter()
            .any(|f| f.role == Role::DrafterOptIn));
        let on = Choices {
            opt_drafter: true,
            ..d
        };
        assert!(l
            .selected_files(&on)
            .iter()
            .any(|f| f.local_name() == "laguna-s-2.1-dflash-Q8_0.gguf"));
        let off = Choices {
            opt_drafter: true,
            drafters: false,
            ..d
        };
        assert_eq!(
            l.selected_files(&off).len(),
            3,
            "the opt-in needs the drafters on"
        );
        assert_eq!(rocm_version(), "10.2.0a20261014");
        assert_eq!(
            parse_rocm_version("# c\n\n  10.1.0 \n# x\n"),
            Some("10.1.0")
        );
        assert_eq!(parse_rocm_version("# only comments\n"), None);
    }

    #[test]
    fn resolve_names() {
        assert_eq!(resolve("1").unwrap().id, "qwen3.8-27b");
        assert_eq!(resolve("glm").unwrap().id, "glm-5.3-flash");
        assert_eq!(resolve("MiMo-V2.6-Flash").unwrap().id, "mimo-v2.6-flash");
        assert!(resolve("6").is_none());
        assert!(resolve("0").is_none());
        let l = parse_list("1, laguna 1").unwrap();
        assert_eq!(l.len(), 2);
        assert_eq!(parse_list("all").unwrap().len(), 5);
        assert!(parse_list("llama").is_err());
    }
}
