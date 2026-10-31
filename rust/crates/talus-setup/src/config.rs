//! Run configs for talus-server, one per model, from the shipped arguments in docs/MODELS.md.
//!
//! These are "our tuned defaults": the TUI shows them as the defaults and its tuning edits start from them, so they
//! carry the shipped arguments and nothing else (an opt-in such as the 27B's 4-bit prompt math is turned on in the
//! TUI, not here). Files are absolute paths into the install.

use crate::manifest::{Choices, ModelInstall, Role};
use crate::state::SetupState;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Everything a run config points at.
#[derive(Clone, Debug)]
pub struct Ctx {
    /// Where talus-cli and libtalus_kernels.so are.
    pub bin_dir: PathBuf,
    /// The ROCm SDK root.
    pub rocm: PathBuf,
    /// The Talus source (data files: expert profile, draft vocabularies, hipBLASLt tables).
    pub src: PathBuf,
    /// `<models_dir>/<model-id>`.
    pub model_dir: PathBuf,
    pub log: PathBuf,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
}

impl Ctx {
    pub fn new(
        st: &SetupState,
        paths: &crate::paths::Paths,
        model_id: &str,
        api_key: Option<String>,
    ) -> Ctx {
        Ctx {
            bin_dir: paths.bin(),
            rocm: st
                .rocm_sdk
                .clone()
                .unwrap_or_else(|| PathBuf::from("/opt/rocm")),
            src: st.src.clone(),
            model_dir: st.model_dir(model_id),
            log: paths.engine_log(model_id),
            host: st.host.clone(),
            port: st.port,
            api_key,
        }
    }

    pub fn lib_dirs(&self) -> Vec<String> {
        [
            self.bin_dir.clone(),
            self.rocm.join("lib"),
            self.rocm.join("lib/rocm_sysdeps/lib"),
            self.rocm.join("lib/llvm/lib"),
        ]
        .iter()
        .map(|p| p.display().to_string())
        .collect()
    }
}

/// Flash-Next's prepared files, inside its model directory.
pub fn flash_next_pack(model_dir: &Path) -> PathBuf {
    model_dir.join("pack")
}
pub fn flash_next_mtp(model_dir: &Path) -> PathBuf {
    model_dir.join("mtp")
}

fn p(path: PathBuf) -> String {
    path.display().to_string()
}

/// The run config for `m` with these choices (drafters on or off, opt-in drafters).
pub fn run_config(m: &ModelInstall, ch: &Choices, c: &Ctx) -> Value {
    let drafters = ch.drafters;
    let f = |role: Role, i: usize| -> String {
        let spec = m
            .files
            .iter()
            .filter(|f| f.role == role)
            .nth(i)
            .expect("manifest file");
        p(c.model_dir.join(spec.local_name()))
    };
    let main = p(c.model_dir.join(m.main_file().local_name()));
    let mut args: Vec<String> = Vec::new();
    let mut env = serde_json::Map::new();
    let mut a = |xs: &[&str]| args.extend(xs.iter().map(|s| s.to_string()));
    let (stops, sampling, tokenizer): (Vec<&str>, Option<Value>, String) = match m.id {
        "qwen3.8-27b" => {
            a(&["--model", &main]);
            if drafters {
                a(&["--mtp-gguf", &f(Role::Drafter, 0), "--mtp", "5"]);
                a(&[
                    "--spec-draft-vocab",
                    &f(Role::DraftVocab, 0),
                    "--ngram",
                    "24",
                ]);
                a(&["--dflash-gguf", &f(Role::Drafter, 1)]);
            }
            (
                vec!["<|im_end|>"],
                Some(json!({"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0})),
                main.clone(),
            )
        }
        "qwen3.8-flash-next" => {
            let pack = p(flash_next_pack(&c.model_dir));
            a(&["--pack", &pack, "--native", &main]);
            a(&[
                "--expert-profile",
                &p(c.src.join("data/expert-profile.bin")),
            ]);
            a(&[
                "--expert-cache",
                "24576",
                "--resident-budget-gib",
                "55",
                "--prefill",
                "auto:16384",
            ]);
            if drafters {
                a(&["--spec", "5", "--spec-min-p", "0.5"]);
                a(&[
                    "--mtp",
                    &p(flash_next_mtp(&c.model_dir).join("rt")),
                    "--mtp-window",
                    "8192",
                ]);
                a(&[
                    "--mtp-draft-vocab",
                    &p(c.src.join("data/draft_vocab_en.bin")),
                ]);
            } else {
                a(&["--suffix-draft", "0"]);
            }
            a(&["--max-context", "262144", "--kv", "int8"]);
            env.insert(
                "STRATA_HIPBLASLT_TUNING".into(),
                json!(p(c.src.join("tools/hip/gfx1151-hipblaslt-100401.txt"))),
            );
            for k in [
                "STRATA_PF_FUSED",
                "STRATA_PF_GEMM",
                "STRATA_HC_UPMIX",
                "STRATA_PA_FAST",
                "STRATA_HIP_WMMA",
                "STRATA_SELECT_WMMA",
                "STRATA_HC_Q8",
            ] {
                env.insert(k.into(), json!("1"));
            }
            env.insert("STRATA_PF_SWITCH_MIN_T".into(), json!("0"));
            env.insert("STRATA_PREFILL_STREAM_MIN".into(), json!("128"));
            (vec!["<|im_end|>"], None, format!("{pack}/tokenizer"))
        }
        "glm-5.3-flash" => {
            a(&["--model", &main, "--ctx", "65536"]);
            if drafters {
                a(&["--draft-vocab", &f(Role::DraftVocab, 0)]);
            }
            (
                vec!["<|user|>", "<|observation|>", "<|endoftext|>"],
                Some(json!({"temperature": 0.8, "top_p": 0.95, "top_k": 40, "min_p": 0.05})),
                main.clone(),
            )
        }
        "mimo-v2.6-flash" => {
            a(&[
                "--model",
                &main,
                "--max-context",
                "196608",
                "--ubatch",
                "2048",
            ]);
            if drafters {
                a(&["--dflash-gguf", &f(Role::Drafter, 0), "--spec", "4"]);
                a(&["--spec-draft-vocab", &f(Role::DraftVocab, 0)]);
            } else {
                a(&["--no-mtp"]);
            }
            (
                vec!["<|im_end|>", "<|endoftext|>"],
                Some(json!({"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0})),
                main.clone(),
            )
        }
        "laguna-s-2.1" => {
            a(&[
                "--model",
                &main,
                "--max-context",
                "262144",
                "--ubatch",
                "2048",
            ]);
            a(&[
                "--eos-ids",
                "2,24",
                "--spec",
                "7",
                "--sparse",
                "1024,16384,128,1024,16",
            ]);
            // the DFlash drafter is an opt-in download (license unclear); prompt lookup drafts either way
            if drafters && ch.opt_drafter {
                a(&["--dflash-gguf", &f(Role::DrafterOptIn, 0)]);
            }
            (
                vec!["\u{3008}|EOS|\u{3009}", "</assistant>"],
                Some(json!({"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0})),
                main.clone(),
            )
        }
        other => panic!("no run config recipe for {other}"),
    };
    let mut v = serde_json::Map::new();
    v.insert("exe".into(), json!(p(c.bin_dir.join("talus-cli"))));
    v.insert("args".into(), json!(args));
    v.insert("env".into(), Value::Object(env));
    v.insert("cwd".into(), json!(p(c.bin_dir.clone())));
    v.insert("tokenizer".into(), json!(tokenizer));
    v.insert("model_name".into(), json!(m.id));
    v.insert("stop_tokens".into(), json!(stops));
    v.insert("log".into(), json!(p(c.log.clone())));
    v.insert("lib_dirs".into(), json!(c.lib_dirs()));
    v.insert("host".into(), json!(c.host));
    v.insert("port".into(), json!(c.port));
    if let Some(k) = &c.api_key {
        v.insert("api_key".into(), json!(k));
    }
    if let Some(s) = sampling {
        v.insert("sampling".into(), s);
    }
    Value::Object(v)
}

/// The run config `talus setup` writes for installed model `id` with the current setup (None: not a known model).
pub fn expected_text(paths: &crate::paths::Paths, st: &SetupState, id: &str) -> Option<String> {
    let m = crate::manifest::get(id)?;
    let c = st.models.get(id).copied().unwrap_or_default();
    let lan = st.host != "127.0.0.1" && st.host != "localhost";
    let key = if lan {
        std::fs::read_to_string(paths.api_key())
            .ok()
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty())
    } else {
        None
    };
    let ctx = Ctx::new(st, paths, id, key);
    Some(to_text(&run_config(m, &c, &ctx)))
}

pub fn to_text(v: &Value) -> String {
    serde_json::to_string_pretty(v).expect("json") + "\n"
}

/// Files a run config names that must exist before the server can start it: the executable, the model files, the
/// drafters, the tokenizer. Returns (what, path).
pub fn referenced_files(v: &Value) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    if let Some(e) = v.get("exe").and_then(Value::as_str) {
        out.push(("engine".to_string(), PathBuf::from(e)));
    }
    if let Some(args) = v.get("args").and_then(Value::as_array) {
        let a: Vec<&str> = args.iter().filter_map(Value::as_str).collect();
        for w in a.windows(2) {
            let what = match w[0] {
                "--model" | "--native" => "model",
                "--pack" => "pack",
                "--mtp-gguf" | "--dflash-gguf" | "--mtp" => "drafter",
                "--spec-draft-vocab" | "--draft-vocab" | "--mtp-draft-vocab" => "draft vocabulary",
                "--expert-profile" => "expert profile",
                _ => continue,
            };
            // `--mtp 5` is a count for the 27B, a directory for Flash-Next
            if w[1].starts_with('/') {
                out.push((what.to_string(), PathBuf::from(w[1])));
            }
        }
    }
    if let Some(t) = v.get("tokenizer").and_then(Value::as_str) {
        out.push(("tokenizer".to_string(), PathBuf::from(t)));
    }
    out
}

/// What is wrong with a run config on disk (empty = fine).
pub fn validate(text: &str, expect_model: &str) -> Vec<String> {
    let v: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => return vec![format!("not valid JSON: {e}")],
    };
    let mut bad = Vec::new();
    if !v.get("args").map(Value::is_array).unwrap_or(false) {
        bad.push("no \"args\" list".to_string());
    }
    match v.get("model_name").and_then(Value::as_str) {
        Some(n) if n == expect_model => {}
        Some(n) => bad.push(format!("model_name is {n}, expected {expect_model}")),
        None => bad.push("no model_name".to_string()),
    }
    for (what, path) in referenced_files(&v) {
        if !path.exists() {
            bad.push(format!("{what} missing: {}", path.display()));
        }
    }
    bad
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest;

    const ON: Choices = Choices {
        drafters: true,
        w4i: false,
        w4i_fast: false,
        opt_drafter: false,
    };
    const OFF: Choices = Choices {
        drafters: false,
        ..ON
    };
    const ALL: Choices = Choices {
        opt_drafter: true,
        ..ON
    };

    fn ctx() -> Ctx {
        Ctx {
            bin_dir: "/t/bin".into(),
            rocm: "/t/rocm".into(),
            src: "/t/src".into(),
            model_dir: "/m/x".into(),
            log: "/t/log".into(),
            host: "127.0.0.1".into(),
            port: 8000,
            api_key: None,
        }
    }

    fn args(v: &Value) -> Vec<String> {
        v["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn shipped_arguments_per_model() {
        let c = ctx();
        let q = run_config(manifest::get("qwen3.8-27b").unwrap(), &ON, &c);
        assert_eq!(
            args(&q).join(" "),
            "--model /m/x/Qwen3.8-27B-UD-IQ4_XS.gguf \
             --mtp-gguf /m/x/mtp-Qwen3.8-27B-Q4_0.gguf --mtp 5 \
             --spec-draft-vocab /m/x/qwen-dv-65536.bin --ngram 24 \
             --dflash-gguf /m/x/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
        );
        assert_eq!(q["env"], json!({}), "no opt-ins in the tuned defaults");
        assert_eq!(q["exe"], json!("/t/bin/talus-cli"));
        assert_eq!(q["tokenizer"], json!("/m/x/Qwen3.8-27B-UD-IQ4_XS.gguf"));
        assert_eq!(
            q["lib_dirs"],
            json!([
                "/t/bin",
                "/t/rocm/lib",
                "/t/rocm/lib/rocm_sysdeps/lib",
                "/t/rocm/lib/llvm/lib"
            ])
        );
        let q0 = run_config(manifest::get("qwen3.8-27b").unwrap(), &OFF, &c);
        assert_eq!(
            args(&q0).join(" "),
            "--model /m/x/Qwen3.8-27B-UD-IQ4_XS.gguf"
        );

        let g = run_config(manifest::get("glm-5.3-flash").unwrap(), &ON, &c);
        assert_eq!(
            args(&g).join(" "),
            "--model /m/x/GLM-5.3-Flash-a4e3s-00001-of-00004.gguf --ctx 65536 --draft-vocab /m/x/glm-dv-65536.bin"
        );
        let mi = run_config(manifest::get("mimo-v2.6-flash").unwrap(), &ON, &c);
        assert_eq!(
            args(&mi).join(" "),
            "--model /m/x/MiMo-V2.6-Flash-a4-x-00001-of-00003.gguf --max-context 196608 --ubatch 2048 \
             --dflash-gguf /m/x/dflash-MiMo-V2.6-Flash-MOPD-Q8_0.gguf --spec 4 --spec-draft-vocab /m/x/mimo-dv-65536.bin"
        );
        let la = run_config(manifest::get("laguna-s-2.1").unwrap(), &ALL, &c);
        assert_eq!(
            args(&la).join(" "),
            "--model /m/x/Laguna-S-2.1-Q4-lean-00001-of-00003.gguf --max-context 262144 --ubatch 2048 \
             --eos-ids 2,24 --spec 7 --sparse 1024,16384,128,1024,16 --dflash-gguf /m/x/laguna-s-2.1-dflash-Q8_0.gguf"
        );
        assert_eq!(la["stop_tokens"], json!(["〈|EOS|〉", "</assistant>"]));
        // default: no DFlash drafter (opt-in), prompt lookup only
        let la0 = run_config(manifest::get("laguna-s-2.1").unwrap(), &ON, &c);
        assert!(!args(&la0).iter().any(|a| a == "--dflash-gguf"));
        let fx = run_config(manifest::get("qwen3.8-flash-next").unwrap(), &ON, &c);
        let fa = args(&fx).join(" ");
        assert!(fa.starts_with(
            "--pack /m/x/pack --native /m/x/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
        ));
        assert!(fa.contains(
            "--mtp /m/x/mtp/rt --mtp-window 8192 --mtp-draft-vocab /t/src/data/draft_vocab_en.bin"
        ));
        assert!(fa.ends_with("--max-context 262144 --kv int8"));
        assert_eq!(fx["env"]["STRATA_PF_FUSED"], json!("1"));
        assert_eq!(fx["tokenizer"], json!("/m/x/pack/tokenizer"));
    }

    #[test]
    fn every_model_has_a_recipe_and_the_tui_defaults_are_no_ops() {
        // Picking every option's default in the TUI must not change the config we install.
        for m in manifest::MODELS {
            let v = run_config(m, &ON, &ctx());
            let mut a = args(&v);
            let mut env: Vec<(String, String)> = v["env"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                .collect();
            let before = (a.clone(), env.clone());
            let spec = m.spec();
            let edits = spec.edits(&talus_presets::Selection::new()).unwrap();
            talus_presets::apply_edits(&mut a, &mut env, &edits);
            assert_eq!((a, env), before, "{}", m.id);
            assert!(validate(&to_text(&v), m.id)
                .iter()
                .all(|e| e.contains("missing")));
        }
    }

    #[test]
    fn api_key_only_when_given() {
        let mut c = ctx();
        let m = manifest::get("laguna-s-2.1").unwrap();
        assert!(run_config(m, &ON, &c).get("api_key").is_none());
        c.api_key = Some("k".repeat(32));
        assert_eq!(run_config(m, &ON, &c)["api_key"], json!("k".repeat(32)));
    }

    #[test]
    fn validation() {
        assert!(validate("{", "x")[0].starts_with("not valid JSON"));
        let e = validate(r#"{"args":[],"model_name":"y"}"#, "x");
        assert_eq!(e, vec!["model_name is y, expected x".to_string()]);
        let e = validate(
            r#"{"args":["--model","/nope/m.gguf","--mtp","5"],"model_name":"x"}"#,
            "x",
        );
        assert_eq!(e, vec!["model missing: /nope/m.gguf".to_string()]);
    }
}
