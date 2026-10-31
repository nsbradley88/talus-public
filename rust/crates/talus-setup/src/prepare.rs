//! Qwen3.8-Flash-Next's preparation after the download (docs/MODELS.md, Qwen3.8-Flash-Next):
//!
//! 1. a Strata pack from the GGUF: `tools/iq_pack.py --gguf <shard 1> --out <pack> --compat-bf16` (it also writes
//!    the tokenizer);
//! 2. with the drafters, the MTP draft layer: `tools/mtp_fetch.py fetch` reads only the MTP tensors of the original
//!    checkpoint (pinned revision, each tensor checked against SHA-256), `tools/mtp_pack.py --experts q2_0` and
//!    `tools/mtp_rt.py` turn them into the runtime files, and `data/draft_vocab.bin` goes next to them.
//!
//! The tools run in a private Python environment (numpy, regex, plus gguf-py's own dependencies pyyaml, tqdm and
//! requests) with llama.cpp's gguf-py at the commit Strata pins.

use crate::config::{flash_next_mtp, flash_next_pack};
use crate::exec::Exec;
use crate::manifest::{Choices, ModelInstall};
use crate::paths::Paths;
use crate::state::SetupState;
use crate::ui;
use std::path::{Path, PathBuf};

/// llama.cpp commit whose gguf-py the MTP tools are written against (Strata's pin).
pub const LLAMA_CPP_COMMIT: &str = "3cf03257f219afbe7334045ff7c6a06ac68c627d";

/// tools/_paths.py, for a source tree that lacks it: the tools import it to find gguf-py.
const PATHS_SHIM: &str = r#""""Shim written by `talus setup`: where the tools find llama.cpp's gguf-py ($STRATA_GGUF_PY)."""
import os
import sys


def gguf_py():
    p = os.environ.get("STRATA_GGUF_PY")
    if not p or not os.path.isdir(os.path.join(p, "gguf")):
        sys.exit("STRATA_GGUF_PY does not point at llama.cpp's gguf-py")
    return p


def add_gguf_py():
    p = gguf_py()
    if p not in sys.path:
        sys.path.insert(0, p)
"#;

pub fn is_prepared(dir: &Path, c: &Choices) -> bool {
    let pack = flash_next_pack(dir);
    pack.join("native_experts.txt").exists()
        && pack.join("tokenizer/vocab.json").exists()
        && (!c.drafters || flash_next_mtp(dir).join("rt/experts.bin").exists())
}

fn s(p: &Path) -> String {
    p.display().to_string()
}

/// The private Python environment and gguf-py. Returns (python, gguf-py dir, shim dir).
fn python_env(x: &Exec, paths: &Paths) -> Result<(PathBuf, PathBuf, PathBuf), String> {
    let env = paths.pyenv();
    let py = env.join("bin/python");
    if !py.exists() {
        x.run(
            &["python3".into(), "-m".into(), "venv".into(), s(&env)],
            "a private Python for the pack tools",
        )?;
    }
    let ok = std::process::Command::new(&py)
        .args(["-c", "import numpy, regex, yaml, tqdm, requests"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        x.run(
            &[
                s(&py),
                "-m".into(),
                "pip".into(),
                "install".into(),
                "-q".into(),
                "numpy".into(),
                "regex".into(),
                "pyyaml".into(),
                "tqdm".into(),
                "requests".into(),
            ],
            "numpy, regex and gguf-py's dependencies for the pack tools",
        )?;
    }
    let llama = env.join(format!("llama.cpp-{}", &LLAMA_CPP_COMMIT[..7]));
    let gguf_py = llama.join("gguf-py");
    if !gguf_py.join("gguf").is_dir() {
        let zip = env.join(format!("llama.cpp-{}.zip", &LLAMA_CPP_COMMIT[..7]));
        x.run(
            &[
                "curl".into(),
                "-fL".into(),
                "-sS".into(),
                "--retry".into(),
                "5".into(),
                "-o".into(),
                s(&zip),
                format!("https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.zip"),
            ],
            "llama.cpp's gguf-py at Strata's pinned commit",
        )?;
        // unpack only gguf-py, with Python's zipfile (no unzip needed)
        let script = String::from(
            "import zipfile,sys,os\nz=zipfile.ZipFile(sys.argv[1])\nfor n in z.namelist():\n    parts=n.split('/',1)\n    \
             if len(parts)==2 and parts[1].startswith('gguf-py/') and not n.endswith('/'):\n        \
             dst=os.path.join(sys.argv[2],parts[1])\n        os.makedirs(os.path.dirname(dst),exist_ok=True)\n        \
             open(dst,'wb').write(z.read(n))\n"
        );
        x.run(
            &[s(&py), "-I".into(), "-c".into(), script, s(&zip), s(&llama)],
            "unpack gguf-py",
        )?;
        x.remove_file(&zip).map_err(|e| e.to_string())?;
    }
    let shim = env.join("shim");
    x.write(
        &shim.join("_paths.py"),
        PATHS_SHIM,
        0o644,
        "lets the tools find gguf-py",
    )
    .map_err(|e| e.to_string())?;
    Ok((py, gguf_py, shim))
}

pub fn prepare_flash_next(
    x: &Exec,
    paths: &Paths,
    st: &SetupState,
    m: &ModelInstall,
    c: &Choices,
) -> Result<(), String> {
    let dir = st.model_dir(m.id);
    if is_prepared(&dir, c) {
        ui::say("Flash-Next pack and draft layer: already prepared");
        return Ok(());
    }
    let (py, gguf_py, shim) = python_env(x, paths)?;
    let tools = st.src.join("tools");
    // tools/_paths.py first if the tree has it, our shim otherwise
    let pythonpath = format!("{}:{}", s(&tools), s(&shim));
    let env = format!("STRATA_GGUF_PY={}", s(&gguf_py));
    let run_tool = |args: Vec<String>, why: &str| -> Result<(), String> {
        let mut a = vec![
            "env".to_string(),
            env.clone(),
            format!("PYTHONPATH={pythonpath}"),
            s(&py),
        ];
        a.extend(args);
        x.run(&a, why)
    };
    let pack = flash_next_pack(&dir);
    let shard1 = dir.join(m.main_file().local_name());
    if !pack.join("native_experts.txt").exists() || !pack.join("tokenizer/vocab.json").exists() {
        ui::say("Building the Strata pack from the GGUF (about a minute) ...");
        run_tool(
            vec![
                s(&tools.join("iq_pack.py")),
                "--gguf".into(),
                s(&shard1),
                "--out".into(),
                s(&pack),
                "--compat-bf16".into(),
            ],
            "the Flash-Next pack",
        )?;
    }
    if c.drafters {
        let mtp = flash_next_mtp(&dir);
        let rt = mtp.join("rt");
        if !rt.join("experts.bin").exists() {
            ui::say("The MTP draft layer comes from the original Qwen checkpoint: only its MTP tensors (a few GB) are read.");
            run_tool(
                vec![
                    s(&tools.join("mtp_fetch.py")),
                    "fetch".into(),
                    "--out".into(),
                    s(&mtp),
                ],
                "the MTP tensors",
            )?;
            run_tool(
                vec![
                    s(&tools.join("mtp_pack.py")),
                    "--src".into(),
                    s(&mtp),
                    "--experts".into(),
                    "q2_0".into(),
                    "--out".into(),
                    s(&mtp.join("mtp-q2_0.gguf")),
                ],
                "the MTP draft GGUF",
            )?;
            run_tool(
                vec![
                    s(&tools.join("mtp_rt.py")),
                    "--gguf".into(),
                    s(&mtp.join("mtp-q2_0.gguf")),
                    "--out".into(),
                    s(&rt),
                ],
                "the MTP runtime files",
            )?;
        }
        let dv = st.src.join("data/draft_vocab.bin");
        x.run(
            &["cp".into(), s(&dv), s(&rt.join("draft_vocab.bin"))],
            "the draft layer's token subset",
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_run_lists_the_tool_chain() {
        let home = crate::testutil::tempdir("prep");
        let paths = Paths::under(&home);
        let st = SetupState::new(&paths, home.join("src"), None);
        let m = crate::manifest::get("qwen3.8-flash-next").unwrap();
        let x = Exec::new(true);
        prepare_flash_next(&x, &paths, &st, m, &Choices::default()).unwrap();
        let a = x.actions().join("\n");
        for needle in [
            "iq_pack.py",
            "--compat-bf16",
            "mtp_fetch.py fetch",
            "--experts q2_0",
            "mtp_rt.py",
            "draft_vocab.bin",
            LLAMA_CPP_COMMIT,
        ] {
            assert!(a.contains(needle), "missing {needle} in:\n{a}");
        }
        assert!(!home.join(".local").exists(), "a dry run writes nothing");
        std::fs::remove_dir_all(home).ok();
    }
}
