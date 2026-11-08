//! Container runtime for swebench_verified: image preparation with an LRU disk cache, the agent container
//! (no network, model bridge only, snapshot + verification + environment scan before the agent, strace audit)
//! and the evaluation container (fresh, from the same image, no network, the official apply chain and eval
//! script). `DockerRuntime` drives the docker CLI; tests use a local runtime with the same scripts.
//!
//! Image preparation, per drawn instance, while no model is involved: pull the official
//! `swebench/sweb.eval.x86_64.*` image, then build a derived image `talus-bench/swe:<id>` that adds strace (for
//! the audit) and a wheelhouse of the package's build requirements (`pip download` into /opt/tb-wheels, with
//! `PIP_NO_INDEX=1 PIP_FIND_LINKS=/opt/tb-wheels`), so the official eval script's `pip install -e .` works
//! offline. Then the gold patch is evaluated once offline in a throwaway container (it must resolve; otherwise the
//! instance's environment is broken and the instance is reported as an environment error for every model).
//! These steps need the network and happen before and apart from any agent run.

use super::audit::STRACE_ARGS;
use super::bridge::FORWARDER_PY;
use super::dataset::Instance;
use super::harness::{
    prepare_eval_script, APPLY_PATCH_FAIL, APPLY_PATCH_PASS, CONTAINER_PATCH_FILE, GIT_APPLY_CMDS,
};
use super::sitepkg::{target_of, SCAN_PY};
use super::snapshot::{SNAPSHOT_SH, VERIFY_SH};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default, Serialize)]
pub struct PrepInfo {
    pub image: String,
    pub pulled: bool,
    pub built: bool,
    pub prep_s: f64,
    /// Gold patch resolves offline in this image (None: not checked).
    pub gold_ok: Option<bool>,
    pub evicted: Vec<String>,
}

/// One evaluation: the test log, and what the tests did on the network (`egress::to_value`, or Null).
#[derive(Clone, Debug, Default)]
pub struct EvalRun {
    pub log: String,
    pub network: Value,
}

#[derive(Clone, Debug)]
pub struct AgentSpec {
    /// The model id the bridge serves (`talus/<model>` for opencode).
    pub model: String,
    pub bridge_socket_dir: PathBuf,
    pub bridge_port: u16,
    pub work_dir: PathBuf,
    pub opencode: PathBuf,
    pub timeout: Duration,
    pub prompt: String,
    /// The served context window and the per-reply output budget (opencode compacts the session before it fills).
    pub context_tokens: u64,
    pub output_tokens: u64,
    /// The mini-SWE-agent harness instead of opencode: the pinned, self-contained bundle (built on the build host,
    /// `<bundle>/python/cpython-*/bin/python3.*` + `<bundle>/site`), bind-mounted read-only at /agent/mini.
    pub mini_bundle: Option<PathBuf>,
    /// The raw problem statement (the mini harness renders it into its own instance template).
    pub problem: String,
}

/// The mini-SWE-agent runner, run as the agent user inside the container: upstream `DefaultAgent` +
/// `LocalEnvironment` + `LitellmModel` with the upstream `config/benchmarks/swebench.yaml` (templates, step limit,
/// command timeout), only re-pointed: the model at the bridge, cost tracking off (a local model has no price), the
/// wall limit from talus-bench's cap. Writes `traj.json` and `result.json` (exit status, submission) to $HOME/mini.
pub const MINI_RUN_PY: &str = r#"import json, os, pathlib, sys
os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import yaml
import minisweagent
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel
import litellm, requests  # the harness's own dependencies, loaded before the environment variables go

# LocalEnvironment runs every command with os.environ: PYTHONPATH=/agent/mini/site would put the bundle's packages
# (requests 2.34, urllib3, ...) ahead of the testbed's in the model's python, and the TB_* variables are ours. The
# harness keeps its sys.path; the commands get the image's environment.
tb = {k: os.environ.pop(k, None) for k in ("TB_MODEL", "TB_PORT", "TB_WALL", "TB_CWD")}
for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE"):
    os.environ.pop(k, None)

out = pathlib.Path(os.environ["HOME"]) / "mini"
cfg = yaml.safe_load((pathlib.Path(minisweagent.__file__).parent / "config/benchmarks/swebench.yaml").read_text())
e = dict(cfg["environment"])
env_vars = dict(e.get("env", {}))
env_vars.pop("BASH_ENV", None)  # /root/.bashrc: the agent user already has the testbed env first on PATH
env = LocalEnvironment(cwd=tb["TB_CWD"] or e.get("cwd", "/testbed"), timeout=int(e.get("timeout", 60)), env=env_vars)
m = dict(cfg["model"])
m["model_name"] = "openai/" + tb["TB_MODEL"]
m["model_kwargs"] = {**m.get("model_kwargs", {}), "api_base": "http://127.0.0.1:%s/v1" % tb["TB_PORT"], "api_key": "set-by-the-bridge"}
m["cost_tracking"] = "ignore_errors"
model = LitellmModel(**m)
a = dict(cfg["agent"])
a["output_path"] = out / "traj.json"
a["wall_time_limit_seconds"] = int(tb["TB_WALL"])
agent = DefaultAgent(model, env, **a)
task = (out / "problem.txt").read_text()
try:
    info = agent.run(task)
except Exception as ex:
    info = {"exit_status": type(ex).__name__, "submission": "", "exception": str(ex)}
(out / "result.json").write_text(json.dumps({"exit_status": info.get("exit_status"), "submission": info.get("submission") or "",
    "n_calls": agent.n_calls, "mini_version": minisweagent.__version__, "exception": info.get("exception", "")}))
"#;

/// The bundle's interpreter, relative to the bundle (`python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`).
pub fn mini_python(bundle: &Path) -> Result<PathBuf, String> {
    let py = bundle.join("python");
    let dir = std::fs::read_dir(&py)
        .map_err(|e| format!("{}: {e}", py.display()))?
        .filter_map(Result::ok)
        // a real directory: uv also leaves a minor-version symlink pointing at an absolute host path, which does
        // not exist inside the container
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir() && !t.is_symlink()))
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("cpython-"))
        })
        .min()
        .ok_or_else(|| format!("{}: no cpython-* interpreter", py.display()))?;
    let bin = dir.join("bin");
    let exe = std::fs::read_dir(&bin)
        .map_err(|e| format!("{}: {e}", bin.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            let n = p
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            n.starts_with("python3.") && !n.ends_with("-config")
        })
        .min()
        .ok_or_else(|| format!("{}: no python3.x", bin.display()))?;
    exe.strip_prefix(bundle)
        .map(Path::to_path_buf)
        .map_err(|e| e.to_string())
}

/// The bash commands of a mini trajectory as opencode-style tool events (one JSON line each), so the event audit
/// applies unchanged.
pub fn mini_events(traj: &str) -> String {
    let v: Value = serde_json::from_str(traj).unwrap_or(Value::Null);
    let mut out = String::new();
    for m in v["messages"].as_array().into_iter().flatten() {
        for a in m["extra"]["actions"].as_array().into_iter().flatten() {
            if let Some(c) = a["command"].as_str() {
                out.push_str(&json!({"part": {"type": "tool", "tool": "bash", "state": {"input": {"command": c}}}}).to_string());
                out.push('\n');
            }
        }
    }
    out
}

#[derive(Clone, Debug, Default)]
pub struct AgentRun {
    /// The snapshot commit (the only commit git may name).
    pub head: String,
    pub patch: String,
    pub events: String,
    pub strace: String,
    /// The strace log on disk instead of in `strace` (streamed by the audit, then gzipped).
    pub strace_file: Option<PathBuf>,
    pub verify: Value,
    pub sitepkg: Value,
    pub exit: String,
    pub wall_s: f64,
}

pub trait Runtime: Send + Sync {
    fn prepare(&self, inst: &Instance) -> Result<PrepInfo, String>;
    fn run_agent(&self, inst: &Instance, spec: &AgentSpec) -> Result<AgentRun, String>;
    /// Fresh container, model patch applied with the official chain, official eval script; returns the log and
    /// the network evidence (Null when the eval ran offline).
    fn run_eval(&self, inst: &Instance, patch: &str) -> Result<EvalRun, String>;
    /// The pre-agent environment scan, run again in a fresh offline container of the instance's image
    /// (`swe-rescore`, when a saved scan predates a scanner change). None: not supported here.
    fn rescan_environment(&self, _inst: &Instance) -> Option<Value> {
        None
    }
    /// The audit policy for this runtime's paths.
    fn policy(
        &self,
        _inst: &Instance,
        module: &str,
        dist: &str,
        bridge_port: u16,
    ) -> super::audit::Policy {
        super::audit::Policy::container(module, dist, bridge_port)
    }
}

/// The opencode configuration inside the agent container: only the bridge provider, web tools denied, no
/// sharing or updates, no paths outside the repo.
pub fn opencode_config(model: &str, port: u16, context: u64, output: u64) -> Value {
    json!({
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": false,
        "share": "disabled",
        // no per-step working-tree snapshots: on django-sized trees they cost GBs of trace and minutes of I/O
        "snapshot": false,
        "model": format!("talus/{model}"),
        "small_model": format!("talus/{model}"),
        "provider": {"talus": {"npm": "@ai-sdk/openai-compatible", "name": "endpoint under test",
            "options": {"baseURL": format!("http://127.0.0.1:{port}/v1"), "apiKey": "set-by-the-bridge"},
            "models": {model: {"name": model, "limit": {"context": context, "output": output}}}}},
        "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny", "websearch": "deny",
                       "external_directory": "deny"},
        "agent": {"build": {"model": format!("talus/{model}")}, "plan": {"model": format!("talus/{model}")}}
    })
}

pub const OPENCODE_ENV: &[(&str, &str)] = &[
    ("OPENCODE_CONFIG", "/agent/config/opencode.json"),
    ("OPENCODE_DISABLE_AUTOUPDATE", "1"),
    ("OPENCODE_DISABLE_MODELS_FETCH", "1"),
    ("OPENCODE_DISABLE_SHARE", "1"),
    ("OPENCODE_DISABLE_LSP_DOWNLOAD", "1"),
    ("OPENCODE_DISABLE_DEFAULT_PLUGINS", "1"),
    ("OPENCODE_DISABLE_EXTERNAL_SKILLS", "1"),
    ("OPENCODE_DISABLE_CLAUDE_CODE", "1"),
    ("OPENCODE_DISABLE_PROJECT_CONFIG", "1"),
    ("HOME", "/agent/home"),
    // the image's pip settings (env -i drops them): offline wheelhouse, no self-update check
    ("PIP_NO_INDEX", "1"),
    ("PIP_FIND_LINKS", "/opt/tb-wheels"),
    ("PIP_DISABLE_PIP_VERSION_CHECK", "1"),
];

/// The agent container's entrypoint (runs as root; the agent itself runs as the unprivileged `agent` user).
pub fn agent_entry(inst: &Instance, spec: &AgentSpec) -> String {
    let (dist, module) = target_of(&inst.repo);
    let env: String = OPENCODE_ENV
        .iter()
        .map(|(k, v)| format!("{k}={v} "))
        .collect();
    let strace = STRACE_ARGS
        .iter()
        .map(|a| format!("'{a}'"))
        .collect::<Vec<_>>()
        .join(" ");
    let path = "PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
    let secs = spec.timeout.as_secs();
    let model = &spec.model;
    let run = match &spec.mini_bundle {
        None => format!(
            r#"timeout {secs} strace {strace} -o $A/trace.log -u agent -- env -i {path} {env} \
  /agent/bin/opencode run --format json -m talus/{model} --dir /testbed "$(cat $T/prompt.txt)" > $A/events.jsonl 2> $A/opencode.err
echo $? > $A/exit
git -C /testbed add -A >/dev/null 2>&1
git -C /testbed diff --cached --binary HEAD > $A/patch.diff"#
        ),
        Some(b) => {
            let py = mini_python(b)
                .map(|p| format!("/agent/mini/{}", p.display()))
                .unwrap_or_else(|_| "/agent/mini/python/MISSING".into());
            // mini's own wall limit ends the loop cleanly (trajectory saved) a little before the hard cap
            let wall = secs.saturating_sub(90).max(60);
            format!(
                r#"M=/agent/home/mini; mkdir -p $M && cp $T/mini_run.py $T/problem.txt $M/ && chown -R agent $M
timeout {secs} strace {strace} -o $A/trace.log -u agent -- env -i {path} HOME=/agent/home PYTHONPATH=/agent/mini/site PYTHONDONTWRITEBYTECODE=1 \
  TB_MODEL={model} TB_PORT={port} TB_WALL={wall} {py} $M/mini_run.py > $A/mini.out 2> $A/mini.err
echo $? > $A/exit
cp $M/traj.json $A/mini_traj.json 2>/dev/null; cp $M/result.json $A/mini_result.json 2>/dev/null
git -C /testbed add -A >/dev/null 2>&1
git -C /testbed diff --cached --binary HEAD > $A/worktree.diff
# the mini harness's patch is its submission (empty when it did not submit), as upstream
python3 -c 'import json,sys; sys.stdout.write(json.load(open("/audit/mini_result.json")).get("submission") or "")' > $A/patch.diff 2>/dev/null || : > $A/patch.diff"#,
                port = spec.bridge_port
            )
        }
    };
    format!(
        r#"#!/bin/bash
set -uo pipefail
T=/run/talus; A=/audit
PY=/opt/miniconda3/envs/testbed/bin/python
bash $T/snapshot.sh /testbed {base} > $A/snapshot.log 2>&1
git -C /testbed rev-parse HEAD > $A/head 2>/dev/null
bash $T/verify.sh /testbed / > $A/verify.json
$PY $T/scan.py {dist} {module} /testbed --roots / --clean --all-envs --quarantine > $A/sitepkg.json 2> $A/sitepkg.err
grep -q '"ok": true' $A/verify.json || {{ echo "verification failed" > $A/exit; exit 0; }}
# a uid no host file belongs to: the bind-mounted audit and bridge directories stay out of the agent's reach
useradd -u 4242 -m -d /agent/home -s /bin/bash agent 2>/dev/null || true
chown -R agent /testbed; chmod 700 /root $A 2>/dev/null; chown -R agent /agent/home
python3 $T/forwarder.py {port} $T/model.sock > $A/forwarder.log 2>&1 &
sleep 1
cd /testbed
{run}
"#,
        base = inst.base_commit,
        port = spec.bridge_port,
        run = run,
    )
}

/// Build-requirement extractor run once at image prep (network allowed then) so offline reinstalls in the eval
/// script find their wheels under `/opt/tb-wheels`.
pub const BUILD_REQS_PY: &str = r#"import re
try:
    t = open("pyproject.toml").read()
    m = re.search(r"\[build-system\][^\[]*?requires\s*=\s*\[((?:[^\[\]]|\[[^\]]*\])*)\]", t, re.S)
    print(" ".join(re.findall(r"[\"']([^\"']+)[\"']", m.group(1))) if m else "")
except Exception:
    print("")
"#;

/// Prep script copied into the derived image (no Dockerfile heredocs: the classic builder rejects them).
pub const PREP_SH: &str = r#"#!/bin/bash
set -ef
. /opt/miniconda3/bin/activate testbed
cd /testbed
mkdir -p /opt/tb-wheels
REQ=$(python /tmp/tb-build-reqs.py)
python -m pip download --dest /opt/tb-wheels setuptools wheel $REQ || true
rm -f /tmp/tb-build-reqs.py /tmp/tb-prep.sh
"#;

/// The derived image's Dockerfile (strace for the audit, an offline wheelhouse for the eval script's reinstall);
/// the build context also holds `tb-prep.sh` and `tb-build-reqs.py`.
pub fn derived_dockerfile(official: &str) -> String {
    format!(
        "FROM {official}\n\
RUN apt-get update && apt-get install -y --no-install-recommends strace && rm -rf /var/lib/apt/lists/*\n\
COPY tb-prep.sh tb-build-reqs.py /tmp/\n\
RUN bash /tmp/tb-prep.sh\n\
ENV PIP_NO_INDEX=1 PIP_FIND_LINKS=/opt/tb-wheels\n\
LABEL talus-bench.swe=1\n"
    )
}

pub fn derived_tag(inst: &Instance) -> String {
    format!(
        "talus-bench/swe:{}",
        inst.instance_id.to_lowercase().replace("__", "-")
    )
}

// ---------------------------------------------------------------------------------------------------------
// LRU image cache

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CacheEntry {
    pub bytes: u64,
    pub last_used: f64,
    pub gold_ok: Option<bool>,
    /// The gold run's network evidence (`egress::NetEvidence`, discover mode): the allowlist for model evals.
    #[serde(default)]
    pub egress: Option<super::egress::NetEvidence>,
}

/// Index of the images talus-bench pulled or built (`<cache>/swebench_images.json`), evicted least recently used
/// first when the total passes the cap. Images it did not pull are never touched.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ImageCache {
    pub cap_bytes: u64,
    pub images: BTreeMap<String, CacheEntry>,
}

impl ImageCache {
    pub fn load(path: &Path, cap_bytes: u64) -> Self {
        let mut c: ImageCache = std::fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        c.cap_bytes = cap_bytes;
        c
    }

    pub fn save(&self, path: &Path) {
        let _ = std::fs::write(path, serde_json::to_string_pretty(self).unwrap_or_default());
    }

    pub fn touch(&mut self, image: &str, bytes: u64, now: f64) {
        let e = self.images.entry(image.to_string()).or_default();
        e.bytes = bytes;
        e.last_used = now;
    }

    pub fn total(&self) -> u64 {
        self.images.values().map(|e| e.bytes).sum()
    }

    /// Images to remove (oldest first) so the total fits the cap; `keep` (in use) is never evicted.
    pub fn evict(&mut self, keep: &[&str]) -> Vec<String> {
        let mut out = Vec::new();
        let mut order: Vec<(f64, String)> = self
            .images
            .iter()
            .map(|(k, v)| (v.last_used, k.clone()))
            .collect();
        order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for (_, img) in order {
            if self.total() <= self.cap_bytes {
                break;
            }
            if keep.contains(&img.as_str()) {
                continue;
            }
            self.images.remove(&img);
            out.push(img);
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------------------
// Docker

pub struct DockerRuntime {
    pub docker: String,
    pub cache_index: PathBuf,
    pub cap_bytes: u64,
    pub eval_timeout: Duration,
    pub validate_gold: bool,
    /// Evaluate with the egress gateway (blackhole default, gold-derived allowlist) instead of `--network none`.
    pub eval_network: bool,
    lock: std::sync::Mutex<()>,
}

fn run(cmd: &mut Command) -> Result<String, String> {
    let o = cmd
        .output()
        .map_err(|e| format!("{:?}: {e}", cmd.get_program()))?;
    if o.status.success() {
        Ok(String::from_utf8_lossy(&o.stdout).to_string())
    } else {
        Err(format!(
            "{:?} failed: {}",
            cmd.get_args().collect::<Vec<_>>(),
            String::from_utf8_lossy(&o.stderr).trim()
        ))
    }
}

impl DockerRuntime {
    pub fn new(cache_dir: &Path, cap_gb: f64, eval_timeout_s: u64, validate_gold: bool) -> Self {
        Self {
            docker: "docker".into(),
            cache_index: cache_dir.join("swebench_images.json"),
            cap_bytes: (cap_gb * 1e9) as u64,
            eval_timeout: Duration::from_secs(eval_timeout_s),
            validate_gold,
            eval_network: true,
            lock: std::sync::Mutex::new(()),
        }
    }

    pub fn with_eval_network(mut self, on: bool) -> Self {
        self.eval_network = on;
        self
    }

    /// The gateway image and its NAT bridge (image prep time: network allowed).
    fn ensure_gateway(&self) -> Result<(), String> {
        use super::egress::{GW_DOCKERFILE, GW_IMAGE, GW_NETWORK};
        if !self.present(GW_IMAGE) {
            let dir = std::env::temp_dir().join(format!("tb-gw-build-{}", std::process::id()));
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            std::fs::write(dir.join("Dockerfile"), GW_DOCKERFILE).map_err(|e| e.to_string())?;
            run(self
                .docker()
                .args(["build", "-q", "-t", GW_IMAGE])
                .arg(&dir))?;
            let _ = std::fs::remove_dir_all(&dir);
        }
        let nets = run(self
            .docker()
            .args(["network", "ls", "--format", "{{.Name}}"]))?;
        if !nets.lines().any(|n| n.trim() == GW_NETWORK) {
            run(self.docker().args(["network", "create", GW_NETWORK]))?;
        }
        Ok(())
    }

    /// Start a gateway in `mode`; returns (gateway name, shared dir, gateway hostname).
    fn start_gateway(
        &self,
        name: &str,
        mode: &super::egress::NetMode,
    ) -> Result<(String, PathBuf, String), String> {
        use super::egress::{NetMode, GW_IMAGE, GW_NETWORK, GW_SH};
        let gw = format!("{name}-gw");
        let dir = std::env::temp_dir().join(&gw);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("gw.sh"), GW_SH).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("resolv.conf"), "nameserver 127.0.0.1\n")
            .map_err(|e| e.to_string())?;
        if let NetMode::Allow(names) = mode {
            std::fs::write(dir.join("allow.txt"), names.join("\n")).map_err(|e| e.to_string())?;
        }
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777));
        let _ = self.docker().args(["rm", "-f", &gw]).output();
        run(self
            .docker()
            .args([
                "run",
                "-d",
                "--name",
                &gw,
                "--cap-add",
                "NET_ADMIN",
                "--network",
                GW_NETWORK,
            ])
            .args([
                "-v",
                &format!("{}:/tbnet", dir.display()),
                GW_IMAGE,
                "sh",
                "/tbnet/gw.sh",
                mode.arg(),
            ]))?;
        let t0 = Instant::now();
        while !dir.join("ready").exists() {
            if dir.join("error").exists() || t0.elapsed() > Duration::from_secs(90) {
                let logs = self
                    .docker()
                    .args(["logs", &gw])
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stderr).to_string())
                    .unwrap_or_default();
                let _ = self.docker().args(["rm", "-f", &gw]).output();
                return Err(format!(
                    "egress gateway did not start: {}",
                    logs.lines().rev().take(3).collect::<Vec<_>>().join(" | ")
                ));
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        let host = run(self
            .docker()
            .args(["inspect", "-f", "{{.Config.Hostname}}", &gw]))?
        .trim()
        .to_string();
        Ok((gw, dir, host))
    }

    /// Collect the gateway's evidence and remove it.
    fn stop_gateway(
        &self,
        gw: &str,
        dir: &Path,
        host: &str,
        mode: &super::egress::NetMode,
    ) -> super::egress::NetEvidence {
        // the last NFLOG / dnsmasq lines (nflog delivers per packet with queue-threshold 1; margin anyway)
        std::thread::sleep(Duration::from_millis(1500));
        let dns = self
            .docker()
            .args(["logs", gw])
            .output()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout).to_string() + &String::from_utf8_lossy(&o.stderr)
            })
            .unwrap_or_default();
        let conn = std::fs::read_to_string(dir.join("conn.log")).unwrap_or_default();
        let pinned = std::fs::read_to_string(dir.join("hosts.pinned")).unwrap_or_default();
        let _ = self.docker().args(["rm", "-f", gw]).output();
        let _ = std::fs::remove_dir_all(dir);
        let mut ev = super::egress::parse(mode.arg(), &dns, &conn, &pinned, host);
        if let super::egress::NetMode::Allow(n) = mode {
            ev.allowlist = n.clone();
        }
        ev
    }

    fn docker(&self) -> Command {
        Command::new(&self.docker)
    }

    fn image_bytes(&self, image: &str) -> u64 {
        run(self
            .docker()
            .args(["image", "inspect", "--format", "{{.Size}}", image]))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
    }

    fn present(&self, image: &str) -> bool {
        self.docker()
            .args(["image", "inspect", image])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// One offline container from the derived image: copy the patch, the official apply chain, the eval script.
    fn eval_in_fresh_container(
        &self,
        inst: &Instance,
        patch: &str,
        tag: &str,
        mode: &super::egress::NetMode,
    ) -> Result<(String, Option<super::egress::NetEvidence>), String> {
        let name = format!(
            "tb-eval-{}-{}",
            inst.instance_id.to_lowercase().replace("__", "-"),
            std::process::id()
        );
        let _ = self.docker().args(["rm", "-f", &name]).output();
        // offline, or inside a gateway's network namespace (no NET_ADMIN here: the rules are the gateway's)
        let gw = match mode {
            super::egress::NetMode::Offline => None,
            m => Some(self.start_gateway(&name, m)?),
        };
        let started = match &gw {
            None => run(self.docker().args([
                "run",
                "-d",
                "--network",
                "none",
                "--name",
                &name,
                tag,
                "sleep",
                "infinity",
            ])),
            Some((g, dir, _)) => run(self
                .docker()
                .args([
                    "run",
                    "-d",
                    "--network",
                    &format!("container:{g}"),
                    "--name",
                    &name,
                ])
                .args([
                    "-v",
                    &format!("{}:/etc/resolv.conf:ro", dir.join("resolv.conf").display()),
                    tag,
                    "sleep",
                    "infinity",
                ])),
        };
        if let Err(e) = started {
            if let Some((g, dir, host)) = &gw {
                let _ = self.stop_gateway(g, dir, host, mode);
            }
            return Err(e);
        }
        let result = (|| -> Result<String, String> {
            let dir = std::env::temp_dir().join(&name);
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            std::fs::write(dir.join("patch.diff"), patch).map_err(|e| e.to_string())?;
            std::fs::write(dir.join("eval.sh"), prepare_eval_script(&inst.eval_script))
                .map_err(|e| e.to_string())?;
            run(self.docker().args([
                "cp",
                &dir.join("patch.diff").display().to_string(),
                &format!("{name}:{CONTAINER_PATCH_FILE}"),
            ]))?;
            run(self.docker().args([
                "cp",
                &dir.join("eval.sh").display().to_string(),
                &format!("{name}:/eval.sh"),
            ]))?;
            let mut log = String::new();
            // an empty patch evaluates the unmodified base (the trial's known-bad control)
            let mut applied = patch.trim().is_empty();
            if applied {
                log.push_str("(no patch: unmodified base)\n");
            }
            for (i, cmd) in GIT_APPLY_CMDS.iter().enumerate() {
                if applied {
                    break;
                }
                if i > 0 {
                    let _ = self
                        .docker()
                        .args([
                            "exec",
                            "-w",
                            "/testbed",
                            &name,
                            "bash",
                            "-c",
                            "git checkout -- . ; git clean -fd",
                        ])
                        .output();
                }
                let o = self
                    .docker()
                    .args([
                        "exec",
                        "-w",
                        "/testbed",
                        &name,
                        "bash",
                        "-c",
                        &format!("{cmd} {CONTAINER_PATCH_FILE}"),
                    ])
                    .output()
                    .map_err(|e| e.to_string())?;
                if o.status.success() {
                    log.push_str(&format!(
                        "{APPLY_PATCH_PASS}:\n{}\n",
                        String::from_utf8_lossy(&o.stdout)
                    ));
                    applied = true;
                    break;
                }
            }
            if !applied {
                let o = self
                    .docker()
                    .args([
                        "exec",
                        "-w",
                        "/testbed",
                        &name,
                        "bash",
                        "-c",
                        &format!("git apply --check --reverse {CONTAINER_PATCH_FILE}"),
                    ])
                    .output()
                    .map_err(|e| e.to_string())?;
                if o.status.success() {
                    log.push_str(&format!("{APPLY_PATCH_PASS}: verified already applied\n"));
                    applied = true;
                }
            }
            if !applied {
                log.push_str(&format!("{APPLY_PATCH_FAIL}\n"));
                return Ok(log);
            }
            let o = Command::new("timeout")
                .arg(self.eval_timeout.as_secs().to_string())
                .arg(&self.docker)
                .args(["exec", &name, "/bin/bash", "/eval.sh"])
                .output()
                .map_err(|e| e.to_string())?;
            log.push_str(&String::from_utf8_lossy(&o.stdout));
            log.push_str(&String::from_utf8_lossy(&o.stderr));
            if o.status.code() == Some(124) {
                log.push_str(&format!("\n{}\n", super::harness::TESTS_TIMEOUT));
            }
            let _ = std::fs::remove_dir_all(&dir);
            Ok(log)
        })();
        let _ = self.docker().args(["rm", "-f", &name]).output();
        let evidence = gw.map(|(g, dir, host)| self.stop_gateway(&g, &dir, &host, mode));
        result.map(|log| (log, evidence))
    }
}

impl Runtime for DockerRuntime {
    fn prepare(&self, inst: &Instance) -> Result<PrepInfo, String> {
        let _g = self.lock.lock().unwrap();
        let t0 = Instant::now();
        let tag = derived_tag(inst);
        let mut cache = ImageCache::load(&self.cache_index, self.cap_bytes);
        let mut info = PrepInfo {
            image: tag.clone(),
            ..Default::default()
        };
        if !self.present(&tag) {
            if !self.present(&inst.image) {
                run(self.docker().args(["pull", &inst.image]))?;
                info.pulled = true;
            }
            let dir = std::env::temp_dir().join(format!("tb-build-{}", std::process::id()));
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            for (name, body) in [
                ("Dockerfile", derived_dockerfile(&inst.image)),
                ("tb-prep.sh", PREP_SH.to_string()),
                ("tb-build-reqs.py", BUILD_REQS_PY.to_string()),
            ] {
                std::fs::write(dir.join(name), body).map_err(|e| e.to_string())?;
            }
            run(self.docker().args(["build", "-q", "-t", &tag]).arg(&dir))?;
            let _ = std::fs::remove_dir_all(&dir);
            info.built = true;
        }
        let now = crate::util::unix_now();
        cache.touch(&tag, self.image_bytes(&tag), now);
        cache.touch(&inst.image, self.image_bytes(&inst.image), now);
        if self.eval_network {
            self.ensure_gateway()?;
        }
        let entry = cache.images.get(&tag).cloned().unwrap_or_default();
        // the gold run decides the instance's environment and, with the eval network, its allowlist
        let known = entry.gold_ok.filter(|_| {
            !self.eval_network
                || entry
                    .egress
                    .as_ref()
                    .is_some_and(|e| e.version == super::egress::EVIDENCE_VERSION)
        });
        info.gold_ok = match known {
            Some(k) => Some(k),
            None if self.validate_gold => {
                let mode = if self.eval_network {
                    super::egress::NetMode::Discover
                } else {
                    super::egress::NetMode::Offline
                };
                let (log, ev) = self.eval_in_fresh_container(inst, &inst.patch, &tag, &mode)?;
                let g = super::harness::grade(
                    &inst.log_parser,
                    &inst.eval_type,
                    &inst.fail_to_pass,
                    &inst.pass_to_pass,
                    &log,
                )?;
                let e = cache.images.get_mut(&tag).unwrap();
                e.gold_ok = Some(g.resolved);
                e.egress = ev;
                Some(g.resolved)
            }
            None => None,
        };
        info.evicted = cache.evict(&[&tag, &inst.image]);
        for img in &info.evicted {
            let _ = self.docker().args(["rmi", img]).output();
        }
        cache.save(&self.cache_index);
        info.prep_s = t0.elapsed().as_secs_f64();
        Ok(info)
    }

    fn run_agent(&self, inst: &Instance, spec: &AgentSpec) -> Result<AgentRun, String> {
        let t0 = Instant::now();
        // docker treats a relative -v source as a named volume: mount absolute paths only
        let w = &std::path::absolute(&spec.work_dir).map_err(|e| e.to_string())?;
        let audit = w.join("audit");
        let cfg = w.join("config");
        for d in [&audit, &cfg] {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        let t = &spec.bridge_socket_dir;
        for (f, body) in [
            ("snapshot.sh", SNAPSHOT_SH.to_string()),
            ("verify.sh", VERIFY_SH.to_string()),
            ("scan.py", SCAN_PY.to_string()),
            ("forwarder.py", FORWARDER_PY.to_string()),
            ("prompt.txt", spec.prompt.clone()),
            ("problem.txt", spec.problem.clone()),
            ("mini_run.py", MINI_RUN_PY.to_string()),
            ("entry.sh", agent_entry(inst, spec)),
        ] {
            std::fs::write(t.join(f), body).map_err(|e| e.to_string())?;
        }
        std::fs::write(
            cfg.join("opencode.json"),
            serde_json::to_string_pretty(&opencode_config(
                &spec.model,
                spec.bridge_port,
                spec.context_tokens,
                spec.output_tokens,
            ))
            .unwrap_or_default(),
        )
        .map_err(|e| e.to_string())?;
        // the config holds no secret (the bridge adds the key); the agent user must read it, and nothing else of ours
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&cfg, std::fs::Permissions::from_mode(0o755));
        let _ = std::fs::set_permissions(
            cfg.join("opencode.json"),
            std::fs::Permissions::from_mode(0o644),
        );
        for d in [&audit, t] {
            let _ = std::fs::set_permissions(d, std::fs::Permissions::from_mode(0o700));
        }
        let name = format!(
            "tb-agent-{}-{}",
            inst.instance_id.to_lowercase().replace("__", "-"),
            std::process::id()
        );
        let _ = self.docker().args(["rm", "-f", &name]).output();
        let o = self
            .docker()
            .args([
                "run",
                "--rm",
                "--network",
                "none",
                // name lookups fail on loopback at once instead of reaching for the host's resolver
                "--dns",
                "127.0.0.1",
                "--name",
                &name,
                "--cap-add",
                "SYS_PTRACE",
            ])
            .args(["-v", &format!("{}:/run/talus", t.display())])
            .args(["-v", &format!("{}:/audit", audit.display())])
            .args(["-v", &format!("{}:/agent/config:ro", cfg.display())])
            .args([
                "-v",
                &format!(
                    "{}:/agent/bin/opencode:ro",
                    std::path::absolute(&spec.opencode)
                        .unwrap_or_else(|_| spec.opencode.clone())
                        .display()
                ),
            ])
            .args(
                spec.mini_bundle
                    .iter()
                    .flat_map(|b| {
                        let b = std::path::absolute(b).unwrap_or_else(|_| b.clone());
                        ["-v".to_string(), format!("{}:/agent/mini:ro", b.display())]
                    })
                    .collect::<Vec<_>>(),
            )
            .arg(derived_tag(inst))
            .args(["bash", "/run/talus/entry.sh"])
            .output()
            .map_err(|e| e.to_string())?;
        let read = |f: &str| std::fs::read_to_string(audit.join(f)).unwrap_or_default();
        Ok(AgentRun {
            head: read("head").trim().to_string(),
            patch: read("patch.diff"),
            events: if spec.mini_bundle.is_some() {
                mini_events(&read("mini_traj.json"))
            } else {
                read("events.jsonl")
            },
            strace: String::new(),
            strace_file: Some(audit.join("trace.log")),
            verify: serde_json::from_str(read("verify.json").trim()).unwrap_or(Value::Null),
            sitepkg: serde_json::from_str(read("sitepkg.json").trim()).unwrap_or(Value::Null),
            exit: {
                let err = String::from_utf8_lossy(&o.stderr);
                let tail: String = err.lines().rev().take(3).collect::<Vec<_>>().join(" | ");
                if o.status.success() || tail.is_empty() {
                    format!("{} docker {}", read("exit").trim(), o.status)
                } else {
                    format!("{} docker {}: {tail}", read("exit").trim(), o.status)
                }
            },
            wall_s: t0.elapsed().as_secs_f64(),
        })
    }

    fn rescan_environment(&self, inst: &Instance) -> Option<Value> {
        let (dist, module) = target_of(&inst.repo);
        let dir = std::env::temp_dir().join(format!("tb-rescan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok()?;
        std::fs::write(dir.join("scan.py"), SCAN_PY).ok()?;
        let o = self
            .docker()
            .args(["run", "--rm", "--network", "none", "-v"])
            .arg(format!("{}:/w:ro", dir.display()))
            .arg(derived_tag(inst))
            .args([
                "/opt/miniconda3/envs/testbed/bin/python",
                "/w/scan.py",
                dist,
                module,
                "/testbed",
                "--roots",
                "/",
                "--clean",
                "--all-envs",
            ])
            .output()
            .ok()?;
        let _ = std::fs::remove_dir_all(&dir);
        serde_json::from_slice(&o.stdout).ok()
    }

    fn run_eval(&self, inst: &Instance, patch: &str) -> Result<EvalRun, String> {
        let tag = derived_tag(inst);
        if !self.eval_network {
            let (log, _) =
                self.eval_in_fresh_container(inst, patch, &tag, &super::egress::NetMode::Offline)?;
            return Ok(EvalRun {
                log,
                network: Value::Null,
            });
        }
        let gold = ImageCache::load(&self.cache_index, self.cap_bytes)
            .images
            .get(&tag)
            .and_then(|e| e.egress.clone())
            .ok_or_else(|| format!("{tag}: no gold network evidence (prepare the image first)"))?;
        let mode = super::egress::NetMode::Allow(gold.allowlist_from_gold());
        let (log, ev) = self.eval_in_fresh_container(inst, patch, &tag, &mode)?;
        let ev = ev.unwrap_or_default();
        Ok(EvalRun {
            log,
            network: super::egress::to_value(&ev, &gold),
        })
    }
}

impl DockerRuntime {
    /// Evaluate a patch with an explicit network mode (`swe-netcheck`).
    pub fn eval_with(
        &self,
        inst: &Instance,
        patch: &str,
        mode: &super::egress::NetMode,
    ) -> Result<(String, Option<super::egress::NetEvidence>), String> {
        self.eval_in_fresh_container(inst, patch, &derived_tag(inst), mode)
    }

    /// The cached gold evidence of an instance (after `prepare`).
    pub fn gold_evidence(&self, inst: &Instance) -> Option<super::egress::NetEvidence> {
        ImageCache::load(&self.cache_index, self.cap_bytes)
            .images
            .get(&derived_tag(inst))
            .and_then(|e| e.egress.clone())
    }

    /// Disk used by the images talus-bench manages (bytes, from its index).
    pub fn cache_bytes(&self) -> u64 {
        ImageCache::load(&self.cache_index, self.cap_bytes).total()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_runner_clears_the_harness_environment_before_the_commands_run() {
        // LocalEnvironment hands os.environ to every model command: PYTHONPATH (the bundle's packages) must be gone
        // before the environment is created, after the harness's own imports
        let pop = MINI_RUN_PY.find("os.environ.pop(k, None)").unwrap();
        assert!(MINI_RUN_PY.find("import litellm, requests").unwrap() < pop);
        assert!(pop < MINI_RUN_PY.find("env = LocalEnvironment(").unwrap());
        assert!(MINI_RUN_PY.contains("\"PYTHONPATH\""));
        assert!(!MINI_RUN_PY.contains("os.environ[\"TB_"));
    }

    #[test]
    fn lru_eviction() {
        let mut c = ImageCache {
            cap_bytes: 10,
            images: BTreeMap::new(),
        };
        c.touch("a", 4, 1.0);
        c.touch("b", 4, 2.0);
        c.touch("c", 4, 3.0);
        assert_eq!(
            c.evict(&["a"]),
            vec!["b"],
            "a is in use; b is the oldest of the rest"
        );
        assert!(c.total() <= 10);
        c.touch("d", 9, 4.0);
        let ev = c.evict(&["d"]);
        assert_eq!(ev, vec!["a", "c"]);
        assert_eq!(c.total(), 9);
    }

    #[test]
    fn mini_harness_contract() {
        let inst = super::super::dataset::Instance::from_row(
            &super::super::dataset::tests::synthetic(&[("django/django", 1)])[0],
        )
        .unwrap();
        // a fake bundle layout
        let b = std::env::temp_dir().join(format!("tb-mini-bundle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&b);
        std::fs::create_dir_all(b.join("python/cpython-3.12.13-linux-x86_64-gnu/bin")).unwrap();
        // uv's minor-version symlink (absolute host target) must not be chosen
        std::os::unix::fs::symlink(
            b.join("python/cpython-3.12.13-linux-x86_64-gnu"),
            b.join("python/cpython-3.12-linux-x86_64-gnu"),
        )
        .ok();
        for f in ["python3.12", "python3.12-config", "python3"] {
            std::fs::write(
                b.join("python/cpython-3.12.13-linux-x86_64-gnu/bin")
                    .join(f),
                "",
            )
            .unwrap();
        }
        assert_eq!(
            mini_python(&b).unwrap(),
            Path::new("python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12")
        );
        let spec = AgentSpec {
            model: "qwen3.8-27b".into(),
            bridge_socket_dir: "/x".into(),
            bridge_port: 18555,
            work_dir: "/w".into(),
            opencode: "/usr/local/bin/opencode".into(),
            timeout: Duration::from_secs(2700),
            prompt: "Fix it".into(),
            context_tokens: 65536,
            output_tokens: 16384,
            mini_bundle: Some(b.clone()),
            problem: inst.problem_statement.clone(),
        };
        let e = agent_entry(&inst, &spec);
        // the same container contract as opencode: snapshot + verification first, unprivileged, traced, clean env
        let i_ver = e.find("verify.sh").unwrap();
        let i_agent = e.find("mini_run.py > $A/mini.out").unwrap();
        assert!(
            i_ver < i_agent
                && e.contains("-u agent")
                && e.contains("env -i")
                && !e.contains("opencode run")
        );
        assert!(e.contains("/agent/mini/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"));
        assert!(
            e.contains("TB_WALL=2610"),
            "mini's own wall limit ends the loop 90 s before the hard cap"
        );
        assert!(e.contains("timeout 2700 strace"));
        // the patch is the submission; the worktree diff is kept beside it
        assert!(e.contains("worktree.diff") && e.contains(r#"get("submission")"#));
        assert!(
            MINI_RUN_PY.contains("config/benchmarks/swebench.yaml")
                && MINI_RUN_PY.contains("ignore_errors")
        );
        // trajectory commands become audit events
        let traj = r#"{"messages": [{"role": "assistant", "extra": {"actions": [{"command": "ls"}, {"command": "git log --all"}]}},
            {"role": "user", "content": "x"}]}"#;
        let ev = mini_events(traj);
        assert_eq!(ev.lines().count(), 2);
        let mut a = super::super::audit::Audit::default();
        super::super::audit::audit_events(
            &ev,
            &super::super::audit::Policy::container("django", "Django", 18555),
            &mut a,
        );
        assert_eq!(a.tool_calls, 2);
        assert!(a.notes.iter().any(|n| n.kind == "git_probe"));
        let _ = std::fs::remove_dir_all(&b);
    }

    #[test]
    fn agent_container_contract() {
        let inst = super::super::dataset::Instance::from_row(
            &super::super::dataset::tests::synthetic(&[("django/django", 1)])[0],
        )
        .unwrap();
        let spec = AgentSpec {
            model: "qwen3.8-27b".into(),
            bridge_socket_dir: "/x".into(),
            bridge_port: 18555,
            work_dir: "/w".into(),
            opencode: "/usr/local/bin/opencode".into(),
            timeout: Duration::from_secs(2700),
            prompt: "Fix it".into(),
            context_tokens: 65536,
            output_tokens: 16384,
            mini_bundle: None,
            problem: inst.problem_statement.clone(),
        };
        let e = agent_entry(&inst, &spec);
        // snapshot and verification precede the agent; the agent runs unprivileged, traced, with a clean env
        let i_snap = e.find("snapshot.sh").unwrap();
        let i_ver = e.find("verify.sh").unwrap();
        let i_agent = e.find("opencode run").unwrap();
        assert!(i_snap < i_ver && i_ver < i_agent);
        assert!(e.contains("-u agent") && e.contains("env -i") && e.contains("chmod 700 /root"));
        assert!(
            e.contains("useradd -u 4242"),
            "the agent uid must differ from the host user owning the bind mounts"
        );
        assert!(e.find("rev-parse HEAD").unwrap() < i_agent);
        assert!(
            e.contains("OPENCODE_DISABLE_AUTOUPDATE=1")
                && e.contains("OPENCODE_DISABLE_MODELS_FETCH=1")
        );
        assert!(
            e.contains("--clean"),
            "target archives are removed before the agent starts"
        );
        // nothing hidden reaches the agent container
        for secret in [
            &inst.test_patch,
            &inst.patch,
            &inst.fail_to_pass[0],
            &inst.pass_to_pass[0],
        ] {
            assert!(!e.contains(secret.as_str()), "agent entry leaks {secret}");
        }
        let cfg = opencode_config("m", 18555, 65536, 16384);
        assert_eq!(cfg["permission"]["webfetch"], "deny");
        assert_eq!(cfg["permission"]["websearch"], "deny");
        assert_eq!(cfg["permission"]["external_directory"], "deny");
        assert_eq!(
            cfg["provider"]["talus"]["models"]["m"]["limit"]["context"],
            65536
        );
        assert_eq!(
            cfg["provider"]["talus"]["options"]["baseURL"],
            "http://127.0.0.1:18555/v1"
        );
        let df = derived_dockerfile("swebench/sweb.eval.x86_64.x:latest");
        assert!(df.contains("PIP_NO_INDEX=1") && df.contains("strace"));
        assert!(!df.contains("<<"), "no heredocs in the Dockerfile");
    }
}

/// The real mini-SWE-agent loop (the pinned bundle) against the mock model: an edit, a patch file, an explicit
/// submit. Opt-in (`TALUS_BENCH_MINI_BUNDLE=<bundle dir>`): CI machines without the bundle skip it.
#[cfg(test)]
mod mini_tests {
    use super::*;

    #[test]
    fn mini_loop_submits_against_the_mock_model() {
        let Ok(bundle) = std::env::var("TALUS_BENCH_MINI_BUNDLE") else {
            eprintln!("TALUS_BENCH_MINI_BUNDLE not set: skipped");
            return;
        };
        let bundle = PathBuf::from(bundle);
        let py = bundle.join(mini_python(&bundle).unwrap());
        let (mock, _) = crate::mock::spawn_openai(0, crate::mock::Degrade::NONE).unwrap();
        let root = std::env::temp_dir().join(format!("tb-mini-loop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let repo = root.join("repo");
        let home = root.join("home");
        std::fs::create_dir_all(home.join("mini")).unwrap();
        std::fs::create_dir_all(&repo).unwrap();
        let sh = |c: &str| {
            assert!(
                Command::new("bash")
                    .arg("-c")
                    .arg(c)
                    .current_dir(&repo)
                    .env("GIT_CONFIG_GLOBAL", "/dev/null")
                    .status()
                    .unwrap()
                    .success(),
                "{c}"
            );
        };
        sh("git init -q && echo 'x = 1' > m.py && git add . && git -c user.name=t -c user.email=t@x commit -qm base");
        std::fs::write(home.join("mini/mini_run.py"), MINI_RUN_PY).unwrap();
        std::fs::write(home.join("mini/problem.txt"), "m.py needs a fix comment").unwrap();
        let o = Command::new(&py)
            .arg(home.join("mini/mini_run.py"))
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", &home)
            .env("PYTHONPATH", bundle.join("site"))
            .env("TB_MODEL", "mock-model")
            .env("TB_PORT", mock.port().to_string())
            .env("TB_WALL", "120")
            .env("TB_CWD", &repo)
            .output()
            .unwrap();
        let res: Value = serde_json::from_str(
            &std::fs::read_to_string(home.join("mini/result.json"))
                .unwrap_or_else(|_| panic!("{}", String::from_utf8_lossy(&o.stderr))),
        )
        .unwrap();
        assert_eq!(
            res["exit_status"],
            "Submitted",
            "{res} {}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(
            res["submission"]
                .as_str()
                .unwrap()
                .contains("+# tb-mock fix"),
            "{res}"
        );
        assert_eq!(res["mini_version"], "2.4.6");
        let ev = mini_events(&std::fs::read_to_string(home.join("mini/traj.json")).unwrap());
        assert_eq!(ev.lines().count(), 3, "{ev}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
