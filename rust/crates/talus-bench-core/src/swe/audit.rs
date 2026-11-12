//! Layer 5, the audit trail. Three logs per instance, all written outside the agent's reach:
//! - `strace -f` of the whole agent process tree (`STRACE_ARGS`): every exec, file access, chdir, process
//!   creation and connect, with per-process working directories tracked so relative paths resolve;
//! - opencode's own event stream (`--format json`): every tool call with its input;
//! - the model bridge's request log (only the allowlisted model API passes).
//!
//! Attribution: opencode and the helpers it runs itself (its repo `git` queries) are the harness; a shell it
//! starts is the model's bash tool, and that shell's whole subtree is audited under the full policy. Harness
//! filesystem probing and harness git queries are recorded (`harness_commands`), not flagged; opencode's own
//! tools (read/grep/glob/edit) are audited from its event stream.
//!
//! Flags are leak vectors: any one marks the instance INVALID (not passed; its patch is still evaluated and
//! reported as `resolved_ignoring_flags`). Notes (`NOTE_KINDS`: `outside_repo`, `git_probe`, history probes
//! inside the verified single-commit snapshot) are informational and reported separately. Kinds:
//! `outside_repo` (a tool process opens a file outside the repo and the system allowlist; stat/access, chdir
//! and directory listings are exploration, not flagged), `swebench_read` (any path, argument or
//! URL naming SWE-bench), `target_copy_read` (code of the target library outside the repo: site-packages,
//! caches, archives, and copies vendored inside another package such as pip/_vendor/requests: pip itself needs its
//! copy, so only pip's own processes may open it, see [`is_pip_exec`]), `network` (a connect to anything but the model bridge, from a tool process),
//! `network_tool` (curl/wget/pip install/npm/ssh/... executed), `audit_missing` (the trace is unreadable),
//! `git_history` (reaching objects beyond the snapshot: fetch/pull/clone/ls-remote, URLs, or naming a commit
//! other than the snapshot's; probes such as --all, reflog or fsck are the `git_probe` note) (formerly: git commands that name other
//! commits or reach for history: hashes other than HEAD, --all, reflog, fetch, remote, clone, fsck, bundle,
//! cat-file), `web_tool` (opencode webfetch/websearch, denied by its config), `bridge_denied` (a request
//! through the bridge to anything but the model API). Network attempts by the opencode process itself (it is
//! the harness, e.g. an update check) are recorded as `harness_network`, informational: the network namespace
//! blocks them and the agent does not control them.

use regex::Regex;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

const SWEBENCH_NAMES: [&str; 4] = ["swe-bench", "swebench", "swe_bench", "princeton-nlp"];

fn mentions_swebench(text: &str) -> bool {
    let low = text.to_lowercase();
    SWEBENCH_NAMES.iter().any(|s| low.contains(s))
}

/// A SWE-bench name inside something that can be fetched or opened: a path or URL (has '/') or a data file
/// name. A bare mention ("# simulate the SWE-bench harness" in the agent's own script) is the model recognising
/// the benchmark (measured by the memorization layer), not access to its data.
fn names_swebench_resource(text: &str) -> bool {
    let data_ext = [
        ".json", ".jsonl", ".parquet", ".csv", ".arrow", ".gz", ".zip", ".tar", ".db", ".sqlite",
        ".pkl",
    ];
    text.split(|c: char| c.is_whitespace() || "\"'`(),;=[]{}<>|&".contains(c))
        .any(|t| {
            let low = t.to_lowercase();
            SWEBENCH_NAMES.iter().any(|s| low.contains(s))
                && (low.contains('/') || data_ext.iter().any(|e| low.contains(e)))
        })
}

/// strace options (the caller adds `-o FILE -u agent -- CMD...`).
pub const STRACE_ARGS: &[&str] = &[
    "-f",
    "-qq",
    "-s",
    "4096",
    "-e",
    "trace=execve,open,openat,openat2,creat,stat,lstat,newfstatat,statx,access,faccessat,faccessat2,readlink,readlinkat,chdir,fchdir,connect,clone,clone3,fork,vfork",
];

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Flag {
    pub kind: String,
    pub detail: String,
    /// strace | events | bridge
    pub source: String,
}

/// Informational kinds: recorded in `notes`, never INVALID. `outside_repo` (a tool opened a file outside the
/// repo and the allowlist: no leak vector by itself; the leak paths have their own kinds) and `git_probe`
/// (history-probing git commands inside the verified single-commit snapshot).
pub const NOTE_KINDS: [&str; 3] = ["outside_repo", "git_probe", "network_private"];

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Audit {
    /// Leak vectors: any one makes the instance INVALID.
    pub flags: Vec<Flag>,
    /// Informational findings (`NOTE_KINDS`), reported separately.
    pub notes: Vec<Flag>,
    pub harness_network: Vec<String>,
    /// Commands the harness (opencode itself, not a tool call) ran, e.g. its own `git` repo queries
    /// (informational, capped at 50).
    pub harness_commands: Vec<String>,
    /// Package-manager commands of tool processes that cannot fetch anything here (apt/conda/npm, `pip install`
    /// of non-target packages): offline by layer 1, recorded (capped at 50), not flagged.
    pub package_commands: Vec<String>,
    pub execs: usize,
    pub file_accesses: usize,
    pub tool_calls: usize,
}

impl Audit {
    pub fn invalid(&self) -> bool {
        !self.flags.is_empty()
    }

    /// A rule result: flags, except the informational `package` kind.
    fn apply(&mut self, kind: &str, detail: String, source: &str) {
        if kind == "package" {
            if self.package_commands.len() < 50 && !self.package_commands.contains(&detail) {
                self.package_commands.push(detail);
            }
        } else {
            self.flag(kind, detail, source);
        }
    }

    /// Record a flag. Capped per kind (50), so noise of one kind can never hide another kind.
    fn flag(&mut self, kind: &str, detail: String, source: &str) {
        let list = if NOTE_KINDS.contains(&kind) {
            &mut self.notes
        } else {
            &mut self.flags
        };
        let same_kind = list.iter().filter(|f| f.kind == kind).count();
        if same_kind < 50 && !list.iter().any(|f| f.kind == kind && f.detail == detail) {
            list.push(Flag {
                kind: kind.into(),
                detail,
                source: source.into(),
            });
        }
    }
}

#[derive(Clone, Debug)]
pub struct Policy {
    pub repo: String,
    /// Path prefixes the agent may touch besides the repo (system, interpreter, its own home).
    pub allowed: Vec<String>,
    /// Paths inside allowed prefixes that are still forbidden (target package copies, caches).
    pub forbidden: Vec<String>,
    pub bridge_port: u16,
    /// The snapshot commit (git may name it).
    pub head: String,
    /// The harness executable (its own network attempts are informational).
    pub harness_exe: String,
    pub target_module: String,
    pub target_dist: String,
}

impl Policy {
    /// The container policy for one instance.
    pub fn container(target_module: &str, target_dist: &str, bridge_port: u16) -> Policy {
        Policy {
            repo: "/testbed".into(),
            allowed: [
                "/tmp",
                "/var/tmp",
                "/dev",
                "/proc",
                "/sys",
                "/usr",
                "/lib",
                "/lib32",
                "/lib64",
                "/bin",
                "/sbin",
                "/etc",
                // system caches and state test suites touch (fontconfig, texmf, ...); the name rules still apply
                "/var/cache",
                "/var/lib",
                "/var/run",
                "/run",
                "/opt/miniconda3",
                "/agent",
                "/run/talus",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            forbidden: vec!["/opt/miniconda3/pkgs".into()],
            bridge_port,
            head: String::new(),
            harness_exe: "/agent/bin/opencode".into(),
            target_module: target_module.into(),
            target_dist: target_dist.into(),
        }
    }

    fn under(p: &str, base: &str) -> bool {
        p == base || p.starts_with(&format!("{}/", base.trim_end_matches('/')))
    }

    /// Classify one accessed path (absolute).
    pub fn check_path(&self, p: &str) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        let low = p.to_lowercase();
        // the pinned mini-SWE-agent bundle (read-only) ships its own prompt config
        // (minisweagent/config/benchmarks/swebench.yaml): harness code, not benchmark data
        if ["swe-bench", "swebench", "swe_bench", "princeton-nlp"]
            .iter()
            .any(|s| low.contains(s))
            && !Self::under(p, "/agent/mini")
        {
            out.push(("swebench_read", p.to_string()));
        }
        if Self::under(p, &self.repo) {
            return out;
        }
        let dist_us = self.target_dist.replace('-', "_");
        // a released copy of the target: an installed package dir in site-packages, a conda package dir
        // `pkgs/<dist>-<version>...`, or a `<dist>-<version>` archive / metadata dir in a cache (not pytest's
        // own `/tmp/pytest-of-<user>/pytest-0/.../.pytest_cache/` when the target is pytest)
        let dist_l = self.target_dist.to_lowercase();
        let dist_us_l = dist_us.to_lowercase();
        let in_cache = low.contains("/.cache/") || low.contains("/cache/");
        let segs: Vec<&str> = low.split('/').collect();
        let versioned = |seg: &str| {
            [&dist_l, &dist_us_l].iter().any(|d| {
                seg.strip_prefix(&format!("{d}-"))
                    .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
            })
        };
        let archive = |seg: &str| {
            [
                ".whl",
                ".tar.gz",
                ".tgz",
                ".zip",
                ".tar.bz2",
                ".egg",
                ".conda",
                ".dist-info",
                ".egg-info",
            ]
            .iter()
            .any(|e| seg.ends_with(e))
        };
        let is_copy = !self.target_module.is_empty()
            && (p.contains(&format!("site-packages/{}/", self.target_module))
                || p.ends_with(&format!("site-packages/{}", self.target_module))
                || segs.windows(2).any(|w| w[0] == "pkgs" && versioned(w[1]))
                || (in_cache && segs.iter().any(|g| versioned(g) && archive(g))));
        if is_copy {
            out.push(("target_copy_read", p.to_string()));
        }
        // the mini-SWE-agent bundle's own copy of the target: a harness dependency (requests for litellm, at a
        // later release than any requests task), so the harness's imports are fine but a model command opening it
        // is a leak; the strace audit judges it for tool processes only (`bundle_copy_read` -> target_copy_read)
        let bundle_seg = p
            .strip_prefix("/agent/mini/site/")
            .and_then(|r| r.split('/').next())
            .map(str::to_lowercase);
        if !self.target_module.is_empty()
            && bundle_seg.as_deref().is_some_and(|seg| {
                seg == self.target_module.to_lowercase()
                    || (versioned(seg)
                        && (seg.ends_with(".dist-info") || seg.ends_with(".egg-info")))
            })
        {
            out.push(("bundle_copy_read", p.to_string()));
        }
        // a copy vendored inside another package (pip/_vendor/requests in every SWE image, at a later release than
        // any requests task), or pip's bundled wheel, which holds the same copy; the strace audit lets pip's own
        // processes open them (`vendored_kind`). Case-sensitive like the site-packages rule: paths, not names.
        let real_segs: Vec<&str> = p.split('/').collect();
        let vendored = !self.target_module.is_empty()
            && real_segs.windows(2).any(|w| {
                ["_vendor", "vendor", "vendored"].contains(&w[0]) && w[1] == self.target_module
            });
        let pip_wheel = self.target_module == "requests"
            && real_segs.windows(3).any(|w| {
                w[0] == "ensurepip"
                    && w[1] == "_bundled"
                    && w[2].starts_with("pip-")
                    && w[2].ends_with(".whl")
            });
        if vendored || pip_wheel {
            out.push(("vendored_copy_read", p.to_string()));
        }
        // ancestor directories of the repo and of allowed prefixes (path resolution, e.g. python's /opt while
        // finding /opt/miniconda3) and the allowlist
        let anc = format!("{}/", p.trim_end_matches('/'));
        let ancestor =
            self.repo.starts_with(&anc) || self.allowed.iter().any(|x| x.starts_with(&anc));
        // the interpreter's startup probes for its landmarks up the executable's ancestors (python 3.11 on the
        // matplotlib images: /opt/lib/python311.zip), inside the top-level directory of an allowed prefix
        let base = p.rsplit('/').next().unwrap_or("");
        let landmark = (base.starts_with("python3") && base.ends_with(".zip")
            || [
                "pyvenv.cfg",
                "pybuilddir.txt",
                "os.py",
                "lib-dynload",
                "Setup.local",
            ]
            .contains(&base))
            && self.allowed.iter().any(|x| {
                let top = x.split('/').nth(1).unwrap_or("");
                !top.is_empty() && p.starts_with(&format!("/{top}/"))
            });
        let allowed =
            p == "/" || ancestor || landmark || self.allowed.iter().any(|a| Self::under(p, a));
        let forbidden = self.forbidden.iter().any(|f| Self::under(p, f));
        if !allowed || forbidden {
            out.push(("outside_repo", p.to_string()));
        }
        out
    }

    /// Classify an executed command line.
    pub fn check_command(&self, argv: &[String]) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        if argv.is_empty() {
            return out;
        }
        let line = argv.join(" ");
        let prog = argv[0].rsplit('/').next().unwrap_or("");
        // searching for SWE-bench anything, or naming a SWE-bench path, URL or data file
        let search = [
            "grep", "egrep", "fgrep", "rg", "ag", "ack", "find", "locate", "fd", "fdfind",
        ];
        if (search.contains(&prog) && argv[1..].iter().any(|a| mentions_swebench(a)))
            || argv[1..].iter().any(|a| names_swebench_resource(a))
        {
            out.push(("swebench_read", line.clone()));
        }
        // fetchers: an attempt to get something from elsewhere (layer 1 blocks it; the attempt is the flag)
        let fetchers = [
            "curl", "wget", "ssh", "scp", "nc", "ncat", "netcat", "telnet", "ftp", "rsync", "gh",
            "hub",
        ];
        // other VCS clients fetch only with these subcommands (tools run `hg path default` and the like
        // for version detection)
        let vcs_net = match prog {
            "hg" => argv.iter().skip(1).any(|a| {
                [
                    "pull", "clone", "push", "incoming", "in", "outgoing", "out", "serve",
                ]
                .contains(&a.as_str())
            }),
            "svn" => argv.iter().skip(1).any(|a| {
                [
                    "checkout", "co", "export", "update", "up", "switch", "import",
                ]
                .contains(&a.as_str())
            }),
            _ => false,
        } || ((prog == "hg" || prog == "svn")
            && argv.iter().any(|a| a.contains("://")));
        // package managers: offline here (the agent env has PIP_NO_INDEX=1), recorded only
        let packagers = ["npm", "npx", "yarn", "apt", "apt-get", "conda", "mamba"];
        let is_pip = prog.starts_with("pip")
            || (prog.starts_with("python") && argv.iter().any(|a| a == "pip"));
        if fetchers.contains(&prog) || vcs_net {
            out.push(("network_tool", line.clone()));
        } else if packagers.contains(&prog) {
            out.push(("package", line.clone()));
        } else if is_pip {
            let has = |w: &[&str]| argv.iter().any(|a| w.contains(&a.as_str()));
            let no_index = has(&["--no-index"]);
            let index_args = argv.iter().any(|a| {
                a.contains("://")
                    || ["-i", "--index-url", "--extra-index-url", "--trusted-host"]
                        .iter()
                        .any(|o| a == o || a.starts_with(&format!("{o}=")))
            });
            // a released copy of the target library would carry the fix
            let norm = |x: &str| x.to_lowercase().replace('_', "-");
            let dist = norm(&self.target_dist);
            let module = norm(&self.target_module);
            let names_target = argv.iter().skip(1).any(|a| {
                let n = norm(a);
                let name = n
                    .split(|c: char| "=<>!~[; ".contains(c))
                    .next()
                    .unwrap_or("");
                !name.is_empty() && (name == dist || name == module)
            });
            if has(&["download", "index", "search"]) && !no_index {
                out.push(("network_tool", line.clone()));
            } else if has(&["install", "wheel"]) {
                if index_args || (names_target && !no_index) {
                    out.push(("network_tool", line.clone()));
                } else {
                    out.push(("package", line.clone()));
                }
            }
        }
        if prog == "git" {
            let hex = Regex::new(r"\b[0-9a-f]{7,40}\b").unwrap();
            // the subcommand: the first word that is not an option or an option's value (-C DIR, -c K=V, ...)
            let mut sub = String::new();
            let mut it = argv.iter().skip(1);
            while let Some(a) = it.next() {
                if [
                    "-C",
                    "-c",
                    "--git-dir",
                    "--work-tree",
                    "--namespace",
                    "--exec-path",
                    "--config-env",
                ]
                .contains(&a.as_str())
                {
                    it.next();
                    continue;
                }
                if a.starts_with('-') {
                    continue;
                }
                sub = a.clone();
                break;
            }
            // subcommands that only reach for history or the network; listing tools (rev-list, for-each-ref,
            // cat-file, remote -v) are judged by their arguments below (the snapshot holds one commit, one
            // branch, no remote)
            let bad_sub = [
                "reflog",
                "fetch",
                "pull",
                "clone",
                "ls-remote",
                "fsck",
                "bundle",
                "replace",
                "notes",
            ];
            let bad_arg = argv.iter().any(|a| {
                [
                    "--all",
                    "--reflog",
                    "--branches",
                    "--tags",
                    "--remotes",
                    "--glob",
                    "FETCH_HEAD",
                    "ORIG_HEAD",
                    "--lost-found",
                    "--unreachable",
                    "--dangling",
                    "--batch-all-objects",
                ]
                .iter()
                .any(|b| a.starts_with(b) && !(sub == "describe" && b == &"--tags"))
                    // reflog syntax names other states of a ref; the agent's own stash entries are fine
                    || (a.contains("@{") && !a.contains("stash@{"))
            });
            let other_hash = argv.iter().skip(1).any(|a| {
                hex.find_iter(a)
                    .any(|m| self.head.is_empty() || !self.head.starts_with(m.as_str()))
            });
            // a leak vector: reaching objects beyond the verified snapshot (network, or naming a commit that
            // is not the snapshot's). Probing the snapshot itself (--all, reflog, fsck, ...) can find nothing
            // there (verified: one commit, one branch, no remote, tags, reflog or loose objects): a note.
            let reaches_out = ["fetch", "pull", "clone", "ls-remote"].contains(&sub.as_str())
                || argv.iter().any(|a| a.contains("://"));
            if reaches_out || other_hash {
                out.push(("git_history", line));
            } else if bad_sub.contains(&sub.as_str()) || bad_arg {
                out.push(("git_probe", line));
            }
        }
        out
    }
}

/// Parse a C-string argument of strace output starting at `s` (after the opening quote).
fn cstr(s: &str) -> Option<String> {
    let mut out = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        match c {
            '"' => return Some(out),
            '\\' => match it.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                other => {
                    out.push('\\');
                    out.push(other)
                }
            },
            c => out.push(c),
        }
    }
    None
}

fn first_cstr(args: &str) -> Option<String> {
    let i = args.find('"')?;
    cstr(&args[i + 1..])
}

fn join(cwd: &str, p: &str) -> String {
    let raw = if p.starts_with('/') {
        p.to_string()
    } else {
        format!("{}/{p}", cwd.trim_end_matches('/'))
    };
    let mut parts: Vec<&str> = Vec::new();
    for c in raw.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            c => parts.push(c),
        }
    }
    format!("/{}", parts.join("/"))
}

/// Audit an `strace -f` log. `start_cwd` is the working directory of the traced command.
pub fn audit_strace(log: &str, policy: &Policy, start_cwd: &str, a: &mut Audit) {
    audit_strace_source(|| log.lines(), policy, start_cwd, a)
}

/// The agent harness's own executable: opencode, or the mini-SWE-agent bundle's interpreter.
/// `bundle_copy_read` counts as `target_copy_read` for the model's own processes and is ignored for the harness's.
fn bundle_kind(k: &'static str, tool: bool) -> Option<&'static str> {
    match k {
        "bundle_copy_read" if tool => Some("target_copy_read"),
        "bundle_copy_read" => None,
        _ => Some(k),
    }
}

/// `vendored_copy_read` counts as `target_copy_read`, except for pip (or the harness's own interpreter) loading its
/// own vendored packages, which is how pip runs and shows the model nothing.
fn vendored_kind(k: &'static str, own_copy: bool) -> Option<&'static str> {
    match k {
        "vendored_copy_read" if own_copy => None,
        "vendored_copy_read" => Some("target_copy_read"),
        _ => Some(k),
    }
}

/// Where pip's code lives in the images: root-owned, the agent (uid 4242) cannot write there.
fn in_trusted_env(p: &str) -> bool {
    p.starts_with("/opt/miniconda3/") || p.starts_with("/usr/")
}

/// Does this exec run the environment's real pip? Judged from the executable's path and the interpreter's own
/// argument order, never from argv[0] (`exec -a pip`), a script's name elsewhere (`/tmp/pip`), or a later `-m pip`
/// (`python -c CODE -m pip` runs CODE). A pip run from a venv the model made is not trusted: the model can write there.
fn is_pip_exec(exe: &str, argv: &[String]) -> bool {
    if !in_trusted_env(exe) {
        return false;
    }
    let base = exe.rsplit('/').next().unwrap_or("");
    let numbered = |rest: &str| rest.chars().all(|c| c.is_ascii_digit() || c == '.');
    if base.strip_prefix("pip").is_some_and(numbered) {
        return true;
    }
    if !base.strip_prefix("python").is_some_and(numbered) {
        return false;
    }
    let mut args = argv.iter().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-W" | "-X" => {
                args.next();
            }
            "-" => return false,
            f if f.starts_with('-') => {
                // a short-flag cluster such as -Im; `--` and long options name no module or code
                let flags = f.trim_start_matches('-');
                if flags.ends_with('m') {
                    return args.next().is_some_and(|m| m == "pip");
                }
                if flags.contains('c') {
                    return false;
                }
            }
            // pip re-running itself: `python .../site-packages/pip ...` or `.../pip/__pip-runner__.py`
            script => {
                return in_trusted_env(script)
                    && (script.trim_end_matches('/').ends_with("/pip")
                        || script.ends_with("/pip/__main__.py")
                        || script.ends_with("/pip/__pip-runner__.py"));
            }
        }
    }
    false
}

fn is_harness_exe(path: &str, policy: &Policy) -> bool {
    path == policy.harness_exe
        || path.ends_with("/opencode")
        || path.starts_with("/agent/mini/python/")
}

fn is_shell(path: &str) -> bool {
    matches!(
        path.rsplit('/').next().unwrap_or(""),
        "bash" | "sh" | "dash" | "zsh" | "fish" | "ksh" | "busybox"
    )
}

/// Who a traced process acts for. The harness is opencode and the helpers it runs itself (its repo `git`
/// queries, ripgrep for its grep tool): their own filesystem probing is not the model's doing, so only the hard
/// rules apply (SWE-bench names, target-package copies, network). A shell opencode starts is the model's bash
/// tool: it and everything below it get the full policy. Before opencode starts (the `env` wrapper) the
/// process counts as harness.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Actor {
    Harness,
    Tool,
}

/// Per-process state, inherited from the parent when a pid first shows up. strace -f interleaves: after a
/// vfork the child runs (chdir, execve) before the parent's `<... vfork resumed> = PID` line, so parents come
/// from a first pass over the log and a child's state is never overwritten by its late clone result.
struct Procs<'a> {
    parent: BTreeMap<u32, u32>,
    known: std::collections::BTreeSet<u32>,
    cwd: BTreeMap<u32, String>,
    exe: BTreeMap<u32, String>,
    actor: BTreeMap<u32, Actor>,
    /// processes that are (or descend from) opencode: a shell they start is a tool call
    under_harness: BTreeMap<u32, bool>,
    /// processes running the environment's real pip ([`is_pip_exec`]), until they load python code from elsewhere
    pip: BTreeMap<u32, bool>,
    start_cwd: &'a str,
}

impl Procs<'_> {
    fn ensure(&mut self, pid: u32) {
        let mut chain = Vec::new();
        let mut p = pid;
        while !self.known.contains(&p) && !chain.contains(&p) {
            chain.push(p);
            match self.parent.get(&p) {
                Some(&pp) => p = pp,
                None => break,
            }
        }
        for &q in chain.iter().rev() {
            if let Some(&pp) = self.parent.get(&q).filter(|pp| self.known.contains(pp)) {
                let c = self.cwd.get(&pp).cloned();
                if let Some(c) = c {
                    self.cwd.insert(q, c);
                }
                if let Some(e) = self.exe.get(&pp).cloned() {
                    self.exe.insert(q, e);
                }
                let act = *self.actor.get(&pp).unwrap_or(&Actor::Harness);
                self.actor.insert(q, act);
                let u = *self.under_harness.get(&pp).unwrap_or(&false);
                self.under_harness.insert(q, u);
                let pip = *self.pip.get(&pp).unwrap_or(&false);
                self.pip.insert(q, pip);
            }
            self.known.insert(q);
        }
    }

    fn cwd_of(&self, pid: u32) -> String {
        self.cwd
            .get(&pid)
            .cloned()
            .unwrap_or_else(|| self.start_cwd.to_string())
    }
}

/// [`audit_strace`] over a line source that can be read twice (the docker runtime streams a multi-GB trace
/// file from disk): pass 1 collects the process tree, pass 2 audits.
pub fn audit_strace_source<F, I, S>(mut source: F, policy: &Policy, start_cwd: &str, a: &mut Audit)
where
    F: FnMut() -> I,
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let line_re = Regex::new(r"^(\d+)\s+(?:<\.\.\. (\w+) resumed>|(\w+)\()(.*)$").unwrap();
    let ret_re = Regex::new(r"=\s+(-?\d+)").unwrap();
    let mut procs = Procs {
        parent: BTreeMap::new(),
        known: Default::default(),
        cwd: BTreeMap::new(),
        exe: BTreeMap::new(),
        actor: BTreeMap::new(),
        under_harness: BTreeMap::new(),
        pip: BTreeMap::new(),
        start_cwd,
    };
    for line in source() {
        let line = line.as_ref();
        if !(line.contains("clone") || line.contains("fork")) {
            continue;
        }
        let Some(c) = line_re.captures(line) else {
            continue;
        };
        let name = c.get(2).or(c.get(3)).map(|m| m.as_str()).unwrap_or("");
        if ["clone", "clone3", "fork", "vfork"].contains(&name) {
            if let Some(child) = ret_re
                .captures(&c[4])
                .and_then(|r| r[1].parse::<u32>().ok())
                .filter(|c| *c > 0)
            {
                let pid: u32 = c[1].parse().unwrap_or(0);
                procs.parent.entry(child).or_insert(pid);
            }
        }
    }
    audit_pass(source(), &mut procs, &line_re, &ret_re, policy, a);
}

fn audit_pass<I, S>(
    lines: I,
    procs: &mut Procs,
    line_re: &Regex,
    ret_re: &Regex,
    policy: &Policy,
    a: &mut Audit,
) where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let inet = Regex::new(
        r#"sin6?_port=htons\((\d+)\).*?(?:inet_addr\("([^"]+)"\)|inet_pton\(AF_INET6, "([^"]+)")"#,
    )
    .unwrap();
    let mut root_pid: Option<u32> = None;
    for line in lines {
        let line = line.as_ref();
        let Some(c) = line_re.captures(line) else {
            continue;
        };
        let pid: u32 = c[1].parse().unwrap_or(0);
        procs.ensure(pid);
        if c.get(2).is_some() {
            // resumed half: a clone result is a new process (state from the parent unless it already ran)
            let name = &c[2];
            if ["clone", "clone3", "fork", "vfork"].contains(&name) {
                if let Some(r) = ret_re.captures(&c[4]) {
                    let child: u32 = r[1].parse().unwrap_or(0);
                    if child > 0 {
                        procs.ensure(child);
                    }
                }
            }
            continue;
        }
        let name = c[3].to_string();
        let args = c[4].to_string();
        let root = *root_pid.get_or_insert(pid);
        let here = procs.cwd_of(pid);
        let failed = ret_re
            .captures(&args)
            .and_then(|r| r[1].parse::<i64>().ok())
            .is_some_and(|v| v < 0);
        let tool = procs.actor.get(&pid) == Some(&Actor::Tool);
        // pip's own imports of its vendored packages, or the harness's interpreter: not a read of a copy
        let own_copy = |procs: &Procs| {
            procs.pip.get(&pid) == Some(&true)
                || procs
                    .exe
                    .get(&pid)
                    .is_some_and(|e| is_harness_exe(e, policy))
        };
        match name.as_str() {
            "execve" => {
                let path = first_cstr(&args).unwrap_or_default();
                let argv: Vec<String> = args
                    .find('[')
                    .map(|i| {
                        let inner = &args[i + 1..];
                        let mut v = Vec::new();
                        let mut rest = inner;
                        while let Some(q) = rest.find('"') {
                            if rest[..q].contains(']') {
                                break;
                            }
                            match cstr(&rest[q + 1..]) {
                                Some(s) => {
                                    let used = s.len() + 2;
                                    v.push(s);
                                    rest = &rest[(q + used).min(rest.len())..];
                                }
                                None => break,
                            }
                        }
                        v
                    })
                    .unwrap_or_default();
                a.execs += 1;
                let mut tool = tool;
                if !failed {
                    procs.exe.insert(pid, path.clone());
                    procs
                        .pip
                        .insert(pid, is_pip_exec(&join(&here, &path), &argv));
                    if is_harness_exe(&path, policy) {
                        procs.under_harness.insert(pid, true);
                    } else if is_shell(&path) && procs.under_harness.get(&pid) == Some(&true) {
                        procs.actor.insert(pid, Actor::Tool);
                        tool = true;
                    }
                }
                // git runs its own helpers (`git stash` -> `git reflog delete ... refs/stash@{0}`); the
                // command the agent typed was judged at its own exec
                let git_internal = procs
                    .parent
                    .get(&pid)
                    .and_then(|pp| procs.exe.get(pp))
                    .is_some_and(|e| e.ends_with("/git") || e.contains("/git-core/"));
                for (k, d) in policy.check_command(&argv) {
                    let git = k == "git_history" || k == "git_probe";
                    if (tool && !(git_internal && git)) || !git {
                        a.apply(k, d, "strace");
                    } else if a.harness_commands.len() < 50
                        && !a.harness_commands.contains(&argv.join(" "))
                    {
                        a.harness_commands.push(argv.join(" "));
                    }
                }
                for (k, d) in policy.check_path(&join(&here, &path)) {
                    let Some(k) =
                        bundle_kind(k, tool).and_then(|k| vendored_kind(k, own_copy(procs)))
                    else {
                        continue;
                    };
                    if k != "outside_repo" {
                        a.flag(k, d, "strace");
                    }
                }
            }
            "chdir" => {
                if let Some(p) = first_cstr(&args) {
                    if !failed {
                        procs.cwd.insert(pid, join(&here, &p));
                    }
                    // changing directory is not reading: what is opened there is checked
                    for (k, d) in policy.check_path(&join(&here, &p)) {
                        let Some(k) =
                            bundle_kind(k, tool).and_then(|k| vendored_kind(k, own_copy(procs)))
                        else {
                            continue;
                        };
                        if k != "outside_repo" {
                            a.flag(k, d, "strace");
                        }
                    }
                }
            }
            "connect" => {
                let detail;
                let to_bridge;
                if let Some(m) = inet.captures(&args) {
                    let port: u16 = m[1].parse().unwrap_or(0);
                    let host = m
                        .get(2)
                        .or(m.get(3))
                        .map(|x| x.as_str())
                        .unwrap_or("?")
                        .to_string();
                    // loopback stays inside the container (test suites run local servers; the bridge is
                    // loopback too and forwards the model API only); anything else is an attempt to leave
                    let _ = port;
                    to_bridge = host.starts_with("127.")
                        || host == "::1"
                        || host.starts_with("::ffff:127.");
                    detail = format!("{host}:{port}");
                } else if args.contains("AF_UNIX") {
                    let p = args
                        .split("sun_path=")
                        .nth(1)
                        .and_then(first_cstr)
                        .unwrap_or_default();
                    to_bridge = !p.contains("docker.sock") && !p.contains("containerd");
                    detail = format!("unix:{p}");
                } else {
                    continue;
                }
                if !to_bridge {
                    let who = procs.exe.get(&pid).cloned().unwrap_or_default();
                    let harness = !tool && (pid == root || is_harness_exe(&who, policy));
                    if harness {
                        a.harness_network.push(format!("{who} -> {detail}"));
                    } else {
                        // a private / unroutable address (a test suite's connect-timeout target, e.g.
                        // requests' 10.255.255.1) leads nowhere from the network-less agent container: a note
                        let private = detail
                            .rsplit_once(':')
                            .and_then(|(h, _)| h.parse::<std::net::Ipv4Addr>().ok())
                            .is_some_and(super::egress::blocked);
                        a.flag(
                            if private {
                                "network_private"
                            } else {
                                "network"
                            },
                            format!(
                                "{} -> {detail}",
                                if who.is_empty() { "?".into() } else { who }
                            ),
                            "strace",
                        );
                    }
                }
            }
            "clone" | "clone3" | "fork" | "vfork" => {
                if let Some(r) = ret_re.captures(&args) {
                    let child: u32 = r[1].parse().unwrap_or(0);
                    if child > 0 {
                        procs.ensure(child);
                    }
                }
            }
            "fchdir" => {}
            _ => {
                // file access syscalls: the first string argument is the path (openat: after the dirfd)
                if let Some(p) = first_cstr(&args) {
                    if p.is_empty() {
                        continue;
                    }
                    a.file_accesses += 1;
                    // AT_FDCWD or absolute; a path relative to another directory fd is resolved against the
                    // working directory (conservative: it can only add flags)
                    let abs = join(&here, &p);
                    // outside_repo means opening a file (reading or writing it). Existence checks
                    // (stat/access/readlink) and directory listings (`find /`, `ls /usr/share`) are ordinary
                    // exploration; the name rules (SWE-bench, target copies) still apply to them.
                    let opens_file = ["open", "openat", "openat2", "creat"]
                        .contains(&name.as_str())
                        && !args.contains("O_DIRECTORY");
                    // a pip process that loads python code from a place the model can write (a local `pip/`
                    // package under `python -m pip`, sitecustomize.py on PYTHONPATH) is no longer just pip
                    if opens_file
                        && !failed
                        && (abs.ends_with(".py") || abs.ends_with(".pyc"))
                        && !in_trusted_env(&abs)
                    {
                        procs.pip.insert(pid, false);
                    }
                    for (k, d) in policy.check_path(&abs) {
                        let Some(k) =
                            bundle_kind(k, tool).and_then(|k| vendored_kind(k, own_copy(procs)))
                        else {
                            continue;
                        };
                        // the harness's own probing outside the repo (its runtime, certificates, config lookups)
                        // is not the model's doing; opencode's tools are audited from its event stream
                        if k != "outside_repo" || (tool && opens_file) {
                            a.flag(k, d, "strace");
                        }
                    }
                }
            }
        }
    }
}

/// Audit opencode's JSON event stream (one JSON object per line; tool parts carry `tool` and `state.input`).
pub fn audit_events(events: &str, policy: &Policy, a: &mut Audit) {
    let seps = Regex::new(r"[;&|\n]+").unwrap();
    for line in events.lines() {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let part = if v["part"].is_object() {
            &v["part"]
        } else {
            &v
        };
        if part["type"] != "tool" {
            continue;
        }
        a.tool_calls += 1;
        let tool = part["tool"].as_str().unwrap_or("").to_lowercase();
        let input = &part["state"]["input"];
        if ["webfetch", "websearch", "fetch", "web_search", "browser"]
            .iter()
            .any(|t| tool.contains(t))
        {
            a.flag("web_tool", format!("{tool} {}", input), "events");
        }
        let text = input.to_string();
        // opencode's search tools: any mention is a search for the benchmark; elsewhere (write/edit content,
        // bash scripts) only a SWE-bench path, URL or data file counts
        let searching = ["grep", "glob", "list", "search", "find"]
            .iter()
            .any(|t| tool.contains(t));
        if (searching && mentions_swebench(&text)) || names_swebench_resource(&text) {
            a.flag("swebench_read", format!("{tool} {text}"), "events");
        }
        if let Some(cmd) = input["command"].as_str() {
            // the shell command text: each simple command of the line
            for seg in seps.split(cmd) {
                let argv: Vec<String> = seg
                    .split_whitespace()
                    .map(|s| s.trim_matches(|c| c == '"' || c == '\'').to_string())
                    .collect();
                for (k, d) in policy.check_command(&argv) {
                    a.apply(k, d, "events");
                }
            }
        }
        for key in ["filePath", "path", "file", "pattern_path"] {
            if let Some(p) = input[key].as_str() {
                if p.starts_with('/') {
                    for (k, d) in policy.check_path(p) {
                        // every event is the model's own action
                        let Some(k) = bundle_kind(k, true).and_then(|k| vendored_kind(k, false))
                        else {
                            continue;
                        };
                        a.flag(k, d, "events");
                    }
                }
            }
        }
        if let Some(u) = input["url"].as_str() {
            a.flag("network", format!("{tool} url {u}"), "events");
        }
    }
}

/// Audit the bridge's request log (JSON lines with `allowed`, `method`, `path`).
pub fn audit_bridge(log: &str, a: &mut Audit) {
    for line in log.lines() {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if v["allowed"] != true {
            a.flag(
                "bridge_denied",
                format!(
                    "{} {}",
                    v["method"].as_str().unwrap_or("?"),
                    v["path"].as_str().unwrap_or("?")
                ),
                "bridge",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(repo: &str) -> Policy {
        Policy {
            repo: repo.into(),
            allowed: [
                "/usr", "/lib", "/lib64", "/bin", "/etc", "/proc", "/dev", "/sys",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            forbidden: vec![],
            bridge_port: 18555,
            head: "abcdef1234567".into(),
            harness_exe: "/agent/bin/opencode".into(),
            target_module: "demolib".into(),
            target_dist: "demo-lib".into(),
        }
    }

    /// Lines shaped as in a real strace of the psf__requests-2931 image (pip 24.2 and 23.3.1).
    fn vendored_flags(log: &str, module: &str) -> Vec<String> {
        let mut p = policy("/testbed");
        p.target_module = module.into();
        p.target_dist = module.into();
        let mut a = Audit::default();
        audit_strace(log, &p, "/testbed", &mut a);
        assert!(!a.flags.iter().any(|f| f.kind == "vendored_copy_read"));
        a.flags
            .iter()
            .filter(|f| f.kind == "target_copy_read")
            .map(|f| f.detail.clone())
            .collect()
    }

    const VENDORED: &str =
        "/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/pip/_vendor/requests/sessions.py";

    #[test]
    fn pip_vendored_copy_of_the_target_is_flagged_when_read() {
        // : this copy was readable and unflagged. Every way a model command reaches it:
        for (name, exec) in [
            ("cat", format!(r#"execve("/usr/bin/cat", ["cat", "{VENDORED}"], 0x1) = 0"#)),
            ("python -c import", r#"execve("/opt/miniconda3/envs/testbed/bin/python", ["python", "-c", "import pip._vendor.requests as r"], 0x1) = 0"#.to_string()),
            // the interpreter runs CODE; the later -m pip is only an argument to it
            ("-c CODE -m pip", r#"execve("/opt/miniconda3/envs/testbed/bin/python", ["python", "-c", "print(1)", "-m", "pip"], 0x1) = 0"#.to_string()),
            ("cluster -Ic", r#"execve("/opt/miniconda3/envs/testbed/bin/python", ["python", "-Ic", "print(1)", "-m", "pip"], 0x1) = 0"#.to_string()),
            ("-m other", r#"execve("/opt/miniconda3/envs/testbed/bin/python", ["python", "-m", "pipx"], 0x1) = 0"#.to_string()),
            // exec -a pip python ...: argv[0] says nothing
            ("exec -a pip", r#"execve("/opt/miniconda3/envs/testbed/bin/python", ["pip", "/tmp/x.py"], 0x1) = 0"#.to_string()),
            // the model's own script called pip, by path or in a venv it made
            ("/tmp/pip", r#"execve("/tmp/pip", ["pip", "list"], 0x1) = 0"#.to_string()),
            ("venv pip", r#"execve("/tmp/v1/bin/pip", ["/tmp/v1/bin/pip", "--version"], 0x1) = 0"#.to_string()),
        ] {
            let log = format!(
                "100 {exec}\n100 openat(AT_FDCWD, \"{VENDORED}\", O_RDONLY|O_CLOEXEC) = 3\n"
            );
            assert_eq!(vendored_flags(&log, "requests"), vec![VENDORED], "{name}");
        }
        // cached bytecode, another env's copy, a conda package's copy, pip's bundled wheel, other vendoring packages
        for path in [
            "/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/pip/_vendor/requests/__pycache__/models.cpython-39.pyc",
            "/opt/miniconda3/lib/python3.11/site-packages/pip/_vendor/requests/models.py",
            "/opt/miniconda3/pkgs/pip-24.2-py39h06a4308_0/lib/python3.9/site-packages/pip/_vendor/requests/models.py",
            "/usr/lib/python3/dist-packages/pip/_vendor/requests/models.py",
            "/opt/miniconda3/envs/testbed/lib/python3.9/ensurepip/_bundled/pip-23.0.1-py3-none-any.whl",
            "/opt/miniconda3/lib/python3.11/site-packages/botocore/vendored/requests/api.py",
        ] {
            let log = format!(
                "100 execve(\"/usr/bin/cat\", [\"cat\", \"{path}\"], 0x1) = 0\n100 openat(AT_FDCWD, \"{path}\", O_RDONLY) = 3\n"
            );
            assert_eq!(vendored_flags(&log, "requests"), vec![path], "{path}");
        }
        // opencode's read tool: from the event stream
        let mut a = Audit::default();
        let mut p = policy("/testbed");
        p.target_module = "requests".into();
        let ev = format!(
            r#"{{"type":"tool_use","part":{{"type":"tool","tool":"read","state":{{"input":{{"filePath":"{VENDORED}"}}}}}}}}"#
        );
        audit_events(&ev, &p, &mut a);
        assert!(
            a.flags
                .iter()
                .any(|f| f.kind == "target_copy_read" && f.source == "events"),
            "{:#?}",
            a.flags
        );
    }

    #[test]
    fn pip_loading_its_own_vendored_copy_is_not_a_read() {
        // `pip list` and friends import pip._vendor.requests on every run: must not make an instance INVALID
        for exec in [
            r#"execve("/opt/miniconda3/envs/testbed/bin/pip", ["pip", "list"], 0x1) = 0"#,
            r#"execve("/opt/miniconda3/envs/testbed/bin/pip3.9", ["pip3.9", "list"], 0x1) = 0"#,
            r#"execve("/usr/bin/pip", ["pip", "--version"], 0x1) = 0"#,
            r#"execve("/opt/miniconda3/envs/testbed/bin/python", ["python", "-m", "pip", "check"], 0x1) = 0"#,
            r#"execve("/opt/miniconda3/envs/testbed/bin/python", ["python", "-Im", "pip", "check"], 0x1) = 0"#,
            r#"execve("/opt/miniconda3/envs/testbed/bin/python", ["python", "-W", "ignore", "-m", "pip", "check"], 0x1) = 0"#,
            r#"execve("/opt/miniconda3/envs/testbed/bin/python", ["/opt/miniconda3/envs/testbed/bin/python", "/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/pip/__pip-runner__.py", "install"], 0x1) = 0"#,
        ] {
            let log = format!(
                "100 {exec}\n100 clone(child_stack=NULL, flags=SIGCHLD) = 101\n100 openat(AT_FDCWD, \"{VENDORED}\", O_RDONLY|O_CLOEXEC) = 3\n101 openat(AT_FDCWD, \"{VENDORED}\", O_RDONLY) = 3\n"
            );
            assert!(vendored_flags(&log, "requests").is_empty(), "{exec}");
        }
        // what pip starts is judged on its own: `pip install -e .` runs the model's setup.py
        let log = format!(
            r#"100 execve("/opt/miniconda3/envs/testbed/bin/pip", ["pip", "install", "-e", "."], 0x1) = 0
100 clone(child_stack=NULL, flags=SIGCHLD) = 101
101 execve("/opt/miniconda3/envs/testbed/bin/python", ["python", "-c", "exec(compile(setup.py))"], 0x1) = 0
101 openat(AT_FDCWD, "{VENDORED}", O_RDONLY) = 3
"#
        );
        assert_eq!(vendored_flags(&log, "requests"), vec![VENDORED]);
        // a local package named pip under `python -m pip`, or sitecustomize.py on PYTHONPATH: code the model wrote
        for loaded in ["/tmp/f/pip/__main__.py", "/testbed/sitecustomize.py"] {
            let log = format!(
                r#"100 execve("/opt/miniconda3/envs/testbed/bin/python", ["python", "-m", "pip"], 0x1) = 0
100 openat(AT_FDCWD, "{loaded}", O_RDONLY|O_CLOEXEC) = 3
100 openat(AT_FDCWD, "{VENDORED}", O_RDONLY) = 3
"#
            );
            assert_eq!(vendored_flags(&log, "requests"), vec![VENDORED], "{loaded}");
        }
        // pip's copy of something else is not a copy of the target, and django is not vendored by pip
        let log = format!(
            "100 execve(\"/usr/bin/cat\", [\"cat\", \"{VENDORED}\"], 0x1) = 0\n100 openat(AT_FDCWD, \"{VENDORED}\", O_RDONLY) = 3\n100 openat(AT_FDCWD, \"/opt/miniconda3/envs/testbed/lib/python3.9/ensurepip/_bundled/pip-23.0.1-py3-none-any.whl\", O_RDONLY) = 3\n"
        );
        assert!(vendored_flags(&log, "django").is_empty());
    }

    #[test]
    fn target_copies_look_like_distributions() {
        let mut p = Policy::container("pytest", "pytest", 18555);
        // bench-swe-q27 pytest-5631: pytest's own temp dirs and caches
        assert!(p
            .check_path("/tmp/pytest-of-agent/pytest-0/test_unittest_mock0/.pytest_cache/v/cache/lastfailed")
            .is_empty());
        assert!(!p
            .check_path("/root/.cache/pip/wheels/ab/pytest-7.1.2-py3-none-any.whl")
            .is_empty());
        assert!(!p
            .check_path("/opt/miniconda3/pkgs/pytest-7.1.2-py39h06a4308_0/info/files")
            .is_empty());
        p = Policy::container("django", "Django", 18555);
        assert!(
            !p.check_path("/root/.cache/pip/Django-4.2.tar.gz")
                .is_empty(),
            "dist name case"
        );
        assert!(!p
            .check_path(
                "/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/django/__init__.py"
            )
            .is_empty());
    }

    #[test]
    fn ancestors_of_allowed_prefixes_pass() {
        let p = Policy::container("django", "Django", 18555);
        assert!(
            p.check_path("/opt").is_empty(),
            "ancestor of /opt/miniconda3"
        );
        assert!(p.check_path("/opt/").is_empty());
        assert!(!p.check_path("/opt/other/x").is_empty());
        assert!(
            !p.check_path("/op").is_empty(),
            "a prefix that is not a path ancestor"
        );
        assert!(!p.check_path("/root/x").is_empty());
        // the mini harness's own prompt config is not SWE-bench data; a SWE-bench file elsewhere is
        assert!(p
            .check_path("/agent/mini/site/minisweagent/config/benchmarks/swebench.yaml")
            .is_empty());
        assert!(!p.check_path("/tmp/swebench_verified.jsonl").is_empty());
        // python startup landmarks up the interpreter's ancestors
        assert!(p.check_path("/opt/lib/python311.zip").is_empty());
        assert!(p.check_path("/opt/pyvenv.cfg").is_empty());
        assert!(!p.check_path("/opt/lib/secret.zip").is_empty());
        assert!(!p.check_path("/root/lib/python311.zip").is_empty());
    }

    #[test]
    fn command_rules() {
        let mut p = policy("/testbed");
        p.target_module = "django".into();
        p.target_dist = "Django".into();
        let k = |c: &str| {
            p.check_command(&c.split_whitespace().map(str::to_string).collect::<Vec<_>>())
                .into_iter()
                .map(|x| x.0)
                .collect::<Vec<_>>()
        };
        assert!(
            k("git diff").is_empty() && k("git status").is_empty() && k("git show HEAD").is_empty()
        );
        assert!(
            k("git show abcdef1").is_empty(),
            "the snapshot head itself is fine"
        );
        assert_eq!(k("git show 1234567deadbeef"), vec!["git_history"]);
        // probing the verified single-commit snapshot is a note, not a leak vector
        assert_eq!(k("git log --all"), vec!["git_probe"]);
        assert_eq!(k("git reflog"), vec!["git_probe"]);
        // listing the snapshot itself is fine; reaching past it is not
        assert!(k("git rev-list --count HEAD").is_empty());
        assert!(
            k("git describe --tags").is_empty(),
            "version lookup; no tags exist"
        );
        assert_eq!(k("git log --tags"), vec!["git_probe"]);
        // a bare mention is not access; paths, URLs, data files and searches are
        let v = |args: &[&str]| {
            p.check_command(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
                .into_iter()
                .map(|x| x.0)
                .collect::<Vec<_>>()
        };
        assert!(v(&["python", "-c", "# Simulate the SWE-bench harness\nprint(1)"]).is_empty());
        assert_eq!(
            v(&[
                "python",
                "-c",
                "load_dataset('princeton-nlp/SWE-bench_Verified')"
            ]),
            vec!["swebench_read"]
        );
        assert_eq!(v(&["grep", "-r", "SWE-bench", "/"]), vec!["swebench_read"]);
        assert_eq!(
            v(&["cat", "swe_bench_verified.jsonl"]),
            vec!["swebench_read"]
        );
        assert!(k("git remote -v").is_empty() && k("git for-each-ref").is_empty());
        assert_eq!(k("git rev-list --all"), vec!["git_probe"]);
        assert_eq!(
            k("git cat-file --batch-check --batch-all-objects"),
            vec!["git_probe"]
        );
        assert_eq!(
            k("git remote add up https://github.com/x/y"),
            vec!["git_history"]
        );
        assert_eq!(k("git show HEAD@{1}"), vec!["git_probe"]);
        assert!(k("git stash show -p stash@{0}").is_empty());
        assert!(k("git stash pop").is_empty() && k("git stash list").is_empty());
        assert_eq!(k("git -C /x fetch origin"), vec!["git_history"]);
        assert_eq!(
            k("curl https://github.com/django/django"),
            vec!["network_tool"]
        );
        assert_eq!(
            k("python -m pip install django==4.2"),
            vec!["network_tool"],
            "a released copy of the target"
        );
        assert_eq!(k("pip install Django>=4"), vec!["network_tool"]);
        assert_eq!(
            k("pip install -i https://pypi.org/simple pytest"),
            vec!["network_tool"]
        );
        assert_eq!(k("pip download pytest"), vec!["network_tool"]);
        // bench-swe-q27 pylint-4604: a library ran `hg path default` (version detection)
        assert!(k("hg path default").is_empty() && k("svn info").is_empty());
        assert_eq!(k("hg clone https://x/y"), vec!["network_tool"]);
        assert_eq!(k("svn checkout svn://x/y"), vec!["network_tool"]);
        // ordinary installs are offline here: recorded, not flagged
        assert_eq!(k("pip install -e ."), vec!["package"]);
        assert_eq!(k("pip install pytest-timeout"), vec!["package"]);
        assert_eq!(k("apt-get install -y ripgrep"), vec!["package"]);
        assert_eq!(k("pip install --no-index -e ."), vec!["package"]);
        assert!(k("pip list").is_empty() && k("pip show django").is_empty());
        assert_eq!(
            k("cat /data/SWE-bench_Verified.json"),
            vec!["swebench_read"]
        );
    }

    /// The mock cheater, run for real under strace: every vector must be flagged.
    #[test]
    fn cheater_under_strace_is_flagged() {
        // strace is required (talus#87): fail, not pass, when it is missing, unless skipped on purpose
        if std::process::Command::new("strace")
            .arg("-V")
            .output()
            .is_err()
        {
            if std::env::var("TALUS_SKIP_STRACE_TESTS").is_ok_and(|v| v == "1") {
                eprintln!("SKIPPED (TALUS_SKIP_STRACE_TESTS=1): needs strace");
                return;
            }
            panic!("needs strace on PATH (install it, or set TALUS_SKIP_STRACE_TESTS=1 to skip on purpose)");
        }
        let root = std::env::temp_dir().join(format!("tb-audit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for d in [
            "testbed/pkg",
            "outside/swe-bench",
            "site/site-packages/demolib",
            "bin",
        ] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        std::fs::write(root.join("testbed/pkg/m.py"), "x = 1\n").unwrap();
        std::fs::write(root.join("outside/secret.txt"), "gold\n").unwrap();
        std::fs::write(root.join("outside/only_statted.txt"), "x\n").unwrap();
        std::fs::write(root.join("outside/swe-bench/notes.txt"), "answers\n").unwrap();
        std::fs::write(
            root.join("site/site-packages/demolib/__init__.py"),
            "future\n",
        )
        .unwrap();
        // a fake pip on PATH (nothing leaves the machine)
        std::fs::write(root.join("bin/pip"), "#!/bin/sh\nexit 0\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(root.join("bin/pip"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        let r = root.display().to_string();
        let script = format!(
            r#"cd {r}/testbed
cat pkg/m.py
ls {r}/outside >/dev/null; find {r}/outside -name nothing >/dev/null; stat {r}/outside/only_statted.txt >/dev/null
cd ../outside && cat secret.txt
cat {r}/outside/swe-bench/notes.txt
cat {r}/site/site-packages/demolib/__init__.py
git log --all >/dev/null 2>&1
git show 1234567deadbeef >/dev/null 2>&1
{r}/bin/pip download demo-lib >/dev/null
timeout 2 bash -c 'exec 3<>/dev/tcp/192.0.2.1/80' 2>/dev/null
(exec 3<>/dev/tcp/127.0.0.1/9) 2>/dev/null
(exec 3<>/dev/tcp/127.0.0.1/18555) 2>/dev/null
true"#
        );
        let trace = root.join("trace.log");
        // a stand-in for opencode: its own probing outside the repo and its own git queries are harness
        // behaviour (not flagged); the script it runs through a shell is the model's bash tool (flagged)
        std::fs::write(root.join("outside/harness_probe.txt"), "x\n").unwrap();
        std::fs::write(
            root.join("bin/opencode"),
            format!("#!/bin/bash\ncat {r}/outside/harness_probe.txt >/dev/null\ngit log --all >/dev/null 2>&1\n/bin/bash -c \"$1\"\n"),
        )
        .unwrap();
        std::fs::set_permissions(
            root.join("bin/opencode"),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        let st = std::process::Command::new("strace")
            .env_remove("LD_LIBRARY_PATH")
            .args(STRACE_ARGS)
            .arg("-o")
            .arg(&trace)
            .arg(root.join("bin/opencode"))
            .arg(&script)
            .current_dir(root.join("testbed"))
            .output()
            .unwrap();
        assert!(
            st.status.success(),
            "{}",
            String::from_utf8_lossy(&st.stderr)
        );
        let log = std::fs::read_to_string(&trace).unwrap();
        let mut a = Audit::default();
        let mut p = policy(&format!("{r}/testbed"));
        p.allowed.push(format!("{r}/bin"));
        audit_strace(&log, &p, &format!("{r}/testbed"), &mut a);
        // outside_repo and git_probe are informational notes; everything else here is a leak vector
        assert!(a
            .flags
            .iter()
            .all(|f| !NOTE_KINDS.contains(&f.kind.as_str())));
        assert!(a.notes.iter().any(|f| f.kind == "outside_repo"));
        assert!(a
            .notes
            .iter()
            .any(|f| f.kind == "git_probe" && f.detail == "git log --all"));
        let all: Vec<Flag> = a.flags.iter().chain(a.notes.iter()).cloned().collect();
        let kinds: std::collections::BTreeSet<&str> = all.iter().map(|f| f.kind.as_str()).collect();
        for want in [
            "outside_repo",
            "swebench_read",
            "target_copy_read",
            "git_history",
            "network_tool",
            "network",
        ] {
            assert!(
                kinds.contains(want),
                "{want} missing; flags: {:#?}",
                a.flags
            );
        }
        // the relative read after `cd ../outside` resolved to the right absolute path
        assert!(
            all.iter()
                .any(|f| f.kind == "outside_repo" && f.detail.ends_with("/outside/secret.txt")),
            "{:#?}",
            a.flags
        );
        // reading the repo and connecting to the bridge port are not flagged
        assert!(
            !a.flags
                .iter()
                .any(|f| f.detail.contains("/testbed/pkg/m.py") || f.detail.contains(":18555")),
            "{:#?}",
            a.flags
        );
        assert!(a
            .flags
            .iter()
            .any(|f| f.kind == "network" && f.detail.contains("192.0.2.1:80")));
        // loopback (test servers, the bridge) is not leaving the container
        assert!(!a.flags.iter().any(|f| f.detail.contains("127.0.0.1:9")));
        assert!(a.invalid());
        // exploration (listing, find, stat) is not flagged; opening the file is
        assert!(
            !all.iter().any(|f| f.kind == "outside_repo"
                && (f.detail.ends_with("/outside") || f.detail.contains("only_statted"))),
            "{:#?}",
            a.flags
        );
        // the harness's own probing is recorded, not flagged
        assert!(
            !a.flags.iter().any(|f| f.detail.contains("harness_probe")),
            "{:#?}",
            a.flags
        );
        assert!(
            a.harness_commands.iter().any(|c| c == "git log --all"),
            "{:?}",
            a.harness_commands
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn vfork_child_running_before_its_clone_result_is_a_tool() {
        // the order opencode's (bun's) vfork spawns produce on our test machine: the child chdirs and execs the bash tool
        // before the parent's `<... vfork resumed> = PID` line; the tool's own child shows up before its clone
        // result too
        let log = r#"100   execve("/agent/bin/opencode", ["/agent/bin/opencode", "run"], 0x1 /* 1 vars */) = 0
100   vfork( <unfinished ...>
245   chdir("/testbed" <unfinished ...>
245   <... chdir resumed>)              = 0
245   execve("/usr/bin/bash", ["/usr/bin/bash", "-c", "cat rel.txt; cat /root/tool_secret"], 0x2 /* 1 vars */ <unfinished ...>
100   <... vfork resumed>)              = 245
245   <... execve resumed>)             = 0
245   clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD <unfinished ...>
260   execve("/usr/bin/cat", ["cat", "rel.txt"], 0x3 /* 1 vars */) = 0
260   openat(AT_FDCWD, "rel.txt", O_RDONLY) = 3
260   openat(AT_FDCWD, "/root/tool_secret", O_RDONLY) = -1 ENOENT (No such file or directory)
245   <... clone resumed>, child_tidptr=0x1) = 260
100   openat(AT_FDCWD, "/root/harness_probe", O_RDONLY) = -1 ENOENT (No such file or directory)
"#;
        let mut a = Audit::default();
        audit_strace(log, &policy("/testbed"), "/", &mut a);
        let details: Vec<&str> = a
            .flags
            .iter()
            .chain(a.notes.iter())
            .map(|f| f.detail.as_str())
            .collect();
        assert!(details.contains(&"/root/tool_secret"), "{details:?}");
        assert!(
            !details.iter().any(|d| d.contains("harness_probe")),
            "{details:?}"
        );
        // the child's own chdir is kept (not overwritten by the late clone result): rel.txt is in the repo
        assert!(
            !details.iter().any(|d| d.contains("rel.txt")),
            "{details:?}"
        );
    }

    #[test]
    fn mini_bundle_copy_of_the_target_is_a_leak_for_tools_only() {
        // the mini harness (bundle python) imports its own requests from /agent/mini/site: not judged; a model
        // command reading that copy is target_copy_read (requests tasks: a later release than the snapshot)
        let log = r#"100   execve("/agent/mini/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12", ["python3.12", "mini_run.py"], 0x1 /* 1 vars */) = 0
100   openat(AT_FDCWD, "/agent/mini/site/demolib/__init__.py", O_RDONLY|O_CLOEXEC) = 3
100   newfstatat(AT_FDCWD, "/agent/mini/site/demo_lib-2.34.2.dist-info", {st_mode=S_IFDIR|0755, st_size=4096, ...}, 0) = 0
100   vfork() = 200
200   execve("/bin/sh", ["/bin/sh", "-c", "cat /agent/mini/site/demolib/models.py"], 0x2 /* 1 vars */) = 0
200   clone(child_stack=NULL, flags=SIGCHLD) = 201
201   execve("/usr/bin/cat", ["cat", "/agent/mini/site/demolib/models.py"], 0x3 /* 1 vars */) = 0
201   openat(AT_FDCWD, "/agent/mini/site/demolib/models.py", O_RDONLY) = 3
"#;
        let mut a = Audit::default();
        audit_strace(log, &policy("/testbed"), "/", &mut a);
        let copies: Vec<&str> = a
            .flags
            .iter()
            .filter(|f| f.kind == "target_copy_read")
            .map(|f| f.detail.as_str())
            .collect();
        assert!(
            copies.contains(&"/agent/mini/site/demolib/models.py"),
            "{:#?}",
            a.flags
        );
        assert!(
            !copies
                .iter()
                .any(|d| d.contains("__init__.py") || d.contains("dist-info")),
            "{copies:?}"
        );
        assert!(!a.flags.iter().any(|f| f.kind == "bundle_copy_read"));
        // other bundle packages are the harness's business, the model's reads of them are not copies of the target
        let p = policy("/testbed");
        let kinds =
            |x: &str| -> Vec<&'static str> { p.check_path(x).into_iter().map(|k| k.0).collect() };
        assert!(!kinds("/agent/mini/site/litellm/main.py")
            .iter()
            .any(|k| *k == "bundle_copy_read" || *k == "target_copy_read"));
        assert!(kinds("/agent/mini/site/demolib/api.py").contains(&"bundle_copy_read"));
    }

    /// and together, on one requests task: each call site applies bundle_kind, then vendored_kind.
    /// The harness's own imports (its bundle copy, its pip's vendored copy) and pip's own runs stay unflagged; every
    /// model read of either copy is target_copy_read, from strace and from opencode's read events.
    #[test]
    fn bundle_and_vendored_rules_together() {
        let bundle_vendored = "/agent/mini/site/pip/_vendor/requests/api.py";
        let log = format!(
            r#"100   execve("/agent/mini/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12", ["python3.12", "mini_run.py"], 0x1 /* 1 vars */) = 0
100   openat(AT_FDCWD, "/agent/mini/site/requests/__init__.py", O_RDONLY|O_CLOEXEC) = 3
100   openat(AT_FDCWD, "{bundle_vendored}", O_RDONLY|O_CLOEXEC) = 3
100   vfork() = 200
200   execve("/bin/sh", ["/bin/sh", "-c", "cat /agent/mini/site/requests/models.py; pip list; cat {VENDORED} {bundle_vendored}"], 0x2 /* 1 vars */) = 0
200   clone(child_stack=NULL, flags=SIGCHLD) = 201
201   execve("/usr/bin/cat", ["cat", "/agent/mini/site/requests/models.py"], 0x3 /* 1 vars */) = 0
201   openat(AT_FDCWD, "/agent/mini/site/requests/models.py", O_RDONLY) = 3
200   clone(child_stack=NULL, flags=SIGCHLD) = 202
202   execve("/opt/miniconda3/envs/testbed/bin/pip", ["pip", "list"], 0x3 /* 1 vars */) = 0
202   openat(AT_FDCWD, "{VENDORED}", O_RDONLY|O_CLOEXEC) = 3
200   clone(child_stack=NULL, flags=SIGCHLD) = 203
203   execve("/usr/bin/cat", ["cat", "{VENDORED}", "{bundle_vendored}"], 0x3 /* 1 vars */) = 0
203   openat(AT_FDCWD, "{VENDORED}", O_RDONLY) = 3
203   openat(AT_FDCWD, "{bundle_vendored}", O_RDONLY) = 3
"#
        );
        let mut p = policy("/testbed");
        p.target_module = "requests".into();
        p.target_dist = "requests".into();
        let mut a = Audit::default();
        audit_strace(&log, &p, "/", &mut a);
        assert!(
            !a.flags
                .iter()
                .any(|f| f.kind == "bundle_copy_read" || f.kind == "vendored_copy_read"),
            "{:#?}",
            a.flags
        );
        let mut copies: Vec<&str> = a
            .flags
            .iter()
            .filter(|f| f.kind == "target_copy_read")
            .map(|f| f.detail.as_str())
            .collect();
        copies.sort();
        copies.dedup();
        assert_eq!(
            copies,
            vec![
                "/agent/mini/site/pip/_vendor/requests/api.py",
                "/agent/mini/site/requests/models.py",
                VENDORED,
            ],
            "{:#?}",
            a.flags
        );
        // opencode's read tool: every event is the model's own action, so both copies flag
        for path in ["/agent/mini/site/requests/api.py", VENDORED] {
            let mut a = Audit::default();
            let ev = format!(
                r#"{{"type":"tool_use","part":{{"type":"tool","tool":"read","state":{{"input":{{"filePath":"{path}"}}}}}}}}"#
            );
            audit_events(&ev, &p, &mut a);
            assert!(
                a.flags.iter().any(|f| f.kind == "target_copy_read"
                    && f.detail == path
                    && f.source == "events"),
                "{path}: {:#?}",
                a.flags
            );
        }
    }

    #[test]
    fn git_internal_helpers_are_not_the_agents_commands() {
        // bench-swe-q27 astropy-14096 (git 2.34 in the image): the model's `git stash pop` runs
        // `/usr/lib/git-core/git reflog delete --updateref --rewrite refs/stash@{0}` as a child process
        let log = r#"100   execve("/agent/bin/opencode", ["/agent/bin/opencode", "run"], 0x1 /* 1 vars */) = 0
100   vfork() = 200
200   execve("/usr/bin/bash", ["/usr/bin/bash", "-c", "git stash pop; git reflog"], 0x2 /* 1 vars */) = 0
200   clone(child_stack=NULL, flags=SIGCHLD) = 201
201   execve("/usr/bin/git", ["git", "stash", "pop"], 0x3 /* 1 vars */) = 0
201   clone(child_stack=NULL, flags=SIGCHLD) = 202
202   execve("/usr/lib/git-core/git", ["/usr/lib/git-core/git", "reflog", "delete", "--updateref", "--rewrite", "refs/stash@{0}"], 0x4 /* 1 vars */) = 0
200   clone(child_stack=NULL, flags=SIGCHLD) = 203
203   execve("/usr/bin/git", ["git", "reflog"], 0x5 /* 1 vars */) = 0
"#;
        let mut a = Audit::default();
        audit_strace(log, &policy("/testbed"), "/testbed", &mut a);
        assert!(a.flags.is_empty(), "{:#?}", a.flags);
        let d: Vec<&str> = a.notes.iter().map(|f| f.detail.as_str()).collect();
        assert_eq!(
            d,
            vec!["git reflog"],
            "only the agent's own reflog, as a note: {d:?}"
        );
    }

    #[test]
    fn private_connect_from_a_test_suite_is_a_note() {
        // bench-swe-q27 requests-2931: the model ran the suite; its connect-timeout tests aim at 10.255.255.1
        let log = r#"100   execve("/agent/bin/opencode", ["/agent/bin/opencode", "run"], 0x1 /* 1 vars */) = 0
100   vfork() = 200
200   execve("/usr/bin/bash", ["/usr/bin/bash", "-c", "pytest test_requests.py"], 0x2 /* 1 vars */) = 0
200   clone(child_stack=NULL, flags=SIGCHLD) = 201
201   execve("/opt/miniconda3/envs/testbed/bin/python", ["python", "-m", "pytest"], 0x3 /* 1 vars */) = 0
201   connect(5, {sa_family=AF_INET, sin_port=htons(80), sin_addr=inet_addr("10.255.255.1")}, 16) = -1 ENETUNREACH (Network is unreachable)
201   connect(6, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("140.82.112.3")}, 16) = -1 ENETUNREACH (Network is unreachable)
"#;
        let mut a = Audit::default();
        audit_strace(log, &policy("/testbed"), "/testbed", &mut a);
        assert!(
            a.notes
                .iter()
                .any(|n| n.kind == "network_private" && n.detail.contains("10.255.255.1:80")),
            "{:#?}",
            a.notes
        );
        assert!(
            a.flags
                .iter()
                .any(|f| f.kind == "network" && f.detail.contains("140.82.112.3:443")),
            "{:#?}",
            a.flags
        );
        assert!(!a.flags.iter().any(|f| f.detail.contains("10.255.255.1")));
    }

    #[test]
    fn events_and_bridge() {
        let p = policy("/testbed");
        let ev = [
            r#"{"type":"tool_use","part":{"type":"tool","tool":"read","state":{"input":{"filePath":"/testbed/m.py"}}}}"#,
            r#"{"type":"tool_use","part":{"type":"tool","tool":"webfetch","state":{"input":{"url":"https://github.com/django/django/pull/1"}}}}"#,
            r#"{"type":"tool_use","part":{"type":"tool","tool":"bash","state":{"input":{"command":"cd /testbed && git log --all | head; curl -s https://x"}}}}"#,
            r#"{"type":"tool_use","part":{"type":"tool","tool":"read","state":{"input":{"filePath":"/root/.cache/pip/wheels/demo_lib-9.whl"}}}}"#,
            r#"{"type":"tool_use","part":{"type":"tool","tool":"grep","state":{"input":{"pattern":"SWE-bench","path":"/"}}}}"#,
        ]
        .join("\n");
        let mut a = Audit::default();
        audit_events(&ev, &p, &mut a);
        let kinds: std::collections::BTreeSet<&str> = a
            .flags
            .iter()
            .chain(a.notes.iter())
            .map(|f| f.kind.as_str())
            .collect();
        for want in [
            "web_tool",
            "git_probe",
            "network_tool",
            "outside_repo",
            "target_copy_read",
            "swebench_read",
            "network",
        ] {
            assert!(kinds.contains(want), "{want}: {:#?}", a.flags);
        }
        assert_eq!(a.tool_calls, 5);
        // the model naming the benchmark in its own code is not access
        let mut c = Audit::default();
        audit_events(
            r##"{"part":{"type":"tool","tool":"write","state":{"input":{"filePath":"/testbed/repro.py","content":"# Simulate the SWE-bench harness\nimport django"}}}}"##,
            &p,
            &mut c,
        );
        assert!(c.flags.is_empty(), "{:#?}", c.flags);
        let mut b = Audit::default();
        audit_bridge("{\"allowed\":true,\"method\":\"POST\",\"path\":\"/v1/chat/completions\"}\n{\"allowed\":false,\"method\":\"GET\",\"path\":\"/v1/talus/status\"}\n", &mut b);
        assert_eq!(b.flags.len(), 1);
        assert_eq!(b.flags[0].kind, "bridge_denied");
        // a clean run
        let mut c = Audit::default();
        audit_events(
            r#"{"part":{"type":"tool","tool":"edit","state":{"input":{"filePath":"/testbed/pkg/m.py"}}}}"#,
            &p,
            &mut c,
        );
        assert!(!c.invalid());
    }
}
