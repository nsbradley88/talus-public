# Guided install

One command installs Talus on a Strix Halo machine, and three more look after it afterwards:

| Command | What it does |
|---|---|
| `curl -fsSL https://raw.githubusercontent.com/nsbradley88/talus-public/main/install.sh \| bash` | Installs everything and starts `talus setup` |
| `talus setup` | Choose, add or remove models; settings |
| `talus doctor` | Checks the install, says what is wrong in plain words, fixes what it safely can |
| `talus update` | Gets the new release, rebuilds, updates the configs, runs the doctor |

Everything goes into your home directory. The installer asks before it uses `sudo` (for build tools from your
distribution), and it changes boot settings only after it has shown you the exact change and you typed `yes`.
It never reboots by itself.

## What you need

- A Strix Halo machine: Ryzen AI Max with the Radeon 8060S (gfx1151) and 128 GB of memory.
- Linux with the amdgpu driver (any current distribution has it). We test Ubuntu 24.04 and newer; the
  release machine runs Ubuntu 26.04. See [Supported distributions](#supported-distributions).
- Disk: about 15 GB for the ROCm SDK and the build, plus the models you choose (17 GB to 118 GB each).
- An internet connection. No accounts or tokens are needed: every download is public.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/nsbradley88/talus-public/main/install.sh | bash
```

Press Enter to accept each suggested answer. The installer:

1. **Checks this computer**: one line per check (operating system, GPU, memory, disk, GPU access). On a computer
   without a gfx1151 GPU it stops here; `--force` continues anyway (for example to prepare a disk).
2. **Installs the build tools**: a C/C++ compiler, git, curl and Python from your distribution (it shows the
   command and asks first), then CMake, Ninja and Rust (rustup, with the version this release pins) in your
   home directory. Your shell configuration is not changed.
3. **Gets Talus and the ROCm SDK**: the source into `~/.local/share/talus/src` and AMD's ROCm SDK (TheRock wheels,
   the exact version this release was built and measured with, named in the repository's `rocm.version`) into
   `~/.local/share/talus/rocm`, a download of several GB from AMD's index
   (`https://nightly.repo.amd.com/rocm/whl-next/`). If you already have a ROCm SDK, point at it with
   `ROCM_SDK=/path/to/rocm`.
4. **Builds Talus**: the GPU kernel library (this takes a while the first time) and the Rust programs. They go to
   `~/.local/share/talus/bin`, with the `talus` and `talus-tui` commands in `~/.local/bin`. With a published release
   you can skip the compile: `TALUS_RELEASE=<tag>` takes that release's prebuilt programs from GitHub
   (`talus-<tag>-linux-x86_64-gfx1151.tar.gz`, checked against its `.sha256`; `TALUS_PREBUILT_URL=<url>` for
   another copy). They find ROCm through the link `~/.local/share/talus/rocm-sdk`, so no `LD_LIBRARY_PATH` is needed.
   The releases and their assets are listed at `https://github.com/nsbradley88/talus-public/releases`; for example
   `curl -fsSL https://raw.githubusercontent.com/nsbradley88/talus-public/main/install.sh | TALUS_RELEASE=<tag> bash`.
5. **Runs `talus setup`**, below.

Run the same command again at any time: finished steps are skipped and a stopped download resumes.

## Choosing models

`talus setup` shows a report of the machine, then the five models:

```
[2/6] Models
  #  Model                Download  Context With desktop   Answer / prompt  License
  1  Qwen3.8-27B             17 GB  64K     yes            .. / ... tok/s   Apache-2.0
  2  Qwen3.8-Flash-Next      94 GB  256K    yes            .. / ... tok/s   Qwen Community License 1.0
  3  GLM-5.3-Flash          118 GB  64K     headless only  .. / ... tok/s   MIT
  4  MiMo-V2.6-Flash         98 GB  192K    tight          .. / ... tok/s   MIT
  5  Laguna-S-2.1            73 GB  256K    yes            .. / ... tok/s   OpenMDW-1.1
  Speeds: tokens per second in chat (answer) and reading a 2K prompt, from the release run.
  Models to install (numbers like 1,5 or all): [1]
```

(The real table shows the speeds of the release run; [RESULTS.md](RESULTS.md) has them all.)

- **Download** is the size with the drafters (speculative decoding), which are on by default. One drafter is not
  downloaded by default: Laguna-S-2.1's DFlash drafter, a community conversion whose license is unclear (check its
  repository, [wimmmm/poolside-Laguna-S-2.1-DFlash-GGUF](https://huggingface.co/wimmmm/poolside-Laguna-S-2.1-DFlash-GGUF),
  before use). Without it Laguna drafts with prompt lookup; `--advanced` shows the source and asks, and
  `--opt-in-drafters` takes it.
- **With desktop** says whether the model fits while a desktop session runs: Qwen3.8-27B, Qwen3.8-Flash-Next and
  Laguna-S-2.1 do; MiMo-V2.6-Flash is tight; GLM-5.3-Flash needs the machine without a desktop session. Under the
  table, setup also says what stops a model from fitting on your machine right now.
- One model serves at a time. Install several to switch between them in `talus-tui`.
- Qwen3.8-Flash-Next's license has a condition for businesses that offer inference as a service; setup shows it.

Setup checks the free disk space before it downloads anything. Downloads come from the Hugging Face repositories
listed in [MODELS.md](MODELS.md), resume where they stopped, and are checked against the SHA-256 their repositories
publish. If you have an `HF_TOKEN` for Hugging Face rate limits, it is used, never printed and never stored.

`talus setup --advanced` asks more: the folder for the model files, the drafters on or off, Qwen3.8-27B's 4-bit
prompt weights (an opt-in with a trade-off: faster prompt reading, slower answers after it; setup shows the
details), its Fast-tier file (which did not pass our quality gate; setup warns), and whether the server should be
reachable from your network (it then makes an API key).

## System settings

GLM-5.3-Flash and MiMo-V2.6-Flash need the GPU to use more of the shared memory than Linux allows by default
([MODELS.md, System setup](MODELS.md#system-setup)). When one of them is chosen, setup shows each kernel parameter,
its current value and why it matters, then the exact change:

```
--- /dev/null
+++ /etc/default/grub.d/90-talus.cfg
+# Talus: kernel parameters for large models (written by `talus setup`).
+...
+GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT ttm.pages_limit=31457280 ttm.page_pool_size=31457280 kho=off amdgpu.lockup_timeout=10000,60000,10000,10000"
```

and the commands it would run (a backup of the boot menu first, then the new file, then `update-grub`). It
applies them only if you answer yes and then type `yes`. Answer no and it prints the lines to run yourself.
Either way it does not reboot: reboot when it suits you. On Fedora it uses `grubby`; on other boot loaders it
prints the parameters for you to add.

To undo: `sudo rm /etc/default/grub.d/90-talus.cfg && sudo update-grub`, then reboot.

## Running Talus

At the end, setup offers to install a user service (the server then starts when you log in) and to start the
server now. Then:

- The API is at `http://127.0.0.1:8000/v1` (OpenAI and Anthropic compatible; see the README).
- When the server is reachable from your network, clients need the API key from `~/.config/talus/api-key`:

  ```bash
  export TALUS_API_KEY=$(cat ~/.config/talus/api-key)
  curl http://<this-machine>:8000/v1/models -H "Authorization: Bearer $TALUS_API_KEY"
  ```

  `talus-tui` reads `TALUS_API_KEY` too, and can make a new key (Server > API key).
- `talus-tui` is the dashboard: speed, memory, switching models, tuning.
- Without the service, `talus serve` starts the server in the terminal.
- To start the service at boot without logging in: `loginctl enable-linger`.
- GLM-5.3-Flash needs the machine without a desktop session:
  `sudo systemctl set-default multi-user.target` and a reboot (`graphical.target` undoes it).

## talus doctor

`talus doctor` prints one line per check, with what to do when something is wrong:

```
  [ ok ] Operating system               Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic
  [ ok ] GPU                            gfx1151 (Radeon 8060S) in AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S
  [ ok ] GPU memory pool                120 GiB (120 GiB needed by GLM and MiMo)
  [FAIL] Qwen3.8-27B run config         ~/.config/talus/models/qwen3.8-27b.json is missing
                                        fix: write the run config of qwen3.8-27b again
  [warn] Qwen3.8-27B files              14 GB present, 1 not checksum-verified yet
```

It checks: the operating system and kernel; the GPU and your access to it (the `render` and `video` groups); the
ROCm SDK and its version; the Talus programs and kernel library; the kernel parameters against what your models
need; free memory for each model; every model file (present, size, SHA-256; `--deep` hashes the large ones too);
the drafters, draft vocabularies and prompt-weight files; each run config; the server on its port (and whether
another program holds the port); the API key file; the service; free disk; and whether a newer Talus exists.

- `talus doctor --fix` applies the fixes: downloads a missing or corrupt file again, writes a run config again,
  restarts the service, tightens a key file's permissions. It asks before each one (`--yes` to skip the
  questions). Changes that need `sudo` (joining the `render` group, kernel parameters) are shown and asked for
  separately, and boot settings always go through the same show-the-change, type-`yes` step as setup.
- `talus doctor --report` prints a report to paste into a GitHub issue. Your home directory, user and host
  names, IP addresses and keys are removed; read it before you post it.
- `talus hw` shows only the hardware checks.

## talus update

`talus update` fetches the new release, shows what changed, rebuilds, rewrites the run configs (a changed one is
backed up next to it), offers to restart the server, and runs the doctor. `talus update --check` only says whether
there is one. Your models are not downloaded again.

## Adding and removing models

Run `talus setup` again: it offers to add models, remove models, check and repair, or change settings. For scripts:

```bash
talus setup --add laguna            # numbers, ids or short names: 1,5  27b,glm  all
talus setup --remove glm
```

## Without questions

- `--yes` takes every suggested answer. It never changes boot settings unless you also give
  `--allow-system-changes`.
- `--dry-run` prints every action (downloads with their URL, size and checksum, files it would write, commands it
  would run) and changes nothing. It works for `install.sh`, `talus setup`, `talus doctor --fix` and `talus update`.

```bash
curl -fsSL https://raw.githubusercontent.com/nsbradley88/talus-public/main/install.sh | bash -s -- --yes
```

## Where everything goes

| Path | What |
|---|---|
| `~/.local/share/talus/src` | the Talus source |
| `~/.local/share/talus/rocm` | the ROCm SDK |
| `~/.local/share/talus/rocm-sdk` | a link to the ROCm SDK the programs use |
| `~/.local/share/talus/bin` | the programs and the kernel library |
| `~/.local/share/talus/models/<model>` | model files (or the folder you chose) |
| `~/.config/talus/setup.json` | what is installed and where |
| `~/.config/talus/models/<model>.json` | one run config per model (the server's control directory) |
| `~/.config/talus/api-key` | only when the server is reachable from the network |
| `~/.config/systemd/user/talus.service` | the optional service |
| `~/.local/bin/talus`, `~/.local/bin/talus-tui` | the commands |

`TALUS_HOME` moves `~/.local/share/talus` elsewhere.

**Uninstall:** `systemctl --user disable --now talus`, then delete the folders above (and the kernel-parameter
file, if setup wrote one, as shown under System settings).

## Supported distributions

| | Distributions |
|---|---|
| Tested | Ubuntu 24.04, 26.04 (and their derivatives) |
| Handled | Ubuntu 22.04, Debian 12 and 13, Fedora 41 and newer, Arch and its derivatives, openSUSE Tumbleweed |
| Other | the installer asks you to install a C/C++ compiler, git, curl and Python 3 with venv yourself, then continues |

Kernel parameters are applied for you on Debian and Ubuntu (GRUB drop-in) and Fedora (`grubby`); elsewhere the
installer prints them.

## Troubleshooting

- **"no AMD GPU compute device"**: the amdgpu driver is not loaded, or this is not a Strix Halo machine.
- **"you are not in the render group"**: `sudo usermod -aG render,video $USER`, then log out and back in.
- **A download stopped**: run the same command again; it resumes.
- **"changed upstream since this Talus release"**: a model file on Hugging Face no longer matches the checksum this
  release expects. Nothing was downloaded; run `talus update`, or report it.
- **The ROCm version this release was built with cannot be downloaded**: the installer tries AMD's nightly index,
  then its older index, and only then offers AMD's stable ROCm 10.1.0 instead (not the version this release was
  measured with; it should work). To use your own SDK: `ROCM_SDK=/path/to/rocm`.
- Anything else: `talus doctor`, and `talus doctor --report` for an issue.
