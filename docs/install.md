# Install &amp; Run — _vailá_ (Linux, Windows, macOS)

Detailed, platform-by-platform install and troubleshooting guide for _vailá_. This
page goes deeper than the root [`README.md`](../README.md) — for the quick one-liners,
GUI/CLI overview, and full button map, start there. This guide only covers install
mechanics; anything already fully documented in README (uninstall steps, the GPU
support matrix) is linked, not duplicated, so the two pages can't drift apart.

_vailá_'s single, official install method on every platform is
**[uv](https://github.com/astral-sh/uv)** — no separate Python distribution or Conda
needed. The install scripts below install `uv`, pin Python `3.12.14`, pick the right
`pyproject_*.toml` template for your hardware, and run `uv sync`.

## Before you start

- **Disk space:** a few GB (Python 3.12 runtime + dependencies; CUDA/Sapiens2 extras add more).
- **Git**: recommended for a clone-first install (lets `git pull` update later); the
  one-line installers work without a prior clone too.
- Windows: PowerShell 5.1 (already on Windows 10/11) is enough to bootstrap — no
  separate `git`/`pwsh`/`node` required upfront.
- macOS: [Homebrew](https://brew.sh) is used for system dependencies.
- Linux: `apt` is used for system dependencies (Debian/Ubuntu family).

## Linux

### One-line install

```bash
wget -qO- https://raw.githubusercontent.com/vaila-multimodaltoolbox/vaila/main/install_vaila_linux.sh | bash
```

### Clone-first install (recommended)

```bash
git clone https://github.com/vaila-multimodaltoolbox/vaila.git
cd vaila
bash install_vaila_linux.sh
```

### What `install_vaila_linux.sh` does

1. Installs system dependencies via `apt` and self-installs `uv` if missing.
2. Pins Python `3.12.14` (`uv python pin 3.12.14`).
3. Detects an NVIDIA GPU and prompts for optional extras (`gpu`, `sam`, `sapiens`, `fifa`).
4. Copies the matching template (`pyproject_linux_cuda12.toml` for NVIDIA CUDA 12.8,
   otherwise `pyproject_universal_cpu.toml`) over `pyproject.toml` — **before** creating
   the virtual environment, since the active template determines what `uv sync`
   installs.
5. Runs `uv lock` / `uv sync --extra ...` with fallback logic, then verifies CUDA
   wheels and the Sapiens2 editable install if those extras were selected.
6. Adds finishing touches: a desktop entry and SSH-transfer setup for the File Manager.

For the auto-detecting alternative (same logic, callable any time you want to
switch templates), see [`bin/setup_pyproject.sh`](../bin/setup_pyproject.sh) in
root `README.md`'s [Installation and Setup](../README.md#installation-and-setup)
section.

### GPU vs CPU-only

NVIDIA CUDA 12.8 + TensorRT is used automatically when a compatible GPU is
detected; otherwise the CPU-only template is used. See root README's
[Cross-Platform support note](../README.md#-engine-powered-by-uv) for the full
matrix, including the one hard exception (SAM 3 video needs CUDA, no CPU/MPS path).

### Troubleshooting

- **`Permission denied (publickey)` during Git LFS / clone / update check:** your
  `origin` remote is set to SSH (`git@github.com:...`), which needs a GitHub SSH key
  most machines don't have. Re-clone with HTTPS instead (as shown above), or add an
  SSH key to your GitHub account.
- If `uv sync` fails partway through an extra (`sam`/`sapiens`/`fifa`), re-run the
  install script — it re-applies the template and retries `uv sync` idempotently.

## Windows

### One-line install

```powershell
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072; irm https://raw.githubusercontent.com/vaila-multimodaltoolbox/vaila/main/install_vaila_win.ps1 | iex
```

### Clone-first install (recommended)

```powershell
git clone https://github.com/vaila-multimodaltoolbox/vaila.git
cd vaila
powershell -ExecutionPolicy Bypass -File .\install_vaila_win.ps1
```

> **SSH clone on Windows:** if checkout fails on a Git LFS asset with
> `Permission denied (publickey)`, clone with **HTTPS** (as shown above) instead of
> an SSH remote.

### What `install_vaila_win.ps1` does

1. Self-installs `uv` (winget or the official installer) if missing.
2. Pins Python `3.12.14`.
3. Detects an NVIDIA GPU and prompts for optional extras (`gpu`, `sam`, `sapiens`, `fifa`).
4. Copies the matching template (`pyproject_win_cuda12.toml` for NVIDIA CUDA 12.1,
   otherwise `pyproject_universal_cpu.toml`) over `pyproject.toml` **before**
   `uv venv`/`uv sync` — same order-sensitivity as Linux/macOS.
5. Runs `uv lock` / `uv sync --extra ...` with fallback logic.
6. Adds finishing touches: PowerShell/batch launchers, Desktop and Start Menu
   shortcuts, and a Windows Terminal profile.

### GPU vs CPU-only

NVIDIA CUDA 12.1 + TensorRT is used automatically when a compatible GPU is
detected; otherwise CPU-only. See root README's Windows section for the
Local/Portable vs profile-install path choice.

### Troubleshooting

- **Script won't run ("running scripts is disabled"):** run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use the
  `-ExecutionPolicy Bypass -File` form shown above.
- **`Permission denied (publickey)` when checking for updates:** your git `origin`
  remote is SSH-based; re-clone with HTTPS, or configure a GitHub SSH key.

## macOS

### One-line install

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/vaila-multimodaltoolbox/vaila/main/install_vaila_mac.sh)"
```

### Clone-first install (recommended)

```bash
git clone https://github.com/vaila-multimodaltoolbox/vaila.git
cd vaila
bash install_vaila_mac.sh
```

### What `install_vaila_mac.sh` does

1. Installs system dependencies via Homebrew and self-installs `uv` if missing.
2. Pins Python `3.12.14`.
3. Prompts for optional extras (`gpu`/`sam`/`sapiens`/`fifa` as applicable to Apple
   hardware).
4. Copies `pyproject_macos.toml` (Metal/MPS acceleration) over `pyproject.toml`
   **before** creating the virtual environment.
5. Runs `uv lock` / `uv sync --extra ...` with fallback logic.
6. Builds a full `.app` bundle (with icon conversion) so _vailá_ can be launched from
   Applications/Launchpad like a native app.

### GPU vs CPU-only

Apple Silicon uses Metal/MPS acceleration automatically via the general PyTorch
stack; Intel Macs fall back to CPU-only. Note the one exception in root README:
[SAM 3 video](../vaila/help/vaila_sam.md) requires NVIDIA CUDA at runtime and has no
MPS/CPU path — it is not usable on macOS.

### Troubleshooting

- **Homebrew missing:** install it from <https://brew.sh> first, then re-run the
  script.
- **"App can't be opened because it is from an unidentified developer":** right-click
  the `.app` in Applications and choose **Open** once to bypass Gatekeeper for an
  unsigned local build.

## Running _vailá_ after install

- **GUI:** `uv run vaila.py`
- **CLI:** see root README's *Running _vailá_ — GUI and CLI* section for per-module
  CLI invocation.

## Updating / uninstalling

Don't duplicate those steps here — see root README's *Staying up to date* note and
*Uninstallation Instructions* section, which stay the single source of truth.

## See also

- [Root README — Installation and Setup](../README.md#installation-and-setup)
- [AGENTS.md](../AGENTS.md) — hybrid CPU/CUDA template workflow, `uv run` recipes
- [Hugging Face setup](huggingface_setup.md) — gated SAM / SAM 3D / Sapiens2 login
- [Help Index](help.html) / [Help Guide](help.md) — back to the docs hub
