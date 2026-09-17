# CLAUDE.md

Guide for **AI Assistants** (Claude Code, Antigravity, Cursor, Windsurf, etc.) working with **vailá** repo.

> See also: [AGENTS.md](./AGENTS.md) — shared rules, all AI agents.

## Project Overview

**vailá** (Versatile Anarcho Integrated Liberation Ánalysis) — open-source Python 3.12 multimodal toolbox, biomechanical data analysis. Integrates IMU, motion capture, markerless tracking (MediaPipe, YOLO), force plates, EMG, GNSS/GPS via Tkinter GUI.

- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila
- **Python:** strictly `>=3.12,<3.13`
- **License:** AGPLv3
- **Build backend:** `hatchling` via [`uv`](https://docs.astral.sh/uv/)

---

## Astral Toolchain

Full [Astral](https://astral.sh) Rust toolchain:

| Tool                                   | Purpose                                                    | Replaces                        |
| -------------------------------------- | ---------------------------------------------------------- | ------------------------------- |
| [`uv`](https://docs.astral.sh/uv/)     | Package manager, venv, Python installer                    | pip, poetry, pyenv, virtualenv  |
| [`ruff`](https://docs.astral.sh/ruff/) | Linter + formatter                                         | flake8, black, isort, pyupgrade |
| [`ty`](https://docs.astral.sh/ty/)     | Static type checker (beta, Rust, 10-100x faster than mypy) | mypy, Pyright                   |

> **Never use** bare `pip install`, `black`, `isort`, `flake8`, `mypy` — always Astral equivalents via `uv run`.

---

## Commands Reference

### uv

```bash
# Run the application
uv run vaila.py

# Sync dependencies (reads uv.lock + pyproject.toml)
uv sync                        # CPU wheels / macOS Metal (default groups: dev + cpu)
uv sync --no-group cpu --group cuda            # NVIDIA CUDA 12.8 wheels + tensorrt + nvidia-ml-py
uv sync --extra sam            # SAM 3 optional stack; video still needs NVIDIA CUDA at runtime
uv sync --extra sapiens        # Sapiens2 Pose (308 kp); bash bin/setup_sapiens2.sh after
uv sync --extra fifa           # FIFA Skeletal Tracking Light (SAM 3D Body + PyTorch Lightning)
uv sync --frozen               # CI mode: fail if lock is outdated

# RECOMMENDED: unified interactive bootstrap (auto-detects OS + NVIDIA + extras)
bash bin/setup_pyproject.sh                                      # Linux / macOS / WSL / Git Bash
pwsh bin/setup_pyproject.ps1                                     # Windows PowerShell
bash bin/setup_pyproject.sh --target=cuda --extras=sam --yes
# The bootstrap only picks `uv sync` arguments — it never rewrites tracked files.

# Manage dependencies
uv add <package>               # Add runtime dependency
uv add --dev <package>         # Add dev dependency
uv remove <package>            # Remove dependency
uv lock                        # Regenerate uv.lock
uv lock --upgrade              # Upgrade all packages

# Python version management
uv python install 3.12         # Install Python 3.12
uv python pin 3.12             # Pin project to 3.12
uv venv --python 3.12          # Create venv with specific version

# Global tools (outside project venv)
uv tool install ruff           # Install ruff globally
uv tool install ty             # Install ty globally
uv tool upgrade ruff           # Upgrade ruff globally
uvx ruff check vaila/          # Run ruff ephemerally (no install)

# Export for legacy tooling
uv export --format requirements-txt > requirements.txt
uv export --format requirements-txt --no-hashes --frozen > requirements.txt
```

### ruff

```bash
# Linting
uv run ruff check vaila/              # Lint all files
uv run ruff check vaila/ --fix        # Lint + auto-fix safe issues
uv run ruff check vaila/ --fix-only   # Apply fixes only, no output
uv run ruff check vaila/ --diff       # Preview what --fix would change

# Formatting (replaces black)
uv run ruff format vaila/             # Format all files
uv run ruff format vaila/ --check     # CI mode: check without writing
uv run ruff format vaila/ --diff      # Preview what format would change

# Single file
uv run ruff check vaila/my_module.py --fix
uv run ruff format vaila/my_module.py
```

**Inline suppression:**

```python
x = some_var  # noqa: F841
x = some_var  # noqa: F841, E501
```

**Config in `pyproject.toml`:**

```toml
[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "NPY", "UP", "B", "C4", "SIM"]
ignore = ["E501", "N806", "N803"]   # scientific uppercase vars are OK

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]            # intentional re-exports
```

### ty

```bash
# Type checking
uv run ty check vaila/                # Check all files
uv run ty check vaila/my_module.py    # Check single file
uv run ty check vaila/ --watch        # Watch mode: re-checks on save

# Override rule severity on CLI
uv run ty check vaila/ --error unresolved-import
uv run ty check vaila/ --warn  possibly-unbound
uv run ty check vaila/ --ignore division-by-zero
```

**Inline suppression:**

```python
x: int = "hello"  # ty: ignore[invalid-assignment]
x: int = "hello"  # ty: ignore[invalid-assignment, unresolved-import]
```

**Config in `pyproject.toml`:**

```toml
[tool.ty.rules]
unresolved-import  = "warn"   # "error" | "warn" | "ignore"
possibly-unbound   = "warn"
division-by-zero   = "error"
unused-ignore-comment = "warn"

[tool.ty.src]
include = ["vaila", "tests"]
exclude = ["vaila/_generated"]
```

> `ty` beta — no drop-in mypy/Pyright replacement; different design/defaults. Use alongside ruff, not instead.

---

## Full QA Pipeline (run before every commit)

```bash
uv run ruff check vaila/ --fix    # fix lint issues
uv run ruff format vaila/         # format code
uv run ty check vaila/            # type check
uv run pytest tests/ -v           # run tests
```

---

## Mandatory: Update metadata on any script change

Edit any Python script (`*.py`) in repo → also update metadata so users see consistent **date/version** across app, docs, help.

### Checklist

- **Edited script header**: update top module docstring/header:
  - **Update Date**: today
  - **Version**: **global vailá version** (same as `vaila.py` header/banner)
- **Main entry point**: change hits GUI/CLI banner → update `vaila.py` header + banner strings.
- **Install scripts**: install/run UX hit → review/update:
  - `install_vaila_linux.sh`, `install_vaila_mac.sh`, `install_vaila_win.ps1`, `install-hooks.sh`
- **Repo README**: update root `README.md` line `Last updated: YYYY-MM-DD` to today.
- **Help docs**:
  - main index `vaila/help/index.md` + `vaila/help/index.html` ("Generated on")
  - changed module help `vaila/help/<module>.md` + `vaila/help/<module>.html` (Version + Updated)

### Writing convention: how to style "vailá"

Write project name **lowercase, italicized** in prose — `*vailá*` Markdown, `<i>vailá</i>` HTML — matches root `README.md` canonical `# _vailá_ - Multimodal Toolbox` title, body usage (`_vailá_`). Never bold (`**vailá**` / `<strong>vailá</strong>`), never capitalize ("Vailá"/"VAILA") in prose. Audit 2026-08-04 found `vaila/help/*.md`/`*.html` mostly plain/unstyled (106/148 `.md`, ~135/148 `.html`), few bolded (9/148 each) or already italicized (3–4/148 each) — no consistent prior norm; italic-lowercase standard going forward. Apply when touching help page for other reason; repo-wide sweep not done yet.

---

## Architecture

### Entry Point & GUI (`vaila.py`)

`vaila.py` defines `Vaila(tk.Tk)`, three frames:

| Frame       | Purpose                                                                             |
| ----------- | ----------------------------------------------------------------------------------- |
| **Frame A** | File Manager — rename, import, export, copy, move, remove, tree, find, SSH transfer |
| **Frame B** | Multimodal Analysis — IMU, MoCap, Markerless 2D/3D, EMG, Force Plate, GNSS          |
| **Frame C** | Tools — CSV editing, C3D conversion, DLT reconstruction, video/image, visualization |

**Lazy imports** in all handler methods, avoid loading full dependency graph at startup.

Two dispatch patterns:

1. **Direct import + call** — same process
2. **Subprocess via `run_vaila_module()`** — separate process (avoids Tkinter conflicts)

**Button grid** (row/col ids match code, e.g. `B1_r1_c4`; run `uv run vaila.py` to see live):

| Area | Buttons |
| --- | --- |
| Frame A (r1) | Rename · Import · Export · Copy · Move · Remove · Tree · Find · Transfer |
| Frame B (r1) | IMU · MoCap Cluster · MoCap Full Body · **Markerless 2D** (coringa: Standard/Advanced/YOLOv26 MediaPipe, Yolo+Markerless_MP, YOLOv26 Tracker/Pose/Seg/Train, SAM 3 video, Sapiens2 Pose, SAM3+Sapiens2 [+Visualize ID], Markerless Hands, MP Angles, Face Mesh, Crop Face, Markerless Live) · **Markerless 3D** (coringa: Standard/Advanced YOLO lift, SAM3+DINOv3 3D [+Visualize ID]) |
| Frame B (r2) | Vector Coding · EMG · Force Plate · GNSS/GPS · MEG/EEG |
| Frame B (r3) | HR/ECG · Vertical Jump · Cube2D · Animal Open Field |
| Frame B (r4) | ML Walkway |
| Frame B (r5) | Ultrasound · Brainstorm · Scout · StartBlock · Pynalty |
| Frame B (r6) | Sprint · tugturn · Soccer Tools (Field KPs AI, Soccer-Field Calib, VEK ElasticKick, FIFA cams→DLT) · Deadlift |
| Frame B (r7) | Treadmill LC (step-based ground-reaction-force workflow, TOML config) |
| Frame C-A (Data Files) | Edit CSV/C3D · C3D↔CSV · Smooth & Filter · **DLT/REC 2D-3D** (coringa: Make DLT2D/DLT3D, Rec2D/Rec3D 1DLT + MultiDLT) · ReID Marker |
| Frame C-B (Video/Image) | Video↔PNG · Planar Geo · Draw Box · Compress Video · Video Stabilizer · Make Sync file · GetPixelCoord · Metadata info · Merge/Split · Distort · Cut · Resize · YT Downloader · Insert Audio · rm Dup PNG |
| Frame C-C (Visualization) | Show C3D/CSV 3D · Plot 2D/3D · Draw Sports · Stroboscopic |

Full ASCII map + descriptions: `README.md` § *vailá Structure and Interface*; per-button docs: `docs/vaila_buttons/`.

### Package Structure (`vaila/`)

~100 self-contained analysis modules. Each module:

- Has `run_*()` or `analyze_*()` entry point called from GUI
- Uses Tkinter `filedialog` for user input prompts
- Reads CSV/C3D via `pandas` / `numpy` / `ezc3d`
- Writes results (CSV + PNG plots) to timestamped output subdirectories

**Key shared modules:**

| Module                                      | Role                                                      |
| ------------------------------------------- | --------------------------------------------------------- |
| `data_processing.py`                        | CSV/C3D reading, auto-header detection                |
| `filtering.py` / `filter_utils.py`          | Butterworth, FIR filter implementations                |
| `readcsv.py`                                 | Header detection (`headersidx`), data reshaping (`reshapedata`), file dialogs |
| `dialogsuser.py` / `dialogsuser_cluster.py` | Reusable Tkinter input dialogs                            |
| `filemanager.py`                            | File management (rename, copy, move, SSH transfer)        |
| `hardware_manager.py`                       | GPU/CPU detection, TensorRT export — **do not duplicate**. First run per model builds VRAM-sized `.engine` (2–5 min, cached); Windows/Linux engines coexist same folder on dual-boot. |
| `interp_smooth_split.py`                    | Interpolation, smoothing, splitting (GUI + CLI)           |

---

## Platform-Specific Configuration

**One `pyproject.toml` + one `uv.lock` for every machine and OS.** No template files, no file copying: the PyTorch backend is a uv **dependency group**, chosen at sync time.

| Hardware                        | Command                                | Wheels                              |
| ------------------------------- | -------------------------------------- | ----------------------------------- |
| Windows/Linux + NVIDIA          | `uv sync --no-group cpu --group cuda`   | `whl/cu128` + tensorrt, nvidia-ml-py |
| CPU-only laptop                 | `uv sync`                              | `whl/cpu`                           |
| macOS Apple Silicon (Metal/MPS) | `uv sync`                              | PyPI wheel (MPS build)              |

`[dependency-groups]` defines `cpu` and `cuda`; `[tool.uv] default-groups = ["dev", "cpu"]` and
`conflicts = [[{group = "cpu"}, {group = "cuda"}]]` make uv resolve **both** variants into the
single committed `uv.lock`. Consequence: a CUDA workstation and a CPU laptop commit byte-identical
`pyproject.toml`/`uv.lock`, so `git pull` / `git commit` never need `git restore`.

Which backend is installed here?

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.11.0+cu128 True  -> cuda group   |   2.11.0+cpu False -> cpu group
```

Install scripts handle it automatically: `install_vaila_linux.sh`, `install_vaila_mac.sh`, `install_vaila_win.ps1`.

---

## Coding Conventions

### Mandatory dual-import pattern

Every module must support package import AND standalone execution:

```python
try:
    from .readcsv import read_csv_file      # package import
    from .filtering import butter_filter
except ImportError:
    from readcsv import read_csv_file       # standalone fallback
    from filtering import butter_filter
```

### Rules

- **GUI framework:** Tkinter only — never introduce Qt, wx, Dear PyGui, etc.
- **Scientific variable names** (X, Y, Z, F, R, T, etc.) valid — suppressed via ruff `N806`/`N803`
- **Output dirs:** always timestamped → `processed_<type>_YYYYMMDD_HHMMSS/`
- **No hard-coded absolute paths**
- **No files ≥20 MiB** (git hook enforced)

---

## Testing

```bash
uv run pytest tests/ -v                              # all tests
uv run pytest tests/test_vaila_and_jump.py -v        # biomechanical calculations
uv run pytest tests/test_tugturn.py -v               # TUG/Turn analysis
uv run pytest tests/test_dlt_rec.py -v               # DLT/Rec math
uv run pytest tests/test_dlt_rec_integration.py -v   # DLT/Rec pipeline
```

Sample data: `tests/vaila_and_jump/` (CSV + TOML).

---

## Common Task Recipes

### Add new analysis module

1. Create `vaila/my_module.py`, `run_my_module()` entry point
2. Apply dual-import pattern at top
3. Use helpers from `dialogsuser.py` for prompts
4. Write results to timestamped output dir
5. Wire button in `vaila.py`, lazy import
6. Lint + type-check: `uv run ruff check vaila/my_module.py --fix && uv run ty check vaila/my_module.py`
7. Add unit test in `tests/`

### Fix all lint + type issues in one shot

```bash
uv run ruff check vaila/ --fix && uv run ruff format vaila/ && uv run ty check vaila/
```

### Run module standalone via CLI

```bash
uv run vaila/interp_smooth_split.py -i /path/to/csv_dir -c smooth_config.toml
```

---

## Security

Open-source under **AGPL-3.0** — never commit API keys, tokens, local credential files. See **[SECURITY.md](SECURITY.md)**, **[CONTRIBUTING.md](CONTRIBUTING.md)**. Use `.env` locally (gitignored); see `.env.example` for safe template.

---

## Agents and skills

Step-by-step workflows, specialized agent roles in `.claude/` dir. For any AI assistant (Claude Code, Antigravity, Cursor, etc.).


### Recent GUI Notes

Per-change notes (what/why/gotchas/validation) for every recent module addition, GUI reorg — Crop Face, Smart Load Tracking CSV, SAM3/Sapiens2/DINOv3 pipelines, DLT/REC family, Markerless 2D/3D chooser reorgs, Geometric ReID v2, GUI→CLI mirror, rec3d Blender export fixes, joint-angle extraction, monocular↔DLT alignment, Sapiens2 3D Pose — moved to **[docs/claude-session-notes-archive.md](docs/claude-session-notes-archive.md)**, keeps this file under char budget. Skim before touching those modules; append new entries there, not here.

### Specialized Agents (`.claude/agents/`)

Role cards, domain experts. Use when task fits their domain:

- [biomechanics-analyst.md](.claude/agents/biomechanics-analyst.md)
- [gui-developer.md](.claude/agents/gui-developer.md)
- [video-processor.md](.claude/agents/video-processor.md)
- [test-writer.md](.claude/agents/test-writer.md)

### Technical Skills (`.claude/skills/`)

Reusable how-to guides, complex workflows:

- **vailá Core**: [create a new analysis module](.claude/skills/create-analysis-module.md), [port a MATLAB algorithm](.claude/skills/port-matlab-algorithm.md), [getpixelvideo-tracking-loader](.claude/skills/getpixelvideo-tracking-loader/SKILL.md) — smart Load Tracking CSV (SAM3 / YOLO auto-detect + bbox → marker anchor prompt), [yolo-fb-gui-cli](.claude/skills/yolo-fb-gui-cli/SKILL.md) — **YOLO + FB** chooser + GUI→CLI terminal mirror (Cursor CLI resume).
- **Sports AI**:
  - [sam3-video](.claude/skills/sam3-video/SKILL.md) — SAM 3 text-prompt video segmentation, GUI help button, prompt presets, **Cross-Chunk Tracklet Linking (v0.3.54)**.
  - [fifa-skeletal-tracking](.claude/skills/fifa-skeletal-tracking/SKILL.md) — FIFA 2026 pipeline (`fifa bootstrap` / `prepare` / `boxes` / `preprocess` / `baseline` / **`dlt-export`** / `pack`), `vaila/fifa_to_dlt.py` (per-frame DLT for **`rec2d.py`/`rec3d.py`** vs fixed-cam **`rec2d_one_dlt2d.py`**), vendored `fifa_starter_lib`, gated SAM 3D Body setup, soccer-field DLT2D calibration.
  - [soccer-field-keypoints-yolo](.claude/skills/soccer-field-keypoints-yolo/SKILL.md) — Ultralytics YOLO **pitch** pose (32 kp), external merged `unified/` tree, `yolo pose train`; see **`docs/fifa_workflow.md` §4.5**, `vaila/help/soccerfield_keypoints_ai.md`.
- **Reports**: [xlsx](.claude/skills/xlsx/SKILL.md) (Excel), [pdf](.claude/skills/pdf/SKILL.md), [pptx](.claude/skills/pptx/SKILL.md) (PowerPoint).
- **Automation**: [mcp-builder](.claude/skills/mcp-builder/SKILL.md) (Model Context Protocol), [webapp-testing](.claude/skills/webapp-testing/SKILL.md).
- **Visualization**: [web-artifacts-builder](.claude/skills/web-artifacts-builder/SKILL.md).

### FIFA Skeletal Tracking Light 2026

vailá ships complete pipeline for
[FIFA Skeletal Tracking Light 2026](https://inside.fifa.com/innovation/innovation-programme/skeletal-tracking)
challenge. One-line setup:

```bash
bash bin/setup_fifa_sam3d.sh              # clone sam_3d_body + gated HF weights
uv run vaila/vaila_sam.py fifa bootstrap \
  --videos-dir /data/FIFA/.../Videos \
  --data-root  /data/FIFA/data
uv run vaila/vaila_sam.py fifa prepare    --data-root data/ --video-source /data/FIFA/.../Videos
uv run vaila/vaila_sam.py fifa boxes      --data-root data/ --sequences data/sequences_val.txt
uv run vaila/vaila_sam.py fifa preprocess --data-root data/ --sequences data/sequences_val.txt  # CUDA
uv run vaila/vaila_sam.py fifa baseline   --data-root data/ --sequences data/sequences_full.txt --output outputs/submission_full.npz  # add --export-camera to refresh cameras/*.npz
uv run vaila/vaila_sam.py fifa dlt-export --cameras-dir data/cameras --output-dir outputs/dlt_per_frame
uv run vaila/vaila_sam.py fifa pack       --submission-full outputs/submission_full.npz --data-root data/ --output-dir outputs/ --split val
```

Companion tool `vaila/soccerfield_calib.py` (button **Soccer-Field Calib**,
Frame C of `vaila.py`) fits DLT2D homography from 29 FIFA keypoints, can
emit `cameras/<stem>_homography.npz` as fallback when sequence has no
official `cameras/*.npz`.

**External unified pitch dataset (YOLO retrain):** `vaila.fifa_dataset_builder` writes `unified/data.yaml` under user-chosen root **outside** git. After QA on `check_all_labels/`, use `vaila.fifa_check_labels_dedupe`, `vaila.fifa_dataset_train_readiness` to align `unified/`, then `yolo pose train data=/ABS/.../unified/data.yaml`. Full recipe: **`docs/fifa_workflow.md` §4.5**.

### Slash Commands (`.claude/commands/`)

Specs, common shortcuts like `/check`, `/new-module`, `/debug` (safe File Manager / downloader diagnosis — `.agents/skills/debug/SKILL.md`).